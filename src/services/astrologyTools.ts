import { type KundliSnapshotInput, fetchTransitChart } from './be1Client.js';
import { env } from '../config/env.js';
import { stableHash } from './hash.js';
import { cacheGetJson, cacheSetJson } from './valkeyCache.js';

export interface ToolFinding {
  name: string;
  status: 'ok' | 'partial' | 'unavailable';
  facts: string[];
  evidencePaths: string[];
  missing?: string[];
  snippets?: string[];
}

export interface TransitToolInput {
  kundli: KundliSnapshotInput;
  question: string;
  referenceTimestamp?: number;
}

export type TransitRequestKind = 'current' | 'point' | 'range';

type TransitRequestPlan =
  | { kind: 'current'; samplePoints: [Date] }
  | { kind: 'point'; samplePoints: [Date] }
  | { kind: 'range'; samplePoints: Date[]; rangeStart: Date; rangeEnd: Date };

interface TransitCacheEntry extends Record<string, unknown> {
  transitAtIso: string;
  cachedAt: number;
  payload: Record<string, unknown>;
}

function normalizeQuestion(question: string): string {
  return question.toLowerCase().replace(/\s+/g, ' ').trim();
}

const MONTH_TOKEN_TO_INDEX: Record<string, number> = {
  jan: 0,
  january: 0,
  feb: 1,
  february: 1,
  mar: 2,
  march: 2,
  apr: 3,
  april: 3,
  may: 4,
  jun: 5,
  june: 5,
  jul: 6,
  july: 6,
  aug: 7,
  august: 7,
  sep: 8,
  sept: 8,
  september: 8,
  oct: 9,
  october: 9,
  nov: 10,
  november: 10,
  dec: 11,
  december: 11,
};

type ParsedClockTime = {
  hour: number;
  minute: number;
  second: number;
};

function normalizeHourByMeridiem(hour: number, meridiem: string | null): number | null {
  if (!Number.isInteger(hour) || hour < 0 || hour > 23) return null;
  if (!meridiem) return hour;

  const meridiemLower = meridiem.toLowerCase();
  if (hour < 1 || hour > 12) return null;
  if (meridiemLower === 'am') {
    return hour === 12 ? 0 : hour;
  }
  if (meridiemLower === 'pm') {
    return hour === 12 ? 12 : hour + 12;
  }
  return null;
}

function parseClockTime(fragment: string): ParsedClockTime | null {
  const withMeridiem = /\b(\d{1,2})\s*[:.]\s*(\d{1,2})(?:\s*[:.]\s*(\d{1,2}))?\s*(am|pm)\b/i.exec(fragment);
  if (withMeridiem) {
    const hourRaw = Number(withMeridiem[1]);
    const minute = Number(withMeridiem[2]);
    const second = withMeridiem[3] ? Number(withMeridiem[3]) : 0;
    const hour = normalizeHourByMeridiem(hourRaw, withMeridiem[4]);

    if (
      hour !== null
      && Number.isInteger(minute)
      && minute >= 0
      && minute <= 59
      && Number.isInteger(second)
      && second >= 0
      && second <= 59
    ) {
      return { hour, minute, second };
    }
  }

  const hourMeridiemOnly = /\b(\d{1,2})\s*(am|pm)\b/i.exec(fragment);
  if (hourMeridiemOnly) {
    const hourRaw = Number(hourMeridiemOnly[1]);
    const hour = normalizeHourByMeridiem(hourRaw, hourMeridiemOnly[2]);
    if (hour !== null) {
      return { hour, minute: 0, second: 0 };
    }
  }

  const twentyFourHour = /\b(\d{1,2})\s*[:.]\s*(\d{1,2})(?:\s*[:.]\s*(\d{1,2}))?\b/.exec(fragment);
  if (twentyFourHour) {
    const hour = Number(twentyFourHour[1]);
    const minute = Number(twentyFourHour[2]);
    const second = twentyFourHour[3] ? Number(twentyFourHour[3]) : 0;
    if (
      Number.isInteger(hour)
      && hour >= 0
      && hour <= 23
      && Number.isInteger(minute)
      && minute >= 0
      && minute <= 59
      && Number.isInteger(second)
      && second >= 0
      && second <= 59
    ) {
      return { hour, minute, second };
    }
  }

  return null;
}

function findClockTimeNearIndex(question: string, index: number): ParsedClockTime | null {
  const from = Math.max(0, index - 48);
  const to = Math.min(question.length, index + 64);
  const nearby = question.slice(from, to);
  return parseClockTime(nearby);
}

function extractAnyClockTime(question: string): ParsedClockTime | null {
  return parseClockTime(question);
}

function buildUtcDate(year: number, monthIndex: number, day: number, time?: ParsedClockTime | null): Date | null {
  if (!Number.isInteger(year) || !Number.isInteger(monthIndex) || !Number.isInteger(day)) {
    return null;
  }
  if (monthIndex < 0 || monthIndex > 11 || day < 1 || day > 31) {
    return null;
  }

  const hour = time?.hour ?? 0;
  const minute = time?.minute ?? 0;
  const second = time?.second ?? 0;

  const candidate = new Date(Date.UTC(year, monthIndex, day, hour, minute, second, 0));
  if (
    candidate.getUTCFullYear() !== year
    || candidate.getUTCMonth() !== monthIndex
    || candidate.getUTCDate() !== day
  ) {
    return null;
  }

  return candidate;
}

function parseIsoLikeToken(token: string): Date | null {
  const normalized = token.trim();
  if (!normalized) return null;

  if (/[zZ]|[+-]\d{2}:?\d{2}$/.test(normalized)) {
    const parsed = new Date(normalized);
    return Number.isNaN(parsed.getTime()) ? null : parsed;
  }

  const raw = /^(20\d{2})-(\d{2})-(\d{2})(?:[t\s](\d{1,2})[:.](\d{1,2})(?::(\d{1,2}))?\s*(am|pm)?)?$/i.exec(normalized);
  if (!raw) {
    const parsed = new Date(normalized);
    return Number.isNaN(parsed.getTime()) ? null : parsed;
  }

  const year = Number(raw[1]);
  const month = Number(raw[2]) - 1;
  const day = Number(raw[3]);

  let clock: ParsedClockTime | null = null;
  if (raw[4] !== undefined && raw[5] !== undefined) {
    const hourRaw = Number(raw[4]);
    const minute = Number(raw[5]);
    const second = raw[6] ? Number(raw[6]) : 0;
    const hour = normalizeHourByMeridiem(hourRaw, raw[7] ?? null);
    if (
      hour !== null
      && Number.isInteger(minute)
      && minute >= 0
      && minute <= 59
      && Number.isInteger(second)
      && second >= 0
      && second <= 59
    ) {
      clock = { hour, minute, second };
    }
  }

  return buildUtcDate(year, month, day, clock);
}

function parseNaturalDateMatch(match: RegExpExecArray, dayIndex: number, monthIndex: number, yearIndex: number, question: string): Date | null {
  const day = Number(match[dayIndex]);
  const monthToken = String(match[monthIndex] ?? '').toLowerCase().replace(/\.$/, '');
  const month = MONTH_TOKEN_TO_INDEX[monthToken];
  const year = Number(match[yearIndex]);

  if (!Number.isInteger(month) || !Number.isInteger(day) || !Number.isInteger(year)) {
    return null;
  }

  const explicitTime = findClockTimeNearIndex(question, match.index ?? 0) ?? extractAnyClockTime(question);
  return buildUtcDate(year, month, day, explicitTime);
}

function buildYearWindow(year: number): { start: Date; end: Date } {
  return {
    start: new Date(Date.UTC(year, 0, 1, 0, 0, 0, 0)),
    end: new Date(Date.UTC(year, 11, 31, 23, 59, 59, 0)),
  };
}

function buildMonthWindow(year: number, monthIndex: number): { start: Date; end: Date } {
  return {
    start: new Date(Date.UTC(year, monthIndex, 1, 0, 0, 0, 0)),
    end: new Date(Date.UTC(year, monthIndex + 1, 0, 23, 59, 59, 0)),
  };
}

function extractLikelyForecastYears(question: string, referenceTimestamp: number): number[] {
  const refYear = new Date(Number.isFinite(referenceTimestamp) ? referenceTimestamp : Date.now()).getUTCFullYear();
  const minYear = refYear - 15;
  const maxYear = refYear + 30;
  const years = new Set<number>();

  for (const match of question.matchAll(/\b(20\d{2})\b/g)) {
    const year = Number(match[1]);
    if (Number.isInteger(year) && year >= minYear && year <= maxYear) {
      years.add(year);
    }
  }

  return [...years].sort((a, b) => a - b);
}

function resolveYearSpan(question: string, referenceTimestamp: number): { start: Date; end: Date } | null {
  const directRange = /\b(20\d{2})\s*[-/]\s*(\d{2}|20\d{2})\b(?!\s*[-/]\s*\d{1,2})/.exec(question);
  if (directRange) {
    const startYear = Number(directRange[1]);
    const rawEnd = directRange[2];
    let endYear = Number(rawEnd);
    if (rawEnd.length === 2) {
      const century = Math.floor(startYear / 100) * 100;
      endYear = century + endYear;
      if (endYear < startYear) {
        endYear += 100;
      }
    }

    if (Number.isInteger(startYear) && Number.isInteger(endYear) && endYear >= startYear) {
      return {
        start: buildYearWindow(startYear).start,
        end: buildYearWindow(endYear).end,
      };
    }
  }

  const fromToRange = /\b(?:from\s+)?(20\d{2})\s+(?:to|until|through|till)\s+(20\d{2})\b/i.exec(question);
  if (fromToRange) {
    const startYear = Number(fromToRange[1]);
    const endYear = Number(fromToRange[2]);
    if (Number.isInteger(startYear) && Number.isInteger(endYear) && endYear >= startYear) {
      return {
        start: buildYearWindow(startYear).start,
        end: buildYearWindow(endYear).end,
      };
    }
  }

  const likelyYears = extractLikelyForecastYears(question, referenceTimestamp);
  const yearPairConnector = /\b(or|and|to|till|until|through|between|from)\b/i.test(question) || /[?,/]/.test(question);
  if (likelyYears.length >= 2 && yearPairConnector) {
    const startYear = likelyYears[0];
    const endYear = likelyYears[likelyYears.length - 1];
    if (endYear >= startYear) {
      return {
        start: buildYearWindow(startYear).start,
        end: buildYearWindow(endYear).end,
      };
    }
  }

  return null;
}

function resolveSingleYearWindow(question: string, referenceTimestamp: number): { start: Date; end: Date } | null {
  const likelyYears = extractLikelyForecastYears(question, referenceTimestamp);
  if (likelyYears.length !== 1) return null;

  const hasYearAnchor = /\b(in|for|during|around|by)\s+20\d{2}\b/i.test(question)
    || /\b20\d{2}\b\s*(?:\?|$|[.,!])/i.test(question);

  if (!hasYearAnchor) return null;

  return buildYearWindow(likelyYears[0]);
}

function resolveNamedMonthYearWindow(question: string, referenceTimestamp: number): { start: Date; end: Date } | null {
  const refYear = new Date(Number.isFinite(referenceTimestamp) ? referenceTimestamp : Date.now()).getUTCFullYear();
  const minYear = refYear - 15;
  const maxYear = refYear + 30;

  const named = /\b(?:in|during|around|for)?\s*(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(20\d{2})\b/i.exec(question);
  if (named) {
    const month = MONTH_TOKEN_TO_INDEX[String(named[1]).toLowerCase().replace(/\.$/, '')];
    const year = Number(named[2]);
    if (Number.isInteger(month) && Number.isInteger(year) && year >= minYear && year <= maxYear) {
      return buildMonthWindow(year, month);
    }
  }

  return null;
}

function resolveCurrentMonthWindow(referenceTimestamp: number): { start: Date; end: Date } {
  const base = new Date(Number.isFinite(referenceTimestamp) ? referenceTimestamp : Date.now());
  return buildMonthWindow(base.getUTCFullYear(), base.getUTCMonth());
}

function resolveCurrentYearWindow(referenceTimestamp: number): { start: Date; end: Date } {
  const base = new Date(Number.isFinite(referenceTimestamp) ? referenceTimestamp : Date.now());
  return buildYearWindow(base.getUTCFullYear());
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function isSameUtcDay(a: Date, b: Date): boolean {
  return a.getUTCFullYear() === b.getUTCFullYear()
    && a.getUTCMonth() === b.getUTCMonth()
    && a.getUTCDate() === b.getUTCDate();
}

function resolveTransitCacheTtlSeconds(transitAt: Date, referenceTimestamp: number): number {
  const timingTtl = Math.max(1, env.TIMING_CACHE_TTL_SECONDS);
  const referenceDate = new Date(Number.isFinite(referenceTimestamp) ? referenceTimestamp : Date.now());
  if (isSameUtcDay(transitAt, referenceDate)) {
    return Math.max(60, Math.floor(timingTtl / 24));
  }
  return timingTtl;
}

function buildTransitCacheKey(kundli: KundliSnapshotInput, transitAt: Date): string {
  const normalizedTransitAt = new Date(transitAt.getTime());
  normalizedTransitAt.setSeconds(0, 0);

  const payload = {
    latitude: kundli.latitude,
    longitude: kundli.longitude,
    year: kundli.year,
    month: kundli.month,
    day: kundli.day,
    hour: kundli.hour,
    min: kundli.min,
    sec: kundli.sec,
    time_zone: kundli.time_zone,
    transitAt: normalizedTransitAt.toISOString(),
  };

  return `transit:${stableHash(JSON.stringify(payload))}`;
}

function addMonths(date: Date, months: number): Date {
  const next = new Date(date.getTime());
  next.setMonth(next.getMonth() + months);
  return next;
}

function addYears(date: Date, years: number): Date {
  const next = new Date(date.getTime());
  next.setFullYear(next.getFullYear() + years);
  return next;
}

function subtractMonths(date: Date, months: number): Date {
  return addMonths(date, -months);
}

function subtractYears(date: Date, years: number): Date {
  return addYears(date, -years);
}

function addByUnit(date: Date, amount: number, unit: string): Date {
  if (unit.startsWith('day')) {
    const next = new Date(date.getTime());
    next.setDate(next.getDate() + amount);
    return next;
  }
  if (unit.startsWith('week')) {
    const next = new Date(date.getTime());
    next.setDate(next.getDate() + amount * 7);
    return next;
  }
  if (unit.startsWith('month')) {
    return addMonths(date, amount);
  }
  return addYears(date, amount);
}

function resolveExplicitDate(question: string): Date | null {
  const isoLike = question.match(/\b(20\d{2}-\d{2}-\d{2}(?:[t\s]\d{1,2}[:.]\d{1,2}(?::\d{1,2})?\s*(?:am|pm)?)?(?:z|[+-]\d{2}:?\d{2})?)\b/i);
  if (isoLike?.[1]) {
    const parsed = parseIsoLikeToken(isoLike[1]);
    if (parsed) return parsed;
  }

  const slashDate = question.match(/\b(\d{1,2})[\/.-](\d{1,2})[\/.-](20\d{2})\b/);
  if (slashDate) {
    const [, dayRaw, monthRaw, yearRaw] = slashDate;
    const explicitTime = findClockTimeNearIndex(question, slashDate.index ?? 0) ?? extractAnyClockTime(question);
    const parsed = buildUtcDate(Number(yearRaw), Number(monthRaw) - 1, Number(dayRaw), explicitTime);
    if (parsed) return parsed;
  }

  const dayMonthYear = /\b(\d{1,2})\s*(?:st|nd|rd|th)?[\s,.-]*(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)[\s,.-]*(20\d{2})\b/i.exec(question);
  if (dayMonthYear) {
    const parsed = parseNaturalDateMatch(dayMonthYear, 1, 2, 3, question);
    if (parsed) return parsed;
  }

  const monthDayYear = /\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s*(\d{1,2})(?:st|nd|rd|th)?[\s,.-]*(20\d{2})\b/i.exec(question);
  if (monthDayYear) {
    const parsed = parseNaturalDateMatch(monthDayYear, 2, 1, 3, question);
    if (parsed) return parsed;
  }

  return null;
}

function extractExplicitDates(question: string): Date[] {
  const found: Date[] = [];

  const isoPattern = /\b(20\d{2}-\d{2}-\d{2}(?:[t\s]\d{1,2}[:.]\d{1,2}(?::\d{1,2})?\s*(?:am|pm)?)?(?:z|[+-]\d{2}:?\d{2})?)\b/gi;
  for (const match of question.matchAll(isoPattern)) {
    const parsed = parseIsoLikeToken(match[1]);
    if (parsed) {
      found.push(parsed);
    }
  }

  const slashPattern = /\b(\d{1,2})[\/.-](\d{1,2})[\/.-](20\d{2})\b/g;
  for (const match of question.matchAll(slashPattern)) {
    const parsed = buildUtcDate(Number(match[3]), Number(match[2]) - 1, Number(match[1]), extractAnyClockTime(question));
    if (parsed) {
      found.push(parsed);
    }
  }

  const dayMonthYearPattern = /\b(\d{1,2})\s*(?:st|nd|rd|th)?[\s,.-]*(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)[\s,.-]*(20\d{2})\b/gi;
  for (const match of question.matchAll(dayMonthYearPattern)) {
    const month = MONTH_TOKEN_TO_INDEX[String(match[2]).toLowerCase().replace(/\.$/, '')];
    const parsed = buildUtcDate(Number(match[3]), month, Number(match[1]), findClockTimeNearIndex(question, match.index ?? 0) ?? extractAnyClockTime(question));
    if (parsed) {
      found.push(parsed);
    }
  }

  const monthDayYearPattern = /\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s*(\d{1,2})(?:st|nd|rd|th)?[\s,.-]*(20\d{2})\b/gi;
  for (const match of question.matchAll(monthDayYearPattern)) {
    const month = MONTH_TOKEN_TO_INDEX[String(match[1]).toLowerCase().replace(/\.$/, '')];
    const parsed = buildUtcDate(Number(match[3]), month, Number(match[2]), findClockTimeNearIndex(question, match.index ?? 0) ?? extractAnyClockTime(question));
    if (parsed) {
      found.push(parsed);
    }
  }

  return found;
}

function getMidpointDate(start: Date, end: Date): Date {
  return new Date(start.getTime() + Math.floor((end.getTime() - start.getTime()) / 2));
}

function dedupeSamplePoints(points: Date[]): Date[] {
  const byMinute = new Map<number, Date>();
  for (const point of points) {
    const rounded = new Date(point.getTime());
    rounded.setSeconds(0, 0);
    byMinute.set(rounded.getTime(), rounded);
  }
  return [...byMinute.values()].sort((a, b) => a.getTime() - b.getTime());
}

function resolveTransitRequestPlan(question: string, referenceTimestamp: number): TransitRequestPlan {
  const normalized = normalizeQuestion(question);
  const explicitDates = extractExplicitDates(question);
  const base = new Date(Number.isFinite(referenceTimestamp) ? referenceTimestamp : Date.now());
  const currentRequest = /\b(current|now|today|at present|currently)\b/.test(normalized);

  if (currentRequest) {
    const explicitCurrentPoint = resolveTransitAt(question, referenceTimestamp);
    return {
      kind: 'current',
      samplePoints: [explicitCurrentPoint ?? new Date(Number.isFinite(referenceTimestamp) ? referenceTimestamp : Date.now())],
    };
  }

  const yearSpan = resolveYearSpan(question, referenceTimestamp);
  if (yearSpan) {
    const samplePoints = dedupeSamplePoints([yearSpan.start, getMidpointDate(yearSpan.start, yearSpan.end), yearSpan.end]);
    return {
      kind: 'range',
      samplePoints,
      rangeStart: yearSpan.start,
      rangeEnd: yearSpan.end,
    };
  }

  const namedMonthYearWindow = resolveNamedMonthYearWindow(question, referenceTimestamp);
  if (namedMonthYearWindow) {
    const samplePoints = dedupeSamplePoints([
      namedMonthYearWindow.start,
      getMidpointDate(namedMonthYearWindow.start, namedMonthYearWindow.end),
      namedMonthYearWindow.end,
    ]);
    return {
      kind: 'range',
      samplePoints,
      rangeStart: namedMonthYearWindow.start,
      rangeEnd: namedMonthYearWindow.end,
    };
  }

  if (/\b(this|current)\s+month\b|\bthis\s+month\b/i.test(normalized)) {
    const monthWindow = resolveCurrentMonthWindow(referenceTimestamp);
    const samplePoints = dedupeSamplePoints([monthWindow.start, getMidpointDate(monthWindow.start, monthWindow.end), monthWindow.end]);
    return {
      kind: 'range',
      samplePoints,
      rangeStart: monthWindow.start,
      rangeEnd: monthWindow.end,
    };
  }

  if (/\b(this|current)\s+year\b|\bthis\s+year\b/i.test(normalized)) {
    const yearWindow = resolveCurrentYearWindow(referenceTimestamp);
    const samplePoints = dedupeSamplePoints([yearWindow.start, getMidpointDate(yearWindow.start, yearWindow.end), yearWindow.end]);
    return {
      kind: 'range',
      samplePoints,
      rangeStart: yearWindow.start,
      rangeEnd: yearWindow.end,
    };
  }

  if (explicitDates.length === 0) {
    const singleYearWindow = resolveSingleYearWindow(question, referenceTimestamp);
    if (singleYearWindow) {
      const samplePoints = dedupeSamplePoints([
        singleYearWindow.start,
        getMidpointDate(singleYearWindow.start, singleYearWindow.end),
        singleYearWindow.end,
      ]);
      return {
        kind: 'range',
        samplePoints,
        rangeStart: singleYearWindow.start,
        rangeEnd: singleYearWindow.end,
      };
    }
  }

  const openEndedTimingIntent = (
    /\b(when|by when|which year|what age|what time|timing|timeline|period|window|phase)\b/.test(normalized)
    || (/(?:\bcan\s+i\b|\bwill\s+i\b|\bshould\s+i\b)/.test(normalized)
      && /\b(marry|marriage|relationship|partner|spouse|career|job|business|promotion|finance|money|health|education|children|property|travel|spiritual|spirituality)\b/.test(normalized))
  );

  if (openEndedTimingIntent && explicitDates.length === 0) {
    const rangeStart = base;
    const rangeEnd = addByUnit(base, 12, 'month');
    const samplePoints = dedupeSamplePoints([rangeStart, getMidpointDate(rangeStart, rangeEnd), rangeEnd]);
    return {
      kind: 'range',
      samplePoints,
      rangeStart,
      rangeEnd,
    };
  }

  const explicitRangeConnector = /\bbetween\b[\s\S]{0,40}\b(?:and|to)\b/.test(normalized)
    || /\bfrom\b[\s\S]{0,40}\b(?:to|till|until|through)\b/.test(normalized);

  if (explicitRangeConnector && explicitDates.length >= 2) {
    const sorted = [...explicitDates].sort((a, b) => a.getTime() - b.getTime());
    const rangeStart = sorted[0];
    const rangeEnd = sorted[sorted.length - 1];
    const samplePoints = dedupeSamplePoints([rangeStart, getMidpointDate(rangeStart, rangeEnd), rangeEnd]);
    return {
      kind: 'range',
      samplePoints,
      rangeStart,
      rangeEnd,
    };
  }

  const explicitFutureWindow = normalized.match(/\b(?:next|upcoming|coming)\s+(\d+)\s+(day|days|week|weeks|month|months|year|years)\b/);
  if (explicitFutureWindow) {
    const amount = Number(explicitFutureWindow[1]);
    const unit = explicitFutureWindow[2];
    const rangeStart = base;
    const rangeEnd = addByUnit(base, amount, unit);
    const samplePoints = dedupeSamplePoints([rangeStart, getMidpointDate(rangeStart, rangeEnd), rangeEnd]);
    return {
      kind: 'range',
      samplePoints,
      rangeStart,
      rangeEnd,
    };
  }

  const explicitPastWindow = normalized.match(/\b(?:last|past|previous)\s+(\d+)\s+(day|days|week|weeks|month|months|year|years)\b/);
  if (explicitPastWindow) {
    const amount = Number(explicitPastWindow[1]);
    const unit = explicitPastWindow[2];
    const rangeEnd = base;
    const rangeStart = addByUnit(base, -amount, unit);
    const samplePoints = dedupeSamplePoints([rangeStart, getMidpointDate(rangeStart, rangeEnd), rangeEnd]);
    return {
      kind: 'range',
      samplePoints,
      rangeStart,
      rangeEnd,
    };
  }

  const point = resolveTransitAt(question, referenceTimestamp)
    ?? new Date(Number.isFinite(referenceTimestamp) ? referenceTimestamp : Date.now());
  return {
    kind: 'point',
    samplePoints: [point],
  };
}

export function resolveTransitRequestKind(question: string, referenceTimestamp: number): TransitRequestKind {
  return resolveTransitRequestPlan(question, referenceTimestamp).kind;
}

export function resolveTransitAt(question: string, referenceTimestamp: number): Date | null {
  const normalized = normalizeQuestion(question);
  const base = new Date(Number.isFinite(referenceTimestamp) ? referenceTimestamp : Date.now());

  const explicit = resolveExplicitDate(question);
  if (explicit) return explicit;

  const currentRequest = /\b(current|now|today|at present|currently)\b/.test(normalized);
  if (currentRequest) {
    const explicitTime = extractAnyClockTime(question);
    if (explicitTime) {
      const currentWithTime = new Date(base.getTime());
      currentWithTime.setUTCHours(explicitTime.hour, explicitTime.minute, explicitTime.second, 0);
      return currentWithTime;
    }
  }

  if (/\btomorrow\b/.test(normalized)) {
    const next = new Date(base.getTime());
    next.setDate(next.getDate() + 1);
    return next;
  }

  if (/\bday after tomorrow\b/.test(normalized)) {
    const next = new Date(base.getTime());
    next.setDate(next.getDate() + 2);
    return next;
  }

  if (/\bnext week\b/.test(normalized)) {
    const next = new Date(base.getTime());
    next.setDate(next.getDate() + 7);
    return next;
  }

  if (/\bnext month\b/.test(normalized)) {
    return addMonths(base, 1);
  }

  if (/\bnext year\b/.test(normalized)) {
    return addYears(base, 1);
  }

  if (/\b(last|past|previous|earlier|ago)\b/.test(normalized)) {
    const decade = normalized.match(/\b(?:last|past|previous|earlier)\s+(\d+)\s+(day|days|week|weeks|month|months|year|years)\b/);
    if (decade) {
      const amount = Number(decade[1]);
      const unit = decade[2];
      if (unit.startsWith('day')) {
        const next = new Date(base.getTime());
        next.setDate(next.getDate() - amount);
        return next;
      }
      if (unit.startsWith('week')) {
        const next = new Date(base.getTime());
        next.setDate(next.getDate() - amount * 7);
        return next;
      }
      if (unit.startsWith('month')) {
        return subtractMonths(base, amount);
      }
      if (unit.startsWith('year')) {
        return subtractYears(base, amount);
      }
    }

    if (/\b(10|ten)\s+years?\s+ago\b/.test(normalized) || /\bpast\s+10\s+years?\b/.test(normalized) || /\blast\s+10\s+years?\b/.test(normalized)) {
      return subtractYears(base, 10);
    }
  }

  const relative = normalized.match(/\bin\s+(\d+)\s+(day|days|week|weeks|month|months|year|years)\b/);
  if (relative) {
    const amount = Number(relative[1]);
    const unit = relative[2];
    const next = new Date(base.getTime());

    if (unit.startsWith('day')) next.setDate(next.getDate() + amount);
    else if (unit.startsWith('week')) next.setDate(next.getDate() + amount * 7);
    else if (unit.startsWith('month')) next.setMonth(next.getMonth() + amount);
    else if (unit.startsWith('year')) next.setFullYear(next.getFullYear() + amount);

    return next;
  }

  return null;
}

function extractTransitBlocks(payload: any): any[] {
  const candidates = [payload?.chart?.transit, payload?.chart?.transits, payload?.chart?.gochar, payload?.transit, payload?.transits, payload?.gochar];
  return candidates.filter(Boolean);
}

async function buildTransitToolFindingForPlan(input: TransitToolInput, plan: TransitRequestPlan): Promise<ToolFinding> {
  const referenceTimestamp = input.referenceTimestamp ?? Date.now();
  const samplePoints = plan.samplePoints;

  try {
    const payloads: Array<{ transitAt: Date; payload: Record<string, unknown>; cacheHit: boolean }> = [];

    for (const transitAt of samplePoints) {
      const cacheKey = buildTransitCacheKey(input.kundli, transitAt);
      const cached = await cacheGetJson<TransitCacheEntry>(cacheKey);
      const cachedPayload = cached && isPlainObject(cached.payload)
        ? cached.payload
        : null;
      const cacheHit = Boolean(cachedPayload);

      const fetchedPayload = cacheHit
        ? null
        : await fetchTransitChart(input.kundli, { transitAt });

      const transitPayload = cachedPayload
        ?? (isPlainObject(fetchedPayload) ? fetchedPayload : { value: fetchedPayload as unknown });

      if (!cacheHit && isPlainObject(transitPayload)) {
        await cacheSetJson(cacheKey, {
          transitAtIso: transitAt.toISOString(),
          cachedAt: Date.now(),
          payload: transitPayload,
        }, resolveTransitCacheTtlSeconds(transitAt, referenceTimestamp));
      }

      payloads.push({
        transitAt,
        payload: transitPayload,
        cacheHit,
      });
    }

    const primary = payloads[0];
    const allBlocksCount = payloads.reduce((acc, item) => acc + extractTransitBlocks(item.payload).length, 0);
    const chart = (primary.payload as { chart?: any })?.chart ?? primary.payload;
    const transitChart = (primary.payload as { chart?: { transit?: unknown; natal?: unknown } })?.chart?.transit
      ?? (primary.payload as { chart?: { transit?: unknown; natal?: unknown } })?.chart;
    const transitLagna = isPlainObject(transitChart)
      ? (getTransitByPath(transitChart, 'lagna.Lg') ?? getTransitByPath(transitChart, 'lagna'))
      : null;
    const transitLagnaRashi = isPlainObject(transitLagna)
      ? parseTransitRashiNumber((transitLagna as Record<string, unknown>).rashi)
      : null;
    const currentSun = chart?.transit?.Su ?? chart?.transit?.Sun ?? chart?.transit?.sun ?? chart?.Su ?? chart?.Sun;

    const hits = payloads.filter((item) => item.cacheHit).length;
    const misses = payloads.length - hits;

    const facts: string[] = [
      `Transit reference timestamp used: ${primary.transitAt.toISOString()}.`,
      plan.kind === 'range'
        ? `Transit range analysis generated for ${plan.rangeStart.toISOString()} to ${plan.rangeEnd.toISOString()} using ${payloads.length} sample point(s).`
        : plan.kind === 'current'
          ? `Current transit snapshot generated from backend transit-chart strategy for ${primary.transitAt.toISOString()}.`
          : `Transit chart generated directly from backend endpoint for ${primary.transitAt.toISOString()}.`,
      misses === 0
        ? 'Transit cache: hit (Valkey).'
        : hits === 0
          ? 'Transit cache: miss (fresh backend fetch).'
          : `Transit cache: partial hit (${hits} hit, ${misses} miss).`,
      allBlocksCount > 0
        ? `Transit/gochar block(s) detected across samples: ${allBlocksCount}.`
        : 'Transit response was returned, but no dedicated transit block was found in the payload shape.',
      'Transit house contract: derive house from transit Lagna rashi and planet rashi using whole-sign formula ((planet_rashi - lagna_rashi + 12) % 12) + 1; treat backend house_number as auxiliary backend metadata.',
    ];

    const wantsFullPlanetSnapshot = /\b((?:current\s+)?transit\s+details?|full\s+transit|all\s+transits?|complete\s+transit|transit\s+snapshot|(?:current\s+)?gochar\s+details?)\b/i.test(input.question);

    if (transitChart && typeof transitChart === 'object') {
      facts.push(...summarizeTransitChart(transitChart, input.question));
      if (plan.kind === 'current' || wantsFullPlanetSnapshot) {
        facts.push('Full transit planet snapshot (ascendant-relative):');
        facts.push(...summarizeFullTransitPlanetSnapshot(transitChart));
      }
      if ((primary.payload as { chart?: { natal?: unknown } })?.chart?.natal) {
        facts.push('Natal + transit charts were returned together; use natal as the baseline and transit as the timing trigger window.');
      }
    }

    if (currentSun && typeof currentSun === 'object') {
      const sun = currentSun as Record<string, unknown>;
      const rashi = sun.rashi ?? sun.sign ?? 'n/a';
      const degree = sun.degree !== undefined ? Number(sun.degree).toFixed(2) : null;
      const derivedHouse = deriveWholeSignHouseFromLagna(rashi, transitLagnaRashi);
      facts.push(`Transit Sun: sign ${String(rashi)}${derivedHouse !== null ? `, house ${derivedHouse} (ascendant-relative)` : ''}${degree ? `, ${degree}°` : ''}.`);
    }

    if (plan.kind === 'range') {
      const focus = getTransitFocusDetails(input.question);
      const focusHouses = new Set(
        focus.houses
          .map((house) => Number(house))
          .filter((house) => Number.isInteger(house) && house >= 1 && house <= 12)
      );
      const intervalSignals = buildTransitIntervalSignals(
        payloads.map((item) => ({ transitAt: item.transitAt, payload: item.payload })),
        focus.planets,
        focusHouses
      );

      if (intervalSignals.length > 0) {
        facts.push(...summarizeTransitIntervalSignals(intervalSignals, focus.label));
        facts.push('Interval prediction note: supportive/pressure windows are probabilistic checkpoints and should be synthesized with dasha context for final timing decisions.');
      }

      facts.push(`Forecast window interpreted from question: ${plan.rangeStart.toISOString()} → ${plan.rangeEnd.toISOString()}.`);
    } else if (/\b(tomorrow|next week|next month|next year|in\s+\d+\s+(day|days|week|weeks|month|months|year|years))\b/i.test(input.question)) {
      facts.push(`Forecast window interpreted from question: ${primary.transitAt.toISOString()}.`);
    } else if (plan.kind === 'current') {
      facts.push('Current transit intent detected and evaluated at reference time.');
    } else {
      facts.push(`No explicit future/past transit window detected; interpreted as current-context transit at ${primary.transitAt.toISOString()}.`);
    }

    return {
      name: 'Transit analyzer',
      status: 'ok',
      facts: [...new Set(facts)],
      evidencePaths: ['backend:/api/transit-chart'],
    };
  } catch (error) {
    return {
      name: 'Transit analyzer',
      status: 'unavailable',
      facts: [`Failed to call transit endpoint directly: ${String(error)}`],
      evidencePaths: ['backend:/api/transit-chart'],
      missing: ['backend transit response'],
    };
  }
}

export async function buildTransitPointToolFinding(input: TransitToolInput): Promise<ToolFinding> {
  const referenceTimestamp = input.referenceTimestamp ?? Date.now();
  const plan = resolveTransitRequestPlan(input.question, referenceTimestamp);
  const pointPlan: TransitRequestPlan = plan.kind === 'range'
    ? { kind: 'point', samplePoints: [plan.samplePoints[0]] }
    : plan;
  const finding = await buildTransitToolFindingForPlan(input, pointPlan);
  finding.facts.unshift('Transit tool mode: point/current.');
  return finding;
}

export async function buildTransitIntervalToolFinding(input: TransitToolInput): Promise<ToolFinding> {
  const referenceTimestamp = input.referenceTimestamp ?? Date.now();
  const plan = resolveTransitRequestPlan(input.question, referenceTimestamp);

  if (plan.kind === 'range') {
    const finding = await buildTransitToolFindingForPlan(input, plan);
    finding.facts.unshift('Transit tool mode: interval.');
    return finding;
  }

  const singlePoint = plan.samplePoints[0] ?? new Date(referenceTimestamp);
  const derivedRangeEnd = addByUnit(singlePoint, 30, 'day');
  const derivedPlan: TransitRequestPlan = {
    kind: 'range',
    rangeStart: singlePoint,
    rangeEnd: derivedRangeEnd,
    samplePoints: dedupeSamplePoints([singlePoint, getMidpointDate(singlePoint, derivedRangeEnd), derivedRangeEnd]),
  };

  const finding = await buildTransitToolFindingForPlan(input, derivedPlan);
  finding.facts.unshift('Transit tool mode: interval (derived 30-day window).');
  return finding;
}

export async function buildTransitToolFinding(input: TransitToolInput): Promise<ToolFinding> {
  const referenceTimestamp = input.referenceTimestamp ?? Date.now();
  const kind = resolveTransitRequestKind(input.question, referenceTimestamp);
  if (kind === 'range') {
    return buildTransitIntervalToolFinding(input);
  }
  return buildTransitPointToolFinding(input);
}

function getTransitFocusDetails(question: string): { label: string; planets: string[]; houses: string[] } {
  const q = question.toLowerCase();

  if (/\b((?:current\s+)?transit\s+details?|full\s+transit|all\s+transits?|complete\s+transit|transit\s+snapshot|(?:current\s+)?gochar\s+details?)\b/.test(q)) {
    return {
      label: 'current full transit snapshot',
      planets: ['Su', 'Mo', 'Ma', 'Me', 'Ju', 'Ve', 'Sa', 'Ra', 'Ke'],
      houses: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
    };
  }

  if (/\b(marriage|relationship|partner|spouse|love|compatibility|romance|dating)\b/.test(q)) {
    return { label: 'relationship timing', planets: ['Ve', 'Ju', 'Mo', 'Sa'], houses: ['5', '7', '9'] };
  }
  if (/\b(career|job|profession|business|work|promotion)\b/.test(q)) {
    return { label: 'career timing', planets: ['Su', 'Me', 'Sa', 'Ju'], houses: ['2', '6', '10', '11'] };
  }
  if (/\b(wealth|money|income|finance|investment|assets|property)\b/.test(q)) {
    return { label: 'finance timing', planets: ['Ju', 'Ve', 'Me', 'Sa'], houses: ['2', '5', '8', '11'] };
  }
  if (/\b(health|disease|illness|medical|recovery|fitness|longevity|lifespan|life span|ayush|ayu|mrityu|death|end of life)\b/.test(q)) {
    return { label: 'health and longevity timing', planets: ['Sa', 'Ma', 'Ra', 'Ke', 'Mo'], houses: ['1', '6', '8', '12'] };
  }
  if (/\b(children|child|kids|pregnancy|pregnant)\b/.test(q)) {
    return { label: 'children timing', planets: ['Ju', 'Mo', 'Ve'], houses: ['5', '9'] };
  }
  if (/\b(property|house|home|land|vehicle|car|real estate|asset)\b/.test(q)) {
    return { label: 'property timing', planets: ['Mo', 'Sa', 'Ve', 'Ma'], houses: ['2', '4', '11'] };
  }
  if (/\b(travel|foreign|abroad|visa|relocation|migration|move)\b/.test(q)) {
    return { label: 'travel timing', planets: ['Ra', 'Ke', 'Mo', 'Sa'], houses: ['3', '9', '12'] };
  }
  if (/\b(spiritual|spirituality|moksha|meditation|religion|faith|guru)\b/.test(q)) {
    return { label: 'spiritual timing', planets: ['Ke', 'Ju', 'Sa', 'Mo'], houses: ['5', '9', '12'] };
  }
  if (/\b(transit|gochar|today|now|tomorrow|this month|next month|this year|next year)\b/.test(q)) {
    return { label: 'transit timing', planets: ['Mo', 'Sa', 'Ju', 'Su', 'Ve', 'Me'], houses: ['1', '4', '7', '10'] };
  }

  return { label: 'general timing', planets: ['Mo', 'Sa', 'Ra', 'Ju', 'Su'], houses: ['1', '4', '7', '10'] };
}

type TransitIntervalSignal = {
  at: Date;
  tracked: number;
  supportive: number;
  pressure: number;
  neutral: number;
  highlights: string[];
};

const TRANSIT_PRESSURE_HOUSES = new Set<number>([6, 8, 12]);

const TRANSIT_PLANET_LABELS: Record<string, string> = {
  Su: 'Sun',
  Mo: 'Moon',
  Ma: 'Mars',
  Me: 'Mercury',
  Ju: 'Jupiter',
  Ve: 'Venus',
  Sa: 'Saturn',
  Ra: 'Rahu',
  Ke: 'Ketu',
};

const ALL_TRANSIT_PLANET_CODES = ['Su', 'Mo', 'Ma', 'Me', 'Ju', 'Ve', 'Sa', 'Ra', 'Ke'] as const;

const TRANSIT_PLANET_ALIASES: Record<string, string[]> = {
  Su: ['Su', 'Sun', 'sun', 'Surya', 'SURYA'],
  Mo: ['Mo', 'Moon', 'moon', 'Chandra', 'CHANDRA'],
  Ma: ['Ma', 'Mars', 'mars', 'Mangal', 'MANGAL'],
  Me: ['Me', 'Mercury', 'mercury', 'Budh', 'BUDH', 'Budha', 'BUDHA'],
  Ju: ['Ju', 'Jupiter', 'jupiter', 'Guru', 'GURU'],
  Ve: ['Ve', 'Venus', 'venus', 'Shukra', 'SHUKRA'],
  Sa: ['Sa', 'Saturn', 'saturn', 'Shani', 'SHANI'],
  Ra: ['Ra', 'Rahu', 'rahu'],
  Ke: ['Ke', 'Ketu', 'ketu'],
};

function getTransitGrahaPlanet(graha: Record<string, unknown>, code: string): unknown {
  const aliases = TRANSIT_PLANET_ALIASES[code] ?? [code];

  for (const alias of aliases) {
    if (alias in graha) {
      return graha[alias];
    }
  }

  for (const [key, value] of Object.entries(graha)) {
    const keyLower = key.toLowerCase();
    for (const alias of aliases) {
      if (keyLower === alias.toLowerCase()) {
        return value;
      }
    }
  }

  return undefined;
}

function getTransitByPath(value: unknown, path: string): unknown {
  const parts = path.split('.');
  let current: unknown = value;

  for (const part of parts) {
    if (current === null || current === undefined || typeof current !== 'object' || Array.isArray(current)) {
      return undefined;
    }

    current = (current as Record<string, unknown>)[part];
    if (current === undefined) return undefined;
  }

  return current;
}

function formatTransitValue(value: unknown): string {
  if (value === undefined) return 'undefined';
  if (value === null) return 'null';
  if (typeof value === 'string') return value;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function formatTransitPlacement(value: unknown): string {
  if (!isPlainObject(value)) return formatTransitValue(value);

  const parts: string[] = [];
  if ('rashi' in value) parts.push(`rashi ${String(value.rashi)}`);
  if ('degree' in value) parts.push(`${Number(value.degree).toFixed(2)}°`);
  if ('speed' in value) parts.push(`speed ${Number(value.speed).toFixed(4)}`);

  return parts.length > 0 ? parts.join(', ') : formatTransitValue(value);
}

function parseTransitRashiNumber(value: unknown): number | null {
  const n = Number(value);
  if (Number.isInteger(n) && n >= 1 && n <= 12) {
    return n;
  }
  return null;
}

function deriveWholeSignHouseFromLagna(planetRashi: unknown, lagnaRashi: unknown): number | null {
  const p = parseTransitRashiNumber(planetRashi);
  const l = parseTransitRashiNumber(lagnaRashi);
  if (p === null || l === null) return null;
  return ((p - l + 12) % 12) + 1;
}

function formatTransitPlacementRelativeToLagna(value: unknown, lagnaRashi: unknown): string {
  if (!isPlainObject(value)) return formatTransitValue(value);

  const parts: string[] = [];
  if ('rashi' in value) parts.push(`rashi ${String(value.rashi)}`);

  const derivedHouse = deriveWholeSignHouseFromLagna(value.rashi, lagnaRashi);
  if (derivedHouse !== null) {
    parts.push(`house ${derivedHouse} (ascendant-relative)`);
  }

  if ('degree' in value) parts.push(`${Number(value.degree).toFixed(2)}°`);
  if ('speed' in value) parts.push(`speed ${Number(value.speed).toFixed(4)}`);

  return parts.length > 0 ? parts.join(', ') : formatTransitValue(value);
}

function summarizeTransitKeyPlacements(node: unknown, limit = 5): string[] {
  if (!isPlainObject(node)) return [];

  return Object.entries(node)
    .slice(0, limit)
    .map(([key, value]) => `${key}: ${formatTransitPlacement(value)}`);
}

function summarizeFullTransitPlanetSnapshot(transitChart: unknown): string[] {
  if (!isPlainObject(transitChart)) {
    return [];
  }

  const lagna = getTransitByPath(transitChart, 'lagna.Lg') ?? getTransitByPath(transitChart, 'lagna');
  const lagnaRashi = isPlainObject(lagna) ? parseTransitRashiNumber((lagna as Record<string, unknown>).rashi) : null;
  const graha = getTransitByPath(transitChart, 'graha') as Record<string, unknown> | undefined;
  if (!isPlainObject(graha)) {
    return [];
  }

  const facts: string[] = [];
  for (const code of ALL_TRANSIT_PLANET_CODES) {
    const label = TRANSIT_PLANET_LABELS[code] ?? code;
    const planet = getTransitGrahaPlanet(graha as Record<string, unknown>, code);
    if (isPlainObject(planet)) {
      facts.push(`Transit ${label}: ${formatTransitPlacementRelativeToLagna(planet, lagnaRashi)}.`);
    } else {
      facts.push(`Transit ${label}: unavailable in backend payload.`);
    }
  }

  return facts;
}

function summarizeTransitChart(transitChart: unknown, question: string): string[] {
  if (!isPlainObject(transitChart)) {
    return [`Transit chart: ${formatTransitValue(transitChart)}`];
  }

  const facts: string[] = [];
  const focus = getTransitFocusDetails(question);
  const lagna = getTransitByPath(transitChart, 'lagna.Lg') ?? getTransitByPath(transitChart, 'lagna');
  const graha = getTransitByPath(transitChart, 'graha') as Record<string, unknown> | undefined;
  const bhava = getTransitByPath(transitChart, 'bhava') as Record<string, unknown> | undefined;
  const lagnaRashi = isPlainObject(lagna) ? parseTransitRashiNumber((lagna as Record<string, unknown>).rashi) : null;

  facts.push(`Transit focus: ${focus.label}.`);

  if (lagna) {
    facts.push(`Transit Lagna: ${formatTransitPlacement(lagna)}.`);
  }

  if (graha && isPlainObject(graha)) {
    const focusPlanets = new Set(focus.planets);
    let matched = 0;
    let derivedHouseCount = 0;

    for (const code of [...focusPlanets]) {
      const planet = getTransitGrahaPlanet(graha, code);
      if (!isPlainObject(planet)) continue;

      matched += 1;
      const markers: string[] = [];
      if ((planet as Record<string, unknown>).gocharastha === true) markers.push('gocharastha');
      if ((planet as Record<string, unknown>).astangata) markers.push('astangata');
      if ((planet as Record<string, unknown>).vargottama) markers.push('vargottama');
      if (typeof (planet as Record<string, unknown>).speed === 'number' && Number((planet as Record<string, unknown>).speed) < 0) {
        markers.push('retrograde');
      }

      if (deriveWholeSignHouseFromLagna((planet as Record<string, unknown>).rashi, lagnaRashi) !== null) {
        derivedHouseCount += 1;
      }

      facts.push(`Transit ${TRANSIT_PLANET_LABELS[code] ?? code}: ${formatTransitPlacementRelativeToLagna(planet, lagnaRashi)}${markers.length > 0 ? ` (${markers.join(', ')})` : ''}.`);
    }

    if (matched === 0) {
      const preview = summarizeTransitKeyPlacements(graha, 5);
      if (preview.length > 0) {
        facts.push(`Transit graha preview: ${preview.join(' | ')}.`);
      }
    }

    const gocharasthaCount = Object.values(graha).filter((planet) => isPlainObject(planet) && (planet as Record<string, unknown>).gocharastha === true).length;
    if (gocharasthaCount > 0) {
      facts.push(`${gocharasthaCount} transit graha(s) are marked gocharastha.`);
    }

    if (lagnaRashi !== null) {
      facts.push(`Transit Lagna rashi used for whole-sign house derivation: ${lagnaRashi}.`);
      facts.push(`Transit planets with ascendant-relative derived house available: ${derivedHouseCount}.`);
    }
  }

  if (bhava && isPlainObject(bhava)) {
    const focusHouses = new Set(focus.houses);
    const housePreview = Object.entries(bhava)
      .filter(([house]) => focusHouses.has(house))
      .slice(0, 4)
      .map(([house, value]) => `House ${house}: ${formatTransitPlacement(value)}`);

    if (housePreview.length > 0) {
      facts.push(`Transit houses: ${housePreview.join(' | ')}.`);
    } else {
      const preview = summarizeTransitKeyPlacements(bhava, 4);
      if (preview.length > 0) {
        facts.push(`Transit bhava preview: ${preview.join(' | ')}.`);
      }
    }
  }

  return facts;
}

function buildTransitIntervalSignals(
  payloads: Array<{ transitAt: Date; payload: Record<string, unknown> }>,
  focusPlanets: string[],
  focusHouses: Set<number>
): TransitIntervalSignal[] {
  const signals: TransitIntervalSignal[] = [];

  for (const item of payloads) {
    const chart = (item.payload as { chart?: unknown })?.chart;
    const transitChart = (isPlainObject(chart) ? (chart as Record<string, unknown>).transit : undefined) ?? chart;
    if (!isPlainObject(transitChart)) {
      continue;
    }

    const lagna = getTransitByPath(transitChart, 'lagna.Lg') ?? getTransitByPath(transitChart, 'lagna');
    const lagnaRashi = isPlainObject(lagna) ? parseTransitRashiNumber((lagna as Record<string, unknown>).rashi) : null;
    const graha = getTransitByPath(transitChart, 'graha');
    if (!isPlainObject(graha)) {
      continue;
    }

    let tracked = 0;
    let supportive = 0;
    let pressure = 0;
    let neutral = 0;
    const highlights: string[] = [];

    for (const code of focusPlanets) {
      const planet = getTransitGrahaPlanet(graha as Record<string, unknown>, code);
      if (!isPlainObject(planet)) continue;

      const house = deriveWholeSignHouseFromLagna((planet as Record<string, unknown>).rashi, lagnaRashi);
      if (house === null) continue;

      tracked += 1;
      const label = TRANSIT_PLANET_LABELS[code] ?? code;

      if (focusHouses.has(house)) {
        supportive += 1;
        highlights.push(`${label}→H${house} supportive`);
        continue;
      }

      if (TRANSIT_PRESSURE_HOUSES.has(house)) {
        pressure += 1;
        highlights.push(`${label}→H${house} pressure`);
        continue;
      }

      neutral += 1;
    }

    if (tracked > 0) {
      signals.push({
        at: item.transitAt,
        tracked,
        supportive,
        pressure,
        neutral,
        highlights: highlights.slice(0, 4),
      });
    }
  }

  return signals.sort((a, b) => a.at.getTime() - b.at.getTime());
}

function summarizeTransitIntervalSignals(signals: TransitIntervalSignal[], focusLabel: string): string[] {
  if (signals.length === 0) {
    return [];
  }

  const facts: string[] = [];
  const net = (signal: TransitIntervalSignal): number => signal.supportive - signal.pressure;

  if (signals.length >= 2) {
    const first = signals[0];
    const last = signals[signals.length - 1];
    const delta = net(last) - net(first);
    const trajectory = delta >= 2
      ? 'improving'
      : delta <= -2
        ? 'tightening'
        : 'mixed/stable';

    facts.push(
      `Transit interval trajectory (${focusLabel}): start net=${net(first)} and end net=${net(last)} (${trajectory}).`
    );
  }

  const supportiveWindow = [...signals].sort((a, b) => {
    const netDiff = net(b) - net(a);
    if (netDiff !== 0) return netDiff;
    return b.supportive - a.supportive;
  })[0];

  const pressureWindow = [...signals].sort((a, b) => {
    const netDiff = net(a) - net(b);
    if (netDiff !== 0) return netDiff;
    return b.pressure - a.pressure;
  })[0];

  if (supportiveWindow) {
    facts.push(
      `Potentially supportive window near ${supportiveWindow.at.toISOString()} (supportive=${supportiveWindow.supportive}, pressure=${supportiveWindow.pressure}${supportiveWindow.highlights.length > 0 ? `; ${supportiveWindow.highlights.join(', ')}` : ''}).`
    );
  }

  if (pressureWindow && pressureWindow.at.getTime() !== supportiveWindow?.at.getTime()) {
    facts.push(
      `Potential pressure window near ${pressureWindow.at.toISOString()} (supportive=${pressureWindow.supportive}, pressure=${pressureWindow.pressure}${pressureWindow.highlights.length > 0 ? `; ${pressureWindow.highlights.join(', ')}` : ''}).`
    );
  }

  return facts;
}
