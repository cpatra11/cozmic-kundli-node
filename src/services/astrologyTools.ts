import { type KundliSnapshotInput, fetchTransitChart } from './be1Client.js';

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

function normalizeQuestion(question: string): string {
  return question.toLowerCase().replace(/\s+/g, ' ').trim();
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

function resolveExplicitDate(question: string): Date | null {
  const isoLike = question.match(/\b(20\d{2}-\d{2}-\d{2}(?:[t\s]\d{2}:\d{2}(?::\d{2})?)?(?:z|[+-]\d{2}:?\d{2})?)\b/i);
  if (isoLike?.[1]) {
    const parsed = new Date(isoLike[1]);
    if (!Number.isNaN(parsed.getTime())) return parsed;
  }

  const slashDate = question.match(/\b(\d{1,2})[\/.-](\d{1,2})[\/.-](20\d{2})\b/);
  if (slashDate) {
    const [, day, month, year] = slashDate;
    const parsed = new Date(Number(year), Number(month) - 1, Number(day));
    if (!Number.isNaN(parsed.getTime())) return parsed;
  }

  return null;
}

export function resolveTransitAt(question: string, referenceTimestamp: number): Date | null {
  const normalized = normalizeQuestion(question);
  const base = new Date(Number.isFinite(referenceTimestamp) ? referenceTimestamp : Date.now());

  const explicit = resolveExplicitDate(question);
  if (explicit) return explicit;

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

export async function buildTransitToolFinding(input: TransitToolInput): Promise<ToolFinding> {
  const referenceTimestamp = input.referenceTimestamp ?? Date.now();
  const requestedTransitAt = resolveTransitAt(input.question, referenceTimestamp) ?? new Date(referenceTimestamp);
  const transitAt = requestedTransitAt;

  try {
    const transitPayload = await fetchTransitChart(input.kundli, { transitAt });
    const blocks = extractTransitBlocks(transitPayload);
    const chart = (transitPayload as { chart?: any })?.chart ?? transitPayload;
    const currentSun = chart?.transit?.Su ?? chart?.transit?.Sun ?? chart?.transit?.sun ?? chart?.Su ?? chart?.Sun;

    const facts: string[] = [
      `Transit chart generated directly from backend endpoint for ${transitAt.toISOString()}.`,
      blocks.length > 0
        ? `Transit/gochar block(s) detected: ${blocks.length}.`
        : 'Transit response was returned, but no dedicated transit block was found in the payload shape.',
    ];

    if (currentSun && typeof currentSun === 'object') {
      const sun = currentSun as Record<string, unknown>;
      const rashi = sun.rashi ?? sun.sign ?? 'n/a';
      const degree = sun.degree !== undefined ? Number(sun.degree).toFixed(2) : null;
      const house = sun.house_number !== undefined ? String(sun.house_number) : null;
      facts.push(`Transit Sun: sign ${String(rashi)}${house ? `, house ${house}` : ''}${degree ? `, ${degree}°` : ''}.`);
    }

    if (/\b(tomorrow|next week|next month|next year|in\s+\d+\s+(day|days|week|weeks|month|months|year|years))\b/i.test(input.question)) {
      facts.push(`Forecast window interpreted from question: ${transitAt.toISOString()}.`);
    }

    return {
      name: 'Transit calculator',
      status: 'ok',
      facts,
      evidencePaths: ['backend:/v1/transit-chart'],
      snippets: [JSON.stringify(transitPayload, null, 2).slice(0, 2000)],
    };
  } catch (error) {
    return {
      name: 'Transit calculator',
      status: 'unavailable',
      facts: [`Failed to call transit endpoint directly: ${String(error)}`],
      evidencePaths: ['backend:/v1/transit-chart'],
      missing: ['backend transit response'],
    };
  }
}
