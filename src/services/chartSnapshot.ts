function isPlainObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

export type ChartSchemaVersion = 'mahadasha-first' | 'legacy-deep-dasha';

export interface ChartSchemaInfo {
  chartSchemaVersion: ChartSchemaVersion;
  dashaDepth: number;
  dashaPeriodKey?: string;
}

const NAKSHATRA_LORD_CYCLE = ['Ketu', 'Venus', 'Sun', 'Moon', 'Mars', 'Rahu', 'Jupiter', 'Saturn', 'Mercury'];

const NAKSHATRA_NAME_TO_NUMBER: Record<string, number> = {
  ashwini: 1,
  ashvini: 1,
  bharani: 2,
  krittika: 3,
  rohini: 4,
  mrigashira: 5,
  mrigasira: 5,
  ardra: 6,
  punarvasu: 7,
  pushya: 8,
  pusya: 8,
  aslesha: 9,
  ashlesha: 9,
  magha: 10,
  purvaphalguni: 11,
  purvaphalghuni: 11,
  pubba: 11,
  uttaraphalguni: 12,
  hasta: 13,
  chitra: 14,
  swati: 15,
  vishakha: 16,
  visakha: 16,
  anuradha: 17,
  jyeshtha: 18,
  jyestha: 18,
  mula: 19,
  moola: 19,
  purvashadha: 20,
  purvaashadha: 20,
  uttarashadha: 21,
  uttaraashadha: 21,
  shravana: 22,
  sravana: 22,
  dhanishtha: 23,
  dhanishta: 23,
  shatabhisha: 24,
  satabhisha: 24,
  purvabhadrapada: 25,
  purvabhadra: 25,
  uttarabhadrapada: 26,
  uttarabhadra: 26,
  revati: 27,
};

function normalizeToken(value: unknown): string {
  return String(value ?? '')
    .toLowerCase()
    .replace(/[^a-z]/g, '');
}

function getString(value: unknown): string | null {
  if (typeof value !== 'string') return null;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function getByPath(value: unknown, path: string): unknown {
  if (!isPlainObject(value)) return undefined;

  return path.split('.').reduce<unknown>((current, segment) => {
    if (!isPlainObject(current)) return undefined;
    return current[segment];
  }, value);
}

function normalizedLongitude(longitude: unknown): number | null {
  const n = Number(longitude);
  if (!Number.isFinite(n)) return null;
  return ((n % 360) + 360) % 360;
}

function deriveNakshatraNumber(entry: Record<string, unknown>): number | null {
  const directCandidates = [
    entry.nakshatra_no,
    entry.nakshatraNumber,
    entry.nak_no,
    isPlainObject(entry.nakshatra) ? entry.nakshatra.number : undefined,
    isPlainObject(entry.nakshatra) ? entry.nakshatra.nakshatra_no : undefined,
  ];

  for (const candidate of directCandidates) {
    const n = Number(candidate);
    if (Number.isFinite(n) && n >= 1 && n <= 27) {
      return Math.floor(n);
    }
  }

  const nakName = getString(entry.nakshatra_name)
    ?? getString(entry.nakshatraName)
    ?? getString(entry.nakshatra)
    ?? (isPlainObject(entry.nakshatra) ? getString(entry.nakshatra.name) : null);
  if (nakName) {
    const mapped = NAKSHATRA_NAME_TO_NUMBER[normalizeToken(nakName)];
    if (mapped) return mapped;
  }

  const longCandidates = [
    entry.global_degree,
    entry.longitude,
    entry.full_longitude,
    entry.lng,
  ];
  for (const candidate of longCandidates) {
    const normalized = normalizedLongitude(candidate);
    if (normalized === null) continue;
    return Math.floor(normalized / (360 / 27)) + 1;
  }

  return null;
}

function deriveNakshatraLordName(entry: Record<string, unknown>): string | null {
  const explicitLord =
    getString(entry.nakshatra_lord)
    ?? getString(entry.nakshatraLord)
    ?? getString(entry.star_lord)
    ?? getString(entry.nak_lord)
    ?? getString(entry.lord_of_nakshatra)
    ?? (isPlainObject(entry.nakshatra)
      ? getString(entry.nakshatra.lord) ?? getString(entry.nakshatra.nakshatra_lord)
      : null);
  if (explicitLord) return explicitLord;

  const nakNumber = deriveNakshatraNumber(entry);
  if (!nakNumber) return null;
  return NAKSHATRA_LORD_CYCLE[(nakNumber - 1) % NAKSHATRA_LORD_CYCLE.length] ?? null;
}

function coerceDegreeFromLongitude(longitude: unknown): number | null {
  const numericLongitude = Number(longitude);
  if (!Number.isFinite(numericLongitude)) return null;
  const degree = numericLongitude % 30;
  return degree >= 0 ? degree : degree + 30;
}

function measureDashaDepth(node: unknown): number {
  if (!isPlainObject(node)) return 0;

  const periods = node.periods;
  if (!isPlainObject(periods)) return 0;

  let deepestChild = 0;
  for (const child of Object.values(periods)) {
    const childDepth = measureDashaDepth(child);
    if (childDepth > deepestChild) {
      deepestChild = childDepth;
    }
  }

  return deepestChild + 1;
}

function isDashaNode(node: unknown): node is Record<string, unknown> {
  return isPlainObject(node)
    && ('periods' in node || 'start' in node || 'end' in node || 'nesting' in node || 'type' in node || 'key' in node);
}

function stripNestedDashaPeriods(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map((item) => stripNestedDashaPeriods(item));
  }

  if (!isPlainObject(value)) {
    return value;
  }

  const next: Record<string, unknown> = {};
  for (const [key, entry] of Object.entries(value)) {
    if (key === 'periods' && isPlainObject(entry)) {
      next.periods = Object.fromEntries(
        Object.entries(entry).map(([periodKey, periodValue]) => [periodKey, stripDashaBranch(periodValue)])
      );
      continue;
    }

    next[key] = stripNestedDashaPeriods(entry);
  }

  return next;
}

function stripDashaBranch(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map((item) => stripNestedDashaPeriods(item));
  }

  if (!isDashaNode(value)) {
    return stripNestedDashaPeriods(value);
  }

  const next: Record<string, unknown> = {};
  for (const [key, entry] of Object.entries(value)) {
    if (key === 'periods') {
      continue;
    }

    next[key] = stripNestedDashaPeriods(entry);
  }

  return next;
}

export function extractChartSchemaInfo(payload: unknown): ChartSchemaInfo {
  const dasha = getByPath(payload, 'chart.dasha') ?? getByPath(payload, 'dasha');
  const dashaDepth = Math.max(1, measureDashaDepth(dasha));
  const periodKey = getString(getByPath(dasha, 'period_key'))
    ?? getString(getByPath(dasha, 'periodKey'))
    ?? getString(getByPath(dasha, 'key'))
    ?? undefined;

  return {
    chartSchemaVersion: dashaDepth > 1 ? 'legacy-deep-dasha' : 'mahadasha-first',
    dashaDepth,
    dashaPeriodKey: periodKey,
  };
}

function hydrateDegrees<T>(value: T, seen = new WeakMap<object, unknown>()): T {
  if (!isPlainObject(value) && !Array.isArray(value)) {
    return value;
  }

  if (typeof value === 'object' && value !== null) {
    const cached = seen.get(value as object);
    if (cached) return cached as T;
  }

  if (Array.isArray(value)) {
    const nextArray = value.map((item) => hydrateDegrees(item, seen)) as T;
    seen.set(value, nextArray);
    return nextArray;
  }

  const source = value as Record<string, unknown>;
  const next: Record<string, unknown> = {};
  seen.set(source, next);

  for (const [key, entry] of Object.entries(source)) {
    next[key] = hydrateDegrees(entry, seen);
  }

  const existingDegree = Number(next.degree);
  if (!Number.isFinite(existingDegree)) {
    const derivedDegree = coerceDegreeFromLongitude(next.longitude);
    if (derivedDegree !== null) {
      next.degree = derivedDegree;
    }
  }

  if (!Number.isFinite(Number(next.normDegree)) && Number.isFinite(Number(next.degree))) {
    next.normDegree = Number(next.degree);
  }

  const derivedNakLord = deriveNakshatraLordName(next);
  if (derivedNakLord) {
    if (!getString(next.nakshatra_lord)) {
      next.nakshatra_lord = derivedNakLord;
    }
    if (!getString(next.nakshatraLord)) {
      next.nakshatraLord = derivedNakLord;
    }

    if (isPlainObject(next.nakshatra)) {
      if (!getString(next.nakshatra.lord)) {
        next.nakshatra.lord = derivedNakLord;
      }
      if (!getString(next.nakshatra.nakshatra_lord)) {
        next.nakshatra.nakshatra_lord = derivedNakLord;
      }
    }
  }

  return next as T;
}

export function buildChartSnapshot<T>(payload: T): T {
  return stripNestedDashaPeriods(hydrateDegrees(payload)) as T;
}