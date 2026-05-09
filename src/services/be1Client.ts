import { env } from '../config/env.js';

let be1FailureCount = 0;
let be1CircuitOpenedAt = 0;

/**
 * Convert decimal timezone offset (e.g. '5.5') to ±HH:MM format (e.g. '+05:30').
 * The BE1 PHP calculator expects offsets like '+05:30', not decimal '5.5'.
 */
function toTimezoneOffset(tz: string): string {
  // Already in ±HH:MM format — return as-is (e.g. '+05:30')
  if (/^[+-]\d{2}:\d{2}$/.test(tz)) return tz;
  // Named timezone like 'Asia/Tehran' — pass through
  const num = parseFloat(tz);
  if (isNaN(num)) return tz;
  const sign = num >= 0 ? '+' : '-';
  const abs = Math.abs(num);
  const h = Math.floor(abs);
  const m = Math.round((abs - h) * 60);
  return `${sign}${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`;
}

export interface KundliSnapshotInput {
  latitude: number;
  longitude: number;
  year: number;
  month: number;
  day: number;
  hour: number;
  min: number;
  sec?: number;
  time_zone: string;
}

function getInternalHeaders(): Record<string, string> {
  if (!env.BE1_CONFIG.internal_api_key) {
    return {};
  }
  return {
    'X-Internal-Api-Key': env.BE1_CONFIG.internal_api_key,
  };
}

function isBe1CircuitOpen(now = Date.now()): boolean {
  if (be1CircuitOpenedAt <= 0) return false;
  const elapsed = now - be1CircuitOpenedAt;
  if (elapsed >= env.BE1_CONFIG.circuit_cooldown_ms) {
    be1CircuitOpenedAt = 0;
    be1FailureCount = 0;
    return false;
  }
  return true;
}

function markBe1Success() {
  be1FailureCount = 0;
  be1CircuitOpenedAt = 0;
}

function markBe1Failure() {
  be1FailureCount += 1;
  if (be1FailureCount >= env.BE1_CONFIG.circuit_fail_threshold) {
    be1CircuitOpenedAt = Date.now();
  }
}

async function fetchWithTimeout(url: string, init: RequestInit): Promise<Response> {
  if (isBe1CircuitOpen()) {
    throw new Error(`be1 circuit open; retry after ${env.BE1_CONFIG.circuit_cooldown_ms}ms cooldown`);
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), env.BE1_CONFIG.request_timeout_ms);
  try {
    const response = await fetch(url, {
      ...init,
      signal: controller.signal,
    });
    if (!response.ok) {
      markBe1Failure();
    } else {
      markBe1Success();
    }
    return response;
  } catch (error) {
    markBe1Failure();
    if (error instanceof Error && error.name === 'AbortError') {
      throw new Error(`be1 request timed out after ${env.BE1_CONFIG.request_timeout_ms}ms`);
    }
    throw error;
  } finally {
    clearTimeout(timeoutId);
  }
}

function buildBe1Url(path: string, params: URLSearchParams): string {
  const normalizedBase = env.BE1_CONFIG.base_url.replace(/\/+$/, '');
  const normalizedPath = path.replace(/^\/+/, '');
  return `${normalizedBase}/${normalizedPath}?${params.toString()}`;
}

export async function fetchBe1Json(path: string, query: Record<string, string | number>) {
  const params = new URLSearchParams(
    Object.entries(query).reduce(
      (acc, [key, value]) => {
        acc[key] = String(value);
        return acc;
      },
      {} as Record<string, string>
    )
  );

  const url = buildBe1Url(path, params);
  const response = await fetchWithTimeout(url, {
    headers: getInternalHeaders(),
  });

  if (!response.ok) {
    throw new Error(`be1 /${path} failed (${response.status})`);
  }

  return response.json();
}

export async function postBe1Json(path: string, body: Record<string, unknown>) {
  const normalizedBase = env.BE1_CONFIG.base_url.replace(/\/+$/, '');
  const normalizedPath = path.replace(/^\/+/, '');
  const url = `${normalizedBase}/${normalizedPath}`;
  const response = await fetchWithTimeout(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...getInternalHeaders(),
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const details = await response.text();
    throw new Error(`be1 /${path} failed (${response.status}): ${details}`);
  }

  return response.json();
}

export interface CalculateChartOptions {
  nesting?: number;
  periodKey?: string;
  infolevel?: string;
  varga?: string;
  ayanamsha?: string;
  nodeType?: 'mean' | 'true';
  dstHour?: number;
  dstMin?: number;
}

export async function fetchCalculatedChart(input: KundliSnapshotInput, options: CalculateChartOptions = {}) {
  return fetchBe1Json('calculate', {
    latitude: input.latitude,
    longitude: input.longitude,
    year: input.year,
    month: input.month,
    day: input.day,
    hour: input.hour,
    min: input.min,
    sec: input.sec ?? 0,
    time_zone: toTimezoneOffset(input.time_zone),
    dst_hour: options.dstHour ?? 0,
    dst_min: options.dstMin ?? 0,
    nesting: options.nesting ?? 1,
    ...(options.periodKey ? { period_key: options.periodKey } : {}),
    infolevel:
      options.infolevel ?? 'basic,ashtakavarga,grahabala,rashibala,yogas,panchanga,dasha,ayanamsa,arudha',
    varga: options.varga ?? 'D1,D2,D3,D4,D6,D7,D9,D10,D12,D16,D20,D24,D27,D30,D40,D45,D60',
    ...(options.ayanamsha ? { ayanamsha: options.ayanamsha } : {}),
    ...(options.nodeType ? { node_type: options.nodeType } : {}),
  });
}

export async function fetchKundliSnapshot(input: KundliSnapshotInput) {
  return fetchBe1Json('calculate', {
    latitude: input.latitude,
    longitude: input.longitude,
    year: input.year,
    month: input.month,
    day: input.day,
    hour: input.hour,
    min: input.min,
    sec: input.sec ?? 0,
    time_zone: toTimezoneOffset(input.time_zone),
    varga: 'D1',
    infolevel: 'basic',
  });
}

export async function fetchTransitChart(
  input: KundliSnapshotInput,
  options?: {
    transitAt?: Date;
    nesting?: number;
  }
) {
  const transitAt = options?.transitAt ?? new Date();

  return postBe1Json('transit-chart', {
    latitude: input.latitude,
    longitude: input.longitude,
    time_zone: toTimezoneOffset(input.time_zone),
    year: input.year,
    month: input.month,
    day: input.day,
    hour: input.hour,
    min: input.min,
    sec: input.sec ?? 0,
    t_year: transitAt.getUTCFullYear(),
    t_month: transitAt.getUTCMonth() + 1,
    t_day: transitAt.getUTCDate(),
    t_hour: transitAt.getUTCHours(),
    t_min: transitAt.getUTCMinutes(),
    t_sec: transitAt.getUTCSeconds(),
    nesting: options?.nesting ?? 4,
  });
}

export interface Be1CalculateOptions {
  varga?: string;
  infolevel?: string;
  nesting?: number;
  ayanamsha?: string;
  nodeType?: 'mean' | 'true';
  dstHour?: number;
  dstMin?: number;
  periodKey?: string;
}

const DEFAULT_VARGA = 'D1,D9,D10';
const DEFAULT_INFOLEVEL = 'basic,dasha,yogas';

export async function fetchBe1Calculate(
  input: KundliSnapshotInput,
  options: Be1CalculateOptions = {}
): Promise<unknown> {
  return fetchBe1Json('calculate', {
    latitude: input.latitude,
    longitude: input.longitude,
    year: input.year,
    month: input.month,
    day: input.day,
    hour: input.hour,
    min: input.min,
    sec: input.sec ?? 0,
    time_zone: toTimezoneOffset(input.time_zone),
    dst_hour: options.dstHour ?? 0,
    dst_min: options.dstMin ?? 0,
    nesting: options.nesting ?? 1,
    ...(options.periodKey ? { period_key: options.periodKey } : {}),
    infolevel: options.infolevel ?? DEFAULT_INFOLEVEL,
    varga: options.varga ?? DEFAULT_VARGA,
    ...(options.ayanamsha ? { ayanamsha: options.ayanamsha } : {}),
    ...(options.nodeType ? { node_type: options.nodeType } : {}),
  });
}

export interface Be1TransitOptions {
  transitAt?: Date;
  nesting?: number;
  ayanamsha?: string;
}

export async function fetchBe1Transit(
  input: KundliSnapshotInput,
  transitAt: Date = new Date(),
  options?: Be1TransitOptions
): Promise<unknown> {
  return postBe1Json('transit-chart', {
    latitude: input.latitude,
    longitude: input.longitude,
    time_zone: toTimezoneOffset(input.time_zone),
    year: input.year,
    month: input.month,
    day: input.day,
    hour: input.hour,
    min: input.min,
    sec: input.sec ?? 0,
    t_year: transitAt.getUTCFullYear(),
    t_month: transitAt.getUTCMonth() + 1,
    t_day: transitAt.getUTCDate(),
    t_hour: transitAt.getUTCHours(),
    t_min: transitAt.getUTCMinutes(),
    t_sec: transitAt.getUTCSeconds(),
    nesting: options?.nesting ?? 4,
    ...(options?.ayanamsha ? { ayanamsha: options.ayanamsha } : {}),
  });
}
