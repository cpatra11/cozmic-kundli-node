import { env } from '../config/env.js';

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

function buildBe1Url(path: string, params: URLSearchParams): string {
  const normalizedBase = env.BE1_BASE_URL.replace(/\/+$/, '');
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
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`be1 /${path} failed (${response.status})`);
  }

  return response.json();
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
    time_zone: input.time_zone,
    varga: 'D1',
    infolevel: 'basic',
  });
}
