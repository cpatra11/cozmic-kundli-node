function isPlainObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function coerceDegreeFromLongitude(longitude: unknown): number | null {
  const numericLongitude = Number(longitude);
  if (!Number.isFinite(numericLongitude)) return null;
  const degree = numericLongitude % 30;
  return degree >= 0 ? degree : degree + 30;
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

  return next as T;
}

export function buildChartSnapshot<T>(payload: T): T {
  return hydrateDegrees(payload);
}