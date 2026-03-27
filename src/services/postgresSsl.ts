import { readFileSync } from 'node:fs';
import { env } from '../config/env.js';

function parseBoolean(value: string | undefined): boolean | undefined {
  if (value === undefined) return undefined;

  const normalized = value.trim().toLowerCase();
  if (!normalized) return undefined;
  if (['1', 'true', 'yes', 'on'].includes(normalized)) return true;
  if (['0', 'false', 'no', 'off'].includes(normalized)) return false;

  throw new Error(
    `DATABASE_SSL_REJECT_UNAUTHORIZED must be a boolean-like value (true/false/1/0), got: ${value}`
  );
}

export function buildPostgresSslConfig(): { ssl?: { ca?: string; rejectUnauthorized?: boolean } } {
  const ssl: { ca?: string; rejectUnauthorized?: boolean } = {};

  const caPath = env.DATABASE_SSL_CA_PATH?.trim();
  if (caPath) {
    ssl.ca = readFileSync(caPath, 'utf8');
  }

  const rejectUnauthorized = parseBoolean(env.DATABASE_SSL_REJECT_UNAUTHORIZED);
  if (typeof rejectUnauthorized === 'boolean') {
    ssl.rejectUnauthorized = rejectUnauthorized;
  }

  return Object.keys(ssl).length > 0 ? { ssl } : {};
}

export function withPostgresSslOverrides(connectionString: string): string {
  const hasSslOverride =
    Boolean(env.DATABASE_SSL_CA_PATH?.trim()) || env.DATABASE_SSL_REJECT_UNAUTHORIZED !== undefined;

  if (!hasSslOverride) {
    return connectionString;
  }

  try {
    const parsed = new URL(connectionString);
    parsed.searchParams.delete('sslmode');
    parsed.searchParams.delete('sslrootcert');
    parsed.searchParams.delete('sslcert');
    parsed.searchParams.delete('sslkey');
    return parsed.toString();
  } catch {
    return connectionString;
  }
}