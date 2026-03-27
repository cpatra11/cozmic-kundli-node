import { Pool } from 'pg';
import { env } from '../config/env.js';
import { buildPostgresSslConfig, withPostgresSslOverrides } from './postgresSsl.js';

let singletonPool: Pool | null = null;

export function getPostgresPool(): Pool | null {
  if (!env.DATABASE_URL?.trim()) {
    return null;
  }

  if (!singletonPool) {
    singletonPool = new Pool({
      connectionString: withPostgresSslOverrides(env.DATABASE_URL),
      ...buildPostgresSslConfig(),
    });
  }

  return singletonPool;
}