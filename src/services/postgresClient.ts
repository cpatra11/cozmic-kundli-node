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
      query_timeout: env.PG_QUERY_TIMEOUT_MS,
      connectionTimeoutMillis: env.PG_CONNECTION_TIMEOUT_MS,
      idleTimeoutMillis: env.PG_IDLE_TIMEOUT_MS,
      keepAlive: true,
      keepAliveInitialDelayMillis: env.PG_KEEPALIVE_INITIAL_DELAY_MS,
      ...buildPostgresSslConfig(),
    });

    singletonPool.on('error', (error) => {
      console.error('[postgres] idle client error (non-fatal)', {
        message: error.message,
        code: (error as NodeJS.ErrnoException).code,
      });
    });
  }

  return singletonPool;
}