import dotenv from 'dotenv';
import { Pool } from 'pg';
import { buildPostgresSslConfig, withPostgresSslOverrides } from '../src/services/postgresSsl.js';
import { resetAndReapplyMigrations } from '../src/services/postgresMigrations.js';

dotenv.config();

function isTruthy(value: string | undefined): boolean {
  return (value ?? '').trim().toLowerCase() === 'true';
}

async function main() {
  const databaseUrl = process.env.DATABASE_URL?.trim();
  if (!databaseUrl) {
    throw new Error('DATABASE_URL is required');
  }

  const embeddingDim = Number(process.env.EMBEDDING_DIM ?? 192);
  const enablePgVector = isTruthy(process.env.PGVECTOR_ENABLED);

  const pool = new Pool({
    connectionString: withPostgresSslOverrides(databaseUrl),
    ...buildPostgresSslConfig(),
  });

  try {
    await resetAndReapplyMigrations(pool);
    console.log('✅ Database reset + schema push completed.');
    console.log(`   - pgvector enabled: ${enablePgVector}`);
    console.log(`   - embedding dim: ${embeddingDim}`);
  } finally {
    await pool.end();
  }
}

main().catch((error) => {
  console.error('❌ Failed to reset/push schema:', error);
  process.exit(1);
});