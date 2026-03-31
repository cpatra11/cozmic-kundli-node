import dotenv from 'dotenv';
import { Pool } from 'pg';
import { buildPostgresSslConfig, withPostgresSslOverrides } from '../src/services/postgresSsl.js';
import { applyPendingMigrations } from '../src/services/postgresMigrations.js';

dotenv.config();

async function main() {
  const databaseUrl = process.env.DATABASE_URL?.trim();
  if (!databaseUrl) {
    throw new Error('DATABASE_URL is required');
  }

  const pool = new Pool({
    connectionString: withPostgresSslOverrides(databaseUrl),
    ...buildPostgresSslConfig(),
  });

  try {
    await applyPendingMigrations(pool);
    console.log('✅ Database migrations applied successfully.');
  } finally {
    await pool.end();
  }
}

main().catch((error) => {
  console.error('❌ Failed to apply migrations:', error);
  process.exit(1);
});
