import { readdir, readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Pool } from 'pg';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const MIGRATIONS_DIR = path.resolve(__dirname, '../db/migrations');

const readyByPool = new WeakMap<Pool, Promise<void>>();

interface MigrationFile {
  name: string;
  sql: string;
}

async function listMigrationFiles(): Promise<MigrationFile[]> {
  const entries = await readdir(MIGRATIONS_DIR, { withFileTypes: true });
  const files = entries
    .filter((entry) => entry.isFile() && entry.name.endsWith('.sql'))
    .map((entry) => entry.name)
    .sort((a, b) => a.localeCompare(b));

  const resolved = await Promise.all(
    files.map(async (name) => {
      const sqlPath = path.join(MIGRATIONS_DIR, name);
      const sql = await readFile(sqlPath, 'utf8');
      return { name, sql };
    })
  );

  return resolved;
}

async function ensureMigrationsTable(pool: Pool): Promise<void> {
  await pool.query(`
    CREATE TABLE IF NOT EXISTS schema_migrations (
      name TEXT PRIMARY KEY,
      applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
  `);
}

export async function applyPendingMigrations(pool: Pool): Promise<void> {
  const existing = readyByPool.get(pool);
  if (existing) {
    await existing;
    return;
  }

  const runPromise = (async () => {
    await ensureMigrationsTable(pool);
    const migrationFiles = await listMigrationFiles();

    for (const migration of migrationFiles) {
      const alreadyApplied = await pool.query<{ name: string }>(
        `SELECT name FROM schema_migrations WHERE name = $1 LIMIT 1`,
        [migration.name]
      );

      if (alreadyApplied.rows[0]) {
        continue;
      }

      const client = await pool.connect();
      try {
        await client.query('BEGIN');
        await client.query(migration.sql);
        await client.query(`INSERT INTO schema_migrations (name) VALUES ($1)`, [migration.name]);
        await client.query('COMMIT');
      } catch (error) {
        await client.query('ROLLBACK').catch(() => {});
        throw error;
      } finally {
        client.release();
      }
    }
  })();

  readyByPool.set(pool, runPromise);

  try {
    await runPromise;
  } catch (error) {
    readyByPool.delete(pool);
    throw error;
  }
}

export async function resetAndReapplyMigrations(pool: Pool): Promise<void> {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');
    await client.query(`DROP TABLE IF EXISTS chart_jobs CASCADE;`);
    await client.query(`DROP TABLE IF EXISTS rag_chunks CASCADE;`);
    await client.query(`DROP TABLE IF EXISTS rag_api_sources CASCADE;`);
    await client.query(`DROP TABLE IF EXISTS rag_profiles CASCADE;`);
    await client.query(`DROP TABLE IF EXISTS chat_messages CASCADE;`);
    await client.query(`DROP TABLE IF EXISTS chat_sessions CASCADE;`);
    await client.query(`DROP TABLE IF EXISTS chart_vectors CASCADE;`);
    await client.query(`DROP TABLE IF EXISTS charts CASCADE;`);
    await client.query(`DROP TABLE IF EXISTS documents CASCADE;`);
    await client.query(`DROP TABLE IF EXISTS auth_users CASCADE;`);
    await client.query(`DROP TABLE IF EXISTS subscriptions CASCADE;`);
    await client.query(`DROP TABLE IF EXISTS schema_migrations CASCADE;`);
    await client.query('COMMIT');
  } catch (error) {
    await client.query('ROLLBACK').catch(() => {});
    throw error;
  } finally {
    client.release();
  }

  readyByPool.delete(pool);
  await applyPendingMigrations(pool);
}