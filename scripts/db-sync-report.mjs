import 'dotenv/config';
import { Pool } from 'pg';
import { buildPostgresSslConfig, withPostgresSslOverrides } from '../dist/services/postgresSsl.js';

async function main() {
  if (!process.env.DATABASE_URL) {
    throw new Error('DATABASE_URL is missing');
  }

  const pool = new Pool({
    connectionString: withPostgresSslOverrides(process.env.DATABASE_URL),
    ...buildPostgresSslConfig(),
  });

  try {
    const mig = await pool.query('SELECT name, applied_at FROM schema_migrations ORDER BY name');
    const tables = await pool.query(
      "SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_type='BASE TABLE' ORDER BY table_name"
    );
    const vectors = await pool.query(
      "SELECT table_name, column_name, udt_name FROM information_schema.columns WHERE table_schema='public' AND udt_name='vector' ORDER BY table_name, column_name"
    );

    const counts = await pool.query(`
      SELECT
        (SELECT COUNT(*)::int FROM auth_users) AS auth_users,
        (SELECT COUNT(*)::int FROM subscriptions) AS subscriptions,
        (SELECT COUNT(*)::int FROM chat_sessions) AS chat_sessions,
        (SELECT COUNT(*)::int FROM chat_messages) AS chat_messages,
        (SELECT COUNT(*)::int FROM rag_profiles) AS rag_profiles,
        (SELECT COUNT(*)::int FROM rag_api_sources) AS rag_api_sources,
        (SELECT COUNT(*)::int FROM rag_chunks) AS rag_chunks,
        (SELECT COUNT(*)::int FROM chart_jobs) AS chart_jobs,
        (SELECT COUNT(*)::int FROM charts) AS charts,
        (SELECT COUNT(*)::int FROM chart_vectors) AS chart_vectors,
        (SELECT COUNT(*)::int FROM documents) AS documents
    `);

    console.log(
      JSON.stringify(
        {
          migrationCount: mig.rowCount,
          migrations: mig.rows,
          tables: tables.rows.map((r) => r.table_name),
          vectorColumns: vectors.rows,
          rowCounts: counts.rows[0],
        },
        null,
        2
      )
    );
  } finally {
    await pool.end();
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
