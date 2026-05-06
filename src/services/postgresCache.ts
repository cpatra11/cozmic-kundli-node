import { Pool } from 'pg';
import { BaseCache, type CacheFullKey, type CacheNamespace } from '@langchain/langgraph-checkpoint';

const CREATE_TABLE_SQL = `
CREATE TABLE IF NOT EXISTS langgraph_node_cache (
  namespace TEXT[] NOT NULL,
  key TEXT NOT NULL,
  value BYTEA NOT NULL,
  encoding TEXT NOT NULL DEFAULT 'json',
  expires_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  PRIMARY KEY (namespace, key)
);
`;

const CREATE_INDEX_SQL = `
CREATE INDEX IF NOT EXISTS idx_langgraph_cache_expiry
  ON langgraph_node_cache(expires_at)
  WHERE expires_at IS NOT NULL;
`;

export class PostgresCache<V = unknown> extends BaseCache<V> {
  private pool: Pool;
  private tableCreated = false;

  constructor(pool: Pool) {
    super();
    this.pool = pool;
  }

  async setup(): Promise<void> {
    if (this.tableCreated) return;
    const client = await this.pool.connect();
    try {
      await client.query(CREATE_TABLE_SQL);
      await client.query(CREATE_INDEX_SQL);
      this.tableCreated = true;
    } finally {
      client.release();
    }
  }

  async get(keys: CacheFullKey[]): Promise<{ key: CacheFullKey; value: V }[]> {
    if (!keys.length) return [];
    await this.setup();

    const results: { key: CacheFullKey; value: V }[] = [];

    for (const fullKey of keys) {
      const [namespace, key] = fullKey;

      const queryResult = await this.pool.query(
        `SELECT value, encoding, expires_at FROM langgraph_node_cache
         WHERE namespace = $1::text[] AND key = $2
         AND (expires_at IS NULL OR expires_at > NOW())`,
        [namespace, key]
      );

      if (queryResult.rows.length === 0) continue;

      const { value: rawValue, encoding } = queryResult.rows[0];

      try {
        const value = await this.serde.loadsTyped(encoding, rawValue);
        results.push({ key: fullKey, value: value as V });
      } catch {
        await this.pool.query(
          `DELETE FROM langgraph_node_cache WHERE namespace = $1::text[] AND key = $2`,
          [namespace, key]
        );
      }
    }

    return results;
  }

  async set(pairs: { key: CacheFullKey; value: V; ttl?: number }[]): Promise<void> {
    if (!pairs.length) return;
    await this.setup();

    for (const { key: fullKey, value, ttl } of pairs) {
      const [namespace, key] = fullKey;
      const [encoding, rawValue] = await this.serde.dumpsTyped(value);

      await this.pool.query(
        `INSERT INTO langgraph_node_cache (namespace, key, value, encoding, expires_at)
         VALUES ($1::text[], $2, $3, $4, CASE WHEN $5 IS NOT NULL THEN NOW() + ($5 || ' seconds')::interval END)
         ON CONFLICT (namespace, key)
         DO UPDATE SET value = EXCLUDED.value, encoding = EXCLUDED.encoding,
                       expires_at = EXCLUDED.expires_at, created_at = NOW()`,
        [namespace, key, rawValue, encoding, ttl != null ? String(ttl) : null]
      );
    }
  }

  async clear(namespaces: CacheNamespace[]): Promise<void> {
    if (!namespaces.length) {
      await this.pool.query(`DELETE FROM langgraph_node_cache`);
      return;
    }

    for (const namespace of namespaces) {
      await this.pool.query(
        `DELETE FROM langgraph_node_cache WHERE namespace = $1::text[]`,
        [namespace]
      );
    }
  }
}
