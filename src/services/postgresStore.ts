import { randomUUID } from 'node:crypto';
import { Pool, type PoolClient } from 'pg';
import { env } from '../config/env.js';
import { buildPostgresSslConfig, withPostgresSslOverrides } from './postgresSsl.js';

type PostgresDocumentValue =
  | string
  | number
  | boolean
  | null
  | Date
  | PostgresDocumentValue[]
  | { [key: string]: PostgresDocumentValue };

export type PostgresDocumentData = { [key: string]: PostgresDocumentValue };

type QueryResult<T> = {
  id: string;
  data: T;
};

type FieldFilterOperator =
  | 'EQUAL'
  | 'NOT_EQUAL'
  | 'GREATER_THAN'
  | 'GREATER_THAN_OR_EQUAL'
  | 'LESS_THAN'
  | 'LESS_THAN_OR_EQUAL'
  | 'IN'
  | 'NOT_IN'
  | 'ARRAY_CONTAINS'
  | 'ARRAY_CONTAINS_ANY';

export interface StructuredFieldFilter {
  field: string;
  op: FieldFilterOperator;
  value: PostgresDocumentValue;
}

export interface StructuredOrderBy {
  field: string;
  direction?: 'ASCENDING' | 'DESCENDING';
}

interface StoredDocumentRow {
  path: string;
  collection: string;
  doc_id: string;
  data: PostgresDocumentData;
  created_at: string;
  updated_at: string;
}

function deepMerge<T>(base: T, patch: unknown): T {
  if (Array.isArray(base) || Array.isArray(patch)) {
    return patch as T;
  }

  if (base && typeof base === 'object' && patch && typeof patch === 'object') {
    const result: Record<string, unknown> = { ...(base as Record<string, unknown>) };
    for (const [key, value] of Object.entries(patch as Record<string, unknown>)) {
      const current = result[key];
      if (current && typeof current === 'object' && !Array.isArray(current) && value && typeof value === 'object' && !Array.isArray(value)) {
        result[key] = deepMerge(current, value);
      } else {
        result[key] = value;
      }
    }
    return result as T;
  }

  return patch as T;
}

function parsePath(path: string): { collection: string; docId: string } {
  const normalized = path.replace(/^\/+|\/+$/g, '');
  const parts = normalized.split('/').filter(Boolean);
  if (parts.length < 2) {
    throw new Error(`Invalid document path: ${path}`);
  }

  return {
    collection: parts[0]!,
    docId: parts.slice(1).join('/'),
  };
}

function getFieldValue(data: unknown, fieldPath: string): unknown {
  if (!data || typeof data !== 'object') return undefined;
  return fieldPath.split('.').reduce<unknown>((current, segment) => {
    if (!current || typeof current !== 'object') return undefined;
    return (current as Record<string, unknown>)[segment];
  }, data);
}

function compareValues(left: unknown, op: FieldFilterOperator, right: unknown): boolean {
  switch (op) {
    case 'EQUAL':
      return JSON.stringify(left) === JSON.stringify(right);
    case 'NOT_EQUAL':
      return JSON.stringify(left) !== JSON.stringify(right);
    case 'GREATER_THAN':
      return Number(left) > Number(right);
    case 'GREATER_THAN_OR_EQUAL':
      return Number(left) >= Number(right);
    case 'LESS_THAN':
      return Number(left) < Number(right);
    case 'LESS_THAN_OR_EQUAL':
      return Number(left) <= Number(right);
    case 'IN':
      return Array.isArray(right) ? right.some((item) => JSON.stringify(item) === JSON.stringify(left)) : false;
    case 'NOT_IN':
      return Array.isArray(right) ? !right.some((item) => JSON.stringify(item) === JSON.stringify(left)) : false;
    case 'ARRAY_CONTAINS':
      return Array.isArray(left) ? left.some((item) => JSON.stringify(item) === JSON.stringify(right)) : false;
    case 'ARRAY_CONTAINS_ANY':
      return Array.isArray(left) && Array.isArray(right)
        ? right.some((candidate) => left.some((item) => JSON.stringify(item) === JSON.stringify(candidate)))
        : false;
    default:
      return false;
  }
}

class PostgresStore {
  private readonly pool: Pool;
  private ready: Promise<void> | null = null;

  constructor(databaseUrl: string) {
    this.pool = new Pool({
      connectionString: withPostgresSslOverrides(databaseUrl),
      ...buildPostgresSslConfig(),
    });
  }

  private async ensureSchema(): Promise<void> {
    if (!this.ready) {
      this.ready = (async () => {
        await this.pool.query(`
          CREATE TABLE IF NOT EXISTS documents (
            path TEXT PRIMARY KEY,
            collection TEXT NOT NULL,
            doc_id TEXT NOT NULL,
            data JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
          );
        `);
        await this.pool.query(`CREATE INDEX IF NOT EXISTS documents_collection_idx ON documents (collection);`);
        await this.pool.query(`CREATE INDEX IF NOT EXISTS documents_collection_owner_idx ON documents (collection, ((data->>'ownerId')));`);
        await this.pool.query(`CREATE INDEX IF NOT EXISTS documents_collection_profile_idx ON documents (collection, ((data->>'profileId')));`);
        await this.pool.query(`CREATE INDEX IF NOT EXISTS documents_collection_updated_idx ON documents (collection, updated_at DESC);`);
      })();
    }

    await this.ready;
  }

  private async readRow(path: string, client?: PoolClient): Promise<StoredDocumentRow | null> {
    const runner = client ?? this.pool;
    const response = await runner.query(
      `SELECT path, collection, doc_id, data, created_at, updated_at FROM documents WHERE path = $1 LIMIT 1`,
      [path]
    );
    return (response.rows[0] as StoredDocumentRow | undefined) ?? null;
  }

  private static rowToQueryResult<T>(row: StoredDocumentRow): QueryResult<T> {
    return {
      id: row.doc_id,
      data: row.data as T,
    };
  }

  async setDocument(path: string, data: object, merge = true): Promise<void> {
    await this.ensureSchema();
    const { collection, docId } = parsePath(path);
    const existing = merge ? await this.readRow(path) : null;
    const nextData = merge && existing ? deepMerge(existing.data, data) : (data as PostgresDocumentData);

    await this.pool.query(
      `
      INSERT INTO documents (path, collection, doc_id, data, created_at, updated_at)
      VALUES ($1, $2, $3, $4::jsonb, NOW(), NOW())
      ON CONFLICT (path)
      DO UPDATE SET
        collection = EXCLUDED.collection,
        doc_id = EXCLUDED.doc_id,
        data = EXCLUDED.data,
        updated_at = NOW();
      `,
      [path, collection, docId, JSON.stringify(nextData)]
    );
  }

  async createDocument(collectionPath: string, data: object, documentId?: string): Promise<string> {
    const docId = documentId ?? randomUUID();
    await this.setDocument(`${collectionPath}/${docId}`, data, false);
    return docId;
  }

  async deleteDocument(path: string): Promise<void> {
    await this.ensureSchema();
    await this.pool.query(`DELETE FROM documents WHERE path = $1`, [path]);
  }

  async getDocument<T extends object>(path: string): Promise<QueryResult<T> | null> {
    await this.ensureSchema();
    const row = await this.readRow(path);
    return row ? PostgresStore.rowToQueryResult<T>(row) : null;
  }

  async runQuery<T extends object>(
    collection: string,
    filters: StructuredFieldFilter[],
    options?: {
      orderBy?: StructuredOrderBy[];
      limit?: number;
    }
  ): Promise<QueryResult<T>[]> {
    await this.ensureSchema();
    const response = await this.pool.query(
      `SELECT path, collection, doc_id, data, created_at, updated_at FROM documents WHERE collection = $1`,
      [collection]
    );

    const rows = response.rows as StoredDocumentRow[];
    const filtered = rows.filter((row) =>
      filters.every((filter) => compareValues(getFieldValue(row.data, filter.field), filter.op, filter.value))
    );

    const ordered = [...filtered].sort((a, b) => {
      const orderBy = options?.orderBy ?? [];
      for (const clause of orderBy) {
        const left = getFieldValue(a.data, clause.field);
        const right = getFieldValue(b.data, clause.field);
        if (left === right) continue;

        const direction = clause.direction === 'ASCENDING' ? 1 : -1;
        if (left === undefined || left === null) return 1 * direction;
        if (right === undefined || right === null) return -1 * direction;

        if (typeof left === 'number' && typeof right === 'number') {
          return (left - right) * direction;
        }

        if (typeof left === 'string' && typeof right === 'string') {
          return left.localeCompare(right) * direction;
        }

        return JSON.stringify(left).localeCompare(JSON.stringify(right)) * direction;
      }
      return 0;
    });

    const limited = typeof options?.limit === 'number' ? ordered.slice(0, options.limit) : ordered;
    return limited.map((row) => PostgresStore.rowToQueryResult<T>(row));
  }
}

let singletonStore: PostgresStore | null = null;

export function getPostgresStore(): PostgresStore {
  if (!env.DATABASE_URL?.trim()) {
    throw new Error('DATABASE_URL is required for Postgres-backed persistence');
  }

  if (!singletonStore) {
    singletonStore = new PostgresStore(env.DATABASE_URL);
  }

  return singletonStore;
}

export { PostgresStore };
