import { randomUUID } from 'node:crypto';
import { Pool, type PoolClient } from 'pg';
import { env } from '../config/env.js';
import { buildPostgresSslConfig, withPostgresSslOverrides } from './postgresSsl.js';
import { applyPendingMigrations } from './postgresMigrations.js';

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

function toJsonPathLiteral(fieldPath: string): string {
  const parts = fieldPath.split('.').map((part) => part.trim()).filter(Boolean);
  if (parts.length === 0) {
    throw new Error(`Invalid field path: ${fieldPath}`);
  }

  for (const part of parts) {
    if (!/^[A-Za-z0-9_]+$/.test(part)) {
      throw new Error(`Unsafe field path segment: ${part}`);
    }
  }

  return `{${parts.join(',')}}`;
}

function buildJsonTextExpr(fieldPath: string): string {
  return `data #>> '${toJsonPathLiteral(fieldPath)}'`;
}

function isScalarComparable(value: PostgresDocumentValue): value is string | number | boolean | null {
  return value === null || typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean';
}

function canUseSqlPushdown(filters: StructuredFieldFilter[]): boolean {
  return filters.every((filter) => filter.op === 'EQUAL' && isScalarComparable(filter.value));
}

class PostgresStore {
  private readonly pool: Pool;
  private ready: Promise<void> | null = null;

  constructor(databaseUrl: string) {
    this.pool = new Pool({
      connectionString: withPostgresSslOverrides(databaseUrl),
      query_timeout: env.PG_QUERY_TIMEOUT_MS,
      connectionTimeoutMillis: env.PG_CONNECTION_TIMEOUT_MS,
      idleTimeoutMillis: env.PG_IDLE_TIMEOUT_MS,
      keepAlive: true,
      keepAliveInitialDelayMillis: env.PG_KEEPALIVE_INITIAL_DELAY_MS,
      ...buildPostgresSslConfig(),
    });

    this.pool.on('error', (error) => {
      console.error('[postgres-store] idle client error (non-fatal)', {
        message: error.message,
        code: (error as NodeJS.ErrnoException).code,
      });
    });
  }

  private isTransientQueryTimeout(error: unknown): boolean {
    const message = String(error ?? '').toLowerCase();
    return (
      message.includes('query read timeout') ||
      message.includes('query timeout') ||
      message.includes('timeout') ||
      message.includes('etimedout')
    );
  }

  private async executeQuery(text: string, values: unknown[] = [], options?: { disableTimeout?: boolean; retries?: number }): Promise<{ rows: unknown[] }> {
    const retries = Math.max(0, options?.retries ?? 1);
    let attempt = 0;

    while (true) {
      try {
        if (options?.disableTimeout) {
          return (await (this.pool as any).query({
            text,
            values,
            query_timeout: 0,
          })) as { rows: unknown[] };
        }

        return (await this.pool.query(text, values)) as { rows: unknown[] };
      } catch (error) {
        if (attempt >= retries || !this.isTransientQueryTimeout(error)) {
          throw error;
        }

        attempt += 1;
        const delayMs = 150 * attempt;
        await new Promise((resolve) => setTimeout(resolve, delayMs));
      }
    }
  }

  private async ensureSchema(): Promise<void> {
    if (!this.ready) {
      this.ready = (async () => {
        await applyPendingMigrations(this.pool);
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

    await this.executeQuery(
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
      [path, collection, docId, JSON.stringify(nextData)],
      { retries: 1 }
    );
  }

  async createDocument(collectionPath: string, data: object, documentId?: string): Promise<string> {
    const docId = documentId ?? randomUUID();
    await this.setDocument(`${collectionPath}/${docId}`, data, false);
    return docId;
  }

  async deleteDocument(path: string): Promise<void> {
    await this.ensureSchema();
    await this.executeQuery(`DELETE FROM documents WHERE path = $1`, [path], { retries: 1 });
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

    if (canUseSqlPushdown(filters)) {
      const params: Array<string | number> = [collection];
      const whereParts: string[] = ['collection = $1'];

      for (const filter of filters) {
        const fieldExpr = buildJsonTextExpr(filter.field);
        if (filter.value === null) {
          whereParts.push(`${fieldExpr} IS NULL`);
          continue;
        }

        params.push(String(filter.value));
        whereParts.push(`${fieldExpr} = $${params.length}`);
      }

      const orderBy = options?.orderBy ?? [];
      const orderBySql = orderBy
        .map((clause) => {
          const direction = clause.direction === 'ASCENDING' ? 'ASC' : 'DESC';
          const fieldExpr = buildJsonTextExpr(clause.field);
          return `${fieldExpr} ${direction}`;
        })
        .join(', ');

      let limitSql = '';
      if (typeof options?.limit === 'number' && Number.isFinite(options.limit) && options.limit > 0) {
        params.push(Math.floor(options.limit));
        limitSql = ` LIMIT $${params.length}`;
      }

      const sql = [
        'SELECT path, collection, doc_id, data, created_at, updated_at',
        'FROM documents',
        `WHERE ${whereParts.join(' AND ')}`,
        orderBySql ? `ORDER BY ${orderBySql}` : '',
        limitSql,
      ]
        .filter(Boolean)
        .join(' ');

      const response = await this.executeQuery(sql, params, { retries: 1 });
      return (response.rows as StoredDocumentRow[]).map((row) => PostgresStore.rowToQueryResult<T>(row));
    }

    const response = await this.executeQuery(
      `SELECT path, collection, doc_id, data, created_at, updated_at FROM documents WHERE collection = $1`,
      [collection],
      { retries: 1 }
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
