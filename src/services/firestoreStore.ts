import fs from 'node:fs';
import { GoogleAuth } from 'google-auth-library';
import { env } from '../config/env.js';

type FirestorePrimitive = string | number | boolean | null;
type FirestoreDocumentValue =
  | FirestorePrimitive
  | Date
  | FirestoreDocumentValue[]
  | { [key: string]: FirestoreDocumentValue };

export type FirestoreDocumentData = { [key: string]: FirestoreDocumentValue };

type FirestoreValue = {
  stringValue?: string;
  integerValue?: string;
  doubleValue?: number;
  booleanValue?: boolean;
  timestampValue?: string;
  arrayValue?: { values?: FirestoreValue[] };
  mapValue?: { fields?: Record<string, FirestoreValue> };
  nullValue?: null;
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
  value: FirestoreDocumentValue;
}

export interface StructuredOrderBy {
  field: string;
  direction?: 'ASCENDING' | 'DESCENDING';
}

interface QueryResult<T> {
  id: string;
  data: T;
}

function toFirestoreValue(value: FirestoreDocumentValue): FirestoreValue {
  if (value === null) return { nullValue: null };
  if (typeof value === 'string') return { stringValue: value };

  if (typeof value === 'number') {
    if (Number.isInteger(value)) {
      return { integerValue: String(value) };
    }
    return { doubleValue: value };
  }

  if (typeof value === 'boolean') return { booleanValue: value };
  if (value instanceof Date) return { timestampValue: value.toISOString() };

  if (Array.isArray(value)) {
    return { arrayValue: { values: value.map((item) => toFirestoreValue(item)) } };
  }

  return {
    mapValue: {
      fields: Object.entries(value).reduce(
        (acc, [key, nested]) => {
          acc[key] = toFirestoreValue(nested);
          return acc;
        },
        {} as Record<string, FirestoreValue>
      ),
    },
  };
}

function fromFirestoreValue(value: FirestoreValue): FirestoreDocumentValue {
  if (value.stringValue !== undefined) return value.stringValue;
  if (value.integerValue !== undefined) return Number(value.integerValue);
  if (value.doubleValue !== undefined) return value.doubleValue;
  if (value.booleanValue !== undefined) return value.booleanValue;
  if (value.timestampValue !== undefined) return value.timestampValue;
  if (value.nullValue !== undefined) return null;

  if (value.arrayValue?.values) {
    return value.arrayValue.values.map((item) => fromFirestoreValue(item));
  }

  if (value.mapValue?.fields) {
    return Object.entries(value.mapValue.fields).reduce(
      (acc, [key, nested]) => {
        acc[key] = fromFirestoreValue(nested);
        return acc;
      },
      {} as Record<string, FirestoreDocumentValue>
    );
  }

  return null;
}

function fromFirestoreFields(fields?: Record<string, FirestoreValue>): FirestoreDocumentData {
  if (!fields) return {};
  return Object.entries(fields).reduce(
    (acc, [key, value]) => {
      acc[key] = fromFirestoreValue(value);
      return acc;
    },
    {} as FirestoreDocumentData
  );
}

export class FirestoreStore {
  private readonly baseUrl: string;
  private readonly auth: GoogleAuth;

  constructor(projectId: string) {
    this.baseUrl = `https://firestore.googleapis.com/v1/projects/${projectId}/databases/default/documents`;

    let credentials: { client_email?: string; private_key?: string } | undefined;
    if (env.FIREBASE_SERVICE_ACCOUNT_JSON?.trim()) {
      credentials = JSON.parse(env.FIREBASE_SERVICE_ACCOUNT_JSON) as { client_email?: string; private_key?: string };
    } else if (env.FIREBASE_SERVICE_ACCOUNT_PATH?.trim()) {
      const raw = fs.readFileSync(env.FIREBASE_SERVICE_ACCOUNT_PATH, 'utf8');
      credentials = JSON.parse(raw) as { client_email?: string; private_key?: string };
    }

    this.auth = new GoogleAuth({
      credentials,
      projectId,
      scopes: ['https://www.googleapis.com/auth/datastore'],
    });
  }

  private async authHeaders(): Promise<Record<string, string>> {
    const token = await this.auth.getAccessToken();

    if (!token) {
      throw new Error('Failed to obtain OAuth access token for Firestore REST API');
    }

    return {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    };
  }

  private parseDocId(documentName: string): string {
    const parts = documentName.split('/');
    return parts[parts.length - 1] ?? documentName;
  }

  async setDocument(path: string, data: object, merge = true): Promise<void> {
    const url = `${this.baseUrl}/${path}`;

    const response = await fetch(url, {
      method: 'PATCH',
      headers: await this.authHeaders(),
      body: JSON.stringify(
        merge
          ? {
              fields: Object.entries(data as Record<string, FirestoreDocumentValue>).reduce(
                (acc, [key, value]) => {
                  acc[key] = toFirestoreValue(value);
                  return acc;
                },
                {} as Record<string, FirestoreValue>
              ),
            }
          : {
              fields: Object.entries(data as Record<string, FirestoreDocumentValue>).reduce(
                (acc, [key, value]) => {
                  acc[key] = toFirestoreValue(value);
                  return acc;
                },
                {} as Record<string, FirestoreValue>
              ),
            }
      ),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Firestore setDocument failed (${response.status}): ${errorText}`);
    }
  }

  async createDocument(collectionPath: string, data: object, documentId?: string): Promise<string> {
    const queryPart = documentId ? `?documentId=${encodeURIComponent(documentId)}` : '';
    const url = `${this.baseUrl}/${collectionPath}${queryPart}`;

    const response = await fetch(url, {
      method: 'POST',
      headers: await this.authHeaders(),
      body: JSON.stringify({
        fields: Object.entries(data as Record<string, FirestoreDocumentValue>).reduce(
          (acc, [key, value]) => {
            acc[key] = toFirestoreValue(value);
            return acc;
          },
          {} as Record<string, FirestoreValue>
        ),
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Firestore createDocument failed (${response.status}): ${errorText}`);
    }

    const created = (await response.json()) as { name: string };
    return this.parseDocId(created.name);
  }

  async getDocument<T extends object>(path: string): Promise<QueryResult<T> | null> {
    const url = `${this.baseUrl}/${path}`;
    const response = await fetch(url, {
      method: 'GET',
      headers: await this.authHeaders(),
    });

    if (response.status === 404) return null;

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Firestore getDocument failed (${response.status}): ${errorText}`);
    }

    const document = (await response.json()) as {
      name: string;
      fields?: Record<string, FirestoreValue>;
    };

    return {
      id: this.parseDocId(document.name),
      data: fromFirestoreFields(document.fields) as T,
    };
  }

  async runQuery<T extends object>(
    collection: string,
    filters: StructuredFieldFilter[],
    options?: {
      orderBy?: StructuredOrderBy[];
      limit?: number;
    }
  ): Promise<QueryResult<T>[]> {
    const url = `${this.baseUrl}:runQuery`;

    const structuredQuery: {
      from: Array<{ collectionId: string }>;
      where?: {
        compositeFilter: {
          op: 'AND';
          filters: Array<{
            fieldFilter: {
              field: { fieldPath: string };
              op: FieldFilterOperator;
              value: FirestoreValue;
            };
          }>;
        };
      };
      orderBy?: Array<{ field: { fieldPath: string }; direction: 'ASCENDING' | 'DESCENDING' }>;
      limit?: number;
    } = {
      from: [{ collectionId: collection }],
    };

    if (filters.length > 0) {
      structuredQuery.where = {
        compositeFilter: {
          op: 'AND',
          filters: filters.map((filter) => ({
            fieldFilter: {
              field: { fieldPath: filter.field },
              op: filter.op,
              value: toFirestoreValue(filter.value),
            },
          })),
        },
      };
    }

    if (options?.orderBy?.length) {
      structuredQuery.orderBy = options.orderBy.map((orderBy) => ({
        field: { fieldPath: orderBy.field },
        direction: orderBy.direction ?? 'DESCENDING',
      }));
    }

    if (typeof options?.limit === 'number') {
      structuredQuery.limit = options.limit;
    }

    const response = await fetch(url, {
      method: 'POST',
      headers: await this.authHeaders(),
      body: JSON.stringify({ structuredQuery }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Firestore runQuery failed (${response.status}): ${errorText}`);
    }

    const rows = (await response.json()) as Array<{
      document?: {
        name: string;
        fields?: Record<string, FirestoreValue>;
      };
    }>;

    return rows
      .filter((row) => row.document)
      .map((row) => {
        const doc = row.document!;
        return {
          id: this.parseDocId(doc.name),
          data: fromFirestoreFields(doc.fields) as T,
        };
      });
  }
}

let singletonStore: FirestoreStore | null = null;

export function getFirestoreStore(): FirestoreStore {
  if (!env.FIREBASE_PROJECT_ID) {
    throw new Error('FIREBASE_PROJECT_ID is required for Firestore REST store');
  }

  if (!singletonStore) {
    singletonStore = new FirestoreStore(env.FIREBASE_PROJECT_ID);
  }

  return singletonStore;
}
