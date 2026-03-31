import type { RagApiSourceDocument } from '../models/firestoreModels.js';
import { getPostgresPool } from '../services/postgresClient.js';
import { applyPendingMigrations } from '../services/postgresMigrations.js';

interface RagApiSourceRow {
  id: string;
  owner_id: string;
  profile_id: string;
  display_name: string | null;
  place: string | null;
  source_type: 'be1';
  endpoint: string;
  request_key: string;
  payload_hash: string;
  raw_payload: unknown;
  chart_snapshot: unknown;
  preview: string;
  tags: unknown;
  created_at: number;
}

function rowToDocument(row: RagApiSourceRow): RagApiSourceDocument {
  return {
    ownerId: row.owner_id,
    profileId: row.profile_id,
    displayName: row.display_name ?? undefined,
    place: row.place ?? undefined,
    sourceType: row.source_type,
    endpoint: row.endpoint,
    requestKey: row.request_key,
    payloadHash: row.payload_hash,
    rawPayload: row.raw_payload,
    chartSnapshot: row.chart_snapshot ?? undefined,
    preview: row.preview,
    tags: Array.isArray(row.tags) ? row.tags.map((tag) => String(tag)) : [],
    createdAt: Number(row.created_at),
  };
}

export interface RagApiSourceRecord {
  id: string;
  data: RagApiSourceDocument;
}

export class RagSourcesRepository {
  private async withPool() {
    const pool = getPostgresPool();
    if (!pool) {
      throw new Error('DATABASE_URL is required for rag_api_sources repository');
    }

    await applyPendingMigrations(pool);
    return pool;
  }

  async getById(id: string): Promise<RagApiSourceRecord | null> {
    const pool = await this.withPool();
    const response = await pool.query<RagApiSourceRow>(
      `
      SELECT id, owner_id, profile_id, display_name, place, source_type, endpoint, request_key, payload_hash, raw_payload, chart_snapshot, preview, tags, created_at
      FROM rag_api_sources
      WHERE id = $1
      LIMIT 1
      `,
      [id]
    );

    const row = response.rows[0];
    return row ? { id: row.id, data: rowToDocument(row) } : null;
  }

  async listByOwnerProfile(ownerId: string, profileId: string): Promise<RagApiSourceRecord[]> {
    const pool = await this.withPool();
    const response = await pool.query<RagApiSourceRow>(
      `
      SELECT id, owner_id, profile_id, display_name, place, source_type, endpoint, request_key, payload_hash, raw_payload, chart_snapshot, preview, tags, created_at
      FROM rag_api_sources
      WHERE owner_id = $1 AND profile_id = $2
      ORDER BY created_at DESC
      `,
      [ownerId, profileId]
    );

    return response.rows.map((row) => ({ id: row.id, data: rowToDocument(row) }));
  }

  async upsert(id: string, source: RagApiSourceDocument): Promise<void> {
    const pool = await this.withPool();
    await pool.query(
      `
      INSERT INTO rag_api_sources (
        id,
        owner_id,
        profile_id,
        display_name,
        place,
        source_type,
        endpoint,
        request_key,
        payload_hash,
        raw_payload,
        chart_snapshot,
        preview,
        tags,
        created_at
      ) VALUES (
        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, $11::jsonb, $12, $13::jsonb, $14
      )
      ON CONFLICT (id)
      DO UPDATE SET
        owner_id = EXCLUDED.owner_id,
        profile_id = EXCLUDED.profile_id,
        display_name = EXCLUDED.display_name,
        place = EXCLUDED.place,
        source_type = EXCLUDED.source_type,
        endpoint = EXCLUDED.endpoint,
        request_key = EXCLUDED.request_key,
        payload_hash = EXCLUDED.payload_hash,
        raw_payload = EXCLUDED.raw_payload,
        chart_snapshot = EXCLUDED.chart_snapshot,
        preview = EXCLUDED.preview,
        tags = EXCLUDED.tags,
        created_at = EXCLUDED.created_at
      `,
      [
        id,
        source.ownerId,
        source.profileId,
        source.displayName ?? null,
        source.place ?? null,
        source.sourceType,
        source.endpoint,
        source.requestKey,
        source.payloadHash,
        JSON.stringify(source.rawPayload),
        source.chartSnapshot ? JSON.stringify(source.chartSnapshot) : null,
        source.preview,
        JSON.stringify(source.tags ?? []),
        source.createdAt,
      ]
    );
  }

  async patchMetadataForOwnerProfile(ownerId: string, profileId: string, patch: { displayName?: string; place?: string }): Promise<void> {
    const pool = await this.withPool();
    await pool.query(
      `
      UPDATE rag_api_sources
      SET
        display_name = COALESCE($3, display_name),
        place = COALESCE($4, place)
      WHERE owner_id = $1 AND profile_id = $2
      `,
      [ownerId, profileId, patch.displayName ?? null, patch.place ?? null]
    );
  }

  async deleteByOwnerProfile(ownerId: string, profileId: string): Promise<void> {
    const pool = await this.withPool();
    await pool.query(
      `DELETE FROM rag_api_sources WHERE owner_id = $1 AND profile_id = $2`,
      [ownerId, profileId]
    );
  }
}

let singletonRagSourcesRepository: RagSourcesRepository | null = null;

export function getRagSourcesRepository(): RagSourcesRepository {
  if (!singletonRagSourcesRepository) {
    singletonRagSourcesRepository = new RagSourcesRepository();
  }

  return singletonRagSourcesRepository;
}
