import type { RagChunkDocument } from '../models/firestoreModels.js';
import { getPostgresPool } from '../services/postgresClient.js';
import { applyPendingMigrations } from '../services/postgresMigrations.js';

interface RagChunkRow {
  id: string;
  owner_id: string;
  profile_id: string;
  source_doc_id: string;
  source_type: 'be1';
  endpoint: string;
  chunk_index: number;
  text: string;
  text_preview: string;
  embedding: string | null;
  embedding_model: string | null;
  embedding_dim: number | null;
  token_estimate: number | null;
  tags: unknown;
  created_at: number;
}

export interface RagChunkRecord {
  id: string;
  data: RagChunkDocument;
}

function parseVectorLiteral(value: string | null): number[] {
  if (!value) return [];
  const trimmed = value.trim();
  if (!trimmed.startsWith('[') || !trimmed.endsWith(']')) return [];

  const inner = trimmed.slice(1, -1).trim();
  if (!inner) return [];

  return inner
    .split(',')
    .map((item) => Number(item.trim()))
    .filter((item) => Number.isFinite(item));
}

function toVectorLiteral(values?: number[]): string | null {
  if (!values || values.length === 0) return null;
  return `[${values.map((value) => (Number.isFinite(value) ? value : 0)).join(',')}]`;
}

function rowToDocument(row: RagChunkRow): RagChunkDocument {
  return {
    ownerId: row.owner_id,
    profileId: row.profile_id,
    sourceDocId: row.source_doc_id,
    sourceType: row.source_type,
    endpoint: row.endpoint,
    chunkIndex: Number(row.chunk_index),
    text: row.text,
    textPreview: row.text_preview,
    embedding: parseVectorLiteral(row.embedding),
    embeddingModel: row.embedding_model ?? 'deterministic-hash-v1',
    embeddingDim: Number(row.embedding_dim ?? 0),
    tokenEstimate: Number(row.token_estimate ?? 0),
    tags: Array.isArray(row.tags) ? row.tags.map((tag) => String(tag)) : [],
    createdAt: Number(row.created_at),
  };
}

export class RagChunksRepository {
  private async withPool() {
    const pool = getPostgresPool();
    if (!pool) {
      throw new Error('DATABASE_URL is required for rag_chunks repository');
    }

    await applyPendingMigrations(pool);
    return pool;
  }

  async upsertMany(records: RagChunkRecord[]): Promise<void> {
    if (records.length === 0) return;

    const pool = await this.withPool();
    const client = await pool.connect();

    try {
      await client.query('BEGIN');
      for (const record of records) {
        const chunk = record.data;
        await client.query(
          `
          INSERT INTO rag_chunks (
            id,
            owner_id,
            profile_id,
            source_doc_id,
            source_type,
            endpoint,
            chunk_index,
            text,
            text_preview,
            embedding,
            embedding_model,
            embedding_dim,
            token_estimate,
            tags,
            created_at
          ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10::vector, $11, $12, $13, $14::jsonb, $15
          )
          ON CONFLICT (id)
          DO UPDATE SET
            owner_id = EXCLUDED.owner_id,
            profile_id = EXCLUDED.profile_id,
            source_doc_id = EXCLUDED.source_doc_id,
            source_type = EXCLUDED.source_type,
            endpoint = EXCLUDED.endpoint,
            chunk_index = EXCLUDED.chunk_index,
            text = EXCLUDED.text,
            text_preview = EXCLUDED.text_preview,
            embedding = EXCLUDED.embedding,
            embedding_model = EXCLUDED.embedding_model,
            embedding_dim = EXCLUDED.embedding_dim,
            token_estimate = EXCLUDED.token_estimate,
            tags = EXCLUDED.tags,
            created_at = EXCLUDED.created_at
          `,
          [
            record.id,
            chunk.ownerId,
            chunk.profileId,
            chunk.sourceDocId,
            chunk.sourceType,
            chunk.endpoint,
            chunk.chunkIndex,
            chunk.text,
            chunk.textPreview,
            toVectorLiteral(chunk.embedding),
            chunk.embeddingModel,
            chunk.embeddingDim,
            chunk.tokenEstimate,
            JSON.stringify(chunk.tags ?? []),
            chunk.createdAt,
          ]
        );
      }
      await client.query('COMMIT');
    } catch (error) {
      await client.query('ROLLBACK').catch(() => {});
      throw error;
    } finally {
      client.release();
    }
  }

  async listByOwnerProfile(ownerId: string, profileId: string, limit = 200): Promise<RagChunkRecord[]> {
    const pool = await this.withPool();
    const response = await pool.query<RagChunkRow>(
      `
      SELECT id, owner_id, profile_id, source_doc_id, source_type, endpoint, chunk_index, text, text_preview, embedding::text AS embedding, embedding_model, embedding_dim, token_estimate, tags, created_at
      FROM rag_chunks
      WHERE owner_id = $1 AND profile_id = $2
      ORDER BY created_at DESC
      LIMIT $3
      `,
      [ownerId, profileId, Math.max(1, Math.floor(limit))]
    );

    return response.rows.map((row) => ({ id: row.id, data: rowToDocument(row) }));
  }

  async listForQuery(ownerId: string, profileId: string | undefined, limit: number): Promise<RagChunkRecord[]> {
    const pool = await this.withPool();
    const hasProfile = Boolean(profileId);
    const response = await pool.query<RagChunkRow>(
      `
      SELECT id, owner_id, profile_id, source_doc_id, source_type, endpoint, chunk_index, text, text_preview, embedding::text AS embedding, embedding_model, embedding_dim, token_estimate, tags, created_at
      FROM rag_chunks
      WHERE owner_id = $1
        AND ($2::text IS NULL OR profile_id = $2)
      ORDER BY created_at DESC
      LIMIT $3
      `,
      [ownerId, hasProfile ? profileId : null, Math.max(1, Math.floor(limit))]
    );

    return response.rows.map((row) => ({ id: row.id, data: rowToDocument(row) }));
  }

  async deleteByOwnerProfile(ownerId: string, profileId: string): Promise<void> {
    const pool = await this.withPool();
    await pool.query(
      `DELETE FROM rag_chunks WHERE owner_id = $1 AND profile_id = $2`,
      [ownerId, profileId]
    );
  }
}

let singletonRagChunksRepository: RagChunksRepository | null = null;

export function getRagChunksRepository(): RagChunksRepository {
  if (!singletonRagChunksRepository) {
    singletonRagChunksRepository = new RagChunksRepository();
  }

  return singletonRagChunksRepository;
}
