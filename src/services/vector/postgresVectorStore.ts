import { env } from '../../config/env.js';
import type { RagChunkResult } from '../../models/firestoreModels.js';
import { getPostgresPool } from '../postgresClient.js';
import type { VectorChunkUpsertRecord, VectorSearchInput, VectorStoreProvider } from './types.js';

function shouldEnablePgVector(): boolean {
  return env.PGVECTOR_ENABLED.trim().toLowerCase() === 'true';
}

function toVectorLiteral(values: number[]): string {
  return `[${values.map((value) => (Number.isFinite(value) ? value : 0)).join(',')}]`;
}

function deriveTextPreview(text: string): string {
  return text.length <= 240 ? text : `${text.slice(0, 237)}...`;
}

function rowsToChunks(rows: any[]): RagChunkResult[] {
  return rows
    .map((row) => ({
      id: String(row.id),
      ownerId: String(row.owner_id),
      profileId: String(row.profile_id ?? row.kundali_id ?? ''),
      kundaliId: String(row.kundali_id),
      sourceDocId: String(row.source_doc_id ?? ''),
      sourceType: (row.source_type ?? 'be1') as 'be1',
      endpoint: String(row.endpoint ?? 'calculate'),
      chunkIndex: Number(row.chunk_index ?? 0),
      text: String(row.text ?? ''),
      textPreview: String(row.text_preview ?? row.section ?? deriveTextPreview(String(row.text ?? ''))),
      embedding: [],
      embeddingModel: 'postgres-pgvector',
      embeddingDim: env.EMBEDDING_DIM,
      tokenEstimate: Number(row.token_estimate ?? 0),
      tags: Array.isArray(row.tags) ? row.tags.map((tag: unknown) => String(tag)) : [],
      createdAt: Number(row.created_at ?? 0),
      similarity: typeof row.similarity === 'number' ? row.similarity : 0,
    }))
    .filter((item) => Boolean(item.id) && Boolean(item.text));
}

export class PostgresVectorStore implements VectorStoreProvider {
  async upsertChunks(records: VectorChunkUpsertRecord[]): Promise<void> {
    const pool = getPostgresPool();
    if (!pool || records.length === 0 || !shouldEnablePgVector()) return;

    const client = await pool.connect();
    try {
      await client.query('BEGIN');

      const firstChunk = records[0]?.data as any;
      const chartOwnerId = String(firstChunk?.ownerId ?? '');
      const kundaliId = String(firstChunk?.kundaliId ?? '');
      const sourceDocId = String(firstChunk?.sourceDocId ?? '');

      if (!chartOwnerId || !kundaliId) {
        throw new Error('Missing ownerId/kundaliId for pgvector upsert');
      }

      const chartLookup = await client.query<{ id: string }>(
        `
          SELECT id
          FROM charts
          WHERE owner_id = $1
            AND kundali_id = $2
            AND deleted_at IS NULL
          LIMIT 1
        `,
        [chartOwnerId, kundaliId]
      );
      const chartId = chartLookup.rows[0]?.id;
      if (!chartId) {
        throw new Error(`Unable to resolve chart id for owner=${chartOwnerId} kundali=${kundaliId}`);
      }

      await client.query('DELETE FROM chart_vectors WHERE owner_id = $1 AND kundali_id = $2 AND source_doc_id = $3', [
        chartOwnerId,
        kundaliId,
        sourceDocId,
      ]);

      for (const record of records) {
        await client.query(
          `
            INSERT INTO chart_vectors (
              chart_id,
              owner_id,
              kundali_id,
              source_doc_id,
              source_type,
              endpoint,
              chunk_index,
              section,
              text,
              text_preview,
              embedding_model,
              embedding_dim,
              token_estimate,
              tags,
              embedding,
              created_at
            ) VALUES (
              $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14::jsonb, $15::vector, $16
            )
          `,
          [
            chartId,
            chartOwnerId,
            kundaliId,
            sourceDocId,
              firstChunk?.sourceType ?? 'be1',
              firstChunk?.endpoint ?? 'calculate',
            record.data.chunkIndex,
            deriveTextPreview(record.data.text),
            record.data.text,
              deriveTextPreview(record.data.text),
              record.data.embeddingModel ?? 'deterministic-hash',
              record.data.embeddingDim ?? env.EMBEDDING_DIM,
              record.data.tokenEstimate ?? 0,
              JSON.stringify(record.data.tags ?? []),
            toVectorLiteral(record.data.embedding),
            record.data.createdAt ?? Date.now(),
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

  async searchChunks(input: VectorSearchInput): Promise<RagChunkResult[]> {
    const pool = getPostgresPool();
    if (!pool || !shouldEnablePgVector()) return [];

    const client = await pool.connect();
    try {
      const candidateWindow = Math.max(input.candidateWindow ?? Math.max(input.topK * 8, 40), input.topK);
      const filters = ['owner_id = $1'];
      const params: unknown[] = [input.ownerId];
      let paramIndex = 2;

      if (input.kundaliId) {
        filters.push(`kundali_id = $${paramIndex}`);
        params.push(input.kundaliId);
        paramIndex += 1;
      }

      params.push(toVectorLiteral(input.queryEmbedding));
      params.push(candidateWindow);

      const result = await client.query(
        `
          SELECT
            id,
            owner_id,
            kundali_id,
            source_doc_id,
            source_type,
            endpoint,
            chunk_index,
            section,
            text,
            text_preview,
            embedding_model,
            embedding_dim,
            token_estimate,
            tags,
            created_at,
            1 - (embedding <=> $${paramIndex}::vector) AS similarity
          FROM chart_vectors
          WHERE ${filters.join(' AND ')}
          ORDER BY embedding <=> $${paramIndex}::vector ASC, created_at DESC
          LIMIT $${paramIndex + 1}
        `,
        params
      );

      return rowsToChunks(result.rows as any[]).slice(0, input.topK).filter((item) => Number.isFinite(item.similarity));
    } finally {
      client.release();
    }
  }
}