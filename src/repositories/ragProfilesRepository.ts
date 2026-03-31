import type { RagProfileDocument } from '../models/firestoreModels.js';
import { getPostgresPool } from '../services/postgresClient.js';
import { applyPendingMigrations } from '../services/postgresMigrations.js';

interface RagProfileRow {
  owner_id: string;
  profile_id: string;
  display_name: string | null;
  place: string | null;
  kundli_signature: string;
  chart_version: string;
  kundli_input: RagProfileDocument['kundliInput'];
  latest_source_doc_id: string;
  source_count: number;
  updated_at: number;
  created_at: number;
}

function rowToDocument(row: RagProfileRow): RagProfileDocument {
  return {
    ownerId: row.owner_id,
    profileId: row.profile_id,
    displayName: row.display_name ?? undefined,
    place: row.place ?? undefined,
    kundliSignature: row.kundli_signature,
    chartVersion: row.chart_version,
    kundliInput: row.kundli_input,
    latestSourceDocId: row.latest_source_doc_id,
    sourceCount: Number(row.source_count),
    updatedAt: Number(row.updated_at),
    createdAt: Number(row.created_at),
  };
}

export class RagProfilesRepository {
  private async withPool() {
    const pool = getPostgresPool();
    if (!pool) {
      throw new Error('DATABASE_URL is required for rag_profiles repository');
    }

    await applyPendingMigrations(pool);
    return pool;
  }

  async getByOwnerAndProfileId(ownerId: string, profileId: string): Promise<RagProfileDocument | null> {
    const pool = await this.withPool();
    const response = await pool.query<RagProfileRow>(
      `
      SELECT owner_id, profile_id, display_name, place, kundli_signature, chart_version, kundli_input, latest_source_doc_id, source_count, updated_at, created_at
      FROM rag_profiles
      WHERE owner_id = $1 AND profile_id = $2
      LIMIT 1
      `,
      [ownerId, profileId]
    );

    const row = response.rows[0];
    return row ? rowToDocument(row) : null;
  }

  async listByOwner(ownerId: string, limit = 100): Promise<RagProfileDocument[]> {
    const pool = await this.withPool();
    const response = await pool.query<RagProfileRow>(
      `
      SELECT owner_id, profile_id, display_name, place, kundli_signature, chart_version, kundli_input, latest_source_doc_id, source_count, updated_at, created_at
      FROM rag_profiles
      WHERE owner_id = $1
      ORDER BY updated_at DESC
      LIMIT $2
      `,
      [ownerId, Math.max(1, Math.floor(limit))]
    );

    return response.rows.map(rowToDocument);
  }

  async upsert(profile: RagProfileDocument): Promise<void> {
    const pool = await this.withPool();
    await pool.query(
      `
      INSERT INTO rag_profiles (
        owner_id,
        profile_id,
        display_name,
        place,
        kundli_signature,
        chart_version,
        kundli_input,
        latest_source_doc_id,
        source_count,
        updated_at,
        created_at
      ) VALUES (
        $1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10, $11
      )
      ON CONFLICT (owner_id, profile_id)
      DO UPDATE SET
        display_name = EXCLUDED.display_name,
        place = EXCLUDED.place,
        kundli_signature = EXCLUDED.kundli_signature,
        chart_version = EXCLUDED.chart_version,
        kundli_input = EXCLUDED.kundli_input,
        latest_source_doc_id = EXCLUDED.latest_source_doc_id,
        source_count = EXCLUDED.source_count,
        updated_at = EXCLUDED.updated_at,
        created_at = LEAST(rag_profiles.created_at, EXCLUDED.created_at)
      `,
      [
        profile.ownerId,
        profile.profileId,
        profile.displayName ?? null,
        profile.place ?? null,
        profile.kundliSignature,
        profile.chartVersion,
        JSON.stringify(profile.kundliInput),
        profile.latestSourceDocId,
        profile.sourceCount,
        profile.updatedAt,
        profile.createdAt,
      ]
    );
  }

  async patchMetadata(ownerId: string, profileId: string, patch: { displayName?: string; place?: string; updatedAt: number }): Promise<void> {
    const pool = await this.withPool();
    await pool.query(
      `
      UPDATE rag_profiles
      SET
        display_name = COALESCE($3, display_name),
        place = COALESCE($4, place),
        updated_at = $5
      WHERE owner_id = $1 AND profile_id = $2
      `,
      [ownerId, profileId, patch.displayName ?? null, patch.place ?? null, patch.updatedAt]
    );
  }

  async deleteByOwnerAndProfileId(ownerId: string, profileId: string): Promise<void> {
    const pool = await this.withPool();
    await pool.query(
      `DELETE FROM rag_profiles WHERE owner_id = $1 AND profile_id = $2`,
      [ownerId, profileId]
    );
  }
}

let singletonRagProfilesRepository: RagProfilesRepository | null = null;

export function getRagProfilesRepository(): RagProfilesRepository {
  if (!singletonRagProfilesRepository) {
    singletonRagProfilesRepository = new RagProfilesRepository();
  }

  return singletonRagProfilesRepository;
}
