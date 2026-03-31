import type { ChartJobDocument } from '../models/firestoreModels.js';
import { getPostgresPool } from '../services/postgresClient.js';
import { applyPendingMigrations } from '../services/postgresMigrations.js';

interface ChartJobRow {
  owner_id: string;
  job_id: string;
  profile_id: string;
  status: ChartJobDocument['status'];
  request: Record<string, unknown>;
  result: unknown;
  error: string | null;
  created_at: number;
  updated_at: number;
}

export interface ChartJobRecord {
  jobId: string;
  data: ChartJobDocument;
}

function rowToDocument(row: ChartJobRow): ChartJobDocument {
  return {
    ownerId: row.owner_id,
    profileId: row.profile_id,
    status: row.status,
    request: row.request,
    result: row.result,
    error: row.error ?? undefined,
    createdAt: Number(row.created_at),
    updatedAt: Number(row.updated_at),
  };
}

export class ChartJobsRepository {
  private async withPool() {
    const pool = getPostgresPool();
    if (!pool) {
      throw new Error('DATABASE_URL is required for chart_jobs repository');
    }

    await applyPendingMigrations(pool);
    return pool;
  }

  async create(ownerId: string, jobId: string, job: ChartJobDocument): Promise<void> {
    const pool = await this.withPool();
    await pool.query(
      `
      INSERT INTO chart_jobs (owner_id, job_id, profile_id, status, request, result, error, created_at, updated_at)
      VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, $7, $8, $9)
      `,
      [
        ownerId,
        jobId,
        job.profileId,
        job.status,
        JSON.stringify(job.request),
        job.result ? JSON.stringify(job.result) : null,
        job.error ?? null,
        job.createdAt,
        job.updatedAt,
      ]
    );
  }

  async patch(ownerId: string, jobId: string, patch: Partial<Pick<ChartJobDocument, 'profileId' | 'status' | 'result' | 'error' | 'updatedAt'>>): Promise<void> {
    const pool = await this.withPool();
    await pool.query(
      `
      UPDATE chart_jobs
      SET
        profile_id = COALESCE($3, profile_id),
        status = COALESCE($4, status),
        result = COALESCE($5::jsonb, result),
        error = COALESCE($6, error),
        updated_at = COALESCE($7, updated_at)
      WHERE owner_id = $1 AND job_id = $2
      `,
      [
        ownerId,
        jobId,
        patch.profileId ?? null,
        patch.status ?? null,
        patch.result ? JSON.stringify(patch.result) : null,
        patch.error ?? null,
        patch.updatedAt ?? null,
      ]
    );
  }

  async get(ownerId: string, jobId: string): Promise<ChartJobRecord | null> {
    const pool = await this.withPool();
    const response = await pool.query<ChartJobRow>(
      `
      SELECT owner_id, job_id, profile_id, status, request, result, error, created_at, updated_at
      FROM chart_jobs
      WHERE owner_id = $1 AND job_id = $2
      LIMIT 1
      `,
      [ownerId, jobId]
    );

    const row = response.rows[0];
    return row ? { jobId: row.job_id, data: rowToDocument(row) } : null;
  }
}

let singletonChartJobsRepository: ChartJobsRepository | null = null;

export function getChartJobsRepository(): ChartJobsRepository {
  if (!singletonChartJobsRepository) {
    singletonChartJobsRepository = new ChartJobsRepository();
  }

  return singletonChartJobsRepository;
}
