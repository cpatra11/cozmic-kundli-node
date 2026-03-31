import type { AuthUserDocument } from '../models/firestoreModels.js';
import { getPostgresPool } from '../services/postgresClient.js';
import { applyPendingMigrations } from '../services/postgresMigrations.js';

interface AuthUserRow {
  owner_id: string;
  email: string | null;
  phone_number: string | null;
  provider: 'firebase';
  created_at: number;
  updated_at: number;
  last_seen_at: number;
}

function rowToDocument(row: AuthUserRow): AuthUserDocument {
  return {
    ownerId: row.owner_id,
    email: row.email ?? undefined,
    phoneNumber: row.phone_number ?? undefined,
    provider: row.provider,
    createdAt: Number(row.created_at),
    updatedAt: Number(row.updated_at),
    lastSeenAt: Number(row.last_seen_at),
  };
}

export class AuthUsersRepository {
  private async withPool() {
    const pool = getPostgresPool();
    if (!pool) {
      throw new Error('DATABASE_URL is required for auth_users repository');
    }

    await applyPendingMigrations(pool);
    return pool;
  }

  async getByOwnerId(ownerId: string): Promise<AuthUserDocument | null> {
    const pool = await this.withPool();
    const response = await pool.query<AuthUserRow>(
      `
      SELECT owner_id, email, phone_number, provider, created_at, updated_at, last_seen_at
      FROM auth_users
      WHERE owner_id = $1
      LIMIT 1
      `,
      [ownerId]
    );

    const row = response.rows[0];
    return row ? rowToDocument(row) : null;
  }

  async upsert(profile: AuthUserDocument): Promise<void> {
    const pool = await this.withPool();
    await pool.query(
      `
      INSERT INTO auth_users (owner_id, email, phone_number, provider, created_at, updated_at, last_seen_at)
      VALUES ($1, $2, $3, $4, $5, $6, $7)
      ON CONFLICT (owner_id)
      DO UPDATE SET
        email = EXCLUDED.email,
        phone_number = EXCLUDED.phone_number,
        provider = EXCLUDED.provider,
        created_at = LEAST(auth_users.created_at, EXCLUDED.created_at),
        updated_at = EXCLUDED.updated_at,
        last_seen_at = EXCLUDED.last_seen_at
      `,
      [
        profile.ownerId,
        profile.email ?? null,
        profile.phoneNumber ?? null,
        profile.provider,
        profile.createdAt,
        profile.updatedAt,
        profile.lastSeenAt,
      ]
    );
  }
}

let singletonAuthUsersRepository: AuthUsersRepository | null = null;

export function getAuthUsersRepository(): AuthUsersRepository {
  if (!singletonAuthUsersRepository) {
    singletonAuthUsersRepository = new AuthUsersRepository();
  }

  return singletonAuthUsersRepository;
}
