import type { UserSubscriptionDocument } from '../models/firestoreModels.js';
import { getPostgresPool } from '../services/postgresClient.js';
import { applyPendingMigrations } from '../services/postgresMigrations.js';

interface SubscriptionRow {
  owner_id: string;
  source: 'expo_iap' | 'app_store' | 'play_store';
  entitlement_id: string;
  is_pro: boolean;
  store: string | null;
  product_id: string | null;
  event_type: string | null;
  expires_at_ms: number | null;
  updated_at: number;
  last_event_at: number;
  last_event_id: string | null;
}

function rowToDocument(row: SubscriptionRow): UserSubscriptionDocument {
  return {
    ownerId: row.owner_id,
    source: row.source,
    entitlementId: row.entitlement_id,
    isPro: row.is_pro,
    store: row.store ?? undefined,
    productId: row.product_id ?? undefined,
    eventType: row.event_type ?? undefined,
    expiresAtMs: row.expires_at_ms ?? undefined,
    updatedAt: Number(row.updated_at),
    lastEventAt: Number(row.last_event_at),
    lastEventId: row.last_event_id ?? undefined,
  };
}

export class SubscriptionsRepository {
  private async withPool() {
    const pool = getPostgresPool();
    if (!pool) {
      throw new Error('DATABASE_URL is required for subscriptions repository');
    }

    await applyPendingMigrations(pool);
    return pool;
  }

  async getByOwnerId(ownerId: string): Promise<UserSubscriptionDocument | null> {
    const pool = await this.withPool();
    const response = await pool.query<SubscriptionRow>(
      `
      SELECT owner_id, source, entitlement_id, is_pro, store, product_id, event_type, expires_at_ms, updated_at, last_event_at, last_event_id
      FROM subscriptions
      WHERE owner_id = $1
      LIMIT 1
      `,
      [ownerId]
    );

    const row = response.rows[0];
    return row ? rowToDocument(row) : null;
  }

  async upsert(subscription: UserSubscriptionDocument): Promise<void> {
    const pool = await this.withPool();
    await pool.query(
      `
      INSERT INTO subscriptions (
        owner_id,
        source,
        entitlement_id,
        is_pro,
        store,
        product_id,
        event_type,
        expires_at_ms,
        updated_at,
        last_event_at,
        last_event_id
      ) VALUES (
        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11
      )
      ON CONFLICT (owner_id)
      DO UPDATE SET
        source = EXCLUDED.source,
        entitlement_id = EXCLUDED.entitlement_id,
        is_pro = EXCLUDED.is_pro,
        store = EXCLUDED.store,
        product_id = EXCLUDED.product_id,
        event_type = EXCLUDED.event_type,
        expires_at_ms = EXCLUDED.expires_at_ms,
        updated_at = EXCLUDED.updated_at,
        last_event_at = EXCLUDED.last_event_at,
        last_event_id = EXCLUDED.last_event_id
      `,
      [
        subscription.ownerId,
        subscription.source,
        subscription.entitlementId,
        subscription.isPro,
        subscription.store ?? null,
        subscription.productId ?? null,
        subscription.eventType ?? null,
        subscription.expiresAtMs ?? null,
        subscription.updatedAt,
        subscription.lastEventAt,
        subscription.lastEventId ?? null,
      ]
    );
  }
}

let singletonSubscriptionsRepository: SubscriptionsRepository | null = null;

export function getSubscriptionsRepository(): SubscriptionsRepository {
  if (!singletonSubscriptionsRepository) {
    singletonSubscriptionsRepository = new SubscriptionsRepository();
  }

  return singletonSubscriptionsRepository;
}
