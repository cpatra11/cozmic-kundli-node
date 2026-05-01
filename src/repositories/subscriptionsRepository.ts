import type { UserSubscriptionDocument } from '../models/firestoreModels.js';
import { getPostgresPool } from '../services/postgresClient.js';
import { applyPendingMigrations } from '../services/postgresMigrations.js';

interface SubscriptionRow {
  owner_id: string;
  source: 'iapkit' | 'app_store' | 'play_store';
  entitlement_id: string;
  is_pro: boolean;
  store: string | null;
  product_id: string | null;
  event_type: string | null;
  purchase_token: string | null;
  transaction_id: string | null;
  iapkit_state: string | null;
  iapkit_valid: boolean | null;
  iapkit_store: 'apple' | 'google' | 'unknown' | null;
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
    purchaseToken: row.purchase_token ?? undefined,
    transactionId: row.transaction_id ?? undefined,
    iapkitState: row.iapkit_state ?? undefined,
    iapkitValid: typeof row.iapkit_valid === 'boolean' ? row.iapkit_valid : undefined,
    iapkitStore: row.iapkit_store ?? undefined,
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
      SELECT owner_id, source, entitlement_id, is_pro, store, product_id, event_type, purchase_token, transaction_id, iapkit_state, iapkit_valid, iapkit_store, expires_at_ms, updated_at, last_event_at, last_event_id
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

    const existingResponse = await pool.query<{ event_type: string | null }>(
      `
      SELECT event_type
      FROM subscriptions
      WHERE owner_id = $1
      LIMIT 1
      `,
      [subscription.ownerId]
    );

    const existingEventType = existingResponse.rows[0]?.event_type ?? null;
    // Allow purchase_update and other valid events to override admin_revoke
    // when iapkitValid is true (verified purchase) OR when isPro is true (device has active subscription)
    const shouldSkipUpdate = existingEventType === 'admin_revoke' && 
      subscription.eventType !== 'admin_revoke' &&
      subscription.iapkitValid !== true &&
      subscription.isPro !== true;
    
    if (shouldSkipUpdate) {
      console.log('[subscriptions] Skipping update due to admin_revoke', {
        ownerId: subscription.ownerId,
        existingEventType,
        newEventType: subscription.eventType,
        iapkitValid: subscription.iapkitValid,
      });
      return;
    }

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
        purchase_token,
        transaction_id,
        iapkit_state,
        iapkit_valid,
        iapkit_store,
        expires_at_ms,
        updated_at,
        last_event_at,
        last_event_id
      ) VALUES (
        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16
      )
      ON CONFLICT (owner_id)
      DO UPDATE SET
        source = EXCLUDED.source,
        entitlement_id = EXCLUDED.entitlement_id,
        is_pro = EXCLUDED.is_pro,
        store = EXCLUDED.store,
        product_id = EXCLUDED.product_id,
        event_type = EXCLUDED.event_type,
        purchase_token = EXCLUDED.purchase_token,
        transaction_id = EXCLUDED.transaction_id,
        iapkit_state = EXCLUDED.iapkit_state,
        iapkit_valid = EXCLUDED.iapkit_valid,
        iapkit_store = EXCLUDED.iapkit_store,
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
        subscription.purchaseToken ?? null,
        subscription.transactionId ?? null,
        subscription.iapkitState ?? null,
        typeof subscription.iapkitValid === 'boolean' ? subscription.iapkitValid : null,
        subscription.iapkitStore ?? null,
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
