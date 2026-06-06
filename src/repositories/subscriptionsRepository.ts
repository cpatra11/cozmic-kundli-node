import type { UserSubscriptionDocument } from '../models/firestoreModels.js';
import { getPostgresPool } from '../services/postgresClient.js';
import { applyPendingMigrations } from '../services/postgresMigrations.js';

interface SubscriptionRow {
  owner_id: string;
  source: 'iapkit' | 'app_store' | 'play_store' | 'dodopayments';
  entitlement_id: string;
  is_pro: boolean;
  store: string | null;
  product_id: string | null;
  event_type: string | null;
  purchase_token: string | null;
  transaction_id: string | null;
  original_transaction_id: string | null;
  iapkit_state: string | null;
  iapkit_valid: boolean | null;
  iapkit_store: 'apple' | 'google' | 'unknown' | null;
  expires_at_ms: number | null;
  billing_anchor_ms: number | null;
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
    originalTransactionId: row.original_transaction_id ?? undefined,
    iapkitState: row.iapkit_state ?? undefined,
    iapkitValid: typeof row.iapkit_valid === 'boolean' ? row.iapkit_valid : undefined,
    iapkitStore: row.iapkit_store ?? undefined,
    expiresAtMs: row.expires_at_ms ?? undefined,
    billingAnchorMs: row.billing_anchor_ms ?? undefined,
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
      SELECT owner_id, source, entitlement_id, is_pro, store, product_id, event_type, purchase_token, transaction_id, original_transaction_id, iapkit_state, iapkit_valid, iapkit_store, expires_at_ms, billing_anchor_ms, updated_at, last_event_at, last_event_id
      FROM subscriptions
      WHERE owner_id = $1
      LIMIT 1
      `,
      [ownerId]
    );

    const row = response.rows[0];
    return row ? rowToDocument(row) : null;
  }

  async getByOriginalTransactionId(originalTransactionId: string): Promise<UserSubscriptionDocument | null> {
    const pool = await this.withPool();
    const response = await pool.query<SubscriptionRow>(
      `
      SELECT owner_id, source, entitlement_id, is_pro, store, product_id, event_type, purchase_token, transaction_id, original_transaction_id, iapkit_state, iapkit_valid, iapkit_store, expires_at_ms, billing_anchor_ms, updated_at, last_event_at, last_event_id
      FROM subscriptions
      WHERE original_transaction_id = $1
      LIMIT 1
      `,
      [originalTransactionId]
    );

    const row = response.rows[0];
    return row ? rowToDocument(row) : null;
  }

  async updateFromAppleWebhook(ownerId: string, changes: {
    isPro: boolean;
    eventType: string;
    expiresAtMs?: number;
    transactionId?: string;
  }): Promise<void> {
    const pool = await this.withPool();

    const existingResponse = await pool.query<{ event_type: string | null }>(
      `SELECT event_type FROM subscriptions WHERE owner_id = $1 LIMIT 1`,
      [ownerId]
    );

    const existingEventType = existingResponse.rows[0]?.event_type ?? null;
    if (existingEventType === 'admin_revoke' && !changes.isPro) {
      return;
    }

    const now = Date.now();

    const setClauses: string[] = [];
    const params: (string | number | boolean | null)[] = [];
    let paramIndex = 1;

    setClauses.push(`is_pro = $${paramIndex++}`);
    params.push(changes.isPro);

    setClauses.push(`event_type = $${paramIndex++}`);
    params.push(changes.eventType);

    setClauses.push(`updated_at = $${paramIndex++}`);
    params.push(now);

    setClauses.push(`last_event_at = $${paramIndex++}`);
    params.push(now);

    if (changes.expiresAtMs !== undefined) {
      setClauses.push(`expires_at_ms = $${paramIndex++}`);
      params.push(changes.expiresAtMs);
    }

    if (changes.transactionId !== undefined) {
      setClauses.push(`transaction_id = $${paramIndex++}`);
      params.push(changes.transactionId);
    }

    params.push(ownerId);

    await pool.query(
      `UPDATE subscriptions SET ${setClauses.join(', ')} WHERE owner_id = $${paramIndex}`,
      params
    );
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
    const isTrustedEvent =
      subscription.eventType === 'iapkit_verified' ||
      subscription.eventType?.startsWith('webhook_') ||
      subscription.source === 'dodopayments';
    const shouldSkipUpdate = existingEventType === 'admin_revoke' &&
      subscription.eventType !== 'admin_revoke' &&
      !isTrustedEvent;
    
    if (shouldSkipUpdate) {
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
        original_transaction_id,
        iapkit_state,
        iapkit_valid,
        iapkit_store,
        expires_at_ms,
        billing_anchor_ms,
        updated_at,
        last_event_at,
        last_event_id
      ) VALUES (
        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18
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
        original_transaction_id = CASE
          WHEN EXCLUDED.original_transaction_id IS NOT NULL THEN EXCLUDED.original_transaction_id
          ELSE subscriptions.original_transaction_id
        END,
        iapkit_state = EXCLUDED.iapkit_state,
        iapkit_valid = EXCLUDED.iapkit_valid,
        iapkit_store = EXCLUDED.iapkit_store,
        expires_at_ms = EXCLUDED.expires_at_ms,
        billing_anchor_ms = EXCLUDED.billing_anchor_ms,
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
        subscription.originalTransactionId ?? null,
        subscription.iapkitState ?? null,
        typeof subscription.iapkitValid === 'boolean' ? subscription.iapkitValid : null,
        subscription.iapkitStore ?? null,
        subscription.expiresAtMs ?? null,
        subscription.billingAnchorMs ?? null,
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
