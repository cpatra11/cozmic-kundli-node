import { Router } from 'express';
import { z } from 'zod';
import { env } from '../config/env.js';
import { requireFirebaseAuth } from '../middleware/auth.js';
import { COLLECTIONS, type UserSubscriptionDocument } from '../models/firestoreModels.js';
import { getPostgresStore } from '../services/postgresStore.js';

const router = Router();

const RevenueCatEventSchema = z.object({
  event: z
    .object({
      id: z.string().optional(),
      type: z.string().optional(),
      app_user_id: z.string().optional(),
      aliases: z.array(z.string()).optional(),
      entitlement_ids: z.array(z.string()).optional(),
      expiration_at_ms: z.number().optional(),
      expires_date_ms: z.number().optional(),
      purchased_at_ms: z.number().optional(),
      event_timestamp_ms: z.number().optional(),
      product_id: z.string().optional(),
      store: z.string().optional(),
    })
    .passthrough(),
});

function isWebhookAuthorized(headers: Record<string, string | string[] | undefined>): boolean {
  const expected = env.REVENUECAT_WEBHOOK_SECRET?.trim();
  if (!expected) return true;

  const authorization = headers.authorization;
  const providedBearer =
    typeof authorization === 'string' && authorization.startsWith('Bearer ')
      ? authorization.slice('Bearer '.length).trim()
      : undefined;

  const providedDirect = typeof headers['x-webhook-secret'] === 'string' ? headers['x-webhook-secret'] : undefined;

  return providedBearer === expected || providedDirect === expected;
}

function computeIsProStatus(event: {
  type?: string;
  entitlement_ids?: string[];
  expiration_at_ms?: number;
  expires_date_ms?: number;
}): { isPro: boolean; expiresAtMs?: number } {
  const entitlementId = env.REVENUECAT_PRO_ENTITLEMENT_ID;
  const type = String(event.type ?? '').toUpperCase();
  const entitlementIds = event.entitlement_ids ?? [];
  const expiresAtMs = event.expiration_at_ms ?? event.expires_date_ms;
  const now = Date.now();

  const hasEntitlement = entitlementIds.includes(entitlementId) || entitlementIds.includes('pro');
  const notExpired = !expiresAtMs || expiresAtMs > now;
  const revokedByType =
    type.includes('EXPIR') ||
    type.includes('CANCEL') ||
    type.includes('REFUND') ||
    type.includes('REVOKE') ||
    type.includes('BILLING_ISSUE');

  return {
    isPro: hasEntitlement && notExpired && !revokedByType,
    expiresAtMs,
  };
}

router.post('/v1/billing/revenuecat/webhook', async (req, res) => {
  try {
    if (!isWebhookAuthorized(req.headers)) {
      return res.status(401).json({ error: 'Unauthorized webhook signature' });
    }

    const parsed = RevenueCatEventSchema.safeParse(req.body ?? {});
    if (!parsed.success) {
      return res.status(400).json({ error: 'Invalid webhook payload', details: parsed.error.flatten() });
    }

    const event = parsed.data.event;
    const ownerId = event.app_user_id?.trim();

    if (!ownerId) {
      return res.status(202).json({ accepted: true, ignored: 'Missing app_user_id' });
    }

    const store = getPostgresStore();
    const now = Date.now();
    const status = computeIsProStatus(event);

    const subscriptionDoc: UserSubscriptionDocument = {
      ownerId,
      source: 'revenuecat',
      entitlementId: env.REVENUECAT_PRO_ENTITLEMENT_ID,
      isPro: status.isPro,
      store: event.store,
      productId: event.product_id,
      eventType: event.type,
      expiresAtMs: status.expiresAtMs,
      updatedAt: now,
      lastEventAt: event.event_timestamp_ms ?? event.purchased_at_ms ?? now,
      lastEventId: event.id,
    };

    await store.setDocument(`${COLLECTIONS.userSubscriptions}/${ownerId}`, subscriptionDoc, true);

    return res.status(200).json({ ok: true });
  } catch (error) {
    return res.status(500).json({ error: 'Failed to process RevenueCat webhook', details: String(error) });
  }
});

router.get('/v1/billing/subscription', requireFirebaseAuth, async (req, res) => {
  try {
    const store = getPostgresStore();
    const ownerId = req.user!.uid;
    const doc = await store.getDocument<UserSubscriptionDocument>(`${COLLECTIONS.userSubscriptions}/${ownerId}`);

    if (!doc) {
      return res.json({
        subscription: {
          isPro: false,
          entitlementId: env.REVENUECAT_PRO_ENTITLEMENT_ID,
        },
      });
    }

    return res.json({ subscription: doc.data });
  } catch (error) {
    return res.status(500).json({ error: 'Failed to load subscription', details: String(error) });
  }
});

export default router;
