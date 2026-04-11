import { Router } from 'express';
import { z } from 'zod';
import { env } from '../config/env.js';
import { requireFirebaseAuth } from '../middleware/auth.js';
import type { UserSubscriptionDocument } from '../models/firestoreModels.js';
import { getSubscriptionsRepository } from '../repositories/subscriptionsRepository.js';
import { getUsageQuotasRepository } from '../repositories/usageQuotasRepository.js';
import { hasActiveProEntitlement } from '../services/subscriptionAccess.js';
import { verifyBillingSyncPayload } from '../services/storePurchaseValidation.js';
import type { BillingDirectVerificationResult } from '../services/storePurchaseValidation.js';

const router = Router();

const BillingSourceSchema = z.enum(['revenuecat', 'expo_iap', 'app_store', 'play_store']);

const RevenueCatEventSchema = z.object({
  event: z
    .object({
      id: z.string().optional(),
      type: z.string().optional(),
      app_user_id: z.string().optional(),
      aliases: z.array(z.string()).optional(),
      entitlement_ids: z.array(z.string()).optional(),
      expiration_at_ms: z.coerce.number().optional(),
      expires_date_ms: z.coerce.number().optional(),
      purchased_at_ms: z.coerce.number().optional(),
      event_timestamp_ms: z.coerce.number().optional(),
      product_id: z.string().optional(),
      store: z.string().optional(),
    })
    .passthrough(),
});

const RevenueCatSyncSchema = z.object({
  entitlementId: z.string().min(1),
  isPro: z.boolean(),
  expiresAtMs: z.number().int().positive().optional(),
  store: z.string().optional(),
  productId: z.string().optional(),
  eventType: z.string().optional(),
  lastEventId: z.string().optional(),
});

const BillingSyncSchema = RevenueCatSyncSchema.extend({
  source: BillingSourceSchema.default('expo_iap'),
  purchaseToken: z.string().optional(),
  transactionId: z.string().optional(),
});

export function shouldRejectUnverifiedSync(
  verification: BillingDirectVerificationResult,
  payload: z.infer<typeof BillingSyncSchema>
): boolean {
  return verification.status === 'skipped' && payload.isPro && !env.BILLING_ALLOW_UNVERIFIED_SYNC;
}

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

function normalizeOwnerId(raw?: string): string | null {
  const trimmed = raw?.trim();
  if (!trimmed) return null;
  if (trimmed.startsWith('$RCAnonymousID:')) return null;
  return trimmed;
}

function collectOwnerIds(event: { app_user_id?: string; aliases?: string[] }): string[] {
  const ids = new Set<string>();

  const primary = normalizeOwnerId(event.app_user_id);
  if (primary) ids.add(primary);

  for (const alias of event.aliases ?? []) {
    const normalized = normalizeOwnerId(alias);
    if (normalized) ids.add(normalized);
  }

  return [...ids];
}

function resolveConfiguredProEntitlementId(): string {
  return env.PRO_ENTITLEMENT_ID || env.REVENUECAT_PRO_ENTITLEMENT_ID || 'pro';
}

function resolveEntitlementId(event: { entitlement_ids?: string[] }): string {
  const expected = resolveConfiguredProEntitlementId();
  const legacy = env.REVENUECAT_PRO_ENTITLEMENT_ID;
  const entitlementIds = event.entitlement_ids ?? [];

  if (entitlementIds.includes(expected)) return expected;
  if (legacy && entitlementIds.includes(legacy)) return legacy;
  if (entitlementIds.includes('pro')) return 'pro';
  if (entitlementIds.length > 0) return entitlementIds[0]!;

  return expected;
}

function computeIsProStatus(event: {
  type?: string;
  entitlement_ids?: string[];
  expiration_at_ms?: number;
  expires_date_ms?: number;
}): { isPro: boolean; expiresAtMs?: number; hasEntitlementEvidence: boolean; revokedByType: boolean } {
  const entitlementId = resolveConfiguredProEntitlementId();
  const legacyEntitlement = env.REVENUECAT_PRO_ENTITLEMENT_ID;
  const type = String(event.type ?? '').toUpperCase();
  const entitlementIds = event.entitlement_ids ?? [];
  const expiresAtMs = event.expiration_at_ms ?? event.expires_date_ms;
  const now = Date.now();

  const hasEntitlement =
    entitlementIds.includes(entitlementId) ||
    (legacyEntitlement ? entitlementIds.includes(legacyEntitlement) : false) ||
    entitlementIds.includes('pro');
  const hasEntitlementEvidence = entitlementIds.length > 0;
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
    hasEntitlementEvidence,
    revokedByType,
  };
}

async function upsertSubscriptionFromSyncPayload(
  ownerId: string,
  payload: z.infer<typeof BillingSyncSchema>
) {
  const subscriptions = getSubscriptionsRepository();
  const usageQuotas = getUsageQuotasRepository();
  const now = Date.now();

  const subscriptionDoc: UserSubscriptionDocument = {
    ownerId,
    source: payload.source,
    entitlementId: payload.entitlementId,
    isPro: payload.isPro,
    store: payload.store,
    productId: payload.productId,
    eventType: payload.eventType ?? 'client_sync',
    expiresAtMs: payload.expiresAtMs,
    updatedAt: now,
    lastEventAt: now,
    lastEventId: payload.lastEventId,
  };

  await subscriptions.upsert(subscriptionDoc);
  const quotaStatus = await usageQuotas.getQuotaStatus(ownerId, payload.isPro);

  return {
    subscription: subscriptionDoc,
    quotaStatus,
  };
}

router.post('/v1/billing/subscription/sync', requireFirebaseAuth, async (req, res) => {
  try {
    const parsed = BillingSyncSchema.safeParse(req.body ?? {});
    if (!parsed.success) {
      return res.status(400).json({ error: 'Invalid billing sync payload', details: parsed.error.flatten() });
    }

    const verification = await verifyBillingSyncPayload(parsed.data);
    if (verification.status === 'invalid') {
      return res.status(400).json({
        error: 'Store purchase verification failed',
        provider: verification.provider,
        details: verification.reason,
      });
    }

    if (shouldRejectUnverifiedSync(verification, parsed.data)) {
      const skippedReason = verification.status === 'skipped' ? verification.reason : 'Store verification was skipped';
      return res.status(400).json({
        error: 'Unverified pro entitlement sync is not allowed',
        details: skippedReason,
        hint: 'Configure direct store validation credentials or set BILLING_ALLOW_UNVERIFIED_SYNC=true for local development only.',
      });
    }

    const normalizedPayload: z.infer<typeof BillingSyncSchema> =
      verification.status === 'verified'
        ? {
            ...parsed.data,
            source: verification.provider,
            isPro: verification.isPro,
            expiresAtMs: verification.expiresAtMs ?? parsed.data.expiresAtMs,
            productId: verification.productId ?? parsed.data.productId,
            store: verification.store,
            eventType: verification.eventType,
            lastEventId: verification.lastEventId ?? parsed.data.lastEventId,
          }
        : parsed.data;

    const result = await upsertSubscriptionFromSyncPayload(req.user!.uid, normalizedPayload);
    return res.json({
      ...result,
      verification,
    });
  } catch (error) {
    return res.status(500).json({ error: 'Failed to sync billing subscription', details: String(error) });
  }
});

router.post('/v1/billing/revenuecat/sync', requireFirebaseAuth, async (req, res) => {
  try {
    const parsed = RevenueCatSyncSchema.safeParse(req.body ?? {});
    if (!parsed.success) {
      return res.status(400).json({ error: 'Invalid sync payload', details: parsed.error.flatten() });
    }

    const payload = {
      ...parsed.data,
      source: 'revenuecat' as const,
    };

    const result = await upsertSubscriptionFromSyncPayload(req.user!.uid, payload);
    return res.json(result);
  } catch (error) {
    return res.status(500).json({ error: 'Failed to sync RevenueCat subscription', details: String(error) });
  }
});

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
    const ownerIds = collectOwnerIds(event);

    if (ownerIds.length === 0) {
      return res.status(202).json({ accepted: true, ignored: 'Missing app_user_id/aliases' });
    }

    const subscriptions = getSubscriptionsRepository();
    const now = Date.now();
    const status = computeIsProStatus(event);

    for (const ownerId of ownerIds) {
      const existing = await subscriptions.getByOwnerId(ownerId);

      let nextIsPro = status.isPro;
      if (!status.hasEntitlementEvidence && !status.revokedByType && existing) {
        // Some event payloads can omit entitlement_ids. In that case, avoid accidental downgrades.
        nextIsPro = hasActiveProEntitlement(existing, now);
      }

      const subscriptionDoc: UserSubscriptionDocument = {
        ownerId,
        source: 'revenuecat',
        entitlementId: resolveEntitlementId(event),
        isPro: nextIsPro,
        store: event.store,
        productId: event.product_id,
        eventType: event.type,
        expiresAtMs: status.expiresAtMs,
        updatedAt: now,
        lastEventAt: event.event_timestamp_ms ?? event.purchased_at_ms ?? now,
        lastEventId: event.id,
      };

      await subscriptions.upsert(subscriptionDoc);
    }

    return res.status(200).json({ ok: true, updatedOwners: ownerIds.length });
  } catch (error) {
    return res.status(500).json({ error: 'Failed to process RevenueCat webhook', details: String(error) });
  }
});

router.get('/v1/billing/subscription', requireFirebaseAuth, async (req, res) => {
  try {
    const subscriptions = getSubscriptionsRepository();
    const usageQuotas = getUsageQuotasRepository();
    const ownerId = req.user!.uid;
    const doc = await subscriptions.getByOwnerId(ownerId);
    const hasPro = hasActiveProEntitlement(doc);
    const quotaStatus = await usageQuotas.getQuotaStatus(ownerId, hasPro);

    if (!doc) {
      return res.json({
        subscription: {
          isPro: false,
          entitlementId: resolveConfiguredProEntitlementId(),
        },
        quotaStatus,
      });
    }

    return res.json({
      subscription: {
        ...doc,
        isPro: hasPro,
      },
      quotaStatus,
    });
  } catch (error) {
    return res.status(500).json({ error: 'Failed to load subscription', details: String(error) });
  }
});

router.get('/v1/billing/quota-status', requireFirebaseAuth, async (req, res) => {
  try {
    const subscriptions = getSubscriptionsRepository();
    const usageQuotas = getUsageQuotasRepository();
    const ownerId = req.user!.uid;
    const subscription = await subscriptions.getByOwnerId(ownerId);
    const hasPro = hasActiveProEntitlement(subscription);
    const quotaStatus = await usageQuotas.getQuotaStatus(ownerId, hasPro);

    return res.json({ quotaStatus });
  } catch (error) {
    return res.status(500).json({ error: 'Failed to load quota status', details: String(error) });
  }
});

export default router;
