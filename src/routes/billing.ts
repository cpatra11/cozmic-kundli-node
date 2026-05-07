import { Router } from 'express';
import { z } from 'zod';
import { env } from '../config/env.js';
import { requireFirebaseAuth } from '../middleware/auth.js';
import type { UserSubscriptionDocument } from '../models/firestoreModels.js';
import { getSubscriptionsRepository } from '../repositories/subscriptionsRepository.js';
import { getUsageQuotasRepository } from '../repositories/usageQuotasRepository.js';
import { hasActiveProEntitlement } from '../services/subscriptionAccess.js';

const router = Router();

const BillingSourceSchema = z.enum(['iapkit', 'app_store', 'play_store']);

const BillingSyncSchema = z.object({
  source: BillingSourceSchema.default('iapkit'),
  entitlementId: z.string().min(1),
  isPro: z.boolean(),
  expiresAtMs: z.number().int().positive().optional(),
  store: z.string().optional(),
  productId: z.string().optional(),
  eventType: z.string().optional(),
  lastEventId: z.string().optional(),
  purchaseToken: z.string().optional(),
  transactionId: z.string().optional(),
  iapkitState: z.string().optional(),
  iapkitValid: z.boolean().optional(),
  iapkitStore: z.enum(['apple', 'google', 'unknown']).optional(),
});

function getDefaultProEntitlementId(): string {
  return env.PRO_ENTITLEMENT_ID || 'pro';
}

async function upsertSubscriptionFromSyncPayload(ownerId: string, payload: z.infer<typeof BillingSyncSchema>) {
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
    purchaseToken: payload.purchaseToken,
    transactionId: payload.transactionId,
    iapkitState: payload.iapkitState,
    iapkitValid: payload.iapkitValid,
    iapkitStore: payload.iapkitStore,
    expiresAtMs: payload.expiresAtMs,
    updatedAt: now,
    lastEventAt: now,
    lastEventId: payload.lastEventId,
  };

  await subscriptions.upsert(subscriptionDoc);
  const storedSubscription = await subscriptions.getByOwnerId(ownerId);
  const effectiveIsPro = hasActiveProEntitlement(storedSubscription);
  const quotaStatus = await usageQuotas.getQuotaStatus(ownerId, effectiveIsPro);

  return {
    subscription: storedSubscription ?? subscriptionDoc,
    quotaStatus,
  };
}

router.post('/v1/billing/subscription/sync', requireFirebaseAuth, async (req, res) => {
  try {
    const parsed = BillingSyncSchema.safeParse(req.body ?? {});
    if (!parsed.success) {
      return res.status(400).json({ error: 'Invalid billing sync payload', details: parsed.error.flatten() });
    }

    const result = await upsertSubscriptionFromSyncPayload(req.user!.uid, parsed.data);
    return res.json({
      ...result,
      verification: {
        status: 'skipped' as const,
        reason: 'IAPKit-only mode: direct store validation is disabled.',
      },
    });
  } catch (error) {
    console.error('[billing/sync] Error', error);
    return res.status(500).json({ error: 'Failed to sync billing subscription', details: String(error) });
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
          entitlementId: getDefaultProEntitlementId(),
          source: 'iapkit' as const,
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
