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

const BillingSourceSchema = z.enum(['expo_iap', 'app_store', 'play_store']);

const BillingSyncSchema = z.object({
  source: BillingSourceSchema.default('expo_iap'),
  entitlementId: z.string().min(1),
  isPro: z.boolean(),
  expiresAtMs: z.number().int().positive().optional(),
  store: z.string().optional(),
  productId: z.string().optional(),
  eventType: z.string().optional(),
  lastEventId: z.string().optional(),
  purchaseToken: z.string().optional(),
  transactionId: z.string().optional(),
});

export function shouldRejectUnverifiedSync(
  verification: BillingDirectVerificationResult,
  payload: z.infer<typeof BillingSyncSchema>
): boolean {
  return verification.status === 'skipped' && payload.isPro && !env.BILLING_ALLOW_UNVERIFIED_SYNC;
}

function getDefaultProEntitlementId(): string {
  return env.PRO_ENTITLEMENT_ID || 'pro';
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
          source: 'expo_iap' as const,
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
