import { Router } from 'express';
import { z } from 'zod';
import { env } from '../config/env.js';
import { requireFirebaseAuth } from '../middleware/auth.js';
import { getSubscriptionsRepository } from '../repositories/subscriptionsRepository.js';
import { hasActiveProEntitlement } from '../services/subscriptionAccess.js';
import { getUsageQuotasRepository } from '../repositories/usageQuotasRepository.js';

const router = Router();

const BILLING_MONTH_MS = 30 * 24 * 60 * 60 * 1000;

// Plan ID → DodoPayments price ID mapping
const DODO_PRICE_MAP: Record<string, string | undefined> = {
  cozmic_pro_monthly: env.DODOPAYMENTS_PRICE_MONTHLY,
  cozmic_pro_yearly: env.DODOPAYMENTS_PRICE_YEARLY,
};

// Plan duration mapping (in ms)
const PLAN_DURATIONS: Record<string, number> = {
  cozmic_pro_monthly: 30 * 24 * 60 * 60 * 1000,
  cozmic_pro_yearly: 365 * 24 * 60 * 60 * 1000,
};

// Create a DodoPayments checkout session
router.post('/v1/billing/dodo-checkout', requireFirebaseAuth, async (req, res) => {
  try {
    const { planId } = z.object({ planId: z.string().min(1) }).parse(req.body);
    const ownerId = req.user!.uid;

    if (!env.DODOPAYMENTS_API_KEY) {
      return res.status(500).json({ error: 'DodoPayments API key not configured' });
    }

    const dodoPriceId = DODO_PRICE_MAP[planId];
    if (!dodoPriceId) {
      return res.status(400).json({ error: 'Invalid plan ID' });
    }

    const baseUrl = env.NODE_ENV === 'production'
      ? 'https://cozmicastro.one'
      : 'http://localhost:8081';

    const response = await fetch('https://api.dodopayments.com/v1/checkout_sessions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${env.DODOPAYMENTS_API_KEY}`,
      },
      body: JSON.stringify({
        price_id: dodoPriceId,
        success_url: `${baseUrl}/pro-success`,
        cancel_url: `${baseUrl}/post-kundli-paywall`,
        metadata: {
          ownerId,
          planId,
        },
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      return res.status(502).json({ error: 'Failed to create checkout session', details: errorText });
    }

    const session = await response.json();
    return res.json({ url: session.url });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return res.status(400).json({ error: 'Invalid request', details: error.flatten() });
    }
    return res.status(500).json({ error: 'Failed to create checkout', details: String(error) });
  }
});

// DodoPayments webhook — processes checkout.completed events
router.post('/v1/billing/dodo-webhook', async (req, res) => {
  try {
    const signature = req.headers['x-dodopayments-signature'] as string | undefined;
    if (!signature) {
      return res.status(401).json({ error: 'Missing webhook signature' });
    }

    const secret = env.DODOPAYMENTS_WEBHOOK_SECRET;
    if (!secret) {
      return res.status(500).json({ error: 'Webhook secret not configured' });
    }

    const rawBody = req.body instanceof Buffer ? req.body : Buffer.from(JSON.stringify(req.body));
    const encoder = new TextEncoder();
    const key = await crypto.subtle.importKey(
      'raw',
      encoder.encode(secret),
      { name: 'HMAC', hash: 'SHA-256' },
      false,
      ['verify']
    );
    const sigBytes = hexToBytes(signature) as unknown as BufferSource;
    const valid = await crypto.subtle.verify('HMAC', key, sigBytes, rawBody as unknown as BufferSource);

    if (!valid) {
      return res.status(401).json({ error: 'Invalid webhook signature' });
    }

    const event = JSON.parse(rawBody.toString('utf8'));
    if (event.type !== 'checkout.session.completed') {
      return res.json({ received: true });
    }

    const session = event.data;
    const ownerId = session.metadata?.ownerId;
    const planId = session.metadata?.planId;

    if (!ownerId) {
      return res.status(400).json({ error: 'Missing ownerId in session metadata' });
    }

    const duration = planId && PLAN_DURATIONS[planId] ? PLAN_DURATIONS[planId] : BILLING_MONTH_MS;
    const now = Date.now();
    const subscriptions = getSubscriptionsRepository();
    const usageQuotas = getUsageQuotasRepository();
    const billingAnchorMs = now - (now % BILLING_MONTH_MS);

    const subscriptionDoc = {
      ownerId,
      source: 'dodopayments' as const,
      entitlementId: env.PRO_ENTITLEMENT_ID,
      isPro: true,
      productId: planId || env.DODOPAYMENTS_PRICE_MONTHLY,
      eventType: 'checkout.session.completed',
      expiresAtMs: now + duration,
      billingAnchorMs,
      updatedAt: now,
      lastEventAt: now,
      lastEventId: session.id,
    };

    await subscriptions.upsert(subscriptionDoc);
    const storedSubscription = await subscriptions.getByOwnerId(ownerId);
    const hasPro = hasActiveProEntitlement(storedSubscription);
    await usageQuotas.getQuotaStatus(ownerId, hasPro, billingAnchorMs);

    return res.json({ received: true });
  } catch (error) {
    return res.status(500).json({ error: 'Webhook processing failed', details: String(error) });
  }
});

function hexToBytes(hex: string): Buffer {
  const bytes = Buffer.alloc(hex.length / 2);
  for (let i = 0; i < hex.length; i += 2) {
    bytes[i / 2] = parseInt(hex.substring(i, i + 2), 16);
  }
  return bytes;
}

export default router;
