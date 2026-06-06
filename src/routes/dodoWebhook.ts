import { Router } from 'express';
import { z } from 'zod';
import { env, dodoApiBaseUrl } from '../config/env.js';
import { createHmac, timingSafeEqual } from 'crypto';
import { requireFirebaseAuth } from '../middleware/auth.js';
import { getSubscriptionsRepository } from '../repositories/subscriptionsRepository.js';
import { hasActiveProEntitlement } from '../services/subscriptionAccess.js';
import { getUsageQuotasRepository } from '../repositories/usageQuotasRepository.js';

const router = Router();

const BILLING_MONTH_MS = 30 * 24 * 60 * 60 * 1000;

const DODO_PRODUCT_MAP: Record<string, string | undefined> = {
  cozmic_pro_monthly: env.DODOPAYMENTS_PRICE_MONTHLY,
  cozmic_pro_yearly: env.DODOPAYMENTS_PRICE_YEARLY,
};

const PLAN_DURATIONS: Record<string, number> = {
  cozmic_pro_monthly: 30 * 24 * 60 * 60 * 1000,
  cozmic_pro_yearly: 365 * 24 * 60 * 60 * 1000,
};

const REVERSE_PRICE_MAP: Record<string, string> = {};
for (const [planId, priceId] of Object.entries(DODO_PRODUCT_MAP)) {
  if (priceId) REVERSE_PRICE_MAP[priceId] = planId;
}

router.post('/v1/billing/dodo-checkout', requireFirebaseAuth, async (req, res) => {
  try {
    const { planId } = z.object({ planId: z.string().min(1) }).parse(req.body);
    const ownerId = req.user!.uid;

    if (!env.DODOPAYMENTS_API_KEY) {
      return res.status(500).json({ error: 'DodoPayments API key not configured' });
    }

    const dodoProductId = DODO_PRODUCT_MAP[planId];
    if (!dodoProductId) {
      return res.status(400).json({ error: 'Invalid plan ID' });
    }

    const baseUrl = env.NODE_ENV === 'production'
      ? 'https://cozmicastro.one'
      : 'http://localhost:8081';

    const response = await fetch(`${dodoApiBaseUrl()}/checkouts`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${env.DODOPAYMENTS_API_KEY}`,
      },
      body: JSON.stringify({
        product_cart: [
          {
            product_id: dodoProductId,
            quantity: 1,
          },
        ],
        return_url: `${baseUrl}/pro-success`,
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
    return res.json({ url: session.checkout_url });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return res.status(400).json({ error: 'Invalid request', details: error.flatten() });
    }
    return res.status(500).json({ error: 'Failed to create checkout', details: String(error) });
  }
});

router.post('/v1/billing/dodo-webhook', async (req, res) => {
  try {
    const webhookId = req.headers['webhook-id'] as string | undefined;
    const webhookTimestamp = req.headers['webhook-timestamp'] as string | undefined;
    const webhookSignature = req.headers['webhook-signature'] as string | undefined;

    if (!webhookId || !webhookTimestamp || !webhookSignature) {
      return res.status(401).json({ error: 'Missing webhook headers' });
    }

    const now = Math.floor(Date.now() / 1000);
    const ts = parseInt(webhookTimestamp, 10);
    if (isNaN(ts) || Math.abs(now - ts) > 300) {
      return res.status(401).json({ error: 'Webhook timestamp out of range' });
    }

    const secret = env.DODOPAYMENTS_WEBHOOK_SECRET;
    if (!secret) {
      return res.status(500).json({ error: 'Webhook secret not configured' });
    }

    const rawBody = req.body instanceof Buffer ? req.body : Buffer.from(JSON.stringify(req.body));
    const signedContent = `${webhookId}.${webhookTimestamp}.${rawBody.toString('utf8')}`;

    const rawSecret = secret.startsWith('whsec_') ? secret.slice(6) : secret;
    const keyBytes = Buffer.from(rawSecret, 'base64');

    const signatures = webhookSignature.split(' ').map(s => {
      const sep = s.indexOf(',');
      if (sep === -1) return null;
      return { version: s.slice(0, sep), value: s.slice(sep + 1) };
    }).filter(Boolean) as { version: string; value: string }[];

    let valid = false;
    for (const sig of signatures) {
      if (sig.version !== 'v1') continue;
      const expected = createHmac('sha256', keyBytes).update(signedContent).digest('base64');
      if (constantTimeEqual(expected, sig.value)) {
        valid = true;
        break;
      }
    }

    if (!valid) {
      return res.status(401).json({ error: 'Invalid webhook signature' });
    }

    const event = JSON.parse(rawBody.toString('utf8'));
    const eventType: string = event.type || event.event_type || '';
    const data = event.data || {};

    console.log('[dodo-webhook] event_type=%s ownerId=%s topKeys=%j', eventType, data.metadata?.ownerId, Object.keys(event).slice(0, 10));

    switch (eventType) {
      case 'payment.succeeded':
        await handleCheckoutCompleted(data);
        break;
      case 'subscription.active':
      case 'subscription.renewed':
        await handleSubscriptionActive(data);
        break;
      case 'subscription.cancelled':
        await handleSubscriptionCancelled(data);
        break;
      case 'subscription.expired':
        await handleSubscriptionExpired(data);
        break;
      case 'subscription.failed':
        await handleSubscriptionFailed(data);
        break;
      case 'subscription.updated':
        await handleSubscriptionUpdated(data);
        break;
      default:
        break;
    }

    return res.json({ received: true });
  } catch (error) {
    return res.status(500).json({ error: 'Webhook processing failed', details: String(error) });
  }
});

async function upsertFromDodoData(data: {
  ownerId: string;
  planId?: string;
  isPro: boolean;
  expiresAtMs: number;
  eventType: string;
  lastEventId: string;
  subscriptionId?: string;
  customerId?: string;
  priceId?: string;
}) {
  const now = Date.now();
  const subscriptions = getSubscriptionsRepository();
  const usageQuotas = getUsageQuotasRepository();
  const billingAnchorMs = now - (now % BILLING_MONTH_MS);

  const subscriptionDoc = {
    ownerId: data.ownerId,
    source: 'dodopayments' as const,
    entitlementId: env.PRO_ENTITLEMENT_ID,
    isPro: data.isPro,
    store: 'dodopayments',
    productId: data.priceId || data.planId || env.DODOPAYMENTS_PRICE_MONTHLY,
    eventType: data.eventType,
    transactionId: data.subscriptionId,
    purchaseToken: data.customerId,
    expiresAtMs: data.expiresAtMs,
    billingAnchorMs,
    updatedAt: now,
    lastEventAt: now,
    lastEventId: data.lastEventId,
  };

  console.log('[dodo-webhook] upsert: ownerId=%s isPro=%s expiresAtMs=%s', data.ownerId, data.isPro, data.expiresAtMs);
  await subscriptions.upsert(subscriptionDoc);
  const storedSubscription = await subscriptions.getByOwnerId(data.ownerId);
  const hasPro = hasActiveProEntitlement(storedSubscription);
  console.log('[dodo-webhook] upsert done: ownerId=%s hasPro=%s', data.ownerId, hasPro);
  await usageQuotas.getQuotaStatus(data.ownerId, hasPro, billingAnchorMs);
}

async function handleCheckoutCompleted(data: any) {
  const ownerId = data.metadata?.ownerId;
  const planId = data.metadata?.planId;

  if (!ownerId) {
    console.log('[dodo-webhook] handleCheckoutCompleted: no ownerId in metadata, dataKeys=%j', Object.keys(data));
    return;
  }

  const duration = planId && PLAN_DURATIONS[planId] ? PLAN_DURATIONS[planId] : BILLING_MONTH_MS;
  const now = Date.now();

  console.log('[dodo-webhook] handleCheckoutCompleted: ownerId=%s planId=%s', ownerId, planId);
  await upsertFromDodoData({
    ownerId,
    planId,
    isPro: true,
    expiresAtMs: now + duration,
    eventType: 'payment.succeeded',
    lastEventId: data.id || data.subscription_id || '',
    subscriptionId: data.subscription_id,
    customerId: data.customer_id,
    priceId: data.price_id,
  });
}

async function handleSubscriptionActive(data: any) {
  const ownerId = data.metadata?.ownerId;
  if (!ownerId) {
    return;
  }

  const periodEndMs = data.current_period_end
    ? typeof data.current_period_end === 'number'
      ? data.current_period_end * 1000
      : new Date(data.current_period_end).getTime()
    : Date.now() + BILLING_MONTH_MS;

  await upsertFromDodoData({
    ownerId,
    isPro: true,
    expiresAtMs: periodEndMs,
    eventType: 'subscription.active',
    lastEventId: data.id || '',
    subscriptionId: data.id,
    customerId: data.customer_id,
    priceId: data.price_id,
  });
}

async function handleSubscriptionCancelled(data: any) {
  const ownerId = data.metadata?.ownerId;
  if (!ownerId) {
    return;
  }

  const periodEndMs = data.current_period_end
    ? typeof data.current_period_end === 'number'
      ? data.current_period_end * 1000
      : new Date(data.current_period_end).getTime()
    : Date.now();

  await upsertFromDodoData({
    ownerId,
    isPro: true,
    expiresAtMs: periodEndMs,
    eventType: 'subscription.cancelled',
    lastEventId: data.id || '',
    subscriptionId: data.id,
    customerId: data.customer_id,
    priceId: data.price_id,
  });
}

async function handleSubscriptionExpired(data: any) {
  const ownerId = data.metadata?.ownerId;
  if (!ownerId) {
    return;
  }

  const now = Date.now();

  await upsertFromDodoData({
    ownerId,
    isPro: false,
    expiresAtMs: now,
    eventType: 'subscription.expired',
    lastEventId: data.id || '',
    subscriptionId: data.id,
    customerId: data.customer_id,
    priceId: data.price_id,
  });
}

async function handleSubscriptionFailed(data: any) {
  const ownerId = data.metadata?.ownerId;
  if (!ownerId) {
    return;
  }

  await upsertFromDodoData({
    ownerId,
    isPro: false,
    expiresAtMs: Date.now(),
    eventType: 'subscription.failed',
    lastEventId: data.id || '',
    subscriptionId: data.id,
    customerId: data.customer_id,
    priceId: data.price_id,
  });
}

async function handleSubscriptionUpdated(data: any) {
  const ownerId = data.metadata?.ownerId;
  if (!ownerId) {
    return;
  }

  const periodEndMs = data.current_period_end
    ? typeof data.current_period_end === 'number'
      ? data.current_period_end * 1000
      : new Date(data.current_period_end).getTime()
    : undefined;

  const isActive = data.status === 'active' || data.status === 'trialing';

  await upsertFromDodoData({
    ownerId,
    isPro: isActive,
    expiresAtMs: periodEndMs || Date.now() + BILLING_MONTH_MS,
    eventType: 'subscription.updated',
    lastEventId: data.id || '',
    subscriptionId: data.id,
    customerId: data.customer_id,
    priceId: data.price_id,
  });
}

function constantTimeEqual(a: string, b: string): boolean {
  const bufA = Buffer.from(a);
  const bufB = Buffer.from(b);
  if (bufA.length !== bufB.length) {
    timingSafeEqual(bufA, bufA);
    return false;
  }
  return timingSafeEqual(bufA, bufB);
}

export default router;
