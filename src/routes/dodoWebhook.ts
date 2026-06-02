import { Router } from 'express';
import { env } from '../config/env.js';
import { getSubscriptionsRepository } from '../repositories/subscriptionsRepository.js';
import { hasActiveProEntitlement } from '../services/subscriptionAccess.js';
import { getUsageQuotasRepository } from '../repositories/usageQuotasRepository.js';

const router = Router();

const BILLING_MONTH_MS = 30 * 24 * 60 * 60 * 1000;

router.post('/v1/billing/dodo-webhook', async (req, res) => {
  try {
    const signature = req.headers['x-dodopayments-signature'] as string | undefined;
    if (!signature) {
      return res.status(401).json({ error: 'Missing webhook signature' });
    }

    // Verify webhook signature using raw body
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
    if (!ownerId) {
      return res.status(400).json({ error: 'Missing ownerId in session metadata' });
    }

    const now = Date.now();
    const subscriptions = getSubscriptionsRepository();
    const usageQuotas = getUsageQuotasRepository();

    const billingAnchorMs = now - (now % BILLING_MONTH_MS);

    const subscriptionDoc = {
      ownerId,
      source: 'dodopayments' as const,
      entitlementId: env.PRO_ENTITLEMENT_ID,
      isPro: true,
      productId: env.DODOPAYMENTS_PRICE_ID,
      eventType: 'checkout.session.completed',
      expiresAtMs: now + BILLING_MONTH_MS,
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
