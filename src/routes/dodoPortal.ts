import { Router } from 'express';
import { env, dodoApiBaseUrl } from '../config/env.js';
import { requireFirebaseAuth } from '../middleware/auth.js';
import { getSubscriptionsRepository } from '../repositories/subscriptionsRepository.js';

const router = Router();

router.post('/v1/billing/dodo-portal', requireFirebaseAuth, async (req, res) => {
  try {
    const ownerId = req.user!.uid;

    if (!env.DODOPAYMENTS_API_KEY) {
      return res.status(500).json({ error: 'DodoPayments API key not configured' });
    }

    const subscriptions = getSubscriptionsRepository();
    const subscription = await subscriptions.getByOwnerId(ownerId);

    const customerId = subscription?.purchaseToken || subscription?.transactionId || null;

    if (!customerId) {
      return res.status(404).json({ error: 'No DodoPayments customer found for this user' });
    }

    const baseUrl = env.NODE_ENV === 'production'
      ? 'https://cozmicastro.one'
      : 'http://localhost:8081';

    const params = new URLSearchParams({ return_url: `${baseUrl}/(tabs)/profile` });
    const response = await fetch(`${dodoApiBaseUrl()}/customers/${encodeURIComponent(customerId)}/customer-portal/session?${params}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${env.DODOPAYMENTS_API_KEY}`,
      },
    });

    if (!response.ok) {
      const errorText = await response.text();
      return res.status(502).json({ error: 'Failed to create customer portal session', details: errorText });
    }

    const portal = await response.json();
    return res.json({ url: portal.link });
  } catch (error) {
    return res.status(500).json({ error: 'Failed to create customer portal', details: String(error) });
  }
});

export default router;
