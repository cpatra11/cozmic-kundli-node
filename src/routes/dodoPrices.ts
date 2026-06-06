import { Router } from 'express';
import { env, dodoApiBaseUrl } from '../config/env.js';

const router = Router();

const PLAN_MAP: Record<string, { productIdEnv: keyof typeof env; identifier: string; title: string; description: string }> = {
  cozmic_pro_monthly: {
    productIdEnv: 'DODOPAYMENTS_PRICE_MONTHLY' as any,
    identifier: 'cozmic_pro_monthly',
    title: 'Monthly',
    description: 'Full access, cancel anytime',
  },
  cozmic_pro_yearly: {
    productIdEnv: 'DODOPAYMENTS_PRICE_YEARLY' as any,
    identifier: 'cozmic_pro_yearly',
    title: 'Yearly',
    description: 'Best value',
  },
};

function formatPrice(cents: number, currency: string): string {
  const symbols: Record<string, string> = { INR: '₹', USD: '$', EUR: '€', GBP: '£' };
  const sym = symbols[currency] || currency + ' ';
  return `${sym}${(cents / 100).toLocaleString('en-IN', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`;
}

router.get('/v1/billing/prices', async (_req, res) => {
  try {
    if (!env.DODOPAYMENTS_API_KEY) {
      return res.status(500).json({ error: 'DodoPayments API key not configured' });
    }

    const results = await Promise.allSettled(
      Object.values(PLAN_MAP).map(async (plan) => {
        const productId = env[plan.productIdEnv] as string | undefined;
        if (!productId) return null;

        const response = await fetch(`${dodoApiBaseUrl()}/products/${productId}`, {
          headers: { Authorization: `Bearer ${env.DODOPAYMENTS_API_KEY}` },
        });

        if (!response.ok) return null;

        const product = await response.json();
        const priceCents = product.price_detail?.price ?? product.price;
        const currency = product.price_detail?.currency ?? product.currency ?? 'USD';
        const frequency = product.price_detail?.payment_frequency_interval?.toLowerCase() ?? 'month';
        const discount = product.price_detail?.discount ?? 0;

        return {
          identifier: plan.identifier,
          title: plan.title,
          description: plan.description,
          priceString: `${formatPrice(priceCents, currency)}/${frequency === 'year' ? 'year' : 'month'}`,
          price: priceCents,
          currencyCode: currency,
          frequency,
          discount,
        };
      })
    );

    const packages = results
      .map((r) => (r.status === 'fulfilled' ? r.value : null))
      .filter(Boolean);

    if (packages.length === 0) {
      return res.status(502).json({ error: 'Failed to fetch prices from DodoPayments' });
    }

    return res.json({ packages });
  } catch (error) {
    return res.status(500).json({ error: 'Failed to fetch prices', details: String(error) });
  }
});

export default router;
