import { env } from '../config/env.js';
import type { UserSubscriptionDocument } from '../models/firestoreModels.js';

export function hasActiveProEntitlement(
  subscription: UserSubscriptionDocument | null | undefined,
  nowMs = Date.now()
): boolean {
  if (!subscription?.isPro) return false;

  if (typeof subscription.expiresAtMs === 'number' && subscription.expiresAtMs <= nowMs) {
    return false;
  }

  const expectedEntitlement = env.PRO_ENTITLEMENT_ID;
  const legacyEntitlement = env.REVENUECAT_PRO_ENTITLEMENT_ID;

  return (
    subscription.entitlementId === expectedEntitlement ||
    subscription.entitlementId === 'pro' ||
    (legacyEntitlement ? subscription.entitlementId === legacyEntitlement : false)
  );
}
