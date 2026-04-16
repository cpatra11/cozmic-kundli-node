BEGIN;

-- Normalize any legacy RevenueCat-backed rows to the new store-native source model.
-- This keeps existing Pro users working while removing the RevenueCat runtime path.
ALTER TABLE subscriptions
  DROP CONSTRAINT IF EXISTS subscriptions_source_chk;

UPDATE subscriptions
SET
  source = CASE
    WHEN source = 'revenuecat' AND lower(COALESCE(store, '')) = 'play_store' THEN 'play_store'
    WHEN source = 'revenuecat' AND lower(COALESCE(store, '')) = 'app_store' THEN 'app_store'
    WHEN source = 'revenuecat' THEN 'expo_iap'
    ELSE source
  END,
  entitlement_id = CASE
    WHEN is_pro THEN 'pro'
    WHEN entitlement_id = 'Cozmic Astrology Pro' THEN 'pro'
    WHEN entitlement_id = 'pro_access' THEN 'pro'
    ELSE entitlement_id
  END
WHERE source = 'revenuecat'
   OR is_pro = TRUE
   OR entitlement_id IN ('Cozmic Astrology Pro', 'pro_access');

ALTER TABLE subscriptions
  ADD CONSTRAINT subscriptions_source_chk
  CHECK (source IN ('expo_iap', 'app_store', 'play_store'));

COMMIT;
