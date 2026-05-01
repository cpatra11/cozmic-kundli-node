-- Convert legacy subscription rows into the new store/IAPKit subscription model.
-- Existing active entitlements are normalized to the canonical "pro" entitlement.

ALTER TABLE subscriptions
  DROP CONSTRAINT IF EXISTS subscriptions_source_chk;

UPDATE subscriptions
SET
  source = CASE
    WHEN source IN ('iapkit', 'app_store', 'play_store') THEN source
    WHEN store IN ('app_store', 'play_store') THEN store
    ELSE 'iapkit'
  END,
  entitlement_id = CASE
    WHEN is_pro THEN 'pro'
    ELSE 'free'
  END;

ALTER TABLE subscriptions
  ALTER COLUMN source SET DEFAULT 'iapkit';

ALTER TABLE subscriptions
  ADD CONSTRAINT subscriptions_source_chk CHECK (source IN ('iapkit', 'app_store', 'play_store'));

ALTER TABLE subscriptions
  ADD COLUMN IF NOT EXISTS purchase_token TEXT,
  ADD COLUMN IF NOT EXISTS transaction_id TEXT,
  ADD COLUMN IF NOT EXISTS iapkit_state TEXT,
  ADD COLUMN IF NOT EXISTS iapkit_valid BOOLEAN,
  ADD COLUMN IF NOT EXISTS iapkit_store TEXT;

ALTER TABLE subscriptions
  DROP CONSTRAINT IF EXISTS subscriptions_iapkit_store_chk;

ALTER TABLE subscriptions
  ADD CONSTRAINT subscriptions_iapkit_store_chk CHECK (iapkit_store IN ('apple', 'google', 'unknown') OR iapkit_store IS NULL);
