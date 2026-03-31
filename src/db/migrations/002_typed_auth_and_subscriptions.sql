CREATE TABLE IF NOT EXISTS auth_users (
  owner_id TEXT PRIMARY KEY,
  email TEXT,
  phone_number TEXT,
  provider TEXT NOT NULL DEFAULT 'firebase',
  created_at BIGINT NOT NULL,
  updated_at BIGINT NOT NULL,
  last_seen_at BIGINT NOT NULL,
  CONSTRAINT auth_users_provider_chk CHECK (provider IN ('firebase'))
);

CREATE UNIQUE INDEX IF NOT EXISTS auth_users_email_unique_idx
ON auth_users (lower(trim(email)))
WHERE email IS NOT NULL AND NULLIF(trim(email), '') IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS auth_users_phone_unique_idx
ON auth_users (trim(phone_number))
WHERE phone_number IS NOT NULL AND NULLIF(trim(phone_number), '') IS NOT NULL;

CREATE TABLE IF NOT EXISTS subscriptions (
  owner_id TEXT PRIMARY KEY,
  source TEXT NOT NULL DEFAULT 'revenuecat',
  entitlement_id TEXT NOT NULL,
  is_pro BOOLEAN NOT NULL DEFAULT FALSE,
  store TEXT,
  product_id TEXT,
  event_type TEXT,
  expires_at_ms BIGINT,
  updated_at BIGINT NOT NULL,
  last_event_at BIGINT NOT NULL,
  last_event_id TEXT,
  CONSTRAINT subscriptions_source_chk CHECK (source IN ('revenuecat'))
);

CREATE INDEX IF NOT EXISTS subscriptions_is_pro_idx ON subscriptions (is_pro);
CREATE INDEX IF NOT EXISTS subscriptions_updated_idx ON subscriptions (updated_at DESC);
