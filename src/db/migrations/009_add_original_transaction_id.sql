-- Add original_transaction_id column for Apple webhook matching
-- This stores Apple's stable subscription identifier (doesn't change on renewal)

ALTER TABLE subscriptions
  ADD COLUMN IF NOT EXISTS original_transaction_id TEXT;

CREATE INDEX IF NOT EXISTS idx_subscriptions_original_transaction_id
  ON subscriptions (original_transaction_id)
  WHERE original_transaction_id IS NOT NULL;

-- Backfill for existing iOS rows: original_transaction_id = transaction_id
-- This is correct for initial purchases (most common case).
-- Renewal rows will be fixed on next client sync.
UPDATE subscriptions
SET original_transaction_id = transaction_id
WHERE iapkit_store = 'apple'
  AND original_transaction_id IS NULL
  AND transaction_id IS NOT NULL;

-- Table for tracking unmatched webhook notifications (user hasn't synced yet)
CREATE TABLE IF NOT EXISTS apple_webhook_orphans (
  original_transaction_id TEXT PRIMARY KEY,
  notification_type TEXT,
  product_id TEXT,
  expires_date BIGINT,
  received_at BIGINT NOT NULL
);
