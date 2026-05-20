ALTER TABLE subscriptions ADD COLUMN IF NOT EXISTS billing_anchor_ms BIGINT;

ALTER TABLE monthly_usage_counters ADD COLUMN IF NOT EXISTS period_start_ms BIGINT;

UPDATE monthly_usage_counters
SET period_start_ms = EXTRACT(EPOCH FROM TO_DATE(year_month || '-01', 'YYYY-MM-DD'))::BIGINT * 1000
WHERE period_start_ms IS NULL;

ALTER TABLE monthly_usage_counters ALTER COLUMN period_start_ms SET NOT NULL;

ALTER TABLE monthly_usage_counters DROP CONSTRAINT IF EXISTS monthly_usage_counters_pkey;

ALTER TABLE monthly_usage_counters ADD PRIMARY KEY (owner_id, period_start_ms);

CREATE INDEX IF NOT EXISTS monthly_usage_counters_period_start_ms_idx
ON monthly_usage_counters (period_start_ms);
