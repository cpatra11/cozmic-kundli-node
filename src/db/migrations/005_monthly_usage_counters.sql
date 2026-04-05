CREATE TABLE IF NOT EXISTS monthly_usage_counters (
  owner_id TEXT NOT NULL,
  year_month TEXT NOT NULL,
  mini_chat_used INTEGER NOT NULL DEFAULT 0,
  pro_chat_used INTEGER NOT NULL DEFAULT 0,
  kundli_generate_used INTEGER NOT NULL DEFAULT 0,
  created_at BIGINT NOT NULL,
  updated_at BIGINT NOT NULL,
  PRIMARY KEY (owner_id, year_month),
  CONSTRAINT monthly_usage_counters_non_negative_chk CHECK (
    mini_chat_used >= 0 AND
    pro_chat_used >= 0 AND
    kundli_generate_used >= 0
  )
);

CREATE INDEX IF NOT EXISTS monthly_usage_counters_year_month_idx
ON monthly_usage_counters (year_month);

CREATE INDEX IF NOT EXISTS monthly_usage_counters_owner_updated_idx
ON monthly_usage_counters (owner_id, updated_at DESC);
