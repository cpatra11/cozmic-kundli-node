ALTER TABLE subscriptions DROP CONSTRAINT IF EXISTS subscriptions_source_chk;

ALTER TABLE subscriptions
  ADD CONSTRAINT subscriptions_source_chk CHECK (source IN ('iapkit', 'app_store', 'play_store', 'dodopayments'));
