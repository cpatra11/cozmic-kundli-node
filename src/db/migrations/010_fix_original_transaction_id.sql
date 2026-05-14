-- Fix original_transaction_id: clear values incorrectly backfilled from transaction_id
-- Migration 009 copied transaction_id (changes on renewal) as original_transaction_id.
-- This breaks Apple webhook matching for users whose first sync was after a renewal.
-- The correct value will be set on the next client sync from the device.

UPDATE subscriptions
SET original_transaction_id = NULL
WHERE iapkit_store = 'apple'
  AND original_transaction_id IS NOT NULL
  AND original_transaction_id = transaction_id;
