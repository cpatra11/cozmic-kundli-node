-- Migration 007: LangGraph node-level cache table
-- Purpose: Store LangGraph node-level cache entries in PostgreSQL
--          Replaces Valkey for decision caching and enables intent-based caching with TTLs

CREATE TABLE IF NOT EXISTS langgraph_node_cache (
  namespace TEXT[] NOT NULL,
  key TEXT NOT NULL,
  value BYTEA NOT NULL,
  encoding TEXT NOT NULL DEFAULT 'json',
  expires_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  PRIMARY KEY (namespace, key)
);

CREATE INDEX IF NOT EXISTS idx_langgraph_cache_expiry
  ON langgraph_node_cache(expires_at)
  WHERE expires_at IS NOT NULL;
