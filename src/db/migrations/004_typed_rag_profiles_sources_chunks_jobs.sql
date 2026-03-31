CREATE TABLE IF NOT EXISTS rag_profiles (
  owner_id TEXT NOT NULL,
  profile_id TEXT NOT NULL,
  display_name TEXT,
  place TEXT,
  kundli_signature TEXT NOT NULL,
  chart_version TEXT NOT NULL,
  kundli_input JSONB NOT NULL,
  latest_source_doc_id TEXT NOT NULL,
  source_count INTEGER NOT NULL DEFAULT 0,
  updated_at BIGINT NOT NULL,
  created_at BIGINT NOT NULL,
  PRIMARY KEY (owner_id, profile_id),
  CONSTRAINT rag_profiles_source_count_chk CHECK (source_count >= 0)
);

CREATE INDEX IF NOT EXISTS rag_profiles_owner_updated_idx ON rag_profiles (owner_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS rag_profiles_owner_signature_idx ON rag_profiles (owner_id, kundli_signature);
CREATE INDEX IF NOT EXISTS rag_profiles_owner_latest_source_idx ON rag_profiles (owner_id, latest_source_doc_id);

CREATE TABLE IF NOT EXISTS rag_api_sources (
  id TEXT PRIMARY KEY,
  owner_id TEXT NOT NULL,
  profile_id TEXT NOT NULL,
  display_name TEXT,
  place TEXT,
  source_type TEXT NOT NULL,
  endpoint TEXT NOT NULL,
  request_key TEXT NOT NULL,
  payload_hash TEXT NOT NULL,
  raw_payload JSONB NOT NULL,
  chart_snapshot JSONB,
  preview TEXT NOT NULL,
  tags JSONB NOT NULL DEFAULT '[]'::jsonb,
  created_at BIGINT NOT NULL,
  CONSTRAINT rag_api_sources_source_type_chk CHECK (source_type IN ('be1')),
  CONSTRAINT rag_api_sources_profile_fk FOREIGN KEY (owner_id, profile_id) REFERENCES rag_profiles(owner_id, profile_id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS rag_api_sources_owner_profile_request_payload_unique_idx
ON rag_api_sources (owner_id, profile_id, request_key, payload_hash);
CREATE INDEX IF NOT EXISTS rag_api_sources_owner_profile_created_idx ON rag_api_sources (owner_id, profile_id, created_at DESC);
CREATE INDEX IF NOT EXISTS rag_api_sources_owner_profile_endpoint_idx ON rag_api_sources (owner_id, profile_id, endpoint);

CREATE TABLE IF NOT EXISTS rag_chunks (
  id TEXT PRIMARY KEY,
  owner_id TEXT NOT NULL,
  profile_id TEXT NOT NULL,
  source_doc_id TEXT NOT NULL,
  source_type TEXT NOT NULL,
  endpoint TEXT NOT NULL,
  chunk_index INTEGER NOT NULL,
  text TEXT NOT NULL,
  text_preview TEXT NOT NULL,
  embedding VECTOR(192),
  embedding_model TEXT,
  embedding_dim INTEGER,
  token_estimate INTEGER,
  tags JSONB NOT NULL DEFAULT '[]'::jsonb,
  created_at BIGINT NOT NULL,
  CONSTRAINT rag_chunks_source_type_chk CHECK (source_type IN ('be1')),
  CONSTRAINT rag_chunks_text_chk CHECK (char_length(text) > 0),
  CONSTRAINT rag_chunks_source_fk FOREIGN KEY (source_doc_id) REFERENCES rag_api_sources(id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS rag_chunks_source_chunk_unique_idx ON rag_chunks (source_doc_id, chunk_index);
CREATE INDEX IF NOT EXISTS rag_chunks_owner_profile_created_idx ON rag_chunks (owner_id, profile_id, created_at DESC);
CREATE INDEX IF NOT EXISTS rag_chunks_owner_profile_source_idx ON rag_chunks (owner_id, profile_id, source_doc_id);
CREATE INDEX IF NOT EXISTS rag_chunks_embedding_hnsw_idx ON rag_chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS chart_jobs (
  owner_id TEXT NOT NULL,
  job_id TEXT NOT NULL,
  profile_id TEXT NOT NULL,
  status TEXT NOT NULL,
  request JSONB NOT NULL,
  result JSONB,
  error TEXT,
  created_at BIGINT NOT NULL,
  updated_at BIGINT NOT NULL,
  PRIMARY KEY (owner_id, job_id),
  CONSTRAINT chart_jobs_status_chk CHECK (status IN ('queued', 'running', 'completed', 'failed'))
);

CREATE INDEX IF NOT EXISTS chart_jobs_owner_created_idx ON chart_jobs (owner_id, created_at DESC);
CREATE INDEX IF NOT EXISTS chart_jobs_owner_status_updated_idx ON chart_jobs (owner_id, status, updated_at DESC);
CREATE INDEX IF NOT EXISTS chart_jobs_owner_profile_updated_idx ON chart_jobs (owner_id, profile_id, updated_at DESC);
