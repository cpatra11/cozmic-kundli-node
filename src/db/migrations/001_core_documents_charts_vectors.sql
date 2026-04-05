CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
  path TEXT PRIMARY KEY,
  collection TEXT NOT NULL,
  doc_id TEXT NOT NULL,
  data JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CONSTRAINT documents_path_consistency_chk CHECK (path = collection || '/' || doc_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS documents_collection_doc_id_unique_idx ON documents (collection, doc_id);
CREATE INDEX IF NOT EXISTS documents_collection_idx ON documents (collection);
CREATE INDEX IF NOT EXISTS documents_collection_owner_idx ON documents (collection, ((data->>'ownerId')));
CREATE INDEX IF NOT EXISTS documents_collection_profile_idx ON documents (collection, ((data->>'profileId')));
CREATE INDEX IF NOT EXISTS documents_collection_updated_idx ON documents (collection, updated_at DESC);
CREATE INDEX IF NOT EXISTS documents_collection_owner_profile_idx ON documents (collection, ((data->>'ownerId')), ((data->>'profileId')));
CREATE INDEX IF NOT EXISTS documents_collection_latest_source_idx ON documents (collection, ((data->>'latestSourceDocId')));
CREATE INDEX IF NOT EXISTS documents_collection_chart_version_idx ON documents (collection, ((data->>'chartVersion')));
CREATE UNIQUE INDEX IF NOT EXISTS documents_auth_users_email_unique_idx ON documents (lower(trim((data->>'email')))) WHERE collection = 'auth_users' AND data ? 'email' AND NULLIF(trim(data->>'email'), '') IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS documents_auth_users_phone_unique_idx ON documents (trim((data->>'phoneNumber'))) WHERE collection = 'auth_users' AND data ? 'phoneNumber' AND NULLIF(trim(data->>'phoneNumber'), '') IS NOT NULL;
CREATE INDEX IF NOT EXISTS documents_chat_sessions_owner_updated_idx ON documents (((data->>'ownerId')), ((data->>'updatedAt')) DESC) WHERE collection = 'chat_sessions';
CREATE INDEX IF NOT EXISTS documents_chat_messages_owner_session_created_idx ON documents (((data->>'ownerId')), ((data->>'sessionId')), ((data->>'createdAt'))) WHERE collection = 'chat_messages';

CREATE TABLE IF NOT EXISTS charts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  owner_id TEXT NOT NULL,
  kundali_id TEXT NOT NULL,
  request_key TEXT,
  ingestion_status TEXT NOT NULL DEFAULT 'ready',
  name TEXT,
  place TEXT,
  display_name TEXT,
  tags JSONB NOT NULL DEFAULT '[]'::jsonb,
  panchanga JSONB,
  chart_data JSONB NOT NULL DEFAULT '{}'::jsonb,
  chart_schema_version TEXT NOT NULL DEFAULT 'mahadasha-first',
  dasha_depth INTEGER NOT NULL DEFAULT 1,
  dasha_period_key TEXT,
  raw_payload_ref TEXT,
  chart_signature TEXT,
  chart_datetime TEXT,
  location_lat DOUBLE PRECISION,
  location_lng DOUBLE PRECISION,
  timezone TEXT,
  tithi TEXT,
  nakshatra TEXT,
  dasha_current TEXT,
  created_at BIGINT NOT NULL,
  updated_at BIGINT NOT NULL,
  deleted_at BIGINT,
  CONSTRAINT charts_owner_kundali_unique UNIQUE (owner_id, kundali_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS charts_owner_request_key_unique_idx ON charts (owner_id, request_key) WHERE request_key IS NOT NULL AND deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS charts_owner_updated_idx ON charts (owner_id, updated_at DESC) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS charts_owner_signature_idx ON charts (owner_id, chart_signature) WHERE deleted_at IS NULL;

CREATE TABLE IF NOT EXISTS chart_vectors (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  chart_id UUID NOT NULL REFERENCES charts(id) ON DELETE CASCADE,
  owner_id TEXT NOT NULL,
  kundali_id TEXT NOT NULL,
  profile_id TEXT GENERATED ALWAYS AS (kundali_id) STORED,
  source_doc_id TEXT NOT NULL,
  chunk_index INTEGER NOT NULL,
  section TEXT,
  text TEXT NOT NULL,
  embedding VECTOR(192) NOT NULL,
  created_at BIGINT NOT NULL,
  source_type TEXT,
  endpoint TEXT,
  text_preview TEXT,
  embedding_model TEXT,
  embedding_dim INTEGER,
  token_estimate INTEGER,
  tags JSONB NOT NULL DEFAULT '[]'::jsonb,
  CONSTRAINT chart_vectors_owner_kundali_source_chunk_unique UNIQUE (owner_id, kundali_id, source_doc_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chart_vectors_owner_kundali_created_idx ON chart_vectors (owner_id, kundali_id, created_at DESC);
CREATE INDEX IF NOT EXISTS chart_vectors_chart_id_created_idx ON chart_vectors (chart_id, created_at DESC);
CREATE INDEX IF NOT EXISTS chart_vectors_source_doc_idx ON chart_vectors (source_doc_id);
CREATE INDEX IF NOT EXISTS chart_vectors_embedding_hnsw_idx ON chart_vectors USING hnsw (embedding vector_cosine_ops);
