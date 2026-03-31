CREATE TABLE IF NOT EXISTS chat_sessions (
  id TEXT PRIMARY KEY,
  owner_id TEXT NOT NULL,
  title TEXT NOT NULL,
  kundali_id TEXT,
  chart_version TEXT,
  last_message_preview TEXT,
  created_at BIGINT NOT NULL,
  updated_at BIGINT NOT NULL,
  CONSTRAINT chat_sessions_owner_title_chk CHECK (char_length(title) > 0)
);

CREATE INDEX IF NOT EXISTS chat_sessions_owner_updated_idx ON chat_sessions (owner_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS chat_sessions_owner_kundali_idx ON chat_sessions (owner_id, kundali_id);

CREATE TABLE IF NOT EXISTS chat_messages (
  id TEXT PRIMARY KEY,
  owner_id TEXT NOT NULL,
  session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
  role TEXT NOT NULL,
  message TEXT NOT NULL,
  mode TEXT,
  model TEXT,
  request_id TEXT,
  kundali_id TEXT,
  binding_id TEXT,
  binding_turn INTEGER,
  binding_chart_version TEXT,
  binding_kundli_signature TEXT,
  embedding VECTOR(192),
  embedding_model TEXT,
  embedding_dim INTEGER,
  created_at BIGINT NOT NULL,
  CONSTRAINT chat_messages_role_chk CHECK (role IN ('user', 'assistant')),
  CONSTRAINT chat_messages_mode_chk CHECK (mode IS NULL OR mode IN ('mini', 'pro')),
  CONSTRAINT chat_messages_message_chk CHECK (char_length(message) > 0)
);

CREATE INDEX IF NOT EXISTS chat_messages_session_created_idx ON chat_messages (session_id, created_at ASC);
CREATE INDEX IF NOT EXISTS chat_messages_owner_session_created_idx ON chat_messages (owner_id, session_id, created_at ASC);
CREATE INDEX IF NOT EXISTS chat_messages_owner_created_idx ON chat_messages (owner_id, created_at DESC);
CREATE INDEX IF NOT EXISTS chat_messages_request_id_idx ON chat_messages (request_id);
CREATE INDEX IF NOT EXISTS chat_messages_embedding_hnsw_idx ON chat_messages USING hnsw (embedding vector_cosine_ops);
