-- Migration 008: Drop obsolete tables from old RAG architecture
-- Run this after verifying the new architecture works correctly in production

-- Drop tables in correct order (respecting foreign key dependencies)
DROP TABLE IF EXISTS rag_chunks CASCADE;
DROP TABLE IF EXISTS rag_api_sources CASCADE;
DROP TABLE IF EXISTS chart_vectors CASCADE;

-- Drop indexes if they weren't dropped with the tables
DROP INDEX IF EXISTS idx_rag_chunks_owner_profile;
DROP INDEX IF EXISTS idx_rag_sources_owner_profile;
DROP INDEX IF EXISTS idx_chart_vectors_owner_kundali;

-- Verify tables are dropped
-- SELECT table_name FROM information_schema.tables 
-- WHERE table_schema = 'public' AND table_name IN ('rag_chunks', 'rag_api_sources', 'chart_vectors');
