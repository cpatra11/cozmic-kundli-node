# Database Schema Overview

This document describes the database tables used by `cozmic-rag-agents`, what each one is for, and how they relate to each other.

## High-level shape

The database uses **three physical tables**:

1. `documents`
2. `charts`
3. `chart_vectors`

The app also uses **logical document collections** stored inside `documents`, such as `auth_users`, `chat_messages`, `chat_sessions`, `rag_profiles`, `rag_api_sources`, `rag_chunks`, and `user_subscriptions`.

## `public.documents`

### What it is
A generic JSON document store. This is the main flexible storage table used by the backend for several logical collections.

### Main columns
- `path` — unique document path, like `auth_users/<userId>` or `chat_messages/<messageId>`
- `collection` — logical collection name
- `doc_id` — document id within that collection
- `data` — JSON payload
- `created_at` — timestamp when the document was created
- `updated_at` — timestamp when the document was last changed

### What it does
This table acts like a Firestore-style document store. The backend uses it for:
- user auth profile records
- chat sessions and messages
- rag profile documents
- rag API source records
- rag chunks
- user subscriptions

### Important logical collections inside `documents`

#### `auth_users`
Stores Firebase-backed user profile records.

Typical fields in the JSON payload:
- `ownerId`
- `email`
- `phoneNumber`
- `provider`
- `createdAt`
- `updatedAt`
- `lastSeenAt`

Used by:
- `GET /v1/me`

#### `chat_sessions`
Stores chat session metadata.

Used by:
- chat session creation and updates
- chat history lookups

#### `chat_messages`
Stores individual chat messages.

Used by:
- chat history and conversation replay

#### `rag_profiles`
Stores per-user RAG profile data.

Used by:
- chart-related retrieval and profile persistence

#### `rag_api_sources`
Stores source payloads and chart snapshots used to build RAG content.

#### `rag_chunks`
Stores chunked text and embeddings for retrieval.

#### `user_subscriptions`
Stores billing/subscription state, including RevenueCat-derived entitlement data.

Used by:
- `GET /v1/billing/subscription`

### Relation notes
- `documents` is the umbrella table for several app features.
- There are **no foreign key constraints** between these logical collections; the links are stored in JSON fields like `ownerId`, `profileId`, and `sessionId`.

---

## `public.charts`

### What it is
A chart metadata table. It stores the main astrology chart record for a user/profile.

### Main columns
- `id` — primary UUID
- `owner_id` — Firebase user / owner identifier
- `kundali_id` — chart/profile identifier used by the app
- `request_key` — deduplication or request correlation key
- `ingestion_status` — current ingestion state
- `name`, `place`, `display_name` — user-facing labels
- `tags`, `panchanga`, `chart_data`, `raw_payload_ref` — JSON payloads and metadata
- `chart_signature` — chart identity fingerprint
- `chart_datetime` — chart date/time
- `location_lat`, `location_lng`, `timezone` — birth/location context
- `tithi`, `nakshatra`, `dasha_current` — astrology-specific metadata
- `created_at`, `updated_at`, `deleted_at` — lifecycle timestamps

### What it does
This table stores the structured chart record that the app can query, display, and enrich.

### Relation notes
- `charts.owner_id` identifies the user who owns the chart.
- `charts.kundali_id` is reused by related chart/vector rows.
- `chart_vectors.chart_id` references `charts.id` logically, though the database does not enforce a foreign key.

---

## `public.chart_vectors`

### What it is
A vector/chunk table for retrieval and semantic search.

### Main columns
- `id` — primary UUID
- `chart_id` — chart row UUID
- `owner_id` — chart owner
- `kundali_id` — chart/profile identifier
- `source_doc_id` — source document id
- `chunk_index` — ordering of the chunk within a source
- `section` — optional section label
- `text` — chunk text
- `embedding` — vector embedding
- `created_at` — creation timestamp
- `source_type`, `endpoint` — provenance metadata
- `text_preview` — short preview text
- `embedding_model`, `embedding_dim` — embedding metadata
- `token_estimate` — estimated token size
- `tags` — JSON tags

### What it does
This table powers chart search and retrieval workflows. It stores chunked chart text plus embeddings so the backend can do similarity-based lookup.

### Relation notes
- `chart_vectors.chart_id` points to a row in `charts.id`.
- `chart_vectors.owner_id` and `chart_vectors.kundali_id` should match the owning chart context.
- `chart_vectors.source_doc_id` often links back to a document stored in `documents`, usually under `rag_api_sources` or another source record.

---

## Relationships between the tables

### 1. `documents` → app feature collections
`documents` is the shared store for:
- auth profile data
- chat data
- rag data
- subscriptions

This is the most flexible table and is used for most non-chart records.

### 2. `charts` → `chart_vectors`
A chart can have many vector rows.

- One `charts` row
- Many `chart_vectors` rows

This is a one-to-many relationship used for semantic retrieval.

### 3. `documents` → `charts` / `chart_vectors`
The JSON documents often provide supporting data used by chart and retrieval workflows.

For example:
- `rag_api_sources` stores source payloads
- `rag_chunks` stores chunk-level data
- `rag_profiles` stores profile-level chart retrieval context

### 4. `auth_users` is the user identity anchor
The `auth_users` collection in `documents` stores the app’s user profile record, including the Firebase UID owner reference. Other user-owned records usually point back to that same owner identity.

---

## Feature-to-table map

### Authentication / profile
- `documents.auth_users`

### Chat history
- `documents.chat_sessions`
- `documents.chat_messages`

### Billing
- `documents.user_subscriptions`

### Astrology charts and retrieval
- `charts`
- `chart_vectors`
- `documents.rag_profiles`
- `documents.rag_api_sources`
- `documents.rag_chunks`

---

## Implemented v2 improvements

The schema has been upgraded with a stronger relational baseline while keeping the document-style flexibility.

### What changed

1. **Centralized schema management in code**
	- All core DDL now lives in `src/services/postgresSchema.ts`.
	- `PostgresStore` and pgvector workflows now rely on this shared schema setup.

2. **Stronger `documents` guarantees**
	- Added path consistency check: `path = collection || '/' || doc_id`.
	- Added unique `(collection, doc_id)` index in addition to primary `path` key.
	- Kept existing targeted indexes and auth uniqueness indexes.

3. **Structured `charts` table with identity constraints**
	- Enforced `UNIQUE(owner_id, kundali_id)`.
	- Added partial unique index on `(owner_id, request_key)` for active rows.
	- Added owner/sort indexes for common read patterns.

4. **Relational `chart_vectors` constraints**
	- Added foreign key: `chart_vectors.chart_id -> charts.id` with `ON DELETE CASCADE`.
	- Added unique chunk identity: `(owner_id, kundali_id, source_doc_id, chunk_index)`.
	- Added lookup indexes and HNSW cosine index for vector similarity search.

5. **Ingestion now writes structured chart rows**
	- `ragPipeline` now upserts a canonical row into `charts` on ingest.
	- pgvector chunks are upserted for vector search when enabled.

### Reset and push command

The DB clear + push workflow is implemented as:

- Script: `scripts/reset-db.ts`
- NPM command: `npm run db:reset:push`

This command:
- drops `chart_vectors`, `charts`, and `documents`
- recreates schema and indexes from `postgresSchema.ts`
- recreates pgvector artifacts when `PGVECTOR_ENABLED=true`

---

## Summary

If you want the short version:

- **`documents`** = flexible JSON store for app records
- **`charts`** = main astrology chart metadata
- **`chart_vectors`** = embeddings/chunks for chart search and retrieval

The tables are related mostly by shared identifiers like `ownerId`, `chart_id`, `kundali_id`, and `sourceDocId`, rather than strict foreign key constraints.

---

## Plain-English summary

If you want the simple version, here it is:

### 1. `documents`
This is the app’s flexible storage box.

It holds things like:
- user login/profile records
- chat messages and chat sessions
- billing/subscription info
- RAG source data and chunk data

Think of it like a general-purpose warehouse for records that do not need their own separate table.

### 2. `charts`
This is the main astrology chart table.

It stores the important details of a chart, like:
- who owns it
- the chart name / display name
- birth and location details
- astrology metadata such as nakshatra or tithi

Think of it as the master chart record.

### 3. `chart_vectors`
This is the search and retrieval table.

It stores:
- chart chunks
- embeddings
- preview text
- metadata for matching and lookup

Think of it as the searchable memory layer for chart content.

### How they connect
- `documents` is the shared store for many features.
- `charts` holds the main chart record.
- `chart_vectors` belongs to a chart and stores searchable pieces of that chart.
- `chart_vectors.chart_id` links back to `charts.id`.
- `documents.auth_users` stores the user’s profile and identity info.

### One-line summary
- `documents` = general app data
- `charts` = astrology chart master records
- `chart_vectors` = searchable chart chunks and embeddings
