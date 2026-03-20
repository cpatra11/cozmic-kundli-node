# Firestore RAG Model (Access-Pattern First)

## Design rules applied

- Access-pattern driven schema; small strategic denormalization.
- Small predictable documents (`rag_chunks.text` capped by chunking).
- No hot counters/documents in write path.
- Composite indexes are explicit (`firestore.indexes.json`).

## Collections

### `rag_profiles`
One profile per `(ownerId, profileId)`.

Core fields:
- `ownerId`, `profileId`
- `kundliSignature`, `kundliInput`
- `latestSourceDocId`
- `sourceCount`, `createdAt`, `updatedAt`

Read pattern:
- list profiles by owner (recent first)

### `rag_api_sources`
Metadata for each BE1 ingestion event.

Core fields:
- `ownerId`, `profileId`
- `sourceType=be1`, `endpoint`
- `requestKey`, `payloadHash`
- `preview` (trimmed payload snapshot)
- `createdAt`

Read pattern:
- inspect ingestion lineage/debugging

### `rag_chunks`
Chunked text units with vectors.

Core fields:
- `ownerId`, `profileId`
- `sourceDocId`, `endpoint`, `chunkIndex`
- `text`, `textPreview`
- `embedding[]`, `embeddingModel`, `embeddingDim`
- `tokenEstimate`, `createdAt`

Read pattern:
- fetch recent candidate chunks by owner/profile
- re-rank in app layer with cosine similarity

## Why this shape

- Duplicates `ownerId/profileId/endpoint` on chunks intentionally to avoid fan-out reads.
- Keeps source docs lightweight (`preview` instead of full payload blob).
- Avoids a central mutable document in ingest path.

## Query flow

1. Ingest kundli from BE1 `/calculate`.
2. Flatten payload -> chunk text lines.
3. Embed each chunk and store to `rag_chunks`.
4. Query retrieves owner/profile candidates and scores by cosine similarity.

## Current vector implementation

- Deterministic hash embedding (`deterministic-hash-v1`), dimension via `EMBEDDING_DIM`.
- Safe scaffold for now; replace with real embedding model later without changing schema.
