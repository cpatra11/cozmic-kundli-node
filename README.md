# cozmic-rag-agents

Standalone backend service for horoscope chat/voice, using Firebase Authentication for identity, Postgres for canonical persistence, and a LangGraph-based grounded agent layer.

## What is implemented now

- Express TypeScript server scaffold
- Firebase ID token middleware via JWKS (`Bearer <Firebase ID token>`)
- Chat session APIs (create/list/send-message)
- Voice APIs (transcribe/synthesize) as contract-first stubs
- `be1` grounding client for `/api/calculate`
- LangGraph grounded-answer runtime (`runKundliAgent`)

## API endpoints

- `GET /health`
- `POST /v1/chat/sessions`
- `GET /v1/chat/sessions`
- `POST /v1/chat/sessions/:sessionId/messages`
- `POST /v1/voice/transcribe`
- `POST /v1/voice/synthesize`
- `POST /v1/chart/generate`
- `POST /v1/rag/ingest`
- `POST /v1/rag/query`

## Local setup

1. Copy `.env.example` into `.env` and fill `FIREBASE_PROJECT_ID` + `DATABASE_URL`.
2. Install dependencies and run dev server.

Optional commands:
- `npm install`
- `npm run dev`

Server default URL: `http://localhost:8787`

## Firebase requirements

Default auth verification mode is JWKS-based and does not require a service-account JSON file.

Set:
- `FIREBASE_PROJECT_ID`
- optional `FIREBASE_JWKS_URL` (defaults to Google Secure Token JWKS)
- optional `JWT_CLOCK_SKEW_SECONDS`

Optional fallback (if you choose Firebase Admin verification):
- `FIREBASE_SERVICE_ACCOUNT_JSON`, or
- `FIREBASE_SERVICE_ACCOUNT_PATH`

## Postgres / LangGraph runtime

Set:

- `DATABASE_URL`
- optional `DATABASE_SSL_CA_PATH` if your database uses a private/self-signed CA bundle
- optional `DATABASE_SSL_REJECT_UNAUTHORIZED=false` only for local/dev debugging

The Postgres database stores canonical chat sessions, messages, chart payloads, and grounding metadata as JSONB documents.

For the grounded chart assistant:

- Set `GEMINI_API_KEY` if you want Gemini-powered responses
- Configure model with `GOOGLE_GENAI_MODEL`

The LangGraph agent uses the saved raw chart payload and selects only the relevant canonical sections for the user's question. It does not use client-side fallback chart details.

