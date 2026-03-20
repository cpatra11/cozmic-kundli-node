# cozmic-rag-agents

Standalone backend service for horoscope chat/voice, using Firebase for identity + persistence and a Google ADK TypeScript-ready agent layer.

## What is implemented now

- Express TypeScript server scaffold
- Firebase Admin token middleware (`Bearer <Firebase ID token>`)
- Chat session APIs (create/list/send-message)
- Voice APIs (transcribe/synthesize) as contract-first stubs
- `be1` grounding client for `/api/calculate`
- Agent runtime adapter scaffold (`runKundliAgent`)

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

1. Copy `.env.example` into `.env` and fill Firebase Admin credentials.
2. Install dependencies and run dev server.

Optional commands:
- `npm install`
- `npm run dev`

Server default URL: `http://localhost:8787`

## Firebase requirements

Set one credential source:
- `FIREBASE_SERVICE_ACCOUNT_JSON`, or
- `FIREBASE_SERVICE_ACCOUNT_PATH`

And set:
- `FIREBASE_PROJECT_ID`
- `FIREBASE_STORAGE_BUCKET`

## ADK / Gemini runtime

For Google ADK TypeScript runtime:

- Install dependency: `@google/adk` (already added in this service)
- Set `GEMINI_API_KEY` (or compatible env used by your runtime)
- Configure model with `GOOGLE_GENAI_MODEL`

## Next implementation step

Replace `src/services/kundliAgent.ts` internals with Google ADK TypeScript runtime calls and retrieval pipeline.
