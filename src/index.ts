import express from 'express';
import cors from 'cors';
import { allowedOrigins, env } from './config/env.js';
import healthRoutes from './routes/health.js';
import chatRoutes from './routes/chat.js';
import meRoutes from './routes/me.js';
import kundaliRoutes from './routes/kundalis.js';
import voiceRoutes from './routes/voice.js';
import ragRoutes from './routes/rag.js';
import billingRoutes from './routes/billing.js';
import { expensiveEndpointRateLimit } from './middleware/rateLimit.js';
import { disconnectValkey } from './services/valkeyCache.js';

// Catch background Postgres connection timeouts during long LLM calls
// These are idle connections timing out — not application errors
process.on('uncaughtException', (error) => {
  if ((error as NodeJS.ErrnoException).code === 'ETIMEDOUT' && error.message?.includes('read')) {
    console.warn('[process] bg pg connection timeout (non-fatal):', error.message);
    return;
  }
  console.error('[process] uncaught exception:', error);
  process.exit(1);
});

process.on('unhandledRejection', (reason) => {
  const err = reason as Error;
  if ((err as NodeJS.ErrnoException).code === 'ETIMEDOUT' && err.message?.includes('read')) {
    console.warn('[process] bg pg rejection (non-fatal):', err.message);
    return;
  }
  console.error('[process] unhandled rejection:', reason);
});

const app = express();

app.use(
  cors({
    origin: (origin, callback) => {
      if (!origin || allowedOrigins.includes(origin)) {
        callback(null, true);
        return;
      }
      callback(new Error(`Origin not allowed: ${origin}`));
    },
  })
);

app.use(express.json({ limit: '8mb' }));

app.use((req, res, next) => {
  const start = process.hrtime.bigint();
  res.on('finish', () => {
    const end = process.hrtime.bigint();
    const durationMs = Number(end - start) / 1_000_000;
    // eslint-disable-next-line no-console
    console.log(`[http] ${req.method} ${req.originalUrl} -> ${res.statusCode} (${durationMs.toFixed(1)}ms)`);
  });
  next();
});

app.use('/v1/chart/generate', expensiveEndpointRateLimit);
app.use('/v1/chart/calculate', expensiveEndpointRateLimit);
app.use('/v1/rag/query', expensiveEndpointRateLimit);

app.use(healthRoutes);
app.use(chatRoutes);
app.use(meRoutes);
app.use(billingRoutes);
app.use(kundaliRoutes);
app.use(voiceRoutes);
app.use(ragRoutes);

app.use((err: unknown, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
  res.status(500).json({
    error: 'Unhandled server error',
    details: String(err),
  });
});

app.listen(env.PORT, () => {
  // eslint-disable-next-line no-console
  console.log(`cozmic-rag-agents listening on http://localhost:${env.PORT}`);
});

async function shutdown() {
  await disconnectValkey();
  process.exit(0);
}

process.on('SIGINT', () => {
  void shutdown();
});

process.on('SIGTERM', () => {
  void shutdown();
});
