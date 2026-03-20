import express from 'express';
import cors from 'cors';
import { allowedOrigins, env } from './config/env.js';
import healthRoutes from './routes/health.js';
import chatRoutes from './routes/chat.js';
import voiceRoutes from './routes/voice.js';
import ragRoutes from './routes/rag.js';

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

app.use(healthRoutes);
app.use(chatRoutes);
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
