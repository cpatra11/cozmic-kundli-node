import { Router } from 'express';
import { env } from '../config/env.js';

const router = Router();

router.get('/health', (_req, res) => {
  res.json({
    status: 'ok',
    service: 'cozmic-rag-agents',
    nodeEnv: env.NODE_ENV,
    timestamp: new Date().toISOString(),
  });
});

export default router;
