import { Router } from 'express';
import { randomUUID } from 'node:crypto';
import { z } from 'zod';
import { requireFirebaseAuth } from '../middleware/auth.js';
import { runKundliAgent } from '../services/kundliAgent.js';
import { getPostgresStore } from '../services/postgresStore.js';
import { COLLECTIONS, type RagProfileDocument } from '../models/firestoreModels.js';

const CreateSessionSchema = z.object({
  title: z.string().min(1).max(120).optional(),
  kundaliId: z.string().min(1).max(120).optional(),
});

const SendMessageSchema = z.object({
  message: z.string().min(1).max(4000),
  profileId: z.string().min(1).max(120).optional(),
  kundaliId: z.string().min(1).max(120).optional(),
  requestId: z.string().min(1).max(120).optional(),
  clientTimestamp: z.number().int().optional(),
  kundli: z
    .object({
      latitude: z.number(),
      longitude: z.number(),
      year: z.number(),
      month: z.number(),
      day: z.number(),
      hour: z.number(),
      min: z.number(),
      sec: z.number().optional(),
      time_zone: z.string(),
    })
    .optional(),
});

const router = Router();

interface ChatSessionDoc {
  ownerId: string;
  title: string;
  kundaliId?: string;
  chartVersion?: string;
  createdAt: number;
  updatedAt: number;
  lastMessagePreview?: string;
}

interface ChatMessageDoc {
  ownerId: string;
  sessionId: string;
  role: 'user' | 'assistant';
  message: string;
  model?: string;
  requestId?: string;
  kundaliId?: string;
  bindingId?: string;
  bindingTurn?: number;
  bindingChartVersion?: string;
  bindingKundliSignature?: string;
  createdAt: number;
}

router.post('/v1/chat/sessions', requireFirebaseAuth, async (req, res) => {
  try {
    const parsed = CreateSessionSchema.safeParse(req.body ?? {});
    if (!parsed.success) {
      return res.status(400).json({ error: 'Invalid body', details: parsed.error.flatten() });
    }

    const store = getPostgresStore();
    const now = Date.now();
    const sessionId = randomUUID();

    const sessionDoc: ChatSessionDoc = {
      ownerId: req.user!.uid,
      title: parsed.data.title ?? 'New horoscope chat',
      kundaliId: parsed.data.kundaliId,
      createdAt: now,
      updatedAt: now,
    };

    await store.setDocument(`chat_sessions/${sessionId}`, sessionDoc, false);

    return res.status(201).json({ id: sessionId });
  } catch (error) {
    return res.status(500).json({ error: 'Failed to create session', details: String(error) });
  }
});

router.get('/v1/chat/sessions', requireFirebaseAuth, async (req, res) => {
  try {
    const store = getPostgresStore();
    const snapshot = await store.runQuery<ChatSessionDoc>(
      'chat_sessions',
      [{ field: 'ownerId', op: 'EQUAL', value: req.user!.uid }],
      {
        orderBy: [{ field: 'updatedAt', direction: 'DESCENDING' }],
        limit: 50,
      }
    );

    const sessions = snapshot.map((doc) => ({ id: doc.id, ...doc.data }));
    return res.json({ sessions });
  } catch (error) {
    return res.status(500).json({ error: 'Failed to list sessions', details: String(error) });
  }
});

router.get('/v1/chat/sessions/:sessionId/messages', requireFirebaseAuth, async (req, res) => {
  try {
    const store = getPostgresStore();
    const sessionIdParam = req.params.sessionId;
    const sessionId = Array.isArray(sessionIdParam) ? sessionIdParam[0] : sessionIdParam;

    if (!sessionId) {
      return res.status(400).json({ error: 'Missing sessionId path parameter' });
    }

    const session = await store.getDocument<ChatSessionDoc>(`chat_sessions/${sessionId}`);
    if (!session) {
      return res.status(404).json({ error: 'Session not found' });
    }

    if (session.data.ownerId !== req.user!.uid) {
      return res.status(403).json({ error: 'Forbidden' });
    }

    const messages = await store.runQuery<ChatMessageDoc>(
      'chat_messages',
      [
        { field: 'ownerId', op: 'EQUAL', value: req.user!.uid },
        { field: 'sessionId', op: 'EQUAL', value: sessionId },
      ],
      {
        orderBy: [{ field: 'createdAt', direction: 'ASCENDING' }],
        limit: 500,
      }
    );

    return res.json({ messages: messages.map((doc) => ({ id: doc.id, ...doc.data })) });
  } catch (error) {
    return res.status(500).json({ error: 'Failed to list messages', details: String(error) });
  }
});

router.post('/v1/chat/sessions/:sessionId/messages', requireFirebaseAuth, async (req, res) => {
  try {
    const parsed = SendMessageSchema.safeParse(req.body ?? {});
    if (!parsed.success) {
      return res.status(400).json({ error: 'Invalid body', details: parsed.error.flatten() });
    }

    const store = getPostgresStore();
    const sessionIdParam = req.params.sessionId;
    const sessionId = Array.isArray(sessionIdParam) ? sessionIdParam[0] : sessionIdParam;

    if (!sessionId) {
      return res.status(400).json({ error: 'Missing sessionId path parameter' });
    }

    const sessionPath = `chat_sessions/${sessionId}`;
    const session = await store.getDocument<ChatSessionDoc>(sessionPath);

    if (!session) {
      return res.status(404).json({ error: 'Session not found' });
    }

    if (session.data.ownerId !== req.user!.uid) {
      return res.status(403).json({ error: 'Forbidden' });
    }

    let effectiveProfileId = parsed.data.profileId ?? parsed.data.kundaliId ?? session.data.kundaliId;

    if (!effectiveProfileId) {
      const latestProfiles = await store.runQuery<RagProfileDocument>(
        COLLECTIONS.ragProfiles,
        [{ field: 'ownerId', op: 'EQUAL', value: req.user!.uid }],
        {
          orderBy: [{ field: 'updatedAt', direction: 'DESCENDING' }],
          limit: 1,
        }
      );

      effectiveProfileId = latestProfiles[0]?.data.profileId;
    }

    if (!effectiveProfileId) {
      return res.status(400).json({
        error: 'Missing canonical chart identity',
        details:
          'Open or save a Kundli first so chat can load the canonical raw payload from Postgres. If none exists yet, generate a chart via /v1/chart/generate first.',
      });
    }

    const now = Date.now();
    const requestId = parsed.data.requestId ?? randomUUID();

    const userMessage: ChatMessageDoc = {
      ownerId: req.user!.uid,
      sessionId,
      role: 'user',
      message: parsed.data.message,
      requestId,
      kundaliId: effectiveProfileId,
      createdAt: now,
    };

    await store.createDocument('chat_messages', userMessage);

    const agent = await runKundliAgent({
      ownerId: req.user!.uid,
      message: parsed.data.message,
      profileId: effectiveProfileId,
      kundli: parsed.data.kundli,
      clientTimestamp: parsed.data.clientTimestamp,
    });

    const assistantMessage: ChatMessageDoc = {
      ownerId: req.user!.uid,
      sessionId,
      role: 'assistant',
      message: agent.answer,
      model: agent.model,
      requestId,
      kundaliId: effectiveProfileId,
      bindingId: agent.grounding?.sourceDocId,
      bindingChartVersion: agent.grounding?.chartVersion,
      bindingKundliSignature: agent.grounding?.kundliSignature,
      createdAt: Date.now(),
    };

    await store.createDocument('chat_messages', assistantMessage);

    await store.setDocument(
      sessionPath,
      {
        updatedAt: Date.now(),
        lastMessagePreview: parsed.data.message.slice(0, 180),
        kundaliId: effectiveProfileId,
        chartVersion: agent.grounding?.chartVersion,
      },
      true
    );

    return res.json({
      answer: agent.answer,
      model: agent.model,
      grounding: agent.grounding,
      sessionId,
      kundaliId: effectiveProfileId,
      chartVersion: agent.grounding?.chartVersion,
      requestId,
    });
  } catch (error) {
    return res.status(500).json({ error: 'Failed to process message', details: String(error) });
  }
});

export default router;
