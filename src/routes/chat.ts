import { Router } from 'express';
import { randomUUID } from 'node:crypto';
import { z } from 'zod';
import { requireFirebaseAuth } from '../middleware/auth.js';
import { runKundliAgent } from '../services/kundliAgent.js';
import { getFirestoreStore } from '../services/firestoreStore.js';

const CreateSessionSchema = z.object({
  title: z.string().min(1).max(120).optional(),
});

const SendMessageSchema = z.object({
  message: z.string().min(1).max(4000),
  profileId: z.string().min(1).max(120).optional(),
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
  createdAt: number;
}

router.post('/v1/chat/sessions', requireFirebaseAuth, async (req, res) => {
  try {
    const parsed = CreateSessionSchema.safeParse(req.body ?? {});
    if (!parsed.success) {
      return res.status(400).json({ error: 'Invalid body', details: parsed.error.flatten() });
    }

    const store = getFirestoreStore();
    const now = Date.now();
    const sessionId = randomUUID();

    const sessionDoc: ChatSessionDoc = {
      ownerId: req.user!.uid,
      title: parsed.data.title ?? 'New horoscope chat',
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
    const store = getFirestoreStore();
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

router.post('/v1/chat/sessions/:sessionId/messages', requireFirebaseAuth, async (req, res) => {
  try {
    const parsed = SendMessageSchema.safeParse(req.body ?? {});
    if (!parsed.success) {
      return res.status(400).json({ error: 'Invalid body', details: parsed.error.flatten() });
    }

    const store = getFirestoreStore();
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

    const now = Date.now();

    const userMessage: ChatMessageDoc = {
      ownerId: req.user!.uid,
      sessionId,
      role: 'user',
      message: parsed.data.message,
      createdAt: now,
    };

    await store.createDocument('chat_messages', userMessage);

    const agent = await runKundliAgent({
      ownerId: req.user!.uid,
      message: parsed.data.message,
      profileId: parsed.data.profileId,
      kundli: parsed.data.kundli,
    });

    const assistantMessage: ChatMessageDoc = {
      ownerId: req.user!.uid,
      sessionId,
      role: 'assistant',
      message: agent.answer,
      model: agent.model,
      createdAt: Date.now(),
    };

    await store.createDocument('chat_messages', assistantMessage);

    await store.setDocument(
      sessionPath,
      {
        updatedAt: Date.now(),
        lastMessagePreview: parsed.data.message.slice(0, 180),
      },
      true
    );

    return res.json({
      answer: agent.answer,
      model: agent.model,
      grounding: agent.grounding,
    });
  } catch (error) {
    return res.status(500).json({ error: 'Failed to process message', details: String(error) });
  }
});

export default router;
