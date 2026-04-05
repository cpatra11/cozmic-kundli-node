import { Router, type Response } from 'express';
import { randomUUID } from 'node:crypto';
import { z } from 'zod';
import { requireFirebaseAuth } from '../middleware/auth.js';
import { runKundliAgent } from '../services/kundliAgent.js';
import { env } from '../config/env.js';
import { buildChatMessageEmbedding, queryRelevantSessionMemories } from '../services/chatMemory.js';
import { getChatRepository } from '../repositories/chatRepository.js';
import { getRagProfilesRepository } from '../repositories/ragProfilesRepository.js';
import { getSubscriptionsRepository } from '../repositories/subscriptionsRepository.js';
import { getUsageQuotasRepository, type QuotaStatusSnapshot } from '../repositories/usageQuotasRepository.js';
import { hasActiveProEntitlement } from '../services/subscriptionAccess.js';

const CreateSessionSchema = z.object({
  title: z.string().min(1).max(120).optional(),
  kundaliId: z.string().min(1).max(120).optional(),
});

const SendMessageSchema = z.object({
  message: z.string().min(1).max(4000),
  mode: z.enum(['mini', 'pro']).optional(),
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
  mode?: 'mini' | 'pro';
  model?: string;
  requestId?: string;
  kundaliId?: string;
  bindingId?: string;
  bindingTurn?: number;
  bindingChartVersion?: string;
  bindingKundliSignature?: string;
  embedding?: number[];
  embeddingModel?: string;
  embeddingDim?: number;
  createdAt: number;
}

interface ChatAccessDecision {
  allowed: boolean;
  hasPro: boolean;
  quotaStatus?: QuotaStatusSnapshot;
  statusCode?: number;
  payload?: Record<string, unknown>;
}

function isObviousFastMessage(message: string): boolean {
  const q = message.trim().toLowerCase();
  return /^(h+i+|hello|hey|namaste|good\s+(morning|afternoon|evening)|thanks|thank\s+you|ok(?:ay)?|cool|bye|goodbye)\b/.test(q);
}

async function evaluateChatAccess(ownerId: string, mode: 'mini' | 'pro'): Promise<ChatAccessDecision> {
  const subscriptionsRepository = getSubscriptionsRepository();
  const usageQuotasRepository = getUsageQuotasRepository();
  const subscription = await subscriptionsRepository.getByOwnerId(ownerId);
  const hasPro = hasActiveProEntitlement(subscription);

  if (!env.QUOTA_ENFORCEMENT_ENABLED) {
    if (mode === 'pro' && !hasPro) {
      return {
        allowed: false,
        hasPro,
        statusCode: 402,
        payload: {
          error: 'Pro subscription required',
          details: 'Pro mode requires an active subscription. Please purchase or restore your plan.',
          code: 'PRO_SUBSCRIPTION_REQUIRED',
        },
      };
    }

    return {
      allowed: true,
      hasPro,
    };
  }

  const quotaType = mode === 'pro' ? 'pro_chat' : 'mini_chat';
  const consumed = await usageQuotasRepository.consumeQuota(ownerId, hasPro, quotaType);

  if (consumed.allowed) {
    return {
      allowed: true,
      hasPro,
      quotaStatus: consumed.status,
    };
  }

  if (mode === 'pro' && !hasPro) {
    return {
      allowed: false,
      hasPro,
      quotaStatus: consumed.status,
      statusCode: 402,
      payload: {
        error: 'Pro subscription required',
        details:
          'Your monthly free Pro trial request is already used. Upgrade to Pro for more Pro-mode requests.',
        code: 'PRO_SUBSCRIPTION_REQUIRED',
        quotaStatus: consumed.status,
      },
    };
  }

  const modeLabel = mode === 'pro' ? 'Pro chat' : 'Mini chat';
  return {
    allowed: false,
    hasPro,
    quotaStatus: consumed.status,
    statusCode: 429,
    payload: {
      error: 'Monthly quota exceeded',
      details: `${modeLabel} monthly quota exceeded. Please wait for reset or switch plans.`,
      code: 'MONTHLY_QUOTA_EXCEEDED',
      quotaType,
      quotaStatus: consumed.status,
    },
  };
}

function writeSseEvent(res: Response, event: string, payload: unknown): void {
  res.write(`event: ${event}\n`);
  res.write(`data: ${JSON.stringify(payload)}\n\n`);
}

router.post('/v1/chat/sessions', requireFirebaseAuth, async (req, res) => {
  try {
    const parsed = CreateSessionSchema.safeParse(req.body ?? {});
    if (!parsed.success) {
      return res.status(400).json({ error: 'Invalid body', details: parsed.error.flatten() });
    }

    const chatRepository = getChatRepository();
    const now = Date.now();
    const sessionId = randomUUID();

    const sessionDoc: ChatSessionDoc = {
      ownerId: req.user!.uid,
      title: parsed.data.title ?? 'New horoscope chat',
      kundaliId: parsed.data.kundaliId,
      createdAt: now,
      updatedAt: now,
    };

    await chatRepository.createSession({
      id: sessionId,
      ownerId: sessionDoc.ownerId,
      title: sessionDoc.title,
      kundaliId: sessionDoc.kundaliId,
      chartVersion: sessionDoc.chartVersion,
      createdAt: sessionDoc.createdAt,
      updatedAt: sessionDoc.updatedAt,
      lastMessagePreview: sessionDoc.lastMessagePreview,
    });

    return res.status(201).json({ id: sessionId });
  } catch (error) {
    return res.status(500).json({ error: 'Failed to create session', details: String(error) });
  }
});

router.get('/v1/chat/sessions', requireFirebaseAuth, async (req, res) => {
  try {
    const chatRepository = getChatRepository();
    const sessions = await chatRepository.listSessionsByOwner(req.user!.uid, 50);
    return res.json({ sessions });
  } catch (error) {
    return res.status(500).json({ error: 'Failed to list sessions', details: String(error) });
  }
});

router.get('/v1/chat/sessions/:sessionId/messages', requireFirebaseAuth, async (req, res) => {
  try {
    const chatRepository = getChatRepository();
    const sessionIdParam = req.params.sessionId;
    const sessionId = Array.isArray(sessionIdParam) ? sessionIdParam[0] : sessionIdParam;

    if (!sessionId) {
      return res.status(400).json({ error: 'Missing sessionId path parameter' });
    }

    const session = await chatRepository.getSessionById(sessionId);
    if (!session) {
      return res.status(404).json({ error: 'Session not found' });
    }

    if (session.ownerId !== req.user!.uid) {
      return res.status(403).json({ error: 'Forbidden' });
    }

    const messages = await chatRepository.listSessionMessages(req.user!.uid, sessionId, 500);
    return res.json({ messages });
  } catch (error) {
    return res.status(500).json({ error: 'Failed to list messages', details: String(error) });
  }
});

router.post('/v1/chat/sessions/:sessionId/messages/stream', requireFirebaseAuth, async (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache, no-transform');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');
  res.flushHeaders();

  let closed = false;
  req.on('close', () => {
    closed = true;
  });

  try {
    const parsed = SendMessageSchema.safeParse(req.body ?? {});
    if (!parsed.success) {
      writeSseEvent(res, 'error', {
        error: 'Invalid body',
        details: parsed.error.flatten(),
        status: 400,
      });
      res.end();
      return;
    }

    const requestedMode = parsed.data.mode ?? 'mini';

    const chatRepository = getChatRepository();
    const ragProfilesRepository = getRagProfilesRepository();

    const sessionIdParam = req.params.sessionId;
    const sessionId = Array.isArray(sessionIdParam) ? sessionIdParam[0] : sessionIdParam;

    if (!sessionId) {
      writeSseEvent(res, 'error', {
        error: 'Missing sessionId path parameter',
        status: 400,
      });
      res.end();
      return;
    }

    const session = await chatRepository.getSessionById(sessionId);

    if (!session) {
      writeSseEvent(res, 'error', {
        error: 'Session not found',
        status: 404,
      });
      res.end();
      return;
    }

    if (session.ownerId !== req.user!.uid) {
      writeSseEvent(res, 'error', {
        error: 'Forbidden',
        status: 403,
      });
      res.end();
      return;
    }

    const accessDecision = await evaluateChatAccess(req.user!.uid, requestedMode);
    if (!accessDecision.allowed) {
      writeSseEvent(res, 'error', {
        ...(accessDecision.payload ?? { error: 'Chat access denied' }),
        status: accessDecision.statusCode ?? 403,
      });
      res.end();
      return;
    }

    let effectiveProfileId = parsed.data.profileId ?? parsed.data.kundaliId ?? session.kundaliId;
    const fastMessage = isObviousFastMessage(parsed.data.message);

    if (!effectiveProfileId && !fastMessage) {
      const latestProfiles = await ragProfilesRepository.listByOwner(req.user!.uid, 1);
      effectiveProfileId = latestProfiles[0]?.profileId;
    }

    const now = Date.now();
    const requestId = parsed.data.requestId ?? randomUUID();

    const relevantMemories = fastMessage
      ? []
      : await queryRelevantSessionMemories({
          ownerId: req.user!.uid,
          sessionId,
          message: parsed.data.message,
          excludeRequestId: requestId,
          topK: 6,
        });

    const userMessage: ChatMessageDoc = {
      ownerId: req.user!.uid,
      sessionId,
      role: 'user',
      message: parsed.data.message,
      mode: requestedMode,
      requestId,
      kundaliId: effectiveProfileId,
      createdAt: now,
      ...buildChatMessageEmbedding(parsed.data.message),
    };

    await chatRepository.createMessage(userMessage);

    writeSseEvent(res, 'ack', {
      requestId,
      sessionId,
      mode: requestedMode,
      kundaliId: effectiveProfileId,
      quotaStatus: accessDecision.quotaStatus,
    });

    const agent = await runKundliAgent({
      ownerId: req.user!.uid,
      message: parsed.data.message,
      mode: requestedMode,
      profileId: effectiveProfileId,
      kundli: parsed.data.kundli,
      clientTimestamp: parsed.data.clientTimestamp,
      conversationContext: relevantMemories.map((m) => `${m.role.toUpperCase()}: ${m.text}`),
      onStage: (stage) => {
        if (!closed) {
          writeSseEvent(res, 'stage', stage);
        }
      },
    });

    const assistantMessage: ChatMessageDoc = {
      ownerId: req.user!.uid,
      sessionId,
      role: 'assistant',
      message: agent.answer,
      mode: requestedMode,
      model: agent.model,
      requestId,
      kundaliId: effectiveProfileId,
      bindingId: agent.grounding?.sourceDocId,
      bindingChartVersion: agent.grounding?.chartVersion,
      bindingKundliSignature: agent.grounding?.kundliSignature,
      createdAt: Date.now(),
      ...buildChatMessageEmbedding(agent.answer),
    };

    await chatRepository.createMessage(assistantMessage);

    await chatRepository.updateSession(sessionId, req.user!.uid, {
      updatedAt: Date.now(),
      lastMessagePreview: parsed.data.message.slice(0, 180),
      kundaliId: effectiveProfileId,
      chartVersion: agent.grounding?.chartVersion,
    });

    writeSseEvent(res, 'done', {
      answer: agent.answer,
      model: agent.model,
      mode: requestedMode,
      executionPlan: agent.executionPlan,
      analysisStages: agent.analysisStages,
      decisionTelemetry: agent.decisionTelemetry,
      grounding: agent.grounding,
      memoryContextUsed: relevantMemories,
      sessionId,
      kundaliId: effectiveProfileId,
      chartVersion: agent.grounding?.chartVersion,
      requestId,
      quotaStatus: accessDecision.quotaStatus,
    });

    res.end();
  } catch (error) {
    const maybeError = error as { status?: unknown; code?: unknown; message?: unknown };
    const status = typeof maybeError.status === 'number' && Number.isFinite(maybeError.status)
      ? maybeError.status
      : 500;

    writeSseEvent(res, 'error', {
      error: 'Failed to process streamed message',
      details: String(error),
      status,
      ...(typeof maybeError.code === 'string' ? { code: maybeError.code } : {}),
    });
    res.end();
  }
});

router.post('/v1/chat/sessions/:sessionId/messages', requireFirebaseAuth, async (req, res) => {
  try {
    const parsed = SendMessageSchema.safeParse(req.body ?? {});
    if (!parsed.success) {
      return res.status(400).json({ error: 'Invalid body', details: parsed.error.flatten() });
    }

    const requestedMode = parsed.data.mode ?? 'mini';

    const chatRepository = getChatRepository();
    const ragProfilesRepository = getRagProfilesRepository();

    const sessionIdParam = req.params.sessionId;
    const sessionId = Array.isArray(sessionIdParam) ? sessionIdParam[0] : sessionIdParam;

    if (!sessionId) {
      return res.status(400).json({ error: 'Missing sessionId path parameter' });
    }

    const session = await chatRepository.getSessionById(sessionId);

    if (!session) {
      return res.status(404).json({ error: 'Session not found' });
    }

    if (session.ownerId !== req.user!.uid) {
      return res.status(403).json({ error: 'Forbidden' });
    }

    const accessDecision = await evaluateChatAccess(req.user!.uid, requestedMode);
    if (!accessDecision.allowed) {
      return res.status(accessDecision.statusCode ?? 403).json(accessDecision.payload ?? { error: 'Chat access denied' });
    }

    let effectiveProfileId = parsed.data.profileId ?? parsed.data.kundaliId ?? session.kundaliId;
    const fastMessage = isObviousFastMessage(parsed.data.message);

    if (!effectiveProfileId && !fastMessage) {
      const latestProfiles = await ragProfilesRepository.listByOwner(req.user!.uid, 1);
      effectiveProfileId = latestProfiles[0]?.profileId;
    }

    const now = Date.now();
    const requestId = parsed.data.requestId ?? randomUUID();

    const relevantMemories = fastMessage
      ? []
      : await queryRelevantSessionMemories({
          ownerId: req.user!.uid,
          sessionId,
          message: parsed.data.message,
          excludeRequestId: requestId,
          topK: 6,
        });

    const userMessage: ChatMessageDoc = {
      ownerId: req.user!.uid,
      sessionId,
      role: 'user',
      message: parsed.data.message,
      mode: requestedMode,
      requestId,
      kundaliId: effectiveProfileId,
      createdAt: now,
      ...buildChatMessageEmbedding(parsed.data.message),
    };

    await chatRepository.createMessage(userMessage);

    const agent = await runKundliAgent({
      ownerId: req.user!.uid,
      message: parsed.data.message,
      mode: requestedMode,
      profileId: effectiveProfileId,
      kundli: parsed.data.kundli,
      clientTimestamp: parsed.data.clientTimestamp,
      conversationContext: relevantMemories.map((m) => `${m.role.toUpperCase()}: ${m.text}`),
    });

    const assistantMessage: ChatMessageDoc = {
      ownerId: req.user!.uid,
      sessionId,
      role: 'assistant',
      message: agent.answer,
      mode: requestedMode,
      model: agent.model,
      requestId,
      kundaliId: effectiveProfileId,
      bindingId: agent.grounding?.sourceDocId,
      bindingChartVersion: agent.grounding?.chartVersion,
      bindingKundliSignature: agent.grounding?.kundliSignature,
      createdAt: Date.now(),
      ...buildChatMessageEmbedding(agent.answer),
    };

    await chatRepository.createMessage(assistantMessage);

    await chatRepository.updateSession(sessionId, req.user!.uid, {
      updatedAt: Date.now(),
      lastMessagePreview: parsed.data.message.slice(0, 180),
      kundaliId: effectiveProfileId,
      chartVersion: agent.grounding?.chartVersion,
    });

    return res.json({
      answer: agent.answer,
      model: agent.model,
      mode: requestedMode,
      executionPlan: agent.executionPlan,
      analysisStages: agent.analysisStages,
      decisionTelemetry: agent.decisionTelemetry,
      grounding: agent.grounding,
      memoryContextUsed: relevantMemories,
      sessionId,
      kundaliId: effectiveProfileId,
      chartVersion: agent.grounding?.chartVersion,
      requestId,
      quotaStatus: accessDecision.quotaStatus,
    });
  } catch (error) {
    return res.status(500).json({ error: 'Failed to process message', details: String(error) });
  }
});

export default router;
