import { Router } from 'express';
import { z } from 'zod';
import { requireFirebaseAuth } from '../middleware/auth.js';
import { runKundliAgent } from '../services/kundliAgent.js';
import { NOVA_SONIC_MODEL_ID, synthesizeVoiceTurn } from '../services/novaSonicSpeech.js';

const router = Router();

const VoiceTranscriptSchema = z.object({
  audioBase64: z.string().min(1),
  mimeType: z.string().default('audio/m4a'),
  durationMs: z.number().optional(),
});

const VoiceSynthesisSchema = z.object({
  text: z.string().min(1).max(3000),
  voice: z.string().default('en-US-Standard-C'),
});

const VoiceTurnSchema = z.object({
  message: z.string().min(1).max(4000),
  profileId: z.string().min(1).max(120).optional(),
  kundli: z.any().optional(),
  clientTimestamp: z.number().optional(),
  systemPrompt: z.string().min(1).max(4000).optional(),
});

router.post('/v1/voice/transcribe', requireFirebaseAuth, async (req, res) => {
  const parsed = VoiceTranscriptSchema.safeParse(req.body ?? {});
  if (!parsed.success) {
    return res.status(400).json({ error: 'Invalid body', details: parsed.error.flatten() });
  }

  // Phase-1 placeholder: backend route + auth contract in place.
  // Phase-2: connect Google Speech-to-Text and Firebase Storage upload.
  return res.json({
    transcript: '[stub] transcription wired; connect Google STT next.',
    bytesReceived: Buffer.from(parsed.data.audioBase64, 'base64').byteLength,
    mimeType: parsed.data.mimeType,
  });
});

router.post('/v1/voice/synthesize', requireFirebaseAuth, async (req, res) => {
  const parsed = VoiceSynthesisSchema.safeParse(req.body ?? {});
  if (!parsed.success) {
    return res.status(400).json({ error: 'Invalid body', details: parsed.error.flatten() });
  }

  // Phase-1 placeholder: contract-first API for mobile integration.
  return res.json({
    audioBase64: '',
    mimeType: 'audio/mp3',
    note: '[stub] synthesis route ready; connect Google TTS next.',
    text: parsed.data.text,
  });
});

router.post('/v1/voice/turn', requireFirebaseAuth, async (req, res) => {
  const parsed = VoiceTurnSchema.safeParse(req.body ?? {});
  if (!parsed.success) {
    return res.status(400).json({ error: 'Invalid body', details: parsed.error.flatten() });
  }

  try {
    const answer = await runKundliAgent({
      ownerId: req.user!.uid,
      message: parsed.data.message,
      profileId: parsed.data.profileId,
      kundli: parsed.data.kundli,
      clientTimestamp: parsed.data.clientTimestamp,
    });

    const speech = await synthesizeVoiceTurn({
      message: answer.answer,
      systemPrompt:
        parsed.data.systemPrompt ??
        'Read the following assistant answer out loud in a calm, warm, conversational tone. Keep the pacing natural and do not add extra commentary.',
    });

    return res.json({
      answer: answer.answer,
      model: answer.model,
      grounding: answer.grounding,
      voiceModel: speech.model || NOVA_SONIC_MODEL_ID,
      audioBase64: speech.audioBase64,
      audioMimeType: speech.mimeType,
      spokenText: speech.text || answer.answer,
    });
  } catch (error) {
    return res.status(500).json({
      error: 'Failed to produce voice turn',
      details: String(error),
    });
  }
});

export default router;
