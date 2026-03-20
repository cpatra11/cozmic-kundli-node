import { Router } from 'express';
import { z } from 'zod';
import { requireFirebaseAuth } from '../middleware/auth.js';

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

export default router;
