import { Router } from 'express';
import { z } from 'zod';
import { requireFirebaseAuth } from '../middleware/auth.js';
import { fetchBe1Json } from '../services/be1Client.js';
import { ingestChartPayloadForProfile, ingestKundliForProfile, queryRagChunks } from '../services/ragPipeline.js';
import { stableHash } from '../services/hash.js';

const router = Router();

const KundliSchema = z.object({
  latitude: z.number(),
  longitude: z.number(),
  year: z.number(),
  month: z.number(),
  day: z.number(),
  hour: z.number(),
  min: z.number(),
  sec: z.number().optional(),
  time_zone: z.string(),
});

const IngestSchema = z.object({
  profileId: z.string().min(1).max(120),
  kundli: KundliSchema,
});

const QuerySchema = z.object({
  profileId: z.string().min(1).max(120).optional(),
  message: z.string().min(1).max(4000),
  topK: z.number().int().min(1).max(20).optional(),
});

const GenerateChartSchema = z.object({
  name: z.string().min(1).max(120).optional(),
  place: z.string().min(1).max(180).optional(),
  profileId: z.string().min(1).max(120).optional(),
  latitude: z.number(),
  longitude: z.number(),
  year: z.number(),
  month: z.number(),
  day: z.number(),
  hour: z.number(),
  min: z.number(),
  sec: z.number().optional(),
  time_zone: z.string(),
  dst_hour: z.number().optional(),
  dst_min: z.number().optional(),
  nesting: z.number().int().min(1).max(6).optional(),
  infolevel: z.string().optional(),
  varga: z.string().optional(),
  ayanamsha: z.string().optional(),
});

router.post('/v1/rag/ingest', requireFirebaseAuth, async (req, res) => {
  try {
    const parsed = IngestSchema.safeParse(req.body ?? {});
    if (!parsed.success) {
      return res.status(400).json({ error: 'Invalid body', details: parsed.error.flatten() });
    }

    const result = await ingestKundliForProfile({
      ownerId: req.user!.uid,
      profileId: parsed.data.profileId,
      kundli: parsed.data.kundli,
    });

    return res.status(201).json({
      ok: true,
      ...result,
    });
  } catch (error) {
    return res.status(500).json({ error: 'Failed to ingest RAG profile', details: String(error) });
  }
});

router.post('/v1/rag/query', requireFirebaseAuth, async (req, res) => {
  try {
    const parsed = QuerySchema.safeParse(req.body ?? {});
    if (!parsed.success) {
      return res.status(400).json({ error: 'Invalid body', details: parsed.error.flatten() });
    }

    const chunks = await queryRagChunks({
      ownerId: req.user!.uid,
      profileId: parsed.data.profileId,
      message: parsed.data.message,
      topK: parsed.data.topK,
    });

    return res.json({
      chunks: chunks.map((chunk) => ({
        id: chunk.id,
        profileId: chunk.profileId,
        endpoint: chunk.endpoint,
        text: chunk.text,
        similarity: chunk.similarity,
        sourceDocId: chunk.sourceDocId,
      })),
    });
  } catch (error) {
    return res.status(500).json({ error: 'Failed to query vectors', details: String(error) });
  }
});

router.post('/v1/chart/generate', requireFirebaseAuth, async (req, res) => {
  try {
    const parsed = GenerateChartSchema.safeParse(req.body ?? {});
    if (!parsed.success) {
      return res.status(400).json({ error: 'Invalid body', details: parsed.error.flatten() });
    }

    const profileId =
      parsed.data.profileId ??
      `p_${stableHash(
        JSON.stringify({
          uid: req.user!.uid,
          lat: parsed.data.latitude,
          lon: parsed.data.longitude,
          y: parsed.data.year,
          m: parsed.data.month,
          d: parsed.data.day,
          h: parsed.data.hour,
          min: parsed.data.min,
          sec: parsed.data.sec ?? 0,
        })
      ).slice(0, 12)}`;

    const query: Record<string, number | string> = {
      latitude: parsed.data.latitude,
      longitude: parsed.data.longitude,
      year: parsed.data.year,
      month: parsed.data.month,
      day: parsed.data.day,
      hour: parsed.data.hour,
      min: parsed.data.min,
      sec: parsed.data.sec ?? 0,
      time_zone: parsed.data.time_zone,
      dst_hour: parsed.data.dst_hour ?? 0,
      dst_min: parsed.data.dst_min ?? 0,
      nesting: parsed.data.nesting ?? 5,
      infolevel:
        parsed.data.infolevel ??
        'basic,ashtakavarga,grahabala,rashibala,yogas,panchanga,dasha,ayanamsa,upagraha,arudha',
      varga: parsed.data.varga ?? 'D1,D2,D3,D4,D7,D9,D10,D12,D16,D20,D24,D27,D30,D40,D45,D60',
    };

    if (parsed.data.ayanamsha) {
      query.ayanamsha = parsed.data.ayanamsha;
    }

    const chartData = await fetchBe1Json('calculate', query);

    const ingestion = await ingestChartPayloadForProfile({
      ownerId: req.user!.uid,
      profileId,
      kundli: {
        latitude: parsed.data.latitude,
        longitude: parsed.data.longitude,
        year: parsed.data.year,
        month: parsed.data.month,
        day: parsed.data.day,
        hour: parsed.data.hour,
        min: parsed.data.min,
        sec: parsed.data.sec ?? 0,
        time_zone: parsed.data.time_zone,
      },
      payload: chartData,
      endpoint: 'calculate',
      tags: ['chart-generate', 'kundli', 'be1', 'calculate'],
    });

    return res.status(201).json({
      ok: true,
      profileId,
      chartData,
      ingestion,
    });
  } catch (error) {
    return res.status(500).json({ error: 'Failed to generate and ingest chart', details: String(error) });
  }
});

export default router;
