import { Router } from 'express';
import { z } from 'zod';
import { env } from '../config/env.js';
import { requireFirebaseAuth } from '../middleware/auth.js';
import { fetchBe1Calculate, fetchTransitChart } from '../services/be1Client.js';

import { ingestKundliForProfile } from '../services/ragPipeline.js';
import { stableHash } from '../services/hash.js';
import { buildChartSnapshot, extractChartSchemaInfo } from '../services/chartSnapshot.js';
import { getSubscriptionsRepository } from '../repositories/subscriptionsRepository.js';
import { getUsageQuotasRepository, type QuotaStatusSnapshot } from '../repositories/usageQuotasRepository.js';
import { hasActiveProEntitlement } from '../services/subscriptionAccess.js';

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
  name: z.string().min(1).max(120).optional(),
  place: z.string().min(1).max(180).optional(),
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
  periodKey: z.string().min(1).max(24).optional(),
  infolevel: z.string().optional(),
  varga: z.string().optional(),
  ayanamsha: z.string().optional(),
  ingest: z.boolean().optional(),
});

const TransitChartSchema = z.object({
  latitude: z.number(),
  longitude: z.number(),
  time_zone: z.string().optional(),
  timezone: z.string().optional(),
  timeZone: z.string().optional(),
  year: z.number().optional(),
  month: z.number().optional(),
  day: z.number().optional(),
  hour: z.number().optional(),
  min: z.number().optional(),
  sec: z.number().optional(),
  t_year: z.number().optional(),
  t_month: z.number().optional(),
  t_day: z.number().optional(),
  t_hour: z.number().optional(),
  t_min: z.number().optional(),
  t_sec: z.number().optional(),
  nesting: z.number().int().min(1).max(6).optional(),
});

async function calculateChartPreview(parsedData: z.infer<typeof GenerateChartSchema>) {
  let chartData: unknown;
  try {
    chartData = await fetchBe1Calculate(
      {
        latitude: parsedData.latitude,
        longitude: parsedData.longitude,
        year: parsedData.year,
        month: parsedData.month,
        day: parsedData.day,
        hour: parsedData.hour,
        min: parsedData.min,
        sec: parsedData.sec ?? 0,
        time_zone: parsedData.time_zone,
      },
      {
        nesting: parsedData.nesting ?? 1,
        periodKey: parsedData.periodKey,
        // Simplified infolevel for faster chart-only generation (without RAG/AI chat)
        infolevel:
          parsedData.infolevel ??
            'basic,ashtakavarga,grahabala,rashibala,yogas,panchanga,dasha,ayanamsa,upagraha,arudha',
        varga: parsedData.varga ?? 'D1,D2,D3,D4,D7,D9,D10,D12,D16,D20,D24,D27,D30,D40,D45,D60',
        ayanamsha: parsedData.ayanamsha,
        dstHour: parsedData.dst_hour ?? 0,
        dstMin: parsedData.dst_min ?? 0,
      }
    );
  } catch (error) {
    throw new Error(`Failed to fetch chart payload from BE1: ${String(error)}`);
  }

  const chartSnapshot = buildChartSnapshot(chartData);
  const chartSchema = extractChartSchemaInfo(chartData);

  return {
    rawChartData: chartData,
    chartSnapshot,
    chartSchema,
  };
}

async function generateAndIngestChart(ownerId: string, parsedData: z.infer<typeof GenerateChartSchema>) {
  const startTime = Date.now();
  const profileId =
    parsedData.profileId ??
    `p_${stableHash(
      JSON.stringify({
        uid: ownerId,
        lat: parsedData.latitude,
        lon: parsedData.longitude,
        y: parsedData.year,
        m: parsedData.month,
        d: parsedData.day,
        h: parsedData.hour,
        min: parsedData.min,
        sec: parsedData.sec ?? 0,
      })
    ).slice(0, 12)}`;

  console.log('[generateAndIngestChart] Starting for profileId:', profileId, 'ingest flag:', parsedData.ingest);
  
  const chartStart = Date.now();
  const { rawChartData, chartSnapshot, chartSchema } = await calculateChartPreview(parsedData);
  console.log('[generateAndIngestChart] calculateChartPreview took:', Date.now() - chartStart, 'ms');

  let ingestion;
  let ingestionError: string | undefined;
  
  // Simplified: only save birth details to rag_profiles, not full chart
  const shouldSaveProfile = parsedData.ingest !== false;
  
  try {
    if (shouldSaveProfile) {
      ingestion = await ingestKundliForProfile({
        ownerId,
        profileId,
        displayName: parsedData.name,
        place: parsedData.place,
        kundli: {
          latitude: parsedData.latitude,
          longitude: parsedData.longitude,
          year: parsedData.year,
          month: parsedData.month,
          day: parsedData.day,
          hour: parsedData.hour,
          min: parsedData.min,
          sec: parsedData.sec ?? 0,
          time_zone: parsedData.time_zone,
        },
      });
      console.log('[generateAndIngestChart] Saved profile for profileId:', profileId);
    } else {
      console.log('[generateAndIngestChart] Skipped profile save for profileId:', profileId);
    }
  } catch (error) {
    ingestionError = `Failed to persist birth details to Postgres: ${String(error)}`;
    console.error('[chart/generate] non-fatal ingestion failure', {
      ownerId,
      profileId,
      error: ingestionError,
    });
  }

  return {
    profileId,
    kundaliId: profileId,
    chartData: chartSnapshot,
    chartSchemaVersion: chartSchema.chartSchemaVersion,
    dashaDepth: chartSchema.dashaDepth,
    dashaPeriodKey: chartSchema.dashaPeriodKey,
    ingestion: ingestion ?? null,
    ingestionStatus: ingestion ? 'ready' : 'degraded',
    ingestionError,
  };
}

router.post('/v1/chart/calculate', async (req, res) => {
  try {
    const parsed = GenerateChartSchema.safeParse(req.body ?? {});
    if (!parsed.success) {
      return res.status(400).json({ error: 'Invalid body', details: parsed.error.flatten() });
    }

    const { chartSnapshot, chartSchema } = await calculateChartPreview(parsed.data);

    return res.status(200).json({
      ok: true,
      chartData: chartSnapshot,
      chartSchemaVersion: chartSchema.chartSchemaVersion,
      dashaDepth: chartSchema.dashaDepth,
      dashaPeriodKey: chartSchema.dashaPeriodKey,
      source: 'preview',
    });
  } catch (error) {
    return res.status(500).json({ error: 'Failed to calculate chart preview', details: String(error) });
  }
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
      displayName: parsed.data.name,
      place: parsed.data.place,
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

router.post('/v1/rag/query', requireFirebaseAuth, async (_req, res) => {
  // Vector search disabled - agent uses direct PHP API calls
  return res.json({ chunks: [] });
});

router.post('/v1/chart/generate', requireFirebaseAuth, async (req, res) => {
  try {
    const parsed = GenerateChartSchema.safeParse(req.body ?? {});
    if (!parsed.success) {
      return res.status(400).json({ error: 'Invalid body', details: parsed.error.flatten() });
    }

    let quotaStatus: QuotaStatusSnapshot | undefined;
    if (env.QUOTA_ENFORCEMENT_ENABLED) {
      const subscriptions = getSubscriptionsRepository();
      const usageQuotas = getUsageQuotasRepository();
      const subscription = await subscriptions.getByOwnerId(req.user!.uid);
      const hasPro = hasActiveProEntitlement(subscription);

      const consumed = await usageQuotas.consumeQuota(req.user!.uid, hasPro, 'kundli_generate');
      if (!consumed.allowed) {
        return res.status(429).json({
          error: 'Monthly quota exceeded',
          details: 'Kundli generation monthly quota exceeded. Please wait for reset or upgrade your plan.',
          code: 'MONTHLY_QUOTA_EXCEEDED',
          quotaType: 'kundli_generate',
          quotaStatus: consumed.status,
        });
      }

      quotaStatus = consumed.status;
    }

    const queryIngest = (req.query as Record<string, unknown>)?.ingest;
    if (queryIngest === 'false' || queryIngest === false) {
      parsed.data.ingest = false;
    }

    const result = await generateAndIngestChart(req.user!.uid, parsed.data);

    return res.status(201).json({
      ok: true,
      ...result,
      quotaStatus,
    });
  } catch (error) {
    return res.status(500).json({ error: 'Failed to generate and ingest chart', details: String(error) });
  }
});

router.get('/api/compatibility', (_req, res) => {
  // Compatibility endpoint disabled - use agent for compatibility questions
  return res.status(501).json({ error: 'Compatibility endpoint deprecated. Use /v1/rag/agent instead.' });
});

router.post('/v1/transit-chart', requireFirebaseAuth, async (req, res) => {
  try {
    const parsed = TransitChartSchema.safeParse(req.body ?? {});
    if (!parsed.success) {
      return res.status(400).json({ error: 'Invalid body', details: parsed.error.flatten() });
    }

    const timeZone = parsed.data.time_zone ?? parsed.data.timezone ?? parsed.data.timeZone;
    if (!timeZone) {
      return res.status(400).json({ error: 'Invalid body', details: { time_zone: ['Required time_zone/timezone/timeZone'] } });
    }

    const natalYear = parsed.data.year ?? parsed.data.t_year;
    const natalMonth = parsed.data.month ?? parsed.data.t_month;
    const natalDay = parsed.data.day ?? parsed.data.t_day;
    const natalHour = parsed.data.hour ?? parsed.data.t_hour;
    const natalMin = parsed.data.min ?? parsed.data.t_min;
    const natalSec = parsed.data.sec ?? 0;

    if (
      natalYear === undefined ||
      natalMonth === undefined ||
      natalDay === undefined ||
      natalHour === undefined ||
      natalMin === undefined
    ) {
      return res.status(400).json({ error: 'Invalid body', details: { year: ['Required natal date/time fields'] } });
    }

    const data = await fetchTransitChart(
      {
        latitude: parsed.data.latitude,
        longitude: parsed.data.longitude,
        year: natalYear,
        month: natalMonth,
        day: natalDay,
        hour: natalHour,
        min: natalMin,
        sec: natalSec,
        time_zone: timeZone,
      },
      { nesting: parsed.data.nesting ?? 4 }
    );

    return res.json(data);
  } catch (error) {
    return res.status(500).json({ error: 'Failed to fetch transit chart', details: String(error) });
  }
});

export default router;
