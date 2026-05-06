import { Router } from 'express';
import { createHash } from 'crypto';
import { requireFirebaseAuth } from '../middleware/auth.js';
import type { RagProfileDocument } from '../models/firestoreModels.js';
import { getRagProfilesRepository } from '../repositories/ragProfilesRepository.js';
import { getChatRepository } from '../repositories/chatRepository.js';
import { getUsageQuotasRepository } from '../repositories/usageQuotasRepository.js';
import { buildChartSnapshot } from '../services/chartSnapshot.js';
import { fetchBe1Calculate } from '../services/be1Client.js';
import { getPostgresPool } from '../services/postgresClient.js';
import { applyPendingMigrations } from '../services/postgresMigrations.js';
import { cacheDelete, cacheGetJson, cacheSetJson } from '../services/valkeyCache.js';

const router = Router();
const ragProfiles = getRagProfilesRepository();
const usageQuotas = getUsageQuotasRepository();

const KUNDALI_CACHE_TTL_MS = 20_000;

function getIfNoneMatchHeader(value: string | string[] | undefined): string | null {
  if (Array.isArray(value)) return value[0] ?? null;
  return typeof value === 'string' ? value : null;
}

function buildKundaliEtag(
  ownerId: string,
  kundaliId: string,
  chartVersion: unknown,
  updatedAt: unknown
): string {
  const payload = `${ownerId}:${kundaliId}:${String(chartVersion ?? '')}:${String(updatedAt ?? '')}`;
  const digest = createHash('sha1').update(payload).digest('hex');
  return `W/"${digest}"`;
}

function buildDocumentId(ownerId: string, profileId: string): string {
  return `${ownerId}__${profileId}`;
}

function extractProfileId(identifier: string, ownerId: string): string {
  if (identifier.startsWith(`${ownerId}__`)) {
    return identifier.slice(ownerId.length + 2);
  }
  return identifier;
}

function buildKundaliSummary(doc: RagProfileDocument) {
  return {
    id: buildDocumentId(doc.ownerId, doc.profileId),
    kundaliId: doc.profileId,
    name: doc.displayName ?? doc.profileId,
    displayName: doc.displayName ?? doc.profileId,
    place: doc.place,
    createdAt: doc.createdAt,
    updatedAt: doc.updatedAt,
    kundliInput: doc.kundliInput,
  };
}

function parseNestingParam(value: unknown, fallback = 5): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(1, Math.min(5, Math.floor(parsed)));
}

router.get('/v1/kundalis', requireFirebaseAuth, async (req, res) => {
  try {
    const rows = await ragProfiles.listByOwner(req.user!.uid, 100);

    res.setHeader('Cache-Control', 'private, max-age=10, stale-while-revalidate=20');
    res.setHeader('Vary', 'Authorization');

    return res.json({
      kundalis: rows.map((doc) => buildKundaliSummary(doc)),
    });
  } catch (error) {
    return res.status(500).json({ error: 'Failed to list kundalis', details: String(error) });
  }
});

router.get('/v1/kundalis/:kundaliId', requireFirebaseAuth, async (req, res) => {
  try {
    const identifier = String(req.params.kundaliId ?? '').trim();
    if (!identifier) {
      return res.status(400).json({ error: 'Missing kundaliId path parameter' });
    }

    const ownerId = req.user!.uid;
    const cacheKey = `kundali:${ownerId}:${identifier}`;

    try {
      const cachedData = await cacheGetJson<{ etag: string; body: Record<string, unknown> }>(cacheKey);
      if (cachedData) {
        const { etag, body } = cachedData;
        const ifNoneMatch = getIfNoneMatchHeader(req.headers['if-none-match']);
        res.setHeader('Cache-Control', 'private, max-age=15, stale-while-revalidate=30');
        res.setHeader('ETag', etag);
        res.setHeader('Vary', 'Authorization');
        if (ifNoneMatch === etag) {
          return res.status(304).end();
        }
        return res.json(body);
      }
    } catch (redisError) {
      console.warn('Redis cache miss or error:', redisError);
    }

    const profileId = extractProfileId(identifier, ownerId);
    const profileDoc = await ragProfiles.getByOwnerAndProfileId(ownerId, profileId);

    if (!profileDoc) {
      return res.status(404).json({ error: 'Kundali not found' });
    }

    const input = profileDoc.kundliInput;
    if (!input) {
      return res.status(422).json({ error: 'Kundli input snapshot unavailable for this profile' });
    }

    const calculated = await fetchBe1Calculate(input);
    const chartData = buildChartSnapshot(calculated);

    const responseBody = {
      kundali: {
        id: buildDocumentId(profileDoc.ownerId, profileDoc.profileId),
        kundaliId: profileDoc.profileId,
        ownerId: profileDoc.ownerId,
        name: profileDoc.displayName ?? profileDoc.profileId,
        displayName: profileDoc.displayName ?? profileDoc.profileId,
        place: profileDoc.place,
        chartData,
        rawPayload: calculated,
        createdAt: profileDoc.createdAt,
        updatedAt: profileDoc.updatedAt,
        kundliInput: profileDoc.kundliInput,
        chartVersion: profileDoc.chartVersion,
      },
    };

    const etag = buildKundaliEtag(
      ownerId,
      profileDoc.profileId,
      profileDoc.chartVersion,
      profileDoc.updatedAt
    );

    const ifNoneMatch = getIfNoneMatchHeader(req.headers['if-none-match']);
    res.setHeader('Cache-Control', 'private, max-age=15, stale-while-revalidate=30');
    res.setHeader('ETag', etag);
    res.setHeader('Vary', 'Authorization');
    if (ifNoneMatch === etag) {
      return res.status(304).end();
    }

    try {
      await cacheSetJson(cacheKey, { etag, body: responseBody }, Math.floor(KUNDALI_CACHE_TTL_MS / 1000));
    } catch (redisError) {
      console.warn('Redis set error:', redisError);
    }

    return res.json(responseBody);
  } catch (error) {
    return res.status(500).json({ error: 'Failed to load kundali', details: String(error) });
  }
});

router.get('/v1/kundalis/:kundaliId/dasha', requireFirebaseAuth, async (req, res) => {
  try {
    const identifier = String(req.params.kundaliId ?? '').trim();
    if (!identifier) {
      return res.status(400).json({ error: 'Missing kundaliId path parameter' });
    }

    const ownerId = req.user!.uid;
    const profileId = extractProfileId(identifier, ownerId);
    const profileDoc = await ragProfiles.getByOwnerAndProfileId(ownerId, profileId);
    if (!profileDoc) {
      return res.status(404).json({ error: 'Kundali not found' });
    }

    const nesting = parseNestingParam((req.query as Record<string, unknown>)?.nesting, 5);
    const rawPeriodKey = String((req.query as Record<string, unknown>)?.periodKey ?? '').trim();
    const periodKey = rawPeriodKey ? rawPeriodKey.slice(0, 4) : undefined;

    const input = profileDoc.kundliInput;
    if (!input) {
      return res.status(422).json({ error: 'Kundli input snapshot unavailable for this profile' });
    }

    const calculated = await fetchBe1Calculate(input, {
      nesting,
      ...(periodKey ? { periodKey } : {}),
    });
    // IMPORTANT: Do NOT run on-demand dasha through buildChartSnapshot here.
    // Snapshot normalization intentionally strips nested dasha periods for storage,
    // but this endpoint must return deep periods for frontend drill-down.
    const dasha = (calculated as any)?.chart?.dasha;

    if (!dasha || typeof dasha !== 'object') {
      return res.status(502).json({ error: 'Dasha payload missing from calculated chart response' });
    }

    return res.json({
      kundaliId: profileDoc.profileId,
      dasha,
      dashaDepth: nesting,
      dashaPeriodKey: periodKey ?? null,
      source: 'on-demand',
    });
  } catch (error) {
    return res.status(500).json({ error: 'Failed to load dasha on demand', details: String(error) });
  }
});

router.patch('/v1/kundalis/:kundaliId', requireFirebaseAuth, async (req, res) => {
  try {
    const identifier = String(req.params.kundaliId ?? '').trim();
    if (!identifier) {
      return res.status(400).json({ error: 'Missing kundaliId path parameter' });
    }

    const displayName = typeof req.body?.displayName === 'string' ? req.body.displayName.trim() : '';
    const place = typeof req.body?.place === 'string' ? req.body.place.trim() : '';

    const profileId = extractProfileId(identifier, req.user!.uid);
    const profileDoc = await ragProfiles.getByOwnerAndProfileId(req.user!.uid, profileId);
    if (!profileDoc) {
      return res.status(404).json({ error: 'Kundali not found' });
    }

    const updatedAt = Date.now();
    await ragProfiles.patchMetadata(req.user!.uid, profileDoc.profileId, {
      displayName: displayName || undefined,
      place: place || undefined,
      updatedAt,
    });

    await cacheDelete(`kundali:${req.user!.uid}:${profileDoc.profileId}`);

    return res.json({
      ok: true,
      kundaliId: profileDoc.profileId,
      updatedAt,
      displayName: displayName || profileDoc.displayName || profileDoc.profileId,
      place: place || profileDoc.place || '',
    });
  } catch (error) {
    return res.status(500).json({ error: 'Failed to update kundali', details: String(error) });
  }
});

router.delete('/v1/kundalis/:kundaliId', requireFirebaseAuth, async (req, res) => {
  try {
    const identifier = String(req.params.kundaliId ?? '').trim();
    if (!identifier) {
      return res.status(400).json({ error: 'Missing kundaliId path parameter' });
    }

    const profileId = extractProfileId(identifier, req.user!.uid);
    const profileDoc = await ragProfiles.getByOwnerAndProfileId(req.user!.uid, profileId);
    if (!profileDoc) {
      return res.status(404).json({ error: 'Kundali not found' });
    }

    const chatRepository = getChatRepository();

    await Promise.all([
      ragProfiles.deleteByOwnerAndProfileId(req.user!.uid, profileDoc.profileId),
      chatRepository.deleteSessionsByOwnerKundali(req.user!.uid, profileDoc.profileId),
    ]);

    const pool = getPostgresPool();
    if (pool) {
      await applyPendingMigrations(pool);
      await pool.query(`DELETE FROM chart_vectors WHERE owner_id = $1 AND kundali_id = $2`, [req.user!.uid, profileDoc.profileId]);
      await pool.query(`DELETE FROM charts WHERE owner_id = $1 AND kundali_id = $2`, [req.user!.uid, profileDoc.profileId]);
    }

    await usageQuotas.refundQuota(req.user!.uid, 'kundli_generate');

    await cacheDelete(`kundali:${req.user!.uid}:${profileDoc.profileId}`);

    return res.status(204).send();
  } catch (error) {
    return res.status(500).json({ error: 'Failed to delete kundali', details: String(error) });
  }
});

export default router;