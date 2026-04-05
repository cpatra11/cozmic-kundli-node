import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import dotenv from 'dotenv';
import { getChatRepository } from '../src/repositories/chatRepository.js';
import { getChartJobsRepository } from '../src/repositories/chartJobsRepository.js';
import { getRagChunksRepository } from '../src/repositories/ragChunksRepository.js';
import { getRagProfilesRepository } from '../src/repositories/ragProfilesRepository.js';
import { getRagSourcesRepository } from '../src/repositories/ragSourcesRepository.js';
import { ingestChartPayloadForProfile } from '../src/services/ragPipeline.js';
import { queryRagChunks } from '../src/services/ragPipeline.js';
import { runKundliAgent } from '../src/services/kundliAgent.js';
import { getPostgresPool } from '../src/services/postgresClient.js';
import { applyPendingMigrations } from '../src/services/postgresMigrations.js';
import { extractChartSchemaInfo } from '../src/services/chartSnapshot.js';

dotenv.config();

function sampleRawPayload() {
  return {
    chart: {
      graha: {
        Su: { rashi: 1, house_number: 1, degree: 10.25 },
        Mo: { rashi: 4, house_number: 4, degree: 18.5, nakshatra: 'Pushya', nakshatra_lord: 'Sa' },
        Ma: { rashi: 10, house_number: 10, degree: 3.1 },
      },
      lagna: {
        Lg: { rashi: 1, house_number: 1, degree: 12.4 },
      },
      varga: {
        D1: {
          graha: {
            Su: { rashi: 1, house_number: 1, degree: 10.25 },
            Mo: { rashi: 4, house_number: 4, degree: 18.5 },
          },
          lagna: {
            Lg: { rashi: 1, house_number: 1, degree: 12.4 },
          },
        },
        D9: {
          graha: {
            Ve: { rashi: 7, house_number: 7, degree: 2.2 },
            Ju: { rashi: 9, house_number: 9, degree: 14.8 },
          },
          lagna: {
            Lg: { rashi: 5, house_number: 1, degree: 20.1 },
          },
        },
      },
      dasha: {
        type: 'vimshottari',
        start: '2020-01-01T00:00:00Z',
        end: '2040-01-01T00:00:00Z',
      },
      panchanga: {
        tithi: 'Shukla Panchami',
        nakshatra: 'Pushya',
      },
    },
  };
}

async function cleanupOwner(ownerId: string): Promise<void> {
  const pool = getPostgresPool();
  if (!pool) return;

  await applyPendingMigrations(pool);

  await pool.query(`DELETE FROM chart_jobs WHERE owner_id = $1`, [ownerId]);
  await pool.query(`DELETE FROM monthly_usage_counters WHERE owner_id = $1`, [ownerId]);
  await pool.query(`DELETE FROM chat_messages WHERE owner_id = $1`, [ownerId]);
  await pool.query(`DELETE FROM chat_sessions WHERE owner_id = $1`, [ownerId]);
  await pool.query(`DELETE FROM rag_chunks WHERE owner_id = $1`, [ownerId]);
  await pool.query(`DELETE FROM rag_api_sources WHERE owner_id = $1`, [ownerId]);
  await pool.query(`DELETE FROM rag_profiles WHERE owner_id = $1`, [ownerId]);
  await pool.query(`DELETE FROM chart_vectors WHERE owner_id = $1`, [ownerId]);
  await pool.query(`DELETE FROM charts WHERE owner_id = $1`, [ownerId]);
}

async function main() {
  if (!process.env.DATABASE_URL) {
    throw new Error('DATABASE_URL is required to run integration tests');
  }

  const ownerId = `it_owner_${Date.now()}_${randomUUID().slice(0, 8)}`;
  const newIngestProfileId = `it_ingest_${randomUUID().slice(0, 8)}`;
  const profileId = `it_profile_${randomUUID().slice(0, 8)}`;
  const sourceDocId = `it_src_${randomUUID().slice(0, 10)}`;

  const ragProfiles = getRagProfilesRepository();
  const ragSources = getRagSourcesRepository();
  const ragChunks = getRagChunksRepository();
  const chat = getChatRepository();
  const chartJobs = getChartJobsRepository();

  const now = Date.now();

  try {
    await cleanupOwner(ownerId);

    // Regression: new-profile ingest must not violate rag_api_sources -> rag_profiles FK.
    const ingestResult = await ingestChartPayloadForProfile({
      ownerId,
      profileId: newIngestProfileId,
      displayName: 'Integration New Ingest Profile',
      place: 'Talcher',
      kundli: {
        latitude: 20.95,
        longitude: 85.22,
        year: 2002,
        month: 3,
        day: 14,
        hour: 10,
        min: 39,
        sec: 0,
        time_zone: '+05:30',
      },
      payload: sampleRawPayload(),
      endpoint: 'calculate',
      tags: ['integration', 'fk-order'],
    });

    assert.equal(ingestResult.profileId, newIngestProfileId, 'Expected ingest to succeed for brand-new profile');
    const ingestedProfile = await ragProfiles.getByOwnerAndProfileId(ownerId, newIngestProfileId);
    assert.ok(ingestedProfile, 'Expected newly ingested profile row to exist');

    await ragProfiles.upsert({
      ownerId,
      profileId,
      displayName: 'Integration Test Profile',
      place: 'Bhubaneswar',
      kundliSignature: `sig_${randomUUID().slice(0, 12)}`,
      chartVersion: `ver_${randomUUID().slice(0, 12)}`,
      kundliInput: {
        latitude: 20.2961,
        longitude: 85.8245,
        year: 1994,
        month: 7,
        day: 14,
        hour: 10,
        min: 30,
        sec: 0,
        time_zone: '+05:30',
      },
      latestSourceDocId: sourceDocId,
      sourceCount: 1,
      createdAt: now,
      updatedAt: now,
    });

    await ragSources.upsert(sourceDocId, {
      ownerId,
      profileId,
      displayName: 'Integration Test Profile',
      place: 'Bhubaneswar',
      sourceType: 'be1',
      endpoint: 'calculate',
      requestKey: `req_${randomUUID().slice(0, 12)}`,
      payloadHash: `hash_${randomUUID().slice(0, 12)}`,
      chartSchemaVersion: extractChartSchemaInfo(sampleRawPayload()).chartSchemaVersion,
      dashaDepth: extractChartSchemaInfo(sampleRawPayload()).dashaDepth,
      dashaPeriodKey: extractChartSchemaInfo(sampleRawPayload()).dashaPeriodKey,
      rawPayload: sampleRawPayload(),
      chartSnapshot: sampleRawPayload(),
      preview: 'integration preview',
      tags: ['integration', 'typed'],
      createdAt: now,
    });

    const storedSource = await ragSources.getById(sourceDocId);
    assert.ok(storedSource, 'Expected saved source document to be readable');
    assert.equal(storedSource?.data.chartSchemaVersion, 'mahadasha-first', 'Saved source should default to mahadasha-first schema');
    assert.equal(storedSource?.data.dashaDepth, 1, 'Saved source should record shallow dasha depth');
    assert.equal(storedSource?.data.dashaPeriodKey, undefined, 'Saved source should not invent a dasha period key');

    await ragChunks.upsertMany([
      {
        id: `it_chk_${randomUUID().slice(0, 10)}`,
        data: {
          ownerId,
          profileId,
          sourceDocId,
          sourceType: 'be1',
          endpoint: 'calculate',
          chunkIndex: 0,
          text: 'Moon in Pushya and 10th house Mars indicate work discipline and structured growth.',
          textPreview: 'Moon in Pushya and 10th house Mars...',
          embedding: Array.from({ length: 192 }, (_, i) => (i % 7 === 0 ? 0.25 : 0.01)),
          embeddingModel: 'integration-fixed-v1',
          embeddingDim: 192,
          tokenEstimate: 24,
          tags: ['integration', 'career'],
          createdAt: now,
        },
      },
    ]);

    const queried = await queryRagChunks({
      ownerId,
      profileId,
      message: 'Tell me about my career pattern from chart data',
      topK: 5,
    });

    assert.ok(queried.length >= 1, 'Expected at least one rag chunk result');
    assert.equal(queried[0]?.profileId, profileId, 'RAG result should match profile id');

    const sessionId = await chat.createSession({
      ownerId,
      title: 'Integration chat session',
      kundaliId: profileId,
      createdAt: now,
      updatedAt: now,
    });

    await chat.createMessage({
      ownerId,
      sessionId,
      role: 'user',
      message: 'How is my marriage outlook?',
      mode: 'mini',
      requestId: `req_${randomUUID().slice(0, 10)}`,
      kundaliId: profileId,
      createdAt: now + 10,
    });

    await chat.createMessage({
      ownerId,
      sessionId,
      role: 'assistant',
      message: 'Based on D1/D9 reference this looks stable with gradual maturity.',
      mode: 'mini',
      model: 'integration-test-model',
      requestId: `req_${randomUUID().slice(0, 10)}`,
      kundaliId: profileId,
      createdAt: now + 20,
    });

    const messages = await chat.listSessionMessages(ownerId, sessionId, 50);
    assert.equal(messages.length, 2, 'Expected 2 persisted chat messages');
    assert.equal(messages[0]?.role, 'user');
    assert.equal(messages[1]?.role, 'assistant');

    const jobId = `it_job_${randomUUID().slice(0, 10)}`;
    await chartJobs.create(ownerId, jobId, {
      ownerId,
      profileId,
      status: 'queued',
      request: { test: true },
      createdAt: now,
      updatedAt: now,
    });

    await chartJobs.patch(ownerId, jobId, {
      status: 'completed',
      result: { ok: true },
      updatedAt: now + 100,
    });

    const job = await chartJobs.get(ownerId, jobId);
    assert.ok(job, 'Expected chart job record');
    assert.equal(job?.data.status, 'completed');

    const agent = await runKundliAgent({
      ownerId,
      mode: 'mini',
      profileId,
      message: 'What does my D9 indicate for relationship dynamics?',
    });

    assert.ok(agent.answer && agent.answer.trim().length > 0, 'Expected non-empty agent answer');
    assert.equal(agent.grounding?.profileId, profileId, 'Agent grounding should use typed profile');

    console.log('✅ Typed persistence integration passed.');
    console.log(`   ownerId: ${ownerId}`);
    console.log(`   profileId: ${profileId}`);
    console.log(`   sessionId: ${sessionId}`);
    console.log(`   jobId: ${jobId}`);
  } finally {
    await cleanupOwner(ownerId);
  }
}

main().catch((error) => {
  console.error('❌ Typed persistence integration failed:', error);
  process.exit(1);
});
