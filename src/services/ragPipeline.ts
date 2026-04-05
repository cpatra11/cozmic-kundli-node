import { env } from '../config/env.js';
import { type RagApiSourceDocument, type RagChunkDocument, type RagChunkResult, type RagProfileDocument } from '../models/firestoreModels.js';
import { getRagChunksRepository } from '../repositories/ragChunksRepository.js';
import { getRagProfilesRepository } from '../repositories/ragProfilesRepository.js';
import { getRagSourcesRepository } from '../repositories/ragSourcesRepository.js';
import { fetchKundliSnapshot, type KundliSnapshotInput } from './be1Client.js';
import { cosineSimilarity, embedTextDeterministic } from './embeddings.js';
import { getPostgresPool } from './postgresClient.js';
import { stableHash, toDocId } from './hash.js';
import { buildChartSnapshot, extractChartSchemaInfo, type ChartSchemaInfo } from './chartSnapshot.js';
import { PostgresVectorStore } from './vector/postgresVectorStore.js';

interface IngestInput {
  ownerId: string;
  profileId: string;
  displayName?: string;
  place?: string;
  kundli: KundliSnapshotInput;
}

interface IngestChartPayloadInput {
  ownerId: string;
  profileId: string;
  displayName?: string;
  place?: string;
  kundli: KundliSnapshotInput;
  payload: unknown;
  endpoint?: string;
  tags?: string[];
}

interface QueryInput {
  ownerId: string;
  profileId?: string;
  message: string;
  topK?: number;
}

interface IngestResult {
  profileId: string;
  sourceDocId: string;
  chunkCount: number;
  endpoint: string;
}

const pgVectorStore = new PostgresVectorStore();
const ragProfiles = getRagProfilesRepository();
const ragSources = getRagSourcesRepository();
const ragChunks = getRagChunksRepository();

function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

function stringValueOf(value: unknown): string {
  if (value === null || value === undefined) return '';
  if (typeof value === 'string') return value;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  return JSON.stringify(value);
}

function flattenPayload(value: unknown, path = 'root', out: string[] = []): string[] {
  if (value === null || value === undefined) {
    out.push(`${path}: null`);
    return out;
  }

  if (Array.isArray(value)) {
    value.forEach((item, index) => {
      flattenPayload(item, `${path}[${index}]`, out);
    });
    return out;
  }

  if (typeof value === 'object') {
    Object.entries(value as Record<string, unknown>).forEach(([key, item]) => {
      flattenPayload(item, `${path}.${key}`, out);
    });
    return out;
  }

  out.push(`${path}: ${stringValueOf(value)}`);
  return out;
}

function chunkLines(lines: string[], maxChars = 900): string[] {
  const chunks: string[] = [];
  let current = '';

  for (const line of lines) {
    const next = current ? `${current}\n${line}` : line;
    if (next.length > maxChars) {
      if (current) chunks.push(current);

      if (line.length <= maxChars) {
        current = line;
      } else {
        const parts = line.match(new RegExp(`.{1,${maxChars}}`, 'g')) ?? [line.slice(0, maxChars)];
        chunks.push(...parts.slice(0, -1));
        current = parts[parts.length - 1] ?? '';
      }
    } else {
      current = next;
    }
  }

  if (current) chunks.push(current);
  return chunks;
}

function makeKundliSignature(kundli: KundliSnapshotInput): string {
  const normalized = {
    latitude: kundli.latitude,
    longitude: kundli.longitude,
    year: kundli.year,
    month: kundli.month,
    day: kundli.day,
    hour: kundli.hour,
    min: kundli.min,
    sec: kundli.sec ?? 0,
    time_zone: kundli.time_zone,
  };

  return stableHash(JSON.stringify(normalized));
}

function toKundliInputDocument(kundli: KundliSnapshotInput): RagProfileDocument['kundliInput'] {
  return {
    latitude: kundli.latitude,
    longitude: kundli.longitude,
    year: kundli.year,
    month: kundli.month,
    day: kundli.day,
    hour: kundli.hour,
    min: kundli.min,
    sec: kundli.sec ?? 0,
    time_zone: kundli.time_zone,
  };
}

function toRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}

function asText(value: unknown): string | null {
  if (typeof value !== 'string') return null;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

async function upsertChartRow(input: {
  ownerId: string;
  profileId: string;
  displayName?: string;
  place?: string;
  sourceDocId: string;
  requestKey: string;
  payloadHash: string;
  chartSnapshot: unknown;
  chartSchema: ChartSchemaInfo;
  kundliInput: RagProfileDocument['kundliInput'];
  tags: string[];
}): Promise<void> {
  const pool = getPostgresPool();
  if (!pool) return;

  const snapshot = toRecord(input.chartSnapshot);
  const panchanga = toRecord(snapshot.panchanga);
  const dasha = toRecord(snapshot.dasha);

  const now = Date.now();
  const chartDateTime = `${input.kundliInput.year}-${String(input.kundliInput.month).padStart(2, '0')}-${String(
    input.kundliInput.day
  ).padStart(2, '0')}T${String(input.kundliInput.hour).padStart(2, '0')}:${String(input.kundliInput.min).padStart(2, '0')}:${String(
    input.kundliInput.sec
  ).padStart(2, '0')}`;

  await pool.query(
    `
      INSERT INTO charts (
        owner_id,
        kundali_id,
        request_key,
        ingestion_status,
        name,
        place,
        display_name,
        tags,
        panchanga,
        chart_data,
        chart_schema_version,
        dasha_depth,
        dasha_period_key,
        raw_payload_ref,
        chart_signature,
        chart_datetime,
        location_lat,
        location_lng,
        timezone,
        tithi,
        nakshatra,
        dasha_current,
        created_at,
        updated_at,
        deleted_at
      ) VALUES (
        $1,
        $2,
        $3,
        'ready',
        $4,
        $5,
        $6,
        $7::jsonb,
        $8::jsonb,
        $9::jsonb,
        $10,
        $11,
        $12,
        $13,
        $14,
        $15,
        $16,
        $17,
        $18,
        $19,
        $20,
        $21,
        $22,
        $23,
        NULL
      )
      ON CONFLICT (owner_id, kundali_id)
      DO UPDATE SET
        request_key = EXCLUDED.request_key,
        ingestion_status = EXCLUDED.ingestion_status,
        name = EXCLUDED.name,
        place = EXCLUDED.place,
        display_name = EXCLUDED.display_name,
        tags = EXCLUDED.tags,
        panchanga = EXCLUDED.panchanga,
        chart_data = EXCLUDED.chart_data,
        chart_schema_version = EXCLUDED.chart_schema_version,
        dasha_depth = EXCLUDED.dasha_depth,
        dasha_period_key = EXCLUDED.dasha_period_key,
        raw_payload_ref = EXCLUDED.raw_payload_ref,
        chart_signature = EXCLUDED.chart_signature,
        chart_datetime = EXCLUDED.chart_datetime,
        location_lat = EXCLUDED.location_lat,
        location_lng = EXCLUDED.location_lng,
        timezone = EXCLUDED.timezone,
        tithi = EXCLUDED.tithi,
        nakshatra = EXCLUDED.nakshatra,
        dasha_current = EXCLUDED.dasha_current,
        updated_at = EXCLUDED.updated_at,
        deleted_at = NULL
    `,
    [
      input.ownerId,
      input.profileId,
      input.requestKey,
      input.displayName ?? null,
      input.place ?? null,
      input.displayName ?? input.profileId,
      JSON.stringify(input.tags),
      Object.keys(panchanga).length > 0 ? JSON.stringify(panchanga) : null,
      JSON.stringify(input.chartSnapshot),
      input.chartSchema.chartSchemaVersion,
      input.chartSchema.dashaDepth,
      input.chartSchema.dashaPeriodKey ?? null,
      input.sourceDocId,
      input.payloadHash,
      chartDateTime,
      input.kundliInput.latitude,
      input.kundliInput.longitude,
      input.kundliInput.time_zone,
      asText(panchanga.tithi),
      asText(panchanga.nakshatra),
      asText(dasha.current),
      now,
      now,
    ]
  );
}

export async function ingestKundliForProfile(input: IngestInput): Promise<IngestResult> {
  const payload = await fetchKundliSnapshot(input.kundli);
  return ingestChartPayloadForProfile({
    ownerId: input.ownerId,
    profileId: input.profileId,
    displayName: input.displayName,
    place: input.place,
    kundli: input.kundli,
    payload,
    endpoint: 'calculate',
    tags: ['kundli', 'be1', 'calculate'],
  });
}

export async function ingestChartPayloadForProfile(input: IngestChartPayloadInput): Promise<IngestResult> {
  const now = Date.now();
  const endpoint = input.endpoint ?? 'calculate';

  const payloadRaw = JSON.stringify(input.payload);
  const payloadHash = stableHash(payloadRaw);
  const chartSchema = extractChartSchemaInfo(input.payload);
  const requestKey = stableHash(
    JSON.stringify({ endpoint, ownerId: input.ownerId, profileId: input.profileId, kundli: toKundliInputDocument(input.kundli) })
  );

  const sourceDocId = toDocId('src', `${input.ownerId}:${input.profileId}:${requestKey}:${payloadHash}`);
  const chartSnapshot = buildChartSnapshot(input.payload);

  const sourceDoc: RagApiSourceDocument = {
    ownerId: input.ownerId,
    profileId: input.profileId,
    displayName: input.displayName,
    place: input.place,
    sourceType: 'be1',
    endpoint,
    requestKey,
    payloadHash,
    chartSchemaVersion: chartSchema.chartSchemaVersion,
    dashaDepth: chartSchema.dashaDepth,
    dashaPeriodKey: chartSchema.dashaPeriodKey,
    rawPayload: input.payload,
    chartSnapshot,
    preview: payloadRaw.slice(0, 1800),
    tags: input.tags ?? ['kundli', 'be1', endpoint],
    createdAt: now,
  };

  const existingProfile = await ragProfiles.getByOwnerAndProfileId(input.ownerId, input.profileId);
  const profileDoc: RagProfileDocument = {
    ownerId: input.ownerId,
    profileId: input.profileId,
    displayName: input.displayName,
    place: input.place,
    kundliSignature: makeKundliSignature(input.kundli),
    chartVersion: payloadHash,
    kundliInput: toKundliInputDocument(input.kundli),
    latestSourceDocId: sourceDocId,
    sourceCount: (existingProfile?.sourceCount ?? 0) + 1,
    updatedAt: now,
    createdAt: existingProfile?.createdAt ?? now,
  };

  // IMPORTANT: Upsert profile first so rag_api_sources FK (owner_id, profile_id) is always satisfied.
  await ragProfiles.upsert(profileDoc);

  await ragSources.upsert(sourceDocId, sourceDoc);

  await upsertChartRow({
    ownerId: input.ownerId,
    profileId: input.profileId,
    displayName: input.displayName,
    place: input.place,
    sourceDocId,
    requestKey,
    payloadHash,
    chartSnapshot,
    chartSchema,
    kundliInput: toKundliInputDocument(input.kundli),
    tags: input.tags ?? ['kundli', 'be1', endpoint],
  });

  const lines = flattenPayload(input.payload);
  const chunkTexts = chunkLines(lines, 900).slice(0, 200);

  const embeddingDim = env.EMBEDDING_DIM;
  const chunkRecords = chunkTexts.map((chunkText, index) => {
    const embedding = embedTextDeterministic(chunkText, embeddingDim);
    const chunkDoc: RagChunkDocument = {
      ownerId: input.ownerId,
      profileId: input.profileId,
      sourceDocId,
      sourceType: 'be1',
      endpoint,
      chunkIndex: index,
      text: chunkText,
      textPreview: chunkText.slice(0, 220),
      embedding: embedding.vector,
      embeddingModel: embedding.model,
      embeddingDim: embedding.dimension,
      tokenEstimate: estimateTokens(chunkText),
      tags: input.tags ?? ['kundli', endpoint],
      createdAt: now,
    };

    const chunkId = toDocId('chk', `${sourceDocId}:${index}`);
    return {
      id: chunkId,
      data: chunkDoc,
    };
  });

  await ragChunks.upsertMany(chunkRecords);

  await pgVectorStore.upsertChunks(
    chunkTexts.map((chunkText, index) => {
      const embedding = embedTextDeterministic(chunkText, embeddingDim);
      return {
        data: {
          ownerId: input.ownerId,
          profileId: input.profileId,
          kundaliId: input.profileId,
          sourceDocId,
          sourceType: 'be1' as const,
          endpoint,
          chunkIndex: index,
          text: chunkText,
          textPreview: chunkText.slice(0, 220),
          embedding: embedding.vector,
          embeddingModel: embedding.model,
          embeddingDim: embedding.dimension,
          tokenEstimate: estimateTokens(chunkText),
          tags: input.tags ?? ['kundli', endpoint],
          createdAt: now,
        },
      };
    })
  );

  return {
    profileId: input.profileId,
    sourceDocId,
    chunkCount: chunkTexts.length,
    endpoint,
  };
}

export async function queryRagChunks(input: QueryInput): Promise<RagChunkResult[]> {
  const topK = Math.min(Math.max(input.topK ?? 8, 1), 20);
  const candidateWindow = Math.max(topK * 8, 40);
  const queryEmbedding = embedTextDeterministic(input.message, env.EMBEDDING_DIM).vector;

  if (env.PGVECTOR_ENABLED.trim().toLowerCase() === 'true') {
    const vectorResults = (await pgVectorStore.searchChunks({
      ownerId: input.ownerId,
      kundaliId: input.profileId,
      queryEmbedding,
      topK,
      candidateWindow,
    })) as RagChunkResult[];

    if (vectorResults.length > 0) {
      return vectorResults;
    }
  }

  const candidates = await ragChunks.listForQuery(input.ownerId, input.profileId, candidateWindow);

  if (candidates.length === 0) {
    return [];
  }

  const scored = candidates.map((candidate) => {
    const similarity = cosineSimilarity(queryEmbedding, candidate.data.embedding);
    return {
      id: candidate.id,
      ...candidate.data,
      similarity,
    };
  });

  return scored
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, topK)
    .filter((item) => Number.isFinite(item.similarity));
}
