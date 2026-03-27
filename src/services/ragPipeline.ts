import { env } from '../config/env.js';
import { COLLECTIONS, type RagApiSourceDocument, type RagChunkDocument, type RagChunkResult, type RagProfileDocument } from '../models/firestoreModels.js';
import { fetchKundliSnapshot, type KundliSnapshotInput } from './be1Client.js';
import { cosineSimilarity, embedTextDeterministic } from './embeddings.js';
import { getPostgresStore } from './postgresStore.js';
import { stableHash, toDocId } from './hash.js';
import { buildChartSnapshot } from './chartSnapshot.js';

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
  const store = getPostgresStore();
  const now = Date.now();
  const endpoint = input.endpoint ?? 'calculate';

  const payloadRaw = JSON.stringify(input.payload);
  const payloadHash = stableHash(payloadRaw);
  const requestKey = stableHash(
    JSON.stringify({ endpoint, ownerId: input.ownerId, profileId: input.profileId, kundli: toKundliInputDocument(input.kundli) })
  );

  const sourceDocId = toDocId('src', `${input.ownerId}:${input.profileId}:${requestKey}:${payloadHash}`);
  const sourceDoc: RagApiSourceDocument = {
    ownerId: input.ownerId,
    profileId: input.profileId,
    displayName: input.displayName,
    place: input.place,
    sourceType: 'be1',
    endpoint,
    requestKey,
    payloadHash,
    rawPayload: input.payload,
    chartSnapshot: buildChartSnapshot(input.payload),
    preview: payloadRaw.slice(0, 1800),
    tags: input.tags ?? ['kundli', 'be1', endpoint],
    createdAt: now,
  };

  await store.setDocument(`${COLLECTIONS.ragApiSources}/${sourceDocId}`, sourceDoc, false);

  const lines = flattenPayload(input.payload);
  const chunkTexts = chunkLines(lines, 900).slice(0, 200);

  const embeddingDim = env.EMBEDDING_DIM;
  const writes = chunkTexts.map(async (chunkText, index) => {
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
    await store.setDocument(`${COLLECTIONS.ragChunks}/${chunkId}`, chunkDoc, false);
  });

  await Promise.all(writes);

  const profileDocPath = `${COLLECTIONS.ragProfiles}/${input.ownerId}__${input.profileId}`;
  const existingProfile = await store.getDocument<RagProfileDocument>(profileDocPath);

  const profileDoc: RagProfileDocument = {
    ownerId: input.ownerId,
    profileId: input.profileId,
    displayName: input.displayName,
    place: input.place,
    kundliSignature: makeKundliSignature(input.kundli),
    chartVersion: payloadHash,
    kundliInput: toKundliInputDocument(input.kundli),
    latestSourceDocId: sourceDocId,
    sourceCount: (existingProfile?.data.sourceCount ?? 0) + 1,
    updatedAt: now,
    createdAt: existingProfile?.data.createdAt ?? now,
  };

  await store.setDocument(profileDocPath, profileDoc, true);

  return {
    profileId: input.profileId,
    sourceDocId,
    chunkCount: chunkTexts.length,
    endpoint,
  };
}

export async function queryRagChunks(input: QueryInput): Promise<RagChunkResult[]> {
  const store = getPostgresStore();
  const topK = Math.min(Math.max(input.topK ?? 8, 1), 20);
  const candidateWindow = Math.max(topK * 8, 40);

  const filters = [{ field: 'ownerId', op: 'EQUAL' as const, value: input.ownerId }];
  if (input.profileId) {
    filters.push({ field: 'profileId', op: 'EQUAL' as const, value: input.profileId });
  }

  const candidates = await store.runQuery<RagChunkDocument>(COLLECTIONS.ragChunks, filters, {
    orderBy: [{ field: 'createdAt', direction: 'DESCENDING' }],
    limit: candidateWindow,
  });

  if (candidates.length === 0) {
    return [];
  }

  const queryEmbedding = embedTextDeterministic(input.message, env.EMBEDDING_DIM).vector;

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
