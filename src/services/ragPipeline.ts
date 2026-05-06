import { env } from '../config/env.js';
import type { RagProfileDocument, RagChunkDocument, RagChunkResult } from '../models/firestoreModels.js';
import { getRagProfilesRepository } from '../repositories/ragProfilesRepository.js';
import { fetchKundliSnapshot, type KundliSnapshotInput } from './be1Client.js';
import { getPostgresPool } from './postgresClient.js';
import { stableHash, toDocId } from './hash.js';
import { buildChartSnapshot, extractChartSchemaInfo, type ChartSchemaInfo } from './chartSnapshot.js';

interface IngestInput {
  ownerId: string;
  profileId: string;
  displayName?: string;
  place?: string;
  kundli: KundliSnapshotInput;
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

const ragProfiles = getRagProfilesRepository();

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
  const payloadHash = stableHash(JSON.stringify(payload));

  const existingProfile = await ragProfiles.getByOwnerAndProfileId(input.ownerId, input.profileId);
  const profileDoc: RagProfileDocument = {
    ownerId: input.ownerId,
    profileId: input.profileId,
    displayName: input.displayName ?? undefined,
    place: input.place ?? undefined,
    kundliSignature: makeKundliSignature(input.kundli),
    chartVersion: payloadHash,
    kundliInput: toKundliInputDocument(input.kundli),
    sourceCount: (existingProfile?.sourceCount ?? 0) + 1,
    updatedAt: Date.now(),
    createdAt: existingProfile?.createdAt ?? Date.now(),
  };

  await ragProfiles.upsert(profileDoc);

  return {
    profileId: input.profileId,
    sourceDocId: input.profileId,
    chunkCount: 0,
    endpoint: 'calculate',
  };
}

export async function queryRagChunks(_input: QueryInput): Promise<RagChunkResult[]> {
  // Vector search is disabled - agent now uses direct PHP API calls
  return [];
}
