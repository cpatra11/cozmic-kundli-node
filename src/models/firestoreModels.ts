export const COLLECTIONS = {
  authUsers: 'auth_users',
  userSubscriptions: 'user_subscriptions',
  ragProfiles: 'rag_profiles',
  ragApiSources: 'rag_api_sources',
  ragChunks: 'rag_chunks',
  chartJobs: 'chart_jobs',
} as const;

export interface AuthUserDocument {
  ownerId: string;
  email?: string;
  provider: 'firebase';
  createdAt: number;
  updatedAt: number;
  lastSeenAt: number;
}

export interface UserSubscriptionDocument {
  ownerId: string;
  source: 'revenuecat';
  entitlementId: string;
  isPro: boolean;
  store?: string;
  productId?: string;
  eventType?: string;
  expiresAtMs?: number;
  updatedAt: number;
  lastEventAt: number;
  lastEventId?: string;
}

export interface RagProfileDocument {
  ownerId: string;
  profileId: string;
  displayName?: string;
  place?: string;
  kundliSignature: string;
  chartVersion: string;
  kundliInput: {
    latitude: number;
    longitude: number;
    year: number;
    month: number;
    day: number;
    hour: number;
    min: number;
    sec: number;
    time_zone: string;
  };
  latestSourceDocId: string;
  sourceCount: number;
  updatedAt: number;
  createdAt: number;
}

export interface RagApiSourceDocument {
  ownerId: string;
  profileId: string;
  displayName?: string;
  place?: string;
  sourceType: 'be1';
  endpoint: string;
  requestKey: string;
  payloadHash: string;
  rawPayload: unknown;
  chartSnapshot?: unknown;
  preview: string;
  tags: string[];
  createdAt: number;
}

export interface RagChunkDocument {
  ownerId: string;
  profileId: string;
  sourceDocId: string;
  sourceType: 'be1';
  endpoint: string;
  chunkIndex: number;
  text: string;
  textPreview: string;
  embedding: number[];
  embeddingModel: string;
  embeddingDim: number;
  tokenEstimate: number;
  tags: string[];
  createdAt: number;
}

export interface RagChunkResult extends RagChunkDocument {
  id: string;
  similarity: number;
}

export interface ChartJobDocument {
  ownerId: string;
  profileId: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  request: Record<string, unknown>;
  result?: unknown;
  error?: string;
  createdAt: number;
  updatedAt: number;
}
