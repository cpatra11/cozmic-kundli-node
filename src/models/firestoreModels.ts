export const COLLECTIONS = {
  authUsers: 'auth_users',
  userSubscriptions: 'user_subscriptions',
  ragProfiles: 'rag_profiles',
  ragApiSources: 'rag_api_sources',
  ragChunks: 'rag_chunks',
  chartJobs: 'chart_jobs',
} as const;

export type ChartSchemaVersion = 'mahadasha-first' | 'legacy-deep-dasha';

export interface AuthUserDocument {
  ownerId: string;
  email?: string;
  phoneNumber?: string;
  provider: 'firebase';
  createdAt: number;
  updatedAt: number;
  lastSeenAt: number;
}

export interface UserSubscriptionDocument {
  ownerId: string;
  source: 'iapkit' | 'app_store' | 'play_store' | 'dodopayments';
  entitlementId: string;
  isPro: boolean;
  store?: string;
  productId?: string;
  eventType?: string;
  purchaseToken?: string;
  transactionId?: string;
  originalTransactionId?: string;
  iapkitState?: string;
  iapkitValid?: boolean;
  iapkitStore?: 'apple' | 'google' | 'unknown';
  expiresAtMs?: number;
  billingAnchorMs?: number;
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
  latestSourceDocId?: string;
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
  chartSchemaVersion: ChartSchemaVersion;
  dashaDepth: number;
  dashaPeriodKey?: string;
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
