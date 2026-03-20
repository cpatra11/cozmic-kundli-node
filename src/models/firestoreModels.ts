export const COLLECTIONS = {
  ragProfiles: 'rag_profiles',
  ragApiSources: 'rag_api_sources',
  ragChunks: 'rag_chunks',
} as const;

export interface RagProfileDocument {
  ownerId: string;
  profileId: string;
  kundliSignature: string;
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
  sourceType: 'be1';
  endpoint: string;
  requestKey: string;
  payloadHash: string;
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
