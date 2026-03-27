export interface VectorChunkUpsertRecord {
  data: {
    ownerId: string;
    profileId: string;
    kundaliId: string;
    sourceDocId: string;
    sourceType?: 'be1';
    endpoint?: string;
    chunkIndex: number;
    text: string;
    textPreview?: string;
    embedding: number[];
    embeddingModel?: string;
    embeddingDim?: number;
    tokenEstimate?: number;
    tags?: string[];
    createdAt?: number;
  };
}

export interface VectorSearchInput {
  ownerId: string;
  kundaliId?: string;
  queryEmbedding: number[];
  topK: number;
  candidateWindow?: number;
}

export interface VectorStoreProvider {
  upsertChunks(records: VectorChunkUpsertRecord[]): Promise<void>;
  searchChunks(input: VectorSearchInput): Promise<unknown[]>;
}