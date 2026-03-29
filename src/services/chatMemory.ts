import { env } from '../config/env.js';
import { cosineSimilarity, embedTextDeterministic } from './embeddings.js';
import { getPostgresStore } from './postgresStore.js';

interface ChatMessageMemoryDoc {
  ownerId: string;
  sessionId: string;
  role: 'user' | 'assistant';
  message: string;
  createdAt: number;
  requestId?: string;
  embedding?: number[];
  embeddingModel?: string;
  embeddingDim?: number;
}

export interface RelevantChatMemory {
  role: 'user' | 'assistant';
  text: string;
  similarity: number;
  createdAt: number;
}

export function buildChatMessageEmbedding(text: string): { embedding: number[]; embeddingModel: string; embeddingDim: number } {
  const result = embedTextDeterministic(text, env.EMBEDDING_DIM);
  return {
    embedding: result.vector,
    embeddingModel: result.model,
    embeddingDim: result.dimension,
  };
}

export async function queryRelevantSessionMemories(input: {
  ownerId: string;
  sessionId: string;
  message: string;
  excludeRequestId?: string;
  topK?: number;
}): Promise<RelevantChatMemory[]> {
  const store = getPostgresStore();
  const topK = Math.min(Math.max(input.topK ?? 6, 1), 12);
  const candidateWindow = Math.max(40, topK * 10);

  const candidates = await store.runQuery<ChatMessageMemoryDoc>(
    'chat_messages',
    [
      { field: 'ownerId', op: 'EQUAL', value: input.ownerId },
      { field: 'sessionId', op: 'EQUAL', value: input.sessionId },
    ],
    {
      orderBy: [{ field: 'createdAt', direction: 'DESCENDING' }],
      limit: candidateWindow,
    }
  );

  if (candidates.length === 0) return [];

  const queryVector = embedTextDeterministic(input.message, env.EMBEDDING_DIM).vector;

  const scored = candidates
    .map((doc) => doc.data)
    .filter((doc) => doc.message && doc.message.trim().length > 0)
    .filter((doc) => (input.excludeRequestId ? doc.requestId !== input.excludeRequestId : true))
    .map((doc) => {
      const vector =
        Array.isArray(doc.embedding) && doc.embedding.length === env.EMBEDDING_DIM
          ? doc.embedding
          : embedTextDeterministic(doc.message, env.EMBEDDING_DIM).vector;

      const similarity = cosineSimilarity(queryVector, vector);
      return {
        role: doc.role,
        text: doc.message,
        similarity,
        createdAt: Number(doc.createdAt) || 0,
      } as RelevantChatMemory;
    })
    .filter((item) => Number.isFinite(item.similarity) && item.similarity > 0.1)
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, topK)
    .sort((a, b) => a.createdAt - b.createdAt);

  return scored;
}
