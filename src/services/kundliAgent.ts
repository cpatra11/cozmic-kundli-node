import { type KundliSnapshotInput } from './be1Client.js';
import { env } from '../config/env.js';
import { ingestKundliForProfile, queryRagChunks } from './ragPipeline.js';
import { stableHash } from './hash.js';

export interface AgentAnswerInput {
  ownerId?: string;
  message: string;
  kundli?: KundliSnapshotInput;
  profileId?: string;
}

async function runWithAdk(params: {
  ownerId: string;
  profileId?: string;
  userMessage: string;
  retrieval: Array<{ id: string; text: string; similarity: number; sourceDocId: string }>;
}): Promise<string> {
  const adk = (await import('@google/adk')) as any;
  const InMemoryRunner = adk.InMemoryRunner;
  const LlmAgent = adk.LlmAgent;

  if (env.GEMINI_API_KEY) {
    process.env.GEMINI_API_KEY = env.GEMINI_API_KEY;
  }

  const profileId = params.profileId ?? 'unknown-profile';
  const retrievalTool = async () => ({
    status: 'success',
    profileId,
    chunks: params.retrieval.map((item) => ({
      id: item.id,
      sourceDocId: item.sourceDocId,
      similarity: item.similarity,
      text: item.text,
    })),
  });

  const agent = new LlmAgent({
    name: 'kundli_rag_advisor',
    description: 'Grounded horoscope assistant that answers only from retrieved chart context.',
    model: env.GOOGLE_GENAI_MODEL,
    instruction: [
      'You are a Vedic astrology assistant.',
      'Always use the retrievalTool context before answering.',
      'If context is insufficient, say what is missing and ask the user to regenerate chart.',
      'Do not invent chart facts.',
      'Answer in concise, practical language.',
    ].join(' '),
    tools: [retrievalTool],
  });

  const runner = new InMemoryRunner({
    appName: 'cozmic-rag-agents',
    agent,
  });

  const session = await runner.sessionService.createSession({
    appName: 'cozmic-rag-agents',
    userId: params.ownerId,
  });

  const newMessage = {
    role: 'user',
    parts: [
      {
        text: `profile_id=${profileId}\nquestion=${params.userMessage}`,
      },
    ],
  };

  const collected: string[] = [];
  for await (const event of runner.runAsync({
    userId: session.userId,
    sessionId: session.id,
    newMessage,
  })) {
    const text = event.content?.parts?.map((part: { text?: string }) => part.text ?? '').join('').trim();
    if (text) {
      collected.push(text);
    }
  }

  const answer = collected[collected.length - 1] ?? '';
  if (!answer) {
    throw new Error('ADK run produced no final text');
  }

  return answer;
}

// Phase-1 adapter: structured placeholder for Google ADK TypeScript integration.
// Replace internals with your chosen google-adk-ts runtime client in Phase-2.
export async function runKundliAgent(input: AgentAnswerInput) {
  const ownerId = input.ownerId ?? 'anonymous';
  let profileId = input.profileId;

  if (!profileId && input.kundli) {
    profileId = `p_${stableHash(JSON.stringify(input.kundli)).slice(0, 10)}`;
  }

  let retrieval = await queryRagChunks({
    ownerId,
    profileId,
    message: input.message,
    topK: 6,
  });

  if (retrieval.length === 0 && profileId && input.kundli) {
    await ingestKundliForProfile({
      ownerId,
      profileId,
      kundli: input.kundli,
    });
    retrieval = await queryRagChunks({
      ownerId,
      profileId,
      message: input.message,
      topK: 6,
    });
  }

  const context = retrieval
    .map((item, index) => `${index + 1}. (${item.similarity.toFixed(3)}) ${item.text}`)
    .join('\n');

  let answer = [
    retrieval.length > 0
      ? 'I retrieved grounded horoscope context from your stored BE1 vectors.'
      : 'No stored grounding chunks yet. Send kundli input once to ingest your profile.',
    `User message: ${input.message}`,
    retrieval.length > 0 ? `Top retrieved context:\n${context}` : '',
  ].join(' ');

  let model = 'rag-retrieval-scaffold-v1';
  if (retrieval.length > 0 && (env.GEMINI_API_KEY || process.env.GEMINI_API_KEY)) {
    try {
      answer = await runWithAdk({
        ownerId,
        profileId,
        userMessage: input.message,
        retrieval: retrieval.map((item) => ({
          id: item.id,
          text: item.text,
          similarity: item.similarity,
          sourceDocId: item.sourceDocId,
        })),
      });
      model = env.GOOGLE_GENAI_MODEL;
    } catch {
      model = 'adk-fallback-rag-scaffold';
    }
  }

  return {
    answer,
    grounding: {
      profileId,
      matches: retrieval.map((item) => ({
        id: item.id,
        similarity: item.similarity,
        sourceDocId: item.sourceDocId,
        textPreview: item.textPreview,
      })),
    },
    model,
  };
}
