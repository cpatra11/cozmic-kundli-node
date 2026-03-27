import { BedrockRuntimeClient, ConverseCommand } from '@aws-sdk/client-bedrock-runtime';
import { env } from '../config/env.js';

let singletonClient: BedrockRuntimeClient | null = null;

function getBedrockClient(): BedrockRuntimeClient {
  if (!singletonClient) {
    singletonClient = new BedrockRuntimeClient({ region: env.AWS_REGION });
  }
  return singletonClient;
}

function resolveDeepSeekModelId(): string {
  return (
    env.BEDROCK_DEEPSEEK_COMPOSER_MODEL_ID?.trim() ||
    env.BEDROCK_DEEPSEEK_PLANNER_MODEL_ID?.trim() ||
    'deepseek.v3.2'
  );
}

function extractTextFromConverseOutput(output: unknown): string {
  const candidate = output as { message?: { content?: Array<{ text?: string }> } };
  const parts = candidate.message?.content ?? [];
  return parts
    .map((part) => part?.text ?? '')
    .join('')
    .trim();
}

export async function invokeDeepSeekBedrock(params: {
  systemPrompt: string;
  userPrompt: string;
  maxTokens?: number;
}): Promise<{ text: string; model: string }> {
  const modelId = resolveDeepSeekModelId();
  const client = getBedrockClient();

  const response = await client.send(
    new ConverseCommand({
      modelId,
      system: [{ text: params.systemPrompt }],
      messages: [
        {
          role: 'user',
          content: [{ text: params.userPrompt }],
        },
      ],
      inferenceConfig: {
        temperature: 0,
        maxTokens: params.maxTokens ?? 1400,
      },
    })
  );

  const text = extractTextFromConverseOutput(response.output);
  if (!text) {
    throw new Error('DeepSeek Bedrock returned an empty response');
  }

  return {
    text,
    model: modelId,
  };
}
