import { BedrockRuntimeClient, ConverseCommand } from '@aws-sdk/client-bedrock-runtime';
import type { ToolConfiguration } from '@aws-sdk/client-bedrock-runtime';
import { env } from '../config/env.js';
import { logLLMCall } from '../utils/debugLog.js';

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

// -- Types for tool-calling conversation --

export interface ConversationContentBlock {
  text?: string;
  toolUse?: { toolUseId: string; name: string; input: Record<string, unknown> };
  toolResult?: { toolUseId: string; content: Array<{ json: unknown }> };
}

export interface ConversationMessage {
  role: 'user' | 'assistant';
  content: ConversationContentBlock[];
}

export interface ConversationResponse {
  content: Array<{ text?: string; toolUse?: { toolUseId: string; name: string; input: Record<string, unknown> } }>;
  stopReason: string;
  model: string;
}

// -- Existing simple invoke (for fast_answer, search_astrology, etc.) --

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

  logLLMCall(params.systemPrompt, params.userPrompt, '');

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

  logLLMCall('', '', text);

  return {
    text,
    model: modelId,
  };
}

// -- New tool-calling conversation function --

function extractContentBlocks(output: unknown): ConversationContentBlock[] {
  const candidate = output as { message?: { content?: Array<Record<string, unknown>> } };
  const rawBlocks = candidate.message?.content ?? [];
  return rawBlocks.map((block: Record<string, unknown>) => {
    if (block.text) return { text: block.text as string };
    if (block.toolUse) {
      const tu = block.toolUse as Record<string, unknown>;
      return {
        toolUse: {
          toolUseId: tu.toolUseId as string,
          name: tu.name as string,
          input: tu.input as Record<string, unknown>,
        },
      };
    }
    return {};
  });
}

export async function invokeDeepSeekConversation(params: {
  systemPrompt: string;
  messages: ConversationMessage[];
  tools?: ToolConfiguration;
  maxTokens?: number;
}): Promise<ConversationResponse> {
  const modelId = resolveDeepSeekModelId();
  const client = getBedrockClient();

  const systemLog = `[${params.messages.length} messages, tools: ${params.tools ? 'yes' : 'no'}]`;
  logLLMCall(params.systemPrompt, systemLog, '');

  const response = await client.send(
    new ConverseCommand({
      modelId,
      system: [{ text: params.systemPrompt }],
      messages: params.messages.map((m) => ({
        role: m.role,
        content: m.content.map((block) => {
          if (block.text) return { text: block.text };
          if (block.toolUse) return { toolUse: block.toolUse };
          if (block.toolResult) return {
            toolResult: { toolUseId: block.toolResult.toolUseId, content: block.toolResult.content },
          };
          return { text: '' };
        }) as any,
      })),
      inferenceConfig: {
        temperature: 0,
        maxTokens: params.maxTokens ?? 4096,
      },
      ...(params.tools ? { toolConfig: params.tools } : {}),
    })
  );

  const content = extractContentBlocks(response.output);
  const stopReason = response.stopReason ?? 'unknown';

  logLLMCall('', `stopReason: ${stopReason}`, JSON.stringify(content.slice(0, 3)));

  return { content, stopReason, model: modelId };
}
