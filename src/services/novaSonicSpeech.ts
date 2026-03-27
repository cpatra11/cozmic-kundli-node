import { Buffer } from 'node:buffer';
import { randomUUID } from 'node:crypto';
import {
  BedrockRuntimeClient,
  InvokeModelWithBidirectionalStreamCommand,
  type InvokeModelWithBidirectionalStreamInput,
} from '@aws-sdk/client-bedrock-runtime';
import { NodeHttp2Handler } from '@smithy/node-http-handler';
import { env } from '../config/env.js';

export const NOVA_SONIC_MODEL_ID = 'amazon.nova-2-sonic-v1:0';

type VoiceTurnInput = {
  message: string;
  systemPrompt?: string;
};

type VoiceTurnOutput = {
  text: string;
  audioBase64: string;
  mimeType: string;
  model: string;
};

const DEFAULT_AUDIO_OUTPUT_CONFIGURATION = {
  mediaType: 'audio/mpeg',
};

const DEFAULT_TEXT_CONFIGURATION = {
  mediaType: 'text/plain',
};

const DEFAULT_INFERENCE_CONFIGURATION = {
  maxTokens: 1024,
  topP: 0.9,
  temperature: 0.7,
};

let singletonClient: BedrockRuntimeClient | null = null;

function getBedrockClient(): BedrockRuntimeClient {
  if (!singletonClient) {
    singletonClient = new BedrockRuntimeClient({
      region: env.AWS_REGION,
      requestHandler: new NodeHttp2Handler({
        requestTimeout: 300000,
        sessionTimeout: 300000,
        disableConcurrentStreams: false,
        maxConcurrentStreams: 20,
      }),
    });
  }

  return singletonClient;
}

function encodeEvent(event: Record<string, unknown>): InvokeModelWithBidirectionalStreamInput {
  return {
    chunk: {
      bytes: new TextEncoder().encode(JSON.stringify({ event })),
    },
  };
}

function createVoiceInputBody(input: VoiceTurnInput): AsyncIterable<InvokeModelWithBidirectionalStreamInput> {
  const promptName = randomUUID();
  const systemContentName = randomUUID();
  const userContentName = randomUUID();

  const systemPrompt =
    input.systemPrompt?.trim() ||
    'You are a warm, concise assistant. Read the answer naturally and clearly for the user.';

  const events: Record<string, unknown>[] = [
    {
      sessionStart: {
        inferenceConfiguration: DEFAULT_INFERENCE_CONFIGURATION,
      },
    },
    {
      promptStart: {
        promptName,
        textOutputConfiguration: DEFAULT_TEXT_CONFIGURATION,
        audioOutputConfiguration: DEFAULT_AUDIO_OUTPUT_CONFIGURATION,
      },
    },
    {
      contentStart: {
        promptName,
        contentName: systemContentName,
        type: 'TEXT',
        interactive: false,
        role: 'SYSTEM',
        textInputConfiguration: DEFAULT_TEXT_CONFIGURATION,
      },
    },
    {
      textInput: {
        promptName,
        contentName: systemContentName,
        content: systemPrompt,
      },
    },
    {
      contentEnd: {
        promptName,
        contentName: systemContentName,
      },
    },
    {
      contentStart: {
        promptName,
        contentName: userContentName,
        type: 'TEXT',
        interactive: true,
        role: 'USER',
        textInputConfiguration: DEFAULT_TEXT_CONFIGURATION,
      },
    },
    {
      textInput: {
        promptName,
        contentName: userContentName,
        content: input.message,
      },
    },
    {
      contentEnd: {
        promptName,
        contentName: userContentName,
      },
    },
    {
      promptEnd: {
        promptName,
      },
    },
    {
      sessionEnd: {},
    },
  ];

  return {
    async *[Symbol.asyncIterator]() {
      for (const event of events) {
        yield encodeEvent(event);
      }
    },
  };
}

function extractTextValue(value: unknown): string {
  if (typeof value === 'string') return value;
  if (!value || typeof value !== 'object') return '';

  const candidate = value as Record<string, unknown>;
  const text = candidate.text ?? candidate.content ?? candidate.value;
  if (typeof text === 'string') return text;

  if (typeof text === 'number' || typeof text === 'boolean') {
    return String(text);
  }

  return '';
}

function extractAudioChunk(value: unknown): Buffer | null {
  if (!value) return null;

  if (typeof value === 'string') {
    return value ? Buffer.from(value, 'base64') : null;
  }

  if (value instanceof Uint8Array) {
    return Buffer.from(value);
  }

  if (Array.isArray(value)) {
    return Buffer.from(value);
  }

  if (typeof value === 'object') {
    const candidate = value as Record<string, unknown>;
    const nested = candidate.bytes ?? candidate.content ?? candidate.audio ?? candidate.data;

    if (typeof nested === 'string') {
      return nested ? Buffer.from(nested, 'base64') : null;
    }

    if (nested instanceof Uint8Array) {
      return Buffer.from(nested);
    }

    if (Array.isArray(nested)) {
      return Buffer.from(nested);
    }
  }

  return null;
}

export async function synthesizeVoiceTurn(input: VoiceTurnInput): Promise<VoiceTurnOutput> {
  const client = getBedrockClient();
  const response = await client.send(
    new InvokeModelWithBidirectionalStreamCommand({
      modelId: NOVA_SONIC_MODEL_ID,
      body: createVoiceInputBody(input),
    })
  );

  const textParts: string[] = [];
  const audioParts: Buffer[] = [];

  for await (const event of response.body as AsyncIterable<{ chunk?: { bytes?: Uint8Array } }>) {
    if (!event.chunk?.bytes) continue;

    const rawText = new TextDecoder().decode(event.chunk.bytes);

    try {
      const parsed = JSON.parse(rawText) as { event?: Record<string, unknown> };
      const responseEvent = parsed.event ?? {};

      if (responseEvent.textOutput) {
        const text = extractTextValue(responseEvent.textOutput);
        if (text) textParts.push(text);
      }

      if (responseEvent.audioOutput) {
        const audio = extractAudioChunk(responseEvent.audioOutput);
        if (audio) audioParts.push(audio);
      }
    } catch {
      // Ignore non-JSON chunks; the SDK stream can include internal framing noise.
    }
  }

  return {
    text: textParts.join('').trim(),
    audioBase64: audioParts.length > 0 ? Buffer.concat(audioParts).toString('base64') : '',
    mimeType: 'audio/mpeg',
    model: NOVA_SONIC_MODEL_ID,
  };
}