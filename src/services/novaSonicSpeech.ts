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

const DEFAULT_INFERENCE_CONFIGURATION = {
  maxTokens: 1024,
  topP: 0.9,
  temperature: 0.7,
};

function getBearerToken(): string {
  const token = process.env.AWS_BEARER_TOKEN_BEDROCK;
  if (!token) {
    throw new Error('AWS_BEARER_TOKEN_BEDROCK environment variable is not set');
  }
  return token;
}

interface NovaSonicResponse {
  output: {
    message: {
      content: Array<{
        type: string;
        text?: string;
        source?: {
          bytes: string;
        };
      }>;
    };
  };
  stopReason: string;
  usage: Record<string, unknown>;
}

export async function synthesizeVoiceTurn(input: VoiceTurnInput): Promise<VoiceTurnOutput> {
  const systemPrompt =
    input.systemPrompt?.trim() ||
    'You are a warm, concise assistant. Read the answer naturally and clearly for the user.';

  const url = `https://bedrock-runtime.${env.AWS_REGION}.amazonaws.com/model/${NOVA_SONIC_MODEL_ID}/invoke`;

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${getBearerToken()}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      schemaVersion: '1.0',
      system: [
        {
          type: 'TEXT',
          content: systemPrompt,
        },
      ],
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'TEXT',
              text: input.message,
            },
          ],
        },
      ],
      inferenceConfig: DEFAULT_INFERENCE_CONFIGURATION,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Bedrock Nova Sonic API error ${response.status}: ${errorText}`);
  }

  const data: NovaSonicResponse = await response.json();

  let text = '';
  let audioBase64 = '';

  for (const block of data.output.message.content) {
    if (block.type === 'TEXT' && block.text) {
      text += block.text;
    }
    if (block.type === 'AUDIO' && block.source?.bytes) {
      audioBase64 = block.source.bytes;
    }
  }

  return {
    text: text.trim(),
    audioBase64,
    mimeType: 'audio/mpeg',
    model: NOVA_SONIC_MODEL_ID,
  };
}
