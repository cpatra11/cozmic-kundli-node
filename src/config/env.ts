import dotenv from 'dotenv';
import { z } from 'zod';

dotenv.config();

const EnvSchema = z.object({
  PORT: z.coerce.number().default(8787),
  NODE_ENV: z.string().default('development'),
  ALLOWED_ORIGINS: z.string().default('http://localhost:8081,http://localhost:19006'),
  FIREBASE_PROJECT_ID: z.string().optional(),
  FIREBASE_STORAGE_BUCKET: z.string().optional(),
  FIREBASE_SERVICE_ACCOUNT_JSON: z.string().optional(),
  FIREBASE_SERVICE_ACCOUNT_PATH: z.string().optional(),
  BE1_BASE_URL: z.string().default('http://localhost:9393/api'),
  GOOGLE_CLOUD_PROJECT: z.string().optional(),
  GOOGLE_CLOUD_LOCATION: z.string().default('us-central1'),
  GOOGLE_GENAI_MODEL: z.string().default('gemini-2.0-flash'),
  GEMINI_API_KEY: z.string().optional(),
  EMBEDDING_DIM: z.coerce.number().default(192),
  MAX_AUDIO_BYTES: z.coerce.number().default(5 * 1024 * 1024),
});

export const env = EnvSchema.parse(process.env);

export const allowedOrigins = env.ALLOWED_ORIGINS.split(',')
  .map((item) => item.trim())
  .filter(Boolean);
