import dotenv from 'dotenv';
import { z } from 'zod';

dotenv.config();

const EnvSchema = z.object({
  PORT: z.coerce.number().default(8787),
  NODE_ENV: z.string().default('development'),
  ALLOWED_ORIGINS: z.string().default('http://localhost:8081,http://localhost:19006'),
  DATABASE_URL: z.string().optional(),
  DATABASE_SSL_CA_PATH: z.string().optional(),
  DATABASE_SSL_REJECT_UNAUTHORIZED: z.string().optional(),
  FIREBASE_PROJECT_ID: z.string().optional(),
  FIREBASE_STORAGE_BUCKET: z.string().optional(),
  FIREBASE_SERVICE_ACCOUNT_JSON: z.string().optional(),
  FIREBASE_SERVICE_ACCOUNT_PATH: z.string().optional(),
  GOOGLE_APPLICATION_CREDENTIALS: z.string().optional(),
  FIREBASE_JWKS_URL: z
    .string()
    .default('https://www.googleapis.com/service_accounts/v1/jwk/securetoken@system.gserviceaccount.com'),
  JWT_CLOCK_SKEW_SECONDS: z.coerce.number().default(60),
  PGVECTOR_ENABLED: z.string().default('false'),
  BE1_BASE_URL: z.string().default('http://localhost:9393/api'),
  BE1_INTERNAL_API_KEY: z.string().optional(),
  BE1_REQUEST_TIMEOUT_MS: z.coerce.number().default(15000),
  BE1_CIRCUIT_FAIL_THRESHOLD: z.coerce.number().default(5),
  BE1_CIRCUIT_COOLDOWN_MS: z.coerce.number().default(30000),
  PG_QUERY_TIMEOUT_MS: z.coerce.number().default(30000),
  PG_CONNECTION_TIMEOUT_MS: z.coerce.number().default(5000),
  RATE_LIMIT_WINDOW_MS: z.coerce.number().default(60000),
  RATE_LIMIT_EXPENSIVE_MAX: z.coerce.number().default(20),
  GOOGLE_CLOUD_PROJECT: z.string().optional(),
  GOOGLE_CLOUD_LOCATION: z.string().default('us-central1'),
  GOOGLE_GENAI_MODEL: z.string().default('gemini-2.0-flash'),
  GEMINI_API_KEY: z.string().optional(),
  LLM_PROVIDER: z.enum(['gemini-adk', 'deepseek-bedrock']).default('deepseek-bedrock'),
  AWS_REGION: z.string().default('us-east-1'),
  REDIS_URL: z.string().optional(),
  VALKEY_ENABLED: z
    .string()
    .optional()
    .transform((value) => (value ?? 'true').toLowerCase() === 'true'),
  VALKEY_COOLDOWN_MS: z.coerce.number().default(60_000),
  BEDROCK_DEEPSEEK_PLANNER_MODEL_ID: z.string().optional(),
  BEDROCK_DEEPSEEK_COMPOSER_MODEL_ID: z.string().optional(),
  EMBEDDING_DIM: z.coerce.number().default(192),
  MAX_AUDIO_BYTES: z.coerce.number().default(5 * 1024 * 1024),
  REVENUECAT_WEBHOOK_SECRET: z.string().optional(),
  REVENUECAT_PRO_ENTITLEMENT_ID: z.string().default('Cozmic Astrology Pro'),
});

export const env = EnvSchema.parse(process.env);

export const allowedOrigins = env.ALLOWED_ORIGINS.split(',')
  .map((item) => item.trim())
  .filter(Boolean);
