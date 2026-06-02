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

  PG_IDLE_TIMEOUT_MS: z.coerce.number().default(30000),
  PG_KEEPALIVE_INITIAL_DELAY_MS: z.coerce.number().default(60000),

  BE1_CONFIG: z
    .string()
    .optional()
    .transform((value) => {
      const defaultValue = {
        base_url: 'http://localhost:9393/api',
        internal_api_key: undefined as string | undefined,
        request_timeout_ms: 15000,
        circuit_fail_threshold: 5,
        circuit_cooldown_ms: 30000,
      };
      if (!value) return defaultValue;
      try {
        return { ...defaultValue, ...JSON.parse(value) };
      } catch {
        return defaultValue;
      }
    }),

  PG_CONFIG: z
    .string()
    .optional()
    .transform((value) => {
      const defaultValue = {
        query_timeout_ms: 120000,
        connection_timeout_ms: 5000,
      };
      if (!value) return defaultValue;
      try {
        return JSON.parse(value);
      } catch {
        return defaultValue;
      }
    }),

  RATE_LIMIT_CONFIG: z
    .string()
    .optional()
    .transform((value) => {
      const defaultValue = {
        window_ms: 60000,
        expensive_max: 20,
      };
      if (!value) return defaultValue;
      try {
        return JSON.parse(value);
      } catch {
        return defaultValue;
      }
    }),

  QUOTA_CONFIG: z
    .string()
    .optional()
    .transform((value) => {
      const defaultValue = {
        enabled: true,
        pro: { mini_requests: 50, pro_requests: 50, kundli_generations: 10 },
        nonpro: { mini_requests: 5, pro_requests: 0, kundli_generations: 3 },
      };
      if (!value) return defaultValue;
      try {
        return JSON.parse(value);
      } catch {
        return defaultValue;
      }
    }),

  LLM_DECISION_CONFIG: z
    .string()
    .optional()
    .transform((value) => {
      const defaultValue = {
        enabled: true,
        mode: 'hybrid',
        route: true,
        intent: true,
        mini_scope: true,
        fast_answer: true,
        temporal: true,
        scope_selector: true,
        plan: true,
        tool_selection: true,
        refinement_router: true,
        coverage: true,
        response_policy: true,
        shadow_mode: false,
      };
      if (!value) return defaultValue;
      try {
        return JSON.parse(value);
      } catch {
        return defaultValue;
      }
    }),
  LLM_PROVIDER: z.enum(['gemini-adk', 'deepseek-bedrock']).default('deepseek-bedrock'),
  AWS_REGION: z.string().default('us-east-1'),
  CACHE_CONFIG: z
    .string()
    .optional()
    .transform((value) => {
      const defaultValue = {
        agent_ttl_seconds: 300,
        timing_ttl_seconds: 21600,
      };
      if (!value) return defaultValue;
      try {
        return JSON.parse(value);
      } catch {
        return defaultValue;
      }
    }),
  BEDROCK_DEEPSEEK_PLANNER_MODEL_ID: z.string().optional(),
  BEDROCK_DEEPSEEK_COMPOSER_MODEL_ID: z.string().optional(),
  EMBEDDING_DIM: z.coerce.number().default(192),
  GOOGLE_GENAI_MODEL: z.string().default('gemini-2.0-flash'),
  PRO_ENTITLEMENT_ID: z.string().default('pro'),
  APPLE_BUNDLE_ID: z.string().optional(),
  APPLE_APP_ID: z.string().optional(),
  CONTACT_SMTP_USER: z.string().optional(),
  CONTACT_SMTP_PASS: z.string().optional(),
  CONTACT_EMAIL_TO: z.string().optional(),

  DODOPAYMENTS_API_KEY: z.string().optional(),
  DODOPAYMENTS_WEBHOOK_SECRET: z.string().optional(),
  DODOPAYMENTS_PRICE_ID: z.string().default('price_pro_monthly'),
});

export const env = EnvSchema.parse(process.env);

export const allowedOrigins = env.ALLOWED_ORIGINS.split(',')
  .map((item) => item.trim())
  .filter(Boolean);