import { GoogleAuth } from 'google-auth-library';
import { decodeJwt, importPKCS8, SignJWT } from 'jose';
import { z } from 'zod';
import { env } from '../config/env.js';

type BillingSource = 'revenuecat' | 'expo_iap' | 'app_store' | 'play_store';

export interface BillingDirectVerificationInput {
  source: BillingSource;
  purchaseToken?: string;
  transactionId?: string;
  productId?: string;
  store?: string;
}

export type BillingDirectVerificationResult =
  | {
      status: 'skipped';
      reason: string;
    }
  | {
      status: 'invalid';
      provider: 'app_store' | 'play_store';
      reason: string;
    }
  | {
      status: 'verified';
      provider: 'app_store' | 'play_store';
      isPro: boolean;
      expiresAtMs?: number;
      productId?: string;
      store: 'app_store' | 'play_store';
      eventType: 'direct_store_validation';
      lastEventId?: string;
    };

const AppleTransactionLookupSchema = z
  .object({
    signedTransactionInfo: z.string().min(1),
  })
  .passthrough();

const GoogleSubscriptionLineItemSchema = z
  .object({
    productId: z.string().optional(),
    expiryTime: z.string().optional(),
  })
  .passthrough();

const GoogleSubscriptionV2Schema = z
  .object({
    subscriptionState: z.string().optional(),
    latestOrderId: z.string().optional(),
    lineItems: z.array(GoogleSubscriptionLineItemSchema).optional(),
  })
  .passthrough();

const GoogleProductPurchaseSchema = z
  .object({
    purchaseState: z.coerce.number().optional(),
    orderId: z.string().optional(),
  })
  .passthrough();

function normalizeOptional(value?: string): string | undefined {
  const trimmed = value?.trim();
  return trimmed ? trimmed : undefined;
}

function parseMillis(value: unknown): number | undefined {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return Math.round(value);
  }

  if (typeof value === 'string') {
    const trimmed = value.trim();
    if (!trimmed) return undefined;

    const asNumber = Number(trimmed);
    if (!Number.isNaN(asNumber)) {
      return Math.round(asNumber);
    }

    const asDate = Date.parse(trimmed);
    if (!Number.isNaN(asDate)) {
      return asDate;
    }
  }

  return undefined;
}

function parseIsoDateToMillis(value?: string): number | undefined {
  if (!value) return undefined;
  const parsed = Date.parse(value);
  return Number.isNaN(parsed) ? undefined : parsed;
}

function looksLikeJwt(value: string): boolean {
  return value.split('.').length === 3;
}

function extractAppleTransactionId(input: BillingDirectVerificationInput): string | undefined {
  const transactionId = normalizeOptional(input.transactionId);
  if (transactionId) return transactionId;

  const purchaseToken = normalizeOptional(input.purchaseToken);
  if (!purchaseToken) return undefined;

  if (!looksLikeJwt(purchaseToken)) {
    return purchaseToken;
  }

  try {
    const decoded = decodeJwt(purchaseToken) as Record<string, unknown>;
    const decodedTransactionId = decoded.transactionId;
    if (typeof decodedTransactionId === 'string' && decodedTransactionId.trim()) {
      return decodedTransactionId;
    }
  } catch {
    // ignore; invalid token shape handled later via API lookup
  }

  return undefined;
}

function normalizePrivateKey(privateKey: string): string {
  return privateKey.replace(/\\n/g, '\n').trim();
}

function isAppleValidationConfigured(): boolean {
  return Boolean(
    env.APPLE_IAP_BUNDLE_ID && env.APPLE_IAP_ISSUER_ID && env.APPLE_IAP_KEY_ID && env.APPLE_IAP_PRIVATE_KEY
  );
}

function isGoogleValidationConfigured(): boolean {
  return Boolean(env.GOOGLE_PLAY_PACKAGE_NAME);
}

function resolveStoreFromPayload(input: BillingDirectVerificationInput): 'app_store' | 'play_store' | null {
  if (input.source === 'app_store' || input.source === 'play_store') {
    return input.source;
  }

  if (input.source !== 'expo_iap') {
    return null;
  }

  const storeHint = normalizeOptional(input.store)?.toLowerCase();
  if (storeHint?.includes('apple') || storeHint?.includes('app_store') || storeHint?.includes('ios')) {
    return 'app_store';
  }
  if (
    storeHint?.includes('google') ||
    storeHint?.includes('play') ||
    storeHint?.includes('android') ||
    storeHint?.includes('play_store')
  ) {
    return 'play_store';
  }

  return null;
}

async function readJsonResponse(response: Response): Promise<unknown> {
  const text = await response.text();
  if (!text.trim()) return {};

  try {
    return JSON.parse(text);
  } catch {
    return { raw: text };
  }
}

async function buildAppleAppStoreToken(): Promise<string> {
  if (!env.APPLE_IAP_BUNDLE_ID || !env.APPLE_IAP_ISSUER_ID || !env.APPLE_IAP_KEY_ID || !env.APPLE_IAP_PRIVATE_KEY) {
    throw new Error('Apple App Store validation credentials are incomplete');
  }

  const now = Math.floor(Date.now() / 1000);
  const privateKey = await importPKCS8(normalizePrivateKey(env.APPLE_IAP_PRIVATE_KEY), 'ES256');

  return new SignJWT({ bid: env.APPLE_IAP_BUNDLE_ID })
    .setProtectedHeader({ alg: 'ES256', kid: env.APPLE_IAP_KEY_ID, typ: 'JWT' })
    .setIssuer(env.APPLE_IAP_ISSUER_ID)
    .setAudience('appstoreconnect-v1')
    .setIssuedAt(now)
    .setExpirationTime(now + 300)
    .sign(privateKey);
}

async function verifyWithApple(input: BillingDirectVerificationInput): Promise<BillingDirectVerificationResult> {
  if (!isAppleValidationConfigured()) {
    return { status: 'skipped', reason: 'Apple validation is not configured' };
  }

  const transactionId = extractAppleTransactionId(input);
  if (!transactionId) {
    return { status: 'skipped', reason: 'Missing Apple transaction identifier for direct validation' };
  }

  const authToken = await buildAppleAppStoreToken();
  const environmentOrder =
    env.APPLE_IAP_ENVIRONMENT === 'production'
      ? ['https://api.storekit.itunes.apple.com']
      : env.APPLE_IAP_ENVIRONMENT === 'sandbox'
      ? ['https://api.storekit-sandbox.itunes.apple.com']
      : ['https://api.storekit.itunes.apple.com', 'https://api.storekit-sandbox.itunes.apple.com'];

  let lastError: string | undefined;

  for (const baseUrl of environmentOrder) {
    const url = `${baseUrl}/inApps/v1/transactions/${encodeURIComponent(transactionId)}`;
    const response = await fetch(url, {
      method: 'GET',
      headers: {
        Authorization: `Bearer ${authToken}`,
        Accept: 'application/json',
      },
    });

    if (response.status === 404) {
      lastError = `Transaction not found in ${baseUrl.includes('sandbox') ? 'sandbox' : 'production'}`;
      continue;
    }

    if (!response.ok) {
      const errorPayload = await readJsonResponse(response);
      return {
        status: 'invalid',
        provider: 'app_store',
        reason: `Apple validation failed (${response.status}): ${JSON.stringify(errorPayload)}`,
      };
    }

    const json = await readJsonResponse(response);
    const parsed = AppleTransactionLookupSchema.safeParse(json);
    if (!parsed.success) {
      return {
        status: 'invalid',
        provider: 'app_store',
        reason: 'Apple validation response is missing signedTransactionInfo',
      };
    }

    let decoded: Record<string, unknown>;
    try {
      decoded = decodeJwt(parsed.data.signedTransactionInfo) as Record<string, unknown>;
    } catch {
      return {
        status: 'invalid',
        provider: 'app_store',
        reason: 'Apple signedTransactionInfo could not be decoded',
      };
    }

    const bundleId = normalizeOptional(typeof decoded.bundleId === 'string' ? decoded.bundleId : undefined);
    if (bundleId && env.APPLE_IAP_BUNDLE_ID && bundleId !== env.APPLE_IAP_BUNDLE_ID) {
      return {
        status: 'invalid',
        provider: 'app_store',
        reason: `Apple transaction bundleId mismatch (${bundleId})`,
      };
    }

    const productId = normalizeOptional(typeof decoded.productId === 'string' ? decoded.productId : undefined);
    if (input.productId && productId && input.productId !== productId) {
      return {
        status: 'invalid',
        provider: 'app_store',
        reason: `Apple productId mismatch (expected ${input.productId}, got ${productId})`,
      };
    }

    const expiresAtMs = parseMillis(decoded.expiresDate);
    const revocationDateMs = parseMillis(decoded.revocationDate);
    const now = Date.now();
    const isPro = !revocationDateMs && (!expiresAtMs || expiresAtMs > now);

    const validatedTransactionId =
      normalizeOptional(typeof decoded.transactionId === 'string' ? decoded.transactionId : undefined) ?? transactionId;

    return {
      status: 'verified',
      provider: 'app_store',
      isPro,
      expiresAtMs,
      productId: productId ?? input.productId,
      store: 'app_store',
      eventType: 'direct_store_validation',
      lastEventId: validatedTransactionId,
    };
  }

  return {
    status: 'invalid',
    provider: 'app_store',
    reason: lastError ?? 'Apple transaction not found',
  };
}

async function getGoogleAccessToken(): Promise<string> {
  const jsonCredentials = normalizeOptional(env.GOOGLE_PLAY_SERVICE_ACCOUNT_JSON ?? env.FIREBASE_SERVICE_ACCOUNT_JSON);
  const credentials = jsonCredentials ? JSON.parse(jsonCredentials) : undefined;

  const auth = new GoogleAuth({
    credentials,
    scopes: ['https://www.googleapis.com/auth/androidpublisher'],
  });

  const token = await auth.getAccessToken();
  if (!token) {
    throw new Error('Unable to obtain Google access token for Play validation');
  }

  return token;
}

function normalizeGoogleSubscriptionState(state?: string): string | undefined {
  return state?.trim().toUpperCase();
}

function isGoogleSubscriptionEntitled(state: string | undefined, expiresAtMs?: number): boolean {
  if (!expiresAtMs || expiresAtMs <= Date.now()) {
    return false;
  }

  if (!state) return true;

  const disallowedStates = new Set([
    'SUBSCRIPTION_STATE_EXPIRED',
    'SUBSCRIPTION_STATE_ON_HOLD',
    'SUBSCRIPTION_STATE_PAUSED',
    'SUBSCRIPTION_STATE_PENDING',
    'SUBSCRIPTION_STATE_UNSPECIFIED',
  ]);

  return !disallowedStates.has(state);
}

async function verifyWithGoogle(input: BillingDirectVerificationInput): Promise<BillingDirectVerificationResult> {
  if (!isGoogleValidationConfigured()) {
    return { status: 'skipped', reason: 'Google Play validation is not configured' };
  }

  const packageName = normalizeOptional(env.GOOGLE_PLAY_PACKAGE_NAME);
  const purchaseToken = normalizeOptional(input.purchaseToken);

  if (!packageName || !purchaseToken) {
    return {
      status: 'skipped',
      reason: 'Missing Google Play package name or purchase token for direct validation',
    };
  }

  const accessToken = await getGoogleAccessToken();

  const subscriptionUrl =
    `https://androidpublisher.googleapis.com/androidpublisher/v3/applications/${encodeURIComponent(packageName)}` +
    `/purchases/subscriptionsv2/tokens/${encodeURIComponent(purchaseToken)}`;

  const subscriptionResponse = await fetch(subscriptionUrl, {
    method: 'GET',
    headers: {
      Authorization: `Bearer ${accessToken}`,
      Accept: 'application/json',
    },
  });

  if (subscriptionResponse.ok) {
    const json = await readJsonResponse(subscriptionResponse);
    const parsed = GoogleSubscriptionV2Schema.safeParse(json);
    if (!parsed.success) {
      return {
        status: 'invalid',
        provider: 'play_store',
        reason: 'Google subscription response is invalid',
      };
    }

    const lineItems = parsed.data.lineItems ?? [];
    const productId = lineItems[0]?.productId ?? input.productId;

    if (input.productId && productId && input.productId !== productId) {
      return {
        status: 'invalid',
        provider: 'play_store',
        reason: `Google productId mismatch (expected ${input.productId}, got ${productId})`,
      };
    }

    const expiresAtMs = lineItems
      .map((item) => parseIsoDateToMillis(item.expiryTime))
      .filter((value): value is number => typeof value === 'number')
      .sort((a, b) => b - a)[0];

    const subscriptionState = normalizeGoogleSubscriptionState(parsed.data.subscriptionState);
    const isPro = isGoogleSubscriptionEntitled(subscriptionState, expiresAtMs);

    return {
      status: 'verified',
      provider: 'play_store',
      isPro,
      expiresAtMs,
      productId,
      store: 'play_store',
      eventType: 'direct_store_validation',
      lastEventId: parsed.data.latestOrderId ?? purchaseToken,
    };
  }

  if (subscriptionResponse.status !== 404) {
    const errorPayload = await readJsonResponse(subscriptionResponse);
    return {
      status: 'invalid',
      provider: 'play_store',
      reason: `Google subscription validation failed (${subscriptionResponse.status}): ${JSON.stringify(errorPayload)}`,
    };
  }

  // Fallback: one-time product purchase endpoint.
  const productId = normalizeOptional(input.productId);
  if (!productId) {
    return {
      status: 'invalid',
      provider: 'play_store',
      reason: 'Google subscription token not found and productId is missing for one-time fallback validation',
    };
  }

  const productUrl =
    `https://androidpublisher.googleapis.com/androidpublisher/v3/applications/${encodeURIComponent(packageName)}` +
    `/purchases/products/${encodeURIComponent(productId)}/tokens/${encodeURIComponent(purchaseToken)}`;

  const productResponse = await fetch(productUrl, {
    method: 'GET',
    headers: {
      Authorization: `Bearer ${accessToken}`,
      Accept: 'application/json',
    },
  });

  if (!productResponse.ok) {
    const errorPayload = await readJsonResponse(productResponse);
    return {
      status: 'invalid',
      provider: 'play_store',
      reason: `Google product validation failed (${productResponse.status}): ${JSON.stringify(errorPayload)}`,
    };
  }

  const json = await readJsonResponse(productResponse);
  const parsed = GoogleProductPurchaseSchema.safeParse(json);
  if (!parsed.success) {
    return {
      status: 'invalid',
      provider: 'play_store',
      reason: 'Google product validation response is invalid',
    };
  }

  const isPro = parsed.data.purchaseState === 0;

  return {
    status: 'verified',
    provider: 'play_store',
    isPro,
    productId,
    store: 'play_store',
    eventType: 'direct_store_validation',
    lastEventId: parsed.data.orderId ?? purchaseToken,
  };
}

export async function verifyBillingSyncPayload(
  input: BillingDirectVerificationInput
): Promise<BillingDirectVerificationResult> {
  if (!env.BILLING_DIRECT_VALIDATION_ENABLED) {
    return { status: 'skipped', reason: 'Direct billing validation is disabled' };
  }

  const targetStore = resolveStoreFromPayload(input);
  if (!targetStore) {
    return { status: 'skipped', reason: 'No direct store validation provider resolved for this payload' };
  }

  if (targetStore === 'app_store') {
    return verifyWithApple(input);
  }

  return verifyWithGoogle(input);
}