import { createClient } from 'redis';
import { env } from '../config/env.js';

type JsonValue = Record<string, unknown>;

type ValkeyClient = ReturnType<typeof createClient>;

let valkeyClient: ValkeyClient | null = null;
let connectPromise: Promise<ValkeyClient | null> | null = null;
let blockedUntil = 0;
let lastBlockReason = '';

function isValkeyEnabled(): boolean {
  return Boolean(env.VALKEY_ENABLED && env.REDIS_URL);
}

function blockTemporarily(reason: string) {
  blockedUntil = Date.now() + env.VALKEY_COOLDOWN_MS;
  lastBlockReason = reason;
  // eslint-disable-next-line no-console
  console.warn(`[valkey] disabled for ${env.VALKEY_COOLDOWN_MS}ms: ${reason}`);
}

function isBlockedNow(): boolean {
  if (Date.now() < blockedUntil) return true;
  if (blockedUntil !== 0) {
    blockedUntil = 0;
    // eslint-disable-next-line no-console
    console.log(`[valkey] re-enabling cache access after cooldown (${lastBlockReason || 'previous failures'})`);
  }
  return false;
}

function getOrCreateClient(): ValkeyClient | null {
  if (!isValkeyEnabled()) return null;
  if (valkeyClient) return valkeyClient;

  const redisUrl = env.REDIS_URL;
  if (!redisUrl) return null;

  const tlsEnabled = redisUrl.startsWith('rediss://');
  const reconnectStrategy = (): false => false;

  const client = createClient({
    url: redisUrl,
    socket: tlsEnabled
      ? { tls: true, reconnectStrategy, connectTimeout: 1500 }
      : { reconnectStrategy, connectTimeout: 1500 },
  });

  client.on('error', (_error) => {
    // We intentionally avoid verbose stack spam here. Connection failures are handled in connect().
  });

  client.on('connect', () => {
    // eslint-disable-next-line no-console
    console.log('[valkey] connecting...');
  });

  client.on('ready', () => {
    // eslint-disable-next-line no-console
    console.log('[valkey] connected');
  });

  valkeyClient = client;
  return client;
}

async function getValkeyClient(): Promise<ValkeyClient | null> {
  if (!isValkeyEnabled()) return null;
  if (isBlockedNow()) return null;

  const client = getOrCreateClient();
  if (!client) return null;
  if (client.isReady) return client;

  if (!connectPromise) {
    connectPromise = client
      .connect()
      .then(() => client)
      .catch((error) => {
        const reason = error instanceof Error ? error.message : String(error);
        blockTemporarily(reason);
        return null;
      })
      .finally(() => {
        connectPromise = null;
      });
  }

  return connectPromise;
}

export async function cacheGetJson<T extends JsonValue>(key: string): Promise<T | null> {
  const client = await getValkeyClient();
  if (!client) return null;

  try {
    const payload = await client.get(key);
    if (!payload) return null;
    return JSON.parse(payload) as T;
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    blockTemporarily(`get failed (${reason})`);
    return null;
  }
}

export async function cacheSetJson(key: string, value: JsonValue, ttlSeconds: number): Promise<void> {
  const client = await getValkeyClient();
  if (!client) return;

  try {
    await client.setEx(key, Math.max(1, ttlSeconds), JSON.stringify(value));
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    blockTemporarily(`set failed (${reason})`);
  }
}

export async function cacheDelete(key: string): Promise<void> {
  const client = await getValkeyClient();
  if (!client) return;

  try {
    await client.del(key);
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    blockTemporarily(`delete failed (${reason})`);
  }
}

export async function disconnectValkey(): Promise<void> {
  if (!valkeyClient) return;

  try {
    await valkeyClient.quit();
  } catch {
    await valkeyClient.disconnect();
  } finally {
    valkeyClient = null;
    connectPromise = null;
  }
}
