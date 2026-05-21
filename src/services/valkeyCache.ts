type JsonValue = Record<string, unknown>;

export async function cacheGetJson<T extends JsonValue>(_key: string): Promise<T | null> {
  return null;
}

export async function cacheSetJson(_key: string, _value: JsonValue, _ttlSeconds: number): Promise<void> {
}

export async function cacheDelete(_key: string): Promise<void> {
}

export async function disconnectValkey(): Promise<void> {
}
