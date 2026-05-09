# Caching Implementation Plan

## Current State

- `compiledGraph` (module-level var) — cached LangGraph DAG, works, no changes needed
- `PostgresCache` (node-level, DB-backed) — near-zero hit rate, negligible benefit

## Goal

Add two in-memory caches using `lru-cache` (npm: [`lru-cache`](https://www.npmjs.com/package/lru-cache)):

1. **BE1 API response cache** — saves ~300-400ms per request
2. **LLM answer cache** — saves ~25-40s per request on repeat questions

## Implementation

### Step 1: Install dependency

```bash
npm install lru-cache
```

### Step 2: Replace `PostgresCache` with in-memory `BaseCache` (LangGraph node cache replacement)

File: `src/services/newAgent.ts`

**Delete** or **comment out** the `PostgresCache` import and usage:

```ts
// DELETE these lines:
import { PostgresCache } from './postgresCache.js';
// ...
const cache = pgPool ? new PostgresCache(pgPool) : undefined;
compiledGraph = graph.compile({ ...(cache ? { cache } : {}) });
```

**Replace** with a lightweight in-memory cache that implements LangGraph's `BaseCache`:

```ts
import { LRUCache } from 'lru-cache';
import type { BaseCache, CacheFullKey, CacheNamespace } from '@langchain/langgraph-checkpoint';

class InMemoryNodeCache<V = unknown> implements BaseCache<V> {
  private store = new LRUCache<string, { value: V; encoding: string }>({ max: 500, ttl: 1000 * 60 * 30 });

  async get(keys: CacheFullKey[]): Promise<{ key: CacheFullKey; value: V }[]> {
    const results: { key: CacheFullKey; value: V }[] = [];
    for (const [namespace, key] of keys) {
      const k = JSON.stringify([namespace, key]);
      const entry = this.store.get(k);
      if (entry) {
        const parsed = entry.encoding === 'json' ? JSON.parse(entry.value as string) : entry.value;
        results.push({ key: [namespace, key], value: parsed as V });
      }
    }
    return results;
  }

  async set(pairs: { key: CacheFullKey; value: V; ttl?: number }[]): Promise<void> {
    for (const { key: [namespace, key], value, ttl } of pairs) {
      const k = JSON.stringify([namespace, key]);
      const encoding = typeof value === 'string' ? 'utf-8' : 'json';
      const stored = encoding === 'json' ? JSON.stringify(value) : value;
      this.store.set(k, { value: stored as any, encoding }, { ttl });
    }
  }

  async clear(namespaces: CacheNamespace[]): Promise<void> {
    // Clear all since we don't namespace LRU keys
    this.store.clear();
  }
}
```

Then use it:

```ts
compiledGraph = graph.compile({
  cache: new InMemoryNodeCache(),
});
```

This gives LangGraph's built-in deduplication a meaningful in-memory store instead of a Postgres round-trip. Hit rate will still be low for extractor nodes (unique inputs per request) but zero cost.

### Step 3: Add BE1 response cache

File: `src/services/newAgent.ts` — inside `loadGroundingNode`

**Add a module-level LRU cache**:

```ts
// Near top of file, after imports:
import { LRUCache } from 'lru-cache';

const be1Cache = new LRUCache<string, any>({ max: 200, ttl: 1000 * 60 * 60 }); // 1 hour TTL
```

**Wrap BE1 calls** with cache check:

```ts
// In loadGroundingNode, before the fetchBe1Calculate call:
const cacheKey = JSON.stringify({ kundli, varga, infolevel, nesting, type: 'calculate' });
const cached = be1Cache.get(cacheKey);
let apiResponse: any;
if (cached) {
  console.log('[load_grounding] BE1 calculate cache HIT');
  apiResponse = cached;
} else {
  console.log('[load_grounding] BE1 calculate cache MISS');
  apiResponse = await fetchBe1Calculate(kundli, { varga, infolevel, nesting });
  be1Cache.set(cacheKey, apiResponse);
}
```

Same for transit:

```ts
const transitCacheKey = JSON.stringify({ kundli, date: new Date().toISOString().slice(0, 10), type: 'transit' });
const cachedTransit = be1Cache.get(transitCacheKey);
if (cachedTransit) {
  // use cached
} else {
  // fetch and cache
}
```

Use `JSON.stringify(kundli)` as key — kundli is a small object with deterministic fields.

### Step 4: Add LLM answer cache

File: `src/services/newAgent.ts` — in `runKundliAgentV2`, before `graph.invoke()`

**Add module-level LRU cache**:

```ts
const llmAnswerCache = new LRUCache<string, string>({ max: 1000, ttl: 1000 * 60 * 60 * 24 }); // 24h TTL
```

**Check before graph invoke**:

```ts
// In runKundliAgentV2, before const result = await compiledGraph.invoke(...)
const cacheKey = JSON.stringify({
  kundli: input.kundli,
  message: input.message?.trim().toLowerCase(),
  mode: input.mode,
  date: new Date().toISOString().slice(0, 10), // daily expiry
});
const cachedAnswer = llmAnswerCache.get(cacheKey);
if (cachedAnswer) {
  console.log('[runKundliAgentV2] LLM answer cache HIT');
  return { answer: cachedAnswer, model: 'cozmic-agent-v2 (cached)' };
}
```

**Store after graph invoke**:

```ts
// After const result = await compiledGraph.invoke(...)
llmAnswerCache.set(cacheKey, finalAnswer);
```

### Summary

| Cache | TTL | Max Size | Saves | Risk |
|-------|-----|----------|-------|------|
| Node-level LangGraph (`InMemoryNodeCache`) | 30 min | 500 entries | Negligible standalone | None |
| BE1 API response | 1 hour | 200 entries | ~300-400ms/req | None (deterministic) |
| LLM answer | 24 hours | 1000 entries | ~25-40s/req | Low (non-deterministic but stable) |

### Edge Cases

- **Kundli updates**: LLM cache key includes full kundli object, so any change in birth data produces a new key
- **Session-specific answers**: Cache key does NOT include sessionId — same question + same chart = same answer across sessions (by design)
- **Cache invalidation on server restart**: All caches are in-memory, lost on process restart. Acceptable for MVP. Can move to Valkey/Redis later for persistence.
