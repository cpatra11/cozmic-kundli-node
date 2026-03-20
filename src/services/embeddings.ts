import { stableHash } from './hash.js';

export interface EmbeddingResult {
  vector: number[];
  model: string;
  dimension: number;
}

const TOKEN_SPLIT_REGEX = /[^\p{L}\p{N}_]+/u;

export function embedTextDeterministic(text: string, dimension: number): EmbeddingResult {
  const tokens = text
    .toLowerCase()
    .split(TOKEN_SPLIT_REGEX)
    .map((token) => token.trim())
    .filter(Boolean);

  const vector = new Array<number>(dimension).fill(0);

  for (const token of tokens) {
    const hA = stableHash(`a:${token}`);
    const hB = stableHash(`b:${token}`);
    const bucket = parseInt(hA.slice(0, 8), 16) % dimension;
    const sign = parseInt(hB.slice(0, 2), 16) % 2 === 0 ? 1 : -1;
    vector[bucket] += sign;
  }

  const norm = Math.sqrt(vector.reduce((acc, value) => acc + value * value, 0)) || 1;
  const normalized = vector.map((value) => value / norm);

  return {
    vector: normalized,
    model: 'deterministic-hash-v1',
    dimension,
  };
}

export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return 0;

  let dot = 0;
  let magA = 0;
  let magB = 0;

  for (let i = 0; i < a.length; i += 1) {
    dot += a[i]! * b[i]!;
    magA += a[i]! * a[i]!;
    magB += b[i]! * b[i]!;
  }

  const denom = Math.sqrt(magA) * Math.sqrt(magB);
  if (denom === 0) return 0;
  return dot / denom;
}
