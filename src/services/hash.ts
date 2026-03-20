export function stableHash(input: string): string {
  let h1 = 0xdeadbeef;
  let h2 = 0x41c6ce57;

  for (let i = 0; i < input.length; i += 1) {
    const ch = input.charCodeAt(i);
    h1 = Math.imul(h1 ^ ch, 2654435761);
    h2 = Math.imul(h2 ^ ch, 1597334677);
  }

  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);

  const part1 = (h2 >>> 0).toString(16).padStart(8, '0');
  const part2 = (h1 >>> 0).toString(16).padStart(8, '0');
  return `${part1}${part2}`;
}

export function toDocId(prefix: string, raw: string): string {
  return `${prefix}_${stableHash(raw).slice(0, 24)}`;
}
