function stableHash(input: string): string {
  let hash = 0;
  for (let i = 0; i < input.length; i++) {
    const char = input.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return Math.abs(hash).toString(36);
}

function getTimeBucketForIntent(intentPrimary: string): number {
  const now = Date.now();
  switch (intentPrimary) {
    case 'transit':
      return Math.floor(now / 60_000);
    case 'dasha':
      return Math.floor(now / 3_600_000);
    default:
      return Math.floor(now / 300_000);
  }
}

interface DataPlan {
  varga: string[];
  infolevel: string[];
  needsTransit: boolean;
  nesting: number;
}

interface QuestionIntent {
  primary: string;
  flags: string[];
  topics?: string[];
  timeDirection?: string;
}

interface AgentStateType {
  profileId: string | null;
  intent: QuestionIntent | null;
  dataPlan: DataPlan | null;
}

export function buildGroundingCacheKey(state: AgentStateType): string {
  const dataPlan = state.dataPlan;
  const intentPrimary = state.intent?.primary ?? 'general';
  const timeBucket = getTimeBucketForIntent(intentPrimary);

  const intentKey = intentPrimary;
  const intentFlags = (state.intent?.flags ?? []).sort().join('+');

  const varga = dataPlan?.varga.join(',') ?? 'D1';
  const infolevel = dataPlan?.infolevel.join(',') ?? 'basic';

  const raw = `${state.profileId}:${varga}:${infolevel}:${intentKey}:${intentFlags}:${timeBucket}`;
  return stableHash(raw);
}

export function buildGroundingCacheKeySimple(
  profileId: string,
  varga: string,
  infolevel: string,
  intentPrimary: string,
  intentFlags: string[] = []
): string {
  const timeBucket = getTimeBucketForIntent(intentPrimary);
  const flagsKey = intentFlags.sort().join('+');
  const raw = `${profileId}:${varga}:${infolevel}:${intentPrimary}:${flagsKey}:${timeBucket}`;
  return stableHash(raw);
}