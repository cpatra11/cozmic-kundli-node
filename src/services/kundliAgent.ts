import { Annotation, END, START, StateGraph } from '@langchain/langgraph';
import { type KundliSnapshotInput } from './be1Client.js';
import { env } from '../config/env.js';
import { COLLECTIONS, type RagApiSourceDocument, type RagProfileDocument } from '../models/firestoreModels.js';
import { getPostgresStore } from './postgresStore.js';
import { stableHash } from './hash.js';
import { invokeDeepSeekBedrock } from './deepseekBedrock.js';
import { buildTransitToolFinding, type ToolFinding } from './astrologyTools.js';

export interface AgentAnswerInput {
  ownerId?: string;
  message: string;
  mode?: 'mini' | 'pro';
  kundli?: KundliSnapshotInput;
  profileId?: string;
  clientTimestamp?: number;
  conversationContext?: string[];
  onStage?: (stage: AnalysisStage) => void;
}

type AgentMode = 'mini' | 'pro';

type IntentPrimary = 'd9' | 'dasha' | 'transit' | 'general';
type QuestionFamily =
  | 'general'
  | 'career'
  | 'marriage'
  | 'relationship'
  | 'finance'
  | 'health'
  | 'longevity'
  | 'education'
  | 'children'
  | 'property'
  | 'travel'
  | 'spirituality'
  | 'timing'
  | 'yoga'
  | 'family';
type ChartLayer = 'D1' | 'D9' | 'D10' | 'D8' | 'D30' | 'D7' | 'D4' | 'D12' | 'D20';
type MicroSignal = 'nakshatra' | 'nakshatra_lord' | 'sign_lord' | 'drishti' | 'degree';
type TimeSource = 'client' | 'server';
type TimeDirection = 'past' | 'future' | 'present';
type TimeUnit = 'day' | 'week' | 'month' | 'year';

type AnalysisStage = {
  id: string;
  label: string;
  status: 'completed';
  details?: string;
};

type DynamicExecutionPlan = {
  family: QuestionFamily;
  chartLayers: ChartLayer[];
  includeTiming: boolean;
  includeTransit: boolean;
  includeDasha: boolean;
  includeCareer: boolean;
  includeRelationship: boolean;
  includeMicroSignals: MicroSignal[];
  seriesNodes: string[];
  parallelBatches: string[][];
};

type CoverageGap = 'varga' | 'd9' | 'dasha' | 'transit' | 'career' | 'longevity';

type QuestionIntent = {
  primary: IntentPrimary;
  flags: string[];
  topics: string[];
  timeDirection: TimeDirection;
  timeValue?: number;
  timeUnit?: TimeUnit;
  timeLabel?: string;
};

type SelectedSection = {
  path: string;
  value: unknown;
};

type AtlasItem = {
  path: string;
  kind: 'object' | 'array' | 'scalar' | 'missing';
  summary: string;
  sampleKeys?: string[];
};

type GroundingContext = {
  ownerId: string;
  profileId: string;
  sourceDocId: string;
  chartVersion: string;
  kundliSignature: string;
  kundli: KundliSnapshotInput;
  requestKey: string;
  payloadHash: string;
  referenceTimestamp: number;
  referenceTimeSource: TimeSource;
  rawPayload: unknown;
  selectedPaths: string[];
  selectedSections: SelectedSection[];
};

export interface AgentAnswer {
  answer: string;
  model: string;
  mode: AgentMode;
  executionPlan?: DynamicExecutionPlan;
  analysisStages?: AnalysisStage[];
  grounding?: {
    ownerId: string;
    profileId: string;
    sourceDocId: string;
    chartVersion: string;
    kundliSignature: string;
    kundli: KundliSnapshotInput;
    requestKey: string;
    payloadHash: string;
    referenceTimestamp: number;
    referenceTimeSource: TimeSource;
    selectedPaths: string[];
  };
}

const AgentState = Annotation.Root({
  ownerId: Annotation<string>,
  profileId: Annotation<string>,
  mode: Annotation<AgentMode>,
  question: Annotation<string>,
  kundliInput: Annotation<KundliSnapshotInput | null>,
  referenceTimestamp: Annotation<number | null>,
  referenceTimeSource: Annotation<TimeSource | null>,
  conversationContext: Annotation<string[]>,
  stageReporter: Annotation<((stage: AnalysisStage) => void) | null>,
  toolIteration: Annotation<number>,
  maxToolIterations: Annotation<number>,
  intent: Annotation<QuestionIntent | null>,
  executionPlan: Annotation<DynamicExecutionPlan | null>,
  coverageGaps: Annotation<CoverageGap[]>,
  grounding: Annotation<GroundingContext | null>,
  analysisStages: Annotation<AnalysisStage[]>,
  toolFindings: Annotation<ToolFinding[]>,
  prompt: Annotation<string | null>,
  answer: Annotation<string | null>,
  model: Annotation<string | null>,
});

type AgentStateType = typeof AgentState.State;
type AgentUpdateType = typeof AgentState.Update;

const CONCISE_ANSWER_MAX_CHARS = 900;
const CONCISE_ANSWER_MIN_LINES = 6;

function normalizeProfileId(input: AgentAnswerInput): string | null {
  if (input.profileId?.trim()) return input.profileId.trim();
  if (input.kundli) {
    return `p_${stableHash(JSON.stringify(input.kundli)).slice(0, 10)}`;
  }
  return null;
}

function getByPath(value: unknown, path: string): unknown {
  const parts = path.split('.');
  let current: unknown = value;

  for (const part of parts) {
    if (current === null || current === undefined || typeof current !== 'object') {
      return undefined;
    }

    const next = (current as Record<string, unknown>)[part];
    if (next === undefined) return undefined;
    current = next;
  }

  return current;
}

function getFirstPathValue(value: unknown, paths: string[]): SelectedSection | null {
  for (const path of paths) {
    const found = getByPath(value, path);
    if (found !== undefined) {
      return { path, value: found };
    }
  }
  return null;
}

function formatValue(value: unknown): string {
  if (value === undefined) return 'undefined';
  if (value === null) return 'null';
  if (typeof value === 'string') return value;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function truncateText(value: string, maxChars: number): string {
  if (value.length <= maxChars) return value;
  return `${value.slice(0, Math.max(0, maxChars - 16))}\n...[truncated]`;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function objectKeys(value: unknown): string[] {
  return isPlainObject(value) ? Object.keys(value) : [];
}

function formatNumber(value: unknown, digits = 2): string {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : String(value ?? 'n/a');
}

function toValidDate(value: string | number | Date | null | undefined): Date | null {
  if (value === null || value === undefined) return null;
  if (value instanceof Date) return Number.isNaN(value.getTime()) ? null : value;

  if (typeof value === 'number') {
    const dt = new Date(value);
    return Number.isNaN(dt.getTime()) ? null : dt;
  }

  const normalized = String(value).trim();
  if (!normalized) return null;

  const candidates = normalized.includes('T') ? [normalized, `${normalized}Z`] : [normalized.replace(' ', 'T'), `${normalized.replace(' ', 'T')}Z`, normalized];

  for (const candidate of candidates) {
    const dt = new Date(candidate);
    if (!Number.isNaN(dt.getTime())) return dt;
  }

  return null;
}

function formatIsoLike(value: string | number | Date | null | undefined): string {
  const dt = toValidDate(value);
  return dt ? dt.toISOString() : 'n/a';
}

function formatPlacement(value: unknown): string {
  if (!isPlainObject(value)) return formatValue(value);
  const parts: string[] = [];
  if ('rashi' in value) parts.push(`rashi ${rashiName(value.rashi)}`);
  if ('house_number' in value) parts.push(`house ${String(value.house_number)}`);
  if ('degree' in value) parts.push(`${formatNumber(value.degree)}°`);
  if ('longitude' in value) parts.push(`lon ${formatNumber(value.longitude)}`);
  if ('speed' in value) parts.push(`speed ${formatNumber(value.speed, 4)}`);
  return parts.length > 0 ? parts.join(', ') : formatValue(value);
}

function formatArudhaPlacement(value: unknown): string {
  if (!isPlainObject(value)) return formatValue(value);

  const parts: string[] = [];
  if ('rashi' in value) {
    parts.push(`rashi ${rashiName(value.rashi)}`);
  }
  if ('degree' in value) parts.push(`${formatNumber(value.degree)}°`);
  if ('longitude' in value) parts.push(`lon ${formatNumber(value.longitude)}`);
  if ('house_number' in value) parts.push(`house ${String(value.house_number)}`);
  if ('speed' in value) parts.push(`speed ${formatNumber(value.speed, 4)}`);

  return parts.length > 0 ? parts.join(', ') : formatValue(value);
}

function summarizeArudhaSection(value: unknown, limit = 12): string[] {
  if (!isPlainObject(value)) {
    return [formatArudhaPlacement(value)];
  }

  return Object.entries(value)
    .slice(0, limit)
    .map(([key, nested]) => {
      if (isPlainObject(nested)) {
        return `${key}: ${formatArudhaPlacement(nested)}`;
      }

      return `${key}: ${formatValue(nested)}`;
    });
}

function asList(value: unknown): string[] {
  if (!value) return [];
  if (Array.isArray(value)) return value.map((item) => String(item));
  if (isPlainObject(value)) return Object.keys(value);
  return [String(value)];
}

function chooseBestPath(rawPayload: unknown, paths: string[]): SelectedSection | null {
  return getFirstPathValue(rawPayload, paths);
}

function resolveReferenceTime(input: AgentAnswerInput): { timestamp: number; source: TimeSource } {
  if (typeof input.clientTimestamp === 'number' && Number.isFinite(input.clientTimestamp)) {
    return { timestamp: input.clientTimestamp, source: 'client' };
  }

  return { timestamp: Date.now(), source: 'server' };
}

function extractRequestedVargaKeys(question: string): string[] {
  const q = question.toLowerCase();
  const keys = new Set<string>();

  const named: Array<{ pattern: RegExp; key: string }> = [
    { pattern: /\bnavamsha\b|\bnavamsa\b|\bd9\b/, key: 'D9' },
    { pattern: /\bdasamsa\b|\bd10\b/, key: 'D10' },
    { pattern: /\bdwadasamsa\b|\bd12\b/, key: 'D12' },
    { pattern: /\bshodasamsa\b|\bd16\b/, key: 'D16' },
    { pattern: /\bvimsamsa\b|\bd20\b/, key: 'D20' },
    { pattern: /\bchaturvimshamsa\b|\bd24\b/, key: 'D24' },
    { pattern: /\bsaptavimshamsa\b|\bd27\b/, key: 'D27' },
    { pattern: /\btrimshamsa\b|\bd30\b/, key: 'D30' },
    { pattern: /\bkhavedamsa\b|\bd40\b/, key: 'D40' },
    { pattern: /\bakshavedamsa\b|\bd45\b/, key: 'D45' },
    { pattern: /\bshashtiamsa\b|\bd60\b/, key: 'D60' },
    { pattern: /\bd2\b|\bhora\b/, key: 'D2' },
    { pattern: /\bd3\b|\bdrekkana\b/, key: 'D3' },
    { pattern: /\bd4\b|\bchaturthamsa\b/, key: 'D4' },
    { pattern: /\bd7\b|\bsaptamsa\b/, key: 'D7' },
    { pattern: /\bd1\b|\brasi\b|\brashi\b|\blagna\b|\bascendant\b/, key: 'D1' },
  ];

  for (const item of named) {
    if (item.pattern.test(q)) keys.add(item.key);
  }

  const explicit = q.match(/\bd(\d{1,2})\b/g) ?? [];
  for (const token of explicit) {
    const num = Number(token.slice(1));
    if (Number.isInteger(num) && num >= 1 && num <= 60) {
      keys.add(`D${num}`);
    }
  }

  return [...keys];
}

function evaluateMiniScope(question: string): { allowed: boolean; reasons: string[] } {
  const reasons: string[] = [];
  const q = question.toLowerCase();

  const vargaKeys = extractRequestedVargaKeys(question);
  if (vargaKeys.some((key) => key !== 'D1' && key !== 'D9')) {
    reasons.push('This asks for advanced divisional charts beyond D1/D9.');
  }

  const intent = classifyQuestionIntent(question);
  const family = determineQuestionFamily(question, intent);
  const allowedFamilies = new Set<QuestionFamily>(['general', 'relationship', 'marriage']);

  if (!allowedFamilies.has(family)) {
    reasons.push(`This falls under ${family} analysis, which is Pro scope.`);
  }

  const proTimingFlags = new Set(['timing', 'dasha', 'transit', 'forecast', 'history', 'career_timing']);
  if (intent.flags.some((flag) => proTimingFlags.has(flag))) {
    reasons.push('This requires timing/predictive analysis (dasha/transit/forecast).');
  }

  if (/(\bashtakavarga\b|\barudha\b|\baruda\b|\bshadbala\b|\bkp\b|\bjaimini\b|\bnadi\b)/.test(q)) {
    reasons.push('This requests advanced systems reserved for Pro mode.');
  }

  return { allowed: reasons.length === 0, reasons };
}

function buildMiniUpgradeResponse(question: string, reasons: string[] = []): string {
  const why = reasons.length > 0
    ? [``, 'Why this is Pro:', ...reasons.slice(0, 3).map((reason) => `- ${reason}`)]
    : [];

  return [
    'You are currently in **Cozmic Mini** mode.',
    'This request needs advanced analysis that is available in **Cozmic Pro**.',
    '',
    'Mini supports: **D1, D9, and basic non-predictive astrology insights**.',
    ...why,
    '',
    `Please switch to **Cozmic Pro** and ask again: "${question.trim()}"`,
  ].join('\n');
}

function summarizeVargaSection(vargaKey: string, value: unknown, question: string): string[] {
  const facts: string[] = [];

  if (!isPlainObject(value)) {
    return [`${vargaKey}: ${formatValue(value)}`];
  }

  const lagna = getByPath(value, 'lagna.Lg') ?? getByPath(value, 'lagna');
  if (lagna) facts.push(`${vargaKey} Lagna: ${formatPlacement(lagna)}`);

  const graha = getByPath(value, 'graha');
  if (isPlainObject(graha)) {
    const lines = summarizeKeyPlacements(graha, 8);
    if (lines.length > 0) facts.push(`${vargaKey} graha: ${lines.join(' | ')}`);
  }

  const bhava = getByPath(value, 'bhava');
  if (isPlainObject(bhava)) {
    const lines = summarizeKeyPlacements(bhava, 8);
    if (lines.length > 0) facts.push(`${vargaKey} bhava: ${lines.join(' | ')}`);
  }

  if (/\b(marriage|relationship|partner|spouse|love|compatibility)\b/i.test(question)) {
    const venus = getByPath(value, 'graha.Ve');
    const jupiter = getByPath(value, 'graha.Ju');
    if (venus) facts.push(`${vargaKey} Venus focus: ${formatPlacement(venus)}`);
    if (jupiter) facts.push(`${vargaKey} Jupiter focus: ${formatPlacement(jupiter)}`);
  }

  if (/\b(career|job|profession|business|work|promotion)\b/i.test(question)) {
    const sun = getByPath(value, 'graha.Su');
    const mercury = getByPath(value, 'graha.Me');
    const saturn = getByPath(value, 'graha.Sa');
    if (sun) facts.push(`${vargaKey} Sun focus: ${formatPlacement(sun)}`);
    if (mercury) facts.push(`${vargaKey} Mercury focus: ${formatPlacement(mercury)}`);
    if (saturn) facts.push(`${vargaKey} Saturn focus: ${formatPlacement(saturn)}`);
  }

  return facts.length > 0 ? facts : [`${vargaKey}: ${formatValue(value)}`];
}

function makeVargaToolFinding(rawPayload: unknown, question: string, mode: AgentMode): ToolFinding {
  const varga = getFirstPathValue(rawPayload, ['chart.varga', 'varga']);
  if (!varga || !isPlainObject(varga.value)) {
    return {
      name: 'Varga analyzer',
      status: 'unavailable',
      facts: ['Canonical payload has no varga section.'],
      evidencePaths: [],
      missing: ['chart.varga'],
    };
  }

  const availableKeys = Object.keys(varga.value)
    .filter((key) => /^D\d{1,2}$/.test(key))
    .sort((a, b) => Number(a.slice(1)) - Number(b.slice(1)));

  const modeAllowedKeys = mode === 'mini' ? ['D1', 'D9'] : availableKeys;
  const filteredAvailableKeys = availableKeys.filter((key) => modeAllowedKeys.includes(key));

  const requestedKeys = extractRequestedVargaKeys(question);
  const selectedKeys = [...new Set(['D1', ...requestedKeys].filter((key) => filteredAvailableKeys.includes(key)))].slice(0, 8);
  const effectiveKeys = selectedKeys.length > 0 ? selectedKeys : filteredAvailableKeys.slice(0, 8);

  const facts: string[] = [];
  const evidencePaths: string[] = [];

  for (const key of effectiveKeys) {
    const section = (varga.value as Record<string, unknown>)[key];
    if (section === undefined) continue;
    evidencePaths.push(`chart.varga.${key}`);
    facts.push(...summarizeVargaSection(key, section, question).slice(0, 8));
  }

  const remaining = filteredAvailableKeys.filter((key) => !effectiveKeys.includes(key));
  if (remaining.length > 0) {
    facts.push(`Additional vargas available: ${remaining.join(', ')}.`);
  }

  if (mode === 'mini') {
    facts.push('Mini mode scope enforced: only D1 and D9 are considered.');
  }

  return {
    name: 'Varga analyzer',
    status: facts.length > 0 ? 'ok' : 'partial',
    facts,
    evidencePaths,
    snippets: effectiveKeys.map((key) => truncateText(JSON.stringify((varga.value as Record<string, unknown>)[key], null, 2), 1200)),
  };
}

function summarizeNode(path: string, value: unknown): AtlasItem {
  if (value === undefined) {
    return { path, kind: 'missing', summary: 'missing' };
  }
  if (Array.isArray(value)) {
    return {
      path,
      kind: 'array',
      summary: `array(${value.length})`,
      sampleKeys: value.slice(0, 6).map((item, index) => `${index}:${typeof item}`),
    };
  }
  if (!isPlainObject(value)) {
    return { path, kind: 'scalar', summary: formatValue(value) };
  }
  const keys = Object.keys(value);
  return {
    path,
    kind: 'object',
    summary: `${keys.length} key(s)`,
    sampleKeys: keys.slice(0, 16),
  };
}

function summarizeChartAtlas(rawPayload: unknown): AtlasItem[] {
  const atlas: AtlasItem[] = [];
  const chart = getByPath(rawPayload, 'chart');
  atlas.push(summarizeNode('chart', chart));

  if (!isPlainObject(chart)) return atlas;

  const chartKeys = Object.keys(chart);
  atlas.push({
    path: 'chart.keys',
    kind: 'array',
    summary: chartKeys.join(', ') || 'no keys',
    sampleKeys: chartKeys,
  });

  const varga = chart.varga;
  atlas.push(summarizeNode('chart.varga', varga));
  if (isPlainObject(varga)) {
    for (const [key, value] of Object.entries(varga)) {
      atlas.push(summarizeNode(`chart.varga.${key}`, value));
    }
  }

  for (const key of ['graha', 'lagna', 'bhava', 'houses', 'panchanga', 'yogas', 'arudha', 'dasha', 'ashtakavarga', 'kala', 'rising', 'ayanamsa', 'grahabala', 'rashibala']) {
    atlas.push(summarizeNode(`chart.${key}`, (chart as Record<string, unknown>)[key]));
  }

  return atlas;
}

function formatSelectedSectionForPrompt(section: SelectedSection): string {
  if (/\barudha\b/i.test(section.path)) {
    const arudhaLines = summarizeArudhaSection(section.value, 16);
    return arudhaLines.length > 0 ? arudhaLines.join(' | ') : formatArudhaPlacement(section.value);
  }

  return truncateText(formatValue(section.value), 900);
}

function makeAtlasToolFinding(rawPayload: unknown): ToolFinding {
  const atlas = summarizeChartAtlas(rawPayload);
  return {
    name: 'Chart atlas',
    status: atlas.length > 0 ? 'ok' : 'partial',
    facts: atlas.slice(0, 30).map((item) => `${item.path}: ${item.summary}${item.sampleKeys?.length ? ` | ${item.sampleKeys.join(', ')}` : ''}`),
    evidencePaths: atlas.slice(0, 30).map((item) => item.path),
    snippets: atlas.slice(0, 20).map((item) => `${item.path} => ${item.summary}`),
  };
}

function summarizeKeyPlacements(node: unknown, limit = 12): string[] {
  if (!isPlainObject(node)) return [];
  return Object.entries(node)
    .slice(0, limit)
    .map(([key, value]) => `${key}: ${formatPlacement(value)}`);
}

function pickQuestionScope(question: string): string[] {
  const q = question.toLowerCase();
  const candidates = new Set<string>();
  const add = (...items: string[]) => items.forEach((item) => candidates.add(item));

  if (/\b(d9|navamsha|navamsa|marriage|relationship|partner|spouse|love|compatibility)\b/.test(q)) add('chart.varga.D9');
  if (/\b(d10|career|job|profession|business|work|promotion)\b/.test(q)) add('chart.varga.D10');
  if (/\b(dasha|dasa|mahadasha|antardasha|vimshottari|period|timing|when)\b/.test(q)) add('chart.dasha');
  if (/\b(longevity|lifespan|life span|how long will i live|length of life|ayush|ayu|mrityu|death|end of life)\b/.test(q)) {
    add('chart.varga.D8', 'chart.varga.D30', 'chart.dasha', 'chart.varga.D1', 'chart.graha', 'chart.bhava');
  }
  if (/\b(transit|gochar|sun transit|current transit|today|now)\b/.test(q)) add('chart.transit', 'chart.transits', 'chart.gochar');
  if (/\b(yoga|yogas)\b/.test(q)) add('chart.yogas');
  if (/\b(ashtakavarga)\b/.test(q)) add('chart.ashtakavarga');
  if (/\b(panchanga|tithi|nakshatra|karana|yoga)\b/.test(q)) add('chart.panchanga');
  if (/\b(arudha|aruda)\b/.test(q)) add('chart.arudha');
  if (/\b(house|bhava|lagna|ascendant|rashi|sign|planet|graha|moon|sun|mars|mercury|jupiter|venus|saturn|rahu|ketu)\b/.test(q)) {
    add('chart.graha', 'chart.lagna', 'chart.bhava', 'chart.houses', 'chart.varga.D1');
  }

  if (candidates.size === 0) {
    add('chart.graha', 'chart.lagna', 'chart.varga.D1');
  }

  return [...candidates];
}

const PLANET_LABELS: Record<string, string> = {
  Su: 'Sun',
  Mo: 'Moon',
  Ma: 'Mars',
  Me: 'Mercury',
  Ju: 'Jupiter',
  Ve: 'Venus',
  Sa: 'Saturn',
  Ra: 'Rahu',
  Ke: 'Ketu',
};

const RASHI_LORDS: Record<number, keyof typeof PLANET_LABELS> = {
  1: 'Ma',
  2: 'Ve',
  3: 'Me',
  4: 'Mo',
  5: 'Su',
  6: 'Me',
  7: 'Ve',
  8: 'Ma',
  9: 'Ju',
  10: 'Sa',
  11: 'Sa',
  12: 'Ju',
};

const RASHI_NAMES: Record<number, string> = {
  1: 'Aries',
  2: 'Taurus',
  3: 'Gemini',
  4: 'Cancer',
  5: 'Leo',
  6: 'Virgo',
  7: 'Libra',
  8: 'Scorpio',
  9: 'Sagittarius',
  10: 'Capricorn',
  11: 'Aquarius',
  12: 'Pisces',
};

function rashiName(value: unknown): string {
  const n = Number(value);
  if (Number.isInteger(n) && n >= 1 && n <= 12) {
    return `${RASHI_NAMES[n]} (${n})`;
  }
  return String(value ?? 'unknown');
}

function parseTemporalWindow(question: string): { direction: TimeDirection; value?: number; unit?: TimeUnit; label?: string } {
  const q = question.toLowerCase();

  const explicitPast = q.match(/\b(?:last|past|previous|earlier|ago)\s+(\d+)\s+(day|days|week|weeks|month|months|year|years)\b/);
  if (explicitPast) {
    const value = Number(explicitPast[1]);
    const unit = explicitPast[2].replace(/s$/, '') as TimeUnit;
    return { direction: 'past', value, unit, label: `past ${value} ${unit}${value === 1 ? '' : 's'}` };
  }

  const explicitFuture = q.match(/\b(?:next|upcoming|coming|in)\s+(\d+)\s+(day|days|week|weeks|month|months|year|years)\b/);
  if (explicitFuture) {
    const value = Number(explicitFuture[1]);
    const unit = explicitFuture[2].replace(/s$/, '') as TimeUnit;
    return { direction: 'future', value, unit, label: `next ${value} ${unit}${value === 1 ? '' : 's'}` };
  }

  if (/\b(10|ten)\s+years?\s+ago\b/.test(q) || /\bpast\s+10\s+years?\b/.test(q) || /\blast\s+10\s+years?\b/.test(q)) {
    return { direction: 'past', value: 10, unit: 'year', label: 'past 10 years' };
  }

  if (/\bnext\s+10\s+years?\b/.test(q) || /\bin\s+10\s+years?\b/.test(q)) {
    return { direction: 'future', value: 10, unit: 'year', label: 'next 10 years' };
  }

  if (/\b(tomorrow|next week|next month|next year|upcoming|future|later|after)\b/.test(q)) {
    return { direction: 'future', label: 'future-oriented' };
  }

  if (/\b(past|last|previous|earlier|before|history|retrospective|back then)\b/.test(q)) {
    return { direction: 'past', label: 'past-oriented' };
  }

  return { direction: 'present', label: 'present' };
}

function classifyQuestionIntent(question: string): QuestionIntent {
  const q = question.toLowerCase();
  const flags = new Set<string>();
  const topics = new Set<string>();
  const temporal = parseTemporalWindow(question);

  if (/\b(d9|navamsha|navamsa)\b/.test(q)) flags.add('d9');
  if (/\b(dasha|dasa|mahadasha|antardasha|vimshottari|period|timing|timeline|when)\b/.test(q)) flags.add('dasha');
  if (/\b(transit|gochar|current transit|current sun|sun transit|today|now|tomorrow|next week|next month|next year|future|past|last|previous|ago)\b/.test(q)) flags.add('transit');

  if (/\b(career|job|profession|business|promotion|work|office|employment|salary|resume|interview)\b/.test(q)) {
    flags.add('career');
    topics.add('career');
  }
  if (/\b(marriage|relationship|partner|spouse|love|compatibility|romance|dating)\b/.test(q)) {
    flags.add('relationship');
    topics.add('relationship');
  }
  if (/\b(wealth|money|income|finance|assets|property|investment|profits|revenue)\b/.test(q)) {
    flags.add('finance');
    topics.add('finance');
  }
  if (/\b(health|disease|illness|medical|surgery|recovery|fitness|stress)\b/.test(q)) {
    flags.add('health');
    topics.add('health');
  }
  if (/\b(longevity|lifespan|life span|how long will i live|length of life|ayush|ayu|mrityu|death|end of life)\b/.test(q)) {
    flags.add('longevity');
    flags.add('timing');
    topics.add('health');
  }
  if (/\b(education|study|studies|exam|exams|degree|college|school|learning|research)\b/.test(q)) {
    topics.add('education');
  }
  if (/\b(children|child|kids|pregnancy|pregnant|pregnancy)\b/.test(q)) {
    topics.add('children');
  }
  if (/\b(property|house|home|land|real estate|vehicle|car|asset)\b/.test(q)) {
    topics.add('property');
  }
  if (/\b(travel|traveling|foreign|abroad|visa|relocation|migration|move)\b/.test(q)) {
    topics.add('travel');
  }
  if (/\b(spiritual|spirituality|moksha|meditation|religion|faith|guru)\b/.test(q)) {
    topics.add('spirituality');
  }
  if (/\b(communication|writing|speech|speaking|media|marketing|technology|coding|tech)\b/.test(q)) {
    topics.add('communication');
  }

  if (temporal.direction !== 'present') {
    flags.add('timing');
  }

  if (topics.has('career') && temporal.direction !== 'present') {
    flags.add('career_timing');
  }

  if (temporal.direction === 'future') {
    flags.add('forecast');
  } else if (temporal.direction === 'past') {
    flags.add('history');
  }

  const primary: IntentPrimary = flags.has('d9')
    ? 'd9'
    : flags.has('dasha') || flags.has('timing') || topics.has('career') || topics.has('finance') || topics.has('education')
      ? 'dasha'
      : flags.has('transit')
        ? 'transit'
        : 'general';

  return {
    primary,
    flags: [...flags],
    topics: [...topics],
    timeDirection: temporal.direction,
    timeValue: temporal.value,
    timeUnit: temporal.unit,
    timeLabel: temporal.label,
  };
}

function determineQuestionFamily(question: string, intent: QuestionIntent): QuestionFamily {
  const q = question.toLowerCase();

  if (intent.flags.includes('longevity') || /\b(longevity|lifespan|life span|how long will i live|length of life|ayush|ayu|mrityu|death|end of life)\b/.test(q)) {
    return 'longevity';
  }

  if (intent.flags.includes('career') || intent.flags.includes('career_timing') || /\b(career|job|profession|business|work|promotion)\b/.test(q)) {
    return 'career';
  }
  if (/\b(marriage|spouse|compatibility|wedding)\b/.test(q)) {
    return 'marriage';
  }
  if (intent.flags.includes('relationship') || /\b(relationship|love|romance|dating|partner)\b/.test(q)) {
    return 'relationship';
  }
  if (intent.flags.includes('finance') || /\b(wealth|money|income|finance|investment|assets)\b/.test(q)) {
    return 'finance';
  }
  if (intent.flags.includes('health') || /\b(health|disease|illness|medical|recovery|fitness)\b/.test(q)) {
    return 'health';
  }
  if (intent.topics.includes('education')) return 'education';
  if (intent.topics.includes('children')) return 'children';
  if (intent.topics.includes('property')) return 'property';
  if (intent.topics.includes('travel')) return 'travel';
  if (intent.topics.includes('spirituality')) return 'spirituality';
  if (intent.flags.includes('dasha') || intent.flags.includes('transit') || intent.flags.includes('timing')) return 'timing';
  if (/\b(yoga|yogas|raja yoga|mahapurusha)\b/.test(q)) return 'yoga';
  if (/\b(family|father|mother|siblings|parents)\b/.test(q)) return 'family';

  return 'general';
}

function buildDynamicExecutionPlan(question: string, intent: QuestionIntent, mode: AgentMode): DynamicExecutionPlan {
  const family = determineQuestionFamily(question, intent);
  const chartLayers = new Set<ChartLayer>(['D1']);

  if (family === 'marriage' || family === 'relationship') chartLayers.add('D9');
  if (family === 'career') chartLayers.add('D10');
  if (family === 'children') chartLayers.add('D7');
  if (family === 'property') chartLayers.add('D4');
  if (family === 'travel') chartLayers.add('D12');
  if (family === 'spirituality') chartLayers.add('D20');
  if (family === 'longevity') {
    chartLayers.add('D8');
    chartLayers.add('D30');
  }
  if (intent.flags.includes('d9')) chartLayers.add('D9');

  const includeTiming = intent.flags.includes('timing') || intent.flags.includes('dasha') || intent.flags.includes('transit') || family === 'timing' || family === 'longevity';
  const includeTransit = mode === 'pro' && (intent.flags.includes('transit') || includeTiming);
  const dashaDrivenFamilies = new Set<QuestionFamily>(['career', 'marriage', 'longevity', 'health', 'timing']);
  const includeDasha = mode === 'pro' && (intent.flags.includes('dasha') || includeTiming || dashaDrivenFamilies.has(family));
  const includeCareer = family === 'career' && mode === 'pro';
  const includeRelationship = family === 'marriage' || family === 'relationship';

  const includeMicroSignals: MicroSignal[] = ['nakshatra', 'nakshatra_lord', 'sign_lord', 'drishti', 'degree'];

  const seriesNodes = ['load_grounding', 'classify_intent', 'plan_execution', 'run_specialized_tools', 'run_general_tools', 'build_prompt', 'answer_with_deepseek', 'condense_answer'];
  const parallelBatches = [
    ['atlas', 'varga', 'placement'],
    [includeDasha ? 'dasha' : '', includeTransit ? 'transit' : '', includeCareer ? 'career' : ''].filter(Boolean),
    ['nakshatra', 'lordship', 'drishti_degree'],
  ].filter((batch) => batch.length > 0);

  return {
    family,
    chartLayers: [...chartLayers],
    includeTiming,
    includeTransit,
    includeDasha,
    includeCareer,
    includeRelationship,
    includeMicroSignals,
    seriesNodes,
    parallelBatches,
  };
}

function appendStage(state: AgentStateType, id: string, label: string, details?: string): AnalysisStage[] {
  const stages = state.analysisStages ?? [];
  const stage: AnalysisStage = { id, label, status: 'completed', details };
  if (state.stageReporter) {
    try {
      state.stageReporter(stage);
    } catch {
      // best effort stage emission; never break analysis flow
    }
  }
  return [...stages, stage];
}

function mergeFindings(existing: ToolFinding[], incoming: ToolFinding[]): ToolFinding[] {
  const rank: Record<ToolFinding['status'], number> = {
    ok: 3,
    partial: 2,
    unavailable: 1,
  };

  const merged = new Map<string, ToolFinding>();

  for (const finding of existing) {
    merged.set(finding.name, finding);
  }

  for (const finding of incoming) {
    const current = merged.get(finding.name);
    if (!current || rank[finding.status] >= rank[current.status]) {
      merged.set(finding.name, finding);
    }
  }

  return [...merged.values()];
}

function getFindingStatus(findings: ToolFinding[], name: string): ToolFinding['status'] | null {
  const found = findings.find((f) => f.name === name);
  return found?.status ?? null;
}

function isCoverageWeak(status: ToolFinding['status'] | null, strict = false): boolean {
  if (status === null) return true;
  if (status === 'unavailable') return true;
  if (strict && status === 'partial') return true;
  return false;
}

function determineCoverageGaps(state: AgentStateType): CoverageGap[] {
  const executionPlan = state.executionPlan ?? (state.intent ? buildDynamicExecutionPlan(state.question, state.intent, state.mode) : null);
  if (!executionPlan) return [];

  const findings = state.toolFindings ?? [];
  const strict = (state.toolIteration ?? 0) === 0;
  const gaps = new Set<CoverageGap>();

  if (isCoverageWeak(getFindingStatus(findings, 'Varga analyzer'), strict)) {
    gaps.add('varga');
  }

  if ((executionPlan.chartLayers.includes('D9') || state.intent?.primary === 'd9') && isCoverageWeak(getFindingStatus(findings, 'D9 analyzer'), strict)) {
    gaps.add('d9');
  }

  if (executionPlan.includeDasha && isCoverageWeak(getFindingStatus(findings, 'Dasha analyzer'), strict)) {
    gaps.add('dasha');
  }

  if (executionPlan.includeTransit && isCoverageWeak(getFindingStatus(findings, 'Transit analyzer'), strict)) {
    gaps.add('transit');
  }

  if (executionPlan.includeCareer && isCoverageWeak(getFindingStatus(findings, 'Career analyzer'), strict)) {
    gaps.add('career');
  }

  if (executionPlan.family === 'longevity' && isCoverageWeak(getFindingStatus(findings, 'Longevity analyzer'), strict)) {
    gaps.add('longevity');
  }

  return [...gaps];
}

function planetPathHints(question: string): string[] {
  const q = question.toLowerCase();
  const hints: string[] = [];
  const map: Array<{ pattern: RegExp; code: string }> = [
    { pattern: /\b(sun|surya)\b/, code: 'Su' },
    { pattern: /\b(moon|chandra)\b/, code: 'Mo' },
    { pattern: /\b(mars|mangal)\b/, code: 'Ma' },
    { pattern: /\b(mercury|budh)\b/, code: 'Me' },
    { pattern: /\b(jupiter|guru|brihaspati)\b/, code: 'Ju' },
    { pattern: /\b(venus|shukra)\b/, code: 'Ve' },
    { pattern: /\b(saturn|shani)\b/, code: 'Sa' },
    { pattern: /\b(rahu)\b/, code: 'Ra' },
    { pattern: /\b(ketu)\b/, code: 'Ke' },
  ];

  for (const item of map) {
    if (item.pattern.test(q)) {
      hints.push(`chart.graha.${item.code}`, `chart.varga.D9.graha.${item.code}`, `chart.varga.D10.graha.${item.code}`);
    }
  }

  return hints;
}

function selectRelevantPaths(question: string, flags: string[] = [], mode: AgentMode = 'pro'): string[] {
  const q = question.toLowerCase();
  const paths = new Set<string>([
    'chart.user',
    'chart.graha',
    'chart.lagna',
    'chart.houses',
    'chart.bhava',
    'chart.varga.D1',
    'chart.panchanga',
    'chart.yogas',
    'chart.ashtakavarga',
    'chart.arudha',
    'chart.dasha',
    'chart.kala',
  ]);

  const add = (...items: string[]) => items.forEach((item) => paths.add(item));

  for (const scope of pickQuestionScope(q)) {
    add(scope);
  }

  if (flags.includes('d9')) add('chart.varga.D9');
  if (flags.includes('dasha')) add('chart.dasha');
  if (flags.includes('transit')) add('chart.transit', 'chart.transits', 'chart.gochar');
  if (flags.includes('career')) add('chart.varga.D10');
  if (flags.includes('relationship')) add('chart.varga.D9');
  if (flags.includes('longevity')) add('chart.varga.D8', 'chart.varga.D30', 'chart.dasha', 'chart.bhava');

  if (/\b(career|job|profession|business|promotion|work|employment|salary|interview|office)\b/.test(q)) {
    add('chart.varga.D10', 'chart.varga.D1', 'chart.varga.D11');
  }
  if (/\b(marriage|relationship|partner|spouse|love|compatibility|romance|dating)\b/.test(q)) {
    add('chart.varga.D9', 'chart.varga.D1', 'chart.varga.D7');
  }
  if (/\b(wealth|money|income|finance|assets|property|investment|profits|revenue)\b/.test(q)) {
    add('chart.varga.D2', 'chart.varga.D11', 'chart.varga.D4');
  }
  if (/\b(education|study|studies|exam|exams|degree|college|school|learning|research)\b/.test(q)) {
    add('chart.varga.D24', 'chart.varga.D4', 'chart.varga.D1');
  }
  if (/\b(children|child|kids|pregnancy|pregnant)\b/.test(q)) {
    add('chart.varga.D7', 'chart.varga.D5', 'chart.varga.D1');
  }
  if (/\b(property|house|home|land|real estate|vehicle|car|asset)\b/.test(q)) {
    add('chart.varga.D4', 'chart.varga.D2', 'chart.varga.D11');
  }
  if (/\b(travel|foreign|abroad|visa|relocation|migration|move)\b/.test(q)) {
    add('chart.varga.D12', 'chart.varga.D9', 'chart.varga.D1');
  }
  if (/\b(spiritual|spirituality|moksha|meditation|religion|faith|guru)\b/.test(q)) {
    add('chart.varga.D20', 'chart.varga.D9', 'chart.varga.D1');
  }
  if (/\b(longevity|lifespan|life span|how long will i live|length of life|ayush|ayu|mrityu|death|end of life)\b/.test(q)) {
    add('chart.varga.D8', 'chart.varga.D30', 'chart.dasha', 'chart.varga.D1', 'chart.graha', 'chart.bhava');
  }
  if (/\b(communication|writing|speech|speaking|media|marketing|technology|coding|tech)\b/.test(q)) {
    add('chart.varga.D1', 'chart.varga.D10', 'chart.varga.D2');
  }

  add(...planetPathHints(question));

  if (mode === 'mini') {
    return [...paths].filter((path) => {
      if (
        path === 'chart.user' ||
        path === 'chart.graha' ||
        path === 'chart.lagna' ||
        path === 'chart.houses' ||
        path === 'chart.bhava' ||
        path === 'chart.panchanga' ||
        path === 'chart.yogas'
      ) {
        return true;
      }
      return path.startsWith('chart.varga.D1') || path.startsWith('chart.varga.D9');
    });
  }

  return [...paths];
}

function selectSections(rawPayload: unknown, question: string, flags: string[] = [], mode: AgentMode = 'pro'): SelectedSection[] {
  const selectedPaths = selectRelevantPaths(question, flags, mode);
  return selectedPaths
    .map((path) => ({ path, value: getByPath(rawPayload, path) }))
    .filter((section) => section.value !== undefined);
}

function normalizeRawPayload(rawPayload: unknown): unknown {
  if (!rawPayload || typeof rawPayload !== 'object') {
    return rawPayload;
  }
  return rawPayload;
}

async function loadCanonicalGrounding(state: AgentStateType): Promise<AgentUpdateType> {
  const store = getPostgresStore();
  const profilePath = `${COLLECTIONS.ragProfiles}/${state.ownerId}__${state.profileId}`;
  const profileDoc = await store.getDocument<RagProfileDocument>(profilePath);

  if (!profileDoc) {
    throw new Error(`No canonical profile snapshot found for ${state.profileId}. Regenerate the Kundli first.`);
  }

  const sourceDoc = await store.getDocument<RagApiSourceDocument>(
    `${COLLECTIONS.ragApiSources}/${profileDoc.data.latestSourceDocId}`
  );

  if (!sourceDoc) {
    throw new Error(`No canonical raw payload found for profile ${state.profileId}. Regenerate the Kundli first.`);
  }

  const rawPayload = normalizeRawPayload(sourceDoc.data.rawPayload);
  const atlas = summarizeChartAtlas(rawPayload);
  const selectedSections = selectSections(rawPayload, state.question, [], state.mode);
  const fallbackSections = selectedSections.length > 0 ? selectedSections : atlas.slice(0, 12).map((item) => ({ path: item.path, value: getByPath(rawPayload, item.path) }));
  const kundli = state.kundliInput ?? profileDoc.data.kundliInput;

  return {
    grounding: {
      ownerId: state.ownerId,
      profileId: state.profileId,
      sourceDocId: profileDoc.data.latestSourceDocId,
      chartVersion: profileDoc.data.chartVersion,
      kundliSignature: profileDoc.data.kundliSignature,
      kundli,
      requestKey: sourceDoc.data.requestKey,
      payloadHash: sourceDoc.data.payloadHash,
      referenceTimestamp: state.referenceTimestamp ?? Date.now(),
      referenceTimeSource: state.referenceTimeSource ?? 'server',
      rawPayload,
      selectedPaths: fallbackSections.map((section) => section.path),
      selectedSections: fallbackSections,
    },
    toolFindings: [{
      name: 'Reference time analyzer',
      status: 'ok',
      facts: [
        `Reference time source: ${state.referenceTimeSource ?? 'server'}.`,
        `Reference timestamp: ${state.referenceTimestamp ?? Date.now()}.`,
        `Reference ISO: ${new Date(state.referenceTimestamp ?? Date.now()).toISOString()}.`,
      ],
      evidencePaths: [],
    }, {
      name: 'Chart atlas',
      status: 'ok',
      facts: atlas.slice(0, 24).map((item) => `${item.path} → ${item.summary}${item.sampleKeys?.length ? ` | ${item.sampleKeys.join(', ')}` : ''}`),
      evidencePaths: atlas.slice(0, 24).map((item) => item.path),
      snippets: atlas.slice(0, 18).map((item) => `${item.path}: ${item.summary}`),
    }],
    analysisStages: appendStage(state, 'load_grounding', 'Analyzing canonical chart payload', `Selected ${fallbackSections.length} canonical section(s).`),
  };
}

function makePlacementToolFinding(rawPayload: unknown, question: string): ToolFinding {
  const q = question.toLowerCase();
  const scopePaths = [
    /\b(d9|navamsha|navamsa|marriage|relationship|partner|spouse|love|compatibility)\b/.test(q) ? 'chart.varga.D9' : null,
    /\b(d10|career|job|profession|business|work|promotion)\b/.test(q) ? 'chart.varga.D10' : null,
    /\b(d1|lagna|ascendant|rashi|sign|planet|graha|house|bhava)\b/.test(q) ? 'chart.varga.D1' : null,
  ].filter(Boolean) as string[];

  const chosen = chooseBestPath(rawPayload, scopePaths.length > 0 ? scopePaths : ['chart.varga.D1', 'chart.graha', 'chart.lagna']);
  if (!chosen) {
    return {
      name: 'Placement analyzer',
      status: 'unavailable',
      facts: ['No suitable placement scope was found in the canonical payload.'],
      evidencePaths: [],
      missing: scopePaths.length > 0 ? scopePaths : ['chart.varga.D1'],
    };
  }

  const value = chosen.value;
  const graha = getByPath(value, 'graha');
  const bhava = getByPath(value, 'bhava');
  const lagna = getByPath(value, 'lagna');

  const facts: string[] = [];
  facts.push(`Scope: ${chosen.path}`);
  if (lagna) facts.push(`Lagna/Ascendant: ${formatValue(lagna)}`);

  const grahaLines = summarizeKeyPlacements(graha, 18);
  if (grahaLines.length > 0) facts.push(...grahaLines);

  const bhavaLines = summarizeKeyPlacements(bhava, 12);
  if (bhavaLines.length > 0) facts.push(`Houses: ${bhavaLines.join(' | ')}`);

  return {
    name: 'Placement analyzer',
    status: facts.length > 1 ? 'ok' : 'partial',
    facts,
    evidencePaths: [chosen.path],
    snippets: [truncateText(JSON.stringify(value, null, 2), 2200)],
  };
}

function makePanchangaToolFinding(rawPayload: unknown): ToolFinding {
  const panchanga = chooseBestPath(rawPayload, ['chart.panchanga']);
  if (!panchanga) {
    return {
      name: 'Panchanga analyzer',
      status: 'unavailable',
      facts: ['Panchanga data is not present in the canonical payload.'],
      evidencePaths: [],
      missing: ['chart.panchanga'],
    };
  }

  const lines = summarizeKeyPlacements(panchanga.value, 12);
  return {
    name: 'Panchanga analyzer',
    status: lines.length > 0 ? 'ok' : 'partial',
    facts: lines.length > 0 ? lines : [formatValue(panchanga.value)],
    evidencePaths: [panchanga.path],
    snippets: [truncateText(JSON.stringify(panchanga.value, null, 2), 1800)],
  };
}

function makeFeatureToolFinding(rawPayload: unknown): ToolFinding {
  const featurePaths = ['chart.yogas', 'chart.ashtakavarga', 'chart.arudha'];
  const sections = featurePaths
    .map((path) => chooseBestPath(rawPayload, [path]))
    .filter((item): item is SelectedSection => Boolean(item));

  if (sections.length === 0) {
    return {
      name: 'Feature analyzer',
      status: 'unavailable',
      facts: ['Yogas, ashtakavarga, and arudha were not found in the canonical payload.'],
      evidencePaths: [],
      missing: featurePaths,
    };
  }

  const facts: string[] = [];
  for (const section of sections) {
    facts.push(`${section.path}: ${formatValue(section.value)}`);
  }

  return {
    name: 'Feature analyzer',
    status: 'ok',
    facts,
    evidencePaths: sections.map((section) => section.path),
    snippets: sections.map((section) => truncateText(JSON.stringify(section.value, null, 2), 1200)),
  };
}

function makeNakshatraLordToolFinding(rawPayload: unknown): ToolFinding {
  const grahaSection = getFirstPathValue(rawPayload, ['chart.graha', 'chart.varga.D1.graha', 'graha']);
  if (!grahaSection || !isPlainObject(grahaSection.value)) {
    return {
      name: 'Nakshatra/lord analyzer',
      status: 'unavailable',
      facts: ['Nakshatra-level planetary data is not available in the canonical payload.'],
      evidencePaths: [],
      missing: ['chart.graha'],
    };
  }

  const facts: string[] = [];
  for (const [code, data] of Object.entries(grahaSection.value as Record<string, unknown>)) {
    if (!isPlainObject(data)) continue;
    const nakshatra = data.nakshatra ?? data.nakshatra_name;
    const nakshatraLord = data.nakshatra_lord ?? data.star_lord;
    const sign = rashiName(data.rashi);
    const signLord = data.sign_lord ?? data.rashi_lord;
    if (nakshatra || nakshatraLord || signLord) {
      facts.push(
        `${PLANET_LABELS[code] ?? code}: ${nakshatra ? `nakshatra ${String(nakshatra)}` : 'nakshatra n/a'}${nakshatraLord ? ` (lord ${String(nakshatraLord)})` : ''}, sign ${sign}${signLord ? ` (lord ${String(signLord)})` : ''}.`
      );
    }
  }

  if (facts.length === 0) {
    return {
      name: 'Nakshatra/lord analyzer',
      status: 'partial',
      facts: ['Planet entries exist but nakshatra or lord fields are missing in this payload shape.'],
      evidencePaths: [grahaSection.path],
      snippets: [truncateText(JSON.stringify(grahaSection.value, null, 2), 1200)],
    };
  }

  return {
    name: 'Nakshatra/lord analyzer',
    status: 'ok',
    facts,
    evidencePaths: [grahaSection.path],
    snippets: [truncateText(JSON.stringify(grahaSection.value, null, 2), 1600)],
  };
}

function makeDrishtiDegreeToolFinding(rawPayload: unknown): ToolFinding {
  const grahaSection = getFirstPathValue(rawPayload, ['chart.graha', 'chart.varga.D1.graha', 'graha']);
  if (!grahaSection || !isPlainObject(grahaSection.value)) {
    return {
      name: 'Drishti/degree analyzer',
      status: 'unavailable',
      facts: ['Planetary drishti/degree details are not available in the canonical payload.'],
      evidencePaths: [],
      missing: ['chart.graha'],
    };
  }

  const graha = grahaSection.value as Record<string, unknown>;
  const facts: string[] = [];

  for (const [code, data] of Object.entries(graha)) {
    if (!isPlainObject(data)) continue;
    const degree = data.degree;
    const drishti = data.drishti ?? data.aspects;
    if (degree !== undefined) {
      facts.push(`${PLANET_LABELS[code] ?? code}: degree ${formatNumber(degree)}°, sign ${rashiName(data.rashi)}${data.house_number !== undefined ? `, house ${String(data.house_number)}` : ''}.`);
    }
    if (drishti) {
      facts.push(`${PLANET_LABELS[code] ?? code}: drishti/aspects ${formatValue(drishti)}.`);
    }
  }

  const entries = Object.entries(graha)
    .map(([code, data]) => ({
      code,
      longitude: isPlainObject(data) ? Number(data.longitude ?? data.degree) : Number.NaN,
    }))
    .filter((item) => Number.isFinite(item.longitude));

  entries.sort((a, b) => a.longitude - b.longitude);
  for (let i = 0; i < entries.length - 1; i += 1) {
    const diff = Math.abs(entries[i + 1].longitude - entries[i].longitude);
    if (diff <= 5) {
      facts.push(`Degree-closeness: ${(PLANET_LABELS[entries[i].code] ?? entries[i].code)} and ${(PLANET_LABELS[entries[i + 1].code] ?? entries[i + 1].code)} within ${diff.toFixed(2)}°.`);
    }
  }

  return {
    name: 'Drishti/degree analyzer',
    status: facts.length > 0 ? 'ok' : 'partial',
    facts: facts.length > 0 ? facts : ['Planetary entries found, but no drishti/degree-rich details could be extracted.'],
    evidencePaths: [grahaSection.path],
    snippets: [truncateText(JSON.stringify(grahaSection.value, null, 2), 1800)],
  };
}

function makeD9ToolFinding(rawPayload: unknown, question: string): ToolFinding {
  const d9 = getFirstPathValue(rawPayload, ['chart.varga.D9', 'varga.D9', 'chart.D9']);
  if (!d9) {
    return {
      name: 'D9 analyzer',
      status: 'unavailable',
      facts: ['D9/Navamsha structure is not present in the canonical payload.'],
      evidencePaths: [],
      missing: ['chart.varga.D9'],
    };
  }

  const value = d9.value as Record<string, unknown>;
  const lagna = getByPath(value, 'lagna.Lg') as Record<string, unknown> | undefined;
  const graha = getByPath(value, 'graha') as Record<string, Record<string, unknown>> | undefined;

  const facts: string[] = [];
  if (lagna) {
    facts.push(`D9 Lagna: ${rashiName(lagna.rashi)} at ${Number(lagna.degree ?? 0).toFixed(2)}°`);
  }

  const keyPlanets = ['Su', 'Mo', 'Ma', 'Me', 'Ju', 'Ve', 'Sa', 'Ra', 'Ke'];
  if (graha && typeof graha === 'object') {
    for (const code of keyPlanets) {
      const p = graha[code];
      if (!p) continue;
      const label = PLANET_LABELS[code] ?? code;
      const house = p.house_number !== undefined ? `, house ${p.house_number}` : '';
      facts.push(`${label} in D9: ${rashiName(p.rashi)}${house}`);
    }
  }

  if (/\b(marriage|relationship|partner|spouse|love|compatibility)\b/i.test(question)) {
    const ve = graha?.Ve;
    const ju = graha?.Ju;
    if (ve) facts.push(`Marriage indicator Venus (D9): ${rashiName(ve.rashi)}, house ${ve.house_number ?? 'n/a'}.`);
    if (ju) facts.push(`Guidance indicator Jupiter (D9): ${rashiName(ju.rashi)}, house ${ju.house_number ?? 'n/a'}.`);
  }

  return {
    name: 'D9 analyzer',
    status: facts.length > 0 ? 'ok' : 'partial',
    facts,
    evidencePaths: [d9.path],
    snippets: [truncateText(JSON.stringify(d9.value, null, 2), 2200)],
  };
}

type ParsedPeriod = {
  key: string;
  start?: string;
  end?: string;
};

function collectPeriods(value: unknown, out: ParsedPeriod[], depth = 0): void {
  if (depth > 6 || value === null || value === undefined) return;
  if (Array.isArray(value)) {
    for (const item of value) collectPeriods(item, out, depth + 1);
    return;
  }
  if (typeof value !== 'object') return;

  const obj = value as Record<string, unknown>;
  const key = String(obj.key ?? obj.name ?? obj.graha ?? obj.planet ?? obj.lord ?? '').trim();
  const start = typeof obj.start === 'string' ? obj.start : undefined;
  const end = typeof obj.end === 'string' ? obj.end : undefined;

  if (key && (start || end)) {
    out.push({ key, start, end });
  }

  for (const nested of Object.values(obj)) {
    if (nested && typeof nested === 'object') collectPeriods(nested, out, depth + 1);
  }
}

function toDate(value: string | undefined): Date | null {
  if (!value) return null;
  const dt = new Date(value.replace(' ', 'T'));
  return Number.isNaN(dt.getTime()) ? null : dt;
}

type ActiveDashaPeriod = {
  code: string;
  key: string;
  type: string;
  level: number;
  start: string;
  end: string;
  remainingMs: number;
  elapsedMs: number;
};

function resolveActiveDashaChain(periods: unknown, at: Date, level = 1): ActiveDashaPeriod[] {
  if (!isPlainObject(periods)) return [];

  const entries = Object.entries(periods)
    .map(([code, node]) => [code, node] as const)
    .filter(([, node]) => isPlainObject(node) && typeof node.start === 'string' && typeof node.end === 'string')
    .sort((a, b) => {
      const aStart = toDate((a[1] as Record<string, unknown>).start as string | undefined)?.getTime() ?? 0;
      const bStart = toDate((b[1] as Record<string, unknown>).start as string | undefined)?.getTime() ?? 0;
      return aStart - bStart;
    });

  for (const [code, nodeUnknown] of entries) {
    const node = nodeUnknown as Record<string, unknown>;
    const start = toDate(node.start as string | undefined);
    const end = toDate(node.end as string | undefined);
    if (!start || !end) continue;
    if (!(start <= at && at <= end)) continue;

    const current: ActiveDashaPeriod = {
      code,
      key: String(node.key ?? code),
      type: String(node.type ?? 'dasha'),
      level,
      start: node.start as string,
      end: node.end as string,
      remainingMs: Math.max(0, end.getTime() - at.getTime()),
      elapsedMs: Math.max(0, at.getTime() - start.getTime()),
    };

    const nested = resolveActiveDashaChain(node.periods, at, level + 1);
    return [current, ...nested];
  }

  return [];
}

function formatDurationMs(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return 'n/a';
  const totalMinutes = Math.floor(ms / 60000);
  const days = Math.floor(totalMinutes / (24 * 60));
  const hours = Math.floor((totalMinutes % (24 * 60)) / 60);
  const minutes = totalMinutes % 60;
  return `${days}d ${hours}h ${minutes}m`;
}

function makeReferenceTimeToolFinding(referenceTimestamp: number, source: TimeSource): ToolFinding {
  const timestamp = Number.isFinite(referenceTimestamp) ? referenceTimestamp : Date.now();
  const dt = new Date(timestamp);

  return {
    name: 'Reference time analyzer',
    status: Number.isNaN(dt.getTime()) ? 'partial' : 'ok',
    facts: [
      `Reference time source: ${source}.`,
      `Reference timestamp: ${timestamp}.`,
      `Reference ISO: ${Number.isNaN(dt.getTime()) ? 'invalid' : dt.toISOString()}.`,
    ],
    evidencePaths: [],
  };
}

function resolveAnalysisTimestamp(referenceTimestamp: number, intent: QuestionIntent | null): number {
  const baseTimestamp = Number.isFinite(referenceTimestamp) ? referenceTimestamp : Date.now();
  if (!intent || intent.timeDirection === 'present' || intent.timeValue === undefined || !intent.timeUnit) {
    return baseTimestamp;
  }

  const date = new Date(baseTimestamp);
  const amount = intent.timeValue;
  const multiplier = intent.timeDirection === 'past' ? -1 : 1;

  switch (intent.timeUnit) {
    case 'day':
      date.setDate(date.getDate() + amount * multiplier);
      break;
    case 'week':
      date.setDate(date.getDate() + amount * 7 * multiplier);
      break;
    case 'month':
      date.setMonth(date.getMonth() + amount * multiplier);
      break;
    case 'year':
      date.setFullYear(date.getFullYear() + amount * multiplier);
      break;
  }

  return date.getTime();
}

function makeCareerToolFinding(rawPayload: unknown, question: string, analysisTimestamp: number): ToolFinding {
  const d10 = getFirstPathValue(rawPayload, ['chart.varga.D10', 'varga.D10']);
  const d1 = getFirstPathValue(rawPayload, ['chart.varga.D1', 'varga.D1']);
  const dashaSection = getFirstPathValue(rawPayload, ['chart.dasha', 'dasha']);

  if (!d10 && !d1 && !dashaSection) {
    return {
      name: 'Career analyzer',
      status: 'unavailable',
      facts: ['Career-focused charts (D10/D1/dasha) are not present in the canonical payload.'],
      evidencePaths: [],
      missing: ['chart.varga.D10', 'chart.varga.D1', 'chart.dasha'],
    };
  }

  const facts: string[] = [];

  if (analysisTimestamp) {
    facts.push(`Career time window evaluated at ${new Date(analysisTimestamp).toISOString()}.`);
  }

  if (d10) {
    const d10Value = d10.value as Record<string, unknown>;
    const lagna = getByPath(d10Value, 'lagna.Lg') ?? getByPath(d10Value, 'lagna');
    const graha = getByPath(d10Value, 'graha');
    const bhava = getByPath(d10Value, 'bhava');

    if (lagna) facts.push(`D10 Lagna: ${formatPlacement(lagna)}.`);

    const keyCareerPlanets = ['Su', 'Me', 'Ju', 'Sa', 'Ra'];
    if (isPlainObject(graha)) {
      for (const code of keyCareerPlanets) {
        const planet = (graha as Record<string, unknown>)[code];
        if (!planet) continue;
        const label = PLANET_LABELS[code] ?? code;
        facts.push(`${label} in D10: ${formatPlacement(planet)}.`);
      }
    }

    if (isPlainObject(bhava)) {
      const tenthHouse = (bhava as Record<string, unknown>)['10'];
      const sixthHouse = (bhava as Record<string, unknown>)['6'];
      const eleventhHouse = (bhava as Record<string, unknown>)['11'];
      if (tenthHouse) facts.push(`10th house focus in D10: ${formatPlacement(tenthHouse)}.`);
      if (sixthHouse) facts.push(`6th house work/service focus in D10: ${formatPlacement(sixthHouse)}.`);
      if (eleventhHouse) facts.push(`11th house gains/networking focus in D10: ${formatPlacement(eleventhHouse)}.`);
    }
  }

  if (d1) {
    const d1Value = d1.value as Record<string, unknown>;
    const sun = getByPath(d1Value, 'graha.Su');
    const mercury = getByPath(d1Value, 'graha.Me');
    const saturn = getByPath(d1Value, 'graha.Sa');
    const jupiter = getByPath(d1Value, 'graha.Ju');
    if (sun) facts.push(`Sun in D1: ${formatPlacement(sun)}.`);
    if (mercury) facts.push(`Mercury in D1: ${formatPlacement(mercury)}.`);
    if (saturn) facts.push(`Saturn in D1: ${formatPlacement(saturn)}.`);
    if (jupiter) facts.push(`Jupiter in D1: ${formatPlacement(jupiter)}.`);
  }

  if (dashaSection) {
    facts.push(`Dasha timeline available for ${question.toLowerCase().includes('past') ? 'retrospective' : 'career timing'} analysis.`);
  }

  return {
    name: 'Career analyzer',
    status: 'ok',
    facts,
    evidencePaths: [
      ...(d10 ? [d10.path] : []),
      ...(d1 ? [d1.path] : []),
      ...(dashaSection ? [dashaSection.path] : []),
    ],
    snippets: [
      ...(d10 ? [truncateText(JSON.stringify(d10.value, null, 2), 2000)] : []),
      ...(dashaSection ? [truncateText(JSON.stringify(dashaSection.value, null, 2), 1600)] : []),
    ],
  };
}

function makeLongevityToolFinding(rawPayload: unknown, referenceTimestamp: number): ToolFinding {
  const d1Section = getFirstPathValue(rawPayload, ['chart.varga.D1', 'varga.D1']);
  const d8Section = getFirstPathValue(rawPayload, ['chart.varga.D8', 'varga.D8']);
  const d30Section = getFirstPathValue(rawPayload, ['chart.varga.D30', 'varga.D30']);
  const dashaSection = getFirstPathValue(rawPayload, ['chart.dasha', 'dasha']);

  if (!d1Section && !d8Section && !dashaSection) {
    return {
      name: 'Longevity analyzer',
      status: 'unavailable',
      facts: ['Longevity-relevant sections (D1/D8/dasha) are not present in the canonical payload.'],
      evidencePaths: [],
      missing: ['chart.varga.D1', 'chart.varga.D8', 'chart.dasha'],
    };
  }

  const facts: string[] = [];
  const evidencePaths: string[] = [];

  if (d1Section) {
    evidencePaths.push(d1Section.path);
    const d1 = d1Section.value as Record<string, unknown>;
    const lagna = (getByPath(d1, 'lagna.Lg') ?? getByPath(d1, 'lagna')) as Record<string, unknown> | undefined;
    const bhava = getByPath(d1, 'bhava') as Record<string, unknown> | undefined;
    const graha = getByPath(d1, 'graha') as Record<string, unknown> | undefined;

    if (lagna) {
      facts.push(`D1 Lagna vitality baseline: ${formatPlacement(lagna)}.`);
      const lagnaRashi = Number(lagna.rashi);
      const lagnaLord = RASHI_LORDS[lagnaRashi];
      if (lagnaLord && graha?.[lagnaLord]) {
        facts.push(`Lagna lord (${PLANET_LABELS[lagnaLord]}) in D1: ${formatPlacement(graha[lagnaLord])}.`);
      }
    }

    const eighthHouse = bhava?.['8'];
    const secondHouse = bhava?.['2'];
    const seventhHouse = bhava?.['7'];
    if (eighthHouse) {
      facts.push(`D1 8th-house (longevity axis): ${formatPlacement(eighthHouse)}.`);
      const eighthRashi = Number((eighthHouse as Record<string, unknown>).rashi);
      const eighthLord = RASHI_LORDS[eighthRashi];
      if (eighthLord && graha?.[eighthLord]) {
        facts.push(`8th-house lord (${PLANET_LABELS[eighthLord]}) in D1: ${formatPlacement(graha[eighthLord])}.`);
      }
    }
    if (secondHouse) facts.push(`D1 maraka house (2nd): ${formatPlacement(secondHouse)}.`);
    if (seventhHouse) facts.push(`D1 maraka house (7th): ${formatPlacement(seventhHouse)}.`);

    if (graha?.Sa) facts.push(`Saturn in D1: ${formatPlacement(graha.Sa)}.`);
    if (graha?.Ra) facts.push(`Rahu in D1: ${formatPlacement(graha.Ra)}.`);
    if (graha?.Ke) facts.push(`Ketu in D1: ${formatPlacement(graha.Ke)}.`);
  }

  if (d8Section) {
    evidencePaths.push(d8Section.path);
    const d8 = d8Section.value as Record<string, unknown>;
    const d8Lagna = getByPath(d8, 'lagna.Lg') ?? getByPath(d8, 'lagna');
    const d8Graha = getByPath(d8, 'graha') as Record<string, unknown> | undefined;
    if (d8Lagna) facts.push(`D8 Lagna: ${formatPlacement(d8Lagna)}.`);
    if (d8Graha?.Sa) facts.push(`Saturn in D8: ${formatPlacement(d8Graha.Sa)}.`);
    if (d8Graha?.Ma) facts.push(`Mars in D8: ${formatPlacement(d8Graha.Ma)}.`);
    if (d8Graha?.Ra) facts.push(`Rahu in D8: ${formatPlacement(d8Graha.Ra)}.`);
    if (d8Graha?.Ke) facts.push(`Ketu in D8: ${formatPlacement(d8Graha.Ke)}.`);
  }

  if (d30Section) {
    evidencePaths.push(d30Section.path);
    const d30 = d30Section.value as Record<string, unknown>;
    const d30Lagna = getByPath(d30, 'lagna.Lg') ?? getByPath(d30, 'lagna');
    if (d30Lagna) facts.push(`D30 stress/suffering profile Lagna: ${formatPlacement(d30Lagna)}.`);
  }

  if (dashaSection) {
    evidencePaths.push(dashaSection.path);
    const dashaFinding = makeDashaToolFinding(rawPayload, referenceTimestamp);
    facts.push(...dashaFinding.facts.slice(0, 4));
  }

  if (facts.length > 0) {
    facts.push('Longevity note: provide risk profile and health trajectory windows, not exact death timing.');
  }

  return {
    name: 'Longevity analyzer',
    status: facts.length > 0 ? 'ok' : 'partial',
    facts: facts.length > 0 ? facts : ['Longevity sections detected but could not extract structured markers.'],
    evidencePaths,
    snippets: [
      ...(d1Section ? [truncateText(JSON.stringify(d1Section.value, null, 2), 1800)] : []),
      ...(d8Section ? [truncateText(JSON.stringify(d8Section.value, null, 2), 1800)] : []),
      ...(dashaSection ? [truncateText(JSON.stringify(dashaSection.value, null, 2), 1400)] : []),
    ],
  };
}

function makeDashaToolFinding(rawPayload: unknown, referenceTimestamp: number): ToolFinding {
  const dashaSection = getFirstPathValue(rawPayload, ['chart.dasha', 'dasha']);
  if (!dashaSection) {
    return {
      name: 'Dasha analyzer',
      status: 'unavailable',
      facts: ['Dasha structure is not present in the canonical payload.'],
      evidencePaths: [],
      missing: ['chart.dasha'],
    };
  }

  const dasha = dashaSection.value as Record<string, unknown>;
  const referenceDate = new Date(Number.isFinite(referenceTimestamp) ? referenceTimestamp : Date.now());
  const facts: string[] = [];

  if (dasha.type) facts.push(`Dasha type: ${String(dasha.type)}.`);
  if (dasha.start && dasha.end) facts.push(`Overall window: ${String(dasha.start)} to ${String(dasha.end)}.`);

  if (typeof dasha.duration === 'number') {
    const years = dasha.duration / (365.2425 * 24 * 3600);
    facts.push(`Overall duration: ${years.toFixed(2)} years.`);
  }

  facts.push(`Reference time: ${referenceDate.toISOString()}.`);

  const activeChain = resolveActiveDashaChain(dasha.periods, referenceDate);

  if (activeChain.length > 0) {
    facts.push(`Active chain: ${activeChain.map((item) => `${item.type}:${item.key} [${item.start} → ${item.end}]`).join(' > ')}.`);
    const deepest = activeChain[activeChain.length - 1];
    facts.push(`Current period: ${deepest.type} ${deepest.key} (level ${deepest.level}) with ${formatDurationMs(deepest.remainingMs)} remaining.`);
    facts.push(
      `Period detail: ${activeChain
        .map((item) => `${item.type} ${item.key}: elapsed ${formatDurationMs(item.elapsedMs)}, remaining ${formatDurationMs(item.remainingMs)}`)
        .join(' | ')}.`
    );
  } else {
    const periods: ParsedPeriod[] = [];
    collectPeriods(dasha.periods, periods);
    collectPeriods(dasha, periods);

    const uniq = periods.filter((item, index, arr) => arr.findIndex((x) => `${x.key}|${x.start}|${x.end}` === `${item.key}|${item.start}|${item.end}`) === index);
    if (uniq.length > 0) {
      const preview = uniq.slice(0, 3).map((item) => `${item.key} (${item.start ?? '?'} → ${item.end ?? '?'})`);
      facts.push(`Available sub-periods (sample): ${preview.join('; ')}`);
    } else {
      facts.push('Detailed dasha sub-period timeline is not present (only top-level dasha metadata available).');
    }
  }

  return {
    name: 'Dasha analyzer',
    status: activeChain.length > 0 ? 'ok' : 'partial',
    facts,
    evidencePaths: [dashaSection.path],
    snippets: [truncateText(JSON.stringify(dashaSection.value, null, 2), 2600)],
  };
}

function makeTransitToolFinding(rawPayload: unknown): ToolFinding {
  const transitSection = getFirstPathValue(rawPayload, [
    'chart.transit',
    'chart.transits',
    'chart.gochar',
    'transit',
    'transits',
    'gochar',
  ]);

  const natalSun = getFirstPathValue(rawPayload, ['chart.graha.Su', 'graha.Su']);

  if (!transitSection) {
    const facts = [
      'Canonical payload has no dedicated live transit/gochar block, so exact current transit cannot be computed from this JSON alone.',
    ];

    if (natalSun) {
      const sun = natalSun.value as Record<string, unknown>;
      facts.push(`Natal Sun reference: ${rashiName(sun.rashi)}${sun.house_number ? `, house ${sun.house_number}` : ''}.`);
    }

    facts.push('To answer current transit accurately, store a transit payload (gochar/transit endpoint) into canonical rawPayload.');

    return {
      name: 'Transit analyzer',
      status: 'partial',
      facts,
      evidencePaths: natalSun ? [natalSun.path] : [],
      missing: ['chart.transit|chart.transits|chart.gochar'],
    };
  }

  const transit = transitSection.value as Record<string, unknown>;
  const sun = (transit.Su ?? transit.Sun ?? transit.sun) as Record<string, unknown> | undefined;
  const facts = ['Transit/gochar block detected in canonical payload.'];

  if (sun) {
    const degree = sun.degree !== undefined ? ` at ${Number(sun.degree).toFixed(2)}°` : '';
    const house = sun.house_number !== undefined ? `, house ${sun.house_number}` : '';
    facts.push(`Current transit Sun: ${rashiName(sun.rashi)}${house}${degree}.`);
  } else {
    facts.push('Transit block exists, but Sun transit entry is missing in that structure.');
  }

  return {
    name: 'Transit analyzer',
    status: sun ? 'ok' : 'partial',
    facts,
    evidencePaths: [transitSection.path],
    snippets: [truncateText(JSON.stringify(transitSection.value, null, 2), 2000)],
  };
}

function makeGeneralToolFinding(rawPayload: unknown, question: string, flags: string[]): ToolFinding {
  const sections = selectSections(rawPayload, question, flags);
  if (sections.length === 0) {
    return {
      name: 'General grounding analyzer',
      status: 'unavailable',
      facts: ['No relevant canonical sections matched this question in the current payload.'],
      evidencePaths: [],
    };
  }

  const snippets = sections
    .slice(0, 8)
    .map((section) => `PATH: ${section.path}\nVALUE: ${formatSelectedSectionForPrompt(section)}`);

  return {
    name: 'General grounding analyzer',
    status: 'ok',
    facts: [`Matched ${sections.length} canonical section(s) relevant to the question.`],
    evidencePaths: sections.map((item) => item.path),
    snippets,
  };
}

function makeArudhaToolFinding(rawPayload: unknown): ToolFinding {
  const arudhaSection = getFirstPathValue(rawPayload, ['chart.arudha', 'arudha']);
  if (!arudhaSection) {
    return {
      name: 'Arudha analyzer',
      status: 'unavailable',
      facts: ['Arudha data is not present in the canonical payload.'],
      evidencePaths: [],
      missing: ['chart.arudha'],
    };
  }

  const facts = summarizeArudhaSection(arudhaSection.value, 16);
  if (facts.length === 0) {
    facts.push('Arudha section is present but no readable entries were found.');
  }

  return {
    name: 'Arudha analyzer',
    status: 'ok',
    facts,
    evidencePaths: [arudhaSection.path],
    snippets: [
      truncateText(
        summarizeArudhaSection(arudhaSection.value, 24)
          .map((line) => `- ${line}`)
          .join('\n'),
        1800
      ),
    ],
  };
}

async function classifyIntentNode(state: AgentStateType): Promise<AgentUpdateType> {
  const intent = classifyQuestionIntent(state.question);
  return {
    intent,
    analysisStages: appendStage(state, 'classify_intent', 'Classifying user question intent', `Primary=${intent.primary}; flags=${intent.flags.join(',') || 'none'}`),
  };
}

async function planExecutionNode(state: AgentStateType): Promise<AgentUpdateType> {
  const intent = state.intent ?? classifyQuestionIntent(state.question);
  const executionPlan = buildDynamicExecutionPlan(state.question, intent, state.mode);
  return {
    executionPlan,
    analysisStages: appendStage(
      state,
      'plan_execution',
      'Planning dynamic analysis path',
      `Family=${executionPlan.family}; layers=${executionPlan.chartLayers.join(',')}; parallelBatches=${executionPlan.parallelBatches.length}`
    ),
  };
}

async function runSpecializedToolsNode(state: AgentStateType): Promise<AgentUpdateType> {
  const grounding = state.grounding;
  if (!grounding) throw new Error('Grounding context missing before specialized tools stage.');
  const intent = state.intent ?? classifyQuestionIntent(state.question);
  const executionPlan = state.executionPlan ?? buildDynamicExecutionPlan(state.question, intent, state.mode);
  const analysisTimestamp = resolveAnalysisTimestamp(grounding.referenceTimestamp, intent);
  const isMini = state.mode === 'mini';

  const includeTransit = executionPlan.includeTransit;
  const includeD9 = executionPlan.chartLayers.includes('D9') || intent.flags.includes('d9') || intent.primary === 'd9';
  const includeDasha = executionPlan.includeDasha;
  const includePlacement = intent.flags.includes('relationship') || intent.flags.includes('career') || intent.flags.includes('health') || intent.flags.includes('finance') || executionPlan.family !== 'general';
  const includeCareer = executionPlan.includeCareer || intent.topics.includes('career') || intent.flags.includes('career') || intent.flags.includes('career_timing');
  const includeLongevity = executionPlan.family === 'longevity' || intent.flags.includes('longevity');

  const baseTasks: Array<Promise<ToolFinding | null>> = [
    Promise.resolve(makeReferenceTimeToolFinding(grounding.referenceTimestamp, grounding.referenceTimeSource)),
    Promise.resolve(makeAtlasToolFinding(grounding.rawPayload)),
    Promise.resolve(makeVargaToolFinding(grounding.rawPayload, state.question, state.mode)),
  ];

  const domainTasks: Array<Promise<ToolFinding | null>> = [
    /\b(arudha|aruda)\b/i.test(state.question) && !isMini ? Promise.resolve(makeArudhaToolFinding(grounding.rawPayload)) : Promise.resolve(null),
    includeD9 ? Promise.resolve(makeD9ToolFinding(grounding.rawPayload, state.question)) : Promise.resolve(null),
    includeDasha && !isMini ? Promise.resolve(makeDashaToolFinding(grounding.rawPayload, analysisTimestamp)) : Promise.resolve(null),
    includeTransit
      && !isMini
      ? buildTransitToolFinding({ kundli: grounding.kundli, question: state.question, referenceTimestamp: grounding.referenceTimestamp })
      : Promise.resolve(null),
    includeCareer && !isMini ? Promise.resolve(makeCareerToolFinding(grounding.rawPayload, state.question, analysisTimestamp)) : Promise.resolve(null),
    includeLongevity && !isMini ? Promise.resolve(makeLongevityToolFinding(grounding.rawPayload, grounding.referenceTimestamp)) : Promise.resolve(null),
    includePlacement ? Promise.resolve(makePlacementToolFinding(grounding.rawPayload, state.question)) : Promise.resolve(null),
  ];

  const microSignalTasks: Array<Promise<ToolFinding | null>> = [
    executionPlan.includeMicroSignals.includes('nakshatra') || executionPlan.includeMicroSignals.includes('nakshatra_lord') || executionPlan.includeMicroSignals.includes('sign_lord')
      ? Promise.resolve(makeNakshatraLordToolFinding(grounding.rawPayload))
      : Promise.resolve(null),
    executionPlan.includeMicroSignals.includes('drishti') || executionPlan.includeMicroSignals.includes('degree')
      ? Promise.resolve(makeDrishtiDegreeToolFinding(grounding.rawPayload))
      : Promise.resolve(null),
  ];

  const batch1 = (await Promise.all(baseTasks)).filter((item): item is ToolFinding => Boolean(item));
  const batch2 = (await Promise.all(domainTasks)).filter((item): item is ToolFinding => Boolean(item));
  const batch3 = (await Promise.all(microSignalTasks)).filter((item): item is ToolFinding => Boolean(item));
  const findings = mergeFindings(state.toolFindings ?? [], [...batch1, ...batch2, ...batch3]);

  return {
    toolFindings: findings,
    analysisStages: appendStage(
      state,
      'run_specialized_tools',
      'Running specialized analyzers',
      `Completed ${executionPlan.parallelBatches.length} parallel batch group(s); findings=${findings.length}`
    ),
  };
}

async function runGeneralToolsNode(state: AgentStateType): Promise<AgentUpdateType> {
  const grounding = state.grounding;
  if (!grounding) throw new Error('Grounding context missing before general tools stage.');
  const intent = state.intent ?? classifyQuestionIntent(state.question);
  const executionPlan = state.executionPlan ?? buildDynamicExecutionPlan(state.question, intent, state.mode);
  const analysisTimestamp = resolveAnalysisTimestamp(grounding.referenceTimestamp, intent);
  const isMini = state.mode === 'mini';

  const includePanchanga = /\b(panchanga|tithi|nakshatra|karana|yoga)\b/i.test(state.question);
  const includeFeature = /\b(yoga|yogas|ashtakavarga|arudha|aruda)\b/i.test(state.question) && !isMini;
  const includeDasha = executionPlan.includeDasha && !intent.flags.includes('dasha') && !isMini;
  const includeTransit = executionPlan.includeTransit && !intent.flags.includes('transit') && !isMini;
  const includeCareer = (executionPlan.includeCareer || intent.topics.includes('career') || /\b(career|job|profession|business|promotion|work|employment|salary|interview)\b/i.test(state.question)) && !isMini;
  const includeLongevity = (executionPlan.family === 'longevity' || intent.flags.includes('longevity') || /\b(longevity|lifespan|life span|how long will i live|length of life|ayush|ayu|mrityu|death|end of life)\b/i.test(state.question)) && !isMini;

  const taskList: Array<Promise<ToolFinding | null>> = [
    Promise.resolve(makeReferenceTimeToolFinding(grounding.referenceTimestamp, grounding.referenceTimeSource)),
    Promise.resolve(makeGeneralToolFinding(grounding.rawPayload, state.question, intent.flags)),
    Promise.resolve(makeVargaToolFinding(grounding.rawPayload, state.question, state.mode)),
    /\b(arudha|aruda)\b/i.test(state.question) && !isMini ? Promise.resolve(makeArudhaToolFinding(grounding.rawPayload)) : Promise.resolve(null),
    Promise.resolve(makePlacementToolFinding(grounding.rawPayload, state.question)),
    includePanchanga ? Promise.resolve(makePanchangaToolFinding(grounding.rawPayload)) : Promise.resolve(null),
    includeFeature ? Promise.resolve(makeFeatureToolFinding(grounding.rawPayload)) : Promise.resolve(null),
    includeDasha ? Promise.resolve(makeDashaToolFinding(grounding.rawPayload, analysisTimestamp)) : Promise.resolve(null),
    includeTransit
      ? buildTransitToolFinding({ kundli: grounding.kundli, question: state.question, referenceTimestamp: grounding.referenceTimestamp })
      : Promise.resolve(null),
    includeCareer ? Promise.resolve(makeCareerToolFinding(grounding.rawPayload, state.question, analysisTimestamp)) : Promise.resolve(null),
    includeLongevity ? Promise.resolve(makeLongevityToolFinding(grounding.rawPayload, grounding.referenceTimestamp)) : Promise.resolve(null),
    executionPlan.includeMicroSignals.includes('nakshatra') || executionPlan.includeMicroSignals.includes('nakshatra_lord') || executionPlan.includeMicroSignals.includes('sign_lord')
      ? Promise.resolve(makeNakshatraLordToolFinding(grounding.rawPayload))
      : Promise.resolve(null),
    executionPlan.includeMicroSignals.includes('drishti') || executionPlan.includeMicroSignals.includes('degree')
      ? Promise.resolve(makeDrishtiDegreeToolFinding(grounding.rawPayload))
      : Promise.resolve(null),
  ];

  const findings = mergeFindings(
    state.toolFindings ?? [],
    (await Promise.all(taskList)).filter((item): item is ToolFinding => Boolean(item))
  );

  return {
    toolFindings: findings,
    analysisStages: appendStage(state, 'run_general_tools', 'Running general analyzers', `Findings total=${findings.length}`),
  };
}

async function evaluateCoverageNode(state: AgentStateType): Promise<AgentUpdateType> {
  const coverageGaps = determineCoverageGaps(state);

  return {
    coverageGaps,
    analysisStages: appendStage(
      state,
      'evaluate_coverage',
      'Evaluating tool coverage quality',
      coverageGaps.length > 0
        ? `Coverage gaps: ${coverageGaps.join(', ')}`
        : 'Coverage is sufficient for answer generation.'
    ),
  };
}

async function refineToolsNode(state: AgentStateType): Promise<AgentUpdateType> {
  const grounding = state.grounding;
  if (!grounding) throw new Error('Grounding context missing before refine-tools stage.');

  const gaps = state.coverageGaps ?? [];
  if (gaps.length === 0) {
    return {
      toolIteration: Math.min((state.toolIteration ?? 0) + 1, state.maxToolIterations ?? 2),
      analysisStages: appendStage(state, 'refine_tools', 'Refining missing analyzers', 'No gaps found during refinement pass.'),
    };
  }

  const intent = state.intent ?? classifyQuestionIntent(state.question);
  const analysisTimestamp = resolveAnalysisTimestamp(grounding.referenceTimestamp, intent);
  const tasks: Array<Promise<ToolFinding | null>> = [];

  for (const gap of gaps) {
    switch (gap) {
      case 'varga':
        tasks.push(Promise.resolve(makeVargaToolFinding(grounding.rawPayload, state.question, state.mode)));
        break;
      case 'd9':
        tasks.push(Promise.resolve(makeD9ToolFinding(grounding.rawPayload, state.question)));
        break;
      case 'dasha':
        tasks.push(Promise.resolve(makeDashaToolFinding(grounding.rawPayload, analysisTimestamp)));
        break;
      case 'transit':
        tasks.push(
          buildTransitToolFinding({
            kundli: grounding.kundli,
            question: state.question,
            referenceTimestamp: grounding.referenceTimestamp,
          })
        );
        break;
      case 'career':
        tasks.push(Promise.resolve(makeCareerToolFinding(grounding.rawPayload, state.question, analysisTimestamp)));
        break;
      case 'longevity':
        tasks.push(Promise.resolve(makeLongevityToolFinding(grounding.rawPayload, grounding.referenceTimestamp)));
        break;
      default:
        break;
    }
  }

  // Always include one broad grounding pass in refinement to capture missed paths.
  tasks.push(Promise.resolve(makeGeneralToolFinding(grounding.rawPayload, state.question, intent.flags)));

  const retryFindings = (await Promise.all(tasks)).filter((item): item is ToolFinding => Boolean(item));
  const merged = mergeFindings(state.toolFindings ?? [], retryFindings);
  const nextIteration = Math.min((state.toolIteration ?? 0) + 1, state.maxToolIterations ?? 2);

  return {
    toolFindings: merged,
    toolIteration: nextIteration,
    analysisStages: appendStage(
      state,
      'refine_tools',
      'Refining missing analyzers',
      `Iteration ${nextIteration}/${state.maxToolIterations ?? 2}; retried ${gaps.length} gap group(s).`
    ),
  };
}

function routeAfterCoverage(state: AgentStateType): 'refine_tools' | 'build_prompt' {
  const gapCount = (state.coverageGaps ?? []).length;
  const iteration = state.toolIteration ?? 0;
  const maxIterations = state.maxToolIterations ?? 2;

  if (gapCount > 0 && iteration < maxIterations) {
    return 'refine_tools';
  }

  return 'build_prompt';
}

function buildPrompt(state: AgentStateType): string {
  const grounding = state.grounding;
  if (!grounding) throw new Error('Grounding context missing while building prompt.');
  const executionPlan = state.executionPlan;

  const findings = state.toolFindings ?? [];
  const conversationContext = (state.conversationContext ?? [])
    .map((item) => String(item).trim())
    .filter(Boolean)
    .slice(-8);

  const findingBlock = findings
    .map((finding) => {
      const facts = finding.facts.map((fact) => `- ${fact}`).join('\n');
      const paths = finding.evidencePaths.length ? `Evidence paths: ${finding.evidencePaths.join(', ')}` : 'Evidence paths: none';
      const missing = finding.missing?.length ? `Missing: ${finding.missing.join(', ')}` : '';
      return `TOOL: ${finding.name} [${finding.status}]\n${facts}\n${paths}${missing ? `\n${missing}` : ''}`;
    })
    .join('\n\n');

  const MAX_SNIPPET_CHARS = 9000;
  let budget = MAX_SNIPPET_CHARS;
  let omitted = 0;

  const snippets: string[] = [];
  for (const finding of findings) {
    for (const snippet of finding.snippets ?? []) {
      if (budget <= 0) {
        omitted += 1;
        continue;
      }
      const piece = truncateText(snippet, Math.min(1200, budget));
      snippets.push(piece);
      budget -= piece.length + 2;
    }
  }

  if (snippets.length === 0) {
    for (const section of grounding.selectedSections.slice(0, 6)) {
      const piece = `PATH: ${section.path}\nVALUE: ${truncateText(formatValue(section.value), 900)}`;
      if (budget <= 0) {
        omitted += 1;
        continue;
      }
      snippets.push(piece);
      budget -= piece.length + 2;
    }
  }

  if (omitted > 0) {
    snippets.push(`[${omitted} additional snippet(s) omitted due to prompt size budget]`);
  }

  return [
    'You are a Vedic astrology assistant grounded in canonical JSON payload data.',
    'Use the tool findings first; they are deterministic extracts from the JSON blob.',
    'Never invent chart facts. If dasha/transit details are missing in payload, explicitly say so.',
    'For compound questions, separate the topic, the time window, and the chart layer before answering.',
    'For varga requests, prefer the specific Dxx chart named by the user and fall back to D1 only when needed.',
    'For dasha requests, report the active chain only when exact period boundaries are available, and include the current timestamp used.',
    'For transit requests, call the backend transit endpoint directly and report the requested forecast window.',
    'If tool findings already include longevity/dasha/varga evidence, do not claim that a full analysis is still pending.',
    'Never write generic lines like "a definitive assessment requires full divisional/dasha analysis" unless findings explicitly show those sections are missing.',
    'Use relevant prior chat messages as soft context for continuity, but never override canonical chart facts.',
    '',
    `profileId: ${grounding.profileId}`,
    `sourceDocId: ${grounding.sourceDocId}`,
    `chartVersion: ${grounding.chartVersion}`,
    `payloadHash: ${grounding.payloadHash}`,
    `referenceTimestamp: ${grounding.referenceTimestamp}`,
    `referenceTimeSource: ${grounding.referenceTimeSource}`,
    ...(executionPlan
      ? [
          `questionFamily: ${executionPlan.family}`,
          `chartLayers: ${executionPlan.chartLayers.join(', ')}`,
          `includeTiming: ${executionPlan.includeTiming}`,
          `includeTransit: ${executionPlan.includeTransit}`,
          `includeDasha: ${executionPlan.includeDasha}`,
          `microSignals: ${executionPlan.includeMicroSignals.join(', ')}`,
        ]
      : []),
    '',
    'Deterministic tool findings:',
    findingBlock || 'No tool findings available.',
    '',
    'Relevant prior chat context (same thread; semantic retrieval):',
    conversationContext.length > 0 ? conversationContext.map((line) => `- ${line}`).join('\n') : 'None',
    '',
    'Canonical snippets:',
    snippets.join('\n\n') || 'No snippets available.',
    '',
    `User question: ${state.question}`,
  ].join('\n');
}

async function buildPromptNode(state: AgentStateType): Promise<AgentUpdateType> {
  return {
    prompt: buildPrompt(state),
    analysisStages: appendStage(state, 'build_prompt', 'Synthesizing evidence for response prompt'),
  };
}

function extractMessageText(content: unknown): string {
  if (typeof content === 'string') return content.trim();
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === 'string') return part;
        if (part && typeof part === 'object' && 'text' in part && typeof (part as { text?: unknown }).text === 'string') {
          return (part as { text: string }).text;
        }
        return '';
      })
      .join('')
      .trim();
  }
  if (content && typeof content === 'object' && 'text' in content && typeof (content as { text?: unknown }).text === 'string') {
    return (content as { text: string }).text.trim();
  }
  return String(content ?? '').trim();
}

function buildDeterministicFallback(state: AgentStateType): string {
  const findings = state.toolFindings ?? [];
  const lines = [
    'I am answering from deterministic canonical extraction because a live model response is unavailable.',
    '',
    ...findings.flatMap((finding) => [`${finding.name} [${finding.status}]`, ...finding.facts.map((fact) => `- ${fact}`), '']),
  ];
  return lines.join('\n').trim();
}

function sanitizeGenericMissingAnalysisClaims(answer: string, findings: ToolFinding[]): string {
  const hasVarga = findings.some((f) => f.name === 'Varga analyzer' && f.status !== 'unavailable');
  const hasDasha = findings.some((f) => f.name === 'Dasha analyzer' && f.status !== 'unavailable');
  const hasLongevity = findings.some((f) => f.name === 'Longevity analyzer' && f.status !== 'unavailable');

  if (!(hasVarga && (hasDasha || hasLongevity))) {
    return answer;
  }

  return answer
    .replace(/A\s+definitive\s+longevity\s+assessment\s+requires\s+a\s+full\s+analysis\s+of\s+divisional\s+charts\s+and\s+dashas\.?/gi, '')
    .replace(/A\s+definitive\s+assessment\s+requires\s+a\s+full\s+analysis\s+of\s+divisional\s+charts\s+and\s+dashas\.?/gi, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

function routeIntent(state: AgentStateType): 'run_specialized_tools' | 'run_general_tools' {
  if (state.mode === 'mini') {
    return 'run_general_tools';
  }

  if (state.executionPlan && state.executionPlan.family !== 'general') {
    return 'run_specialized_tools';
  }

  const primary = state.intent?.primary ?? 'general';
  return primary === 'general' ? 'run_general_tools' : 'run_specialized_tools';
}

async function answerWithDeepSeekNode(state: AgentStateType): Promise<AgentUpdateType> {
  const grounding = state.grounding;
  if (!grounding) {
    throw new Error('Grounding context missing; loadCanonicalGrounding must run first.');
  }

  if (!state.prompt) {
    throw new Error('Prompt missing before DeepSeek call.');
  }

  try {
    const deepSeek = await invokeDeepSeekBedrock({
      systemPrompt: state.prompt,
      userPrompt: 'Answer using deterministic tool findings and canonical snippets. Be decisive, specific, and avoid generic disclaimers unless payload data is actually missing.',
    });

    const sanitized = sanitizeGenericMissingAnalysisClaims(deepSeek.text, state.toolFindings ?? []);

    return {
      answer: sanitized,
      model: deepSeek.model,
      analysisStages: appendStage(state, 'answer_with_deepseek', 'Generating grounded response text'),
    };
  } catch (error) {
    return {
      answer: `${buildDeterministicFallback(state)}\n\nModel error: ${String(error)}`,
      model: 'deepseek-bedrock-fallback',
      analysisStages: appendStage(state, 'answer_with_deepseek', 'Generating grounded response text', 'Fell back to deterministic output due to model error.'),
    };
  }
}

async function condenseAnswerNode(state: AgentStateType): Promise<AgentUpdateType> {
  if (!state.answer) {
    return {};
  }

  const rawAnswer = state.answer.trim();
  const lineCount = rawAnswer.split('\n').filter(Boolean).length;
  if (rawAnswer.length <= CONCISE_ANSWER_MAX_CHARS && lineCount <= CONCISE_ANSWER_MIN_LINES) {
    return {};
  }

  try {
    const concise = await invokeDeepSeekBedrock({
      systemPrompt: [
        'You are a concise response editor.',
        'Rewrite the provided answer so it is short, direct, and easy to scan.',
        'Keep only the essential facts, timing, and next steps.',
        'Do not add any new facts or explanations.',
        'Do not mention that you are summarizing.',
        'Return plain text only, ideally 3-5 bullet points or 2 short paragraphs.',
      ].join(' '),
      userPrompt: `Condense this answer for the user:\n\n${rawAnswer}`,
      maxTokens: 320,
    });

    const compact = concise.text.trim();
    if (!compact) {
      return {};
    }

    return {
      answer: compact,
      model: concise.model,
      analysisStages: appendStage(state, 'condense_answer', 'Condensing final response'),
    };
  } catch {
    return {};
  }
}

const graph = new StateGraph(AgentState)
  .addNode('load_grounding', loadCanonicalGrounding)
  .addNode('classify_intent', classifyIntentNode)
  .addNode('plan_execution', planExecutionNode)
  .addNode('run_specialized_tools', runSpecializedToolsNode)
  .addNode('run_general_tools', runGeneralToolsNode)
  .addNode('evaluate_coverage', evaluateCoverageNode)
  .addNode('refine_tools', refineToolsNode)
  .addNode('build_prompt', buildPromptNode)
  .addNode('answer_with_deepseek', answerWithDeepSeekNode)
  .addNode('condense_answer', condenseAnswerNode)
  .addEdge(START, 'load_grounding')
  .addEdge('load_grounding', 'classify_intent')
  .addEdge('classify_intent', 'plan_execution')
  .addConditionalEdges('plan_execution', routeIntent)
  .addEdge('run_specialized_tools', 'run_general_tools')
  .addEdge('run_general_tools', 'evaluate_coverage')
  .addConditionalEdges('evaluate_coverage', routeAfterCoverage)
  .addEdge('refine_tools', 'evaluate_coverage')
  .addEdge('build_prompt', 'answer_with_deepseek')
  .addEdge('answer_with_deepseek', 'condense_answer')
  .addEdge('condense_answer', END)
  .compile();

// Phase-2: LangGraph-based grounded assistant that only reads canonical raw payload sections from Postgres.
export async function runKundliAgent(input: AgentAnswerInput): Promise<AgentAnswer> {
  const ownerId = input.ownerId ?? 'anonymous';
  const mode: AgentMode = input.mode ?? 'mini';

  if (mode === 'mini') {
    const miniScope = evaluateMiniScope(input.message);
    if (!miniScope.allowed) {
    return {
        answer: buildMiniUpgradeResponse(input.message, miniScope.reasons),
        model: 'cozmic-mini-guard',
        mode,
      };
    }
  }

  const profileId = normalizeProfileId(input);
  const referenceTime = resolveReferenceTime(input);

  if (!profileId) {
    throw new Error('A profileId or kundli snapshot is required to load canonical chart data.');
  }

  const finalState = (await graph.invoke({
    ownerId,
    profileId,
    mode,
    question: input.message,
    kundliInput: input.kundli ?? null,
    referenceTimestamp: referenceTime.timestamp,
    referenceTimeSource: referenceTime.source,
    conversationContext: input.conversationContext ?? [],
    stageReporter: input.onStage ?? null,
    toolIteration: 0,
    maxToolIterations: 2,
    intent: null,
    executionPlan: null,
    coverageGaps: [],
    analysisStages: [],
    toolFindings: [],
    prompt: null,
  })) as AgentStateType;

  if (!finalState.answer || !finalState.grounding) {
    throw new Error('LangGraph execution completed without a grounded answer.');
  }

  return {
    answer: finalState.answer,
    model: finalState.model ?? env.GOOGLE_GENAI_MODEL,
    mode,
    executionPlan: finalState.executionPlan ?? undefined,
    analysisStages: finalState.analysisStages ?? undefined,
    grounding: {
      ownerId,
      profileId,
      sourceDocId: finalState.grounding.sourceDocId,
      chartVersion: finalState.grounding.chartVersion,
      kundliSignature: finalState.grounding.kundliSignature,
      kundli: finalState.grounding.kundli,
      requestKey: finalState.grounding.requestKey,
      payloadHash: finalState.grounding.payloadHash,
      referenceTimestamp: finalState.grounding.referenceTimestamp,
      referenceTimeSource: finalState.grounding.referenceTimeSource,
      selectedPaths: finalState.grounding.selectedPaths,
    },
  };
}
