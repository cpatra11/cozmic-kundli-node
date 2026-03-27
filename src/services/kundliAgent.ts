import { Annotation, END, START, StateGraph } from '@langchain/langgraph';
import { type KundliSnapshotInput } from './be1Client.js';
import { env } from '../config/env.js';
import { COLLECTIONS, type RagApiSourceDocument, type RagProfileDocument } from '../models/firestoreModels.js';
import { getPostgresStore } from './postgresStore.js';
import { stableHash } from './hash.js';
import { invokeDeepSeekBedrock } from './deepseekBedrock.js';

export interface AgentAnswerInput {
  ownerId?: string;
  message: string;
  kundli?: KundliSnapshotInput;
  profileId?: string;
  clientTimestamp?: number;
}

type IntentPrimary = 'd9' | 'dasha' | 'transit' | 'general';
type TimeSource = 'client' | 'server';

type QuestionIntent = {
  primary: IntentPrimary;
  flags: string[];
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

type ToolFinding = {
  name: string;
  status: 'ok' | 'partial' | 'unavailable';
  facts: string[];
  evidencePaths: string[];
  missing?: string[];
  snippets?: string[];
};

type GroundingContext = {
  ownerId: string;
  profileId: string;
  sourceDocId: string;
  chartVersion: string;
  kundliSignature: string;
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
  grounding: {
    ownerId: string;
    profileId: string;
    sourceDocId: string;
    chartVersion: string;
    kundliSignature: string;
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
  question: Annotation<string>,
  referenceTimestamp: Annotation<number | null>,
  referenceTimeSource: Annotation<TimeSource | null>,
  intent: Annotation<QuestionIntent | null>,
  grounding: Annotation<GroundingContext | null>,
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

function makeVargaToolFinding(rawPayload: unknown, question: string): ToolFinding {
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

  const requestedKeys = extractRequestedVargaKeys(question);
  const selectedKeys = [...new Set(['D1', ...requestedKeys].filter((key) => availableKeys.includes(key)))].slice(0, 8);
  const effectiveKeys = selectedKeys.length > 0 ? selectedKeys : availableKeys.slice(0, 8);

  const facts: string[] = [];
  const evidencePaths: string[] = [];

  for (const key of effectiveKeys) {
    const section = (varga.value as Record<string, unknown>)[key];
    if (section === undefined) continue;
    evidencePaths.push(`chart.varga.${key}`);
    facts.push(...summarizeVargaSection(key, section, question).slice(0, 8));
  }

  const remaining = availableKeys.filter((key) => !effectiveKeys.includes(key));
  if (remaining.length > 0) {
    facts.push(`Additional vargas available: ${remaining.join(', ')}.`);
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

function classifyQuestionIntent(question: string): QuestionIntent {
  const q = question.toLowerCase();
  const flags = new Set<string>();

  if (/\b(d9|navamsha|navamsa)\b/.test(q)) flags.add('d9');
  if (/\b(dasha|dasa|mahadasha|antardasha|vimshottari|period)\b/.test(q)) flags.add('dasha');
  if (/\b(transit|gochar|current transit|current sun|sun transit|today|now)\b/.test(q)) flags.add('transit');

  if (/\b(career|job|profession|business|promotion|work)\b/.test(q)) flags.add('career');
  if (/\b(marriage|relationship|partner|spouse|love|compatibility)\b/.test(q)) flags.add('relationship');
  if (/\b(wealth|money|income|finance|assets|property)\b/.test(q)) flags.add('finance');
  if (/\b(health|disease|illness|medical)\b/.test(q)) flags.add('health');

  const primary: IntentPrimary = flags.has('d9')
    ? 'd9'
    : flags.has('dasha')
      ? 'dasha'
      : flags.has('transit')
        ? 'transit'
        : 'general';

  return {
    primary,
    flags: [...flags],
  };
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

function selectRelevantPaths(question: string, flags: string[] = []): string[] {
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

  add(...planetPathHints(question));
  return [...paths];
}

function selectSections(rawPayload: unknown, question: string, flags: string[] = []): SelectedSection[] {
  const selectedPaths = selectRelevantPaths(question, flags);
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
  const selectedSections = selectSections(rawPayload, state.question, []);
  const fallbackSections = selectedSections.length > 0 ? selectedSections : atlas.slice(0, 12).map((item) => ({ path: item.path, value: getByPath(rawPayload, item.path) }));

  return {
    grounding: {
      ownerId: state.ownerId,
      profileId: state.profileId,
      sourceDocId: profileDoc.data.latestSourceDocId,
      chartVersion: profileDoc.data.chartVersion,
      kundliSignature: profileDoc.data.kundliSignature,
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
  return {
    intent: classifyQuestionIntent(state.question),
  };
}

async function runSpecializedToolsNode(state: AgentStateType): Promise<AgentUpdateType> {
  const grounding = state.grounding;
  if (!grounding) throw new Error('Grounding context missing before specialized tools stage.');
  const intent = state.intent ?? classifyQuestionIntent(state.question);

  const findings: ToolFinding[] = [];
  findings.push(makeReferenceTimeToolFinding(grounding.referenceTimestamp, grounding.referenceTimeSource));
  findings.push(makeAtlasToolFinding(grounding.rawPayload));
  findings.push(makeVargaToolFinding(grounding.rawPayload, state.question));
  if (/\b(arudha|aruda)\b/i.test(state.question)) findings.push(makeArudhaToolFinding(grounding.rawPayload));

  if (intent.flags.includes('d9') || intent.primary === 'd9') findings.push(makeD9ToolFinding(grounding.rawPayload, state.question));
  if (intent.flags.includes('dasha') || intent.primary === 'dasha') findings.push(makeDashaToolFinding(grounding.rawPayload, grounding.referenceTimestamp));
  if (intent.flags.includes('transit') || intent.primary === 'transit') findings.push(makeTransitToolFinding(grounding.rawPayload));

  if (intent.flags.includes('relationship') || intent.flags.includes('career') || intent.flags.includes('health') || intent.flags.includes('finance')) {
    findings.push(makePlacementToolFinding(grounding.rawPayload, state.question));
  }

  return {
    toolFindings: [...(state.toolFindings ?? []), ...findings],
  };
}

async function runGeneralToolsNode(state: AgentStateType): Promise<AgentUpdateType> {
  const grounding = state.grounding;
  if (!grounding) throw new Error('Grounding context missing before general tools stage.');
  const intent = state.intent ?? classifyQuestionIntent(state.question);

  const findings = [...(state.toolFindings ?? [])];
  findings.push(makeReferenceTimeToolFinding(grounding.referenceTimestamp, grounding.referenceTimeSource));
  findings.push(makeGeneralToolFinding(grounding.rawPayload, state.question, intent.flags));
  findings.push(makeVargaToolFinding(grounding.rawPayload, state.question));
  if (/\b(arudha|aruda)\b/i.test(state.question)) findings.push(makeArudhaToolFinding(grounding.rawPayload));
  findings.push(makePlacementToolFinding(grounding.rawPayload, state.question));

  if (/\b(panchanga|tithi|nakshatra|karana|yoga)\b/i.test(state.question)) {
    findings.push(makePanchangaToolFinding(grounding.rawPayload));
  }

  if (/\b(yoga|yogas|ashtakavarga|arudha|aruda)\b/i.test(state.question)) {
    findings.push(makeFeatureToolFinding(grounding.rawPayload));
  }

  if (!findings.some((item) => item.name === 'Dasha analyzer') && /\b(dasha|dasa|mahadasha|antardasha|period)\b/i.test(state.question)) {
    findings.push(makeDashaToolFinding(grounding.rawPayload, grounding.referenceTimestamp));
  }
  if (!findings.some((item) => item.name === 'Transit analyzer') && /\b(transit|gochar|sun transit|today|now)\b/i.test(state.question)) {
    findings.push(makeTransitToolFinding(grounding.rawPayload));
  }

  return {
    toolFindings: findings,
  };
}

function buildPrompt(state: AgentStateType): string {
  const grounding = state.grounding;
  if (!grounding) throw new Error('Grounding context missing while building prompt.');

  const findings = state.toolFindings ?? [];

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
    'For varga requests, prefer the specific Dxx chart named by the user and fall back to D1 only when needed.',
    'For dasha requests, report the active chain only when exact period boundaries are available, and include the current timestamp used.',
    'For transit requests, only report current transit if a transit/gochar block exists in payload.',
    '',
    `profileId: ${grounding.profileId}`,
    `sourceDocId: ${grounding.sourceDocId}`,
    `chartVersion: ${grounding.chartVersion}`,
    `payloadHash: ${grounding.payloadHash}`,
    `referenceTimestamp: ${grounding.referenceTimestamp}`,
    `referenceTimeSource: ${grounding.referenceTimeSource}`,
    '',
    'Deterministic tool findings:',
    findingBlock || 'No tool findings available.',
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

function routeIntent(state: AgentStateType): 'run_specialized_tools' | 'run_general_tools' {
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

    return {
      answer: deepSeek.text,
      model: deepSeek.model,
    };
  } catch (error) {
    return {
      answer: `${buildDeterministicFallback(state)}\n\nModel error: ${String(error)}`,
      model: 'deepseek-bedrock-fallback',
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
    };
  } catch {
    return {};
  }
}

const graph = new StateGraph(AgentState)
  .addNode('load_grounding', loadCanonicalGrounding)
  .addNode('classify_intent', classifyIntentNode)
  .addNode('run_specialized_tools', runSpecializedToolsNode)
  .addNode('run_general_tools', runGeneralToolsNode)
  .addNode('build_prompt', buildPromptNode)
  .addNode('answer_with_deepseek', answerWithDeepSeekNode)
  .addNode('condense_answer', condenseAnswerNode)
  .addEdge(START, 'load_grounding')
  .addEdge('load_grounding', 'classify_intent')
  .addConditionalEdges('classify_intent', routeIntent)
  .addEdge('run_specialized_tools', 'run_general_tools')
  .addEdge('run_general_tools', 'build_prompt')
  .addEdge('build_prompt', 'answer_with_deepseek')
  .addEdge('answer_with_deepseek', 'condense_answer')
  .addEdge('condense_answer', END)
  .compile();

// Phase-2: LangGraph-based grounded assistant that only reads canonical raw payload sections from Postgres.
export async function runKundliAgent(input: AgentAnswerInput): Promise<AgentAnswer> {
  const ownerId = input.ownerId ?? 'anonymous';
  const profileId = normalizeProfileId(input);
  const referenceTime = resolveReferenceTime(input);

  if (!profileId) {
    throw new Error('A profileId or kundli snapshot is required to load canonical chart data.');
  }

  const finalState = (await graph.invoke({
    ownerId,
    profileId,
    question: input.message,
    referenceTimestamp: referenceTime.timestamp,
    referenceTimeSource: referenceTime.source,
    intent: null,
    toolFindings: [],
    prompt: null,
  })) as AgentStateType;

  if (!finalState.answer || !finalState.grounding) {
    throw new Error('LangGraph execution completed without a grounded answer.');
  }

  return {
    answer: finalState.answer,
    model: finalState.model ?? env.GOOGLE_GENAI_MODEL,
    grounding: {
      ownerId,
      profileId,
      sourceDocId: finalState.grounding.sourceDocId,
      chartVersion: finalState.grounding.chartVersion,
      kundliSignature: finalState.grounding.kundliSignature,
      requestKey: finalState.grounding.requestKey,
      payloadHash: finalState.grounding.payloadHash,
      referenceTimestamp: finalState.grounding.referenceTimestamp,
      referenceTimeSource: finalState.grounding.referenceTimeSource,
      selectedPaths: finalState.grounding.selectedPaths,
    },
  };
}
