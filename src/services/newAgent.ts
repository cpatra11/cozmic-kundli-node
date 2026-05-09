import { Annotation, Command, END, Send, START, StateGraph, MessagesAnnotation } from '@langchain/langgraph';
import type { KundliSnapshotInput } from './be1Client.js';
import { fetchBe1Calculate, fetchBe1Transit } from './be1Client.js';
import { invokeDeepSeekBedrock } from './deepseekBedrock.js';
import { getPostgresPool } from './postgresClient.js';
import { PostgresCache } from './postgresCache.js';
import { logNodeStart, logNodeEnd, logAgentStart } from '../utils/debugLog.js';
import type { RelevantChatMemory } from './chatMemory.js';

// -- State with LangGraph reducers --

const AgentState = Annotation.Root({
  ...MessagesAnnotation.spec,

  ownerId: Annotation<string>,
  profileId: Annotation<string | null>,
  mode: Annotation<'mini' | 'pro'>,
  kundliInput: Annotation<KundliSnapshotInput | null>,

  topLevelRoute: Annotation<'pipeline' | 'smalltalk' | 'general_astro' | null>,

  grounding: Annotation<any>,

  findings: Annotation<string[]>({
    reducer: (a, b) => [...a, ...b],
    default: () => [],
  }),

  answer: Annotation<string | null>,
  finalAnswer: Annotation<string | null>,

  qualityRetryCount: Annotation<number>,
  shouldRegenerate: Annotation<boolean>,

  selectedVargas: Annotation<string[]>,
  selectedInfolevels: Annotation<string[]>,
  fetchTransit: Annotation<boolean>,

  dashaNesting: Annotation<number>,
  timingReasoning: Annotation<string | null>,
  relevantMemories: Annotation<RelevantChatMemory[]>,
});

type AgentStateType = typeof AgentState.State;

// -- Helpers --

function extractQuestionFromMessages(messages: any[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    const role = msg.role || msg.kwargs?.role || (msg.type === 'human' ? 'user' : msg.type === 'ai' ? 'assistant' : undefined);
    const content = msg.content || msg.kwargs?.content;
    if (role === 'user') {
      return Array.isArray(content) ? (content[0]?.text || '') : (content || '');
    }
  }
  return '';
}

function parseJsonSafely(text: string): any {
  try {
    return JSON.parse(text.replace(/```json|```/g, '').trim());
  } catch {
    return null;
  }
}

// -- Inline routing (replaces intentClassifier.ts) --

function isObviousSmalltalk(q: string): boolean {
  return /^(hello|hi|hey|thanks|thank you|bye|goodbye|great|ok|okay|nice|good)\b/i.test(q.trim());
}

function isIdentityQuestion(q: string): boolean {
  return /\b(who are you|what are you|what can you|your name|tell me about yourself|capabilities|help)\b/i.test(q);
}

function shouldForcePipelineRoute(q: string): boolean {
  const patterns = [
    'chart','kundli','horoscope','astrology','planet','house','rashi','nakshatra','dasha','transit','birth','janam','bhav',
    'varga','yoga','karaka','arudha','mangal','shani','guru','rahu','ketu','career','finance','marriage','health','education',
    'property','travel','remedy','prediction','sarkari','naukri','upsc','ssc','banking','railway','defence','ias','ips','officer',
    'jee','neet','gate','cat','entrance','manglik','kaal','sarp','pitra','kuja','upay','upaya','mantra','puja','vastu',
    'muhurta','engineering','medical','doctor','engineer','videsh','nri','gemstone','ratna','stone','dhaiya',
  ];
  const phrases = [
    '\\bD\\d+\\b','government\\s+job','civil\\s+service','competitive\\s+exam',
    'love\\s+marriage','arranged\\s+marriage','intercaste','love\\s+match',
    'mangal\\s+dosh','kaal\\s+sarp','pitra\\s+dosh','kuja\\s+dosh',
    'shubh\\s+(?:time|samay|muhurat)','kundli\\s+matching','gun\\s+milan',
    'business\\s+vs\\s+service','service\\s+or\\s+business',
    'foreign\\s+settlement','abroad\\s+study','settle\\s+abroad',
    'sade\\s+sati','shani\\s+sade','shani\\s+dhaiya',
  ];
  return new RegExp([...patterns, ...phrases].join('|'), 'i').test(q);
}

const RASHI_NAMES = ['Aries','Taurus','Gemini','Cancer','Leo','Virgo','Libra','Scorpio','Sagittarius','Capricorn','Aquarius','Pisces'];

function getRashiName(r: number): string {
  return RASHI_NAMES[(r - 1 + 12) % 12] || `Rashi ${r}`;
}

function fmtDeg(d: number): string {
  return `${d.toFixed(1)}°`;
}

// -- Varga/infolevel selection constants --

const VARGA_MAP: Record<string, { name: string; keywords: RegExp }> = {
  D2:  { name: 'Hora',           keywords: /wealth|finance|financial|financially|money|income|prosperity|rich|poor|poverty|afford|debt/i },
  D3:  { name: 'Drekkana',       keywords: /sibling|brother|sister|courage|co-born/i },
  D4:  { name: 'Chaturthamsa',   keywords: /property|real estate|land|house|home|accommodation|fortune|immovable/i },
  D6:  { name: 'Shashtamsha',    keywords: /health|disease|illness|sickness|litigation|legal|court|disputes|immunity|recovery|chronic|surgery|hospital|bimari/i },
  D7:  { name: 'Saptamsa',       keywords: /child|children|progeny|offspring|creativity|fertility|son|daughter/i },
  D9:  { name: 'Navamsa',        keywords: /marriage|spouse|partner|love|relationship|divorce|husband|wife|navamsa|d9/i },
  D10: { name: 'Dasamsa',        keywords: /career|profession|work|job|business|employment|occupation|promotion|dasamsa|d10/i },
  D12: { name: 'Dwadasamsa',     keywords: /parent|mother|father|family|ancestor|lineage/i },
  D16: { name: 'Shodasamsa',     keywords: /travel|vehicle|\bcar\b|journey|commute|abroad|foreign|trip|conveyance/i },
  D20: { name: 'Vimsamsa',       keywords: /spiritual|spirituality|devotion|religious|god|temple|meditation|guru/i },
  D24: { name: 'Siddhamamsa',    keywords: /education|wisdom|knowledge|learning|study|student|exam|degree|intellect|intelligence/i },
  D27: { name: 'Bhamsa',         keywords: /strength|weakness|power|ability|talent|skill|capability|strong/i },
  D30: { name: 'Trimamsa',       keywords: /obstacle|misfortune|evil|problem|difficulty|trouble|suffering|adversity|enemy|struggl|hardship|blockage|setback|failure/i },
  D40: { name: 'Khavedamsa',     keywords: /maternal|mother.*lineage/i },
  D45: { name: 'Akshvedamsa',    keywords: /paternal|father.*lineage/i },
  D60: { name: 'Shashtiamsa',    keywords: /karma|destiny|fate|shashtiamsa|d60/i },
};

const VARGA_NAMES: Record<string, string> = {
  D1: 'Rasi', D2: 'Hora', D3: 'Drekkana', D4: 'Chaturthamsa',
  D6: 'Shashtamsha', D7: 'Saptamsa', D9: 'Navamsa', D10: 'Dasamsa', D12: 'Dwadasamsa',
  D16: 'Shodasamsa', D20: 'Vimsamsa', D24: 'Siddhamamsa', D27: 'Bhamsa',
  D30: 'Trimamsa', D40: 'Khavedamsa', D45: 'Akshvedamsa', D60: 'Shashtiamsa',
};

const VALID_VARGAS = new Set(Object.keys(VARGA_NAMES));

const TIMING_KEYWORDS = /when|prediction|future|dasha|period|timing|transit|gochar|mahadasha|antardasha|bhukti|forecast|upcoming|will\s+I/i;
const STRENGTH_KEYWORDS = /strong|weak|strength|power|bala|ashtakavarga|grahabala|powerful|capable/i;
const ARUDHA_KEYWORDS = /arudha|pada|perception|illusion|reflection|image/i;

// -- Routing helper: detect general astrology concept questions --

function isGeneralAstroQuestion(q: string): boolean {
  const hasAstro = /\b(house|planet|rashi|nakshatra|dasha|yoga|varga|bhava|karaka|bala|arudha|aspect|conjunction|retrograde|exaltation|debiliation|navamsa|dasamsa|saptamsa|drekkana|hora|shashtiamsa|trimamsa|siddhamamsa|D\d+|sun|moon|mars|mercury|jupiter|venus|saturn|rahu|ketu|graha)\b/i.test(q);
  const noPersonal = !/\b(my|mine|my\s+chart|my\s+birth|do\s+I|am\s+I|will\s+I|have\s+I|my\s+kundli)\b/i.test(q);
  const isConceptual = /\b(what is|what does|tell me about|explain|meaning of|significance of|describe|define|how does|why does|what are)\b/i.test(q);
  return hasAstro && noPersonal && isConceptual;
}

// -- Varga selection --

function selectVargas(question: string, mode: 'mini' | 'pro'): string[] {
  const selected = new Set<string>(['D1']);

  if (mode === 'pro') {
    selected.add('D9');
    selected.add('D10');
  }

  // Explicit D\d+ mentions (case-insensitive)
  const dChartMatches = question.matchAll(/\b(D\d+)\b/gi);
  for (const match of dChartMatches) {
    const vk = match[1].toUpperCase();
    if (VALID_VARGAS.has(vk)) {
      selected.add(vk);
    }
  }

  // Keyword matching (Pro only)
  if (mode === 'pro') {
    for (const [vk, entry] of Object.entries(VARGA_MAP)) {
      if (entry.keywords.test(question)) {
        selected.add(vk);
      }
    }
  }

  // Mini always returns only D1
  if (mode === 'mini') {
    return ['D1'];
  }

  return [...selected].sort();
}

// -- Infolevel + transit selection --

function selectInfolevelsAndTransit(question: string, mode: 'mini' | 'pro'):
  { infolevels: string[]; fetchTransit: boolean }
{
  if (mode === 'mini') {
    return { infolevels: ['basic'], fetchTransit: false };
  }

  const levels = new Set<string>(['basic', 'panchanga', 'yogas', 'ayanamsa']);

  if (TIMING_KEYWORDS.test(question)) {
    levels.add('dasha');
  }
  if (STRENGTH_KEYWORDS.test(question)) {
    levels.add('ashtakavarga');
    levels.add('grahabala');
  }
  if (ARUDHA_KEYWORDS.test(question)) {
    levels.add('arudha');
  }

  // Dosha/remedy keywords trigger strength analysis
  if (/manglik|mangal\s+dosh|kaal\s+sarp|pitra\s+dosh|kuja\s+dosh|upay|upaya|ratna|gemstone|mantra|remedy|dosha|dosh/i.test(question)) {
    levels.add('grahabala');
    levels.add('ashtakavarga');
  }

  const fetchTransit = TIMING_KEYWORDS.test(question);

  return { infolevels: [...levels], fetchTransit };
}

// -- Check if Mini user mentioned a Pro-only varga --

function miniMentionedProVarga(question: string): boolean {
  const matches = question.matchAll(/\b(D\d+)\b/gi);
  for (const match of matches) {
    const vk = match[1].toUpperCase();
    if (vk !== 'D1' && VALID_VARGAS.has(vk)) return true;
  }
  return false;
}

// -- Dasha nesting selection --

const FINE_SCALE = /\b(exact date|specific date|this week|next week|within\s+(day|week)|when\s+exactly|precise\s+date|immediate)\b/i;
const MEDIUM_SCALE = /\b(this month|next month|within\s+month|coming month|this year|next year|within\s+year|by\s+(next|this)\s+(month|year)|soon|near\s+future)\b/i;

function selectDashaNesting(question: string, mode: 'mini' | 'pro'): { nesting: number; reasoning: string } {
  if (mode === 'mini') return { nesting: 1, reasoning: 'Mini mode: mahadasha overview only' };

  if (FINE_SCALE.test(question))
    return { nesting: 4, reasoning: `Fine-scale timing (weeks/days). Nesting=4 enables sookshmantardasha-level analysis (~days resolution).` };

  if (MEDIUM_SCALE.test(question))
    return { nesting: 3, reasoning: `Medium-scale timing (months). Nesting=3 adds pratyantardasha granularity (~weeks to months).` };

  return { nesting: 2, reasoning: `Year-scale or general prediction. Nesting=2 (mahadasha+antardasha) provides optimal balance — 13KB vs 10MB at n=5.` };
}

function formatRemaining(endDate: string): string {
  const now = Date.now();
  const end = new Date(endDate).getTime();
  if (!end || end <= now) return 'ended';
  const totalMs = end - now;
  const totalDays = Math.floor(totalMs / 86400000);
  const years = Math.floor(totalDays / 365.25);
  const months = Math.floor((totalDays % 365.25) / 30.44);
  const days = Math.floor(totalDays % 30.44);
  const parts: string[] = [];
  if (years > 0) parts.push(`${years}y`);
  if (months > 0) parts.push(`${months}m`);
  if (days > 0 || parts.length === 0) parts.push(`${days}d`);
  return `${parts.join(' ')} remaining`;
}

// -- Node: route --

async function routeNode(state: AgentStateType): Promise<Partial<AgentStateType>> {
  const question = extractQuestionFromMessages(state.messages || []);
  logNodeStart('route', { question, mode: state.mode });
  console.log('[route] input:', { question, mode: state.mode });

  if (!question?.trim()) {
    const g = 'Hello! Please share your birth details (date, time, place) or ask about your chart.';
    console.log('[route] output: no question -> smalltalk');
    return { topLevelRoute: 'smalltalk', answer: g, finalAnswer: g };
  }

  if (isObviousSmalltalk(question)) { console.log('[route] output: smalltalk'); return { topLevelRoute: 'smalltalk' }; }
  if (isIdentityQuestion(question)) { console.log('[route] output: general_astro (identity)'); return { topLevelRoute: 'general_astro' }; }

  // NEW: General astrology concept question — answer from LLM knowledge, no chart
  if (isGeneralAstroQuestion(question)) {
    console.log('[route] output: general_astro (conceptual)');
    return { topLevelRoute: 'general_astro' };
  }

  // Chart-specific question
  if (shouldForcePipelineRoute(question)) {
    const selectedVargas = selectVargas(question, state.mode);
    const { infolevels, fetchTransit } = selectInfolevelsAndTransit(question, state.mode);
    const { nesting, reasoning } = selectDashaNesting(question, state.mode);

    console.log('[route] output: pipeline', { selectedVargas, infolevels, fetchTransit, dashaNesting: nesting });

    return {
      topLevelRoute: 'pipeline',
      selectedVargas,
      selectedInfolevels: infolevels,
      fetchTransit,
      dashaNesting: nesting,
      timingReasoning: reasoning,
    };
  }

  // Fallback
  console.log('[route] output: smalltalk (fallback)');
  return { topLevelRoute: 'smalltalk' };
}

function routeAfterRoute(state: AgentStateType): string {
  switch (state.topLevelRoute) {
    case 'pipeline': return 'load_grounding';
    default: return 'fast_answer';
  }
}

// -- Node: fast_answer (smalltalk / general_astro) --

async function fastAnswerNode(state: AgentStateType): Promise<Partial<AgentStateType>> {
  if (state.finalAnswer) return {};

  const question = extractQuestionFromMessages(state.messages || []);
  const isSmallTalk = state.topLevelRoute === 'smalltalk';
  console.log('[fast_answer] input:', { question, isSmallTalk });

  if (!question?.trim()) {
    console.log('[fast_answer] output: no question');
    return { finalAnswer: 'Hello! How can I help you with your chart today?' };
  }

  console.log('[fast_answer] calling LLM...');
  const result = await invokeDeepSeekBedrock({
    systemPrompt: isSmallTalk
      ? 'You are Cozmic, a friendly Vedic astrology assistant. Respond naturally to casual chat. Keep it brief and warm.'
      : 'You are Cozmic, a Vedic astrology assistant. Answer conceptual questions clearly and concisely (3-5 paragraphs max).',
    userPrompt: question,
    maxTokens: 1024,
  });
  console.log('[fast_answer] LLM response length:', result.text.length);

  return { finalAnswer: result.text };
}

// -- Node: load_grounding --

function extractDashaPeriods(node: any, depth: number, maxDepth: number): any[] {
  const periods: any[] = [];
  if (!node || !node.periods) return periods;
  for (const [key, period] of Object.entries(node.periods) as [string, any][]) {
    const entry: any = { key, type: period.type || 'period', start: period.start, end: period.end, duration: period.duration };
    periods.push(entry);
    if (depth < maxDepth && period.periods) {
      entry.subPeriods = extractDashaPeriods(period, depth + 1, maxDepth);
    }
  }
  return periods;
}

function computeTransitDates(dashaTree: any, maxDates: number): Date[] {
  const dates = new Set<number>([Date.now()]);
  const now = Date.now();
  const lookAhead = 10 * 365.25 * 86400000; // 10 years
  const cutoff = now + lookAhead;

  function walk(node: any): void {
    if (!node || !node.periods) return;
    for (const period of Object.values(node.periods) as any[]) {
      const start = new Date(period.start).getTime();
      const end = new Date(period.end).getTime();
      if (start && start >= now && start <= cutoff) dates.add(start);
      if (end && end >= now && end <= cutoff) dates.add(end);
      const mid = start && end ? new Date((start + end) / 2).getTime() : 0;
      if (mid && mid >= now && mid <= cutoff) dates.add(mid);
      if (period.periods) walk(period);
    }
  }

  if (dashaTree) walk(dashaTree);

  return [...dates]
    .sort((a, b) => a - b)
    .slice(0, maxDates)
    .map(ts => new Date(ts));
}

function computeNatalHouseMap(transitGraha: Record<string, any>, natalLagnaRashi: number): Record<string, number> {
  const map: Record<string, number> = {};
  if (typeof natalLagnaRashi !== 'number') return map;
  for (const [planet, data] of Object.entries(transitGraha)) {
    const r = data?.rashi;
    if (typeof r === 'number') {
      map[planet] = ((r - natalLagnaRashi + 12) % 12) + 1;
    }
  }
  return map;
}

async function loadGroundingNode(state: AgentStateType): Promise<Partial<AgentStateType>> {
  const kundli = state.kundliInput;
  logNodeStart('load_grounding', { hasKundli: !!kundli, mode: state.mode });
  console.log('[load_grounding] input:', { hasKundli: !!kundli, mode: state.mode });

  if (!kundli) {
    console.log('[load_grounding] no kundli input -> null grounding');
    return { grounding: null };
  }

  // Use on-demand selected vargas/infolevels + dynamic nesting from routeNode
  const varga = (state.selectedVargas?.length ? state.selectedVargas : ['D1']).join(',');
  const infolevel = (state.selectedInfolevels?.length ? state.selectedInfolevels : ['basic']).join(',');
  const nesting = state.dashaNesting ?? (state.mode === 'pro' ? 2 : 1);

  logNodeStart('load_grounding', { varga, infolevel, nesting, mode: state.mode });
  console.log('[load_grounding] calling BE1:', { varga, infolevel, nesting });

  try {
    const t0 = Date.now();
    const apiResponse = await fetchBe1Calculate(kundli, { varga, infolevel, nesting });
    console.log('[load_grounding] BE1 calculate took', Date.now() - t0, 'ms');

    let rawPayload: any = apiResponse;
    const transitSnapshots: Record<string, any> = {};

    // Fetch transit data at multiple dates when timing-related question
    if (state.fetchTransit) {
      const payload = getPayload(rawPayload);
      const dashaTree = payload?.dasha;

      // Compute transit dates from dasha periods
      const transitDates = computeTransitDates(dashaTree, 10);
      console.log('[load_grounding] computed', transitDates.length, 'transit dates from dasha tree');

      try {
        const t1 = Date.now();
        const transitResults = await Promise.all(
          transitDates.map(date => fetchBe1Transit(kundli, date, { nesting: 1 }))
        );
        console.log('[load_grounding] all transit fetches took', Date.now() - t1, 'ms');

        const payload2 = getPayload(rawPayload);
        const natLagna = payload2?.lagna?.Lg || payload2?.lagna;
        const natalLagnaRashi = natLagna?.rashi;
        const natalHouseMaps: Record<string, Record<string, number>> = {};

        for (let i = 0; i < transitDates.length; i++) {
          const date = transitDates[i];
          const dateKey = date.toISOString().slice(0, 10);
          const obj = transitResults[i] as any;
          const chart = obj.chart || obj;
          const transitData = chart?.transit || chart;

          if (transitData?.graha) {
            const grahaData = transitData.graha as Record<string, any>;
            const houseMap = computeNatalHouseMap(grahaData, natalLagnaRashi);
            natalHouseMaps[dateKey] = houseMap;
            transitSnapshots[dateKey] = {
              graha: grahaData,
              houseMap,
            };
          }
        }
        console.log('[load_grounding] transit snapshots:', Object.keys(transitSnapshots).length);
      } catch (tErr) {
        console.log('[load_grounding] multi-transit fetch failed:', String(tErr));
        logNodeStart('load_grounding', { transitError: String(tErr) });
      }
    }

    // Log what data we got
    const payload = getPayload(rawPayload);
    const planetCount = Object.keys(payload.graha || {}).length;
    const vargaCount = Object.keys(payload.varga || {}).length;
    const hasDasha = !!(payload.dasha);
    const hasYogas = !!(payload.yogas || []).length;
    const hasAshtakavarga = !!payload.ashtakavarga;
    const hasTransit = Object.keys(transitSnapshots).length > 0;

    console.log('[load_grounding] data summary:', { planetCount, vargaCount, hasDasha, hasYogas, hasAshtakavarga, hasTransit, nesting });

    logNodeStart('load_grounding', {
      planets: planetCount,
      vargaCharts: vargaCount,
      hasDasha,
      hasYoga: hasYogas,
      hasAshtakavarga,
      hasTransit,
    });

    logNodeEnd('load_grounding', { varga, infolevel, nesting, hasTransit });

    return {
      grounding: { rawPayload, transitSnapshots },
    };
  } catch (error) {
    console.log('[load_grounding] BE1 calculate failed:', String(error));
    logNodeEnd('load_grounding', { error: String(error) });
    return { grounding: { rawPayload: null } };
  }
}

// -- Parallel extractors (launched via Send) --

function fmtObj(val: any): string {
  if (!val) return '';
  if (typeof val === 'string') return val;
  if (typeof val === 'number') return String(val);
  if (val.name) return val.name;
  if (val.nakshatra?.name) return val.nakshatra.name;
  // Fallback: try to convert to string safely
  try {
    return JSON.stringify(val);
  } catch {
    return '';
  }
}

function getPayload(raw: any): any {
  return raw?.chart || raw;
}

// Extractor 1: PLANETS + BHAVA + LAGNA + PANCHANDA (runs for both pro and mini)
function extractPlanetsNode(state: AgentStateType): Partial<AgentStateType> {
  const payload = getPayload(state.grounding?.rawPayload);
  if (!payload) { console.log('[extract_planets] no payload'); return { findings: [] }; }

  logNodeStart('extract_planets', { hasPayload: !!payload });

  const lines: string[] = [];
  const graha = payload.graha as Record<string, any>;
  const bhava = payload.bhava as Record<string, any>;
  const lagna = payload.lagna?.Lg || payload.lagna;
  const panchanga = payload.panchanga;

  const planetCount = Object.keys(graha || {}).length;
  const houseCount = Object.keys(bhava || {}).length;
  logNodeStart('extract_planets', { planetCount, houseCount, hasLagna: !!lagna, hasPanchanga: !!panchanga });

  // PLANETS
  if (graha) {
    lines.push('=== GRAHA POSITIONS ===');
    for (const [planet, data] of Object.entries(graha)) {
      const r = data?.rashi;
      const deg = data?.longitude;
      const nk = data?.nakshatra;
      const hs = data?.house_number ?? data?.house;
      const retro = data?.retrograde ? ' (R)' : '';
      const parts = [`${planet}${retro}: ${typeof r === 'number' ? getRashiName(r) : '?'}`];
      if (typeof deg === 'number') parts.push(fmtDeg(deg));
      if (fmtObj(nk)) parts.push(fmtObj(nk));
      if (typeof hs === 'number') parts.push(`House ${hs}`);
      lines.push(parts.join(' — '));
    }
  }

  // BHAVA (houses)
  if (bhava) {
    lines.push('=== BHAVA (HOUSE CUSPS) ===');
    for (const [num, data] of Object.entries(bhava)) {
      const r = data?.rashi;
      lines.push(`House ${num}: ${typeof r === 'number' ? getRashiName(r) : '?'}`);
    }
  }

  // LAGNA
  if (lagna?.rashi) {
    lines.push(`=== LAGNA (ASCENDANT) ===`);
    const rName = typeof lagna.rashi === 'number' ? getRashiName(lagna.rashi) : String(lagna.rashi);
    const nkName = fmtObj(lagna.nakshatra);
    lines.push(`Lagna: ${rName}${nkName ? `, ${nkName} nakshatra` : ''}`);
  }

  // PANCHANDA
  if (panchanga) {
    lines.push('=== PANCHANG ===');
    if (fmtObj(panchanga.tithi)) lines.push(`Tithi: ${fmtObj(panchanga.tithi)}`);
    if (fmtObj(panchanga.vaara)) lines.push(`Vaara: ${fmtObj(panchanga.vaara)}`);
    if (fmtObj(panchanga.nakshatra)) lines.push(`Nakshatra: ${fmtObj(panchanga.nakshatra)}`);
    if (fmtObj(panchanga.yoga)) lines.push(`Yoga: ${fmtObj(panchanga.yoga)}`);
    if (fmtObj(panchanga.karana)) lines.push(`Karana: ${fmtObj(panchanga.karana)}`);
  }

  logNodeEnd('extract_planets', { findingsCount: lines.length });
  console.log('[extract_planets] output lines:', lines.length);
  return { findings: lines.length ? [lines.join('\n')] : [] };
}

// Extractor 2: VARGA CHARTS (pro only)
function extractVargaNode(state: AgentStateType): Partial<AgentStateType> {
  const payload = getPayload(state.grounding?.rawPayload);
  const varga = payload?.varga as Record<string, any>;
  if (!varga) { logNodeStart('extract_varga', { charts: 0 }); console.log('[extract_varga] no varga data'); return { findings: [] }; }

  logNodeStart('extract_varga', { charts: Object.keys(varga).length });
  console.log('[extract_varga] varga charts:', Object.keys(varga).length);

  const lines: string[] = ['=== ALL VARGA CHARTS ==='];
  for (const [vk, data] of Object.entries(varga)) {
    const lg = data?.lagna?.Lg || data?.lagna;
    const planetsInHouses: string[] = [];
    const lgRashiNum = typeof lg?.rashi === 'number' ? lg.rashi : 0;
    const graha = data?.graha as Record<string, any> || {};
    for (const [planet, pdata] of Object.entries(graha)) {
      const h = pdata?.house_number ?? pdata?.house;
      if (typeof h === 'number') {
        const rNum = pdata?.rashi || ((lgRashiNum > 0) ? ((lgRashiNum + h - 2 + 12) % 12 + 1) : 0);
        const rName = rNum > 0 ? getRashiName(rNum) : '?';
        planetsInHouses.push(`${planet}:House${h}(${rName})`);
      }
    }
    const rName = lg?.rashi ? (typeof lg.rashi === 'number' ? getRashiName(lg.rashi) : String(lg.rashi)) : '?';
    lines.push(`${vk}: Lagna ${rName} | ${planetsInHouses.join(', ') || 'no planets'}`);
  }

  logNodeEnd('extract_varga', { chartEntries: lines.length });
  console.log('[extract_varga] output lines:', lines.length);
  return { findings: lines };
}

// Extractor 3: TIMING - DASHA + TRANSIT (pro only)
function extractTimingNode(state: AgentStateType): Partial<AgentStateType> {
  const raw = state.grounding?.rawPayload;
  const payload = getPayload(raw);
  const lines: string[] = [];
  const now = Date.now();

  logNodeStart('extract_timing', {});

  // DASHA timeline - handle nested object structure
  const dashaRoot = payload?.dasha as any;
  const timingReasoning = state.timingReasoning;
  const hasDasha = dashaRoot && (dashaRoot.periods || dashaRoot.nesting);

  if (hasDasha) {
    lines.push('=== DASHA TIMELINE ===');

    if (timingReasoning) {
      lines.push(`Analysis: ${timingReasoning}`);
    }

    // Recursively format dasha tree
    function formatPeriodTree(node: any, depth: number, maxDepth: number, prefix: string, isLast: boolean): void {
      if (!node || !node.periods) return;
      const periodKeys = Object.keys(node.periods);
      let idx = 0;
      for (const key of periodKeys) {
        const period = node.periods[key];
        idx++;
        const isLastChild = idx === periodKeys.length;
        const connector = isLastChild ? '  └ ' : '  ├ ';
        const childPrefix = isLast ? '     ' : '  │  ';

        const startStr = period.start ? period.start.slice(0, 10) : '?';
        const endStr = period.end ? period.end.slice(0, 10) : '?';
        const levelName = nestingNames[period.nesting] || period.type || 'period';

        // Check if current
        const pStart = new Date(period.start).getTime();
        const pEnd = new Date(period.end).getTime();
        const isCurrent = pStart <= now && pEnd > now;
        const remaining = isCurrent && period.end ? ` — ${formatRemaining(period.end)}` : '';
        const isFuture = pStart > now;
        const isPast = pEnd <= now;

        let label = `${prefix}${connector}${key} ${levelName} (${startStr} → ${endStr})`;
        if (isCurrent) label += ' ← ACTIVE';
        if (isFuture) label += ' ← upcoming';
        label += remaining;

        lines.push(label);

        // Recurse into subperiods if within nesting depth
        if (depth < maxDepth && period.periods && Object.keys(period.periods).length > 0) {
          const nextPrefix = isLast ? `${prefix}     ` : `${prefix}  │  `;
          formatPeriodTree(period, depth + 1, maxDepth, nextPrefix, isLast);
        }
      }
    }

    const nestingNames: Record<number, string> = {
      1: 'mahadasha', 2: 'antardasha', 3: 'pratyantardasha',
      4: 'sookshmantardasha', 5: 'pranantardasha',
    };

    // Start from root level
    const maxDisplayDepth = 2; // Always show at least mahadasha + antardasha
    formatPeriodTree(dashaRoot, 1, Math.max(maxDisplayDepth, state.dashaNesting ?? 2), '', true);
  } else {
    logNodeStart('extract_timing', { dashaPeriods: 0 });
  }

  // TRANSIT snapshots
  const transitSnapshots = state.grounding?.transitSnapshots as Record<string, { graha: Record<string, any>; houseMap: Record<string, number> }>;
  if (transitSnapshots && Object.keys(transitSnapshots).length > 0) {
    const dateKeys = Object.keys(transitSnapshots).sort();
    console.log('[extract_timing] transit snapshots:', dateKeys);
    lines.push('');
    lines.push('=== GOCHAR SNAPSHOTS ===');

    for (const dateKey of dateKeys) {
      const snap = transitSnapshots[dateKey];
      if (!snap?.graha) continue;

      const isCurrent = dateKey === new Date().toISOString().slice(0, 10);
      lines.push(`${dateKey}${isCurrent ? ' (current)' : ''}`);
      for (const [planet, data] of Object.entries(snap.graha)) {
        const r = data?.rashi;
        const nh = snap.houseMap?.[planet];
        if (r || nh) {
          lines.push(`  ${planet}: ${r ? getRashiName(r) : '?'} House ${nh || '?'}`);
        }
      }
    }
  } else {
    logNodeStart('extract_timing', { transitPlanets: 0 });
    console.log('[extract_timing] no transit data');
  }

  // SADE SATI + SHAHI DHAIYA — Moon birth rashi vs transit Saturn
  const natalGraha = payload?.graha as Record<string, any> | undefined;
  const moonRashi = natalGraha?.Mo?.rashi as number | undefined;
  if (typeof moonRashi === 'number' && transitSnapshots && Object.keys(transitSnapshots).length > 0) {
    const saturnLines: string[] = [];
    for (const [dateKey, snap] of Object.entries(transitSnapshots)) {
      const satRashi = snap?.graha?.Sa?.rashi as number | undefined;
      if (typeof satRashi !== 'number') continue;
      const diff = ((satRashi - moonRashi) + 12) % 12;
      let phase = '';
      if (diff === 11) phase = 'Sade Sati FIRST phase (12th from Moon)';
      else if (diff === 0) phase = 'Sade Sati PEAK phase (1st from Moon)';
      else if (diff === 1) phase = 'Sade Sati LAST phase (2nd from Moon)';
      else if (diff === 3) phase = 'Shani Dhaiya (4th from Moon)';
      else if (diff === 7) phase = 'Ashtama Shani (8th from Moon)';
      if (phase) {
        const isCurrent = dateKey === new Date().toISOString().slice(0, 10);
        saturnLines.push(`${dateKey}${isCurrent ? ' (current)' : ''}: Saturn in ${getRashiName(satRashi)} — ${phase}`);
      }
    }
    if (saturnLines.length > 0) {
      lines.push('');
      lines.push('=== SADE SATI / DHAIYA ===');
      lines.push(`Moon birth rashi: ${getRashiName(moonRashi)}`);
      lines.push(...saturnLines);
    }
  }

  const hasData = lines.length > 0;
  logNodeEnd('extract_timing', { lines: lines.length });
  console.log('[extract_timing] lines produced:', hasData ? lines.length : 0);
  return hasData ? { findings: [lines.join('\n')] } : { findings: [] };
}

// Extractor 4: SPECIAL - YOGAS + ASHTAKAVARGA + GRAHABALA + ARUDHA (pro only)
function extractSpecialNode(state: AgentStateType): Partial<AgentStateType> {
  const payload = getPayload(state.grounding?.rawPayload);
  const lines: string[] = [];

  logNodeStart('extract_special', {});

  // YOGAS
  const yogas = payload?.yogas as any[];
  if (yogas?.length) {
    logNodeStart('extract_special', { yogas: yogas.length });
    console.log('[extract_special] yogas:', yogas.length);
    lines.push('=== YOGAS (planetary combinations) ===');
    for (const y of yogas.slice(0, 10)) {
      lines.push(`${y.name || 'Yoga'}: ${y.description || ''}`);
    }
  }

  // ASHTAKAVARGA
  const ashtakavarga = payload?.ashtakavarga;
  if (ashtakavarga) {
    logNodeStart('extract_special', { ashtakavarga: Object.keys(ashtakavarga).length });
    console.log('[extract_special] ashtakavarga planets:', Object.keys(ashtakavarga).length);
    lines.push('=== ASHTAKAVARGA (points) ===');
    for (const [planet, pts] of Object.entries(ashtakavarga)) {
      if (typeof pts === 'number') {
        lines.push(`${planet}: ${pts} points`);
      }
    }
  }

  // GRAHABALA (strength)
  const grahabala = payload?.grahabala;
  if (grahabala) {
    logNodeStart('extract_special', { grahabala: Object.keys(grahabala).length });
    console.log('[extract_special] grahabala planets:', Object.keys(grahabala).length);
    lines.push('=== GRAHABALA (planetary strength) ===');
    for (const [planet, strength] of Object.entries(grahabala)) {
      lines.push(`${planet}: ${fmtObj(strength)}`);
    }
  }

  // ARUDHA
  const arudha = payload?.arudha as Record<string, any> || {};
  if (Object.keys(arudha).length) {
    logNodeStart('extract_special', { arudha: Object.keys(arudha).length });
    lines.push('=== ARUDHA PADS ===');
    for (const [house, data] of Object.entries(arudha)) {
      const r = data?.rashi;
      lines.push(`House ${house}: Arudha Lagna ${typeof r === 'number' ? getRashiName(r) : '?'}`);
    }
  }

  // DOSHA ANALYSIS
  const graha = payload?.graha as Record<string, any>;
  const doshaLines: string[] = [];

  if (graha) {
    // Manglik Dosha — Mars in house 1,4,7,8,12
    const maHouse = graha.Ma?.house_number ?? graha.Ma?.house;
    if (typeof maHouse === 'number' && [1, 4, 7, 8, 12].includes(maHouse)) {
      doshaLines.push(`Manglik Dosha: ACTIVE — Mars in House ${maHouse} (${getRashiName(graha.Ma.rashi)})`);
      // Check for Jupiter mitigation
      const juHouse = graha.Ju?.house_number ?? graha.Ju?.house;
      if (typeof juHouse === 'number' && [1, 5, 9].includes(juHouse)) {
        doshaLines.push(`  Mitigation: Jupiter in ${getRashiName(graha.Ju.rashi)} (House ${juHouse}) aspects/cancels Manglik Dosha`);
      }
    }

    // Kaal Sarp Dosha — all planets between Rahu and Ketu in zodiac order
    const raRashi = graha.Ra?.rashi;
    const keRashi = graha.Ke?.rashi;
    if (typeof raRashi === 'number' && typeof keRashi === 'number' && raRashi !== keRashi) {
      let planetsInside = 0;
      const innerPlanets: string[] = [];
      // Traverse from Ra rashi forward to Ke rashi
      let r = raRashi;
      const target = keRashi + (keRashi <= raRashi ? 12 : 0);
      while (r < target) {
        const checkRashi = ((r - 1) % 12) + 1;
        for (const [pk, pd] of Object.entries(graha)) {
          if (pk === 'Ra' || pk === 'Ke') continue;
          if (pd?.rashi === checkRashi) {
            planetsInside++;
            innerPlanets.push(pk);
          }
        }
        r++;
      }
      if (planetsInside > 0 && innerPlanets.length > 0) {
        doshaLines.push(`Kaal Sarp Dosha: ACTIVE — ${innerPlanets.join(', ')} between Rahu (${getRashiName(raRashi)}) and Ketu (${getRashiName(keRashi)})`);
      }
    }

    // Pitra Dosha — Sun/Saturn/Rahu in 9th house
    const afflicting9: string[] = [];
    for (const pk of ['Su', 'Sa', 'Ra']) {
      const pd = graha[pk];
      const h = pd?.house_number ?? pd?.house;
      if (typeof h === 'number' && h === 9) {
        afflicting9.push(pk);
      }
    }
    if (afflicting9.length > 0) {
      doshaLines.push(`Pitra Dosha: ACTIVE — ${afflicting9.join(', ')} in 9th house`);
    }
  }

  if (doshaLines.length > 0) {
    lines.push('=== DOSHA ANALYSIS ===');
    lines.push(...doshaLines);
  }

  logNodeEnd('extract_special', { lines: lines.length });
  console.log('[extract_special] lines:', lines.length);
  return lines.length ? { findings: [lines.join('\n')] } : { findings: [] };
}

// -- Node: generate_answer --

async function generateAnswerNode(state: AgentStateType): Promise<Partial<AgentStateType>> {
  const question = extractQuestionFromMessages(state.messages || []);
  logNodeStart('generate_answer', { question, findingCount: state.findings.length, mode: state.mode });
  console.log('[generate_answer] input:', { question, findingsLength: state.findings.length, isRetry: (state.qualityRetryCount || 0) > 0 });

  if (!question?.trim()) {
    console.log('[generate_answer] no question');
    return { answer: 'Please tell me about your chart question.', finalAnswer: 'Please tell me about your chart question.' };
  }

  const isRetry = (state.qualityRetryCount || 0) > 0;
  const chartData = state.findings.join('\n\n');
  const memoryContext = (state.relevantMemories || [])
    .slice(-5)
    .map(m => `[${m.role === 'user' ? 'User' : 'Assistant'}] ${m.text}`)
    .join('\n');
  console.log('[generate_answer] chartData length:', chartData.length, 'memoryContext length:', memoryContext.length);

  // Reactive + Bonus instruction
  const modeInstruction = state.mode === 'mini'
    ? 'Keep answers brief (2-3 sentences), focus on the key insight.'
    : 'Provide detailed Vedic astrology analysis. Answer the specific question thoroughly, then optionally add 1-2 brief bonus insights if other chart factors are genuinely relevant. Keep bonus observations short (1-2 sentences each, start with "Bonus insight:" or "Also notable:").';

  const currentDate = new Date().toISOString().slice(0, 10);

  const availableCharts = (state.selectedVargas || ['D1'])
    .map((v: string) => `${v}(${VARGA_NAMES[v] || v})`)
    .join(', ');
  const proVargaMentioned = state.mode === 'mini' && miniMentionedProVarga(question);
  const upgradeNote = proVargaMentioned
    ? `\nCRITICAL: The user asked about a chart not available on Mini plan. You ONLY have D1(Rasi) data. You MUST NOT invent or fabricate positions for other charts. Base your answer entirely on the D1 data below, then in 1 sentence mention that Pro mode unlocks all 16 divisional charts.`
    : '';

  const systemPrompt = `You are an expert Vedic astrologer. Answer the user's question based on the chart data provided.

${modeInstruction}

AVAILABLE CHARTS: ${availableCharts}
YOU ONLY HAVE DATA FOR THE CHARTS LISTED ABOVE. You must invent or fabricate ZERO positions for any other chart. You have no data for charts not listed. Base every claim strictly on the chart data provided below.${upgradeNote}

IMPORTANT RULES:
- Answer the user's specific question first and thoroughly.
- Then, if other chart factors are genuinely relevant to their question, add 1-2 brief bonus observations.
  Format bonus as: "Bonus insight:" or "Also notable:" — keep it to 1-2 sentences.
- Today's date is ${currentDate}. Use this as your reference for all timing (transits, dasha periods, etc). Compute year ranges relative to this date.
- Ground every claim in specific planetary positions from the data.
- NEVER mention missing data, unavailable tools, or backend limitations.
- NEVER suggest consulting a professional astrologer.
- End with actionable insight or forward-looking guidance.
- VARGA CHARTS: Each varga chart line starts with the chart name and its OWN LAGNA (e.g. 'D30: Lagna Scorpio | Su:House12(Libra)...'). All house numbers in that line are relative to THAT varga's lagna only. Transit/ashtakavarga/grahabala/arudha sections use D1 lagna houses. For example, in D30 with Scorpio Lagna, House 2 means the 2nd sign from Scorpio = Sagittarius, NOT the D1 lagna's 2nd house.
- VARGA CHART NAMES: D24 = Siddhamamsa (Chaturvimsamsa, education/wisdom), D30 = Trimamsa (Trisamsa, evil/inauspicious influences). Do NOT call D30 Siddhamamsa or Siddhamsha.
- INDIAN ASTROLOGY CONTEXT: Dosha analysis — Manglik (Mars in 1/4/7/8/12), Kaal Sarp (planets between Ra-Ke), Pitra Dosha (Sun/Saturn/Rahu in 9th). Explain severity based on D1 + D9. For Sade Sati, Saturn transits 12th/1st/2nd from Moon's birth rashi; each phase lasts ~2.5 years; first (12th) = subtle, peak (1st) = intense, last (2nd) = easing. Shani Dhaiya = Saturn in 4th or 8th from Moon. For government jobs/exams, check D10 (career) and D24 (education) — favorable dasha of 10th/6th lord supports exam success. For business vs service: D10 10th lord + 11th for profit suggests business; strong 6th/10th lord connection points to service. For marriage type: D9 + D1 5th/7th — Venus-Jupiter suggests love marriage, Saturn in 7th points to arranged/late marriage. Use Hinglish naturally where appropriate (e.g., "Shani ki mahadasha chal rahi hai" alongside English). REMEDIES: ONLY suggest remedies if the user explicitly asks for upay/remedies/mantra/gemstones/ratna. When asked, use grahabala strength scores to recommend: Su <100 → Ruby (Manik)/Sunday, Mo <80 → Pearl (Moti)/Monday, Ma <80 → Red Coral (Moonga)/Tuesday, Me <80 → Emerald (Panna)/Wednesday, Ju <80 → Yellow Sapphire (Pukhraj)/Thursday, Ve <80 → Diamond (Heera)/Friday, Sa <80 → Blue Sapphire (Neelam)/Saturday, Ra <60 → Hessonite (Gomed), Ke <60 → Cat's Eye (Lehsunia). Methods: mantra japa, ratna, vrat, daan, puja, vastu adjustments. Ashtakavarga: 30+/56 bindus = strong area, <25 = challenges. Kundli Matching (Gun Milan) feature is coming soon — do not fabricate matching analysis.
- PREVIOUS CONVERSATION CONTEXT (from this thread, ranked by relevance): Use this only if the current question references it (e.g., "when will it end" needs "it" resolved). Ignore if the question is self-contained.
${memoryContext}
${isRetry ? '- The previous answer was insufficient. Provide a more detailed, specific, and actionable response.' : ''}

CHART DATA:
${chartData || 'No chart data available. Ask the user for their birth details.'}`;

  console.log('[generate_answer] calling LLM with maxTokens:', state.mode === 'mini' ? 500 : 3000);
  const t0 = Date.now();
  const result = await invokeDeepSeekBedrock({
    systemPrompt,
    userPrompt: `Question: ${question}`,
    maxTokens: state.mode === 'mini' ? 500 : 3000,
  });
  console.log('[generate_answer] LLM took', Date.now() - t0, 'ms, response length:', result.text.length);

  logNodeEnd('generate_answer', { answer: result.text });

  return { answer: result.text };
}

// -- Node: finalize (quality check) --

async function finalizeNode(state: AgentStateType): Promise<Partial<AgentStateType>> {
  const answer = state.answer || '';
  const retries = state.qualityRetryCount || 0;

  logNodeStart('finalize', { answerLength: answer.length, retries });
  console.log('[finalize] input:', { answerLength: answer.length, retries });

  if (!answer?.trim()) {
    console.log('[finalize] empty answer');
    return { finalAnswer: 'I apologize, but I was unable to generate a response.', shouldRegenerate: false, qualityRetryCount: retries };
  }

  if (retries >= 1) {
    console.log('[finalize] max retries reached, accepting answer');
    return { finalAnswer: answer, shouldRegenerate: false, qualityRetryCount: retries };
  }

  const question = extractQuestionFromMessages(state.messages || []);
  console.log('[finalize] calling LLM quality check...');

  try {
    const t0 = Date.now();
    const result = await invokeDeepSeekBedrock({
      systemPrompt: `Review the answer and return JSON:
{
  "finalAnswer": "string (the final answer, possibly refined)",
  "qualityFlags": {
    "addressesQuestion": boolean,
    "noMissingDataLanguage": boolean,
    "hasForwardGuidance": boolean,
    "consistentWithPrior": boolean
  },
  "extractedClaims": string[]
}
No markdown. JSON only.`,
      userPrompt: `Original answer: ${answer.slice(0, 3000)}\nUser question: ${question}`,
      maxTokens: 2048,
    });
    console.log('[finalize] LLM quality check took', Date.now() - t0, 'ms');

    const parsed = parseJsonSafely(result.text);
    if (!parsed || !parsed.qualityFlags) {
      console.log('[finalize] could not parse quality check JSON, accepting answer');
      return { finalAnswer: answer, shouldRegenerate: false, qualityRetryCount: retries };
    }

    const qf = parsed.qualityFlags;
    const needsRegen = !qf.addressesQuestion || !qf.noMissingDataLanguage;

    console.log('[finalize] quality flags:', qf, 'needsRegen:', needsRegen);

    if (needsRegen) {
      logNodeEnd('finalize', { needsRegen: true, retryCount: retries + 1 });
      return {
        finalAnswer: null,
        answer: parsed.finalAnswer || answer,
        shouldRegenerate: true,
        qualityRetryCount: retries + 1,
      };
    }

    console.log('[finalize] quality passed, final answer length:', (parsed.finalAnswer || answer).length);
    logNodeEnd('finalize', { finalAnswer: parsed.finalAnswer || answer });
    return {
      finalAnswer: parsed.finalAnswer || answer,
      shouldRegenerate: false,
      qualityRetryCount: retries,
    };
  } catch (e) {
    console.log('[finalize] quality check error:', String(e));
    return { finalAnswer: answer, shouldRegenerate: false, qualityRetryCount: retries };
  }
}

// -- Build Graph --

function buildGraph() {
  const graph = new StateGraph(AgentState)
    .addNode('route', routeNode)
    .addNode('fast_answer', fastAnswerNode)
    .addNode('load_grounding', loadGroundingNode)

    .addNode('extract_planets', extractPlanetsNode)
    .addNode('extract_varga', extractVargaNode)
    .addNode('extract_timing', extractTimingNode)
    .addNode('extract_special', extractSpecialNode)

    .addNode('generate_answer', generateAnswerNode)
    .addNode('finalize', finalizeNode)

    .addEdge(START, 'route')
    .addConditionalEdges('route', routeAfterRoute, {
      fast_answer: 'fast_answer',
      load_grounding: 'load_grounding',
    })
    .addEdge('fast_answer', END)

    // Mini: only planets. Pro: all 4 extractors run in parallel
    .addConditionalEdges('load_grounding', (state) => {
      const sends: Send<string, Record<string, unknown>>[] = [
        new Send('extract_planets', { grounding: state.grounding }),
      ];
      if (state.mode === 'pro') {
        sends.push(
          new Send('extract_varga', { grounding: state.grounding }),
          new Send('extract_timing', { grounding: state.grounding }),
          new Send('extract_special', { grounding: state.grounding }),
        );
      }
      return sends;
    })

    .addEdge('extract_planets', 'generate_answer')
    .addEdge('extract_varga', 'generate_answer')
    .addEdge('extract_timing', 'generate_answer')
    .addEdge('extract_special', 'generate_answer')

    .addEdge('generate_answer', 'finalize')
    .addConditionalEdges('finalize', (state) => {
      if (state.shouldRegenerate) return 'generate_answer';
      return END;
    });

  return graph;
}

// -- Entry point --

let compiledGraph: any = null;

export async function runKundliAgentV2(
  input: {
    message: string;
    ownerId?: string;
    profileId?: string;
    kundli?: KundliSnapshotInput;
    mode?: 'mini' | 'pro';
    sessionId?: string;
    relevantMemories?: RelevantChatMemory[];
  }
): Promise<{ answer: string; model?: string }> {
  console.log('[runKundliAgentV2] ENTRY', { message: input.message?.slice(0, 100), mode: input.mode, hasKundli: !!input.kundli, sessionId: input.sessionId, memoryCount: input.relevantMemories?.length });

  if (!compiledGraph) {
    console.log('[runKundliAgentV2] building graph for first time...');
    const graph = buildGraph();
    const pgPool = getPostgresPool();
    const cache = pgPool ? new PostgresCache(pgPool) : undefined;

    compiledGraph = graph.compile({
      ...(cache ? { cache } : {}),
    });
    console.log('[runKundliAgentV2] graph compiled');
  }

  if (!input.message?.trim()) {
    console.log('[runKundliAgentV2] empty message');
    return { answer: 'Please provide a message to chat about your chart.', model: 'cozmic-agent-v2' };
  }

  // Clear debug log and log agent start
  logAgentStart({
    message: input.message,
    mode: input.mode || 'mini',
    hasKundli: !!input.kundli,
    sessionId: input.sessionId,
  });

  try {
    const initialState: any = {
      messages: [{ role: 'user', content: input.message }],
      ownerId: input.ownerId || 'anonymous',
      profileId: input.profileId || null,
      mode: input.mode || 'mini',
      kundliInput: input.kundli || null,
      // Reset per-invocation state to prevent duplicate accumulation via checkpointer
      findings: [],
      answer: null,
      finalAnswer: null,
      shouldRegenerate: false,
      qualityRetryCount: 0,
      topLevelRoute: null,
      grounding: null,
      selectedVargas: ['D1'],
      selectedInfolevels: ['basic'],
      fetchTransit: false,
      dashaNesting: input.mode === 'pro' ? 2 : 1,
      timingReasoning: null,
      relevantMemories: input.relevantMemories || [],
    };

    const thread_id = input.sessionId || `anon_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    const config: any = { configurable: { thread_id } };

    console.log('[runKundliAgentV2] invoking graph...');
    const t0 = Date.now();
    const result = await compiledGraph.invoke(initialState, config);
    const elapsed = Date.now() - t0;

    const finalAnswer = result.finalAnswer || result.answer || 'I apologize, but I was unable to generate a response.';
    console.log('[runKundliAgentV2] graph completed in', elapsed, 'ms, answer length:', finalAnswer.length);

    return {
      answer: finalAnswer,
      model: 'cozmic-agent-v2',
    };
  } catch (error) {
    console.log('[runKundliAgentV2] graph invoke error:', (error as Error).message);
    return {
      answer: `I encountered an error: ${(error as Error).message}`,
      model: 'cozmic-agent-v2',
    };
  }
}

// Exports for testing
export { isGeneralAstroQuestion, selectVargas, selectInfolevelsAndTransit, miniMentionedProVarga };
