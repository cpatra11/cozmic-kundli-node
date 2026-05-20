import { Annotation, END, START, StateGraph, MessagesAnnotation } from '@langchain/langgraph';
import type { KundliSnapshotInput } from './be1Client.js';
import { fetchBe1Calculate, fetchBe1Transit } from './be1Client.js';
import { invokeDeepSeekBedrock, invokeDeepSeekConversation } from './deepseekBedrock.js';
import { CHART_TOOL_CONFIG } from '../utils/toolDefs.js';
import {
  formatGrahaBhavaLagna, formatVargaCharts, formatPanchanga,
  formatDashaTimeline, formatYogas, formatAshtakavarga,
  formatGrahabala, formatArudha, formatDoshaAnalysis,
  formatTransitSnapshots, formatSadeSati,
  computeNatalHouseMap, getRashiName,
} from '../utils/formatChartData.js';
import { getPostgresPool } from './postgresClient.js';
import { PostgresCache } from './postgresCache.js';
import { logNodeStart, logNodeEnd, logAgentStart, appendDebugLog } from '../utils/debugLog.js';
import type { RelevantChatMemory } from './chatMemory.js';

export interface ChatHistoryMessage {
  role: 'user' | 'assistant';
  message: string;
}

// -- State --

const AgentState = Annotation.Root({
  ...MessagesAnnotation.spec,

  ownerId: Annotation<string>,
  profileId: Annotation<string | null>,
  mode: Annotation<'mini' | 'pro'>,
  isExplicitMode: Annotation<boolean>,
  kundliInput: Annotation<KundliSnapshotInput | null>,

  topLevelRoute: Annotation<'pipeline' | 'smalltalk' | 'general_astro' | null>,

  chartData: Annotation<Record<string, unknown> | null>,
  transitSnapshots: Annotation<Record<string, Record<string, unknown>> | null>,

  answer: Annotation<string | null>,
  finalAnswer: Annotation<string | null>,

  qualityRetryCount: Annotation<number>,
  shouldRegenerate: Annotation<boolean>,

  relevantMemories: Annotation<RelevantChatMemory[]>,

  conversationHistory: Annotation<ChatHistoryMessage[]>,
  needsContext: Annotation<boolean>,
  contextMessages: Annotation<ChatHistoryMessage[]>,
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

async function classifyQuestionMode(question: string): Promise<'mini' | 'pro'> {
  try {
    const result = await invokeDeepSeekBedrock({
      systemPrompt: `Classify the user's Vedic astrology question as one of:
- "mini": Simple question answerable with D1 (Rasi) chart only. Examples: "What is my lagna?", "Tell me about my chart", "Read my kundli", "Basic analysis", "What is my moon sign?"
- "pro": Complex question needing D9/D10/D60 charts, timing/dasha analysis, transit details, or specific life area depth. Examples: "When will I get married?", "Career prospects analysis", "Health issues timeline", "Marriage compatibility", "Detailed analysis with D9 and D10", "What does my D9 chart say?"

Return JSON: {"mode": "mini" | "pro"}
No markdown. JSON only.`,
      userPrompt: question,
      maxTokens: 64,
    });
    const parsed = parseJsonSafely(result.text);
    if (parsed?.mode === 'pro') return 'pro';
  } catch {
    // fall through
  }
  return 'mini';
}

// -- Fast routing --

function isObviousSmalltalk(q: string): boolean {
  return /^(hello|hi|hey|thanks|thank you|bye|goodbye|great|ok|okay|nice|good)\b/i.test(q.trim());
}

function isIdentityQuestion(q: string): boolean {
  return /\b(who are you|what are you|what can you|your name|tell me about yourself|capabilities|help)\b/i.test(q);
}

// -- Node: route --

async function routeNode(state: AgentStateType): Promise<Partial<AgentStateType>> {
  const question = extractQuestionFromMessages(state.messages || []);
  logNodeStart('route', { question, mode: state.mode });

  if (!question?.trim()) {
    return { topLevelRoute: 'smalltalk', answer: 'Hello! Please share your birth details (date, time, place) or ask about your chart.', finalAnswer: '' };
  }

  if (isObviousSmalltalk(question)) return { topLevelRoute: 'smalltalk' };
  if (isIdentityQuestion(question)) return { topLevelRoute: 'general_astro' };

  // LLM classifier for remaining cases
  try {
    const result = await invokeDeepSeekBedrock({
      systemPrompt: `Classify the user's question into exactly one:
- "pipeline": personal chart question — ANY question using "my/mine" with chart/kundli/D1/D9/D10/horoscope/birth/lagna, or asking about personal life areas (career/marriage/health/finance timing). Examples: "Analyse my D1 chart" = pipeline. "Will I get married?" = pipeline. "Read my kundli" = pipeline. If unsure, use pipeline.
- "general_astro": conceptual question about astrology with NO personal reference — "what is D9 chart?", "explain manglik dosha", "how does vimshottari dasha work?"
- "smalltalk": greeting, chitchat, thank you, feedback, off-topic

Return JSON: {"route": "pipeline" | "general_astro" | "smalltalk"}
No markdown. JSON only.`,
      userPrompt: question,
      maxTokens: 64,
    });
    const parsed = parseJsonSafely(result.text);
    const route = parsed?.route;
    if (route === 'pipeline' || route === 'general_astro') {
      return { topLevelRoute: route };
    }
  } catch {
    // fall through to default
  }

  return { topLevelRoute: 'pipeline' };
}

function routeAfterRoute(state: AgentStateType): string {
  switch (state.topLevelRoute) {
    case 'pipeline': return 'agent';
    default: return 'fast_answer';
  }
}

// -- Node: classify_context (determine if follow-up context needed) --

const EXPLICIT_CONTEXT_PATTERNS = [
  /you (mentioned|said|wrote|talked|stated|indicated|replied|answered)/i,
  /what did you mean by/i,
  /tell me more about/i,
  /earlier you/i,
  /previously you/i,
  /as you said/i,
  /as mentioned/i,
  /regarding (the|your|that|this)/i,
  /go deeper into/i,
  /explain further/i,
  /elaborate on/i,
];

const AMBIGUOUS_CONTEXT_PATTERNS = [
  /^(why|how|what|when|where)\b/i,
  /^(and\b|but\b|so\b|then\b)/i,
  /\b(it|that|this|those|these)\b/i,
  /^(is it|are they|does it|can you|could you|would you)/i,
];

function isObviousFollowUp(question: string): boolean {
  return EXPLICIT_CONTEXT_PATTERNS.some(p => p.test(question));
}

function isPossiblyFollowUp(question: string): boolean {
  if (question.trim().length < 30) return true;
  return AMBIGUOUS_CONTEXT_PATTERNS.some(p => p.test(question));
}

function getLastContextualMessages(
  history: ChatHistoryMessage[],
  maxTurns: number
): ChatHistoryMessage[] {
  const relevant: ChatHistoryMessage[] = [];
  for (let i = history.length - 1; i >= 0 && relevant.length < maxTurns * 2; i--) {
    relevant.unshift(history[i]);
  }
  return relevant;
}

async function classifyContextNode(state: AgentStateType): Promise<Partial<AgentStateType>> {
  const question = extractQuestionFromMessages(state.messages || []);
  const history = state.conversationHistory || [];
  logNodeStart('classify_context', { question, historyLength: history.length });

  if (!question?.trim() || history.length < 2) {
    return { needsContext: false, contextMessages: [] };
  }

  if (isObviousFollowUp(question)) {
    const contextMessages = getLastContextualMessages(history, 2);
    logNodeEnd('classify_context', { needsContext: true, reason: 'explicit_pattern', messages: contextMessages.length });
    return { needsContext: true, contextMessages };
  }

  if (!isPossiblyFollowUp(question)) {
    return { needsContext: false, contextMessages: [] };
  }

  // Ambiguous — use cheap LLM classifier
  const lastExchange = history.slice(-2).map(m =>
    `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.message.slice(0, 200)}`
  ).join('\n');

  try {
    const result = await invokeDeepSeekBedrock({
      systemPrompt: `You are a classifier. Determine if the user's new question is a follow-up to the previous conversation or a new standalone question.
- Return "follow-up" if the question refers to or builds upon something mentioned in the previous exchange.
- Return "new" if the question is self-contained and does not depend on prior context.
- Return JSON: {"verdict": "follow-up" | "new"}
No markdown. JSON only.`,
      userPrompt: `Previous exchange:\n${lastExchange}\n\nNew question: ${question}`,
      maxTokens: 128,
    });

    const parsed = parseJsonSafely(result.text);
    const needsContext = parsed?.verdict === 'follow-up';

    if (needsContext) {
      const contextMessages = getLastContextualMessages(history, 2);
      logNodeEnd('classify_context', { needsContext: true, reason: 'llm_classifier', messages: contextMessages.length });
      return { needsContext, contextMessages };
    }

    logNodeEnd('classify_context', { needsContext: false, reason: 'llm_classifier' });
    return { needsContext: false, contextMessages: [] };
  } catch {
    // On error, don't inject context
    return { needsContext: false, contextMessages: [] };
  }
}

// -- Node: fast_answer (smalltalk / general_astro) --

async function fastAnswerNode(state: AgentStateType): Promise<Partial<AgentStateType>> {
  if (state.finalAnswer) return {};
  try {
    const question = extractQuestionFromMessages(state.messages || []);
    const isSmallTalk = state.topLevelRoute === 'smalltalk';

    if (!question?.trim()) {
      return { finalAnswer: 'Hello! How can I help you with your chart today?' };
    }

    const result = await invokeDeepSeekBedrock({
      systemPrompt: isSmallTalk
        ? 'You are Cozmic, a friendly Vedic astrology assistant. Respond naturally to casual chat. Keep it brief and warm.'
        : 'You are Cozmic, a Vedic astrology assistant. Answer conceptual questions clearly and concisely (3-5 paragraphs max).',
      userPrompt: question,
      maxTokens: 1024,
    });

    return { finalAnswer: result.text };
  } catch (error) {
    console.error('[fastAnswerNode] failed', { message: (error as Error).message, stack: (error as Error).stack?.split('\n').slice(0, 4).join('\n') });
    return { finalAnswer: 'I apologize, but I was unable to process your request.' };
  }
}

// -- Tool execution --

const NESTING_NAMES: Record<number, string> = {
  1: 'mahadasha', 2: 'antardasha', 3: 'pratyantardasha',
  4: 'sookshmantardasha', 5: 'pranantardasha',
};

function truncatePastDashaPeriods(payload: any, maxAgeYears = 5): any {
  const target = payload?.chart || payload;
  if (!target?.dasha?.periods) return payload;

  const cutoff = Date.now() - maxAgeYears * 365.25 * 86400000;
  const now = Date.now();
  let omittedCount = 0;

  function walk(node: any): any {
    if (!node?.periods) return node;
    const result: any = { ...node, periods: {} };
    const keys = Object.keys(node.periods);
    for (const key of keys) {
      const period = node.periods[key] as any;
      const pEnd = new Date(period.end).getTime();
      const pStart = new Date(period.start).getTime();
      if (pEnd > cutoff || pStart > now || (pStart <= now && pEnd > now)) {
        result.periods[key] = walk(period);
      } else {
        omittedCount++;
      }
    }
    return result;
  }

  const modified = walk(target);
  if (omittedCount > 0) {
    modified.dasha._omittedPastPeriods = omittedCount;
  }
  const resultPayload = structuredClone(payload);
  const rTarget = resultPayload?.chart || resultPayload;
  rTarget.dasha = modified.dasha;
  return resultPayload;
}

function findActiveDashaPeriods(
  payload: any
): Array<{ lord: string; level: string; start: string; end: string; nesting: number }> {
  const p = payload?.chart || payload;
  const dashaRoot = p?.dasha;
  if (!dashaRoot?.periods) return [];

  const now = Date.now();
  const active: Array<{ lord: string; level: string; start: string; end: string; nesting: number }> = [];

  function walk(node: any, depth: number) {
    if (!node?.periods) return;
    const entries = Object.entries(node.periods) as Array<[string, any]>;
    for (const [key, period] of entries) {
      const pStart = new Date(period.start).getTime();
      const pEnd = new Date(period.end).getTime();
      if (pStart <= now && pEnd > now) {
        active.push({
          lord: key,
          level: NESTING_NAMES[period.nesting] || `level_${depth}`,
          start: period.start,
          end: period.end,
          nesting: period.nesting || depth,
        });
      }
      walk(period, depth + 1);
    }
  }

  walk(dashaRoot, 1);
  return active;
}

function generateTimingHints(
  planetsPayload: Record<string, unknown>,
  activePeriods: Array<{ lord: string; level: string; start: string; end: string; nesting: number }>,
  transitSnapshots?: Record<string, Record<string, unknown>>
): string {
  const payload = (planetsPayload as any)?.chart || planetsPayload;
  const graha = payload?.graha as Record<string, any> | undefined;
  const bhava = payload?.bhava as Record<string, any> | undefined;
  const lagna = payload?.lagna?.Lg || payload?.lagna;
  const natalLagnaRashi = lagna?.rashi as number | undefined;

  if (!graha || !bhava || !activePeriods.length) return '';

  const lines: string[] = ['📊 Dasha Lord → Bhava Ownership → Gochar Activation'];

  for (const period of activePeriods) {
    const lordData = graha[period.lord];
    if (!lordData) continue;

    const lordRashi = lordData.rashi as number | undefined;
    const lordNakshatra = lordData.nakshatra as any;
    const nkStr = lordNakshatra?.name ? ` (${lordNakshatra.name})` : '';

    // Find which houses this lord owns in the bhava map
    const ownedHouses: number[] = [];
    for (const [bhNum, bhData] of Object.entries(bhava)) {
      const lordKey = (bhData as any)?.lord;
      if (lordKey === period.lord) {
        ownedHouses.push(parseInt(bhNum));
      }
    }

    let line = `${period.lord} ${period.level}: Rashi ${lordRashi ?? '?'}${nkStr}`;
    if (ownedHouses.length > 0) {
      line += ` | Svāmī of houses ${ownedHouses.join(', ')}`;
    }
    if (period.start && period.end) {
      const start = period.start.slice(0, 10);
      const end = period.end.slice(0, 10);
      line += ` | Period: ${start} → ${end}`;
    }
    lines.push(line);
  }

  // If transit data is missing, note it so the LLM knows to fetch it
  if (!transitSnapshots || Object.keys(transitSnapshots).length === 0) {
    lines.push('');
    lines.push('[Gochar data not yet fetched. Call fetch_transit with today\'s date for current planetary transit positions.]');
  }

  // Add transit activation hints if transit data is available
  if (transitSnapshots && Object.keys(transitSnapshots).length > 0 && typeof natalLagnaRashi === 'number') {
    const dateKeys = Object.keys(transitSnapshots).sort();
    lines.push('');
    lines.push('🔄 Current Gochar Activation:');
    for (const dateKey of dateKeys) {
      const snapData = transitSnapshots[dateKey];
      const grahaData = snapData?.graha as Record<string, any> | undefined;
      if (!grahaData) continue;

      const houseMap = computeNatalHouseMap(grahaData, natalLagnaRashi);
      const isCurrent = dateKey === new Date().toISOString().slice(0, 10);
      const dateLabel = isCurrent ? `${dateKey} (current)` : dateKey;
      lines.push(`  ${dateLabel}:`);

      for (const [planet, data] of Object.entries(grahaData)) {
        const h = houseMap?.[planet];
        const r = data?.rashi as number | undefined;
        if (h) {
          lines.push(`    ${planet}: House ${h} (${r ? getRashiName(r) : '?'})`);
        }
      }
    }
  }

  return lines.join('\n');
}

async function executeFetchSection(
  kundli: KundliSnapshotInput | null,
  section: string,
  input: Record<string, unknown>,
  options?: { mode?: 'mini' | 'pro'; isExplicitMode?: boolean }
): Promise<Record<string, unknown>> {
  if (!kundli) {
    return { error: 'No birth details available. Please provide birth date, time, and place.' };
  }

  const vargas = (input.vargas as string[]) || ['D1'];
  const nesting = Math.max(1, Math.min(5, (input.nesting as number) ?? 2));

  // Hard block non-D1 vargas only in explicit mini mode (caller explicitly chose mini)
  const isHardMini = options?.mode === 'mini' && options?.isExplicitMode;
  if (isHardMini && section === 'basic' && vargas.some(v => v !== 'D1')) {
    return { error: 'Mini mode: D1 chart only. Upgrade to Pro for multi-varga access.' };
  }

  // lordshipVarga validation for dasha
  const lordshipVarga = section === 'dasha' ? ((input.lordshipVarga as string) || 'D1') : undefined;
  if (section === 'dasha' && lordshipVarga && !/^D\d+$/i.test(lordshipVarga)) {
    return { error: `Invalid lordshipVarga '${lordshipVarga}'. Must be like D1, D9, D10, etc.` };
  }

  try {
    // Map section name to backend infolevel
    const infolevelMap: Record<string, string> = {
      basic: 'basic',
      panchanga: 'panchanga',
      yogas: 'yogas',
      dasha: 'dasha',
      ashtakavarga: 'ashtakavarga',
      grahabala: 'grahabala',
      arudha: 'arudha',
    };
    const infolevel = infolevelMap[section];
    if (!infolevel) {
      return { error: `Unknown section: ${section}` };
    }

    // For dasha, include lordshipVarga in vargas so raw payload has bhava data
    const effectiveVargas = section === 'dasha' && lordshipVarga
      ? Array.from(new Set([...vargas, lordshipVarga]))
      : vargas;

    const apiResponse = await fetchBe1Calculate(kundli, {
      varga: effectiveVargas.join(','),
      infolevel,
      nesting: section === 'dasha' ? nesting : 1,
    });

    let formattedData = '';
    switch (section) {
      case 'basic': {
        formattedData = formatGrahaBhavaLagna(apiResponse);
        if (vargas.length > 1) {
          const vc = formatVargaCharts(apiResponse);
          if (vc) formattedData += '\n\n' + vc;
        }
        const dosha = formatDoshaAnalysis(apiResponse);
        if (dosha) formattedData += '\n\n' + dosha;
        break;
      }
      case 'panchanga':
        formattedData = formatPanchanga(apiResponse);
        break;
      case 'yogas':
        formattedData = formatYogas(apiResponse);
        break;
      case 'dasha': {
        // Truncate past periods for display, keep full raw payload for hints
        const truncated = truncatePastDashaPeriods(structuredClone(apiResponse));
        formattedData = formatDashaTimeline(truncated, nesting);
        break;
      }
      case 'ashtakavarga':
        formattedData = formatAshtakavarga(apiResponse);
        break;
      case 'grahabala':
        formattedData = formatGrahabala(apiResponse);
        break;
      case 'arudha':
        formattedData = formatArudha(apiResponse);
        break;
    }

    return {
      section,
      vargas: effectiveVargas,
      infolevel,
      nesting,
      lordshipVarga,
      formattedData: formattedData || `No ${section} data returned.`,
      rawPayload: apiResponse as Record<string, unknown>,
      rawAvailable: true,
    };
  } catch (error) {
    return { error: `Failed to fetch ${section} data: ${String(error)}` };
  }
}

async function executeFetchTransit(
  kundli: KundliSnapshotInput | null,
  input: Record<string, unknown>,
  chartPayload?: Record<string, unknown>
): Promise<Record<string, unknown>> {
  if (!kundli) {
    return { error: 'No birth details available.' };
  }

  const dateStrings = (input.dates as string[]) || [];
  if (!dateStrings.length) {
    return { error: 'No dates provided for transit fetch.' };
  }

  try {
    const dates = dateStrings.map(d => new Date(d));
    const transitNesting = Math.max(1, Math.min(5, (input.nesting as number) ?? 2));
    const transitResults = await Promise.all(
      dates.map(date => fetchBe1Transit(kundli, date, { nesting: transitNesting }))
    );

    const snapshots: Record<string, Record<string, unknown>> = {};
    for (let i = 0; i < dates.length; i++) {
      const dateKey = dates[i].toISOString().slice(0, 10);
      const obj = transitResults[i] as any;
      const chart = obj?.chart || obj;
      const transitData = chart?.transit || chart;
      if (transitData?.graha) {
        const cleanedGraha: Record<string, unknown> = {};
        for (const [planet, data] of Object.entries(transitData.graha as Record<string, any>)) {
          const { house_number, ...rest } = data;
          cleanedGraha[planet] = rest;
        }
        snapshots[dateKey] = { graha: cleanedGraha };
      }
    }

    // Format transit data with natal-lagna-relative house positions
    let transitText = '';
    if (chartPayload && Object.keys(snapshots).length > 0) {
      transitText = formatTransitSnapshots(chartPayload, snapshots);

      // Gate Sade Sati: only compute when Saturn is near Moon's natal rashi
      const p = ((chartPayload as any)?.chart || chartPayload) as any;
      const graha = p?.graha as Record<string, any> | undefined;
      const moonRashi = graha?.Mo?.rashi as number | undefined;
      if (typeof moonRashi === 'number') {
        const hasSaturnNearMoon = Object.values(snapshots).some(snap => {
          const satRashi = (snap?.graha as any)?.Sa?.rashi as number | undefined;
          if (typeof satRashi !== 'number') return false;
          const diff = ((satRashi - moonRashi) + 12) % 12;
          return diff <= 2 || diff >= 10; // ±2 rashi from Moon
        });
        if (hasSaturnNearMoon) {
          const sadeSati = formatSadeSati(chartPayload, snapshots);
          if (sadeSati) transitText += '\n' + sadeSati;
        }
      }
    }

    return {
      transitText: transitText || 'No transit data available for the requested dates.',
      dates: dateStrings,
      _snapshots: snapshots,
    };
  } catch (error) {
    return { error: `Failed to fetch transit data: ${String(error)}` };
  }
}

async function executeSearchAstrology(query: string): Promise<Record<string, unknown>> {
  try {
    const result = await invokeDeepSeekBedrock({
      systemPrompt: 'You are a Vedic astrology teacher. Explain the following concept concisely and accurately in 2-3 paragraphs. Focus on classical principles.',
      userPrompt: query,
      maxTokens: 1024,
    });

    return { explanation: result.text, query };
  } catch (error) {
    return { error: `Search failed: ${String(error)}`, query };
  }
}

// -- System prompt builder --

function buildAgentSystemPrompt(state: AgentStateType, chartDataCached: boolean): string {
  const question = extractQuestionFromMessages(state.messages || []);
  const memories = state.relevantMemories || [];
  const isRetry = (state.qualityRetryCount || 0) > 0;
  const currentDate = new Date().toISOString().slice(0, 10);

  const modeInstruction = state.mode === 'mini'
    ? 'Keep answers short (1-2 sentences). Answer ONLY what was asked. Never add unsolicited details.'
    : 'Answer the specific question directly and concisely. Provide supporting data only when it proves your answer. If the question is narrow (e.g. "What is my lagna?"), answer narrow — do not dump the entire chart. Add at most 1 brief bonus insight (1 sentence, start with "Bonus:"), and only if genuinely relevant to the question topic.';

  const hasKundli = !!state.kundliInput;
  const profileNote = state.profileId ? `\nUser has a saved birth chart (profileId: ${state.profileId}).` : '';

  const upgradeNote = state.mode === 'mini'
    ? `\nMini mode: D1 (Rasi) chart data available. If the question needs divisional charts (D9, D10, etc.), work with what you have from D1 — do not mention mode limitations or suggest upgrades.`
    : '';

  const retryNote = isRetry
    ? '\nThe previous answer was insufficient. Provide a more detailed, specific, and actionable response. Address the question directly with concrete chart references.'
    : '';

  const dataNote = chartDataCached
    ? '\nChart data was already fetched in a previous attempt and is available in your conversation history below. You do not need to call fetch tools again unless you need different parameters. Simply review the existing data and improve the answer.'
    : '';

  return `You are Cozmic, an expert Vedic astrologer. You analyse Vedic astrology birth charts (kundli/horoscope). "D1" means the Rasi chart (main birth chart), NOT a financial trading chart. The user is ALWAYS asking about Vedic astrology.

${modeInstruction}
${profileNote}
${dataNote}
${upgradeNote}
${retryNote}


TOOL GUIDE:
- fetch_planets([vargas]): planet positions, houses, lagna, dosha. D1 = main Vedic birth chart.
- fetch_dasha({nesting, lordshipVarga}): Vimshottari dasha timeline.
  nesting=1 (Mahadasha, ~2KB, years-level), nesting=2 (Antardasha, ~13KB, months-years),
  nesting=3 (Pratyantardasha, ~200KB, weeks-months), nesting=4 (Sookshmantardasha, days-weeks),
  nesting=5 (Pranantardasha, ~very large). Choose based on time granularity needed.
  lordshipVarga: D1 (general), D9 (marriage), D10 (career).
- fetch_transit({dates, nesting}): current sky positions for specific dates.
- fetch_ashtakavarga, fetch_grahabala: planetary/house strength.
- fetch_yogas: special planetary combinations.
- fetch_arudha: perception/reflection points.
- fetch_panchanga: daily panchanga (tithi, nakshatra, yoga).
- search_astrology: lookup astrology concepts.

FETCH STRATEGY:
- Fetch only what's needed. Start with fetch_planets (D1 covers most questions).
- For timing questions: fetch_planets + fetch_dasha together. Choose nesting based on how specific the timing needs to be.
- After analyzing dasha, call fetch_transit with relevant dates.
- Need deeper detail? Re-fetch dasha at higher nesting or fetch transit for more dates.
- Need a different varga? Re-fetch planets with the needed vargas.
- You have enough iterations for multi-step analysis — plan your fetches across iterations.

ANSWER PROPORTIONALLY:
- Answer ONLY what the user asked. If they ask about lagna, tell them the lagna — not all 9 planets.
- Match the scope of your answer to the scope of the question. Narrow question → narrow answer.
- Supporting data is for proving your claim, not for dumping everything you know.
- Before responding, ask yourself: "Does this detail directly answer the question?" If not, leave it out.

TIMING INTERPRETATION RULES:
1. Daśā svāmī (dasha lord) activates the bhavas it owns in the selected varga
2. Gochar (transit) shows where this activation is currently manifesting
3. Strong event indication: dasha lord + key planet (Guru/Shani) both transiting the same relevant bhava
4. Weak indication: dasha lord owns the bhava but transiting an unrelated bhava
5. Sade Sati: Shani 12ve/1le/2re bhava se Chandra rashi ka gochar — each phase ~2.5 years

RULES:
- A graha placed in a bhava does NOT make it the lord of that bhava. Lordship is determined by the rashi on the bhava cusp.
- NEVER mention missing data, unavailable tools, or backend limitations.
- NEVER suggest consulting a professional astrologer.
- NEVER invent or fabricate planetary positions.
- Ground every claim in specific data. CITATION: "Shani 10ve bhava mein (Makara, 15°42')"
- End with actionable insight or forward-looking guidance.
- VARGA CHARTS: Bhava numbers are relative to THAT varga's own lagna, NOT D1's lagna.
- Remedies: ONLY suggest when user explicitly asks for upay/remedies/mantra/gemstones.
- Today's date: ${currentDate}. Use as reference for all timing.`;
}

// -- Node: agent (ReAct loop with tool calling) --

async function agentNode(state: AgentStateType): Promise<Partial<AgentStateType>> {
  try {
    const question = extractQuestionFromMessages(state.messages || []);
    logNodeStart('agent', { question, mode: state.mode, hasKundli: !!state.kundliInput });

    if (!question?.trim()) {
      return { answer: 'Please tell me about your chart question.' };
    }

    const systemPrompt = buildAgentSystemPrompt(state, !!state.chartData);
    const maxIterations = 5;

    // Inject conversation context as actual messages for pronoun resolution
    const messages: Array<{ role: string; content: Array<Record<string, unknown>> }> = [];
    if (state.needsContext && state.contextMessages?.length) {
      for (const ctx of state.contextMessages) {
        messages.push({
          role: ctx.role === 'assistant' ? 'assistant' : 'user',
          content: [{ text: ctx.message }],
        });
      }
    }
    messages.push({ role: 'user', content: [{ text: question }] });

    // Accumulated data across iterations for cross-section correlation
    let collectedChartPayload: Record<string, unknown> | null = state.chartData || null;
    const sectionRawPayloads: Record<string, Record<string, unknown>> = {};

    // Seed sectionRawPayloads from state.chartData if available
    if (state.chartData && (state.chartData as any).rawPayload) {
      sectionRawPayloads.basic = (state.chartData as any).rawPayload as Record<string, unknown>;
    }

    let collectedTransitSnapshots: Record<string, Record<string, unknown>> | null = state.transitSnapshots || null;
    let timingHintsInjected = false;
    let lastTransitSnapshotCount = state.transitSnapshots ? Object.keys(state.transitSnapshots).length : 0;
    const calledTools: Set<string> = new Set();

    for (let iteration = 0; iteration < maxIterations; iteration++) {
      let response;
      try {
        response = await invokeDeepSeekConversation({
          systemPrompt,
          messages: messages as any,
          tools: CHART_TOOL_CONFIG,
          maxTokens: 4096,
        });
      } catch (err) {
        if (iteration === 0) {
          // Retry once on transient API errors
          response = await invokeDeepSeekConversation({
            systemPrompt,
            messages: messages as any,
            tools: CHART_TOOL_CONFIG,
            maxTokens: 4096,
          });
        } else {
          throw err;
        }
      }

      const toolUseBlocks = response.content.filter(c => c.toolUse);
      const textBlocks = response.content.filter(c => c.text);

      if (toolUseBlocks.length === 0) {
        // First iteration with no chart data fetched — instruct the LLM to fetch data before answering.
        if (iteration === 0 && !collectedChartPayload && !state.chartData) {
          messages.push({
            role: 'user',
            content: [{ text: 'This is a personal chart question. You must call fetch_planets (or other relevant data tool) before answering. Do this now with appropriate parameters for the question.' }],
          });
          continue;
        }

        // Transit question but no transit data fetched — force fetch
        if (collectedChartPayload && !collectedTransitSnapshots && iteration < maxIterations - 1) {
          const q = (question || '').toLowerCase();
          if (/\b(transit|gochar|current|today)\b/.test(q)) {
            messages.push({
              role: 'user',
              content: [{ text: 'You have chart data but not current transit positions. The user is asking about transits. Call fetch_transit with today\'s date now before answering.' }],
            });
            continue;
          }
        }

        // Post-processing: inject timing hints if we have planets + dasha + transit
        if (!timingHintsInjected && collectedChartPayload) {
          const planetsRaw = sectionRawPayloads.basic;
          const dashaRaw = sectionRawPayloads.dasha;
          const rawForPeriods = dashaRaw || planetsRaw;
          if (planetsRaw && rawForPeriods) {
            const activePeriods = findActiveDashaPeriods(rawForPeriods);
            const allTransitSnapshots = collectedTransitSnapshots || {};
            const hints = generateTimingHints(
              planetsRaw,
              activePeriods,
              Object.keys(allTransitSnapshots).length > 0 ? allTransitSnapshots : undefined
            );
            if (hints) {
              messages.push({
                role: 'user',
                content: [{ text: '[REFERENCE DATA: Dasha × Gochar Viśleṣaṇ]\n' + hints }],
              });
              if (Object.keys(allTransitSnapshots).length > 0) {
                timingHintsInjected = true;
                const currentCount = Object.keys(allTransitSnapshots).length;
                if (currentCount > lastTransitSnapshotCount) {
                  timingHintsInjected = false; // re-inject if new transit data arrived
                }
                lastTransitSnapshotCount = currentCount;
              }
            }
          }
        }

        // LLM finished — text response is the answer
        let answerText = textBlocks.map(c => c.text).join('').trim();

        // If LLM returned empty despite having data, prod it to respond
        if (!answerText && collectedChartPayload && iteration < maxIterations - 1) {
          messages.push({
            role: 'user',
            content: [{ text: 'Please provide your analysis based on the chart data and reference data above.' }],
          });
          continue;
        }

        logNodeEnd('agent', { answerLength: answerText.length, iterations: iteration + 1 });
        return {
          answer: answerText,
          chartData: collectedChartPayload,
          transitSnapshots: collectedTransitSnapshots,
        };
      }

      // Process tool calls
      const assistantContent: Array<Record<string, unknown>> = [];
      for (const block of toolUseBlocks) {
        if (block.toolUse) {
          assistantContent.push({ toolUse: block.toolUse });
        }
      }
      if (textBlocks.length > 0) {
        assistantContent.push({ text: textBlocks.map(c => c.text).join('') });
      }

      messages.push({ role: 'assistant', content: assistantContent });

      // Execute all tool calls in parallel
      const toolResults = await Promise.all(
        toolUseBlocks.map(async (block) => {
          const tu = block.toolUse;
          if (!tu) return null;

          const toolSig = `${tu.name}:${JSON.stringify(tu.input)}`;
          if (calledTools.has(toolSig)) {
            return {
              toolUseId: tu.toolUseId,
              content: [{ toolResult: { toolUseId: tu.toolUseId, content: [{ text: 'Already fetched with identical parameters. Use existing data.' }] } }],
            };
          }
          calledTools.add(toolSig);

          let result: Record<string, unknown>;

          switch (tu.name) {
            case 'fetch_planets':
            case 'fetch_panchanga':
            case 'fetch_yogas':
            case 'fetch_dasha':
            case 'fetch_ashtakavarga':
            case 'fetch_grahabala':
            case 'fetch_arudha': {
              const sectionMap: Record<string, string> = {
                fetch_planets: 'basic',
                fetch_panchanga: 'panchanga',
                fetch_yogas: 'yogas',
                fetch_dasha: 'dasha',
                fetch_ashtakavarga: 'ashtakavarga',
                fetch_grahabala: 'grahabala',
                fetch_arudha: 'arudha',
              };
              const section = sectionMap[tu.name] || 'basic';
              result = await executeFetchSection(state.kundliInput, section, tu.input, {
                mode: state.mode,
                isExplicitMode: state.isExplicitMode,
              });
              if (result.rawPayload) {
                sectionRawPayloads[section] = result.rawPayload as Record<string, unknown>;
                const rawPayloadsCopy = { ...sectionRawPayloads };
                collectedChartPayload = { ...(collectedChartPayload || {}), ...result, _rawPayloads: rawPayloadsCopy };
              }
              break;
            }
            case 'fetch_transit': {
              const natalPayload = sectionRawPayloads.basic;
              result = await executeFetchTransit(state.kundliInput, tu.input, natalPayload);
              const snapshots = result._snapshots as Record<string, Record<string, unknown>> | undefined;
              if (snapshots && Object.keys(snapshots).length > 0) {
                collectedTransitSnapshots = { ...(collectedTransitSnapshots || {}), ...snapshots };
              }
              // Strip internal fields from result exposed to LLM
              const { _snapshots: _ss, ...cleanResult } = result;
              result = cleanResult;
              break;
            }
            case 'search_astrology': {
              const query = typeof tu.input?.query === 'string' ? tu.input.query : JSON.stringify(tu.input);
              result = await executeSearchAstrology(query);
              break;
            }
            default: {
              result = { error: `Unknown tool: ${tu.name}` };
            }
          }

          return {
            toolUseId: tu.toolUseId,
            content: [{ toolResult: { toolUseId: tu.toolUseId, content: [{ json: result }] } }],
          };
        })
      );

      // Combine all tool results into a single user message (Bedrock requires all toolResults
      // for one assistant message to be in a single user message with multiple content blocks)
      const allToolContent: Array<Record<string, unknown>> = toolResults
        .filter(tr => tr !== null)
        .flatMap(tr => tr!.content as Array<Record<string, unknown>>);
      if (allToolContent.length > 0) {
        messages.push({
          role: 'user',
          content: allToolContent,
        });
      }

      // Post-processing: inject timing hints after each iteration when planets + dasha + transit data exist
      if (!timingHintsInjected && collectedChartPayload) {
        const planetsRaw = sectionRawPayloads.basic;
        const dashaRaw = sectionRawPayloads.dasha;
        const rawForPeriods = dashaRaw || planetsRaw;
        if (planetsRaw && rawForPeriods) {
          const activePeriods = findActiveDashaPeriods(rawForPeriods);
          const allTransitSnapshots = collectedTransitSnapshots || {};
          const hints = generateTimingHints(
            planetsRaw,
            activePeriods,
            Object.keys(allTransitSnapshots).length > 0 ? allTransitSnapshots : undefined
          );
          if (hints) {
            messages.push({
              role: 'user',
              content: [{ text: '[REFERENCE DATA: Dasha × Gochar Viśleṣaṇ]\n' + hints }],
            });
            if (Object.keys(allTransitSnapshots).length > 0) {
              timingHintsInjected = true;
              const currentCount = Object.keys(allTransitSnapshots).length;
              if (currentCount > lastTransitSnapshotCount) {
                timingHintsInjected = false;
              }
              lastTransitSnapshotCount = currentCount;
            }
          }
        }
      }
    }

    // Max iterations reached
    const lastText = messages
      .filter(m => m.role === 'assistant')
      .reverse()
      .find(m => m.content.some(c => c.text));
    const fallback = lastText
      ? (lastText.content.find(c => c.text)?.text as string) || ''
      : '';

    return {
      answer: fallback || 'I apologize, but I was unable to complete the analysis within the allowed steps.',
      chartData: collectedChartPayload,
      transitSnapshots: collectedTransitSnapshots,
    };
  } catch (error) {
    const errMsg = (error as Error).message;
    const errStack = (error as Error).stack?.split('\n').slice(0, 6).join('\n');
    console.error('[agentNode] failed', { iteration: 'unknown', message: errMsg, stack: errStack });
    appendDebugLog('agent', 'ERROR', `Error: ${errMsg}\nStack: ${errStack}`);
    return {
      answer: 'The chart analysis service is temporarily unavailable. Please try again.',
      chartData: state.chartData || null,
      transitSnapshots: state.transitSnapshots || null,
    };
  }
}

// -- Node: finalize (quality check) --

async function finalizeNode(state: AgentStateType): Promise<Partial<AgentStateType>> {
  const answer = state.answer || '';
  const retries = state.qualityRetryCount || 0;

  logNodeStart('finalize', { answerLength: answer.length, retries });

  if (!answer?.trim()) {
    return { finalAnswer: 'I apologize, but I was unable to generate a response.', shouldRegenerate: false, qualityRetryCount: retries };
  }

  if (retries >= 1) {
    return { finalAnswer: answer, shouldRegenerate: false, qualityRetryCount: retries };
  }

  const question = extractQuestionFromMessages(state.messages || []);

  try {
    const result = await invokeDeepSeekBedrock({
      systemPrompt: `Review the answer and return JSON. Evaluate isAstrologyDomain FIRST — this is the most critical check:
{
  "isAstrologyDomain": boolean (true if answer is about Vedic astrology birth charts, false if it discusses financial trading, stock markets, candlestick patterns, support/resistance levels, volume, or any non-astrology topics),
  "finalAnswer": "string (the final answer, possibly refined)",
  "qualityFlags": {
    "addressesQuestion": boolean,
    "noMissingDataLanguage": boolean,
    "hasForwardGuidance": boolean,
    "consistentWithPrior": boolean,
    "isProportional": boolean (true if answer scope matches question scope — does not dump extra unrelated data)
  },
  "extractedClaims": string[]
}
No markdown. JSON only.`,
      userPrompt: `Original answer: ${answer.slice(0, 3000)}\nUser question: ${question}`,
      maxTokens: 2048,
    });

    const parsed = parseJsonSafely(result.text);
    if (!parsed || !parsed.qualityFlags) {
      return { finalAnswer: answer, shouldRegenerate: false, qualityRetryCount: retries };
    }

    const qf = parsed.qualityFlags;
    const needsRegen = !qf.isAstrologyDomain || !qf.addressesQuestion || !qf.noMissingDataLanguage || !qf.isProportional;

    if (needsRegen) {
      return {
        finalAnswer: null,
        answer: parsed.finalAnswer || answer,
        shouldRegenerate: true,
        qualityRetryCount: retries + 1,
      };
    }

    return {
      finalAnswer: parsed.finalAnswer || answer,
      shouldRegenerate: false,
      qualityRetryCount: retries,
    };
  } catch (e) {
    return { finalAnswer: answer, shouldRegenerate: false, qualityRetryCount: retries };
  }
}

// -- Node: regenerate (retry with cached data) --

async function regenerateNode(state: AgentStateType): Promise<Partial<AgentStateType>> {
  try {
    const question = extractQuestionFromMessages(state.messages || []);
    const previousAnswer = state.answer || '';
    const memories = state.relevantMemories || [];

    let chartDataText = '';
    let dashaText = '';
    let transitBlock = '';
    if (state.chartData) {
      const cd = state.chartData as any;
      const rawPayloads = cd._rawPayloads;
      // Reconstruct from raw payloads (not stale formattedData)
      if (rawPayloads?.basic) {
        chartDataText = formatGrahaBhavaLagna(rawPayloads.basic);
        const vc = formatVargaCharts(rawPayloads.basic);
        if (vc) chartDataText += '\n\n' + vc;
        const dosha = formatDoshaAnalysis(rawPayloads.basic);
        if (dosha) chartDataText += '\n\n' + dosha;
      } else if (cd.formattedData) {
        chartDataText = cd.formattedData;
      }
      if (rawPayloads?.dasha) {
        const formatted = formatDashaTimeline(rawPayloads.dasha, 3);
        if (formatted) dashaText = formatted;
      }
    }
    if (state.transitSnapshots && state.chartData) {
      const rawChart = (state.chartData as any)?._rawPayloads?.basic;
      if (rawChart) {
        const formattedSnapshots = formatTransitSnapshots(rawChart, state.transitSnapshots);
        const sadeSati = formatSadeSati(rawChart, state.transitSnapshots);
        transitBlock = formattedSnapshots;
        if (sadeSati) transitBlock += '\n' + sadeSati;
      }
    }

    const memoryContext = memories.slice(-5).map(m => `[${m.role}] ${m.text}`).join('\n');

    const modeDetail = state.mode === 'mini'
      ? 'Keep answers brief (1-2 sentences), focus on the key insight.'
      : 'Answer the specific question directly and concisely. Add context only when needed. Limit bonus insights to 1 sentence.';

    const systemPrompt = `You are Cozmic, an expert Vedic astrologer. You analyse Vedic astrology birth charts (kundli/horoscope). "D1" means the Rasi chart (main birth chart), NOT a financial trading chart. The user is ALWAYS asking about Vedic astrology.

${modeDetail}

The previous answer was flagged as insufficient. Provide a better, more specific answer based on the chart data below.

CHART DATA (already fetched, do NOT call tools):
${chartDataText || 'No chart data available.'}

${dashaText ? `\n${dashaText}` : ''}

${transitBlock ? `FORMATTED TRANSIT DATA:\n${transitBlock}` : ''}

${memoryContext ? `PREVIOUS CONVERSATION:\n${memoryContext}` : ''}

ANSWER PROPORTIONALLY:
- Answer ONLY what was asked. If the question is narrow, answer narrow.
- Supporting data is for proving your claim, not for dumping everything you know.

CRITICAL RULES:
- A graha placed in a bhava does NOT make it the lord of that bhava. Lordship is determined by the rashi on the bhava cusp.
- NEVER mention missing data, unavailable tools, or backend limitations.
- NEVER suggest consulting a professional astrologer.
- NEVER invent or fabricate planetary positions.
- Ground every claim in specific data from the fetched chart. CITATION: Format as "Shani 10ve bhava mein (Makara, 15°42')". Never state a position without having fetched it.
- End with actionable insight or forward-looking guidance.
- VARGA CHARTS: Bhava numbers are relative to THAT varga's own lagna, NOT D1's lagna.
- Sade Sati: Shani 12ve/1le/2re bhava se Chandra rashi ka gochar; each phase ~2.5 years.
- Manglik: Mars in 1/4/7/8/12. Kaal Sarp: all planets between Ra-Ke. Pitra: Sun/Saturn/Rahu in 9th.
- Remedies: ONLY suggest when user explicitly asks for upay/remedies/mantra/gemstones.
- Today is ${new Date().toISOString().slice(0, 10)}.

Before returning your answer, verify:
1. Is this about Vedic astrology (not financial trading, stock markets, or candlestick patterns)?
2. Did you avoid suggesting remedies unless the user explicitly asked?
3. Are all claims grounded in the provided chart data?
4. Does the answer match the scope of the question?
If any check fails, fix the answer before returning.`;

    const result = await invokeDeepSeekBedrock({
      systemPrompt,
      userPrompt: `User question: ${question}\n\nProvide a comprehensive, specific answer.`,
      maxTokens: 3000,
    });

    return { answer: result.text, finalAnswer: null, shouldRegenerate: false };
  } catch (error) {
    console.error('[regenerateNode] failed', { message: (error as Error).message, stack: (error as Error).stack?.split('\n').slice(0, 4).join('\n') });
    return { answer: state.answer || 'Regeneration failed.', finalAnswer: null, shouldRegenerate: false };
  }
}

// -- Build Graph --

function buildGraph() {
  const graph = new StateGraph(AgentState)
    .addNode('route', routeNode)
    .addNode('classify_context', classifyContextNode)
    .addNode('fast_answer', fastAnswerNode)
    .addNode('agent', agentNode)
    .addNode('finalize', finalizeNode)
    .addNode('regenerate', regenerateNode)

    .addEdge(START, 'route')
    .addConditionalEdges('route', routeAfterRoute, {
      fast_answer: 'fast_answer',
      agent: 'classify_context',
    })
    .addEdge('classify_context', 'agent')
    .addEdge('fast_answer', END)
    .addEdge('agent', 'finalize')
    .addConditionalEdges('finalize', (state) => {
      if (state.shouldRegenerate) return 'regenerate';
      return END;
    })
    .addEdge('regenerate', 'finalize');

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
    mode?: 'mini' | 'pro' | 'auto';
    sessionId?: string;
    relevantMemories?: RelevantChatMemory[];
    conversationHistory?: ChatHistoryMessage[];
  }
): Promise<{ answer: string; model?: string }> {
  if (!compiledGraph) {
    const graph = buildGraph();
    const pgPool = getPostgresPool();
    const cache = pgPool ? new PostgresCache(pgPool) : undefined;

    compiledGraph = graph.compile({
      ...(cache ? { cache } : {}),
    });
  }

  if (!input.message?.trim()) {
    return { answer: 'Please provide a message to chat about your chart.', model: 'cozmic-agent-v2' };
  }

  // Resolve mode: auto → classify, explicit → use as-is
  const callerMode = input.mode || 'auto';
  const isExplicitMode = callerMode !== 'auto';
  const resolvedMode = isExplicitMode ? callerMode : await classifyQuestionMode(input.message);

  logAgentStart({
    message: input.message,
    mode: resolvedMode,
    hasKundli: !!input.kundli,
    sessionId: input.sessionId,
  });

  try {
    const initialState: any = {
      messages: [{ role: 'user', content: input.message }],
      ownerId: input.ownerId || 'anonymous',
      profileId: input.profileId || null,
      mode: resolvedMode,
      isExplicitMode,
      kundliInput: input.kundli || null,
      chartData: null,
      transitSnapshots: null,
      answer: null,
      finalAnswer: null,
      shouldRegenerate: false,
      qualityRetryCount: 0,
      topLevelRoute: null,
      relevantMemories: input.relevantMemories || [],
      conversationHistory: input.conversationHistory || [],
      needsContext: false,
      contextMessages: [],
    };

    const thread_id = input.sessionId || `anon_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    const config: any = { configurable: { thread_id } };

    const result = await compiledGraph.invoke(initialState, config);

    const finalAnswer = result.finalAnswer || result.answer || 'I apologize, but I was unable to generate a response.';

    return {
      answer: finalAnswer,
      model: 'cozmic-agent-v2',
    };
  } catch (error) {
    const err = error as Error;
    console.error('[runKundliAgentV2] graph execution failed', {
      name: err.name,
      message: err.message,
      stack: err.stack?.split('\n').slice(0, 6).join('\n'),
      cause: err.cause instanceof Error
        ? { name: err.cause.name, message: err.cause.message }
        : err.cause,
    });
    return {
      answer: `I encountered an error: ${err.message}`,
      model: 'cozmic-agent-v2',
    };
  }
}
