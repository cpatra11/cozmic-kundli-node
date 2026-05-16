import { Annotation, END, START, StateGraph, MessagesAnnotation } from '@langchain/langgraph';
import type { KundliSnapshotInput } from './be1Client.js';
import { fetchBe1Calculate, fetchBe1Transit } from './be1Client.js';
import { invokeDeepSeekBedrock, invokeDeepSeekConversation } from './deepseekBedrock.js';
import { CHART_TOOL_CONFIG } from '../utils/toolDefs.js';
import { formatAllChartData, formatTransitSnapshots, formatSadeSati } from '../utils/formatChartData.js';
import { getPostgresPool } from './postgresClient.js';
import { PostgresCache } from './postgresCache.js';
import { logNodeStart, logNodeEnd, logAgentStart } from '../utils/debugLog.js';
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

function extractDashaTransitionDates(apiResponse: any): string[] {
  const dates = new Set<string>();
  dates.add(new Date().toISOString().slice(0, 10));
  const dashaRoot = apiResponse?.chart?.dasha;
  if (!dashaRoot?.periods) return Array.from(dates);
  function walk(node: any, depth: number) {
    if (!node?.periods || depth > 3) return;
    for (const key of Object.keys(node.periods)) {
      const p = node.periods[key];
      if (p.start) dates.add(p.start.slice(0, 10));
      if (p.end) dates.add(p.end.slice(0, 10));
      if (depth < 2) walk(p, depth + 1);
    }
  }
  walk(dashaRoot, 1);
  return Array.from(dates).sort();
}

// -- Fast routing --

function isObviousSmalltalk(q: string): boolean {
  return /^(hello|hi|hey|thanks|thank you|bye|goodbye|great|ok|okay|nice|good)\b/i.test(q.trim());
}

function isIdentityQuestion(q: string): boolean {
  return /\b(who are you|what are you|what can you|your name|tell me about yourself|capabilities|help)\b/i.test(q);
}

// (removed isGeneralAstroQuestion and shouldForcePipelineRoute — replaced by LLM classifier below)

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
      systemPrompt: `Classify the user's astrology question into exactly one:
- "pipeline": personal chart question (mentions "my", "mine", "will I", birth details, or any specific life area like career/marriage/health). Also route here if unsure.
- "general_astro": conceptual question about astrology without personal reference (e.g. "what is D9", "explain manglik dosha")
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

async function executeFetchChartData(
  kundli: KundliSnapshotInput | null,
  input: Record<string, unknown>
): Promise<Record<string, unknown>> {
  if (!kundli) {
    return { error: 'No birth details available. Please provide birth date, time, and place.' };
  }

  const vargas = (input.vargas as string[]) || ['D1'];
  const infolevels = (input.infolevels as string[]) || ['basic'];
  const nesting = (input.nesting as number) ?? 2;
  const autoTransit = input.autoTransit === true;

  try {
    const apiResponse = await fetchBe1Calculate(kundli, {
      varga: vargas.join(','),
      infolevel: infolevels.join(','),
      nesting,
    });

    const formatted = formatAllChartData(apiResponse, infolevels);
    const result: Record<string, unknown> = {
      vargas,
      infolevels,
      nesting,
      formattedData: formatted,
      rawPayload: apiResponse,
      rawAvailable: true,
    };

    if (autoTransit && infolevels.includes('dasha')) {
      const dates = extractDashaTransitionDates(apiResponse);
      if (dates.length > 0) {
        try {
          const transitResult = await executeFetchTransit(kundli, { dates }, apiResponse as Record<string, unknown>);
          if (transitResult.transitText) {
            result.transitData = transitResult.transitText;
          }
        } catch {
          // non-fatal
        }
      }
    }

    return result;
  } catch (error) {
    return { error: `Failed to fetch chart data: ${String(error)}` };
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
    const transitResults = await Promise.all(
      dates.map(date => fetchBe1Transit(kundli, date, { nesting: 1 }))
    );

    const snapshots: Record<string, Record<string, unknown>> = {};
    for (let i = 0; i < dates.length; i++) {
      const dateKey = dates[i].toISOString().slice(0, 10);
      const obj = transitResults[i] as any;
      const chart = obj?.chart || obj;
      const transitData = chart?.transit || chart;
      if (transitData?.graha) {
        snapshots[dateKey] = { graha: transitData.graha as Record<string, unknown> };
      }
    }

    // Format transit data with natal-lagna-relative house positions
    let transitText = '';
    if (chartPayload && Object.keys(snapshots).length > 0) {
      transitText = formatTransitSnapshots(chartPayload, snapshots);
      const sadeSati = formatSadeSati(chartPayload, snapshots);
      if (sadeSati) transitText += '\n' + sadeSati;
    }

    return { transitText: transitText || 'No transit data available for the requested dates.', dates: dateStrings };
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
    ? 'Keep answers brief (2-3 sentences), focus on the key insight.'
    : 'Provide detailed Vedic astrology analysis. Answer the specific question thoroughly, then optionally add 1-2 brief bonus insights if other chart factors are genuinely relevant. Keep bonus observations short (1-2 sentences each, start with "Bonus insight:" or "Also notable:").';

  const hasKundli = !!state.kundliInput;
  const profileNote = state.profileId ? `\nUser has a saved birth chart (profileId: ${state.profileId}).` : '';

  const upgradeNote = state.mode === 'mini'
    ? `\nMini mode: You only have D1 (Rasi) chart data available. If user asks about other D-charts, you MUST NOT invent positions. Base answer on D1 data, suggest upgrading to Pro for full chart access.`
    : '';

  const retryNote = isRetry
    ? '\nThe previous answer was insufficient. Provide a more detailed, specific, and actionable response. Address the question directly with concrete chart references.'
    : '';

  const dataNote = chartDataCached
    ? '\nChart data was already fetched in a previous attempt and is available in your conversation history below. You do not need to call fetch_chart_data again unless you need different parameters. Simply review the existing data and improve the answer.'
    : '';

  return `You are Cozmic, an expert Vedic astrologer. Answer the user's question based on chart data you fetch via tools.

${modeInstruction}
${profileNote}
${dataNote}
${upgradeNote}
${retryNote}

AVAILABLE DATA — Choose what to fetch based on the question:
CHARTS (vargas):
D1(Rasi/base)  D2(wealth)  D3(siblings)  D4(property)  D5(fame)
D6(health)  D7(children)  D8(sudden)  D9(marriage)  D10(career)
D11(destruction)  D12(parents)  D16(travel)  D20(spirituality)
D24(education)  D27(talent)  D30(obstacles)  D40(maternal)  D45(paternal)  D60(karma)

DATA SECTIONS (infolevels):
basic(planets/houses)  panchanga(tithi/nakshatra)  yogas(combinations)
dasha(timing)  ashtakavarga(strength)  grahabala(shadbala)  arudha(pada)  ayanamsa

DASHA NESTING (choose based on time scale):
1=mahadasha only(~2KB, decade-level)
2=+antardasha(~13KB, month-level) — default for most life questions
3=+pratyantardasha(~200KB, week-level) — use for "when will X happen"
4=+sookshmantardasha — day-level precision (large)
5=+pranantardasha — hour-level precision (very large, rarely needed)

TIMING GRANULARITY — Choose nesting based on the question's time scale:
• Years/months ("when will I get married/get a job/buy a house") → nesting 3 (pratyantar)
• Broad window ("will I marry in 2026") → nesting 2 (antar) is often sufficient
• Days/weeks ("will this month be good") → nesting 4 (sookshmantar)
• Specific days ("is next Tuesday good for trip") → nesting 5 (pranantar)
• General reading (no timing) → nesting 2

INSTRUCTIONS:
1. PLAN: Decide which charts, data sections, and nesting depth are appropriate for the question.
2. FETCH: Call fetch_chart_data with your chosen parameters. For timing questions, set autoTransit: true to auto-fetch transit for dasha period dates.
3. REVIEW: Examine returned data. If transit is missing or you need different parameters, call again.
4. ANALYZE: Synthesize all data into a clear answer.
5. For Mini mode: you may NOT call fetch_chart_data with vargas other than D1. The tool will reject Mini requests for D9+.

EFFICIENCY: You have 3 tool-calling rounds max. Plan all needed data before your first call. For timing questions: fetch D1+dasha+nesting 3+autoTransit:true in ONE call.

CRITICAL RULES:
- A planet being placed in a house does NOT make it the lord of that house. The house lord is determined by the rashi sign ruling that house cusp. For example, Mercury in the 10th house does NOT make Mercury the 10th lord.
- NEVER mention missing data, unavailable tools, or backend limitations.
- NEVER suggest consulting a professional astrologer.
- NEVER invent or fabricate planetary positions.
- Ground every claim in specific data from the fetched chart. CITATION: Format as "Saturn in 10th house (Capricorn, 15°42')". Never state a position without having fetched it.
- End with actionable insight or forward-looking guidance.
- VARGA CHARTS: House numbers are relative to THAT varga's own lagna, NOT D1's lagna.
- D24 = Siddhamamsa (Chaturvimsamsa, education/wisdom). D30 = Trimamsa (Trisamsa, evil/inauspicious). Do NOT confuse them.
- Sade Sati: Saturn transits 12th/1st/2nd from Moon's birth rashi; each phase ~2.5 years.
- Manglik: Mars in 1/4/7/8/12. Kaal Sarp: all planets between Ra-Ke. Pitra: Sun/Saturn/Rahu in 9th.
- Remedies: ONLY suggest when user explicitly asks for upay/remedies/mantra/gemstones.
- Today's date: ${currentDate}. Use as reference for all timing.
- Use Hinglish naturally where appropriate (e.g., "Shani ki mahadasha chal rahi hai" alongside English).

CHART SUMMARY NOTES:
- Basic infolevel gives you: planet positions (rashi, longitude, nakshatra, house), house cusps, lagna
- Each varga chart line shows: CHART_NAME: Lagna X | Planet1:HouseN(Rashi), Planet2:HouseN(Rashi)
- Dasha timeline shows: PERIOD_NAME period_type (start → end) with ← ACTIVE or ← upcoming markers
- Transit shows planet positions at specific dates with house placement relative to natal lagna`;
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
    const maxIterations = 3;

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

    let collectedChartPayload: Record<string, unknown> | null = state.chartData || null;
    let collectedTransitSnapshots: Record<string, Record<string, unknown>> | null = state.transitSnapshots || null;
    const calledTools: Set<string> = new Set();

    for (let iteration = 0; iteration < maxIterations; iteration++) {
      const response = await invokeDeepSeekConversation({
        systemPrompt,
        messages: messages as any,
        tools: CHART_TOOL_CONFIG,
        maxTokens: 4096,
      });

      const toolUseBlocks = response.content.filter(c => c.toolUse);
      const textBlocks = response.content.filter(c => c.text);

      if (toolUseBlocks.length === 0) {
        // LLM finished — text response is the answer
        const answerText = textBlocks.map(c => c.text).join('').trim();
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

      // Execute each tool and add results
      for (const block of toolUseBlocks) {
        const tu = block.toolUse;
        if (!tu) continue;

        const toolSig = `${tu.name}:${JSON.stringify(tu.input)}`;
        if (calledTools.has(toolSig)) {
          messages.push({
            role: 'user',
            content: [{ toolResult: { toolUseId: tu.toolUseId, content: [{ text: 'Already fetched with identical parameters. Use existing data.' }] } }],
          });
          continue;
        }
        calledTools.add(toolSig);

        let result: Record<string, unknown>;

        switch (tu.name) {
          case 'fetch_chart_data': {
            result = await executeFetchChartData(state.kundliInput, tu.input);
            if (result.formattedData) {
              collectedChartPayload = { ...(collectedChartPayload || {}), ...result };
            }
            break;
          }
          case 'fetch_transit': {
            const natalPayload = collectedChartPayload?.rawPayload as Record<string, unknown> | undefined;
            result = await executeFetchTransit(state.kundliInput, tu.input, natalPayload);
            const transitText = result.transitText as string | undefined;
            if (transitText) {
              collectedTransitSnapshots = collectedTransitSnapshots || {};
            }
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

        messages.push({
          role: 'user',
          content: [{ toolResult: { toolUseId: tu.toolUseId, content: [{ json: result }] } }],
        });
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
    console.error('[agentNode] failed', {
      iteration: 'unknown',
      message: (error as Error).message,
      stack: (error as Error).stack?.split('\n').slice(0, 6).join('\n'),
    });
    return {
      answer: 'I encountered an error while analyzing your chart.',
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

    const parsed = parseJsonSafely(result.text);
    if (!parsed || !parsed.qualityFlags) {
      return { finalAnswer: answer, shouldRegenerate: false, qualityRetryCount: retries };
    }

    const qf = parsed.qualityFlags;
    const needsRegen = !qf.addressesQuestion || !qf.noMissingDataLanguage;

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

    const chartDataJson = state.chartData ? JSON.stringify(state.chartData, null, 2).slice(0, 8000) : '';
    const transitJson = state.transitSnapshots ? JSON.stringify(state.transitSnapshots, null, 2).slice(0, 4000) : '';

    const memoryContext = memories.slice(-5).map(m => `[${m.role}] ${m.text}`).join('\n');

    const systemPrompt = `You are Cozmic, an expert Vedic astrologer. The previous answer was flagged as insufficient. Provide a better, more detailed answer.

PREVIOUS ANSWER (needs improvement):
${previousAnswer.slice(0, 2000)}

CHART DATA (already fetched, do NOT call tools):
${chartDataJson || 'No chart data available.'}

${transitJson ? `TRANSIT DATA:\n${transitJson}` : ''}

${memoryContext ? `PREVIOUS CONVERSATION:\n${memoryContext}` : ''}

CRITICAL RULES:
- Answer the user's specific question thoroughly.
- Ground every claim in chart data.
- NEVER mention missing data or suggest consulting a professional.
- End with actionable insight.
- Today is ${new Date().toISOString().slice(0, 10)}.`;

    const result = await invokeDeepSeekBedrock({
      systemPrompt,
      userPrompt: `User question: ${question}\n\nProvide a comprehensive, specific answer.`,
      maxTokens: 3000,
    });

    return { finalAnswer: result.text, shouldRegenerate: false };
  } catch (error) {
    console.error('[regenerateNode] failed', { message: (error as Error).message, stack: (error as Error).stack?.split('\n').slice(0, 4).join('\n') });
    return { finalAnswer: state.answer || 'Regeneration failed.', shouldRegenerate: false };
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
    .addEdge('regenerate', END);

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
