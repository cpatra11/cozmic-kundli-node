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

// -- Fast routing --

function isObviousSmalltalk(q: string): boolean {
  return /^(hello|hi|hey|thanks|thank you|bye|goodbye|great|ok|okay|nice|good)\b/i.test(q.trim());
}

function isIdentityQuestion(q: string): boolean {
  return /\b(who are you|what are you|what can you|your name|tell me about yourself|capabilities|help)\b/i.test(q);
}

function isGeneralAstroQuestion(q: string): boolean {
  const hasAstro = /\b(house|planet|rashi|nakshatra|dasha|yoga|varga|bhava|karaka|bala|arudha|aspect|conjunction|retrograde|exaltation|debiliation|navamsa|dasamsa|saptamsa|drekkana|hora|shashtiamsa|trimamsa|siddhamamsa|D\d+|sun|moon|mars|mercury|jupiter|venus|saturn|rahu|ketu|graha)\b/i.test(q);
  const noPersonal = !/\b(my|mine|my\s+chart|my\s+birth|do\s+I|am\s+I|will\s+I|have\s+I|my\s+kundli)\b/i.test(q);
  const isConceptual = /\b(what is|what does|tell me about|explain|meaning of|significance of|describe|define|how does|why does|what are)\b/i.test(q);
  return hasAstro && noPersonal && isConceptual;
}

function shouldForcePipelineRoute(q: string): boolean {
  const patterns = [
    'chart', 'kundli', 'horoscope', 'astrology', 'planet', 'house', 'rashi', 'nakshatra', 'dasha', 'transit', 'birth', 'janam', 'bhav',
    'varga', 'yoga', 'karaka', 'arudha', 'mangal', 'shani', 'guru', 'rahu', 'ketu', 'career', 'finance', 'marriage', 'health', 'education',
    'property', 'travel', 'remedy', 'prediction', 'sarkari', 'naukri', 'upsc', 'ssc', 'banking', 'railway', 'defence', 'ias', 'ips', 'officer',
    'jee', 'neet', 'gate', 'cat', 'entrance', 'manglik', 'kaal', 'sarp', 'pitra', 'kuja', 'upay', 'upaya', 'mantra', 'puja', 'vastu',
    'muhurta', 'engineering', 'medical', 'doctor', 'engineer', 'videsh', 'nri', 'gemstone', 'ratna', 'stone', 'dhaiya',
  ];
  const phrases = [
    '\\bD\\s*\\d+\\b', 'government\\s+job', 'civil\\s+service', 'competitive\\s+exam',
    'love\\s+marriage', 'arranged\\s+marriage', 'intercaste', 'love\\s+match',
    'mangal\\s+dosh', 'kaal\\s+sarp', 'pitra\\s+dosh', 'kuja\\s+dosh',
    'shubh\\s+(?:time|samay|muhurat)', 'kundli\\s+matching', 'gun\\s+milan',
    'business\\s+vs\\s+service', 'service\\s+or\\s+business',
    'foreign\\s+settlement', 'abroad\\s+study', 'settle\\s+abroad',
    'sade\\s+sati', 'shani\\s+sade', 'shani\\s+dhaiya',
  ];
  return new RegExp([...patterns, ...phrases].join('|'), 'i').test(q);
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
  if (isGeneralAstroQuestion(question)) return { topLevelRoute: 'general_astro' };

  if (shouldForcePipelineRoute(question)) {
    return { topLevelRoute: 'pipeline' };
  }

  return { topLevelRoute: 'smalltalk' };
}

function routeAfterRoute(state: AgentStateType): string {
  switch (state.topLevelRoute) {
    case 'pipeline': return 'agent';
    default: return 'fast_answer';
  }
}

// -- Node: fast_answer (smalltalk / general_astro) --

async function fastAnswerNode(state: AgentStateType): Promise<Partial<AgentStateType>> {
  if (state.finalAnswer) return {};

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

  try {
    const apiResponse = await fetchBe1Calculate(kundli, {
      varga: vargas.join(','),
      infolevel: infolevels.join(','),
      nesting,
    });

    const formatted = formatAllChartData(apiResponse, infolevels);

    return {
      vargas,
      infolevels,
      nesting,
      formattedData: formatted,
      rawAvailable: true,
    };
  } catch (error) {
    return { error: `Failed to fetch chart data: ${String(error)}` };
  }
}

async function executeFetchTransit(
  kundli: KundliSnapshotInput | null,
  input: Record<string, unknown>
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

    return { snapshots, dates: dateStrings };
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

  const memoryContext = memories.length > 0
    ? '\nPREVIOUS CONVERSATION CONTEXT (from this thread, ranked by relevance):\n' +
      memories.slice(-5).map(m => `[${m.role === 'user' ? 'User' : 'Assistant'}] ${m.text}`).join('\n')
    : '';

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
${memoryContext}

AVAILABLE TOOLS:
1. fetch_chart_data — Fetch birth chart data. Parameters: vargas (array of chart IDs), infolevels (array of data types), nesting (1-5 dasha depth). Always include "basic" in infolevels.
2. fetch_transit — Fetch transit positions for specific dates. Parameters: dates (array of YYYY-MM-DD strings).
3. search_astrology — Look up general astrological concepts. Parameters: query (your question).

INSTRUCTIONS:
1. PLAN: First decide what data you need based on the user's question. Think about which vargas, infolevels, and nesting depth are appropriate.
2. FETCH: Call fetch_chart_data with your chosen parameters.
3. REVIEW: Examine the returned chart data. If you need more detail (e.g., transit for timing), call fetch_transit or fetch_chart_data again with different params.
4. ANALYZE: Synthesize all data into a clear answer.
5. For Mini mode: you may NOT call fetch_chart_data with vargas other than D1. The tool will reject Mini requests for D9+.

CRITICAL RULES:
- NEVER mention missing data, unavailable tools, or backend limitations.
- NEVER suggest consulting a professional astrologer.
- NEVER invent or fabricate planetary positions.
- Ground every claim in specific data from the fetched chart.
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
  const question = extractQuestionFromMessages(state.messages || []);
  logNodeStart('agent', { question, mode: state.mode, hasKundli: !!state.kundliInput });

  if (!question?.trim()) {
    return { answer: 'Please tell me about your chart question.' };
  }

  const systemPrompt = buildAgentSystemPrompt(state, !!state.chartData);
  const maxIterations = 10;

  const messages: Array<{ role: string; content: Array<Record<string, unknown>> }> = [
    { role: 'user', content: [{ text: question }] },
  ];

  let collectedChartPayload: Record<string, unknown> | null = state.chartData || null;
  let collectedTransitSnapshots: Record<string, Record<string, unknown>> | null = state.transitSnapshots || null;

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
          result = await executeFetchTransit(state.kundliInput, tu.input);
          const snapshots = result.snapshots as Record<string, Record<string, unknown>> | undefined;
          if (snapshots) {
            collectedTransitSnapshots = { ...(collectedTransitSnapshots || {}), ...snapshots };
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
}

// -- Build Graph --

function buildGraph() {
  const graph = new StateGraph(AgentState)
    .addNode('route', routeNode)
    .addNode('fast_answer', fastAnswerNode)
    .addNode('agent', agentNode)
    .addNode('finalize', finalizeNode)
    .addNode('regenerate', regenerateNode)

    .addEdge(START, 'route')
    .addConditionalEdges('route', routeAfterRoute, {
      fast_answer: 'fast_answer',
      agent: 'agent',
    })
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
    return {
      answer: `I encountered an error: ${(error as Error).message}`,
      model: 'cozmic-agent-v2',
    };
  }
}
