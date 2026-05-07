import { Annotation, END, START, StateGraph, MessagesAnnotation } from '@langchain/langgraph';
import type { KundliSnapshotInput } from './be1Client.js';
import { fetchBe1Calculate, fetchBe1Transit } from './be1Client.js';
import { buildGroundingCacheKey, getTimeBucketForIntent } from './cacheKey.js';
import { invokeDeepSeekBedrock } from './deepseekBedrock.js';
import { getPostgresPool } from './postgresClient.js';
import { PostgresCache } from './postgresCache.js';
import { PostgresSaver } from '@langchain/langgraph-checkpoint-postgres';
import { logNodeStart, logNodeEnd } from '../utils/debugLog.js';

// -- State definition --
const AgentState = Annotation.Root({
  ...MessagesAnnotation.spec,

  ownerId: Annotation<string>,
  profileId: Annotation<string | null>,
  mode: Annotation<'mini' | 'pro'>,
  kundliInput: Annotation<KundliSnapshotInput | null>,

  route: Annotation<'pipeline' | 'smalltalk' | 'general_astro' | 'clarify' | null>,
  intent: Annotation<any>,
  dataPlan: Annotation<any>,
  toolGroups: Annotation<string[]>,
  routeConfidence: Annotation<number>,

  grounding: Annotation<any>,
  atlas: Annotation<any[]>,
  toolFindings: Annotation<any[]>,

  priorClaims: Annotation<string[]>,

  answerTemplate: Annotation<string | null>,
  answer: Annotation<string | null>,
  coverageGaps: Annotation<string[]>,

  finalAnswer: Annotation<string | null>,
  shouldCondense: Annotation<boolean | null>,
  condensedAnswer: Annotation<string | null>,
  qualityFlags: Annotation<any>,

  clarificationQuestion: Annotation<string | null>,

  decisionTelemetry: Annotation<any[]>,
  stageReporter: Annotation<any>,
});

type AgentStateType = typeof AgentState.State;

// -- Helpers --
function getLastHumanMessage(messages: any[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    const directRole = msg.role;
    const kwargRole = msg.kwargs?.role;
    const directContent = msg.content;
    const kwargContent = msg.kwargs?.content;
    
    // LangChain uses msg.type: "human" | "ai" for message roles
    const msgType = msg.type;
    const isHumanMessage = msgType === 'human';
    const isAIMessage = msgType === 'ai';
    
    const role = directRole || kwargRole || (isHumanMessage ? 'user' : isAIMessage ? 'assistant' : undefined);
    const content = directContent || kwargContent;
    
    if (role === 'user') {
      if (Array.isArray(content)) {
        return content[0]?.text || '';
      }
      return content || '';
    }
  }
  return '';
}

function parseJsonSafely(text: string): any {
  try {
    const cleaned = text.replace(/```json|```/g, '').trim();
    return JSON.parse(cleaned);
  } catch {
    return null;
  }
}

function extractClaims(text: string): string[] {
  const claims: string[] = [];
  const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 10);
  for (const sentence of sentences) {
    if (/will|is|are|has|have|shows|indicates|suggests/i.test(sentence)) {
      claims.push(sentence.trim());
    }
  }
  return claims.slice(0, 10);
}

// -- Node 1: route_and_plan --
async function routeAndPlan(state: AgentStateType): Promise<Partial<AgentStateType>> {
  const question = getLastHumanMessage(state.messages || []);
  
  logNodeStart('route_and_plan', { question, state: { route: state.route, intent: state.intent, dataPlan: state.dataPlan } });
  
  // Handle empty question
  if (!question?.trim()) {
    return {
      route: 'smalltalk',
      routeConfidence: 0.5,
      answer: 'Hello! Please share your birth details (date, time, place) or ask me about your chart.',
      finalAnswer: 'Hello! Please share your birth details (date, time, place) or ask me about your chart.',
    };
  }
  
  const priorContext = (state.priorClaims || []).slice(-5).join(' ');

  const systemPrompt = `You are the routing and planning engine for a Vedic astrology assistant called Cozmic.

Analyze the user's message and return a JSON object with these fields:
- "route": "pipeline" | "smalltalk" | "general_astro" | "clarify"
- "routeReason": string explaining the routing decision
- "intent" (only for pipeline): { "primary": "d9"|"dasha"|"transit"|"general", "flags": string[], "domains": string[], "isFollowUp": boolean }
- "dataPlan" (only for pipeline): { "varga": string[], "infolevel": string[], "needsTransit": boolean, "nesting": number }
- "toolGroups" (only for pipeline): string[]
- "clarificationQuestion" (only for clarify route): string
- "confidence": number between 0 and 1

ROUTING RULES:
- pipeline: Personal chart analysis, timing, predictions, remedies, "my chart"
- smalltalk: Greetings, thanks, farewells, casual chat
- general_astro: Concept questions ("What is D9?"), capabilities, "who are you"
- clarify: Ambiguous — needs follow-up (use sparingly)

PRIOR CLAIMS (be consistent with these):
${priorContext}

INTENT FLAGS: marriage, career, health, finance, timing, transit, dasha, forecast, education, children, property, travel, spirituality, longevity, remedies

DATA PLANNING RULES:
- Marriage/relationship → varga: ["D1","D9"], infolevel: ["basic","dasha"]
- Career/profession → varga: ["D1","D10"], infolevel: ["basic","dasha","yogas"]
- Health → varga: ["D1","D8","D30"], infolevel: ["basic","grahabala"]
- Timing questions → needsTransit: true
- Pro mode → add more varga divisions
- Mini mode → varga: ["D1"] only

Return ONLY valid JSON.`;

  const result = await invokeDeepSeekBedrock({
    systemPrompt,
    userPrompt: question,
    maxTokens: 1024,
  });

  const parsed = parseJsonSafely(result.text);
  
  // Default to pipeline for chart-related questions, not smalltalk
  const isChartRelated = /chart|d9|d1|d7|d10| astrology|kundli|planet|house|sign| nakshatra|dasha|transit|arudha/i.test(question);
  
  if (!parsed) {
    // If LLM fails to return valid JSON, still try to route based on question content
    return {
      route: isChartRelated ? 'pipeline' : 'smalltalk',
      routeConfidence: 0.3,
      intent: isChartRelated ? { primary: 'general', flags: [], domains: [], isFollowUp: false } : null,
      dataPlan: isChartRelated ? { varga: ['D1'], infolevel: ['basic'], needsTransit: false, nesting: 1 } : null,
      answer: isChartRelated ? null : result.text,
      finalAnswer: isChartRelated ? null : result.text,
    };
  }

  const update: Partial<AgentStateType> = {
    // If LLM says smalltalk but question is chart-related, override to pipeline
    route: (parsed.route === 'smalltalk' && isChartRelated) ? 'pipeline' : (parsed.route || (isChartRelated ? 'pipeline' : 'smalltalk')),
    routeConfidence: parsed.confidence || 0.5,
    decisionTelemetry: [
      ...(state.decisionTelemetry || []),
      { node: 'route_and_plan', model: result.model, confidence: parsed.confidence },
    ],
  };

  logNodeEnd('route_and_plan', { parsedJson: parsed, update });

  if (parsed.intent) update.intent = parsed.intent;
  if (parsed.dataPlan) update.dataPlan = parsed.dataPlan;
  if (parsed.toolGroups) update.toolGroups = parsed.toolGroups;
  if (parsed.clarificationQuestion) update.clarificationQuestion = parsed.clarificationQuestion;

  // If we overrode to pipeline, ensure intent and dataPlan are set
  if (isChartRelated && update.route === 'pipeline' && !update.intent) {
    update.intent = { primary: 'general', flags: [], domains: [], isFollowUp: false };
    update.dataPlan = { varga: ['D1'], infolevel: ['basic'], needsTransit: false, nesting: 1 };
    update.toolGroups = [];
  }

  return update;
}

function routeAfterPlan(state: AgentStateType) {
  switch (state.route) {
    case 'smalltalk':
    case 'general_astro':
      return 'fast_answer';
    case 'clarify':
      return 'send_clarification';
    case 'pipeline':
      return 'load_grounding';
    default:
      return 'fast_answer';
  }
}

// -- Node 2: fast_answer --
async function fastAnswer(state: AgentStateType): Promise<Partial<AgentStateType>> {
  const question = getLastHumanMessage(state.messages || []);
  
  logNodeStart('fast_answer', { question, route: state.route });
  
  // Handle empty question
  if (!question?.trim()) {
    const answer = 'Hello! How can I help you with your chart today?';
    logNodeEnd('fast_answer', { answer });
    return {
      answer,
      finalAnswer: answer,
    };
  }
  
  const isSmalltalk = state.route === 'smalltalk';

  const systemPrompt = isSmalltalk
    ? 'You are Cozmic, a friendly Vedic astrology assistant. Respond naturally to casual chat. Keep it brief and warm.'
    : 'You are Cozmic, a Vedic astrology assistant. Answer the user\'s conceptual question clearly and accurately. Keep responses concise (3-5 paragraphs max).';

  const result = await invokeDeepSeekBedrock({
    systemPrompt,
    userPrompt: question,
    maxTokens: 1024,
  });

  logNodeEnd('fast_answer', { answer: result.text });

  return {
    answer: result.text,
    finalAnswer: result.text,
  };
}

// -- Node 3: send_clarification --
async function sendClarification(state: AgentStateType): Promise<Partial<AgentStateType>> {
  const clarification = state.clarificationQuestion || 'Could you clarify what area of life you\'d like me to focus on?';
  return {
    answer: clarification,
    finalAnswer: clarification,
  };
}

// -- Node 4: load_grounding --
async function loadGrounding(state: AgentStateType): Promise<Partial<AgentStateType>> {
  const kundli = state.kundliInput;
  const dataPlan = state.dataPlan;
  const intent = state.intent;

  logNodeStart('load_grounding', { kundli, dataPlan, intent });

  if (!kundli || !dataPlan) {
    logNodeEnd('load_grounding', { grounding: null, reason: 'no kundli or dataPlan' });
    return {
      grounding: null,
      toolFindings: [{ name: 'Grounding', status: 'unavailable', facts: ['No birth data available.'] }],
    };
  }

  try {
    const apiResponse = await fetchBe1Calculate(kundli, {
      varga: dataPlan.varga?.join(',') || 'D1',
      infolevel: dataPlan.infolevel?.join(',') || 'basic',
      nesting: dataPlan.nesting || 1,
    });

    let rawPayload: any = apiResponse;
    let transitData: any = null;

    if (dataPlan.needsTransit) {
      const transitResponse = await fetchBe1Transit(kundli, new Date(), { nesting: 1 });
      transitData = (transitResponse as any).transit;
      rawPayload = { ...rawPayload, transit: transitData };
    }

    const cacheKey = buildGroundingCacheKey({
      profileId: state.profileId,
      intent: intent,
      dataPlan: dataPlan,
    } as any);

    logNodeEnd('load_grounding', { 
      grounding: { cacheKey, varga: dataPlan.varga, infolevel: dataPlan.infolevel },
      hasTransitData: !!transitData 
    });

    return {
      grounding: { rawPayload, cacheKey },
      atlas: [],
    };
  } catch (error) {
    logNodeEnd('load_grounding', { error: (error as Error).message });
    return {
      grounding: null,
      toolFindings: [{ name: 'Grounding', status: 'error', facts: [(error as Error).message] }],
    };
  }
}

// -- Node 5: gather_data --
async function gatherData(state: AgentStateType): Promise<Partial<AgentStateType>> {
  const toolGroups = state.toolGroups || [];
  const rawPayload = state.grounding?.rawPayload;
  const intent = state.intent;

  logNodeStart('gather_data', { toolGroups, intent });

  if (toolGroups.length === 0 || !rawPayload) {
    logNodeEnd('gather_data', { toolFindings: [], reason: 'no toolGroups or rawPayload' });
    return { toolFindings: [] };
  }

  const findings: any[] = [];

  for (const group of toolGroups) {
    try {
      const finding = await analyzeToolGroup(group, rawPayload, intent);
      findings.push(finding);
    } catch (error) {
      findings.push({
        name: group,
        status: 'error',
        facts: [`Failed to analyze: ${String(error)}`],
        evidencePaths: [],
      });
    }
  }

  logNodeEnd('gather_data', { toolFindings: findings });

  return { toolFindings: findings };
}

async function analyzeToolGroup(group: string, rawPayload: any, intent: any): Promise<any> {
  const isTransit = intent?.flags?.includes('transit') || intent?.primary === 'transit';
  const transitData = rawPayload.transit;
  
  const focusArea = intent?.flags?.join(', ') || 'general analysis';
  
  let chartDataPreview = JSON.stringify(rawPayload).slice(0, 2500);
  
  if (isTransit && transitData) {
    chartDataPreview = `=== NATAL CHART ===\n${JSON.stringify(rawPayload.chart || rawPayload).slice(0, 1200)}\n\n=== CURRENT TRANSITS ===\n${JSON.stringify(transitData).slice(0, 1200)}`;
  }

  const systemPrompt = `You are a Vedic astrology data analyzer.

Analyze the chart data for the tool group: "${group}"

Return a JSON object with:
- "status": "ok" | "partial" | "insufficient_data"
- "facts": string[] (3-7 key facts from the data)
- "evidencePaths": string[] (JSON paths to supporting data)

Focus on: ${focusArea}

${isTransit ? 'IMPORTANT: Analyze CURRENT TRANSITS (not natal chart). Compare transit positions to natal chart houses to show how transiting planets affect the native.' : ''}

Chart data preview: ${chartDataPreview}`;

  const userPrompt = isTransit
    ? `Analyze ${group} - focus on how CURRENT planetary transits (not natal chart) are affecting the native right now. Use the transit data to identify active transit influences.`
    : `Analyze ${group} based on the chart data.`;

  const result = await invokeDeepSeekBedrock({
    systemPrompt,
    userPrompt,
    maxTokens: 1024,
  });

  const parsed = parseJsonSafely(result.text);
  if (!parsed) {
    return {
      name: group,
      status: 'partial',
      facts: [result.text.slice(0, 200)],
      evidencePaths: [],
    };
  }

  return {
    name: group,
    status: parsed.status || 'ok',
    facts: parsed.facts || [],
    evidencePaths: parsed.evidencePaths || [],
  };
}

// -- Node 6: generate_answer --
async function generateAnswer(state: AgentStateType): Promise<Partial<AgentStateType>> {
  const question = getLastHumanMessage(state.messages || []);
  
  logNodeStart('generate_answer', { question, intent: state.intent, dataPlan: state.dataPlan, toolFindings: state.toolFindings });
  
  // Handle empty question
  if (!question?.trim()) {
    const answer = 'Hello! Please tell me about your chart question.';
    logNodeEnd('generate_answer', { answer });
    return {
      answer,
      answerTemplate: 'general_chart_reading',
    };
  }
  
  const toolFindings = state.toolFindings || [];
  const priorClaims = state.priorClaims || [];
  const isTransit = state.intent?.primary === 'transit' || state.intent?.flags?.includes('transit');

  const context = toolFindings.map((f: any) => `${f.name}: ${f.facts?.join(', ') || 'N/A'}`).join('\n');
  const priorContext = priorClaims.slice(-5).join('\n');

  const transitInstruction = isTransit
    ? `\n\nIMPORTANT: The user asked about CURRENT TRANSITS. In your answer:
- Focus on how transiting planets (from transit.graha) are affecting the natal chart houses
- Compare transit positions to natal chart to show their impact
- Use the current date (May 2026) for timing - explain what's happening NOW
- Don't use old transit periods like 2020-2023 - use 2025-2027 instead`
    : '';

  const systemPrompt = `You are an expert Vedic astrologer. Answer the user's question based on the chart data.

RULES:
- ALWAYS provide a direct, confident answer using available evidence
- Use specific year ranges for timing (e.g., "2025-2027")
- Ground every claim in specific planetary positions from the data
- If evidence is partial, synthesize insights from what you DO have
- NEVER mention missing data, unavailable tools, or backend limitations
- NEVER suggest consulting a professional astrologer
- Be consistent with prior conversation — do not contradict previous answers
- End with actionable insight or forward-looking guidance${transitInstruction}

PRIOR CLAIMS (be consistent with these):
${priorContext}

CONTEXT:
${context}`;

  const result = await invokeDeepSeekBedrock({
    systemPrompt,
    userPrompt: `Question: ${question}`,
    maxTokens: 2048,
  });

  logNodeEnd('generate_answer', { answer: result.text, context });

  return {
    answer: result.text,
    answerTemplate: 'general_chart_reading',
  };
}

// -- Node 7: finalize --
async function finalize(state: AgentStateType): Promise<Partial<AgentStateType>> {
  const answer = state.answer || '';
  const question = getLastHumanMessage(state.messages || []);
  
  logNodeStart('finalize', { question, answer, priorClaims: state.priorClaims });
  
  // Skip finalization if no answer
  if (!answer?.trim()) {
    const finalAnswer = 'I apologize, but I was unable to generate a response.';
    logNodeEnd('finalize', { finalAnswer });
    return {
      finalAnswer,
    };
  }
  
  const priorClaims = state.priorClaims || [];

  const systemPrompt = `Review the answer and ensure quality:
1. Does it address the user's question?
2. Does it avoid mentioning "missing data", "unavailable", or "consult a professional"?
3. Does it provide forward guidance or actionable insights?
4. Is it consistent with prior conversation context?

Return a JSON object with:
- "finalAnswer": string (the final answer, possibly refined)
- "shouldCondense": boolean
- "condensedAnswer": string (only if shouldCondense is true)
- "qualityFlags": { "addressesQuestion": boolean, "noMissingDataLanguage": boolean, "hasForwardGuidance": boolean, "consistentWithPrior": boolean }
- "extractedClaims": string[] (key claims made in the answer for future reference)`;

  const result = await invokeDeepSeekBedrock({
    systemPrompt,
    userPrompt: `Original answer: ${answer}\nUser question: ${question}`,
    maxTokens: 2048,
  });

  const parsed = parseJsonSafely(result.text);
  if (!parsed) {
    logNodeEnd('finalize', { finalAnswer: answer, parsedFailed: true });
    return {
      finalAnswer: answer,
      priorClaims: [...priorClaims, ...extractClaims(answer)],
    };
  }

  const newClaims = extractClaims(parsed.finalAnswer || answer);

  logNodeEnd('finalize', { parsedJson: parsed, finalAnswer: parsed.finalAnswer || answer, newClaims });

  return {
    finalAnswer: parsed.shouldCondense && parsed.condensedAnswer
      ? parsed.condensedAnswer
      : parsed.finalAnswer || answer,
    shouldCondense: parsed.shouldCondense || false,
    condensedAnswer: parsed.condensedAnswer || null,
    qualityFlags: parsed.qualityFlags || null,
    priorClaims: [...priorClaims, ...newClaims],
  };
}

// -- Build Graph --
function buildGraph() {
  const graph = new StateGraph(AgentState)
    .addNode('route_and_plan', routeAndPlan)
    .addNode('fast_answer', fastAnswer)
    .addNode('send_clarification', sendClarification)
    .addNode('load_grounding', loadGrounding)
    .addNode('gather_data', gatherData)
    .addNode('generate_answer', generateAnswer)
    .addNode('finalize', finalize)
    .addEdge(START, 'route_and_plan')
    .addConditionalEdges('route_and_plan', routeAfterPlan, {
      fast_answer: 'fast_answer',
      send_clarification: 'send_clarification',
      load_grounding: 'load_grounding',
    })
    .addEdge('load_grounding', 'gather_data')
    .addEdge('gather_data', 'generate_answer')
    .addEdge('generate_answer', 'finalize')
    .addEdge('fast_answer', END)
    .addEdge('send_clarification', END)
    .addEdge('finalize', END);

  return graph;
}

let compiledGraph: any = null;
let checkpointer: PostgresSaver | null = null;

async function getCheckpointer(): Promise<PostgresSaver | undefined> {
  if (checkpointer) return checkpointer;

  const pool = getPostgresPool();
  if (!pool) return undefined;

  checkpointer = new PostgresSaver(pool);
  await checkpointer.setup();
  return checkpointer;
}

export async function runKundliAgentV2(
  input: {
    message: string;
    ownerId?: string;
    profileId?: string;
    kundli?: KundliSnapshotInput;
    mode?: 'mini' | 'pro';
    sessionId?: string;
  }
): Promise<{ answer: string; model?: string }> {
  if (!compiledGraph) {
    const graph = buildGraph();
    // Skip checkpointer for now - causes message serialization issues
    // const checkpointer = await getCheckpointer();
    const pgPool = getPostgresPool();
    const cache = pgPool ? new PostgresCache(pgPool) : undefined;

    compiledGraph = graph.compile({
      // ...(checkpointer ? { checkpointer } : {}),
      ...(cache ? { cache } : {}),
    });

    // Preload graph so subsequent calls don't recompile
    // This also ensures the graph is ready before any state issues
  }

  // Skip thread_id config to avoid checkpointer loading old state
  const config = {};

  // Validate input
  if (!input.message?.trim()) {
    return {
      answer: 'Please provide a message to chat about your chart.',
      model: 'cozmic-agent-v2',
    };
  }

  try {
    // Use simple string format for messages - avoids ContentBlock complexity
    const initialState: any = {
      messages: [{ role: 'user', content: input.message }],
      ownerId: input.ownerId || 'anonymous',
      profileId: input.profileId || null,
      mode: input.mode || 'mini',
      kundliInput: input.kundli || null,
    };

    const result = await compiledGraph.invoke(initialState, config);

    return {
      answer: result.finalAnswer || result.answer || 'I apologize, but I was unable to generate a response.',
      model: 'cozmic-agent-v2',
    };
  } catch (error) {
    return {
      answer: `I encountered an error: ${(error as Error).message}`,
      model: 'cozmic-agent-v2',
    };
  }
}
