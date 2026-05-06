import { Annotation, END, START, StateGraph } from '@langchain/langgraph';
import type { KundliSnapshotInput } from './be1Client.js';
import { fetchBe1Calculate, fetchBe1Transit } from './be1Client.js';
import { buildGroundingCacheKey } from './cacheKey.js';
import { invokeDeepSeekBedrock } from './deepseekBedrock.js';
import { getPostgresPool } from './postgresClient.js';
import { PostgresCache } from './postgresCache.js';

// -- State definition --
const AgentState = Annotation.Root({
  messages: Annotation<any[]>,
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

  priorClaims: Annotation<any[]>,

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
    if (messages[i].role === 'user') return messages[i].content || '';
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

// -- Node 1: route_and_plan --
async function routeAndPlan(state: AgentStateType): Promise<Partial<AgentStateType>> {
  const question = getLastHumanMessage(state.messages || []);

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
  if (!parsed) {
    return {
      route: 'smalltalk',
      routeConfidence: 0.5,
      answer: result.text,
      finalAnswer: result.text,
    };
  }

  const update: Partial<AgentStateType> = {
    route: parsed.route || 'smalltalk',
    routeConfidence: parsed.confidence || 0.5,
    decisionTelemetry: [
      ...(state.decisionTelemetry || []),
      { node: 'route_and_plan', model: result.model, confidence: parsed.confidence },
    ],
  };

  if (parsed.intent) update.intent = parsed.intent;
  if (parsed.dataPlan) update.dataPlan = parsed.dataPlan;
  if (parsed.toolGroups) update.toolGroups = parsed.toolGroups;
  if (parsed.clarificationQuestion) update.clarificationQuestion = parsed.clarificationQuestion;

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
  const isSmalltalk = state.route === 'smalltalk';

  const systemPrompt = isSmalltalk
    ? 'You are Cozmic, a friendly Vedic astrology assistant. Respond naturally to casual chat. Keep it brief and warm.'
    : 'You are Cozmic, a Vedic astrology assistant. Answer the user\'s conceptual question clearly and accurately. Keep responses concise (3-5 paragraphs max).';

  const result = await invokeDeepSeekBedrock({
    systemPrompt,
    userPrompt: question,
    maxTokens: 1024,
  });

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

  if (!kundli || !dataPlan) {
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

    if (dataPlan.needsTransit) {
      const transitResponse = await fetchBe1Transit(kundli, new Date(), { nesting: 1 });
      rawPayload = { ...rawPayload, transit: (transitResponse as any).transit };
    }

    return {
      grounding: { rawPayload },
      atlas: [],
    };
  } catch (error) {
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

  if (toolGroups.length === 0 || !rawPayload) {
    return { toolFindings: [] };
  }

  const findings = toolGroups.map((group: string) => ({
    name: group,
    status: 'ok',
    facts: [],
    evidencePaths: [],
  }));

  return { toolFindings: findings };
}

// -- Node 6: generate_answer --
async function generateAnswer(state: AgentStateType): Promise<Partial<AgentStateType>> {
  const question = getLastHumanMessage(state.messages || []);
  const toolFindings = state.toolFindings || [];

  const context = toolFindings.map((f: any) => `${f.name}: ${f.facts?.join(', ') || 'N/A'}`).join('\n');

  const systemPrompt = `You are an expert Vedic astrologer. Answer the user's question based on the chart data.

RULES:
- ALWAYS provide a direct, confident answer using available evidence
- Use specific year ranges for timing (e.g., "2025-2027")
- Ground every claim in specific planetary positions from the data
- If evidence is partial, synthesize insights from what you DO have
- NEVER mention missing data, unavailable tools, or backend limitations
- NEVER suggest consulting a professional astrologer
- Be consistent with prior conversation — do not contradict previous answers
- End with actionable insight or forward-looking guidance

CONTEXT:
${context}`;

  const result = await invokeDeepSeekBedrock({
    systemPrompt,
    userPrompt: `Question: ${question}`,
    maxTokens: 2048,
  });

  return {
    answer: result.text,
    answerTemplate: 'general_chart_reading',
  };
}

// -- Node 7: finalize --
async function finalize(state: AgentStateType): Promise<Partial<AgentStateType>> {
  const answer = state.answer || '';
  const question = getLastHumanMessage(state.messages || []);

  const systemPrompt = `Review the answer and ensure quality:
1. Does it address the user's question?
2. Does it avoid mentioning "missing data", "unavailable", or "consult a professional"?
3. Does it provide forward guidance or actionable insights?
4. Is it consistent with prior conversation context?

Return a JSON object with:
- "finalAnswer": string (the final answer, possibly refined)
- "shouldCondense": boolean
- "condensedAnswer": string (only if shouldCondense is true)
- "qualityFlags": { "addressesQuestion": boolean, "noMissingDataLanguage": boolean, "hasForwardGuidance": boolean, "consistentWithPrior": boolean }`;

  const result = await invokeDeepSeekBedrock({
    systemPrompt,
    userPrompt: `Original answer: ${answer}\nUser question: ${question}`,
    maxTokens: 2048,
  });

  const parsed = parseJsonSafely(result.text);
  if (!parsed) {
    return { finalAnswer: answer };
  }

  return {
    finalAnswer: parsed.shouldCondense && parsed.condensedAnswer
      ? parsed.condensedAnswer
      : parsed.finalAnswer || answer,
    shouldCondense: parsed.shouldCondense || false,
    condensedAnswer: parsed.condensedAnswer || null,
    qualityFlags: parsed.qualityFlags || null,
  };
}

// -- Build Graph --
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

let compiledGraph: any = null;

export async function runKundliAgentV2(
  input: {
    message: string;
    ownerId?: string;
    profileId?: string;
    kundli?: KundliSnapshotInput;
    mode?: 'mini' | 'pro';
  }
): Promise<{ answer: string; model?: string }> {
  if (!compiledGraph) {
    const pool = getPostgresPool();
    const cache = pool ? new PostgresCache(pool) : undefined;

    compiledGraph = graph.compile({
      ...(cache ? { cache } : {}),
    });
  }

  try {
    const result = await compiledGraph.invoke({
      messages: [{ role: 'user', content: input.message }],
      ownerId: input.ownerId || 'anonymous',
      profileId: input.profileId || null,
      mode: input.mode || 'mini',
      kundliInput: input.kundli || null,
      route: null,
      intent: null,
      dataPlan: null,
      toolGroups: [],
      routeConfidence: 0,
      grounding: null,
      atlas: [],
      toolFindings: [],
      priorClaims: [],
      answerTemplate: null,
      answer: null,
      coverageGaps: [],
      finalAnswer: null,
      shouldCondense: null,
      condensedAnswer: null,
      qualityFlags: null,
      clarificationQuestion: null,
      decisionTelemetry: [],
      stageReporter: null,
    });

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
