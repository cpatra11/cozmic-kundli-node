import { Annotation, END, START, StateGraph } from '@langchain/langgraph';
import { fetchBe1Calculate, fetchCalculatedChart, type KundliSnapshotInput } from './be1Client.js';
import { env } from '../config/env.js';
import type { RagProfileDocument } from '../models/firestoreModels.js';
import { getRagProfilesRepository } from '../repositories/ragProfilesRepository.js';
import { stableHash } from './hash.js';
import { invokeDeepSeekBedrock } from './deepseekBedrock.js';
import { buildChartSnapshot } from './chartSnapshot.js';
import {
  buildTransitIntervalToolFinding,
  buildTransitPointToolFinding,
  resolveTransitRequestKind,
  type ToolFinding,
} from './astrologyTools.js';
import { cacheGetJson, cacheSetJson } from './valkeyCache.js';
import { z } from 'zod';

interface RagApiSourceRecord {
  data: {
    chartSnapshot?: unknown;
    rawPayload: unknown;
    requestKey: string;
    payloadHash: string;
  };
}

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

type TopLevelRoute = 'pipeline' | 'smalltalk' | 'general_astro';

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
  | 'remedies'
  | 'relocation'
  | 'past_life'
  | 'pregnancy_fertility'
  | 'legal'
  | 'education'
  | 'children'
  | 'property'
  | 'travel'
  | 'spirituality'
  | 'timing'
  | 'yoga'
  | 'family';
type ChartLayer = 'D1' | 'D2' | 'D3' | 'D4' | 'D5' | 'D6' | 'D7' | 'D8' | 'D9' | 'D10' | 'D11' | 'D12' | 'D16' | 'D20' | 'D24' | 'D27' | 'D30';
type MicroSignal = 'nakshatra' | 'nakshatra_lord' | 'sign_lord' | 'drishti' | 'degree';
type TimeSource = 'client' | 'server';
type TimeDirection = 'past' | 'future' | 'present';
type TimeUnit = 'day' | 'week' | 'month' | 'year';
type ResponseStyle = 'micro' | 'brief' | 'normal' | 'expand';
type FastAnswerIntentKind = 'identity' | 'capability' | 'smalltalk' | 'general_astro';
type QuestionTypeRoute = 'timing_marriage' | 'timing_career' | 'timing_general' | 'relationship_general' | 'career_general' | 'other';
type MiniScopeEnforcementMode = 'full' | 'restricted' | 'blocked';
type ToolGroupKey =
  | 'reference_time'
  | 'atlas'
  | 'varga'
  | 'arudha'
  | 'd9'
  | 'dasha'
  | 'transit'
  | 'career'
  | 'remedies'
  | 'relocation'
  | 'past_life'
  | 'pregnancy_fertility'
  | 'legal'
  | 'finance'
  | 'health'
  | 'education'
  | 'children'
  | 'property'
  | 'travel'
  | 'spirituality'
  | 'longevity'
  | 'placement'
  | 'panchanga'
  | 'feature'
  | 'general_grounding'
  | 'nakshatra_lord'
  | 'drishti_degree';

type LlmDecisionMode = 'deterministic' | 'hybrid' | 'llm_first';
type ToolCostClass = 'low' | 'medium' | 'high';

type ToolCapability = {
  group: ToolGroupKey;
  domains: string[];
  requiredScopes: string[];
  minMode: AgentMode;
  costClass: ToolCostClass;
  fallbackGroup?: ToolGroupKey;
};

type ToolAvailabilityPreflight = {
  availableGroups: ToolGroupKey[];
  blockedByMode: ToolGroupKey[];
  missingByData: ToolGroupKey[];
  notes: string[];
};

type DecisionBundle = {
  source: 'deterministic' | 'llm' | 'hybrid';
  topRoute?: TopLevelRoute;
  responseStyle?: ResponseStyle;
  continuityIntent?: boolean;
  intentPrimary?: IntentPrimary;
  questionFamily?: QuestionFamily;
  timeDirection?: TimeDirection;
  requiredScopes?: string[];
  requiredToolGroups?: ToolGroupKey[];
  miniEnforcementMode?: MiniScopeEnforcementMode;
  confidence?: number;
  reason?: string;
};

type AnalysisStage = {
  id: string;
  label: string;
  status: 'completed';
  details?: string;
};

type DecisionTelemetry = {
  node: string;
  model: string;
  latencyMs: number;
  confidence?: number;
  usedFallback: boolean;
  fallbackReason?: string;
  shadowComparison?: string;
};

const IntentRouteDecisionSchema = z.object({
  topRoute: z.enum(['pipeline', 'smalltalk', 'general_astro']),
  responseStyle: z.enum(['micro', 'brief', 'normal', 'expand']).default('brief'),
  continuityIntent: z.boolean().default(false),
  confidence: z.number().min(0).max(1).default(0.5),
  reasoningBrief: z.string().default(''),
  requiresPersonalChart: z.boolean().default(false),
});

type IntentRouteDecision = z.infer<typeof IntentRouteDecisionSchema>;

const IntentDecisionSchema = z.object({
  primary: z.enum(['d9', 'dasha', 'transit', 'general']).default('general'),
  flags: z.array(z.string()).default([]),
  topics: z.array(z.string()).default([]),
  timeDirection: z.enum(['past', 'future', 'present']).default('present'),
  timeValue: z.number().int().positive().optional(),
  timeUnit: z.enum(['day', 'week', 'month', 'year']).optional(),
  timeLabel: z.string().optional(),
  confidence: z.number().min(0).max(1).default(0.5),
});

type IntentDecision = z.infer<typeof IntentDecisionSchema>;

const ExecutionPlanDecisionSchema = z.object({
  questionFamily: z.string(),
  requiredChartLayers: z.array(z.string()).default([]),
  includeMicroSignals: z.array(z.string()).default([]),
  includeTiming: z.boolean().default(false),
  includeTransit: z.boolean().default(false),
  includeDasha: z.boolean().default(false),
  includeCareer: z.boolean().default(false),
  includeRelationship: z.boolean().default(false),
  confidence: z.number().min(0).max(1).default(0.5),
});

type ExecutionPlanDecision = z.infer<typeof ExecutionPlanDecisionSchema>;

const EvidenceGateDecisionSchema = z.object({
  sufficientCoverageAchieved: z.boolean().default(false),
  gapsIdentified: z.array(z.string()).default([]),
  shouldRetry: z.boolean().default(false),
  nextAction: z.enum(['refine_tools', 'build_prompt']).default('build_prompt'),
  confidence: z.number().min(0).max(1).default(0.5),
});

type EvidenceGateDecision = z.infer<typeof EvidenceGateDecisionSchema>;

const ResponsePolicyDecisionSchema = z.object({
  shouldCondense: z.boolean().default(false),
  tone: z.enum(['confident', 'balanced', 'cautious']).default('balanced'),
  addDisclaimer: z.boolean().default(false),
  confidence: z.number().min(0).max(1).default(0.5),
});

const ToolSelectionDecisionSchema = z.object({
  selectedToolGroups: z.array(z.string()).default([]),
  confidence: z.number().min(0).max(1).default(0.5),
});

type ToolSelectionDecision = z.infer<typeof ToolSelectionDecisionSchema>;

const PlanAndToolsDecisionSchema = z.object({
  questionFamily: z.string(),
  requiredChartLayers: z.array(z.string()).default([]),
  includeMicroSignals: z.array(z.string()).default([]),
  includeTiming: z.boolean().default(false),
  includeTransit: z.boolean().default(false),
  includeDasha: z.boolean().default(false),
  includeCareer: z.boolean().default(false),
  includeRelationship: z.boolean().default(false),
  selectedToolGroups: z.array(z.string()).default([]),
  confidence: z.number().min(0).max(1).default(0.5),
});

type PlanAndToolsDecision = z.infer<typeof PlanAndToolsDecisionSchema>;

const MiniScopeDecisionSchema = z.object({
  allowed: z.boolean().default(true),
  enforcementMode: z.enum(['full', 'restricted', 'blocked']).default('full'),
  reasons: z.array(z.string()).default([]),
  suggestedAlternative: z.string().optional(),
  confidence: z.number().min(0).max(1).default(0.5),
});

type MiniScopeDecision = z.infer<typeof MiniScopeDecisionSchema>;

const FastAnswerIntentDecisionSchema = z.object({
  intentKind: z.enum(['identity', 'capability', 'smalltalk', 'general_astro']).default('general_astro'),
  responseStyle: z.enum(['brief', 'normal']).default('brief'),
  confidence: z.number().min(0).max(1).default(0.5),
});

type FastAnswerIntentDecision = z.infer<typeof FastAnswerIntentDecisionSchema>;

const TemporalWindowDecisionSchema = z.object({
  direction: z.enum(['past', 'future', 'present']).default('present'),
  timeValue: z.number().int().positive().optional(),
  timeUnit: z.enum(['day', 'week', 'month', 'year']).optional(),
  timeLabel: z.string().optional(),
  absStartTs: z.number().int().optional(),
  absEndTs: z.number().int().optional(),
  dashaPeriodRef: z.string().optional(),
  confidence: z.number().min(0).max(1).default(0.5),
});

type TemporalWindowDecision = z.infer<typeof TemporalWindowDecisionSchema>;

const ScopeSelectionDecisionSchema = z.object({
  questionType: z.enum(['timing_marriage', 'timing_career', 'timing_general', 'relationship_general', 'career_general', 'other']).default('other'),
  requiredScopes: z.array(z.string()).default([]),
  requiredTools: z.array(z.string()).default([]),
  needsDasha: z.boolean().default(false),
  needsTransit: z.boolean().default(false),
  needsD9: z.boolean().default(false),
  needsD10: z.boolean().default(false),
  needsLongevity: z.boolean().default(false),
  needsGeneral: z.boolean().default(true),
  confidence: z.number().min(0).max(1).default(0.5),
});

type ScopeSelectionDecision = z.infer<typeof ScopeSelectionDecisionSchema>;

const UnifiedIntentScopeDecisionSchema = z.object({
  primary: z.enum(['d9', 'dasha', 'transit', 'general']).default('general'),
  flags: z.array(z.string()).default([]),
  topics: z.array(z.string()).default([]),
  timeDirection: z.enum(['past', 'future', 'present']).default('present'),
  timeValue: z.number().int().positive().optional(),
  timeUnit: z.enum(['day', 'week', 'month', 'year']).optional(),
  timeLabel: z.string().optional(),
  questionType: z.enum(['timing_marriage', 'timing_career', 'timing_general', 'relationship_general', 'career_general', 'other']).default('other'),
  requiredScopes: z.array(z.string()).default([]),
  requiredTools: z.array(z.string()).default([]),
  needsDasha: z.boolean().default(false),
  needsTransit: z.boolean().default(false),
  needsD9: z.boolean().default(false),
  needsD10: z.boolean().default(false),
  needsLongevity: z.boolean().default(false),
  needsGeneral: z.boolean().default(true),
  confidence: z.number().min(0).max(1).default(0.5),
});

type UnifiedIntentScopeDecision = z.infer<typeof UnifiedIntentScopeDecisionSchema>;

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

type CoverageGap =
  | 'varga'
  | 'd9'
  | 'dasha'
  | 'transit'
  | 'career'
  | 'remedies'
  | 'relocation'
  | 'past_life'
  | 'pregnancy_fertility'
  | 'legal'
  | 'finance'
  | 'health'
  | 'education'
  | 'children'
  | 'property'
  | 'travel'
  | 'spirituality'
  | 'longevity';

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
  decisionTelemetry?: DecisionTelemetry[];
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
  profileId: Annotation<string | null>,
  mode: Annotation<AgentMode>,
  question: Annotation<string>,
  kundliInput: Annotation<KundliSnapshotInput | null>,
  referenceTimestamp: Annotation<number | null>,
  referenceTimeSource: Annotation<TimeSource | null>,
  conversationContext: Annotation<string[]>,
  topLevelRoute: Annotation<TopLevelRoute | null>,
  topLevelRouteConfidence: Annotation<number | null>,
  responseStyleHint: Annotation<ResponseStyle | null>,
  continuationIntent: Annotation<boolean | null>,
  decisionTelemetry: Annotation<DecisionTelemetry[]>,
  stageReporter: Annotation<((stage: AnalysisStage) => void) | null>,
  toolIteration: Annotation<number>,
  maxToolIterations: Annotation<number>,
  temporalWindow: Annotation<TemporalWindowDecision | null>,
  intent: Annotation<QuestionIntent | null>,
  scopeSelection: Annotation<ScopeSelectionDecision | null>,
  selectedToolGroups: Annotation<ToolGroupKey[] | null>,
  executionPlan: Annotation<DynamicExecutionPlan | null>,
  coverageGaps: Annotation<CoverageGap[]>,
  coverageShouldRetry: Annotation<boolean | null>,
  coverageDecisionConfidence: Annotation<number | null>,
  refinementNextAction: Annotation<'refine_tools' | 'build_prompt' | null>,
  grounding: Annotation<GroundingContext | null>,
  analysisStages: Annotation<AnalysisStage[]>,
  toolFindings: Annotation<ToolFinding[]>,
  prompt: Annotation<string | null>,
  answer: Annotation<string | null>,
  responseShouldCondense: Annotation<boolean | null>,
  responsePolicyTone: Annotation<'confident' | 'balanced' | 'cautious' | null>,
  responsePolicyAddDisclaimer: Annotation<boolean | null>,
  responsePolicyConfidence: Annotation<number | null>,
  model: Annotation<string | null>,
  decisionBundle: Annotation<DecisionBundle | null>,
  toolAvailabilityPreflight: Annotation<ToolAvailabilityPreflight | null>,
});

type AgentStateType = typeof AgentState.State;
type AgentUpdateType = typeof AgentState.Update;

const CONCISE_ANSWER_MAX_CHARS = 900;
const CONCISE_ANSWER_MIN_LINES = 6;

const QUESTION_FAMILY_ALLOWLIST = new Set<QuestionFamily>([
  'general',
  'career',
  'marriage',
  'relationship',
  'finance',
  'health',
  'longevity',
  'remedies',
  'relocation',
  'past_life',
  'pregnancy_fertility',
  'legal',
  'education',
  'children',
  'property',
  'travel',
  'spirituality',
  'timing',
  'yoga',
  'family',
]);

const CHART_LAYER_ALLOWLIST = new Set<ChartLayer>(['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D16', 'D20', 'D24', 'D27', 'D30']);
const MICRO_SIGNAL_ALLOWLIST = new Set<MicroSignal>(['nakshatra', 'nakshatra_lord', 'sign_lord', 'drishti', 'degree']);
const COVERAGE_GAP_ALLOWLIST = new Set<CoverageGap>([
  'varga',
  'd9',
  'dasha',
  'transit',
  'career',
  'remedies',
  'relocation',
  'past_life',
  'pregnancy_fertility',
  'legal',
  'finance',
  'health',
  'education',
  'children',
  'property',
  'travel',
  'spirituality',
  'longevity',
]);
const TOOL_GROUP_ALLOWLIST = new Set<ToolGroupKey>([
  'reference_time',
  'atlas',
  'varga',
  'arudha',
  'd9',
  'dasha',
  'transit',
  'career',
  'remedies',
  'relocation',
  'past_life',
  'pregnancy_fertility',
  'legal',
  'finance',
  'health',
  'education',
  'children',
  'property',
  'travel',
  'spirituality',
  'longevity',
  'placement',
  'panchanga',
  'feature',
  'general_grounding',
  'nakshatra_lord',
  'drishti_degree',
]);

const MODE_RANK: Record<AgentMode, number> = { mini: 0, pro: 1 };

const TOOL_CAPABILITY_MANIFEST: Record<ToolGroupKey, ToolCapability> = {
  reference_time: {
    group: 'reference_time',
    domains: ['timing', 'general'],
    requiredScopes: [],
    minMode: 'mini',
    costClass: 'low',
  },
  atlas: {
    group: 'atlas',
    domains: ['general', 'discovery'],
    requiredScopes: ['chart'],
    minMode: 'mini',
    costClass: 'low',
  },
  varga: {
    group: 'varga',
    domains: ['general', 'varga'],
    requiredScopes: ['chart.varga'],
    minMode: 'mini',
    costClass: 'medium',
  },
  arudha: {
    group: 'arudha',
    domains: ['advanced', 'arudha'],
    requiredScopes: ['chart.arudha'],
    minMode: 'pro',
    costClass: 'medium',
    fallbackGroup: 'general_grounding',
  },
  d9: {
    group: 'd9',
    domains: ['relationship', 'marriage', 'varga'],
    requiredScopes: ['chart.varga.D9'],
    minMode: 'mini',
    costClass: 'medium',
    fallbackGroup: 'varga',
  },
  dasha: {
    group: 'dasha',
    domains: ['timing', 'forecast'],
    requiredScopes: ['chart.dasha'],
    minMode: 'pro',
    costClass: 'high',
    fallbackGroup: 'reference_time',
  },
  transit: {
    group: 'transit',
    domains: ['timing', 'forecast', 'gochar'],
    // Transit analyzer fetches live transit data from backend:/api/transit-chart;
    // it must not depend on chart.transit existing in canonical payload.
    requiredScopes: [],
    minMode: 'pro',
    costClass: 'high',
    fallbackGroup: 'reference_time',
  },
  career: {
    group: 'career',
    domains: ['career'],
    requiredScopes: ['chart.varga.D10'],
    minMode: 'pro',
    costClass: 'high',
    fallbackGroup: 'varga',
  },
  remedies: {
    group: 'remedies',
    domains: ['remedies', 'healing', 'mitigation'],
    requiredScopes: ['chart.varga.D16', 'chart.varga.D27', 'chart.dasha'],
    minMode: 'pro',
    costClass: 'medium',
    fallbackGroup: 'general_grounding',
  },
  relocation: {
    group: 'relocation',
    domains: ['relocation', 'migration', 'travel'],
    requiredScopes: ['chart.varga.D4', 'chart.varga.D12', 'chart.dasha'],
    minMode: 'pro',
    costClass: 'high',
    fallbackGroup: 'travel',
  },
  past_life: {
    group: 'past_life',
    domains: ['karma', 'past_life', 'spirituality'],
    requiredScopes: ['chart.varga.D1', 'chart.graha', 'chart.bhava'],
    minMode: 'pro',
    costClass: 'medium',
    fallbackGroup: 'spirituality',
  },
  pregnancy_fertility: {
    group: 'pregnancy_fertility',
    domains: ['children', 'fertility', 'pregnancy'],
    requiredScopes: ['chart.varga.D5', 'chart.varga.D7', 'chart.dasha'],
    minMode: 'pro',
    costClass: 'high',
    fallbackGroup: 'children',
  },
  legal: {
    group: 'legal',
    domains: ['legal', 'litigation', 'disputes'],
    requiredScopes: ['chart.varga.D3', 'chart.dasha'],
    minMode: 'pro',
    costClass: 'high',
    fallbackGroup: 'general_grounding',
  },
  finance: {
    group: 'finance',
    domains: ['finance', 'wealth', 'assets'],
    requiredScopes: ['chart.varga.D2', 'chart.varga.D11'],
    minMode: 'pro',
    costClass: 'high',
    fallbackGroup: 'varga',
  },
  health: {
    group: 'health',
    domains: ['health', 'wellbeing', 'risk_profile'],
    requiredScopes: ['chart.varga.D6', 'chart.varga.D8', 'chart.varga.D30'],
    minMode: 'pro',
    costClass: 'high',
    fallbackGroup: 'general_grounding',
  },
  education: {
    group: 'education',
    domains: ['education', 'study', 'academics'],
    requiredScopes: ['chart.varga.D24', 'chart.varga.D4'],
    minMode: 'pro',
    costClass: 'medium',
    fallbackGroup: 'varga',
  },
  children: {
    group: 'children',
    domains: ['children', 'progeny', 'family_growth'],
    requiredScopes: ['chart.varga.D7', 'chart.varga.D5'],
    minMode: 'pro',
    costClass: 'medium',
    fallbackGroup: 'varga',
  },
  property: {
    group: 'property',
    domains: ['property', 'home', 'assets'],
    requiredScopes: ['chart.varga.D4', 'chart.varga.D2'],
    minMode: 'pro',
    costClass: 'medium',
    fallbackGroup: 'varga',
  },
  travel: {
    group: 'travel',
    domains: ['travel', 'foreign', 'relocation'],
    requiredScopes: ['chart.varga.D12', 'chart.varga.D9'],
    minMode: 'pro',
    costClass: 'medium',
    fallbackGroup: 'varga',
  },
  spirituality: {
    group: 'spirituality',
    domains: ['spirituality', 'sadhana', 'moksha'],
    requiredScopes: ['chart.varga.D20', 'chart.varga.D9'],
    minMode: 'pro',
    costClass: 'medium',
    fallbackGroup: 'general_grounding',
  },
  longevity: {
    group: 'longevity',
    domains: ['longevity', 'risk_profile'],
    requiredScopes: ['chart.varga.D8', 'chart.varga.D30', 'chart.dasha'],
    minMode: 'pro',
    costClass: 'high',
    fallbackGroup: 'general_grounding',
  },
  placement: {
    group: 'placement',
    domains: ['general', 'planetary_positions'],
    requiredScopes: ['chart.varga.D1'],
    minMode: 'mini',
    costClass: 'low',
  },
  panchanga: {
    group: 'panchanga',
    domains: ['calendar', 'muhurta', 'nakshatra'],
    requiredScopes: ['chart.panchanga'],
    minMode: 'mini',
    costClass: 'low',
    fallbackGroup: 'general_grounding',
  },
  feature: {
    group: 'feature',
    domains: ['advanced', 'yoga', 'ashtakavarga'],
    requiredScopes: ['chart.yogas', 'chart.ashtakavarga', 'chart.arudha'],
    minMode: 'pro',
    costClass: 'medium',
    fallbackGroup: 'general_grounding',
  },
  general_grounding: {
    group: 'general_grounding',
    domains: ['general'],
    requiredScopes: ['chart.graha', 'chart.lagna'],
    minMode: 'mini',
    costClass: 'low',
  },
  nakshatra_lord: {
    group: 'nakshatra_lord',
    domains: ['nakshatra', 'lordship'],
    requiredScopes: ['chart.graha'],
    minMode: 'mini',
    costClass: 'low',
    fallbackGroup: 'general_grounding',
  },
  drishti_degree: {
    group: 'drishti_degree',
    domains: ['drishti', 'degree'],
    requiredScopes: ['chart.graha'],
    minMode: 'mini',
    costClass: 'medium',
    fallbackGroup: 'general_grounding',
  },
};

function getToolManifestForMode(mode: AgentMode): ToolCapability[] {
  return Object.values(TOOL_CAPABILITY_MANIFEST).filter((capability) => MODE_RANK[mode] >= MODE_RANK[capability.minMode]);
}

function hasAnyPath(rawPayload: unknown, paths: string[]): boolean {
  return paths.some((path) => getByPath(rawPayload, path) !== undefined);
}

function buildToolAvailabilityPreflight(rawPayload: unknown, mode: AgentMode): ToolAvailabilityPreflight {
  const availableGroups: ToolGroupKey[] = [];
  const blockedByMode: ToolGroupKey[] = [];
  const missingByData: ToolGroupKey[] = [];
  const notes: string[] = [];

  for (const capability of Object.values(TOOL_CAPABILITY_MANIFEST)) {
    const modeBlocked = MODE_RANK[mode] < MODE_RANK[capability.minMode];
    if (modeBlocked) {
      blockedByMode.push(capability.group);
      continue;
    }

    const hasRequiredData = capability.requiredScopes.length === 0 || hasAnyPath(rawPayload, capability.requiredScopes);
    if (hasRequiredData) {
      availableGroups.push(capability.group);
    } else {
      missingByData.push(capability.group);
    }
  }

  if (mode === 'mini' && blockedByMode.length > 0) {
    notes.push(`mini-mode restrictions active for: ${blockedByMode.join(', ')}`);
  }

  if (missingByData.length > 0) {
    notes.push(`payload gaps affecting tools: ${missingByData.join(', ')}`);
  }

  return {
    availableGroups: [...new Set(availableGroups)],
    blockedByMode: [...new Set(blockedByMode)],
    missingByData: [...new Set(missingByData)],
    notes,
  };
}

function resolveDecisionMode(): LlmDecisionMode {
  return (env.LLM_DECISION_MODE as LlmDecisionMode) ?? 'hybrid';
}

function isDecisionNodeEnabled(legacyNodeFlag: boolean): boolean {
  if (!env.LLM_DECISION_ENABLED) {
    return false;
  }

  const mode = resolveDecisionMode();
  if (mode === 'deterministic') {
    return false;
  }

  if (mode === 'llm_first') {
    return true;
  }

  return legacyNodeFlag;
}

function mergeDecisionBundle(state: AgentStateType, patch: Partial<DecisionBundle>): DecisionBundle {
  const current = state.decisionBundle ?? { source: 'deterministic' as const };
  return { ...current, ...patch };
}

function normalizeToken(value: string): string {
  return value.trim().toLowerCase().replace(/[\s-]+/g, '_');
}

function clampQuestionFamily(value: string, fallback: QuestionFamily): QuestionFamily {
  const normalized = normalizeToken(value);
  return QUESTION_FAMILY_ALLOWLIST.has(normalized as QuestionFamily) ? (normalized as QuestionFamily) : fallback;
}

function clampChartLayers(values: string[], fallback: ChartLayer[]): ChartLayer[] {
  const next = values
    .map((item) => String(item).trim().toUpperCase())
    .filter((item): item is ChartLayer => CHART_LAYER_ALLOWLIST.has(item as ChartLayer));

  const unique = [...new Set<ChartLayer>(['D1', ...next])];
  return unique.length > 0 ? unique : fallback;
}

function clampMicroSignals(values: string[], fallback: MicroSignal[]): MicroSignal[] {
  const next = values
    .map((item) => normalizeToken(String(item)))
    .filter((item): item is MicroSignal => MICRO_SIGNAL_ALLOWLIST.has(item as MicroSignal));
  return next.length > 0 ? [...new Set(next)] : fallback;
}

function clampCoverageGaps(values: string[], fallback: CoverageGap[]): CoverageGap[] {
  const next = values
    .map((item) => normalizeToken(String(item)))
    .filter((item): item is CoverageGap => COVERAGE_GAP_ALLOWLIST.has(item as CoverageGap));
  return next.length > 0 ? [...new Set(next)] : fallback;
}

function clampToolGroups(values: string[], fallback: ToolGroupKey[], mode: AgentMode): ToolGroupKey[] {
  const next = values
    .map((item) => normalizeToken(String(item)))
    .filter((item): item is ToolGroupKey => TOOL_GROUP_ALLOWLIST.has(item as ToolGroupKey));

  const unique = [...new Set(next)];
  const withDefaults = unique.length > 0 ? unique : fallback;

  if (mode !== 'mini') {
    return withDefaults;
  }

  const miniBlocked = new Set<ToolGroupKey>([
    'dasha',
    'transit',
    'career',
    'remedies',
    'relocation',
    'past_life',
    'pregnancy_fertility',
    'legal',
    'finance',
    'health',
    'education',
    'children',
    'property',
    'travel',
    'spirituality',
    'longevity',
    'feature',
    'arudha',
  ]);
  const miniSafe = withDefaults.filter((group) => !miniBlocked.has(group));
  if (miniSafe.length > 0) {
    return miniSafe;
  }

  return ['reference_time', 'general_grounding', 'varga', 'placement'];
}

function buildDataGapDisclaimer(findings: ToolFinding[]): string | null {
  const missing = findings
    .filter((f) => f.status === 'unavailable')
    .slice(0, 3)
    .map((f) => f.name);

  if (missing.length === 0) return null;
  return `Note: some analyzers are unavailable right now (${missing.join(', ')}), so this answer is based on the evidence that could be fetched.`;
}

function appendDecisionTelemetry(state: AgentStateType, telemetry: DecisionTelemetry): DecisionTelemetry[] {
  const current = state.decisionTelemetry ?? [];
  return [...current, telemetry];
}

function appendDecisionTelemetries(state: AgentStateType, telemetry: DecisionTelemetry[]): DecisionTelemetry[] {
  const current = state.decisionTelemetry ?? [];
  return [...current, ...telemetry];
}

function clampRequiredScopes(values: string[], fallback: string[], mode: AgentMode): string[] {
  const normalized = values
    .map((item) => String(item).trim())
    .filter((item) => /^chart(?:\.[A-Za-z0-9_]+)+$/.test(item));

  const modeFiltered = mode === 'mini'
    ? normalized.filter((path) => {
      if (
        path === 'chart.user'
        || path === 'chart.graha'
        || path === 'chart.lagna'
        || path === 'chart.houses'
        || path === 'chart.bhava'
        || path === 'chart.panchanga'
        || path === 'chart.yogas'
      ) {
        return true;
      }

      return path.startsWith('chart.varga.D1') || path.startsWith('chart.varga.D9');
    })
    : normalized;

  const withD1 = [...new Set(['chart.varga.D1', ...modeFiltered])];
  if (withD1.length > 0) return withD1;
  return [...new Set(fallback.length > 0 ? fallback : ['chart.varga.D1', 'chart.graha', 'chart.lagna'])];
}

function parseJsonObject(text: string): unknown {
  const direct = text.trim();
  if (!direct) throw new Error('Empty JSON text');

  try {
    return JSON.parse(direct);
  } catch {
    const fenced = direct.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
    if (fenced?.[1]) {
      return JSON.parse(fenced[1]);
    }
    const firstBrace = direct.indexOf('{');
    const lastBrace = direct.lastIndexOf('}');
    if (firstBrace >= 0 && lastBrace > firstBrace) {
      return JSON.parse(direct.slice(firstBrace, lastBrace + 1));
    }
    throw new Error('No JSON object found in model response');
  }
}

interface DecisionNodeCacheEntry extends Record<string, unknown> {
  cachedAt: number;
  model: string;
  decision: Record<string, unknown>;
}

interface GroundingProfileCacheEntry extends Record<string, unknown> {
  cachedAt: number;
  doc: RagProfileDocument;
}

interface GroundingSourceCacheEntry extends Record<string, unknown> {
  cachedAt: number;
  record: RagApiSourceRecord;
}

function buildDecisionNodeCacheKey(node: string, input: Record<string, unknown>): string | null {
  try {
    return `agent:decision:v1:${node}:${stableHash(JSON.stringify(input))}`;
  } catch {
    return null;
  }
}

function buildGroundingProfileCacheKey(ownerId: string, profileId: string): string {
  return `agent:grounding:profile:${stableHash(JSON.stringify({ ownerId, profileId }))}`;
}

function buildGroundingSourceCacheKey(sourceDocId: string): string {
  return `agent:grounding:source:${stableHash(sourceDocId)}`;
}

async function invokeDecisionNode<T extends Record<string, unknown>>(params: {
  node: string;
  schema: z.ZodType<T, z.ZodTypeDef, unknown>;
  input: Record<string, unknown>;
  fallback: () => T;
}): Promise<{ decision: T; model: string; usedFallback: boolean; latencyMs: number }> {
  const startedAt = Date.now();

  const decisionCacheKey = buildDecisionNodeCacheKey(params.node, params.input);
  if (decisionCacheKey) {
    const cached = await cacheGetJson<DecisionNodeCacheEntry>(decisionCacheKey);
    if (cached?.decision && isPlainObject(cached.decision)) {
      try {
        const parsed = params.schema.parse(cached.decision);
        return {
          decision: parsed,
          model: `${cached.model}|valkey-hit`,
          usedFallback: false,
          latencyMs: Date.now() - startedAt,
        };
      } catch {
        // ignore malformed cache entry and continue to fresh model inference
      }
    }
  }

  try {
    const response = await invokeDeepSeekBedrock({
      systemPrompt: [
        'You are a strict JSON decision engine for an astrology agent.',
        'Return ONE JSON object only. No markdown. No prose.',
        `Decision node: ${params.node}`,
      ].join(' '),
      userPrompt: JSON.stringify(params.input),
      maxTokens: 300,
    });

    const parsed = params.schema.parse(parseJsonObject(response.text));

    if (decisionCacheKey) {
      await cacheSetJson(
        decisionCacheKey,
        {
          cachedAt: Date.now(),
          model: response.model,
          decision: parsed,
        },
        Math.max(1, env.AGENT_CACHE_TTL_SECONDS)
      );
    }

    return {
      decision: parsed,
      model: response.model,
      usedFallback: false,
      latencyMs: Date.now() - startedAt,
    };
  } catch {
    return {
      decision: params.fallback(),
      model: 'decision-fallback',
      usedFallback: true,
      latencyMs: Date.now() - startedAt,
    };
  }
}

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

function formatNumber(value: unknown, digits = 2): string {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : String(value ?? 'n/a');
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
    { pattern: /\bdasamsa\b|\bdashamsha\b|\bd10\b/, key: 'D10' },
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

function miniScopeSeverity(mode: MiniScopeEnforcementMode): number {
  switch (mode) {
    case 'full':
      return 0;
    case 'restricted':
      return 1;
    case 'blocked':
      return 2;
    default:
      return 2;
  }
}

function isMiniTransitTimingHardBlock(question: string): boolean {
  const q = question.toLowerCase();

  const explicitTransit = /\b(transit|gochar)\b/.test(q);
  const explicitTimingAnalysis = /\b(timing\s+analysis|predictive\s+analysis|prediction\s+analysis|forecast\s+analysis|future\s+prediction|astrological\s+prediction|transit\s+analysis|gochar\s+analysis)\b/.test(q);
  const predictiveQuestionCue = /\b(when\s+will|by\s+when|which\s+year|what\s+age)\b/.test(q)
    && /\b(marriage|relationship|partner|spouse|career|job|business|promotion|finance|money|health|property|travel|children|pregnancy|fertility)\b/.test(q);
  const explicitDashaTiming = /\b(dasha|dasa|mahadasha|antardasha|vimshottari)\b/.test(q);

  return explicitTransit || explicitTimingAnalysis || predictiveQuestionCue || explicitDashaTiming;
}

function evaluateMiniScope(question: string): {
  allowed: boolean;
  enforcementMode: MiniScopeEnforcementMode;
  reasons: string[];
  suggestedAlternative?: string;
} {
  const restrictedReasons: string[] = [];
  const blockedReasons: string[] = [];
  const q = question.toLowerCase();

  const vargaKeys = extractRequestedVargaKeys(question);
  const requestedNonMiniVargas = vargaKeys.filter((key) => key !== 'D1' && key !== 'D9');
  if (requestedNonMiniVargas.length > 0) {
    blockedReasons.push(`Mini supports D1/D9 analysis only. Requested: ${requestedNonMiniVargas.join(', ')}.`);
  }

  const explicitProSections = [
    /\b(dasha|dasa|mahadasha|antardasha|vimshottari|transit|gochar|forecast|prediction|predictive timing)\b/,
    /\b(ashtakavarga|arudha|aruda|shadbala|kp|jaimini|nadi)\b/,
    /\b(longevity|lifespan|life span|how long will i live|length of life|ayush|ayu|mrityu|death|end of life)\b/,
    /\b(remedy|remedies|upay|upaya|mantra|gemstone|puja|pooja|relocation|migration|past life|karmic|pregnancy|fertility|conception|legal|litigation|court|lawsuit|dispute)\b/,
    /\b(d10|dashamsha|dasamsa|d8|d30|career\s+chart|profession\s+chart)\b/,
  ];

  if (explicitProSections.some((pattern) => pattern.test(q))) {
    blockedReasons.push('This request is in Pro-only analysis scope.');
  }

  const hasAstroCue = /\b(kundli|chart|horoscope|astrology|vedic|rashi|lagna|nakshatra|graha|planet|d1|d9|dasha|gochar|transit|marriage|career|finance|health|remedy|relationship)\b/.test(q);
  const hasTransitCue = /\b(transit|gochar)\b/.test(q);
  const hasTimingCue = /\b(timing|timeline|prediction|predict|forecast|when\s+will|by\s+when|which\s+year|what\s+age|next\s+(week|month|year)|this\s+(week|month|year)|today|tomorrow|future|past)\b/.test(q);
  const transitOrTimingAnalysisRequest = isMiniTransitTimingHardBlock(question) || hasTransitCue || (hasTimingCue && hasAstroCue);

  if (transitOrTimingAnalysisRequest) {
    blockedReasons.push('Transit, timing, and predictive analysis are available only in Pro mode.');
  }

  const intent = classifyQuestionIntent(question);
  const family = determineQuestionFamily(question, intent);
  const restrictedFamilies = new Set<QuestionFamily>([
    'career',
    'finance',
    'health',
    'education',
    'children',
    'property',
    'travel',
    'spirituality',
    'family',
  ]);

  const proOnlyFamilies = new Set<QuestionFamily>([
    'longevity',
    'remedies',
    'relocation',
    'past_life',
    'pregnancy_fertility',
    'legal',
  ]);

  if (proOnlyFamilies.has(family)) {
    blockedReasons.push(`${family.replace(/_/g, ' ')} analysis is Pro scope.`);
  } else if (restrictedFamilies.has(family)) {
    restrictedReasons.push(`This is ${family} analysis; mini will use foundational D1/D9 scope.`);
  }

  const proTimingFlags = new Set(['timing', 'dasha', 'transit', 'forecast', 'history', 'career_timing']);
  if (intent.flags.some((flag) => proTimingFlags.has(flag)) && (hasAstroCue || hasTransitCue || intent.topics.length > 0)) {
    blockedReasons.push('This requires predictive timing analysis (dasha/transit/forecast), which is Pro scope.');
  }

  if (/(\bashtakavarga\b|\barudha\b|\baruda\b|\bshadbala\b|\bkp\b|\bjaimini\b|\bnadi\b)/.test(q)) {
    blockedReasons.push('This requests advanced systems reserved for Pro mode.');
  }

  const enforcementMode: MiniScopeEnforcementMode = blockedReasons.length > 0
    ? 'blocked'
    : restrictedReasons.length > 0
      ? 'restricted'
      : 'full';

  const reasons = [...new Set([...blockedReasons, ...restrictedReasons])];

  return {
    allowed: enforcementMode !== 'blocked',
    enforcementMode,
    reasons,
    suggestedAlternative:
      enforcementMode === 'restricted'
        ? 'I can provide D1/D9-based foundational guidance in Mini mode.'
        : undefined,
  };
}

function buildMiniUpgradeResponse(question: string, reasons: string[] = []): string {
  const transitTimingBlock = isMiniTransitTimingHardBlock(question)
    || reasons.some((reason) => /transit|gochar|timing|predict/i.test(reason));

  if (transitTimingBlock) {
    return [
      'Transit, timing, and predictive analysis are not available in **Cozmic Mini**.',
      'In Mini mode, I can help with small talk and basic D1/D9 astrology only.',
      'Switch to **Cozmic Pro** for transit/gochar and timing analysis.',
    ].join('\n');
  }

  const leadReason = reasons[0]?.replace(/[^\x09\x0A\x0D\x20-\x7E]/g, ' ').trim();
  return [
    'This analysis is not available in **Cozmic Mini**.',
    leadReason ? `Reason: ${truncateText(leadReason, 220)}` : 'Mini supports D1/D9 foundational insights only.',
    'Switch to **Cozmic Pro** to use this analysis.',
  ].join('\n');
}

function buildMiniRestrictedNotice(reasons: string[] = [], suggestedAlternative?: string): string {
  const leadReason = reasons[0]?.trim();
  return [
    'Mini scope note: answer constrained to D1/D9 foundational guidance.',
    leadReason ? `Reason: ${leadReason}` : (suggestedAlternative ?? 'For full advanced analysis, switch to Cozmic Pro.'),
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
  if (/\b(remedy|remedies|upay|upaya|mantra|gemstone|puja|pooja|fasting)\b/.test(q)) {
    add('chart.varga.D16', 'chart.varga.D27', 'chart.dasha', 'chart.varga.D1', 'chart.graha', 'chart.bhava');
  }
  if (/\b(relocation|migrate|migration|settle abroad|foreign settlement|move abroad)\b/.test(q)) {
    add('chart.varga.D4', 'chart.varga.D12', 'chart.dasha', 'chart.transit', 'chart.transits', 'chart.gochar', 'chart.varga.D1');
  }
  if (/\b(past life|past-life|karma|karmic|reincarnation|soul purpose)\b/.test(q)) {
    add('chart.varga.D1', 'chart.graha', 'chart.bhava');
  }
  if (/\b(pregnancy|fertility|conceive|conception|childbirth|delivery|baby)\b/.test(q)) {
    add('chart.varga.D5', 'chart.varga.D7', 'chart.dasha', 'chart.transit', 'chart.transits', 'chart.gochar', 'chart.varga.D1');
  }
  if (/\b(legal|court|litigation|lawsuit|dispute|case)\b/.test(q)) {
    add('chart.varga.D3', 'chart.dasha', 'chart.transit', 'chart.transits', 'chart.gochar', 'chart.varga.D1', 'chart.bhava');
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

function extractLikelyTemporalYears(question: string): number[] {
  const currentYear = new Date().getUTCFullYear();
  const minYear = currentYear - 15;
  const maxYear = currentYear + 30;
  const years = new Set<number>();

  for (const match of question.matchAll(/\b(20\d{2})\b/g)) {
    const year = Number(match[1]);
    if (Number.isInteger(year) && year >= minYear && year <= maxYear) {
      years.add(year);
    }
  }

  return [...years].sort((a, b) => a - b);
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

  const ageBasedFuture = q.match(/\b(?:after|from)\s+age\s+(\d{1,2})\b/);
  if (ageBasedFuture) {
    const age = Number(ageBasedFuture[1]);
    return { direction: 'future', label: `after age ${age}` };
  }

  const rangePattern = q.match(/\b(?:from|between)\b[\s\S]{0,40}\b(?:to|and)\b/);
  if (rangePattern) {
    return { direction: 'future', label: 'explicit date range' };
  }

  const compactYearRange = /\b(20\d{2})\s*[-/]\s*(\d{2}|20\d{2})\b(?!\s*[-/]\s*\d{1,2})/.exec(q);
  if (compactYearRange) {
    const startYear = Number(compactYearRange[1]);
    const endRaw = compactYearRange[2];
    let endYear = Number(endRaw);
    if (endRaw.length === 2) {
      const century = Math.floor(startYear / 100) * 100;
      endYear = century + endYear;
      if (endYear < startYear) endYear += 100;
    }
    const spanYears = Math.max(1, endYear - startYear + 1);
    return { direction: endYear >= new Date().getUTCFullYear() ? 'future' : 'past', value: spanYears, unit: 'year', label: `${startYear}-${endYear}` };
  }

  const likelyYears = extractLikelyTemporalYears(question);
  if (likelyYears.length >= 2 && (/\b(or|and|to|till|until|through|between|from)\b/.test(q) || /[?,/]/.test(q))) {
    const startYear = likelyYears[0];
    const endYear = likelyYears[likelyYears.length - 1];
    const spanYears = Math.max(1, endYear - startYear + 1);
    return { direction: endYear >= new Date().getUTCFullYear() ? 'future' : 'past', value: spanYears, unit: 'year', label: `${startYear}-${endYear}` };
  }

  if (likelyYears.length === 1 && /\b(in|for|during|around|by)\s+20\d{2}\b/.test(q)) {
    const year = likelyYears[0];
    return { direction: year >= new Date().getUTCFullYear() ? 'future' : 'past', value: 1, unit: 'year', label: String(year) };
  }

  if (likelyYears.length === 1 && /\b20\d{2}\b\s*(?:\?|$|[.,!])/.test(q)) {
    const year = likelyYears[0];
    return { direction: year >= new Date().getUTCFullYear() ? 'future' : 'past', value: 1, unit: 'year', label: String(year) };
  }

  if (likelyYears.length >= 2 && /\b20\d{2}\b/.test(q)) {
    const startYear = likelyYears[0];
    const endYear = likelyYears[likelyYears.length - 1];
    const spanYears = Math.max(1, endYear - startYear + 1);
    return { direction: endYear >= new Date().getUTCFullYear() ? 'future' : 'past', value: spanYears, unit: 'year', label: `${startYear}-${endYear}` };
  }

  if (/\b(10|ten)\s+years?\s+ago\b/.test(q) || /\bpast\s+10\s+years?\b/.test(q) || /\blast\s+10\s+years?\b/.test(q)) {
    return { direction: 'past', value: 10, unit: 'year', label: 'past 10 years' };
  }

  if (/\bnext\s+10\s+years?\b/.test(q) || /\bin\s+10\s+years?\b/.test(q)) {
    return { direction: 'future', value: 10, unit: 'year', label: 'next 10 years' };
  }

  if (/\b(tomorrow|next week|next month|next year|upcoming|future|later|after|when will|by when|which year|what age|will it improve|improve in future)\b/.test(q)) {
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
  if (/\b(dasha|dasa|mahadasha|antardasha|vimshottari|period|timing|timeline|when|improve|improvement phase|bad phase|difficult phase)\b/.test(q)) flags.add('dasha');
  if (/\b(transit|gochar|current transit|current sun|sun transit|today|now|tomorrow|next week|next month|next year|future|past|last|previous|ago|from|between)\b/.test(q)) flags.add('transit');

  if (/\b(career|job|profession|business|promotion|work|office|employment|salary|resume|interview)\b/.test(q)) {
    flags.add('career');
    topics.add('career');
  }
  if (/\b(upsc|ias|ips|ifs|ssc|psc|civil service|civil services|government job|govt job|state job|psu|bank po|sarkari)\b/.test(q)) {
    flags.add('career');
    flags.add('career_govt');
    topics.add('career');
  }
  if (/\b(startup|corporate|mnc|private sector|product company|private job|entrepreneurship)\b/.test(q)) {
    flags.add('career');
    flags.add('career_private');
    topics.add('career');
  }
  if (/\b(marriage|relationship|partner|spouse|love|compatibility|romance|dating)\b/.test(q)) {
    flags.add('relationship');
    topics.add('relationship');
  }
  if (/\b(cheated|betray(?:ed|al)|heartbreak|breakup|separation|infidelity)\b/.test(q)) {
    flags.add('relationship');
    flags.add('hardship');
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
  if (/\b(bad\s+time|difficult\s+time|hard\s+time|rough\s+phase|loss|grief|depressed|depression|anxiety|panic|suffering)\b/.test(q)) {
    flags.add('hardship');
    flags.add('timing');
  }
  if (/\b(longevity|lifespan|life span|how long will i live|length of life|ayush|ayu|mrityu|death|end of life)\b/.test(q)) {
    flags.add('longevity');
    flags.add('timing');
    topics.add('health');
  }
  if (/\b(remedy|remedies|upay|upaya|mantra|gemstone|puja|pooja|fasting)\b/.test(q)) {
    flags.add('remedies');
    topics.add('spirituality');
  }
  if (/\b(relocation|migrate|migration|settle abroad|foreign settlement|move abroad)\b/.test(q)) {
    flags.add('relocation');
    flags.add('timing');
    topics.add('travel');
  }
  if (/\b(past life|past-life|karma|karmic|reincarnation|soul purpose)\b/.test(q)) {
    flags.add('past_life');
    topics.add('spirituality');
  }
  if (/\b(pregnancy|fertility|conceive|conception|childbirth|delivery|baby)\b/.test(q)) {
    flags.add('pregnancy_fertility');
    flags.add('timing');
    topics.add('children');
  }
  if (/\b(legal|court|litigation|lawsuit|dispute|case)\b/.test(q)) {
    flags.add('legal');
    flags.add('timing');
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

  if (/\b(time\s+went\s+wrong|bad\s+time|difficult\s+time|why\s+this\s+phase|when\s+will\s+my\s+time\s+improve|how\s+to\s+improve)\b/.test(q)) {
    flags.add('timing');
    flags.add('forecast');
  }

  if (/\b(when|by when|which year|what age|timeline|timing|period|window|phase)\b/.test(q)) {
    flags.add('timing');
  }

  if (/(\bcan\s+i\b|\bwill\s+i\b|\bshould\s+i\b)/.test(q)
      && /\b(marry|marriage|relationship|partner|spouse|career|job|business|promotion|finance|money|health|education|children|property|travel|spiritual|spirituality)\b/.test(q)) {
    flags.add('timing');
    flags.add('forecast');
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

function buildDeterministicTemporalWindow(question: string): TemporalWindowDecision {
  const temporal = parseTemporalWindow(question);
  return {
    direction: temporal.direction,
    timeValue: temporal.value,
    timeUnit: temporal.unit,
    timeLabel: temporal.label,
    confidence: 0.35,
  };
}

function buildDeterministicScopeSelection(question: string, intent: QuestionIntent, mode: AgentMode): ScopeSelectionDecision {
  const q = question.toLowerCase();
  const hasTiming = intent.flags.includes('timing') || intent.flags.includes('dasha') || /\b(when|timing|timeline|period)\b/.test(q);
  const hasCareer = intent.flags.includes('career') || intent.topics.includes('career') || /\b(career|job|profession|business|work|promotion)\b/.test(q);
  const hasRelationship = intent.flags.includes('relationship') || intent.topics.includes('relationship') || /\b(marriage|relationship|partner|spouse|love|compatibility|romance|dating)\b/.test(q);
  const hasLongevity = intent.flags.includes('longevity') || /\b(longevity|lifespan|life span|how long will i live|length of life|ayush|ayu|mrityu|death|end of life)\b/.test(q);
  const hasTransit = intent.flags.includes('transit') || intent.flags.includes('timing') || /\b(transit|gochar|today|now|tomorrow|this month|next month|this year|next year|future|past|last|previous|ago|from|between)\b/.test(q);

  const questionType: QuestionTypeRoute = hasTiming && hasRelationship
    ? 'timing_marriage'
    : hasTiming && hasCareer
      ? 'timing_career'
      : hasTiming
        ? 'timing_general'
        : hasRelationship
          ? 'relationship_general'
          : hasCareer
            ? 'career_general'
            : 'other';

  const requiredScopes = clampRequiredScopes(pickQuestionScope(question), ['chart.varga.D1', 'chart.graha', 'chart.lagna'], mode);
  const needsDasha = hasTiming;
  const needsTransit = hasTransit;
  const needsD9 = hasRelationship || intent.flags.includes('d9');
  const needsD10 = hasCareer;
  const needsLongevity = hasLongevity;
  const needsGeneral = true;

  const requiredTools = [
    needsGeneral ? 'general' : '',
    needsDasha ? 'dasha' : '',
    needsTransit ? 'transit' : '',
    needsD9 ? 'd9' : '',
    needsD10 ? 'career' : '',
    needsLongevity ? 'longevity' : '',
  ].filter(Boolean);

  return {
    questionType,
    requiredScopes,
    requiredTools,
    needsDasha,
    needsTransit,
    needsD9,
    needsD10,
    needsLongevity,
    needsGeneral,
    confidence: 0.35,
  };
}

function applyTemporalToIntent(intent: QuestionIntent, temporal: TemporalWindowDecision): QuestionIntent {
  const nextFlags = new Set(intent.flags);

  if (temporal.direction !== 'present') {
    nextFlags.add('timing');
  }

  if (temporal.direction === 'future') {
    nextFlags.add('forecast');
    nextFlags.delete('history');
  } else if (temporal.direction === 'past') {
    nextFlags.add('history');
    nextFlags.delete('forecast');
  }

  if (intent.topics.includes('career') && temporal.direction !== 'present') {
    nextFlags.add('career_timing');
  }

  return {
    ...intent,
    flags: [...nextFlags],
    timeDirection: temporal.direction,
    timeValue: temporal.timeValue,
    timeUnit: temporal.timeUnit,
    timeLabel: temporal.timeLabel,
  };
}

async function decideTemporalWindowDetailed(
  message: string,
  mode: AgentMode,
  conversationContext: string[] = [],
  referenceTimestamp: number = Date.now()
): Promise<{ decision: TemporalWindowDecision; model: string; usedFallback: boolean; latencyMs: number }> {
  const deterministic = buildDeterministicTemporalWindow(message);

  if (!isDecisionNodeEnabled(env.LLM_DECISION_TEMPORAL_ENABLED)) {
    return {
      decision: deterministic,
      model: 'temporal-disabled',
      usedFallback: true,
      latencyMs: 0,
    };
  }

  const result = await invokeDecisionNode<TemporalWindowDecision>({
    node: 'temporal_parser',
    schema: TemporalWindowDecisionSchema,
    input: {
      question: message,
      mode,
      referenceTimestamp,
      conversationContext: conversationContext.slice(-6),
      instruction:
        'Extract temporal window for astrology intent routing. Return structured time direction and optional value/unit. Use present when uncertain.',
    },
    fallback: () => deterministic,
  });

  return result;
}

async function decideScopeSelectionDetailed(
  question: string,
  mode: AgentMode,
  intent: QuestionIntent,
  temporal: TemporalWindowDecision,
  conversationContext: string[] = []
): Promise<{ decision: ScopeSelectionDecision; model: string; usedFallback: boolean; latencyMs: number }> {
  const deterministic = buildDeterministicScopeSelection(question, applyTemporalToIntent(intent, temporal), mode);

  if (!isDecisionNodeEnabled(env.LLM_DECISION_SCOPE_SELECTOR_ENABLED)) {
    return {
      decision: deterministic,
      model: 'scope-disabled',
      usedFallback: true,
      latencyMs: 0,
    };
  }

  const result = await invokeDecisionNode<ScopeSelectionDecision>({
    node: 'intent_and_scope_router',
    schema: ScopeSelectionDecisionSchema,
    input: {
      question,
      mode,
      intent,
      temporal,
      conversationContext: conversationContext.slice(-6),
      deterministicScopes: deterministic.requiredScopes,
      instruction:
        'Classify question type and choose required canonical scopes/tools for astrology analysis. Prioritize precision and avoid unrelated scopes.',
    },
    fallback: () => deterministic,
  });

  return {
    ...result,
    decision: {
      ...result.decision,
      requiredScopes: clampRequiredScopes(result.decision.requiredScopes, deterministic.requiredScopes, mode),
      requiredTools: [...new Set(result.decision.requiredTools.map((item) => normalizeToken(item)))],
    },
  };
}

function buildDeterministicUnifiedIntentScopeDecision(question: string, mode: AgentMode): UnifiedIntentScopeDecision {
  const deterministicTemporal = buildDeterministicTemporalWindow(question);
  const deterministicIntent = applyTemporalToIntent(classifyQuestionIntent(question), deterministicTemporal);
  const deterministicScope = buildDeterministicScopeSelection(question, deterministicIntent, mode);

  return {
    primary: deterministicIntent.primary,
    flags: deterministicIntent.flags,
    topics: deterministicIntent.topics,
    timeDirection: deterministicTemporal.direction,
    timeValue: deterministicTemporal.timeValue,
    timeUnit: deterministicTemporal.timeUnit,
    timeLabel: deterministicTemporal.timeLabel,
    questionType: deterministicScope.questionType,
    requiredScopes: deterministicScope.requiredScopes,
    requiredTools: deterministicScope.requiredTools,
    needsDasha: deterministicScope.needsDasha,
    needsTransit: deterministicScope.needsTransit,
    needsD9: deterministicScope.needsD9,
    needsD10: deterministicScope.needsD10,
    needsLongevity: deterministicScope.needsLongevity,
    needsGeneral: deterministicScope.needsGeneral,
    confidence: 0.35,
  };
}

async function decideUnifiedIntentScopeDetailed(
  question: string,
  mode: AgentMode,
  conversationContext: string[] = [],
  referenceTimestamp: number = Date.now()
): Promise<{ decision: UnifiedIntentScopeDecision; model: string; usedFallback: boolean; latencyMs: number }> {
  const deterministic = buildDeterministicUnifiedIntentScopeDecision(question, mode);
  const legacyIntentEnabled = isDecisionNodeEnabled(env.LLM_DECISION_INTENT_ENABLED);
  const legacyTemporalEnabled = isDecisionNodeEnabled(env.LLM_DECISION_TEMPORAL_ENABLED);
  const legacyScopeEnabled = isDecisionNodeEnabled(env.LLM_DECISION_SCOPE_SELECTOR_ENABLED);
  const unifiedEnabled = legacyIntentEnabled || legacyTemporalEnabled || legacyScopeEnabled;

  if (!unifiedEnabled) {
    return {
      decision: deterministic,
      model: 'intent-scope-unified-disabled',
      usedFallback: true,
      latencyMs: 0,
    };
  }

  const result = await invokeDecisionNode<UnifiedIntentScopeDecision>({
    node: 'intent_scope_unified',
    schema: UnifiedIntentScopeDecisionSchema,
    input: {
      question,
      mode,
      referenceTimestamp,
      conversationContext: conversationContext.slice(-8),
      deterministic,
      instruction:
        'Extract intent, temporal window, and scope/tool requirements in one JSON. Keep output conservative, mode-aware, and aligned to deterministic hints when uncertain.',
    },
    fallback: () => deterministic,
  });

  return {
    ...result,
    decision: {
      ...result.decision,
      flags: [...new Set(result.decision.flags.map((flag) => normalizeToken(flag)))],
      topics: [...new Set(result.decision.topics.map((topic) => normalizeToken(topic)))],
      requiredScopes: clampRequiredScopes(result.decision.requiredScopes, deterministic.requiredScopes, mode),
      requiredTools: [...new Set(result.decision.requiredTools.map((item) => normalizeToken(item)))],
    },
  };
}

async function decideMiniScopeDetailed(
  question: string,
  mode: AgentMode,
  conversationContext: string[] = []
): Promise<{ decision: MiniScopeDecision; model: string; usedFallback: boolean; latencyMs: number }> {
  if (mode === 'pro') {
    return {
      decision: { allowed: true, enforcementMode: 'full', reasons: [], confidence: 1 },
      model: 'mini-scope-not-required',
      usedFallback: false,
      latencyMs: 0,
    };
  }

  const deterministic = evaluateMiniScope(question);
  const deterministicDecision: MiniScopeDecision = {
    allowed: deterministic.allowed,
    enforcementMode: deterministic.enforcementMode,
    reasons: deterministic.reasons,
    suggestedAlternative: deterministic.suggestedAlternative,
    confidence: 0.35,
  };

  if (!isDecisionNodeEnabled(env.LLM_DECISION_MINI_SCOPE_ENABLED)) {
    return {
      decision: deterministicDecision,
      model: 'mini-scope-disabled',
      usedFallback: true,
      latencyMs: 0,
    };
  }

  const result = await invokeDecisionNode<MiniScopeDecision>({
    node: 'mini_scope_decision',
    schema: MiniScopeDecisionSchema,
    input: {
      question,
      mode,
      conversationContext: conversationContext.slice(-6),
      deterministicReasons: deterministic.reasons,
      instruction:
        'Decide mini-scope enforcement mode. Modes: full (allowed), restricted (D1/D9 foundational), blocked (requires Pro). Mini allows only D1/D9 foundational non-predictive guidance. Block all pro-only analysis requests including advanced varga (anything beyond D1/D9), dasha, transit, longevity, arudha/ashtakavarga, and advanced systems.',
    },
    fallback: () => deterministicDecision,
  });

  const clampedMode = miniScopeSeverity(result.decision.enforcementMode) >= miniScopeSeverity(deterministicDecision.enforcementMode)
    ? result.decision.enforcementMode
    : deterministicDecision.enforcementMode;
  const forcedAllowed = clampedMode !== 'blocked';
  const reasons = [...new Set([...deterministic.reasons, ...result.decision.reasons])];
  const suggestedAlternative = result.decision.suggestedAlternative ?? deterministicDecision.suggestedAlternative;

  return {
    ...result,
    decision: {
      ...result.decision,
      enforcementMode: clampedMode,
      allowed: forcedAllowed,
      reasons,
      suggestedAlternative,
    },
  };
}

function deriveDeterministicFastAnswerIntent(message: string, route: TopLevelRoute): FastAnswerIntentDecision {
  let intentKind: FastAnswerIntentKind;
  if (isIdentityQuestion(message)) {
    intentKind = 'identity';
  } else if (isCapabilityQuestion(message)) {
    intentKind = 'capability';
  } else if (route === 'smalltalk') {
    intentKind = 'smalltalk';
  } else {
    intentKind = 'general_astro';
  }

  return {
    intentKind,
    responseStyle: 'brief',
    confidence: 0.35,
  };
}

function isObviousSmalltalk(message: string): boolean {
  const q = message.trim().toLowerCase();
  return /^(h+i+|hello|hey|namaste|good\s+(morning|afternoon|evening)|thanks|thank\s+you|ok(?:ay)?|cool|bye|goodbye|gn|good\s+night)\b/.test(q);
}

function isContinuationRequest(message: string): boolean {
  const q = message.trim().toLowerCase();
  return /\b(continue|go on|go ahead|tell me more|more details|elaborate|expand|carry on)\b/.test(q);
}

function isYesNoQuestion(message: string): boolean {
  const q = message.trim().toLowerCase();
  return /^(is|are|am|can|could|should|will|would|did|do|does|has|have)\b/.test(q);
}

function deriveResponseStyleHint(message: string, conversationContext: string[] = []): ResponseStyle {
  if (isContinuationRequest(message)) return 'expand';
  if (isYesNoQuestion(message)) return 'micro';
  if ((conversationContext?.length ?? 0) > 5) return 'brief';
  return 'brief';
}

async function decideFastAnswerIntentDetailed(
  message: string,
  mode: AgentMode,
  route: TopLevelRoute,
  conversationContext: string[] = []
): Promise<{ decision: FastAnswerIntentDecision; model: string; usedFallback: boolean; latencyMs: number }> {
  const deterministic = deriveDeterministicFastAnswerIntent(message, route);

  if (!isDecisionNodeEnabled(env.LLM_DECISION_FAST_ANSWER_ENABLED)) {
    return {
      decision: deterministic,
      model: 'fast-answer-disabled',
      usedFallback: true,
      latencyMs: 0,
    };
  }

  const result = await invokeDecisionNode<FastAnswerIntentDecision>({
    node: 'fast_answer_intent',
    schema: FastAnswerIntentDecisionSchema,
    input: {
      message,
      mode,
      route,
      conversationContext: conversationContext.slice(-6),
      instruction:
        'Classify the fast-answer intent for a non-pipeline astrology chat message. Pick exactly one intent kind. Mini mode guardrail: never route transit/gochar/timing/predictive questions as fast-answer general_astro; those must be blocked via Pro-scope path.',
    },
    fallback: () => deterministic,
  });

  return result;
}

function decideTopLevelRouteDeterministic(message: string): TopLevelRoute {
  // DEPRECATED: Kept for reference only. Direct LLM routing is now used.
  // This function is no longer critical to the routing decision.
  const q = message.trim().toLowerCase();

  if (!q || /^(h+i+|hello|hey|namaste|good\s+(morning|afternoon|evening)|thanks|thank you|bye|goodbye)\b/.test(q)) {
    return 'smalltalk';
  }

  if (/\b(what\s+can\s+you|how\s+can\s+you|features|capabilities)\b/.test(q)) {
    return 'general_astro';
  }

  // Default to pipeline for anything with personal chart cues
  if (/\b(my|mine|me|for me|my chart|my kundli|when will i|when can i|by when|which year|will i|can i|should i|timing|timeline|period)\b/.test(q)) {
    return 'pipeline';
  }

  return 'general_astro'; // Safe default
}

function shouldForcePipelineRoute(message: string): boolean {
  const q = message.trim().toLowerCase();
  if (!q) return false;

  if (isObviousSmalltalk(message) || isIdentityQuestion(message) || isCapabilityQuestion(message)) {
    return false;
  }

  const strongTimingOrTransit = /\b(transit|gochar|dasha|dasa|mahadasha|antardasha|vimshottari|timing|timeline|forecast|prediction|predictive\s+timing|when\s+will|by\s+when|which\s+year|what\s+age|this\s+month|next\s+month|this\s+year|next\s+year|today|tomorrow|next\s+week|past|future)\b/.test(q);
  if (strongTimingOrTransit) {
    return true;
  }

  const hardshipPersonalCue = /\b(i|my|me|for me)\b/.test(q)
    && /\b(cheated|betray(?:ed|al)|heartbreak|breakup|separation|bad\s+time|difficult\s+time|hard\s+time|rough\s+phase|loss|grief|depressed|depression|anxiety)\b/.test(q);
  if (hardshipPersonalCue) {
    return true;
  }

  const personalCue = /\b(my|mine|me|for me|my chart|my kundli|my horoscope|from my chart|based on my chart)\b/.test(q);
  const astrologyCue = /\b(kundli|chart|horoscope|lagna|rashi|nakshatra|d1|d9|d10|planet|jupiter|saturn|venus|mars|mercury|moon|sun|rahu|ketu)\b/.test(q);

  return personalCue && astrologyCue;
}

function hasExplicitTransitCue(message: string): boolean {
  const q = message.trim().toLowerCase();
  return /\b(transit|gochar|current\s+transit|transit\s+details?|today\b|now\b|real[-\s]?time\s+snapshot)\b/.test(q);
}

// DIRECT LLM ROUTER - Connects message → decision → response immediately
async function directLLMRoute(
  message: string,
  mode: AgentMode,
  conversationContext: string[] = []
): Promise<{ route: TopLevelRoute; confidence: number; latencyMs: number; responseStyle: ResponseStyle; continuityIntent: boolean; model: string; usedFallback: boolean }> {
  const startTime = Date.now();
  const deterministicRoute = decideTopLevelRouteDeterministic(message);
  const deterministicStyle = deriveResponseStyleHint(message, conversationContext);

  if (isObviousSmalltalk(message)) {
    return {
      route: 'smalltalk',
      confidence: 0.99,
      latencyMs: Date.now() - startTime,
      responseStyle: 'micro',
      continuityIntent: false,
      model: 'deterministic-smalltalk-router',
      usedFallback: false,
    };
  }

  if (mode === 'mini') {
    const miniScope = evaluateMiniScope(message);
    if (!miniScope.allowed || miniScope.enforcementMode === 'blocked') {
      return {
        route: 'pipeline',
        confidence: 0.99,
        latencyMs: Date.now() - startTime,
        responseStyle: deterministicStyle,
        continuityIntent: isContinuationRequest(message),
        model: 'deterministic-mini-scope-route-guard',
        usedFallback: true,
      };
    }
  }
  
  const result = await invokeDecisionNode<IntentRouteDecision>({
    node: 'route_top_level',
    schema: IntentRouteDecisionSchema,
    input: {
      message,
      mode,
      conversationContext: conversationContext.slice(-8),
      allowedRoutes: ['pipeline', 'smalltalk', 'general_astro'],
      instruction: `You are Cozmic AI's root routing node. Analyze the user message and decide ONE route:

**smalltalk** (instant response): Greetings (hi, hello), thanks/acks (thanks, okay, cool), farewells (bye, goodbye), casual chat
    **general_astro** (brief astro answer): Concept questions (what is nakshatra, planets), capabilities (what can you do), features, general astrology, "who are you / who built you"
**pipeline** (full analysis): Personal requests with "my" (my chart, my career, my marriage), predictions (when will I), timing requests

    Mode policy:
    - mini: allow only smalltalk and basic astrology explanations limited to D1/D9 foundations.
    - mini: DO NOT route transit/gochar/timing/prediction/forecast questions to general_astro or smalltalk; route those to pipeline so mini-scope guard can block.
    - pro: allow all kinds of astrology answers and advanced topics.

    Brand rule:
    - If user asks identity (who are you / who built you), route to general_astro so response can clearly state: "I am Cozmic AI."

Message: "${message}"

Return strict JSON with: topRoute, responseStyle (micro|brief|normal|expand), continuityIntent (boolean), confidence (0-1).`,
    },
    fallback: () => ({
      topRoute: deterministicRoute,
  responseStyle: deterministicStyle,
  continuityIntent: isContinuationRequest(message),
      confidence: deterministicRoute === 'pipeline' ? 0.85 : 0.7,
      reasoningBrief: 'llm fallback deterministic',
      requiresPersonalChart: deterministicRoute === 'pipeline',
    }),
  });

  const forcePipeline = shouldForcePipelineRoute(message);
  const routeWasOverridden = forcePipeline && result.decision.topRoute !== 'pipeline';
  const resolvedRoute: TopLevelRoute = forcePipeline ? 'pipeline' : result.decision.topRoute;

  return {
    route: resolvedRoute,
    confidence: routeWasOverridden ? Math.max(result.decision.confidence, 0.9) : result.decision.confidence,
    latencyMs: Date.now() - startTime,
    responseStyle: result.decision.responseStyle,
    continuityIntent: result.decision.continuityIntent,
    model: routeWasOverridden ? `${result.model}|pipeline-guardrail` : result.model,
    usedFallback: result.usedFallback || routeWasOverridden,
  };
}

// IMMEDIATE RESPONSE GENERATORS - Called after LLM route decision


async function generatePipelineResponse(message: string, mode: AgentMode): Promise<Pick<AgentAnswer, 'answer' | 'model' | 'mode'>> {
  // Route to full pipeline - handled by langgraph
  return {
    answer: 'Loading your chart analysis...',
    model: 'cozmic-pipeline-router',
    mode,
  };
}

async function decideTopLevelRoute(message: string, mode: AgentMode = 'mini', conversationContext: string[] = []): Promise<IntentRouteDecision> {
  // Direct LLM decision - no pattern detection
  const result = await directLLMRoute(message, mode, conversationContext);
  return {
    topRoute: result.route,
    responseStyle: result.responseStyle,
    continuityIntent: result.continuityIntent,
    confidence: result.confidence,
    reasoningBrief: `direct_llm (${result.latencyMs}ms)`,
    requiresPersonalChart: result.route === 'pipeline',
  };
}

async function decideTopLevelRouteDetailed(
  message: string,
  mode: AgentMode = 'mini',
  conversationContext: string[] = []
): Promise<{ decision: IntentRouteDecision; model: string; usedFallback: boolean; latencyMs: number }> {
  const result = await directLLMRoute(message, mode, conversationContext);
  return {
    decision: {
      topRoute: result.route,
      responseStyle: result.responseStyle,
      continuityIntent: result.continuityIntent,
      confidence: result.confidence,
      reasoningBrief: 'direct_llm_router',
      requiresPersonalChart: result.route === 'pipeline',
    },
    model: result.model,
    usedFallback: result.usedFallback,
    latencyMs: result.latencyMs,
  };
}

export async function shouldBypassChartPipeline(message: string, mode: AgentMode = 'mini', conversationContext: string[] = []): Promise<boolean> {
  if (mode === 'mini') {
    const miniScope = evaluateMiniScope(message);
    if (!miniScope.allowed || miniScope.enforcementMode === 'blocked') {
      return false;
    }
  }

  if (shouldForcePipelineRoute(message)) {
    return false;
  }

  const route = await decideTopLevelRoute(message, mode, conversationContext);
  return route.topRoute !== 'pipeline';
}

function isIdentityQuestion(message: string): boolean {
  const q = message.toLowerCase().trim();
  return /\b(who\s+(are\s+)?you|who\s+built\s+you|who\s+created\s+you|what\s+are\s+you|who\s+made\s+you|tell\s+me\s+about\s+yourself|your\s+name|introduce\s+yourself)\b/.test(q);
}

function isCapabilityQuestion(message: string): boolean {
  const q = message.toLowerCase().trim();
  return /\b(what\s+can\s+you\s+do|how\s+can\s+you\s+help|your\s+capabilities|features|what\s+do\s+you\s+do|help\s+me\s+with)\b/.test(q);
}

// Response generators for immediate answers after LLM route decision
async function answerIdentityQuestion(message: string, mode: AgentMode): Promise<string> {
  const identityAnswers = {
    mini: 'I am Cozmic AI. In Mini mode, I provide D1/D9-based astrology guidance.',
    pro: 'I am Cozmic AI. In Pro mode, I provide full advanced astrology analysis.',
  };
  return identityAnswers[mode];
}

async function answerCapabilityQuestion(message: string, mode: AgentMode): Promise<string> {
  const capabilityAnswers = {
    mini: 'Mini: D1/D9 foundational insights and basic guidance. Pro-only analysis includes advanced timing and deep chart systems.',
    pro: 'Pro: full chart layers, timing analysis, and advanced astrological systems.',
  };
  return capabilityAnswers[mode];
}

async function generateSmallTalkResponse(
  message: string,
  mode: AgentMode,
  conversationContext: string[] = [],
  responseStyle: ResponseStyle = 'brief'
): Promise<string> {
  const styleInstruction = responseStyle === 'micro'
    ? 'Reply in one short line.'
    : responseStyle === 'expand'
      ? 'Reply in 2-4 short lines and continue only from prior context if relevant.'
      : 'Reply in 1-2 short lines.';
  const maxTokens = responseStyle === 'micro' ? 50 : responseStyle === 'expand' ? 140 : 90;

  try {
    const response = await invokeDeepSeekBedrock({
      systemPrompt: [
        'You are Cozmic AI, a friendly Vedic astrology assistant.',
        styleInstruction,
        'Answer only what the user asked. Do not add extra sections.',
        'If asked identity (who are you / who built you), explicitly say: "I am Cozmic AI."',
        mode === 'mini'
          ? 'Mini mode scope: mention D1, D9, and basic astrology guidance only.'
          : 'Pro mode scope: you may mention comprehensive astrology capabilities.',
      ].join(' '),
      userPrompt: [
        `Message: ${message}`,
        conversationContext.length > 0 ? `Recent context: ${conversationContext.slice(-4).join(' | ')}` : 'Recent context: none',
      ].join('\n'),
      maxTokens,
    });

    const text = response.text.trim();
    if (text) return text;
  } catch {
    // fall through to deterministic fallback
  }

  return 'I am Cozmic AI. Ask me anything about your chart, astrology concepts, or life guidance.';
}

async function generateGeneralAstroResponse(
  message: string,
  mode: AgentMode,
  conversationContext: string[] = [],
  responseStyle: ResponseStyle = 'brief'
): Promise<string> {
  if (mode === 'mini') {
    const miniScope = evaluateMiniScope(message);
    if (!miniScope.allowed || miniScope.enforcementMode === 'blocked') {
      return buildMiniUpgradeResponse(message, miniScope.reasons);
    }
  }

  const styleInstruction = responseStyle === 'micro'
    ? 'Return a direct answer in 1-2 lines max.'
    : responseStyle === 'expand'
      ? 'Return focused detail in 4-7 lines and continue from prior context only when user asked for more.'
      : 'Return concise answer in 3-6 lines.';
  const maxTokens = responseStyle === 'micro' ? 90 : responseStyle === 'expand' ? 260 : 150;

  // LLM for complex concept questions
  try {
    const response = await invokeDeepSeekBedrock({
      systemPrompt: `You are Cozmic AI, a Vedic astrology assistant. ${styleInstruction}
Identity rule: If the user asks who you are or who built you, say clearly: "I am Cozmic AI."
${mode === 'mini'
  ? 'Mini mode policy: answer only smalltalk or basic D1/D9 foundational astrology. Never provide transit/gochar/timing/predictive analysis. If user asks those, reply that it is unavailable in Cozmic Mini and suggest Cozmic Pro.'
  : 'Pro mode policy: provide all kinds of astrology answers, including advanced divisional charts, dasha, transit, yogas, and timing.'}
Answer only what user asked. Avoid extra sections unless explicitly requested.
If chart-specific analysis is requested but Kundli context is unavailable, do NOT ask for date/time/place of birth. Ask the user to open or generate a Kundli in the app.` ,
      userPrompt: [
        `User message: ${message}`,
        conversationContext.length > 0 ? `Recent context: ${conversationContext.slice(-6).join(' | ')}` : 'Recent context: none',
      ].join('\n'),
      maxTokens,
    });
    return response.text.trim();
  } catch {
    return 'I can explain astrology concepts. What would you like to know about planets, houses, signs, nakshatras, or timing?';
  }
}

async function answerSimpleWithoutChart(
  message: string,
  mode: AgentMode,
  route: TopLevelRoute,
  conversationContext: string[] = [],
  responseStyleHint: ResponseStyle = 'brief'
): Promise<Pick<AgentAnswer, 'answer' | 'model' | 'mode'>> {
  try {
    if (mode === 'mini') {
      const miniScope = evaluateMiniScope(message);
      if (!miniScope.allowed || miniScope.enforcementMode === 'blocked') {
        return {
          answer: buildMiniUpgradeResponse(message, miniScope.reasons),
          model: 'cozmic-mini-guard',
          mode,
        };
      }
    }

    const shouldShortcut = route === 'smalltalk' && isObviousSmalltalk(message);
    const decisionResult = shouldShortcut
      ? {
          decision: deriveDeterministicFastAnswerIntent(message, route),
          model: 'fast-answer-deterministic-shortcut',
          usedFallback: false,
          latencyMs: 0,
        }
      : await decideFastAnswerIntentDetailed(message, mode, route, conversationContext);
    const decision = decisionResult.decision;
    const responseStyle: ResponseStyle = responseStyleHint === 'expand'
      ? 'expand'
      : decision.responseStyle === 'normal'
        ? 'normal'
        : responseStyleHint;

    let answer = '';
    let handlerModel = 'cozmic-fallback-response';

    switch (decision.intentKind) {
      case 'identity':
        answer = await answerIdentityQuestion(message, mode);
        handlerModel = 'cozmic-identity-response';
        break;
      case 'capability':
        answer = await answerCapabilityQuestion(message, mode);
        handlerModel = 'cozmic-capability-response';
        break;
      case 'smalltalk':
        answer = await generateSmallTalkResponse(message, mode, conversationContext, responseStyle);
        handlerModel = 'cozmic-smalltalk-response';
        break;
      case 'general_astro':
      default:
        answer = await generateGeneralAstroResponse(message, mode, conversationContext, responseStyle);
        handlerModel = 'cozmic-general-astro-response';
        break;
    }

    answer = enforceGroundingAnswerContract(answer, [], message);

    return {
      answer: answer.trim() || 'Ask me an astrology question, and I will help!',
      model: `${handlerModel}|intent=${decision.intentKind}|router=${decisionResult.model}`,
      mode,
    };
  } catch {
    return {
      answer: 'I can help with astrology. What would you like to know?',
      model: 'cozmic-error-fallback',
      mode,
    };
  }
}

function determineQuestionFamily(question: string, intent: QuestionIntent): QuestionFamily {
  const q = question.toLowerCase();

  if (intent.flags.includes('remedies') || /\b(remedy|remedies|upay|upaya|mantra|gemstone|puja|pooja|fasting)\b/.test(q)) {
    return 'remedies';
  }
  if (intent.flags.includes('relocation') || /\b(relocation|migrate|migration|settle abroad|foreign settlement|move abroad)\b/.test(q)) {
    return 'relocation';
  }
  if (intent.flags.includes('past_life') || /\b(past life|past-life|karma|karmic|reincarnation|soul purpose)\b/.test(q)) {
    return 'past_life';
  }
  if (intent.flags.includes('pregnancy_fertility') || /\b(pregnancy|fertility|conceive|conception|childbirth|delivery|baby)\b/.test(q)) {
    return 'pregnancy_fertility';
  }
  if (intent.flags.includes('legal') || /\b(legal|court|litigation|lawsuit|dispute|case)\b/.test(q)) {
    return 'legal';
  }

  if (intent.flags.includes('longevity') || /\b(longevity|lifespan|life span|how long will i live|length of life|ayush|ayu|mrityu|death|end of life)\b/.test(q)) {
    return 'longevity';
  }

  if (intent.flags.includes('relationship') || /\b(cheated|betray(?:ed|al)|heartbreak|breakup|separation|infidelity)\b/.test(q)) {
    return 'relationship';
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
  if (family === 'remedies') {
    chartLayers.add('D16');
    chartLayers.add('D27');
  }
  if (family === 'relocation') {
    chartLayers.add('D4');
    chartLayers.add('D12');
  }
  if (family === 'pregnancy_fertility') {
    chartLayers.add('D5');
    chartLayers.add('D7');
  }
  if (family === 'legal') {
    chartLayers.add('D3');
  }
  if (family === 'finance') {
    chartLayers.add('D2');
    chartLayers.add('D11');
  }
  if (family === 'health') {
    chartLayers.add('D6');
    chartLayers.add('D8');
    chartLayers.add('D30');
  }
  if (family === 'education') {
    chartLayers.add('D4');
    chartLayers.add('D24');
  }
  if (family === 'children') chartLayers.add('D7');
  if (family === 'property') chartLayers.add('D4');
  if (family === 'travel') chartLayers.add('D12');
  if (family === 'spirituality') chartLayers.add('D20');
  if (family === 'family') chartLayers.add('D4');
  if (family === 'longevity') {
    chartLayers.add('D8');
    chartLayers.add('D30');
  }
  if (intent.flags.includes('d9')) chartLayers.add('D9');

  const aspectTimingFamilies = new Set<QuestionFamily>([
    'career',
    'remedies',
    'relocation',
    'pregnancy_fertility',
    'legal',
    'marriage',
    'relationship',
    'finance',
    'health',
    'education',
    'children',
    'property',
    'travel',
    'spirituality',
    'family',
    'longevity',
  ]);

  const includeTiming = intent.flags.includes('timing')
    || intent.flags.includes('dasha')
    || intent.flags.includes('transit')
    || family === 'timing'
    || family === 'longevity'
    || (mode === 'pro' && aspectTimingFamilies.has(family));
  const includeTransit = mode === 'pro' && (intent.flags.includes('transit') || includeTiming);
  const dashaDrivenFamilies = new Set<QuestionFamily>([
    'career',
    'remedies',
    'relocation',
    'pregnancy_fertility',
    'legal',
    'marriage',
    'relationship',
    'finance',
    'health',
    'education',
    'children',
    'property',
    'travel',
    'spirituality',
    'family',
    'longevity',
    'timing',
  ]);
  const includeDasha = mode === 'pro' && (intent.flags.includes('dasha') || includeTiming || dashaDrivenFamilies.has(family));
  const includeCareer = family === 'career' && mode === 'pro';
  const includeRelationship = family === 'marriage' || family === 'relationship';

  const includeMicroSignals: MicroSignal[] = ['nakshatra', 'nakshatra_lord', 'sign_lord', 'drishti', 'degree'];

  const seriesNodes = ['classify_intent', 'load_grounding', 'plan_and_tools', 'run_specialized_tools', 'run_general_tools', 'build_prompt', 'answer_with_remedy_specialist', 'answer_with_deepseek', 'condense_answer'];
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

  if (executionPlan.family === 'remedies' && isCoverageWeak(getFindingStatus(findings, 'Remedies analyzer'), strict)) {
    gaps.add('remedies');
  }

  if (executionPlan.family === 'relocation' && isCoverageWeak(getFindingStatus(findings, 'Relocation analyzer'), strict)) {
    gaps.add('relocation');
  }

  if (executionPlan.family === 'past_life' && isCoverageWeak(getFindingStatus(findings, 'Past-life analyzer'), strict)) {
    gaps.add('past_life');
  }

  if (executionPlan.family === 'pregnancy_fertility' && isCoverageWeak(getFindingStatus(findings, 'Pregnancy/Fertility analyzer'), strict)) {
    gaps.add('pregnancy_fertility');
  }

  if (executionPlan.family === 'legal' && isCoverageWeak(getFindingStatus(findings, 'Legal analyzer'), strict)) {
    gaps.add('legal');
  }

  if (executionPlan.family === 'finance' && isCoverageWeak(getFindingStatus(findings, 'Finance analyzer'), strict)) {
    gaps.add('finance');
  }

  if (executionPlan.family === 'health' && isCoverageWeak(getFindingStatus(findings, 'Health analyzer'), strict)) {
    gaps.add('health');
  }

  if (executionPlan.family === 'education' && isCoverageWeak(getFindingStatus(findings, 'Education analyzer'), strict)) {
    gaps.add('education');
  }

  if (executionPlan.family === 'children' && isCoverageWeak(getFindingStatus(findings, 'Children analyzer'), strict)) {
    gaps.add('children');
  }

  if (executionPlan.family === 'property' && isCoverageWeak(getFindingStatus(findings, 'Property analyzer'), strict)) {
    gaps.add('property');
  }

  if (executionPlan.family === 'travel' && isCoverageWeak(getFindingStatus(findings, 'Travel analyzer'), strict)) {
    gaps.add('travel');
  }

  if (executionPlan.family === 'spirituality' && isCoverageWeak(getFindingStatus(findings, 'Spirituality analyzer'), strict)) {
    gaps.add('spirituality');
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

function selectRelevantPaths(
  question: string,
  flags: string[] = [],
  mode: AgentMode = 'pro',
  scopeSelection: ScopeSelectionDecision | null = null
): string[] {
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

  if (scopeSelection?.requiredScopes?.length) {
    add(...scopeSelection.requiredScopes);
  } else {
    for (const scope of pickQuestionScope(q)) {
      add(scope);
    }
  }

  if (flags.includes('d9')) add('chart.varga.D9');
  if (flags.includes('dasha')) add('chart.dasha');
  if (flags.includes('transit')) add('chart.transit', 'chart.transits', 'chart.gochar');
  if (flags.includes('career')) add('chart.varga.D10');
  if (flags.includes('relationship')) add('chart.varga.D9');
  if (flags.includes('longevity')) add('chart.varga.D8', 'chart.varga.D30', 'chart.dasha', 'chart.bhava');

  if (scopeSelection) {
    if (scopeSelection.needsDasha) add('chart.dasha');
    if (scopeSelection.needsTransit) add('chart.transit', 'chart.transits', 'chart.gochar');
    if (scopeSelection.needsD9) add('chart.varga.D9');
    if (scopeSelection.needsD10) add('chart.varga.D10');
    if (scopeSelection.needsLongevity) add('chart.varga.D8', 'chart.varga.D30');
  }

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
  if (/\b(remedy|remedies|upay|upaya|mantra|gemstone|puja|pooja|fasting)\b/.test(q)) {
    add('chart.varga.D16', 'chart.varga.D27', 'chart.varga.D1', 'chart.graha', 'chart.bhava', 'chart.dasha');
  }
  if (/\b(relocation|migrate|migration|settle abroad|foreign settlement|move abroad)\b/.test(q)) {
    add('chart.varga.D4', 'chart.varga.D12', 'chart.varga.D1', 'chart.dasha', 'chart.transit', 'chart.transits', 'chart.gochar');
  }
  if (/\b(past life|past-life|karma|karmic|reincarnation|soul purpose)\b/.test(q)) {
    add('chart.varga.D1', 'chart.graha', 'chart.bhava');
  }
  if (/\b(pregnancy|fertility|conceive|conception|childbirth|delivery|baby)\b/.test(q)) {
    add('chart.varga.D5', 'chart.varga.D7', 'chart.varga.D1', 'chart.dasha', 'chart.transit', 'chart.transits', 'chart.gochar');
  }
  if (/\b(legal|court|litigation|lawsuit|dispute|case)\b/.test(q)) {
    add('chart.varga.D3', 'chart.varga.D1', 'chart.bhava', 'chart.dasha', 'chart.transit', 'chart.transits', 'chart.gochar');
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

function selectSections(
  rawPayload: unknown,
  question: string,
  flags: string[] = [],
  mode: AgentMode = 'pro',
  scopeSelection: ScopeSelectionDecision | null = null
): SelectedSection[] {
  const selectedPaths = selectRelevantPaths(question, flags, mode, scopeSelection);
  return selectedPaths
    .map((path) => ({ path, value: getByPath(rawPayload, path) }))
    .filter((section) => section.value !== undefined);
}

function normalizeRawPayload(rawPayload: unknown): unknown {
  if (!rawPayload || typeof rawPayload !== 'object') {
    return rawPayload;
  }
  return buildChartSnapshot(rawPayload);
}

async function loadCanonicalGrounding(state: AgentStateType): Promise<AgentUpdateType> {
  const profileId = state.profileId;
  if (!profileId) {
    throw new Error('Profile identity is required before grounding can be loaded.');
  }

  const ragProfiles = getRagProfilesRepository();

  const profileCacheKey = buildGroundingProfileCacheKey(state.ownerId, profileId);
  const cachedProfile = await cacheGetJson<GroundingProfileCacheEntry>(profileCacheKey);
  const profileDoc = cachedProfile?.doc ?? await ragProfiles.getByOwnerAndProfileId(state.ownerId, profileId);

  if (!cachedProfile && profileDoc) {
    await cacheSetJson(
      profileCacheKey,
      { cachedAt: Date.now(), doc: profileDoc },
      Math.max(1, env.AGENT_CACHE_TTL_SECONDS)
    );
  }

  if (!profileDoc) {
    if (state.kundliInput) {
      const rawPayload = {};
      const atlas: SelectedSection[] = [];
      const toolAvailabilityPreflight = buildToolAvailabilityPreflight(rawPayload, state.mode);

      return {
        grounding: {
          ownerId: state.ownerId,
          profileId,
          sourceDocId: 'inline-kundli',
          chartVersion: 'inline',
          kundliSignature: `inline_${stableHash(JSON.stringify(state.kundliInput)).slice(0, 10)}`,
          kundli: state.kundliInput,
          requestKey: 'inline',
          payloadHash: 'inline',
          referenceTimestamp: state.referenceTimestamp ?? Date.now(),
          referenceTimeSource: state.referenceTimeSource ?? 'server',
          rawPayload,
          selectedPaths: [],
          selectedSections: [],
        },
        toolAvailabilityPreflight,
      };
    }

    throw new Error(`No canonical profile snapshot found for ${profileId}. Regenerate the Kundli first.`);
  }

  const latestSourceDocId = profileDoc.latestSourceDocId;
  if (!latestSourceDocId) {
    if (state.kundliInput) {
      const rawPayload = {};
      const toolAvailabilityPreflight = buildToolAvailabilityPreflight(rawPayload, state.mode);
      return {
        grounding: {
          ownerId: state.ownerId,
          profileId,
          sourceDocId: 'inline-kundli',
          chartVersion: 'inline',
          kundliSignature: `inline_${stableHash(JSON.stringify(state.kundliInput)).slice(0, 10)}`,
          kundli: state.kundliInput,
          requestKey: 'inline',
          payloadHash: 'inline',
          referenceTimestamp: state.referenceTimestamp ?? Date.now(),
          referenceTimeSource: state.referenceTimeSource ?? 'server',
          rawPayload,
          selectedPaths: [],
          selectedSections: [],
        },
        toolAvailabilityPreflight,
      };
    }
    throw new Error(`No canonical raw payload found for profile ${profileId}. Regenerate the Kundali first.`);
  }

  const sourceCacheKey = buildGroundingSourceCacheKey(latestSourceDocId);
  const cachedSource = await cacheGetJson<GroundingSourceCacheEntry>(sourceCacheKey);

  let sourceDoc: RagApiSourceRecord | null = cachedSource?.record ?? null;

  if (!cachedSource && !sourceDoc) {
    await cacheSetJson(
      sourceCacheKey,
      { cachedAt: Date.now(), record: sourceDoc },
      Math.max(1, env.AGENT_CACHE_TTL_SECONDS)
    );
  }

  if (!sourceDoc) {
    if (state.kundliInput) {
      const rawPayload = {};
      const toolAvailabilityPreflight = buildToolAvailabilityPreflight(rawPayload, state.mode);
      return {
        grounding: {
          ownerId: state.ownerId,
          profileId,
          sourceDocId: 'inline-kundli',
          chartVersion: 'inline',
          kundliSignature: `inline_${stableHash(JSON.stringify(state.kundliInput)).slice(0, 10)}`,
          kundli: state.kundliInput,
          requestKey: 'inline',
          payloadHash: 'inline',
          referenceTimestamp: state.referenceTimestamp ?? Date.now(),
          referenceTimeSource: state.referenceTimeSource ?? 'server',
          rawPayload,
          selectedPaths: [],
          selectedSections: [],
        },
        toolAvailabilityPreflight,
      };
    }

    throw new Error(`No canonical raw payload found for profile ${profileId}. Regenerate the Kundali first.`);
  }

  const rawPayload = normalizeRawPayload(sourceDoc.data.chartSnapshot ?? sourceDoc.data.rawPayload);
  const atlas = summarizeChartAtlas(rawPayload);
  const toolAvailabilityPreflight = buildToolAvailabilityPreflight(rawPayload, state.mode);
  const selectedSections = selectSections(rawPayload, state.question, state.intent?.flags ?? [], state.mode, state.scopeSelection ?? null);
  const fallbackSections = selectedSections.length > 0 ? selectedSections : atlas.slice(0, 12).map((item) => ({ path: item.path, value: getByPath(rawPayload, item.path) }));
  const kundli = state.kundliInput ?? profileDoc.kundliInput;

  return {
    grounding: {
      ownerId: state.ownerId,
      profileId,
      sourceDocId: profileDoc.latestSourceDocId ?? 'unknown',
      chartVersion: profileDoc.chartVersion,
      kundliSignature: profileDoc.kundliSignature,
      kundli,
      requestKey: sourceDoc.data.requestKey,
      payloadHash: sourceDoc.data.payloadHash,
      referenceTimestamp: state.referenceTimestamp ?? Date.now(),
      referenceTimeSource: state.referenceTimeSource ?? 'server',
      rawPayload,
      selectedPaths: fallbackSections.map((section) => section.path),
      selectedSections: fallbackSections,
    },
    toolAvailabilityPreflight,
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

interface DashaFindingCacheEntry extends Record<string, unknown> {
  cachedAt: number;
  referenceBucket: number;
  finding: ToolFinding;
}

const DASHA_LEVEL_LABELS: Record<number, string> = {
  1: 'mahadasha',
  2: 'antardasha',
  3: 'pratyantardasha',
  4: 'sookshma',
  5: 'prana',
};

const DASHA_TYPE_ALIASES: Record<string, string> = {
  maha: 'mahadasha',
  mahadasha: 'mahadasha',
  antardasha: 'antardasha',
  antar: 'antardasha',
  pratyantar: 'pratyantardasha',
  pratyantardasha: 'pratyantardasha',
  sookshma: 'sookshma',
  sukshma: 'sookshma',
  prana: 'prana',
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

function normalizeDashaType(rawType: unknown, level: number): string {
  const normalizedRaw = String(rawType ?? '')
    .trim()
    .toLowerCase()
    .replace(/\s+/g, '')
    .replace(/[-_]/g, '');

  if (normalizedRaw) {
    const alias = DASHA_TYPE_ALIASES[normalizedRaw];
    if (alias) {
      return alias;
    }
  }

  return DASHA_LEVEL_LABELS[level] ?? 'dasha';
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
      type: normalizeDashaType(node.type, level),
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
  const q = question.toLowerCase();
  const governmentCue = /\b(upsc|ias|ips|ifs|ssc|psc|civil service|civil services|government job|govt job|state job|psu|bank po|sarkari)\b/.test(q);
  const privateCue = /\b(startup|corporate|mnc|private sector|product company|private job|entrepreneurship)\b/.test(q);

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

  const scoringSections = [d10?.value, d1?.value].filter((item): item is Record<string, unknown> => isPlainObject(item));
  let governmentRaw = 0;
  let privateRaw = 0;

  for (const section of scoringSections) {
    const graha = getByPath(section, 'graha') as Record<string, Record<string, unknown>> | undefined;
    if (!graha || !isPlainObject(graha)) continue;

    for (const [code, placement] of Object.entries(graha)) {
      if (!isPlainObject(placement)) continue;
      const house = Number(placement.house_number);

      if (['Su', 'Sa', 'Ju'].includes(code)) {
        governmentRaw += 2;
      }
      if (['Me', 'Ma', 'Ve', 'Ra'].includes(code)) {
        privateRaw += 2;
      }

      if ([6, 10].includes(house)) {
        governmentRaw += 1;
      }
      if ([3, 7, 10, 11].includes(house)) {
        privateRaw += 1;
      }
    }
  }

  const normalizeScore = (value: number): number => {
    if (value <= 0) return 50;
    const max = Math.max(12, governmentRaw + privateRaw);
    return Math.max(35, Math.min(95, Math.round((value / max) * 100)));
  };

  const governmentScore = normalizeScore(governmentRaw);
  const privateScore = normalizeScore(privateRaw);
  const recommendation = governmentScore > privateScore + 8
    ? 'government-focused track stronger'
    : privateScore > governmentScore + 8
      ? 'private/corporate track stronger'
      : 'balanced between government and private tracks';

  if (governmentCue || privateCue) {
    facts.push(`India career split heuristic: governmentScore=${governmentScore}/100, privateScore=${privateScore}/100 (${recommendation}).`);
    if (governmentCue) {
      facts.push('Question intent indicates Indian government/public sector path (UPSC/IAS/SSC/PSU-like trajectory).');
    }
    if (privateCue) {
      facts.push('Question intent indicates Indian private/corporate path (startup/MNC/product/company trajectory).');
    }
    facts.push('Govt/private suitability is heuristic guidance based on D1/D10 placements plus timing overlays, not a deterministic guarantee.');
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

// Test-only helper to validate deterministic career scoring/rationale behavior in regression scripts.
export function __testOnlyMakeCareerToolFinding(rawPayload: unknown, question: string, analysisTimestamp: number): ToolFinding {
  return makeCareerToolFinding(rawPayload, question, analysisTimestamp);
}

function makeRemediesToolFinding(rawPayload: unknown, analysisTimestamp: number, question = ''): ToolFinding {
  const base = makeAspectToolFinding(rawPayload, analysisTimestamp, {
    name: 'Remedies analyzer',
    vargaKeys: ['D16', 'D27', 'D1'],
    focusPlanets: ['Sa', 'Ma', 'Ra', 'Ke', 'Su', 'Mo'],
    focusHouses: ['6', '8', '12'],
    intentLabel: 'Remedies',
  });

  if (base.status === 'unavailable') {
    return base;
  }

  const d1 = getFirstPathValue(rawPayload, ['chart.varga.D1', 'varga.D1']);
  const afflicted: string[] = [];
  const afflictedCodes: string[] = [];
  const graha = d1 ? (getByPath(d1.value, 'graha') as Record<string, Record<string, unknown>> | undefined) : undefined;
  if (graha && isPlainObject(graha)) {
    for (const [code, placement] of Object.entries(graha)) {
      if (!isPlainObject(placement)) continue;
      const house = Number(placement.house_number);
      const retro = Boolean(placement.retrograde);
      if (([6, 8, 12].includes(house) && ['Sa', 'Ma', 'Ra', 'Ke'].includes(code)) || retro) {
        afflicted.push(`${PLANET_LABELS[code] ?? code} (${house || 'n/a'}H${retro ? ', retrograde' : ''})`);
        afflictedCodes.push(code);
      }
    }
  }

  const remedyPlaybook: Record<string, string> = {
    Sa: 'Saturn remedy track: Saturday discipline, service to elderly/workers, and consistent routine commitments.',
    Ma: 'Mars remedy track: Tuesday Hanuman practice, physical training, and anger/impulse regulation habits.',
    Ra: 'Rahu remedy track: Durga/Kaal Bhairav prayers, reduce intoxicants/overstimulation, strengthen boundaries.',
    Ke: 'Ketu remedy track: Ganesha prayers, meditation/journaling, and focused detachment from chaotic influences.',
    Su: 'Sun remedy track: sunrise arghya, Aditya Hridayam/Gayatri discipline, leadership through responsibility.',
    Mo: 'Moon remedy track: Monday moon practices, sleep/emotional hygiene, and steady family nourishment routines.',
  };

  const facts = [...base.facts];
  facts.push('Remedy prioritization heuristic uses D1 + D16 + D27 stress indicators with dasha timing context.');
  if (afflicted.length > 0) {
    facts.push(`Potentially afflicted planets for remedies: ${afflicted.slice(0, 4).join(', ')}.`);

    const prioritizedCodes = [...new Set(afflictedCodes)].slice(0, 3);
    if (prioritizedCodes.length > 0) {
      facts.push('Recommended remedy tracks (prioritized):');
      for (const code of prioritizedCodes) {
        const label = PLANET_LABELS[code] ?? code;
        const remedyLine = remedyPlaybook[code] ?? `${label} remedy track: mantra, discipline, and service-based corrective actions.`;
        facts.push(`${label}: ${remedyLine}`);
      }
      facts.push('Remedy cadence: apply daily micro-remedies + weekly anchor practice for at least 6-8 weeks before reassessment.');
    }
  } else {
    facts.push('No high-severity affliction cluster detected in sampled placements; use general stabilizing remedies (discipline, prayer/mantra consistency, sleep and routine hygiene).');
  }

  if (/\b(cheated|betray(?:ed|al)|heartbreak|breakup|separation|bad\s+time|difficult\s+time|hard\s+time)\b/i.test(question)) {
    facts.push('Hardship support remedy layer: combine spiritual remedies with practical boundaries, trusted support network, and routine stabilization.');
  }

  facts.push('Remedies are supportive and probabilistic, not deterministic guarantees; apply with practical judgment.');

  return {
    ...base,
    facts,
  };
}

function makeRelocationToolFinding(rawPayload: unknown, analysisTimestamp: number): ToolFinding {
  const base = makeAspectToolFinding(rawPayload, analysisTimestamp, {
    name: 'Relocation analyzer',
    vargaKeys: ['D4', 'D12', 'D1'],
    focusPlanets: ['Ra', 'Ke', 'Mo', 'Sa', 'Ju'],
    focusHouses: ['4', '9', '12'],
    intentLabel: 'Relocation',
  });

  if (base.status === 'unavailable') {
    return base;
  }

  return {
    ...base,
    facts: [
      ...base.facts,
      'Relocation suitability is interpreted from D4/D12 foundations with timing overlays (dasha/transit when available).',
    ],
  };
}

function makePastLifeToolFinding(rawPayload: unknown, analysisTimestamp: number): ToolFinding {
  const d1 = getFirstPathValue(rawPayload, ['chart.varga.D1', 'varga.D1']);
  if (!d1) {
    return {
      name: 'Past-life analyzer',
      status: 'unavailable',
      facts: ['Past-life analysis requires D1 graha/bhava details, but they are unavailable.'],
      evidencePaths: [],
      missing: ['chart.varga.D1', 'chart.graha', 'chart.bhava'],
    };
  }

  const facts: string[] = [`Past-life window evaluated at ${new Date(analysisTimestamp).toISOString()}.`];
  const graha = getByPath(d1.value, 'graha') as Record<string, unknown> | undefined;
  const bhava = getByPath(d1.value, 'bhava') as Record<string, unknown> | undefined;

  if (graha?.Ra) facts.push(`Rahu karmic vector (D1): ${formatPlacement(graha.Ra)}.`);
  if (graha?.Ke) facts.push(`Ketu carry-over vector (D1): ${formatPlacement(graha.Ke)}.`);
  if (bhava?.['12']) facts.push(`12th-house release pattern (D1): ${formatPlacement(bhava['12'])}.`);
  if (bhava?.['8']) facts.push(`8th-house transformation pattern (D1): ${formatPlacement(bhava['8'])}.`);

  return {
    name: 'Past-life analyzer',
    status: facts.length > 1 ? 'ok' : 'partial',
    facts,
    evidencePaths: [d1.path],
    snippets: [truncateText(JSON.stringify(d1.value, null, 2), 1800)],
  };
}

function makePregnancyFertilityToolFinding(rawPayload: unknown, analysisTimestamp: number): ToolFinding {
  const base = makeAspectToolFinding(rawPayload, analysisTimestamp, {
    name: 'Pregnancy/Fertility analyzer',
    vargaKeys: ['D5', 'D7', 'D1'],
    focusPlanets: ['Ju', 'Ve', 'Mo'],
    focusHouses: ['5', '7'],
    intentLabel: 'Pregnancy/Fertility',
  });

  if (base.status === 'unavailable') {
    return base;
  }

  return {
    ...base,
    facts: [
      ...base.facts,
      'Pregnancy/fertility guidance is probabilistic and should be read with practical medical guidance where relevant.',
    ],
  };
}

function makeLegalToolFinding(rawPayload: unknown, analysisTimestamp: number): ToolFinding {
  const base = makeAspectToolFinding(rawPayload, analysisTimestamp, {
    name: 'Legal analyzer',
    vargaKeys: ['D3', 'D1'],
    focusPlanets: ['Ma', 'Sa', 'Ra', 'Ju'],
    focusHouses: ['6', '7', '8'],
    intentLabel: 'Legal',
  });

  if (base.status === 'unavailable') {
    return base;
  }

  return {
    ...base,
    facts: [
      ...base.facts,
      'Legal/litigation analysis is indicative timing guidance, not legal advice.',
    ],
  };
}

type AspectAnalyzerConfig = {
  name: string;
  vargaKeys: string[];
  focusPlanets: string[];
  focusHouses: string[];
  intentLabel: string;
};

function makeAspectToolFinding(rawPayload: unknown, analysisTimestamp: number, config: AspectAnalyzerConfig): ToolFinding {
  const vargaSections = config.vargaKeys
    .map((key) => getFirstPathValue(rawPayload, [`chart.varga.${key}`, `varga.${key}`]))
    .filter((section): section is SelectedSection => Boolean(section));

  const dashaSection = getFirstPathValue(rawPayload, ['chart.dasha', 'dasha']);
  const transitSection = getFirstPathValue(rawPayload, ['chart.transit', 'chart.transits', 'chart.gochar', 'transit', 'transits', 'gochar']);

  if (vargaSections.length === 0 && !dashaSection && !transitSection) {
    return {
      name: config.name,
      status: 'unavailable',
      facts: [`No ${config.intentLabel}-relevant varga/timing sections were found in canonical payload.`],
      evidencePaths: [],
      missing: config.vargaKeys.map((key) => `chart.varga.${key}`),
    };
  }

  const facts: string[] = [`${config.intentLabel} time window evaluated at ${new Date(analysisTimestamp).toISOString()}.`];
  const evidencePaths: string[] = [];

  for (const section of vargaSections) {
    evidencePaths.push(section.path);
    const sectionValue = section.value as Record<string, unknown>;
    const lagna = getByPath(sectionValue, 'lagna.Lg') ?? getByPath(sectionValue, 'lagna');
    const graha = getByPath(sectionValue, 'graha') as Record<string, unknown> | undefined;
    const bhava = getByPath(sectionValue, 'bhava') as Record<string, unknown> | undefined;

    if (lagna) {
      facts.push(`${section.path} Lagna: ${formatPlacement(lagna)}.`);
    }

    if (graha) {
      for (const code of config.focusPlanets) {
        const planet = graha[code];
        if (!planet) continue;
        facts.push(`${PLANET_LABELS[code] ?? code} in ${section.path}: ${formatPlacement(planet)}.`);
      }
    }

    if (bhava) {
      for (const house of config.focusHouses) {
        const houseValue = bhava[house];
        if (!houseValue) continue;
        facts.push(`House ${house} in ${section.path}: ${formatPlacement(houseValue)}.`);
      }
    }
  }

  if (dashaSection) {
    evidencePaths.push(dashaSection.path);
    facts.push('Dasha timeline is available for timing overlay in this life-area analysis.');
  }

  if (transitSection) {
    evidencePaths.push(transitSection.path);
    facts.push('Transit/gochar block is available for near-term timing overlay.');
  }

  return {
    name: config.name,
    status: facts.length > 1 ? 'ok' : 'partial',
    facts,
    evidencePaths,
    snippets: [
      ...vargaSections.slice(0, 2).map((section) => truncateText(JSON.stringify(section.value, null, 2), 1800)),
      ...(dashaSection ? [truncateText(JSON.stringify(dashaSection.value, null, 2), 1200)] : []),
    ],
  };
}

function makeFinanceToolFinding(rawPayload: unknown, analysisTimestamp: number): ToolFinding {
  return makeAspectToolFinding(rawPayload, analysisTimestamp, {
    name: 'Finance analyzer',
    vargaKeys: ['D2', 'D11', 'D1'],
    focusPlanets: ['Ju', 'Ve', 'Me', 'Sa', 'Ra'],
    focusHouses: ['2', '11', '8', '5'],
    intentLabel: 'Finance',
  });
}

function makeHealthToolFinding(rawPayload: unknown, analysisTimestamp: number): ToolFinding {
  return makeAspectToolFinding(rawPayload, analysisTimestamp, {
    name: 'Health analyzer',
    vargaKeys: ['D6', 'D8', 'D30', 'D1'],
    focusPlanets: ['Sa', 'Ma', 'Ra', 'Ke', 'Mo'],
    focusHouses: ['1', '6', '8', '12'],
    intentLabel: 'Health',
  });
}

function makeEducationToolFinding(rawPayload: unknown, analysisTimestamp: number): ToolFinding {
  return makeAspectToolFinding(rawPayload, analysisTimestamp, {
    name: 'Education analyzer',
    vargaKeys: ['D24', 'D4', 'D1'],
    focusPlanets: ['Me', 'Ju', 'Mo', 'Su'],
    focusHouses: ['4', '5', '9'],
    intentLabel: 'Education',
  });
}

function makeChildrenToolFinding(rawPayload: unknown, analysisTimestamp: number): ToolFinding {
  return makeAspectToolFinding(rawPayload, analysisTimestamp, {
    name: 'Children analyzer',
    vargaKeys: ['D7', 'D5', 'D1'],
    focusPlanets: ['Ju', 'Ve', 'Mo'],
    focusHouses: ['5', '9'],
    intentLabel: 'Children',
  });
}

function makePropertyToolFinding(rawPayload: unknown, analysisTimestamp: number): ToolFinding {
  return makeAspectToolFinding(rawPayload, analysisTimestamp, {
    name: 'Property analyzer',
    vargaKeys: ['D4', 'D2', 'D11', 'D1'],
    focusPlanets: ['Ma', 'Ve', 'Sa', 'Mo'],
    focusHouses: ['4', '2', '11'],
    intentLabel: 'Property',
  });
}

function makeTravelToolFinding(rawPayload: unknown, analysisTimestamp: number): ToolFinding {
  return makeAspectToolFinding(rawPayload, analysisTimestamp, {
    name: 'Travel analyzer',
    vargaKeys: ['D12', 'D9', 'D1'],
    focusPlanets: ['Ra', 'Ke', 'Mo', 'Sa'],
    focusHouses: ['3', '9', '12'],
    intentLabel: 'Travel',
  });
}

function makeSpiritualityToolFinding(rawPayload: unknown, analysisTimestamp: number): ToolFinding {
  return makeAspectToolFinding(rawPayload, analysisTimestamp, {
    name: 'Spirituality analyzer',
    vargaKeys: ['D20', 'D9', 'D1'],
    focusPlanets: ['Ju', 'Ke', 'Sa', 'Su'],
    focusHouses: ['5', '9', '12'],
    intentLabel: 'Spirituality',
  });
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
  if (typeof dasha.nesting === 'number') facts.push(`Dasha nesting depth available in payload: ${dasha.nesting}.`);
  if (dasha.start && dasha.end) facts.push(`Overall window: ${String(dasha.start)} to ${String(dasha.end)}.`);

  if (typeof dasha.duration === 'number') {
    const years = dasha.duration / (365.2425 * 24 * 3600);
    facts.push(`Overall duration: ${years.toFixed(2)} years.`);
  }

  facts.push(`Reference time: ${referenceDate.toISOString()}.`);
  facts.push('Dasha levels normalized for analysis: mahadasha > antardasha > pratyantardasha > sookshma > prana.');

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

export function shouldFetchDeepDasha(question: string): boolean {
  const q = question.toLowerCase();
  const currentOnly = /\b(current|which|what|show)\b.*\b(maha|mahadasha|dasha|dasa)\b/.test(q) || /\bcurrent\s+mahadasha\b/.test(q);
  const timingFocus = /\b(when|timing|marriage|career|promotion|job|future|forecast|period|improve|better|bad phase|difficult phase|transit|gochar)\b/.test(q);
  return timingFocus && !currentOnly;
}

function isToolFindingLike(value: unknown): value is ToolFinding {
  if (!isPlainObject(value)) return false;
  if (typeof value.name !== 'string') return false;
  if (!(value.status === 'ok' || value.status === 'partial' || value.status === 'unavailable')) return false;
  if (!Array.isArray(value.facts) || !Array.isArray(value.evidencePaths)) return false;
  return true;
}

function cloneToolFinding(finding: ToolFinding): ToolFinding {
  return {
    ...finding,
    facts: [...finding.facts],
    evidencePaths: [...finding.evidencePaths],
    missing: finding.missing ? [...finding.missing] : undefined,
    snippets: finding.snippets ? [...finding.snippets] : undefined,
  };
}

function buildDashaCacheKey(input: { ownerId: string; profileId: string; payloadHash: string; referenceTimestamp: number }): { key: string; referenceBucket: number } {
  const bucketMs = 60 * 60 * 1000;
  const referenceBucket = Math.floor(input.referenceTimestamp / bucketMs);
  const seed = {
    ownerId: input.ownerId,
    profileId: input.profileId,
    payloadHash: input.payloadHash,
    referenceBucket,
    version: 1,
  };
  return {
    key: `dasha_chain:${stableHash(JSON.stringify(seed))}`,
    referenceBucket,
  };
}

async function buildDashaToolFindingCached(input: {
  rawPayload: unknown;
  referenceTimestamp: number;
  ownerId: string;
  profileId: string;
  payloadHash: string;
  question: string;
  kundli?: KundliSnapshotInput | null;
}): Promise<ToolFinding> {
  const useDeepFetch = shouldFetchDeepDasha(input.question);
  const cacheKeyInput = {
    ownerId: input.ownerId,
    profileId: input.profileId,
    payloadHash: `${input.payloadHash}:${useDeepFetch ? 'deep' : 'shallow'}`,
    referenceTimestamp: input.referenceTimestamp,
  };
  const { key, referenceBucket } = buildDashaCacheKey(cacheKeyInput);
  const cached = await cacheGetJson<DashaFindingCacheEntry>(key);

  if (cached && isToolFindingLike(cached.finding)) {
    const finding = cloneToolFinding(cached.finding);
    finding.facts.push('Dasha cache: hit (Valkey).');
    return finding;
  }

  const payload = useDeepFetch && input.kundli
    ? await fetchCalculatedChart(input.kundli, {
        nesting: 5,
        infolevel: 'basic,panchanga,dasha',
        varga: 'D1',
      })
    : input.rawPayload;

  const finding = makeDashaToolFinding(payload, input.referenceTimestamp);
  await cacheSetJson(
    key,
    {
      cachedAt: Date.now(),
      referenceBucket,
      finding,
    },
    Math.max(300, Math.floor(Math.max(1, env.TIMING_CACHE_TTL_SECONDS) / 6))
  );

  const decorated = cloneToolFinding(finding);
  decorated.facts.push('Dasha cache: miss (computed fresh).');
  return decorated;
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
      'Live transit is fetched on demand from the backend transit analyzer, not from canonical chart data.',
    ];

    if (natalSun) {
      const sun = natalSun.value as Record<string, unknown>;
      facts.push(`Natal Sun reference: ${rashiName(sun.rashi)}${sun.house_number ? `, house ${sun.house_number}` : ''}.`);
    }

    facts.push('Use the live transit-chart endpoint for timing forecasts; canonical chart data is only the natal baseline.');

    return {
      name: 'Transit analyzer',
      status: 'partial',
      facts,
      evidencePaths: natalSun ? [natalSun.path] : [],
      missing: ['backend transit-chart payload'],
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

function makeGeneralToolFinding(
  rawPayload: unknown,
  question: string,
  flags: string[],
  mode: AgentMode = 'pro',
  scopeSelection: ScopeSelectionDecision | null = null
): ToolFinding {
  const sections = selectSections(rawPayload, question, flags, mode, scopeSelection);
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
  const deterministicUnified = buildDeterministicUnifiedIntentScopeDecision(state.question, state.mode);
  const result = await decideUnifiedIntentScopeDetailed(
    state.question,
    state.mode,
    state.conversationContext ?? [],
    state.referenceTimestamp ?? Date.now()
  );

  const legacyIntentEnabled = isDecisionNodeEnabled(env.LLM_DECISION_INTENT_ENABLED);
  const legacyTemporalEnabled = isDecisionNodeEnabled(env.LLM_DECISION_TEMPORAL_ENABLED);
  const legacyScopeEnabled = isDecisionNodeEnabled(env.LLM_DECISION_SCOPE_SELECTOR_ENABLED);

  const temporalWindow: TemporalWindowDecision = {
    direction: legacyTemporalEnabled ? result.decision.timeDirection : deterministicUnified.timeDirection,
    timeValue: legacyTemporalEnabled ? result.decision.timeValue : deterministicUnified.timeValue,
    timeUnit: legacyTemporalEnabled ? result.decision.timeUnit : deterministicUnified.timeUnit,
    timeLabel: legacyTemporalEnabled ? result.decision.timeLabel : deterministicUnified.timeLabel,
    confidence: legacyTemporalEnabled ? result.decision.confidence : 0.35,
  };

  const llmIntentBase: QuestionIntent = {
    primary: result.decision.primary,
    flags: [...new Set(result.decision.flags.map((flag) => normalizeToken(flag)))],
    topics: [...new Set(result.decision.topics.map((topic) => normalizeToken(topic)))],
    timeDirection: temporalWindow.direction,
    timeValue: temporalWindow.timeValue,
    timeUnit: temporalWindow.timeUnit,
    timeLabel: temporalWindow.timeLabel,
  };

  const deterministicIntentBase = classifyQuestionIntent(state.question);
  const intentBase = legacyIntentEnabled ? llmIntentBase : deterministicIntentBase;
  const intent = applyTemporalToIntent(intentBase, temporalWindow);

  const deterministicScope = buildDeterministicScopeSelection(state.question, intent, state.mode);
  const scopeSelection: ScopeSelectionDecision = legacyScopeEnabled
    ? {
        questionType: result.decision.questionType,
        requiredScopes: clampRequiredScopes(result.decision.requiredScopes, deterministicScope.requiredScopes, state.mode),
        requiredTools: (() => {
          const normalized = [...new Set(result.decision.requiredTools.map((item) => normalizeToken(item)))];
          return normalized.length > 0 ? normalized : deterministicScope.requiredTools;
        })(),
        needsDasha: result.decision.needsDasha,
        needsTransit: result.decision.needsTransit,
        needsD9: result.decision.needsD9,
        needsD10: result.decision.needsD10,
        needsLongevity: result.decision.needsLongevity,
        needsGeneral: result.decision.needsGeneral,
        confidence: result.decision.confidence,
      }
    : {
        ...deterministicScope,
        confidence: 0.35,
      };

  const deterministicIntentWithTemporal = applyTemporalToIntent(
    classifyQuestionIntent(state.question),
    {
      direction: deterministicUnified.timeDirection,
      timeValue: deterministicUnified.timeValue,
      timeUnit: deterministicUnified.timeUnit,
      timeLabel: deterministicUnified.timeLabel,
      confidence: deterministicUnified.confidence,
    }
  );

  const usedAnyLlm = (legacyIntentEnabled || legacyTemporalEnabled || legacyScopeEnabled) && !result.usedFallback;
  const source: DecisionBundle['source'] = usedAnyLlm
    ? 'llm'
    : (legacyIntentEnabled || legacyTemporalEnabled || legacyScopeEnabled)
      ? 'hybrid'
      : 'deterministic';

  const shadowIntentComparison = env.LLM_DECISION_SHADOW_MODE
    ? `deterministicPrimary=${deterministicIntentWithTemporal.primary}; llmPrimary=${result.decision.primary}; finalPrimary=${intent.primary}; match=${deterministicIntentWithTemporal.primary === intent.primary}`
    : undefined;

  return {
    temporalWindow,
    intent,
    scopeSelection,
    decisionBundle: mergeDecisionBundle(state, {
      source,
      intentPrimary: intent.primary,
      questionFamily: determineQuestionFamily(state.question, intent),
      timeDirection: temporalWindow.direction,
      requiredScopes: scopeSelection.requiredScopes,
      requiredToolGroups: clampToolGroups(
        scopeSelection.requiredTools,
        ['reference_time', 'general_grounding', 'varga', 'placement'],
        state.mode
      ),
      miniEnforcementMode: state.mode === 'mini' ? evaluateMiniScope(state.question).enforcementMode : 'full',
      confidence: result.decision.confidence,
      reason: 'unified intent+temporal+scope decision',
    }),
    decisionTelemetry: appendDecisionTelemetry(state, {
      node: 'intent_scope_unified',
      model: result.model,
      latencyMs: result.latencyMs,
      confidence: result.decision.confidence,
      usedFallback: result.usedFallback,
      fallbackReason: result.usedFallback ? 'unified intent/scope fallback' : undefined,
      shadowComparison: shadowIntentComparison,
    }),
    analysisStages: appendStage(
      state,
      'classify_intent',
      'Classifying user question intent',
      `Primary=${intent.primary}; flags=${intent.flags.join(',') || 'none'}; temporal=${temporalWindow.direction}; questionType=${scopeSelection.questionType}; model=${result.model}; latencyMs=${result.latencyMs}; confidence=${result.decision.confidence.toFixed(2)}`
    ),
  };
}

async function routeTopLevelNode(state: AgentStateType): Promise<AgentUpdateType> {
  if (state.topLevelRoute) {
    return {
      analysisStages: appendStage(
        state,
        'route_top_level',
        'Routing question at top-level',
        `route=${state.topLevelRoute}; style=${state.responseStyleHint ?? 'brief'}; continuation=${Boolean(state.continuationIntent)}`
      ),
    };
  }

  const result = await decideTopLevelRouteDetailed(state.question, state.mode, state.conversationContext ?? []);
  const decision = result.decision;
  const deterministicRoute = decideTopLevelRouteDeterministic(state.question);
  const shadowComparison = env.LLM_DECISION_SHADOW_MODE
    ? `deterministic=${deterministicRoute}; llm=${decision.topRoute}; match=${deterministicRoute === decision.topRoute}`
    : undefined;
  return {
    topLevelRoute: decision.topRoute,
    topLevelRouteConfidence: decision.confidence,
    responseStyleHint: decision.responseStyle,
    continuationIntent: decision.continuityIntent,
    decisionBundle: mergeDecisionBundle(state, {
      source: result.usedFallback ? 'hybrid' : 'llm',
      topRoute: decision.topRoute,
      responseStyle: decision.responseStyle,
      continuityIntent: decision.continuityIntent,
      confidence: decision.confidence,
      reason: 'top-level routing',
    }),
    decisionTelemetry: appendDecisionTelemetry(state, {
      node: 'route_top_level',
      model: result.model,
      latencyMs: result.latencyMs,
      confidence: decision.confidence,
      usedFallback: result.usedFallback,
      fallbackReason: result.usedFallback ? 'route decision fallback' : undefined,
      shadowComparison,
    }),
    analysisStages: appendStage(
      state,
      'route_top_level',
      'Routing question at top-level',
      `route=${decision.topRoute}; style=${decision.responseStyle}; continuation=${decision.continuityIntent}; confidence=${decision.confidence.toFixed(2)}; model=${result.model}; latencyMs=${result.latencyMs}`
    ),
  };
}

function routeFromTopLevel(state: AgentStateType): 'fast_answer' | 'classify_intent' {
  if (state.mode === 'mini') {
    const miniScope = evaluateMiniScope(state.question);
    if (!miniScope.allowed || miniScope.enforcementMode === 'blocked') {
      return 'classify_intent';
    }
  }

  if (shouldForcePipelineRoute(state.question)) {
    return 'classify_intent';
  }

  return state.topLevelRoute === 'pipeline' ? 'classify_intent' : 'fast_answer';
}

async function fastAnswerNode(state: AgentStateType): Promise<AgentUpdateType> {
  if (state.mode === 'mini') {
    const miniScope = evaluateMiniScope(state.question);
    if (!miniScope.allowed || miniScope.enforcementMode === 'blocked') {
      return {
        answer: buildMiniUpgradeResponse(state.question, miniScope.reasons),
        model: 'cozmic-mini-guard',
        executionPlan: {
          family: 'general',
          chartLayers: [],
          includeTiming: false,
          includeTransit: false,
          includeDasha: false,
          includeCareer: false,
          includeRelationship: false,
          includeMicroSignals: [],
          seriesNodes: ['route_top_level', 'fast_answer'],
          parallelBatches: [],
        },
        analysisStages: appendStage(
          state,
          'fast_answer',
          'Returning direct LLM answer for non-pipeline route',
          'mini scope blocked (transit/timing or pro-only request)'
        ),
      };
    }
  }

  const route = state.topLevelRoute ?? 'smalltalk';
  const styleHint = state.responseStyleHint ?? deriveResponseStyleHint(state.question, state.conversationContext ?? []);
  const fast = await answerSimpleWithoutChart(state.question, state.mode, route, state.conversationContext ?? [], styleHint);
  return {
    answer: fast.answer,
    model: fast.model,
    executionPlan: {
      family: 'general',
      chartLayers: [],
      includeTiming: false,
      includeTransit: false,
      includeDasha: false,
      includeCareer: false,
      includeRelationship: false,
      includeMicroSignals: [],
      seriesNodes: ['route_top_level', 'fast_answer'],
      parallelBatches: [],
    },
    analysisStages: appendStage(state, 'fast_answer', 'Returning direct LLM answer for non-pipeline route', `route=${route}`),
  };
}

async function planExecutionNode(state: AgentStateType): Promise<AgentUpdateType> {
  const intent = state.intent ?? classifyQuestionIntent(state.question);
  const deterministicPlan = buildDynamicExecutionPlan(state.question, intent, state.mode);

  if (!isDecisionNodeEnabled(env.LLM_DECISION_PLAN_ENABLED)) {
    return {
      executionPlan: deterministicPlan,
      decisionTelemetry: appendDecisionTelemetry(state, {
        node: 'plan_execution',
        model: 'plan-disabled',
        latencyMs: 0,
        confidence: 0.25,
        usedFallback: true,
      }),
      analysisStages: appendStage(
        state,
        'plan_execution',
        'Planning dynamic analysis path',
        `Family=${deterministicPlan.family}; layers=${deterministicPlan.chartLayers.join(',')}; parallelBatches=${deterministicPlan.parallelBatches.length}; model=plan-disabled`
      ),
    };
  }

  const result = await invokeDecisionNode<ExecutionPlanDecision>({
    node: 'plan_execution',
    schema: ExecutionPlanDecisionSchema,
    input: {
      question: state.question,
      mode: state.mode,
      intent,
      allowedFamilies: [...QUESTION_FAMILY_ALLOWLIST],
      allowedChartLayers: [...CHART_LAYER_ALLOWLIST],
      allowedMicroSignals: [...MICRO_SIGNAL_ALLOWLIST],
      instruction:
        'Return execution plan fields for chart analysis. Use conservative defaults when uncertain and avoid unsupported layers.',
    },
    fallback: () => ({
      questionFamily: deterministicPlan.family,
      requiredChartLayers: deterministicPlan.chartLayers,
      includeMicroSignals: deterministicPlan.includeMicroSignals,
      includeTiming: deterministicPlan.includeTiming,
      includeTransit: deterministicPlan.includeTransit,
      includeDasha: deterministicPlan.includeDasha,
      includeCareer: deterministicPlan.includeCareer,
      includeRelationship: deterministicPlan.includeRelationship,
      confidence: 0.35,
    }),
  });

  const executionPlan: DynamicExecutionPlan = {
    ...deterministicPlan,
    family: clampQuestionFamily(result.decision.questionFamily, deterministicPlan.family),
    chartLayers: clampChartLayers(result.decision.requiredChartLayers, deterministicPlan.chartLayers),
    includeMicroSignals: clampMicroSignals(result.decision.includeMicroSignals, deterministicPlan.includeMicroSignals),
    includeTiming: result.decision.includeTiming,
    includeTransit: state.mode === 'pro' ? result.decision.includeTransit : false,
    includeDasha: state.mode === 'pro' ? result.decision.includeDasha : false,
    includeCareer: state.mode === 'pro' ? result.decision.includeCareer : false,
    includeRelationship: result.decision.includeRelationship,
  };

  return {
    executionPlan,
    decisionTelemetry: appendDecisionTelemetry(state, {
      node: 'plan_execution',
      model: result.model,
      latencyMs: result.latencyMs,
      confidence: result.decision.confidence,
      usedFallback: result.usedFallback,
    }),
    analysisStages: appendStage(
      state,
      'plan_execution',
      'Planning dynamic analysis path',
      `Family=${executionPlan.family}; layers=${executionPlan.chartLayers.join(',')}; parallelBatches=${executionPlan.parallelBatches.length}; model=${result.model}; latencyMs=${result.latencyMs}; confidence=${result.decision.confidence.toFixed(2)}`
    ),
  };
}

async function planAndToolsNode(state: AgentStateType): Promise<AgentUpdateType> {
  const intent = state.intent ?? classifyQuestionIntent(state.question);
  const deterministicPlan = buildDynamicExecutionPlan(state.question, intent, state.mode);
  const deterministicGroups = buildDeterministicToolGroupsFromInputs({
    question: state.question,
    mode: state.mode,
    intent,
    executionPlan: deterministicPlan,
    scopeSelection: state.scopeSelection,
  });

  const availableByManifest = getToolManifestForMode(state.mode).map((item) => item.group);
  const preflight = state.toolAvailabilityPreflight;
  const preflightAvailable = preflight?.availableGroups ?? availableByManifest;
  const preflightBlocked = preflight?.blockedByMode ?? [];
  const deterministicManifestSafe = deterministicGroups.filter((group) => preflightAvailable.includes(group) && !preflightBlocked.includes(group));
  const deterministicForDecision = deterministicManifestSafe.length > 0 ? deterministicManifestSafe : deterministicGroups;

  const planEnabled = isDecisionNodeEnabled(env.LLM_DECISION_PLAN_ENABLED);
  const toolEnabled = isDecisionNodeEnabled(env.LLM_DECISION_TOOL_SELECTION_ENABLED);

  if (!planEnabled && !toolEnabled) {
    return {
      executionPlan: deterministicPlan,
      selectedToolGroups: deterministicForDecision,
      decisionBundle: mergeDecisionBundle(state, {
        source: 'deterministic',
        questionFamily: deterministicPlan.family,
        requiredToolGroups: deterministicForDecision,
        confidence: 0.35,
        reason: 'plan_and_tools deterministic path',
      }),
      decisionTelemetry: appendDecisionTelemetry(state, {
        node: 'plan_and_tools',
        model: 'plan-and-tools-disabled',
        latencyMs: 0,
        confidence: 0.25,
        usedFallback: true,
      }),
      analysisStages: appendStage(
        state,
        'plan_and_tools',
        'Planning execution and selecting tools',
        `Family=${deterministicPlan.family}; layers=${deterministicPlan.chartLayers.join(',')}; selected=${deterministicForDecision.join(', ')}; model=plan-and-tools-disabled`
      ),
    };
  }

  const result = await invokeDecisionNode<PlanAndToolsDecision>({
    node: 'plan_and_tools',
    schema: PlanAndToolsDecisionSchema,
    input: {
      question: state.question,
      mode: state.mode,
      intent,
      scopeSelection: state.scopeSelection,
      deterministicPlan: {
        questionFamily: deterministicPlan.family,
        requiredChartLayers: deterministicPlan.chartLayers,
        includeMicroSignals: deterministicPlan.includeMicroSignals,
        includeTiming: deterministicPlan.includeTiming,
        includeTransit: deterministicPlan.includeTransit,
        includeDasha: deterministicPlan.includeDasha,
        includeCareer: deterministicPlan.includeCareer,
        includeRelationship: deterministicPlan.includeRelationship,
      },
      deterministicGroups: deterministicForDecision,
      allowedFamilies: [...QUESTION_FAMILY_ALLOWLIST],
      allowedChartLayers: [...CHART_LAYER_ALLOWLIST],
      allowedMicroSignals: [...MICRO_SIGNAL_ALLOWLIST],
      allowedGroups: [...TOOL_GROUP_ALLOWLIST],
      toolManifest: getToolManifestForMode(state.mode).map((item) => ({
        group: item.group,
        minMode: item.minMode,
        domains: item.domains,
        requiredScopes: item.requiredScopes,
        costClass: item.costClass,
        fallbackGroup: item.fallbackGroup,
      })),
      preflight,
      instruction:
        'Return one JSON containing execution-plan fields and selected tool groups. Keep selection minimal but sufficient, respect mode and available data, and prefer deterministic hints when uncertain.',
    },
    fallback: () => ({
      questionFamily: deterministicPlan.family,
      requiredChartLayers: deterministicPlan.chartLayers,
      includeMicroSignals: deterministicPlan.includeMicroSignals,
      includeTiming: deterministicPlan.includeTiming,
      includeTransit: deterministicPlan.includeTransit,
      includeDasha: deterministicPlan.includeDasha,
      includeCareer: deterministicPlan.includeCareer,
      includeRelationship: deterministicPlan.includeRelationship,
      selectedToolGroups: deterministicForDecision,
      confidence: 0.35,
    }),
  });

  const executionPlan: DynamicExecutionPlan = planEnabled
    ? {
        ...deterministicPlan,
        family: clampQuestionFamily(result.decision.questionFamily, deterministicPlan.family),
        chartLayers: clampChartLayers(result.decision.requiredChartLayers, deterministicPlan.chartLayers),
        includeMicroSignals: clampMicroSignals(result.decision.includeMicroSignals, deterministicPlan.includeMicroSignals),
        includeTiming: result.decision.includeTiming,
        includeTransit: state.mode === 'pro' ? result.decision.includeTransit : false,
        includeDasha: state.mode === 'pro' ? result.decision.includeDasha : false,
        includeCareer: state.mode === 'pro' ? result.decision.includeCareer : false,
        includeRelationship: result.decision.includeRelationship,
      }
    : deterministicPlan;

  const llmSelectedToolGroups = clampToolGroups(result.decision.selectedToolGroups, deterministicForDecision, state.mode)
    .filter((group) => preflightAvailable.includes(group) && !preflightBlocked.includes(group));
  const selectedToolGroups = toolEnabled
    ? [...new Set([...llmSelectedToolGroups, ...deterministicForDecision])]
    : deterministicForDecision;

  const shadowComparison = env.LLM_DECISION_SHADOW_MODE
    ? `planFamily deterministic=${deterministicPlan.family}; llm=${executionPlan.family}; tools deterministic=${deterministicForDecision.join('|')}; llm=${selectedToolGroups.join('|')}`
    : undefined;

  return {
    executionPlan,
    selectedToolGroups,
    decisionBundle: mergeDecisionBundle(state, {
      source: result.usedFallback ? 'hybrid' : 'llm',
      questionFamily: executionPlan.family,
      requiredToolGroups: selectedToolGroups,
      confidence: result.decision.confidence,
      reason: 'merged plan_and_tools node',
    }),
    decisionTelemetry: appendDecisionTelemetry(state, {
      node: 'plan_and_tools',
      model: result.model,
      latencyMs: result.latencyMs,
      confidence: result.decision.confidence,
      usedFallback: result.usedFallback,
      fallbackReason: result.usedFallback ? 'plan_and_tools fallback' : undefined,
      shadowComparison,
    }),
    analysisStages: appendStage(
      state,
      'plan_and_tools',
      'Planning execution and selecting tools',
      `Family=${executionPlan.family}; layers=${executionPlan.chartLayers.join(',')}; selected=${selectedToolGroups.join(', ')}; model=${result.model}; latencyMs=${result.latencyMs}; confidence=${result.decision.confidence.toFixed(2)}`
    ),
  };
}

function buildDeterministicToolGroupsFromInputs(params: {
  question: string;
  mode: AgentMode;
  intent: QuestionIntent;
  executionPlan: DynamicExecutionPlan;
  scopeSelection: ScopeSelectionDecision | null;
}): ToolGroupKey[] {
  const { question, mode, intent, executionPlan, scopeSelection: scope } = params;
  const groups = new Set<ToolGroupKey>(['reference_time', 'general_grounding', 'varga', 'placement']);

  if (executionPlan.family !== 'general') {
    groups.add('atlas');
  }

  if (scope?.needsD9 || executionPlan.chartLayers.includes('D9') || intent.flags.includes('d9') || intent.flags.includes('relationship')) {
    groups.add('d9');
  }

  if (mode === 'pro') {
    if (scope?.needsDasha || executionPlan.includeDasha || intent.flags.includes('dasha') || intent.flags.includes('timing')) {
      groups.add('dasha');
    }

    if (scope?.needsTransit || executionPlan.includeTransit || intent.flags.includes('transit') || hasExplicitTransitCue(question)) {
      groups.add('transit');
    }

    if (scope?.needsD10 || executionPlan.includeCareer || intent.flags.includes('career') || intent.flags.includes('career_timing')) {
      groups.add('career');
    }

    if (executionPlan.family === 'remedies' || intent.flags.includes('remedies') || intent.flags.includes('hardship')) {
      groups.add('remedies');
    }

    if (executionPlan.family === 'relocation' || intent.flags.includes('relocation')) {
      groups.add('relocation');
    }

    if (executionPlan.family === 'past_life' || intent.flags.includes('past_life')) {
      groups.add('past_life');
    }

    if (executionPlan.family === 'pregnancy_fertility' || intent.flags.includes('pregnancy_fertility')) {
      groups.add('pregnancy_fertility');
    }

    if (executionPlan.family === 'legal' || intent.flags.includes('legal')) {
      groups.add('legal');
    }

    if (executionPlan.family === 'finance' || intent.flags.includes('finance') || intent.topics.includes('finance')) {
      groups.add('finance');
    }

    if (executionPlan.family === 'health' || intent.flags.includes('health') || intent.topics.includes('health')) {
      groups.add('health');
    }

    if (executionPlan.family === 'education' || intent.topics.includes('education')) {
      groups.add('education');
    }

    if (executionPlan.family === 'children' || intent.topics.includes('children')) {
      groups.add('children');
    }

    if (executionPlan.family === 'property' || intent.topics.includes('property')) {
      groups.add('property');
    }

    if (executionPlan.family === 'travel' || intent.topics.includes('travel')) {
      groups.add('travel');
    }

    if (executionPlan.family === 'spirituality' || intent.topics.includes('spirituality')) {
      groups.add('spirituality');
    }

    if (scope?.needsLongevity || executionPlan.family === 'longevity' || intent.flags.includes('longevity')) {
      groups.add('longevity');
    }
  }

  if (/\b(arudha|aruda)\b/i.test(question) && mode === 'pro') {
    groups.add('arudha');
  }

  if (/\b(panchanga|tithi|nakshatra|karana|yoga)\b/i.test(question)) {
    groups.add('panchanga');
  }

  if (/\b(yoga|yogas|ashtakavarga|arudha|aruda)\b/i.test(question) && mode === 'pro') {
    groups.add('feature');
  }

  if (executionPlan.includeMicroSignals.includes('nakshatra') || executionPlan.includeMicroSignals.includes('nakshatra_lord') || executionPlan.includeMicroSignals.includes('sign_lord')) {
    groups.add('nakshatra_lord');
  }

  if (executionPlan.includeMicroSignals.includes('drishti') || executionPlan.includeMicroSignals.includes('degree')) {
    groups.add('drishti_degree');
  }

  return clampToolGroups([...groups], ['reference_time', 'general_grounding', 'varga', 'placement'], mode);
}

function buildDeterministicToolGroups(state: AgentStateType): ToolGroupKey[] {
  const intent = state.intent ?? classifyQuestionIntent(state.question);
  const executionPlan = state.executionPlan ?? buildDynamicExecutionPlan(state.question, intent, state.mode);
  return buildDeterministicToolGroupsFromInputs({
    question: state.question,
    mode: state.mode,
    intent,
    executionPlan,
    scopeSelection: state.scopeSelection,
  });
}

async function selectToolGroupsNode(state: AgentStateType): Promise<AgentUpdateType> {
  const deterministicGroups = buildDeterministicToolGroups(state);
  const availableByManifest = getToolManifestForMode(state.mode).map((item) => item.group);
  const preflight = state.toolAvailabilityPreflight;
  const preflightAvailable = preflight?.availableGroups ?? availableByManifest;
  const preflightBlocked = preflight?.blockedByMode ?? [];
  const deterministicManifestSafe = deterministicGroups.filter((group) => preflightAvailable.includes(group) && !preflightBlocked.includes(group));
  const deterministicForDecision = deterministicManifestSafe.length > 0 ? deterministicManifestSafe : deterministicGroups;

  if (!isDecisionNodeEnabled(env.LLM_DECISION_TOOL_SELECTION_ENABLED)) {
    return {
      selectedToolGroups: deterministicForDecision,
      decisionBundle: mergeDecisionBundle(state, {
        source: 'deterministic',
        requiredToolGroups: deterministicForDecision,
        confidence: 0.35,
        reason: 'deterministic tool selection (node disabled)',
      }),
      decisionTelemetry: appendDecisionTelemetry(state, {
        node: 'tool_selection',
        model: 'tool-selection-disabled',
        latencyMs: 0,
        confidence: 0.25,
        usedFallback: true,
        fallbackReason: 'LLM_DECISION_TOOL_SELECTION_ENABLED=false',
      }),
      analysisStages: appendStage(
        state,
        'tool_selection',
        'Selecting tool groups for analysis',
        `selected=${deterministicForDecision.join(', ')}; model=tool-selection-disabled`
      ),
    };
  }

  const result = await invokeDecisionNode<ToolSelectionDecision>({
    node: 'tool_selection',
    schema: ToolSelectionDecisionSchema,
    input: {
      question: state.question,
      mode: state.mode,
      intent: state.intent,
      scopeSelection: state.scopeSelection,
      executionPlan: state.executionPlan,
      deterministicGroups: deterministicForDecision,
      allowedGroups: [...TOOL_GROUP_ALLOWLIST],
      toolManifest: getToolManifestForMode(state.mode).map((item) => ({
        group: item.group,
        minMode: item.minMode,
        domains: item.domains,
        requiredScopes: item.requiredScopes,
        costClass: item.costClass,
        fallbackGroup: item.fallbackGroup,
      })),
      preflight,
      instruction:
        'Select minimal but sufficient tool groups for this query. Prefer deterministicGroups when uncertain and avoid unnecessary expensive groups.',
    },
    fallback: () => ({
      selectedToolGroups: deterministicForDecision,
      confidence: 0.35,
    }),
  });

  const selectedToolGroups = clampToolGroups(result.decision.selectedToolGroups, deterministicForDecision, state.mode)
    .filter((group) => preflightAvailable.includes(group) && !preflightBlocked.includes(group));
  const finalGroups = [...new Set([...selectedToolGroups, ...deterministicForDecision])];
  const shadowComparison = env.LLM_DECISION_SHADOW_MODE
    ? `deterministic=${deterministicForDecision.join('|')}; llm=${finalGroups.join('|')}; match=${deterministicForDecision.join('|') === finalGroups.join('|')}`
    : undefined;

  return {
    selectedToolGroups: finalGroups,
    decisionBundle: mergeDecisionBundle(state, {
      source: result.usedFallback ? 'hybrid' : 'llm',
      requiredToolGroups: finalGroups,
      confidence: result.decision.confidence,
      reason: 'tool selection with manifest + preflight',
    }),
    decisionTelemetry: appendDecisionTelemetry(state, {
      node: 'tool_selection',
      model: result.model,
      latencyMs: result.latencyMs,
      confidence: result.decision.confidence,
      usedFallback: result.usedFallback,
      fallbackReason: result.usedFallback ? 'tool selection fallback' : undefined,
      shadowComparison,
    }),
    analysisStages: appendStage(
      state,
      'tool_selection',
      'Selecting tool groups for analysis',
      `selected=${finalGroups.join(', ')}; model=${result.model}; latencyMs=${result.latencyMs}; confidence=${result.decision.confidence.toFixed(2)}`
    ),
  };
}

type ToolTaskContext = {
  grounding: GroundingContext;
  question: string;
  mode: AgentMode;
  intent: QuestionIntent;
  executionPlan: DynamicExecutionPlan;
  scopeSelection: ScopeSelectionDecision | null;
  analysisTimestamp: number;
};

function createToolTaskRegistry(ctx: ToolTaskContext): Record<ToolGroupKey, () => Promise<ToolFinding | null>> {
  const { grounding, question, mode, intent, analysisTimestamp, scopeSelection } = ctx;
  const isMini = mode === 'mini';

  return {
    reference_time: () => Promise.resolve(makeReferenceTimeToolFinding(grounding.referenceTimestamp, grounding.referenceTimeSource)),
    atlas: () => Promise.resolve(makeAtlasToolFinding(grounding.rawPayload)),
    varga: () => Promise.resolve(makeVargaToolFinding(grounding.rawPayload, question, mode)),
    arudha: () => (isMini ? Promise.resolve(null) : Promise.resolve(makeArudhaToolFinding(grounding.rawPayload))),
    d9: () => Promise.resolve(makeD9ToolFinding(grounding.rawPayload, question)),
    dasha: () => (isMini
      ? Promise.resolve(null)
      : buildDashaToolFindingCached({
          rawPayload: grounding.rawPayload,
          referenceTimestamp: analysisTimestamp,
          ownerId: grounding.ownerId,
          profileId: grounding.profileId,
          payloadHash: grounding.payloadHash,
          question,
          kundli: grounding.kundli,
        })),
    transit: () => (isMini
      ? Promise.resolve(null)
      : resolveTransitRequestKind(question, grounding.referenceTimestamp) === 'range'
        ? buildTransitIntervalToolFinding({ kundli: grounding.kundli, question, referenceTimestamp: grounding.referenceTimestamp })
        : buildTransitPointToolFinding({ kundli: grounding.kundli, question, referenceTimestamp: grounding.referenceTimestamp })),
    career: () => (isMini ? Promise.resolve(null) : Promise.resolve(makeCareerToolFinding(grounding.rawPayload, question, analysisTimestamp))),
    remedies: () => (isMini ? Promise.resolve(null) : Promise.resolve(makeRemediesToolFinding(grounding.rawPayload, analysisTimestamp, question))),
    relocation: () => (isMini ? Promise.resolve(null) : Promise.resolve(makeRelocationToolFinding(grounding.rawPayload, analysisTimestamp))),
    past_life: () => (isMini ? Promise.resolve(null) : Promise.resolve(makePastLifeToolFinding(grounding.rawPayload, analysisTimestamp))),
    pregnancy_fertility: () => (isMini ? Promise.resolve(null) : Promise.resolve(makePregnancyFertilityToolFinding(grounding.rawPayload, analysisTimestamp))),
    legal: () => (isMini ? Promise.resolve(null) : Promise.resolve(makeLegalToolFinding(grounding.rawPayload, analysisTimestamp))),
    finance: () => (isMini ? Promise.resolve(null) : Promise.resolve(makeFinanceToolFinding(grounding.rawPayload, analysisTimestamp))),
    health: () => (isMini ? Promise.resolve(null) : Promise.resolve(makeHealthToolFinding(grounding.rawPayload, analysisTimestamp))),
    education: () => (isMini ? Promise.resolve(null) : Promise.resolve(makeEducationToolFinding(grounding.rawPayload, analysisTimestamp))),
    children: () => (isMini ? Promise.resolve(null) : Promise.resolve(makeChildrenToolFinding(grounding.rawPayload, analysisTimestamp))),
    property: () => (isMini ? Promise.resolve(null) : Promise.resolve(makePropertyToolFinding(grounding.rawPayload, analysisTimestamp))),
    travel: () => (isMini ? Promise.resolve(null) : Promise.resolve(makeTravelToolFinding(grounding.rawPayload, analysisTimestamp))),
    spirituality: () => (isMini ? Promise.resolve(null) : Promise.resolve(makeSpiritualityToolFinding(grounding.rawPayload, analysisTimestamp))),
    longevity: () => (isMini ? Promise.resolve(null) : Promise.resolve(makeLongevityToolFinding(grounding.rawPayload, grounding.referenceTimestamp))),
    placement: () => Promise.resolve(makePlacementToolFinding(grounding.rawPayload, question)),
    panchanga: () => Promise.resolve(makePanchangaToolFinding(grounding.rawPayload)),
    feature: () => (isMini ? Promise.resolve(null) : Promise.resolve(makeFeatureToolFinding(grounding.rawPayload))),
    general_grounding: () => Promise.resolve(makeGeneralToolFinding(grounding.rawPayload, question, intent.flags, mode, scopeSelection)),
    nakshatra_lord: () => Promise.resolve(makeNakshatraLordToolFinding(grounding.rawPayload)),
    drishti_degree: () => Promise.resolve(makeDrishtiDegreeToolFinding(grounding.rawPayload)),
  };
}

async function executeToolGroups(
  groups: ToolGroupKey[],
  registry: Record<ToolGroupKey, () => Promise<ToolFinding | null>>
): Promise<ToolFinding[]> {
  const uniqueGroups = [...new Set(groups)];
  const findings = await Promise.all(uniqueGroups.map((group) => registry[group]()));
  return findings.filter((item): item is ToolFinding => Boolean(item));
}

async function runSpecializedToolsNode(state: AgentStateType): Promise<AgentUpdateType> {
  const grounding = state.grounding;
  if (!grounding) throw new Error('Grounding context missing before specialized tools stage.');
  const intent = state.intent ?? classifyQuestionIntent(state.question);
  const executionPlan = state.executionPlan ?? buildDynamicExecutionPlan(state.question, intent, state.mode);
  const scopeSelection = state.scopeSelection;
  const selectedGroups = new Set(state.selectedToolGroups ?? []);
  const useSelection = selectedGroups.size > 0;
  const shouldRun = (group: ToolGroupKey, fallback: boolean) => (useSelection ? selectedGroups.has(group) : fallback);
  const analysisTimestamp = resolveAnalysisTimestamp(grounding.referenceTimestamp, intent);
  const isMini = state.mode === 'mini';

  const includeTransit = shouldRun('transit', executionPlan.includeTransit || Boolean(scopeSelection?.needsTransit));
  const includeD9 = shouldRun('d9', executionPlan.chartLayers.includes('D9') || intent.flags.includes('d9') || intent.primary === 'd9' || Boolean(scopeSelection?.needsD9));
  const includeDasha = shouldRun('dasha', executionPlan.includeDasha || Boolean(scopeSelection?.needsDasha));
  const includePlacement = shouldRun('placement', intent.flags.includes('relationship') || intent.flags.includes('career') || intent.flags.includes('health') || intent.flags.includes('finance') || executionPlan.family !== 'general');
  const includeCareer = shouldRun('career', executionPlan.includeCareer || intent.topics.includes('career') || intent.flags.includes('career') || intent.flags.includes('career_timing') || Boolean(scopeSelection?.needsD10));
  const includeRemedies = shouldRun('remedies', executionPlan.family === 'remedies' || intent.flags.includes('remedies') || intent.flags.includes('hardship'));
  const includeRelocation = shouldRun('relocation', executionPlan.family === 'relocation' || intent.flags.includes('relocation'));
  const includePastLife = shouldRun('past_life', executionPlan.family === 'past_life' || intent.flags.includes('past_life'));
  const includePregnancyFertility = shouldRun('pregnancy_fertility', executionPlan.family === 'pregnancy_fertility' || intent.flags.includes('pregnancy_fertility'));
  const includeLegal = shouldRun('legal', executionPlan.family === 'legal' || intent.flags.includes('legal'));
  const includeFinance = shouldRun('finance', executionPlan.family === 'finance' || intent.topics.includes('finance') || intent.flags.includes('finance'));
  const includeHealth = shouldRun('health', executionPlan.family === 'health' || intent.topics.includes('health') || intent.flags.includes('health'));
  const includeEducation = shouldRun('education', executionPlan.family === 'education' || intent.topics.includes('education'));
  const includeChildren = shouldRun('children', executionPlan.family === 'children' || intent.topics.includes('children'));
  const includeProperty = shouldRun('property', executionPlan.family === 'property' || intent.topics.includes('property'));
  const includeTravel = shouldRun('travel', executionPlan.family === 'travel' || intent.topics.includes('travel'));
  const includeSpirituality = shouldRun('spirituality', executionPlan.family === 'spirituality' || intent.topics.includes('spirituality'));
  const includeLongevity = shouldRun('longevity', executionPlan.family === 'longevity' || intent.flags.includes('longevity') || Boolean(scopeSelection?.needsLongevity));
  const includeArudha = shouldRun('arudha', /\b(arudha|aruda)\b/i.test(state.question) && !isMini);
  const includeNakshatraLord = shouldRun(
    'nakshatra_lord',
    executionPlan.includeMicroSignals.includes('nakshatra') || executionPlan.includeMicroSignals.includes('nakshatra_lord') || executionPlan.includeMicroSignals.includes('sign_lord')
  );
  const includeDrishtiDegree = shouldRun('drishti_degree', executionPlan.includeMicroSignals.includes('drishti') || executionPlan.includeMicroSignals.includes('degree'));

  const groupsToRun: ToolGroupKey[] = [
    shouldRun('reference_time', true) ? 'reference_time' : null,
    shouldRun('atlas', true) ? 'atlas' : null,
    shouldRun('varga', true) ? 'varga' : null,
    includeArudha ? 'arudha' : null,
    includeD9 ? 'd9' : null,
    includeDasha && !isMini ? 'dasha' : null,
    includeTransit && !isMini ? 'transit' : null,
    includeCareer && !isMini ? 'career' : null,
    includeRemedies && !isMini ? 'remedies' : null,
    includeRelocation && !isMini ? 'relocation' : null,
    includePastLife && !isMini ? 'past_life' : null,
    includePregnancyFertility && !isMini ? 'pregnancy_fertility' : null,
    includeLegal && !isMini ? 'legal' : null,
    includeFinance && !isMini ? 'finance' : null,
    includeHealth && !isMini ? 'health' : null,
    includeEducation && !isMini ? 'education' : null,
    includeChildren && !isMini ? 'children' : null,
    includeProperty && !isMini ? 'property' : null,
    includeTravel && !isMini ? 'travel' : null,
    includeSpirituality && !isMini ? 'spirituality' : null,
    includeLongevity && !isMini ? 'longevity' : null,
    includePlacement ? 'placement' : null,
    includeNakshatraLord ? 'nakshatra_lord' : null,
    includeDrishtiDegree ? 'drishti_degree' : null,
  ].filter((item): item is ToolGroupKey => Boolean(item));

  const registry = createToolTaskRegistry({
    grounding,
    question: state.question,
    mode: state.mode,
    intent,
    executionPlan,
    scopeSelection,
    analysisTimestamp,
  });

  const findings = mergeFindings(state.toolFindings ?? [], await executeToolGroups(groupsToRun, registry));

  return {
    toolFindings: findings,
    analysisStages: appendStage(
      state,
      'run_specialized_tools',
      'Running specialized analyzers',
      `Executed groups=${groupsToRun.join(', ') || 'none'}; findings=${findings.length}`
    ),
  };
}

async function runGeneralToolsNode(state: AgentStateType): Promise<AgentUpdateType> {
  const grounding = state.grounding;
  if (!grounding) throw new Error('Grounding context missing before general tools stage.');
  const intent = state.intent ?? classifyQuestionIntent(state.question);
  const executionPlan = state.executionPlan ?? buildDynamicExecutionPlan(state.question, intent, state.mode);
  const scopeSelection = state.scopeSelection;
  const selectedGroups = new Set(state.selectedToolGroups ?? []);
  const useSelection = selectedGroups.size > 0;
  const shouldRun = (group: ToolGroupKey, fallback: boolean) => (useSelection ? selectedGroups.has(group) : fallback);
  const analysisTimestamp = resolveAnalysisTimestamp(grounding.referenceTimestamp, intent);
  const isMini = state.mode === 'mini';

  const includePanchanga = shouldRun('panchanga', /\b(panchanga|tithi|nakshatra|karana|yoga)\b/i.test(state.question));
  const includeFeature = shouldRun('feature', /\b(yoga|yogas|ashtakavarga|arudha|aruda)\b/i.test(state.question) && !isMini);
  const includeDasha = shouldRun('dasha', (executionPlan.includeDasha || Boolean(scopeSelection?.needsDasha)) && !intent.flags.includes('dasha') && !isMini);
  const includeTransit = shouldRun('transit', (executionPlan.includeTransit || Boolean(scopeSelection?.needsTransit) || hasExplicitTransitCue(state.question)) && !isMini);
  const includeCareer = shouldRun('career', (executionPlan.includeCareer || Boolean(scopeSelection?.needsD10) || intent.topics.includes('career') || /\b(career|job|profession|business|promotion|work|employment|salary|interview)\b/i.test(state.question)) && !isMini);
  const includeRemedies = shouldRun('remedies', (executionPlan.family === 'remedies' || intent.flags.includes('remedies') || intent.flags.includes('hardship') || /\b(remedy|remedies|upay|upaya|mantra|gemstone|puja|pooja|fasting)\b/i.test(state.question)) && !isMini);
  const includeRelocation = shouldRun('relocation', (executionPlan.family === 'relocation' || intent.flags.includes('relocation') || /\b(relocation|migrate|migration|settle abroad|foreign settlement|move abroad)\b/i.test(state.question)) && !isMini);
  const includePastLife = shouldRun('past_life', (executionPlan.family === 'past_life' || intent.flags.includes('past_life') || /\b(past life|past-life|karma|karmic|reincarnation|soul purpose)\b/i.test(state.question)) && !isMini);
  const includePregnancyFertility = shouldRun('pregnancy_fertility', (executionPlan.family === 'pregnancy_fertility' || intent.flags.includes('pregnancy_fertility') || /\b(pregnancy|fertility|conceive|conception|childbirth|delivery|baby)\b/i.test(state.question)) && !isMini);
  const includeLegal = shouldRun('legal', (executionPlan.family === 'legal' || intent.flags.includes('legal') || /\b(legal|court|litigation|lawsuit|dispute|case)\b/i.test(state.question)) && !isMini);
  const includeFinance = shouldRun('finance', (executionPlan.family === 'finance' || intent.topics.includes('finance') || /\b(wealth|money|income|finance|assets|property|investment|profits|revenue)\b/i.test(state.question)) && !isMini);
  const includeHealth = shouldRun('health', (executionPlan.family === 'health' || intent.topics.includes('health') || /\b(health|disease|illness|medical|recovery|fitness|stress)\b/i.test(state.question)) && !isMini);
  const includeEducation = shouldRun('education', (executionPlan.family === 'education' || intent.topics.includes('education') || /\b(education|study|studies|exam|exams|degree|college|school|learning|research)\b/i.test(state.question)) && !isMini);
  const includeChildren = shouldRun('children', (executionPlan.family === 'children' || intent.topics.includes('children') || /\b(children|child|kids|pregnancy|pregnant)\b/i.test(state.question)) && !isMini);
  const includeProperty = shouldRun('property', (executionPlan.family === 'property' || intent.topics.includes('property') || /\b(property|house|home|land|real estate|vehicle|car|asset)\b/i.test(state.question)) && !isMini);
  const includeTravel = shouldRun('travel', (executionPlan.family === 'travel' || intent.topics.includes('travel') || /\b(travel|traveling|foreign|abroad|visa|relocation|migration|move)\b/i.test(state.question)) && !isMini);
  const includeSpirituality = shouldRun('spirituality', (executionPlan.family === 'spirituality' || intent.topics.includes('spirituality') || /\b(spiritual|spirituality|moksha|meditation|religion|faith|guru)\b/i.test(state.question)) && !isMini);
  const includeLongevity = shouldRun('longevity', (executionPlan.family === 'longevity' || Boolean(scopeSelection?.needsLongevity) || intent.flags.includes('longevity') || /\b(longevity|lifespan|life span|how long will i live|length of life|ayush|ayu|mrityu|death|end of life)\b/i.test(state.question)) && !isMini);
  const includeArudha = shouldRun('arudha', /\b(arudha|aruda)\b/i.test(state.question) && !isMini);
  const includeGeneralGrounding = shouldRun('general_grounding', true);
  const includeVarga = shouldRun('varga', true);
  const includePlacement = shouldRun('placement', true);
  const includeReferenceTime = shouldRun('reference_time', true);
  const includeNakshatraLord = shouldRun(
    'nakshatra_lord',
    executionPlan.includeMicroSignals.includes('nakshatra') || executionPlan.includeMicroSignals.includes('nakshatra_lord') || executionPlan.includeMicroSignals.includes('sign_lord')
  );
  const includeDrishtiDegree = shouldRun('drishti_degree', executionPlan.includeMicroSignals.includes('drishti') || executionPlan.includeMicroSignals.includes('degree'));

  const groupsToRun: ToolGroupKey[] = [
    includeReferenceTime ? 'reference_time' : null,
    includeGeneralGrounding ? 'general_grounding' : null,
    includeVarga ? 'varga' : null,
    includeArudha ? 'arudha' : null,
    includePlacement ? 'placement' : null,
    includePanchanga ? 'panchanga' : null,
    includeFeature ? 'feature' : null,
    includeDasha ? 'dasha' : null,
    includeTransit ? 'transit' : null,
    includeCareer ? 'career' : null,
    includeRemedies ? 'remedies' : null,
    includeRelocation ? 'relocation' : null,
    includePastLife ? 'past_life' : null,
    includePregnancyFertility ? 'pregnancy_fertility' : null,
    includeLegal ? 'legal' : null,
    includeFinance ? 'finance' : null,
    includeHealth ? 'health' : null,
    includeEducation ? 'education' : null,
    includeChildren ? 'children' : null,
    includeProperty ? 'property' : null,
    includeTravel ? 'travel' : null,
    includeSpirituality ? 'spirituality' : null,
    includeLongevity ? 'longevity' : null,
    includeNakshatraLord ? 'nakshatra_lord' : null,
    includeDrishtiDegree ? 'drishti_degree' : null,
  ].filter((item): item is ToolGroupKey => Boolean(item));

  const registry = createToolTaskRegistry({
    grounding,
    question: state.question,
    mode: state.mode,
    intent,
    executionPlan,
    scopeSelection,
    analysisTimestamp,
  });

  const findings = mergeFindings(state.toolFindings ?? [], await executeToolGroups(groupsToRun, registry));

  return {
    toolFindings: findings,
    analysisStages: appendStage(state, 'run_general_tools', 'Running general analyzers', `Executed groups=${groupsToRun.join(', ') || 'none'}; findings=${findings.length}`),
  };
}

async function evidenceGateNode(state: AgentStateType): Promise<AgentUpdateType> {
  const deterministicGaps = determineCoverageGaps(state);
  const gapCount = deterministicGaps.length;
  const iteration = state.toolIteration ?? 0;
  const maxIterations = state.maxToolIterations ?? 2;
  const deterministicShouldRetry = gapCount > 0;
  const deterministicAction: 'refine_tools' | 'build_prompt' = deterministicShouldRetry && iteration < maxIterations
    ? 'refine_tools'
    : 'build_prompt';

  // Hard stop guardrail always wins.
  if (iteration >= maxIterations || gapCount === 0) {
    const shouldRetry = gapCount > 0 && iteration < maxIterations;
    return {
      coverageGaps: deterministicGaps,
      coverageShouldRetry: shouldRetry,
      coverageDecisionConfidence: 1,
      refinementNextAction: 'build_prompt',
      decisionTelemetry: appendDecisionTelemetry(state, {
        node: 'evidence_gate',
        model: 'evidence-hard-stop',
        latencyMs: 0,
        confidence: 1,
        usedFallback: true,
        fallbackReason: 'hard iteration/gap guardrail',
      }),
      analysisStages: appendStage(
        state,
        'evidence_gate',
        'Evaluating evidence and deciding refinement',
        `nextAction=build_prompt; reason=hard-stop; iteration=${iteration}/${maxIterations}; gaps=${gapCount}`
      ),
    };
  }

  const currentFindings = (state.toolFindings ?? []).map((finding) => ({
    name: finding.name,
    status: finding.status,
    factCount: finding.facts.length,
  }));

  const coverageEnabled = isDecisionNodeEnabled(env.LLM_DECISION_COVERAGE_ENABLED);
  const refinementEnabled = isDecisionNodeEnabled(env.LLM_DECISION_REFINEMENT_ROUTER_ENABLED);
  if (!coverageEnabled && !refinementEnabled) {
    return {
      coverageGaps: deterministicGaps,
      coverageShouldRetry: deterministicShouldRetry,
      coverageDecisionConfidence: 0.25,
      refinementNextAction: deterministicAction,
      decisionTelemetry: appendDecisionTelemetry(state, {
        node: 'evidence_gate',
        model: 'evidence-gate-disabled',
        latencyMs: 0,
        confidence: 0.25,
        usedFallback: true,
        fallbackReason: 'LLM_DECISION_COVERAGE_ENABLED=false and LLM_DECISION_REFINEMENT_ROUTER_ENABLED=false',
      }),
      analysisStages: appendStage(
        state,
        'evidence_gate',
        'Evaluating evidence and deciding refinement',
        `nextAction=${deterministicAction}; gaps=${deterministicGaps.join(',') || 'none'}; model=evidence-gate-disabled; iteration=${iteration}/${maxIterations}`
      ),
    };
  }

  const result = await invokeDecisionNode<EvidenceGateDecision>({
    node: 'evidence_gate',
    schema: EvidenceGateDecisionSchema,
    input: {
      question: state.question,
      mode: state.mode,
      executionPlan: state.executionPlan,
      deterministicGaps,
      deterministicShouldRetry,
      deterministicAction,
      currentIteration: iteration,
      maxIterations,
      findings: currentFindings,
      instruction:
        'Evaluate evidence coverage and choose next action. Use refine_tools only if critical gaps remain and another iteration is likely to improve evidence. Respect deterministic hints when uncertain.',
    },
    fallback: () => ({
      sufficientCoverageAchieved: deterministicGaps.length === 0,
      gapsIdentified: deterministicGaps,
      shouldRetry: deterministicShouldRetry,
      nextAction: deterministicAction,
      confidence: 0.35,
    }),
  });

  const coverageGaps = clampCoverageGaps(result.decision.gapsIdentified, deterministicGaps);
  const shouldRetry = coverageGaps.length > 0 && result.decision.shouldRetry && iteration < maxIterations;
  const nextAction = shouldRetry && result.decision.nextAction === 'refine_tools' ? 'refine_tools' : 'build_prompt';
  const shadowComparison = env.LLM_DECISION_SHADOW_MODE
    ? `deterministic=${deterministicAction}; llm=${nextAction}; match=${deterministicAction === nextAction}`
    : undefined;

  return {
    coverageGaps,
    coverageShouldRetry: shouldRetry,
    coverageDecisionConfidence: result.decision.confidence,
    refinementNextAction: nextAction,
    decisionTelemetry: appendDecisionTelemetry(state, {
      node: 'evidence_gate',
      model: result.model,
      latencyMs: result.latencyMs,
      confidence: result.decision.confidence,
      usedFallback: result.usedFallback,
      fallbackReason: result.usedFallback ? 'evidence gate fallback' : undefined,
      shadowComparison,
    }),
    analysisStages: appendStage(
      state,
      'evidence_gate',
      'Evaluating evidence and deciding refinement',
      `nextAction=${nextAction}; gaps=${coverageGaps.join(',') || 'none'}; shouldRetry=${shouldRetry}; model=${result.model}; latencyMs=${result.latencyMs}; confidence=${result.decision.confidence.toFixed(2)}; iteration=${iteration}/${maxIterations}`
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
        tasks.push(
          buildDashaToolFindingCached({
            rawPayload: grounding.rawPayload,
            referenceTimestamp: analysisTimestamp,
            ownerId: grounding.ownerId,
            profileId: grounding.profileId,
            payloadHash: grounding.payloadHash,
            question: state.question,
            kundli: grounding.kundli,
          })
        );
        break;
      case 'transit':
        tasks.push(
          resolveTransitRequestKind(state.question, grounding.referenceTimestamp) === 'range'
            ? buildTransitIntervalToolFinding({
                kundli: grounding.kundli,
                question: state.question,
                referenceTimestamp: grounding.referenceTimestamp,
              })
            : buildTransitPointToolFinding({
                kundli: grounding.kundli,
                question: state.question,
                referenceTimestamp: grounding.referenceTimestamp,
              })
        );
        break;
      case 'career':
        tasks.push(Promise.resolve(makeCareerToolFinding(grounding.rawPayload, state.question, analysisTimestamp)));
        break;
      case 'remedies':
        tasks.push(Promise.resolve(makeRemediesToolFinding(grounding.rawPayload, analysisTimestamp, state.question)));
        break;
      case 'relocation':
        tasks.push(Promise.resolve(makeRelocationToolFinding(grounding.rawPayload, analysisTimestamp)));
        break;
      case 'past_life':
        tasks.push(Promise.resolve(makePastLifeToolFinding(grounding.rawPayload, analysisTimestamp)));
        break;
      case 'pregnancy_fertility':
        tasks.push(Promise.resolve(makePregnancyFertilityToolFinding(grounding.rawPayload, analysisTimestamp)));
        break;
      case 'legal':
        tasks.push(Promise.resolve(makeLegalToolFinding(grounding.rawPayload, analysisTimestamp)));
        break;
      case 'finance':
        tasks.push(Promise.resolve(makeFinanceToolFinding(grounding.rawPayload, analysisTimestamp)));
        break;
      case 'health':
        tasks.push(Promise.resolve(makeHealthToolFinding(grounding.rawPayload, analysisTimestamp)));
        break;
      case 'education':
        tasks.push(Promise.resolve(makeEducationToolFinding(grounding.rawPayload, analysisTimestamp)));
        break;
      case 'children':
        tasks.push(Promise.resolve(makeChildrenToolFinding(grounding.rawPayload, analysisTimestamp)));
        break;
      case 'property':
        tasks.push(Promise.resolve(makePropertyToolFinding(grounding.rawPayload, analysisTimestamp)));
        break;
      case 'travel':
        tasks.push(Promise.resolve(makeTravelToolFinding(grounding.rawPayload, analysisTimestamp)));
        break;
      case 'spirituality':
        tasks.push(Promise.resolve(makeSpiritualityToolFinding(grounding.rawPayload, analysisTimestamp)));
        break;
      case 'longevity':
        tasks.push(Promise.resolve(makeLongevityToolFinding(grounding.rawPayload, grounding.referenceTimestamp)));
        break;
      default:
        break;
    }
  }

  // Always include one broad grounding pass in refinement to capture missed paths.
  tasks.push(Promise.resolve(makeGeneralToolFinding(grounding.rawPayload, state.question, intent.flags, state.mode, state.scopeSelection ?? null)));

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

function routeAfterEvidenceGate(state: AgentStateType): 'refine_tools' | 'build_prompt' {
  return state.refinementNextAction === 'refine_tools' ? 'refine_tools' : 'build_prompt';
}

function rankFindingsForPrompt(state: AgentStateType, findings: ToolFinding[]): ToolFinding[] {
  const selectedGroups = new Set(state.selectedToolGroups ?? []);
  const q = state.question.toLowerCase();

  const groupToToolName: Partial<Record<ToolGroupKey, string>> = {
    reference_time: 'Reference time analyzer',
    atlas: 'Chart atlas',
    varga: 'Varga analyzer',
    arudha: 'Arudha analyzer',
    d9: 'D9 analyzer',
    dasha: 'Dasha analyzer',
    transit: 'Transit analyzer',
    career: 'Career analyzer',
    remedies: 'Remedies analyzer',
    relocation: 'Relocation analyzer',
    past_life: 'Past-life analyzer',
    pregnancy_fertility: 'Pregnancy/Fertility analyzer',
    legal: 'Legal analyzer',
    finance: 'Finance analyzer',
    health: 'Health analyzer',
    education: 'Education analyzer',
    children: 'Children analyzer',
    property: 'Property analyzer',
    travel: 'Travel analyzer',
    spirituality: 'Spirituality analyzer',
    longevity: 'Longevity analyzer',
    placement: 'Placement analyzer',
    panchanga: 'Panchanga analyzer',
    feature: 'Feature analyzer',
    general_grounding: 'General grounding analyzer',
    nakshatra_lord: 'Nakshatra/lord analyzer',
    drishti_degree: 'Drishti/degree analyzer',
  };

  const selectedToolNames = new Set(
    [...selectedGroups]
      .map((group) => groupToToolName[group])
      .filter((name): name is string => Boolean(name))
  );

  const getScore = (finding: ToolFinding): number => {
    let score = 0;

    if (finding.status === 'ok') score += 4;
    else if (finding.status === 'partial') score += 2;

    if (selectedToolNames.has(finding.name)) score += 6;

    const name = finding.name.toLowerCase();
    if ((/career|job|profession|business|work/.test(q) && /career/.test(name))
      || (/marriage|relationship|partner|spouse|love|compatibility/.test(q) && /d9|varga|placement/.test(name))
      || (/when|timing|dasha|period|timeline/.test(q) && /dasha|reference time/.test(name))
      || (/transit|gochar|today|now/.test(q) && /transit/.test(name))
      || (/longevity|lifespan|ayush|mrityu|death/.test(q) && /longevity|d8|d30|dasha/.test(name))) {
      score += 5;
    }

    score += Math.min(4, Math.floor((finding.facts?.length ?? 0) / 6));
    return score;
  };

  return [...findings].sort((a, b) => getScore(b) - getScore(a));
}

function computeDynamicSnippetBudget(state: AgentStateType, findings: ToolFinding[]): { totalChars: number; perSnippetCap: number } {
  const base = state.mode === 'pro' ? 9000 : 7000;
  const questionPenalty = Math.min(2200, state.question.length * 2);
  const contextPenalty = Math.min(1200, (state.conversationContext?.length ?? 0) * 140);
  const findingBoost = Math.min(3200, findings.length * 280);

  const totalChars = Math.max(4500, Math.min(14000, base + findingBoost - questionPenalty - contextPenalty));
  const perSnippetCap = Math.max(600, Math.min(1800, Math.floor(totalChars / Math.max(6, findings.length * 1.4))));
  return { totalChars, perSnippetCap };
}

function buildPrompt(state: AgentStateType): string {
  const grounding = state.grounding;
  if (!grounding) throw new Error('Grounding context missing while building prompt.');
  const executionPlan = state.executionPlan;
  const mode = state.mode;

  const findings = state.toolFindings ?? [];
  const styleHint = state.responseStyleHint ?? 'brief';
  const rankedFindings = rankFindingsForPrompt(state, findings);
  const maxFindings = mode === 'pro' ? 12 : 8;
  const promptFindings = rankedFindings.slice(0, maxFindings);
  const conversationContext = (state.conversationContext ?? [])
    .map((item) => String(item).trim())
    .filter(Boolean)
    .slice(-8);

  const findingBlock = promptFindings
    .map((finding) => {
      const facts = finding.facts.map((fact) => `- ${fact}`).join('\n');
      const paths = finding.evidencePaths.length ? `Evidence paths: ${finding.evidencePaths.join(', ')}` : 'Evidence paths: none';
      const missing = finding.missing?.length ? `Missing: ${finding.missing.join(', ')}` : '';
      return `TOOL: ${finding.name} [${finding.status}]\n${facts}\n${paths}${missing ? `\n${missing}` : ''}`;
    })
    .join('\n\n');

  const availability = getToolAvailabilityIndex(findings);
  const availableNames = [...availability.available];
  const partialNames = [...availability.partial];
  const unavailableNames = [...availability.unavailable];

  const snippetBudget = computeDynamicSnippetBudget(state, promptFindings);
  let budget = snippetBudget.totalChars;
  let omitted = 0;

  const snippets: string[] = [];
  for (const finding of promptFindings) {
    for (const snippet of finding.snippets ?? []) {
      if (budget <= 0) {
        omitted += 1;
        continue;
      }
      const piece = truncateText(snippet, Math.min(snippetBudget.perSnippetCap, budget));
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

  // Mode-specific instructions
  const modeInstructions =
    mode === 'mini'
      ? [
          'MODE: MINI - Basic Astrological Insights',
          'In mini mode, focus insights on D1 (Rashi) and D9 (Navamsha) charts only.',
          'Provide foundational interpretations only for allowed mini scope.',
          'Never provide timing, forecast, prediction, transit/gochar, D10, D8, D30, dasha, longevity, arudha, or other pro-only analysis in mini mode.',
          'If a pro-only analysis is requested, respond with one short deny line and Pro upgrade direction.',
          'Keep guidance practical and accessible for users new to astrology.',
        ]
      : [
          'MODE: PRO - Comprehensive Astrological Analysis',
          'In pro mode, provide thorough analysis using all relevant divisional charts.',
          'Analyze D1, D9, D10, D8, D30, D7, D4, D12, D20 as needed for complete insights.',
          'Include advanced techniques: dasha periods, transits, yoga formations, micro-signals.',
          'Synthesize multiple layers of evidence for nuanced, multi-dimensional predictions.',
        ];

  return [
    'You are Cozmic AI, a Vedic astrology assistant grounded in canonical JSON payload data.',
    'Brand identity rule: if asked "who are you" / "who built you", state clearly: "I am Cozmic AI."',
    'Your goal is to analyze personal birth charts and provide accurate, evidence-based astrological insights.',
    '',
    ...modeInstructions,
    '',
    'Core analysis guidelines:',
    'Answer only what the user asked. Do not add extra sections unless user asks for details.',
    'Default response length: concise (3-6 lines).',
    `Response style hint: ${styleHint}.`,
    styleHint === 'micro' ? 'Start with a direct yes/no first line when applicable.' : 'Use direct first-line answer, then minimal supporting detail.',
    'Use the tool findings first; they are deterministic extracts from the JSON blob.',
    'Grounding contract: every material claim must be grounded in one or more tool findings listed below.',
    'Career contract: India government/private career split is heuristic suitability guidance, not a deterministic guarantee.',
    'Grounding contract: do NOT claim data is missing when that tool is marked ok/partial.',
    'Grounding contract: only mention missing data if the related tool is marked unavailable.',
    'Never ask user for raw birth details (date of birth, birth time, birthplace). Use the Kundli context already provided.',
    'If personal chart context is missing, ask the user to open or generate a Kundli instead of requesting birth details.',
    'Hardship-response contract: for user pain reports (betrayal, heartbreak, bad phase, grief, loss), start with one empathetic line and avoid dismissive openings like "No".',
    'Past-event contract: if user provides an explicit month/year (for example, September 2025), analyze that exact window using transit/dasha evidence before concluding.',
    'Never invent chart facts.',
    'For transit questions, if Transit analyzer is ok/partial, treat backend transit data as available even if canonical raw payload lacks chart.transit.',
    'Only claim transit data is missing when Transit analyzer is unavailable.',
    'If user asks for "current transit details", treat it as a current snapshot request anchored to reference timestamp; do not redirect to a separate transit feature.',
    'Transit mapping contract: determine transit houses relative to transit ascendant (Lagna) using whole-sign mapping (house = ((planet_rashi - lagna_rashi + 12) % 12) + 1). Never present backend house_number as user-facing transit house truth.',
    'Transit wording contract: do not write "from natal" for transit houses unless natal/transit comparative evidence is explicitly present in tool findings.',
    'Rashi consistency contract: if you mention sign names, they must match the rashi number exactly (1 Aries, 2 Taurus, 3 Gemini, 4 Cancer, 5 Leo, 6 Virgo, 7 Libra, 8 Scorpio, 9 Sagittarius, 10 Capricorn, 11 Aquarius, 12 Pisces); if uncertain, report rashi number only.',
    'For transit point questions without an explicit future/past range, anchor interpretation to the transit reference timestamp from tool findings (current context).',
    'Do not shift planet ingress dates into the future unless the user explicitly asked for a future window.',
    'For compound questions, separate the topic, the time window, and the chart layer before answering.',
    'For varga requests, prefer the specific Dxx chart named by the user and fall back to D1 only when needed.',
    'For dasha requests, report the active chain only when exact period boundaries are available, and include the current timestamp used.',
    'For transit requests, call the backend transit endpoint directly and report the requested forecast window.',
    'Never say transit is unavailable because canonical chart data lacks chart.transit; live transit is fetched on demand from the backend analyzer.',
    'Never use phrases like "canonical chart data", "Transit analyzer tool is required", or "dedicated transit forecast feature" in the user-facing answer.',
    'For timing-related questions, always synthesize: reference time context + current natal baseline (D1) + relevant varga chart(s) + dasha timeline + transit window + domain analyzer findings + chart atlas/info sections.',
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
    `userMode: ${mode}`,
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
    `Tool availability summary: ok=[${availableNames.join(', ') || 'none'}]; partial=[${partialNames.join(', ') || 'none'}]; unavailable=[${unavailableNames.join(', ') || 'none'}]`,
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

function shouldUseRemedySpecialist(state: AgentStateType): boolean {
  if (state.mode === 'mini') {
    return false;
  }

  const q = state.question.toLowerCase();
  const family = state.executionPlan?.family;
  const flags = new Set(state.intent?.flags ?? []);

  if (family === 'remedies') {
    return true;
  }

  if (flags.has('remedies') || flags.has('hardship')) {
    return true;
  }

  if (/\b(remedy|remedies|upay|upaya|mantra|gemstone|puja|pooja|fasting|cheated|betray(?:ed|al)|heartbreak|breakup|separation|bad\s+time|difficult\s+time|hard\s+time|rough\s+phase|loss|grief|depressed|depression|anxiety)\b/.test(q)) {
    return true;
  }

  return (state.toolFindings ?? []).some((finding) => finding.name === 'Remedies analyzer' && finding.status !== 'unavailable');
}

function routeAfterPrompt(state: AgentStateType): 'answer_with_remedy_specialist' | 'answer_with_deepseek' {
  return shouldUseRemedySpecialist(state) ? 'answer_with_remedy_specialist' : 'answer_with_deepseek';
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

function getToolAvailabilityIndex(findings: ToolFinding[]): {
  available: Set<string>;
  partial: Set<string>;
  unavailable: Set<string>;
} {
  const available = new Set<string>();
  const partial = new Set<string>();
  const unavailable = new Set<string>();

  for (const finding of findings) {
    if (finding.status === 'ok') {
      available.add(finding.name);
    } else if (finding.status === 'partial') {
      partial.add(finding.name);
    } else {
      unavailable.add(finding.name);
    }
  }

  return { available, partial, unavailable };
}

export function sanitizeMissingDataContradictions(answer: string, findings: ToolFinding[]): string {
  const { available, partial } = getToolAvailabilityIndex(findings);
  const hasData = (name: string) => available.has(name) || partial.has(name);
  const transitHasData = hasData('Transit analyzer');
  const transitUnavailable = findings.some((f) => f.name === 'Transit analyzer' && f.status === 'unavailable');

  const containsTransitToolsetTemplate = (text: string): boolean => (
    /\btransit\s+analy[sz]er\s+tool\s+is\s+unavailable\b/i.test(text)
    || /\bcurrent\s+transit\s+details?\s+are\s+not\s+available\b/i.test(text)
    || /\bcurrent\s+transit\s+details?\s+require\s+a\s+live\s+transit\s+forecast\b/i.test(text)
    || /\bcurrent\s+transit\s+details?\s+cannot\s+be\s+provided\b/i.test(text)
    || /\bcanonical\s+chart\s+data\b/i.test(text)
    || /\bstatic\s+chart\s+data\b/i.test(text)
    || /\blive\s+forecast\s+data\b/i.test(text)
    || /\blive\s+transit\s+forecast\b/i.test(text)
    || /\bthe\s+transit\s+analy[sz]er\s+tool\s+is\s+required\s+for\s+this\s+forecast\b/i.test(text)
    || /\bavailable\s+tools?\s+only\s+contain\s+your\s+natal\s+chart\s+data\b/i.test(text)
    || /\bask\s+a\s+specific\s+timing\s+question\b/i.test(text)
    || /\bnatal\s+chart\s+and\s+dasha\s+periods\b/i.test(text)
    || /\bplease\s+try\s+again\s+or\s+use\s+the\s+app'?s\s+transit\s+feature\b/i.test(text)
    || /\breal[-\s]?time\s+snapshot\b/i.test(text)
    || /\bnot\s+supported\s+by\s+the\s+current\s+toolset\b/i.test(text)
    || /\bfor\s+transit\s+analysis,\s+please\s+use\s+a\s+dedicated\s+transit\s+forecast\s+feature\b/i.test(text)
    || /\breference\s+time\s+for\s+your\s+chart\s+is\s+[0-9TZ:.-]+\b/i.test(text)
  );

  let next = answer;

  const removeIfHasData = (toolName: string, patterns: RegExp[]) => {
    if (!hasData(toolName)) return;
    for (const pattern of patterns) {
      next = next.replace(pattern, '');
    }
  };

  removeIfHasData('Dasha analyzer', [
    /\b(?:no|missing|unavailable)\s+dasha(?:\s+data)?\b[^.]*\.?/gi,
    /\b(?:cannot|can't|unable to)\s+(?:analy[sz]e|determine|predict)\b[^.]*\bdasha\b[^.]*\.?/gi,
  ]);

  removeIfHasData('Transit analyzer', [
    /\b(?:no|missing|unavailable)\s+(?:transit|gochar)(?:\s+data)?\b[^.]*\.?/gi,
    /\b(?:cannot|can't|unable to)\s+(?:analy[sz]e|determine|compute)\b[^.]*\b(?:transit|gochar)\b[^.]*\.?/gi,
    /\byour\s+current\s+transit\s+details?\s+are\s+not\s+available\b[^.]*\.?/gi,
    /\byour\s+current\s+transit\s+details?\s+require\s+a\s+live\s+transit\s+forecast\b[^.]*\.?/gi,
    /\byour\s+current\s+transit\s+details?\s+cannot\s+be\s+provided\b[^.]*\.?/gi,
    /\bcurrent\s+transit\s+details?\s+cannot\s+be\s+provided\b[^.]*\.?/gi,
    /\byour\s+current\s+transit\s+details?\s+are\s+not\s+available\s+in\s+the\s+canonical\s+chart\s+data\b[^.]*\.?/gi,
    /\bi\s+cannot\s+fetch\b[^.]*\bstatic\s+chart\s+data\b[^.]*\.?/gi,
    /\bstatic\s+chart\s+data\b[^.]*\.?/gi,
    /\bprovided\s+chart\s+data\b[^.]*\bdoes\s+not\s+include\b[^.]*\btransit\b[^.]*\.?/gi,
    /\bthe\s+transit\s+analy[sz]er\s+tool\s+is\s+required\s+for\s+this\s+forecast\b[^.]*\.?/gi,
    /\bthe\s+transit\s+analy[sz]er\s+tool\s+is\s+currently\s+unavailable\b[^.]*\.?/gi,
    /\bto\s+get\s+a\s+transit\s+report\b[^.]*\.?/gi,
    /\bplease\s+use\s+the\s+dedicated\s+transit\s+forecast\s+feature\s+in\s+the\s+app\b[^.]*\.?/gi,
    /\bplease\s+try\s+again\s+or\s+use\s+the\s+app'?s\s+transit\s+feature\b[^.]*\.?/gi,
    /\breal[-\s]?time\s+snapshot\b[^.]*\.?/gi,
    /\bor\s+ask\s+a\s+specific\s+timing\s+question\b[^.]*\.?/gi,
    /\bi\s+can\s+analy[sz]e\s+it\s+using\s+your\s+natal\s+chart\s+and\s+dasha\s+periods\b[^.]*\.?/gi,
    /\btransit\s+analy[sz]er\s+tool\s+is\s+unavailable\b[^.]*\.?/gi,
    /\bthe\s+available\s+tools?\s+only\s+contain\s+your\s+natal\s+chart\s+data\b[^.]*\.?/gi,
    /\bnot\s+supported\s+by\s+the\s+current\s+toolset\b[^.]*\.?/gi,
    /\breference\s+time\s+for\s+your\s+chart\s+is\s+[0-9TZ:.-]+\b[^.]*\.?/gi,
    /\bfor\s+transit\s+analysis,\s+please\s+use\s+a\s+dedicated\s+transit\s+forecast\s+feature\b[^.]*\.?/gi,
  ]);

  // Replace hostile-template wording with user-safe fallback language.
  // This also guards non-pipeline/fast-answer branches where tool findings can be absent.
  const looksLikeToolsetTemplate = containsTransitToolsetTemplate(next);
  if (looksLikeToolsetTemplate) {
    next = next
      .replace(/\b(?:i\s+cannot|i\s+can'?t|i\s+am\s+unable\s+to)\b[^\n]*\btransit\b[^\n]*\.?/gi, '')
      .replace(/\byour\s+current\s+transit\s+details?\s+require\s+a\s+live\s+transit\s+forecast\b[^\n]*\.?/gi, '')
      .replace(/\b(?:your\s+)?current\s+transit\s+details?\s+cannot\s+be\s+provided\b[^\n]*[.,!?]?/gi, '')
      .replace(/\bcannot\s+be\s+provided\s+because\s+a\b[^\n]*[.,!?]?/gi, '')
      .replace(/\bi\s+cannot\s+fetch\b[^\n]*\bstatic\s+chart\s+data\b[^\n]*\.?/gi, '')
      .replace(/\byour\s+current\s+transit\s+details?\s+are\s+not\s+available[^\n]*\.?/gi, '')
      .replace(/\bcurrent\s+transit\s+details?\s+are\s+not\s+available[^\n]*\.?/gi, '')
      .replace(/\bthe\s+transit\s+analy[sz]er\s+tool\s+is\s+required\s+for\s+this\s+forecast[^\n]*\.?/gi, '')
      .replace(/\bthe\s+transit\s+analy[sz]er\s+tool\s+is\s+currently\s+unavailable[^\n]*\.?/gi, '')
      .replace(/\bthe\s+available\s+tools?\s+only\s+contain\s+your\s+natal\s+chart\s+data[^\n]*\.?/gi, '')
      .replace(/\bcanonical\s+chart\s+data[^\n]*\.?/gi, '')
      .replace(/\bstatic\s+chart\s+data[^\n]*\.?/gi, '')
      .replace(/\blive\s+forecast\s+data[^\n]*\.?/gi, '')
      .replace(/\blive\s+transit\s+forecast[^\n]*\.?/gi, '')
      .replace(/\bfor\s+transit\s+analysis,\s+please\s+use\s+a\s+dedicated\s+transit\s+forecast\s+feature[^\n]*\.?/gi, '')
      .replace(/\bplease\s+use\s+the\s+dedicated\s+transit\s+forecast\s+feature\s+in\s+the\s+app[^\n]*\.?/gi, '')
      .replace(/\bplease\s+try\s+again\s+or\s+use\s+the\s+app'?s\s+transit\s+feature[^\n]*\.?/gi, '')
      .replace(/\breal[-\s]?time\s+snapshot[^\n]*\.?/gi, '')
      .replace(/\bdedicated\s+transit\s+forecast\s+feature[^\n]*\.?/gi, '')
      .replace(/\bor\s+ask\s+a\s+specific\s+timing\s+question[^\n]*\.?/gi, '')
      .replace(/\bask\s+a\s+specific\s+timing\s+question[^\n]*\.?/gi, '')
      .replace(/\bi\s+can\s+analy[sz]e\s+it\s+using\s+your\s+natal\s+chart\s+and\s+dasha\s+periods[^\n]*\.?/gi, '')
      .replace(/\bto\s+get\s+(?:your\s+)?(?:current\s+)?transit\s+details?[^\n]*[.,!?]?/gi, '')
      .replace(/^\s*i\s+am\s+cozmic\s+ai\.?\s*$/gmi, '')
      .replace(/\bnot\s+supported\s+by\s+the\s+current\s+toolset[^\n]*\.?/gi, '')
      .replace(/\breference\s+time\s+for\s+your\s+chart\s+is\s+[0-9TZ:.-]+[^\n]*\.?/gi, '')
      .replace(/^\s*(Your|The|And)\s*$/gmi, '')
      .replace(/[ \t]+\n/g, '\n')
      .replace(/\n{3,}/g, '\n\n')
      .trim();

    const unavailableFallback = 'I could not fetch live transit from the backend right now. Please retry this question, and I will compute transit from your saved Kundli context.';
    const availableFallback = 'I can compute transit using your saved Kundli context. Ask for today, this month, this year, or an exact date/time.';
    const fallback = transitUnavailable || !transitHasData ? unavailableFallback : availableFallback;
    const identityOnly = /^i\s+am\s+cozmic\s+ai\.?$/i.test(next);

    if (!next || identityOnly) {
      next = fallback;
    } else if (!/could\s+not\s+fetch\s+live\s+transit|i\s+can\s+compute\s+transit\s+using\s+your\s+saved\s+kundli\s+context/i.test(next)) {
      next = `${next}\n\n${fallback}`;
    }
  }

  removeIfHasData('D9 analyzer', [
    /\b(?:no|missing|unavailable)\s+(?:d9|navamsha|navamsa)(?:\s+data)?\b[^.]*\.?/gi,
    /\b(?:need|requires?)\b[^.]*\b(?:d9|navamsha|navamsa)\b[^.]*\b(?:for|to)\b[^.]*\.?/gi,
  ]);

  removeIfHasData('Career analyzer', [
    /\b(?:no|missing|unavailable)\s+(?:d10|career chart|career data)\b[^.]*\.?/gi,
    /\b(?:cannot|can't|unable to)\s+(?:analy[sz]e|assess|evaluate)\b[^.]*\bcareer\b[^.]*\.?/gi,
  ]);

  removeIfHasData('Longevity analyzer', [
    /\b(?:cannot|can't|unable to)\s+(?:analy[sz]e|assess|evaluate)\b[^.]*\blongevity\b[^.]*\.?/gi,
    /\b(?:need|requires?)\b[^.]*\b(?:d8|d30)\b[^.]*\blongevity\b[^.]*\.?/gi,
  ]);

  return next
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

function sanitizeBirthDetailRequests(answer: string): string {
  const original = answer;
  let next = answer;

  const birthDetailRequestPatterns = [
    /\b(?:please\s+)?(?:share|provide|tell|send|enter|give)\b[^.?!\n]*(?:date of birth|dob|birth time|time of birth|birthplace|place of birth|birth details)[^.?!\n]*[.?!]?/gi,
    /\b(?:i\s+need|i(?:'| )?ll\s+need|we\s+need)\b[^.?!\n]*(?:date of birth|dob|birth time|time of birth|birthplace|place of birth|birth details)[^.?!\n]*[.?!]?/gi,
    /\b(?:what(?:'s| is)\s+your|can you share your)\b[^.?!\n]*(?:date of birth|dob|birth time|time of birth|birthplace|place of birth|birth details)[^.?!\n]*[?]?/gi,
  ];

  for (const pattern of birthDetailRequestPatterns) {
    next = next.replace(pattern, '');
  }

  next = next
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();

  if (next === original.trim()) {
    return next;
  }

  const kundliFallback = 'For a personal chart reading, please open or generate a Kundli in the app.';
  if (next.length === 0 && !/open\s+or\s+generate\s+a\s+kundli/i.test(next)) {
    next = kundliFallback;
  }

  return next;
}

function isTransitDetailsQuestion(question: string): boolean {
  const q = question.toLowerCase();
  return /\b(transit|gochar)\b/.test(q)
    && /\b(details?|snapshot|current|full|all|complete)\b/.test(q);
}

function getTransitAnalyzerFinding(findings: ToolFinding[]): ToolFinding | null {
  const finding = findings.find((item) => item.name === 'Transit analyzer' && item.status !== 'unavailable');
  return finding ?? null;
}

function extractTransitPlanetFact(findings: ToolFinding[], planetLabel: string): string | null {
  const transitFinding = getTransitAnalyzerFinding(findings);
  if (!transitFinding) return null;

  const line = transitFinding.facts.find((fact) => new RegExp(`^Transit\\s+${planetLabel}:`, 'i').test(fact));
  return line?.trim() ?? null;
}

function sanitizeTransitProfileGateLeak(answer: string, findings: ToolFinding[], question: string): string {
  if (!isTransitDetailsQuestion(question)) {
    return answer;
  }

  if (!getTransitAnalyzerFinding(findings)) {
    return answer;
  }

  return answer
    .replace(/^\s*For a personal chart reading, please open or generate a Kundli in the app\.\s*$/gim, '')
    .replace(/^\s*To answer this as a personal chart reading, I need your Kundli context first\.\s*$/gim, '')
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

function ensureTransitMercuryMention(answer: string, findings: ToolFinding[], question: string): string {
  if (!isTransitDetailsQuestion(question)) {
    return answer;
  }

  if (/\bMercury\b/i.test(answer)) {
    return answer;
  }

  const mercuryFact = extractTransitPlanetFact(findings, 'Mercury');
  if (!mercuryFact) {
    return answer;
  }

  const trimmed = answer.trim();
  if (!trimmed) {
    return mercuryFact;
  }

  return `${trimmed}\n- ${mercuryFact}`;
}

function isHardshipQuestion(question: string): boolean {
  const q = question.toLowerCase();
  return /\b(bad time|difficult time|hard time|rough phase|cheated|betray(ed|al)|heartbreak|breakup|abandon(ed|ment)|loss|grief|depressed|depression|anxiety|panic|suffering|stressed|stressful|why is this happening|why me)\b/.test(q);
}

function sanitizeHardshipTone(answer: string, question: string): string {
  if (!question || !isHardshipQuestion(question)) {
    return answer;
  }

  let next = answer.trim();
  next = next.replace(
    /the\s+chart\s+does\s+not\s+show\s+a\s+definitive\s+astrological\s+signature[^.]*\./i,
    'Charts cannot prove a single event with certainty, but they can highlight stress patterns and timing windows.'
  );

  if (/^no[,\.\s]/i.test(next)) {
    next = next.replace(/^no[,\.\s]*/i, '');
  }

  if (!/^i['’]m\s+sorry\s+you\s+(went\s+through|are\s+facing)/i.test(next)) {
    next = `I’m sorry you went through this.\n\n${next}`;
  }

  return next
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

export function enforceGroundingAnswerContract(answer: string, findings: ToolFinding[], question = ''): string {
  const step0 = sanitizeBirthDetailRequests(answer);
  const step1 = sanitizeGenericMissingAnalysisClaims(step0, findings);
  const step2 = sanitizeMissingDataContradictions(step1, findings);
  const step3 = sanitizeTransitProfileGateLeak(step2, findings, question);
  const step4 = ensureTransitMercuryMention(step3, findings, question);
  const step5 = sanitizeHardshipTone(step4, question);
  return step5;
}

function buildKendraSignsFromLagna(lagnaRashi: number): { h1: number; h4: number; h7: number; h10: number } {
  const normalize = (n: number): number => ((n - 1 + 12) % 12) + 1;
  return {
    h1: normalize(lagnaRashi),
    h4: normalize(lagnaRashi + 3),
    h7: normalize(lagnaRashi + 6),
    h10: normalize(lagnaRashi + 9),
  };
}

function maybeBuildDeterministicCurrentTransitAnswer(
  question: string,
  findings: ToolFinding[],
  referenceTimestamp: number
): string | null {
  if (!/\b(transit|gochar)\b/i.test(question)) return null;
  if (resolveTransitRequestKind(question, referenceTimestamp) !== 'current') return null;

  const transitFinding = findings.find((finding) => finding.name === 'Transit analyzer' && finding.status !== 'unavailable');
  if (!transitFinding) return null;

  const lagnaLine = transitFinding.facts.find((fact) => /^Transit Lagna:/i.test(fact));
  const lagnaMatch = lagnaLine?.match(/rashi\s+(\d+)(?:,\s*([0-9]+(?:\.[0-9]+)?)°)?/i);
  const lagnaRashi = lagnaMatch ? Number(lagnaMatch[1]) : null;
  const lagnaDegree = lagnaMatch?.[2] ?? null;

  const referenceLine = transitFinding.facts.find((fact) => /^Transit reference timestamp used:/i.test(fact));
  const referenceIso = referenceLine?.match(/used:\s*([0-9T:.\-Z]+)/i)?.[1] ?? null;

  const preferredOrder = ['Sun', 'Moon', 'Mars', 'Mercury', 'Jupiter', 'Venus', 'Saturn', 'Rahu', 'Ketu'];
  const parsedPlanets: Array<{ name: string; rashi: number; house: number }> = [];

  for (const fact of transitFinding.facts) {
    const m = fact.match(/^Transit\s+(Sun|Moon|Mars|Mercury|Jupiter|Venus|Saturn|Rahu|Ketu):\s*rashi\s+(\d+),\s*house\s+(\d+)\s*\(ascendant-relative\)/i);
    if (!m) continue;
    parsedPlanets.push({
      name: m[1],
      rashi: Number(m[2]),
      house: Number(m[3]),
    });
  }

  if (parsedPlanets.length === 0) return null;

  const byOrder = [...parsedPlanets].sort((a, b) => preferredOrder.indexOf(a.name) - preferredOrder.indexOf(b.name));
  const headlinePlanets = byOrder.slice(0, 6);

  const lines: string[] = [];
  lines.push('I am Cozmic AI. Here are your current transit details (ascendant-relative houses):');
  if (referenceIso) {
    lines.push(`Reference time: ${referenceIso}.`);
  }

  if (lagnaRashi !== null) {
    lines.push(`Transit Lagna: rashi ${lagnaRashi}${lagnaDegree ? ` at ${lagnaDegree}°` : ''}.`);
    const kendra = buildKendraSignsFromLagna(lagnaRashi);
    lines.push(`Derived house-sign axis (whole-sign): 1st=rashi ${kendra.h1}, 4th=rashi ${kendra.h4}, 7th=rashi ${kendra.h7}, 10th=rashi ${kendra.h10}.`);
  }

  for (const planet of headlinePlanets) {
    lines.push(`${planet.name}: rashi ${planet.rashi}, house ${planet.house}.`);
  }

  lines.push('House mapping note: houses are computed from transit Lagna with whole-sign mapping, not backend house_number metadata.');
  return lines.join('\n');
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
  if (state.mode === 'pro' && hasExplicitTransitCue(state.question)) {
    return 'run_specialized_tools';
  }

  const selected = new Set(state.selectedToolGroups ?? []);
  if (selected.size > 0) {
    const specialized = ['d9', 'dasha', 'transit', 'career', 'remedies', 'relocation', 'past_life', 'pregnancy_fertility', 'legal', 'finance', 'health', 'education', 'children', 'property', 'travel', 'spirituality', 'longevity', 'feature', 'arudha'].some((group) => selected.has(group as ToolGroupKey));
    return specialized ? 'run_specialized_tools' : 'run_general_tools';
  }

  if (state.mode === 'mini') {
    return 'run_general_tools';
  }

  if (state.executionPlan && state.executionPlan.family !== 'general') {
    return 'run_specialized_tools';
  }

  const primary = state.intent?.primary ?? 'general';
  return primary === 'general' ? 'run_general_tools' : 'run_specialized_tools';
}

async function answerWithRemedySpecialistNode(state: AgentStateType): Promise<AgentUpdateType> {
  const grounding = state.grounding;
  if (!grounding) {
    throw new Error('Grounding context missing; loadCanonicalGrounding must run first.');
  }

  if (!state.prompt) {
    throw new Error('Prompt missing before remedy specialist call.');
  }

  try {
    const findings = state.toolFindings ?? [];
    const availability = getToolAvailabilityIndex(findings);
    const styleHint = state.responseStyleHint ?? 'brief';

    const deepSeek = await invokeDeepSeekBedrock({
      systemPrompt: state.prompt,
      userPrompt: [
        'You are Cozmic AI Remedy Specialist node.',
        `Response style: ${styleHint}.`,
        'Deliver high-agency remedy guidance tied to deterministic findings only.',
        'Output contract (in this order): (1) one empathy line when user describes pain/hardship; (2) likely timing/context trigger in one short line; (3) 3-5 remedy actions split between practical and spiritual tracks; (4) nearest supportive and pressure windows from interval transit evidence when available; (5) a 7-day starter plan in concise bullets.',
        'Interval prediction contract: for marriage, relationship, career, business, finance, health, children, property, travel, spirituality, and family questions, translate transit interval evidence into probable windows using month-year wording when possible.',
        'If interval findings include supportive/pressure windows, surface them explicitly and keep claims probabilistic.',
        'Always include one line: These remedies are supportive and probabilistic, not deterministic guarantees.',
        'Do not prescribe harmful actions, fear language, or absolute claims.',
        'Do not ask for date of birth, birth time, or birthplace details in chat responses.',
        'If personal chart context is missing, ask user to open or generate a Kundli.',
        'For transit placements, derive house from transit Lagna and planet rashi with whole-sign mapping; never present backend house_number as user-facing house truth.',
        `Available analyzers: ${[...availability.available, ...availability.partial].join(', ') || 'none'}`,
        `Unavailable analyzers: ${[...availability.unavailable].join(', ') || 'none'}`,
        'Be specific, compassionate, and action-focused.',
      ].join(' '),
    });

    const finalAnswer = enforceGroundingAnswerContract(deepSeek.text, findings, state.question);

    return {
      answer: finalAnswer,
      model: `${deepSeek.model}|remedy-specialist`,
      analysisStages: appendStage(state, 'answer_with_remedy_specialist', 'Generating remedy-specialist grounded response'),
    };
  } catch (error) {
    return {
      answer: `${buildDeterministicFallback(state)}\n\nModel error: ${String(error)}`,
      model: 'remedy-specialist-fallback',
      analysisStages: appendStage(state, 'answer_with_remedy_specialist', 'Generating remedy-specialist grounded response', 'Fell back to deterministic output due to model error.'),
    };
  }
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
    const findings = state.toolFindings ?? [];
    const availability = getToolAvailabilityIndex(findings);
    const styleHint = state.responseStyleHint ?? 'brief';
    const deepSeek = await invokeDeepSeekBedrock({
      systemPrompt: state.prompt,
      userPrompt: [
        'Answer using deterministic tool findings and canonical snippets.',
        `Response style: ${styleHint}.`,
        'Answer only the asked question. Do not add extra sections unless user explicitly asks for details.',
        'If question is yes/no, start with a direct yes/no first line when possible.',
        'Contract: every important claim must be grounded in available tool findings.',
        'Contract: India government/private career split should be framed as heuristic suitability guidance (not deterministic outcome).',
        'Contract: do not claim missing dasha/transit/D9/D10/longevity data when related analyzer status is ok/partial.',
        'Contract: if data is missing, name the exact unavailable analyzer/tool and continue with available evidence.',
        'Contract: never ask for date of birth, birth time, or birthplace details in chat responses.',
        'Contract: if personal chart context is missing, ask user to open or generate a Kundli.',
        'Contract: never say transit is unavailable because canonical chart data lacks chart.transit; if transit fetch fails, say live transit could not be fetched right now and ask to retry.',
        'Contract: if user asks for current transit details, treat it as a current snapshot request and do not redirect to any separate feature.',
        'Contract: for transit placements, derive house from transit Lagna and planet rashi with whole-sign mapping; never present backend house_number as user-facing house truth.',
        'Contract: if the user reports a painful experience (cheating, betrayal, heartbreak, bad phase), begin with one empathetic sentence and avoid dismissive openings.',
        'Contract: if the user references a specific month/year in the past, use that exact window in transit/dasha interpretation before giving a conclusion.',
        'Contract: do not claim "from natal" transit houses unless tool findings explicitly include natal-vs-transit comparative mapping.',
        'Contract: if sign name is mentioned, it must match rashi number exactly; otherwise use numeric rashi only.',
        'Contract: if transit findings indicate point/current context, treat that timestamp as present context; do not project future ingress dates unless user requested a future range.',
        `Available analyzers: ${[...availability.available, ...availability.partial].join(', ') || 'none'}`,
        `Unavailable analyzers: ${[...availability.unavailable].join(', ') || 'none'}`,
        'Be decisive, specific, and avoid generic disclaimers.',
      ].join(' '),
    });

    const finalAnswer = enforceGroundingAnswerContract(deepSeek.text, findings, state.question);

    return {
      answer: finalAnswer,
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

async function responsePolicyNode(state: AgentStateType): Promise<AgentUpdateType> {
  if (!state.answer) {
    return {};
  }

  const styleHint = state.responseStyleHint ?? 'brief';
  const contextDepth = state.conversationContext?.length ?? 0;
  const maxChars = styleHint === 'micro'
    ? 320
    : styleHint === 'expand'
      ? 1300
      : contextDepth > 3
        ? 700
        : CONCISE_ANSWER_MAX_CHARS;
  const maxLines = styleHint === 'micro'
    ? 3
    : styleHint === 'expand'
      ? 10
      : contextDepth > 3
        ? 5
        : CONCISE_ANSWER_MIN_LINES;

  const defaultShouldCondense = (() => {
    const raw = state.answer ?? '';
    const lineCount = raw.split('\n').filter(Boolean).length;
    return raw.length > maxChars || lineCount > maxLines;
  })();

  if (!isDecisionNodeEnabled(env.LLM_DECISION_RESPONSE_POLICY_ENABLED)) {
    return {
      responseShouldCondense: defaultShouldCondense,
      responsePolicyTone: 'balanced',
      responsePolicyAddDisclaimer: false,
      responsePolicyConfidence: 0.25,
      decisionTelemetry: appendDecisionTelemetry(state, {
        node: 'response_policy',
        model: 'response-policy-disabled',
        latencyMs: 0,
        confidence: 0.25,
        usedFallback: true,
      }),
      analysisStages: appendStage(
        state,
        'response_policy',
        'Applying response policy',
        `shouldCondense=${defaultShouldCondense}; tone=balanced; model=response-policy-disabled`
      ),
    };
  }

  const result = await invokeDecisionNode<z.infer<typeof ResponsePolicyDecisionSchema>>({
    node: 'response_policy',
    schema: ResponsePolicyDecisionSchema,
    input: {
      question: state.question,
      mode: state.mode,
      responseStyleHint: styleHint,
      contextDepth,
      maxChars,
      maxLines,
      answer: state.answer,
      findings: (state.toolFindings ?? []).map((f) => ({ name: f.name, status: f.status })),
      defaultShouldCondense,
      instruction:
        'Set condensation strategy and confidence tone. Keep response short and directly scoped to user request unless expansion was explicitly asked. Use disclaimer only when key findings are unavailable.',
    },
    fallback: () => ({
      shouldCondense: defaultShouldCondense,
      tone: 'balanced',
      addDisclaimer: false,
      confidence: 0.35,
    }),
  });

  const disclaimer = result.decision.addDisclaimer ? buildDataGapDisclaimer(state.toolFindings ?? []) : null;
  const nextAnswer = disclaimer && state.answer && !state.answer.includes(disclaimer)
    ? `${disclaimer}\n\n${state.answer}`
    : state.answer;

  return {
    answer: nextAnswer,
    responseShouldCondense: result.decision.shouldCondense,
    responsePolicyTone: result.decision.tone,
    responsePolicyAddDisclaimer: result.decision.addDisclaimer,
    responsePolicyConfidence: result.decision.confidence,
    decisionTelemetry: appendDecisionTelemetry(state, {
      node: 'response_policy',
      model: result.model,
      latencyMs: result.latencyMs,
      confidence: result.decision.confidence,
      usedFallback: result.usedFallback,
    }),
    analysisStages: appendStage(
      state,
      'response_policy',
      'Applying response policy',
      `shouldCondense=${result.decision.shouldCondense}; tone=${result.decision.tone}; model=${result.model}; latencyMs=${result.latencyMs}; confidence=${result.decision.confidence.toFixed(2)}`
    ),
  };
}

async function condenseAnswerNode(state: AgentStateType): Promise<AgentUpdateType> {
  if (!state.answer) {
    return {};
  }

  const rawAnswer = state.answer.trim();
  const lineCount = rawAnswer.split('\n').filter(Boolean).length;
  const shouldCondense = state.responseShouldCondense ?? (rawAnswer.length > CONCISE_ANSWER_MAX_CHARS || lineCount > CONCISE_ANSWER_MIN_LINES);

  if (!shouldCondense) {
    return {};
  }

  try {
    const styleHint = state.responseStyleHint ?? 'brief';
    const concise = await invokeDeepSeekBedrock({
      systemPrompt: [
        'You are a concise response editor.',
        `Rewrite for style=${styleHint}. Keep it short, direct, and easy to scan.`,
        'Keep only the essential facts, timing, and next steps.',
        'Answer only what the user asked. Remove extra explanatory sections not requested by user.',
        'Do not add any new facts or explanations.',
        'Do not mention that you are summarizing.',
        styleHint === 'micro'
          ? 'Return plain text only in 1-2 short lines.'
          : 'Return plain text only, ideally 3-5 bullet points or 2 short paragraphs.',
      ].join(' '),
      userPrompt: `Condense this answer for the user:\n\n${rawAnswer}`,
      maxTokens: 320,
    });

    const compact = concise.text.trim();
    if (!compact) {
      return {};
    }

    const findings = state.toolFindings ?? [];
    const sanitizedCompact = enforceGroundingAnswerContract(compact, findings, state.question);

    return {
      answer: sanitizedCompact || compact,
      model: concise.model,
      analysisStages: appendStage(state, 'condense_answer', 'Condensing final response'),
    };
  } catch {
    return {};
  }
}

const graph = new StateGraph(AgentState)
  .addNode('route_top_level', routeTopLevelNode)
  .addNode('fast_answer', fastAnswerNode)
  .addNode('load_grounding', loadCanonicalGrounding)
  .addNode('classify_intent', classifyIntentNode)
  .addNode('plan_and_tools', planAndToolsNode)
  .addNode('run_specialized_tools', runSpecializedToolsNode)
  .addNode('run_general_tools', runGeneralToolsNode)
  .addNode('evidence_gate', evidenceGateNode)
  .addNode('refine_tools', refineToolsNode)
  .addNode('build_prompt', buildPromptNode)
  .addNode('answer_with_remedy_specialist', answerWithRemedySpecialistNode)
  .addNode('answer_with_deepseek', answerWithDeepSeekNode)
  .addNode('response_policy', responsePolicyNode)
  .addNode('condense_answer', condenseAnswerNode)
  .addEdge(START, 'route_top_level')
  .addConditionalEdges('route_top_level', routeFromTopLevel)
  .addEdge('fast_answer', END)
  .addEdge('classify_intent', 'load_grounding')
  .addEdge('load_grounding', 'plan_and_tools')
  .addConditionalEdges('plan_and_tools', routeIntent)
  .addEdge('run_specialized_tools', 'evidence_gate')
  .addEdge('run_general_tools', 'evidence_gate')
  .addConditionalEdges('evidence_gate', routeAfterEvidenceGate)
  .addEdge('refine_tools', 'evidence_gate')
  .addConditionalEdges('build_prompt', routeAfterPrompt)
  .addEdge('answer_with_remedy_specialist', 'response_policy')
  .addEdge('answer_with_deepseek', 'response_policy')
  .addEdge('response_policy', 'condense_answer')
  .addEdge('condense_answer', END)
  .compile();

// Phase-2: LangGraph-based grounded assistant that only reads canonical raw payload sections from Postgres.
export async function runKundliAgent(input: AgentAnswerInput): Promise<AgentAnswer> {
  const ownerId = input.ownerId ?? 'anonymous';
  const mode: AgentMode = input.mode ?? 'mini';

  if (mode === 'mini') {
    const deterministicMiniScope = evaluateMiniScope(input.message);
    if (!deterministicMiniScope.allowed || deterministicMiniScope.enforcementMode === 'blocked') {
      return {
        answer: buildMiniUpgradeResponse(input.message, deterministicMiniScope.reasons),
        model: 'cozmic-mini-guard',
        mode,
      };
    }
  }

  const topRouteResult = await decideTopLevelRouteDetailed(input.message, mode, input.conversationContext ?? []);
  const topRouteDecision = topRouteResult.decision;
  const topLevelRoute = topRouteDecision.topRoute;
  const deterministicTopRoute = decideTopLevelRouteDeterministic(input.message);

  let miniScopeTelemetry: DecisionTelemetry | null = null;
  let miniScopeDecision: MiniScopeDecision | null = null;

  if (mode === 'mini') {
    const miniScopeResult = await decideMiniScopeDetailed(input.message, mode, input.conversationContext ?? []);
    const deterministicMiniScope = evaluateMiniScope(input.message);
    miniScopeDecision = miniScopeResult.decision;
    miniScopeTelemetry = {
      node: 'mini_scope_decision',
      model: miniScopeResult.model,
      latencyMs: miniScopeResult.latencyMs,
      confidence: miniScopeResult.decision.confidence,
      usedFallback: miniScopeResult.usedFallback,
      fallbackReason: miniScopeResult.usedFallback ? 'mini scope decision fallback' : undefined,
      shadowComparison: env.LLM_DECISION_SHADOW_MODE
        ? `deterministicMode=${deterministicMiniScope.enforcementMode}; llmMode=${miniScopeResult.decision.enforcementMode}; deterministicAllowed=${deterministicMiniScope.allowed}; llmAllowed=${miniScopeResult.decision.allowed}`
        : undefined,
    };

    if (!miniScopeResult.decision.allowed || miniScopeResult.decision.enforcementMode === 'blocked') {
      return {
        answer: buildMiniUpgradeResponse(input.message, miniScopeResult.decision.reasons),
        model: 'cozmic-mini-guard',
        mode,
      };
    }
  }

  const profileId = normalizeProfileId(input);
  const referenceTime = resolveReferenceTime(input);

  if (!profileId && !input.kundli && topLevelRoute === 'pipeline') {
    return {
      answer: [
        'To answer this as a personal chart reading, I need your Kundli context first.',
        'I will not ask you for birth date/time/place in chat.',
        'Please open or generate a Kundli and try again.',
        '',
        'You can still ask general astrology questions (concepts, D1/D9 basics, or small talk) without loading a chart.',
      ].join('\n'),
      model: 'cozmic-profile-gate',
      mode,
      decisionTelemetry: [
        {
          node: 'profile_gate',
          model: 'deterministic-profile-gate',
          latencyMs: 0,
          confidence: 1,
          usedFallback: false,
        },
      ],
    };
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
    topLevelRoute,
    topLevelRouteConfidence: topRouteDecision.confidence,
    responseStyleHint: topRouteDecision.responseStyle,
    continuationIntent: topRouteDecision.continuityIntent,
    decisionTelemetry: [
      {
        node: 'route_top_level',
        model: topRouteResult.model,
        latencyMs: topRouteResult.latencyMs,
        confidence: topRouteDecision.confidence,
        usedFallback: topRouteResult.usedFallback,
        fallbackReason: topRouteResult.usedFallback ? 'route preflight fallback' : undefined,
        shadowComparison: env.LLM_DECISION_SHADOW_MODE
          ? `deterministic=${deterministicTopRoute}; llm=${topLevelRoute}; match=${deterministicTopRoute === topLevelRoute}`
          : undefined,
      },
      ...(miniScopeTelemetry ? [miniScopeTelemetry] : []),
    ],
    stageReporter: input.onStage ?? null,
    toolIteration: 0,
    maxToolIterations: 2,
    temporalWindow: null,
    intent: null,
    scopeSelection: null,
    selectedToolGroups: null,
    executionPlan: null,
    coverageGaps: [],
    coverageShouldRetry: null,
    coverageDecisionConfidence: null,
    refinementNextAction: null,
    analysisStages: [],
    toolFindings: [],
    prompt: null,
    responseShouldCondense: null,
    responsePolicyTone: null,
    responsePolicyAddDisclaimer: null,
    responsePolicyConfidence: null,
    decisionBundle: {
      source: topRouteResult.usedFallback ? 'hybrid' : 'llm',
      topRoute: topLevelRoute,
      responseStyle: topRouteDecision.responseStyle,
      continuityIntent: topRouteDecision.continuityIntent,
      miniEnforcementMode: miniScopeDecision?.enforcementMode,
      confidence: topRouteDecision.confidence,
      reason: 'preflight route decision',
    },
    toolAvailabilityPreflight: null,
  })) as AgentStateType;

  if (!finalState.answer) {
    throw new Error('LangGraph execution completed without a grounded answer.');
  }

  if (finalState.topLevelRoute === 'pipeline' && !finalState.grounding) {
    throw new Error('LangGraph execution completed without grounding for pipeline route.');
  }

  const restrictedMiniNotice =
    mode === 'mini' && miniScopeDecision?.enforcementMode === 'restricted'
      ? buildMiniRestrictedNotice(miniScopeDecision.reasons, miniScopeDecision.suggestedAlternative)
      : null;

  const finalAnswer = restrictedMiniNotice
    ? `${restrictedMiniNotice}\n\n${finalState.answer}`
    : finalState.answer;
  const groundedFinalAnswer = enforceGroundingAnswerContract(finalAnswer, finalState.toolFindings ?? [], input.message);

  return {
    answer: groundedFinalAnswer,
    model: finalState.model ?? env.GOOGLE_GENAI_MODEL,
    mode,
    executionPlan: finalState.executionPlan ?? undefined,
    analysisStages: finalState.analysisStages ?? undefined,
    decisionTelemetry: finalState.decisionTelemetry ?? undefined,
    grounding: finalState.grounding
      ? {
          ownerId,
          profileId: finalState.grounding.profileId,
          sourceDocId: finalState.grounding.sourceDocId,
          chartVersion: finalState.grounding.chartVersion,
          kundliSignature: finalState.grounding.kundliSignature,
          kundli: finalState.grounding.kundli,
          requestKey: finalState.grounding.requestKey,
          payloadHash: finalState.grounding.payloadHash,
          referenceTimestamp: finalState.grounding.referenceTimestamp,
          referenceTimeSource: finalState.grounding.referenceTimeSource,
          selectedPaths: finalState.grounding.selectedPaths,
        }
      : undefined,
  };
}
