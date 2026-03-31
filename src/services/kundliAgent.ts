import { Annotation, END, START, StateGraph } from '@langchain/langgraph';
import { type KundliSnapshotInput } from './be1Client.js';
import { env } from '../config/env.js';
import { getRagProfilesRepository } from '../repositories/ragProfilesRepository.js';
import { getRagSourcesRepository } from '../repositories/ragSourcesRepository.js';
import { stableHash } from './hash.js';
import { invokeDeepSeekBedrock } from './deepseekBedrock.js';
import { buildTransitToolFinding, type ToolFinding } from './astrologyTools.js';
import { z } from 'zod';

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
  'education',
  'children',
  'property',
  'travel',
  'spirituality',
  'timing',
  'yoga',
  'family',
]);

const CHART_LAYER_ALLOWLIST = new Set<ChartLayer>(['D1', 'D9', 'D10', 'D8', 'D30', 'D7', 'D4', 'D12', 'D20']);
const MICRO_SIGNAL_ALLOWLIST = new Set<MicroSignal>(['nakshatra', 'nakshatra_lord', 'sign_lord', 'drishti', 'degree']);
const COVERAGE_GAP_ALLOWLIST = new Set<CoverageGap>(['varga', 'd9', 'dasha', 'transit', 'career', 'longevity']);
const TOOL_GROUP_ALLOWLIST = new Set<ToolGroupKey>([
  'reference_time',
  'atlas',
  'varga',
  'arudha',
  'd9',
  'dasha',
  'transit',
  'career',
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
    requiredScopes: ['chart.transit'],
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
  return value.trim().toLowerCase().replace(/\s+/g, '_');
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

  const miniBlocked = new Set<ToolGroupKey>(['dasha', 'transit', 'career', 'longevity', 'feature', 'arudha']);
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
  return `Note: some sections are missing in current data (${missing.join(', ')}), so this answer is based on available canonical evidence.`;
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

async function invokeDecisionNode<T>(params: {
  node: string;
  schema: z.ZodType<T, z.ZodTypeDef, unknown>;
  input: Record<string, unknown>;
  fallback: () => T;
}): Promise<{ decision: T; model: string; usedFallback: boolean; latencyMs: number }> {
  const startedAt = Date.now();
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
  const advancedVargas = vargaKeys.filter((key) => key !== 'D1' && key !== 'D9');
  const onlyCareerAdvanced = advancedVargas.length > 0 && advancedVargas.every((key) => key === 'D10');

  if (advancedVargas.length > 0) {
    if (onlyCareerAdvanced) {
      restrictedReasons.push('D10 is Pro-level; mini will provide D1/D9-based career guidance only.');
    } else {
      blockedReasons.push('This asks for advanced divisional charts beyond mini scope (D1/D9).');
    }
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

  if (family === 'longevity') {
    blockedReasons.push('Longevity analysis (D8/D30 + advanced timing) is Pro scope.');
  } else if (restrictedFamilies.has(family)) {
    restrictedReasons.push(`This is ${family} analysis; mini will use foundational D1/D9 scope.`);
  }

  const proTimingFlags = new Set(['timing', 'dasha', 'transit', 'forecast', 'history', 'career_timing']);
  if (intent.flags.some((flag) => proTimingFlags.has(flag))) {
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

function buildMiniRestrictedNotice(reasons: string[] = [], suggestedAlternative?: string): string {
  const why = reasons.length > 0
    ? ['Why this was restricted in Mini:', ...reasons.slice(0, 2).map((reason) => `- ${reason}`)]
    : [];

  return [
    'Mini scope note: this answer is intentionally constrained to **D1/D9 foundational guidance**.',
    ...why,
    suggestedAlternative ?? 'For full predictive/advanced chart analysis, switch to Cozmic Pro.',
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
  const hasTransit = intent.flags.includes('transit') || /\b(transit|gochar|today|now|tomorrow)\b/.test(q);

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
        'Decide mini-scope enforcement mode. Modes: full (allowed as-is), restricted (answer with D1/D9 foundational scope), blocked (requires Pro). Mini allows D1/D9 and basic non-predictive guidance. Block advanced systems and predictive timing.',
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
        'Classify the fast-answer intent for a non-pipeline astrology chat message. Pick exactly one intent kind.',
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
  if (/\b(my|mine|me|for me|my chart|my kundli|when will i|will i|should i)\b/.test(q)) {
    return 'pipeline';
  }

  return 'general_astro'; // Safe default
}

// DIRECT LLM ROUTER - Connects message → decision → response immediately
async function directLLMRoute(
  message: string,
  mode: AgentMode,
  conversationContext: string[] = []
): Promise<{ route: TopLevelRoute; confidence: number; latencyMs: number }> {
  const startTime = Date.now();
  const deterministicRoute = decideTopLevelRouteDeterministic(message);
  
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
    - mini: keep non-pipeline astrology scope to D1, D9, and basic astrological insights.
    - pro: allow all kinds of astrology answers and advanced topics.

    Brand rule:
    - If user asks identity (who are you / who built you), route to general_astro so response can clearly state: "I am Cozmic AI."

Message: "${message}"

Reply with just the route name and confidence score (0-1).`,
    },
    fallback: () => ({
      topRoute: deterministicRoute,
      confidence: deterministicRoute === 'pipeline' ? 0.85 : 0.7,
      reasoningBrief: 'llm fallback deterministic',
      requiresPersonalChart: deterministicRoute === 'pipeline',
    }),
  });

  return {
    route: result.decision.topRoute,
    confidence: result.decision.confidence,
    latencyMs: Date.now() - startTime,
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
      confidence: result.confidence,
      reasoningBrief: 'direct_llm_router',
      requiresPersonalChart: result.route === 'pipeline',
    },
    model: 'cozmic-direct-llm-router',
    usedFallback: false,
    latencyMs: result.latencyMs,
  };
}

export async function shouldBypassChartPipeline(message: string, mode: AgentMode = 'mini', conversationContext: string[] = []): Promise<boolean> {
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
    mini: 'I am Cozmic AI, your Vedic astrology assistant. I analyze your birth chart (D1 & D9) for personality, relationships, and compatibility insights. In mini mode, I focus on the main chart and divisional chart 9 (marriage). Share your birth details to get started!',
    pro: 'I am Cozmic AI, your comprehensive Vedic astrology assistant. I analyze all divisional charts (D1-D30), dasha periods, transits, yogas, and advanced techniques for personality, relationships, career, finances, health, and life timing. Share your birth details for deep cosmic insights!',
  };
  return identityAnswers[mode];
}

async function answerCapabilityQuestion(message: string, mode: AgentMode): Promise<string> {
  const capabilityAnswers = {
    mini: 'In **mini mode**, I analyze your **D1 (Main Chart)** for core personality and **D9 (Navamsha)** for relationships and hidden traits. You get personality insights, relationship compatibility, and basic life timing. Upgrade to **Pro** for career analysis (D10), longevity (D8/D30), and advanced techniques.',
    pro: 'I provide complete Vedic astrology analysis: **all divisional charts** (D1, D9, D10, D8, D30, D7, D4, D12, D20), **dasha periods** (life timing), **transits** (current planetary cycles), **yogas** (auspicious combinations), and advanced astrological techniques. I cover personality, relationships, career, finances, health, remedies, and timing.',
  };
  return capabilityAnswers[mode];
}

async function generateSmallTalkResponse(message: string, mode: AgentMode, conversationContext: string[] = []): Promise<string> {
  try {
    const response = await invokeDeepSeekBedrock({
      systemPrompt: [
        'You are Cozmic AI, a friendly Vedic astrology assistant.',
        'Respond briefly (1-2 short sentences).',
        'If asked identity (who are you / who built you), explicitly say: "I am Cozmic AI."',
        mode === 'mini'
          ? 'Mini mode scope: mention D1, D9, and basic astrology guidance only.'
          : 'Pro mode scope: you may mention comprehensive astrology capabilities.',
      ].join(' '),
      userPrompt: [
        `Message: ${message}`,
        conversationContext.length > 0 ? `Recent context: ${conversationContext.slice(-4).join(' | ')}` : 'Recent context: none',
      ].join('\n'),
      maxTokens: 90,
    });

    const text = response.text.trim();
    if (text) return text;
  } catch {
    // fall through to deterministic fallback
  }

  return 'I am Cozmic AI. Ask me anything about your chart, astrology concepts, or life guidance.';
}

async function generateGeneralAstroResponse(message: string, mode: AgentMode, conversationContext: string[] = []): Promise<string> {
  // LLM for complex concept questions
  try {
    const response = await invokeDeepSeekBedrock({
      systemPrompt: `You are Cozmic AI, a Vedic astrology assistant. Answer clearly and accurately in 2-4 short sentences.
Identity rule: If the user asks who you are or who built you, say clearly: "I am Cozmic AI."
${mode === 'mini'
  ? 'Mini mode policy: answer using D1, D9, and basic astrology insights only. Avoid deep advanced techniques.'
  : 'Pro mode policy: provide all kinds of astrology answers, including advanced divisional charts, dasha, transit, yogas, and timing.'}
If question needs personal chart-specific analysis but birth details are missing, ask for birth details.` ,
      userPrompt: [
        `User message: ${message}`,
        conversationContext.length > 0 ? `Recent context: ${conversationContext.slice(-6).join(' | ')}` : 'Recent context: none',
      ].join('\n'),
      maxTokens: 150,
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
  conversationContext: string[] = []
): Promise<Pick<AgentAnswer, 'answer' | 'model' | 'mode'>> {
  try {
    const decisionResult = await decideFastAnswerIntentDetailed(message, mode, route, conversationContext);
    const decision = decisionResult.decision;

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
        answer = await generateSmallTalkResponse(message, mode, conversationContext);
        handlerModel = 'cozmic-smalltalk-response';
        break;
      case 'general_astro':
      default:
        answer = await generateGeneralAstroResponse(message, mode, conversationContext);
        handlerModel = 'cozmic-general-astro-response';
        break;
    }

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

  const seriesNodes = ['classify_intent', 'load_grounding', 'plan_and_tools', 'run_specialized_tools', 'run_general_tools', 'build_prompt', 'answer_with_deepseek', 'condense_answer'];
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
  return rawPayload;
}

async function loadCanonicalGrounding(state: AgentStateType): Promise<AgentUpdateType> {
  const profileId = state.profileId;
  if (!profileId) {
    throw new Error('Profile identity is required before grounding can be loaded.');
  }

  const ragProfiles = getRagProfilesRepository();
  const ragSources = getRagSourcesRepository();
  const profileDoc = await ragProfiles.getByOwnerAndProfileId(state.ownerId, profileId);

  if (!profileDoc) {
    throw new Error(`No canonical profile snapshot found for ${profileId}. Regenerate the Kundli first.`);
  }

  const sourceDoc = await ragSources.getById(profileDoc.latestSourceDocId);

  if (!sourceDoc) {
    throw new Error(`No canonical raw payload found for profile ${profileId}. Regenerate the Kundli first.`);
  }

  const rawPayload = normalizeRawPayload(sourceDoc.data.rawPayload);
  const atlas = summarizeChartAtlas(rawPayload);
  const toolAvailabilityPreflight = buildToolAvailabilityPreflight(rawPayload, state.mode);
  const selectedSections = selectSections(rawPayload, state.question, state.intent?.flags ?? [], state.mode, state.scopeSelection ?? null);
  const fallbackSections = selectedSections.length > 0 ? selectedSections : atlas.slice(0, 12).map((item) => ({ path: item.path, value: getByPath(rawPayload, item.path) }));
  const kundli = state.kundliInput ?? profileDoc.kundliInput;

  return {
    grounding: {
      ownerId: state.ownerId,
      profileId,
      sourceDocId: profileDoc.latestSourceDocId,
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
      analysisStages: appendStage(state, 'route_top_level', 'Routing question at top-level', `route=${state.topLevelRoute}`),
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
    decisionBundle: mergeDecisionBundle(state, {
      source: result.usedFallback ? 'hybrid' : 'llm',
      topRoute: decision.topRoute,
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
      `route=${decision.topRoute}; confidence=${decision.confidence.toFixed(2)}; model=${result.model}; latencyMs=${result.latencyMs}`
    ),
  };
}

function routeFromTopLevel(state: AgentStateType): 'fast_answer' | 'classify_intent' {
  return state.topLevelRoute === 'pipeline' ? 'classify_intent' : 'fast_answer';
}

async function fastAnswerNode(state: AgentStateType): Promise<AgentUpdateType> {
  const route = state.topLevelRoute ?? 'smalltalk';
  const fast = await answerSimpleWithoutChart(state.question, state.mode, route, state.conversationContext ?? []);
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
    ? (llmSelectedToolGroups.length > 0 ? llmSelectedToolGroups : deterministicForDecision)
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

    if (scope?.needsTransit || executionPlan.includeTransit || intent.flags.includes('transit')) {
      groups.add('transit');
    }

    if (scope?.needsD10 || executionPlan.includeCareer || intent.flags.includes('career') || intent.flags.includes('career_timing')) {
      groups.add('career');
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
  const finalGroups = selectedToolGroups.length > 0 ? selectedToolGroups : deterministicForDecision;
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
    dasha: () => (isMini ? Promise.resolve(null) : Promise.resolve(makeDashaToolFinding(grounding.rawPayload, analysisTimestamp))),
    transit: () => (isMini
      ? Promise.resolve(null)
      : buildTransitToolFinding({ kundli: grounding.kundli, question, referenceTimestamp: grounding.referenceTimestamp })),
    career: () => (isMini ? Promise.resolve(null) : Promise.resolve(makeCareerToolFinding(grounding.rawPayload, question, analysisTimestamp))),
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
  const includeTransit = shouldRun('transit', (executionPlan.includeTransit || Boolean(scopeSelection?.needsTransit)) && !intent.flags.includes('transit') && !isMini);
  const includeCareer = shouldRun('career', (executionPlan.includeCareer || Boolean(scopeSelection?.needsD10) || intent.topics.includes('career') || /\b(career|job|profession|business|promotion|work|employment|salary|interview)\b/i.test(state.question)) && !isMini);
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
          'Provide foundational interpretations: planetary placements, sign/nakshatra meanings, basic timing.',
          'Avoid deep divisional chart analysis (D10, D8, D30, etc.) unless explicitly requested.',
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
    'Use the tool findings first; they are deterministic extracts from the JSON blob.',
    'Grounding contract: every material claim must be grounded in one or more tool findings listed below.',
    'Grounding contract: do NOT claim data is missing when that tool is marked ok/partial.',
    'Grounding contract: only mention missing data if the related tool is marked unavailable.',
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

function sanitizeMissingDataContradictions(answer: string, findings: ToolFinding[]): string {
  const { available, partial } = getToolAvailabilityIndex(findings);
  const hasData = (name: string) => available.has(name) || partial.has(name);

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
  ]);

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

function enforceGroundingAnswerContract(answer: string, findings: ToolFinding[]): string {
  const step1 = sanitizeGenericMissingAnalysisClaims(answer, findings);
  const step2 = sanitizeMissingDataContradictions(step1, findings);
  return step2;
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
  const selected = new Set(state.selectedToolGroups ?? []);
  if (selected.size > 0) {
    const specialized = ['d9', 'dasha', 'transit', 'career', 'longevity', 'feature', 'arudha'].some((group) => selected.has(group as ToolGroupKey));
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
    const deepSeek = await invokeDeepSeekBedrock({
      systemPrompt: state.prompt,
      userPrompt: [
        'Answer using deterministic tool findings and canonical snippets.',
        'Contract: every important claim must be grounded in available tool findings.',
        'Contract: do not claim missing dasha/transit/D9/D10/longevity data when related analyzer status is ok/partial.',
        'Contract: if data is missing, name the exact unavailable analyzer/tool and continue with available evidence.',
        `Available analyzers: ${[...availability.available, ...availability.partial].join(', ') || 'none'}`,
        `Unavailable analyzers: ${[...availability.unavailable].join(', ') || 'none'}`,
        'Be decisive, specific, and avoid generic disclaimers.',
      ].join(' '),
    });

    const sanitized = enforceGroundingAnswerContract(deepSeek.text, findings);

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

async function responsePolicyNode(state: AgentStateType): Promise<AgentUpdateType> {
  if (!state.answer) {
    return {};
  }

  const defaultShouldCondense = (() => {
    const raw = state.answer ?? '';
    const lineCount = raw.split('\n').filter(Boolean).length;
    return raw.length > CONCISE_ANSWER_MAX_CHARS || lineCount > CONCISE_ANSWER_MIN_LINES;
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
      answer: state.answer,
      findings: (state.toolFindings ?? []).map((f) => ({ name: f.name, status: f.status })),
      defaultShouldCondense,
      instruction:
        'Set condensation strategy and confidence tone based on evidence quality. Use disclaimer only when key findings are unavailable.',
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
  .addEdge('build_prompt', 'answer_with_deepseek')
  .addEdge('answer_with_deepseek', 'response_policy')
  .addEdge('response_policy', 'condense_answer')
  .addEdge('condense_answer', END)
  .compile();

// Phase-2: LangGraph-based grounded assistant that only reads canonical raw payload sections from Postgres.
export async function runKundliAgent(input: AgentAnswerInput): Promise<AgentAnswer> {
  const ownerId = input.ownerId ?? 'anonymous';
  const mode: AgentMode = input.mode ?? 'mini';

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

  if (!profileId && topLevelRoute === 'pipeline') {
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
    topLevelRoute,
    topLevelRouteConfidence: topRouteDecision.confidence,
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

  return {
    answer: finalAnswer,
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
