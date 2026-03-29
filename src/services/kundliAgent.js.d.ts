export interface AgentAnswerInput {
  ownerId?: string;
  message: string;
  mode?: 'mini' | 'pro';
  kundli?: unknown;
  profileId?: string;
  clientTimestamp?: number;
  conversationContext?: string[];
  onStage?: (stage: AnalysisStage) => void;
}

export interface AnalysisStage {
  id: string;
  label: string;
  status: 'completed';
  details?: string;
}

export interface DynamicExecutionPlan {
  family: string;
  chartLayers: string[];
  includeTiming: boolean;
  includeTransit: boolean;
  includeDasha: boolean;
  includeCareer: boolean;
  includeRelationship: boolean;
  includeMicroSignals: string[];
  seriesNodes: string[];
  parallelBatches: string[][];
}

export interface AgentAnswer {
  answer: string;
  model: string;
  mode: 'mini' | 'pro';
  executionPlan?: DynamicExecutionPlan;
  analysisStages?: AnalysisStage[];
  grounding?: {
    ownerId: string;
    profileId: string;
    sourceDocId: string;
    chartVersion: string;
    kundliSignature: string;
    kundli: unknown;
    requestKey: string;
    payloadHash: string;
    referenceTimestamp: number;
    referenceTimeSource: 'client' | 'server';
    selectedPaths: string[];
  };
}

export declare function runKundliAgent(input: AgentAnswerInput): Promise<AgentAnswer>;