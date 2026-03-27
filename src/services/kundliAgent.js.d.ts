export interface AgentAnswerInput {
  ownerId?: string;
  message: string;
  kundli?: unknown;
  profileId?: string;
  clientTimestamp?: number;
}

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
    referenceTimeSource: 'client' | 'server';
    selectedPaths: string[];
  };
}

export declare function runKundliAgent(input: AgentAnswerInput): Promise<AgentAnswer>;