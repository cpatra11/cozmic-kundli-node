import assert from 'node:assert/strict';
import { runKundliAgent, shouldBypassChartPipeline } from '../src/services/kundliAgent.js';

async function main(): Promise<void> {
  const routeCases: Array<{ q: string; expectedBypass: boolean }> = [
    { q: 'Hi', expectedBypass: true },
    { q: 'thanks', expectedBypass: true },
    { q: 'What is Saturn return?', expectedBypass: true },
    { q: 'When will I marry from my kundli?', expectedBypass: false },
  ];

  for (const testCase of routeCases) {
    const bypass = await shouldBypassChartPipeline(testCase.q, 'mini', []);
    assert.equal(
      bypass,
      testCase.expectedBypass,
      `Unexpected bypass for question: ${testCase.q}; got ${bypass}; expected ${testCase.expectedBypass}`
    );
  }

  const fast = await runKundliAgent({
    ownerId: 'smoke',
    mode: 'mini',
    message: 'What is Saturn return?',
    conversationContext: [],
  });

  assert.ok(fast.answer.length > 0, 'Fast-path answer should not be empty');
  assert.ok(Array.isArray(fast.analysisStages), 'Fast-path analysisStages should be present');
  assert.ok(
    fast.analysisStages?.some((stage) => stage.id === 'route_top_level'),
    'Fast-path should include route_top_level stage'
  );
  assert.ok(
    fast.analysisStages?.some((stage) => stage.id === 'fast_answer'),
    'Fast-path should include fast_answer stage'
  );

  assert.ok(Array.isArray(fast.decisionTelemetry), 'Decision telemetry should be present');
  assert.ok(
    fast.decisionTelemetry?.some((item) => item.node === 'route_top_level'),
    'Decision telemetry should include route_top_level entry'
  );

  for (const entry of fast.decisionTelemetry ?? []) {
    assert.equal(typeof entry.node, 'string', 'Telemetry node must be a string');
    assert.equal(typeof entry.model, 'string', 'Telemetry model must be a string');
    assert.equal(typeof entry.latencyMs, 'number', 'Telemetry latency must be numeric');
    if (entry.confidence !== undefined) {
      assert.ok(entry.confidence >= 0 && entry.confidence <= 1, 'Telemetry confidence must be in [0,1]');
    }
  }

  console.log('agent decision smoke tests passed');
}

main().catch((error) => {
  console.error('agent decision smoke tests failed', error);
  process.exitCode = 1;
});
