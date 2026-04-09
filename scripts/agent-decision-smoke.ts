import assert from 'node:assert/strict';
import { runKundliAgent, shouldBypassChartPipeline } from '../src/services/kundliAgent.js';

async function main(): Promise<void> {
  const routeCases: Array<{ q: string; expectedBypass: boolean }> = [
    { q: 'Hi', expectedBypass: true },
    { q: 'thanks', expectedBypass: true },
    { q: 'What is Saturn return?', expectedBypass: true },
    { q: 'From my chart, what is my D10 dashamsha?', expectedBypass: false },
    { q: 'When will my marriage happen?', expectedBypass: false },
    { q: 'Can I marry in 2026? 2027?', expectedBypass: false },
    { q: 'When will I get a new job?', expectedBypass: false },
    { q: 'Will my finances improve in 2027?', expectedBypass: false },
    { q: 'When will I marry from my kundli?', expectedBypass: false },
    { q: 'What is my current transit details?', expectedBypass: false },
    { q: 'How will my Jupiter transit affect me?', expectedBypass: false },
    { q: 'From my chart, should I prepare for UPSC or private corporate job?', expectedBypass: false },
    { q: 'From my chart, what remedies should I do this year?', expectedBypass: false },
    { q: 'From my chart, can I relocate abroad in 2027?', expectedBypass: false },
    { q: 'From my chart, what does my past life karma indicate?', expectedBypass: false },
    { q: 'From my chart, can I conceive in 2026?', expectedBypass: false },
    { q: 'From my chart, is there legal dispute risk in 2027?', expectedBypass: false },
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

  const miniBlocked = await runKundliAgent({
    ownerId: 'smoke',
    mode: 'mini',
    message: 'From my chart, what is my D10 dashamsha?',
    conversationContext: [],
  });

  assert.equal(miniBlocked.model, 'cozmic-mini-guard', 'Mini mode should block explicit D10/dashamsha analysis');
  assert.ok(
    /not available in \*\*Cozmic Mini\*\*/i.test(miniBlocked.answer),
    'Mini block message should clearly state unavailability in Mini mode'
  );

  const miniBlockedRemedies = await runKundliAgent({
    ownerId: 'smoke',
    mode: 'mini',
    message: 'From my chart, what remedies should I do to improve career?',
    conversationContext: [],
  });

  assert.equal(miniBlockedRemedies.model, 'cozmic-mini-guard', 'Mini mode should block remedies analysis');
  assert.ok(
    /not available in \*\*Cozmic Mini\*\*/i.test(miniBlockedRemedies.answer),
    'Mini remedies block message should clearly state unavailability in Mini mode'
  );

  const miniBlockedTransitTiming = await runKundliAgent({
    ownerId: 'smoke',
    mode: 'mini',
    message: 'How will my Jupiter transit affect my marriage timing this year?',
    conversationContext: [],
  });

  assert.equal(
    miniBlockedTransitTiming.model,
    'cozmic-mini-guard',
    'Mini mode should block transit/timing analysis and force upgrade response'
  );
  assert.ok(
    /not available in \*\*Cozmic Mini\*\*/i.test(miniBlockedTransitTiming.answer),
    'Mini transit/timing block message should clearly state unavailability in Mini mode'
  );

  const miniBlockedTimingAnalysis = await runKundliAgent({
    ownerId: 'smoke',
    mode: 'mini',
    message: 'Please do timing analysis for my marriage.',
    conversationContext: [],
  });

  assert.equal(
    miniBlockedTimingAnalysis.model,
    'cozmic-mini-guard',
    'Mini mode should hard-block explicit timing analysis phrasing'
  );
  assert.ok(
    /not available in \*\*Cozmic Mini\*\*/i.test(miniBlockedTimingAnalysis.answer),
    'Mini timing-analysis block message should clearly state unavailability in Mini mode'
  );
  assert.ok(
    !/open or generate a Kundli in the app/i.test(miniBlockedTimingAnalysis.answer),
    'Mini timing-analysis hard block should not fall through to profile gate wording'
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
