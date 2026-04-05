import assert from 'node:assert/strict';
import {
  __testOnlyMakeCareerToolFinding,
  runKundliAgent,
  shouldBypassChartPipeline,
} from '../src/services/kundliAgent.js';

type CareerSplit = {
  governmentScore: number;
  privateScore: number;
  recommendation: string;
};

function extractCareerSplit(facts: string[]): CareerSplit {
  const heuristicLine = facts.find((fact) => fact.startsWith('India career split heuristic:'));
  assert.ok(heuristicLine, 'Career finding should include India career split heuristic line for govt/private cue questions');

  const match = heuristicLine.match(/governmentScore=(\d+)\/100,\s*privateScore=(\d+)\/100\s*\(([^)]+)\)/i);
  assert.ok(match, `Could not parse heuristic score line: ${heuristicLine}`);

  return {
    governmentScore: Number(match[1]),
    privateScore: Number(match[2]),
    recommendation: match[3],
  };
}

async function main(): Promise<void> {
  const routingCases: Array<{ q: string; expectedBypass: boolean }> = [
    { q: 'From my chart, what remedies should I do for career stress?', expectedBypass: false },
    { q: 'From my chart, can I relocate abroad in 2027?', expectedBypass: false },
    { q: 'From my chart, what does my past-life karma indicate?', expectedBypass: false },
    { q: 'From my chart, can I conceive in 2026?', expectedBypass: false },
    { q: 'From my chart, is there legal dispute risk this year?', expectedBypass: false },
    { q: 'From my chart, should I prepare for UPSC or private corporate job?', expectedBypass: false },
  ];

  for (const testCase of routingCases) {
    const bypass = await shouldBypassChartPipeline(testCase.q, 'mini', []);
    assert.equal(
      bypass,
      testCase.expectedBypass,
      `Unexpected bypass for question: ${testCase.q}; got ${bypass}; expected ${testCase.expectedBypass}`
    );
  }

  const miniBlockedCases = [
    'From my chart, what remedies should I do this year?',
    'From my chart, can I relocate abroad in 2027?',
    'From my chart, what does my past life karma indicate?',
    'From my chart, can I conceive in 2026?',
    'From my chart, is there legal dispute risk in 2027?',
  ];

  for (const question of miniBlockedCases) {
    const response = await runKundliAgent({
      ownerId: 'family-regression',
      mode: 'mini',
      message: question,
      conversationContext: [],
    });

    assert.equal(response.model, 'cozmic-mini-guard', `Mini mode should block advanced family analysis: ${question}`);
    assert.ok(
      /not available in \*\*Cozmic Mini\*\*/i.test(response.answer),
      `Mini blocked answer should clearly state unavailability in Mini mode for: ${question}`
    );
  }

  const referenceTs = Date.parse('2026-04-04T00:00:00.000Z');

  const governmentLeaningPayload = {
    chart: {
      varga: {
        D10: {
          lagna: { Lg: { rashi: 1, degree: 12.2 } },
          graha: {
            Su: { rashi: 10, house_number: 6, degree: 5.2 },
            Sa: { rashi: 6, house_number: 6, degree: 11.7 },
            Ju: { rashi: 9, house_number: 6, degree: 2.4 },
          },
        },
        D1: {
          graha: {
            Su: { rashi: 11, house_number: 6, degree: 14.1 },
            Sa: { rashi: 8, house_number: 6, degree: 22.6 },
            Ju: { rashi: 5, house_number: 6, degree: 9.7 },
          },
        },
      },
      dasha: {
        type: 'vimshottari',
        periods: {},
      },
    },
  };

  const governmentFinding = __testOnlyMakeCareerToolFinding(
    governmentLeaningPayload,
    'From my chart, should I prepare for UPSC or private corporate job?',
    referenceTs
  );

  assert.equal(governmentFinding.status, 'ok', 'Career analyzer should produce an ok finding for synthetic D1/D10 payload');
  assert.ok(
    governmentFinding.facts.some((fact) => /Question intent indicates Indian government\/public sector path/i.test(fact)),
    'Govt-cue career question should include government/public sector rationale line'
  );
  assert.ok(
    governmentFinding.facts.some((fact) => /heuristic guidance/i.test(fact)),
    'Govt/private output should include explicit heuristic disclaimer'
  );

  const governmentSplit = extractCareerSplit(governmentFinding.facts);
  assert.ok(
    governmentSplit.governmentScore > governmentSplit.privateScore,
    `Expected government score to exceed private score for government-leaning payload; got gov=${governmentSplit.governmentScore}, private=${governmentSplit.privateScore}`
  );
  assert.match(
    governmentSplit.recommendation,
    /government-focused track stronger/i,
    `Expected government recommendation for government-leaning payload; got: ${governmentSplit.recommendation}`
  );

  const privateLeaningPayload = {
    chart: {
      varga: {
        D10: {
          lagna: { Lg: { rashi: 7, degree: 19.3 } },
          graha: {
            Me: { rashi: 3, house_number: 11, degree: 27.6 },
            Ve: { rashi: 2, house_number: 11, degree: 20.1 },
            Ma: { rashi: 1, house_number: 7, degree: 3.1 },
            Ra: { rashi: 8, house_number: 3, degree: 11.4 },
          },
        },
        D1: {
          graha: {
            Me: { rashi: 3, house_number: 11, degree: 12.6 },
            Ve: { rashi: 6, house_number: 11, degree: 9.1 },
            Ma: { rashi: 2, house_number: 7, degree: 22.7 },
            Ra: { rashi: 5, house_number: 3, degree: 4.2 },
          },
        },
      },
      dasha: {
        type: 'vimshottari',
        periods: {},
      },
    },
  };

  const privateFinding = __testOnlyMakeCareerToolFinding(
    privateLeaningPayload,
    'From my chart, should I target private sector startup or MNC role?',
    referenceTs
  );

  assert.equal(privateFinding.status, 'ok', 'Career analyzer should produce an ok finding for synthetic private-leaning payload');
  assert.ok(
    privateFinding.facts.some((fact) => /Question intent indicates Indian private\/corporate path/i.test(fact)),
    'Private-cue career question should include private/corporate rationale line'
  );

  const privateSplit = extractCareerSplit(privateFinding.facts);
  assert.ok(
    privateSplit.privateScore > privateSplit.governmentScore,
    `Expected private score to exceed government score for private-leaning payload; got gov=${privateSplit.governmentScore}, private=${privateSplit.privateScore}`
  );
  assert.match(
    privateSplit.recommendation,
    /private\/corporate track stronger/i,
    `Expected private recommendation for private-leaning payload; got: ${privateSplit.recommendation}`
  );

  for (const score of [governmentSplit.governmentScore, governmentSplit.privateScore, privateSplit.governmentScore, privateSplit.privateScore]) {
    assert.ok(score >= 35 && score <= 95, `Heuristic score should be within normalized bounds [35,95]; got ${score}`);
  }

  console.log('family regression tests passed');
}

main().catch((error) => {
  console.error('family regression tests failed', error);
  process.exitCode = 1;
});
