import assert from 'node:assert/strict';
import {
  buildTransitIntervalToolFinding,
  buildTransitPointToolFinding,
  resolveTransitAt,
  resolveTransitRequestKind,
} from '../src/services/astrologyTools.js';
import { sanitizeMissingDataContradictions } from '../src/services/kundliAgent.js';

async function main(): Promise<void> {
  const referenceTimestamp = Date.parse('2026-04-03T00:00:00Z');

  const currentQuery = 'give me the current transit details with the time 1. 06AM. Is the current time and date is 3 April 2026. And the birth. 14 March. 2002. 10:39 AM.20.9501° N, 85.2168° E';
  const currentResolved = resolveTransitAt(currentQuery, referenceTimestamp);

  assert.equal(resolveTransitRequestKind(currentQuery, referenceTimestamp), 'current', 'Current transit query should resolve to current mode');
  assert.ok(currentResolved, 'Current transit query should resolve a concrete datetime');
  assert.equal(currentResolved?.toISOString(), '2026-04-03T01:06:00.000Z', 'Current transit query should honor explicit date+time from question');

  const betweenQuery = 'transit between 03/04/2026 and 03/04/2027';
  assert.equal(resolveTransitRequestKind(betweenQuery, referenceTimestamp), 'range', 'Explicit between-dates query should resolve to range mode');

  const yearSpanQuery = 'how will be my career in 2026-27';
  assert.equal(resolveTransitRequestKind(yearSpanQuery, referenceTimestamp), 'range', 'Year span query should resolve to range mode');

  const thisMonthQuery = 'what does this month look like for my career';
  assert.equal(
    resolveTransitRequestKind(thisMonthQuery, referenceTimestamp),
    'range',
    'This month timing query should resolve to range mode'
  );

  const thisYearQuery = 'what does this year look like for my marriage';
  assert.equal(
    resolveTransitRequestKind(thisYearQuery, referenceTimestamp),
    'range',
    'This year timing query should resolve to range mode'
  );

  const hardshipMonthYearQuery = 'I got cheated in September 2025, what was happening then?';
  assert.equal(
    resolveTransitRequestKind(hardshipMonthYearQuery, referenceTimestamp),
    'range',
    'Named month-year hardship query should resolve to range mode for exact period analysis'
  );

  const marriageTwoYearQuery = 'Can I marry in 2026? 2027';
  assert.equal(
    resolveTransitRequestKind(marriageTwoYearQuery, referenceTimestamp),
    'range',
    'Marriage timing query with two years should resolve to range mode'
  );

  const marriageSingleYearQuery = 'Can I marry in 2026';
  assert.equal(
    resolveTransitRequestKind(marriageSingleYearQuery, referenceTimestamp),
    'range',
    'Marriage timing query with a single explicit year should resolve to range mode'
  );

  const noTimeQuery = 'when will my marriage happen';
  assert.equal(
    resolveTransitRequestKind(noTimeQuery, referenceTimestamp),
    'range',
    'No-time marriage timing query should resolve to range mode for timing window analysis'
  );
  assert.equal(
    resolveTransitAt(noTimeQuery, referenceTimestamp),
    null,
    'No-time query should not fabricate explicit datetime in resolver'
  );

  const careerTimingNoDateQuery = 'when will i get a new job';
  assert.equal(
    resolveTransitRequestKind(careerTimingNoDateQuery, referenceTimestamp),
    'range',
    'Open-ended career timing query should resolve to range mode'
  );

  const jupiterTransitQuery = 'How will my Jupiter transit affect me?';
  assert.equal(
    resolveTransitRequestKind(jupiterTransitQuery, referenceTimestamp),
    'point',
    'Jupiter transit impact query without explicit time window should resolve to point/current-context mode'
  );

  const financeTimingYearQuery = 'will my finances improve in 2027';
  assert.equal(
    resolveTransitRequestKind(financeTimingYearQuery, referenceTimestamp),
    'range',
    'Year-anchored finance timing query should resolve to range mode'
  );

  const pointQuery = 'transit on 2026-04-03 01:06';
  const pointResolved = resolveTransitAt(pointQuery, referenceTimestamp);
  assert.equal(resolveTransitRequestKind(pointQuery, referenceTimestamp), 'point', 'Explicit point datetime query should resolve to point mode');
  assert.ok(pointResolved, 'Point transit query should resolve datetime');
  assert.equal(pointResolved?.toISOString(), '2026-04-03T01:06:00.000Z', 'Point transit query should preserve explicit datetime');

  const kundli = {
    latitude: 20.9501,
    longitude: 85.2168,
    year: 2002,
    month: 3,
    day: 14,
    hour: 10,
    min: 39,
    sec: 0,
    time_zone: 'Asia/Kolkata',
  };

  const pointFinding = await buildTransitPointToolFinding({
    kundli,
    question: currentQuery,
    referenceTimestamp,
  });

  assert.notEqual(
    pointFinding.status,
    'unavailable',
    `Point transit finding should be available; got: ${pointFinding.facts.join(' | ')}`
  );
  assert.ok(
    pointFinding.facts.some((fact) => /Transit focus:/i.test(fact)),
    'Point transit finding should include a transit focus summary'
  );
  assert.ok(
    pointFinding.evidencePaths.includes('backend:/api/transit-chart'),
    'Point transit finding should reference backend transit-chart endpoint'
  );

  const intervalFinding = await buildTransitIntervalToolFinding({
    kundli,
    question: yearSpanQuery,
    referenceTimestamp,
  });

  assert.notEqual(
    intervalFinding.status,
    'unavailable',
    `Interval transit finding should be available; got: ${intervalFinding.facts.join(' | ')}`
  );
  assert.ok(
    intervalFinding.facts.some((fact) => /Forecast window interpreted from question/i.test(fact)),
    'Interval transit finding should include interpreted forecast window'
  );

  const marriageIntervalFinding = await buildTransitIntervalToolFinding({
    kundli,
    question: marriageTwoYearQuery,
    referenceTimestamp,
  });

  assert.notEqual(
    marriageIntervalFinding.status,
    'unavailable',
    `Marriage interval transit finding should be available; got: ${marriageIntervalFinding.facts.join(' | ')}`
  );
  assert.ok(
    marriageIntervalFinding.facts.some((fact) => /2026-01-01T00:00:00.000Z/.test(fact) || /2027-12-31T23:59:59.000Z/.test(fact)),
    'Marriage interval finding should include the interpreted year window'
  );

  const noTimeIntervalFinding = await buildTransitIntervalToolFinding({
    kundli,
    question: noTimeQuery,
    referenceTimestamp,
  });

  assert.notEqual(
    noTimeIntervalFinding.status,
    'unavailable',
    `No-time interval transit finding should be available; got: ${noTimeIntervalFinding.facts.join(' | ')}`
  );
  assert.ok(
    noTimeIntervalFinding.facts.some((fact) => /2026-04-03T00:00:00.000Z/.test(fact)),
    'No-time timing query should anchor interval window at current reference timestamp'
  );

  const monthIntervalFinding = await buildTransitIntervalToolFinding({
    kundli,
    question: thisMonthQuery,
    referenceTimestamp,
  });

  assert.notEqual(
    monthIntervalFinding.status,
    'unavailable',
    `This-month interval transit finding should be available; got: ${monthIntervalFinding.facts.join(' | ')}`
  );
  assert.ok(
    monthIntervalFinding.facts.some((fact) => /2026-04-01T00:00:00.000Z/.test(fact) && /2026-04-30T23:59:59.000Z/.test(fact)),
    'This-month interval finding should include the full current month window'
  );

  const yearIntervalFinding = await buildTransitIntervalToolFinding({
    kundli,
    question: thisYearQuery,
    referenceTimestamp,
  });

  assert.notEqual(
    yearIntervalFinding.status,
    'unavailable',
    `This-year interval transit finding should be available; got: ${yearIntervalFinding.facts.join(' | ')}`
  );
  assert.ok(
    yearIntervalFinding.facts.some((fact) => /2026-01-01T00:00:00.000Z/.test(fact) && /2026-12-31T23:59:59.000Z/.test(fact)),
    'This-year interval finding should include the full current year window'
  );

  const hardshipMonthIntervalFinding = await buildTransitIntervalToolFinding({
    kundli,
    question: hardshipMonthYearQuery,
    referenceTimestamp,
  });

  assert.notEqual(
    hardshipMonthIntervalFinding.status,
    'unavailable',
    `Named month-year hardship interval finding should be available; got: ${hardshipMonthIntervalFinding.facts.join(' | ')}`
  );
  assert.ok(
    hardshipMonthIntervalFinding.facts.some((fact) => /2025-09-01T00:00:00.000Z/.test(fact) && /2025-09-30T23:59:59.000Z/.test(fact)),
    'Named month-year hardship interval finding should include exact September 2025 window'
  );

  const jupiterPointFinding = await buildTransitPointToolFinding({
    kundli,
    question: jupiterTransitQuery,
    referenceTimestamp,
  });

  assert.notEqual(
    jupiterPointFinding.status,
    'unavailable',
    `Jupiter transit point finding should be available; got: ${jupiterPointFinding.facts.join(' | ')}`
  );
  assert.ok(
    jupiterPointFinding.facts.some((fact) => /Transit reference timestamp used:\s*2026-04-03T00:00:00.000Z/i.test(fact)),
    'Jupiter transit point finding should expose the exact reference timestamp used for current-context interpretation'
  );
  assert.ok(
    jupiterPointFinding.facts.some((fact) => /Transit house contract:/i.test(fact)),
    'Transit point finding should include explicit ascendant-relative house contract to avoid sign-as-house confusion'
  );
  assert.ok(
    jupiterPointFinding.facts.some((fact) => /Transit Jupiter:.*house\s+\d+/i.test(fact)),
    'Transit point finding should include Jupiter house details in facts'
  );
  assert.ok(
    jupiterPointFinding.facts.some((fact) => /Transit Jupiter:.*ascendant-relative/i.test(fact)),
    'Transit point finding should expose ascendant-relative transit house derivation for Jupiter'
  );
  assert.ok(
    !jupiterPointFinding.facts.some((fact) => /^Transit\s+(Sun|Moon|Mars|Mercury|Jupiter|Venus|Saturn|Rahu|Ketu):.*backend\s+house_number/i.test(fact)),
    'Transit point finding should not leak backend house_number as user-facing transit house truth'
  );
  assert.ok(
    jupiterPointFinding.facts.some((fact) => /No explicit future\/past transit window detected/i.test(fact)),
    'Jupiter transit point finding should state that no explicit future/past window was requested'
  );

  const globalWithFetch = globalThis as typeof globalThis & { fetch: typeof fetch };
  const originalFetch = globalWithFetch.fetch;

  globalWithFetch.fetch = (async () => {
    const mockedPayload = {
      chart: {
        transit: {
          lagna: {
            Lg: { rashi: 2, degree: 8.07 },
          },
          graha: {
            Su: { rashi: 12, degree: 19, house_number: 4 },
            Sa: { rashi: 12, degree: 11, house_number: 4 },
            Ma: { rashi: 12, degree: 1, house_number: 4 },
            Me: { rashi: 11, degree: 15, house_number: 3 },
            Mo: { rashi: 7, degree: 9, house_number: 11 },
            Ju: { rashi: 3, degree: 21, house_number: 7 },
            Ve: { rashi: 1, degree: 10, house_number: 5 },
            Ra: { rashi: 11, degree: 6, house_number: 3 },
            Ke: { rashi: 5, degree: 6, house_number: 9 },
          },
        },
      },
    };

    return new Response(JSON.stringify(mockedPayload), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });
  }) as typeof fetch;

  try {
    const mockKundli = {
      ...kundli,
      min: 41,
    };

    const currentDetailsFinding = await buildTransitPointToolFinding({
      kundli: mockKundli,
      question: 'give me my current transit details',
      referenceTimestamp,
    });

    assert.equal(currentDetailsFinding.status, 'ok', 'Current transit details finding should be available with mocked transit payload');
    assert.ok(
      currentDetailsFinding.facts.some((fact) => /Transit Sun:.*house\s+11\s*\(ascendant-relative\)/i.test(fact)),
      'Sun in rashi 12 from lagna rashi 2 should resolve to ascendant-relative house 11'
    );
    assert.ok(
      currentDetailsFinding.facts.some((fact) => /Transit Moon:.*house\s+6\s*\(ascendant-relative\)/i.test(fact)),
      'Moon in rashi 7 from lagna rashi 2 should resolve to ascendant-relative house 6'
    );
    assert.ok(
      currentDetailsFinding.facts.some((fact) => /Transit Jupiter:.*house\s+2\s*\(ascendant-relative\)/i.test(fact)),
      'Jupiter in rashi 3 from lagna rashi 2 should resolve to ascendant-relative house 2'
    );
    assert.ok(
      currentDetailsFinding.facts.some((fact) => /Transit Venus:.*house\s+12\s*\(ascendant-relative\)/i.test(fact)),
      'Venus in rashi 1 from lagna rashi 2 should resolve to ascendant-relative house 12'
    );
    assert.ok(
      currentDetailsFinding.facts.some((fact) => /Transit Mars:.*house\s+11\s*\(ascendant-relative\)/i.test(fact)),
      'Mars in rashi 12 from lagna rashi 2 should resolve to ascendant-relative house 11'
    );
    assert.ok(
      currentDetailsFinding.facts.some((fact) => /Transit Mercury:.*house\s+10\s*\(ascendant-relative\)/i.test(fact)),
      'Mercury in rashi 11 from lagna rashi 2 should resolve to ascendant-relative house 10'
    );
    assert.ok(
      currentDetailsFinding.facts.some((fact) => /Transit Rahu:.*house\s+10\s*\(ascendant-relative\)/i.test(fact)),
      'Rahu in rashi 11 from lagna rashi 2 should resolve to ascendant-relative house 10'
    );
    assert.ok(
      currentDetailsFinding.facts.some((fact) => /Transit Ketu:.*house\s+4\s*\(ascendant-relative\)/i.test(fact)),
      'Ketu in rashi 5 from lagna rashi 2 should resolve to ascendant-relative house 4'
    );
    assert.ok(
      !currentDetailsFinding.facts.some((fact) => /^Transit\s+(Sun|Moon|Mars|Mercury|Jupiter|Venus|Saturn|Rahu|Ketu):.*backend\s+house_number/i.test(fact)),
      'Current transit details finding should not expose backend house_number metadata in user-facing facts'
    );

    const intervalDetailsFinding = await buildTransitIntervalToolFinding({
      kundli: mockKundli,
      question: 'Can I marry in 2026-2027?',
      referenceTimestamp,
    });

    assert.equal(
      intervalDetailsFinding.status,
      'ok',
      `Interval transit finding should be available with mocked payload; got: ${intervalDetailsFinding.facts.join(' | ')}`
    );
    assert.ok(
      intervalDetailsFinding.facts.some((fact) => /Transit interval trajectory\s*\(/i.test(fact)),
      'Interval transit finding should include deterministic interval trajectory summary'
    );
    assert.ok(
      intervalDetailsFinding.facts.some((fact) => /Potentially supportive window near/i.test(fact)),
      'Interval transit finding should include a deterministic supportive window checkpoint'
    );
  } finally {
    globalWithFetch.fetch = originalFetch;
  }

  const staleTransitAnswer = [
    'Your current transit details are not available in the canonical chart data. The Transit analyzer tool is required for this forecast, and it is currently unavailable.',
    '',
    'The reference time for your chart is 2026-04-03T21:55:05.535Z. For transit analysis, please use a dedicated transit forecast feature.',
  ].join('\n');

  const sanitizedTransitAnswer = sanitizeMissingDataContradictions(staleTransitAnswer, [
    {
      name: 'Transit analyzer',
      status: 'unavailable',
      facts: ['backend transit response missing'],
      evidencePaths: ['backend:/api/transit-chart'],
      missing: ['backend transit-chart payload'],
    } as any,
  ]);

  assert.equal(
    sanitizedTransitAnswer,
    'I could not fetch live transit from the backend right now. Please retry this question, and I will compute transit from your saved Kundli context.',
    'Stale canonical-chart transit fallback should be rewritten to a live-fetch retry message'
  );

  const staleTransitAnswerLiveForecast = [
    'Your current transit details are not available in the live forecast data. The Transit analyzer tool is required for this forecast, and it is currently unavailable.',
    '',
    'The reference time for your chart is 2026-04-03T22:06:43.502Z. For transit analysis, please use a dedicated transit forecast feature.',
  ].join('\n');

  const sanitizedTransitAnswerLiveForecast = sanitizeMissingDataContradictions(staleTransitAnswerLiveForecast, [
    {
      name: 'Transit analyzer',
      status: 'unavailable',
      facts: ['backend transit response missing'],
      evidencePaths: ['backend:/api/transit-chart'],
      missing: ['backend transit-chart payload'],
    } as any,
  ]);

  assert.equal(
    sanitizedTransitAnswerLiveForecast,
    'I could not fetch live transit from the backend right now. Please retry this question, and I will compute transit from your saved Kundli context.',
    'Stale live-forecast transit fallback should be rewritten to a live-fetch retry message'
  );

  const sanitizedTransitAnswerNoFindings = sanitizeMissingDataContradictions(
    staleTransitAnswerLiveForecast,
    []
  );

  assert.equal(
    sanitizedTransitAnswerNoFindings,
    'I could not fetch live transit from the backend right now. Please retry this question, and I will compute transit from your saved Kundli context.',
    'Stale transit template should be rewritten even when tool findings are absent'
  );

  const staticChartTransitFallback = [
    'I am Cozmic AI. Your current transit details require a live transit forecast, which I cannot fetch directly from the static chart data.',
    '',
    'To get your current transit details, please use the dedicated transit forecast feature in the app or ask a specific timing question (e.g., "How will Jupiter\'s transit affect me?"), and I can analyze it using your natal chart and dasha periods.',
  ].join('\n');

  const sanitizedStaticChartTransitFallback = sanitizeMissingDataContradictions(staticChartTransitFallback, []);

  assert.equal(
    sanitizedStaticChartTransitFallback,
    'I could not fetch live transit from the backend right now. Please retry this question, and I will compute transit from your saved Kundli context.',
    'Static-chart transit refusal template should be rewritten to live-fetch retry message'
  );

  const mixedTransitFallbackLeak = [
    'I am Cozmic AI. Your current transit details cannot be provided because a',
    '',
    "Please try again or use the app's transit feature for a real-time snapshot.",
    '',
    'I could not fetch live transit from the backend right now. Please retry this question, and I will compute transit from your saved Kundli context.',
  ].join('\n');

  const sanitizedMixedTransitFallbackLeak = sanitizeMissingDataContradictions(mixedTransitFallbackLeak, []);

  assert.equal(
    sanitizedMixedTransitFallbackLeak,
    'I could not fetch live transit from the backend right now. Please retry this question, and I will compute transit from your saved Kundli context.',
    'Mixed transit fallback leakage should collapse to a single clean live-fetch retry message'
  );

  console.log('transit regression tests passed');
}

main().catch((error) => {
  console.error('transit regression tests failed', error);
  process.exitCode = 1;
});
