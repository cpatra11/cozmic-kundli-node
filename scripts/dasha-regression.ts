import assert from 'node:assert/strict';
import { fetchCalculatedChart } from '../src/services/be1Client.js';
import { shouldFetchDeepDasha } from '../src/services/kundliAgent.js';
import { buildChartSnapshot, extractChartSchemaInfo } from '../src/services/chartSnapshot.js';

async function main(): Promise<void> {
  const legacyDeepPayload = {
    chart: {
      dasha: {
        nesting: 0,
        type: 'vimshottari',
        key: '',
        periods: {
          Ju: {
            nesting: 1,
            type: 'mahadasha',
            key: 'Ju',
            periods: {
              Sa: {
                nesting: 2,
                type: 'antardasha',
                key: 'JuSa',
              },
            },
          },
        },
      },
    },
  };

  const legacySchema = extractChartSchemaInfo(legacyDeepPayload);
  assert.equal(legacySchema.chartSchemaVersion, 'legacy-deep-dasha', 'Legacy nested payload should be tagged as legacy deep dasha');
  assert.ok(legacySchema.dashaDepth > 1, 'Legacy nested payload should report depth greater than 1');

  const shallowSnapshot = buildChartSnapshot(legacyDeepPayload) as typeof legacyDeepPayload;
  assert.equal(
    shallowSnapshot.chart.dasha.periods.Ju.periods,
    undefined,
    'Normalized snapshot should strip nested dasha periods from the mahadasha branch'
  );

  const routingCases = [
    { question: 'What is my current mahadasha?', expected: false },
    { question: 'Which dasha am I in now?', expected: false },
    { question: 'Show me the current dasha only', expected: false },
    { question: 'When will I get married?', expected: true },
    { question: 'Tell me the timing for career promotion', expected: true },
    { question: 'Will my finances improve in 2027?', expected: true },
  ];

  for (const testCase of routingCases) {
    assert.equal(
      shouldFetchDeepDasha(testCase.question),
      testCase.expected,
      `Unexpected deep-dasha routing for: ${testCase.question}`
    );
  }

  const globalWithFetch = globalThis as typeof globalThis & { fetch: typeof fetch };
  const originalFetch = globalWithFetch.fetch;
  const calls: Array<{ url: string; init?: RequestInit }> = [];

  globalWithFetch.fetch = (async (input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
    calls.push({ url, init });
    return new Response(JSON.stringify({ ok: true, url }), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });
  }) as typeof fetch;

  try {
    const kundli = {
      latitude: 20.9501,
      longitude: 85.2168,
      year: 2002,
      month: 3,
      day: 14,
      hour: 10,
      min: 39,
      sec: 0,
      time_zone: '+05:30',
    };

    await fetchCalculatedChart(kundli);
    assert.equal(calls.length, 1, 'Expected one BE1 request for default chart fetch');

    const firstUrl = new URL(calls[0]!.url);
    assert.equal(firstUrl.searchParams.get('nesting'), '1', 'Default chart fetch should request mahadasha-only payload');
    assert.equal(firstUrl.searchParams.get('period_key'), null, 'Default chart fetch should not include a period key');

    await fetchCalculatedChart(kundli, { nesting: 5, periodKey: 'Ju' });
    assert.equal(calls.length, 2, 'Expected two BE1 requests after explicit deep fetch');

    const secondUrl = new URL(calls[1]!.url);
    assert.equal(secondUrl.searchParams.get('nesting'), '5', 'Explicit deep fetch should keep requested nesting');
    assert.equal(secondUrl.searchParams.get('period_key'), 'Ju', 'Explicit deep fetch should forward the requested period key');
  } finally {
    globalWithFetch.fetch = originalFetch;
  }

  console.log('dasha regression tests passed');
}

main().catch((error) => {
  console.error('dasha regression tests failed', error);
  process.exitCode = 1;
});