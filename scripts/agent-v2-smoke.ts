import assert from 'node:assert/strict';
import { runKundliAgentV2 } from '../src/services/newAgent.js';

const SAMPLE_KUNDLI = {
  latitude: 28.6139,
  longitude: 77.2090,
  year: 1990,
  month: 6,
  day: 15,
  hour: 10,
  min: 30,
  sec: 0,
  time_zone: 'Asia/Kolkata',
};

async function main() {
  // 1. Smalltalk
  const { answer: st } = await runKundliAgentV2({ message: 'Hi there!', mode: 'mini' });
  assert.ok(st.length > 0, 'smalltalk answer non-empty');
  console.log(`[PASS] smalltalk: ${st.slice(0, 60)}...`);

  // 2. Identity / conceptual
  const { answer: id } = await runKundliAgentV2({ message: 'What is Vedic astrology?', mode: 'mini' });
  assert.ok(id.length > 0, 'identity answer non-empty');
  console.log(`[PASS] general_astro: ${id.slice(0, 60)}...`);

  // 3. Mini blocked: pro keyword triggers upgrade response
  const { answer: blocked } = await runKundliAgentV2({ message: 'What is my dasha period?', mode: 'mini' });
  assert.ok(blocked.length > 0, 'blocked answer non-empty');
  assert.ok(
    /pro|upgrade|mini plan/i.test(blocked),
    `mini blocked should mention upgrade. Got: ${blocked.slice(0, 100)}`
  );
  console.log(`[PASS] mini blocked dasha: ${blocked.slice(0, 60)}...`);

  // 4. Pipeline with kundli (hits BE1)
  const { answer: chart } = await runKundliAgentV2({
    message: 'Tell me about my chart planets',
    mode: 'mini',
    kundli: SAMPLE_KUNDLI,
  });
  assert.ok(chart.length > 0, 'chart answer non-empty');
  assert.ok(
    !/missing data|unavailable|consult a professional/i.test(chart),
    'no missing data language in answer'
  );
  console.log(`[PASS] chart mini (kundli): ${chart.slice(0, 80)}...`);

  // 5. Pro pipeline with transit
  const { answer: pro } = await runKundliAgentV2({
    message: 'How is my current transit?',
    mode: 'pro',
    kundli: SAMPLE_KUNDLI,
  });
  assert.ok(pro.length > 0, 'pro answer non-empty');
  assert.ok(!/missing data|unavailable|consult a professional/i.test(pro), 'no missing data language');
  console.log(`[PASS] pro transit: ${pro.slice(0, 80)}...`);

  console.log('\n=== ALL V2 SMOKE TESTS PASSED ===');
}

main().catch((err) => {
  console.error('V2 SMOKE TEST FAILED:', err);
  process.exitCode = 1;
});
