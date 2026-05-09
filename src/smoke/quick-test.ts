import { runKundliAgentV2 } from '../services/newAgent.js';

async function main() {
  const sessionId = `test_${Date.now()}`;
  const kundli = {
    latitude: 28.6139,
    longitude: 77.209,
    year: 1995,
    month: 6,
    day: 15,
    hour: 14,
    min: 30,
    sec: 0,
    time_zone: '5.5',
  };

  const r = await runKundliAgentV2({
    message: 'Analyse my d60 chart',
    mode: 'pro',
    ownerId: 'test-owner',
    profileId: 'test-profile',
    sessionId,
    kundli,
  });
  console.log('=== ANSWER ===');
  console.log(r.answer);
}

main().catch(console.error);
