import dotenv from 'dotenv';
import admin from 'firebase-admin';

dotenv.config({ path: '.env' });

const serviceAccount = JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT_JSON);
admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
  projectId: process.env.FIREBASE_PROJECT_ID,
  storageBucket: process.env.FIREBASE_STORAGE_BUCKET,
});

async function getIdToken() {
  const uid = `rag-test-${Date.now()}`;
  const customToken = await admin.auth().createCustomToken(uid);
  const tokenResponse = await fetch(
    `https://identitytoolkit.googleapis.com/v1/accounts:signInWithCustomToken?key=${process.env.FIREBASE_API_KEY}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token: customToken, returnSecureToken: true }),
    }
  );

  if (!tokenResponse.ok) {
    throw new Error(await tokenResponse.text());
  }

  const body = await tokenResponse.json();
  return body.idToken;
}

async function main() {
  const idToken = await getIdToken();

  const ingestResponse = await fetch('http://localhost:8787/v1/rag/ingest', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${idToken}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      profileId: 'demo-profile',
      kundli: {
        latitude: 28.6139,
        longitude: 77.209,
        year: 1994,
        month: 7,
        day: 14,
        hour: 10,
        min: 30,
        sec: 0,
        time_zone: '+05:30',
      },
    }),
  });

  const ingestBody = await ingestResponse.json();
  console.log('Ingest status:', ingestResponse.status);
  console.log('Ingest body:', JSON.stringify(ingestBody, null, 2));

  const queryResponse = await fetch('http://localhost:8787/v1/rag/query', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${idToken}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      profileId: 'demo-profile',
      message: 'What can you tell me about my moon and career pattern?',
      topK: 5,
    }),
  });

  const queryBody = await queryResponse.json();
  console.log('Query status:', queryResponse.status);
  console.log('Query chunks:', queryBody.chunks?.length ?? 0);
  console.log('Query body:', JSON.stringify(queryBody, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
