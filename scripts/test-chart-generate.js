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
  const uid = `chart-test-${Date.now()}`;
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

  const response = await fetch('http://localhost:8787/v1/chart/generate', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${idToken}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      name: 'Test User',
      place: 'Bhubaneswar',
      latitude: 20.2961,
      longitude: 85.8245,
      year: 1994,
      month: 7,
      day: 14,
      hour: 10,
      min: 30,
      sec: 0,
      time_zone: '+05:30',
      ayanamsha: 'Lahiri',
    }),
  });

  const data = await response.json();
  console.log('Status:', response.status);
  console.log('Body keys:', Object.keys(data));
  if (data.error) {
    console.log('Error:', data.error);
  }
  if (data.details) {
    console.log('Details:', data.details);
  }
  if (data.ingestionStatus) {
    console.log('Ingestion status:', data.ingestionStatus);
  }
  if (data.ingestionError) {
    console.log('Ingestion error:', data.ingestionError);
  }
  console.log('Profile ID:', data.profileId);
  console.log('Ingestion:', data.ingestion);
  console.log('Has chartData.chart:', Boolean(data.chartData?.chart));
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
