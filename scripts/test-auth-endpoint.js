import dotenv from 'dotenv';
import admin from 'firebase-admin';

dotenv.config({ path: '.env' });

// Initialize Admin SDK
const serviceAccount = JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT_JSON);
admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
  projectId: process.env.FIREBASE_PROJECT_ID,
  storageBucket: process.env.FIREBASE_STORAGE_BUCKET,
});

const auth = admin.auth();

async function testAuthEndpoint() {
  try {
    console.log('Creating custom JWT token...');
    
    const uid = 'test-user-' + Date.now();
    const customToken = await auth.createCustomToken(uid);
    console.log('✓ Custom token created');

    // Exchange custom token for ID token via Firebase REST API
    console.log('\nExchanging custom token for ID token...');
    const tokenResponse = await fetch(
      `https://identitytoolkit.googleapis.com/v1/accounts:signInWithCustomToken?key=${process.env.FIREBASE_API_KEY}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token: customToken, returnSecureToken: true }),
      }
    );

    if (!tokenResponse.ok) {
      const error = await tokenResponse.json();
      throw new Error(`Failed to exchange token: ${JSON.stringify(error)}`);
    }

    const { idToken } = await tokenResponse.json();
    console.log('✓ ID token obtained:', idToken.substring(0, 50) + '...');

    // Now test the backend endpoint with this ID token
    console.log('\nTesting GET /v1/chat/sessions with ID token...');
    const response = await fetch('http://localhost:8787/v1/chat/sessions', {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${idToken}`,
      },
    });

    console.log('Response Status:', response.status);
    const data = await response.json();
    console.log('Response:', JSON.stringify(data, null, 2));

    if (response.status === 200) {
      console.log('\n✓ AUTH ENDPOINT TEST PASSED - Backend fully functional!');
    } else {
      console.log('\n✗ Unexpected status code');
    }
  } catch (error) {
    console.error('\n✗ Error:', error.message);
  } finally {
    process.exit(0);
  }
}

testAuthEndpoint();
