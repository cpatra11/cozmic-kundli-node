import dotenv from 'dotenv';
import { GoogleAuth } from 'google-auth-library';
import * as fs from 'fs';

dotenv.config();

const sa = JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT_JSON);
const projectId = process.env.FIREBASE_PROJECT_ID;

console.log('Converting to Native mode...');

const databaseId = 'default';
const url = `https://firestore.googleapis.com/v1/projects/${projectId}/databases/${databaseId}?updateMask=type`;

console.log(`Attempting conversion for project: ${projectId}`);

// Get auth token
const tempFile = '/tmp/firebase-sa.json';
fs.writeFileSync(tempFile, JSON.stringify(sa));

const auth = new GoogleAuth({
  keyFile: tempFile,
  scopes: ['https://www.googleapis.com/auth/cloud-platform'],
});

const client = await auth.getClient();
const token = await client.getAccessToken();
const accessToken = token.token;

console.log('Token obtained');

const response = await fetch(url, {
  method: 'PATCH',
  headers: {
    'Authorization': `Bearer ${accessToken}`,
    'Accept': 'application/json',
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({ type: 1 }), // 1 = FIRESTORE_NATIVE
});

const data = await response.json();
console.log('\nResponse status:', response.status);
console.log('Response:', JSON.stringify(data, null, 2));

fs.unlinkSync(tempFile);

if (response.status === 200 || response.status === 201) {
  console.log('\n✓ Database conversion to Native mode initiated successfully!');
} else if (response.status === 400 && data.error?.message?.includes('empty')) {
  console.error('\n✗ Database must be empty before conversion.');
  console.error('To clear all Datastore entities:');
  console.error('1. Go to Firebase Console → Datastore Mode');
  console.error('2. Manually delete all entities/collections, OR');
  console.error('3. Create a new Firebase project in Native mode');
  process.exit(1);
} else if (response.status === 403) {
  console.error('\n✗ Permission denied. Service account lacks required IAM role.');
  console.error('Contact your Firebase project owner to grant this service account:');
  console.error('- Cloud Datastore Owner role, OR');
  console.error('- Create a new project where this SA has Owner permissions');
  process.exit(1);
} else {
  console.error('\n✗ Conversion failed');
  process.exit(1);
}
