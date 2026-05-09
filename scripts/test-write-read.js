import dotenv from 'dotenv';
import { promisify } from 'util';
import { exec } from 'child_process';
const execAsync = promisify(exec);

dotenv.config();
const projectId = process.env.FIREBASE_PROJECT_ID;
const databaseId = 'default';

const { stdout } = await execAsync('gcloud auth print-access-token');
const accessToken = stdout.trim();

const testDocId = 'test_' + Date.now();
const createUrl = `https://firestore.googleapis.com/v1/projects/${projectId}/databases/${databaseId}/documents`;

console.log('Writing test document via REST API...');

// Create document
const createResponse = await fetch(`${createUrl}/_connectivity?documentId=${testDocId}`, {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${accessToken}`,
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    fields: {
      ok: { booleanValue: true },
      timestamp: { stringValue: new Date().toISOString() },
    },
  }),
});

const createData = await createResponse.json();
console.log('Write Status:', createResponse.status);
console.log('Write Response:', JSON.stringify(createData, null, 2));

if (!createResponse.ok) {
  console.error('✗ Write failed');
  process.exit(1);
}

console.log('\n✓ Write successful!');

// Read document
console.log('\nReading test document via REST API...');
const getUrl = `${createUrl}/_connectivity/${testDocId}`;

const getResponse = await fetch(getUrl, {
  headers: { 'Authorization': `Bearer ${accessToken}` },
});

const getData = await getResponse.json();
console.log('Read Status:', getResponse.status);
console.log('Read Response:', JSON.stringify(getData, null, 2));

if (getResponse.ok) {
  console.log('\n✓ Read successful!');
  console.log(`\n✓ Firestore Native mode is FULLY OPERATIONAL!`);
} else {
  console.error('✗ Read failed');
  process.exit(1);
}
