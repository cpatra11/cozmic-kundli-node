import dotenv from 'dotenv';
import { promisify } from 'util';
import { exec } from 'child_process';
const execAsync = promisify(exec);

dotenv.config();
const projectId = process.env.FIREBASE_PROJECT_ID;
const databaseId = 'default';

// Get user's access token
const { stdout } = await execAsync('gcloud auth print-access-token');
const accessToken = stdout.trim();

// Test via REST API
const url = `https://firestore.googleapis.com/v1/projects/${projectId}/databases/${databaseId}/documents/_health`;

console.log(`Testing Firestore API for project: ${projectId}`);

const response = await fetch(url, {
  headers: { 'Authorization': `Bearer ${accessToken}` },
});

const data = await response.json();
console.log('Status:', response.status);
console.log('Response:', JSON.stringify(data, null, 2));

if (response.ok || response.status === 404) {
  console.log('\n✓ Firestore API is responding!');
} else {
  console.log('\n✗ API error');
}
