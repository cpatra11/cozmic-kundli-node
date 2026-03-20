import dotenv from 'dotenv';
import { GoogleAuth } from 'google-auth-library';
import * as fs from 'fs';
import * as path from 'path';

dotenv.config();

// Write service account to temp file for GoogleAuth to read
const sa = JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT_JSON);
const tempFile = '/tmp/firebase-sa.json';
fs.writeFileSync(tempFile, JSON.stringify(sa));

// Create GoogleAuth instance
const auth = new GoogleAuth({
  keyFile: tempFile,
  scopes: ['https://www.googleapis.com/auth/cloud-platform'],
});

// Get authenticated client
const client = await auth.getClient();

// Get access token
const token = await client.getAccessToken();
const accessToken = token.token;
console.log('Token obtained:', accessToken.substring(0, 50) + '...');

// Clean up
fs.unlinkSync(tempFile);

// Make REST API call to convert database to NATIVE_MODE
const projectId = process.env.FIREBASE_PROJECT_ID;
const databaseId = 'default';
const url = `https://firestore.googleapis.com/v1/projects/${projectId}/databases/${databaseId}?updateMask=type`;

const response = await fetch(url, {
  method: 'PATCH',
  headers: {
    'Authorization': `Bearer ${accessToken}`,
    'Accept': 'application/json',
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({ type: 1 }),
});

const data = await response.json();
console.log('Response status:', response.status);
console.log('Response:', JSON.stringify(data, null, 2));

if (!response.ok) {
  process.exit(1);
}
