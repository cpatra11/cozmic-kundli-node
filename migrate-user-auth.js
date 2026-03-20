import dotenv from 'dotenv';

dotenv.config();

const projectId = process.env.FIREBASE_PROJECT_ID;

console.log('Converting to Native mode using authenticated user...');

// Get user's access token
import { promisify } from 'util';
import { exec } from 'child_process';
const execAsync = promisify(exec);

const { stdout } = await execAsync('gcloud auth print-access-token');
const accessToken = stdout.trim();

console.log('User token obtained');

const databaseId = 'default';
const url = `https://firestore.googleapis.com/v1/projects/${projectId}/databases/${databaseId}?updateMask=type`;

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

if (response.status === 200 || response.status === 201) {
  console.log('\n✓ Database conversion to Native mode initiated successfully!');
  console.log('Note: Conversion may take a few minutes. Check Firebase Console to verify.');
} else if (response.status === 400 && data.error?.message?.includes('empty')) {
  console.error('\n✗ Database must be empty before conversion.');
  console.error('Please delete all Datastore entities from Firebase Console and try again.');
  process.exit(1);
} else {
  console.error('\n✗ Conversion failed');
  process.exit(1);
}
