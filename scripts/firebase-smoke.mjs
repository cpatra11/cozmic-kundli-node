import dotenv from 'dotenv';
import { existsSync } from 'node:fs';

dotenv.config({ path: new URL('../.env', import.meta.url) });

const hasProjectId = Boolean(process.env.FIREBASE_PROJECT_ID?.trim());
const hasServiceAccountJson = Boolean(process.env.FIREBASE_SERVICE_ACCOUNT_JSON?.trim());
const serviceAccountPath = process.env.FIREBASE_SERVICE_ACCOUNT_PATH?.trim() || '';
const hasServiceAccountPath = Boolean(serviceAccountPath) && existsSync(serviceAccountPath);
const hasGoogleApplicationCredentials = Boolean(process.env.GOOGLE_APPLICATION_CREDENTIALS?.trim());

const summary = {
  hasProjectId,
  hasServiceAccountJson,
  hasServiceAccountPath,
  hasGoogleApplicationCredentials,
};

try {
  const { getFirebaseClients } = await import('../dist/firebase/admin.js');
  const { auth } = getFirebaseClients();
  const token = await auth.createCustomToken(`smoke_${Date.now()}`);

  console.log(
    JSON.stringify(
      {
        ...summary,
        initStatus: 'ok',
        createCustomTokenStatus: token ? 'ok' : 'empty',
      },
      null,
      2
    )
  );
} catch (error) {
  console.log(
    JSON.stringify(
      {
        ...summary,
        initStatus: 'failed',
        error: String(error),
      },
      null,
      2
    )
  );
  process.exit(1);
}
