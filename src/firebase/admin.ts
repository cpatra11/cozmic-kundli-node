import fs from 'node:fs';
import { initializeApp, getApps, cert, applicationDefault, type App } from 'firebase-admin/app';
import { getAuth } from 'firebase-admin/auth';
import { env } from '../config/env.js';

let firebaseApp: App | null = null;

function readServiceAccount(): Record<string, unknown> | null {
  if (env.FIREBASE_SERVICE_ACCOUNT_JSON?.trim()) {
    try {
      return JSON.parse(env.FIREBASE_SERVICE_ACCOUNT_JSON);
    } catch {
      throw new Error('FIREBASE_SERVICE_ACCOUNT_JSON is not valid JSON');
    }
  }

  if (env.FIREBASE_SERVICE_ACCOUNT_PATH?.trim()) {
    const raw = fs.readFileSync(env.FIREBASE_SERVICE_ACCOUNT_PATH, 'utf8');
    return JSON.parse(raw);
  }

  return null;
}

export function getFirebaseApp(): App {
  if (firebaseApp) return firebaseApp;

  if (getApps().length > 0) {
    firebaseApp = getApps()[0]!;
    return firebaseApp;
  }

  const serviceAccount = readServiceAccount();
  firebaseApp = initializeApp({
    credential: serviceAccount
      ? cert(serviceAccount as Parameters<typeof cert>[0])
      : applicationDefault(),
    projectId: env.FIREBASE_PROJECT_ID,
    storageBucket: env.FIREBASE_STORAGE_BUCKET,
  });

  return firebaseApp;
}

export function getFirebaseClients() {
  const app = getFirebaseApp();
  return {
    auth: getAuth(app),
  };
}
