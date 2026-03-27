import { getFirebaseClients } from '../firebase/admin.js';

export interface VerifiedFirebaseToken {
  uid: string;
  email?: string;
}

export async function verifyFirebaseIdToken(token: string): Promise<VerifiedFirebaseToken> {
  const { auth } = getFirebaseClients();
  const decoded = await auth.verifyIdToken(token, true);

  return {
    uid: decoded.uid,
    email: decoded.email,
  };
}