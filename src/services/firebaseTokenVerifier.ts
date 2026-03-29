import { getFirebaseClients } from '../firebase/admin.js';

export interface VerifiedFirebaseToken {
  uid: string;
  email?: string;
  phoneNumber?: string;
  name?: string;
  signInProvider?: string;
}

export async function verifyFirebaseIdToken(token: string): Promise<VerifiedFirebaseToken> {
  const { auth } = getFirebaseClients();
  const decoded = await auth.verifyIdToken(token, true);

  return {
    uid: decoded.uid,
    email: decoded.email,
    phoneNumber: decoded.phone_number,
    name: decoded.name,
    signInProvider: decoded.firebase?.sign_in_provider,
  };
}