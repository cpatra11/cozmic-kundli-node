import type { Request, Response, NextFunction } from 'express';
import { verifyFirebaseIdToken } from '../services/firebaseTokenVerifier.js';

declare module 'express-serve-static-core' {
  interface Request {
    user?: {
      uid: string;
      email?: string;
      phoneNumber?: string;
      name?: string;
      signInProvider?: string;
    };
  }
}

export async function requireFirebaseAuth(req: Request, res: Response, next: NextFunction) {
  try {
    const authHeader = req.headers.authorization;
    const token = authHeader?.startsWith('Bearer ') ? authHeader.slice(7) : null;

    if (!token) {
      return res.status(401).json({ error: 'Missing Bearer token' });
    }

    const decoded = await verifyFirebaseIdToken(token);

    req.user = {
      uid: decoded.uid,
      email: decoded.email,
      phoneNumber: decoded.phoneNumber,
      name: decoded.name,
      signInProvider: decoded.signInProvider,
    };

    return next();
  } catch (error) {
    return res.status(401).json({ error: 'Invalid Firebase token', details: String(error) });
  }
}
