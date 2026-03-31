import { Router } from 'express';
import { requireFirebaseAuth } from '../middleware/auth.js';
import type { AuthUserDocument } from '../models/firestoreModels.js';
import { getAuthUsersRepository } from '../repositories/authUsersRepository.js';

const router = Router();

function normalizeEmail(value: string | undefined | null) {
  const normalized = value?.trim().toLowerCase();
  return normalized ? normalized : undefined;
}

function normalizePhoneNumber(value: string | undefined | null) {
  const normalized = value?.trim();
  return normalized ? normalized : undefined;
}

router.get('/v1/me', requireFirebaseAuth, async (req, res) => {
  try {
    const authUsers = getAuthUsersRepository();
    const now = Date.now();
    const ownerId = req.user!.uid;
    const existing = await authUsers.getByOwnerId(ownerId);
    const email = normalizeEmail(req.user?.email ?? existing?.email);
    const phoneNumber = normalizePhoneNumber(req.user?.phoneNumber ?? existing?.phoneNumber);

    const profile: AuthUserDocument = {
      ownerId,
      email,
      phoneNumber,
      provider: 'firebase',
      createdAt: existing?.createdAt ?? now,
      updatedAt: now,
      lastSeenAt: now,
    };

    try {
      await authUsers.upsert(profile);
    } catch (error) {
      const code = error instanceof Error ? (error as Error & { code?: string }).code : undefined;
      if (code === '23505') {
        return res.status(409).json({
          error: 'Email address or phone number already exists',
          details: 'Use a different email or phone number.',
        });
      }

      throw error;
    }

    return res.json({
      user: {
        uid: ownerId,
        email: profile.email ?? null,
        phoneNumber: profile.phoneNumber ?? null,
      },
      profile,
    });
  } catch (error) {
    return res.status(500).json({ error: 'Failed to load Firebase auth profile', details: String(error) });
  }
});

export default router;