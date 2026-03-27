import { Router } from 'express';
import { requireFirebaseAuth } from '../middleware/auth.js';
import { COLLECTIONS, type AuthUserDocument } from '../models/firestoreModels.js';
import { getPostgresStore } from '../services/postgresStore.js';

const router = Router();

router.get('/v1/me', requireFirebaseAuth, async (req, res) => {
  try {
    const store = getPostgresStore();
    const now = Date.now();
    const ownerId = req.user!.uid;
    const path = `${COLLECTIONS.authUsers}/${ownerId}`;
    const existing = await store.getDocument<AuthUserDocument>(path);

    const profile: AuthUserDocument = {
      ownerId,
      email: req.user?.email ?? existing?.data.email,
      provider: 'firebase',
      createdAt: existing?.data.createdAt ?? now,
      updatedAt: now,
      lastSeenAt: now,
    };

    await store.setDocument(path, profile, true);

    return res.json({
      user: {
        uid: ownerId,
        email: profile.email ?? null,
      },
      profile,
    });
  } catch (error) {
    return res.status(500).json({ error: 'Failed to load Firebase auth profile', details: String(error) });
  }
});

export default router;