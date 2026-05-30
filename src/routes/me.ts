import { Router } from 'express';
import { requireFirebaseAuth } from '../middleware/auth.js';
import type { AuthUserDocument } from '../models/firestoreModels.js';
import { getAuthUsersRepository } from '../repositories/authUsersRepository.js';
import { getPostgresPool } from '../services/postgresClient.js';
import { getFirebaseClients } from '../firebase/admin.js';

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

router.delete('/v1/me', requireFirebaseAuth, async (req, res) => {
  const pool = getPostgresPool();
  if (!pool) {
    return res.status(500).json({ error: 'Database connection not available' });
  }

  const ownerId = req.user!.uid;

  try {
    // Delete in FK-safe order (child tables first)
    // documents (JSONB store, no FK)
    await pool.query(
      `DELETE FROM documents WHERE collection IN ('rag_profiles', 'chat_sessions', 'chat_messages', 'auth_users') AND data->>'ownerId' = $1`,
      [ownerId]
    );

    // chat_messages (FK -> chat_sessions, delete explicitly for safety)
    await pool.query(`DELETE FROM chat_messages WHERE owner_id = $1`, [ownerId]);

    // chat_sessions
    await pool.query(`DELETE FROM chat_sessions WHERE owner_id = $1`, [ownerId]);

    // rag_profiles
    await pool.query(`DELETE FROM rag_profiles WHERE owner_id = $1`, [ownerId]);

    // chart_jobs
    await pool.query(`DELETE FROM chart_jobs WHERE owner_id = $1`, [ownerId]);

    // charts
    await pool.query(`DELETE FROM charts WHERE owner_id = $1`, [ownerId]);

    // monthly_usage_counters
    await pool.query(`DELETE FROM monthly_usage_counters WHERE owner_id = $1`, [ownerId]);

    // subscriptions
    await pool.query(`DELETE FROM subscriptions WHERE owner_id = $1`, [ownerId]);

    // auth_users (root record, deleted last)
    await pool.query(`DELETE FROM auth_users WHERE owner_id = $1`, [ownerId]);

    // Firebase Auth
    try {
      const { auth } = getFirebaseClients();
      await auth.deleteUser(ownerId);
    } catch {
      // User may not exist in Firebase Auth — non-fatal
    }

    return res.json({ ok: true });
  } catch (error) {
    return res.status(500).json({ error: 'Failed to delete account', details: String(error) });
  }
});

export default router;