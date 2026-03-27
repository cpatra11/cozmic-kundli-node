import { Router } from 'express';
import { requireFirebaseAuth } from '../middleware/auth.js';
import { COLLECTIONS, type RagApiSourceDocument, type RagChunkDocument, type RagProfileDocument } from '../models/firestoreModels.js';
import { buildChartSnapshot } from '../services/chartSnapshot.js';
import { getPostgresStore } from '../services/postgresStore.js';

const router = Router();

function buildKundaliSummary(doc: RagProfileDocument, id: string) {
  return {
    id,
    kundaliId: doc.profileId,
    name: doc.displayName ?? doc.profileId,
    displayName: doc.displayName ?? doc.profileId,
    place: doc.place,
    createdAt: doc.createdAt,
    updatedAt: doc.updatedAt,
    kundliInput: doc.kundliInput,
  };
}

router.get('/v1/kundalis', requireFirebaseAuth, async (req, res) => {
  try {
    const store = getPostgresStore();
    const rows = await store.runQuery<RagProfileDocument>(
      COLLECTIONS.ragProfiles,
      [{ field: 'ownerId', op: 'EQUAL', value: req.user!.uid }],
      {
        orderBy: [{ field: 'updatedAt', direction: 'DESCENDING' }],
        limit: 100,
      }
    );

    return res.json({
      kundalis: rows.map((doc) => buildKundaliSummary(doc.data, doc.id)),
    });
  } catch (error) {
    return res.status(500).json({ error: 'Failed to list kundalis', details: String(error) });
  }
});

router.get('/v1/kundalis/:kundaliId', requireFirebaseAuth, async (req, res) => {
  try {
    const identifier = String(req.params.kundaliId ?? '').trim();
    if (!identifier) {
      return res.status(400).json({ error: 'Missing kundaliId path parameter' });
    }

    const store = getPostgresStore();

    const directPath = `${COLLECTIONS.ragProfiles}/${req.user!.uid}__${identifier}`;
    let profileDoc = await store.getDocument<RagProfileDocument>(directPath);

    if (!profileDoc) {
      const matches = await store.runQuery<RagProfileDocument>(
        COLLECTIONS.ragProfiles,
        [
          { field: 'ownerId', op: 'EQUAL', value: req.user!.uid },
          { field: 'profileId', op: 'EQUAL', value: identifier },
        ],
        { limit: 1 }
      );
      profileDoc = matches[0] ?? null;
    }

    if (!profileDoc) {
      return res.status(404).json({ error: 'Kundali not found' });
    }

    const sourceDoc = await store.getDocument<RagApiSourceDocument>(
      `${COLLECTIONS.ragApiSources}/${profileDoc.data.latestSourceDocId}`
    );

    const rawPayload = sourceDoc?.data.rawPayload;
    const chartData = sourceDoc?.data.chartSnapshot ?? buildChartSnapshot(rawPayload);

    return res.json({
      kundali: {
        id: profileDoc.id,
        kundaliId: profileDoc.data.profileId,
        ownerId: profileDoc.data.ownerId,
        name: profileDoc.data.displayName ?? profileDoc.data.profileId,
        displayName: profileDoc.data.displayName ?? profileDoc.data.profileId,
        place: profileDoc.data.place,
        chartData,
        rawPayload,
        createdAt: profileDoc.data.createdAt,
        updatedAt: profileDoc.data.updatedAt,
        kundliInput: profileDoc.data.kundliInput,
        chartVersion: profileDoc.data.chartVersion,
        latestSourceDocId: profileDoc.data.latestSourceDocId,
      },
    });
  } catch (error) {
    return res.status(500).json({ error: 'Failed to load kundali', details: String(error) });
  }
});

router.patch('/v1/kundalis/:kundaliId', requireFirebaseAuth, async (req, res) => {
  try {
    const identifier = String(req.params.kundaliId ?? '').trim();
    if (!identifier) {
      return res.status(400).json({ error: 'Missing kundaliId path parameter' });
    }

    const displayName = typeof req.body?.displayName === 'string' ? req.body.displayName.trim() : '';
    const place = typeof req.body?.place === 'string' ? req.body.place.trim() : '';

    const store = getPostgresStore();
    const matches = await store.runQuery<RagProfileDocument>(
      COLLECTIONS.ragProfiles,
      [
        { field: 'ownerId', op: 'EQUAL', value: req.user!.uid },
        { field: 'profileId', op: 'EQUAL', value: identifier },
      ],
      { limit: 1 }
    );

    const profileDoc = matches[0];
    if (!profileDoc) {
      return res.status(404).json({ error: 'Kundali not found' });
    }

    const profilePatch: Record<string, unknown> = {
      updatedAt: Date.now(),
    };
    if (displayName) profilePatch.displayName = displayName;
    if (place) profilePatch.place = place;

    await store.setDocument(`${COLLECTIONS.ragProfiles}/${profileDoc.id}`, profilePatch, true);

    const sourceDocs = await store.runQuery<RagApiSourceDocument>(
      COLLECTIONS.ragApiSources,
      [
        { field: 'ownerId', op: 'EQUAL', value: req.user!.uid },
        { field: 'profileId', op: 'EQUAL', value: profileDoc.data.profileId },
      ]
    );

    await Promise.all(
      sourceDocs.map((doc) =>
        store.setDocument(
          `${COLLECTIONS.ragApiSources}/${doc.id}`,
          {
            ...(displayName ? { displayName } : {}),
            ...(place ? { place } : {}),
          },
          true
        )
      )
    );

    return res.json({
      ok: true,
      kundaliId: profileDoc.data.profileId,
      updatedAt: profilePatch.updatedAt,
      displayName: displayName || profileDoc.data.displayName || profileDoc.data.profileId,
      place: place || profileDoc.data.place || '',
    });
  } catch (error) {
    return res.status(500).json({ error: 'Failed to update kundali', details: String(error) });
  }
});

router.delete('/v1/kundalis/:kundaliId', requireFirebaseAuth, async (req, res) => {
  try {
    const identifier = String(req.params.kundaliId ?? '').trim();
    if (!identifier) {
      return res.status(400).json({ error: 'Missing kundaliId path parameter' });
    }

    const store = getPostgresStore();

    const profileMatches = await store.runQuery<RagProfileDocument>(
      COLLECTIONS.ragProfiles,
      [
        { field: 'ownerId', op: 'EQUAL', value: req.user!.uid },
        { field: 'profileId', op: 'EQUAL', value: identifier },
      ],
      { limit: 1 }
    );

    const profileDoc = profileMatches[0];
    if (!profileDoc) {
      return res.status(404).json({ error: 'Kundali not found' });
    }

    const profilePath = `${COLLECTIONS.ragProfiles}/${profileDoc.id}`;
    const sourceDocs = await store.runQuery<RagApiSourceDocument>(
      COLLECTIONS.ragApiSources,
      [
        { field: 'ownerId', op: 'EQUAL', value: req.user!.uid },
        { field: 'profileId', op: 'EQUAL', value: profileDoc.data.profileId },
      ]
    );

    const chunkDocs = await store.runQuery<RagChunkDocument>(
      COLLECTIONS.ragChunks,
      [
        { field: 'ownerId', op: 'EQUAL', value: req.user!.uid },
        { field: 'profileId', op: 'EQUAL', value: profileDoc.data.profileId },
      ]
    );

    await Promise.all([
      store.deleteDocument(profilePath),
      ...sourceDocs.map((doc) => store.deleteDocument(`${COLLECTIONS.ragApiSources}/${doc.id}`)),
      ...chunkDocs.map((doc) => store.deleteDocument(`${COLLECTIONS.ragChunks}/${doc.id}`)),
    ]);

    return res.status(204).send();
  } catch (error) {
    return res.status(500).json({ error: 'Failed to delete kundali', details: String(error) });
  }
});

export default router;