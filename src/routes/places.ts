import { Router } from 'express';
import { z } from 'zod';
import { env } from '../config/env.js';

const router = Router();

const GOOGLE_BASE = 'https://maps.googleapis.com/maps/api';

router.get('/v1/places/autocomplete', async (req, res) => {
  try {
    const { input, types } = z.object({
      input: z.string().min(1),
      types: z.string().optional(),
    }).parse(req.query);

    if (!env.GOOGLE_API_KEY) {
      return res.status(500).json({ error: 'Google API key not configured', predictions: [] });
    }

    const url = `${GOOGLE_BASE}/place/autocomplete/json?input=${encodeURIComponent(input)}&types=${encodeURIComponent(types || 'geocode')}&key=${env.GOOGLE_API_KEY}`;
    const response = await fetch(url);
    const json = await response.json();
    return res.json(json);
  } catch (error) {
    if (error instanceof z.ZodError) {
      return res.status(400).json({ error: 'Invalid request', predictions: [] });
    }
    return res.status(502).json({ error: 'Places API error', predictions: [] });
  }
});

router.get('/v1/places/details', async (req, res) => {
  try {
    const { place_id } = z.object({
      place_id: z.string().min(1),
    }).parse(req.query);

    if (!env.GOOGLE_API_KEY) {
      return res.status(500).json({ error: 'Google API key not configured' });
    }

    const url = `${GOOGLE_BASE}/place/details/json?place_id=${encodeURIComponent(place_id)}&fields=geometry&key=${env.GOOGLE_API_KEY}`;
    const response = await fetch(url);
    const json = await response.json();
    return res.json(json);
  } catch (error) {
    if (error instanceof z.ZodError) {
      return res.status(400).json({ error: 'Invalid request' });
    }
    return res.status(502).json({ error: 'Places API error' });
  }
});

router.get('/v1/places/timezone', async (req, res) => {
  try {
    const { lat, lon } = z.object({
      lat: z.coerce.number(),
      lon: z.coerce.number(),
    }).parse(req.query);

    if (!env.GOOGLE_API_KEY) {
      return res.status(500).json({ error: 'Google API key not configured' });
    }

    const ts = Math.floor(Date.now() / 1000);
    const url = `${GOOGLE_BASE}/timezone/json?location=${lat},${lon}&timestamp=${ts}&key=${env.GOOGLE_API_KEY}`;
    const response = await fetch(url);
    const json = await response.json();
    return res.json(json);
  } catch (error) {
    if (error instanceof z.ZodError) {
      return res.status(400).json({ error: 'Invalid request' });
    }
    return res.status(502).json({ error: 'Timezone API error' });
  }
});

export default router;
