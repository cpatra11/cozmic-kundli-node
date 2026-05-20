import type { ToolConfiguration } from '@aws-sdk/client-bedrock-runtime';

const FETCH_PLANETS_DEF = {
  toolSpec: {
    name: 'fetch_planets',
    description: `Fetch planet positions, houses, lagna, and varga chart summaries. "D1" = Rasi (main Vedic birth chart), NOT a financial trading chart.

For most "analyze my chart" questions, this is the ONLY tool you need (~2KB for D1 alone). Also returns varga chart summaries when multiple vargas requested.

Varga guide (add when topic matches):
- D9 (Navamsa): marriage, spouse, relationships
- D10 (Dasamsa): career, profession, job
- D2 (Hora): wealth, finance
- D3 (Drekkana): siblings, courage
- D4 (Chaturthamsa): property, real estate
- D6 (Shashtamsha): health, disease, litigation
- D7 (Saptamsa): children, creativity
- D12 (Dwadasamsa): parents, family
- D16 (Shodasamsa): travel, vehicles
- D20 (Vimsamsa): spirituality
- D24 (Siddhamamsa): education, wisdom
- D27 (Bhamsa): talent, skill
- D30 (Trimamsa): obstacles, misfortune
- D60 (Shashtiamsa): overall karma, destiny

Mini mode: D1 only. Pro default: D1 + D9 + D10.`,
    inputSchema: {
      json: {
        type: 'object',
        properties: {
          vargas: {
            type: 'array',
            items: { type: 'string' },
            description: 'Divisional chart IDs. Always include D1. Add D9/D10/D60 etc based on topic. Mini mode: D1 only.',
          },
        },
        required: ['vargas'],
      },
    },
  },
};

const FETCH_PANCHANGA_DEF = {
  toolSpec: {
    name: 'fetch_panchanga',
    description: `Fetch Panchanga (5 limbs): tithi (lunar day), vaara (weekday), nakshatra (constellation), yoga, karana.

Use when the user asks about current auspicious timings, muhurta, or daily astrological conditions. Fetches only panchanga data (~200 bytes).`,
    inputSchema: {
      json: {
        type: 'object',
        properties: {},
      },
    },
  },
};

const FETCH_YOGAS_DEF = {
  toolSpec: {
    name: 'fetch_yogas',
    description: `Fetch planetary combinations (yogas) formed in the chart. Use when the user asks about specific yogas, combinations, or special formations.

Do NOT fetch unless the question explicitly asks about yogas/combinat ions. Most chart readings don't need this.`,
    inputSchema: {
      json: {
        type: 'object',
        properties: {},
      },
    },
  },
};

const FETCH_DASHA_DEF = {
  toolSpec: {
    name: 'fetch_dasha',
    description: `Fetch Vimshottari Dasha timeline — the planetary timing periods. Use for ALL timing/prediction questions ("when will X happen", "which period is active now").

Choose nesting based on timing granularity needed. nesting=1 (Mahadasha, ~2KB), nesting=2 (Antardasha, ~13KB), nesting=3 (Pratyantardasha, ~200KB). nesting=3+ is ~200KB+ — only use when fine detail is required.

lordshipVarga: Which varga's bhava (house) mapping to use for dasha lord house lordship analysis. D1 for general, D9 for marriage, D10 for career.

CRITICAL: After analyzing the dasha timeline, YOU decide which specific dates matter. Do NOT fetch transit automatically — call fetch_transit explicitly with your chosen dates in the next iteration.

For "what's happening now" / "current period" questions: fetch_dasha with nesting=1, then call fetch_transit with today's date.`,
    inputSchema: {
      json: {
        type: 'object',
        properties: {
          vargas: {
            type: 'array',
            items: { type: 'string' },
            description: 'Divisional charts for bhava mapping (D1 default, D9 for marriage, D10 for career). Ignored if fetch_planets already loaded the needed varga.',
          },
          nesting: {
            type: 'number',
            description: 'Dasha nesting depth 1-5. 1=Mahadasha (~2KB), 2=Antardasha (~13KB), 3=Pratyantardasha (~200KB), 4=Sookshmantardasha, 5=Pranantardasha.',
          },
          lordshipVarga: {
            type: 'string',
            description: 'Which varga to use for house lordship analysis. D1 (general), D9 (marriage), D10 (career). Default D1.',
          },
        },
        required: [],
      },
    },
  },
};

const FETCH_ASHTAKAVARGA_DEF = {
  toolSpec: {
    name: 'fetch_ashtakavarga',
    description: `Fetch Ashtakavarga — the binding/strength point system for each planet and house.

Use when the user asks about planetary strength, dosha intensity, remedial measures, or house strength. Fetches only ashtakavarga data (~200 bytes).

Do NOT fetch unless the question specifically involves strength analysis.`,
    inputSchema: {
      json: {
        type: 'object',
        properties: {},
      },
    },
  },
};

const FETCH_GRAHABALA_DEF = {
  toolSpec: {
    name: 'fetch_grahabala',
    description: `Fetch Grahabala (Shadbala) — the sixfold planetary strength scores.

Use when the user asks about planetary strength, weakness, or combat (yuddha). Fetches only grahabala data (~200 bytes).

Do NOT fetch unless the question specifically involves strength analysis.`,
    inputSchema: {
      json: {
        type: 'object',
        properties: {},
      },
    },
  },
};

const FETCH_ARUDHA_DEF = {
  toolSpec: {
    name: 'fetch_arudha',
    description: `Fetch Arudha Padas — the perception/reflection points of each house in the chart.

Use when the user specifically asks about Arudha, Pada, perception, or how others perceive them in specific life areas. Fetches only arudha data.

Do NOT fetch unless the question explicitly mentions arudha/pada.`,
    inputSchema: {
      json: {
        type: 'object',
        properties: {},
      },
    },
  },
};

const FETCH_TRANSIT_DEF = {
  toolSpec: {
    name: 'fetch_transit',
    description: `Fetch transit (gochar) planetary positions for specific dates. Use for timing, predictions, and current influences.

YOU decide which dates matter. After analyzing the dasha timeline, pick the specific dates relevant to the question and fetch transit for JUST those dates.

Examples:
- For a specific date: ["2026-06-15"]
- For a month: generate 4-8 evenly-spaced ISO dates spanning that period e.g. ["2027-06-01","2027-06-08","2027-06-15","2027-06-22","2027-06-30"]
- For "what's happening now / today": ["<today's date>"]
- For "next 3 months" (from Apr 2026): ["2026-04-17","2026-05-01","2026-05-17","2026-06-01","2026-06-17","2026-07-01"]

Key transits: Sade Sati (Saturn 12th/1st/2nd from Moon), Dhaiya (Saturn 4th/8th from Moon), Graha house transits.
Use optional nesting parameter for finer transit analysis (higher nesting = more detail per date).`,
    inputSchema: {
      json: {
        type: 'object',
        properties: {
          dates: {
            type: 'array',
            items: { type: 'string' },
            description: 'ISO date strings (YYYY-MM-DD) for transit positions. 1-12 dates per call. For time ranges, generate evenly-spaced dates.',
          },
          nesting: {
            type: 'number',
            description: 'Detail depth 1-5 (default 2). Higher nesting = finer precision for transit positions.',
          },
        },
        required: ['dates'],
      },
    },
  },
};

const SEARCH_ASTROLOGY_DEF = {
  toolSpec: {
    name: 'search_astrology',
    description: `Look up astrological concepts, terms, or principles. Use to explain concepts or verify rules.

Examples: "What is D60?", "Explain Manglik Dosha", "What does 12th lord in 5th mean?", "How does Kaal Sarp form?".

Use when the user asks about general concepts that don't need their specific chart. Do NOT use for personal chart analysis.`,
    inputSchema: {
      json: {
        type: 'object',
        properties: {
          query: {
            type: 'string',
            description: 'The concept or question to look up.',
          },
        },
        required: ['query'],
      },
    },
  },
};

export const CHART_TOOL_CONFIG: ToolConfiguration = {
  tools: [
    FETCH_PLANETS_DEF,
    FETCH_PANCHANGA_DEF,
    FETCH_YOGAS_DEF,
    FETCH_DASHA_DEF,
    FETCH_ASHTAKAVARGA_DEF,
    FETCH_GRAHABALA_DEF,
    FETCH_ARUDHA_DEF,
    FETCH_TRANSIT_DEF,
    SEARCH_ASTROLOGY_DEF,
  ],
};
