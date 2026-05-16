import type { ToolConfiguration } from '@aws-sdk/client-bedrock-runtime';

const FETCH_CHART_DATA_DEF = {
  toolSpec: {
    name: 'fetch_chart_data',
    description: `Fetch detailed birth chart data. Always start with D1 + basic + panchanga + yogas. For Pro mode, add D9 and D10 by default. Add more vargas based on the question topic. Tip: set autoTransit: true when including dasha infolevel to auto-fetch transit for period boundary dates.

Varga guide (fetch when topic matches):
- D2 (Hora): wealth, finance, money, income
- D3 (Drekkana): siblings, courage, co-born
- D4 (Chaturthamsa): property, real estate, land, home
- D6 (Shashtamsha): health, disease, litigation, court
- D7 (Saptamsa): children, progeny, creativity, fertility
- D9 (Navamsa): marriage, spouse, relationships, divorce
- D10 (Dasamsa): career, profession, job, business
- D12 (Dwadasamsa): parents, family, ancestors
- D16 (Shodasamsa): travel, vehicles, foreign journeys
- D20 (Vimsamsa): spirituality, devotion, religious
- D24 (Siddhamamsa): education, knowledge, wisdom, learning
- D27 (Bhamsa): strength, talent, ability, skill
- D30 (Trimamsa): obstacles, misfortune, enemies, struggles
- D40 (Khavedamsa): maternal lineage
- D45 (Akshvedamsa): paternal lineage
- D60 (Shashtiamsa): overall karma, destiny

Infolevels:
- basic: planet positions, houses, lagna (ALWAYS include)
- panchanga: tithi, vaara, nakshatra, yoga, karana
- yogas: planetary combinations
- dasha: timing periods (include for timing/prediction questions)
- ashtakavarga: strength points (include for strength/dosha/remedy)
- grahabala: planetary strength scores (include for strength/dosha/remedy)
- arudha: pada/perception (include when specifically asked)
- ayanamsa: precession info

Dasha nesting:
- 1: mahadasha only (~2KB) — broad decade/year-level overview
- 2: + antardasha (~13KB) — month-level timing, default for most life questions
- 3: + pratyantardasha (~200KB) — week-level timing, use for "when will X happen"
- 4: + sookshmantardasha — day-level precision (large)
- 5: + pranantardasha — hour-level precision (very large, rarely needed)`,
    inputSchema: {
      json: {
        type: 'object',
        properties: {
          vargas: {
            type: 'array',
            items: { type: 'string' },
            description: 'Divisional chart IDs. Always include D1. Pro default: ["D1","D9","D10"]. Add more as needed.',
          },
          infolevels: {
            type: 'array',
            items: { type: 'string' },
            description: 'Information levels. Always include "basic". Add based on question topic.',
          },
          nesting: {
            type: 'number',
            description: 'Dasha nesting depth (1-5). Default: 2 for most questions.',
          },
          autoTransit: {
            type: 'boolean',
            description: 'If true and dasha infolevel is included, auto-fetch transit for period boundary dates.',
          },
        },
        required: ['vargas', 'infolevels'],
      },
    },
  },
};

const FETCH_TRANSIT_DEF = {
  toolSpec: {
    name: 'fetch_transit',
    description: `Fetch transit (gochar) planetary positions for specific dates. Use for timing, predictions, and current influences.

When the user asks about a time period (e.g. "June 2027", "next 3 months", "remaining May"), GENERATE 4-12 evenly spaced ISO dates spanning that period.

Examples:
  "June 2027" → ["2027-06-01","2027-06-08","2027-06-15","2027-06-22","2027-06-30"]
  "next 3 months" (from Apr 2026) → ["2026-04-17","2026-05-01","2026-05-17","2026-06-01","2026-06-17","2026-07-01"]

Call AFTER fetch_chart_data so you have birth chart for comparison.

Key transits: Sade Sati (Saturn 12th/1st/2nd from Moon), Dhaiya (Saturn 4th/8th from Moon), house transits.`,
    inputSchema: {
      json: {
        type: 'object',
        properties: {
          dates: {
            type: 'array',
            items: { type: 'string' },
            description: 'ISO date strings (YYYY-MM-DD) for transit positions. For time ranges, generate 4-12 evenly spaced dates.',
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

Use when the user asks about general concepts that don't need their specific chart.`,
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
  tools: [FETCH_CHART_DATA_DEF, FETCH_TRANSIT_DEF, SEARCH_ASTROLOGY_DEF],
};
