import type { ToolConfiguration } from '@aws-sdk/client-bedrock-runtime';

const FETCH_CHART_DATA_DEF = {
  toolSpec: {
    name: 'fetch_chart_data',
    description: `Fetch detailed birth chart data. Always start with D1 + basic + panchanga + yogas. For Pro mode, add D9 and D10 by default. Add more vargas based on the question topic.

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
- 1: mahadasha only (~2KB)
- 2: + antardasha (~13KB) — good for most questions
- 3: + pratyantardasha (~200KB) — detailed timing
- 4: + sookshmantardasha — precise day-level timing
- 5: full detail (very large) — rarely needed`,
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
        },
        required: ['vargas', 'infolevels'],
      },
    },
  },
};

const FETCH_TRANSIT_DEF = {
  toolSpec: {
    name: 'fetch_transit',
    description: `Fetch current or future transit (gochar) planetary positions. Use when the question involves timing, predictions, or current planetary influences.

The transit shows where planets are now or at specific dates. Compare to birth chart for timing analysis.

Key uses:
- Sade Sati: Saturn transiting 12th/1st/2nd from natal Moon
- Dhaiya: Saturn transiting 4th/8th from natal Moon
- Current planetary transits through houses
- Dasha period analysis with transit support

Call this AFTER fetch_chart_data so you have birth chart for comparison.`,
    inputSchema: {
      json: {
        type: 'object',
        properties: {
          dates: {
            type: 'array',
            items: { type: 'string' },
            description: 'ISO date strings (YYYY-MM-DD) for transit positions. Include today and key dasha period dates.',
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
