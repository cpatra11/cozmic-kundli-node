# Transit Analysis Fix Plan

## Root Cause Analysis

The transit API response structure is:
```
{ chart: { ...natal..., transit: { lagna: {Lg: {rashi:...}}, graha: {Su: {...transit pos...}} } } }
```

**Bug in `load_grounding` (newAgent.ts:287-288):** Current code does:
```js
const rawTransit = transitResponseObj.transit || transitResponseObj;  // WRONG path
transitData = rawTransit.chart || rawTransit;                          // Gets NATAL chart by accident
```

The mature kundliAgent (`astrologyTools.ts:795-797`) correctly extracts:
```js
const transitChart = payload.chart?.transit ?? payload.chart;
```

**Consequence:** `rawPayload.transit` gets set to the **natal chart** data. The LLM sees natal positions labeled as "transit", gets confused, and hallucinates the house mapping.

**Secondary issue:** Even if transit data were correct, `analyzeToolGroup` dumps raw 1500-char JSON blobs and relies on the LLM to compute `((planet_rashi - natal_lagna_rashi + 12) % 12) + 1` — but it never includes the natal lagna rashi in the prompt, so the LLM hallucinates one.

---

## Fix 1: Correct transit extraction in `load_grounding`

Swap `newAgent.ts:286-294` to match the proven pattern from `astrologyTools.ts:795-797`:

```typescript
// Load_grounding transit extraction:
const transitResponseObj = transitResponse as any;
const chart = transitResponseObj.chart || transitResponseObj;
const transitChart = (chart as any)?.transit || chart;
transitData = transitChart;

if (transitData?.graha) {
    rawPayload = { ...rawPayload, transit: transitData };
}
```

Also add logging to verify the transit lagna and transit planet positions at runtime.

---

## Fix 2: Pre-compute natal house for each transit planet server-side

In `load_grounding`, after the transit fetch:

```typescript
// Compute which NATAL house each transit planet is transiting
function computeTransitToNatalHouseMap(
  transitGraha: Record<string, any>,
  natalLagnaRashi: number
): Record<string, number> {
  const map: Record<string, number> = {};
  for (const [planet, data] of Object.entries(transitGraha)) {
    const rashi = data?.rashi;
    if (typeof rashi === 'number') {
      map[planet] = ((rashi - natalLagnaRashi + 12) % 12) + 1;
    }
  }
  return map;
}

// Extract natal lagna from the natal chart data
const natalLagna = (rawPayload as any)?.lagna?.Lg || (rawPayload as any)?.lagna;
const natalLagnaRashi = natalLagna?.rashi;

if (transitData?.graha && natalLagnaRashi) {
  const transitHouseMap = computeTransitToNatalHouseMap(transitData.graha, natalLagnaRashi);
  rawPayload = { ...rawPayload, transitHouseMap };
}
```

This is the **key insight from kundliAgent.ts line 5814**:
```
'Transit mapping contract: determine transit houses relative to transit ascendant (Lagna)
using whole-sign mapping... Never present backend house_number as user-facing transit house truth.'
```

We apply the same principle but compute **natal** houses being transited (not transit houses).

---

## Fix 3: Structured prompt in `analyzeToolGroup`

Replace the 1500-char raw JSON dump with a human-readable table:

```typescript
function buildTransitPrompt(rawPayload: any): string {
  const transitGraha = rawPayload.transit?.graha;
  const transitLagna = rawPayload.transit?.lagna?.Lg || rawPayload.transit?.lagna;
  const transitLagnaRashi = transitLagna?.rashi;
  const transitHouseMap = rawPayload.transitHouseMap || {};

  const natalLagna = rawPayload.lagna?.Lg || rawPayload.lagna;
  const natalLagnaRashi = natalLagna?.rashi;

  const rashiNames = ['Aries','Taurus','Gemini','Cancer','Leo','Virgo','Libra','Scorpio','Sagittarius','Capricorn','Aquarius','Pisces'];

  let result = `Natal Lagna: ${rashiNames[natalLagnaRashi - 1]} (rashi ${natalLagnaRashi})\n\n`;
  result += `Transit Lagna: ${rashiNames[transitLagnaRashi - 1]} (rashi ${transitLagnaRashi})\n\n`;
  result += `Planet | In Sign | Longitude | Transiting NATAL House\n`;
  result += `-------|---------|-----------|-------------------------\n`;

  for (const [planet, data] of Object.entries(transitGraha || {})) {
    const r = (data as any).rashi;
    const lon = (data as any).longitude;
    const natalHouse = transitHouseMap[planet] || '?';
    result += `${planet.padEnd(7)}| ${rashiNames[r-1].padEnd(7)}| ${(lon||'').toFixed(1)}°   | House ${natalHouse}\n`;
  }

  return result;
}
```

This gets injected into the system prompt for `analyzeToolGroup` when `isTransit` is true. The LLM no longer needs to compute or guess anything — the natal house being transited is explicitly shown.

---

## Fix 4: Kill port 8787 process

Before restarting:
```bash
lsof -ti:8787 | xargs kill -9 2>/dev/null; echo "done"
```

---

## Fix 5: Add natal lagna extraction to `analyzeToolGroup`

In the non-transit case, always include the natal lagna position in the prompt data so the LLM can contextualize houses properly:

```typescript
const lagna = rawPayload.lagna?.Lg || rawPayload.lagna;
const lagnaInfo = lagna ? `Natal Lagna: ${rashiName(lagna.rashi)} (rashi ${lagna.rashi})` : '';
```

---

## Files Modified

| File | Changes |
|------|---------|
| `newAgent.ts` | Fix transit extraction path (line 287-288), add `computeTransitToNatalHouseMap`, add `buildTransitPrompt`, add natal lagna to chart data in `analyzeToolGroup` |

---

## Testing Steps

1. Kill process on port 8787, restart the service
2. Send "Give my current transit analysis" with a test kundli
3. Verify `debug-log.txt` shows:
   - `hasTransitData: true` in `load_grounding` output
   - Transit chart with correct planet positions (Sun ~30-60° for May 2026, NOT 329.56°)
   - Structured prompt table instead of raw JSON dump
   - `transitHouseMap` with correct natal house numbers
4. Verify final answer uses real transit positions with correct natal house mapping
5. Run lint/typecheck

---

## Success Criteria

- Transit Sun for May 2026 shows Sun ~30-60° (Aries/Taurus) NOT 329.5° (Aquarius)
- Each transit planet correctly identifies which NATAL house it's transiting
- No hallucinated lagna positions
- No "missing data" language in final answer
- `house_number` from backend is NEVER used directly as user-facing house (per kundliAgent.ts line 5814)
