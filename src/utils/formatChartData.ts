const RASHI_NAMES = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces'];

function getRashiName(r: number): string {
  return RASHI_NAMES[(r - 1 + 12) % 12] || `Rashi ${r}`;
}

function fmtObj(val: any): string {
  if (!val) return '';
  if (typeof val === 'string') return val;
  if (typeof val === 'number') return String(val);
  if (val.name) return val.name;
  if (val.nakshatra?.name) return val.nakshatra.name;
  try { return JSON.stringify(val); } catch { return ''; }
}

function getPayload(raw: any): any {
  return raw?.chart || raw;
}

export function formatGrahaBhavaLagna(payload: any): string {
  const p = getPayload(payload);
  if (!p) return '';

  const lines: string[] = [];
  const graha = p.graha as Record<string, any> | undefined;
  const bhava = p.bhava as Record<string, any> | undefined;
  const lagna = p.lagna?.Lg || p.lagna;

  if (graha) {
    lines.push('=== GRAHA (PLANET) POSITIONS ===');
    for (const [planet, data] of Object.entries(graha)) {
      const r = data?.rashi;
      const deg = data?.longitude;
      const nk = data?.nakshatra;
      const hs = data?.house_number ?? data?.house;
      const retro = data?.retrograde ? ' (R)' : '';
      const parts = [`${planet}${retro}: ${typeof r === 'number' ? getRashiName(r) : '?'}`];
      if (typeof deg === 'number') parts.push(`${deg.toFixed(1)}°`);
      if (fmtObj(nk)) parts.push(fmtObj(nk));
      if (typeof hs === 'number') parts.push(`House ${hs}`);
      lines.push(parts.join(' — '));
    }
  }

  if (bhava) {
    lines.push('');
    lines.push('=== BHAVA (HOUSE CUSPS) ===');
    for (const [num, data] of Object.entries(bhava)) {
      const r = data?.rashi;
      lines.push(`House ${num}: ${typeof r === 'number' ? getRashiName(r) : '?'}`);
    }
  }

  if (lagna?.rashi) {
    lines.push('');
    lines.push('=== LAGNA (ASCENDANT) ===');
    const rName = typeof lagna.rashi === 'number' ? getRashiName(lagna.rashi) : String(lagna.rashi);
    const nkName = fmtObj(lagna.nakshatra);
    lines.push(`Lagna: ${rName}${nkName ? `, ${nkName} nakshatra` : ''}`);
  }

  return lines.join('\n');
}

export function formatPanchanga(payload: any): string {
  const p = getPayload(payload);
  const panchanga = p?.panchanga;
  if (!panchanga) return '';

  const lines: string[] = ['=== PANCHANG ==='];
  if (fmtObj(panchanga.tithi)) lines.push(`Tithi: ${fmtObj(panchanga.tithi)}`);
  if (fmtObj(panchanga.vaara)) lines.push(`Vaara: ${fmtObj(panchanga.vaara)}`);
  if (fmtObj(panchanga.nakshatra)) lines.push(`Nakshatra: ${fmtObj(panchanga.nakshatra)}`);
  if (fmtObj(panchanga.yoga)) lines.push(`Yoga: ${fmtObj(panchanga.yoga)}`);
  if (fmtObj(panchanga.karana)) lines.push(`Karana: ${fmtObj(panchanga.karana)}`);
  return lines.join('\n');
}

export function formatVargaCharts(payload: any): string {
  const p = getPayload(payload);
  const varga = p?.varga as Record<string, any> | undefined;
  if (!varga) return '';

  const lines: string[] = ['=== VARGA (DIVISIONAL) CHARTS ==='];
  for (const [vk, data] of Object.entries(varga)) {
    const lg = data?.lagna?.Lg || data?.lagna;
    const planetsInHouses: string[] = [];
    const lgRashiNum = typeof lg?.rashi === 'number' ? lg.rashi : 0;
    const graha = data?.graha as Record<string, any> || {};
    for (const [planet, pdata] of Object.entries(graha)) {
      const h = pdata?.house_number ?? pdata?.house;
      if (typeof h === 'number') {
        const rNum = pdata?.rashi || ((lgRashiNum > 0) ? ((lgRashiNum + h - 2 + 12) % 12 + 1) : 0);
        const rName = rNum > 0 ? getRashiName(rNum) : '?';
        planetsInHouses.push(`${planet}:House${h}(${rName})`);
      }
    }
    const rName = lg?.rashi ? (typeof lg.rashi === 'number' ? getRashiName(lg.rashi) : String(lg.rashi)) : '?';
    lines.push(`${vk}: Lagna ${rName} | ${planetsInHouses.join(', ') || 'no planets'}`);
  }

  return lines.join('\n');
}

function formatRemaining(endDate: string): string {
  const now = Date.now();
  const end = new Date(endDate).getTime();
  if (!end || end <= now) return 'ended';
  const totalMs = end - now;
  const totalDays = Math.floor(totalMs / 86400000);
  const years = Math.floor(totalDays / 365.25);
  const months = Math.floor((totalDays % 365.25) / 30.44);
  const days = Math.floor(totalDays % 30.44);
  const parts: string[] = [];
  if (years > 0) parts.push(`${years}y`);
  if (months > 0) parts.push(`${months}m`);
  if (days > 0 || parts.length === 0) parts.push(`${days}d`);
  return `${parts.join(' ')} remaining`;
}

const NESTING_NAMES: Record<number, string> = {
  1: 'mahadasha', 2: 'antardasha', 3: 'pratyantardasha',
  4: 'sookshmantardasha', 5: 'pranantardasha',
};

function formatPeriodTree(node: any, depth: number, maxDepth: number, prefix: string, isLast: boolean, lines: string[]): void {
  if (!node || !node.periods) return;
  const periodKeys = Object.keys(node.periods);
  const now = Date.now();
  let idx = 0;
  for (const key of periodKeys) {
    const period = node.periods[key];
    idx++;
    const isLastChild = idx === periodKeys.length;
    const connector = isLastChild ? '  └ ' : '  ├ ';
    const childPrefix = isLast ? '     ' : '  │  ';

    const startStr = period.start ? period.start.slice(0, 10) : '?';
    const endStr = period.end ? period.end.slice(0, 10) : '?';
    const levelName = NESTING_NAMES[period.nesting] || period.type || 'period';

    const pStart = new Date(period.start).getTime();
    const pEnd = new Date(period.end).getTime();
    const isCurrent = pStart <= now && pEnd > now;
    const remaining = isCurrent && period.end ? ` — ${formatRemaining(period.end)}` : '';
    const isFuture = pStart > now;

    let label = `${prefix}${connector}${key} ${levelName} (${startStr} → ${endStr})`;
    if (isCurrent) label += ' ← ACTIVE';
    if (isFuture) label += ' ← upcoming';
    label += remaining;

    lines.push(label);

    if (depth < maxDepth && period.periods && Object.keys(period.periods).length > 0) {
      const nextPrefix = isLast ? `${prefix}     ` : `${prefix}  │  `;
      formatPeriodTree(period, depth + 1, maxDepth, nextPrefix, isLast, lines);
    }
  }
}

export function formatDashaTimeline(payload: any, nesting: number): string {
  const p = getPayload(payload);
  const dashaRoot = p?.dasha as any;
  if (!dashaRoot || !(dashaRoot.periods || dashaRoot.nesting)) return '';

  const lines: string[] = ['=== DASHA TIMELINE ==='];
  const maxDisplayDepth = 2;
  formatPeriodTree(dashaRoot, 1, Math.max(maxDisplayDepth, nesting ?? 2), '', true, lines);
  return lines.join('\n');
}

export function formatYogas(payload: any): string {
  const p = getPayload(payload);
  const yogas = p?.yogas as any[];
  if (!yogas?.length) return '';

  const lines: string[] = ['=== YOGAS (planetary combinations) ==='];
  for (const y of yogas.slice(0, 10)) {
    lines.push(`${y.name || 'Yoga'}: ${y.description || ''}`);
  }
  return lines.join('\n');
}

export function formatAshtakavarga(payload: any): string {
  const p = getPayload(payload);
  const ashtakavarga = p?.ashtakavarga;
  if (!ashtakavarga) return '';

  const lines: string[] = ['=== ASHTAKAVARGA (points) ==='];
  for (const [planet, pts] of Object.entries(ashtakavarga)) {
    if (typeof pts === 'number') {
      lines.push(`${planet}: ${pts} points`);
    }
  }
  return lines.join('\n');
}

export function formatGrahabala(payload: any): string {
  const p = getPayload(payload);
  const grahabala = p?.grahabala;
  if (!grahabala) return '';

  const lines: string[] = ['=== GRAHABALA (planetary strength) ==='];
  for (const [planet, strength] of Object.entries(grahabala)) {
    lines.push(`${planet}: ${fmtObj(strength)}`);
  }
  return lines.join('\n');
}

export function formatArudha(payload: any): string {
  const p = getPayload(payload);
  const arudha = p?.arudha as Record<string, any> || {};
  if (!Object.keys(arudha).length) return '';

  const lines: string[] = ['=== ARUDHA PADS ==='];
  for (const [house, data] of Object.entries(arudha)) {
    const r = data?.rashi;
    lines.push(`House ${house}: Arudha Lagna ${typeof r === 'number' ? getRashiName(r) : '?'}`);
  }
  return lines.join('\n');
}

export function formatDoshaAnalysis(payload: any): string {
  const p = getPayload(payload);
  const graha = p?.graha as Record<string, any> | undefined;
  if (!graha) return '';

  const lines: string[] = [];
  const maHouse = graha.Ma?.house_number ?? graha.Ma?.house;
  if (typeof maHouse === 'number' && [1, 4, 7, 8, 12].includes(maHouse)) {
    lines.push(`Manglik Dosha: ACTIVE — Mars in House ${maHouse} (${getRashiName(graha.Ma.rashi)})`);
    const juHouse = graha.Ju?.house_number ?? graha.Ju?.house;
    if (typeof juHouse === 'number' && [1, 5, 9].includes(juHouse)) {
      lines.push(`  Mitigation: Jupiter in ${getRashiName(graha.Ju.rashi)} (House ${juHouse}) aspects/cancels Manglik Dosha`);
    }
  }

  const raRashi = graha.Ra?.rashi;
  const keRashi = graha.Ke?.rashi;
  if (typeof raRashi === 'number' && typeof keRashi === 'number' && raRashi !== keRashi) {
    let planetsInside = 0;
    const innerPlanets: string[] = [];
    let r = raRashi;
    const target = keRashi + (keRashi <= raRashi ? 12 : 0);
    while (r < target) {
      const checkRashi = ((r - 1) % 12) + 1;
      for (const [pk, pd] of Object.entries(graha)) {
        if (pk === 'Ra' || pk === 'Ke') continue;
        if (pd?.rashi === checkRashi) {
          planetsInside++;
          innerPlanets.push(pk);
        }
      }
      r++;
    }
    if (planetsInside > 0 && innerPlanets.length > 0) {
      lines.push(`Kaal Sarp Dosha: ACTIVE — ${innerPlanets.join(', ')} between Rahu (${getRashiName(raRashi)}) and Ketu (${getRashiName(keRashi)})`);
    }
  }

  const afflicting9: string[] = [];
  for (const pk of ['Su', 'Sa', 'Ra']) {
    const pd = graha[pk];
    const h = pd?.house_number ?? pd?.house;
    if (typeof h === 'number' && h === 9) {
      afflicting9.push(pk);
    }
  }
  if (afflicting9.length > 0) {
    lines.push(`Pitra Dosha: ACTIVE — ${afflicting9.join(', ')} in 9th house`);
  }

  if (!lines.length) return '';
  return '=== DOSHA ANALYSIS ===\n' + lines.join('\n');
}

function computeNatalHouseMap(transitGraha: Record<string, any>, natalLagnaRashi: number): Record<string, number> {
  const map: Record<string, number> = {};
  if (typeof natalLagnaRashi !== 'number') return map;
  for (const [planet, data] of Object.entries(transitGraha)) {
    const r = data?.rashi;
    if (typeof r === 'number') {
      map[planet] = ((r - natalLagnaRashi + 12) % 12) + 1;
    }
  }
  return map;
}

export function formatTransitSnapshots(
  payload: any,
  transitSnapshots: Record<string, Record<string, any>>
): string {
  if (!transitSnapshots || !Object.keys(transitSnapshots).length) return '';

  const p = getPayload(payload);
  const natLagna = p?.lagna?.Lg || p?.lagna;
  const natalLagnaRashi = natLagna?.rashi;

  const lines: string[] = ['=== GOCHAR (TRANSIT) SNAPSHOTS ==='];
  const dateKeys = Object.keys(transitSnapshots).sort();

  for (const dateKey of dateKeys) {
    const snapData = transitSnapshots[dateKey];
    const graha = snapData?.graha as Record<string, any> | undefined;
    if (!graha) continue;

    const isCurrent = dateKey === new Date().toISOString().slice(0, 10);
    const houseMap = computeNatalHouseMap(graha, natalLagnaRashi);
    lines.push(`${dateKey}${isCurrent ? ' (current)' : ''}`);
    for (const [planet, data] of Object.entries(graha)) {
      const r = data?.rashi;
      const nh = houseMap?.[planet];
      if (r || nh) {
        lines.push(`  ${planet}: ${r ? getRashiName(r) : '?'} House ${nh || '?'}`);
      }
    }
  }

  return lines.join('\n');
}

export function formatSadeSati(
  payload: any,
  transitSnapshots: Record<string, Record<string, any>>
): string {
  if (!transitSnapshots || !Object.keys(transitSnapshots).length) return '';

  const p = getPayload(payload);
  const graha = p?.graha as Record<string, any> | undefined;
  const moonRashi = graha?.Mo?.rashi as number | undefined;
  if (typeof moonRashi !== 'number') return '';

  const lines: string[] = [];
  for (const [dateKey, snap] of Object.entries(transitSnapshots)) {
    const satRashi = snap?.graha?.Sa?.rashi as number | undefined;
    if (typeof satRashi !== 'number') continue;
    const diff = ((satRashi - moonRashi) + 12) % 12;
    let phase = '';
    if (diff === 11) phase = 'Sade Sati FIRST phase (12th from Moon)';
    else if (diff === 0) phase = 'Sade Sati PEAK phase (1st from Moon)';
    else if (diff === 1) phase = 'Sade Sati LAST phase (2nd from Moon)';
    else if (diff === 3) phase = 'Shani Dhaiya (4th from Moon)';
    else if (diff === 7) phase = 'Ashtama Shani (8th from Moon)';
    if (phase) {
      const isCurrent = dateKey === new Date().toISOString().slice(0, 10);
      lines.push(`${dateKey}${isCurrent ? ' (current)' : ''}: Saturn in ${getRashiName(satRashi)} — ${phase}`);
    }
  }

  if (!lines.length) return '';
  return `=== SADE SATI / DHAIYA ANALYSIS ===\nMoon birth rashi: ${getRashiName(moonRashi)}\n${lines.join('\n')}`;
}

export function formatAllChartData(
  payload: any,
  infolevels: string[],
  transitSnapshots?: Record<string, Record<string, any>>
): string {
  const parts: string[] = [];

  const grahaSection = formatGrahaBhavaLagna(payload);
  if (grahaSection) parts.push(grahaSection);

  if (infolevels.includes('panchanga')) {
    const pSection = formatPanchanga(payload);
    if (pSection) parts.push(pSection);
  }

  const vargaSection = formatVargaCharts(payload);
  if (vargaSection) parts.push(vargaSection);

  if (infolevels.includes('dasha')) {
    const nesting = (payload as any)?.chart?.dasha?.nesting || 2;
    const dashaSection = formatDashaTimeline(payload, nesting);
    if (dashaSection) parts.push(dashaSection);
  }

  if (infolevels.includes('yogas')) {
    const yogaSection = formatYogas(payload);
    if (yogaSection) parts.push(yogaSection);
  }

  if (infolevels.includes('ashtakavarga')) {
    const asSection = formatAshtakavarga(payload);
    if (asSection) parts.push(asSection);
  }

  if (infolevels.includes('grahabala')) {
    const gbSection = formatGrahabala(payload);
    if (gbSection) parts.push(gbSection);
  }

  if (infolevels.includes('arudha')) {
    const arSection = formatArudha(payload);
    if (arSection) parts.push(arSection);
  }

  const doshaSection = formatDoshaAnalysis(payload);
  if (doshaSection) parts.push(doshaSection);

  if (transitSnapshots && Object.keys(transitSnapshots).length > 0) {
    const transitSection = formatTransitSnapshots(payload, transitSnapshots);
    if (transitSection) parts.push(transitSection);

    const sadeSatiSection = formatSadeSati(payload, transitSnapshots);
    if (sadeSatiSection) parts.push(sadeSatiSection);
  }

  return parts.join('\n\n');
}
