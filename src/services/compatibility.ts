import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import path from 'node:path';

type NakshatraRecord = {
  id: number;
  name: string;
  lord: string;
  gana: string;
  yoni: string;
  yoni_sex?: string;
  nadi: string;
  varna: string;
  vashya: string;
};

type CompatibilityRow = {
  boy_nakshatra: number;
  boy_paadham: number;
  girl_nakshatra: number;
  girl_paadham: number;
  score: number;
  ettu?: Array<number | string>;
  naalu?: {
    mahendra?: boolean;
    vedha?: boolean;
    rajju?: boolean;
    shreedheerga?: boolean;
  };
};

type CompatibilityFlag = 'mahendra' | 'vedha' | 'rajju' | 'shreedheerga';

type CompatibilityQuery = {
  boyNak?: number | null;
  boyPad?: number | null;
  girlNak?: number | null;
  girlPad?: number | null;
  minScore?: number | null;
  checkMahendra?: boolean | null;
  checkVedha?: boolean | null;
  checkRajju?: boolean | null;
  checkShreedheerga?: boolean | null;
  mode?: string | null;
};

type AshtakootaValues = {
  varna: number;
  vashya: number;
  gana: number;
  tara: number;
  yoni: number;
  adhipathi: number;
  rasi: number;
  nadi: number;
};

type AshtakootaResult = {
  values: AshtakootaValues;
  reasons: Record<keyof AshtakootaValues, string>;
  total: number;
  max: number;
  percent: number;
  passed: boolean;
};

type CompatibilityMatch = {
  boy_nakshatra: number;
  boy_paadham: number;
  girl_nakshatra: number;
  girl_paadham: number;
  partner_nakshatra: number;
  partner_paadham: number;
  score: number;
  ettu: AshtakootaValues;
  ashtakoota: AshtakootaResult;
  naalu: {
    mahendra: boolean;
    vedha: boolean;
    rajju: boolean;
    shreedheerga: boolean;
  };
};

const DATA_DIR_CANDIDATES = [
  path.resolve(process.cwd(), '../../be1/api/data'),
  path.resolve(process.cwd(), '../be1/api/data'),
  path.resolve(process.cwd(), 'be1/api/data'),
];

const ASHTAKOOTA_KEYS: Array<keyof AshtakootaValues> = [
  'varna',
  'vashya',
  'gana',
  'tara',
  'yoni',
  'adhipathi',
  'rasi',
  'nadi',
];

let nakshatraCachePromise: Promise<Map<number, NakshatraRecord>> | null = null;
let compatibilityCachePromise: Promise<CompatibilityRow[]> | null = null;

function resolveDataFile(fileName: string): string {
  for (const baseDir of DATA_DIR_CANDIDATES) {
    const candidate = path.join(baseDir, fileName);
    if (existsSync(candidate)) {
      return candidate;
    }
  }

  throw new Error(`Unable to locate compatibility data file: ${fileName}`);
}

async function readJsonFile<T>(fileName: string): Promise<T> {
  const filePath = resolveDataFile(fileName);
  const raw = await readFile(filePath, 'utf8');
  return JSON.parse(raw) as T;
}

function toFiniteNumber(value: unknown): number | undefined {
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : undefined;
  }

  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : undefined;
  }

  return undefined;
}

function toOptionalBoolean(value: unknown): boolean | undefined {
  if (typeof value === 'boolean') {
    return value;
  }

  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (['1', 'true', 'yes', 'on'].includes(normalized)) {
      return true;
    }
    if (['0', 'false', 'no', 'off'].includes(normalized)) {
      return false;
    }
  }

  return undefined;
}

function normalizeRow(row: CompatibilityRow): CompatibilityRow {
  return {
    boy_nakshatra: Number(row.boy_nakshatra),
    boy_paadham: Number(row.boy_paadham),
    girl_nakshatra: Number(row.girl_nakshatra),
    girl_paadham: Number(row.girl_paadham),
    score: Number(row.score ?? 0),
    ettu: Array.isArray(row.ettu) ? row.ettu.map((value) => Number(value ?? 0)) : undefined,
    naalu: {
      mahendra: Boolean(row.naalu?.mahendra),
      vedha: Boolean(row.naalu?.vedha),
      rajju: Boolean(row.naalu?.rajju),
      shreedheerga: Boolean(row.naalu?.shreedheerga),
    },
  };
}

async function loadNakshatras(): Promise<Map<number, NakshatraRecord>> {
  if (!nakshatraCachePromise) {
    nakshatraCachePromise = readJsonFile<NakshatraRecord[]>('nakshatras.json').then((rows) => {
      const byId = new Map<number, NakshatraRecord>();
      for (const row of rows) {
        if (row && Number.isFinite(Number(row.id))) {
          byId.set(Number(row.id), row);
        }
      }
      return byId;
    });
  }

  return nakshatraCachePromise;
}

async function loadCompatibilityRows(): Promise<CompatibilityRow[]> {
  if (!compatibilityCachePromise) {
    compatibilityCachePromise = readJsonFile<CompatibilityRow[]>('all_nak_pad_boy_girl.json').then((rows) => rows.map(normalizeRow));
  }

  return compatibilityCachePromise;
}

function getRashiFromNakshatraAndPada(nak: number, pad: number): number {
  const clampedNak = Math.max(1, Math.min(27, nak));
  const clampedPad = Math.max(1, Math.min(4, pad));
  const index = (clampedNak - 1) * 4 + (clampedPad - 1);
  return Math.floor(index / 9) + 1;
}

function buildAshtakootaValues(ettu: Array<number | string> | undefined): AshtakootaValues {
  const values = ettu ?? [];
  return {
    varna: Number(values[0] ?? 0),
    vashya: Number(values[1] ?? 0),
    gana: Number(values[2] ?? 0),
    tara: Number(values[3] ?? 0),
    yoni: Number(values[4] ?? 0),
    adhipathi: Number(values[5] ?? 0),
    rasi: Number(values[6] ?? 0),
    nadi: Number(values[7] ?? 0),
  };
}

function buildAshtakootaResult(
  boy: NakshatraRecord,
  girl: NakshatraRecord,
  boyNak: number,
  boyPad: number,
  girlNak: number,
  girlPad: number,
  row?: CompatibilityRow,
  overrideTotal?: number
): AshtakootaResult {
  const values = buildAshtakootaValues(row?.ettu);
  const total = overrideTotal ?? ASHTAKOOTA_KEYS.reduce((sum, key) => sum + Number(values[key] ?? 0), 0);
  const max = 36.0;
  const percent = max > 0 ? (total / max) * 100 : 0;

  return {
    values,
    reasons: {
      varna: `Boy varna ${boy.varna}, Girl varna ${girl.varna}`,
      vashya: `Boy vashya ${boy.vashya}, Girl vashya ${girl.vashya}`,
      gana: `Boy gana ${boy.gana}, Girl gana ${girl.gana}`,
      tara: `Tara difference between Nak ${boyNak} and ${girlNak}`,
      yoni: `Boy yoni ${boy.yoni}, Girl yoni ${girl.yoni}`,
      adhipathi: `Boy lord ${boy.lord}, Girl lord ${girl.lord}`,
      rasi: `Boy rashi ${getRashiFromNakshatraAndPada(boyNak, boyPad)}, Girl rashi ${getRashiFromNakshatraAndPada(girlNak, girlPad)}`,
      nadi: `Boy nadi ${boy.nadi}, Girl nadi ${girl.nadi}`,
    },
    total,
    max,
    percent,
    passed: total >= 18.0,
  };
}

async function getNakshatraById(id: number): Promise<NakshatraRecord | undefined> {
  const nakshatras = await loadNakshatras();
  return nakshatras.get(id);
}

async function buildAshtakootaResultForPair(boyNak: number, boyPad: number, girlNak: number, girlPad: number): Promise<AshtakootaResult> {
  const boy = await getNakshatraById(boyNak);
  const girl = await getNakshatraById(girlNak);

  if (!boy || !girl) {
    throw new Error('Invalid nakshatra id');
  }

  const rows = await loadCompatibilityRows();
  const row = rows.find(
    (entry) =>
      entry.boy_nakshatra === boyNak &&
      entry.boy_paadham === boyPad &&
      entry.girl_nakshatra === girlNak &&
      entry.girl_paadham === girlPad
  );

  return buildAshtakootaResult(boy, girl, boyNak, boyPad, girlNak, girlPad, row);
}

function matchesFlagFilters(row: CompatibilityRow, query: CompatibilityQuery): boolean {
  const requestedFlags: CompatibilityFlag[] = [];
  if (query.checkMahendra) requestedFlags.push('mahendra');
  if (query.checkVedha) requestedFlags.push('vedha');
  if (query.checkRajju) requestedFlags.push('rajju');
  if (query.checkShreedheerga) requestedFlags.push('shreedheerga');

  for (const flag of requestedFlags) {
    if (!row.naalu?.[flag]) {
      return false;
    }
  }

  return true;
}

function matchesFilters(row: CompatibilityRow, query: CompatibilityQuery): boolean {
  let matches = true;

  if (query.boyNak != null) {
    matches = matches && row.boy_nakshatra === query.boyNak;
    if (query.boyPad != null) {
      matches = matches && row.boy_paadham === query.boyPad;
    }
  }

  if (query.girlNak != null) {
    matches = matches && row.girl_nakshatra === query.girlNak;
    if (query.girlPad != null) {
      matches = matches && row.girl_paadham === query.girlPad;
    }
  }

  return matches;
}

export async function matchCompatibility(query: CompatibilityQuery): Promise<{ matches: CompatibilityMatch[]; count: number }> {
  const rows = await loadCompatibilityRows();
  const nakshatras = await loadNakshatras();
  const mode = String(query.mode ?? 'rules').trim().toLowerCase();
  const useLegacy = ['csv', 'legacy'].includes(mode);

  const results: CompatibilityMatch[] = [];
  for (const row of rows) {
    if (!matchesFilters(row, query) || !matchesFlagFilters(row, query)) {
      continue;
    }

    const boy = nakshatras.get(row.boy_nakshatra);
    const girl = nakshatras.get(row.girl_nakshatra);
    if (!boy || !girl) {
      continue;
    }

    const ashtakoota = useLegacy
      ? buildAshtakootaResult(
          boy,
          girl,
          row.boy_nakshatra,
          row.boy_paadham,
          row.girl_nakshatra,
          row.girl_paadham,
          row,
          Number(row.score ?? 0)
        )
      : buildAshtakootaResult(boy, girl, row.boy_nakshatra, row.boy_paadham, row.girl_nakshatra, row.girl_paadham, row);

    const score = useLegacy ? Number(row.score ?? 0) : ashtakoota.total;
    if (typeof query.minScore === 'number' && Number.isFinite(query.minScore) && score < query.minScore) {
      continue;
    }

    const partnerNak = query.boyNak != null ? row.girl_nakshatra : row.boy_nakshatra;
    const partnerPad = query.boyPad != null ? row.girl_paadham : row.boy_paadham;

    results.push({
      boy_nakshatra: row.boy_nakshatra,
      boy_paadham: row.boy_paadham,
      girl_nakshatra: row.girl_nakshatra,
      girl_paadham: row.girl_paadham,
      partner_nakshatra: partnerNak,
      partner_paadham: partnerPad,
      score,
      ettu: ashtakoota.values,
      ashtakoota,
      naalu: {
        mahendra: Boolean(row.naalu?.mahendra),
        vedha: Boolean(row.naalu?.vedha),
        rajju: Boolean(row.naalu?.rajju),
        shreedheerga: Boolean(row.naalu?.shreedheerga),
      },
    });
  }

  return {
    matches: results,
    count: results.length,
  };
}

export function parseCompatibilityQuery(rawQuery: Record<string, unknown>): CompatibilityQuery {
  return {
    boyNak: toFiniteNumber(rawQuery.boy_nak),
    boyPad: toFiniteNumber(rawQuery.boy_pad),
    girlNak: toFiniteNumber(rawQuery.girl_nak),
    girlPad: toFiniteNumber(rawQuery.girl_pad),
    minScore: toFiniteNumber(rawQuery.min_score),
    checkMahendra: toOptionalBoolean(rawQuery.check_mahendra),
    checkVedha: toOptionalBoolean(rawQuery.check_vedha),
    checkRajju: toOptionalBoolean(rawQuery.check_rajju),
    checkShreedheerga: toOptionalBoolean(rawQuery.check_shreedheerga),
    mode: typeof rawQuery.mode === 'string' ? rawQuery.mode : undefined,
  };
}

export type { CompatibilityMatch, CompatibilityQuery, AshtakootaResult, AshtakootaValues };
