import { env } from '../config/env.js';
import { getPostgresPool } from '../services/postgresClient.js';
import { applyPendingMigrations } from '../services/postgresMigrations.js';

export type UsageQuotaType = 'mini_chat' | 'pro_chat' | 'kundli_generate';
export type PlanTier = 'pro' | 'free';

interface MonthlyUsageCounterRow {
  owner_id: string;
  period_start_ms: number;
  mini_chat_used: number;
  pro_chat_used: number;
  kundli_generate_used: number;
  created_at: number;
  updated_at: number;
}

interface MonthlyQuotaLimits {
  miniChat: number;
  proChat: number;
  kundliGenerations: number;
}

export interface QuotaBucketStatus {
  used: number;
  limit: number;
  remaining: number;
  exhausted: boolean;
}

export interface QuotaStatusSnapshot {
  ownerId: string;
  planTier: PlanTier;
  periodStartMs: number;
  yearMonth: string;
  resetAtMs: number;
  enforcementEnabled: boolean;
  miniChat: QuotaBucketStatus;
  proChat: QuotaBucketStatus;
  kundliGenerations: QuotaBucketStatus;
}

export interface QuotaConsumeResult {
  allowed: boolean;
  quotaType: UsageQuotaType;
  status: QuotaStatusSnapshot;
}

const BILLING_MONTH_MS = 30 * 24 * 60 * 60 * 1000;

const QUOTA_FIELD_BY_TYPE: Record<UsageQuotaType, keyof MonthlyUsageCounterRow> = {
  mini_chat: 'mini_chat_used',
  pro_chat: 'pro_chat_used',
  kundli_generate: 'kundli_generate_used',
};

function toSafeLimit(value: number, fallback: number): number {
  if (!Number.isFinite(value)) return fallback;
  return Math.max(0, Math.floor(value));
}

function getMonthlyQuotaLimits(isPro: boolean): MonthlyQuotaLimits {
  const quotaConfig = env.QUOTA_CONFIG;
  if (isPro) {
    return {
      miniChat: toSafeLimit(quotaConfig.pro.mini_requests, 50),
      proChat: toSafeLimit(quotaConfig.pro.pro_requests, 50),
      kundliGenerations: toSafeLimit(quotaConfig.pro.kundli_generations, 10),
    };
  }

  return {
    miniChat: toSafeLimit(quotaConfig.nonpro.mini_requests, 5),
    proChat: toSafeLimit(quotaConfig.nonpro.pro_requests, 0),
    kundliGenerations: toSafeLimit(quotaConfig.nonpro.kundli_generations, 3),
  };
}

export function resolveMonthlyWindow(nowMs: number, anchorDateMs?: number): { periodStartMs: number; resetAtMs: number } {
  if (anchorDateMs !== undefined) {
    const monthsSinceAnchor = Math.floor((nowMs - anchorDateMs) / BILLING_MONTH_MS);
    const periodStartMs = anchorDateMs + monthsSinceAnchor * BILLING_MONTH_MS;
    const resetAtMs = periodStartMs + BILLING_MONTH_MS;
    return { periodStartMs, resetAtMs };
  }

  const current = new Date(nowMs);
  const year = current.getUTCFullYear();
  const monthIndex = current.getUTCMonth();
  const periodStartMs = Date.UTC(year, monthIndex, 1, 0, 0, 0, 0);
  const resetAtMs = Date.UTC(year, monthIndex + 1, 1, 0, 0, 0, 0);
  return { periodStartMs, resetAtMs };
}

function buildEmptyCounterRow(ownerId: string, periodStartMs: number, nowMs: number): MonthlyUsageCounterRow {
  return {
    owner_id: ownerId,
    period_start_ms: periodStartMs,
    mini_chat_used: 0,
    pro_chat_used: 0,
    kundli_generate_used: 0,
    created_at: nowMs,
    updated_at: nowMs,
  };
}

function toBucketStatus(usedRaw: number, limitRaw: number): QuotaBucketStatus {
  const used = Math.max(0, Number(usedRaw) || 0);
  const limit = Math.max(0, Number(limitRaw) || 0);
  const remaining = Math.max(0, limit - used);

  return {
    used,
    limit,
    remaining,
    exhausted: remaining <= 0,
  };
}

function toPlanTier(isPro: boolean): PlanTier {
  return isPro ? 'pro' : 'free';
}

function buildQuotaStatusSnapshot(args: {
  ownerId: string;
  planTier: PlanTier;
  periodStartMs: number;
  resetAtMs: number;
  limits: MonthlyQuotaLimits;
  row: MonthlyUsageCounterRow;
}): QuotaStatusSnapshot {
  const periodDate = new Date(args.periodStartMs);
  const yearMonth = `${periodDate.getUTCFullYear()}-${String(periodDate.getUTCMonth() + 1).padStart(2, '0')}`;

  return {
    ownerId: args.ownerId,
    planTier: args.planTier,
    periodStartMs: args.periodStartMs,
    yearMonth,
    resetAtMs: args.resetAtMs,
    enforcementEnabled: env.QUOTA_CONFIG.enabled,
    miniChat: toBucketStatus(args.row.mini_chat_used, args.limits.miniChat),
    proChat: toBucketStatus(args.row.pro_chat_used, args.limits.proChat),
    kundliGenerations: toBucketStatus(args.row.kundli_generate_used, args.limits.kundliGenerations),
  };
}

function pickBucket(status: QuotaStatusSnapshot, quotaType: UsageQuotaType): QuotaBucketStatus {
  if (quotaType === 'mini_chat') return status.miniChat;
  if (quotaType === 'pro_chat') return status.proChat;
  return status.kundliGenerations;
}

export class UsageQuotasRepository {
  private async withPool() {
    const pool = getPostgresPool();
    if (!pool) {
      throw new Error('DATABASE_URL is required for usage quotas repository');
    }

    await applyPendingMigrations(pool);
    return pool;
  }

  async getQuotaStatus(ownerId: string, isPro: boolean, anchorDateMs?: number, nowMs = Date.now()): Promise<QuotaStatusSnapshot> {
    const pool = await this.withPool();
    const { periodStartMs, resetAtMs } = resolveMonthlyWindow(nowMs, anchorDateMs);
    const limits = getMonthlyQuotaLimits(isPro);

    const response = await pool.query<MonthlyUsageCounterRow>(
      `
      SELECT owner_id, period_start_ms, mini_chat_used, pro_chat_used, kundli_generate_used, created_at, updated_at
      FROM monthly_usage_counters
      WHERE owner_id = $1 AND period_start_ms = $2
      LIMIT 1
      `,
      [ownerId, periodStartMs]
    );

    const row = response.rows[0] ?? buildEmptyCounterRow(ownerId, periodStartMs, nowMs);

    return buildQuotaStatusSnapshot({
      ownerId,
      planTier: toPlanTier(isPro),
      periodStartMs,
      resetAtMs,
      limits,
      row,
    });
  }

  async consumeQuota(ownerId: string, isPro: boolean, quotaType: UsageQuotaType, anchorDateMs?: number, nowMs = Date.now()): Promise<QuotaConsumeResult> {
    const pool = await this.withPool();
    const { periodStartMs, resetAtMs } = resolveMonthlyWindow(nowMs, anchorDateMs);
    const limits = getMonthlyQuotaLimits(isPro);
    const planTier = toPlanTier(isPro);
    const quotaColumn = QUOTA_FIELD_BY_TYPE[quotaType];

    const client = await pool.connect();
    try {
      await client.query('BEGIN');

      await client.query(
        `
        INSERT INTO monthly_usage_counters (
          owner_id,
          period_start_ms,
          mini_chat_used,
          pro_chat_used,
          kundli_generate_used,
          created_at,
          updated_at
        ) VALUES ($1, $2, 0, 0, 0, $3, $3)
        ON CONFLICT (owner_id, period_start_ms)
        DO NOTHING
        `,
        [ownerId, periodStartMs, nowMs]
      );

      const lockedRowResponse = await client.query<MonthlyUsageCounterRow>(
        `
        SELECT owner_id, period_start_ms, mini_chat_used, pro_chat_used, kundli_generate_used, created_at, updated_at
        FROM monthly_usage_counters
        WHERE owner_id = $1 AND period_start_ms = $2
        LIMIT 1
        FOR UPDATE
        `,
        [ownerId, periodStartMs]
      );

      const lockedRow = lockedRowResponse.rows[0] ?? buildEmptyCounterRow(ownerId, periodStartMs, nowMs);
      const currentStatus = buildQuotaStatusSnapshot({
        ownerId,
        planTier,
        periodStartMs,
        resetAtMs,
        limits,
        row: lockedRow,
      });

      if (pickBucket(currentStatus, quotaType).remaining <= 0) {
        await client.query('COMMIT');
        return {
          allowed: false,
          quotaType,
          status: currentStatus,
        };
      }

      const updatedResponse = await client.query<MonthlyUsageCounterRow>(
        `
        UPDATE monthly_usage_counters
        SET ${quotaColumn} = ${quotaColumn} + 1,
            updated_at = $3
        WHERE owner_id = $1 AND period_start_ms = $2
        RETURNING owner_id, period_start_ms, mini_chat_used, pro_chat_used, kundli_generate_used, created_at, updated_at
        `,
        [ownerId, periodStartMs, nowMs]
      );

      await client.query('COMMIT');

      const updatedRow = updatedResponse.rows[0] ?? lockedRow;
      return {
        allowed: true,
        quotaType,
        status: buildQuotaStatusSnapshot({
          ownerId,
          planTier,
          periodStartMs,
          resetAtMs,
          limits,
          row: updatedRow,
        }),
      };
    } catch (error) {
      await client.query('ROLLBACK').catch(() => {});
      throw error;
    } finally {
      client.release();
    }
  }

  async refundQuota(ownerId: string, quotaType: UsageQuotaType, anchorDateMs?: number, nowMs = Date.now()): Promise<void> {
    const pool = await this.withPool();
    const { periodStartMs } = resolveMonthlyWindow(nowMs, anchorDateMs);
    const quotaColumn = QUOTA_FIELD_BY_TYPE[quotaType];

    await pool.query(
      `UPDATE monthly_usage_counters
       SET ${quotaColumn} = GREATEST(0, ${quotaColumn} - 1),
           updated_at = $3
       WHERE owner_id = $1 AND period_start_ms = $2`,
      [ownerId, periodStartMs, nowMs]
    );
  }
}

let singletonUsageQuotasRepository: UsageQuotasRepository | null = null;

export function getUsageQuotasRepository(): UsageQuotasRepository {
  if (!singletonUsageQuotasRepository) {
    singletonUsageQuotasRepository = new UsageQuotasRepository();
  }

  return singletonUsageQuotasRepository;
}
