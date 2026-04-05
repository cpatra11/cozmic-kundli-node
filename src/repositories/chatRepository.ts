import { randomUUID } from 'node:crypto';
import { getPostgresPool } from '../services/postgresClient.js';
import { applyPendingMigrations } from '../services/postgresMigrations.js';

export interface ChatSessionRecord {
  id: string;
  ownerId: string;
  title: string;
  kundaliId?: string;
  chartVersion?: string;
  lastMessagePreview?: string;
  createdAt: number;
  updatedAt: number;
}

export interface ChatMessageRecord {
  id: string;
  ownerId: string;
  sessionId: string;
  role: 'user' | 'assistant';
  message: string;
  mode?: 'mini' | 'pro';
  model?: string;
  requestId?: string;
  kundaliId?: string;
  bindingId?: string;
  bindingTurn?: number;
  bindingChartVersion?: string;
  bindingKundliSignature?: string;
  embedding?: number[];
  embeddingModel?: string;
  embeddingDim?: number;
  createdAt: number;
}

interface ChatSessionRow {
  id: string;
  owner_id: string;
  title: string;
  kundali_id: string | null;
  chart_version: string | null;
  last_message_preview: string | null;
  created_at: number;
  updated_at: number;
}

interface ChatMessageRow {
  id: string;
  owner_id: string;
  session_id: string;
  role: 'user' | 'assistant';
  message: string;
  mode: 'mini' | 'pro' | null;
  model: string | null;
  request_id: string | null;
  kundali_id: string | null;
  binding_id: string | null;
  binding_turn: number | null;
  binding_chart_version: string | null;
  binding_kundli_signature: string | null;
  embedding: string | null;
  embedding_model: string | null;
  embedding_dim: number | null;
  created_at: number;
}

function parseVectorLiteral(value: string | null): number[] | undefined {
  if (!value) return undefined;
  const trimmed = value.trim();
  if (!trimmed.startsWith('[') || !trimmed.endsWith(']')) return undefined;

  const inner = trimmed.slice(1, -1).trim();
  if (!inner) return [];

  return inner
    .split(',')
    .map((item) => Number(item.trim()))
    .filter((item) => Number.isFinite(item));
}

function toVectorLiteral(values?: number[]): string | null {
  if (!values || values.length === 0) return null;
  return `[${values.map((value) => (Number.isFinite(value) ? value : 0)).join(',')}]`;
}

function mapSessionRow(row: ChatSessionRow): ChatSessionRecord {
  return {
    id: row.id,
    ownerId: row.owner_id,
    title: row.title,
    kundaliId: row.kundali_id ?? undefined,
    chartVersion: row.chart_version ?? undefined,
    lastMessagePreview: row.last_message_preview ?? undefined,
    createdAt: Number(row.created_at),
    updatedAt: Number(row.updated_at),
  };
}

function mapMessageRow(row: ChatMessageRow): ChatMessageRecord {
  return {
    id: row.id,
    ownerId: row.owner_id,
    sessionId: row.session_id,
    role: row.role,
    message: row.message,
    mode: row.mode ?? undefined,
    model: row.model ?? undefined,
    requestId: row.request_id ?? undefined,
    kundaliId: row.kundali_id ?? undefined,
    bindingId: row.binding_id ?? undefined,
    bindingTurn: row.binding_turn ?? undefined,
    bindingChartVersion: row.binding_chart_version ?? undefined,
    bindingKundliSignature: row.binding_kundli_signature ?? undefined,
    embedding: parseVectorLiteral(row.embedding),
    embeddingModel: row.embedding_model ?? undefined,
    embeddingDim: row.embedding_dim ?? undefined,
    createdAt: Number(row.created_at),
  };
}

export class ChatRepository {
  private async withPool() {
    const pool = getPostgresPool();
    if (!pool) {
      throw new Error('DATABASE_URL is required for chat repository');
    }

    await applyPendingMigrations(pool);
    return pool;
  }

  async createSession(input: Omit<ChatSessionRecord, 'id'> & { id?: string }): Promise<string> {
    const pool = await this.withPool();
    const id = input.id ?? randomUUID();

    await pool.query(
      `
      INSERT INTO chat_sessions (id, owner_id, title, kundali_id, chart_version, last_message_preview, created_at, updated_at)
      VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
      `,
      [
        id,
        input.ownerId,
        input.title,
        input.kundaliId ?? null,
        input.chartVersion ?? null,
        input.lastMessagePreview ?? null,
        input.createdAt,
        input.updatedAt,
      ]
    );

    return id;
  }

  async getSessionById(id: string): Promise<ChatSessionRecord | null> {
    const pool = await this.withPool();
    const response = await pool.query<ChatSessionRow>(
      `
      SELECT id, owner_id, title, kundali_id, chart_version, last_message_preview, created_at, updated_at
      FROM chat_sessions
      WHERE id = $1
      LIMIT 1
      `,
      [id]
    );

    const row = response.rows[0];
    return row ? mapSessionRow(row) : null;
  }

  async listSessionsByOwner(ownerId: string, limit = 50): Promise<ChatSessionRecord[]> {
    const pool = await this.withPool();
    const response = await pool.query<ChatSessionRow>(
      `
      SELECT id, owner_id, title, kundali_id, chart_version, last_message_preview, created_at, updated_at
      FROM chat_sessions
      WHERE owner_id = $1
      ORDER BY updated_at DESC
      LIMIT $2
      `,
      [ownerId, Math.max(1, Math.floor(limit))]
    );

    return response.rows.map(mapSessionRow);
  }

  async updateSession(id: string, ownerId: string, patch: Partial<Omit<ChatSessionRecord, 'id' | 'ownerId' | 'createdAt'>>): Promise<void> {
    const pool = await this.withPool();
    await pool.query(
      `
      UPDATE chat_sessions
      SET
        title = COALESCE($3, title),
        kundali_id = COALESCE($4, kundali_id),
        chart_version = COALESCE($5, chart_version),
        last_message_preview = COALESCE($6, last_message_preview),
        updated_at = $7
      WHERE id = $1 AND owner_id = $2
      `,
      [
        id,
        ownerId,
        patch.title ?? null,
        patch.kundaliId ?? null,
        patch.chartVersion ?? null,
        patch.lastMessagePreview ?? null,
        patch.updatedAt ?? Date.now(),
      ]
    );
  }

  async createMessage(input: Omit<ChatMessageRecord, 'id'> & { id?: string }): Promise<string> {
    const pool = await this.withPool();
    const id = input.id ?? randomUUID();

    await pool.query(
      `
      INSERT INTO chat_messages (
        id,
        owner_id,
        session_id,
        role,
        message,
        mode,
        model,
        request_id,
        kundali_id,
        binding_id,
        binding_turn,
        binding_chart_version,
        binding_kundli_signature,
        embedding,
        embedding_model,
        embedding_dim,
        created_at
      ) VALUES (
        $1,
        $2,
        $3,
        $4,
        $5,
        $6,
        $7,
        $8,
        $9,
        $10,
        $11,
        $12,
        $13,
        $14::vector,
        $15,
        $16,
        $17
      )
      `,
      [
        id,
        input.ownerId,
        input.sessionId,
        input.role,
        input.message,
        input.mode ?? null,
        input.model ?? null,
        input.requestId ?? null,
        input.kundaliId ?? null,
        input.bindingId ?? null,
        input.bindingTurn ?? null,
        input.bindingChartVersion ?? null,
        input.bindingKundliSignature ?? null,
        toVectorLiteral(input.embedding),
        input.embeddingModel ?? null,
        input.embeddingDim ?? null,
        input.createdAt,
      ]
    );

    return id;
  }

  async listSessionMessages(ownerId: string, sessionId: string, limit = 500): Promise<ChatMessageRecord[]> {
    const pool = await this.withPool();
    const response = await pool.query<ChatMessageRow>(
      `
      SELECT
        id,
        owner_id,
        session_id,
        role,
        message,
        mode,
        model,
        request_id,
        kundali_id,
        binding_id,
        binding_turn,
        binding_chart_version,
        binding_kundli_signature,
        embedding::text AS embedding,
        embedding_model,
        embedding_dim,
        created_at
      FROM chat_messages
      WHERE owner_id = $1 AND session_id = $2
      ORDER BY created_at ASC
      LIMIT $3
      `,
      [ownerId, sessionId, Math.max(1, Math.floor(limit))]
    );

    return response.rows.map(mapMessageRow);
  }

  async listMessagesForMemory(ownerId: string, sessionId: string, limit = 120): Promise<ChatMessageRecord[]> {
    const pool = await this.withPool();
    const response = await pool.query<ChatMessageRow>(
      `
      SELECT
        id,
        owner_id,
        session_id,
        role,
        message,
        mode,
        model,
        request_id,
        kundali_id,
        binding_id,
        binding_turn,
        binding_chart_version,
        binding_kundli_signature,
        embedding::text AS embedding,
        embedding_model,
        embedding_dim,
        created_at
      FROM chat_messages
      WHERE owner_id = $1 AND session_id = $2
      ORDER BY created_at DESC
      LIMIT $3
      `,
      [ownerId, sessionId, Math.max(1, Math.floor(limit))]
    );

    return response.rows.map(mapMessageRow);
  }

  async deleteSessionsByOwnerKundali(ownerId: string, kundaliId: string): Promise<void> {
    const pool = await this.withPool();
    await pool.query(
      `
      DELETE FROM chat_sessions
      WHERE owner_id = $1 AND kundali_id = $2
      `,
      [ownerId, kundaliId]
    );
  }
}

let singletonChatRepository: ChatRepository | null = null;

export function getChatRepository(): ChatRepository {
  if (!singletonChatRepository) {
    singletonChatRepository = new ChatRepository();
  }

  return singletonChatRepository;
}
