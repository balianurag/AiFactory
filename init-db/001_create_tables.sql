-- =============================================================================
-- AiFactory — Initial Database Schema
-- =============================================================================
-- Creates the core tables for persistent task storage, agent tracking,
-- and pipeline metrics. Run automatically by PostgreSQL docker-entrypoint
-- or manually via: psql -d aifactory -f init-db/001_create_tables.sql
-- =============================================================================

-- ── Tasks ────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS tasks (
    task_id       TEXT PRIMARY KEY,
    status        TEXT NOT NULL DEFAULT 'pending',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    progress      REAL NOT NULL DEFAULT 0.0,
    request       JSONB NOT NULL DEFAULT '{}',
    result        JSONB,
    error         TEXT,
    metrics       JSONB NOT NULL DEFAULT '{}'
);

-- ── Agents (per-task agent hierarchy snapshots) ──────────────────────────────
CREATE TABLE IF NOT EXISTS task_agents (
    id            SERIAL PRIMARY KEY,
    task_id       TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    agent_id      TEXT NOT NULL,
    name          TEXT NOT NULL,
    role          TEXT NOT NULL,
    model         TEXT NOT NULL,
    specialty     TEXT,
    parent_id     TEXT,
    status        TEXT NOT NULL DEFAULT 'idle',
    tokens_used   INTEGER NOT NULL DEFAULT 0,
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd      REAL NOT NULL DEFAULT 0.0,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_task_agents_task_id ON task_agents(task_id);

-- ── Messages (agent-to-agent messages for replay/debugging) ──────────────────
CREATE TABLE IF NOT EXISTS task_messages (
    id            SERIAL PRIMARY KEY,
    task_id       TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    agent_id      TEXT,
    message_type  TEXT NOT NULL,
    content       JSONB NOT NULL DEFAULT '{}',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_task_messages_task_id ON task_messages(task_id);

-- ── Function to auto-update updated_at ───────────────────────────────────────
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER trg_tasks_updated_at
    BEFORE UPDATE ON tasks
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();
