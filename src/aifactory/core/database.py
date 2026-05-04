"""
Async + sync PostgreSQL persistence layer for AiFactory.

Async engine (asyncpg)  — used by the FastAPI layer for tasks/agents/messages.
Sync engine  (psycopg2) — used by RL scorer, RunStore, FactsStore, ResearchCache
                          which are called from sync or mixed sync/async contexts.

Tables (all created by setup_db()):
  users          — user accounts (email, password_hash, display_name)
  organizations  — team/org accounts (name, slug, plan, owner_id)
  org_members    — user ↔ org membership with role
  api_keys       — per-user API keys (prefix + hash stored; raw shown once)
  refresh_tokens — JWT refresh token store (hashed)
  tasks          — HTTP API task metadata, status, result, metrics
  task_agents    — per-task agent hierarchy snapshot
  task_messages  — agent-to-agent messages for replay
  runs           — unified CLI + API pipeline run records (single source of truth)
  rl_performance — RL-lite model quality scores per (model_id, task_type)
  facts          — REMEMBER: fact persistence (cross-run, multi-process safe)
  research_cache — researcher agent output cache (multi-process safe)
"""

from __future__ import annotations

import contextlib
import json
import logging
from datetime import datetime
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from aifactory.config.settings import get_settings

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# DDL — all tables created with IF NOT EXISTS (idempotent)
# ─────────────────────────────────────────────────────────────────────────────

_DDL = """
-- ── User Account Management ─────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email           TEXT UNIQUE NOT NULL,
    email_verified  BOOLEAN NOT NULL DEFAULT FALSE,
    password_hash   TEXT NOT NULL,
    display_name    TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_login_at   TIMESTAMPTZ,
    is_active       BOOLEAN NOT NULL DEFAULT TRUE
);
CREATE INDEX IF NOT EXISTS users_email_idx ON users (email);

CREATE TABLE IF NOT EXISTS organizations (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name       TEXT NOT NULL,
    slug       TEXT UNIQUE NOT NULL,
    owner_id   UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    plan       TEXT NOT NULL DEFAULT 'starter',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS org_members (
    user_id   UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    org_id    UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    role      TEXT NOT NULL DEFAULT 'member',
    joined_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, org_id)
);

CREATE TABLE IF NOT EXISTS api_keys (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    org_id       UUID REFERENCES organizations(id) ON DELETE SET NULL,
    name         TEXT NOT NULL,
    key_prefix   TEXT NOT NULL,
    key_hash     TEXT UNIQUE NOT NULL,
    scopes       TEXT[] NOT NULL DEFAULT '{}',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_used_at TIMESTAMPTZ,
    expires_at   TIMESTAMPTZ,
    is_active    BOOLEAN NOT NULL DEFAULT TRUE
);
CREATE INDEX IF NOT EXISTS api_keys_user_idx  ON api_keys (user_id);
CREATE INDEX IF NOT EXISTS api_keys_hash_idx  ON api_keys (key_hash);

CREATE TABLE IF NOT EXISTS refresh_tokens (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash   TEXT UNIQUE NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at   TIMESTAMPTZ NOT NULL,
    revoked_at   TIMESTAMPTZ,
    ip_addr      TEXT,
    user_agent   TEXT
);
CREATE INDEX IF NOT EXISTS refresh_tokens_user_idx ON refresh_tokens (user_id);
CREATE INDEX IF NOT EXISTS refresh_tokens_hash_idx ON refresh_tokens (token_hash);

-- ── Pipeline / Task Tables ──────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS tasks (
    task_id     TEXT PRIMARY KEY,
    user_id     UUID REFERENCES users(id) ON DELETE SET NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    request     JSONB,
    result      JSONB,
    metrics     JSONB,
    progress    NUMERIC DEFAULT 0,
    error       TEXT
);
CREATE INDEX IF NOT EXISTS tasks_user_idx ON tasks (user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS task_agents (
    id            BIGSERIAL PRIMARY KEY,
    task_id       TEXT NOT NULL,
    agent_id      TEXT NOT NULL,
    name          TEXT NOT NULL,
    role          TEXT,
    model         TEXT,
    specialty     TEXT,
    parent_id     TEXT,
    status        TEXT DEFAULT 'idle',
    tokens_used   INT DEFAULT 0,
    input_tokens  INT DEFAULT 0,
    output_tokens INT DEFAULT 0,
    cost_usd      NUMERIC(10,6) DEFAULT 0
);

CREATE TABLE IF NOT EXISTS task_messages (
    id           BIGSERIAL PRIMARY KEY,
    task_id      TEXT NOT NULL,
    agent_id     TEXT,
    message_type TEXT,
    content      JSONB,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    user_id       UUID REFERENCES users(id) ON DELETE SET NULL,
    source        TEXT NOT NULL DEFAULT 'cli',
    status        TEXT NOT NULL DEFAULT 'running',
    task          TEXT,
    mode          TEXT,
    gate          TEXT,
    started_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at  TIMESTAMPTZ,
    elapsed_s     NUMERIC(10,3),
    total_tokens  INT DEFAULT 0,
    final_preview TEXT,
    judge_status  TEXT,
    quality       NUMERIC(4,2),
    metadata      JSONB DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS runs_started_at_idx ON runs (started_at DESC);
CREATE INDEX IF NOT EXISTS runs_source_started_idx ON runs (source, started_at DESC);
CREATE INDEX IF NOT EXISTS runs_user_idx ON runs (user_id, started_at DESC);

CREATE TABLE IF NOT EXISTS rl_performance (
    id            BIGSERIAL PRIMARY KEY,
    model_id      TEXT NOT NULL,
    task_type     TEXT NOT NULL,
    quality_score NUMERIC(5,3) NOT NULL,
    tokens_used   INT DEFAULT 0,
    run_id        TEXT DEFAULT '',
    recorded_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS rl_perf_task_model_idx ON rl_performance (task_type, model_id);

CREATE TABLE IF NOT EXISTS facts (
    id         BIGSERIAL PRIMARY KEY,
    content    TEXT NOT NULL UNIQUE,
    from_agent TEXT DEFAULT '',
    run_id     TEXT DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS research_cache (
    id          BIGSERIAL PRIMARY KEY,
    query_key   TEXT NOT NULL UNIQUE,
    query_text  TEXT NOT NULL,
    summary     TEXT NOT NULL,
    embedding   JSONB,
    run_id      TEXT DEFAULT '',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS research_cache_key_idx ON research_cache (query_key);
"""


# ─────────────────────────────────────────────────────────────────────────────
# Async engine (asyncpg) — FastAPI / aiohttp layer
# ─────────────────────────────────────────────────────────────────────────────

_engine: AsyncEngine | None = None


def get_engine() -> AsyncEngine:
    """Return (or create) the async SQLAlchemy engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        url = settings.database_url
        if not url:
            raise RuntimeError(
                "DATABASE_URL is not set — cannot connect to PostgreSQL. "
                "Set DATABASE_URL in .env or as an environment variable."
            )
        url = _to_asyncpg_url(url)
        _engine = create_async_engine(
            url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            echo=False,
        )
        logger.info("PostgreSQL async engine created: %s", _safe_url(url))
    return _engine


async def close_engine() -> None:
    """Dispose the async engine (call on app shutdown)."""
    global _engine
    if _engine:
        await _engine.dispose()
        _engine = None
        logger.info("PostgreSQL async engine disposed")


def reset_engine() -> None:
    """Reset the async engine reference (for testing)."""
    global _engine
    _engine = None


# ─────────────────────────────────────────────────────────────────────────────
# Sync engine (psycopg2) — RL scorer, RunStore, FactsStore, ResearchCache
# ─────────────────────────────────────────────────────────────────────────────

_sync_engine: Any = None  # sqlalchemy.engine.Engine


def get_sync_engine() -> Any:
    """
    Return (or create) the synchronous SQLAlchemy engine (psycopg2).

    Returns None if DATABASE_URL is not configured — callers must handle None
    gracefully and fall back to their own local persistence.
    """
    global _sync_engine
    if _sync_engine is None:
        settings = get_settings()
        url = settings.database_url
        if not url:
            return None
        url = _to_psycopg2_url(url)
        try:
            _sync_engine = create_engine(
                url,
                pool_size=3,
                max_overflow=5,
                pool_pre_ping=True,
                echo=False,
            )
            logger.info("PostgreSQL sync engine created: %s", _safe_url(url))
        except Exception as exc:
            logger.warning("PostgreSQL sync engine creation failed: %s", exc)
            return None
    return _sync_engine


def reset_sync_engine() -> None:
    """Reset the sync engine reference (for testing)."""
    global _sync_engine
    _sync_engine = None


# ─────────────────────────────────────────────────────────────────────────────
# DDL / Migrations
# ─────────────────────────────────────────────────────────────────────────────


# Additive migrations — no FK to users (avoids ordering issues on existing DBs).
# Each statement is run individually so one failure doesn't block the rest.
_DDL_MIGRATIONS: list[str] = [
    "ALTER TABLE tasks ADD COLUMN IF NOT EXISTS metrics JSONB",
    "ALTER TABLE tasks ADD COLUMN IF NOT EXISTS progress NUMERIC DEFAULT 0",
    "ALTER TABLE tasks ADD COLUMN IF NOT EXISTS error TEXT",
    "ALTER TABLE runs ADD COLUMN IF NOT EXISTS gate TEXT",
    "ALTER TABLE runs ADD COLUMN IF NOT EXISTS quality NUMERIC(4,2)",
    "ALTER TABLE runs ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'",
    "CREATE INDEX IF NOT EXISTS runs_started_at_idx ON runs (started_at DESC)",
    "CREATE INDEX IF NOT EXISTS runs_source_started_idx ON runs (source, started_at DESC)",
    "CREATE INDEX IF NOT EXISTS rl_perf_task_model_idx ON rl_performance (task_type, model_id)",
    "CREATE INDEX IF NOT EXISTS research_cache_key_idx ON research_cache (query_key)",
]


# Separate DDL split into individual statements for statement-by-statement execution
def _split_ddl(ddl: str) -> list[str]:
    return [s.strip() for s in ddl.split(";") if s.strip()]


def setup_db() -> None:
    """
    Create all tables if they do not already exist, then apply additive migrations.

    Executes each statement individually so a pre-existing table (or column)
    never blocks the rest of the schema from being applied.
    Safe to call multiple times (idempotent).
    """
    engine = get_sync_engine()
    if engine is None:
        logger.warning(
            "DATABASE_URL not set — skipping database setup. "
            "Persistent storage will be unavailable."
        )
        return

    statements = _split_ddl(_DDL)
    ok = 0
    for stmt in statements:
        try:
            with engine.begin() as conn:
                conn.execute(text(stmt))
            ok += 1
        except Exception as exc:
            # Most failures here are "already exists" — non-fatal
            logger.debug("DDL skipped (non-fatal): %s | %s", exc.__class__.__name__, str(exc)[:120])

    logger.info("Database schema applied (%d/%d statements succeeded)", ok, len(statements))

    for stmt in _DDL_MIGRATIONS:
        try:
            with engine.begin() as conn:
                conn.execute(text(stmt))
        except Exception as exc:
            logger.debug("Migration skipped: %s | %s", exc.__class__.__name__, str(exc)[:80])


# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────


async def pg_health() -> bool:
    """Return True if PostgreSQL is reachable (async)."""
    try:
        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.warning("PostgreSQL health check failed: %s", e)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Tasks — async CRUD (FastAPI layer, unchanged)
# ─────────────────────────────────────────────────────────────────────────────


async def create_task(
    task_id: str,
    status: str,
    created_at: str,
    request: dict[str, Any],
) -> None:
    """Insert a new task row."""
    engine = get_engine()
    ts = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO tasks (task_id, status, created_at, updated_at, request) "
                "VALUES (:task_id, :status, :created_at, :created_at, :request)"
            ),
            {
                "task_id": task_id,
                "status": status,
                "created_at": ts,
                "request": json.dumps(request),
            },
        )


async def update_task(task_id: str, **fields: Any) -> None:
    """Update task fields dynamically. JSONB fields are auto-serialised."""
    if not fields:
        return
    engine = get_engine()
    set_clauses = []
    params: dict[str, Any] = {"task_id": task_id}
    for key, value in fields.items():
        if key in ("result", "metrics", "request"):
            params[key] = json.dumps(value) if value is not None else None
        else:
            params[key] = value
        set_clauses.append(f"{key} = :{key}")
    sql = f"UPDATE tasks SET {', '.join(set_clauses)} WHERE task_id = :task_id"
    async with engine.begin() as conn:
        await conn.execute(text(sql), params)


async def get_task(task_id: str) -> dict[str, Any] | None:
    """Fetch a single task by ID. Returns None if not found."""
    engine = get_engine()
    async with engine.connect() as conn:
        row = (
            (
                await conn.execute(
                    text("SELECT * FROM tasks WHERE task_id = :task_id"),
                    {"task_id": task_id},
                )
            )
            .mappings()
            .first()
        )
    if row is None:
        return None
    return _row_to_task(dict(row))


async def list_tasks(
    status_filter: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List tasks, newest first. Optional status filter."""
    engine = get_engine()
    params: dict[str, Any] = {"limit": limit}
    sql = "SELECT * FROM tasks"
    if status_filter:
        sql += " WHERE status = :status"
        params["status"] = status_filter
    sql += " ORDER BY created_at DESC LIMIT :limit"
    async with engine.connect() as conn:
        rows = (await conn.execute(text(sql), params)).mappings().all()
    return [_row_to_task(dict(r)) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Task Agents — async CRUD
# ─────────────────────────────────────────────────────────────────────────────


async def upsert_agents(task_id: str, agents: list[dict[str, Any]]) -> None:
    """Replace the agent hierarchy snapshot for a task."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM task_agents WHERE task_id = :task_id"),
            {"task_id": task_id},
        )
        for a in agents:
            await conn.execute(
                text(
                    "INSERT INTO task_agents "
                    "(task_id, agent_id, name, role, model, specialty, parent_id, "
                    " status, tokens_used, input_tokens, output_tokens, cost_usd) "
                    "VALUES (:task_id, :agent_id, :name, :role, :model, :specialty, "
                    " :parent_id, :status, :tokens_used, :input_tokens, :output_tokens, :cost_usd)"
                ),
                {
                    "task_id": task_id,
                    "agent_id": a.get("agent_id", ""),
                    "name": a.get("name", ""),
                    "role": a.get("role", ""),
                    "model": a.get("model", ""),
                    "specialty": a.get("specialty"),
                    "parent_id": a.get("parent_id"),
                    "status": a.get("status", "idle"),
                    "tokens_used": a.get("tokens_used", 0),
                    "input_tokens": a.get("input_tokens", 0),
                    "output_tokens": a.get("output_tokens", 0),
                    "cost_usd": a.get("cost_usd", 0.0),
                },
            )


async def get_agents(task_id: str) -> list[dict[str, Any]]:
    """Fetch the agent hierarchy for a task."""
    engine = get_engine()
    async with engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    text(
                        "SELECT agent_id, name, role, model, specialty, parent_id, "
                        "status, tokens_used, input_tokens, output_tokens, cost_usd "
                        "FROM task_agents WHERE task_id = :task_id ORDER BY id"
                    ),
                    {"task_id": task_id},
                )
            )
            .mappings()
            .all()
        )
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Task Messages — async CRUD
# ─────────────────────────────────────────────────────────────────────────────


async def add_message(
    task_id: str,
    agent_id: str | None,
    message_type: str,
    content: dict[str, Any],
) -> None:
    """Append a message to the task message log."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO task_messages (task_id, agent_id, message_type, content) "
                "VALUES (:task_id, :agent_id, :message_type, :content)"
            ),
            {
                "task_id": task_id,
                "agent_id": agent_id,
                "message_type": message_type,
                "content": json.dumps(content),
            },
        )


async def get_messages(task_id: str) -> list[dict[str, Any]]:
    """Fetch all messages for a task."""
    engine = get_engine()
    async with engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    text(
                        "SELECT agent_id, message_type, content, created_at "
                        "FROM task_messages WHERE task_id = :task_id ORDER BY id"
                    ),
                    {"task_id": task_id},
                )
            )
            .mappings()
            .all()
        )
    result = []
    for r in rows:
        d = dict(r)
        if isinstance(d.get("content"), str):
            d["content"] = json.loads(d["content"])
        if hasattr(d.get("created_at"), "isoformat"):
            d["created_at"] = d["created_at"].isoformat()
        result.append(d)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Runs — sync CRUD (unified CLI + API run history)
# ─────────────────────────────────────────────────────────────────────────────


def save_run(
    run_id: str,
    source: str,
    task: str,
    mode: str,
    gate: str = "",
    status: str = "running",
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Insert a new run record.  Call at pipeline start with status='running'.

    Non-fatal — logs and returns on any error.
    """
    engine = get_sync_engine()
    if engine is None:
        return
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO runs (run_id, source, status, task, mode, gate, metadata) "
                    "VALUES (:run_id, :source, :status, :task, :mode, :gate, :metadata) "
                    "ON CONFLICT (run_id) DO NOTHING"
                ),
                {
                    "run_id": run_id,
                    "source": source,
                    "status": status,
                    "task": task[:2000] if task else "",
                    "mode": mode,
                    "gate": gate,
                    "metadata": json.dumps(metadata or {}),
                },
            )
    except Exception as exc:
        logger.debug("save_run failed (non-fatal): %s", exc)


def update_run_status(
    run_id: str,
    status: str,
    elapsed_s: float = 0.0,
    total_tokens: int = 0,
    final_preview: str = "",
    judge_status: str = "",
    quality: float | None = None,
) -> None:
    """
    Update a run record on completion or failure.  Non-fatal.
    """
    engine = get_sync_engine()
    if engine is None:
        return
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE runs SET status = :status, completed_at = NOW(), "
                    "elapsed_s = :elapsed_s, total_tokens = :total_tokens, "
                    "final_preview = :preview, judge_status = :judge_status, "
                    "quality = :quality "
                    "WHERE run_id = :run_id"
                ),
                {
                    "run_id": run_id,
                    "status": status,
                    "elapsed_s": round(elapsed_s, 3),
                    "total_tokens": total_tokens,
                    "preview": final_preview[:500] if final_preview else "",
                    "judge_status": judge_status,
                    "quality": quality,
                },
            )
    except Exception as exc:
        logger.debug("update_run_status failed (non-fatal): %s", exc)


def get_run_pg(run_id: str) -> dict[str, Any] | None:
    """Fetch a single run record by ID.  Returns None if not found."""
    engine = get_sync_engine()
    if engine is None:
        return None
    try:
        with engine.connect() as conn:
            row = (
                conn.execute(
                    text("SELECT * FROM runs WHERE run_id = :run_id"),
                    {"run_id": run_id},
                )
                .mappings()
                .first()
            )
        return _row_to_run(dict(row)) if row else None
    except Exception as exc:
        logger.debug("get_run_pg failed: %s", exc)
        return None


def list_runs_pg(
    limit: int = 20,
    source: str | None = None,
) -> list[dict[str, Any]]:
    """List runs newest-first.  Optional source filter ('cli' | 'api')."""
    engine = get_sync_engine()
    if engine is None:
        return []
    try:
        params: dict[str, Any] = {"limit": limit}
        sql = "SELECT * FROM runs"
        if source:
            sql += " WHERE source = :source"
            params["source"] = source
        sql += " ORDER BY started_at DESC LIMIT :limit"
        with engine.connect() as conn:
            rows = conn.execute(text(sql), params).mappings().all()
        return [_row_to_run(dict(r)) for r in rows]
    except Exception as exc:
        logger.debug("list_runs_pg failed: %s", exc)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# RL Performance — sync CRUD
# ─────────────────────────────────────────────────────────────────────────────


def insert_rl_record(
    model_id: str,
    task_type: str,
    quality_score: float,
    tokens_used: int = 0,
    run_id: str = "",
) -> None:
    """Insert one RL performance record.  Non-fatal."""
    engine = get_sync_engine()
    if engine is None:
        return
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO rl_performance "
                    "(model_id, task_type, quality_score, tokens_used, run_id) "
                    "VALUES (:model_id, :task_type, :quality_score, :tokens_used, :run_id)"
                ),
                {
                    "model_id": model_id,
                    "task_type": task_type,
                    "quality_score": round(float(quality_score), 3),
                    "tokens_used": int(tokens_used),
                    "run_id": run_id,
                },
            )
    except Exception as exc:
        logger.debug("insert_rl_record failed (non-fatal): %s", exc)


def query_rl_scores(
    task_type: str,
    min_samples: int = 3,
) -> dict[str, float]:
    """
    Return average quality scores keyed by model_id for a task type.

    Only includes models with >= min_samples records.
    Returns empty dict if database unavailable or no qualifying records.
    """
    engine = get_sync_engine()
    if engine is None:
        return {}
    try:
        with engine.connect() as conn:
            rows = (
                conn.execute(
                    text(
                        "SELECT model_id, AVG(quality_score) AS avg_q, COUNT(*) AS cnt "
                        "FROM rl_performance "
                        "WHERE task_type = :task_type "
                        "GROUP BY model_id "
                        "HAVING COUNT(*) >= :min_samples"
                    ),
                    {"task_type": task_type, "min_samples": min_samples},
                )
                .mappings()
                .all()
            )
        return {r["model_id"]: round(float(r["avg_q"]), 3) for r in rows}
    except Exception as exc:
        logger.debug("query_rl_scores failed: %s", exc)
        return {}


def get_all_rl_stats() -> dict[str, Any]:
    """Return aggregated RL stats for CLI display."""
    engine = get_sync_engine()
    if engine is None:
        return {"total_records": 0, "by_task_type": {}}
    try:
        with engine.connect() as conn:
            total = conn.execute(text("SELECT COUNT(*) FROM rl_performance")).scalar() or 0
            if total == 0:
                return {"total_records": 0, "by_task_type": {}}

            rows = (
                conn.execute(
                    text(
                        "SELECT task_type, model_id, "
                        "AVG(quality_score) AS avg_q, COUNT(*) AS cnt "
                        "FROM rl_performance "
                        "GROUP BY task_type, model_id "
                        "ORDER BY task_type, avg_q DESC"
                    )
                )
                .mappings()
                .all()
            )

        # Aggregate by task_type
        by_type: dict[str, dict[str, Any]] = {}
        for r in rows:
            tt = r["task_type"]
            mid = r["model_id"]
            avg = round(float(r["avg_q"]), 2)
            cnt = int(r["cnt"])
            if tt not in by_type:
                by_type[tt] = {
                    "records": 0,
                    "models": 0,
                    "best_model": mid,
                    "best_avg_score": avg,
                    "model_avgs": {},
                }
            by_type[tt]["records"] += cnt
            by_type[tt]["models"] += 1
            by_type[tt]["model_avgs"][mid] = avg
            if avg > by_type[tt]["best_avg_score"]:
                by_type[tt]["best_model"] = mid
                by_type[tt]["best_avg_score"] = avg

        return {"total_records": int(total), "by_task_type": by_type}
    except Exception as exc:
        logger.debug("get_all_rl_stats failed: %s", exc)
        return {"total_records": 0, "by_task_type": {}, "error": str(exc)}


def delete_rl_records(task_type: str | None = None) -> int:
    """Delete RL records.  Returns row count deleted.  Non-fatal (returns 0 on error)."""
    engine = get_sync_engine()
    if engine is None:
        return 0
    try:
        with engine.begin() as conn:
            if task_type:
                result = conn.execute(
                    text("DELETE FROM rl_performance WHERE task_type = :tt"),
                    {"tt": task_type},
                )
            else:
                result = conn.execute(text("DELETE FROM rl_performance"))
        return result.rowcount or 0
    except Exception as exc:
        logger.debug("delete_rl_records failed: %s", exc)
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Facts — sync CRUD
# ─────────────────────────────────────────────────────────────────────────────


def upsert_fact(content: str, from_agent: str = "", run_id: str = "") -> None:
    """
    Store a REMEMBER: fact.  ON CONFLICT (content) DO NOTHING deduplicates.
    Raises on DB error — callers (FactsStore) catch and fall back to file.
    """
    engine = get_sync_engine()
    if engine is None:
        raise RuntimeError("No sync engine — DATABASE_URL not set")
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO facts (content, from_agent, run_id) "
                "VALUES (:content, :from_agent, :run_id) "
                "ON CONFLICT (content) DO NOTHING"
            ),
            {"content": content.strip(), "from_agent": from_agent, "run_id": run_id},
        )


def load_all_facts() -> list[str]:
    """Return all stored fact content strings, oldest first.  Returns [] on error."""
    engine = get_sync_engine()
    if engine is None:
        return []
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT content FROM facts ORDER BY created_at ASC")
            ).fetchall()
        return [r[0] for r in rows if r[0]]
    except Exception as exc:
        logger.debug("load_all_facts failed: %s", exc)
        return []


def count_facts() -> int:
    """Return number of stored facts."""
    engine = get_sync_engine()
    if engine is None:
        return 0
    try:
        with engine.connect() as conn:
            return conn.execute(text("SELECT COUNT(*) FROM facts")).scalar() or 0
    except Exception as exc:
        logger.debug("count_facts failed: %s", exc)
        return 0


def clear_facts() -> int:
    """Delete all facts. Returns row count."""
    engine = get_sync_engine()
    if engine is None:
        return 0
    try:
        with engine.begin() as conn:
            result = conn.execute(text("DELETE FROM facts"))
        return result.rowcount or 0
    except Exception as exc:
        logger.debug("clear_facts failed: %s", exc)
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Research Cache — sync CRUD
# ─────────────────────────────────────────────────────────────────────────────


def upsert_research_entry(
    query_key: str,
    query_text: str,
    summary: str,
    embedding: list[float] | None = None,
    run_id: str = "",
) -> None:
    """
    Store a research cache entry.  ON CONFLICT (query_key) updates the summary.
    Raises on DB error — callers (ResearchCache) catch and fall back to file.
    """
    engine = get_sync_engine()
    if engine is None:
        raise RuntimeError("No sync engine — DATABASE_URL not set")
    emb_json = json.dumps(embedding) if embedding else None
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO research_cache "
                "(query_key, query_text, summary, embedding, run_id) "
                "VALUES (:key, :query, :summary, :emb, :run_id) "
                "ON CONFLICT (query_key) DO UPDATE SET "
                "summary = EXCLUDED.summary, embedding = EXCLUDED.embedding, "
                "run_id = EXCLUDED.run_id"
            ),
            {
                "key": query_key,
                "query": query_text,
                "summary": summary,
                "emb": emb_json,
                "run_id": run_id,
            },
        )


def get_research_entry(query_key: str) -> dict[str, Any] | None:
    """Fetch a research cache entry by normalised key.  Returns None on miss."""
    engine = get_sync_engine()
    if engine is None:
        return None
    try:
        with engine.connect() as conn:
            row = (
                conn.execute(
                    text(
                        "SELECT query_key, query_text, summary, embedding, run_id "
                        "FROM research_cache WHERE query_key = :key"
                    ),
                    {"key": query_key},
                )
                .mappings()
                .first()
            )
        if row is None:
            return None
        d = dict(row)
        if isinstance(d.get("embedding"), str):
            with contextlib.suppress(Exception):
                d["embedding"] = json.loads(d["embedding"])
        return d
    except Exception as exc:
        logger.debug("get_research_entry failed: %s", exc)
        return None


def get_all_research_entries() -> list[dict[str, Any]]:
    """Return all research cache entries (for word-overlap fallback scan)."""
    engine = get_sync_engine()
    if engine is None:
        return []
    try:
        with engine.connect() as conn:
            rows = (
                conn.execute(
                    text(
                        "SELECT query_key, query_text, summary, embedding, run_id "
                        "FROM research_cache ORDER BY created_at DESC"
                    )
                )
                .mappings()
                .all()
            )
        result = []
        for r in rows:
            d = dict(r)
            if isinstance(d.get("embedding"), str):
                with contextlib.suppress(Exception):
                    d["embedding"] = json.loads(d["embedding"])
            result.append(d)
        return result
    except Exception as exc:
        logger.debug("get_all_research_entries failed: %s", exc)
        return []


def clear_research_entries() -> int:
    """Delete all research cache entries. Returns row count."""
    engine = get_sync_engine()
    if engine is None:
        return 0
    try:
        with engine.begin() as conn:
            result = conn.execute(text("DELETE FROM research_cache"))
        return result.rowcount or 0
    except Exception as exc:
        logger.debug("clear_research_entries failed: %s", exc)
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _to_asyncpg_url(url: str) -> str:
    """Convert any PostgreSQL URL variant to postgresql+asyncpg://."""
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if url.startswith("postgresql+psycopg2://"):
        return url.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)
    return url  # assume already correct or custom scheme


def _to_psycopg2_url(url: str) -> str:
    """Convert any PostgreSQL URL variant to postgresql+psycopg2://."""
    if url.startswith("postgresql+asyncpg://"):
        return url.replace("postgresql+asyncpg://", "postgresql+psycopg2://", 1)
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg2://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return url


def _safe_url(url: str) -> str:
    """Strip credentials from URL for safe logging."""
    return url.split("@")[-1] if "@" in url else url


def _row_to_task(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a raw DB row to the task dict format used by the API."""
    for col in ("request", "result", "metrics"):
        val = row.get(col)
        if isinstance(val, str):
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                row[col] = json.loads(val)
    for col in ("created_at", "updated_at"):
        val = row.get(col)
        if hasattr(val, "isoformat"):
            row[col] = val.isoformat().replace("+00:00", "Z")
    return row


def _row_to_run(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a raw runs row to a plain dict with ISO timestamps."""
    for col in ("started_at", "completed_at"):
        val = row.get(col)
        if hasattr(val, "isoformat"):
            row[col] = val.isoformat().replace("+00:00", "Z")
    if isinstance(row.get("metadata"), str):
        with contextlib.suppress(Exception):
            row["metadata"] = json.loads(row["metadata"])
    return row


# ─────────────────────────────────────────────────────────────────────────────
# User Account CRUD (async — used by FastAPI auth routes)
# ─────────────────────────────────────────────────────────────────────────────


async def create_user(
    email: str,
    password_hash: str,
    display_name: str | None = None,
) -> dict[str, Any]:
    """Insert a new user row.  Raises ValueError on duplicate email."""
    async with get_engine().begin() as conn:
        row = await conn.execute(
            text(
                "INSERT INTO users (email, password_hash, display_name) "
                "VALUES (:email, :ph, :dn) "
                "RETURNING id, email, display_name, created_at, is_active, email_verified"
            ),
            {"email": email.lower().strip(), "ph": password_hash, "dn": display_name},
        )
        result = row.mappings().one_or_none()
        if result is None:
            raise ValueError(f"User with email {email!r} already exists")
        return _row_to_user(dict(result))


async def get_user_by_email(email: str) -> dict[str, Any] | None:
    """Return a user row by email, or None."""
    async with get_engine().connect() as conn:
        row = await conn.execute(
            text(
                "SELECT id, email, display_name, password_hash, "
                "created_at, last_login_at, is_active, email_verified "
                "FROM users WHERE email = :email"
            ),
            {"email": email.lower().strip()},
        )
        result = row.mappings().one_or_none()
        return _row_to_user(dict(result)) if result else None


async def get_user_by_id(user_id: str) -> dict[str, Any] | None:
    """Return a user row by UUID, or None."""
    async with get_engine().connect() as conn:
        row = await conn.execute(
            text(
                "SELECT id, email, display_name, created_at, last_login_at, "
                "is_active, email_verified "
                "FROM users WHERE id = :uid"
            ),
            {"uid": user_id},
        )
        result = row.mappings().one_or_none()
        return _row_to_user(dict(result)) if result else None


async def touch_user_login(user_id: str) -> None:
    """Update last_login_at to NOW()."""
    async with get_engine().begin() as conn:
        await conn.execute(
            text("UPDATE users SET last_login_at = NOW(), updated_at = NOW() WHERE id = :uid"),
            {"uid": user_id},
        )


# ─────────────────────────────────────────────────────────────────────────────
# Refresh Token CRUD
# ─────────────────────────────────────────────────────────────────────────────


async def create_refresh_token_row(
    user_id: str,
    token_hash: str,
    expires_at: datetime,
    ip_addr: str = "",
    user_agent: str = "",
) -> str:
    """Insert a refresh token row.  Returns the new row id (UUID str)."""
    async with get_engine().begin() as conn:
        row = await conn.execute(
            text(
                "INSERT INTO refresh_tokens (user_id, token_hash, expires_at, ip_addr, user_agent) "
                "VALUES (:uid, :th, :exp, :ip, :ua) RETURNING id"
            ),
            {
                "uid": user_id,
                "th": token_hash,
                "exp": expires_at,
                "ip": ip_addr,
                "ua": user_agent,
            },
        )
        return str(row.scalar_one())


async def get_refresh_token_row(token_hash: str) -> dict[str, Any] | None:
    """Return a refresh token row by hash, or None."""
    async with get_engine().connect() as conn:
        row = await conn.execute(
            text(
                "SELECT id, user_id, expires_at, revoked_at "
                "FROM refresh_tokens WHERE token_hash = :th"
            ),
            {"th": token_hash},
        )
        result = row.mappings().one_or_none()
        return dict(result) if result else None


async def revoke_refresh_token(token_hash: str) -> None:
    """Mark a single refresh token as revoked."""
    async with get_engine().begin() as conn:
        await conn.execute(
            text("UPDATE refresh_tokens SET revoked_at = NOW() WHERE token_hash = :th"),
            {"th": token_hash},
        )


async def revoke_all_user_tokens(user_id: str) -> None:
    """Revoke all refresh tokens for a user (logout everywhere)."""
    async with get_engine().begin() as conn:
        await conn.execute(
            text(
                "UPDATE refresh_tokens SET revoked_at = NOW() "
                "WHERE user_id = :uid AND revoked_at IS NULL"
            ),
            {"uid": user_id},
        )


# ─────────────────────────────────────────────────────────────────────────────
# API Key CRUD
# ─────────────────────────────────────────────────────────────────────────────


async def create_api_key_row(
    user_id: str,
    name: str,
    key_prefix: str,
    key_hash: str,
    scopes: list[str] | None = None,
    org_id: str | None = None,
    expires_at: datetime | None = None,
) -> dict[str, Any]:
    """Insert a new API key row.  Returns the row (without key_hash)."""
    async with get_engine().begin() as conn:
        row = await conn.execute(
            text(
                "INSERT INTO api_keys "
                "(user_id, org_id, name, key_prefix, key_hash, scopes, expires_at) "
                "VALUES (:uid, :oid, :name, :pfx, :kh, :scopes, :exp) "
                "RETURNING id, name, key_prefix, scopes, created_at, expires_at, is_active"
            ),
            {
                "uid": user_id,
                "oid": org_id,
                "name": name,
                "pfx": key_prefix,
                "kh": key_hash,
                "scopes": scopes or [],
                "exp": expires_at,
            },
        )
        result = row.mappings().one()
        return dict(result)


async def list_api_keys(user_id: str) -> list[dict[str, Any]]:
    """Return all active API keys for a user (no key_hash exposed)."""
    async with get_engine().connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT id, name, key_prefix, scopes, created_at, last_used_at, "
                "expires_at, is_active "
                "FROM api_keys WHERE user_id = :uid ORDER BY created_at DESC"
            ),
            {"uid": user_id},
        )
        return [dict(r) for r in rows.mappings().all()]


async def lookup_api_key(key_hash: str) -> dict[str, Any] | None:
    """
    Return (user_id, org_id, scopes, is_active) for a given key hash.
    Used by the auth middleware to resolve X-API-Key requests.
    """
    async with get_engine().connect() as conn:
        row = await conn.execute(
            text(
                "SELECT ak.user_id, ak.org_id, ak.scopes, ak.is_active, "
                "ak.expires_at, u.is_active AS user_active "
                "FROM api_keys ak "
                "JOIN users u ON u.id = ak.user_id "
                "WHERE ak.key_hash = :kh"
            ),
            {"kh": key_hash},
        )
        result = row.mappings().one_or_none()
        return dict(result) if result else None


async def revoke_api_key(key_id: str, user_id: str) -> bool:
    """Deactivate an API key.  Returns True if a row was updated."""
    async with get_engine().begin() as conn:
        result = await conn.execute(
            text("UPDATE api_keys SET is_active = FALSE WHERE id = :kid AND user_id = :uid"),
            {"kid": key_id, "uid": user_id},
        )
        return (result.rowcount or 0) > 0


async def touch_api_key(key_hash: str) -> None:
    """Update last_used_at for an API key (fire-and-forget)."""
    with contextlib.suppress(Exception):
        async with get_engine().begin() as conn:
            await conn.execute(
                text("UPDATE api_keys SET last_used_at = NOW() WHERE key_hash = :kh"),
                {"kh": key_hash},
            )


# ─────────────────────────────────────────────────────────────────────────────
# Organization CRUD
# ─────────────────────────────────────────────────────────────────────────────


async def create_org(name: str, slug: str, owner_id: str, plan: str = "starter") -> dict[str, Any]:
    """Create an org and add the owner as a member with role 'owner'."""
    async with get_engine().begin() as conn:
        row = await conn.execute(
            text(
                "INSERT INTO organizations (name, slug, owner_id, plan) "
                "VALUES (:name, :slug, :oid, :plan) "
                "RETURNING id, name, slug, plan, created_at"
            ),
            {"name": name, "slug": slug.lower(), "oid": owner_id, "plan": plan},
        )
        org = dict(row.mappings().one())
        await conn.execute(
            text(
                "INSERT INTO org_members (user_id, org_id, role) "
                "VALUES (:uid, :oid, 'owner') ON CONFLICT DO NOTHING"
            ),
            {"uid": owner_id, "oid": str(org["id"])},
        )
    return org


async def get_org(org_id: str) -> dict[str, Any] | None:
    """Return an org row by UUID, or None."""
    async with get_engine().connect() as conn:
        row = await conn.execute(
            text("SELECT id, name, slug, plan, created_at FROM organizations WHERE id = :oid"),
            {"oid": org_id},
        )
        result = row.mappings().one_or_none()
        return dict(result) if result else None


async def get_user_orgs(user_id: str) -> list[dict[str, Any]]:
    """Return all orgs the user is a member of."""
    async with get_engine().connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT o.id, o.name, o.slug, o.plan, o.created_at, om.role "
                "FROM organizations o "
                "JOIN org_members om ON om.org_id = o.id "
                "WHERE om.user_id = :uid ORDER BY o.created_at"
            ),
            {"uid": user_id},
        )
        return [dict(r) for r in rows.mappings().all()]


async def add_org_member(user_id: str, org_id: str, role: str = "member") -> None:
    """Add a user to an org.  Silently ignores duplicate."""
    async with get_engine().begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO org_members (user_id, org_id, role) "
                "VALUES (:uid, :oid, :role) ON CONFLICT (user_id, org_id) "
                "DO UPDATE SET role = EXCLUDED.role"
            ),
            {"uid": user_id, "oid": org_id, "role": role},
        )


async def list_org_members(org_id: str) -> list[dict[str, Any]]:
    """Return all members of an org."""
    async with get_engine().connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT u.id, u.email, u.display_name, om.role, om.joined_at "
                "FROM org_members om JOIN users u ON u.id = om.user_id "
                "WHERE om.org_id = :oid ORDER BY om.joined_at"
            ),
            {"oid": org_id},
        )
        return [dict(r) for r in rows.mappings().all()]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _row_to_user(row: dict[str, Any]) -> dict[str, Any]:
    """Normalise a users row — stringify UUID and ISO-format timestamps."""
    row["id"] = str(row["id"]) if row.get("id") else None
    for col in ("created_at", "last_login_at", "updated_at"):
        val = row.get(col)
        if hasattr(val, "isoformat"):
            row[col] = val.isoformat().replace("+00:00", "Z")
    return row
