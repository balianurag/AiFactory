"""
Async PostgreSQL persistence layer for AiFactory tasks.

Uses SQLAlchemy 2.0 async engine + asyncpg driver.
All task CRUD flows through this module — no more in-memory dict.

Tables:
  tasks          — task metadata, status, result, metrics
  task_agents    — per-task agent hierarchy snapshot
  task_messages  — agent-to-agent messages for replay
"""

from __future__ import annotations

import contextlib
import json
import logging
from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from aifactory.config.settings import get_settings

logger = logging.getLogger(__name__)

# ── Module-level engine (lazy singleton) ──────────────────────────────────────

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
        # Convert postgres:// → postgresql+asyncpg://
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif not url.startswith("postgresql+asyncpg://"):
            url = f"postgresql+asyncpg://{url}"

        _engine = create_async_engine(
            url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            echo=False,
        )
        logger.info("PostgreSQL engine created: %s", url.split("@")[-1] if "@" in url else url)
    return _engine


async def close_engine() -> None:
    """Dispose the engine (call on app shutdown)."""
    global _engine
    if _engine:
        await _engine.dispose()
        _engine = None
        logger.info("PostgreSQL engine disposed")


def reset_engine() -> None:
    """Reset the engine reference (for testing)."""
    global _engine
    _engine = None


# ── Health Check ──────────────────────────────────────────────────────────────


async def pg_health() -> bool:
    """Return True if PostgreSQL is reachable."""
    try:
        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.warning("PostgreSQL health check failed: %s", e)
        return False


# ── Task CRUD ─────────────────────────────────────────────────────────────────


async def create_task(
    task_id: str,
    status: str,
    created_at: str,
    request: dict[str, Any],
) -> None:
    """Insert a new task row."""
    engine = get_engine()
    # Convert ISO string to datetime for asyncpg's TIMESTAMPTZ column
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
    """
    Update task fields dynamically.

    Supports: status, progress, result, error, metrics.
    JSONB fields are auto-serialised.
    """
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


# ── Agent Hierarchy CRUD ──────────────────────────────────────────────────────


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


# ── Messages CRUD ─────────────────────────────────────────────────────────────


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


# ── Helpers ───────────────────────────────────────────────────────────────────


def _row_to_task(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a raw DB row to the task dict format used by the API."""
    # JSONB columns come back as dicts already from asyncpg, but may be strings
    for col in ("request", "result", "metrics"):
        val = row.get(col)
        if isinstance(val, str):
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                row[col] = json.loads(val)
    # Convert datetime objects to ISO strings
    for col in ("created_at", "updated_at"):
        val = row.get(col)
        if hasattr(val, "isoformat"):
            row[col] = val.isoformat().replace("+00:00", "Z")
    return row
