"""
AiFactory FastAPI HTTP/WebSocket API
Wraps the LangGraph single-team pipeline for web consumption.

Endpoints:
  GET  /api/health          — liveness + version + DB/Redis/Qdrant status
  GET  /api/models          — allowed model registry per agent role
  POST /api/tasks           — submit a new task, returns task_id
  GET  /api/tasks           — list all tasks (for DashboardView / ProjectsView)
  GET  /api/tasks/{task_id} — poll task status + partial results
  GET  /api/tasks/{task_id}/agents — live agent hierarchy for a task
  GET  /api/memory/stats    — memory layer statistics
  WS   /api/ws/{task_id}    — stream real-time agent events

Persistence: PostgreSQL (tasks, agents, messages) + Redis (hot cache).
Render deployment: uvicorn aifactory.api.main:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from slowapi.errors import RateLimitExceeded

from aifactory.core import database as db
from aifactory.core import redis_client as rc
from aifactory.core.observability import setup_langsmith
from aifactory.core.rate_limiter import (
    RATE_LIMIT_MEMORY,
    RATE_LIMIT_READ,
    RATE_LIMIT_TASKS,
    get_rate_limit_info,
    limiter,
    rate_limit_exceeded_handler,
)

logger = logging.getLogger(__name__)


# ── Lifecycle (connect/disconnect DB + Redis) ─────────────────────────────────


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Startup: verify DB connectivity. Shutdown: close connections."""
    # ── Startup ──
    ok = await db.pg_health()
    if ok:
        logger.info("✅ PostgreSQL connected")
    else:
        logger.warning("⚠️  PostgreSQL unreachable — tasks will NOT be persisted")

    rok = await rc.redis_health()
    if rok:
        logger.info("✅ Redis connected")
    else:
        logger.info("Redis unavailable -- running without cache")

    yield

    # ── Shutdown ──
    await db.close_engine()
    await rc.close_redis()


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AiFactory API",
    description="Self-assembling multi-agent orchestration system",
    version="0.3.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# ── Rate Limiting ─────────────────────────────────────────────────────────────

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)  # type: ignore[arg-type]

# ── LangSmith Observability ───────────────────────────────────────────────────

setup_langsmith()

# ── CORS (allow Vercel frontend + local dev) ──────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://frontend:3000",  # Docker internal
        "https://aifactory-ui.vercel.app",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory WebSocket subscribers (not persisted — ephemeral by nature) ─────

_ws_subscribers: dict[str, list[WebSocket]] = {}
_background_tasks: set[asyncio.Task[None]] = set()

# ── Model registry loader ──────────────────────────────────────────────────────

_MODELS_YAML = Path(__file__).parent.parent / "config" / "models.yaml"


@functools.lru_cache(maxsize=1)
def _load_models_registry() -> dict[str, Any]:
    """Load and cache models.yaml. Cached so the file is only read once."""
    with open(_MODELS_YAML) as f:
        return yaml.safe_load(f)


def _build_models_response() -> dict[str, Any]:
    """Build the /api/models response from models.yaml."""
    registry = _load_models_registry()
    allowed = registry.get("allowed_models", {})
    metadata = registry.get("metadata", {})

    result: dict[str, Any] = {}
    for role, role_data in allowed.items():
        models_list = role_data.get("models", []) if isinstance(role_data, dict) else role_data
        result[role] = {
            "description": role_data.get("description", "") if isinstance(role_data, dict) else "",
            "models": [
                {
                    "id": model_id,
                    **metadata.get(
                        model_id,
                        {"display": model_id.split("/")[-1], "provider": "Unknown"},
                    ),
                }
                for model_id in models_list
            ],
        }
    return result


def _get_allowed_model_ids() -> set[str]:
    """Flat set of all allowed model IDs across all roles."""
    registry = _load_models_registry()
    ids: set[str] = set()
    for role_data in registry.get("allowed_models", {}).values():
        models = role_data.get("models", []) if isinstance(role_data, dict) else role_data
        ids.update(models)
    return ids


# ── Schemas ───────────────────────────────────────────────────────────────────


class TaskRequest(BaseModel):
    task: str
    context: str = ""
    mode: str | None = None  # "best" | "balanced" | "cost_efficient" (default: from graph.yaml)

    # Legacy fields — kept for backward compatibility with existing clients
    num_workers: int = 3  # ignored by dynamic pipeline (analyser decides)
    pm_model: str | None = None  # ignored (ability-scoring picks models)
    worker_tier: str = "budget"  # ignored (ability-scoring picks models)

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str | None) -> str | None:
        if v is None:
            return v
        allowed = {"best", "balanced", "cost_efficient"}
        if v not in allowed:
            raise ValueError(f"mode must be one of {allowed}")
        return v


class TaskResponse(BaseModel):
    task_id: str
    status: str
    created_at: str


class TaskStatus(BaseModel):
    task_id: str
    status: str  # pending | running | completed | failed
    created_at: str
    updated_at: str
    progress: float = 0.0  # 0.0 - 1.0
    agents: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    result: dict[str, Any] | None = None
    error: str | None = None
    metrics: dict[str, Any] = {}


# ── Routes ────────────────────────────────────────────────────────────────────


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


@app.get("/api/health")
@limiter.limit(RATE_LIMIT_READ)
async def health(request: Request) -> dict[str, Any]:
    """Liveness probe for Render — includes DB/Redis/Qdrant connectivity."""
    pg_ok = await db.pg_health()
    redis_ok = await rc.redis_health()
    return {
        "status": "ok",
        "version": "0.3.0",
        "phase": "3 — Persistent Storage + Real Embeddings",
        "timestamp": _now(),
        "services": {
            "postgres": "connected" if pg_ok else "unavailable",
            "redis": "connected" if redis_ok else "unavailable",
        },
        "rate_limiting": get_rate_limit_info(),
    }


@app.get("/api/models")
@limiter.limit(RATE_LIMIT_READ)
async def get_models(request: Request) -> dict[str, Any]:
    """
    Return the curated allowed model registry, structured by agent role.
    Used by the UI model picker and CLI `aifactory models list`.
    """
    return {"roles": _build_models_response()}


@app.get("/api/tasks")
@limiter.limit(RATE_LIMIT_READ)
async def list_tasks(
    request: Request,
    status: str | None = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200),
) -> dict[str, Any]:
    """List all tasks. Used by DashboardView and ProjectsView."""
    # Try Redis cache first
    cached = await rc.get_cached_task_list(status)
    if cached is not None:
        return {"tasks": cached[:limit], "total": len(cached), "source": "cache"}

    # Fallback to Postgres
    rows = await db.list_tasks(status_filter=status, limit=limit)
    items = []
    for t in rows:
        agents = await db.get_agents(t["task_id"])
        items.append(
            {
                "task_id": t["task_id"],
                "status": t["status"],
                "created_at": t.get("created_at", ""),
                "updated_at": t.get("updated_at", ""),
                "progress": t.get("progress", 0),
                "task_description": t.get("request", {}).get("task", "")[:120],
                "agents": agents,
                "metrics": t.get("metrics", {}),
                "error": t.get("error"),
                "has_result": t.get("result") is not None,
            }
        )

    # Cache for next time
    await rc.cache_task_list(items, status)
    return {"tasks": items[:limit], "total": len(items)}


@app.post("/api/tasks", response_model=TaskResponse, status_code=202)
@limiter.limit(RATE_LIMIT_TASKS)
async def submit_task(request: Request, req: TaskRequest) -> TaskResponse:
    """
    Submit a task for the multi-agent pipeline to execute.
    Returns a task_id immediately; use GET /api/tasks/{task_id} to poll.
    Rate limited to 10 requests/minute per IP (configurable).
    """
    task_id = str(uuid.uuid4())
    now = _now()

    # Persist to PostgreSQL
    await db.create_task(
        task_id=task_id,
        status="pending",
        created_at=now,
        request=req.model_dump(),
    )
    # Initialise WebSocket subscriber list
    _ws_subscribers[task_id] = []

    # Invalidate task list cache
    await rc.invalidate_task_list()

    # Fire-and-forget: run pipeline in background
    bg = asyncio.create_task(_run_pipeline(task_id, req))
    _background_tasks.add(bg)
    bg.add_done_callback(_background_tasks.discard)

    return TaskResponse(task_id=task_id, status="pending", created_at=now)


@app.get("/api/tasks/{task_id}", response_model=TaskStatus)
@limiter.limit(RATE_LIMIT_READ)
async def get_task(request: Request, task_id: str) -> TaskStatus:
    """Poll task status and partial results. Checks Redis cache first."""
    # Try Redis cache
    cached = await rc.get_cached_task(task_id)
    if cached:
        return TaskStatus(**{k: v for k, v in cached.items() if k not in ("request",)})

    # Fallback to Postgres
    t = await db.get_task(task_id)
    if t is None:
        raise HTTPException(status_code=404, detail="Task not found")

    agents = await db.get_agents(task_id)
    messages = await db.get_messages(task_id)

    task_data = {
        "task_id": t["task_id"],
        "status": t["status"],
        "created_at": t.get("created_at", ""),
        "updated_at": t.get("updated_at", ""),
        "progress": t.get("progress", 0),
        "agents": agents,
        "messages": messages,
        "result": t.get("result"),
        "error": t.get("error"),
        "metrics": t.get("metrics", {}),
    }
    # Cache for future polls
    await rc.cache_task(task_id, task_data)
    return TaskStatus(**task_data)


@app.get("/api/tasks/{task_id}/agents")
@limiter.limit(RATE_LIMIT_READ)
async def get_task_agents(request: Request, task_id: str) -> dict[str, Any]:
    """Return the live agent hierarchy for a running/completed task."""
    t = await db.get_task(task_id)
    if t is None:
        raise HTTPException(status_code=404, detail="Task not found")
    agents = await db.get_agents(task_id)
    messages = await db.get_messages(task_id)
    return {
        "task_id": task_id,
        "agents": agents,
        "messages": messages,
    }


# ── Shared Memory Router (singleton, in-memory fallback when Qdrant is down) ───

_memory_router_instance: Any = None


def _get_memory_router() -> Any:
    """Return a shared MemoryRouter so data persists across API calls."""
    global _memory_router_instance
    if _memory_router_instance is None:
        from aifactory.core.memory_router import MemoryRouter

        _memory_router_instance = MemoryRouter()
    return _memory_router_instance


@app.get("/api/memory/stats")
@limiter.limit(RATE_LIMIT_MEMORY)
async def memory_stats(request: Request) -> dict[str, Any]:
    """Return memory layer statistics. Used by MemoryView and DashboardView."""
    try:
        router = _get_memory_router()
        stats = router.get_stats()
        return {
            "layers": {level: {"entries": count} for level, count in stats.items()},
            "total_entries": sum(stats.values()),
            "status": "connected",
        }
    except Exception as e:
        return {
            "layers": {"agent": {"entries": 0}, "team": {"entries": 0}, "global": {"entries": 0}},
            "total_entries": 0,
            "status": f"error: {e}",
        }


@app.post("/api/memory/store")
@limiter.limit(RATE_LIMIT_MEMORY)
async def memory_store(request: Request, body: dict[str, Any]) -> dict[str, Any]:
    """
    Store a memory entry.

    Body:
        level: "agent" | "team" | "global"
        content: str
        agent_id?: str
        team_id?: str
        entry_type?: str (default "observation")
        turn_number?: int (default 1)
    """
    try:
        from aifactory.core.schemas import MemoryEntry, MemoryLevel

        router = _get_memory_router()
        entry = MemoryEntry(
            level=MemoryLevel(body["level"]),
            content=body["content"],
            agent_id=body.get("agent_id"),
            team_id=body.get("team_id"),
            entry_type=body.get("entry_type", "observation"),
            turn_number=body.get("turn_number", 1),
            metadata=body.get("metadata", {}),
        )
        point_id = router.store(entry)
        return {"status": "stored", "point_id": point_id, "level": body["level"]}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/api/memory/retrieve")
@limiter.limit(RATE_LIMIT_MEMORY)
async def memory_retrieve(request: Request, body: dict[str, Any]) -> dict[str, Any]:
    """
    Retrieve memories by semantic similarity.

    Body:
        query: str
        levels?: ["agent", "team", "global"] (default: all)
        agent_id?: str
        team_id?: str
        top_k?: int (default 5)
        entry_type?: str
    """
    try:
        from aifactory.core.schemas import MemoryLevel

        router = _get_memory_router()
        levels = [MemoryLevel(lv) for lv in body.get("levels", ["agent", "team", "global"])]
        results = router.retrieve(
            query=body["query"],
            levels=levels,
            agent_id=body.get("agent_id"),
            team_id=body.get("team_id"),
            top_k=body.get("top_k", 5),
            entry_type=body.get("entry_type"),
        )
        return {
            "status": "ok",
            "count": len(results),
            "results": [r.model_dump(mode="json") for r in results],
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/api/memory/promote")
@limiter.limit(RATE_LIMIT_MEMORY)
async def memory_promote(request: Request, body: dict[str, Any]) -> dict[str, Any]:
    """
    Promote memories from one layer to the next.

    Body:
        agent_id: str
        team_id: str
        summary: str
        from_level?: "agent" | "team" (default "agent")
        to_level?: "team" | "global" (default "team")
    """
    try:
        from aifactory.core.schemas import MemoryLevel

        router = _get_memory_router()
        entry_id = router.promote(
            agent_id=body["agent_id"],
            team_id=body["team_id"],
            summary=body["summary"],
            from_level=MemoryLevel(body.get("from_level", "agent")),
            to_level=MemoryLevel(body.get("to_level", "team")),
        )
        return {
            "status": "promoted",
            "entry_id": entry_id,
            "from": body.get("from_level", "agent"),
            "to": body.get("to_level", "team"),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.delete("/api/memory/agent/{agent_id}")
@limiter.limit(RATE_LIMIT_MEMORY)
async def memory_clear_agent(request: Request, agent_id: str) -> dict[str, Any]:
    """Clear all memories for a specific agent."""
    try:
        router = _get_memory_router()
        router.clear_agent(agent_id)
        return {"status": "cleared", "agent_id": agent_id}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── Observability Endpoints ────────────────────────────────────────────────────


@app.get("/api/observability/status")
@limiter.limit(RATE_LIMIT_READ)
async def observability_status(request: Request) -> dict[str, Any]:
    """Check if LangSmith tracing is enabled and return config."""
    import os

    from aifactory.config.settings import get_settings
    from aifactory.core.observability import is_tracing_enabled

    settings = get_settings()
    # Detect if running in a deployed environment (Render, Railway, etc.)
    is_deployed = bool(
        os.environ.get("RENDER")
        or os.environ.get("RAILWAY_ENVIRONMENT")
        or os.environ.get("CI")
        or os.environ.get("VERCEL")
        or os.environ.get("FLY_APP_NAME")
    )
    return {
        "enabled": is_tracing_enabled(),
        "project": settings.langchain_project if is_tracing_enabled() else None,
        "endpoint": settings.langchain_endpoint if is_tracing_enabled() else None,
        "environment": "deployed" if is_deployed else "local",
    }


@app.get("/api/observability/runs")
@limiter.limit(RATE_LIMIT_READ)
async def list_runs(
    request: Request,
    limit: int = Query(20, ge=1, le=100),
) -> dict[str, Any]:
    """List recent LangSmith runs. Cached for 60s to avoid slow LangSmith calls."""
    import time

    from aifactory.core.observability import get_recent_runs, is_tracing_enabled

    if not is_tracing_enabled():
        return {"runs": [], "enabled": False, "message": "LangSmith not configured"}

    # Simple TTL cache — avoid hammering the slow LangSmith API
    cache = getattr(list_runs, "_cache", None)
    now = time.time()
    if cache and cache["limit"] == limit and now - cache["ts"] < 60:
        return cache["data"]

    runs = get_recent_runs(limit=limit)
    data = {"runs": runs, "enabled": True, "total": len(runs)}
    list_runs._cache = {"limit": limit, "ts": now, "data": data}  # type: ignore[attr-defined]
    return data


@app.get("/api/observability/runs/{run_id}")
@limiter.limit(RATE_LIMIT_READ)
async def get_run(request: Request, run_id: str) -> dict[str, Any]:
    """Get detail for a single LangSmith run."""
    from aifactory.core.observability import get_run_detail, is_tracing_enabled

    if not is_tracing_enabled():
        raise HTTPException(status_code=503, detail="LangSmith not configured")
    detail = get_run_detail(run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return detail


@app.get("/api/tasks/{task_id}/traces")
@limiter.limit(RATE_LIMIT_READ)
async def get_task_traces(
    request: Request,
    task_id: str,
    limit: int = Query(20, ge=1, le=100),
) -> dict[str, Any]:
    """
    Return LangSmith traces linked to a specific task.

    Searches for runs whose metadata contains the given task_id.
    Returns empty list gracefully when LangSmith is not configured.
    """
    from aifactory.core.observability import get_task_traces as _get_task_traces
    from aifactory.core.observability import is_tracing_enabled

    # Verify task exists
    t = await db.get_task(task_id)
    if t is None:
        raise HTTPException(status_code=404, detail="Task not found")

    if not is_tracing_enabled():
        return {"task_id": task_id, "traces": [], "enabled": False}

    traces = _get_task_traces(task_id, limit=limit)
    return {"task_id": task_id, "traces": traces, "enabled": True, "total": len(traces)}


@app.websocket("/api/ws/{task_id}")
async def websocket_task(websocket: WebSocket, task_id: str) -> None:
    """
    Real-time WebSocket stream for a task.
    Client receives JSON events as agents execute.

    Fixed: TimeoutError no longer kills the connection — it continues
    the loop so long-running pipelines can stream events for minutes.
    Also replays current task state on connect to avoid race conditions
    (pipeline may start before the WS is established).
    """
    await websocket.accept()

    t = await db.get_task(task_id)
    if t is None:
        await websocket.send_json({"type": "error", "message": "Task not found"})
        await websocket.close()
        return

    # ── Replay current state so late-connecting clients catch up ──────
    # This fixes the race condition where _run_pipeline starts before
    # the WebSocket connection is established.
    if t.get("status") == "completed" and t.get("result"):
        # Task already finished — send the result immediately
        agents = await db.get_agents(task_id)
        await websocket.send_json(
            {
                "type": "completed",
                "progress": 1.0,
                "result": t["result"],
                "agents": agents,
            }
        )
        await websocket.close()
        return

    if t.get("status") == "failed":
        await websocket.send_json(
            {
                "type": "error",
                "message": t.get("error", "Task failed"),
            }
        )
        await websocket.close()
        return

    # Replay in-progress state (agents assembled so far, current progress)
    if t.get("status") == "running":
        agents = await db.get_agents(task_id)
        if agents:
            await websocket.send_json(
                {
                    "type": "agent_hierarchy",
                    "message": f"Reconnected — {len(agents)} agents active",
                    "progress": t.get("progress", 0),
                    "agents": agents,
                }
            )

    if task_id not in _ws_subscribers:
        _ws_subscribers[task_id] = []
    _ws_subscribers[task_id].append(websocket)

    try:
        # Keep connection alive; events are pushed by _run_pipeline.
        # TimeoutError is caught INSIDE the loop so the connection
        # survives idle periods (pipelines can run for many minutes).
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30)
            except TimeoutError:
                # Client didn't send anything for 30s — that's fine,
                # just continue waiting. The pipeline will push events.
                continue
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in _ws_subscribers.get(task_id, []):
            _ws_subscribers[task_id].remove(websocket)


# ── Pipeline runner ───────────────────────────────────────────────────────────


async def _broadcast(task_id: str, event: dict[str, Any]) -> None:
    """Push an event to all WebSocket subscribers for a task."""
    dead: list[WebSocket] = []
    for ws in _ws_subscribers.get(task_id, []):
        try:
            await ws.send_json(event)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in _ws_subscribers.get(task_id, []):
            _ws_subscribers[task_id].remove(ws)


async def _run_pipeline(task_id: str, req: TaskRequest) -> None:
    """
    Execute the AiFactory dynamic 8-node pipeline.

    Architecture (ported from agent-arch):
      master_receive → judge_brief → analyse
        → [researcher → reanalyse]?   (pre-analysis research loop)
        → [researcher]?               (execution research)
        → execution_lead              (spawns N workers dynamically)
        → judge_review
        → master_respond → END

    Each agent's model is selected via ability-scoring (model_registry.yaml).
    Persists all state to PostgreSQL, caches hot data in Redis,
    and broadcasts live events via WebSocket.
    """
    import os

    # Tag all LangSmith traces with this task_id so /api/tasks/{id}/traces works
    _prev_tags = os.environ.get("LANGCHAIN_TAGS", "")
    _prev_metadata = os.environ.get("LANGCHAIN_METADATA", "")
    os.environ["LANGCHAIN_TAGS"] = task_id
    os.environ["LANGCHAIN_METADATA"] = f'{{"task_id": "{task_id}"}}'

    async def update(**patch: Any) -> None:
        """Persist task updates to Postgres + Redis + invalidate list cache."""
        await db.update_task(task_id, **patch)
        await rc.invalidate_task(task_id)
        await rc.invalidate_task_list()

    try:
        await update(status="running", progress=0.05)
        await _broadcast(task_id, {"type": "status", "status": "running", "progress": 0.05})

        # Import here to avoid slow startup at module load
        from aifactory.core.model_selector import cost_usd_from_registry
        from aifactory.graphs.dynamic_pipeline import build_dynamic_pipeline, write_dump

        # ── Build the dynamic graph ──────────────────────────────────
        await _broadcast(
            task_id,
            {"type": "status", "message": "Building agent hierarchy...", "progress": 0.08},
        )

        compiled_graph, registry = build_dynamic_pipeline(
            mode=req.mode or "balanced", run_id=task_id
        )

        # Broadcast initial agent hierarchy
        agents_info = [
            {
                "agent_id": f"{task_id}_{a.name}",
                "name": a.name,
                "role": a.cfg.get("role", a.name),
                "model": a.model,
                "specialty": None,
                "status": "active",
                "tokens_used": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "parent_id": None,
            }
            for a in registry.all()
        ]
        await db.upsert_agents(task_id, agents_info)
        await update(progress=0.10)
        await _broadcast(
            task_id,
            {
                "type": "agent_hierarchy",
                "message": f"Assembled {registry.count()} agents: "
                + ", ".join(a.name for a in registry.all()),
                "progress": 0.10,
                "agents": agents_info,
            },
        )

        # ── Execute the graph ────────────────────────────────────────
        await _broadcast(
            task_id,
            {"type": "status", "message": "Pipeline executing...", "progress": 0.15},
        )
        await update(progress=0.15)

        # Stream graph execution events
        initial_state = {
            "user_query": req.task,
            "shared_context": {"task_id": task_id, "context": req.context or ""},
            "conversation_history": [],
        }

        config = {"configurable": {"thread_id": task_id}}

        # Use astream_events to broadcast node progress to WS clients
        node_progress = {
            "master_receive": 0.20,
            "judge_brief": 0.25,
            "analyse": 0.35,
            "researcher": 0.45,
            "reanalyse": 0.50,
            "execution_lead": 0.70,
            "judge_review": 0.85,
            "master_respond": 0.95,
        }

        final_state = None
        async for event in compiled_graph.astream_events(
            initial_state, config=config, version="v2"
        ):
            kind = event.get("event", "")
            name = event.get("name", "")

            if kind == "on_chain_start" and name in node_progress:
                progress = node_progress[name]
                await update(progress=progress)
                await _broadcast(
                    task_id,
                    {
                        "type": "agent_active",
                        "agent": name,
                        "message": f"{name} executing...",
                        "progress": progress,
                    },
                )

            if kind == "on_chain_end" and name in node_progress:
                # Refresh agent info with updated token counts
                agents_info = _build_agents_info(task_id, registry)
                await db.upsert_agents(task_id, agents_info)

            # Capture final graph state
            if kind == "on_chain_end" and name == "LangGraph":
                final_state = event.get("data", {}).get("output", {})

        # If astream_events didn't capture final state, invoke directly
        if final_state is None:
            final_state = await compiled_graph.ainvoke(initial_state, config=config)

        # ── Extract results ──────────────────────────────────────────
        final_response = final_state.get("final_response", "")
        task_results = final_state.get("task_results", [])
        judge_metrics = final_state.get("judge_metrics", {})
        agent_assessments = final_state.get("agent_assessments", [])
        direct_answer = final_state.get("direct_answer", "")

        # Write JSON run dump
        try:
            from aifactory.core.model_selector import load_graph_config

            _cfg = load_graph_config()
            write_dump(task_id, _cfg, final_state, registry, 0.0, req.task)
        except Exception as dump_exc:
            logger.warning("Failed to write run dump: %s", dump_exc)

        # Calculate total tokens and cost per agent
        total_tokens = 0
        total_in = 0
        total_out = 0
        total_cost = 0.0

        agents_info = []
        for agent in registry.all():
            a_cost = cost_usd_from_registry(
                agent.model, agent.prompt_tokens, agent.completion_tokens
            )
            info = {
                "agent_id": f"{task_id}_{agent.name}",
                "name": agent.name,
                "role": agent.cfg.get("role", agent.name),
                "model": agent.model,
                "specialty": None,
                "status": "completed",
                "tokens_used": agent.total_tokens,
                "input_tokens": agent.prompt_tokens,
                "output_tokens": agent.completion_tokens,
                "cost_usd": round(a_cost, 6),
                "parent_id": None,
            }
            agents_info.append(info)
            total_tokens += agent.total_tokens
            total_in += agent.prompt_tokens
            total_out += agent.completion_tokens
            total_cost += a_cost

        result = {
            "title": req.task[:100],
            "summary": final_response,
            "artifacts": [
                {
                    "name": r.get("task_id", "result"),
                    "content": r.get("result", ""),
                    "type": "text",
                }
                for r in task_results
                if not r.get("skipped")
            ],
            "quality_assessment": judge_metrics.get("overall_status", "completed"),
            "open_issues": [],
            "recommendations": [judge_metrics.get("recommendations", "")]
            if judge_metrics.get("recommendations")
            else [],
            "metrics": {
                "complexity": "dynamic",
                "route": "direct_answer" if direct_answer else "dynamic_pipeline",
                "total_tokens": total_tokens,
                "total_input_tokens": total_in,
                "total_output_tokens": total_out,
                "total_cost_usd": round(total_cost, 6),
                "total_agents": registry.count(),
                "tasks_completed": len([r for r in task_results if not r.get("skipped")]),
                "judge_assessments": agent_assessments,
                "pipeline_efficiency": judge_metrics.get("pipeline_efficiency", ""),
            },
        }

        # ── Persist + broadcast completion ────────────────────────────
        await db.upsert_agents(task_id, agents_info)
        await update(
            status="completed",
            progress=1.0,
            result=result,
            metrics=result["metrics"],
        )
        await _broadcast(
            task_id,
            {
                "type": "completed",
                "progress": 1.0,
                "result": result,
                "agents": agents_info,
            },
        )

    except Exception as exc:
        error_msg = str(exc)
        await update(status="failed", error=error_msg)
        await _broadcast(
            task_id,
            {
                "type": "error",
                "message": error_msg,
            },
        )
    finally:
        # Restore LangSmith env vars
        if _prev_tags:
            os.environ["LANGCHAIN_TAGS"] = _prev_tags
        else:
            os.environ.pop("LANGCHAIN_TAGS", None)
        if _prev_metadata:
            os.environ["LANGCHAIN_METADATA"] = _prev_metadata
        else:
            os.environ.pop("LANGCHAIN_METADATA", None)


def _build_agents_info(task_id: str, registry: Any) -> list[dict[str, Any]]:
    """Build agent info dicts from the registry for DB persistence."""
    from aifactory.core.model_selector import cost_usd_from_registry

    return [
        {
            "agent_id": f"{task_id}_{a.name}",
            "name": a.name,
            "role": a.cfg.get("role", a.name),
            "model": a.model,
            "specialty": None,
            "status": "active",
            "tokens_used": a.total_tokens,
            "input_tokens": a.prompt_tokens,
            "output_tokens": a.completion_tokens,
            "cost_usd": round(
                cost_usd_from_registry(a.model, a.prompt_tokens, a.completion_tokens), 6
            ),
            "parent_id": None,
        }
        for a in registry.all()
    ]
