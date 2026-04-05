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
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from aifactory.core import database as db
from aifactory.core import redis_client as rc
from aifactory.core.observability import setup_langsmith

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
    num_workers: int = 3
    pm_model: str | None = None  # override PM model (must be in allowed list)
    worker_tier: str = "budget"  # "premium" | "standard" | "budget"

    @field_validator("worker_tier")
    @classmethod
    def validate_worker_tier(cls, v: str) -> str:
        allowed = {"premium", "standard", "budget"}
        if v not in allowed:
            raise ValueError(f"worker_tier must be one of {allowed}")
        return v

    @field_validator("pm_model")
    @classmethod
    def validate_pm_model(cls, v: str | None) -> str | None:
        if v is None:
            return v
        allowed = _get_allowed_model_ids()
        if allowed and v not in allowed:
            raise ValueError(f"pm_model '{v}' is not in the allowed model registry")
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
async def health() -> dict[str, Any]:
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
    }


@app.get("/api/models")
async def get_models() -> dict[str, Any]:
    """
    Return the curated allowed model registry, structured by agent role.
    Used by the UI model picker and CLI `aifactory models list`.
    """
    return {"roles": _build_models_response()}


@app.get("/api/tasks")
async def list_tasks(
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
async def submit_task(req: TaskRequest) -> TaskResponse:
    """
    Submit a task for the multi-agent pipeline to execute.
    Returns a task_id immediately; use GET /api/tasks/{task_id} to poll.
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
async def get_task(task_id: str) -> TaskStatus:
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
async def get_task_agents(task_id: str) -> dict[str, Any]:
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
async def memory_stats() -> dict[str, Any]:
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
async def memory_store(body: dict[str, Any]) -> dict[str, Any]:
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
async def memory_retrieve(body: dict[str, Any]) -> dict[str, Any]:
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
async def memory_promote(body: dict[str, Any]) -> dict[str, Any]:
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
async def memory_clear_agent(agent_id: str) -> dict[str, Any]:
    """Clear all memories for a specific agent."""
    try:
        router = _get_memory_router()
        router.clear_agent(agent_id)
        return {"status": "cleared", "agent_id": agent_id}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── Observability Endpoints ────────────────────────────────────────────────────


@app.get("/api/observability/status")
async def observability_status() -> dict[str, Any]:
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
async def list_runs(
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
async def get_run(run_id: str) -> dict[str, Any]:
    """Get detail for a single LangSmith run."""
    from aifactory.core.observability import get_run_detail, is_tracing_enabled

    if not is_tracing_enabled():
        raise HTTPException(status_code=503, detail="LangSmith not configured")
    detail = get_run_detail(run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return detail


@app.get("/api/tasks/{task_id}/traces")
async def get_task_traces(
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
    """
    await websocket.accept()

    t = await db.get_task(task_id)
    if t is None:
        await websocket.send_json({"type": "error", "message": "Task not found"})
        await websocket.close()
        return

    if task_id not in _ws_subscribers:
        _ws_subscribers[task_id] = []
    _ws_subscribers[task_id].append(websocket)

    try:
        # Keep connection alive; events are pushed by _run_pipeline
        while True:
            # Receive any client messages (e.g. pause/resume) — ignore for now
            await asyncio.wait_for(websocket.receive_text(), timeout=30)
    except (WebSocketDisconnect, TimeoutError):
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
    Execute the AiFactory pipeline with intelligent complexity routing.

    Flow:
      1. Assess task complexity (cheap gpt-4o-mini call)
      2. Route:
         - trivial  → single Worker (skip PM + TL)
         - moderate/complex → full hierarchy (PM → TL → Workers)
      3. Track input/output tokens + cost in USD per agent
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
        from aifactory.core.assessor import (
            TaskComplexity,
            assess_complexity,
            cost_usd,
        )
        from aifactory.core.schemas import WorkerSpecialty
        from aifactory.graphs.single_team import SingleTeamPipeline

        specialties = [
            WorkerSpecialty.CODER,
            WorkerSpecialty.RESEARCHER,
            WorkerSpecialty.WRITER,
        ][: req.num_workers]

        # Resolve models
        from aifactory.config.settings import get_settings

        settings = get_settings()
        tier_model_map = {
            "premium": settings.premium_model,
            "standard": settings.standard_model,
            "budget": settings.budget_model,
        }
        pm_model_resolved = req.pm_model or settings.premium_model
        worker_model_resolved = tier_model_map.get(req.worker_tier, settings.budget_model)

        # ── Step 1: Assess complexity ─────────────────────────────────
        await update(progress=0.08)
        await _broadcast(
            task_id,
            {"type": "status", "message": "Assessing task complexity...", "progress": 0.08},
        )

        assessment = await assess_complexity(req.task)
        is_trivial = assessment.complexity == TaskComplexity.TRIVIAL

        await _broadcast(
            task_id,
            {
                "type": "assessment",
                "complexity": assessment.complexity.value,
                "reasoning": assessment.reasoning,
                "route": "direct_worker" if is_trivial else "full_pipeline",
                "progress": 0.12,
            },
        )

        # ── Step 2: Route based on complexity ─────────────────────────
        if is_trivial:
            # ── TRIVIAL PATH: single Worker, no PM/TL overhead ───────
            from aifactory.agents.worker import WorkerAgent
            from aifactory.core.schemas import AgentConfig, AgentRole

            specialty = WorkerSpecialty.CODER
            if assessment.suggested_specialties:
                with contextlib.suppress(ValueError):
                    specialty = WorkerSpecialty(assessment.suggested_specialties[0])

            worker_config = AgentConfig(
                role=AgentRole.WORKER,
                name="Worker-Direct",
                model=worker_model_resolved,
                specialty=specialty,
                team_id="direct",
                max_iterations=10,
                token_budget=8000,
                temperature=0.2,
            )
            worker = WorkerAgent(config=worker_config)

            agents_info = [
                {
                    "agent_id": worker.config.agent_id,
                    "name": worker.config.name,
                    "role": "worker",
                    "model": worker.config.model,
                    "specialty": specialty.value,
                    "status": "active",
                    "tokens_used": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                    "parent_id": None,
                }
            ]
            await db.upsert_agents(task_id, agents_info)
            await update(progress=0.15)
            await _broadcast(
                task_id,
                {
                    "type": "agent_hierarchy",
                    "message": f"Trivial task → direct Worker ({specialty.value})",
                    "progress": 0.15,
                    "agents": agents_info,
                },
            )

            await update(progress=0.3)
            await _broadcast(
                task_id,
                {"type": "status", "message": "Worker executing task...", "progress": 0.3},
            )

            task_result = await worker.execute_task(req.task, context=req.context)

            # Cost tracking
            w_in = worker.total_input_tokens
            w_out = worker.total_output_tokens
            w_cost = cost_usd(worker.config.model, w_in, w_out)

            agents_info[0].update(
                {
                    "tokens_used": worker.total_tokens_used,
                    "input_tokens": w_in,
                    "output_tokens": w_out,
                    "cost_usd": round(w_cost, 6),
                    "status": "completed",
                }
            )

            result = {
                "title": req.task[:100],
                "summary": task_result.summary,
                "artifacts": [a.model_dump() for a in task_result.artifacts],
                "quality_assessment": f"Direct worker execution (trivial). "
                f"Assessment: {assessment.reasoning}",
                "open_issues": task_result.issues_found,
                "recommendations": [],
                "metrics": {
                    "complexity": assessment.complexity.value,
                    "route": "direct_worker",
                    "total_iterations": worker.iteration,
                    "total_tokens": worker.total_tokens_used,
                    "total_input_tokens": w_in,
                    "total_output_tokens": w_out,
                    "total_cost_usd": round(w_cost, 6),
                    "total_teams": 0,
                    "total_agents": 1,
                },
            }

        else:
            # ── FULL PIPELINE PATH: PM → TL → Workers ────────────────
            pipeline = SingleTeamPipeline(
                num_workers=req.num_workers,
                worker_specialties=specialties or None,
                pm_model=pm_model_resolved,
                worker_model=worker_model_resolved,
                skip_assessment=True,  # Already assessed above
            )

            # Setup agents manually so we can stream hierarchy to UI
            initial_state = {
                "task": req.task,
                "context": req.context,
                "assessment": assessment,
                "route": "full",
                "pm": None,
                "team_leads": [],
                "workers": [],
                "delivery": None,
                "status": "starting",
                "error": None,
                "total_tokens": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cost_usd": 0.0,
            }

            setup_result = await pipeline._setup_agents(initial_state)
            pm = setup_result["pm"]
            team_leads = setup_result["team_leads"]

            # Build agent info for UI
            agents_info = []
            if pm:
                agents_info.append(
                    {
                        "agent_id": pm.config.agent_id,
                        "name": pm.config.name,
                        "role": pm.config.role.value,
                        "model": pm.config.model,
                        "specialty": None,
                        "status": "active",
                        "tokens_used": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cost_usd": 0.0,
                        "parent_id": None,
                    }
                )
            for tl in team_leads:
                agents_info.append(
                    {
                        "agent_id": tl.config.agent_id,
                        "name": tl.config.name,
                        "role": tl.config.role.value,
                        "model": tl.config.model,
                        "specialty": None,
                        "status": "active",
                        "tokens_used": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cost_usd": 0.0,
                        "parent_id": pm.config.agent_id if pm else None,
                    }
                )
                for w in tl.workers:
                    agents_info.append(
                        {
                            "agent_id": w.config.agent_id,
                            "name": w.config.name,
                            "role": w.config.role.value,
                            "model": w.config.model,
                            "specialty": w.config.specialty.value if w.config.specialty else None,
                            "status": "idle",
                            "tokens_used": 0,
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "cost_usd": 0.0,
                            "parent_id": tl.config.agent_id,
                        }
                    )

            await db.upsert_agents(task_id, agents_info)
            await update(progress=0.15)
            await _broadcast(
                task_id,
                {
                    "type": "agent_hierarchy",
                    "message": f"Assembled: 1 PM, {len(team_leads)} TL, "
                    f"{sum(len(tl.workers) for tl in team_leads)} Workers",
                    "progress": 0.15,
                    "agents": agents_info,
                },
            )

            await update(progress=0.2)
            await _broadcast(
                task_id,
                {
                    "type": "status",
                    "status": "running",
                    "message": "PM is planning the project...",
                    "progress": 0.2,
                },
            )

            delivery = await pm.execute_project(
                project_description=req.task,
                context=req.context,
            )

            # Calculate total tokens and cost per agent
            total_tokens = 0
            total_in = 0
            total_out = 0
            total_cost = 0.0

            for info in agents_info:
                agent = None
                if pm and info["agent_id"] == pm.config.agent_id:
                    agent = pm
                for tl in team_leads:
                    if info["agent_id"] == tl.config.agent_id:
                        agent = tl
                    for w in tl.workers:
                        if info["agent_id"] == w.config.agent_id:
                            agent = w
                if agent:
                    a_cost = cost_usd(
                        agent.config.model,
                        agent.total_input_tokens,
                        agent.total_output_tokens,
                    )
                    info["tokens_used"] = agent.total_tokens_used
                    info["input_tokens"] = agent.total_input_tokens
                    info["output_tokens"] = agent.total_output_tokens
                    info["cost_usd"] = round(a_cost, 6)
                    info["status"] = "completed"
                    total_tokens += agent.total_tokens_used
                    total_in += agent.total_input_tokens
                    total_out += agent.total_output_tokens
                    total_cost += a_cost

            result = {
                "title": delivery.project_title,
                "summary": delivery.summary,
                "artifacts": [a.model_dump() for a in delivery.deliverables],
                "quality_assessment": delivery.quality_assessment,
                "open_issues": delivery.open_issues,
                "recommendations": delivery.recommendations,
                "metrics": {
                    "complexity": assessment.complexity.value,
                    "route": "full_pipeline",
                    "total_iterations": delivery.metrics.get("total_iterations", 0),
                    "total_tokens": delivery.metrics.get("total_tokens", total_tokens),
                    "total_input_tokens": total_in,
                    "total_output_tokens": total_out,
                    "total_cost_usd": round(total_cost, 6),
                    "total_teams": delivery.metrics.get("teams_used", len(team_leads)),
                    "total_agents": delivery.metrics.get(
                        "agents_used",
                        1 + len(team_leads) + sum(len(tl.workers) for tl in team_leads),
                    ),
                },
            }

        # ── Common: persist to Postgres + cache + broadcast ────────────
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
