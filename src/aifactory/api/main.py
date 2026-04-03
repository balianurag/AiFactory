"""
AiFactory FastAPI HTTP/WebSocket API
Wraps the LangGraph single-team pipeline for web consumption.

Endpoints:
  GET  /api/health          — liveness + version
  GET  /api/models          — allowed model registry per agent role
  POST /api/tasks           — submit a new task, returns task_id
  GET  /api/tasks           — list all tasks (for DashboardView / ProjectsView)
  GET  /api/tasks/{task_id} — poll task status + partial results
  GET  /api/tasks/{task_id}/agents — live agent hierarchy for a task
  GET  /api/memory/stats    — memory layer statistics
  WS   /api/ws/{task_id}    — stream real-time agent events

Render deployment: uvicorn aifactory.api.main:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import asyncio
import functools
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from aifactory.core.observability import setup_langsmith

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AiFactory API",
    description="Self-assembling multi-agent orchestration system",
    version="0.2.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
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

# ── In-memory task store (replace with Redis/DB in production) ────────────────

tasks: dict[str, dict[str, Any]] = {}
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
    """Liveness probe for Render."""
    return {
        "status": "ok",
        "version": "0.2.0",
        "phase": "2 — LangSmith + Assessment Team",
        "timestamp": _now(),
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
    items = []
    for t in tasks.values():
        if status and t.get("status") != status:
            continue
        items.append(
            {
                "task_id": t["task_id"],
                "status": t["status"],
                "created_at": t["created_at"],
                "updated_at": t["updated_at"],
                "progress": t.get("progress", 0),
                "task_description": t.get("request", {}).get("task", "")[:120],
                "agents": t.get("agents", []),
                "metrics": t.get("metrics", {}),
                "error": t.get("error"),
                "has_result": t.get("result") is not None,
            }
        )
    # Most recent first
    items.sort(key=lambda x: x["created_at"], reverse=True)
    return {"tasks": items[:limit], "total": len(items)}


@app.post("/api/tasks", response_model=TaskResponse, status_code=202)
async def submit_task(req: TaskRequest) -> TaskResponse:
    """
    Submit a task for the multi-agent pipeline to execute.
    Returns a task_id immediately; use GET /api/tasks/{task_id} to poll.
    """
    task_id = str(uuid.uuid4())
    now = _now()

    tasks[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "created_at": now,
        "updated_at": now,
        "progress": 0.0,
        "agents": [],
        "messages": [],
        "result": None,
        "error": None,
        "metrics": {},
        "request": req.model_dump(),
        "subscribers": [],
    }

    # Fire-and-forget: run pipeline in background
    bg = asyncio.create_task(_run_pipeline(task_id, req))
    _background_tasks.add(bg)
    bg.add_done_callback(_background_tasks.discard)

    return TaskResponse(task_id=task_id, status="pending", created_at=now)


@app.get("/api/tasks/{task_id}", response_model=TaskStatus)
async def get_task(task_id: str) -> TaskStatus:
    """Poll task status and partial results."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    t = tasks[task_id]
    return TaskStatus(**{k: v for k, v in t.items() if k not in ("subscribers", "request")})


@app.get("/api/tasks/{task_id}/agents")
async def get_task_agents(task_id: str) -> dict[str, Any]:
    """Return the live agent hierarchy for a running/completed task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return {
        "task_id": task_id,
        "agents": tasks[task_id].get("agents", []),
        "messages": tasks[task_id].get("messages", []),
    }


@app.get("/api/memory/stats")
async def memory_stats() -> dict[str, Any]:
    """Return memory layer statistics. Used by MemoryView and DashboardView."""
    try:
        from aifactory.core.memory_router import MemoryRouter

        router = MemoryRouter()
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
    """List recent LangSmith runs. Used by the Traces view in the UI."""
    from aifactory.core.observability import get_recent_runs, is_tracing_enabled

    if not is_tracing_enabled():
        return {"runs": [], "enabled": False, "message": "LangSmith not configured"}
    runs = get_recent_runs(limit=limit)
    return {"runs": runs, "enabled": True, "total": len(runs)}


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


@app.websocket("/api/ws/{task_id}")
async def websocket_task(websocket: WebSocket, task_id: str) -> None:
    """
    Real-time WebSocket stream for a task.
    Client receives JSON events as agents execute.
    """
    await websocket.accept()

    if task_id not in tasks:
        await websocket.send_json({"type": "error", "message": "Task not found"})
        await websocket.close()
        return

    tasks[task_id]["subscribers"].append(websocket)

    try:
        # Keep connection alive; events are pushed by _run_pipeline
        while True:
            # Receive any client messages (e.g. pause/resume) — ignore for now
            await asyncio.wait_for(websocket.receive_text(), timeout=30)
    except (WebSocketDisconnect, TimeoutError):
        pass
    finally:
        if websocket in tasks[task_id].get("subscribers", []):
            tasks[task_id]["subscribers"].remove(websocket)


# ── Pipeline runner ───────────────────────────────────────────────────────────


async def _broadcast(task_id: str, event: dict[str, Any]) -> None:
    """Push an event to all WebSocket subscribers for a task."""
    dead: list[WebSocket] = []
    for ws in tasks.get(task_id, {}).get("subscribers", []):
        try:
            await ws.send_json(event)
        except Exception:
            dead.append(ws)
    for ws in dead:
        tasks[task_id]["subscribers"].remove(ws)


async def _run_pipeline(task_id: str, req: TaskRequest) -> None:
    """
    Execute the AiFactory pipeline and stream events.
    Phase 1: SingleTeamPipeline — PM → Team Lead → Workers.
    Broadcasts live agent hierarchy, messages, and metrics via WebSocket.
    """

    def update(patch: dict[str, Any]) -> None:
        tasks[task_id].update({**patch, "updated_at": _now()})

    try:
        update({"status": "running", "progress": 0.05})
        await _broadcast(task_id, {"type": "status", "status": "running", "progress": 0.05})

        # Import here to avoid slow startup at module load
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

        # Build pipeline (so we can inspect agent hierarchy)
        pipeline = SingleTeamPipeline(
            num_workers=req.num_workers,
            worker_specialties=specialties or None,
            pm_model=pm_model_resolved,
            worker_model=worker_model_resolved,
        )

        # Run setup manually to capture agent hierarchy
        initial_state = {
            "task": req.task,
            "context": req.context,
            "pm": None,
            "team_leads": [],
            "workers": [],
            "delivery": None,
            "status": "starting",
            "error": None,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
        }

        # Setup agents — this creates the hierarchy
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
                        "parent_id": tl.config.agent_id,
                    }
                )

        update({"agents": agents_info, "progress": 0.1})

        # Broadcast agent hierarchy
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

        # Now execute the project — reuse the already-created agents
        update({"progress": 0.2})
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

        # Update agent tokens after execution
        total_tokens = pm.total_tokens_used
        for tl in team_leads:
            total_tokens += tl.total_tokens_used
            for w in tl.workers:
                total_tokens += w.total_tokens_used

        # Update agents_info with final token counts
        for info in agents_info:
            if pm and info["agent_id"] == pm.config.agent_id:
                info["tokens_used"] = pm.total_tokens_used
                info["status"] = "completed"
            for tl in team_leads:
                if info["agent_id"] == tl.config.agent_id:
                    info["tokens_used"] = tl.total_tokens_used
                    info["status"] = "completed"
                for w in tl.workers:
                    if info["agent_id"] == w.config.agent_id:
                        info["tokens_used"] = w.total_tokens_used
                        info["status"] = "completed"

        result = {
            "title": delivery.project_title,
            "summary": delivery.summary,
            "artifacts": [a.model_dump() for a in delivery.deliverables],
            "quality_assessment": delivery.quality_assessment,
            "open_issues": delivery.open_issues,
            "recommendations": delivery.recommendations,
            "metrics": {
                "total_iterations": delivery.metrics.get("total_iterations", 0),
                "total_tokens": delivery.metrics.get("total_tokens", total_tokens),
                "total_teams": delivery.metrics.get("teams_used", len(team_leads)),
                "total_agents": delivery.metrics.get(
                    "agents_used",
                    1 + len(team_leads) + sum(len(tl.workers) for tl in team_leads),
                ),
            },
        }

        update(
            {
                "status": "completed",
                "progress": 1.0,
                "result": result,
                "agents": agents_info,
                "metrics": result["metrics"],
            }
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
        update({"status": "failed", "error": error_msg})
        await _broadcast(
            task_id,
            {
                "type": "error",
                "message": error_msg,
            },
        )
