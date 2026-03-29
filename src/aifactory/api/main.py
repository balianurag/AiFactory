"""
AiFactory FastAPI HTTP/WebSocket API
Wraps the LangGraph single-team pipeline for web consumption.

MVP endpoints:
  GET  /api/health          — liveness + version
  POST /api/tasks           — submit a new task, returns task_id
  GET  /api/tasks/{task_id} — poll task status + partial results
  WS   /api/ws/{task_id}    — stream real-time agent events

Render deployment: uvicorn aifactory.api.main:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AiFactory API",
    description="Self-assembling multi-agent orchestration system",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# ── CORS (allow Vercel frontend + local dev) ──────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://frontend:3000",  # Docker internal
        "https://*.vercel.app",
        "https://aifactory-ui.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory task store (replace with Redis/DB in production) ────────────────

tasks: dict[str, dict[str, Any]] = {}
_background_tasks: set[asyncio.Task[None]] = set()

# ── Schemas ───────────────────────────────────────────────────────────────────


class TaskRequest(BaseModel):
    task: str
    context: str = ""
    num_workers: int = 3


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


@app.get("/api/health")
async def health() -> dict[str, Any]:
    """Liveness probe for Render."""
    return {
        "status": "ok",
        "version": "0.1.0",
        "phase": "1 — Single Team MVP",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@app.post("/api/tasks", response_model=TaskResponse, status_code=202)
async def submit_task(req: TaskRequest) -> TaskResponse:
    """
    Submit a task for the multi-agent pipeline to execute.
    Returns a task_id immediately; use GET /api/tasks/{task_id} to poll.
    """
    task_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat() + "Z"

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
    return TaskStatus(**{k: v for k, v in t.items() if k != "subscribers" and k != "request"})


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
    """
    now = lambda: datetime.utcnow().isoformat() + "Z"  # noqa: E731

    def update(patch: dict[str, Any]) -> None:
        tasks[task_id].update({**patch, "updated_at": now()})

    try:
        update({"status": "running", "progress": 0.05})
        await _broadcast(task_id, {"type": "status", "status": "running", "progress": 0.05})

        # Import here to avoid slow startup at module load
        from aifactory.core.schemas import WorkerSpecialty
        from aifactory.graphs.single_team import run_single_team

        specialties = [
            WorkerSpecialty.CODER,
            WorkerSpecialty.RESEARCHER,
            WorkerSpecialty.WRITER,
        ][: req.num_workers]

        # Broadcast that agents are being assembled
        await _broadcast(
            task_id,
            {
                "type": "agent_hierarchy",
                "message": "Assembling agent hierarchy...",
                "progress": 0.1,
            },
        )
        update({"progress": 0.1})

        delivery = await run_single_team(
            task=req.task,
            context=req.context,
            num_workers=req.num_workers,
            worker_specialties=specialties or None,
        )

        result = {
            "title": delivery.title,
            "summary": delivery.summary,
            "artifacts": [a.model_dump() for a in delivery.artifacts],
            "quality_assessment": delivery.quality_assessment,
            "open_issues": delivery.open_issues,
            "recommendations": delivery.recommendations,
            "metrics": {
                "total_iterations": delivery.total_iterations,
                "total_tokens": delivery.total_tokens,
                "total_teams": delivery.total_teams,
                "total_agents": delivery.total_agents,
            },
        }

        update(
            {
                "status": "completed",
                "progress": 1.0,
                "result": result,
                "metrics": result["metrics"],
            }
        )
        await _broadcast(
            task_id,
            {
                "type": "completed",
                "progress": 1.0,
                "result": result,
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
