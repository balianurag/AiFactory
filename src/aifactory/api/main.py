"""
AiFactory FastAPI HTTP/WebSocket API
Wraps the LangGraph single-team pipeline for web consumption.

Endpoints:
  POST /api/auth/register   — create user account
  POST /api/auth/login      — get access + refresh tokens
  POST /api/auth/refresh    — rotate refresh token
  POST /api/auth/logout     — revoke refresh token
  GET  /api/auth/me         — current user profile
  GET  /api/users/api-keys  — list API keys
  POST /api/users/api-keys  — create API key
  DELETE /api/users/api-keys/{id} — revoke API key
  POST /api/orgs            — create organisation
  GET  /api/orgs            — list user's orgs
  GET  /api/orgs/{id}       — org details
  POST /api/orgs/{id}/members — add member
  GET  /api/orgs/{id}/members — list members
  GET  /api/health          — liveness + version + DB/Redis/Qdrant status
  GET  /api/models          — allowed model registry per agent role
  POST /api/tasks           — submit a new task, returns task_id
  GET  /api/tasks           — list all tasks (for DashboardView / ProjectsView)
  GET  /api/tasks/{task_id} — poll task status + partial results
  GET  /api/tasks/{task_id}/agents — live agent hierarchy for a task
  GET  /api/memory/stats    — memory layer statistics
  GET  /api/integrations    — list integrations + status
  POST /api/integrations/{name}/test — test integration connectivity
  POST /api/integrations/execute — direct tool execution (UI/debug)
  WS   /api/ws/{task_id}    — stream real-time agent events

Authentication:
  JWT Bearer  — Authorization: Bearer <access_token>
  API Key     — X-API-Key: aif_<key>  (user-scoped, created via /api/users/api-keys)
  Shared Key  — X-API-Key: <AIFACTORY_API_KEY>  (legacy service-to-service, backward compat)

Persistence: PostgreSQL (tasks, agents, messages, users, orgs) + Redis (hot cache).
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
from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from slowapi.errors import RateLimitExceeded

from aifactory.core.auth import decode_access_token, hash_api_key

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

# ── Auth ──────────────────────────────────────────────────────────────────────


class UserContext(BaseModel):
    """Resolved identity attached to each authenticated request."""
    user_id: str
    org_id: str | None = None
    scopes: list[str] = []
    auth_type: str  # "jwt" | "api_key" | "shared_key"


async def get_current_user(request: Request) -> UserContext | None:
    """
    Resolve the current user from the request.

    Accepts (in priority order):
      1. Authorization: Bearer <jwt>  — user session (access token)
      2. X-API-Key: aif_<key>          — user-scoped API key (starts with "aif_")
      3. X-API-Key: <shared_key>        — legacy AIFACTORY_API_KEY (service-to-service)

    Returns None when no credentials are present and AIFACTORY_REQUIRE_AUTH is False.
    Raises HTTP 401 when credentials are present but invalid, or when
    AIFACTORY_REQUIRE_AUTH=True and no credentials are provided.
    """
    from aifactory.config.settings import get_settings
    from jose import JWTError

    settings = get_settings()

    authorization = request.headers.get("Authorization", "")
    x_api_key = request.headers.get("X-API-Key", "")

    # ── JWT Bearer ────────────────────────────────────────────────────────────
    if authorization.startswith("Bearer "):
        token = authorization[len("Bearer "):]
        try:
            payload = decode_access_token(token)
        except JWTError as exc:
            raise HTTPException(status_code=401, detail=f"Invalid token: {exc}") from exc
        return UserContext(
            user_id=payload["sub"],
            auth_type="jwt",
        )

    # ── API Key ───────────────────────────────────────────────────────────────
    if x_api_key:
        # User-scoped API key (prefixed "aif_")
        if x_api_key.startswith("aif_"):
            key_hash = hash_api_key(x_api_key)
            row = await db.lookup_api_key(key_hash)
            if not row:
                raise HTTPException(status_code=401, detail="Invalid API key")
            if not row.get("is_active") or not row.get("user_active"):
                raise HTTPException(status_code=401, detail="API key is inactive or account disabled")
            expires = row.get("expires_at")
            if expires and expires.replace(tzinfo=UTC) < datetime.now(UTC):
                raise HTTPException(status_code=401, detail="API key has expired")
            # Fire-and-forget last_used update
            asyncio.create_task(db.touch_api_key(key_hash))
            return UserContext(
                user_id=str(row["user_id"]),
                org_id=str(row["org_id"]) if row.get("org_id") else None,
                scopes=list(row.get("scopes") or []),
                auth_type="api_key",
            )

        # Legacy shared key (backward compat — AIFACTORY_API_KEY)
        expected = settings.api_key.strip()
        if expected and x_api_key == expected:
            return UserContext(user_id="service", auth_type="shared_key")
        if expected:
            raise HTTPException(status_code=401, detail="Invalid X-API-Key")

    # ── No credentials ────────────────────────────────────────────────────────
    if settings.require_auth:
        raise HTTPException(status_code=401, detail="Authentication required")
    return None


async def require_auth(request: Request) -> UserContext:
    """Dependency: raises 401 if the request is not authenticated."""
    ctx = await get_current_user(request)
    if ctx is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return ctx


# Backward-compat shim — existing endpoints use Depends(verify_api_key)
async def verify_api_key(request: Request) -> None:
    """
    Backward-compatible auth check for existing mutating endpoints.

    Accepts JWT, user API key, or legacy shared key when AIFACTORY_API_KEY is set.
    No-op when no auth is configured (dev mode).
    """
    from aifactory.config.settings import get_settings

    settings = get_settings()
    has_jwt = request.headers.get("Authorization", "").startswith("Bearer ")
    has_key = bool(request.headers.get("X-API-Key", ""))

    # If any auth is provided, resolve it properly (raises on bad creds)
    if has_jwt or has_key:
        await get_current_user(request)
        return

    # No credentials: only fail when legacy shared key is configured
    if settings.api_key.strip() and not settings.require_auth:
        raise HTTPException(status_code=401, detail="X-API-Key header required")


# ── Lifecycle (connect/disconnect DB + Redis) ─────────────────────────────────


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Startup: verify DB connectivity. Shutdown: close connections."""
    # ── Startup ──
    ok = await db.pg_health()
    if ok:
        logger.info("✅ PostgreSQL connected")
        # Idempotent DDL — creates tables if they don't exist
        try:
            await asyncio.to_thread(db.setup_db)
            logger.info("✅ Database schema ready")
        except Exception as _ddl_exc:
            logger.warning("⚠️  setup_db failed (non-fatal): %s", _ddl_exc)
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
    description=(
        "Self-assembling multi-agent orchestration system.\n\n"
        "**Authentication**: Mutating endpoints (`POST`, `DELETE`) require an "
        "`X-API-Key` header when `AIFACTORY_API_KEY` is set. "
        "Auth is disabled in dev mode (empty key)."
    ),
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
    swagger_ui_parameters={"persistAuthorization": True},
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

# ── Auth + User routes ────────────────────────────────────────────────────────

from aifactory.api.routes.auth import router as _auth_router
from aifactory.api.routes.users import router as _users_router
from aifactory.api.routes.orgs import router as _orgs_router

app.include_router(_auth_router)
app.include_router(_users_router)
app.include_router(_orgs_router)

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
    plan: str | None = None  # "starter" | "pro" | "enterprise" — overrides mode

    # D1: Multi-modal attachments — HTTPS image URLs or data URIs (data:image/jpeg;base64,...)
    # Vision-capable workers will receive these alongside their task description.
    attachments: list[str] = []

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

    @field_validator("plan")
    @classmethod
    def validate_plan(cls, v: str | None) -> str | None:
        if v is None:
            return v
        allowed = {"starter", "pro", "enterprise"}
        if v not in allowed:
            raise ValueError(f"plan must be one of {allowed}")
        return v

    @property
    def effective_mode(self) -> str:
        """Resolve the pipeline mode: plan → mode → graph.yaml default."""
        if self.plan:
            return _get_plan(self.plan)["mode"]
        return self.mode or "balanced"


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


# ── Subscription Plans ────────────────────────────────────────────────────────

# Plans map subscriptions to pipeline modes + budget caps.
# The mode field drives model_selector.select_model() scoring:
#   cost_efficient -> fastest/cheapest models  (alpha=0.75 -> speed & cost weighted)
#   balanced       -> blended quality/cost      (alpha=0.35 -> task abilities lead)
#   best           -> most capable models        (alpha=1.0  -> max quality)

SUBSCRIPTION_PLANS: list[dict[str, Any]] = [
    {
        "id": "starter",
        "name": "Starter",
        "mode": "cost_efficient",
        "description": "Fast, budget-friendly AI teams for everyday tasks",
        "price_usd": 0,
        "price_label": "Free",
        "token_budget": 100_000,
        "cost_budget_usd": 0.50,
        "max_agents": 4,
        "features": [
            "GPT-4o Mini & Gemini Flash workers",
            "Up to 100k tokens per task",
            "Basic quality assessment",
        ],
        "badge_color": "green",
    },
    {
        "id": "pro",
        "name": "Pro",
        "mode": "balanced",
        "description": "Balanced quality & cost for professional workloads",
        "price_usd": 29,
        "price_label": "$29/mo",
        "token_budget": 500_000,
        "cost_budget_usd": 5.00,
        "max_agents": 8,
        "features": [
            "GPT-4o, Claude Sonnet & Gemini Pro",
            "Up to 500k tokens per task",
            "Full quality assessment + judge review",
            "Priority support",
        ],
        "badge_color": "cyan",
    },
    {
        "id": "enterprise",
        "name": "Enterprise",
        "mode": "best",
        "description": "Maximum quality — best models for every agent role",
        "price_usd": 99,
        "price_label": "$99/mo",
        "token_budget": 2_000_000,
        "cost_budget_usd": 25.00,
        "max_agents": 16,
        "features": [
            "GPT-5, Claude Opus & Gemini 2.5 Pro",
            "Up to 2M tokens per task",
            "Full judge review + RL performance scoring",
            "Custom agent configurations",
            "Dedicated support",
        ],
        "badge_color": "violet",
    },
]

_PLAN_MAP: dict[str, dict[str, Any]] = {p["id"]: p for p in SUBSCRIPTION_PLANS}


def _get_plan(plan_id: str) -> dict[str, Any]:
    """Resolve a plan by id, defaulting to starter."""
    return _PLAN_MAP.get(plan_id, _PLAN_MAP["starter"])


@app.get("/api/plans")
@limiter.limit(RATE_LIMIT_READ)
async def get_plans(request: Request) -> dict[str, Any]:
    """
    Return available subscription plans with their mode mappings.
    The frontend uses this to show the plan picker and send the
    correct ``mode`` when submitting tasks.
    """
    return {"plans": SUBSCRIPTION_PLANS}


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
async def submit_task(
    request: Request,
    req: TaskRequest,
    _auth: None = Depends(verify_api_key),
) -> TaskResponse:
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
async def memory_store(
    request: Request,
    body: dict[str, Any],
    _auth: None = Depends(verify_api_key),
) -> dict[str, Any]:
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
async def memory_retrieve(
    request: Request,
    body: dict[str, Any],
    _auth: None = Depends(verify_api_key),
) -> dict[str, Any]:
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
async def memory_promote(
    request: Request,
    body: dict[str, Any],
    _auth: None = Depends(verify_api_key),
) -> dict[str, Any]:
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
async def memory_clear_agent(
    request: Request,
    agent_id: str,
    _auth: None = Depends(verify_api_key),
) -> dict[str, Any]:
    """Clear all memories for a specific agent."""
    try:
        router = _get_memory_router()
        router.clear_agent(agent_id)
        return {"status": "cleared", "agent_id": agent_id}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── Integration Endpoints ──────────────────────────────────────────────────────

_integration_registry_instance: Any = None


def _get_integration_registry() -> Any:
    """Lazy-init singleton IntegrationRegistry."""
    global _integration_registry_instance
    if _integration_registry_instance is None:
        from aifactory.integrations.registry import IntegrationRegistry

        _integration_registry_instance = IntegrationRegistry()
    return _integration_registry_instance


class ExecuteToolRequest(BaseModel):
    """Request body for direct tool execution."""

    integration: str = "composio"
    tool: str
    params: dict[str, Any] = {}


@app.get("/api/integrations")
@limiter.limit(RATE_LIMIT_READ)
async def list_integrations(request: Request) -> dict[str, Any]:
    """List all registered integrations and their status."""
    try:
        registry = _get_integration_registry()
        return {
            "integrations": registry.info(),
            "summary": registry.summary(),
        }
    except Exception as e:
        return {"integrations": [], "summary": {}, "error": str(e)}


@app.post("/api/integrations/{name}/test")
@limiter.limit(RATE_LIMIT_READ)
async def test_integration(request: Request, name: str) -> dict[str, Any]:
    """Test connectivity for a specific integration."""
    try:
        registry = _get_integration_registry()
        integration = registry.get(name)
        if not integration:
            raise HTTPException(status_code=404, detail=f"Integration '{name}' not found")
        if not integration.is_connected:
            ok = await integration.connect()
            if not ok:
                return {
                    "name": name,
                    "status": "error",
                    "detail": f"Connection failed: {integration.status}",
                }
        healthy = await integration.health_check()
        return {
            "name": name,
            "status": "healthy" if healthy else "unhealthy",
            "info": integration.info(),
        }
    except HTTPException:
        raise
    except Exception as e:
        return {"name": name, "status": "error", "detail": str(e)}


@app.post("/api/integrations/execute")
@limiter.limit(RATE_LIMIT_TASKS)
async def execute_integration_tool(request: Request, body: ExecuteToolRequest) -> dict[str, Any]:
    """Execute a tool directly (for UI/debug use). Rate-limited as task."""
    try:
        registry = _get_integration_registry()
        result = await registry.execute(body.integration, body.tool, body.params)
        return {
            "success": result.success,
            "action": result.action,
            "integration": result.integration,
            "data": result.data,
            "error": result.error,
            "url": result.url,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/integrations/connect/{toolkit}")
@limiter.limit(RATE_LIMIT_READ)
async def connect_integration_toolkit(request: Request, toolkit: str) -> dict[str, Any]:
    """
    Initiate OAuth connection for a toolkit (GitHub, GitLab, Slack, etc.).

    Returns a redirect_url for the user to complete authorization.
    After auth, the connection becomes ACTIVE and tools are available.
    """
    try:
        import os

        from composio import Composio

        key = os.getenv("COMPOSIO_API_KEY", "")
        if not key:
            raise HTTPException(status_code=500, detail="COMPOSIO_API_KEY not configured")

        client = Composio(api_key=key)

        # Find or create auth config for this toolkit
        auth_config_id = None
        existing = client.auth_configs.list()
        for c in existing.items:
            d = c.model_dump() if hasattr(c, "model_dump") else {}
            if d.get("toolkit", {}).get("slug") == toolkit:
                auth_config_id = d["id"]
                break

        if not auth_config_id:
            cfg = client.auth_configs.create(
                toolkit=toolkit,
                options={"type": "use_composio_managed_auth", "name": f"{toolkit}-aifactory"},
            )
            auth_config_id = cfg.id

        # Initiate connection
        conn = client.connected_accounts.initiate(
            user_id="default",
            auth_config_id=auth_config_id,
        )

        return {
            "toolkit": toolkit,
            "connection_id": conn.id,
            "status": conn.status,
            "redirect_url": conn.redirect_url,
        }
    except HTTPException:
        raise
    except ImportError:
        raise HTTPException(status_code=500, detail="composio package not installed") from None
    except Exception as e:
        return {"toolkit": toolkit, "error": str(e)}


@app.get("/api/integrations/connections")
@limiter.limit(RATE_LIMIT_READ)
async def list_integration_connections(request: Request) -> dict[str, Any]:
    """List all connected toolkit accounts and their status."""
    try:
        import os

        from composio import Composio

        key = os.getenv("COMPOSIO_API_KEY", "")
        if not key:
            return {"connections": [], "error": "COMPOSIO_API_KEY not configured"}

        client = Composio(api_key=key)
        resp = client.connected_accounts.list()

        connections = []
        for a in resp.items:
            d = a.model_dump() if hasattr(a, "model_dump") else {}
            connections.append(
                {
                    "id": d.get("id"),
                    "toolkit": d.get("toolkit", {}).get("slug", "unknown"),
                    "status": d.get("status", "unknown"),
                    "user_id": d.get("user_id", ""),
                    "created_at": d.get("created_at", ""),
                }
            )

        return {
            "connections": connections,
            "total": len(connections),
            "active": sum(1 for c in connections if c["status"] == "ACTIVE"),
        }
    except ImportError:
        return {"connections": [], "error": "composio package not installed"}
    except Exception as e:
        return {"connections": [], "error": str(e)}


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

    if rc.get_redis() is not None:
        # Redis mode: subscribe to pub/sub and forward events to this client.
        # Works across all API replicas — events published by any replica arrive here.
        async def _redis_forward() -> None:
            async for event in rc.subscribe_events(task_id):
                try:
                    await websocket.send_json(event)
                    if event.get("type") in ("completed", "error"):
                        break
                except Exception:
                    break

        redis_fwd = asyncio.create_task(_redis_forward())
        _background_tasks.add(redis_fwd)
        redis_fwd.add_done_callback(_background_tasks.discard)
        try:
            while True:
                try:
                    await asyncio.wait_for(websocket.receive_text(), timeout=30)
                except TimeoutError:
                    continue
        except WebSocketDisconnect:
            pass
        finally:
            redis_fwd.cancel()
        return

    # Redis unavailable — fall back to in-process subscriber dict
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
    """Push an event to all WebSocket subscribers for a task.

    When Redis is available: publishes to pub/sub for cross-replica fan-out
    (WebSocket clients subscribe directly; no in-process dict needed).
    When Redis is unavailable: falls back to direct in-process delivery.
    """
    if rc.get_redis() is not None:
        await rc.publish_event(task_id, event)
        return

    # Redis fallback — direct in-process send
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

        # A3: event callback streams worker_start/worker_complete to WebSocket clients
        def _worker_event_cb(event: dict[str, Any]) -> None:
            """Sync callback fired by _run_worker_task; schedules async broadcast."""
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(_broadcast(task_id, event))  # noqa: RUF006
            except Exception:
                pass  # never break the pipeline over a broadcast failure

        compiled_graph, registry = build_dynamic_pipeline(
            mode=req.effective_mode,
            run_id=task_id,
            event_callback=_worker_event_cb,
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

        import time as _time

        _pipeline_t0 = _time.perf_counter()

        # Stream graph execution events
        initial_state: dict[str, Any] = {
            "user_query": req.task,
            "shared_context": {"task_id": task_id, "context": req.context or ""},
            "conversation_history": [],
        }
        if req.attachments:
            initial_state["attachments"] = req.attachments

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

        _pipeline_elapsed = _time.perf_counter() - _pipeline_t0

        # Write JSON run dump
        try:
            from aifactory.core.model_selector import load_graph_config

            _cfg = load_graph_config()
            write_dump(task_id, _cfg, final_state, registry, _pipeline_elapsed, req.task)
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
                "elapsed_seconds": round(_pipeline_elapsed, 2),
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
