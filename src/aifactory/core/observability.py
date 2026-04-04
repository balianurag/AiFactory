"""
LangSmith Observability — trace every LLM call, graph node, and memory op.

Setup:
  1. Set LANGCHAIN_TRACING_V2=true in .env
  2. Set LANGCHAIN_API_KEY=lsv2_pt_xxxxx in .env
  3. Set LANGCHAIN_PROJECT=aifactory (or any project name)

When LANGCHAIN_API_KEY is absent, all tracing is silently disabled (no-op).
This means dev environments without a key work identically — zero overhead.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# ── State ──────────────────────────────────────────────────────────────────────

_initialised = False
_tracing_enabled = False


def setup_langsmith() -> bool:
    """
    Initialise LangSmith tracing from settings.

    Call once at app startup (e.g. in api/main.py or cli.py).
    Safe to call multiple times — only runs once.

    Returns True if tracing was successfully enabled.
    """
    global _initialised, _tracing_enabled

    if _initialised:
        return _tracing_enabled

    _initialised = True

    from aifactory.config.settings import get_settings

    settings = get_settings()

    if not settings.langchain_api_key:
        logger.info("LangSmith: no API key — tracing disabled")
        _tracing_enabled = False
        return False

    # Set env vars that langsmith/langchain read automatically
    os.environ["LANGCHAIN_TRACING_V2"] = "true" if settings.langchain_tracing else "false"
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint

    _tracing_enabled = True
    logger.info(
        "LangSmith: tracing enabled → project=%s endpoint=%s",
        settings.langchain_project,
        settings.langchain_endpoint,
    )
    return True


def is_tracing_enabled() -> bool:
    """Check if LangSmith tracing is active."""
    return _tracing_enabled


# ── Traceable Decorator ────────────────────────────────────────────────────────


def traceable(
    name: str | None = None,
    run_type: str = "chain",
    metadata: dict[str, Any] | None = None,
) -> Callable[[F], F]:
    """
    Decorator that wraps a function with LangSmith tracing.

    When tracing is disabled (no API key), this is a zero-cost passthrough.

    Args:
        name: Display name in LangSmith (defaults to function name).
        run_type: One of "llm", "chain", "tool", "retriever".
        metadata: Extra key-value metadata attached to the trace.

    Usage:
        @traceable(name="agent_think", run_type="llm")
        async def think(self, ...):
            ...
    """

    def decorator(fn: F) -> F:
        @wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not _tracing_enabled:
                return await fn(*args, **kwargs)

            try:
                from langsmith import traceable as ls_traceable

                traced_fn = ls_traceable(
                    name=name or fn.__name__,
                    run_type=run_type,
                    metadata=metadata or {},
                )(fn)
                return await traced_fn(*args, **kwargs)
            except ImportError:
                return await fn(*args, **kwargs)
            except Exception:
                # Never let tracing break the actual function
                return await fn(*args, **kwargs)

        @wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not _tracing_enabled:
                return fn(*args, **kwargs)

            try:
                from langsmith import traceable as ls_traceable

                traced_fn = ls_traceable(
                    name=name or fn.__name__,
                    run_type=run_type,
                    metadata=metadata or {},
                )(fn)
                return traced_fn(*args, **kwargs)
            except ImportError:
                return fn(*args, **kwargs)
            except Exception:
                return fn(*args, **kwargs)

        import asyncio

        if asyncio.iscoroutinefunction(fn):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper  # type: ignore[return-value]

    return decorator


# ── LangSmith Client Helpers ───────────────────────────────────────────────────


def get_langsmith_client() -> Any | None:
    """
    Return a LangSmith Client instance, or None if tracing is disabled.
    Used by API endpoints to fetch run data.
    """
    if not _tracing_enabled:
        return None

    try:
        from langsmith import Client

        return Client()
    except Exception as e:
        logger.warning("Failed to create LangSmith client: %s", e)
        return None


def get_recent_runs(
    limit: int = 20,
    project_name: str | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch recent runs from LangSmith.

    Returns a list of dicts with run summary info.
    Returns empty list if tracing is disabled or API fails.
    """
    client = get_langsmith_client()
    if client is None:
        return []

    from aifactory.config.settings import get_settings

    proj = project_name or get_settings().langchain_project

    try:
        runs = list(
            client.list_runs(
                project_name=proj,
                is_root=True,
                limit=limit,
            )
        )
        return [
            {
                "run_id": str(r.id),
                "name": r.name or "unnamed",
                "run_type": r.run_type,
                "status": r.status,
                "start_time": r.start_time.isoformat() if r.start_time else None,
                "end_time": r.end_time.isoformat() if r.end_time else None,
                "latency_ms": (
                    int((r.end_time - r.start_time).total_seconds() * 1000)
                    if r.end_time and r.start_time
                    else None
                ),
                "total_tokens": r.total_tokens or 0,
                "prompt_tokens": r.prompt_tokens or 0,
                "completion_tokens": r.completion_tokens or 0,
                "total_cost": _safe_float(getattr(r, "total_cost", None)),
                "error": r.error if r.status == "error" else None,
                "tags": r.tags or [],
                "metadata": dict(r.extra.get("metadata", {})) if r.extra else {},
            }
            for r in runs
        ]
    except Exception as e:
        logger.warning("Failed to fetch LangSmith runs: %s", e)
        return []


def get_run_detail(run_id: str) -> dict[str, Any] | None:
    """Fetch detail for a single run. Returns None if not found."""
    client = get_langsmith_client()
    if client is None:
        return None

    try:
        r = client.read_run(run_id)
        # Get child runs for the trace tree
        children = list(
            client.list_runs(
                trace_id=r.trace_id,
                limit=100,
            )
        )
        return {
            "run_id": str(r.id),
            "trace_id": str(r.trace_id) if r.trace_id else None,
            "name": r.name or "unnamed",
            "run_type": r.run_type,
            "status": r.status,
            "start_time": r.start_time.isoformat() if r.start_time else None,
            "end_time": r.end_time.isoformat() if r.end_time else None,
            "latency_ms": (
                int((r.end_time - r.start_time).total_seconds() * 1000)
                if r.end_time and r.start_time
                else None
            ),
            "total_tokens": r.total_tokens or 0,
            "prompt_tokens": r.prompt_tokens or 0,
            "completion_tokens": r.completion_tokens or 0,
            "total_cost": _safe_float(getattr(r, "total_cost", None)),
            "error": r.error,
            "inputs": dict(r.inputs) if r.inputs else {},
            "outputs": dict(r.outputs) if r.outputs else {},
            "tags": r.tags or [],
            "children": [
                {
                    "run_id": str(c.id),
                    "name": c.name or "unnamed",
                    "run_type": c.run_type,
                    "status": c.status,
                    "latency_ms": (
                        int((c.end_time - c.start_time).total_seconds() * 1000)
                        if c.end_time and c.start_time
                        else None
                    ),
                    "total_tokens": c.total_tokens or 0,
                    "total_cost": _safe_float(getattr(c, "total_cost", None)),
                }
                for c in children
                if str(c.id) != str(r.id)
            ],
        }
    except Exception as e:
        logger.warning("Failed to fetch run detail %s: %s", run_id, e)
        return None


def get_task_traces(
    task_id: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """
    Fetch LangSmith runs linked to a specific task.

    Searches for root runs whose metadata or tags contain the task_id.
    Returns empty list if tracing is disabled or API fails.
    """
    client = get_langsmith_client()
    if client is None:
        return []

    from aifactory.config.settings import get_settings

    proj = get_settings().langchain_project

    try:
        # Search for runs that have this task_id in metadata or filter by tag
        all_runs = list(
            client.list_runs(
                project_name=proj,
                is_root=True,
                limit=limit * 3,  # Over-fetch then filter
            )
        )
        # Filter to runs whose metadata contains this task_id
        matched = []
        for r in all_runs:
            meta = r.extra.get("metadata", {}) if r.extra else {}
            tags = r.tags or []
            run_name = r.name or ""
            # Match on metadata task_id, tags, or run name
            if meta.get("task_id") == task_id or task_id in tags or task_id in run_name:
                matched.append(
                    {
                        "run_id": str(r.id),
                        "trace_id": str(r.trace_id) if r.trace_id else None,
                        "name": r.name or "unnamed",
                        "run_type": r.run_type,
                        "status": r.status,
                        "start_time": r.start_time.isoformat() if r.start_time else None,
                        "end_time": r.end_time.isoformat() if r.end_time else None,
                        "latency_ms": (
                            int((r.end_time - r.start_time).total_seconds() * 1000)
                            if r.end_time and r.start_time
                            else None
                        ),
                        "total_tokens": r.total_tokens or 0,
                        "prompt_tokens": r.prompt_tokens or 0,
                        "completion_tokens": r.completion_tokens or 0,
                        "total_cost": _safe_float(getattr(r, "total_cost", None)),
                        "error": r.error if r.status == "error" else None,
                        "tags": tags,
                    }
                )
            if len(matched) >= limit:
                break
        return matched
    except Exception as e:
        logger.warning("Failed to fetch task traces for %s: %s", task_id, e)
        return []


def _safe_float(v: Any) -> float:
    """Safely convert to float, default 0.0."""
    try:
        return float(v) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0
