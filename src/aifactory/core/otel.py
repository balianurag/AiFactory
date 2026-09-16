"""
OpenTelemetry instrumentation for AiFactory.

Provides vendor-neutral distributed tracing alongside LangSmith.
Spans are emitted for:
  - PipelineAgent.think()       → span "agent.think"
  - _run_worker_task()          → span "worker.task"
  - sandbox execute()           → span "sandbox.execute"
  - Each pipeline graph node    → span "pipeline.node.<name>"

Configuration (env vars):
  AIFACTORY_OTEL_ENABLED=true       Enable OTel (default false — zero cost when off)
  AIFACTORY_OTEL_ENDPOINT=...       OTLP HTTP endpoint (e.g. http://localhost:4318)
  AIFACTORY_OTEL_SERVICE_NAME=...   Service name in traces (default "aifactory")

When opentelemetry-api / opentelemetry-sdk are not installed, all decorators
become transparent no-ops — no import error, no broken pipeline.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Generator
from contextlib import contextmanager, suppress
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# ── Singleton tracer (initialised lazily) ─────────────────────────────────────

_tracer: Any = None
_initialized: bool = False


def _get_tracer() -> Any:
    """Return the OTel tracer singleton, or None if OTel is disabled/unavailable."""
    global _tracer, _initialized
    if _initialized:
        return _tracer

    _initialized = True
    try:
        from aifactory.config.settings import get_settings

        s = get_settings()
        if not s.otel_enabled:
            _tracer = None
            return None

        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": s.otel_service_name})
        provider = TracerProvider(resource=resource)

        if s.otel_endpoint:
            exporter = OTLPSpanExporter(endpoint=s.otel_endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))

        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("aifactory", "1.0.0")
        logger.info(
            "OTel tracer initialised (endpoint=%s, service=%s)",
            s.otel_endpoint or "(none)",
            s.otel_service_name,
        )
    except ImportError:
        logger.debug(
            "opentelemetry-sdk not installed — OTel instrumentation disabled. "
            "pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http"
        )
        _tracer = None
    except Exception as exc:
        logger.warning("OTel init failed (non-fatal): %s", exc)
        _tracer = None

    return _tracer


# ── Context manager ───────────────────────────────────────────────────────────


@contextmanager
def span(name: str, attributes: dict[str, Any] | None = None) -> Generator[Any]:
    """
    Context manager that wraps a block in an OTel span.

    Usage:
        with span("worker.task", {"task.id": tid, "worker.model": model}):
            ...

    No-op when OTel is disabled or not installed.
    """
    tracer = _get_tracer()
    if tracer is None:
        yield None
        return

    with tracer.start_as_current_span(name) as _span:
        if attributes and _span.is_recording():
            for k, v in attributes.items():
                with suppress(Exception):
                    _span.set_attribute(k, str(v))
        yield _span


# ── Context propagation for asyncio.gather workers ────────────────────────────
#
# When workers are spawned via asyncio.gather() inside execution_lead, each
# coroutine runs in the same thread but with its own implicit OTel context.
# Without explicitly attaching the parent context, the worker spans become
# orphans and the trace tree breaks.  These helpers capture the current
# context at dispatch time and re-attach it inside each worker coroutine.


def capture_current_context() -> Any:
    """
    Snapshot the active OTel context, or None when OTel is unavailable.

    The returned token is opaque — pass it to ``attach_context()`` inside
    the worker coroutine to restore parent-span linkage.
    """
    if _get_tracer() is None:
        return None
    try:
        from opentelemetry import context as ot_context

        return ot_context.get_current()
    except Exception:
        return None


@contextmanager
def attach_context(token: Any) -> Generator[None]:
    """
    Re-attach a captured OTel context for the duration of a block.

    Pair with ``capture_current_context()``:
        ctx = capture_current_context()
        async def coro():
            with attach_context(ctx):
                with span("worker.task", attrs):
                    ...
    """
    if token is None or _get_tracer() is None:
        yield
        return

    from opentelemetry import context as ot_context

    detach_token = None
    try:
        detach_token = ot_context.attach(token)
        yield
    finally:
        if detach_token is not None:
            with suppress(Exception):
                ot_context.detach(detach_token)


@contextmanager
def worker_span(
    *,
    worker_id: str,
    wave_index: int,
    task_type: str,
    parent_context: Any = None,
    extra: dict[str, Any] | None = None,
) -> Generator[Any]:
    """
    Convenience wrapper used by ``_run_worker_task``.

    Combines ``attach_context()`` with a span named ``worker.task`` carrying
    the structured worker attributes the design canvas calls for.  Stays a
    pure passthrough when OTel is disabled.
    """
    with attach_context(parent_context):
        attrs: dict[str, Any] = {
            "worker.id": worker_id,
            "wave.index": wave_index,
            "task.type": task_type,
        }
        if extra:
            attrs.update(extra)
        with span("worker.task", attrs) as _span:
            yield _span


# ── Decorators ────────────────────────────────────────────────────────────────


def trace_async(span_name: str, **static_attrs: Any) -> Callable[[F], F]:
    """
    Decorator for async functions: wraps the call in an OTel span.

    Usage:
        @trace_async("agent.think", agent_role="worker")
        async def think(self, msg, ctx):
            ...
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = _get_tracer()
            if tracer is None:
                return await fn(*args, **kwargs)

            with tracer.start_as_current_span(span_name) as _span:
                if _span.is_recording():
                    for k, v in static_attrs.items():
                        _span.set_attribute(k, str(v))
                try:
                    result = await fn(*args, **kwargs)
                    return result
                except Exception as exc:
                    if _span.is_recording():
                        _span.record_exception(exc)
                        _span.set_status(
                            __import__(
                                "opentelemetry.trace", fromlist=["StatusCode"]
                            ).StatusCode.ERROR
                        )
                    raise

        return wrapper  # type: ignore[return-value]

    return decorator


def trace_sync(span_name: str, **static_attrs: Any) -> Callable[[F], F]:
    """
    Decorator for sync functions: wraps the call in an OTel span.

    Usage:
        @trace_sync("sandbox.execute", backend="local")
        def execute_python(code, *, timeout=30):
            ...
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = _get_tracer()
            if tracer is None:
                return fn(*args, **kwargs)

            with tracer.start_as_current_span(span_name) as _span:
                if _span.is_recording():
                    for k, v in static_attrs.items():
                        _span.set_attribute(k, str(v))
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    if _span.is_recording():
                        _span.record_exception(exc)
                    raise

        return wrapper  # type: ignore[return-value]

    return decorator
