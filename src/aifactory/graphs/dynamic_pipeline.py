"""
Dynamic Pipeline — The config-driven 8-node LangGraph agent hierarchy.

Ported from agent-arch/main.py into production with full integration:
  - LiteLLM for cross-provider LLM calls
  - Ability-based model selection per agent and per worker task
  - YAML-driven agent configs (config/agents/*.yaml)
  - 3-layer Qdrant memory (Agent → Team → Global)
  - SHARE: protocol for lightweight inter-agent context broadcasting
  - LangGraph MemorySaver checkpointing for human-in-the-loop interrupt()
  - WebSocket-compatible event broadcasting

Graph topology:
  START → master_receive → judge_brief → analyse
        → [researcher → reanalyse]?   (pre-analysis research loop)
        → [researcher]?               (execution research)
        → execution_lead              (spawns N workers dynamically)
        → judge_review
        → master_respond → END

Three conditional edges:
  1. After analyse   → direct_answer | researcher | execution_lead
  2. After researcher → reanalyse (if pre-analysis) | execution_lead
  3. After reanalyse  → researcher (if still needs research) | execution_lead
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

# LangGraph interrupt type — captured gracefully when parallel workers are gathered
try:
    from langgraph.errors import GraphInterrupt as _GraphInterrupt

    _GRAPH_INTERRUPT_TYPES: tuple[type, ...] = (_GraphInterrupt,)
except ImportError:
    _GRAPH_INTERRUPT_TYPES = ()

import litellm
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from aifactory.core.agent_factory import (
    AgentRegistry,
    build_shared_context_block,
    create_worker_config,
    extract_findings,
    extract_remember,
    extract_share,
    format_findings_context,
    merge_share,
    parse_json_robust,
)
from aifactory.core.budget_tracker import BudgetExceededError, BudgetTracker
from aifactory.core.memory_bridge import MemoryBridge
from aifactory.core.model_selector import (
    ABILITIES,
    best_model,
    load_agent_cfg,
    load_available_models,
    load_graph_config,
    load_models,
    model_for_agent,
)
from aifactory.core.observability import traceable
from aifactory.core.output_validator import validate_worker_output
from aifactory.core.retry import is_circuit_open, record_success, retry_with_backoff
from aifactory.core.sandbox import execute_code_blocks
from aifactory.integrations.executor import execute_integration_actions

logger = logging.getLogger(__name__)

# Dumps directory for run/judge artefacts
DUMPS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "dumps"

# ── B4: Axis-targeted self-correction prompts ──────────────────────────────────
# Each task type gets a dedicated self-check that focuses on the axes that matter
# most for that domain.  Falls back to the generic prompt for unknown types.

_SELF_CHECK_PROMPTS: dict[str, str] = {
    "codebase_audit": (
        "Self-check: Review your codebase audit:\n"
        "1. **FILE_READ evidence** — Did you emit FILE_READ for every file you cited? "
        "Findings based only on the structural index are NOT acceptable — you must have "
        "read the actual file content. If you cited a file without reading it, "
        "emit FILE_READ: <path> now and revise with real evidence.\n"
        "2. **Specificity** — Does every finding cite a file path AND function name?\n"
        "3. **Accuracy** — Are findings grounded in what you actually read, not guessed?\n"
        "Reply VALIDATED only if ALL findings are backed by FILE_READ evidence.\n"
        "Reply REVISION: <note>\\n<revised audit with FILE_READ: <path> for any unread files>"
    ),
    "coding": (
        "Self-check: Review your code against these axes:\n"
        "1. **Correctness** — Does it handle edge cases and produce the right output?\n"
        "2. **Completeness** — Are all requirements from the task implemented?\n"
        "3. **Quality** — Is it readable, efficient, and follows good practices?\n"
        "Reply VALIDATED if all checks pass.\n"
        "Reply REVISION: <brief note>\\n<full corrected answer> if any fix is needed."
    ),
    "math": (
        "Self-check: Verify your solution:\n"
        "1. **Step accuracy** — Check each arithmetic/algebraic step.\n"
        "2. **Unit consistency** — Are units correct and consistent throughout?\n"
        "3. **Reasonableness** — Does the final answer make intuitive sense?\n"
        "Reply VALIDATED if correct.\n"
        "Reply REVISION: <brief note>\\n<corrected full solution> if not."
    ),
    "writing": (
        "Self-check: Review your writing for:\n"
        "1. **Clarity** — Is every sentence easy to understand?\n"
        "2. **Structure** — Does it flow logically from start to finish?\n"
        "3. **Completeness** — Does it address all aspects of the request?\n"
        "Reply VALIDATED if all checks pass.\n"
        "Reply REVISION: <brief note>\\n<revised text> if needed."
    ),
    "research": (
        "Self-check: Evaluate your research output:\n"
        "1. **Source quality** — Are claims well-supported and credible?\n"
        "2. **Coverage** — Are there important gaps or missing perspectives?\n"
        "3. **Accuracy** — Are all facts stated correctly?\n"
        "Reply VALIDATED if satisfied.\n"
        "Reply REVISION: <brief note>\\n<improved answer> if needed."
    ),
    "analysis": (
        "Self-check: Review your analysis:\n"
        "1. **Evidence** — Is each conclusion supported by concrete data or reasoning?\n"
        "2. **Coverage** — Have you considered all relevant angles?\n"
        "3. **Objectivity** — Have you avoided unsupported assumptions?\n"
        "Reply VALIDATED if complete.\n"
        "Reply REVISION: <brief note>\\n<improved analysis> if needed."
    ),
}

_DEFAULT_SELF_CHECK = (
    "Self-check: Review your answer to this task.\n"
    "Is it complete, accurate, and fully addresses what was asked?\n"
    "Reply VALIDATED if yes.\n"
    "Reply REVISION: <brief fix note>\\n<your corrected full answer> if it needs fixing."
)

_SELF_CHECK_KEYWORDS: dict[str, list[str]] = {
    "coding": [
        "code",
        "implement",
        "function",
        "class",
        "algorithm",
        "script",
        "program",
        "debug",
        "test",
        "compile",
        "build",
        "refactor",
    ],
    "math": [
        "calculate",
        "compute",
        "solve",
        "math",
        "formula",
        "equation",
        "integral",
        "derivative",
        "probability",
        "statistics",
    ],
    "writing": [
        "write",
        "draft",
        "compose",
        "summarize",
        "summarise",
        "document",
        "essay",
        "blog",
        "report",
        "email",
        "letter",
    ],
    "research": [
        "research",
        "investigate",
        "find",
        "search",
        "look up",
        "gather",
        "explore",
        "survey",
        "review",
    ],
    "codebase_audit": [
        "audit security",
        "audit reliability",
        "audit code quality",
        "audit the codebase",
        "scan the codebase",
        "find flaws",
        "find vulnerabilities",
        "find bugs",
        "review the codebase",
    ],
    "analysis": [
        "analyze",
        "analyse",
        "evaluate",
        "assess",
        "compare",
        "examine",
        "diagnose",
        "breakdown",
    ],
}


def _build_self_check_prompt(task_description: str) -> str:
    """Return the axis-targeted self-check prompt for the given task description."""
    desc_lower = task_description.lower()
    # codebase_audit must be checked first — its task descriptions contain
    # "code" and "CODE_ANALYSIS" which would otherwise match the coding bucket.
    if any(kw in desc_lower for kw in _SELF_CHECK_KEYWORDS["codebase_audit"]):
        return _SELF_CHECK_PROMPTS["codebase_audit"]
    for task_type, keywords in _SELF_CHECK_KEYWORDS.items():
        if task_type == "codebase_audit":
            continue
        if any(kw in desc_lower for kw in keywords):
            return _SELF_CHECK_PROMPTS[task_type]
    return _DEFAULT_SELF_CHECK


# Suppress litellm debug logs and disable SSL verification for internal proxies
litellm.suppress_debug_info = True
litellm.ssl_verify = False

# ── LiteLLM proxy config (populated from env, injected into every call) ───────

import contextlib  # noqa: E402
import os as _os  # noqa: E402

_LITELLM_EXTRA: dict[str, str] = {}
_api_base = _os.getenv("LITELLM_API_BASE", "").strip()
_api_key = _os.getenv("LITELLM_API_KEY", "").strip()
if _api_base:
    _LITELLM_EXTRA["base_url"] = _api_base
    # Only pass LITELLM_API_KEY when routing through a proxy.
    # Without a base_url, litellm uses the native provider key (OPENAI_API_KEY, etc.)
    if _api_key:
        _LITELLM_EXTRA["api_key"] = _api_key


def _proxy_model(model_id: str) -> str:
    """
    When routing through a LiteLLM proxy, prefix with 'openai/' so litellm
    calls the proxy's /v1/chat/completions instead of a native provider API.
    When no proxy is configured, return the model_id unchanged.
    """
    if not _api_base:
        return model_id
    if model_id.startswith("openai/"):
        return model_id
    return f"openai/{model_id}"


# =============================================================================
# Helpers
# =============================================================================


def _elapsed(agent: PipelineAgent) -> str:
    """Sum per-call elapsed times recorded in each action, formatted as a string."""
    if not agent._actions:
        return "-"
    total = sum(act.get("elapsed_s", 0.0) for act in agent._actions)
    return f"{total:.1f}s"


# ── Token Efficiency helpers ────────────────────────────────────────────────

# Anthropic prompt-caching detection + system-content builder are shared with
# every other LiteLLM call site via aifactory.core.prompt_cache. Kept re-exported
# as ``_is_anthropic_model`` because existing tests / callers import it here.
from aifactory.core.prompt_cache import (  # noqa: E402  (re-export)
    build_cached_system as _build_cached_system,
    is_anthropic_model as _is_anthropic_model,
)


_VISION_MODEL_PATTERNS = (
    "gpt-4o",  # gpt-4o, gpt-4o-mini, gpt-4o-2024-*
    "gpt-4-turbo",  # gpt-4-turbo (vision)
    "claude-3",  # claude-3-*, claude-3-5-*
    "gemini",  # gemini-1.5-*, gemini-2.0-*
    "pixtral",
    "llava",
)


def _model_supports_vision(model: str) -> bool:
    """True when the model is known to support image inputs."""
    m = model.lower()
    return any(pat in m for pat in _VISION_MODEL_PATTERNS)


def _build_multimodal_content(text: str, image_urls: list[str]) -> list[dict[str, Any]]:
    """
    Build a LiteLLM-compatible multi-modal content block.

    Images can be:
      - HTTP/HTTPS URLs:   {"type": "image_url", "image_url": {"url": "https://..."}}
      - Data URIs (base64): {"type": "image_url", "image_url": {"url": "data:image/..."}}

    Returns a list of content parts suitable for the "content" field of a user message.
    """
    parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for url in image_urls:
        parts.append({"type": "image_url", "image_url": {"url": url}})
    return parts


def _init_response_cache() -> None:
    """
    Initialise LiteLLM response cache for ALL providers and tiers.

    Prefers Redis (persistent across process restarts) when REDIS_URL is set
    and reachable.  Falls back to in-memory if Redis is unavailable.

    When enabled, identical (model + messages) calls return the cached response
    at 0 tokens — useful for QUESTION/ANSWER retries, repeated testing, and
    any case where the same prompt hits the LLM twice.

    Safe to call multiple times — idempotent (skips if already initialised).
    """
    try:
        from aifactory.config.settings import get_settings as _gs

        cfg = _gs()
        if not cfg.response_cache_enabled:
            return
        # Skip if already initialised to avoid resetting a live cache mid-run
        if getattr(litellm, "cache", None) is not None:
            return
        from litellm import Cache

        ttl = cfg.response_cache_ttl if cfg.response_cache_ttl > 0 else None

        # Try Redis first — persistent across process restarts
        _redis_url = getattr(cfg, "redis_url", "") or ""
        _used_redis = False
        if _redis_url:
            try:
                from urllib.parse import urlparse

                _p = urlparse(_redis_url)
                _host = _p.hostname or "localhost"
                _port = str(_p.port or 6379)
                _password = _p.password or ""
                # Quick connectivity probe before committing to Redis cache
                import redis as _redis_lib  # type: ignore[import]

                _probe = _redis_lib.Redis(
                    host=_host,
                    port=int(_port),
                    password=_password or None,
                    socket_connect_timeout=1,
                )
                _probe.ping()
                _probe.close()
                litellm.cache = Cache(
                    type="redis",
                    host=_host,
                    port=_port,
                    password=_password,
                    ttl=ttl,
                    namespace="aifactory_llm",
                )
                litellm.cache_responses = True
                logger.info(
                    "LiteLLM response cache initialised (Redis %s, TTL=%s)",
                    f"{_host}:{_port}",
                    f"{ttl}s" if ttl else "none",
                )
                _used_redis = True
            except Exception as _redis_exc:
                logger.debug("Redis cache unavailable (%s) — falling back to in-memory", _redis_exc)

        if not _used_redis:
            litellm.cache = Cache(ttl=ttl)
            litellm.cache_responses = True
            logger.info(
                "LiteLLM response cache initialised (in-memory, TTL=%s)",
                f"{ttl}s" if ttl else "none",
            )
    except Exception as exc:
        logger.debug("Response cache init skipped: %s", exc)


#: Context keys that are always retained during pruning (ordered by priority)
_CTX_PRIORITY_KEYS: frozenset[str] = frozenset(
    {
        "objective",
        "tasks_count",
        "connected_integrations",
        "research_snippet",
        "analysis_research",
    }
)


def _prune_shared_context(ctx: dict[str, str], max_keys: int = 25) -> dict[str, str]:
    """
    Trim shared_context to at most max_keys entries to prevent prompt bloat.

    Priority keys are always kept (see _CTX_PRIORITY_KEYS). When pruning is
    needed, the oldest non-priority keys are dropped first — most recently
    inserted keys are kept since they carry the freshest context.

    Returns ctx unchanged when len(ctx) <= max_keys.
    """
    if len(ctx) <= max_keys:
        return ctx

    priority = {k: v for k, v in ctx.items() if k in _CTX_PRIORITY_KEYS}
    others = [(k, v) for k, v in ctx.items() if k not in _CTX_PRIORITY_KEYS]

    slots = max(0, max_keys - len(priority))
    kept_others = dict(others[-slots:])  # most recently added keys

    dropped = len(ctx) - len(priority) - len(kept_others)
    logger.debug("shared_context pruned: %d key(s) dropped (now %d)", dropped, max_keys)
    return {**priority, **kept_others}


def _build_dependency_waves(
    tasks: list[dict[str, Any]],
    dependencies_map: dict[str, list[str]],
) -> list[list[dict[str, Any]]]:
    """
    Split tasks into ordered execution waves respecting dependency constraints.

    Tasks in the same wave have no dependencies on each other and can run
    concurrently.  Tasks in later waves wait for all earlier waves to complete.

    If a circular dependency is detected (no tasks become ready), the first
    blocked task is forced into its own wave to prevent an infinite loop.
    """
    if not any(deps for deps in dependencies_map.values()):
        # No dependencies declared → single wave, all tasks in parallel
        return [tasks]

    completed_ids: set[str] = set()
    waves: list[list[dict[str, Any]]] = []
    remaining = list(tasks)

    while remaining:
        ready = [
            t
            for t in remaining
            if all(dep in completed_ids for dep in dependencies_map.get(t["id"], []))
        ]
        if not ready:
            # Circular dep or unknown dep ID — force first task to unblock
            logger.warning(
                "Dependency resolution stalled for %s — forcing %s into its own wave",
                [t["id"] for t in remaining],
                remaining[0]["id"],
            )
            ready = [remaining[0]]
        waves.append(ready)
        ready_ids = {t["id"] for t in ready}
        completed_ids |= ready_ids
        remaining = [t for t in remaining if t["id"] not in ready_ids]

    return waves


# =============================================================================
# Lightweight Agent — single class for all roles (config-driven)
# =============================================================================


class AgentBudgetExceededError(Exception):
    """Raised when a PipelineAgent exceeds its per-agent token budget."""


class _NullAsyncContext:
    """No-op async context manager used when no asyncio.Lock is provided."""

    async def __aenter__(self) -> _NullAsyncContext:
        return self

    async def __aexit__(self, *_: Any) -> None:
        pass


class PipelineAgent:
    """
    A single LLM agent with private message history.

    This is the production version of agent-arch's Agent class, integrated with:
      - litellm.acompletion for cross-provider LLM calls
      - Token tracking per agent
      - Actions audit log for judge review and run dumps
      - SHARE: protocol support
      - Per-agent token budget enforcement
      - LangSmith tracing via @traceable

    Role-specific behavior comes entirely from the YAML config + system_prompt.
    No Python subclasses needed — one class handles all roles.
    """

    #: Default per-agent token budget. Can be overridden via cfg["token_budget"].
    DEFAULT_TOKEN_BUDGET: int = 50_000

    def __init__(
        self,
        name: str,
        cfg: dict[str, Any],
        model: str | None = None,
        memory: MemoryBridge | None = None,
    ) -> None:
        self.name = name
        self.model = model or cfg.get("model", "gpt-4o-mini")
        self.cfg = cfg
        self._system_prompt: str = cfg.get("system_prompt", "")
        self._messages: list[dict[str, str]] = []  # private history
        self._actions: list[dict[str, Any]] = []  # audit log
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.token_budget: int = int(cfg.get("token_budget", self.DEFAULT_TOKEN_BUDGET))
        # Phase 2.1: 3-layer memory bridge (None = disabled, noop in all ops)
        self.memory: MemoryBridge | None = memory
        # Phase 5 hardening: fallback model when circuit opens, budget tracker
        self.fallback_model: str = cfg.get("fallback_model", "gpt-4o-mini")
        self._budget_tracker: BudgetTracker | None = None
        # Token streaming: callable(token_text) → None; set by build_dynamic_pipeline
        self.on_token: Any = None

    @traceable(name="pipeline_agent_think", run_type="llm")
    # D2: OTel span wrapping is applied via the context manager inside think()
    async def think(
        self,
        user_msg: str,
        shared_ctx: dict[str, str] | None = None,
    ) -> str:
        """
        Send a message to the LLM and get a response.

        Injects shared_context and relevant memory into the system prompt.
        Maintains private conversation history. Tracks tokens per call.
        Stores the reply in agent-layer memory after each call.

        Raises:
            AgentBudgetExceededError: when total_tokens reaches token_budget.
        """
        if self.total_tokens >= self.token_budget:
            raise AgentBudgetExceededError(
                f"Agent {self.name} exceeded token budget ({self.total_tokens}/{self.token_budget})"
            )

        # Phase 2.1: retrieve relevant memories before the LLM call (defense-in-depth
        # try/except here: MemoryBridge is already fail-safe, but guard think() too
        # so a bridge bug never kills the pipeline)
        memory_context = ""
        if self.memory is not None:
            try:
                from aifactory.config.settings import get_settings

                _top_k = get_settings().memory_context_top_k
                memory_context = self.memory.retrieve_context(
                    agent_id=self.name,
                    query=user_msg,
                    top_k=_top_k,
                )
            except Exception as _mem_exc:
                logger.debug("[%s] memory retrieve skipped: %s", self.name, _mem_exc)

        system = self._build_system(shared_ctx, memory_context)
        self._messages.append({"role": "user", "content": user_msg})

        # Phase 3.5: Anthropic prompt caching — inject cache_control on the
        # system message for Claude models.  Delegated to the shared helper so
        # every LiteLLM call site (BaseAgent.think, think_multimodal, the
        # complexity assessor, …) applies the exact same gating rules.
        sys_content: list[dict[str, Any]] | str = _build_cached_system(
            system, self.model, api_base=_api_base
        )

        # ── Phase 5: Budget check + Circuit breaker + Retry ─────────────
        # 1. Hard budget cap — raise before any tokens are consumed
        if self._budget_tracker is not None:
            self._budget_tracker.check()

        # 2. Circuit breaker — switch to fallback if primary model has too many failures
        try:
            from aifactory.config.settings import get_settings as _gs_retry

            _s_retry = _gs_retry()
            _cb_threshold = int(_s_retry.circuit_breaker_threshold)
            _retry_attempts = int(_s_retry.llm_retry_max_attempts)
            _retry_base = float(_s_retry.llm_retry_base_delay_s)
        except Exception:
            _cb_threshold, _retry_attempts, _retry_base = 3, 3, 1.0

        _effective_model = self.model
        if is_circuit_open(self.model, threshold=_cb_threshold):
            logger.warning(
                "[%s] Circuit OPEN for '%s' — switching to fallback '%s'",
                self.name,
                self.model,
                self.fallback_model,
            )
            _effective_model = self.fallback_model

        _messages_snapshot = [{"role": "system", "content": sys_content}, *self._messages]
        _max_tokens = self.cfg.get("max_tokens", 1000)

        def _make_llm_call(use_model: str):
            return litellm.acompletion(
                model=_proxy_model(use_model),
                messages=_messages_snapshot,
                max_tokens=_max_tokens,
                caching=False,  # user turns must always hit the model live
                **_LITELLM_EXTRA,
            )

        t_start = datetime.now(UTC)
        # D2: OTel span around the LLM call
        try:
            from aifactory.core.otel import span as _otel_span
        except Exception:
            _otel_span = None  # type: ignore[assignment]
        _otel_ctx = (
            _otel_span("agent.think", {"agent.name": self.name, "agent.model": _effective_model})
            if _otel_span is not None
            else None
        )
        if _otel_ctx is not None:
            _otel_ctx.__enter__()

        # ── Token streaming path ──────────────────────────────────────
        # When on_token callback is set, stream tokens to the caller in real-time.
        # Single attempt with fallback (no retry — partial output can't be unwound).
        _streaming_active = self.on_token is not None
        if _streaming_active:
            raw_content = await self._think_stream(
                _effective_model, _messages_snapshot, _max_tokens
            )
            elapsed_s = round((datetime.now(UTC) - t_start).total_seconds(), 2)
            # Streaming token counts estimated via litellm.token_counter
            try:
                import litellm as _ll

                _prompt_toks = _ll.token_counter(
                    model=_effective_model, messages=_messages_snapshot
                )
                _compl_toks = _ll.token_counter(model=_effective_model, text=raw_content)
            except Exception:
                _prompt_toks = 0
                _compl_toks = len(raw_content.split()) * 4 // 3  # rough estimate
            usage_prompt = _prompt_toks
            usage_completion = _compl_toks
            usage_total = usage_prompt + usage_completion
            if _otel_ctx is not None:
                with contextlib.suppress(Exception):
                    _otel_ctx.__exit__(None, None, None)
        else:
            # ── Standard (non-streaming) path ────────────────────────
            try:
                resp = await retry_with_backoff(
                    lambda: _make_llm_call(_effective_model),
                    max_retries=_retry_attempts,
                    base_delay=_retry_base,
                    model_id=_effective_model,
                )
            except Exception as _primary_exc:
                # All retries on primary model exhausted — one final attempt with fallback
                if _effective_model != self.fallback_model:
                    logger.error(
                        "[%s] All retries exhausted for '%s' — trying fallback '%s'",
                        self.name,
                        _effective_model,
                        self.fallback_model,
                    )
                    resp = await _make_llm_call(self.fallback_model)
                    record_success(self.fallback_model)
                else:
                    if _otel_ctx is not None:
                        _otel_ctx.__exit__(type(_primary_exc), _primary_exc, None)
                    raise
            finally:
                if _otel_ctx is not None:
                    with contextlib.suppress(Exception):
                        _otel_ctx.__exit__(None, None, None)
            elapsed_s = round((datetime.now(UTC) - t_start).total_seconds(), 2)
            raw_content = resp.choices[0].message.content
            if raw_content is None:
                raw_content = getattr(resp.choices[0].message, "refusal", None) or ""
            usage = resp.usage
            usage_prompt = usage.prompt_tokens
            usage_completion = usage.completion_tokens
            usage_total = usage.total_tokens

        # Alias for shared tracking code below
        if not _streaming_active:
            pass  # raw_content already set inside the else branch above

        if raw_content is None:
            if not _streaming_active:
                # Non-streaming: resp is available for diagnostics
                logger.warning(
                    "[%s] message.content is None (finish_reason=%s) — checking alternatives",
                    self.name,
                    getattr(resp.choices[0], "finish_reason", "?"),  # type: ignore[possibly-undefined]
                )
                raw_content = getattr(resp.choices[0].message, "refusal", None) or ""  # type: ignore[possibly-undefined]
            else:
                raw_content = ""

        reply = raw_content.strip()
        self.total_tokens += usage_total
        self.prompt_tokens += usage_prompt
        self.completion_tokens += usage_completion

        # Phase 5: record cost against run budget tracker
        if self._budget_tracker is not None:
            self._budget_tracker.record(_effective_model, usage_prompt, usage_completion)
        self._messages.append({"role": "assistant", "content": reply})

        logger.info(
            "[%s] prompt=%d completion=%d total=%d time=%.1fs%s",
            self.name,
            usage_prompt,
            usage_completion,
            usage_total,
            elapsed_s,
            " [stream]" if _streaming_active else "",
        )

        self._actions.append(
            {
                "timestamp": t_start.isoformat(),
                "elapsed_s": elapsed_s,
                "input": user_msg,
                "output": reply,
                "shared_context_snapshot": dict(shared_ctx) if shared_ctx else {},
                "tokens": {
                    "prompt": usage_prompt,
                    "completion": usage_completion,
                    "total": usage_total,
                },
            }
        )

        # Phase 2.1: store reply in agent-layer memory + trigger promotion check
        if self.memory is not None:
            try:
                turn = len(self._actions)  # 1-based turn counter
                self.memory.store_output(
                    agent_id=self.name,
                    content=reply,
                    turn_number=turn,
                )
                # Compact summary for promotion: input snippet → output snippet
                _inp_snip = user_msg[:80].replace("\n", " ")
                _out_snip = reply[:120].replace("\n", " ")
                promotion_summary = f"[{self.name}] turn={turn} | q: {_inp_snip} | a: {_out_snip}"
                self.memory.maybe_promote(agent_id=self.name, summary=promotion_summary)
            except Exception as _mem_exc:
                logger.debug("[%s] memory store skipped: %s", self.name, _mem_exc)

        return reply

    async def _think_stream(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
    ) -> str:
        """
        Stream LLM tokens and call self.on_token for each chunk.

        Single-attempt (no retry) — partial output can't be unwound.
        Falls back to the fallback model silently if streaming fails before
        any tokens are emitted; raises if failure occurs mid-stream.
        """
        collected: list[str] = []
        try:
            stream_resp = await litellm.acompletion(
                model=_proxy_model(model),
                messages=messages,
                max_tokens=max_tokens,
                stream=True,
                caching=False,
                **_LITELLM_EXTRA,
            )
            async for chunk in stream_resp:
                delta = ""
                with contextlib.suppress(IndexError, AttributeError):
                    delta = chunk.choices[0].delta.content or ""
                if delta:
                    collected.append(delta)
                    if callable(self.on_token):
                        with contextlib.suppress(Exception):
                            self.on_token(delta)
        except Exception as exc:
            if not collected:
                # Nothing emitted yet — fall back to non-streaming call
                logger.warning(
                    "[%s] Streaming failed before first token (%s) — retrying non-streaming",
                    self.name,
                    exc,
                )
                fallback_resp = await litellm.acompletion(
                    model=_proxy_model(self.fallback_model),
                    messages=messages,
                    max_tokens=max_tokens,
                    caching=False,
                    **_LITELLM_EXTRA,
                )
                content = (fallback_resp.choices[0].message.content or "").strip()
                # Emit full response as one token so caller UI still updates
                if content and callable(self.on_token):
                    with contextlib.suppress(Exception):
                        self.on_token(content)
                return content
            else:
                # Partial stream — log and return what we have
                logger.warning(
                    "[%s] Streaming interrupted after %d chars: %s",
                    self.name,
                    sum(len(c) for c in collected),
                    exc,
                )
        return "".join(collected)

    async def think_multimodal(
        self,
        user_msg: str,
        image_urls: list[str],
        shared_ctx: dict[str, str] | None = None,
    ) -> str:
        """
        Send a multi-modal message (text + images) to a vision-capable LLM.

        Falls back to text-only think() if the assigned model doesn't support vision,
        so callers never need to check capability themselves.

        Args:
            user_msg:   The text portion of the message.
            image_urls: List of HTTPS URLs or data URIs (data:image/jpeg;base64,...).
            shared_ctx: Shared pipeline context dict.
        """
        if not image_urls or not _model_supports_vision(self.model):
            # Graceful fallback: note that images were provided but ignored
            if image_urls:
                logger.info(
                    "[%s] model '%s' does not support vision — running text-only",
                    self.name,
                    self.model,
                )
                user_msg = f"[Note: {len(image_urls)} image(s) were attached but this model cannot process them. Describe what you would need to know from the image(s) to complete the task.]\n\n{user_msg}"  # noqa: E501
            return await self.think(user_msg, shared_ctx)

        # Inject vision content into the user message (list format)
        memory_context = ""
        if self.memory is not None:
            try:
                from aifactory.config.settings import get_settings

                _top_k = get_settings().memory_context_top_k
                memory_context = self.memory.retrieve_context(
                    agent_id=self.name,
                    query=user_msg,
                    top_k=_top_k,
                )
            except Exception as _me:
                logger.debug("[%s] memory retrieve skipped: %s", self.name, _me)

        system = self._build_system(shared_ctx, memory_context)
        multimodal_content = _build_multimodal_content(user_msg, image_urls)
        self._messages.append({"role": "user", "content": multimodal_content})  # type: ignore[arg-type]

        # Phase 3.5: same Anthropic prompt-cache gating as think(). Vision
        # calls with a stable system prompt still qualify for cache_control
        # (Claude Vision honours ephemeral markers on system blocks).
        _sys_content = _build_cached_system(system, self.model, api_base=_api_base)
        _messages_snapshot = [{"role": "system", "content": _sys_content}, *self._messages]

        try:
            _max_tokens = self.cfg.get("max_tokens", 1000)
            resp = await litellm.acompletion(
                model=_proxy_model(self.model),
                messages=_messages_snapshot,
                max_tokens=_max_tokens,
                **_LITELLM_EXTRA,
            )
        except Exception as exc:
            logger.warning("[%s] vision LLM call failed: %s — retrying text-only", self.name, exc)
            # Strip the multimodal message we just added and fall back
            self._messages.pop()
            return await self.think(user_msg, shared_ctx)

        reply = (resp.choices[0].message.content or "").strip()
        usage = (
            getattr(resp, "usage", None)
            or type("U", (), {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})()
        )
        self.prompt_tokens += getattr(usage, "prompt_tokens", 0)
        self.completion_tokens += getattr(usage, "completion_tokens", 0)
        self.total_tokens += getattr(usage, "total_tokens", 0)
        self._messages.append({"role": "assistant", "content": reply})
        return reply

    def _build_system(
        self,
        shared_ctx: dict[str, str] | None,
        memory_context: str = "",
    ) -> str:
        base = self._system_prompt.strip()
        ctx_block = build_shared_context_block(shared_ctx or {})
        system = base + ctx_block if ctx_block else base
        # Memory context appended last so it's closest to the user message
        return system + memory_context if memory_context else system

    def dump(self) -> dict[str, Any]:
        """Return a complete snapshot of this agent for run dumps."""
        try:
            from aifactory.core.model_selector import cost_usd_from_registry as _cost_fn

            cost_usd: float | None = _cost_fn(
                self.model.removeprefix("openai/"),
                self.prompt_tokens,
                self.completion_tokens,
            )
        except Exception:
            cost_usd = None
        return {
            "name": self.name,
            "model": self.model,
            "role": self.cfg.get("role", self.name),
            "system_prompt": self._system_prompt[:800] + "..."
            if len(self._system_prompt) > 800
            else self._system_prompt,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "actions_count": len(self._actions),
            "cost_usd": cost_usd,
        }


# =============================================================================
# LangGraph State
# =============================================================================


class DynamicPipelineState(TypedDict, total=False):
    """State that flows through the 8-node LangGraph pipeline."""

    # Input
    user_query: str
    final_response: str
    direct_answer: str  # set by analyser → bypasses all workers/judge

    # Shared context (SHARE: protocol)
    shared_context: dict[str, str]

    # Task breakdown
    tasks: list[dict[str, Any]]

    # Pre-analysis research
    research_for_analysis: bool
    research_for_analysis_query: str
    analysis_research_summary: str

    # Execution research
    research_needed: bool
    research_query: str
    research_summary: str

    # Results
    task_results: list[dict[str, Any]]
    agent_assessments: list[dict[str, Any]]
    judge_metrics: dict[str, Any]

    # Conversation history (for post-pipeline chat)
    conversation_history: list[dict[str, str]]

    # Safety flags (Phase 5 hardening)
    budget_exceeded: bool  # set True when BudgetTracker halts the run mid-execution
    quality_flagged_tasks: list[str]  # task IDs where output quality gate flagged after retry

    # D1: Multi-modal attachments (image URLs or data URIs forwarded to vision workers)
    attachments: list[str]


# =============================================================================
# Node factories
# =============================================================================


def _build_integration_context() -> dict[str, str]:
    """Detect connected integrations and build context entries for shared_ctx.

    Only stores lightweight info (toolkit names) — the full tool catalog is
    injected directly into worker prompts by create_worker_config().
    """
    ctx: dict[str, str] = {}
    try:
        from aifactory.integrations.registry import IntegrationRegistry

        reg = IntegrationRegistry()
        toolkits = reg.get_connected_toolkit_names()
        if toolkits:
            ctx["connected_integrations"] = ", ".join(toolkits)
    except Exception:
        pass
    return ctx


async def _ensure_integrations_connected(workspace: str | None = None) -> None:
    """
    Auto-register and connect integrations if not already done.

    Called once at the start of a pipeline run (in master_receive).
    Only does the lightweight connect (health check) — NOT the expensive
    per-toolkit tool fetching.  Tool fetching is deferred to the execution
    lead node, which only fetches the specific toolkits that workers need.
    Safe to call multiple times — skips if already connected.

    Args:
        workspace: Optional workspace path for LocalShellProvider. Defaults to cwd.
    """
    try:
        from aifactory.integrations.providers.local_shell import LocalShellProvider
        from aifactory.integrations.registry import IntegrationRegistry

        reg = IntegrationRegistry()

        # Always register LocalShellProvider — no API key needed
        if not reg.get("local_shell"):
            _local_shell = LocalShellProvider(workspace=workspace or _os.getcwd())
            reg.register(_local_shell)
            await _local_shell.connect()
            logger.debug("LocalShellProvider registered (workspace=%s)", _local_shell._workspace)

        # Skip if already has a connected integration
        if reg.get_connected():
            return

        # Auto-register MCP catalog if enabled
        reg.auto_setup_mcp()

        results = await reg.connect_all()
        for name, ok in results.items():
            if ok:
                logger.info("Auto-connected integration: %s", name)
            else:
                logger.debug("Integration %s not available (missing credentials?)", name)
    except Exception as e:
        logger.debug("Integration auto-connect skipped: %s", e)


def make_master_receive(agent: PipelineAgent, semantic_cache: Any = None):
    async def master_receive(state: DynamicPipelineState) -> dict[str, Any]:
        query = state["user_query"]
        # Preserve any caller-injected keys (e.g. workspace, context) — only
        # add "objective" on top, never wipe the whole dict.
        shared_ctx: dict[str, str] = {**state.get("shared_context", {}), "objective": query}

        # Semantic cache check — return cached result immediately if intent matches
        if semantic_cache is not None:
            try:
                hit = semantic_cache.lookup(query)
                if hit:
                    logger.info(
                        "SemanticCache HIT (score=%.3f) — returning cached response",
                        hit.get("score", 0),
                    )
                    return {
                        "shared_context": {"objective": query, "semantic_cache_hit": "true"},
                        "final_response": hit["final_response"],
                        "direct_answer": hit["final_response"],
                        "task_results": [],
                        "conversation_history": [
                            {"role": "assistant", "content": hit["final_response"]}
                        ],
                    }
            except Exception as _sc_exc:
                logger.debug("SemanticCache lookup error: %s", _sc_exc)

        # ── Parallel run-start I/O ──────────────────────────────────────────
        # On semantic-cache miss, integration auto-connect (async, MCP
        # handshakes) and global-memory retrieval (sync Qdrant search) are
        # independent I/O.  Fan out concurrently with asyncio.gather instead
        # of running serially.  Global memory is wrapped in asyncio.to_thread
        # since it's a blocking Qdrant call.  The cache-HIT fast path above
        # is preserved untouched — we never pay the fan-out cost on hits.
        async def _run_integrations() -> None:
            await _ensure_integrations_connected(workspace=shared_ctx.get("workspace"))

        async def _run_global_memory() -> str:
            if agent.memory is None:
                return ""
            return await asyncio.to_thread(
                agent.memory.retrieve_global_context, query, 2
            )

        _integrations_result, _global_ctx_result = await asyncio.gather(
            _run_integrations(),
            _run_global_memory(),
            return_exceptions=True,
        )

        if isinstance(_integrations_result, BaseException):
            logger.debug("Integration auto-connect skipped: %s", _integrations_result)
        # Inject connected integration context so all agents know what tools exist
        shared_ctx.update(_build_integration_context())

        # A1: cross-run global memory result → shared_ctx
        if isinstance(_global_ctx_result, BaseException):
            logger.debug("Global memory retrieval skipped: %s", _global_ctx_result)
        elif _global_ctx_result:
            # Strip section header for cleaner storage in shared_ctx
            _header = "\n\n## Relevant Memory\n"
            shared_ctx["prior_knowledge"] = _global_ctx_result.replace(_header, "").strip()
            logger.info(
                "Global memory: injected %d chars into shared context",
                len(shared_ctx["prior_knowledge"]),
            )

        # D1: note image attachments in shared context so downstream agents are aware
        attachments: list[str] = state.get("attachments") or []
        if attachments:
            shared_ctx["attachments_count"] = str(len(attachments))
            shared_ctx["attachments_note"] = (
                f"{len(attachments)} image attachment(s) provided. "
                "Vision-capable workers will receive the image(s) alongside their task."
            )
            logger.info("Pipeline received %d image attachment(s)", len(attachments))

        # Record a lightweight action for the audit log (no LLM call needed)
        agent._actions.append(
            {
                "input": f"Request received: {query}",
                "output": "Acknowledged. Coordinating team.",
                "elapsed_s": 0.0,
            }
        )
        return {"shared_context": shared_ctx}

    return master_receive


def make_analyse(agent: PipelineAgent):
    async def analyse(state: DynamicPipelineState) -> dict[str, Any]:
        shared_ctx = state.get("shared_context", {})

        # Build integration awareness for the analyser
        integration_hint = ""
        connected = shared_ctx.get("connected_integrations", "")
        if connected:
            integration_hint = (
                f"\nINTEGRATION AWARENESS: Workers have access to these connected services: "
                f"{connected}. "
                "When decomposing tasks that involve external services, reference the "
                "correct platform (e.g. if gitlab is connected but not github, "
                "tasks should mention GitLab, not GitHub). "
                "Workers can invoke tools via ACTION: blocks.\n"
            )

        # Strip prior_knowledge from the analyser's context — it changes every run
        # as Qdrant accumulates memories, which would prevent LiteLLM cache hits.
        # The analyser only needs objective + connected_integrations to decompose.
        _analyser_ctx = {k: v for k, v in shared_ctx.items() if k != "prior_knowledge"}

        raw = await agent.think(
            f"Break down this request into minimal tasks.\n\n"
            f'Request: "{state["user_query"]}"\n\n'
            "Each task description must be an actionable goal — what the worker must accomplish. "
            "Never a statement of completed fact, never a method. "
            "Include ALL specifics from the user's request (names, IDs, usernames, paths) "
            "in the task description so workers don't need to re-ask.\n"
            "PREFER FEWER TASKS: If a request is a single tool action (list/create/update "
            "on a connected service), it should be ONE task, not split into discovery + "
            "fetch + format. Only split when work is genuinely independent and parallel.\n\n"
            "CODEBASE AUDIT RULE (apply when the request is to audit, review, scan, or find "
            "flaws/issues/bugs in an existing codebase): decompose by concern area — NOT by file. "
            "Create exactly 3 tasks: (1) security audit, (2) reliability audit, "
            "(3) code quality audit. "
            "Every task description MUST instruct the worker to: "
            "first emit 'CODE_ANALYSIS: <path>' to get the structural index, "
            "then emit 'FILE_READ: <path>' for each file needing deeper inspection, "
            "and cite specific file path + function name for every finding. "
            "Never a generic task like 'scan the codebase' — always by concern area.\n\n"
            f"{integration_hint}"
            "RESEARCH FLAGS — set these carefully:\n"
            "- research_needed = true when the request involves researching, investigating, "
            "exploring, analysing, or comparing real-world topics/entities, OR when workers "
            "would benefit from a pre-gathered factual summary. If the user explicitly "
            "says 'research', always set this to true.\n"
            "- research_query = a concise query the researcher should answer.\n"
            "- research_for_analysis = true when YOU need background knowledge before "
            "you can decompose the request into tasks properly.\n"
            "- research_for_analysis_query = what you need the researcher to look up.\n\n"
            "Output a JSON object with keys: direct_answer (str, empty if not direct), "
            "research_for_analysis (bool), research_for_analysis_query (str), "
            "research_needed (bool), research_query (str), tasks (array of {id, description}), "
            "graph_spec (object with keys: topology (one of MINIMAL/STANDARD/FULL/EXTENDED), "
            "needs_research (bool), needs_judge (bool), needs_integration (bool), "
            "worker_strategy (parallel/sequential/single), task_count (int), "
            "reasoning (one-sentence justification)). "
            "Topology guidance: MINIMAL for direct single-task answers with no quality review; "
            "STANDARD for clear 1-3 task requests that benefit from a judge but no research loop; "
            "FULL for multi-task, research-heavy, or ambiguous requests; "
            "EXTENDED reserved for tasks needing dynamic mid-run graph expansion (rare).",
            _analyser_ctx,
        )

        try:
            parsed = parse_json_robust(raw)
        except ValueError as e:
            logger.warning("Analyser JSON parse failed — falling back to single task: %s", e)
            parsed = {
                "tasks": [{"id": "task_1", "description": state["user_query"]}],
                "research_needed": False,
                "research_query": "",
            }

        direct_answer: str = parsed.get("direct_answer") or ""
        if direct_answer:
            logger.info("Analyser: DIRECT ANSWER — skipping workers")
            return {
                "direct_answer": direct_answer,
                "tasks": [],
                "research_for_analysis": False,
                "research_for_analysis_query": "",
                "research_needed": False,
                "research_query": "",
                "shared_context": shared_ctx,
            }

        raw_tasks: list[dict] = parsed.get("tasks", [])
        tasks = [
            {
                "id": str(t.get("id") or f"task_{i + 1}"),
                "description": t.get("description") or t.get("task") or str(t),
            }
            for i, t in enumerate(raw_tasks)
        ]

        research_needed = bool(parsed.get("research_needed", False))
        research_query = parsed.get("research_query") or ""

        # Safety net: if the user explicitly asks for research but the
        # analyser missed it, override to ensure the researcher runs.
        research_signals = ("research", "investigate", "look up", "find out about", "deep dive")
        query_lower = state["user_query"].lower()
        if not research_needed and any(sig in query_lower for sig in research_signals):
            research_needed = True
            research_query = research_query or state["user_query"]
            logger.info("Analyser safety-net: overriding research_needed → True")

        # Phase 3.5: prune shared_context before returning to keep prompts lean
        try:
            from aifactory.config.settings import get_settings as _gs

            _max_ctx_keys = _gs().shared_context_max_keys
        except Exception:
            _max_ctx_keys = 25

        # Extract graph_spec from analyser output and store in shared context
        _graph_spec_raw = parsed.get("graph_spec")
        _ctx_update: dict[str, Any] = {"tasks_count": str(len(tasks))}
        if _graph_spec_raw and isinstance(_graph_spec_raw, dict):
            _ctx_update["graph_spec"] = _graph_spec_raw
            logger.info(
                "Analyser graph_spec: topology=%s, needs_research=%s, task_count=%d",
                _graph_spec_raw.get("topology", "FULL"),
                _graph_spec_raw.get("needs_research", False),
                _graph_spec_raw.get("task_count", len(tasks)),
            )

        updated_ctx = _prune_shared_context({**shared_ctx, **_ctx_update}, _max_ctx_keys)
        return {
            "tasks": tasks,
            "research_for_analysis": bool(parsed.get("research_for_analysis", False)),
            "research_for_analysis_query": parsed.get("research_for_analysis_query") or "",
            "research_needed": research_needed,
            "research_query": research_query,
            "shared_context": updated_ctx,
        }

    return analyse


def make_researcher(
    agent: PipelineAgent,
    bridge: MemoryBridge | None = None,
    research_cache: Any | None = None,
    run_id: str = "",
    embed_fn: Any | None = None,
):
    async def researcher(state: DynamicPipelineState) -> dict[str, Any]:
        for_analysis = state.get("research_for_analysis", False)
        if for_analysis:
            query = state.get("research_for_analysis_query") or state["user_query"]
        else:
            query = state.get("research_query") or state["user_query"]

        shared_ctx = state.get("shared_context", {})

        # S2: Semantic research cache — check before calling LLM
        # Use embed_fn for Tier 2 cosine matching when available; falls back
        # to word-overlap (Tier 3) automatically inside ResearchCache.lookup().
        summary: str = ""
        cache_hit = False
        if research_cache is not None:
            _query_vec = None
            if embed_fn is not None:
                try:
                    _query_vec = embed_fn(query)
                except Exception:
                    pass
            cached = research_cache.lookup(query, embedding=_query_vec)
            if cached and cached.get("summary"):
                summary = cached["summary"]
                cache_hit = True
                logger.info(
                    "ResearchCache hit for %r — skipping LLM call (%d chars cached)",
                    query[:60],
                    len(summary),
                )

        if not cache_hit:
            raw = await agent.think(
                f"Research: {query}\n\n"
                "Return a dense, actionable summary — facts and findings only.",
                shared_ctx,
            )
            summary, updates = extract_share(raw)
            shared_ctx = merge_share(shared_ctx, updates, agent.name)

            # Phase 2.1: persist SHARE: updates to team memory so future runs can recall
            if bridge and updates:
                bridge.store_share_updates(agent.name, updates)

            # Store in research cache for future runs
            if research_cache is not None:
                research_cache.store(query, summary, run_id=run_id)
        else:
            # Cache hit: no SHARE: updates to process, but still update team memory
            updates: dict[str, str] = {}

        # Phase 3.5: prune shared_context after research updates
        try:
            from aifactory.config.settings import get_settings as _gs

            _max_ctx_keys = _gs().shared_context_max_keys
        except Exception:
            _max_ctx_keys = 25

        if for_analysis:
            final_ctx = _prune_shared_context(
                {**shared_ctx, "analysis_research": summary[:200]}, _max_ctx_keys
            )
            return {
                "analysis_research_summary": summary,
                "shared_context": final_ctx,
            }
        else:
            final_ctx = _prune_shared_context(
                {**shared_ctx, "research_snippet": summary[:200]}, _max_ctx_keys
            )
            return {
                "research_summary": summary,
                "shared_context": final_ctx,
            }

    return researcher


def make_reanalyse(agent: PipelineAgent):
    async def reanalyse(state: DynamicPipelineState) -> dict[str, Any]:
        shared_ctx = state.get("shared_context", {})
        research = state.get("analysis_research_summary", "")

        raw = await agent.think(
            f"Refine your task breakdown using this research:\n\n{research}\n\n"
            f'Request: "{state["user_query"]}"\n\n'
            "Output a JSON object with keys: research_needed (bool), research_query (str), "
            "tasks (array of {id, description}).",
            shared_ctx,
        )

        try:
            parsed = parse_json_robust(raw)
        except ValueError as e:
            logger.warning("Reanalyse JSON parse failed — keeping original tasks: %s", e)
            return {"research_for_analysis": False}

        raw_tasks: list[dict] = parsed.get("tasks", [])
        tasks = [
            {
                "id": str(t.get("id") or f"task_{i + 1}"),
                "description": t.get("description") or t.get("task") or str(t),
            }
            for i, t in enumerate(raw_tasks)
        ]

        return {
            "tasks": tasks if tasks else state.get("tasks", []),
            "research_for_analysis": False,
            "research_needed": bool(parsed.get("research_needed", False)),
            "research_query": parsed.get("research_query") or "",
            "shared_context": {**shared_ctx, "tasks_count": str(len(tasks))},
        }

    return reanalyse


# =============================================================================
# Worker task executor (module-level coroutine for parallel dispatch)
# =============================================================================


async def _run_worker_task(
    task: dict[str, Any],
    shared_ctx_in: dict[str, str],
    *,
    worker_prompts: dict[str, str],
    ability_reqs: dict[str, dict[str, float]],
    toolkit_assignments: dict[str, list[str]],
    tool_slug_assignments: dict[str, list[str]],
    completed: dict[str, dict[str, Any]],
    workers_cache: dict[str, PipelineAgent],
    prior_results: list[dict[str, Any]],
    mode: str,
    registry: AgentRegistry,
    bridge: MemoryBridge | None,
    event_callback: Any | None = None,
    budget_tracker: BudgetTracker | None = None,
    attachments: list[str] | None = None,
    researcher: PipelineAgent | None = None,
    findings_context: str = "",
    code_processor: Any | None = None,
    structural_index_cache: dict[str, str] | None = None,
    structural_index_lock: asyncio.Lock | None = None,
    otel_parent_context: Any | None = None,
    wave_index: int = 0,
) -> tuple[dict[str, Any], dict[str, str]]:
    """
    Execute a single worker task end-to-end.

    Each worker gets an independent copy of shared_ctx so parallel workers
    in the same wave cannot clobber each other's updates mid-run.
    Updates are returned as ctx_delta and merged by the caller after the wave.

    OTel tracing: when otel_parent_context is supplied, the worker's span is
    re-parented to the captured execution_lead context so traces remain a
    single tree across asyncio.gather() boundaries.  Spans carry worker_id,
    wave_index, and task_type as attributes for filtering downstream.

    Returns:
        (result_entry, ctx_delta) — result dict + net changes to shared context.
    """
    tid = task["id"]
    worker_name = f"Worker-{tid}"

    # Infer the task type once so we can attach it to the span attributes.
    try:
        from aifactory.core.rl_scorer import infer_task_type as _infer_task_type

        _task_type = _infer_task_type(task.get("description", ""))
    except Exception:  # noqa: BLE001 — fail-soft to "general"
        _task_type = "general"

    from aifactory.core.otel import worker_span as _worker_span

    with _worker_span(
        worker_id=worker_name,
        wave_index=wave_index,
        task_type=_task_type,
        parent_context=otel_parent_context,
    ):
        return await _run_worker_task_inner(
            task=task,
            shared_ctx_in=shared_ctx_in,
            worker_prompts=worker_prompts,
            ability_reqs=ability_reqs,
            toolkit_assignments=toolkit_assignments,
            tool_slug_assignments=tool_slug_assignments,
            completed=completed,
            workers_cache=workers_cache,
            prior_results=prior_results,
            mode=mode,
            registry=registry,
            bridge=bridge,
            event_callback=event_callback,
            budget_tracker=budget_tracker,
            attachments=attachments,
            researcher=researcher,
            findings_context=findings_context,
            code_processor=code_processor,
            structural_index_cache=structural_index_cache,
            structural_index_lock=structural_index_lock,
            worker_name=worker_name,
            tid=tid,
        )


async def _run_worker_task_inner(
    *,
    task: dict[str, Any],
    shared_ctx_in: dict[str, str],
    worker_prompts: dict[str, str],
    ability_reqs: dict[str, dict[str, float]],
    toolkit_assignments: dict[str, list[str]],
    tool_slug_assignments: dict[str, list[str]],
    completed: dict[str, dict[str, Any]],
    workers_cache: dict[str, PipelineAgent],
    prior_results: list[dict[str, Any]],
    mode: str,
    registry: AgentRegistry,
    bridge: MemoryBridge | None,
    event_callback: Any | None,
    budget_tracker: BudgetTracker | None,
    attachments: list[str] | None,
    researcher: PipelineAgent | None,
    findings_context: str,
    code_processor: Any | None,
    structural_index_cache: dict[str, str] | None,
    structural_index_lock: asyncio.Lock | None,
    worker_name: str,
    tid: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Body of _run_worker_task — extracted so we can wrap it in a span cleanly."""
    # Phase 5: emit worker_start event for streaming
    if event_callback is not None:
        with contextlib.suppress(Exception):
            event_callback(
                {
                    "type": "worker_start",
                    "worker": worker_name,
                    "task": task["description"][:120],
                }
            )

    # Each worker operates on its own snapshot to prevent parallel context races
    shared_ctx: dict[str, str] = dict(shared_ctx_in)

    # Build prior-results context.
    # When compact FINDINGS: bullets are available they fully REPLACE the full-prose
    # prior-results block — this is the 80-90% token savings the protocol delivers.
    # Fall back to full prose only when no findings exist yet (first wave, or workers
    # that haven't adopted the protocol yet).
    if findings_context:
        task_input = findings_context + "\n\nExecute your assigned task."
    else:
        prior = "\n\n".join(
            f"[{r['task_id']}]\n{r['result']}"
            for r in prior_results
            if r.get("result") and r["result"].strip() != "NOT_NEEDED"
        )
        task_input = "Execute your assigned task." + (
            f"\n\nPrior task results:\n{prior}" if prior else ""
        )

    # C3: inject cross-run global memory — "how did we solve similar tasks before?"
    if bridge is not None:
        try:
            _global_ctx = bridge.retrieve_global_context(query=task["description"], top_k=2)
            if _global_ctx:
                # Strip the "## Relevant Memory" header for cleaner injection
                _global_ctx_clean = _global_ctx.replace("\n\n## Relevant Memory\n", "").strip()
                task_input += f"\n\nRelevant prior solutions:\n{_global_ctx_clean}"
                logger.debug(
                    "[%s] injected %d chars from global memory",
                    worker_name,
                    len(_global_ctx_clean),
                )
        except Exception as _cm_exc:
            logger.debug("[%s] cross-run memory retrieval failed: %s", worker_name, _cm_exc)

    # Code-analysis primer: if the task involves codebase analysis AND workspace is
    # known, prepend an explicit CODE_ANALYSIS: trigger hint.  Models that receive
    # a task like "Analyse the codebase at /path" without this hint often skip the
    # signal entirely — the primer ensures they emit it before doing anything else.
    # Skip the primer (and the extra LLM round-trip) when the structural index has
    # already been built for this workspace in a prior worker within the same run.
    _ws = shared_ctx.get("workspace", "")
    _td_lower = task.get("description", "").lower()
    _is_code_task = _ws and any(
        kw in _td_lower
        for kw in (
            "codebase",
            "code",
            "flaw",
            "review",
            "analyze",
            "analyse",
            "audit",
            "source",
            "repo",
        )
    )
    _cached_index = (structural_index_cache or {}).get(_ws, "") if _ws else ""
    if _is_code_task:
        if _cached_index:
            # Inject the cached index directly — no extra LLM round-trip needed.
            task_input = (
                f"## Codebase Structural Index (cached)\n{_cached_index}\n\n"
                "Using the structural index above, complete your task. "
                "Reference specific file paths and function names from the index. "
                "Do NOT include raw source code in your response.\n\n"
            ) + task_input
            logger.debug(
                "[%s] CODE_ANALYSIS: using cached structural index (%d chars)",
                worker_name,
                len(_cached_index),
            )
        else:
            task_input = (
                f"IMPORTANT: Start by emitting this signal on its own line before doing anything else:\n"
                f"CODE_ANALYSIS: {_ws}\n"
                f"You will receive a safe structural summary of the codebase injected back, then complete your task based on that.\n\n"
            ) + task_input

    # Lazy-fetch toolkit metadata only for toolkits assigned to this task.
    # Use .get(tid) — None means "not in plan" (auto-detect), [] means "lead
    # said no tools" (skip integration entirely).  Do NOT default to [] here
    # or the distinction is lost and auto-detect fires for every task.
    assigned_tks = toolkit_assignments.get(tid)
    if assigned_tks:
        try:
            from aifactory.integrations.registry import IntegrationRegistry

            _reg = IntegrationRegistry()
            for integration in _reg.get_connected():
                if hasattr(integration, "fetch_available_tools"):
                    await integration.fetch_available_tools(toolkits=assigned_tks)
                    break
        except Exception as _exc:
            logger.debug("Toolkit metadata fetch failed for %s: %s", worker_name, _exc)

    assigned_slugs = tool_slug_assignments.get(tid)
    w_config = create_worker_config(
        task_id=tid,
        task_description=task["description"],
        ability_requirements=ability_reqs.get(tid),
        worker_prompt=worker_prompts.get(tid),
        mode=mode,
        assigned_toolkits=assigned_tks,  # None=auto-detect, []=no tools, ['x']=use x
        assigned_tool_slugs=assigned_slugs,
    )
    chosen_model = w_config["model"]

    # D1: determine if this is a vision task (task has image attachments + vision keywords)
    _vision_keywords = (
        "image",
        "photo",
        "picture",
        "screenshot",
        "diagram",
        "chart",
        "figure",
        "visual",
        "look at",
        "analyze the",
        "describe the",
    )
    _task_lower = task.get("description", "").lower()
    _has_vision = bool(attachments) and (
        any(kw in _task_lower for kw in _vision_keywords) or len(task.get("description", "")) < 80
    )

    # Reuse cached worker (interrupt resume path) or create fresh one
    cached_worker = workers_cache.get(tid)
    if cached_worker is not None:
        last_action = cached_worker._actions[-1] if cached_worker._actions else None
        if last_action and re.search(r"(?m)^QUESTION:", last_action["output"], re.IGNORECASE):
            worker = cached_worker
            raw = last_action["output"]
            logger.info("[%s] resuming from cached question (no re-call)", worker_name)
        else:
            worker = cached_worker
            if _has_vision:
                raw = await worker.think_multimodal(task_input, attachments or [], shared_ctx)
            else:
                raw = await worker.think(task_input, shared_ctx)
    else:
        worker = registry.register(
            PipelineAgent(worker_name, w_config, chosen_model, memory=bridge)
        )
        workers_cache[tid] = worker
        if _has_vision:
            raw = await worker.think_multimodal(task_input, attachments or [], shared_ctx)
        else:
            raw = await worker.think(task_input, shared_ctx)

    # Phase 5: wire budget tracker + token streaming into the worker
    if budget_tracker is not None:
        worker._budget_tracker = budget_tracker
    if callable(event_callback):
        try:
            from aifactory.config.settings import get_settings as _gs_w

            _w_stream = _gs_w().stream_tokens_enabled
        except Exception:
            _w_stream = True
        if _w_stream:
            _wname = worker.name

            def _worker_on_token(token: str, _wn: str = _wname) -> None:
                event_callback({"type": "token", "agent": _wn, "text": token})

            worker.on_token = _worker_on_token

    # ── RESEARCH: inline signal — worker requests mid-task research ──
    # Worker emits "RESEARCH: <query>" when it needs external facts before
    # it can complete its task.  A single inline researcher call is made and
    # the result is injected back so the worker can proceed without a full
    # pipeline restart.  Only one research round is allowed per worker to
    # prevent recursive spirals.
    if researcher is not None:
        _research_match = re.search(r"(?m)^RESEARCH:\s*(.+)", raw, re.IGNORECASE)
        if _research_match:
            _research_query = _research_match.group(1).strip()
            logger.info("[%s] RESEARCH: signal detected: %s", worker_name, _research_query[:80])
            try:
                # Budget check before inline researcher LLM call
                if budget_tracker is not None:
                    budget_tracker.check(estimated_tokens=1500)
                _inline_research = await researcher.think(
                    f"Quick research query from a worker:\n{_research_query}", shared_ctx
                )
                _inline_clean, _ = extract_share(_inline_research)
                logger.debug("[%s] inline research → %d chars", worker_name, len(_inline_clean))
                raw = await worker.think(
                    task_input + f"\n\nInline research result:\n{_inline_clean[:1500]}",
                    shared_ctx,
                )
            except Exception as _ri_exc:
                # Surface inline research failures so operators know context was missing
                logger.warning(
                    "[%s] RESEARCH: inline research failed — worker will proceed without it: %s",
                    worker_name,
                    _ri_exc,
                )

    # ── CODE_ANALYSIS: inline signal — worker requests local codebase analysis ──
    # Worker emits "CODE_ANALYSIS: /path/to/project" when it needs to reason over
    # local source code without leaking raw code to external APIs.
    # CodeSecurityProcessor reads + analyses locally; only the safe summary is
    # injected back so the worker can complete its task.
    # Raw source code NEVER reaches any external API boundary.
    if code_processor is not None:
        _ca_match = re.search(r"(?m)^CODE_ANALYSIS:\s*(.+)", raw, re.IGNORECASE)
        if _ca_match:
            _ca_path = _ca_match.group(1).strip()
            _injection = ""

            # Cache read — lock ensures only one coroutine builds the index per path.
            # All other workers wait for the first to finish, then get a cache hit.
            _lock = structural_index_lock
            async with _lock if _lock is not None else _NullAsyncContext():
                _injection = (structural_index_cache or {}).get(_ca_path, "")
                if _injection:
                    logger.debug(
                        "[%s] CODE_ANALYSIS: cache hit for %s (%d chars)",
                        worker_name,
                        _ca_path[:60],
                        len(_injection),
                    )
                    if event_callback is not None:
                        with contextlib.suppress(Exception):
                            event_callback(
                                {
                                    "type": "code_analysis_hit",
                                    "worker": worker_name,
                                    "path": _ca_path,
                                }
                            )
                else:
                    logger.info(
                        "[%s] CODE_ANALYSIS: building structural index for %s",
                        worker_name,
                        _ca_path[:120],
                    )
                    if event_callback is not None:
                        with contextlib.suppress(Exception):
                            event_callback(
                                {
                                    "type": "code_analysis_start",
                                    "worker": worker_name,
                                    "path": _ca_path,
                                }
                            )
                    try:
                        _codebase = code_processor.read_from_path(_ca_path)
                        if _codebase:
                            _safe_data = await code_processor.process_codebase(
                                _codebase, max_injection_chars=10_000
                            )
                            _safe_summary = _safe_data.get("safe_summary", "")
                            _file_count = _safe_data.get("file_count", 0)
                            _langs = ", ".join(_safe_data.get("languages", []))
                            _injection = (
                                f"## Codebase Structural Index\n"
                                f"Files: {_file_count}  Languages: {_langs}\n\n"
                                f"{_safe_summary}"
                            )
                            if structural_index_cache is not None:
                                structural_index_cache[_ca_path] = _injection
                            logger.debug(
                                "[%s] CODE_ANALYSIS: built %d-char structural index, cached",
                                worker_name,
                                len(_injection),
                            )
                        else:
                            logger.warning(
                                "[%s] CODE_ANALYSIS: path yielded no files — worker will "
                                "proceed without codebase context: %s",
                                worker_name,
                                _ca_path,
                            )
                    except Exception as _ca_exc:
                        logger.warning(
                            "[%s] CODE_ANALYSIS: failed — worker will proceed without "
                            "codebase context: %s",
                            worker_name,
                            _ca_exc,
                        )

            if _injection:
                raw = await worker.think(
                    task_input + f"\n\n{_injection}\n\n"
                    "## GROUNDED ANALYSIS RULES (mandatory)\n"
                    "You have the structural index above. Now complete your task.\n\n"
                    "REQUIRED:\n"
                    "- Every finding or claim MUST cite a specific file path AND "
                    "function/class name from the index above.\n"
                    "- If you cannot point to a concrete location in the index, "
                    "do not include that finding.\n\n"
                    "FORBIDDEN:\n"
                    "- Do NOT write Python/code to analyse the summary — reason "
                    "directly from the structural index text.\n"
                    "- Do NOT produce generic advice (e.g. 'consider adding error "
                    "handling') that applies to any project — every point must be "
                    "grounded in what you actually see in this codebase.\n"
                    "- Do NOT include raw source code in your response.\n"
                    "- Do NOT fabricate function names or file paths not present "
                    "in the index.",
                    shared_ctx,
                )
            else:
                # Inform the worker explicitly so it doesn't hallucinate file structure
                raw = await worker.think(
                    task_input + "\n\n[NOTE: Codebase structural index was unavailable. "
                    "Base your response on what is known from FINDINGS and shared context. "
                    "Do not invent file paths or function names.]",
                    shared_ctx,
                )

    # ── FILE_READ: signal — worker requests full content of a specific file ──
    # Worker emits "FILE_READ: src/aifactory/core/database.py" after reviewing
    # the structural index and needing deeper context on a specific file.
    # Processed in a loop so workers can read multiple files in one turn.
    # Secrets are stripped by CodeSecurityProcessor.read_file_safe().
    _fr_rounds = 0
    _fr_max = 5  # cap: prevent runaway multi-file reads
    if code_processor is not None:
        while _fr_rounds < _fr_max:
            _fr_match = re.search(r"(?m)^FILE_READ:\s*(.+)", raw, re.IGNORECASE)
            if not _fr_match:
                if _fr_rounds > 0 and event_callback is not None:
                    with contextlib.suppress(Exception):
                        event_callback({"type": "file_read_tokens_resume", "worker": worker_name})
                break
            _fr_path = _fr_match.group(1).strip()
            _fr_rounds += 1
            logger.info("[%s] FILE_READ: %s (round %d)", worker_name, _fr_path[:120], _fr_rounds)
            if _fr_rounds == 1 and event_callback is not None:
                with contextlib.suppress(Exception):
                    event_callback({"type": "file_read_tokens_suppress", "worker": worker_name})
            if event_callback is not None:
                with contextlib.suppress(Exception):
                    event_callback(
                        {"type": "file_read_start", "worker": worker_name, "path": _fr_path}
                    )
            # Resolve project root from shared_ctx workspace or CODE_ANALYSIS path
            _fr_root = shared_ctx.get("workspace") or shared_ctx.get("project_root") or ""
            _fr_result = code_processor.read_file_safe(_fr_path, project_root=_fr_root or None)
            if event_callback is not None:
                with contextlib.suppress(Exception):
                    event_callback(
                        {
                            "type": "file_read_done",
                            "worker": worker_name,
                            "path": _fr_result.get("path", _fr_path),
                            "lang": _fr_result.get("lang", ""),
                            "size_kb": _fr_result.get("size_kb", 0),
                            "truncated": _fr_result.get("truncated", False),
                            "error": _fr_result.get("error"),
                        }
                    )

            if "error" in _fr_result:
                _fr_injection = (
                    f"## FILE_READ result for `{_fr_path}`\n"
                    f"Error: {_fr_result['error']}\n"
                    "Choose a different file path from the structural index."
                )
            else:
                _fr_header = (
                    f"## FILE_READ: `{_fr_result['path']}` "
                    f"({_fr_result['lang']}, {_fr_result['size_kb']} KB"
                    + (", truncated" if _fr_result["truncated"] else "")
                    + ")\n"
                )
                if _fr_result["functions"]:
                    _fr_header += f"Functions: {', '.join(_fr_result['functions'][:20])}\n"
                if _fr_result["classes"]:
                    _fr_header += f"Classes: {', '.join(_fr_result['classes'][:10])}\n"
                _fr_injection = _fr_header + "\n```\n" + _fr_result["content"] + "\n```"

            raw = await worker.think(
                _fr_injection + "\n\nContinue your task using the file content above. "
                "If you need another file, emit FILE_READ: <path>. "
                "Otherwise produce your final answer.",
                shared_ctx,
            )

    # ── QUESTION: interrupt handling ────────────────────────────────
    has_tools = bool(tool_slug_assignments.get(tid) or toolkit_assignments.get(tid))
    question_retries = 0
    while True:
        q_match = re.search(r"(?m)^QUESTION:\s*(.+)", raw, re.IGNORECASE)
        if not q_match:
            break
        question = q_match.group(1).strip()
        logger.info("[%s] QUESTION: %s", worker_name, question)

        if has_tools and question_retries < 1:
            logger.info("[%s] Suppressing QUESTION (tool task) — auto-retrying", worker_name)
            question_retries += 1
            raw = await worker.think(
                "Do NOT ask questions. The integration is already connected "
                "and authenticated. Use the ACTION block to call the tools "
                "directly with the information from your task. Proceed now.",
                shared_ctx,
            )
            continue

        answer: str = interrupt(f"[{worker_name}] {question}")
        logger.info("[%s] ANSWER: %s", worker_name, answer)
        qa_key = f"qa{sum(1 for k in shared_ctx if k.startswith('qa'))}"
        shared_ctx = {**shared_ctx, qa_key: f"Q:{question}|A:{answer}"}
        raw = await worker.think(
            f'The human answered: "{answer}"\n\nNow complete your task fully.',
            shared_ctx,
        )

    result, updates = extract_share(raw)
    shared_ctx = merge_share(shared_ctx, updates, worker_name)
    if bridge and updates:
        bridge.store_share_updates(worker_name, updates)

    if not result.strip() and raw.strip():
        # Only fall back to raw if it doesn't start with a protocol marker —
        # a FINDINGS-only or REMEMBER-only response should be handled by the
        # protocol extraction below, not restored verbatim here.
        _raw_upper = raw.strip().upper()
        if not any(_raw_upper.startswith(m) for m in ("FINDINGS:", "REMEMBER:", "SHARE:")):
            logger.warning(
                "[%s] extract_share returned empty from %d-char response — using raw",
                worker_name,
                len(raw),
            )
            result = raw.strip()

    # ── Iterative tool-calling loop ──────────────────────────────────
    _tks = toolkit_assignments.get(tid, [])
    is_simple_tool_task = len(_tks) == 1
    max_tool_rounds = 1 if is_simple_tool_task else 3
    all_code_parts: list[str] = []
    all_code_outputs: list[str] = []  # just the raw output text, no code blocks
    all_integ_parts: list[str] = []
    current_raw = raw
    _had_break_after_follow = False  # set True when code succeeded and loop exited early

    # A4: adaptive sandbox timeout — coding/math tasks get more time
    _task_desc_lower = task.get("description", "").lower()
    _is_code_task = any(
        kw in _task_desc_lower
        for kw in (
            "code",
            "implement",
            "function",
            "class",
            "algorithm",
            "script",
            "program",
            "debug",
            "test",
            "compile",
            "build",
            "calculate",
            "compute",
            "math",
            "formula",
            "equation",
        )
    )
    try:
        from aifactory.config.settings import get_settings as _gs_sb

        _sb_settings = _gs_sb()
        _sandbox_timeout = (
            _sb_settings.sandbox_timeout_coding_s if _is_code_task else _sb_settings.sandbox_timeout
        )
    except Exception:
        _sandbox_timeout = 90 if _is_code_task else 30

    for tool_round in range(max_tool_rounds):
        round_feedback: list[str] = []

        _workspace = shared_ctx.get("workspace") or None
        code_runs = await execute_code_blocks(
            current_raw, timeout=_sandbox_timeout, workspace=_workspace
        )
        if code_runs:
            for lang, _code_str, exec_result in code_runs:
                status = "✓" if exec_result.success else "✗"
                logger.info(
                    "[%s] code exec round %d [%s/%s]: %s",
                    worker_name,
                    tool_round + 1,
                    lang,
                    exec_result.backend,
                    status,
                )
                _code_output = exec_result.output[:1500]
                # Include the code block itself so the final result shows both
                # the code and its output, not just the numeric result.
                part = (
                    f"[code:{lang} {status}]\n```{lang}\n{_code_str}\n```\nOutput:\n{_code_output}"
                )
                all_code_parts.append(part)
                all_code_outputs.append(_code_output)
                round_feedback.append(part)

        integration_runs = await execute_integration_actions(current_raw)
        if integration_runs:
            for tool_slug, action_result in integration_runs:
                status = "✓" if action_result.success else "✗"
                logger.info(
                    "[%s] integration round %d [%s/%s]: %s",
                    worker_name,
                    tool_round + 1,
                    action_result.integration,
                    tool_slug,
                    status,
                )
                _out_limit = 1500
                part = f"[{tool_slug} {status}]\n{action_result.output[:_out_limit]}"
                all_integ_parts.append(part)
                round_feedback.append(part)

        if not round_feedback:
            break

        feedback_text = "\n\n".join(round_feedback)
        logger.info(
            "[%s] tool round %d → %d results, re-prompting worker",
            worker_name,
            tool_round + 1,
            len(round_feedback),
        )
        all_code_succeeded = (
            bool(code_runs) and not integration_runs and all(r.success for _, _, r in code_runs)
        )

        # Detect invented slugs (ToolNotFound 404) and produce a targeted re-prompt
        # so the worker corrects itself rather than retrying with the same wrong slug.
        _not_found_slugs = [
            slug
            for slug, ar in (integration_runs or [])
            if not ar.success and "ToolNotFound" in str(ar.error or "")
        ]
        _slug_correction = ""
        if _not_found_slugs:
            logger.warning(
                "[%s] invented slug(s) detected: %s — injecting catalog reminder",
                worker_name,
                _not_found_slugs,
            )
            _slug_correction = (
                f"STOP — the slug(s) {_not_found_slugs} do NOT exist. "
                "You invented them. Open the TOOL CATALOG in your system instructions "
                "and copy a slug exactly as written — character for character. "
                "Do NOT guess, combine, or modify slug names.\n\n"
            )

        # When all code executed cleanly, skip the follow-up LLM call entirely.
        # The worker's round-1 response already has the code; the actual execution
        # output is appended below after the loop. Saves ~2000–3000 tokens per run.
        if all_code_succeeded and not _slug_correction:
            _had_break_after_follow = True
            break

        # For slug-correction, max-rounds, or partial failures: do the follow-up call.
        if _slug_correction or tool_round + 1 >= max_tool_rounds:
            follow_instruction = _slug_correction + (
                "Present your final answer now using the results above. Do NOT call more tools."
            )
        else:
            follow_instruction = (
                "Continue with your task. You may call more tools or provide your final answer."
            )
        current_raw = await worker.think(
            f"Tool execution results (round {tool_round + 1}):\n\n"
            f"{feedback_text}\n\n{follow_instruction}",
            shared_ctx,
        )
        follow_result, follow_updates = extract_share(current_raw)
        shared_ctx = merge_share(shared_ctx, follow_updates, worker_name)
        result = follow_result if follow_result.strip() else result

    # ── NOT_NEEDED guard — reject when tool calls have failed ────────────────
    # NOT_NEEDED is valid only when the worker genuinely has nothing to do
    # (another worker already covered it).  If the worker attempted tools and
    # they failed, returning NOT_NEEDED hides the failure.  Force a recovery
    # pass so the run report reflects reality.
    _had_integ_failures = any(
        " \u2717]" in p.split("\n")[0]  # "[SLUG ✗]" header
        for p in all_integ_parts
    )
    if result.strip().upper() == "NOT_NEEDED" and _had_integ_failures:
        _fail_count = sum(1 for p in all_integ_parts if " \u2717]" in p.split("\n")[0])
        logger.warning(
            "[%s] NOT_NEEDED rejected — %d tool failure(s) detected, running recovery",
            worker_name,
            _fail_count,
        )
        try:
            _recovery_raw = await worker.think(
                f"{_fail_count} of your tool call(s) failed (see results above). "
                "Returning NOT_NEEDED is invalid when tool calls were attempted. "
                "Describe what failed, identify the correct slug from your TOOL CATALOG, "
                "and summarise the task outcome honestly.",
                shared_ctx,
            )
            _recovery_clean, _rec_updates = extract_share(_recovery_raw)
            shared_ctx = merge_share(shared_ctx, _rec_updates, worker_name)
            if _recovery_clean.strip().upper() != "NOT_NEEDED":
                result = _recovery_clean.strip()
        except Exception as _rec_exc:
            logger.debug("[%s] NOT_NEEDED recovery skipped (non-fatal): %s", worker_name, _rec_exc)
        if result.strip().upper() == "NOT_NEEDED":
            # Worker insisted — convert to explicit failure description
            result = (
                "INCOMPLETE: Tool calls failed and the task could not be completed.\n\n"
                "--- Integration Results ---\n" + "\n\n".join(all_integ_parts)
            )
            _quality_flagged = True

    # ── Self-validation and correction (B4: axis-targeted per task type) ──
    # Workers review their own output and can revise once.
    # Skipped for:
    #  - NOT_NEEDED skips
    #  - Pure tool/integration tasks (tool success is ground truth)
    #  - Tasks with successful code execution (output is objective ground truth;
    #    skipping saves ~2000 tokens and avoids a redundant re-run)
    not_needed_check = result.strip().upper() == "NOT_NEEDED"
    _had_successful_code = bool(all_code_parts) and not any(
        " \u2717]" in p.split("\n")[0] for p in all_code_parts
    )
    if not not_needed_check and not has_tools and not _had_successful_code and result.strip():
        try:
            # B4: choose self-check prompt based on inferred task type
            _self_check_prompt = _build_self_check_prompt(task.get("description", ""))
            val_raw = await worker.think(
                _self_check_prompt,
                shared_ctx,
            )
            if val_raw.strip().upper().startswith("REVISION:"):
                first_line, _, remainder = val_raw.strip().partition("\n")
                note = first_line[len("REVISION:") :].strip()
                corrected = remainder.strip()
                if corrected:
                    rev_result, rev_updates = extract_share(corrected)
                    shared_ctx = merge_share(shared_ctx, rev_updates, worker_name)
                    result = rev_result if rev_result.strip() else corrected
                    logger.info("[%s] self-corrected: %s", worker_name, note[:100])
        except Exception as _sc_exc:
            logger.debug("[%s] self-correction skipped (non-fatal): %s", worker_name, _sc_exc)

    # Append execution output to final result.
    # When _had_break_after_follow: code succeeded and we skipped the follow-up
    # LLM call, so append just the actual execution output (no code block — it's
    # already in the worker's round-1 prose). This avoids duplicating the code.
    # Otherwise: append the full code+output block for failed/partial runs.
    if _had_break_after_follow and all_code_outputs:
        result = f"{result}\n\nOutput:\n" + "\n".join(all_code_outputs)
    elif all_code_parts and not _had_break_after_follow:
        result = f"{result}\n\n--- Code Execution Results ---\n" + "\n\n".join(all_code_parts)
    if all_integ_parts:
        result = f"{result}\n\n--- Integration Results ---\n" + "\n\n".join(all_integ_parts)

    # ── Phase 5: Output quality gate ─────────────────────────────────
    # Validate prose results before accepting. Skip for: tool tasks (quality
    # is determined by tool success), NOT_NEEDED self-skips, and empty results
    # that already have execution parts (code/integration output appended above).
    _quality_flagged = False
    _is_not_needed = result.strip().upper() == "NOT_NEEDED"
    if not has_tools and not _is_not_needed and result.strip():
        _valid, _reason = validate_worker_output(result, task["description"])
        if not _valid:
            logger.warning("[%s] quality gate FAILED (%s) — retrying once", worker_name, _reason)
            try:
                _ca_hint = ""
                if "workspace" in shared_ctx or "access" in _reason or "refusal" in _reason:
                    _ca_hint = (
                        f"\nIf your task involves existing code, emit:\n"
                        f"CODE_ANALYSIS: {shared_ctx.get('workspace', '/path/to/project')}\n"
                        "A safe structural summary will be injected so you can proceed.\n"
                    )
                _retry_raw = await worker.think(
                    f"Your previous response was rejected because: {_reason}.\n"
                    f"{_ca_hint}"
                    "Please provide a complete, substantive answer to your assigned task. "
                    "Do not refuse or give a generic template — use CODE_ANALYSIS: if you need file access.",
                    shared_ctx,
                )
                _retry_result, _retry_updates = extract_share(_retry_raw)
                shared_ctx = merge_share(shared_ctx, _retry_updates, worker_name)
                _retry_result = _retry_result if _retry_result.strip() else _retry_raw.strip()
                _retry_valid, _ = validate_worker_output(_retry_result, task["description"])
                if _retry_valid:
                    result = _retry_result
                    logger.info("[%s] quality gate PASSED after retry", worker_name)
                else:
                    # Accept the retry anyway — one attempt is the limit
                    result = _retry_result or result
                    _quality_flagged = True
                    logger.warning(
                        "[%s] quality gate still FAILED after retry — accepting with flag",
                        worker_name,
                    )
            except Exception as _qg_exc:
                _quality_flagged = True
                logger.debug(
                    "[%s] quality gate retry skipped (non-fatal): %s", worker_name, _qg_exc
                )

    # ── FINDINGS: structured facts for downstream workers ───────────
    result, findings_facts = extract_findings(result)

    # If the worker replied with ONLY a FINDINGS block and no prose, synthesise
    # a brief user-facing summary from the facts rather than passing raw protocol
    # markup to master_respond (which would leak FINDINGS: into the final answer).
    if not result.strip() and findings_facts:
        result = "\n".join(f"- {f}" for f in findings_facts)
        logger.info(
            "[%s] worker returned only FINDINGS — synthesised prose from %d fact(s)",
            worker_name,
            len(findings_facts),
        )

    # ── REMEMBER: persist cross-run facts to global LTM ─────────────
    result, remember_facts = extract_remember(result)
    if bridge and remember_facts:
        bridge.store_remember_facts(worker_name, remember_facts)
        logger.debug(
            "[%s] stored %d REMEMBER: fact(s) to global memory", worker_name, len(remember_facts)
        )

    not_needed = result.strip().upper() == "NOT_NEEDED"
    result_entry: dict[str, Any] = {
        "task_id": tid,
        "agent": worker_name,
        "model": chosen_model,
        "description": task["description"],
        "result": result,
        "skipped": not_needed,
        "quality_flagged": _quality_flagged,
        "findings": findings_facts,
    }
    completed[tid] = result_entry

    # Phase 5: emit worker_complete event for streaming
    if event_callback is not None:
        with contextlib.suppress(Exception):
            event_callback(
                {
                    "type": "worker_complete",
                    "worker": worker_name,
                    "task": task["description"][:120],
                    "skipped": not_needed,
                    "quality_flagged": _quality_flagged,
                    "result_preview": result[:200] if result and not not_needed else "",
                }
            )

    # Return net context changes so parallel workers don't clobber each other
    ctx_delta = {k: v for k, v in shared_ctx.items() if shared_ctx_in.get(k) != v}
    return result_entry, ctx_delta


def make_execution_lead(
    lead: PipelineAgent,
    worker_cfg: dict[str, Any],
    models: list[dict[str, Any]],
    mode: str,
    registry: AgentRegistry,
    bridge: MemoryBridge | None = None,
    event_callback: Any | None = None,
    budget_tracker: BudgetTracker | None = None,
    researcher: PipelineAgent | None = None,
    code_processor: Any | None = None,
):
    async def execution_lead(state: DynamicPipelineState) -> dict[str, Any]:
        shared_ctx = dict(state.get("shared_context", {}))
        tasks = state.get("tasks") or [{"id": "task_1", "description": state["user_query"]}]
        research = state.get("research_summary", "")

        # ── Planning call — skipped on resume after a worker QUESTION interrupt ─
        # LangGraph re-runs the entire node after interrupt() resume, so we cache
        # the plan on the lead agent object (same Python instance in the closure).
        # Cache key = task content, so a new request with different tasks re-plans.
        cache_key = tuple((t["id"], t["description"]) for t in tasks)
        cached_plan = getattr(lead, "_plan_cache", None)
        if cached_plan and cached_plan["key"] == cache_key:
            logger.info("Execution lead: resuming (plan cached)")
            worker_prompts: dict[str, str] = cached_plan["worker_prompts"]
            ability_reqs: dict[str, dict[str, float]] = cached_plan["ability_reqs"]
        else:
            worker_prompts = {}
            ability_reqs = {}
            toolkit_assignments: dict[str, list[str]] = {}
            tool_slug_assignments: dict[str, list[str]] = {}
            dependencies_map: dict[str, list[str]] = {}

            # ── Single-task fast-path: focused prompt instead of full dispatch plan ─
            # When there is exactly one task, we still call the lead LLM but with a
            # minimal prompt: "write one optimised worker instruction for this task".
            # This costs ~250 tokens (vs ~1000 for a full plan) and guarantees the
            # worker receives a properly crafted prompt rather than the raw analyser
            # task description. Quality is measurably better for code and research tasks.
            _is_single_task = len(tasks) == 1
            if _is_single_task:
                _tid = tasks[0]["id"]
                _task_desc = tasks[0]["description"]
                _research_hint = (
                    f"\n\nResearch context to pass to worker:\n{research[:400]}" if research else ""
                )
                _single_prompt_raw = await lead.think(
                    f"Write a single optimised worker instruction for this task. "
                    f"Plain prose only — no JSON, no structure.\n\n"
                    f"Task: {_task_desc}{_research_hint}",
                    shared_ctx,
                )
                _single_prompt, _sp_updates = extract_share(_single_prompt_raw)
                shared_ctx = merge_share(shared_ctx, _sp_updates, lead.name)
                worker_prompts = {_tid: _single_prompt.strip() or _task_desc}
                ability_reqs = {_tid: {a: 1.5 for a in ABILITIES}}
                # Suppress integration block unless the task explicitly references a service
                _integ_kw: frozenset[str] = frozenset(
                    {
                        "github",
                        "gitlab",
                        "jira",
                        "linear",
                        "notion",
                        "slack",
                        "confluence",
                        "repo",
                        "repository",
                        "issue",
                        "ticket",
                        "pull request",
                        "merge request",
                        "pr ",
                        "send message",
                        "create issue",
                        "list repos",
                        "create repo",
                    }
                )
                if not any(kw in _task_desc.lower() for kw in _integ_kw):
                    toolkit_assignments = {_tid: []}
                logger.info("Execution lead: single-task — used focused prompt call (~250 tokens)")
            else:
                tasks_list = "\n".join(f"  {t['id']}: {t['description']}" for t in tasks)
                research_block = (
                    f"\nResearch summary (share with workers):\n{research[:500]}"
                    if research
                    else ""
                )

                # The lead sees connected toolkit names via shared_ctx["connected_integrations"].
                # It assigns toolkits per task; workers receive the full filtered catalog.
                # No catalog injection into the lead prompt — keeps the lead as a lean planner.
                plan_raw = await lead.think(
                    f"Plan worker dispatch:{research_block}\n\n{tasks_list}\n\n"
                    "JSON only — no prose:\n"
                    '{"plan": [{"id": "task_1", "reasoning": 0-3, "coding": 0-3, '
                    '"knowledge": 0-3, "writing": 0-3, "math": 0-3, '
                    '"instruction_following": 0-3, "tool_use": 0-3, '
                    '"speed": 0-3, "cost_efficiency": 0-3, "context_handling": 0-3, '
                    '"toolkits": ["only_if_needed"], '
                    '"dependencies": ["task_ids_this_depends_on"]}],\n'
                    ' "prompts": {"task_1": "<concise worker instruction>"}}',
                    shared_ctx,
                )

                try:
                    parsed_plan = parse_json_robust(plan_raw)
                    for row in parsed_plan.get("plan", []):
                        tid = str(row.get("id", ""))
                        ability_reqs[tid] = {a: float(row.get(a, 0.0)) for a in ABILITIES}
                        raw_tks = row.get("toolkits", [])
                        if isinstance(raw_tks, list):
                            toolkit_assignments[tid] = [str(t) for t in raw_tks]
                        # Extract exact slug assignments (new — takes priority over toolkits)
                        raw_slugs = row.get("tool_slugs", [])
                        if isinstance(raw_slugs, list):
                            tool_slug_assignments[tid] = [str(s) for s in raw_slugs if s]
                        # Extract task dependencies for wave-based parallel scheduling
                        raw_deps = row.get("dependencies", [])
                        if isinstance(raw_deps, list):
                            deps = [str(d) for d in raw_deps if str(d) != tid]
                            if deps:
                                dependencies_map[tid] = deps
                    worker_prompts = {
                        str(k): str(v) for k, v in parsed_plan.get("prompts", {}).items()
                    }
                except ValueError as e:
                    logger.warning("Execution lead plan parse failed: %s", e)

            # ── Auto-resolve tool slugs for toolkit assignments ──────────────
            # When the lead assigns a toolkit but no specific slugs, we
            # deterministically select the top-K slugs by keyword scoring.
            # This narrows the worker's catalog from ~58 tools to ~6, making
            # slug hallucination almost impossible.
            try:
                from aifactory.core.agent_factory import resolve_slugs_for_task as _resolve_slugs

                for _tid, _tks in toolkit_assignments.items():
                    if _tks and not tool_slug_assignments.get(_tid):
                        _task_desc = next((t["description"] for t in tasks if t["id"] == _tid), "")
                        _auto_slugs = _resolve_slugs(_task_desc, _tks)
                        if _auto_slugs:
                            tool_slug_assignments[_tid] = _auto_slugs
                            logger.info(
                                "[execution_lead] auto-resolved slugs for %s (toolkit=%s): %s",
                                _tid,
                                _tks,
                                _auto_slugs,
                            )
            except Exception as _sr_exc:
                logger.debug("Slug auto-resolve skipped (non-fatal): %s", _sr_exc)

            logger.info(
                "Execution lead assignments: toolkits=%s slugs=%s deps=%s",
                {tid: tks for tid, tks in toolkit_assignments.items()},
                {tid: sl for tid, sl in tool_slug_assignments.items()},
                dependencies_map or "none",
            )

            lead._plan_cache = {
                "key": cache_key,
                "worker_prompts": worker_prompts,
                "ability_reqs": ability_reqs,
                "toolkit_assignments": toolkit_assignments,
                "tool_slug_assignments": tool_slug_assignments,
                "dependencies_map": dependencies_map,
                "completed": {},  # tid -> result dict, persists across interrupt resumes
            }

        completed: dict[str, dict[str, Any]] = lead._plan_cache["completed"]
        workers_cache: dict[str, PipelineAgent] = lead._plan_cache.setdefault("workers", {})
        toolkit_assignments: dict[str, list[str]] = lead._plan_cache.get("toolkit_assignments", {})
        tool_slug_assignments: dict[str, list[str]] = lead._plan_cache.get(
            "tool_slug_assignments", {}
        )
        dependencies_map: dict[str, list[str]] = lead._plan_cache.get("dependencies_map", {})

        # ── Wave-based parallel worker dispatch ──────────────────────
        # Tasks with no dependency on each other run concurrently via asyncio.gather().
        # Tasks that depend on earlier tasks run in subsequent waves, receiving those
        # results as prior_results.  Interrupt-resume safety: tasks that already
        # completed before an interrupt are seeded into task_results and skipped.
        task_results: list[dict[str, Any]] = []

        # Seed results from tasks completed before an interrupt resume
        for task in tasks:
            if task["id"] in completed:
                entry = completed[task["id"]]
                if not entry.get("skipped"):
                    task_results.append(entry)
                logger.info("[Worker-%s] already completed (pre-interrupt) — skipping", task["id"])

        # Build dependency waves for tasks still pending
        pending = [t for t in tasks if t["id"] not in completed]
        waves = _build_dependency_waves(pending, dependencies_map) if pending else []

        # D1: pass image attachments from state down to workers
        _attachments: list[str] = state.get("attachments") or []
        if _attachments:
            logger.info(
                "Execution lead: %d image attachment(s) will be forwarded to vision workers",
                len(_attachments),
            )

        # Shared structural index cache — built once per unique path per run,
        # reused by all subsequent workers that emit CODE_ANALYSIS: for the same path.
        # The accompanying lock serialises concurrent build attempts so only one
        # coroutine reads the filesystem and calls process_codebase(); all others
        # wait and then get a cache hit.  Saves ~3k tokens per extra code worker.
        _structural_index_cache: dict[str, str] = {}
        _structural_index_lock: asyncio.Lock = asyncio.Lock()

        # Adaptive Mesh W4: Worker Auction — run per pipeline (not per task) to set
        # the quality floor that WaveGovernor will enforce.  Generates default bids
        # from keyword signals in the first task description; stores the floor in
        # shared_ctx["auction_floor"] for WaveGovernor to consume.
        try:
            from aifactory.core.worker_auction import WorkerAuction, bids_for_task

            _auction = WorkerAuction()
            _first_task_desc = tasks[0]["description"] if tasks else ""
            _model_ids = [m.get("id", "") for m in models if isinstance(m, dict)]
            _bids = bids_for_task(_first_task_desc, _model_ids, mode=mode)
            if _bids:
                from aifactory.core.rl_scorer import infer_task_type as _infer_tt

                _auction_result = _auction.run(
                    task_description=_first_task_desc,
                    bids=_bids,
                    task_type=_infer_tt(_first_task_desc),
                )
                _auction_ctx = _auction_result.to_shared_ctx_update()
                shared_ctx = {**shared_ctx, **_auction_ctx}
                logger.debug(
                    "Auction: winner=%s floor=%.2f",
                    _auction_result.winner.worker_id,
                    _auction_result.second_price_floor,
                )
        except Exception as _auction_exc:
            logger.debug("WorkerAuction skipped: %s", _auction_exc)

        # Shared kwargs passed to every _run_worker_task call
        _worker_kwargs: dict[str, Any] = dict(
            worker_prompts=worker_prompts,
            ability_reqs=ability_reqs,
            toolkit_assignments=toolkit_assignments,
            tool_slug_assignments=tool_slug_assignments,
            completed=completed,
            workers_cache=workers_cache,
            mode=mode,
            registry=registry,
            bridge=bridge,
            event_callback=event_callback,
            budget_tracker=budget_tracker,
            attachments=_attachments or None,
            researcher=researcher,
            code_processor=code_processor,
            structural_index_cache=_structural_index_cache,
            structural_index_lock=_structural_index_lock,
        )

        # FINDINGS: accumulated facts from completed workers (injected into subsequent waves)
        findings_by_task: dict[str, list[str]] = {}

        # Capture the active OTel context once so all worker spans (across
        # asyncio.gather waves) are re-parented to the execution_lead span.
        # Without this, opentelemetry.context inside each coroutine starts
        # fresh and the trace tree breaks.  Zero-cost when OTel is disabled.
        from aifactory.core.otel import capture_current_context as _capture_otel_ctx

        _otel_parent_ctx = _capture_otel_ctx()

        _total_waves = len(waves)
        for wave_idx, wave_tasks in enumerate(waves):
            logger.info(
                "Execution wave %d/%d: tasks=%s",
                wave_idx + 1,
                len(waves),
                [t["id"] for t in wave_tasks],
            )
            # Emit wave_start so the CLI can show "Wave 1/3 — 2 parallel tasks"
            if event_callback is not None:
                with contextlib.suppress(Exception):
                    event_callback(
                        {
                            "type": "wave_start",
                            "wave": wave_idx + 1,
                            "total_waves": _total_waves,
                            "task_count": len(wave_tasks),
                            "parallel": len(wave_tasks) > 1,
                            "task_ids": [t["id"] for t in wave_tasks],
                        }
                    )
            # Each wave sees results of all previously completed waves
            prior_results_snapshot = list(task_results)

            # Compute compact findings context from all prior completed workers
            _findings_ctx = format_findings_context(findings_by_task)

            if len(wave_tasks) == 1:
                # Single task — run directly to preserve interrupt() semantics cleanly
                # (no gather wrapper, so GraphInterrupt propagates without wrapping)
                entry, ctx_delta = await _run_worker_task(
                    wave_tasks[0],
                    shared_ctx,
                    prior_results=prior_results_snapshot,
                    findings_context=_findings_ctx,
                    otel_parent_context=_otel_parent_ctx,
                    wave_index=wave_idx,
                    **_worker_kwargs,
                )
                shared_ctx = {**shared_ctx, **ctx_delta}
                if not entry["skipped"]:
                    task_results.append(entry)
                # Collect findings for subsequent waves
                if entry.get("findings"):
                    findings_by_task[entry["task_id"]] = entry["findings"]

            else:
                # Multiple independent tasks — run in parallel.
                # Each worker gets a snapshot of shared_ctx at wave start so parallel
                # workers don't clobber each other's context mid-execution.
                # After the wave, ctx_deltas are merged back sequentially.
                snapshot = dict(shared_ctx)
                coros = [
                    _run_worker_task(
                        task,
                        snapshot,
                        prior_results=prior_results_snapshot,
                        findings_context=_findings_ctx,
                        otel_parent_context=_otel_parent_ctx,
                        wave_index=wave_idx,
                        **_worker_kwargs,
                    )
                    for task in wave_tasks
                ]
                wave_outcomes = await asyncio.gather(*coros, return_exceptions=True)

                interrupt_exc: BaseException | None = None
                for task, outcome in zip(wave_tasks, wave_outcomes, strict=False):
                    if isinstance(outcome, BaseException):
                        # GraphInterrupt must propagate so LangGraph can checkpoint.
                        # Save all successfully completed wave peers first, then re-raise.
                        is_graph_interrupt = (
                            bool(_GRAPH_INTERRUPT_TYPES)
                            and isinstance(outcome, _GRAPH_INTERRUPT_TYPES)
                        ) or "Interrupt" in type(outcome).__name__
                        if is_graph_interrupt:
                            logger.info(
                                "Worker %s raised interrupt — saving wave peers then re-raising",
                                task["id"],
                            )
                            interrupt_exc = outcome
                        elif isinstance(outcome, BudgetExceededError):
                            # Budget exceeded — stop all remaining workers, return partial results
                            logger.error(
                                "Budget cap exceeded at Worker %s: %s", task["id"], outcome
                            )
                            raise outcome
                        else:
                            logger.error(
                                "Worker %s raised unexpected exception: %s", task["id"], outcome
                            )
                            error_entry: dict[str, Any] = {
                                "task_id": task["id"],
                                "agent": f"Worker-{task['id']}",
                                "model": "unknown",
                                "description": task["description"],
                                "result": f"[Worker error: {outcome!r}]",
                                "skipped": False,
                                "quality_flagged": False,
                            }
                            completed[task["id"]] = error_entry
                            task_results.append(error_entry)
                    else:
                        entry, ctx_delta = outcome
                        shared_ctx = {**shared_ctx, **ctx_delta}
                        if not entry["skipped"]:
                            task_results.append(entry)
                        # Collect findings for subsequent waves
                        if entry.get("findings"):
                            findings_by_task[entry["task_id"]] = entry["findings"]

                if interrupt_exc is not None:
                    raise interrupt_exc

            # ── C2: Adaptive worker spawning — revision pass ──────────────
            # After each wave completes, check for quality-flagged results.
            # If budget allows, spawn a targeted revision worker for each.
            _flagged_entries = [e for e in task_results if e.get("quality_flagged")]
            if _flagged_entries:
                _remaining_budget = budget_tracker.remaining_usd if budget_tracker else None
                _budget_ok = _remaining_budget is None or _remaining_budget > 0.10
                if _budget_ok:
                    logger.info(
                        "Adaptive spawn: %d quality-flagged task(s) → revision workers",
                        len(_flagged_entries),
                    )
                    for _flagged in _flagged_entries:
                        _rev_task = {
                            "id": f"{_flagged['task_id']}_rev",
                            "description": (
                                f"REVISION NEEDED for: {_flagged['description']}\n\n"
                                f"Previous attempt was flagged as low quality. "
                                f"Produce a complete, high-quality answer. "
                                f"Prior attempt (for reference only): "
                                f"{_flagged['result'][:300]}"
                            ),
                            "dependencies": [_flagged["task_id"]],
                        }
                        try:
                            _rev_entry, _rev_delta = await _run_worker_task(
                                _rev_task,
                                shared_ctx,
                                otel_parent_context=_otel_parent_ctx,
                                wave_index=wave_idx,
                                **_worker_kwargs,
                            )
                            if not _rev_entry["skipped"] and not _rev_entry.get("quality_flagged"):
                                # Replace original flagged result with revision
                                for i, e in enumerate(task_results):
                                    if e["task_id"] == _flagged["task_id"]:
                                        task_results[i] = {
                                            **_rev_entry,
                                            "task_id": _flagged["task_id"],
                                        }
                                        break
                                shared_ctx = {**shared_ctx, **_rev_delta}
                                logger.info(
                                    "Adaptive spawn: revision for %s accepted",
                                    _flagged["task_id"],
                                )
                        except Exception as _rev_exc:
                            logger.debug(
                                "Revision worker for %s failed (non-fatal): %s",
                                _flagged["task_id"],
                                _rev_exc,
                            )

        # ── Consolidation ────────────────────────────────────────────
        # Skip consolidation LLM call when there's only 1 worker — pass through directly
        if len(task_results) == 1:
            clean_consolidated = task_results[0]["result"]
        else:
            # Truncate each result to limit consolidation input tokens
            max_chars = 600
            results_text = "\n\n".join(
                f"### {r['task_id']}: {r['description']}\n"
                f"{r['result'][:max_chars]}{'…' if len(r['result']) > max_chars else ''}"
                for r in task_results
            )
            consolidated = await lead.think(
                f"Consolidate into one coherent answer. Plain text only.\n\n{results_text}",
                shared_ctx,
            )
            clean_consolidated, updates = extract_share(consolidated)
            shared_ctx = merge_share(shared_ctx, updates, lead.name)
            # Phase 2.1: persist consolidation SHARE: updates
            if bridge and updates:
                bridge.store_share_updates(lead.name, updates)

        # Phase 3.5: prune shared_context before returning from execution_lead
        try:
            from aifactory.config.settings import get_settings as _gs

            _max_ctx_keys = _gs().shared_context_max_keys
        except Exception:
            _max_ctx_keys = 25
        shared_ctx = _prune_shared_context(shared_ctx, _max_ctx_keys)

        return {
            "task_results": task_results,
            "shared_context": shared_ctx,
            "final_response": clean_consolidated,
        }

    return execution_lead


def _synthesize_judge_metrics(
    registry: Any,
    task_results: list[dict],
    total_tokens: int,
) -> tuple[list[dict], dict]:
    """
    Build assessments + metrics from run metadata without an LLM call.

    Used by (a) the smart-judge fast path for single-task simple runs and
    (b) the fallback when the real LLM judge fails or returns unparseable JSON.
    Guarantees a non-empty metrics dict with composite_quality so quality is
    always measured — silent judge_metrics={} loses the RL signal and the
    user-facing quality gate.
    """
    _tool_fail_count = sum(
        1
        for r in task_results
        if " ✗]" in r.get("result", "") or r.get("quality_flagged")
    )
    _all_no_work = bool(task_results) and all(
        r.get("skipped") or r.get("result", "").startswith("INCOMPLETE") for r in task_results
    )

    if _tool_fail_count > 0 and _tool_fail_count >= len(task_results):
        _synth_status, _synth_eff, _synth_cost = "failed", "low", "1"
        _synth_rec = (
            f"{_tool_fail_count} tool failure(s) — check slug names and "
            "API connectivity. See integration results for details."
        )
    elif _tool_fail_count > 0:
        _synth_status, _synth_eff, _synth_cost = "partial", "medium", "2"
        _synth_rec = f"{_tool_fail_count} of {len(task_results)} task(s) had tool failures."
    elif _all_no_work:
        _synth_status, _synth_eff, _synth_cost = "partial", "medium", "2"
        _synth_rec = "All workers self-skipped without completing assigned work."
    else:
        _quality_flagged_count = sum(1 for r in task_results if r.get("quality_flagged"))
        if _quality_flagged_count > 0:
            _synth_status, _synth_eff, _synth_cost = "partial", "medium", "2"
            _synth_rec = (
                f"{_quality_flagged_count} worker(s) produced low-quality or evasive output "
                f"(quality gate flagged). Use CODE_ANALYSIS: for codebase tasks."
            )
        else:
            _synth_status, _synth_eff, _synth_cost = "successful", "high", "3"
            _synth_rec = "Auto-assessed run — no improvements identified."

    assessments: list[dict] = []
    for a in registry.all():
        _agent_status = "successful" if (a.total_tokens > 0 or a._actions) else "not_required"
        _matching_result = next((r for r in task_results if a.name in r.get("task_id", "")), None)
        _agent_quality = (
            "1" if (_matching_result and _matching_result.get("quality_flagged")) else "2"
        )
        assessments.append(
            {
                "agent": a.name,
                "status": _agent_status,
                "quality": _agent_quality,
                "tokens": str(a.total_tokens),
                "notes": "auto-assessed",
            }
        )

    _status_score = {"successful": 10, "partial": 5, "failed": 2}.get(_synth_status, 5)
    _cost_score = {"3": 10, "2": 7, "1": 4, "0": 2}.get(_synth_cost, 5)
    _flagged_penalty = sum(1 for r in task_results if r.get("quality_flagged")) * 2.0
    _composite = round(max(0.0, (_status_score * 0.7 + _cost_score * 0.3) - _flagged_penalty), 1)
    metrics = {
        "overall_status": _synth_status,
        "total_tokens": str(total_tokens),
        "pipeline_efficiency": _synth_eff,
        "cost_rating": _synth_cost,
        "composite_quality": str(_composite),
        "recommendations": _synth_rec,
    }
    return assessments, metrics


def make_judge_review(
    agent: PipelineAgent,
    registry: AgentRegistry,
    run_id: str = "unknown",
    lite_mode: bool = False,
    event_callback: Any | None = None,
):
    async def judge_review(state: DynamicPipelineState) -> dict[str, Any]:
        shared_ctx = state.get("shared_context", {})

        # S3: Lite mode — return synthetic metrics, skip LLM call (~700-1000 tokens saved)
        if lite_mode:
            task_results_lite = state.get("task_results", [])
            total_tokens_lite = sum(a.total_tokens for a in registry.all())
            n_completed = sum(1 for r in task_results_lite if not r.get("skipped"))
            return {
                "judge_metrics": {
                    "overall_status": "done" if n_completed > 0 else "partial",
                    "total_tokens": total_tokens_lite,
                    "pipeline_efficiency": "lite",
                    "cost_rating": 3,
                    "recommendations": "Lite mode — judge_review skipped.",
                    "composite_quality": 7.0,
                },
            }

        # Build run summary for judge (include elapsed times)
        agent_rows = "\n".join(
            f"{a.name},{a.cfg.get('role', a.name)},{len(a._actions)},{a.total_tokens},{_elapsed(a)}"
            for a in registry.all()
        )
        task_results = state.get("task_results", [])
        task_rows = (
            "\n".join(
                f"  {r['task_id']}: {r['description'][:70]}"
                f" | {len(r['result'])} chars | model: {r['model']}"
                for r in task_results
            )
            or "  none"
        )
        total_tokens = sum(a.total_tokens for a in registry.all())

        run_summary = (
            f"agents ({len(registry.all())}):\n{agent_rows}\n\n"
            f"tasks_completed: {len(task_results)}\n"
            f"task_detail:\n{task_rows}\n\n"
            f"research_for_analysis_ran: {bool(state.get('analysis_research_summary'))}\n"
            f"execution_research_ran: {bool(state.get('research_summary'))}\n"
            f"final_output_chars: {len(state.get('final_response', ''))}\n"
            f"total_tokens: {total_tokens}"
        )

        # Phase 3.5: smart judge — skip LLM call for single-task simple runs.
        # A synthetic assessment is generated from run metadata (tokens, models, etc.)
        # saving ~800-1200 tokens per simple query. The report format is identical.
        try:
            from aifactory.config.settings import get_settings as _gs

            _skip_simple = _gs().judge_skip_simple_runs
        except Exception:
            _skip_simple = True

        _research_ran = bool(state.get("research_summary")) or bool(
            state.get("analysis_research_summary")
        )
        # Never treat audit/scan/review requests as simple — they need real LLM evaluation
        # to catch groundedness failures (generic output instead of file:function citations).
        _audit_keywords = ("audit", "scan", "review", "flaw", "bug", "issue", "vuln", "refactor")
        _query_lower = state.get("user_query", "").lower()
        _is_audit_query = any(kw in _query_lower for kw in _audit_keywords)
        _is_simple_run = (
            _skip_simple and len(task_results) <= 1 and not _research_ran and not _is_audit_query
        )

        assessments: list[dict] = []
        metrics: dict = {}

        if _is_simple_run:
            logger.info(
                "Judge: simple run (%d task(s)) — using synthetic assessment", len(task_results)
            )
            # Inspect actual task outcomes before issuing verdict — a synthetic
            # "successful" was wrong when all tools 404'd, so metrics reflect reality
            # (tool failures, no-work, quality flags) for honest RL signal.
            assessments, metrics = _synthesize_judge_metrics(registry, task_results, total_tokens)
        else:
            agent_count = len(registry.all())
            # Derive success criteria from the query itself — judge_brief no longer
            # pre-seeds this, so the judge bootstraps criteria inline.
            raw = await agent.think(
                f"Evaluate this pipeline run.\n\n"
                f'Request: "{state["user_query"]}"\n\n'
                f"First, infer what a successful response looks like for this request "
                f"(2 sentences max). Then assess the run below against those criteria.\n\n"
                f"{run_summary}\n\n"
                f"Output a JSON object with two keys:\n"
                f"- assessments: array of {agent_count} objects, one per agent, each with: "
                f"agent (str), status (successful|unsuccessful|not_required), quality (0-3), "
                f"tokens (int), notes (str)\n"
                f"- metrics: object with: overall_status "
                f"(successful|partial|failed), total_tokens (int), "
                f"pipeline_efficiency (high|medium|low), "
                f"cost_rating (0-3), recommendations (str)",
                shared_ctx,
            )
            try:
                parsed = parse_json_robust(raw)
                assessments = [
                    {
                        "agent": str(r.get("agent", "?")),
                        "status": str(r.get("status", "?")),
                        "quality": str(r.get("quality", "?")),
                        "tokens": str(r.get("tokens", "?")),
                        "notes": str(r.get("notes", "")),
                    }
                    for r in parsed.get("assessments", [])
                ]
                raw_metrics = parsed.get("metrics", {})
                if isinstance(raw_metrics, dict):
                    metrics = {k: str(v) for k, v in raw_metrics.items()}
                # Derive composite_quality if the LLM didn't include it
                if "composite_quality" not in metrics:
                    _st = metrics.get("overall_status", "partial")
                    _cr = metrics.get("cost_rating", "2")
                    _ss = {"successful": 10, "partial": 5, "failed": 2}.get(_st, 5)
                    _cs = {"3": 10, "2": 7, "1": 4, "0": 2}.get(str(_cr), 5)
                    metrics["composite_quality"] = str(round(_ss * 0.7 + _cs * 0.3, 1))
            except (ValueError, Exception) as e:
                logger.warning("Judge JSON parse failed: %s", e)

            # Fallback: the real judge failed (flaky model, empty output, or
            # unparseable JSON). Never leave metrics empty — a multi-worker run
            # with judge_metrics={} silently loses quality measurement and the
            # RL signal. Synthesize honest metrics from run metadata instead.
            if not metrics:
                logger.warning(
                    "Judge produced no metrics — falling back to synthetic assessment"
                )
                assessments, metrics = _synthesize_judge_metrics(
                    registry, task_results, total_tokens
                )

        # ── Deep output quality evaluation (LLM-as-judge) ────────────
        # When judge_deep_eval_enabled=True, run an extra focused LLM call
        # that reads the *actual* final_response and scores it on 4 dimensions.
        # This produces a truthful composite_quality — not metadata-derived.
        try:
            from aifactory.config.settings import get_settings as _gs_deep

            _deep_eval = _gs_deep().judge_deep_eval_enabled
            _eval_model = _gs_deep().judge_eval_model.strip()
        except Exception:
            _deep_eval, _eval_model = False, ""

        _final_resp = state.get("final_response", "") or state.get("direct_answer", "")
        if _deep_eval and _final_resp.strip():
            try:
                # Auto-select cheapest available model for the eval call
                if not _eval_model:
                    try:
                        from aifactory.core.model_selector import best_model as _bm
                        from aifactory.core.model_selector import load_available_models as _lam

                        _eval_model = (
                            _bm(_lam(), mode="cost_efficient") if load_available_models else ""
                        )
                    except Exception:
                        pass
                    if not _eval_model:
                        _eval_model = "gpt-4o-mini"

                _eval_prompt = (
                    f"You are evaluating an AI-generated response. Score it 0–10 on each dimension.\n\n"  # noqa: E501
                    f"ORIGINAL REQUEST:\n{state['user_query']}\n\n"
                    f"AI RESPONSE (first 2000 chars):\n{_final_resp[:2000]}\n\n"
                    "Output ONLY a JSON object:\n"
                    '{"relevance": 0-10, "accuracy": 0-10, "completeness": 0-10, '
                    '"actionability": 0-10, "summary": "one sentence"}'
                )
                _eval_raw = await agent.think(_eval_prompt, {})
                _eval_parsed = parse_json_robust(_eval_raw)
                _r = float(_eval_parsed.get("relevance", 7))
                _a = float(_eval_parsed.get("accuracy", 7))
                _c = float(_eval_parsed.get("completeness", 7))
                _ac = float(_eval_parsed.get("actionability", 7))
                _deep_score = round((_r * 0.3 + _a * 0.3 + _c * 0.25 + _ac * 0.15), 1)

                metrics["composite_quality"] = str(_deep_score)
                metrics["eval_relevance"] = str(round(_r, 1))
                metrics["eval_accuracy"] = str(round(_a, 1))
                metrics["eval_completeness"] = str(round(_c, 1))
                metrics["eval_actionability"] = str(round(_ac, 1))
                metrics["eval_summary"] = str(_eval_parsed.get("summary", ""))
                logger.info(
                    "Judge deep eval: relevance=%.1f accuracy=%.1f completeness=%.1f "
                    "actionability=%.1f → composite=%.1f",
                    _r,
                    _a,
                    _c,
                    _ac,
                    _deep_score,
                )
            except Exception as _de_exc:
                logger.debug("Judge deep eval failed (non-fatal): %s", _de_exc)

        # ── Build hierarchical plain-text report ─────────────────────
        _pricing = {m["id"]: m for m in load_models()}

        def _fmt_model(m: str) -> str:
            m = m.removeprefix("openai/")
            return m[-36:] if len(m) > 36 else m

        def _status_sym(s: str) -> str:
            return {"successful": "✓", "unsuccessful": "✗", "not_required": "—"}.get(s, "—")

        def _agent_cost(ag: PipelineAgent) -> str:
            model_id = ag.model.removeprefix("openai/")
            p = _pricing.get(model_id, {})
            inp = p.get("input_cost_per_1m")
            out = p.get("output_cost_per_1m")
            if inp is None or out is None:
                return "—"
            cost = (ag.prompt_tokens * inp + ag.completion_tokens * out) / 1_000_000
            return f"${cost:.4f}"

        col_w = 82
        sep = "═" * col_w
        line = "─" * col_w
        assess_map = {a["agent"]: a for a in assessments}
        ag_map = {a.name: a for a in registry.all()}

        def _agent_row(name: str, indent: str = "") -> list[str]:
            ag = ag_map.get(name)
            asmnt = assess_map.get(name, {})
            status = asmnt.get("status", "—")
            quality = "—" if status == "not_required" else asmnt.get("quality", "-")
            tokens = ag.total_tokens if ag else 0
            elapsed_str = _elapsed(ag) if ag else "—"
            model = _fmt_model(ag.model) if ag else "—"
            cost = _agent_cost(ag) if ag else "—"
            sym = _status_sym(status)
            label = f"{indent}{name}"
            return [
                f"  {label:<28} {model:<36} {tokens:>5} tok"
                f"  {elapsed_str:>6}  {cost:>9}  {sym:<12} {quality}"
            ]

        report_lines: list[str] = [
            sep,
            f"  RUN    {run_id}",
            f"  QUERY  {state.get('user_query', '')}",
            sep,
            "",
            "  EXECUTION PIPELINE",
            line,
            f"  {'agent':<28} {'model':<36} {'tokens':>9}"
            f"  {'time':>6}  {'cost':>9}  {'status':<12} quality",
            line,
        ]

        named_order = ["master", "judge", "analyser", "researcher", "execution_lead"]
        for name in named_order:
            if name not in ag_map:
                asmnt = assess_map.get(name, {})
                status = asmnt.get("status", "not_required")
                sym = _status_sym(status)
                report_lines.append(
                    f"  {name:<28} {'—':<36} {'—':>9}  {'—':>6}  {'—':>9}  {sym:<12} —"
                )
            else:
                report_lines.extend(_agent_row(name))
            if name == "execution_lead":
                for r in task_results:
                    wname = r["agent"]
                    report_lines.extend(_agent_row(wname, indent="  └─ "))
                    desc = r["description"][:60]
                    report_lines.append(f"     {'':28} task: {desc}")

        total_cost_known = sum(
            (
                ag.prompt_tokens * _pricing[mid].get("input_cost_per_1m", 0)
                + ag.completion_tokens * _pricing[mid].get("output_cost_per_1m", 0)
            )
            / 1_000_000
            for ag in registry.all()
            for mid in [ag.model.removeprefix("openai/")]
            if mid in _pricing and _pricing[mid].get("input_cost_per_1m") is not None
        )
        total_cost_str = f"${total_cost_known:.4f}" if total_cost_known > 0 else "—"

        report_lines += [
            line,
            f"  {'TOTAL':<28} {'':36} {total_tokens:>5} tok  {'':>6}  {total_cost_str:>9}",
            "",
            "  PIPELINE METRICS",
            line,
        ]
        for k, v in metrics.items():
            report_lines.append(f"  {k:<28} {v}")
        report_lines += ["", sep]

        report = "\n".join(report_lines)

        # ── Write judge dump (plain text report) ─────────────────────
        judge_dir = DUMPS_DIR / "judge"
        judge_dir.mkdir(parents=True, exist_ok=True)
        dump_path = judge_dir / f"{run_id}.txt"
        dump_path.write_text(report, encoding="utf-8")
        logger.info("Judge report saved: %s", dump_path)
        if callable(event_callback):
            event_callback({"type": "report", "text": report})

        # ── RL-lite: record worker performance for adaptive model selection ──
        try:
            from aifactory.core.rl_scorer import record_pipeline_performances

            record_pipeline_performances(
                task_results=task_results,
                assessments=assessments,
                run_id=run_id,
            )
        except Exception as _rl_exc:
            logger.debug("RL scorer record failed (non-fatal): %s", _rl_exc)

        # ── W5: Auction → RL feedback loop ─────────────────────────────────
        # When auction metadata is present in shared_ctx (written by execution_lead),
        # record the actual run quality back to the RLScorer so future auctions
        # can adjust RL-weighted confidences for the winning model.
        try:
            _auction_winner_model = shared_ctx.get("auction_winner_model", "")
            if _auction_winner_model:
                from aifactory.core.rl_scorer import infer_task_type, record_performance

                _first_task_desc = (
                    task_results[0].get("task_description", "") if task_results else ""
                )
                _task_type = infer_task_type(_first_task_desc)
                _composite = assessments.get("composite_quality", 5.0)
                record_performance(
                    model_id=_auction_winner_model,
                    task_type=_task_type,
                    quality_score=float(_composite),
                )
                logger.debug(
                    "W5 auction→RL feedback: model=%s task_type=%s quality=%.1f",
                    _auction_winner_model,
                    _task_type,
                    _composite,
                )
        except Exception as _w5_exc:
            logger.debug("W5 auction→RL feedback skipped: %s", _w5_exc)

        # ── Auto-calibration: self-heal ability scores every N runs ──────
        # Fires as a non-blocking background task — never delays the pipeline.
        # Controlled by AIFACTORY_CALIBRATE_EVERY_N_RUNS (0 = disabled).
        try:
            import asyncio as _asyncio

            from aifactory.config.settings import get_settings as _gs_cal

            _cal_n = _gs_cal().calibrate_every_n_runs
            if _cal_n > 0:
                from aifactory.core.run_store import get_run_store as _grs

                _run_count = len(_grs().list_recent(n=10_000))
                if _run_count > 0 and _run_count % _cal_n == 0:
                    logger.info(
                        "Auto-calibration triggered (run_count=%d, every_n=%d)",
                        _run_count,
                        _cal_n,
                    )

                    async def _auto_calibrate() -> None:
                        try:
                            from aifactory.core.model_calibrator import (
                                apply_patches,
                                get_calibration_report,
                            )

                            report = await get_calibration_report()
                            if report["patches"]:
                                apply_patches(report["patches"], dry_run=False)
                                logger.info(
                                    "Auto-calibration: patched %d model(s) in model_registry.yaml",
                                    len(report["patches"]),
                                )
                            else:
                                logger.debug("Auto-calibration: no patches needed")
                        except Exception as _inner_exc:
                            logger.debug("Auto-calibration task failed (non-fatal): %s", _inner_exc)

                    _asyncio.create_task(_auto_calibrate())
        except Exception as _cal_exc:
            logger.debug("Auto-calibration check failed (non-fatal): %s", _cal_exc)

        # ── W6: Gate self-calibration — nudge LITE/EXTENDED thresholds ──────
        # After every AIFACTORY_GATE_CALIBRATE_EVERY_N_RUNS pipeline runs,
        # GateCalibrator reads run history and adjusts complexity_score
        # thresholds so topology choices that produce poor quality are
        # selected less aggressively next time.  Non-blocking background task.
        try:
            import asyncio as _asyncio_gate

            from aifactory.config.settings import get_settings as _gs_gate

            _gate_cal_n = getattr(_gs_gate(), "gate_calibrate_every_n_runs", 0)
            if _gate_cal_n > 0:
                from aifactory.core.run_store import get_run_store as _grs_gate

                _gate_run_count = len(_grs_gate().list_recent(n=10_000))
                if _gate_run_count > 0 and _gate_run_count % _gate_cal_n == 0:
                    logger.info(
                        "W6 gate calibration triggered (run_count=%d, every_n=%d)",
                        _gate_run_count,
                        _gate_cal_n,
                    )

                    async def _gate_calibrate() -> None:
                        try:
                            from aifactory.core.gate_calibrator import (
                                apply_patches as _gate_apply,
                            )
                            from aifactory.core.gate_calibrator import (
                                get_calibration_report as _gate_report,
                            )

                            _report = await _gate_report()
                            _has_changes = any(v is not None for v in _report["patches"].values())
                            if _has_changes:
                                _gate_apply(_report["patches"], dry_run=False)
                                logger.info(
                                    "W6 gate calibration applied: %s",
                                    _report["reasoning"],
                                )
                            else:
                                logger.debug("W6 gate calibration: no threshold changes")
                        except Exception as _gc_inner:
                            logger.debug("W6 gate calibration task failed: %s", _gc_inner)

                    _asyncio_gate.create_task(_gate_calibrate())
        except Exception as _gc_exc:
            logger.debug("W6 gate calibration check failed (non-fatal): %s", _gc_exc)

        return {"agent_assessments": assessments, "judge_metrics": metrics}

    return judge_review


def make_master_respond(
    agent: PipelineAgent,
    semantic_cache: Any = None,
    run_id: str = "",
    registry: Any = None,
):
    async def master_respond(state: DynamicPipelineState) -> dict[str, Any]:
        shared_ctx = state.get("shared_context", {})
        history: list[dict] = list(state.get("conversation_history") or [])

        direct = state.get("direct_answer", "")
        if direct:
            logger.info("Master: direct answer — no consolidation needed")
            history.append({"role": "assistant", "content": direct})
            return {"final_response": direct, "conversation_history": history}

        team_output = state.get("final_response", "")

        # Always call master — it translates worker-internal language to clean user prose.
        # The 400-token cost is always worth the UX improvement; bypassing it was the
        # cause of protocol leakage (FINDINGS:, VALIDATED, etc.) reaching the user.
        task_results = state.get("task_results", [])
        # Strip protocol markers from team_output before handing to master
        _clean_output = team_output
        for _marker in ("FINDINGS:", "REMEMBER:", "SHARE:", "VALIDATED"):
            if _marker in _clean_output:
                _clean_output = _clean_output.split(_marker)[0].strip()
        final = await agent.think(
            f'Request: "{state["user_query"]}"\n\n'
            f"Team output:\n\n{_clean_output or team_output}\n\n"
            "Deliver this to the human as a clean, direct answer. "
            "No mention of agents, teams, workers, or internal process.\n"
            "FIDELITY RULES (critical):\n"
            "- Preserve every concrete fact, number, figure, name, and unit EXACTLY "
            "as it appears in the team output. Do NOT round, rescale, re-estimate, "
            "or invent any value.\n"
            "- If the team output contains conflicting figures for the same thing, "
            "surface the conflict explicitly (e.g. 'estimates range from X to Y') "
            "rather than silently picking one.\n"
            "- Do not add facts, statistics, or claims that are not present in the "
            "team output.",
            shared_ctx,
        )
        # Fallback: if master's synthesis is empty (truncated), use team output
        if not final.strip() and team_output.strip():
            logger.warning("Master respond returned empty — falling back to team output")
            final = team_output
        history.append({"role": "assistant", "content": final})

        # Quality guarantee: lite/MINIMAL topologies omit the judge node entirely,
        # so a multi-worker run can reach here with judge_metrics={} — zero quality
        # measurement. Master runs in EVERY topology, so synthesize metrics here as
        # the last line of defence. This keeps the RL signal and quality gate honest
        # regardless of which graph shape the gate selected.
        _judge_metrics = state.get("judge_metrics", {})
        _synth_assessments: list[dict] = []
        if not _judge_metrics and registry is not None:
            _synth_assessments, _judge_metrics = _synthesize_judge_metrics(
                registry, task_results, sum(a.total_tokens for a in registry.all())
            )
            logger.info(
                "Master: judge node was absent (topology=%s) — synthesized quality metrics",
                state.get("shared_context", {}).get("graph_spec", {}).get("topology", "?"),
            )

        # A1: promote high-quality run to global memory for cross-run learning
        if agent.memory is not None:
            try:
                _run_quality = float(_judge_metrics.get("composite_quality", 10.0))
                _promo_summary = (
                    f"Request: {state['user_query'][:200]}\n"
                    f"Solution: {final[:400]}\n"
                    f"Quality: {_run_quality:.1f}/10"
                )
                agent.memory.promote_to_global(
                    agent_id="master",
                    summary=_promo_summary,
                    run_quality=_run_quality,
                )
            except Exception as _gp_exc:
                logger.debug("Global memory promotion skipped: %s", _gp_exc)

        # Store in semantic cache for future runs (only for full-pipeline results)
        if semantic_cache is not None and final.strip():
            _sc_meta = state.get("shared_context", {})
            try:
                semantic_cache.store(
                    state["user_query"],
                    final,
                    metadata={
                        "mode": _sc_meta.get("mode", ""),
                        "task_count": len(state.get("task_results", [])),
                        "run_id": run_id,
                    },
                )
            except Exception as _sc_store_exc:
                logger.debug("SemanticCache store skipped: %s", _sc_store_exc)

        _out: dict[str, Any] = {"final_response": final, "conversation_history": history}
        # Surface synthesized metrics so the dump / RL scorer see them
        if _synth_assessments or (not state.get("judge_metrics") and _judge_metrics):
            _out["judge_metrics"] = _judge_metrics
            if _synth_assessments:
                _out["agent_assessments"] = _synth_assessments
        return _out

    return master_respond


# =============================================================================
# Dump — full JSON run artefact
# =============================================================================


def write_dump(
    run_id: str,
    cfg: dict[str, Any],
    result: dict[str, Any],
    registry: AgentRegistry,
    elapsed: float,
    request: str = "",
    gate: str = "",
) -> Path:
    """
    Write a complete JSON dump of the pipeline run.

    Args:
        run_id: Unique run identifier (timestamp-based).
        cfg: Graph config dict (mode, model, etc.).
        result: Final pipeline state dict.
        registry: AgentRegistry with all agents.
        elapsed: Wall-clock time in seconds.
        request: Original human request text.
        gate: Complexity gate result: 'bypass', 'fast_llm', or '' (full pipeline).

    Returns:
        Path to the written dump file.
    """
    runs_dir = DUMPS_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    dump_path = runs_dir / f"{run_id}.json"

    payload = {
        "run_id": run_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "gate": gate,
        "config": {
            "model": cfg.get("model", "unknown"),
            "mode": cfg.get("mode", "balanced"),
            "request": request,
        },
        "shared_context_final": result.get("shared_context", {}),
        "tasks": result.get("tasks", []),
        "research_needed": result.get("research_needed", False),
        "research_summary": result.get("research_summary", ""),
        "task_results": result.get("task_results", []),
        "agent_assessments": result.get("agent_assessments", []),
        "judge_metrics": result.get("judge_metrics", {}),
        "final_response": result.get("final_response", ""),
        "agents": [a.dump() for a in registry.all()],
        "totals": {
            "agents": len(registry.all()),
            "total_tokens": sum(a.total_tokens for a in registry.all()),
            "total_actions": sum(len(a._actions) for a in registry.all()),
        },
    }

    dump_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info("Run dump saved: %s", dump_path)

    # B2: persist run record to run_store for meta-agent + cross-run learning
    try:
        from aifactory.core.run_store import build_run_record, get_run_store

        _record = build_run_record(
            run_id=run_id,
            query=request,
            mode=cfg.get("mode", "balanced"),
            elapsed=elapsed,
            gate=gate,
            state=result,
            registry=registry,
        )
        get_run_store().save(_record)
        logger.debug("RunStore: saved run %s (quality=%.1f)", run_id, _record.composite_quality)
    except Exception as _rs_exc:
        logger.debug("RunStore save failed (non-fatal): %s", _rs_exc)

    return dump_path


# =============================================================================
# Routing
# =============================================================================


def route_after_analysis(state: DynamicPipelineState) -> str:
    if state.get("direct_answer"):
        return "master_respond"
    if state.get("research_for_analysis"):
        return "researcher"
    if state.get("research_needed"):
        return "researcher"
    return "execution_lead"


def route_after_researcher(state: DynamicPipelineState) -> str:
    return "reanalyse" if state.get("research_for_analysis") else "execution_lead"


def route_after_researcher_lite(state: DynamicPipelineState) -> str:
    """Lite-mode variant: always skip reanalyse, go straight to execution_lead."""
    return "execution_lead"


def route_after_reanalyse(state: DynamicPipelineState) -> str:
    return "researcher" if state.get("research_needed") else "execution_lead"


# =============================================================================
# Graph builder
# =============================================================================


def build_dynamic_pipeline(
    mode: str | None = None,
    registry: AgentRegistry | None = None,
    checkpointer: MemorySaver | None = None,
    run_id: str | None = None,
    memory_bridge: MemoryBridge | None = None,
    event_callback: Any | None = None,
    gate_complexity_score: float | None = None,
) -> tuple[Any, AgentRegistry]:
    """
    Build and compile the 8-node dynamic pipeline graph.

    Args:
        mode: "best", "balanced", or "cost_efficient". Auto-loaded if None.
        registry: Optional AgentRegistry. Created if None.
        checkpointer: Optional MemorySaver for interrupt/resume support.
        run_id: Optional run identifier for dump files. Auto-generated if None.
        memory_bridge: Optional MemoryBridge. Auto-created from settings if None.
            Pass MemoryBridge(router=None) to explicitly disable memory for a run.

    Returns:
        (compiled_graph, registry) — the compiled LangGraph app and the agent registry.
    """
    if registry is None:
        registry = AgentRegistry()
    if run_id is None:
        run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    # Phase 3.5: initialise LiteLLM response cache (all providers, idempotent)
    _init_response_cache()

    # Phase 2.1: auto-create memory bridge per run when none is provided
    if memory_bridge is None:
        from aifactory.config.settings import get_settings as _get_settings

        _settings = _get_settings()
        if _settings.memory_pipeline_enabled:
            try:
                from aifactory.core.memory_router import MemoryRouter as _MemoryRouter

                _router = _MemoryRouter()
                memory_bridge = MemoryBridge(
                    router=_router,
                    run_id=run_id,
                    max_context_chars=_settings.memory_context_max_chars,
                )
                logger.info("Memory bridge initialised for run %s", run_id)
                # Sprint 1: surface in-memory mode warning so users know memory won't persist
                if getattr(_router, "in_memory_mode", False):
                    logger.warning(
                        "Qdrant is running in-memory mode — memory will NOT persist between runs. "
                        "Set QDRANT_URL=http://localhost:6333 and start Qdrant to enable persistence."  # noqa: E501
                    )
                    if callable(event_callback):
                        event_callback({"type": "qdrant_in_memory"})
            except Exception as _exc:
                logger.warning("Memory bridge init failed — running without memory: %s", _exc)

    # Semantic query cache — check before pipeline, store after master_respond
    _semantic_cache = None
    try:
        if memory_bridge is not None and memory_bridge.router is not None:
            from aifactory.core.semantic_cache import build_semantic_cache as _bsc

            _semantic_cache = _bsc(memory_bridge.router.client)
            if _semantic_cache and _semantic_cache._ready:
                logger.debug("SemanticCache ready (size=%d)", _semantic_cache.size())
    except Exception as _sc_exc:
        logger.debug("SemanticCache init skipped: %s", _sc_exc)

    # S3: Topology auto-selection from gate complexity score + settings overrides.
    # gate_complexity_score < 0.35  → MINIMAL (lite, single-worker, no research)
    # gate_complexity_score > 0.80  → EXTENDED (WaveGovernor active, mode escalation)
    # 0.35–0.80                     → FULL (default production path)
    # Settings flag AIFACTORY_LITE_MODE=true forces MINIMAL regardless of score.
    # W6: load calibrated thresholds from GateCalibrator (falls back to defaults)
    try:
        from aifactory.core.gate_calibrator import load_thresholds as _load_thr

        _thr = _load_thr()
        _LITE_THRESHOLD = _thr["lite_threshold"]
        _EXTENDED_THRESHOLD = _thr["extended_threshold"]
        logger.debug(
            "Gate thresholds loaded: lite=%.3f extended=%.3f",
            _LITE_THRESHOLD,
            _EXTENDED_THRESHOLD,
        )
    except Exception:
        _LITE_THRESHOLD = 0.35
        _EXTENDED_THRESHOLD = 0.80

    _lite_mode = False
    _extended_mode = False
    try:
        from aifactory.config.settings import get_settings as _gs_lite

        _settings_lite = _gs_lite()
        if _settings_lite.lite_mode:
            _lite_mode = True
            logger.info("Lite mode enabled — forced by AIFACTORY_LITE_MODE setting")
        elif gate_complexity_score is not None and gate_complexity_score < _LITE_THRESHOLD:
            _lite_mode = True
            logger.info(
                "Lite mode auto-selected — gate complexity %.2f < %.2f threshold",
                gate_complexity_score,
                _LITE_THRESHOLD,
            )
        elif gate_complexity_score is not None and gate_complexity_score > _EXTENDED_THRESHOLD:
            _extended_mode = True
            logger.info(
                "EXTENDED topology auto-selected — gate complexity %.2f > %.2f threshold",
                gate_complexity_score,
                _EXTENDED_THRESHOLD,
            )
    except Exception:
        if gate_complexity_score is not None and gate_complexity_score < _LITE_THRESHOLD:
            _lite_mode = True
        elif gate_complexity_score is not None and gate_complexity_score > _EXTENDED_THRESHOLD:
            _extended_mode = True

    # S2: Semantic research cache — skip repeated researcher LLM calls across runs
    _research_cache = None
    try:
        from aifactory.core.research_cache import ResearchCache as _ResearchCache

        _research_cache = _ResearchCache()
        logger.debug("ResearchCache ready (%d cached entries)", _research_cache.size())
    except Exception as _rc_exc:
        logger.debug("ResearchCache init skipped: %s", _rc_exc)

    # Code leakage prevention — local codebase analysis processor
    _code_processor = None
    try:
        from aifactory.agents.code_security_processor import get_processor as _get_csp

        _code_processor = _get_csp()
        logger.debug("CodeSecurityProcessor ready")
    except Exception as _csp_exc:
        logger.debug("CodeSecurityProcessor init skipped: %s", _csp_exc)

    # Adaptive Mesh W2: KnowledgeGraph — populated from CSP structural index each run
    _knowledge_graph = None
    try:
        from aifactory.core.knowledge_graph import KnowledgeGraph as _KG

        _knowledge_graph = _KG()
        if _code_processor is not None:
            # Build structural index text from CSP (zero LLM cost), then ingest
            _codebase = getattr(_code_processor, "_codebase", None)
            if _codebase:
                _idx_text = _code_processor.build_structural_index(_codebase)
                _knowledge_graph.ingest_structural_index(_idx_text)
                logger.debug("KnowledgeGraph populated: %s", _knowledge_graph.stats())
    except Exception as _kg_exc:
        logger.debug("KnowledgeGraph init skipped: %s", _kg_exc)

    graph_cfg = load_graph_config()
    if mode is None:
        mode = graph_cfg.get("mode", "balanced")

    fallback_model = graph_cfg.get("model", "gpt-4o-mini")

    # Phase 5: per-run budget tracker — enforces settings.max_budget_usd hard cap
    from aifactory.config.settings import get_settings as _get_budget_settings

    _budget_cap = _get_budget_settings().max_budget_usd
    budget_tracker = BudgetTracker(cap_usd=_budget_cap)

    # Sprint 1: 80% budget warning — surface to console via event_callback or logger
    def _budget_warn_cb(spent: float, cap: float) -> None:
        import logging as _log

        _log.getLogger(__name__).warning(
            "⚠  Budget alert: $%.4f spent (%.0f%% of $%.2f cap) — pipeline will halt at cap.",
            spent,
            (spent / cap * 100) if cap > 0 else 0,
            cap,
        )
        if callable(event_callback):
            event_callback({"type": "budget_warning", "spent": spent, "cap": cap})

    budget_tracker.on_warn = _budget_warn_cb
    models = load_available_models()

    # Load agent configs
    agent_cfgs = {
        n: load_agent_cfg(n)
        for n in ["master", "judge", "analyser", "researcher", "execution_lead", "worker"]
    }

    # Master always uses best_model (mode-independent)
    master_model = best_model(models)

    # Named agents: ability profile x mode bias
    judge_model = model_for_agent(agent_cfgs["judge"], models, mode, fallback_model)
    analyser_model = model_for_agent(agent_cfgs["analyser"], models, mode, fallback_model)
    researcher_model = model_for_agent(agent_cfgs["researcher"], models, mode, fallback_model)
    lead_model = model_for_agent(agent_cfgs["execution_lead"], models, mode, fallback_model)

    logger.info(
        "Pipeline models [mode=%s]: master=%s, judge=%s, analyser=%s, researcher=%s, lead=%s",
        mode,
        master_model,
        judge_model,
        analyser_model,
        researcher_model,
        lead_model,
    )

    # Create agent instances — Phase 2.1: inject memory bridge into every agent
    master = registry.register(
        PipelineAgent(
            "master", {**agent_cfgs["master"], "model": master_model}, master_model, memory_bridge
        )
    )
    judge = registry.register(
        PipelineAgent(
            "judge", {**agent_cfgs["judge"], "model": judge_model}, judge_model, memory_bridge
        )
    )
    analyser = registry.register(
        PipelineAgent(
            "analyser",
            {**agent_cfgs["analyser"], "model": analyser_model},
            analyser_model,
            memory_bridge,
        )
    )
    researcher_ag = registry.register(
        PipelineAgent(
            "researcher",
            {**agent_cfgs["researcher"], "model": researcher_model},
            researcher_model,
            memory_bridge,
        )
    )
    lead = registry.register(
        PipelineAgent(
            "execution_lead",
            {**agent_cfgs["execution_lead"], "model": lead_model},
            lead_model,
            memory_bridge,
        )
    )

    # Phase 5: inject fallback model + budget tracker into all named agents
    for _ag in (master, judge, analyser, researcher_ag, lead):
        _ag.fallback_model = fallback_model
        _ag._budget_tracker = budget_tracker

    # Token streaming: inject on_token callback when streaming is enabled + event_callback exists
    try:
        from aifactory.config.settings import get_settings as _gs_stream

        _stream_enabled = _gs_stream().stream_tokens_enabled
    except Exception:
        _stream_enabled = True

    if _stream_enabled and callable(event_callback):

        def _make_token_cb(agent_name: str):
            def _on_token(token: str) -> None:
                event_callback({"type": "token", "agent": agent_name, "text": token})

            return _on_token

        for _ag in (master, judge, analyser, researcher_ag, lead):
            _ag.on_token = _make_token_cb(_ag.name)

    # S2: embed_fn for researcher semantic cache lookup
    _embed_fn_for_researcher: Any = None
    try:
        from aifactory.core.embed_provider import build_embed_fn as _bef

        _embed_fn_for_researcher, _ = _bef()
    except Exception:
        pass

    # GraphCompiler: delegate graph construction to topology-aware compiler.
    # When _lite_mode is forced (settings or gate score), we override the spec
    # to MINIMAL; otherwise the analyser's graph_spec drives topology selection.
    from aifactory.graphs.graph_compiler import GraphCompiler, GraphSpec, TopologyShape

    _forced_spec: GraphSpec | None = None
    if _lite_mode:
        _forced_spec = GraphSpec(topology=TopologyShape.MINIMAL)
    elif _extended_mode:
        _forced_spec = GraphSpec(topology=TopologyShape.EXTENDED, needs_research=True)

    compiler = GraphCompiler(
        master=master,
        analyser=analyser,
        researcher=researcher_ag,
        lead=lead,
        judge=judge,
        semantic_cache=_semantic_cache,
        memory_bridge=memory_bridge,
        research_cache=_research_cache,
        embed_fn=_embed_fn_for_researcher,
        run_id=run_id,
        event_callback=event_callback,
        budget_tracker=budget_tracker,
        agent_cfgs=agent_cfgs,
        models=models,
        mode=mode,
        registry=registry,
        code_processor=_code_processor,
        checkpointer=checkpointer,
        knowledge_graph=_knowledge_graph,
    )

    compile_result = compiler.compile(_forced_spec)
    compiled = compile_result.graph
    _mode_label = "lite" if compile_result.lite_mode else "standard"
    logger.info(
        "Dynamic pipeline compiled [topology=%s, nodes=%d, mode=%s]",
        compile_result.topology.value,
        compile_result.node_count,
        _mode_label,
    )
    return compiled, registry
