"""
Anthropic prompt-caching helpers — shared across every LiteLLM call site.

Anthropic's Claude models support prompt caching: the caller marks a stable
prefix (typically the system message) with ``cache_control: {"type": "ephemeral"}``
and Anthropic reuses the cached prefix for subsequent calls in the same
session, cutting input token cost by ~90% and TTFT by ~85%.

This module centralises two concerns previously duplicated / missing at
call sites:

  * ``is_anthropic_model(model)`` — detect Claude routes (direct + OpenRouter)
    that will actually honour ``cache_control``.
  * ``build_cached_system(system, model, api_base)`` — return either a
    plain string (when caching does not apply) or the LiteLLM content-block
    list with ``cache_control`` attached.

Gating rules (identical to ``PipelineAgent.think`` — do NOT diverge):

  1. Setting ``prompt_cache_enabled`` must be True.
  2. The model must be an Anthropic Claude route (respecting
     ``prompt_cache_openrouter`` for OpenRouter routes).
  3. When a local LiteLLM proxy is in use (``LITELLM_API_BASE`` set at
     import time in ``dynamic_pipeline`` OR the caller passes a non-empty
     ``api_base``), skip caching — proxy pass-through of ``cache_control``
     is not guaranteed.

Non-Anthropic models MUST receive the plain-string system message; the
content-block format either no-ops or errors depending on the provider.
"""

from __future__ import annotations

import logging
import os as _os
from typing import Any

logger = logging.getLogger(__name__)


def is_anthropic_model(model: str, include_openrouter: bool = True) -> bool:
    """
    True when the model is an Anthropic Claude model that supports cache_control.

    Covers:
      - Direct Anthropic: "claude-*" or "anthropic/claude-*"
      - OpenRouter-routed: "openrouter/anthropic/claude-*" (OpenRouter passes
        cache_control through to Anthropic when include_openrouter=True)

    Does NOT match generic LiteLLM proxy models (openai/claude-*) since proxy
    pass-through of cache_control varies and can't be guaranteed.
    """
    m = model.lower()
    if m.startswith("claude-") or m.startswith("anthropic/claude-"):
        return True
    return bool(include_openrouter and "openrouter" in m and "anthropic/claude" in m)


def _proxy_active(api_base: str | None) -> bool:
    """
    True when a LiteLLM proxy will handle the call.

    Callers may pass an explicit ``api_base``; if omitted, fall back to the
    module-level ``LITELLM_API_BASE`` env var (same source that
    ``dynamic_pipeline`` reads at import time).
    """
    # An explicit ``api_base`` (including empty string) is authoritative — an
    # empty string means "caller has already checked, no proxy is active" and
    # must NOT fall back to reading the env var (that would mis-fire when a
    # test patches the module-level _api_base but the surrounding process
    # still has LITELLM_API_BASE exported).
    if api_base is not None:
        return bool(api_base.strip())
    return bool(_os.getenv("LITELLM_API_BASE", "").strip())


def build_cached_system(
    system: str,
    model: str,
    *,
    api_base: str | None = None,
    variable_suffix: str = "",
) -> str | list[dict[str, Any]]:
    """
    Return the ``system`` payload to hand to LiteLLM.

    When Anthropic prompt caching applies, returns a content-block list with
    ``cache_control: ephemeral`` attached to the ``system`` block; otherwise
    returns the concatenated plain string.

    ``variable_suffix`` (Opt #6 — prompt-cache prefix stability):
        Content that MUST NOT participate in the cached prefix — e.g. the
        rendered ``shared_context`` (grows monotonically wave-over-wave) and
        the per-turn ``memory_context`` (query-dependent).  When caching
        applies, this content is emitted as a SECOND, un-marked content
        block so the stable ``system`` prefix keeps cache-hitting across
        every subsequent worker turn.  When caching does not apply the
        two strings are simply concatenated — behaviour identical to
        callers that don't pass ``variable_suffix``.

    Safe to call from every LiteLLM call site — the gating rules are applied
    inside so no bypass path can accidentally skip caching or accidentally
    inject ``cache_control`` on a non-Claude provider.
    """
    # Settings lookup is guarded — a missing / broken settings module must not
    # break the LLM call. Default to caching on (matches PipelineAgent.think).
    try:
        from aifactory.config.settings import get_settings

        s = get_settings()
        cache_on = bool(s.prompt_cache_enabled)
        openrouter_ok = bool(getattr(s, "prompt_cache_openrouter", True))
    except Exception:
        cache_on = True
        openrouter_ok = True

    _caching_off = (
        not cache_on
        or _proxy_active(api_base)
        or not is_anthropic_model(model, include_openrouter=openrouter_ok)
    )
    if _caching_off:
        return system + variable_suffix if variable_suffix else system

    blocks: list[dict[str, Any]] = [
        {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}},
    ]
    if variable_suffix:
        # Second block is intentionally NOT cache-marked — Anthropic caches
        # up to and including the last cache_control checkpoint, so the
        # stable ``system`` prefix survives even when this suffix mutates.
        blocks.append({"type": "text", "text": variable_suffix})
    return blocks
