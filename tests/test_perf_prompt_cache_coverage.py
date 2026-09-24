"""
Perf Opt #4 — Anthropic prompt-cache coverage audit.

Goal: every LiteLLM chat-completion call site in ``src/aifactory`` either
routes through the shared ``build_cached_system`` helper (which attaches
``cache_control: ephemeral`` on Claude system messages) or is on the
documented exemption allowlist below.

The audit is enforced two ways:

  1. Static: grep the source for ``litellm.(a)completion`` and assert every
     hit either mentions ``cache_control`` / ``build_cached_system`` /
     ``_build_cached_system`` / ``sys_content`` in the surrounding block OR
     the file:line is on the exemption allowlist.

  2. Functional: monkeypatch ``litellm.acompletion`` on the newly-fixed
     call sites (BaseAgent.think, PipelineAgent.think_multimodal, the
     complexity assessor) and assert the outgoing system message includes
     the ``cache_control`` block when the model is Claude, and does NOT
     when it's non-Claude / caching is disabled.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aifactory.core.prompt_cache import build_cached_system, is_anthropic_model

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "aifactory"

# ─── Exemption allowlist ────────────────────────────────────────────────────
# Each entry: (relative path, line, one-line justification).
# Adding to this list is fine — but the justification must state why
# cache_control legitimately would not fire or would not help.
EXEMPT_CALL_SITES: set[tuple[str, int]] = {
    # Dynamic prompt: rebuilt every call from fresh memory entries. No stable
    # system prefix to cache. Called at most ~once per promotion (~every 10
    # agent turns) so even if we forced a Claude system, cache hit rate = 0.
    ("core/base_agent.py", 404),
    # PipelineAgent._think_stream: reuses the ``messages`` list that
    # PipelineAgent.think already built via _build_cached_system — the same
    # cache_control-tagged system block flows straight into the streaming call.
    ("graphs/dynamic_pipeline.py", 795),
    # Same call site, streaming-fallback branch — identical ``messages`` list.
    ("graphs/dynamic_pipeline.py", 820),
}

# Any embedding / non-chat entry points — cache_control does not apply.
EMBEDDING_PATTERN = re.compile(r"litellm\.a?embedding\b")
# Require an opening paren so we don't false-match docstring / comment mentions.
CHAT_PATTERN = re.compile(r"litellm\.a?completion\(")


def _iter_chat_call_sites() -> list[tuple[Path, int, str]]:
    """Yield (path, line_number, line_text) for every chat-completion call site."""
    hits: list[tuple[Path, int, str]] = []
    for py in SRC_ROOT.rglob("*.py"):
        text = py.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), start=1):
            if CHAT_PATTERN.search(line):
                hits.append((py, i, line))
    return hits


# =============================================================================
# 1. Static audit — every call site is covered or explicitly exempt
# =============================================================================


def test_every_completion_call_site_is_cached_or_exempt() -> None:
    """
    Fail loudly if a new ``litellm.(a)completion`` call site appears that is
    neither wrapped in ``build_cached_system`` / ``_build_cached_system`` nor
    on the exemption allowlist.
    """
    unaccounted: list[str] = []
    for path, lineno, _line in _iter_chat_call_sites():
        rel = str(path.relative_to(SRC_ROOT))
        if (rel, lineno) in EXEMPT_CALL_SITES:
            continue
        # Look at a small window around the call for cache_control wiring.
        text = path.read_text(encoding="utf-8", errors="replace").splitlines()
        window_start = max(0, lineno - 40)
        window_end = min(len(text), lineno + 5)
        window = "\n".join(text[window_start:window_end])
        if any(
            marker in window
            for marker in (
                "build_cached_system",
                "_build_cached_system",
                "cache_control",
                # PipelineAgent stream + retry paths reuse the already-cached
                # ``_messages_snapshot`` built by think() above.
                "_messages_snapshot",
            )
        ):
            continue
        unaccounted.append(f"{rel}:{lineno} — {_line.strip()}")

    assert not unaccounted, (
        "New unaudited litellm.completion / acompletion call site(s) — "
        "either route through build_cached_system or add to "
        "EXEMPT_CALL_SITES with justification:\n  " + "\n  ".join(unaccounted)
    )


def test_exempt_allowlist_lines_still_point_at_completion_calls() -> None:
    """Prevent stale allowlist entries drifting off the real call site."""
    for rel, lineno in EXEMPT_CALL_SITES:
        path = SRC_ROOT / rel
        text = path.read_text(encoding="utf-8", errors="replace").splitlines()
        # Allow ±10 line drift before we consider the entry stale (the call
        # itself may span several lines; we just need a completion() nearby).
        window = "\n".join(text[max(0, lineno - 10) : lineno + 10])
        assert CHAT_PATTERN.search(window), (
            f"EXEMPT_CALL_SITES entry {rel}:{lineno} no longer near a "
            "litellm.(a)completion call — update or remove it."
        )


# =============================================================================
# 2. Helper unit tests — gating rules
# =============================================================================


class TestBuildCachedSystem:
    def test_claude_returns_content_block(self) -> None:
        out = build_cached_system("SYS", "claude-3-5-haiku-20241022", api_base="")
        assert isinstance(out, list)
        assert out[0]["cache_control"] == {"type": "ephemeral"}
        assert out[0]["text"] == "SYS"

    def test_openrouter_claude_returns_content_block(self) -> None:
        out = build_cached_system(
            "SYS", "openrouter/anthropic/claude-3-5-haiku", api_base=""
        )
        assert isinstance(out, list)

    def test_gpt_returns_plain_string(self) -> None:
        out = build_cached_system("SYS", "gpt-4o-mini", api_base="")
        assert out == "SYS"

    def test_proxy_active_disables_caching(self) -> None:
        out = build_cached_system(
            "SYS", "claude-3-5-haiku-20241022", api_base="http://proxy:4000"
        )
        assert out == "SYS"

    def test_setting_disabled_skips_caching(self) -> None:
        fake = MagicMock()
        fake.prompt_cache_enabled = False
        fake.prompt_cache_openrouter = True
        with patch("aifactory.config.settings.get_settings", return_value=fake):
            out = build_cached_system(
                "SYS", "claude-3-5-haiku-20241022", api_base=""
            )
        assert out == "SYS"

    def test_openrouter_gated_off_via_setting(self) -> None:
        fake = MagicMock()
        fake.prompt_cache_enabled = True
        fake.prompt_cache_openrouter = False
        with patch("aifactory.config.settings.get_settings", return_value=fake):
            out = build_cached_system(
                "SYS", "openrouter/anthropic/claude-3-5-haiku", api_base=""
            )
        assert out == "SYS"

    def test_is_anthropic_model_negative_cases(self) -> None:
        assert not is_anthropic_model("gpt-4o-mini")
        assert not is_anthropic_model("openai/gpt-4o")
        assert not is_anthropic_model("openai/claude-3-haiku")  # proxy route
        assert not is_anthropic_model(
            "openrouter/anthropic/claude-3-haiku", include_openrouter=False
        )


# =============================================================================
# 3. Functional: newly fixed call sites actually emit cache_control
# =============================================================================


def _fake_completion_resp(content: str = "ok", ptok: int = 10, ctok: int = 5):
    usage = MagicMock(prompt_tokens=ptok, completion_tokens=ctok, total_tokens=ptok + ctok)
    msg = MagicMock(content=content, refusal=None)
    choice = MagicMock(message=msg, finish_reason="stop")
    resp = MagicMock(choices=[choice], usage=usage)
    return resp


# ── BaseAgent.think ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_base_agent_think_injects_cache_control_for_claude(worker_config) -> None:
    from aifactory.core.base_agent import BaseAgent

    worker_config.model = "claude-3-5-haiku-20241022"
    agent = BaseAgent(config=worker_config)

    with patch("aifactory.core.base_agent.litellm.acompletion", new_callable=AsyncMock) as mock:
        mock.return_value = _fake_completion_resp("hi")
        await agent.think("ping")

    messages = mock.call_args.kwargs["messages"]
    sys_msg = next(m for m in messages if m["role"] == "system")
    assert isinstance(sys_msg["content"], list), "Claude BaseAgent must use content-blocks"
    assert sys_msg["content"][0]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_base_agent_think_no_cache_control_for_openai(worker_config) -> None:
    from aifactory.core.base_agent import BaseAgent

    worker_config.model = "openai/gpt-4o-mini"
    agent = BaseAgent(config=worker_config)

    with patch("aifactory.core.base_agent.litellm.acompletion", new_callable=AsyncMock) as mock:
        mock.return_value = _fake_completion_resp("hi")
        await agent.think("ping")

    sys_msg = next(m for m in mock.call_args.kwargs["messages"] if m["role"] == "system")
    assert isinstance(sys_msg["content"], str), "Non-Claude models must stay plain-string"


# ── PipelineAgent.think_multimodal ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_pipeline_agent_think_multimodal_injects_cache_control() -> None:
    from aifactory.graphs.dynamic_pipeline import PipelineAgent

    agent = PipelineAgent(
        "vision",
        {"system_prompt": "you see images", "max_tokens": 100},
        model="claude-3-5-sonnet-20241022",
    )

    with (
        patch("litellm.acompletion", new_callable=AsyncMock) as mock,
        patch("aifactory.graphs.dynamic_pipeline._api_base", ""),
    ):
        mock.return_value = _fake_completion_resp("described")
        await agent.think_multimodal(
            "what is this?", ["https://example.com/pic.jpg"], {}
        )

    messages = mock.call_args.kwargs["messages"]
    sys_msg = next(m for m in messages if m["role"] == "system")
    assert isinstance(sys_msg["content"], list)
    assert sys_msg["content"][0]["cache_control"] == {"type": "ephemeral"}


# ── assessor.assess_complexity ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_assess_complexity_injects_cache_control_for_claude() -> None:
    from aifactory.core import assessor

    with patch("aifactory.core.assessor.litellm.acompletion", new_callable=AsyncMock) as mock:
        mock.return_value = _fake_completion_resp('{"complexity":"trivial"}')
        await assessor.assess_complexity("write hello world", model="claude-3-5-haiku-20241022")

    sys_msg = next(m for m in mock.call_args.kwargs["messages"] if m["role"] == "system")
    assert isinstance(sys_msg["content"], list)
    assert sys_msg["content"][0]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_assess_complexity_stays_plain_string_for_gpt4o_mini() -> None:
    from aifactory.core import assessor

    with patch("aifactory.core.assessor.litellm.acompletion", new_callable=AsyncMock) as mock:
        mock.return_value = _fake_completion_resp('{"complexity":"trivial"}')
        await assessor.assess_complexity("write hello world", model="gpt-4o-mini")

    sys_msg = next(m for m in mock.call_args.kwargs["messages"] if m["role"] == "system")
    assert isinstance(sys_msg["content"], str), "gpt-4o-mini must not get cache_control"
