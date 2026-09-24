"""
Benchmark + correctness tests: per-agent-role ``shared_context`` filtering.

Perf Opt #5 (follows a651f2d, 0210ac2, e9552c8, b335093).  Previously
``_prune_shared_context`` trimmed to 25 keys but every agent's system
prompt got the whole thing — even a worker that only needs the objective
+ FINDINGS.  This test suite guards two things:

  1. The filter measurably reduces the rendered token count on a
     realistic mid-pipeline shared_context (>=50 % on the worker filter).
  2. With ``context_keys=None`` the rendered block is byte-identical to
     the pre-Opt-#5 behaviour — no regression for the analyser / master
     / execution_lead who intentionally see everything.

Uses whitespace-split as a lightweight token proxy — good enough for the
compression assertion and independent of the tiktoken install state.
"""

from __future__ import annotations

import statistics
import time
from typing import Any

import pytest
import yaml

from aifactory.core.agent_factory import (
    _match_context_key,
    build_shared_context_block,
    filter_shared_context,
)
from aifactory.core.schemas import AgentConfig, AgentRole

_ITERS = 5_000
_WARMUP = 200


# The worker filter that ships in worker.yaml.  Kept in sync deliberately —
# if this drifts from the YAML, the benchmark under-reports the real win.
_WORKER_FILTER: list[str] = [
    "objective",
    "connected_integrations",
    "findings_*",
    "prior_knowledge",
    "attachments_note",
    "attachments_count",
]


def _make_shared_ctx() -> dict[str, str]:
    """
    A realistic mid-pipeline shared_context (~15 keys) built from what
    dynamic_pipeline actually writes: objective, integration snapshot,
    per-wave findings, research/analysis summaries, attachments,
    auction bookkeeping, SHARE'd worker facts, workspace, tasks_count.
    """
    return {
        "objective": (
            "Build a token-efficient per-role shared_context filter and "
            "benchmark it end-to-end so the worker prompt shrinks by half."
        ),
        "connected_integrations": "github, slack, linear, notion",
        "prior_knowledge": (
            "Prior runs settled on FastAPI + Redis for async task queues; "
            "auth is JWT with 1h expiry and refresh via /auth/refresh."
        ),
        "attachments_note": "3 files attached — see the pipeline state.",
        "attachments_count": "3",
        "workspace": "/Users/dev/projects/aifactory-perf",
        "tasks_count": "7",
        "research_summary": (
            "Anthropic prompt caching hits ~90% of the input tokens when the "
            "system prefix is stable; recompute cost is amortised across the "
            "5-minute cache TTL. LiteLLM forwards cache_control markers via "
            "the anthropic-beta header when the model is a Claude family. "
            "For streaming responses the cache_control tag must appear on "
            "the last content block of the system message, otherwise "
            "Anthropic responds with cache_creation_input_tokens=0 and "
            "the request is billed at full input-token rate."
        ),
        "research_snippet": (
            "See commit b335093 for the cache_control helper wiring — the "
            "helper gates on Claude-only + no-proxy + settings flag so the "
            "same call site is safe to reuse from BaseAgent, PipelineAgent, "
            "and the complexity assessor without duplicated logic."
        ),
        "analysis_research": (
            "Analyser reviewed four candidate designs for per-role context "
            "filtering: (a) hard-coded Python whitelists per role, "
            "(b) YAML-driven config_keys, (c) runtime introspection of the "
            "system prompt, (d) LLM-driven relevance scoring. Chose (b) — "
            "config-driven, backwards-compatible, zero new deps, no hot-path "
            "overhead beyond a dict comprehension per agent per turn."
        ),
        "findings_wave_0": (
            "Baseline shared_context has ~15 keys mid-pipeline; ~1.2 KB rendered."
        ),
        "findings_wave_1": (
            "Filter shrinks worker prompt by >50 %; analyser unchanged."
        ),
        "findings_wave_2": (
            "Backwards-compat requires context_keys=None to be byte-identical."
        ),
        "auction_winner_model": "anthropic/claude-sonnet-4-20250514",
        "auction_floor": "0.42",
        "qa0": "Q: which cache?|A: Anthropic ephemeral cache_control on system.",
        "framework": "FastAPI",  # SHARE'd fact from a prior worker
        "db": "PostgreSQL",  # SHARE'd fact
    }


# ────────────────────────────────────────────────────────────────────────
# 1. Benchmark — worker filter drops token count by >=50 %
# ────────────────────────────────────────────────────────────────────────


@pytest.mark.perf
def test_worker_filter_halves_rendered_token_count() -> None:
    ctx = _make_shared_ctx()

    baseline = build_shared_context_block(ctx, allowed_patterns=None)
    optimized = build_shared_context_block(ctx, allowed_patterns=_WORKER_FILTER)

    baseline_tokens = len(baseline.split())
    optimized_tokens = len(optimized.split())
    reduction_pct = 100.0 * (baseline_tokens - optimized_tokens) / max(baseline_tokens, 1)

    # Wall-time — filter is a dict comp, must not be materially slower.
    def _bench(patterns: list[str] | None) -> float:
        for _ in range(_WARMUP):
            build_shared_context_block(ctx, allowed_patterns=patterns)
        samples: list[float] = []
        for _ in range(_ITERS):
            t0 = time.perf_counter()
            build_shared_context_block(ctx, allowed_patterns=patterns)
            samples.append(time.perf_counter() - t0)
        return statistics.mean(samples)

    baseline_ns = _bench(None) * 1e9
    optimized_ns = _bench(_WORKER_FILTER) * 1e9

    print(
        "\n[perf] shared_context per-role filter — worker profile\n"
        f"  input keys       : {len(ctx)}\n"
        f"  baseline tokens  : {baseline_tokens}\n"
        f"  optimized tokens : {optimized_tokens}\n"
        f"  reduction        : {reduction_pct:.1f}%\n"
        f"  iterations       : {_ITERS} (warmup {_WARMUP})\n"
        f"  baseline mean    : {baseline_ns:.0f} ns\n"
        f"  optimized mean   : {optimized_ns:.0f} ns\n"
    )

    assert reduction_pct >= 50.0, (
        f"Expected >=50% token reduction, got {reduction_pct:.1f}% "
        f"({baseline_tokens} -> {optimized_tokens})"
    )
    # Filter overhead is negligible in absolute terms — this render happens
    # once per agent per turn, so anything under 50 µs is dwarfed by the LLM
    # call it feeds into.  The real win is the token-count drop above.
    assert optimized_ns < 50_000.0, (
        f"Filter render is unexpectedly slow: {optimized_ns:.0f} ns "
        f"(baseline {baseline_ns:.0f} ns)"
    )


# ────────────────────────────────────────────────────────────────────────
# 2. Backwards-compat — None is byte-identical to prior behaviour
# ────────────────────────────────────────────────────────────────────────


def test_context_keys_none_is_byte_identical() -> None:
    """
    Agents without a ``context_keys`` field (analyser, master, execution_lead,
    meta) MUST see the exact same rendered block as before Opt #5.
    """
    ctx = _make_shared_ctx()

    # Emulate the pre-Opt-#5 rendering: iterate + join, no filter step.
    pre_opt_rendered = "\n\n## Shared Context\n" + "\n".join(
        f"{k}: {v}" for k, v in ctx.items() if v
    )

    post_opt_rendered = build_shared_context_block(ctx, allowed_patterns=None)
    # And the no-arg call must match too — that's what every existing caller uses.
    post_opt_default = build_shared_context_block(ctx)

    assert post_opt_rendered == pre_opt_rendered
    assert post_opt_default == pre_opt_rendered

    # Empty ctx path unchanged
    assert build_shared_context_block({}) == ""
    assert build_shared_context_block({}, allowed_patterns=_WORKER_FILTER) == ""


# ────────────────────────────────────────────────────────────────────────
# 3. Pattern matching correctness
# ────────────────────────────────────────────────────────────────────────


def test_pattern_matching_exact_wildcard_and_missing() -> None:
    ctx = {
        "objective": "solve X",
        "prior_knowledge": "PK",
        "findings_wave_0": "f0",
        "findings_wave_1": "f1",
        "findings_wave_2": "f2",
        "connected_integrations": "github",
        "random_unrelated": "noise",
    }

    # Exact-only
    exact = filter_shared_context(ctx, ["objective", "prior_knowledge"])
    assert set(exact) == {"objective", "prior_knowledge"}

    # Wildcard suffix
    wild = filter_shared_context(ctx, ["findings_*"])
    assert set(wild) == {"findings_wave_0", "findings_wave_1", "findings_wave_2"}

    # Mix + missing key silently ignored (no KeyError)
    mixed = filter_shared_context(
        ctx,
        ["objective", "findings_*", "nonexistent_key", "attachments_note"],
    )
    assert set(mixed) == {
        "objective",
        "findings_wave_0",
        "findings_wave_1",
        "findings_wave_2",
    }

    # Empty pattern list = agent opts out
    assert filter_shared_context(ctx, []) == {}

    # None = passthrough (identity, not a copy — cheap on hot path)
    assert filter_shared_context(ctx, None) is ctx

    # Insertion order preserved (matters for prompt-cache stability)
    ordered = filter_shared_context(ctx, ["findings_*", "objective"])
    assert list(ordered.keys()) == [
        "objective",
        "findings_wave_0",
        "findings_wave_1",
        "findings_wave_2",
    ]

    # _match_context_key unit-level
    assert _match_context_key("findings_wave_0", "findings_*")
    assert not _match_context_key("findings_wave_0", "finding_*")
    assert _match_context_key("objective", "objective")
    assert not _match_context_key("objective", "objectives")


# ────────────────────────────────────────────────────────────────────────
# 4. End-to-end — PipelineAgent._build_system honours context_keys
# ────────────────────────────────────────────────────────────────────────


def test_pipeline_agent_system_prompt_only_includes_allowed_keys() -> None:
    """
    Wire the YAML-declared filter through the actual PipelineAgent code
    path.  A worker cfg with ``context_keys`` set MUST produce a system
    prompt that mentions only the whitelisted keys — no research_summary,
    no auction internals, no unrelated SHARE'd facts.
    """
    from aifactory.graphs.dynamic_pipeline import PipelineAgent

    cfg: dict[str, Any] = {
        "role": "Worker",
        "system_prompt": "You are a worker. Do the task.",
        "context_keys": list(_WORKER_FILTER),
        "model": "gpt-4o-mini",
    }
    agent = PipelineAgent(name="worker-1", cfg=cfg, model="gpt-4o-mini")

    ctx = _make_shared_ctx()
    system = agent._build_system(ctx)

    # Allowed keys are present
    assert "objective:" in system
    assert "connected_integrations:" in system
    assert "prior_knowledge:" in system
    assert "attachments_count:" in system
    assert "findings_wave_0:" in system
    assert "findings_wave_1:" in system
    assert "findings_wave_2:" in system

    # Disallowed keys are absent
    for banned in (
        "research_summary:",
        "research_snippet:",
        "analysis_research:",
        "auction_winner_model:",
        "auction_floor:",
        "framework:",
        "db:",
        "workspace:",
        "qa0:",
        "tasks_count:",
    ):
        assert banned not in system, f"leaked banned key {banned!r} into worker prompt"

    # Same code path with an unfiltered agent (analyser-like) sees everything
    full_cfg: dict[str, Any] = {
        "role": "Analyser",
        "system_prompt": "You are the analyser.",
        # No context_keys => full context
        "model": "gpt-4o-mini",
    }
    full_agent = PipelineAgent(name="analyser", cfg=full_cfg)
    full_system = full_agent._build_system(ctx)
    for k in ctx:
        assert f"{k}:" in full_system, f"analyser missing key {k}"


# ────────────────────────────────────────────────────────────────────────
# 5. AgentConfig schema — new field is optional and defaults to None
# ────────────────────────────────────────────────────────────────────────


def test_agent_config_context_keys_optional_default_none() -> None:
    cfg = AgentConfig(role=AgentRole.WORKER, name="w", model="gpt-4o-mini")
    assert cfg.context_keys is None

    cfg2 = AgentConfig(
        role=AgentRole.WORKER,
        name="w2",
        model="gpt-4o-mini",
        context_keys=["objective", "findings_*"],
    )
    assert cfg2.context_keys == ["objective", "findings_*"]


# ────────────────────────────────────────────────────────────────────────
# 6. YAML wiring — the four unfiltered roles MUST NOT declare context_keys
# ────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "unfiltered_role",
    ["analyser", "master", "execution_lead", "meta"],
)
def test_full_context_roles_have_no_context_keys_filter(unfiltered_role: str) -> None:
    """Guard against a well-intentioned refactor breaking the analyser."""
    from aifactory.core.model_selector import load_agent_cfg

    cfg = load_agent_cfg(unfiltered_role)
    assert cfg.get("context_keys") is None, (
        f"{unfiltered_role}.yaml must NOT declare context_keys — it needs the "
        f"full shared_context. Found: {cfg.get('context_keys')!r}"
    )


def test_worker_and_judge_yamls_declare_expected_filter() -> None:
    from aifactory.core.model_selector import load_agent_cfg

    worker = load_agent_cfg("worker")
    assert worker.get("context_keys") == _WORKER_FILTER

    judge = load_agent_cfg("judge")
    assert judge.get("context_keys") == [
        "objective",
        "connected_integrations",
        "findings_*",
    ]

    # Silence unused-import guard: yaml import is used above indirectly
    # (load_agent_cfg wraps yaml.safe_load).
    _ = yaml
