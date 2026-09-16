"""
DynamicPipelineStateV2 — TypedDict state with reducer-backed fields.

Opt-in via ``settings.aifactory_state_reducers_enabled``. When the flag is
False (the default), the pipeline uses the legacy ``DynamicPipelineState``
in ``dynamic_pipeline.py`` and reducer merging is a no-op.

Design rules:
  - Fields that should retain last-writer-wins semantics stay plain (no
    Annotated reducer).
  - List-valued accumulators use ``operator.add`` so parallel Send-API workers
    each contribute rows without clobbering peers.
  - ``shared_context`` uses ``_merge_shared_ctx`` — right-wins per key. This
    preserves the union of keys written by concurrent workers, matching the
    SHARE: protocol semantics.
  - ``conversation_history`` uses LangGraph's ``add_messages`` which
    de-duplicates by message id.

Callers should NEVER import both state classes into the same node factory;
``graph_compiler.py`` picks one at compile time based on the flag.
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


def _merge_shared_ctx(left: dict[str, str] | None, right: dict[str, str] | None) -> dict[str, str]:
    """
    Reducer for ``shared_context`` — right-wins per key, union of keys.

    Matches the SHARE: protocol semantics used by ``merge_share`` in
    ``core.agent_factory``: concurrent workers may write disjoint keys and
    all of them are preserved; when two workers write the same key, the
    latest write wins (deterministic in single-wave fan-out because Send
    updates are applied in the order LangGraph receives them).
    """
    return {**(left or {}), **(right or {})}


class DynamicPipelineStateV2(TypedDict, total=False):
    """
    Reducer-aware V2 pipeline state.

    Mirrors the field set of ``DynamicPipelineState`` in
    ``dynamic_pipeline.py`` so node factories that only touch immutable
    fields work with either state class.
    """

    # ── Immutable / last-wins fields (identical to V1) ───────────────────
    user_query: str
    final_response: str
    direct_answer: str
    attachments: list[str]
    judge_metrics: dict[str, Any]
    budget_exceeded: bool
    research_for_analysis: bool
    research_for_analysis_query: str
    analysis_research_summary: str
    research_needed: bool
    research_query: str
    research_summary: str

    # ── Reducer-backed fields ────────────────────────────────────────────
    shared_context: Annotated[dict[str, str], _merge_shared_ctx]
    tasks: Annotated[list[dict[str, Any]], add]
    task_results: Annotated[list[dict[str, Any]], add]
    agent_assessments: Annotated[list[dict[str, Any]], add]
    conversation_history: Annotated[list[dict[str, str]], add_messages]
    quality_flagged_tasks: Annotated[list[str], add]


def current_wave_results(
    state: DynamicPipelineStateV2 | dict[str, Any], wave_idx: int
) -> list[dict[str, Any]]:
    """
    Filter ``task_results`` down to entries produced in ``wave_idx``.

    Because ``task_results`` now accumulates via a reducer, the wave_governor
    must not treat the whole accumulator as "this wave's output". Every
    result dict emitted by the worker fan-out carries a ``wave_index`` key;
    this helper returns only the slice matching the current wave.

    Missing ``wave_index`` is treated as wave 0 (first wave) so legacy
    results still surface for round-1 escalation decisions.
    """
    results: list[dict[str, Any]] = list(state.get("task_results") or [])
    return [r for r in results if int(r.get("wave_index", 0)) == int(wave_idx)]
