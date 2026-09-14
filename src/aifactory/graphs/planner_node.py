"""
planner_node — reducer-topology entry point that fans out workers via Send.

This node replaces the monolithic ``execution_lead`` in the V2 topology.
It reuses the lead agent's planning path (``lead.think``) — exactly the
same LLM call the legacy node makes — then, instead of running workers
via ``asyncio.gather`` inside a single node, it returns a list of
``Send("worker", ...)`` objects. LangGraph fans them out in parallel and
merges their state deltas back through the V2 reducers.

Dependency-wave scheduling is preserved by planning wave 0 up-front and
letting the wave_governor loop re-invoke the planner with the accumulated
``task_results`` in state; subsequent invocations advance one wave at a
time. When all waves are exhausted, the router edge ``route_after_planner``
directs execution to ``judge_review`` (or ``master_respond`` in STANDARD).

This module intentionally leaves the deepest orchestration paths
(auction, adaptive re-spawns, findings compression) as follow-up work
under the same feature flag — the scaffolding here is safe to enable and
covers the common wave-0 single-worker fan-out that the majority of runs
use.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from langgraph.types import Send

from aifactory.core.agent_factory import (
    extract_share,
    merge_share,
    parse_json_robust,
)
from aifactory.core.model_selector import ABILITIES as _ABILITY_KEYS

logger = logging.getLogger(__name__)


@dataclass
class ExecutionContext:
    """
    Container for the collaborators the planner and worker share.

    Instances are constructed once per pipeline run in
    ``graph_compiler._compile_reducer_topology`` and threaded through Send
    payloads so the individual worker nodes do not need to re-resolve their
    dependencies.
    """

    lead: Any
    registry: Any
    bridge: Any = None
    event_callback: Any = None
    budget_tracker: Any = None
    researcher: Any = None
    code_processor: Any = None
    models: list[Any] | None = None


async def _plan_once(
    ctx: ExecutionContext,
    tasks: list[dict[str, Any]],
    shared_ctx: dict[str, str],
    research: str,
) -> tuple[dict[str, str], dict[str, dict[str, float]], dict[str, str]]:
    """
    Ask the lead agent for worker prompts + ability requirements.

    Returns ``(worker_prompts, ability_reqs, updated_shared_ctx)``.
    Mirrors the light-weight planning path in ``make_execution_lead`` so we
    don't duplicate the full-fat plan cache logic — the reducer topology
    replans on every node invocation and lets the wave_governor loop
    control iteration.
    """
    worker_prompts: dict[str, str] = {}
    ability_reqs: dict[str, dict[str, float]] = {}

    if not tasks:
        return worker_prompts, ability_reqs, shared_ctx

    if len(tasks) == 1:
        tid = tasks[0]["id"]
        desc = tasks[0]["description"]
        research_hint = (
            f"\n\nResearch context to pass to worker:\n{research[:400]}" if research else ""
        )
        raw = await ctx.lead.think(
            f"Write a single optimised worker instruction for this task. "
            f"Plain prose only — no JSON, no structure.\n\n"
            f"Task: {desc}{research_hint}",
            shared_ctx,
        )
        prompt, deltas = extract_share(raw)
        shared_ctx = merge_share(shared_ctx, deltas, ctx.lead.name)
        worker_prompts = {tid: prompt.strip() or desc}
        ability_reqs = {tid: {a: 1.5 for a in _ABILITY_KEYS}}
        return worker_prompts, ability_reqs, shared_ctx

    tasks_list = "\n".join(f"  {t['id']}: {t['description']}" for t in tasks)
    research_block = (
        f"\nResearch summary (share with workers):\n{research[:500]}" if research else ""
    )
    plan_raw = await ctx.lead.think(
        f"Plan worker dispatch:{research_block}\n\n{tasks_list}\n\n"
        "JSON only — no prose:\n"
        '{"plan": [{"id": "task_1", "reasoning": 0-3, "coding": 0-3, '
        '"knowledge": 0-3, "writing": 0-3, "math": 0-3, '
        '"instruction_following": 0-3, "tool_use": 0-3, '
        '"speed": 0-3, "cost_efficiency": 0-3, "context_handling": 0-3}],\n'
        ' "prompts": {"task_1": "<concise worker instruction>"}}',
        shared_ctx,
    )
    try:
        parsed = parse_json_robust(plan_raw)
        for row in parsed.get("plan", []):
            tid = str(row.get("id", ""))
            ability_reqs[tid] = {a: float(row.get(a, 0.0)) for a in _ABILITY_KEYS}
        worker_prompts = {str(k): str(v) for k, v in parsed.get("prompts", {}).items()}
    except Exception as exc:
        logger.warning("planner_node: plan parse failed (%s) — using raw descriptions", exc)
        worker_prompts = {t["id"]: t["description"] for t in tasks}
        ability_reqs = {t["id"]: {a: 1.5 for a in _ABILITY_KEYS} for t in tasks}
    return worker_prompts, ability_reqs, shared_ctx


def make_planner_node(ctx: ExecutionContext, *, mode: str) -> Any:
    """
    Return a LangGraph node coroutine that plans a wave and stores the
    per-task worker prompts + ability requirements in ``shared_context``.

    Send fan-out itself is performed by :func:`plan_router` on the
    outgoing conditional edge — LangGraph 1.0 requires Send objects to
    come from an edge router, not a node body.
    """

    async def planner_node(state: dict[str, Any]) -> dict[str, Any]:
        tasks = list(state.get("tasks") or [])
        if not tasks:
            tasks = [{"id": "task_1", "description": state.get("user_query", "")}]
        shared_ctx: dict[str, str] = dict(state.get("shared_context") or {})
        research: str = state.get("research_summary", "") or ""

        worker_prompts, ability_reqs, shared_ctx = await _plan_once(
            ctx, tasks, shared_ctx, research
        )

        # Determine wave to run: max(wave_index in task_results) + 1, or 0.
        prior_results = list(state.get("task_results") or [])
        seen_wave = -1
        completed_ids: set[str] = set()
        for r in prior_results:
            seen_wave = max(seen_wave, int(r.get("wave_index", 0)))
            completed_ids.add(str(r.get("task_id", "")))
        current_wave = seen_wave + 1 if prior_results else 0

        pending = [t for t in tasks if t["id"] not in completed_ids]

        # Store the plan in shared_context so plan_router can construct Sends.
        # We stash the payloads on the module cache keyed by run_id + wave
        # to avoid inflating shared_context with non-serialisable objects
        # (ExecutionContext holds live agent handles).
        plan_key = f"__plan_{id(ctx)}_{current_wave}"
        _PLAN_CACHE[plan_key] = {
            "pending": pending,
            "worker_prompts": worker_prompts,
            "ability_reqs": ability_reqs,
            "mode": mode,
            "wave_index": current_wave,
            "worker_ctx": ctx,
        }
        logger.info(
            "planner_node: planned wave %d with %d pending task(s) (key=%s)",
            current_wave,
            len(pending),
            plan_key,
        )
        return {
            "shared_context": {**shared_ctx, "_plan_key": plan_key},
        }

    return planner_node


# Module-level cache — planner writes here, plan_router reads. Small, bounded
# by wave count; entries are consumed once per graph invocation. Non-picklable
# ``ExecutionContext`` handles live here rather than in graph state so LangGraph
# checkpoints stay JSON-serialisable.
_PLAN_CACHE: dict[str, dict[str, Any]] = {}


def plan_router(state: dict[str, Any]) -> list[Send] | str:
    """
    Conditional-edge router — reads the pending wave and emits Send objects.

    Returns the string ``"__end_wave__"`` when there is nothing to dispatch;
    callers should map that key to the next node (judge_review or
    master_respond).
    """
    ctx_dict = state.get("shared_context") or {}
    plan_key = ctx_dict.get("_plan_key")
    plan = _PLAN_CACHE.pop(plan_key, None) if plan_key else None
    if not plan or not plan["pending"]:
        return "__end_wave__"

    ctx = plan["worker_ctx"]
    mode = plan["mode"]
    wave_index = plan["wave_index"]
    shared_snapshot = {k: v for k, v in ctx_dict.items() if k != "_plan_key"}

    sends: list[Send] = []
    for task in plan["pending"]:
        tid = task["id"]
        sends.append(
            Send(
                "worker",
                {
                    "task": task,
                    "worker_prompt": plan["worker_prompts"].get(tid, task["description"]),
                    "ability_req": plan["ability_reqs"].get(tid, {}),
                    "toolkit_assignment": [],
                    "tool_slug_assignment": [],
                    "mode": mode,
                    "wave_index": wave_index,
                    "worker_ctx": ctx,
                    "shared_context": shared_snapshot,
                    "attachments": state.get("attachments") or [],
                    "task_results": list(state.get("task_results") or []),
                },
            )
        )
    return sends
