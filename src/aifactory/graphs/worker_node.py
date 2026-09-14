"""
worker_node — a single worker rendered as a first-class LangGraph node.

Used by the reducer-enabled V2 topology as the fan-out target of Send()
objects emitted by :mod:`aifactory.graphs.planner_node`. Each Send delivers
a per-worker slice of state (`task`, `worker_prompt`, `ability_req`, etc.);
this node executes it using the same helpers the legacy ``execution_lead``
uses, then returns a state delta that the reducer merges back into the
canonical accumulator lists.

The heavy lifting (SHARE / FINDINGS / RESEARCH / CODE_ANALYSIS parsing,
self-check prompts, retry, budget, sandbox) is NOT reimplemented here — we
delegate to :func:`aifactory.graphs.dynamic_pipeline._run_worker_task`.
The one wrapper responsibility is to package the returned tuple into a
state delta whose lists are single-element (the reducer union merges them
with concurrent Send peers).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


async def worker_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Execute a single worker task delivered via the Send API.

    Expected state keys (populated by planner_node's Send payload):
      * ``task``               – task dict {"id", "description", ...}
      * ``worker_prompt``      – prompt string for this worker
      * ``ability_req``        – dict of ability → weight
      * ``toolkit_assignment`` – list of toolkit slugs
      * ``tool_slug_assignment``– list of specific tool slugs
      * ``mode``               – pipeline mode string
      * ``wave_index``         – 0-based wave index
      * ``worker_ctx``         – ExecutionContext object (see planner_node)

    Returns a state delta with single-element lists that reducers merge
    into the accumulators.
    """
    task = state.get("task") or {}
    prompt = state.get("worker_prompt") or task.get("description", "")
    ability_req = state.get("ability_req") or {}
    toolkits = state.get("toolkit_assignment") or []
    tool_slugs = state.get("tool_slug_assignment") or []
    mode = state.get("mode") or "balanced"
    wave_index = int(state.get("wave_index", 0))
    ctx_obj = state.get("worker_ctx")
    shared_ctx: dict[str, str] = dict(state.get("shared_context") or {})

    if not task or ctx_obj is None:
        logger.warning("worker_node invoked without task/worker_ctx — no-op")
        return {}

    # Late import to keep this module import-cheap during graph compile.
    from aifactory.graphs.dynamic_pipeline import _run_worker_task

    tid = str(task.get("id", "unknown"))
    completed: dict[str, dict[str, Any]] = {}
    workers_cache: dict[str, Any] = {}
    structural_index_cache: dict[str, str] = {}
    structural_index_lock = asyncio.Lock()

    try:
        entry, ctx_delta = await _run_worker_task(
            task,
            shared_ctx,
            worker_prompts={tid: prompt},
            ability_reqs={tid: dict(ability_req) if ability_req else {}},
            toolkit_assignments={tid: list(toolkits)},
            tool_slug_assignments={tid: list(tool_slugs)},
            completed=completed,
            workers_cache=workers_cache,
            prior_results=list(state.get("task_results") or []),
            mode=mode,
            registry=ctx_obj.registry,
            bridge=ctx_obj.bridge,
            event_callback=ctx_obj.event_callback,
            budget_tracker=ctx_obj.budget_tracker,
            attachments=state.get("attachments"),
            researcher=ctx_obj.researcher,
            findings_context="",
            code_processor=ctx_obj.code_processor,
            structural_index_cache=structural_index_cache,
            structural_index_lock=structural_index_lock,
            otel_parent_context=None,
            wave_index=wave_index,
        )
    except Exception as exc:
        logger.exception("worker_node[%s] failed: %s", tid, exc)
        return {
            "task_results": [
                {
                    "task_id": tid,
                    "agent": f"Worker-{tid}",
                    "model": "unknown",
                    "description": task.get("description", ""),
                    "result": f"[Worker error: {exc!r}]",
                    "skipped": False,
                    "quality_flagged": False,
                    "wave_index": wave_index,
                }
            ],
        }

    # Stamp the wave index so wave_governor / current_wave_results can filter.
    entry.setdefault("wave_index", wave_index)

    delta: dict[str, Any] = {"shared_context": ctx_delta or {}}
    if not entry.get("skipped"):
        delta["task_results"] = [entry]
    if entry.get("quality_flagged"):
        delta["quality_flagged_tasks"] = [tid]
    return delta
