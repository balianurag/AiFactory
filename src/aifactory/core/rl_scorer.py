"""
RL-lite Scorer — adaptive model selection from historical performance.

Records (model_id, task_type) → quality scores after each pipeline run.
The judge_review node writes records; create_worker_config() reads them
to give proven models a scoring boost in the next run.

Storage: PostgreSQL `rl_performance` table (via database.py sync engine).
Fallback: in-memory dict when DATABASE_URL is not set (same behaviour as
the previous Qdrant in-memory fallback, but without a dummy vector dimension).

Task types: coding, math, research, writing, analysis, tool_use, general
Score range: 0–10 (normalised from judge's 0–3 quality rating × 10/3)

Design:
  - record_performance()          ← called by judge_review after each run
  - get_model_rl_scores()         ← called by create_worker_config() before selection
  - record_pipeline_performances()← convenience batch wrapper
  - get_all_stats()               ← CLI display helper
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ── Task type inference ───────────────────────────────────────────────────────

_TASK_TYPE_KEYWORDS: dict[str, list[str]] = {
    "coding": [
        "code", "implement", "function", "class", "algorithm", "script",
        "python", "javascript", "typescript", "write a", "build a", "sort",
        "parse", "refactor", "test", "unit test", "debug", "fix bug", "compile",
    ],
    "math": [
        "calculate", "compute", "solve", "math", "equation", "formula", "proof",
        "derivative", "integral", "matrix", "statistics", "probability",
        "calculus", "algebra",
    ],
    "research": [
        "research", "find", "search", "gather", "investigate", "look up",
        "what is", "how does", "explain", "overview", "summarize about",
        "tell me about",
    ],
    "writing": [
        "write", "draft", "compose", "essay", "report", "document", "blog post",
        "email draft", "summarize", "describe", "narrate",
    ],
    "analysis": [
        "analyze", "analyse", "compare", "evaluate", "assess", "review",
        "critique", "pros and cons", "trade-offs", "benchmark", "audit",
    ],
    "tool_use": [
        "github", "gitlab", "jira", "slack", "email", "create issue",
        "pull request", "send message", "create ticket", "post to", "teams",
        "atlassian", "confluence", "notion", "linear",
    ],
}


def infer_task_type(description: str) -> str:
    """
    Infer a task type from a natural-language description.

    Returns one of: coding, math, research, writing, analysis, tool_use, general.
    Picks the type with the most keyword hits; ties go to the first match.
    """
    desc_lower = description.lower()
    scores: dict[str, int] = {
        tt: sum(1 for kw in keywords if kw in desc_lower)
        for tt, keywords in _TASK_TYPE_KEYWORDS.items()
    }
    best_type = max(scores, key=lambda t: scores[t])
    return best_type if scores[best_type] > 0 else "general"


# ── In-memory fallback (used when PostgreSQL is unavailable) ──────────────────
# List of (model_id, task_type, quality_score) tuples

_mem_records: list[dict[str, Any]] = []


# ── Core API ─────────────────────────────────────────────────────────────────


def record_performance(
    model_id: str,
    task_type: str,
    quality_score: float,
    tokens_used: int = 0,
    run_id: str = "",
) -> None:
    """
    Store one performance record for a (model_id, task_type) pair.

    Called by judge_review after each pipeline run.  Non-fatal — any
    storage error is logged at DEBUG level and swallowed.

    Args:
        model_id:      LiteLLM model identifier.
        task_type:     One of coding, math, research, writing, analysis,
                       tool_use, general.
        quality_score: 0–10 (normalise from judge's 0–3 via × 10/3).
        tokens_used:   Worker token count (informational).
        run_id:        Pipeline run identifier for traceability.
    """
    try:
        from aifactory.core.database import insert_rl_record

        insert_rl_record(
            model_id=model_id,
            task_type=task_type,
            quality_score=quality_score,
            tokens_used=tokens_used,
            run_id=run_id,
        )
        logger.debug(
            "RL: recorded model=%s task_type=%s score=%.1f",
            model_id, task_type, quality_score,
        )
    except Exception as exc:
        logger.debug("RL: record_performance DB failed — storing in-memory: %s", exc)
        _mem_records.append({
            "model_id": model_id,
            "task_type": task_type,
            "quality_score": float(quality_score),
        })


def get_model_rl_scores(
    task_type: str,
    min_samples: int = 3,
) -> dict[str, float]:
    """
    Return average quality scores (0–10) keyed by model_id for a task type.

    Only includes models that have at least `min_samples` records.
    Returns empty dict if no qualifying records exist.
    """
    try:
        from aifactory.core.database import get_sync_engine, query_rl_scores

        if get_sync_engine() is not None:
            return query_rl_scores(task_type, min_samples)
    except Exception as exc:
        logger.debug("RL: get_model_rl_scores DB failed: %s", exc)

    # In-memory fallback
    if not _mem_records:
        return {}
    filtered = [r for r in _mem_records if r["task_type"] == task_type]
    model_lists: dict[str, list[float]] = {}
    for r in filtered:
        model_lists.setdefault(r["model_id"], []).append(r["quality_score"])
    return {
        mid: round(sum(scores) / len(scores), 3)
        for mid, scores in model_lists.items()
        if len(scores) >= min_samples
    }


def get_all_stats() -> dict[str, Any]:
    """
    Return aggregated RL stats for CLI display.

    Returned dict keys:
      total_records: int
      by_task_type:  dict[task_type → {records, models, best_model, best_avg_score}]
    """
    try:
        from aifactory.core.database import get_all_rl_stats, get_sync_engine

        if get_sync_engine() is not None:
            return get_all_rl_stats()
    except Exception as exc:
        logger.debug("RL: get_all_stats DB failed: %s", exc)

    # In-memory fallback
    if not _mem_records:
        return {"total_records": 0, "by_task_type": {}}

    by_type: dict[str, dict[str, list[float]]] = {}
    for r in _mem_records:
        by_type.setdefault(r["task_type"], {}).setdefault(r["model_id"], []).append(r["quality_score"])

    summary: dict[str, Any] = {}
    for tt, model_score_lists in by_type.items():
        model_avgs = {mid: sum(s) / len(s) for mid, s in model_score_lists.items()}
        best = max(model_avgs, key=lambda m: model_avgs[m])
        summary[tt] = {
            "records": sum(len(s) for s in model_score_lists.values()),
            "models": len(model_avgs),
            "best_model": best,
            "best_avg_score": round(model_avgs[best], 2),
            "model_avgs": {mid: round(v, 2) for mid, v in model_avgs.items()},
        }
    return {"total_records": len(_mem_records), "by_task_type": summary}


def record_pipeline_performances(
    task_results: list[dict[str, Any]],
    assessments: list[dict[str, Any]],
    run_id: str = "",
) -> None:
    """
    Batch-record worker performances from a completed pipeline run.

    Called at the end of judge_review. Matches task_results ↔ assessments
    by agent name, normalises quality 0–3 → 0–10, and records each worker.
    Skipped workers (NOT_NEEDED) are excluded. Non-fatal.
    """
    try:
        from aifactory.config.settings import get_settings as _gs

        if not _gs().rl_enabled:
            return
    except Exception:
        pass

    assess_by_agent: dict[str, dict[str, Any]] = {a["agent"]: a for a in assessments}

    for tr in task_results:
        agent_name = tr.get("agent", "")
        model_id = tr.get("model", "")
        description = tr.get("description", "")

        if not model_id or tr.get("skipped"):
            continue

        assessment = assess_by_agent.get(agent_name, {})
        raw_quality = assessment.get("quality", "1")
        try:
            quality_score = float(raw_quality) * (10.0 / 3.0)
        except (ValueError, TypeError):
            quality_score = 5.0

        tokens_used = 0
        with contextlib.suppress(ValueError, TypeError):
            tokens_used = int(assessment.get("tokens", 0))

        task_type = infer_task_type(description)
        record_performance(
            model_id=model_id,
            task_type=task_type,
            quality_score=quality_score,
            tokens_used=tokens_used,
            run_id=run_id,
        )
        logger.info(
            "RL: %s  model=%s  task_type=%s  quality=%.1f/10",
            agent_name, model_id, task_type, quality_score,
        )
