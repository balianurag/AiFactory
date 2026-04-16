"""
RL-lite Scorer — adaptive model selection from historical performance.

Records (model_id, task_type) → quality scores after each pipeline run.
The judge_review node writes records; create_worker_config() reads them
to give proven models a scoring boost in the next run.

Storage: dedicated Qdrant collection "aifactory_rl_performance".
Uses the same Qdrant client settings as MemoryRouter (remote → in-memory fallback).

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
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

# ── Collection constants ──────────────────────────────────────────────────────

_RL_COLLECTION = "aifactory_rl_performance"
_VECTOR_DIM = 1  # dummy — all queries use payload filters, not vector search

# ── Task type inference ───────────────────────────────────────────────────────

_TASK_TYPE_KEYWORDS: dict[str, list[str]] = {
    "coding": [
        "code",
        "implement",
        "function",
        "class",
        "algorithm",
        "script",
        "python",
        "javascript",
        "typescript",
        "write a",
        "build a",
        "sort",
        "parse",
        "refactor",
        "test",
        "unit test",
        "debug",
        "fix bug",
        "compile",
    ],
    "math": [
        "calculate",
        "compute",
        "solve",
        "math",
        "equation",
        "formula",
        "proof",
        "derivative",
        "integral",
        "matrix",
        "statistics",
        "probability",
        "calculus",
        "algebra",
    ],
    "research": [
        "research",
        "find",
        "search",
        "gather",
        "investigate",
        "look up",
        "what is",
        "how does",
        "explain",
        "overview",
        "summarize about",
        "tell me about",
    ],
    "writing": [
        "write",
        "draft",
        "compose",
        "essay",
        "report",
        "document",
        "blog post",
        "email draft",
        "summarize",
        "describe",
        "narrate",
    ],
    "analysis": [
        "analyze",
        "analyse",
        "compare",
        "evaluate",
        "assess",
        "review",
        "critique",
        "pros and cons",
        "trade-offs",
        "benchmark",
        "audit",
    ],
    "tool_use": [
        "github",
        "gitlab",
        "jira",
        "slack",
        "email",
        "create issue",
        "pull request",
        "send message",
        "create ticket",
        "post to",
        "teams",
        "atlassian",
        "confluence",
        "notion",
        "linear",
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


# ── Qdrant client management ──────────────────────────────────────────────────

_rl_client: Any = None  # QdrantClient — module-level singleton


def _build_rl_client() -> Any:
    """
    Construct a Qdrant client for RL storage.

    Tries the configured remote URL first; falls back to in-memory if
    the server is unreachable (mirrors MemoryRouter's fallback pattern).
    """
    from qdrant_client import QdrantClient  # type: ignore[import]
    from qdrant_client.models import Distance, VectorParams  # type: ignore[import]

    from aifactory.config.settings import get_settings

    settings = get_settings()
    client: QdrantClient

    if settings.qdrant_url.startswith("http"):
        try:
            client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key or None,
            )
            client.get_collections()  # connectivity probe
        except Exception:
            logger.warning(
                "RL scorer: Qdrant unreachable at %s — falling back to in-memory",
                settings.qdrant_url,
            )
            client = QdrantClient(":memory:")
    else:
        client = QdrantClient(":memory:")

    # Ensure the RL collection exists
    existing_names = {c.name for c in client.get_collections().collections}
    if _RL_COLLECTION not in existing_names:
        client.create_collection(
            _RL_COLLECTION,
            vectors_config=VectorParams(size=_VECTOR_DIM, distance=Distance.COSINE),
        )
        logger.debug("RL scorer: created collection '%s'", _RL_COLLECTION)

    return client


def _get_rl_client() -> Any:
    """Return (and lazily initialise) the module-level Qdrant client."""
    global _rl_client
    if _rl_client is None:
        _rl_client = _build_rl_client()
    return _rl_client


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

    Called by judge_review after each pipeline run. Non-fatal — any
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
        from qdrant_client.models import PointStruct  # type: ignore[import]

        client = _get_rl_client()
        client.upsert(
            collection_name=_RL_COLLECTION,
            points=[
                PointStruct(
                    id=str(uuid4()),
                    vector=[0.0],
                    payload={
                        "model_id": model_id,
                        "task_type": task_type,
                        "quality_score": float(quality_score),
                        "tokens_used": int(tokens_used),
                        "run_id": run_id,
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                )
            ],
        )
        logger.debug(
            "RL: recorded model=%s task_type=%s score=%.1f",
            model_id,
            task_type,
            quality_score,
        )
    except Exception as exc:
        logger.debug("RL: record_performance failed (non-fatal): %s", exc)


def get_model_rl_scores(
    task_type: str,
    min_samples: int = 3,
) -> dict[str, float]:
    """
    Return average quality scores (0–10) keyed by model_id for a task type.

    Only includes models that have at least `min_samples` records — prevents
    a single lucky/unlucky run from dominating selection.

    Returns empty dict if Qdrant is unavailable or no qualifying records exist.
    """
    try:
        from qdrant_client.models import FieldCondition, Filter, MatchValue  # type: ignore[import]

        client = _get_rl_client()
        all_records: list[dict[str, Any]] = []
        offset = None

        while True:
            results, next_offset = client.scroll(
                collection_name=_RL_COLLECTION,
                scroll_filter=Filter(
                    must=[FieldCondition(key="task_type", match=MatchValue(value=task_type))]
                ),
                limit=200,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            all_records.extend(r.payload for r in results if r.payload)
            if next_offset is None:
                break
            offset = next_offset

        # Aggregate per model
        model_score_lists: dict[str, list[float]] = {}
        for rec in all_records:
            mid = rec.get("model_id", "")
            score = rec.get("quality_score", 0.0)
            if mid:
                model_score_lists.setdefault(mid, []).append(float(score))

        return {
            mid: round(sum(scores) / len(scores), 3)
            for mid, scores in model_score_lists.items()
            if len(scores) >= min_samples
        }
    except Exception as exc:
        logger.debug("RL: get_model_rl_scores failed (non-fatal): %s", exc)
        return {}


def get_all_stats() -> dict[str, Any]:
    """
    Return aggregated RL stats for CLI display.

    Returned dict keys:
      total_records: int
      by_task_type:  dict[task_type → {records, models, best_model, best_avg_score}]
    """
    try:
        client = _get_rl_client()
        total = client.count(collection_name=_RL_COLLECTION, exact=True).count

        if total == 0:
            return {"total_records": 0, "by_task_type": {}}

        # Fetch all records (capped at 2000 for CLI display)
        all_records: list[dict[str, Any]] = []
        offset = None
        while True:
            results, next_offset = client.scroll(
                collection_name=_RL_COLLECTION,
                limit=500,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            all_records.extend(r.payload for r in results if r.payload)
            if next_offset is None or len(all_records) >= 2000:
                break
            offset = next_offset

        # Aggregate by task_type → model_id → scores
        by_type: dict[str, dict[str, list[float]]] = {}
        for rec in all_records:
            tt = rec.get("task_type", "general")
            mid = rec.get("model_id", "unknown")
            score = float(rec.get("quality_score", 0.0))
            by_type.setdefault(tt, {}).setdefault(mid, []).append(score)

        summary: dict[str, Any] = {}
        for tt, model_score_lists in by_type.items():
            model_avgs = {
                mid: sum(scores) / len(scores) for mid, scores in model_score_lists.items()
            }
            best = max(model_avgs, key=lambda m: model_avgs[m])
            summary[tt] = {
                "records": sum(len(s) for s in model_score_lists.values()),
                "models": len(model_avgs),
                "best_model": best,
                "best_avg_score": round(model_avgs[best], 2),
                "model_avgs": {mid: round(v, 2) for mid, v in model_avgs.items()},
            }

        return {"total_records": total, "by_task_type": summary}
    except Exception as exc:
        logger.debug("RL: get_all_stats failed: %s", exc)
        return {"total_records": 0, "by_task_type": {}, "error": str(exc)}


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
            # Judge quality is 0–3; normalise to 0–10
            quality_score = float(raw_quality) * (10.0 / 3.0)
        except (ValueError, TypeError):
            quality_score = 5.0  # neutral default

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
            agent_name,
            model_id,
            task_type,
            quality_score,
        )
