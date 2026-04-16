"""
Run Store — persistent record of every pipeline execution.

Stores plan, model selections, quality outcomes, and task results for each run
in a Qdrant collection ("aifactory_run_store").  This powers:
  - The meta-agent (C1): reads run history to recommend system improvements
  - Replay on failure: re-run from persisted plan without re-planning
  - Similarity search: find prior runs for similar queries (cross-run learning)
  - CLI: aifactory runs list / show / search

Fallback: when Qdrant is unavailable, stores to JSONL file at
~/.aifactory/run_store.jsonl (append-only, lightweight).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Qdrant collection for run records
_COLLECTION = "aifactory_run_store"

# Fallback JSONL file when Qdrant is unavailable
_FALLBACK_FILE = Path.home() / ".aifactory" / "run_store.jsonl"

# Max chars stored for query/response in Qdrant payload (not the embedding)
_MAX_TEXT_CHARS = 2000


@dataclass
class RunRecord:
    """Complete record of a single pipeline run."""

    run_id: str
    timestamp: str
    query: str
    mode: str
    elapsed_seconds: float
    gate: str  # "bypass" | "fast_llm" | ""
    model_selections: dict[str, str]  # {role: model_id}
    task_count: int
    quality_flagged_count: int
    composite_quality: float  # 0–10, from judge_metrics
    task_types: list[str]  # inferred task types
    final_response_preview: str  # first 500 chars of final response
    quality_per_task: list[dict[str, Any]] = field(default_factory=list)
    plan: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunRecord:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class RunStore:
    """
    Persist and retrieve pipeline run records.

    Usage:
        store = RunStore()
        store.save(record)
        records = store.list_recent(n=10)
        similar = store.search_similar("build a REST API", top_k=3)
    """

    def __init__(self) -> None:
        self._qdrant: Any = None
        self._embed_fn: Any = None
        self._dim: int = 1536
        self._init_backends()

    # ── Initialisation ────────────────────────────────────────────────

    def _init_backends(self) -> None:
        """Try to initialise Qdrant; fall back silently to JSONL."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            from aifactory.config.settings import get_settings
            from aifactory.core.memory_router import _embed_stub

            s = get_settings()
            try:
                client = QdrantClient(
                    url=s.qdrant_url,
                    api_key=s.qdrant_api_key or None,
                    timeout=5,
                )
                # Verify connectivity
                client.get_collections()
                self._qdrant = client
            except Exception:
                # Fall back to in-memory Qdrant
                client = QdrantClient(":memory:")
                self._qdrant = client

            # Ensure collection exists
            existing = [c.name for c in self._qdrant.get_collections().collections]
            if _COLLECTION not in existing:
                self._qdrant.create_collection(
                    collection_name=_COLLECTION,
                    vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
                )
            self._embed_fn = _embed_stub
            logger.debug("RunStore: Qdrant backend initialised (%s)", _COLLECTION)
        except Exception as exc:
            logger.debug("RunStore: Qdrant unavailable, using JSONL fallback: %s", exc)

    # ── Public API ────────────────────────────────────────────────────

    def save(self, record: RunRecord) -> None:
        """Persist a run record.  Fail-safe — never raises."""
        try:
            self._save_qdrant(record)
        except Exception as exc:
            logger.debug("RunStore Qdrant save failed: %s — writing to JSONL", exc)
            self._save_jsonl(record)

    def list_recent(self, n: int = 20) -> list[RunRecord]:
        """Return up to n most recent runs (newest first)."""
        try:
            if self._qdrant:
                return self._list_qdrant(n)
        except Exception as exc:
            logger.debug("RunStore Qdrant list failed: %s", exc)
        return self._list_jsonl(n)

    def search_similar(self, query: str, top_k: int = 5) -> list[RunRecord]:
        """Return runs with queries semantically similar to the given text."""
        try:
            if self._qdrant and self._embed_fn:
                return self._search_qdrant(query, top_k)
        except Exception as exc:
            logger.debug("RunStore similarity search failed: %s", exc)
        return []

    def get(self, run_id: str) -> RunRecord | None:
        """Retrieve a specific run by ID."""
        try:
            if self._qdrant:
                results, _ = self._qdrant.scroll(
                    collection_name=_COLLECTION,
                    scroll_filter=self._id_filter(run_id),
                    limit=1,
                    with_payload=True,
                    with_vectors=False,
                )
                if results:
                    return RunRecord.from_dict(results[0].payload)
        except Exception:
            pass
        # Fallback: scan JSONL
        for rec in self._list_jsonl(n=10_000):
            if rec.run_id == run_id:
                return rec
        return None

    # ── Internal: Qdrant ─────────────────────────────────────────────

    def _save_qdrant(self, record: RunRecord) -> None:
        from qdrant_client.models import PointStruct

        vec = self._embed_fn(record.query[:500]) if self._embed_fn else [0.0] * self._dim
        payload = record.to_dict()
        # Truncate large fields to stay within Qdrant payload limits
        payload["final_response_preview"] = payload["final_response_preview"][:500]
        payload["plan"] = payload["plan"][:20]  # keep at most 20 tasks

        import hashlib

        point_id = int(hashlib.md5(record.run_id.encode()).hexdigest()[:8], 16)

        self._qdrant.upsert(
            collection_name=_COLLECTION,
            points=[PointStruct(id=point_id, vector=vec, payload=payload)],
        )

    def _list_qdrant(self, n: int) -> list[RunRecord]:
        results, _ = self._qdrant.scroll(
            collection_name=_COLLECTION,
            limit=n,
            with_payload=True,
            with_vectors=False,
        )
        records = [RunRecord.from_dict(r.payload) for r in results]
        records.sort(key=lambda r: r.timestamp, reverse=True)
        return records[:n]

    def _search_qdrant(self, query: str, top_k: int) -> list[RunRecord]:
        vec = self._embed_fn(query[:500])
        hits = self._qdrant.search(
            collection_name=_COLLECTION,
            query_vector=vec,
            limit=top_k,
            with_payload=True,
        )
        return [RunRecord.from_dict(h.payload) for h in hits]

    @staticmethod
    def _id_filter(run_id: str) -> Any:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        return Filter(must=[FieldCondition(key="run_id", match=MatchValue(value=run_id))])

    # ── Internal: JSONL fallback ──────────────────────────────────────

    def _save_jsonl(self, record: RunRecord) -> None:
        try:
            _FALLBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
            with _FALLBACK_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record.to_dict()) + "\n")
        except Exception as exc:
            logger.debug("RunStore JSONL write failed: %s", exc)

    def _list_jsonl(self, n: int) -> list[RunRecord]:
        records: list[RunRecord] = []
        try:
            if not _FALLBACK_FILE.exists():
                return []
            lines = _FALLBACK_FILE.read_text(encoding="utf-8").strip().splitlines()
            for line in reversed(lines[-n * 2 :]):
                try:
                    records.append(RunRecord.from_dict(json.loads(line)))
                except Exception:
                    continue
                if len(records) >= n:
                    break
        except Exception:
            pass
        return records


# ── Module-level singleton ────────────────────────────────────────────────────

_store: RunStore | None = None


def get_run_store() -> RunStore:
    """Lazy singleton — created on first call."""
    global _store
    if _store is None:
        _store = RunStore()
    return _store


# ── Helper: build RunRecord from pipeline state ───────────────────────────────


def build_run_record(
    run_id: str,
    query: str,
    mode: str,
    elapsed: float,
    gate: str,
    state: dict[str, Any],
    registry: Any,
) -> RunRecord:
    """
    Construct a RunRecord from the pipeline's final state dict.

    Called by write_dump() and make_judge_review() after the run completes.
    """
    task_results = state.get("task_results", [])
    judge_metrics = state.get("judge_metrics", {})

    quality_flagged_count = sum(1 for t in task_results if t.get("quality_flagged"))
    composite_quality = float(judge_metrics.get("composite_quality", 10.0))

    # Gather model selections from the registry
    model_selections: dict[str, str] = {}
    try:
        for agent in registry.all():
            model_selections[agent.name] = agent.model
    except Exception:
        pass

    # Infer task types from task descriptions
    task_types: list[str] = []
    for task in state.get("tasks", []):
        desc = task.get("description", "").lower()
        if any(
            k in desc for k in ("code", "implement", "function", "class", "algorithm", "script")
        ):
            task_types.append("coding")
        elif any(k in desc for k in ("research", "investigate", "find", "search", "look up")):
            task_types.append("research")
        elif any(k in desc for k in ("write", "draft", "compose", "summarize", "document")):
            task_types.append("writing")
        elif any(k in desc for k in ("analyze", "analyse", "evaluate", "assess", "compare")):
            task_types.append("analysis")
        elif any(k in desc for k in ("calculate", "compute", "solve", "math", "formula")):
            task_types.append("math")
        else:
            task_types.append("general")

    final_response = state.get("final_response", "")

    return RunRecord(
        run_id=run_id,
        timestamp=datetime.now(UTC).isoformat(),
        query=query[:_MAX_TEXT_CHARS],
        mode=mode,
        elapsed_seconds=round(elapsed, 2),
        gate=gate,
        model_selections=model_selections,
        task_count=len(task_results),
        quality_flagged_count=quality_flagged_count,
        composite_quality=composite_quality,
        task_types=list(set(task_types)),
        final_response_preview=final_response[:500],
        quality_per_task=[
            {
                "task_id": t.get("task_id", ""),
                "quality_flagged": t.get("quality_flagged", False),
                "model": t.get("model", ""),
            }
            for t in task_results
        ],
        plan=state.get("tasks", [])[:20],
    )
