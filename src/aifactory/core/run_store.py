"""
Run Store — persistent record of every pipeline execution.

Primary backend: PostgreSQL `runs` table (via database.py sync engine).
Fallback: JSONL file at ~/.aifactory/run_store.jsonl (append-only).

Semantic similarity search still uses Qdrant (via the shared singleton from
qdrant_pool.py) because PostgreSQL has no native vector search.  All other
operations (save, list, get) route through PostgreSQL.

Powers:
  - Meta-agent (aifactory meta review) — reads run history
  - CLI (aifactory runs list / show / search)
  - Auto-calibration trigger (every N runs)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Fallback JSONL file when PostgreSQL is unavailable
_FALLBACK_FILE = Path.home() / ".aifactory" / "run_store.jsonl"

# Max chars stored for query/response payload
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
        self._pg: bool = False
        self._qdrant: Any = None
        self._embed_fn: Any = None
        self._dim: int = 1536
        self._init_backends()

    # ── Initialisation ────────────────────────────────────────────────

    def _init_backends(self) -> None:
        """Initialise PostgreSQL (primary) and Qdrant (similarity search only)."""
        # PostgreSQL
        try:
            from aifactory.core.database import get_sync_engine

            if get_sync_engine() is not None:
                self._pg = True
                logger.debug("RunStore: PostgreSQL backend ready")
        except Exception as exc:
            logger.debug("RunStore: PostgreSQL unavailable: %s", exc)

        # Qdrant — for search_similar() only
        try:
            from qdrant_client.models import Distance, VectorParams

            from aifactory.core.memory_router import _stub_embed
            from aifactory.core.qdrant_pool import get_qdrant

            _COLLECTION = "aifactory_run_store"
            client = get_qdrant()
            existing = [c.name for c in client.get_collections().collections]
            if _COLLECTION not in existing:
                client.create_collection(
                    collection_name=_COLLECTION,
                    vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
                )
            self._qdrant = client
            self._embed_fn = _stub_embed
            logger.debug("RunStore: Qdrant similarity search ready")
        except Exception as exc:
            logger.debug("RunStore: Qdrant unavailable for similarity search: %s", exc)

    # ── Public API ────────────────────────────────────────────────────

    def save(self, record: RunRecord) -> None:
        """Persist a run record.  Fail-safe — never raises."""
        saved = False
        if self._pg:
            try:
                self._save_pg(record)
                saved = True
            except Exception as exc:
                logger.debug("RunStore PG save failed: %s — writing to JSONL", exc)
        if not saved:
            self._save_jsonl(record)

        # Also upsert to Qdrant for similarity search (best-effort)
        if self._qdrant and self._embed_fn:
            try:
                self._save_qdrant(record)
            except Exception as exc:
                logger.debug("RunStore Qdrant save failed (non-fatal): %s", exc)

    def list_recent(self, n: int = 20) -> list[RunRecord]:
        """Return up to n most recent runs (newest first)."""
        if self._pg:
            try:
                return self._list_pg(n)
            except Exception as exc:
                logger.debug("RunStore PG list failed: %s", exc)
        return self._list_jsonl(n)

    def search_similar(self, query: str, top_k: int = 5) -> list[RunRecord]:
        """Return runs with queries semantically similar to the given text."""
        if self._qdrant and self._embed_fn:
            try:
                return self._search_qdrant(query, top_k)
            except Exception as exc:
                logger.debug("RunStore similarity search failed: %s", exc)
        return []

    def get(self, run_id: str) -> RunRecord | None:
        """Retrieve a specific run by ID."""
        if self._pg:
            try:
                from aifactory.core.database import get_run_pg

                row = get_run_pg(run_id)
                if row:
                    return self._row_to_record(row)
            except Exception:
                pass
        # Fallback: scan JSONL
        for rec in self._list_jsonl(n=10_000):
            if rec.run_id == run_id:
                return rec
        return None

    # ── Internal: PostgreSQL ──────────────────────────────────────────

    def _save_pg(self, record: RunRecord) -> None:
        """
        Write to the `runs` table.  The full RunRecord is serialised into
        the `metadata` JSONB column; the key scalar fields are lifted to
        dedicated columns for indexed queries.
        """
        from aifactory.core.database import save_run, update_run_status

        # save_run inserts the row (ON CONFLICT DO NOTHING — may already exist)
        save_run(
            run_id=record.run_id,
            source="cli",
            task=record.query,
            mode=record.mode,
            gate=record.gate,
            status="completed",
            metadata=record.to_dict(),
        )
        # Update the completion columns
        update_run_status(
            run_id=record.run_id,
            status="completed",
            elapsed_s=record.elapsed_seconds,
            total_tokens=sum(r.get("tokens_used", 0) for r in record.quality_per_task),
            final_preview=record.final_response_preview[:500],
            quality=record.composite_quality,
        )

    def _list_pg(self, n: int) -> list[RunRecord]:
        from aifactory.core.database import list_runs_pg

        rows = list_runs_pg(limit=n)
        return [self._row_to_record(r) for r in rows]

    @staticmethod
    def _row_to_record(row: dict[str, Any]) -> RunRecord:
        """Reconstruct a RunRecord from a `runs` table row."""
        meta = row.get("metadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}

        # If full RunRecord dict is stored in metadata, use it directly
        if meta and "run_id" in meta:
            try:
                return RunRecord.from_dict(meta)
            except Exception:
                pass

        # Otherwise reconstruct from scalar columns (minimal record)
        started = row.get("started_at", "")
        if hasattr(started, "isoformat"):
            started = started.isoformat()

        return RunRecord(
            run_id=row.get("run_id", ""),
            timestamp=str(started),
            query=row.get("task", ""),
            mode=row.get("mode", "balanced"),
            elapsed_seconds=float(row.get("elapsed_s") or 0),
            gate=row.get("gate") or "",
            model_selections={},
            task_count=0,
            quality_flagged_count=0,
            composite_quality=float(row.get("quality") or 0),
            task_types=[],
            final_response_preview=row.get("final_preview") or "",
        )

    # ── Internal: Qdrant (similarity search only) ─────────────────────

    _COLLECTION = "aifactory_run_store"

    def _save_qdrant(self, record: RunRecord) -> None:
        import hashlib

        from qdrant_client.models import PointStruct

        vec = self._embed_fn(record.query[:500])
        payload = record.to_dict()
        payload["final_response_preview"] = payload["final_response_preview"][:500]
        payload["plan"] = payload["plan"][:20]

        point_id = int(hashlib.sha256(record.run_id.encode()).hexdigest()[:8], 16)
        self._qdrant.upsert(
            collection_name=self._COLLECTION,
            points=[PointStruct(id=point_id, vector=vec, payload=payload)],
        )

    def _search_qdrant(self, query: str, top_k: int) -> list[RunRecord]:
        vec = self._embed_fn(query[:500])
        response = self._qdrant.query_points(
            collection_name=self._COLLECTION,
            query=vec,
            limit=top_k,
            with_payload=True,
        )
        hits = response.points if hasattr(response, "points") else list(response)
        return [RunRecord.from_dict(h.payload) for h in hits]

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

    model_selections: dict[str, str] = {}
    try:
        for agent in registry.all():
            model_selections[agent.name] = agent.model
    except Exception:
        pass

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
