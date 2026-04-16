"""
Research Cache — Persistent semantic cache for researcher agent outputs.

Ported from experiments/main.py LongTermMemory.get_research() / save_research().

Two-tier lookup:
  1. Exact normalised-key match (zero cost)
  2. Cosine similarity on stored embeddings (threshold 0.82)
  3. Word-overlap fallback when no embeddings available (threshold 0.72)

Persistence: JSON file at ~/.aifactory/memories/research.json (default).
Override with AIFACTORY_MEMORIES_DIR env var or by passing cache_dir.

Design: all methods are fail-safe — exceptions are caught and logged, never raised.
A cache miss is always safe; the researcher will just call the LLM as normal.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Similarity thresholds (matching experiments/main.py values)
_COSINE_THRESHOLD = 0.82
_WORD_OVERLAP_THRESHOLD = 0.72

# Default persistence directory
_DEFAULT_DIR = Path(os.environ.get("AIFACTORY_MEMORIES_DIR", "~/.aifactory/memories"))


# ── Similarity helpers ──────────────────────────────────────────────────────


def _norm_query(q: str) -> str:
    """Normalise a query string for stable dict-key lookup."""
    return re.sub(r"\s+", " ", q.lower().strip())


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length float vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _word_overlap(a: str, b: str) -> float:
    """Jaccard word-overlap similarity between two query strings."""
    wa = set(a.split())
    wb = set(b.split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


# ── ResearchCache ───────────────────────────────────────────────────────────


class ResearchCache:
    """
    Persistent semantic cache for researcher agent outputs.

    Before the researcher LLM is called:
        hit = cache.lookup(query, embedding=embed(query))
        if hit:
            return hit["summary"]   # free

    After the researcher LLM responds:
        cache.store(query, summary, run_id, embedding=embed(query))

    Lookup order:
      1. Exact normalised-key match
      2. Cosine similarity ≥ 0.82 (requires stored embeddings)
      3. Word-overlap Jaccard ≥ 0.72 (no embeddings needed, weaker signal)

    Persistence: ``~/.aifactory/memories/research.json`` (one JSON object,
    keys = normalised query strings, values = entry dicts).
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._dir = (cache_dir or _DEFAULT_DIR).expanduser()
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.debug("ResearchCache: could not create cache dir %s: %s", self._dir, exc)
        self._path = self._dir / "research.json"

    # ── Internal I/O ──────────────────────────────────────────────────

    def _load(self) -> dict[str, Any]:
        try:
            if not self._path.exists():
                return {}
            raw = self._path.read_text(encoding="utf-8")
            return json.loads(raw) if raw.strip() else {}
        except Exception as exc:
            logger.debug("ResearchCache: load failed: %s", exc)
            return {}

    def _save(self, data: dict[str, Any]) -> None:
        try:
            self._path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            logger.debug("ResearchCache: save failed: %s", exc)

    # ── Public API ────────────────────────────────────────────────────

    def lookup(
        self,
        query: str,
        embedding: list[float] | None = None,
    ) -> dict[str, Any] | None:
        """
        Return a cached research entry for *query*, or ``None`` on miss.

        The returned dict always has a ``"summary"`` key with the researcher's
        output text, plus ``"query"``, ``"run_id"``, and ``"timestamp"``.

        Args:
            query:     The research query string.
            embedding: Optional float vector for the query (enables cosine tier).
        """
        try:
            cache = self._load()
            if not cache:
                return None

            key = _norm_query(query)

            # Tier 1: exact key match
            if key in cache:
                logger.debug("ResearchCache: exact hit for %r", query[:60])
                return cache[key]

            # Tier 2: cosine similarity (requires stored + query embeddings)
            if embedding:
                best_score, best_entry = 0.0, None
                for entry in cache.values():
                    emb = entry.get("embedding")
                    if emb:
                        score = _cosine(embedding, emb)
                        if score > best_score:
                            best_score, best_entry = score, entry
                if best_score >= _COSINE_THRESHOLD:
                    logger.debug(
                        "ResearchCache: cosine hit (score=%.3f) for %r", best_score, query[:60]
                    )
                    return best_entry

            # Tier 3: word-overlap fallback
            best_score, best_entry = 0.0, None
            for cached_key, entry in cache.items():
                score = _word_overlap(key, cached_key)
                if score > best_score:
                    best_score, best_entry = score, entry
            if best_score >= _WORD_OVERLAP_THRESHOLD:
                logger.debug(
                    "ResearchCache: word-overlap hit (score=%.3f) for %r",
                    best_score,
                    query[:60],
                )
                return best_entry

            return None

        except Exception as exc:
            logger.debug("ResearchCache: lookup error: %s", exc)
            return None

    def store(
        self,
        query: str,
        summary: str,
        run_id: str = "",
        embedding: list[float] | None = None,
    ) -> None:
        """
        Persist a researcher output in the cache.

        Args:
            query:     The research query string.
            summary:   The researcher agent's output text.
            run_id:    Pipeline run ID (for provenance).
            embedding: Optional float vector for the query.
        """
        try:
            cache = self._load()
            entry: dict[str, Any] = {
                "query": query,
                "summary": summary,
                "run_id": run_id,
                "timestamp": datetime.now(UTC).isoformat(),
            }
            if embedding:
                entry["embedding"] = embedding
            cache[_norm_query(query)] = entry
            self._save(cache)
            logger.debug("ResearchCache: stored entry for %r", query[:60])
        except Exception as exc:
            logger.debug("ResearchCache: store error: %s", exc)

    def size(self) -> int:
        """Number of cached research entries."""
        return len(self._load())

    def clear(self) -> None:
        """Remove all cached entries."""
        self._save({})
        logger.debug("ResearchCache: cleared")
