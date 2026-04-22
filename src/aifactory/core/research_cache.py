"""
Research Cache — persistent semantic cache for researcher agent outputs.

Lookup tiers (fastest to slowest):
  0. Redis hot cache  — exact key match, O(1), 1h TTL (multi-process safe)
  1. PostgreSQL       — exact normalised-key match (multi-process safe)
  2. Cosine similarity on stored embeddings (threshold 0.82) — PG entries
  3. Word-overlap Jaccard fallback (threshold 0.72) — PG or file entries

Write path: Redis (hot tier, 1h TTL) + PostgreSQL (cold tier, permanent)
            + local JSON file (dev fallback when both are unavailable).

Persistence:
  PostgreSQL `research_cache` table — primary, multi-process safe
  ~/.aifactory/memories/research.json — local dev fallback

Design: all methods are fail-safe — exceptions are caught and logged, never raised.
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

_COSINE_THRESHOLD = 0.82
_WORD_OVERLAP_THRESHOLD = 0.72
_REDIS_TTL = 3600  # 1 hour hot cache

_DEFAULT_DIR = Path(os.environ.get("AIFACTORY_MEMORIES_DIR", "~/.aifactory/memories"))


# ── Similarity helpers ──────────────────────────────────────────────────────


def _norm_query(q: str) -> str:
    return re.sub(r"\s+", " ", q.lower().strip())


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _word_overlap(a: str, b: str) -> float:
    wa = set(a.split())
    wb = set(b.split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


# ── Sync Redis singleton ─────────────────────────────────────────────────────

_sync_redis: Any = None
_redis_checked: bool = False


def _get_redis() -> Any:
    """Return a sync redis.Redis client, or None if Redis is unavailable."""
    global _sync_redis, _redis_checked
    if _redis_checked:
        return _sync_redis
    _redis_checked = True
    redis_url = os.getenv("REDIS_URL", "")
    if not redis_url:
        return None
    try:
        import redis as _redis_mod
        client = _redis_mod.from_url(
            redis_url, decode_responses=True, socket_connect_timeout=1
        )
        client.ping()
        _sync_redis = client
    except Exception as exc:
        logger.debug("ResearchCache: Redis unavailable (%s) — using PG/file only", exc)
    return _sync_redis


def _redis_key(query_key: str) -> str:
    return f"aifactory:research:{query_key[:200]}"


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
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._dir = (cache_dir or _DEFAULT_DIR).expanduser()
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.debug("ResearchCache: could not create cache dir %s: %s", self._dir, exc)
        self._path = self._dir / "research.json"

        # PostgreSQL availability flag
        self._pg: bool = False
        try:
            from aifactory.core.database import get_sync_engine
            if get_sync_engine() is not None:
                self._pg = True
        except Exception:
            pass

    # ── Public API ────────────────────────────────────────────────────

    def lookup(
        self,
        query: str,
        embedding: list[float] | None = None,
    ) -> dict[str, Any] | None:
        """
        Return a cached research entry for *query*, or None on miss.

        The returned dict always has a "summary" key with the researcher's
        output text, plus "query", "run_id", and "timestamp".

        Lookup order:
          0. Redis hot cache (exact key, O(1))
          1. PostgreSQL exact key match
          2. Cosine similarity ≥ 0.82 (PG entries with stored embeddings)
          3. Word-overlap Jaccard ≥ 0.72 (PG or file entries)
        """
        try:
            key = _norm_query(query)

            # Tier 0: Redis hot cache
            r = _get_redis()
            if r is not None:
                try:
                    raw = r.get(_redis_key(key))
                    if raw:
                        entry = json.loads(raw)
                        logger.debug("ResearchCache: Redis hit for %r", query[:60])
                        return entry
                except Exception:
                    pass

            # Tier 1: PostgreSQL exact key
            if self._pg:
                try:
                    from aifactory.core.database import get_research_entry
                    entry = get_research_entry(key)
                    if entry:
                        logger.debug("ResearchCache: PG exact hit for %r", query[:60])
                        return self._pg_row_to_entry(entry)
                except Exception as exc:
                    logger.debug("ResearchCache: PG exact lookup failed: %s", exc)

                # Tier 2 + 3: scan all PG entries for similarity
                try:
                    from aifactory.core.database import get_all_research_entries
                    all_entries = get_all_research_entries()
                    if all_entries:
                        result = self._similarity_scan(key, all_entries, embedding)
                        if result:
                            return result
                except Exception as exc:
                    logger.debug("ResearchCache: PG scan failed: %s", exc)

            # Fallback: JSON file tiers
            cache = self._load_file()
            if cache:
                if key in cache:
                    logger.debug("ResearchCache: file exact hit for %r", query[:60])
                    return cache[key]
                result = self._similarity_scan_dict(key, cache, embedding)
                if result:
                    return result

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
        Persist a researcher output.

        Writes to Redis (hot tier, 1h TTL), PostgreSQL (cold tier), and
        JSON file (local fallback).  All writes are fail-safe.
        """
        try:
            key = _norm_query(query)
            entry: dict[str, Any] = {
                "query": query,
                "summary": summary,
                "run_id": run_id,
                "timestamp": datetime.now(UTC).isoformat(),
            }
            if embedding:
                entry["embedding"] = embedding

            # Redis hot tier
            r = _get_redis()
            if r is not None:
                try:
                    r.setex(_redis_key(key), _REDIS_TTL, json.dumps(entry))
                except Exception:
                    pass

            # PostgreSQL cold tier
            if self._pg:
                try:
                    from aifactory.core.database import upsert_research_entry
                    upsert_research_entry(
                        query_key=key,
                        query_text=query,
                        summary=summary,
                        embedding=embedding,
                        run_id=run_id,
                    )
                    logger.debug("ResearchCache: stored via PG for %r", query[:60])
                    return
                except Exception as exc:
                    logger.debug("ResearchCache: PG store failed — writing to file: %s", exc)

            # JSON file fallback
            cache = self._load_file()
            cache[key] = entry
            self._save_file(cache)
            logger.debug("ResearchCache: stored via file for %r", query[:60])

        except Exception as exc:
            logger.debug("ResearchCache: store error: %s", exc)

    def size(self) -> int:
        """Number of cached research entries."""
        if self._pg:
            try:
                from aifactory.core.database import get_all_research_entries
                return len(get_all_research_entries())
            except Exception:
                pass
        return len(self._load_file())

    def clear(self) -> None:
        """Remove all cached entries from all backends."""
        # Redis
        r = _get_redis()
        if r is not None:
            try:
                keys = list(r.scan_iter("aifactory:research:*"))
                if keys:
                    r.delete(*keys)
            except Exception:
                pass
        # PostgreSQL
        if self._pg:
            try:
                from aifactory.core.database import clear_research_entries
                clear_research_entries()
            except Exception:
                pass
        # JSON file
        self._save_file({})
        logger.debug("ResearchCache: cleared")

    # ── Similarity helpers ────────────────────────────────────────────

    def _similarity_scan(
        self,
        key: str,
        entries: list[dict[str, Any]],
        embedding: list[float] | None,
    ) -> dict[str, Any] | None:
        """Scan PG entries for cosine + word-overlap similarity."""
        best_score, best_entry = 0.0, None

        # Tier 2: cosine
        if embedding:
            for row in entries:
                emb = row.get("embedding")
                if emb and isinstance(emb, list):
                    score = _cosine(embedding, emb)
                    if score > best_score:
                        best_score, best_entry = score, row
            if best_score >= _COSINE_THRESHOLD:
                logger.debug("ResearchCache: PG cosine hit (score=%.3f)", best_score)
                return self._pg_row_to_entry(best_entry)

        # Tier 3: word-overlap
        best_score, best_entry = 0.0, None
        for row in entries:
            cached_key = row.get("query_key", "")
            score = _word_overlap(key, cached_key)
            if score > best_score:
                best_score, best_entry = score, row
        if best_score >= _WORD_OVERLAP_THRESHOLD:
            logger.debug("ResearchCache: PG word-overlap hit (score=%.3f)", best_score)
            return self._pg_row_to_entry(best_entry)

        return None

    def _similarity_scan_dict(
        self,
        key: str,
        cache: dict[str, Any],
        embedding: list[float] | None,
    ) -> dict[str, Any] | None:
        """Scan file cache dict for cosine + word-overlap similarity."""
        # Cosine tier
        if embedding:
            best_score, best_entry = 0.0, None
            for entry in cache.values():
                emb = entry.get("embedding")
                if emb:
                    score = _cosine(embedding, emb)
                    if score > best_score:
                        best_score, best_entry = score, entry
            if best_score >= _COSINE_THRESHOLD:
                logger.debug("ResearchCache: file cosine hit (score=%.3f)", best_score)
                return best_entry

        # Word-overlap tier
        best_score, best_entry = 0.0, None
        for cached_key, entry in cache.items():
            score = _word_overlap(key, cached_key)
            if score > best_score:
                best_score, best_entry = score, entry
        if best_score >= _WORD_OVERLAP_THRESHOLD:
            logger.debug("ResearchCache: file word-overlap hit (score=%.3f)", best_score)
            return best_entry

        return None

    @staticmethod
    def _pg_row_to_entry(row: dict[str, Any]) -> dict[str, Any]:
        """Convert a PG research_cache row to the entry dict format."""
        ts = row.get("created_at", "")
        if hasattr(ts, "isoformat"):
            ts = ts.isoformat()
        return {
            "query": row.get("query_text", ""),
            "summary": row.get("summary", ""),
            "run_id": row.get("run_id", ""),
            "timestamp": str(ts),
            "embedding": row.get("embedding"),
        }

    # ── JSON file fallback ────────────────────────────────────────────

    def _load_file(self) -> dict[str, Any]:
        try:
            if not self._path.exists():
                return {}
            raw = self._path.read_text(encoding="utf-8")
            return json.loads(raw) if raw.strip() else {}
        except Exception as exc:
            logger.debug("ResearchCache: file load failed: %s", exc)
            return {}

    def _save_file(self, data: dict[str, Any]) -> None:
        try:
            self._path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:
            logger.debug("ResearchCache: file save failed: %s", exc)
