"""
Semantic Query Cache — Qdrant-backed similarity cache for pipeline results.

Unlike the LiteLLM response cache (byte-identical message matching), this cache
matches queries by semantic intent. Queries that mean the same thing — even with
different phrasing — hit the same cached result.

Examples:
  "write a fibonacci function in python"  → stored
  "fibonacci python function"             → HIT  (similarity ≈ 0.96)
  "fib sequence code python"              → HIT  (similarity ≈ 0.91)
  "build a REST API with FastAPI"         → MISS (different intent, similarity ≈ 0.3)

Architecture:
  Qdrant collection : aifactory_semantic_cache
  Each entry        : { query_embedding, query_text, final_response, metadata, stored_at }
  Lookup            : embed(query) → kNN(limit=1) → if score ≥ threshold → cache HIT
  Storage           : after successful full pipeline run
  TTL               : entries older than ttl_days are deleted on first access

Settings (all optional, have sensible defaults):
  AIFACTORY_SEMANTIC_CACHE_ENABLED    bool  default=True
  AIFACTORY_SEMANTIC_CACHE_THRESHOLD  float default=0.92
  AIFACTORY_SEMANTIC_CACHE_TTL_DAYS   int   default=7
  AIFACTORY_SEMANTIC_CACHE_MAX_SIZE   int   default=10000
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

_COLLECTION = "aifactory_semantic_cache"


class SemanticCache:
    """
    Qdrant-backed semantic similarity cache.

    Thread-safety: the Qdrant client handles concurrent access. Multiple
    concurrent pipeline runs can read/write safely.
    """

    def __init__(
        self,
        qdrant_client: Any,
        embed_fn: Callable[[str], list[float]],
        dimensions: int = 1536,
        threshold: float = 0.92,
        ttl_days: int = 7,
        max_size: int = 10_000,
    ) -> None:
        self.client = qdrant_client
        self.embed_fn = embed_fn
        self.dimensions = dimensions
        self.threshold = threshold
        self.ttl_seconds = ttl_days * 86_400
        self.max_size = max_size
        self._ready = False
        self._init_collection()

    def _init_collection(self) -> None:
        try:
            from qdrant_client.models import Distance, VectorParams

            existing = {c.name for c in self.client.get_collections().collections}
            if _COLLECTION not in existing:
                self.client.create_collection(
                    collection_name=_COLLECTION,
                    vectors_config=VectorParams(size=self.dimensions, distance=Distance.COSINE),
                )
            self._ready = True
            logger.debug(
                "SemanticCache: collection '%s' ready (dim=%d)", _COLLECTION, self.dimensions
            )
        except Exception as exc:
            logger.debug("SemanticCache: init failed (non-fatal) — %s", exc)

    # ─── Lookup ───────────────────────────────────────────────────────────

    def lookup(self, query: str) -> dict | None:
        """
        Search for a semantically similar cached result.

        Returns a dict with keys:
            query          : original cached query text
            final_response : cached pipeline response
            metadata       : dict with mode, elapsed_s, total_tokens, etc.
            score          : cosine similarity (0–1)

        Returns None on miss, expired entry, or any error.
        """
        if not self._ready:
            return None
        try:
            vector = self.embed_fn(query)
            # qdrant-client ≥1.x uses query_points; result has .points attribute
            response = self.client.query_points(
                collection_name=_COLLECTION,
                query=vector,
                limit=1,
                with_payload=True,
                score_threshold=self.threshold if self.threshold > -1 else None,
            )
            results = response.points if hasattr(response, "points") else list(response)
            if not results:
                return None

            hit = results[0]
            payload = hit.payload or {}

            # TTL check
            if self.ttl_seconds > 0:
                age = time.time() - payload.get("stored_at", 0)
                if age > self.ttl_seconds:
                    self._delete_point(hit.id)
                    return None

            logger.info(
                "SemanticCache HIT  score=%.3f  original='%.60s...'",
                hit.score,
                payload.get("query", ""),
            )
            return {
                "query": payload.get("query", ""),
                "final_response": payload.get("final_response", ""),
                "metadata": payload.get("metadata", {}),
                "score": hit.score,
            }
        except Exception as exc:
            logger.debug("SemanticCache lookup error (non-fatal): %s", exc)
            return None

    # ─── Store ────────────────────────────────────────────────────────────

    def store(
        self,
        query: str,
        final_response: str,
        metadata: dict | None = None,
    ) -> None:
        """
        Persist a query + response in the semantic cache.

        Uses the SHA-256 hash of the query as the Qdrant point ID so that
        re-running the identical query overwrites the cached entry (upsert).
        """
        if not self._ready or not final_response.strip():
            return
        try:
            from qdrant_client.models import PointStruct

            vector = self.embed_fn(query)
            point_id = int(hashlib.sha256(query.encode()).hexdigest()[:16], 16) % (2**63)

            self.client.upsert(
                collection_name=_COLLECTION,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "query": query[:500],
                            "final_response": final_response[:8_000],
                            "stored_at": time.time(),
                            "metadata": metadata or {},
                        },
                    )
                ],
            )
            logger.debug("SemanticCache: stored '%.50s...'", query)
        except Exception as exc:
            logger.debug("SemanticCache store error (non-fatal): %s", exc)

    # ─── Maintenance ──────────────────────────────────────────────────────

    def invalidate(self, query: str) -> None:
        """Remove a specific query from the cache."""
        if not self._ready:
            return
        try:
            point_id = int(hashlib.sha256(query.encode()).hexdigest()[:16], 16) % (2**63)
            self._delete_point(point_id)
        except Exception:
            pass

    def clear(self) -> None:
        """Delete all entries from the semantic cache."""
        if not self._ready:
            return
        try:
            from qdrant_client.models import Distance, VectorParams

            self.client.delete_collection(_COLLECTION)
            self.client.create_collection(
                collection_name=_COLLECTION,
                vectors_config=VectorParams(size=self.dimensions, distance=Distance.COSINE),
            )
            logger.info("SemanticCache: cleared")
        except Exception as exc:
            logger.warning("SemanticCache clear failed: %s", exc)

    def size(self) -> int:
        """Return the number of cached entries."""
        if not self._ready:
            return 0
        try:
            info = self.client.get_collection(_COLLECTION)
            return info.points_count or 0
        except Exception:
            return 0

    def _delete_point(self, point_id: Any) -> None:
        try:
            from qdrant_client.models import PointIdsList

            self.client.delete(
                collection_name=_COLLECTION,
                points_selector=PointIdsList(points=[point_id]),
            )
        except Exception:
            pass


# ─── Factory ──────────────────────────────────────────────────────────────────


def build_semantic_cache(qdrant_client: Any) -> SemanticCache | None:
    """
    Build a SemanticCache from settings if enabled. Returns None if disabled.

    Uses the same Qdrant client as the rest of the system to avoid extra
    connections. Uses embed_provider.build_embed_fn() for auto-selected embeddings.
    """
    try:
        from aifactory.config.settings import get_settings

        s = get_settings()
        if not s.semantic_cache_enabled:
            return None

        from aifactory.core.embed_provider import build_embed_fn

        embed_fn, dims = build_embed_fn()

        return SemanticCache(
            qdrant_client=qdrant_client,
            embed_fn=embed_fn,
            dimensions=dims,
            threshold=s.semantic_cache_threshold,
            ttl_days=s.semantic_cache_ttl_days,
            max_size=s.semantic_cache_max_size,
        )
    except Exception as exc:
        logger.debug("SemanticCache build failed (non-fatal): %s", exc)
        return None
