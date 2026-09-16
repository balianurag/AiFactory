"""
MilvusStore — thin adapter presenting the subset of the Qdrant-client
interface that :class:`aifactory.core.memory_router.MemoryRouter` needs.

Supported constructor URIs (``settings.aifactory_milvus_uri``):
  * Local path (e.g. ``"./milvus.db"``) → Milvus Lite (embedded, file-backed).
    Ideal for local dev, CI, and tests — no server required.
  * URI (``"milvus://host:19530"`` or ``"http://host:19530"``) → full Milvus
    over gRPC/HTTP.

Fail-soft: every method wraps its pymilvus calls in try/except so a Milvus
outage never crashes the pipeline. On failure we log a warning and return
an empty / benign result. The caller (MemoryRouter) treats missing hits the
same as a cold cache and continues.

Only the methods MemoryRouter actually touches are implemented:
  - ``get_collections``, ``create_collection``, ``create_payload_index``
  - ``upsert``, ``query_points``, ``scroll``, ``set_payload``, ``count``,
    ``delete``, ``get_collection``

The return values mirror the shapes qdrant-client returns so MemoryRouter's
existing code paths work without special-casing per backend.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ── Shape-compatible response objects (mimic qdrant-client's Pydantic) ───────


@dataclass
class _CollectionDescriptor:
    name: str


@dataclass
class _CollectionsResponse:
    collections: list[_CollectionDescriptor]


@dataclass
class _CollectionInfo:
    points_count: int | None


@dataclass
class _ScoredPoint:
    id: int
    score: float | None
    payload: dict[str, Any]


@dataclass
class _QueryResponse:
    points: list[_ScoredPoint]


@dataclass
class _ScrollPoint:
    id: int
    payload: dict[str, Any]


@dataclass
class _CountResponse:
    count: int


# ── Adapter ──────────────────────────────────────────────────────────────────


class MilvusStore:
    """
    Fail-soft Milvus adapter compatible with the qdrant-client subset used by
    :class:`MemoryRouter`.
    """

    def __init__(
        self,
        uri: str | None = None,
        *,
        dimensions: int = 1536,
    ) -> None:
        self._uri = uri or "./milvus.db"
        self._dimensions = dimensions
        self._client: Any = None
        self._known_collections: set[str] = set()
        self._connect()

    # ── Setup ────────────────────────────────────────────────────────────

    def _connect(self) -> None:
        try:
            from pymilvus import MilvusClient  # type: ignore[import-untyped]

            self._client = MilvusClient(uri=self._uri)
            logger.info("MilvusStore: connected to %s", self._uri)
        except Exception as exc:
            # Fail-soft — any operation later will log its own warning and
            # return an empty result, keeping the pipeline alive.
            logger.warning(
                "MilvusStore: connection to %s failed (%s) — adapter disabled",
                self._uri,
                exc,
            )
            self._client = None

    @property
    def is_ready(self) -> bool:
        return self._client is not None

    # ── Collection management ────────────────────────────────────────────

    def get_collections(self) -> _CollectionsResponse:
        if not self.is_ready:
            return _CollectionsResponse(collections=[])
        try:
            names = list(self._client.list_collections())
            self._known_collections.update(names)
            return _CollectionsResponse(collections=[_CollectionDescriptor(name=n) for n in names])
        except Exception as exc:
            logger.warning("MilvusStore.get_collections failed: %s", exc)
            return _CollectionsResponse(collections=[])

    def create_collection(
        self,
        collection_name: str,
        vectors_config: Any | None = None,
        **_: Any,
    ) -> None:
        if not self.is_ready:
            return
        # Derive dimension from qdrant VectorParams if supplied
        dim = self._dimensions
        try:
            if vectors_config is not None and hasattr(vectors_config, "size"):
                dim = int(vectors_config.size)
        except Exception as _exc:
            logger.debug("MilvusStore: dim resolution fell back to default (%s)", _exc)
        try:
            if self._client.has_collection(collection_name):
                self._known_collections.add(collection_name)
                return
            self._client.create_collection(
                collection_name=collection_name,
                dimension=dim,
                metric_type="COSINE",
                auto_id=False,
                enable_dynamic_field=True,
            )
            self._known_collections.add(collection_name)
            logger.info(
                "MilvusStore: created collection %s (%d-dim, COSINE)",
                collection_name,
                dim,
            )
        except Exception as exc:
            logger.warning("MilvusStore.create_collection(%s) failed: %s", collection_name, exc)

    def create_payload_index(self, *_: Any, **__: Any) -> None:
        # Milvus supports scalar indexes but MemoryRouter's use is best-effort;
        # no-op keeps the adapter fail-soft.
        return None

    def get_collection(self, collection_name: str) -> _CollectionInfo:
        if not self.is_ready:
            return _CollectionInfo(points_count=0)
        try:
            stats = self._client.get_collection_stats(collection_name)
            return _CollectionInfo(points_count=int(stats.get("row_count", 0)))
        except Exception as exc:
            logger.debug("MilvusStore.get_collection(%s): %s", collection_name, exc)
            return _CollectionInfo(points_count=0)

    # ── Writes ───────────────────────────────────────────────────────────

    def upsert(self, collection_name: str, points: list[Any]) -> None:
        if not self.is_ready:
            return
        try:
            data = []
            for p in points:
                payload = dict(getattr(p, "payload", {}) or {})
                data.append(
                    {
                        "id": int(getattr(p, "id", 0)),
                        "vector": list(getattr(p, "vector", [])),
                        **payload,
                    }
                )
            if data:
                self._client.upsert(collection_name=collection_name, data=data)
        except Exception as exc:
            logger.warning("MilvusStore.upsert(%s) failed: %s", collection_name, exc)

    def set_payload(
        self,
        collection_name: str,
        payload: dict[str, Any],
        points: list[Any] | None = None,
        **_: Any,
    ) -> None:
        # Milvus doesn't have a direct partial-update API for dynamic fields;
        # the promotion flag is best-effort so we no-op fail-soft.
        return None

    def delete(
        self,
        collection_name: str,
        points_selector: Any | None = None,
        **_: Any,
    ) -> None:
        if not self.is_ready:
            return
        # Best-effort delete by filter — MemoryRouter passes a FilterSelector
        # wrapping a Qdrant Filter. We don't attempt to translate it; instead
        # we no-op and log so callers know cold-tier deletion is a follow-up.
        logger.debug(
            "MilvusStore.delete(%s): filter-based delete not implemented (no-op)",
            collection_name,
        )
        return None

    # ── Reads ────────────────────────────────────────────────────────────

    def query_points(
        self,
        collection_name: str,
        query: list[float] | None = None,
        query_filter: Any | None = None,
        limit: int = 5,
        with_payload: bool = True,
        **_: Any,
    ) -> _QueryResponse:
        if not self.is_ready or query is None:
            return _QueryResponse(points=[])
        try:
            res = self._client.search(
                collection_name=collection_name,
                data=[list(query)],
                limit=int(limit),
                output_fields=["*"] if with_payload else None,
            )
            hits = res[0] if res else []
            points: list[_ScoredPoint] = []
            for h in hits:
                pid = int(h.get("id", 0)) if isinstance(h, dict) else int(getattr(h, "id", 0))
                score = h.get("distance") if isinstance(h, dict) else getattr(h, "distance", None)
                entity = h.get("entity") if isinstance(h, dict) else getattr(h, "entity", {})
                payload = dict(entity or {})
                # Milvus returns `id` and `vector` inside entity too; strip them
                payload.pop("vector", None)
                payload.pop("id", None)
                points.append(_ScoredPoint(id=pid, score=score, payload=payload))
            return _QueryResponse(points=points)
        except Exception as exc:
            logger.warning("MilvusStore.query_points(%s) failed: %s", collection_name, exc)
            return _QueryResponse(points=[])

    def scroll(
        self,
        collection_name: str,
        scroll_filter: Any | None = None,
        limit: int = 100,
        with_payload: bool = True,
        with_vectors: bool = False,
        **_: Any,
    ) -> tuple[list[_ScrollPoint], Any | None]:
        if not self.is_ready:
            return ([], None)
        try:
            res = self._client.query(
                collection_name=collection_name,
                filter="",
                output_fields=["*"] if with_payload else None,
                limit=int(limit),
            )
            points = []
            for row in res or []:
                pid = int(row.get("id", 0))
                payload = {k: v for k, v in row.items() if k not in ("id", "vector")}
                points.append(_ScrollPoint(id=pid, payload=payload))
            return (points, None)
        except Exception as exc:
            logger.debug("MilvusStore.scroll(%s) failed: %s", collection_name, exc)
            return ([], None)

    def count(
        self,
        collection_name: str,
        count_filter: Any | None = None,
        **_: Any,
    ) -> _CountResponse:
        if not self.is_ready:
            return _CountResponse(count=0)
        try:
            stats = self._client.get_collection_stats(collection_name)
            return _CountResponse(count=int(stats.get("row_count", 0)))
        except Exception as exc:
            logger.debug("MilvusStore.count(%s) failed: %s", collection_name, exc)
            return _CountResponse(count=0)
