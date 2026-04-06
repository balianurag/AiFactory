"""
Memory Router — Three-layer memory system powered by LlamaIndex + Qdrant.

Architecture:
  Agent Layer  → private STM + episodic (what only this agent needs now)
  Team Layer   → shared working memory + contracts + intermediate artifacts
  Global Layer → stable, curated, cross-project knowledge (LTM)

All layers use Qdrant as the vector store, with LlamaIndex VectorStoreIndex
on top for advanced retrieval (query engines, reranking, metadata filtering).

Embedding providers:
  - HuggingFace (default): BAAI/bge-small-en-v1.5 — fast, free, 384-dim
  - OpenAI (optional): text-embedding-3-small — 1536-dim, paid

Promotion pipeline: Agent → (summarise + judge) → Team → (curate) → Global.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
    models,
)

from aifactory.config.settings import get_settings
from aifactory.core.observability import traceable
from aifactory.core.schemas import MemoryEntry, MemoryLevel

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Embedding Provider — LlamaIndex-managed
# ═════════════════════════════════════════════════════════════════════════════

_embed_model_cache: Any = None

# Sentinel value indicating we're using stub embeddings (no real model loaded)
_STUB_SENTINEL = object()


def _get_embed_model() -> Any:
    """
    Build the LlamaIndex embed model singleton.

    - 'huggingface' → BAAI/bge-small-en-v1.5 (local, free, 384-dim)
    - 'openai'      → text-embedding-3-small  (API, paid, 1536-dim)
    """
    global _embed_model_cache
    if _embed_model_cache is not None:
        return _embed_model_cache

    settings = get_settings()
    provider = settings.llamaindex_embedding_provider.lower()

    use_openai = provider == "openai" and settings.openai_api_key

    # If HuggingFace requested, check if the package is actually installed.
    # It lives in the [embeddings-local] optional group (pulls torch ~5GB)
    # and is NOT installed in Docker/CI.  Auto-fallback to OpenAI if missing.
    if not use_openai:
        try:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            _embed_model_cache = HuggingFaceEmbedding(
                model_name=settings.llamaindex_hf_model,
            )
            logger.info(
                "LlamaIndex embedding: HuggingFace %s (%d-dim)",
                settings.llamaindex_hf_model,
                settings.llamaindex_hf_dimensions,
            )
        except ImportError:
            if settings.openai_api_key:
                logger.warning(
                    "llama-index-embeddings-huggingface not installed "
                    "(optional dep). Falling back to OpenAI embeddings."
                )
                use_openai = True
            else:
                # Neither HuggingFace package nor OpenAI key available.
                # Fall back to stub embeddings — memory will work but
                # semantic search quality will be random (hash-based).
                # The core pipeline (LLM calls) is unaffected.
                logger.warning(
                    "No embedding provider available: "
                    "llama-index-embeddings-huggingface not installed and "
                    "OPENAI_API_KEY not set. Using stub embeddings. "
                    "Memory store/retrieve will work but semantic search "
                    "quality will be degraded."
                )
                _embed_model_cache = _STUB_SENTINEL
                return _embed_model_cache

    if use_openai:
        from llama_index.embeddings.openai import OpenAIEmbedding

        _embed_model_cache = OpenAIEmbedding(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
            dimensions=settings.embedding_dimensions,
        )
        logger.info(
            "LlamaIndex embedding: OpenAI %s (%d-dim)",
            settings.embedding_model,
            settings.embedding_dimensions,
        )

    return _embed_model_cache


def _get_embed_dimensions() -> int:
    """Return the active embedding dimension count."""
    settings = get_settings()
    provider = settings.llamaindex_embedding_provider.lower()
    if provider == "openai" and settings.openai_api_key:
        return settings.embedding_dimensions

    # Check if HuggingFace package is actually available
    try:
        import llama_index.embeddings.huggingface  # noqa: F401

        return settings.llamaindex_hf_dimensions
    except ImportError:
        # Will fallback to OpenAI or stub at model-load time
        if settings.openai_api_key:
            return settings.embedding_dimensions
        # Stub mode — use HF dimensions (384) for collection sizing
        return settings.llamaindex_hf_dimensions


# ═════════════════════════════════════════════════════════════════════════════
# Stub embedding — for testing without any model loaded
# ═════════════════════════════════════════════════════════════════════════════


def _stub_embed(text: str) -> list[float]:
    """
    Deterministic pseudo-embedding for offline development & tests.
    Uses SHA-256 hash bytes spread across the vector dimensions.
    NOT suitable for production — semantic similarity is random.
    """
    settings = get_settings()
    dim = settings.llamaindex_hf_dimensions  # 384 by default for tests
    h = hashlib.sha256(text.encode()).digest()
    extended = (h * ((dim // len(h)) + 1))[:dim]
    return [(b / 127.5) - 1.0 for b in extended]


# Also expose openai_embed for backward compat
def openai_embed(text: str) -> list[float]:
    """Legacy wrapper — embeds via the active LlamaIndex model."""
    model = _get_embed_model()
    return model.get_text_embedding(text)


# ═════════════════════════════════════════════════════════════════════════════
# Memory Router
# ═════════════════════════════════════════════════════════════════════════════


class MemoryRouter:
    """
    Unified interface to the 3-layer memory system.

    Every agent calls store() and retrieve() through this router.
    The router handles:
      - Layer selection (agent / team / global)
      - Qdrant collection management
      - LlamaIndex VectorStoreIndex for advanced query engines
      - Optional reranking (BGE reranker)
      - Summarisation before promotion
      - Cross-layer retrieval with metadata filtering
      - Deduplication via content hashing
    """

    def __init__(
        self,
        qdrant_client: QdrantClient | None = None,
        embedding_fn: Any = None,
    ) -> None:
        settings = get_settings()

        # Qdrant client
        if qdrant_client:
            self.client = qdrant_client
        else:
            if settings.qdrant_url.startswith("http"):
                self.client = QdrantClient(
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key or None,
                )
            else:
                self.client = QdrantClient(":memory:")

        self.prefix = settings.qdrant_collection_prefix
        self.top_k = settings.memory_retrieval_top_k
        self.promotion_interval = settings.memory_promotion_interval

        # Embedding: pluggable function for tests, or LlamaIndex model for prod
        self._embed_fn = embedding_fn  # None means use LlamaIndex model
        self._dimensions = len(embedding_fn("test")) if embedding_fn else _get_embed_dimensions()

        # LlamaIndex index cache — one per collection
        self._index_cache: dict[str, Any] = {}

        # Ensure collections exist (fall back to in-memory if remote Qdrant unreachable)
        try:
            self._init_collections()
        except Exception:
            if qdrant_client is None:
                logger.warning(
                    "Qdrant at %s unreachable — falling back to in-memory mode",
                    settings.qdrant_url,
                )
                self.client = QdrantClient(":memory:")
                self._init_collections()
            else:
                raise

    @property
    def dimensions(self) -> int:
        return self._dimensions

    # ─── Embedding ────────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        """Embed text using the configured provider."""
        if self._embed_fn is not None:
            return self._embed_fn(text)
        model = _get_embed_model()
        if model is _STUB_SENTINEL:
            return _stub_embed(text)
        return model.get_text_embedding(text)

    # ─── Collection Management ────────────────────────────────────────

    def _collection_name(self, level: MemoryLevel) -> str:
        return f"{self.prefix}_{level.value}"

    def _init_collections(self) -> None:
        """Create Qdrant collections for each memory level if they don't exist."""
        for level in MemoryLevel:
            name = self._collection_name(level)
            existing = [c.name for c in self.client.get_collections().collections]
            if name not in existing:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=self._dimensions,
                        distance=Distance.COSINE,
                    ),
                )
                # Create payload indexes for fast filtering
                for field in ["agent_id", "team_id", "entry_type", "promoted"]:
                    self.client.create_payload_index(
                        collection_name=name,
                        field_name=field,
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                logger.info("Created Qdrant collection: %s (%d-dim)", name, self._dimensions)

    # ─── LlamaIndex Index (lazy per collection) ──────────────────────

    def _get_index(self, level: MemoryLevel) -> Any:
        """
        Get or create a LlamaIndex VectorStoreIndex for a collection.

        This is the bridge between Qdrant and LlamaIndex query engines.
        """
        name = self._collection_name(level)
        if name in self._index_cache:
            return self._index_cache[name]

        from llama_index.core import Settings as LISettings
        from llama_index.core import StorageContext, VectorStoreIndex
        from llama_index.vector_stores.qdrant import QdrantVectorStore

        # Configure LlamaIndex embed model when not using stub
        if self._embed_fn is None:
            LISettings.embed_model = _get_embed_model()

        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=name,
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
        )

        self._index_cache[name] = index
        return index

    # ─── Query Engine Factory ─────────────────────────────────────────

    def get_query_engine(
        self,
        layer: MemoryLevel | str = MemoryLevel.AGENT,
        agent_id: str | None = None,
        team_id: str | None = None,
        top_k: int | None = None,
    ) -> Any:
        """
        Build a LlamaIndex query engine for a memory layer.

        This is the primary interface agents use for retrieval:
            query_engine = memory_router.get_query_engine(
                layer="team", agent_id=state["current_agent_id"]
            )
            response = query_engine.query("Latest auth decisions")

        Returns:
            A LlamaIndex query engine with metadata filters and optional reranking.
        """
        if isinstance(layer, str):
            layer = MemoryLevel(layer)

        k = top_k or self.top_k
        index = self._get_index(layer)

        # Build Qdrant metadata filters
        from llama_index.core.vector_stores import (
            ExactMatchFilter,
            MetadataFilters,
        )

        filters_list: list[ExactMatchFilter] = []
        if agent_id and layer == MemoryLevel.AGENT:
            filters_list.append(ExactMatchFilter(key="agent_id", value=agent_id))
        if team_id and layer in (MemoryLevel.TEAM, MemoryLevel.AGENT):
            filters_list.append(ExactMatchFilter(key="team_id", value=team_id))

        metadata_filters = MetadataFilters(filters=filters_list) if filters_list else None

        # Build node postprocessors (reranking)
        node_postprocessors = []
        settings = get_settings()
        if settings.llamaindex_rerank_enabled and self._embed_fn is None:
            try:
                from llama_index.postprocessor.flag_embedding_reranker import (
                    FlagEmbeddingReranker,
                )

                reranker = FlagEmbeddingReranker(
                    model=settings.llamaindex_rerank_model,
                    top_n=settings.llamaindex_rerank_top_n,
                )
                node_postprocessors.append(reranker)
            except Exception as e:
                logger.warning("Reranker init failed, skipping: %s", e)

        query_engine = index.as_query_engine(
            similarity_top_k=k,
            filters=metadata_filters,
            node_postprocessors=node_postprocessors if node_postprocessors else None,
        )
        return query_engine

    # ─── Store ────────────────────────────────────────────────────────

    @traceable(name="memory_store", run_type="tool", metadata={"component": "memory_router"})
    def store(
        self,
        entry: MemoryEntry,
    ) -> str:
        """
        Store a memory entry at the appropriate layer.

        Uses raw Qdrant upsert for speed and deduplication control.
        The LlamaIndex index sees the data on next query (shared Qdrant store).

        Returns the point ID.
        """
        collection = self._collection_name(entry.level)
        vector = self._embed(entry.content)
        point_id = self._content_hash(
            entry.content + (entry.agent_id or "") + str(entry.turn_number)
        )

        payload = entry.model_dump(mode="json")
        payload["_content_hash"] = point_id

        self.client.upsert(
            collection_name=collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )
        logger.debug(
            "Stored memory [%s] in %s: %s...",
            entry.level.value,
            collection,
            entry.content[:60],
        )
        return str(point_id)

    # ─── Retrieve ─────────────────────────────────────────────────────

    @traceable(
        name="memory_retrieve",
        run_type="retriever",
        metadata={"component": "memory_router"},
    )
    def retrieve(
        self,
        query: str,
        levels: list[MemoryLevel] | None = None,
        agent_id: str | None = None,
        team_id: str | None = None,
        top_k: int | None = None,
        entry_type: str | None = None,
    ) -> list[MemoryEntry]:
        """
        Search across one or more memory layers using Qdrant directly.

        This is the backward-compatible retrieval path used by BaseAgent.think().
        For LlamaIndex query engine access, use get_query_engine() instead.

        Args:
            query: Natural language query to search for.
            levels: Which layers to search. Default: all three.
            agent_id: Filter to a specific agent's memories.
            team_id: Filter to a specific team's memories.
            top_k: Max results per level. Default from settings.
            entry_type: Filter by entry type (observation, decision, etc.).

        Returns:
            List of MemoryEntry sorted by relevance (highest first).
        """
        if levels is None:
            levels = list(MemoryLevel)
        k = top_k or self.top_k
        query_vector = self._embed(query)

        all_results: list[tuple[float, MemoryEntry]] = []

        for level in levels:
            collection = self._collection_name(level)

            # Build filters
            conditions: list[FieldCondition] = []
            if agent_id and level == MemoryLevel.AGENT:
                conditions.append(FieldCondition(key="agent_id", match=MatchValue(value=agent_id)))
            if team_id and level in (MemoryLevel.TEAM, MemoryLevel.AGENT):
                conditions.append(FieldCondition(key="team_id", match=MatchValue(value=team_id)))
            if entry_type:
                conditions.append(
                    FieldCondition(key="entry_type", match=MatchValue(value=entry_type))
                )

            query_filter = Filter(must=conditions) if conditions else None

            try:
                results = self.client.query_points(
                    collection_name=collection,
                    query=query_vector,
                    query_filter=query_filter,
                    limit=k,
                    with_payload=True,
                )
                for point in results.points:
                    score = point.score if point.score is not None else 0.0
                    entry_obj = MemoryEntry(**point.payload)  # type: ignore[arg-type]
                    all_results.append((score, entry_obj))
            except Exception as e:
                logger.warning(
                    "Memory retrieval failed for %s: %s",
                    collection,
                    e,
                )

        # Sort by score descending, return entries
        all_results.sort(key=lambda x: x[0], reverse=True)
        return [entry_obj for _, entry_obj in all_results[:k]]

    # ─── LlamaIndex Query (advanced) ─────────────────────────────────

    def query(
        self,
        query_text: str,
        layer: MemoryLevel | str = MemoryLevel.TEAM,
        agent_id: str | None = None,
        team_id: str | None = None,
        top_k: int | None = None,
    ) -> str:
        """
        High-level query via LlamaIndex query engine.

        Returns the synthesised response text (LlamaIndex handles retrieval,
        reranking, and response synthesis).

        Usage in LangGraph nodes:
            context = memory_router.query(
                "Latest decisions about authentication flow",
                layer="team",
                team_id=state["team_id"],
            )
        """
        engine = self.get_query_engine(
            layer=layer,
            agent_id=agent_id,
            team_id=team_id,
            top_k=top_k,
        )
        try:
            response = engine.query(query_text)
            return str(response)
        except Exception as e:
            logger.warning("LlamaIndex query failed: %s", e)
            return ""

    # ─── Promote ──────────────────────────────────────────────────────

    @traceable(name="memory_promote", run_type="tool", metadata={"component": "memory_router"})
    def promote(
        self,
        agent_id: str,
        team_id: str,
        summary: str,
        from_level: MemoryLevel = MemoryLevel.AGENT,
        to_level: MemoryLevel = MemoryLevel.TEAM,
    ) -> str:
        """
        Promote summarised memory from one layer to the next.

        Typically called after N turns:
          Agent → Team (with summarisation)
          Team → Global (with curation + quality check)

        Args:
            agent_id: The agent whose memories are being promoted.
            team_id: The team scope.
            summary: Pre-summarised content to store at the higher level.
            from_level: Source layer.
            to_level: Destination layer.

        Returns:
            ID of the new entry in the target layer.
        """
        promoted_entry = MemoryEntry(
            level=to_level,
            agent_id=agent_id,
            team_id=team_id,
            content=summary,
            summary=summary,
            entry_type="promoted_summary",
            metadata={"promoted_from": from_level.value},
        )
        entry_id = self.store(promoted_entry)

        # Mark source entries as promoted (best-effort)
        try:
            source_collection = self._collection_name(from_level)
            results = self.client.scroll(
                collection_name=source_collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="agent_id", match=MatchValue(value=agent_id)),
                        FieldCondition(key="promoted", match=MatchValue(value=False)),
                    ]
                ),
                limit=100,
                with_payload=True,
                with_vectors=False,
            )
            point_ids = [p.id for p in results[0]]
            if point_ids:
                self.client.set_payload(
                    collection_name=source_collection,
                    payload={"promoted": True},
                    points=point_ids,
                )
                logger.info(
                    "Promoted %d entries from %s → %s for agent %s",
                    len(point_ids),
                    from_level.value,
                    to_level.value,
                    agent_id,
                )
        except Exception as e:
            logger.warning("Failed to mark source entries as promoted: %s", e)

        return entry_id

    # ─── Utilities ────────────────────────────────────────────────────

    def get_agent_turn_count(self, agent_id: str) -> int:
        """Count how many unpromoted entries an agent has (proxy for turn count)."""
        try:
            collection = self._collection_name(MemoryLevel.AGENT)
            result = self.client.count(
                collection_name=collection,
                count_filter=Filter(
                    must=[
                        FieldCondition(key="agent_id", match=MatchValue(value=agent_id)),
                        FieldCondition(key="promoted", match=MatchValue(value=False)),
                    ]
                ),
            )
            return result.count
        except Exception:
            return 0

    def should_promote(self, agent_id: str) -> bool:
        """Check if an agent's memories should be promoted (every N turns)."""
        count = self.get_agent_turn_count(agent_id)
        return count >= self.promotion_interval

    def clear_agent(self, agent_id: str) -> None:
        """Clear all memories for a specific agent (used for cleanup)."""
        for level in MemoryLevel:
            collection = self._collection_name(level)
            try:
                self.client.delete(
                    collection_name=collection,
                    points_selector=models.FilterSelector(
                        filter=Filter(
                            must=[FieldCondition(key="agent_id", match=MatchValue(value=agent_id))]
                        )
                    ),
                )
            except Exception as e:
                logger.debug("Clear failed for %s/%s: %s", collection, agent_id, e)

        # Invalidate index cache since data changed
        self._index_cache.clear()

    def get_stats(self) -> dict[str, int]:
        """Get point counts per collection."""
        stats = {}
        for level in MemoryLevel:
            name = self._collection_name(level)
            try:
                info = self.client.get_collection(name)
                stats[level.value] = info.points_count or 0
            except Exception:
                stats[level.value] = 0
        return stats

    # ─── Embedding Helpers ────────────────────────────────────────────

    @staticmethod
    def _content_hash(content: str) -> int:
        """Generate a stable integer ID from content for deduplication."""
        h = hashlib.sha256(content.encode()).hexdigest()
        return int(h[:16], 16)
