"""
Memory Router — The three-layer memory system for the AI Factory.

Architecture:
  Agent Layer  → private STM + episodic (what only this agent needs now)
  Team Layer   → shared working memory + contracts + intermediate artifacts
  Global Layer → stable, curated, cross-project knowledge (LTM)

All layers use Qdrant with namespace isolation (collection + payload filters).
Promotion pipeline: Agent → (summarise + judge) → Team → (curate) → Global.

Uses OpenAI text-embedding-3-small for real semantic embeddings.
Falls back to deterministic hash stub only when OPENAI_API_KEY is not set.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import openai
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from aifactory.config.settings import get_settings
from aifactory.core.schemas import MemoryEntry, MemoryLevel

logger = logging.getLogger(__name__)


# ── OpenAI Embedding Client (module-level, reused) ────────────────────────────

_embed_client: openai.OpenAI | None = None


def _get_embed_client() -> openai.OpenAI | None:
    """Return a sync OpenAI client for embeddings, or None if no API key."""
    global _embed_client
    if _embed_client is not None:
        return _embed_client
    settings = get_settings()
    if not settings.openai_api_key:
        return None
    _embed_client = openai.OpenAI(api_key=settings.openai_api_key)
    return _embed_client


def openai_embed(text: str) -> list[float]:
    """
    Generate embeddings via OpenAI text-embedding-3-small.

    Falls back to deterministic hash stub if API key is not available.
    """
    client = _get_embed_client()
    if client is None:
        logger.debug("No OPENAI_API_KEY — using stub embeddings")
        return _stub_embed(text)
    try:
        settings = get_settings()
        response = client.embeddings.create(
            model=settings.embedding_model,
            input=text[:8000],  # API limit safety
            dimensions=settings.embedding_dimensions,
        )
        return response.data[0].embedding
    except Exception as e:
        logger.warning("OpenAI embedding failed, using stub: %s", e)
        return _stub_embed(text)


def _stub_embed(text: str) -> list[float]:
    """
    Deterministic pseudo-embedding for offline development.
    Uses SHA-256 hash bytes spread across the vector dimensions.
    NOT suitable for production — semantic similarity is random.
    """
    settings = get_settings()
    dim = settings.embedding_dimensions
    h = hashlib.sha256(text.encode()).digest()
    extended = (h * ((dim // len(h)) + 1))[:dim]
    return [(b / 127.5) - 1.0 for b in extended]


class MemoryRouter:
    """
    Unified interface to the 3-layer memory system.

    Every agent calls `store()` and `retrieve()` through this router.
    The router handles:
      - Layer selection (agent / team / global)
      - Qdrant collection management
      - Summarisation before promotion
      - Cross-layer retrieval with reranking
      - Deduplication via content hashing
    """

    def __init__(
        self,
        qdrant_client: QdrantClient | None = None,
        embedding_fn: Any = None,
    ) -> None:
        settings = get_settings()

        # Qdrant client — in-memory for MVP / local dev
        if qdrant_client:
            self.client = qdrant_client
        else:
            if settings.qdrant_url.startswith("http"):
                try:
                    self.client = QdrantClient(
                        url=settings.qdrant_url,
                        api_key=settings.qdrant_api_key or None,
                    )
                except Exception:
                    logger.warning(
                        "Qdrant server not reachable at %s — falling back to in-memory mode",
                        settings.qdrant_url,
                    )
                    self.client = QdrantClient(":memory:")
            else:
                self.client = QdrantClient(":memory:")

        self.prefix = settings.qdrant_collection_prefix
        self.dimensions = settings.embedding_dimensions
        self.top_k = settings.memory_retrieval_top_k
        self.promotion_interval = settings.memory_promotion_interval

        # Pluggable embedding function — defaults to OpenAI text-embedding-3-small.
        # Falls back to stub if no API key is set.
        self._embed = embedding_fn or openai_embed

        # Ensure collections exist
        self._init_collections()

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
                        size=self.dimensions,
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
                logger.info("Created Qdrant collection: %s", name)

    # ─── Store ────────────────────────────────────────────────────────

    def store(
        self,
        entry: MemoryEntry,
    ) -> str:
        """
        Store a memory entry at the appropriate layer.

        Returns the point ID.
        """
        collection = self._collection_name(entry.level)
        vector = self._embed(entry.content)
        point_id = self._content_hash(
            entry.content + (entry.agent_id or "") + str(entry.turn_number)
        )

        payload = entry.model_dump(mode="json")
        # Remove content from payload to avoid duplication — it's in the vector
        # Actually keep it — we need it for retrieval display
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
        Search across one or more memory layers.

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
                    entry = MemoryEntry(**point.payload)  # type: ignore[arg-type]
                    all_results.append((score, entry))
            except Exception as e:
                logger.warning(
                    "Memory retrieval failed for %s: %s",
                    collection,
                    e,
                )

        # Sort by score descending, return entries
        all_results.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in all_results[:k]]

    # ─── Promote ──────────────────────────────────────────────────────

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
            # Find unpromoted entries for this agent
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
        """Count how many entries an agent has in its layer (proxy for turn count)."""
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
        # Qdrant needs positive int or UUID; use first 16 hex chars as int
        return int(h[:16], 16)
