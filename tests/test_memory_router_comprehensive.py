"""
Comprehensive tests for core/memory_router.py — 3-layer memory system.
"""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient

from aifactory.core.memory_router import MemoryRouter
from aifactory.core.schemas import MemoryEntry, MemoryLevel


# =============================================================================
# Collection Initialization
# =============================================================================


class TestCollectionInit:
    def test_creates_three_collections(self, memory: MemoryRouter) -> None:
        """All three memory layers should have Qdrant collections."""
        stats = memory.get_stats()
        assert "agent" in stats
        assert "team" in stats
        assert "global" in stats

    def test_collections_start_empty(self, memory: MemoryRouter) -> None:
        stats = memory.get_stats()
        assert stats["agent"] == 0
        assert stats["team"] == 0
        assert stats["global"] == 0

    def test_idempotent_init(self, qdrant_client: QdrantClient) -> None:
        """Creating MemoryRouter twice should not fail."""
        MemoryRouter(qdrant_client=qdrant_client)
        MemoryRouter(qdrant_client=qdrant_client)  # Should not raise


# =============================================================================
# Store
# =============================================================================


class TestStore:
    def test_store_agent_entry(self, memory: MemoryRouter) -> None:
        entry = MemoryEntry(
            level=MemoryLevel.AGENT,
            agent_id="agent_001",
            content="Working on auth module",
            turn_number=1,
        )
        point_id = memory.store(entry)
        assert point_id
        assert memory.get_stats()["agent"] == 1

    def test_store_team_entry(self, memory: MemoryRouter) -> None:
        entry = MemoryEntry(
            level=MemoryLevel.TEAM,
            team_id="team_alpha",
            content="Team using PostgreSQL",
            turn_number=1,
        )
        memory.store(entry)
        assert memory.get_stats()["team"] == 1

    def test_store_global_entry(self, memory: MemoryRouter) -> None:
        entry = MemoryEntry(
            level=MemoryLevel.GLOBAL,
            content="Best practice: use bcrypt",
            entry_type="learning",
            turn_number=1,
        )
        memory.store(entry)
        assert memory.get_stats()["global"] == 1

    def test_store_multiple_levels(self, memory: MemoryRouter) -> None:
        memory.store(MemoryEntry(level=MemoryLevel.AGENT, agent_id="a1", content="Agent mem", turn_number=1))
        memory.store(MemoryEntry(level=MemoryLevel.TEAM, team_id="t1", content="Team mem", turn_number=1))
        memory.store(MemoryEntry(level=MemoryLevel.GLOBAL, content="Global mem", turn_number=1))

        stats = memory.get_stats()
        assert stats["agent"] >= 1
        assert stats["team"] >= 1
        assert stats["global"] >= 1

    def test_deduplication(self, memory: MemoryRouter) -> None:
        """Same content + agent_id + turn_number should deduplicate."""
        entry = MemoryEntry(
            level=MemoryLevel.AGENT,
            agent_id="agent_001",
            content="Exact same content",
            turn_number=1,
        )
        memory.store(entry)
        memory.store(entry)

        stats = memory.get_stats()
        assert stats["agent"] == 1

    def test_different_content_not_deduped(self, memory: MemoryRouter) -> None:
        memory.store(
            MemoryEntry(
                level=MemoryLevel.AGENT,
                agent_id="a1",
                content="Content A",
                turn_number=1,
            )
        )
        memory.store(
            MemoryEntry(
                level=MemoryLevel.AGENT,
                agent_id="a1",
                content="Content B",
                turn_number=2,
            )
        )
        assert memory.get_stats()["agent"] == 2


# =============================================================================
# Retrieve
# =============================================================================


class TestRetrieve:
    def test_retrieve_by_query(self, memory: MemoryRouter) -> None:
        memory.store(
            MemoryEntry(
                level=MemoryLevel.AGENT,
                agent_id="a1",
                content="Building a FastAPI CRUD application with PostgreSQL",
                turn_number=1,
            )
        )
        results = memory.retrieve(
            query="FastAPI CRUD app",
            levels=[MemoryLevel.AGENT],
            agent_id="a1",
        )
        assert len(results) >= 1
        assert "FastAPI" in results[0].content

    def test_retrieve_filters_by_agent(self, memory: MemoryRouter) -> None:
        memory.store(
            MemoryEntry(
                level=MemoryLevel.AGENT,
                agent_id="a1",
                content="Agent 1 memory",
                turn_number=1,
            )
        )
        memory.store(
            MemoryEntry(
                level=MemoryLevel.AGENT,
                agent_id="a2",
                content="Agent 2 memory",
                turn_number=1,
            )
        )

        results = memory.retrieve(
            query="memory",
            levels=[MemoryLevel.AGENT],
            agent_id="a1",
        )
        # All results should belong to a1
        for r in results:
            assert r.agent_id == "a1"

    def test_retrieve_filters_by_team(self, memory: MemoryRouter) -> None:
        memory.store(
            MemoryEntry(
                level=MemoryLevel.TEAM,
                team_id="team_alpha",
                content="Alpha team using React",
                turn_number=1,
            )
        )
        memory.store(
            MemoryEntry(
                level=MemoryLevel.TEAM,
                team_id="team_beta",
                content="Beta team using Vue",
                turn_number=1,
            )
        )

        results = memory.retrieve(
            query="frontend framework",
            levels=[MemoryLevel.TEAM],
            team_id="team_alpha",
        )
        for r in results:
            assert r.team_id == "team_alpha"

    def test_retrieve_across_levels(self, memory: MemoryRouter) -> None:
        memory.store(
            MemoryEntry(
                level=MemoryLevel.AGENT,
                agent_id="a1",
                content="Agent level: working on auth",
                turn_number=1,
            )
        )
        memory.store(
            MemoryEntry(
                level=MemoryLevel.GLOBAL,
                content="Global: always validate JWT signatures",
                turn_number=1,
            )
        )

        results = memory.retrieve(
            query="authentication",
            levels=[MemoryLevel.AGENT, MemoryLevel.GLOBAL],
        )
        assert len(results) >= 1

    def test_retrieve_empty_results(self, memory: MemoryRouter) -> None:
        results = memory.retrieve(
            query="something that doesn't exist",
            levels=[MemoryLevel.AGENT],
        )
        assert results == []

    def test_retrieve_respects_top_k(self, memory: MemoryRouter) -> None:
        for i in range(20):
            memory.store(
                MemoryEntry(
                    level=MemoryLevel.AGENT,
                    agent_id="a1",
                    content=f"Memory entry number {i} about authentication",
                    turn_number=i,
                )
            )

        results = memory.retrieve(
            query="auth",
            levels=[MemoryLevel.AGENT],
            top_k=3,
        )
        assert len(results) <= 3

    def test_retrieve_by_entry_type(self, memory: MemoryRouter) -> None:
        memory.store(
            MemoryEntry(
                level=MemoryLevel.AGENT,
                agent_id="a1",
                content="Observed the API is slow",
                entry_type="observation",
                turn_number=1,
            )
        )
        memory.store(
            MemoryEntry(
                level=MemoryLevel.AGENT,
                agent_id="a1",
                content="Decided to use caching",
                entry_type="decision",
                turn_number=2,
            )
        )

        results = memory.retrieve(
            query="API performance",
            levels=[MemoryLevel.AGENT],
            entry_type="decision",
        )
        for r in results:
            assert r.entry_type == "decision"


# =============================================================================
# Promotion
# =============================================================================


class TestPromotion:
    def test_promote_agent_to_team(self, memory: MemoryRouter) -> None:
        for i in range(5):
            memory.store(
                MemoryEntry(
                    level=MemoryLevel.AGENT,
                    agent_id="a1",
                    team_id="team_alpha",
                    content=f"Working on feature {i}",
                    turn_number=i,
                )
            )

        memory.promote(
            agent_id="a1",
            team_id="team_alpha",
            summary="Completed features 0-4",
            from_level=MemoryLevel.AGENT,
            to_level=MemoryLevel.TEAM,
        )

        team_results = memory.retrieve(
            query="completed features",
            levels=[MemoryLevel.TEAM],
            team_id="team_alpha",
        )
        assert len(team_results) >= 1

    def test_promote_marks_source_entries(self, memory: MemoryRouter) -> None:
        for i in range(3):
            memory.store(
                MemoryEntry(
                    level=MemoryLevel.AGENT,
                    agent_id="a1",
                    team_id="t1",
                    content=f"Entry {i}",
                    turn_number=i,
                )
            )

        memory.promote(
            agent_id="a1",
            team_id="t1",
            summary="Summary",
            from_level=MemoryLevel.AGENT,
            to_level=MemoryLevel.TEAM,
        )

        # Source entries should now be marked as promoted
        # After promotion, should_promote should return False for turn count
        # because promoted entries are excluded from the count
        turn_count = memory.get_agent_turn_count("a1")
        assert turn_count == 0  # All marked as promoted

    def test_promote_team_to_global(self, memory: MemoryRouter) -> None:
        memory.store(
            MemoryEntry(
                level=MemoryLevel.TEAM,
                agent_id="a1",
                team_id="t1",
                content="Team learned about caching",
                turn_number=1,
            )
        )

        memory.promote(
            agent_id="a1",
            team_id="t1",
            summary="Caching strategy established",
            from_level=MemoryLevel.TEAM,
            to_level=MemoryLevel.GLOBAL,
        )

        assert memory.get_stats()["global"] >= 1


# =============================================================================
# Auto-promotion Threshold
# =============================================================================


class TestShouldPromote:
    def test_below_threshold(self, memory: MemoryRouter) -> None:
        for i in range(5):
            memory.store(
                MemoryEntry(
                    level=MemoryLevel.AGENT,
                    agent_id="a_test",
                    content=f"Entry {i}",
                    turn_number=i,
                )
            )
        assert not memory.should_promote("a_test")

    def test_at_threshold(self, memory: MemoryRouter) -> None:
        # Default promotion interval is 10
        for i in range(10):
            memory.store(
                MemoryEntry(
                    level=MemoryLevel.AGENT,
                    agent_id="a_test",
                    content=f"Turn {i} observation data",
                    turn_number=i,
                )
            )
        assert memory.should_promote("a_test")

    def test_unknown_agent_returns_false(self, memory: MemoryRouter) -> None:
        assert not memory.should_promote("nonexistent_agent")


# =============================================================================
# Clear & Stats
# =============================================================================


class TestClearAndStats:
    def test_clear_agent(self, memory: MemoryRouter) -> None:
        for i in range(3):
            memory.store(
                MemoryEntry(
                    level=MemoryLevel.AGENT,
                    agent_id="a_clear",
                    content=f"Memory {i}",
                    turn_number=i,
                )
            )

        memory.clear_agent("a_clear")

        results = memory.retrieve(
            query="Memory",
            levels=[MemoryLevel.AGENT],
            agent_id="a_clear",
        )
        assert len(results) == 0

    def test_clear_only_affects_target_agent(self, memory: MemoryRouter) -> None:
        memory.store(
            MemoryEntry(
                level=MemoryLevel.AGENT,
                agent_id="a_keep",
                content="Keep this",
                turn_number=1,
            )
        )
        memory.store(
            MemoryEntry(
                level=MemoryLevel.AGENT,
                agent_id="a_delete",
                content="Delete this",
                turn_number=1,
            )
        )

        memory.clear_agent("a_delete")

        results = memory.retrieve(
            query="Keep",
            levels=[MemoryLevel.AGENT],
            agent_id="a_keep",
        )
        assert len(results) >= 1

    def test_get_stats_accuracy(self, memory: MemoryRouter) -> None:
        memory.store(MemoryEntry(level=MemoryLevel.AGENT, agent_id="a1", content="A1", turn_number=1))
        memory.store(MemoryEntry(level=MemoryLevel.AGENT, agent_id="a2", content="A2", turn_number=1))
        memory.store(MemoryEntry(level=MemoryLevel.TEAM, team_id="t1", content="T1", turn_number=1))
        memory.store(MemoryEntry(level=MemoryLevel.GLOBAL, content="G1", turn_number=1))

        stats = memory.get_stats()
        assert stats["agent"] == 2
        assert stats["team"] == 1
        assert stats["global"] == 1

    def test_get_agent_turn_count(self, memory: MemoryRouter) -> None:
        for i in range(7):
            memory.store(
                MemoryEntry(
                    level=MemoryLevel.AGENT,
                    agent_id="a_count",
                    content=f"Turn {i}",
                    turn_number=i,
                )
            )
        assert memory.get_agent_turn_count("a_count") == 7


# =============================================================================
# Embedding Helpers
# =============================================================================


class TestEmbedding:
    def test_stub_embed_returns_correct_dimensions(self) -> None:
        vec = MemoryRouter._stub_embed("Test text")
        from aifactory.config.settings import get_settings

        settings = get_settings()
        assert len(vec) == settings.embedding_dimensions

    def test_stub_embed_deterministic(self) -> None:
        v1 = MemoryRouter._stub_embed("Same text")
        v2 = MemoryRouter._stub_embed("Same text")
        assert v1 == v2

    def test_stub_embed_different_text(self) -> None:
        v1 = MemoryRouter._stub_embed("Text A")
        v2 = MemoryRouter._stub_embed("Text B")
        assert v1 != v2

    def test_stub_embed_values_in_range(self) -> None:
        vec = MemoryRouter._stub_embed("Test")
        assert all(-1.0 <= v <= 1.0 for v in vec)

    def test_content_hash_deterministic(self) -> None:
        h1 = MemoryRouter._content_hash("Same content")
        h2 = MemoryRouter._content_hash("Same content")
        assert h1 == h2

    def test_content_hash_different(self) -> None:
        h1 = MemoryRouter._content_hash("Content A")
        h2 = MemoryRouter._content_hash("Content B")
        assert h1 != h2

    def test_content_hash_is_positive_int(self) -> None:
        h = MemoryRouter._content_hash("Test")
        assert isinstance(h, int)
        assert h > 0
