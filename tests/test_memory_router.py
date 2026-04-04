"""
Tests for the Memory Router.
"""

from __future__ import annotations

import pytest

from aifactory.core.memory_router import MemoryRouter, _stub_embed
from aifactory.core.schemas import MemoryEntry, MemoryLevel


@pytest.fixture
def memory() -> MemoryRouter:
    """Create an in-memory MemoryRouter for testing (with stub embeddings)."""
    from qdrant_client import QdrantClient

    client = QdrantClient(":memory:")
    return MemoryRouter(qdrant_client=client, embedding_fn=_stub_embed)


def test_store_and_retrieve(memory: MemoryRouter) -> None:
    """Store an entry and retrieve it."""
    entry = MemoryEntry(
        level=MemoryLevel.AGENT,
        agent_id="agent_001",
        team_id="team_alpha",
        content="The user wants a FastAPI CRUD app with PostgreSQL",
        entry_type="observation",
        turn_number=1,
    )
    memory.store(entry)

    results = memory.retrieve(
        query="FastAPI CRUD application",
        levels=[MemoryLevel.AGENT],
        agent_id="agent_001",
    )

    assert len(results) >= 1
    assert "FastAPI" in results[0].content


def test_store_multiple_levels(memory: MemoryRouter) -> None:
    """Store entries at different levels and retrieve from specific ones."""
    # Agent level
    memory.store(
        MemoryEntry(
            level=MemoryLevel.AGENT,
            agent_id="agent_001",
            content="Working on authentication module",
            turn_number=1,
        )
    )

    # Team level
    memory.store(
        MemoryEntry(
            level=MemoryLevel.TEAM,
            team_id="team_alpha",
            content="Team is building a web app with auth",
            turn_number=1,
        )
    )

    # Global level
    memory.store(
        MemoryEntry(
            level=MemoryLevel.GLOBAL,
            content="Auth best practice: use bcrypt for password hashing",
            entry_type="learning",
            turn_number=1,
        )
    )

    stats = memory.get_stats()
    assert stats["agent"] >= 1
    assert stats["team"] >= 1
    assert stats["global"] >= 1


def test_promotion(memory: MemoryRouter) -> None:
    """Test promoting entries from agent → team layer."""
    # Store several agent entries
    for i in range(5):
        memory.store(
            MemoryEntry(
                level=MemoryLevel.AGENT,
                agent_id="agent_001",
                team_id="team_alpha",
                content=f"Observation {i}: working on feature {i}",
                turn_number=i,
            )
        )

    # Promote
    memory.promote(
        agent_id="agent_001",
        team_id="team_alpha",
        summary="Agent completed features 0-4 successfully.",
        from_level=MemoryLevel.AGENT,
        to_level=MemoryLevel.TEAM,
    )

    # Verify team layer has the promoted summary
    team_results = memory.retrieve(
        query="completed features",
        levels=[MemoryLevel.TEAM],
        team_id="team_alpha",
    )
    assert len(team_results) >= 1


def test_should_promote(memory: MemoryRouter) -> None:
    """Test the auto-promotion threshold check."""
    # Default promotion interval is 10
    for i in range(9):
        memory.store(
            MemoryEntry(
                level=MemoryLevel.AGENT,
                agent_id="agent_test",
                content=f"Turn {i} observation",
                turn_number=i,
            )
        )

    # Not yet at threshold
    assert not memory.should_promote("agent_test")

    # Add one more
    memory.store(
        MemoryEntry(
            level=MemoryLevel.AGENT,
            agent_id="agent_test",
            content="Turn 9 observation",
            turn_number=9,
        )
    )

    assert memory.should_promote("agent_test")


def test_deduplication(memory: MemoryRouter) -> None:
    """Storing the same content twice should not create duplicates."""
    entry = MemoryEntry(
        level=MemoryLevel.AGENT,
        agent_id="agent_001",
        content="Exact same content",
        turn_number=1,
    )

    memory.store(entry)
    memory.store(entry)

    stats = memory.get_stats()
    assert stats["agent"] == 1  # Should be deduplicated


def test_clear_agent(memory: MemoryRouter) -> None:
    """Test clearing all memories for an agent."""
    for i in range(3):
        memory.store(
            MemoryEntry(
                level=MemoryLevel.AGENT,
                agent_id="agent_to_clear",
                content=f"Memory {i}",
                turn_number=i,
            )
        )

    memory.clear_agent("agent_to_clear")

    results = memory.retrieve(
        query="Memory",
        levels=[MemoryLevel.AGENT],
        agent_id="agent_to_clear",
    )
    assert len(results) == 0
