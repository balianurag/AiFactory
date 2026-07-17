"""
Tests for MemoryBridge.retrieve_global_context relevance filtering.

Regression: stale cross-run facts (e.g. old coding facts) were being injected
into unrelated queries (business research) because the filesystem fallback did
a blind [:N] slice with no relevance check. See CLAUDE.md quality-fix notes.
"""

from __future__ import annotations

from aifactory.core.memory_bridge import MemoryBridge


class _FakeFactsStore:
    def __init__(self, facts: list[str]) -> None:
        self._facts = facts

    def load_all(self) -> list[str]:
        return self._facts


def _bridge_with_facts(facts: list[str]) -> MemoryBridge:
    # router=None forces the filesystem FactsStore fallback path
    bridge = MemoryBridge(router=None)
    bridge._facts_store = _FakeFactsStore(facts)
    return bridge


class TestGlobalContextRelevance:
    def test_irrelevant_facts_filtered_out(self) -> None:
        # Old coding facts must NOT leak into a business research query
        stale = [
            "Python uses GIL for thread safety",
            "FastAPI is built on Starlette",
            "Redis uses in-memory data structures",
            "Kubernetes uses etcd for state storage",
        ]
        bridge = _bridge_with_facts(stale)
        ctx = bridge.retrieve_global_context(
            query="top micro-SaaS business ideas for Indian SMBs revenue potential",
            top_k=2,
        )
        # None of these coding facts share meaningful vocabulary with the query
        assert "GIL" not in ctx
        assert "Starlette" not in ctx
        assert "etcd" not in ctx

    def test_relevant_facts_retained(self) -> None:
        facts = [
            "Indian SMB market projected to reach $1 trillion by 2025",
            "Python uses GIL for thread safety",
        ]
        bridge = _bridge_with_facts(facts)
        ctx = bridge.retrieve_global_context(
            query="Indian SMB market size and business revenue potential",
            top_k=2,
        )
        assert "trillion" in ctx  # relevant fact kept
        assert "GIL" not in ctx  # irrelevant fact dropped

    def test_empty_when_nothing_relevant(self) -> None:
        bridge = _bridge_with_facts(["Python uses GIL for thread safety"])
        ctx = bridge.retrieve_global_context(query="quarterly sales forecast", top_k=2)
        assert ctx == ""
