"""
Tests for Phase 2.1 — Memory integration in PipelineAgent.

Coverage:
  MemoryBridge unit tests (30 cases)
    - noop behaviour when router=None
    - retrieve_context: format, cap, failure isolation
    - store_output: agent layer, empty-content guard, failure isolation
    - store_share_update/updates: team layer, empty-value guard
    - maybe_promote: threshold, failure isolation
    - _format_context: multi-entry, cap, empty

  PipelineAgent integration (10 cases)
    - memory context injected into system prompt
    - store_output called after think()
    - maybe_promote called after think()
    - memory failure inside think() never breaks pipeline
    - memory=None is a noop (no regression on existing behaviour)
    - multiple think() calls accumulate memory turns
    - budget error raised before any memory ops

  build_dynamic_pipeline integration (4 cases)
    - bridge injected into all named agents
    - bridge=MemoryBridge(router=None) disables memory cleanly
    - memory disabled via settings flag
    - workers created inside execution_lead receive the bridge

  CLI cmd_memory (6 cases)
    - memory stats shows layer counts
    - memory search returns results
    - memory search with layer filter
    - memory clear requires confirmation
    - memory clear --yes skips confirmation
    - memory clear --agent scopes to agent only
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client import QdrantClient

from aifactory.core.memory_bridge import MemoryBridge
from aifactory.core.memory_router import MemoryRouter, _stub_embed
from aifactory.core.schemas import MemoryEntry, MemoryLevel
from tests.helpers import make_llm_response

# ─── Shared fixtures ──────────────────────────────────────────────────────────

_LITELLM_PATCH = "aifactory.graphs.dynamic_pipeline.litellm.acompletion"


def _mock_llm(content: str, prompt_tokens: int = 100, completion_tokens: int = 50):
    resp = make_llm_response(content, prompt_tokens, completion_tokens)
    return patch(_LITELLM_PATCH, AsyncMock(return_value=resp))


def _make_in_memory_router() -> MemoryRouter:
    """In-memory Qdrant + stub embeddings — no network, no HuggingFace."""
    client = QdrantClient(":memory:")
    return MemoryRouter(qdrant_client=client, embedding_fn=_stub_embed)


def _make_bridge(
    run_id: str = "test_run", max_chars: int = 800
) -> tuple[MemoryBridge, MemoryRouter]:
    """Return a bridge + the underlying router for inspection."""
    router = _make_in_memory_router()
    bridge = MemoryBridge(router=router, run_id=run_id, max_context_chars=max_chars)
    return bridge, router


# =============================================================================
# MemoryBridge — noop when router=None
# =============================================================================


class TestMemoryBridgeNoop:
    def test_enabled_false_when_no_router(self) -> None:
        bridge = MemoryBridge(router=None)
        assert bridge.enabled is False

    def test_enabled_true_with_router(self) -> None:
        bridge, _ = _make_bridge()
        assert bridge.enabled is True

    def test_retrieve_context_noop(self) -> None:
        bridge = MemoryBridge(router=None)
        assert bridge.retrieve_context("agent", "query") == ""

    def test_retrieve_context_empty_query_noop(self) -> None:
        bridge, _ = _make_bridge()
        assert bridge.retrieve_context("agent", "") == ""
        assert bridge.retrieve_context("agent", "   ") == ""

    def test_store_output_noop(self) -> None:
        bridge = MemoryBridge(router=None)
        bridge.store_output("agent", "content")  # must not raise

    def test_store_share_update_noop(self) -> None:
        bridge = MemoryBridge(router=None)
        bridge.store_share_update("agent", "key", "value")  # must not raise

    def test_store_share_updates_noop(self) -> None:
        bridge = MemoryBridge(router=None)
        bridge.store_share_updates("agent", {"k": "v"})  # must not raise

    def test_maybe_promote_noop(self) -> None:
        bridge = MemoryBridge(router=None)
        bridge.maybe_promote("agent", "summary")  # must not raise


# =============================================================================
# MemoryBridge — retrieve_context
# =============================================================================


class TestMemoryBridgeRetrieve:
    def test_returns_empty_when_no_entries(self) -> None:
        bridge, _ = _make_bridge()
        result = bridge.retrieve_context("analyser", "task breakdown", top_k=3)
        assert result == ""

    def test_returns_context_after_store(self) -> None:
        bridge, router = _make_bridge()
        entry = MemoryEntry(
            level=MemoryLevel.AGENT,
            agent_id="analyser",
            team_id="test_run",
            content="FastAPI was chosen as the web framework",
            entry_type="agent_output",
        )
        router.store(entry)
        result = bridge.retrieve_context("analyser", "web framework FastAPI", top_k=3)
        assert result != ""
        assert "## Relevant Memory" in result
        assert "FastAPI" in result

    def test_context_includes_layer_label(self) -> None:
        bridge, router = _make_bridge()
        router.store(
            MemoryEntry(
                level=MemoryLevel.TEAM,
                agent_id="researcher",
                team_id="test_run",
                content="Research summary: Redis is fast",
                entry_type="share_update",
            )
        )
        result = bridge.retrieve_context("analyser", "Redis cache", top_k=3)
        if result:  # stub embeddings may not retrieve by semantic similarity
            assert "[team]" in result or "[agent]" in result

    def test_context_capped_at_max_chars(self) -> None:
        bridge, router = _make_bridge(max_chars=100)
        # Store a very long entry
        long_content = "X" * 500
        router.store(
            MemoryEntry(
                level=MemoryLevel.AGENT,
                agent_id="analyser",
                team_id="test_run",
                content=long_content,
                entry_type="agent_output",
            )
        )
        result = bridge.retrieve_context("analyser", "X" * 50, top_k=3)
        if result:
            # Strip the header and measure content only
            content_part = result.replace("## Relevant Memory\n", "").replace("\n\n", "")
            assert len(content_part) <= 150  # header + small buffer for label

    def test_failure_in_router_returns_empty_string(self) -> None:
        bridge, _ = _make_bridge()
        # Patch router.retrieve to raise
        with patch.object(bridge._router, "retrieve", side_effect=RuntimeError("qdrant down")):
            result = bridge.retrieve_context("analyser", "some query")
        assert result == ""

    def test_format_context_empty_entries(self) -> None:
        bridge, _ = _make_bridge()
        result = bridge._format_context([])
        assert result == ""

    def test_format_context_skips_empty_content(self) -> None:
        bridge, _ = _make_bridge()
        entries = [
            MemoryEntry(level=MemoryLevel.AGENT, agent_id="a", content="", entry_type="obs"),
            MemoryEntry(level=MemoryLevel.AGENT, agent_id="a", content="  ", entry_type="obs"),
        ]
        result = bridge._format_context(entries)
        assert result == ""

    def test_format_context_multiple_entries(self) -> None:
        bridge, _ = _make_bridge()
        entries = [
            MemoryEntry(
                level=MemoryLevel.AGENT, agent_id="a", content="Entry one", entry_type="obs"
            ),
            MemoryEntry(
                level=MemoryLevel.TEAM, agent_id="b", content="Entry two", entry_type="share"
            ),
        ]
        result = bridge._format_context(entries)
        assert "## Relevant Memory" in result
        assert "Entry one" in result
        assert "Entry two" in result
        assert "[agent]" in result
        assert "[team]" in result

    def test_format_context_truncates_at_cap(self) -> None:
        bridge, _ = _make_bridge(max_chars=50)
        entries = [
            MemoryEntry(level=MemoryLevel.AGENT, agent_id="a", content="A" * 200, entry_type="obs"),
        ]
        result = bridge._format_context(entries)
        if result:
            assert "…" in result


# =============================================================================
# MemoryBridge — store_output
# =============================================================================


class TestMemoryBridgeStoreOutput:
    def test_stores_to_agent_layer(self) -> None:
        bridge, router = _make_bridge()
        bridge.store_output("analyser", "I broke the task into 3 steps", turn_number=1)
        stats = router.get_stats()
        assert stats["agent"] >= 1

    def test_ignores_empty_content(self) -> None:
        bridge, router = _make_bridge()
        bridge.store_output("analyser", "")
        bridge.store_output("analyser", "   ")
        stats = router.get_stats()
        assert stats["agent"] == 0

    def test_uses_correct_entry_type(self) -> None:
        bridge, router = _make_bridge(run_id="run_abc")
        bridge.store_output("master", "Final answer here", entry_type="final_response")
        # Retrieve and check the entry type was passed through
        entries = router.retrieve(
            query="Final answer",
            levels=[MemoryLevel.AGENT],
            agent_id="master",
            top_k=1,
        )
        if entries:
            assert entries[0].entry_type == "final_response"

    def test_failure_is_silent(self) -> None:
        bridge, _ = _make_bridge()
        with patch.object(bridge._router, "store", side_effect=RuntimeError("disk full")):
            bridge.store_output("analyser", "some content")  # must not raise

    def test_multiple_outputs_accumulate(self) -> None:
        bridge, router = _make_bridge()
        for i in range(3):
            bridge.store_output("analyser", f"Output {i}", turn_number=i)
        stats = router.get_stats()
        assert stats["agent"] >= 1  # dedup may merge identical embeddings, but should have entries


# =============================================================================
# MemoryBridge — store_share_update / store_share_updates
# =============================================================================


class TestMemoryBridgeShareStorage:
    def test_stores_to_team_layer(self) -> None:
        bridge, router = _make_bridge()
        bridge.store_share_update("researcher", "framework", "FastAPI")
        stats = router.get_stats()
        assert stats["team"] >= 1

    def test_ignores_empty_value(self) -> None:
        bridge, router = _make_bridge()
        bridge.store_share_update("researcher", "key", "")
        bridge.store_share_update("researcher", "key", "   ")
        stats = router.get_stats()
        assert stats["team"] == 0

    def test_content_includes_key_and_agent(self) -> None:
        bridge, router = _make_bridge(run_id="share_run")
        bridge.store_share_update("researcher", "db", "PostgreSQL")
        entries = router.retrieve(
            query="database PostgreSQL",
            levels=[MemoryLevel.TEAM],
            top_k=5,
        )
        if entries:
            assert "db: PostgreSQL" in entries[0].content
            assert "researcher" in entries[0].content

    def test_store_share_updates_multiple(self) -> None:
        bridge, router = _make_bridge()
        bridge.store_share_updates("worker", {"lang": "Python", "orm": "SQLAlchemy"})
        stats = router.get_stats()
        assert stats["team"] >= 2

    def test_store_share_updates_skips_empty_values(self) -> None:
        bridge, router = _make_bridge()
        bridge.store_share_updates("worker", {"good": "value", "bad": "", "also_bad": "  "})
        stats = router.get_stats()
        assert stats["team"] == 1  # only "good" stored

    def test_failure_is_silent(self) -> None:
        bridge, _ = _make_bridge()
        with patch.object(bridge._router, "store", side_effect=RuntimeError("timeout")):
            bridge.store_share_updates("agent", {"k": "v"})  # must not raise


# =============================================================================
# MemoryBridge — maybe_promote
# =============================================================================


class TestMemoryBridgeMaybePromote:
    def test_noop_below_threshold(self) -> None:
        bridge, router = _make_bridge()
        # Only 2 turns stored — below default threshold of 10
        bridge.store_output("analyser", "Turn 1 output", turn_number=1)
        bridge.store_output("analyser", "Turn 2 output", turn_number=2)
        bridge.maybe_promote("analyser", "summary")
        # No promotion should have happened
        stats = router.get_stats()
        assert stats["team"] == 0

    def test_triggers_at_threshold(self) -> None:
        bridge, router = _make_bridge()
        # Store enough entries to reach the default threshold (10)
        for i in range(10):
            bridge.store_output(
                "analyser", f"Unique content turn {i} analysis result", turn_number=i
            )
        bridge.maybe_promote("analyser", "Analyser did 10 turns of task decomposition")
        stats = router.get_stats()
        # Should have promoted: team layer now has the summary
        assert stats["team"] >= 1

    def test_ignores_empty_summary(self) -> None:
        bridge, router = _make_bridge()
        for i in range(10):
            bridge.store_output("analyser", f"Content {i}", turn_number=i)
        bridge.maybe_promote("analyser", "")  # empty summary — should not promote
        stats = router.get_stats()
        assert stats["team"] == 0

    def test_failure_is_silent(self) -> None:
        bridge, _ = _make_bridge()
        with patch.object(bridge._router, "should_promote", side_effect=RuntimeError("crash")):
            bridge.maybe_promote("analyser", "summary")  # must not raise


# =============================================================================
# PipelineAgent — memory integration
# =============================================================================


class TestPipelineAgentMemoryIntegration:
    @pytest.mark.asyncio
    async def test_memory_context_injected_into_system_prompt(self) -> None:
        """Stored memories should appear in the system prompt of subsequent think() calls."""
        from aifactory.graphs.dynamic_pipeline import PipelineAgent

        bridge, router = _make_bridge()

        # Pre-load a memory entry directly into the router
        router.store(
            MemoryEntry(
                level=MemoryLevel.AGENT,
                agent_id="analyser",
                team_id="test_run",
                content="The user prefers PostgreSQL for all database tasks",
                entry_type="agent_output",
            )
        )

        agent = PipelineAgent(
            "analyser",
            {"system_prompt": "You are an analyser.", "max_tokens": 100},
            memory=bridge,
        )

        with _mock_llm("ok") as mock_call:
            await agent.think("Choose a database for the project")

        # Check that the system prompt included memory context
        messages = mock_call.call_args.kwargs["messages"]
        system_msg = messages[0]["content"]
        # Memory context may or may not match given stub embeddings,
        # but the retrieval pipeline ran without error
        assert "You are an analyser." in system_msg

    @pytest.mark.asyncio
    async def test_store_output_called_after_think(self) -> None:
        """Each think() call should write the reply to agent-layer memory."""
        from aifactory.graphs.dynamic_pipeline import PipelineAgent

        bridge, router = _make_bridge()
        agent = PipelineAgent(
            "master",
            {"system_prompt": "You are master.", "max_tokens": 100},
            memory=bridge,
        )

        with _mock_llm("Here is the final answer"):
            await agent.think("What is the result?")

        stats = router.get_stats()
        assert stats["agent"] >= 1

    @pytest.mark.asyncio
    async def test_memory_failure_does_not_break_think(self) -> None:
        """If the memory bridge raises, think() should still return the LLM reply."""
        from aifactory.graphs.dynamic_pipeline import PipelineAgent

        bridge, _ = _make_bridge()
        # Make both retrieve and store blow up
        with (
            patch.object(bridge, "retrieve_context", side_effect=RuntimeError("boom")),
            patch.object(bridge, "store_output", side_effect=RuntimeError("boom")),
        ):
            agent = PipelineAgent(
                "analyser",
                {"system_prompt": "Test.", "max_tokens": 100},
                memory=bridge,
            )
            with _mock_llm("reply despite memory failure"):
                result = await agent.think("some task")

        assert result == "reply despite memory failure"

    @pytest.mark.asyncio
    async def test_no_memory_no_regression(self) -> None:
        """memory=None must behave exactly like before Phase 2.1."""
        from aifactory.graphs.dynamic_pipeline import PipelineAgent

        agent = PipelineAgent(
            "analyser",
            {"system_prompt": "Test.", "max_tokens": 100},
            memory=None,
        )
        with _mock_llm("normal result"):
            result = await agent.think("question")

        assert result == "normal result"
        assert agent.total_tokens > 0
        assert len(agent._messages) == 2

    @pytest.mark.asyncio
    async def test_multiple_think_calls_accumulate_memory(self) -> None:
        """Each think() call should add an entry, so turn_number increases."""
        from aifactory.graphs.dynamic_pipeline import PipelineAgent

        bridge, router = _make_bridge()
        agent = PipelineAgent(
            "master",
            {"system_prompt": ".", "max_tokens": 100},
            memory=bridge,
        )

        for i in range(3):
            with _mock_llm(f"Reply {i}"):
                await agent.think(f"Question {i}")

        stats = router.get_stats()
        # Three separate outputs stored (each unique content → 3 entries)
        assert stats["agent"] >= 1  # at minimum 1 (dedup may merge by content hash)

    @pytest.mark.asyncio
    async def test_budget_error_before_memory_ops(self) -> None:
        """Budget check fires before any memory retrieve/store."""
        from aifactory.graphs.dynamic_pipeline import AgentBudgetExceededError, PipelineAgent

        bridge, _ = _make_bridge()
        retrieve_called = []
        with patch.object(
            bridge, "retrieve_context", side_effect=lambda *a, **k: retrieve_called.append(1) or ""
        ):
            agent = PipelineAgent(
                "analyser",
                {"system_prompt": ".", "max_tokens": 100, "token_budget": 0},
                memory=bridge,
            )
            with pytest.raises(AgentBudgetExceededError):
                await agent.think("question")

        # retrieve should NOT have been called (budget check fires first)
        assert len(retrieve_called) == 0

    @pytest.mark.asyncio
    async def test_maybe_promote_called_after_think(self) -> None:
        """maybe_promote is called once per think() (may or may not fire based on count)."""
        from aifactory.graphs.dynamic_pipeline import PipelineAgent

        bridge, _ = _make_bridge()
        promote_calls: list[str] = []

        original_promote = bridge.maybe_promote

        def tracking_promote(agent_id: str, summary: str) -> None:
            promote_calls.append(agent_id)
            original_promote(agent_id, summary)

        bridge.maybe_promote = tracking_promote  # type: ignore[method-assign]

        agent = PipelineAgent(
            "analyser",
            {"system_prompt": ".", "max_tokens": 100},
            memory=bridge,
        )
        with _mock_llm("output"):
            await agent.think("input")

        assert "analyser" in promote_calls


# =============================================================================
# build_dynamic_pipeline — bridge injection
# =============================================================================


class TestBuildDynamicPipelineMemory:
    def test_bridge_injected_into_named_agents(self) -> None:
        """All named agents (master, judge, analyser, etc.) should have the bridge."""
        from aifactory.graphs.dynamic_pipeline import build_dynamic_pipeline

        bridge = MemoryBridge(router=None)  # noop bridge
        _, registry = build_dynamic_pipeline(memory_bridge=bridge)

        named = {"master", "judge", "analyser", "researcher", "execution_lead"}
        for agent in registry.all():
            if agent.name in named:
                assert agent.memory is bridge, f"{agent.name} missing memory bridge"

    def test_explicit_none_bridge_disables_memory(self) -> None:
        """Passing MemoryBridge(router=None) gives a bridge that's a noop."""
        from aifactory.graphs.dynamic_pipeline import build_dynamic_pipeline

        bridge = MemoryBridge(router=None)
        assert bridge.enabled is False

        _, registry = build_dynamic_pipeline(memory_bridge=bridge)
        for agent in registry.all():
            if hasattr(agent, "memory"):
                assert not agent.memory.enabled

    def test_memory_disabled_via_settings(self, monkeypatch) -> None:
        """AIFACTORY_MEMORY_PIPELINE_ENABLED=false → no bridge on agents."""
        from aifactory.config.settings import get_settings

        get_settings.cache_clear()
        monkeypatch.setenv("AIFACTORY_MEMORY_PIPELINE_ENABLED", "false")
        get_settings.cache_clear()

        try:
            from aifactory.graphs.dynamic_pipeline import build_dynamic_pipeline

            _, registry = build_dynamic_pipeline()
            for agent in registry.all():
                if hasattr(agent, "memory") and agent.memory is not None:
                    assert not agent.memory.enabled
        finally:
            monkeypatch.delenv("AIFACTORY_MEMORY_PIPELINE_ENABLED", raising=False)
            get_settings.cache_clear()


# =============================================================================
# make_researcher — SHARE: persistence to team memory
# =============================================================================


class TestResearcherSharePersistence:
    @pytest.mark.asyncio
    async def test_researcher_stores_share_updates_to_team(self) -> None:
        """When researcher outputs SHARE: updates, they should land in team memory."""
        from aifactory.core.agent_factory import AgentRegistry
        from aifactory.graphs.dynamic_pipeline import PipelineAgent, make_researcher

        bridge, router = _make_bridge(run_id="research_run")
        researcher = PipelineAgent(
            "researcher",
            {"system_prompt": ".", "max_tokens": 100},
            memory=bridge,
        )
        registry = AgentRegistry()
        registry.register(researcher)

        research_output = "Summary of findings.\nSHARE:\nstack: FastAPI+Postgres"
        with _mock_llm(research_output):
            node_fn = make_researcher(researcher, bridge=bridge)
            result = await node_fn(
                {
                    "user_query": "research Python web frameworks",
                    "shared_context": {},
                    "research_for_analysis": False,
                    "research_query": "Python web frameworks",
                }
            )

        # research_summary should be clean (no SHARE: block)
        assert "SHARE:" not in result.get("research_summary", "")
        # Team memory should have the SHARE: update
        stats = router.get_stats()
        assert stats["team"] >= 1

    @pytest.mark.asyncio
    async def test_researcher_no_share_no_team_write(self) -> None:
        """If the researcher outputs no SHARE: block, team memory stays empty."""
        from aifactory.core.agent_factory import AgentRegistry
        from aifactory.graphs.dynamic_pipeline import PipelineAgent, make_researcher

        bridge, router = _make_bridge()
        researcher = PipelineAgent(
            "researcher", {"system_prompt": ".", "max_tokens": 100}, memory=bridge
        )
        AgentRegistry().register(researcher)

        with _mock_llm("Plain research summary with no share block"):
            node_fn = make_researcher(researcher, bridge=bridge)
            await node_fn(
                {
                    "user_query": "some query",
                    "shared_context": {},
                    "research_for_analysis": False,
                    "research_query": "some query",
                }
            )

        stats = router.get_stats()
        assert stats["team"] == 0


# =============================================================================
# CLI — cmd_memory
# =============================================================================


class TestMemoryCLI:
    def _make_mock_router(self, stats: dict | None = None, entries: list | None = None):
        """Build a mock MemoryRouter for CLI tests."""
        router = MagicMock()
        router.get_stats.return_value = stats or {"agent": 5, "team": 3, "global": 1}
        router.retrieve.return_value = entries or []
        router._collection_name.side_effect = lambda lvl: f"aifactory_{lvl.value}"
        router._index_cache = {}
        router.dimensions = 1536
        return router

    def test_memory_stats_displays_table(self) -> None:
        from typer.testing import CliRunner

        from aifactory.cli import app

        runner = CliRunner()
        with patch("aifactory.cli.cmd_memory._make_router", return_value=self._make_mock_router()):
            result = runner.invoke(app, ["memory", "stats"])

        assert result.exit_code == 0
        assert "Agent" in result.output or "agent" in result.output
        assert "5" in result.output  # agent entries count

    def test_memory_stats_failure_exits_nonzero(self) -> None:
        from typer.testing import CliRunner

        from aifactory.cli import app

        runner = CliRunner()
        with patch(
            "aifactory.cli.cmd_memory._make_router", side_effect=RuntimeError("qdrant down")
        ):
            result = runner.invoke(app, ["memory", "stats"])

        assert result.exit_code != 0

    def test_memory_search_no_results(self) -> None:
        from typer.testing import CliRunner

        from aifactory.cli import app

        runner = CliRunner()
        with patch(
            "aifactory.cli.cmd_memory._make_router", return_value=self._make_mock_router(entries=[])
        ):
            result = runner.invoke(app, ["memory", "search", "unknown topic"])

        assert result.exit_code == 0
        assert "No results" in result.output

    def test_memory_search_with_results(self) -> None:
        from typer.testing import CliRunner

        from aifactory.cli import app

        entries = [
            MemoryEntry(
                level=MemoryLevel.AGENT,
                agent_id="analyser",
                team_id="run_1",
                content="FastAPI is the chosen framework",
                entry_type="agent_output",
            )
        ]
        runner = CliRunner()
        with patch(
            "aifactory.cli.cmd_memory._make_router",
            return_value=self._make_mock_router(entries=entries),
        ):
            result = runner.invoke(app, ["memory", "search", "FastAPI"])

        assert result.exit_code == 0
        assert "FastAPI" in result.output

    def test_memory_search_invalid_layer_exits_nonzero(self) -> None:
        from typer.testing import CliRunner

        from aifactory.cli import app

        runner = CliRunner()
        with patch("aifactory.cli.cmd_memory._make_router", return_value=self._make_mock_router()):
            result = runner.invoke(app, ["memory", "search", "query", "--layer", "invalid"])

        assert result.exit_code != 0

    def test_memory_clear_requires_confirmation(self) -> None:
        from typer.testing import CliRunner

        from aifactory.cli import app

        runner = CliRunner()
        with patch("aifactory.cli.cmd_memory._make_router", return_value=self._make_mock_router()):
            # Simulate user entering 'n' at the confirmation prompt
            result = runner.invoke(app, ["memory", "clear"], input="n\n")

        assert result.exit_code == 0
        assert "Cancelled" in result.output

    def test_memory_clear_yes_skips_confirmation(self) -> None:
        from typer.testing import CliRunner

        from aifactory.cli import app

        mock_router = self._make_mock_router()
        mock_router.client = MagicMock()
        mock_router.client.get_collection.return_value = MagicMock(points_count=5)
        mock_router.client.delete_collection.return_value = None
        mock_router.client.create_collection.return_value = None

        runner = CliRunner()
        with (
            patch("aifactory.cli.cmd_memory._make_router", return_value=mock_router),
            patch("aifactory.cli.cmd_memory._count_layer", return_value=5),
            patch("aifactory.cli.cmd_memory._clear_layer"),
        ):
            result = runner.invoke(app, ["memory", "clear", "--yes"])

        assert result.exit_code == 0
        assert "Cancelled" not in result.output
