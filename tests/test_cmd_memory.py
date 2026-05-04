"""
Tests for cli/cmd_memory.py — memory stats / search / health / clear commands.

Covers:
  - memory_stats: shows FactsStore count row
  - memory_search: includes FactsStore results for global/all layers
  - memory_health: shows Persistent Facts section
  - memory_clear: basic invocation (no crash)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from aifactory.cli.cmd_memory import memory_app

runner = CliRunner()


# =============================================================================
# Helpers
# =============================================================================


def _make_router_mock(stats: dict | None = None) -> MagicMock:
    """Return a MemoryRouter mock with configurable stats."""
    router = MagicMock()
    router.get_stats.return_value = stats or {"agent": 3, "team": 5, "global": 2}
    router.client = MagicMock()
    # get_collections returns empty list (no semantic cache collection)
    router.client.get_collections.return_value = MagicMock(collections=[])
    return router


def _patch_router(stats: dict | None = None):
    """Context manager: patch _make_router() in cmd_memory to return a mock."""
    return patch(
        "aifactory.cli.cmd_memory._make_router",
        return_value=_make_router_mock(stats),
    )


# =============================================================================
# memory stats
# =============================================================================


class TestMemoryStats:
    def test_stats_shows_layer_counts(self) -> None:
        with _patch_router({"agent": 10, "team": 5, "global": 2}):
            with patch("aifactory.core.facts_store.FactsStore.size", return_value=0):
                result = runner.invoke(memory_app, ["stats"])
        assert result.exit_code == 0
        assert "10" in result.output
        assert "5" in result.output

    def test_stats_shows_facts_store_row(self, tmp_path: Path) -> None:
        from aifactory.core.facts_store import FactsStore

        fs = FactsStore(store_dir=tmp_path)
        fs.store("agent1", "Python uses GIL for thread safety", run_id="r1")
        fs.store("agent2", "FastAPI is built on Starlette", run_id="r1")

        with (
            _patch_router(),
            patch(
                "aifactory.core.facts_store.FactsStore",
                return_value=fs,
            ),
        ):
            result = runner.invoke(memory_app, ["stats"])

        assert result.exit_code == 0
        # Should show "2" somewhere in the FactsStore row
        assert "2" in result.output

    def test_stats_facts_zero_when_empty(self, tmp_path: Path) -> None:
        from aifactory.core.facts_store import FactsStore

        fs = FactsStore(store_dir=tmp_path)
        with _patch_router():
            with patch("aifactory.core.facts_store.FactsStore", return_value=fs):
                result = runner.invoke(memory_app, ["stats"])
        assert result.exit_code == 0

    def test_stats_verbose_flag_accepted(self) -> None:
        with _patch_router():
            with patch("aifactory.core.facts_store.FactsStore.size", return_value=0):
                with patch("aifactory.cli.cmd_memory._print_verbose_stats") as mock_verbose:
                    result = runner.invoke(memory_app, ["stats", "--verbose"])
        assert result.exit_code == 0
        mock_verbose.assert_called_once()


# =============================================================================
# memory search
# =============================================================================


class TestMemorySearch:
    def _mock_entry(self, content: str, level_value: str = "agent") -> MagicMock:
        entry = MagicMock()
        entry.content = content
        entry.level = MagicMock()
        entry.level.value = level_value
        entry.agent_id = "test-agent"
        entry.entry_type = "observation"
        return entry

    def test_search_shows_qdrant_results(self) -> None:
        entries = [self._mock_entry("FastAPI handles HTTP requests")]
        with _patch_router():
            with patch("aifactory.cli.cmd_memory._make_router") as mock_make_router:
                router = _make_router_mock()
                router.retrieve.return_value = entries
                mock_make_router.return_value = router
                result = runner.invoke(memory_app, ["search", "FastAPI"])

        assert result.exit_code == 0

    def test_search_includes_facts_for_global_layer(self, tmp_path: Path) -> None:
        from aifactory.core.facts_store import FactsStore

        fs = FactsStore(store_dir=tmp_path)
        fs.store("worker1", "Redis uses in-memory data structures", run_id="r1")

        with patch("aifactory.cli.cmd_memory._make_router") as mock_make_router:
            router = _make_router_mock()
            router.retrieve.return_value = []
            mock_make_router.return_value = router
            with patch("aifactory.core.facts_store.FactsStore", return_value=fs):
                result = runner.invoke(memory_app, ["search", "Redis memory", "--layer", "global"])

        assert result.exit_code == 0
        assert "Redis" in result.output

    def test_search_includes_facts_for_all_layers(self, tmp_path: Path) -> None:
        from aifactory.core.facts_store import FactsStore

        fs = FactsStore(store_dir=tmp_path)
        fs.store("w1", "Kubernetes uses etcd for state storage", run_id="r1")

        with patch("aifactory.cli.cmd_memory._make_router") as mock_make_router:
            router = _make_router_mock()
            router.retrieve.return_value = []
            mock_make_router.return_value = router
            with patch("aifactory.core.facts_store.FactsStore", return_value=fs):
                result = runner.invoke(memory_app, ["search", "Kubernetes etcd", "--layer", "all"])

        assert result.exit_code == 0
        assert "Kubernetes" in result.output

    def test_search_no_facts_for_agent_layer(self, tmp_path: Path) -> None:
        """FactsStore should NOT appear when searching agent layer only."""
        from aifactory.core.facts_store import FactsStore

        fs = FactsStore(store_dir=tmp_path)
        fs.store("w1", "exclusive agent fact about kubernetes", run_id="r1")

        with patch("aifactory.cli.cmd_memory._make_router") as mock_make_router:
            router = _make_router_mock()
            router.retrieve.return_value = []
            mock_make_router.return_value = router
            with patch("aifactory.core.facts_store.FactsStore", return_value=fs):
                result = runner.invoke(memory_app, ["search", "kubernetes", "--layer", "agent"])

        assert result.exit_code == 0
        # "exclusive agent fact" should NOT appear (facts not queried for agent layer)
        assert "exclusive agent fact" not in result.output

    def test_search_no_results_message(self, tmp_path: Path) -> None:
        from aifactory.core.facts_store import FactsStore

        fs = FactsStore(store_dir=tmp_path)  # empty
        with patch("aifactory.cli.cmd_memory._make_router") as mock_make_router:
            router = _make_router_mock()
            router.retrieve.return_value = []
            mock_make_router.return_value = router
            with patch("aifactory.core.facts_store.FactsStore", return_value=fs):
                result = runner.invoke(memory_app, ["search", "xyzzy unmatched query 999"])
        assert result.exit_code == 0
        assert "No results" in result.output

    def test_search_invalid_layer_exits_nonzero(self) -> None:
        result = runner.invoke(memory_app, ["search", "anything", "--layer", "invalid"])
        assert result.exit_code != 0

    def test_search_top_k_respected(self) -> None:
        with patch("aifactory.cli.cmd_memory._make_router") as mock_make_router:
            router = _make_router_mock()
            router.retrieve.return_value = []
            mock_make_router.return_value = router
            with patch("aifactory.core.facts_store.FactsStore.load_all", return_value=[]):
                result = runner.invoke(memory_app, ["search", "query", "--top-k", "3"])
        assert result.exit_code == 0
        router.retrieve.assert_called_once()
        _, kwargs = router.retrieve.call_args
        assert kwargs.get("top_k") == 3


# =============================================================================
# memory health
# =============================================================================


class TestMemoryHealth:
    def test_health_runs_without_error(self) -> None:
        with patch("aifactory.core.memory_router.MemoryRouter") as mock_router_cls:
            router = MagicMock()
            router.in_memory_mode = True
            router.client = MagicMock()
            router.client.get_collections.return_value = MagicMock(collections=[])
            mock_router_cls.return_value = router
            with (
                patch(
                    "aifactory.core.embed_provider.resolve_provider",
                    return_value=("stub", "stub", 1536),
                ),
                patch("aifactory.core.facts_store.FactsStore.size", return_value=0),
            ):
                result = runner.invoke(memory_app, ["health"])
        assert result.exit_code == 0

    def test_health_shows_facts_section(self, tmp_path: Path) -> None:
        from aifactory.core.facts_store import FactsStore

        fs = FactsStore(store_dir=tmp_path)
        fs.store("w1", "a stored fact", run_id="r1")

        with patch("aifactory.core.memory_router.MemoryRouter") as mock_router_cls:
            router = MagicMock()
            router.in_memory_mode = True
            router.client = MagicMock()
            router.client.get_collections.return_value = MagicMock(collections=[])
            mock_router_cls.return_value = router
            with (
                patch(
                    "aifactory.core.embed_provider.resolve_provider",
                    return_value=("stub", "stub", 1536),
                ),
                patch("aifactory.core.facts_store.FactsStore", return_value=fs),
            ):
                result = runner.invoke(memory_app, ["health"])

        assert result.exit_code == 0
        assert "1 facts" in result.output or "facts" in result.output.lower()

    def test_health_shows_qdrant_in_memory_warning(self) -> None:
        with patch("aifactory.core.memory_router.MemoryRouter") as mock_router_cls:
            router = MagicMock()
            router.in_memory_mode = True
            router.client = MagicMock()
            router.client.get_collections.return_value = MagicMock(collections=[])
            mock_router_cls.return_value = router
            with (
                patch(
                    "aifactory.core.embed_provider.resolve_provider",
                    return_value=("stub", "stub", 1536),
                ),
                patch("aifactory.core.facts_store.FactsStore.size", return_value=0),
            ):
                result = runner.invoke(memory_app, ["health"])
        assert result.exit_code == 0
        assert "in-memory" in result.output


# =============================================================================
# memory clear
# =============================================================================


class TestMemoryClear:
    def test_clear_with_yes_flag(self) -> None:
        with patch("aifactory.cli.cmd_memory._make_router") as mock_make_router:
            router = _make_router_mock()
            mock_make_router.return_value = router
            with patch("aifactory.cli.cmd_memory._count_layer", return_value=3):
                with patch("aifactory.cli.cmd_memory._clear_layer"):
                    result = runner.invoke(memory_app, ["clear", "--yes"])
        assert result.exit_code == 0

    def test_clear_specific_agent(self) -> None:
        with patch("aifactory.cli.cmd_memory._make_router") as mock_make_router:
            router = _make_router_mock()
            mock_make_router.return_value = router
            with patch("aifactory.cli.cmd_memory._delete_agent_from_layer"):
                result = runner.invoke(memory_app, ["clear", "--agent", "analyser", "--yes"])
        assert result.exit_code == 0

    def test_clear_invalid_layer(self) -> None:
        result = runner.invoke(memory_app, ["clear", "--layer", "bogus", "--yes"])
        assert result.exit_code != 0
