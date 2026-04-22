"""
Tests for cli/cmd_runs.py — list, show, search commands.

Covers:
  - list_runs: no records, N records, quality colouring, row content
  - show_run: found, not found (exits 1)
  - search_runs: no results, results shown with preview
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from aifactory.cli.cmd_runs import runs_app
from aifactory.core.run_store import RunRecord


runner = CliRunner()


# =============================================================================
# Helpers
# =============================================================================


def _make_record(**overrides) -> RunRecord:
    defaults = dict(
        run_id="run-20260416-001",
        timestamp=datetime.now(UTC).isoformat(),
        query="Build a REST API for a bookshop",
        mode="balanced",
        elapsed_seconds=12.3,
        gate="",
        model_selections={"master": "gpt-4o"},
        task_count=3,
        quality_flagged_count=0,
        composite_quality=8.5,
        task_types=["coding"],
        final_response_preview="Here is the REST API implementation...",
    )
    defaults.update(overrides)
    return RunRecord(**defaults)


def _mock_store(records: list[RunRecord] | None = None, *, record: RunRecord | None = None):
    store = MagicMock()
    store.list_recent.return_value = records or []
    store.get.return_value = record
    store.search_similar.return_value = records or []
    return store


# =============================================================================
# list_runs
# =============================================================================


class TestListRuns:
    def test_no_records_shows_message(self) -> None:
        with patch("aifactory.core.run_store.get_run_store", return_value=_mock_store()):
            result = runner.invoke(runs_app, ["list"])
        assert result.exit_code == 0
        assert "No run history" in result.output

    def test_records_shown_in_table(self) -> None:
        records = [_make_record()]
        with patch("aifactory.core.run_store.get_run_store", return_value=_mock_store(records)):
            result = runner.invoke(runs_app, ["list"])
        assert result.exit_code == 0
        assert "run-20260416-001" in result.output or "REST API" in result.output

    def test_multiple_records_shown(self) -> None:
        records = [
            _make_record(run_id=f"run-{i}", query=f"task {i}") for i in range(5)
        ]
        with patch("aifactory.core.run_store.get_run_store", return_value=_mock_store(records)):
            result = runner.invoke(runs_app, ["list"])
        assert result.exit_code == 0

    def test_last_flag_passed_to_store(self) -> None:
        mock_store = _mock_store()
        with patch("aifactory.core.run_store.get_run_store", return_value=mock_store):
            runner.invoke(runs_app, ["list", "--last", "5"])
        mock_store.list_recent.assert_called_once_with(n=5)

    def test_default_last_is_20(self) -> None:
        mock_store = _mock_store()
        with patch("aifactory.core.run_store.get_run_store", return_value=mock_store):
            runner.invoke(runs_app, ["list"])
        mock_store.list_recent.assert_called_once_with(n=20)

    def test_high_quality_renders_without_error(self) -> None:
        # CliRunner strips Rich markup, just verify no crash
        records = [_make_record(composite_quality=9.0)]
        with patch("aifactory.core.run_store.get_run_store", return_value=_mock_store(records)):
            result = runner.invoke(runs_app, ["list"])
        assert result.exit_code == 0

    def test_low_quality_renders_without_error(self) -> None:
        records = [_make_record(composite_quality=2.0)]
        with patch("aifactory.core.run_store.get_run_store", return_value=_mock_store(records)):
            result = runner.invoke(runs_app, ["list"])
        assert result.exit_code == 0

    def test_flagged_count_shown(self) -> None:
        records = [_make_record(quality_flagged_count=2)]
        with patch("aifactory.core.run_store.get_run_store", return_value=_mock_store(records)):
            result = runner.invoke(runs_app, ["list"])
        assert result.exit_code == 0
        assert "2" in result.output


# =============================================================================
# show_run
# =============================================================================


class TestShowRun:
    def test_found_run_shows_details(self) -> None:
        rec = _make_record()
        store = _mock_store(record=rec)
        with patch("aifactory.core.run_store.get_run_store", return_value=store):
            result = runner.invoke(runs_app, ["show", "run-20260416-001"])
        assert result.exit_code == 0
        assert "run-20260416-001" in result.output
        assert "balanced" in result.output

    def test_not_found_exits_1(self) -> None:
        store = _mock_store(record=None)
        with patch("aifactory.core.run_store.get_run_store", return_value=store):
            result = runner.invoke(runs_app, ["show", "nonexistent-run"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_shows_model_selections(self) -> None:
        rec = _make_record(model_selections={"master": "gpt-4o", "worker": "gpt-4o-mini"})
        store = _mock_store(record=rec)
        with patch("aifactory.core.run_store.get_run_store", return_value=store):
            result = runner.invoke(runs_app, ["show", rec.run_id])
        assert result.exit_code == 0
        assert "gpt-4o" in result.output

    def test_shows_response_preview(self) -> None:
        rec = _make_record(final_response_preview="Here is the implementation...")
        store = _mock_store(record=rec)
        with patch("aifactory.core.run_store.get_run_store", return_value=store):
            result = runner.invoke(runs_app, ["show", rec.run_id])
        assert result.exit_code == 0
        assert "Here is the implementation" in result.output

    def test_no_model_selections_no_crash(self) -> None:
        rec = _make_record(model_selections={})
        store = _mock_store(record=rec)
        with patch("aifactory.core.run_store.get_run_store", return_value=store):
            result = runner.invoke(runs_app, ["show", rec.run_id])
        assert result.exit_code == 0

    def test_empty_response_preview_no_crash(self) -> None:
        rec = _make_record(final_response_preview="")
        store = _mock_store(record=rec)
        with patch("aifactory.core.run_store.get_run_store", return_value=store):
            result = runner.invoke(runs_app, ["show", rec.run_id])
        assert result.exit_code == 0


# =============================================================================
# search_runs
# =============================================================================


class TestSearchRuns:
    def test_no_results_shows_message(self) -> None:
        store = _mock_store([])
        with patch("aifactory.core.run_store.get_run_store", return_value=store):
            result = runner.invoke(runs_app, ["search", "some query"])
        assert result.exit_code == 0
        assert "No similar runs" in result.output

    def test_results_shown(self) -> None:
        records = [_make_record(query="Build a REST API")]
        store = _mock_store(records)
        with patch("aifactory.core.run_store.get_run_store", return_value=store):
            result = runner.invoke(runs_app, ["search", "REST API"])
        assert result.exit_code == 0
        assert "run-20260416-001" in result.output or "REST API" in result.output

    def test_top_k_passed_to_store(self) -> None:
        store = _mock_store([])
        with patch("aifactory.core.run_store.get_run_store", return_value=store):
            runner.invoke(runs_app, ["search", "query", "--top-k", "3"])
        store.search_similar.assert_called_once_with("query", top_k=3)

    def test_default_top_k_is_5(self) -> None:
        store = _mock_store([])
        with patch("aifactory.core.run_store.get_run_store", return_value=store):
            runner.invoke(runs_app, ["search", "query"])
        store.search_similar.assert_called_once_with("query", top_k=5)

    def test_response_preview_shown(self) -> None:
        records = [_make_record(final_response_preview="Here is the code for your REST API")]
        store = _mock_store(records)
        with patch("aifactory.core.run_store.get_run_store", return_value=store):
            result = runner.invoke(runs_app, ["search", "REST API"])
        assert result.exit_code == 0
        assert "Here is the code" in result.output

    def test_empty_preview_shown_without_crash(self) -> None:
        records = [_make_record(final_response_preview="")]
        store = _mock_store(records)
        with patch("aifactory.core.run_store.get_run_store", return_value=store):
            result = runner.invoke(runs_app, ["search", "anything"])
        assert result.exit_code == 0
