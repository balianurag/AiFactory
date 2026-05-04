"""
Tests for cli/cmd_batch.py — batch pipeline orchestration.

Covers:
  - JSONL parsing: valid, invalid JSON, missing 'query', comments/blank lines
  - batch_run CLI: --dry-run flag, file not found
  - _run_single(): gate bypass short-circuit, failed pipeline error handling
  - Output file written with correct format
  - Summary table rendered without error
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from aifactory.cli.cmd_batch import _run_single, batch_app

runner = CliRunner()


# =============================================================================
# Helpers
# =============================================================================


def _write_jsonl(path: Path, items: list[dict]) -> None:
    with path.open("w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


# =============================================================================
# JSONL parsing (via --dry-run)
# =============================================================================


class TestBatchRunParsing:
    def test_valid_tasks_parsed(self, tmp_path: Path) -> None:
        tasks = tmp_path / "tasks.jsonl"
        _write_jsonl(tasks, [{"query": "Build a REST API"}, {"query": "Write unit tests"}])
        result = runner.invoke(batch_app, [str(tasks), "--dry-run"])
        assert result.exit_code == 0
        assert "Build a REST API" in result.output or "tasks" in result.output.lower()

    def test_file_not_found_exits_1(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.jsonl"
        result = runner.invoke(batch_app, [str(missing), "--dry-run"])
        assert result.exit_code == 1

    def test_missing_query_field_skipped(self, tmp_path: Path) -> None:
        tasks = tmp_path / "tasks.jsonl"
        with tasks.open("w") as f:
            f.write('{"not_query": "hello"}\n')
            f.write('{"query": "valid task"}\n')
        result = runner.invoke(batch_app, [str(tasks), "--dry-run"])
        assert result.exit_code == 0
        # Should mention skipping the invalid line
        assert "skipped" in result.output.lower() or "valid task" in result.output

    def test_invalid_json_line_skipped(self, tmp_path: Path) -> None:
        tasks = tmp_path / "tasks.jsonl"
        with tasks.open("w") as f:
            f.write("{{not valid json}}\n")
            f.write('{"query": "good task"}\n')
        result = runner.invoke(batch_app, [str(tasks), "--dry-run"])
        assert result.exit_code == 0

    def test_blank_and_comment_lines_ignored(self, tmp_path: Path) -> None:
        tasks = tmp_path / "tasks.jsonl"
        with tasks.open("w") as f:
            f.write("\n")
            f.write("# This is a comment\n")
            f.write('{"query": "real task"}\n')
        result = runner.invoke(batch_app, [str(tasks), "--dry-run"])
        assert result.exit_code == 0

    def test_no_valid_tasks_exits_0(self, tmp_path: Path) -> None:
        tasks = tmp_path / "tasks.jsonl"
        with tasks.open("w") as f:
            f.write("# just a comment\n")
        result = runner.invoke(batch_app, [str(tasks), "--dry-run"])
        assert result.exit_code == 0
        assert "No valid tasks" in result.output

    def test_dry_run_shows_task_list(self, tmp_path: Path) -> None:
        tasks = tmp_path / "tasks.jsonl"
        _write_jsonl(tasks, [{"query": "alpha"}, {"query": "beta"}])
        result = runner.invoke(batch_app, [str(tasks), "--dry-run"])
        assert result.exit_code == 0
        assert "alpha" in result.output
        assert "beta" in result.output


# =============================================================================
# _run_single — unit tests
# =============================================================================


class TestRunSingle:
    async def test_bypass_gate_returns_immediately(self) -> None:
        """If ComplexityGate.bypass is True, _run_single should short-circuit."""
        mock_gate = MagicMock()
        mock_gate.bypass = True
        mock_gate.method = "math"
        mock_gate.answer = "42"

        sem = asyncio.Semaphore(1)

        with patch("aifactory.core.complexity_gate.evaluate_gate", return_value=mock_gate):
            result = await _run_single({"query": "2 + 2", "id": "test-id"}, "balanced", sem)

        assert result["status"] == "completed"
        assert result["gate"] == "math"
        assert result["result"] == "42"
        assert result["id"] == "test-id"

    async def test_pipeline_exception_returns_failed(self) -> None:
        mock_gate = MagicMock()
        mock_gate.bypass = False

        sem = asyncio.Semaphore(1)

        with (
            patch("aifactory.core.complexity_gate.evaluate_gate", return_value=mock_gate),
            patch(
                "aifactory.graphs.dynamic_pipeline.build_dynamic_pipeline",
                side_effect=Exception("pipeline exploded"),
            ),
        ):
            result = await _run_single({"query": "complex task"}, "balanced", sem)

        assert result["status"] == "failed"
        assert "pipeline exploded" in result["error"]

    async def test_auto_id_assigned_when_missing(self) -> None:
        mock_gate = MagicMock()
        mock_gate.bypass = True
        mock_gate.method = "facts"
        mock_gate.answer = "Paris"

        sem = asyncio.Semaphore(1)

        with patch("aifactory.core.complexity_gate.evaluate_gate", return_value=mock_gate):
            result = await _run_single({"query": "Capital of France?"}, "balanced", sem)

        assert "id" in result
        assert result["id"]  # non-empty

    async def test_elapsed_seconds_recorded(self) -> None:
        mock_gate = MagicMock()
        mock_gate.bypass = True
        mock_gate.method = "bypass"
        mock_gate.answer = "done"

        sem = asyncio.Semaphore(1)

        with patch("aifactory.core.complexity_gate.evaluate_gate", return_value=mock_gate):
            result = await _run_single({"query": "test"}, "balanced", sem)

        assert "elapsed_s" in result
        assert result["elapsed_s"] >= 0

    async def test_successful_pipeline_returns_completed(self) -> None:
        mock_gate = MagicMock()
        mock_gate.bypass = False

        mock_compiled = AsyncMock()
        mock_compiled.ainvoke = AsyncMock(
            return_value={
                "final_response": "Pipeline result",
                "task_results": [],
                "judge_metrics": {},
            }
        )
        mock_registry = MagicMock()

        sem = asyncio.Semaphore(1)

        with (
            patch("aifactory.core.complexity_gate.evaluate_gate", return_value=mock_gate),
            patch(
                "aifactory.graphs.dynamic_pipeline.build_dynamic_pipeline",
                return_value=(mock_compiled, mock_registry),
            ),
            patch("aifactory.graphs.dynamic_pipeline.write_dump"),
        ):
            result = await _run_single({"query": "multi-step task"}, "balanced", sem)

        assert result["status"] == "completed"
        assert result["result"] == "Pipeline result"
        assert result["gate"] == "full"


# =============================================================================
# Output file
# =============================================================================


class TestBatchOutput:
    def test_output_jsonl_written(self, tmp_path: Path) -> None:
        tasks = tmp_path / "tasks.jsonl"
        out = tmp_path / "results.jsonl"
        _write_jsonl(tasks, [{"query": "2 + 2", "id": "math-task"}])

        mock_gate = MagicMock()
        mock_gate.bypass = True
        mock_gate.method = "math"
        mock_gate.answer = "4"

        with patch("aifactory.core.complexity_gate.evaluate_gate", return_value=mock_gate):
            result = runner.invoke(batch_app, [str(tasks), "--output", str(out)])

        assert result.exit_code == 0
        assert out.exists()
        lines = [json.loads(l) for l in out.read_text().strip().splitlines()]
        assert len(lines) == 1
        assert lines[0]["status"] == "completed"

    def test_default_output_path_derived_from_input(self, tmp_path: Path) -> None:
        tasks = tmp_path / "tasks.jsonl"
        _write_jsonl(tasks, [{"query": "2 + 2"}])

        mock_gate = MagicMock()
        mock_gate.bypass = True
        mock_gate.method = "math"
        mock_gate.answer = "4"

        with patch("aifactory.core.complexity_gate.evaluate_gate", return_value=mock_gate):
            result = runner.invoke(batch_app, [str(tasks)])

        expected = tmp_path / "tasks.results.jsonl"
        assert expected.exists()
