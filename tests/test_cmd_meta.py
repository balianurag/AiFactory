"""
Tests for cli/cmd_meta.py — review and calibrate commands.

All imports in cmd_meta.py are inside nested async functions, so patches must
target the source modules, not the cli module itself.

Covers:
  - review: no run history → early return message
  - review: with records → calls LLM agent and prints report
  - review: --output writes report to file
  - calibrate: no RL data → dry run with "no patches" message
  - calibrate: with patches → shows calibration report
  - calibrate: --apply writes patches and shows confirmation
  - calibrate: --no-backup skips .bak creation
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from aifactory.cli.cmd_meta import app
from aifactory.core.run_store import RunRecord


runner = CliRunner()


# =============================================================================
# Helpers
# =============================================================================


def _make_record(**overrides) -> RunRecord:
    defaults = dict(
        run_id="run-meta-001",
        timestamp=datetime.now(UTC).isoformat(),
        query="Build a REST API",
        mode="balanced",
        elapsed_seconds=10.0,
        gate="",
        model_selections={"master": "gpt-4o"},
        task_count=2,
        quality_flagged_count=0,
        composite_quality=8.0,
        task_types=["coding"],
        final_response_preview="Here is the API...",
    )
    defaults.update(overrides)
    return RunRecord(**defaults)


def _empty_store():
    store = MagicMock()
    store.list_recent.return_value = []
    return store


def _store_with_records(records: list[RunRecord]):
    store = MagicMock()
    store.list_recent.return_value = records
    return store


# =============================================================================
# review command
# =============================================================================


class TestReview:
    def test_no_run_history_message(self) -> None:
        with patch("aifactory.core.run_store.get_run_store", return_value=_empty_store()):
            result = runner.invoke(app, ["review"])
        assert result.exit_code == 0
        assert "No run history" in result.output

    def test_with_records_calls_agent(self) -> None:
        records = [_make_record()]

        mock_agent = MagicMock()
        mock_agent.think = AsyncMock(return_value="Here are the recommendations...")

        with (
            patch("aifactory.core.run_store.get_run_store", return_value=_store_with_records(records)),
            patch("aifactory.core.rl_scorer.get_model_rl_scores", new_callable=AsyncMock, return_value={}),
            patch("aifactory.core.model_selector.load_agent_cfg", return_value={"system_prompt": "analyze", "max_tokens": 1000}),
            patch("aifactory.graphs.dynamic_pipeline.PipelineAgent", return_value=mock_agent),
        ):
            result = runner.invoke(app, ["review"])

        assert result.exit_code == 0
        assert mock_agent.think.called
        assert "Here are the recommendations" in result.output

    def test_output_written_to_file(self, tmp_path: Path) -> None:
        records = [_make_record()]
        out = tmp_path / "report.md"

        mock_agent = MagicMock()
        mock_agent.think = AsyncMock(return_value="Agent recommendations here")

        with (
            patch("aifactory.core.run_store.get_run_store", return_value=_store_with_records(records)),
            patch("aifactory.core.rl_scorer.get_model_rl_scores", new_callable=AsyncMock, return_value={}),
            patch("aifactory.core.model_selector.load_agent_cfg", return_value={"system_prompt": "analyze"}),
            patch("aifactory.graphs.dynamic_pipeline.PipelineAgent", return_value=mock_agent),
        ):
            result = runner.invoke(app, ["review", "--output", str(out)])

        assert result.exit_code == 0
        assert out.exists()
        assert "Agent recommendations here" in out.read_text()

    def test_last_flag_controls_records_fetched(self) -> None:
        mock_store = _empty_store()

        with patch("aifactory.core.run_store.get_run_store", return_value=mock_store):
            runner.invoke(app, ["review", "--last", "5"])

        mock_store.list_recent.assert_called_once_with(n=5)

    def test_rl_scores_fetched_for_multiple_task_types(self) -> None:
        records = [_make_record()]

        mock_agent = MagicMock()
        mock_agent.think = AsyncMock(return_value="OK")
        rl_call_tracker: list[str] = []

        def _mock_rl(task_type: str, min_samples: int = 1) -> dict:
            rl_call_tracker.append(task_type)
            return {}

        with (
            patch("aifactory.core.run_store.get_run_store", return_value=_store_with_records(records)),
            patch("aifactory.core.rl_scorer.get_model_rl_scores", side_effect=_mock_rl),
            patch("aifactory.core.model_selector.load_agent_cfg", return_value={"system_prompt": "analyze"}),
            patch("aifactory.graphs.dynamic_pipeline.PipelineAgent", return_value=mock_agent),
        ):
            runner.invoke(app, ["review"])

        # Should call RL for all standard task types
        expected_types = {"coding", "math", "research", "writing", "analysis", "general"}
        assert expected_types.issubset(set(rl_call_tracker))

    def test_meta_config_fallback_on_missing_agent_config(self) -> None:
        records = [_make_record()]

        mock_agent = MagicMock()
        mock_agent.think = AsyncMock(return_value="fallback response")

        with (
            patch("aifactory.core.run_store.get_run_store", return_value=_store_with_records(records)),
            patch("aifactory.core.rl_scorer.get_model_rl_scores", new_callable=AsyncMock, return_value={}),
            patch("aifactory.core.model_selector.load_agent_cfg", side_effect=Exception("no meta yaml")),
            patch("aifactory.graphs.dynamic_pipeline.PipelineAgent", return_value=mock_agent),
        ):
            result = runner.invoke(app, ["review"])

        assert result.exit_code == 0
        # Agent was still created despite config load failure
        assert mock_agent.think.called


# =============================================================================
# calibrate command
# =============================================================================


class TestCalibrate:
    def test_no_patches_shows_no_patches_message(self) -> None:
        empty_report = {
            "summary": "RL Calibration Report\n========\nNo patches needed.",
            "patches": {},
            "models_eligible": 0,
        }
        with patch(
            "aifactory.core.model_calibrator.get_calibration_report",
            new_callable=AsyncMock,
            return_value=empty_report,
        ):
            result = runner.invoke(app, ["calibrate"])

        assert result.exit_code == 0
        assert "No patches" in result.output or "Dry run" in result.output

    def test_dry_run_shows_report_without_writing(self) -> None:
        report = {
            "summary": "RL Calibration Report\n====\ngpt-4o: coding 1.0 → 3.0 ↑",
            "patches": {"gpt-4o": {"coding": 3.0}},
            "models_eligible": 1,
        }
        with (
            patch(
                "aifactory.core.model_calibrator.get_calibration_report",
                new_callable=AsyncMock,
                return_value=report,
            ),
            patch("aifactory.core.model_calibrator.apply_patches") as mock_apply,
        ):
            result = runner.invoke(app, ["calibrate"])

        assert result.exit_code == 0
        mock_apply.assert_not_called()
        assert "Dry run" in result.output or "apply" in result.output.lower()

    def test_apply_flag_writes_patches(self) -> None:
        report = {
            "summary": "RL Calibration Report\n====\ngpt-4o patched",
            "patches": {"gpt-4o": {"coding": 2.8}},
            "models_eligible": 1,
        }
        apply_result = {
            "applied": 1,
            "skipped": 0,
            "path": "/fake/model_registry.yaml",
            "dry_run": False,
        }
        with (
            patch(
                "aifactory.core.model_calibrator.get_calibration_report",
                new_callable=AsyncMock,
                return_value=report,
            ),
            patch(
                "aifactory.core.model_calibrator.apply_patches",
                return_value=apply_result,
            ) as mock_apply,
        ):
            result = runner.invoke(app, ["calibrate", "--apply"])

        assert result.exit_code == 0
        mock_apply.assert_called_once_with(
            report["patches"],
            dry_run=False,
            backup=True,
        )
        assert "Applied 1 model patches" in result.output

    def test_no_backup_flag(self) -> None:
        report = {
            "summary": "report",
            "patches": {"gpt-4o": {"coding": 2.8}},
            "models_eligible": 1,
        }
        apply_result = {"applied": 1, "skipped": 0, "path": "/fake/reg.yaml", "dry_run": False}

        with (
            patch(
                "aifactory.core.model_calibrator.get_calibration_report",
                new_callable=AsyncMock,
                return_value=report,
            ),
            patch(
                "aifactory.core.model_calibrator.apply_patches",
                return_value=apply_result,
            ) as mock_apply,
        ):
            runner.invoke(app, ["calibrate", "--apply", "--no-backup"])

        mock_apply.assert_called_once_with(
            report["patches"],
            dry_run=False,
            backup=False,
        )

    def test_min_samples_passed_to_report(self) -> None:
        with patch(
            "aifactory.core.model_calibrator.get_calibration_report",
            new_callable=AsyncMock,
            return_value={"summary": "ok", "patches": {}, "models_eligible": 0},
        ) as mock_report:
            runner.invoke(app, ["calibrate", "--min-samples", "10"])

        mock_report.assert_called_once_with(min_samples=10)

    def test_backup_path_shown_in_output(self) -> None:
        report = {
            "summary": "patched",
            "patches": {"gpt-4o": {"coding": 2.8}},
            "models_eligible": 1,
        }
        apply_result = {
            "applied": 1,
            "skipped": 0,
            "path": "/fake/model_registry.yaml",
            "dry_run": False,
        }
        with (
            patch(
                "aifactory.core.model_calibrator.get_calibration_report",
                new_callable=AsyncMock,
                return_value=report,
            ),
            patch(
                "aifactory.core.model_calibrator.apply_patches",
                return_value=apply_result,
            ),
        ):
            result = runner.invoke(app, ["calibrate", "--apply"])

        assert result.exit_code == 0
        assert ".bak" in result.output or "Backup" in result.output
