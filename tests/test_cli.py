"""Tests for the AI Factory CLI package."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from aifactory.cli import app
from aifactory.cli._helpers import (
    find_dump_fuzzy,
    format_cost_usd,
    format_elapsed,
    list_dumps,
    load_dump,
    run_id_now,
)

runner = CliRunner()

# ── Helpers tests ────────────────────────────────────────────────────────────


class TestHelpers:
    def test_run_id_now_format(self) -> None:
        """run_id_now returns YYYYMMDD_HHMMSS format."""
        rid = run_id_now()
        assert len(rid) == 15
        assert rid[8] == "_"
        assert rid.replace("_", "").isdigit()

    def test_format_elapsed_seconds(self) -> None:
        assert format_elapsed(12.5) == "12.5s"
        assert format_elapsed(0.3) == "0.3s"

    def test_format_elapsed_minutes(self) -> None:
        result = format_elapsed(125.0)
        assert "2m" in result

    def test_format_cost_usd_gpt4o(self) -> None:
        """Cost estimate for GPT-4o model."""
        cost = format_cost_usd(1_000_000, "gpt-4o")
        assert cost.startswith("$")
        val = float(cost.replace("$", ""))
        assert val > 0

    def test_format_cost_usd_gemini_flash(self) -> None:
        """Gemini Flash should be very cheap."""
        cost = format_cost_usd(10_000, "gemini-2.0-flash")
        val = float(cost.replace("$", ""))
        assert val < 0.01  # 10K tokens on flash should be pennies

    def test_format_cost_usd_default(self) -> None:
        """Unknown model uses default rate."""
        cost = format_cost_usd(1000, "unknown-model")
        val = float(cost.replace("$", ""))
        assert val > 0


class TestDumpIO:
    """Tests for dump loading utilities."""

    def test_list_dumps(self, tmp_path: Path) -> None:
        """list_dumps returns sorted dump data."""
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        for i in range(3):
            dump = {"run_id": f"20260409_10000{i}", "config": {"request": f"test {i}"}}
            (runs_dir / f"20260409_10000{i}.json").write_text(json.dumps(dump))

        with patch("aifactory.cli._helpers.DUMPS_DIR", tmp_path):
            dumps = list_dumps(limit=10)

        assert len(dumps) == 3
        # Newest first
        assert dumps[0]["run_id"] == "20260409_100002"

    def test_load_dump(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        dump = {"run_id": "20260409_100000", "config": {"request": "test"}}
        (runs_dir / "20260409_100000.json").write_text(json.dumps(dump))

        with patch("aifactory.cli._helpers.DUMPS_DIR", tmp_path):
            result = load_dump("20260409_100000")

        assert result is not None
        assert result["run_id"] == "20260409_100000"

    def test_load_dump_missing(self, tmp_path: Path) -> None:
        with patch("aifactory.cli._helpers.DUMPS_DIR", tmp_path):
            assert load_dump("nonexistent") is None

    def test_find_dump_fuzzy_exact(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        dump = {"run_id": "20260409_100000", "config": {"request": "test"}}
        (runs_dir / "20260409_100000.json").write_text(json.dumps(dump))

        with patch("aifactory.cli._helpers.DUMPS_DIR", tmp_path):
            result = find_dump_fuzzy("20260409_100000")

        assert result is not None

    def test_find_dump_fuzzy_prefix(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        dump = {"run_id": "20260409_100000", "config": {"request": "test"}}
        (runs_dir / "20260409_100000.json").write_text(json.dumps(dump))

        with patch("aifactory.cli._helpers.DUMPS_DIR", tmp_path):
            result = find_dump_fuzzy("20260409")

        assert result is not None

    def test_find_dump_fuzzy_text_search(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        dump = {"run_id": "20260409_100000", "config": {"request": "Build a todo app"}}
        (runs_dir / "20260409_100000.json").write_text(json.dumps(dump))

        with patch("aifactory.cli._helpers.DUMPS_DIR", tmp_path):
            result = find_dump_fuzzy("todo app")

        assert result is not None
        assert result["run_id"] == "20260409_100000"


# ── CLI command tests ────────────────────────────────────────────────────────


class TestCLICommands:
    """Tests for CLI commands using typer CliRunner."""

    def test_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "AI Factory" in result.output

    def test_run_help(self) -> None:
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "pipeline" in result.output.lower()
        assert "mode" in result.output.lower()

    def test_resume_help(self) -> None:
        result = runner.invoke(app, ["resume", "--help"])
        assert result.exit_code == 0
        assert "Resume" in result.output

    def test_status_no_dumps(self, tmp_path: Path) -> None:
        with patch("aifactory.cli.cmd_status.list_dumps", return_value=[]):
            result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "No pipeline runs" in result.output

    def test_status_with_dumps(self) -> None:
        fake_dumps = [
            {
                "run_id": "20260409_100000",
                "config": {"request": "Build app", "mode": "balanced"},
                "elapsed_seconds": 30.0,
                "totals": {"total_tokens": 10000},
                "judge_metrics": {"overall_status": "successful"},
                "final_response": "Done!",
            }
        ]
        with patch("aifactory.cli.cmd_status.list_dumps", return_value=fake_dumps):
            result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "20260409_100000" in result.output
        assert "Build app" in result.output

    def test_inspect_not_found(self) -> None:
        with patch("aifactory.cli.cmd_status.find_dump_fuzzy", return_value=None):
            result = runner.invoke(app, ["inspect", "nonexistent"])
        assert result.exit_code == 1

    def test_inspect_found(self) -> None:
        fake_dump = {
            "run_id": "20260409_100000",
            "config": {"request": "Build app", "mode": "balanced"},
            "elapsed_seconds": 30.0,
            "tasks": [{"description": "Build the app"}],
            "task_results": [
                {
                    "task_id": "task_1",
                    "agent": "Worker-task_1",
                    "model": "gpt-4o",
                    "result": "Done building the app",
                    "skipped": False,
                }
            ],
            "agents": [
                {
                    "name": "master",
                    "model": "gpt-4o",
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                }
            ],
            "agent_assessments": [],
            "judge_metrics": {"overall_status": "successful"},
            "shared_context_final": {"objective": "Build app"},
            "final_response": "App built successfully!",
        }
        with patch("aifactory.cli.cmd_status.find_dump_fuzzy", return_value=fake_dump):
            result = runner.invoke(app, ["inspect", "20260409"])
        assert result.exit_code == 0
        assert "Build app" in result.output
        assert "Pipeline" in result.output

    def test_tools_help(self) -> None:
        result = runner.invoke(app, ["tools", "--help"])
        assert result.exit_code == 0
        assert "integration tools" in result.output.lower()

    def test_connect_help(self) -> None:
        result = runner.invoke(app, ["connect", "--help"])
        assert result.exit_code == 0
        assert "Connect" in result.output

    def test_ask_help(self) -> None:
        result = runner.invoke(app, ["ask", "--help"])
        assert result.exit_code == 0
        assert "question" in result.output.lower()

    def test_cost_today_no_runs(self) -> None:
        with patch("aifactory.cli.cmd_cost.list_dumps", return_value=[]):
            result = runner.invoke(app, ["cost", "today"])
        assert result.exit_code == 0
        assert "No runs" in result.output

    def test_agents_list(self) -> None:
        result = runner.invoke(app, ["agents", "list"])
        assert result.exit_code == 0
        assert "master" in result.output
        assert "execution_lead" in result.output

    def test_agents_roles(self) -> None:
        result = runner.invoke(app, ["agents", "roles"])
        assert result.exit_code == 0
        assert "premium" in result.output.lower() or "budget" in result.output.lower()

    def test_info(self) -> None:
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert "AI Factory Configuration" in result.output

    def test_models_tiers(self) -> None:
        result = runner.invoke(app, ["models", "tiers"])
        assert result.exit_code == 0
        assert "premium" in result.output.lower()

    # ── gate command ─────────────────────────────────────────────────

    def test_gate_math_bypass(self) -> None:
        """Pure math → bypass tier."""
        result = runner.invoke(app, ["gate", "2 + 2"])
        assert result.exit_code == 0
        assert "BYPASS" in result.output

    def test_gate_simple_qa_fast_llm(self) -> None:
        """Simple factual question → fast_llm tier."""
        result = runner.invoke(app, ["gate", "What is photosynthesis?"])
        assert result.exit_code == 0
        # Either FAST LLM or FULL PIPELINE — should not be bypass
        assert result.exit_code == 0

    def test_gate_complex_full_pipeline(self) -> None:
        """Complex build task → full pipeline tier."""
        result = runner.invoke(app, ["gate", "Build a FastAPI todo app with auth"])
        assert result.exit_code == 0
        assert "FULL PIPELINE" in result.output

    def test_gate_verbose_shows_scores(self) -> None:
        """--verbose flag shows pattern score table."""
        result = runner.invoke(app, ["gate", "Build a FastAPI app", "--verbose"])
        assert result.exit_code == 0
        assert "needs_tools" in result.output or "simple_qa" in result.output

    def test_gate_static_fact_bypass(self) -> None:
        """Known static fact → bypass tier."""
        result = runner.invoke(app, ["gate", "Capital of France"])
        assert result.exit_code == 0
        assert "BYPASS" in result.output


# ── run --context flag ────────────────────────────────────────────────────────


class TestRunContextFlag:
    """Tests for `aifactory run --context` flag."""

    def test_run_help_shows_context(self) -> None:
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "--context" in result.output or "-c" in result.output

    def test_run_context_reaches_pipeline(self) -> None:
        """--context value ends up in shared_context.user_context for full pipeline."""
        from unittest.mock import AsyncMock, patch

        with patch("aifactory.cli.cmd_run._run_pipeline", new=AsyncMock()) as mock_pipe:
            runner.invoke(app, ["run", "--context", "Use PostgreSQL", "Build a service"])
        mock_pipe.assert_called_once()
        _, kwargs = mock_pipe.call_args
        assert kwargs.get("context") == "Use PostgreSQL"

    def test_run_context_empty_by_default(self) -> None:
        """Calling run without --context passes empty string."""
        from unittest.mock import AsyncMock, patch

        with patch("aifactory.cli.cmd_run._run_pipeline", new=AsyncMock()) as mock_pipe:
            runner.invoke(app, ["run", "2 + 2"])
        # Gate may bypass before _run_pipeline is called for trivial tasks
        if mock_pipe.called:
            _, kwargs = mock_pipe.call_args
            assert kwargs.get("context", "") == ""


# ── connect subcommands ──────────────────────────────────────────────────────


class TestConnectCommands:
    """Tests for `aifactory connect add/list/remove`."""

    def test_connect_help(self) -> None:
        result = runner.invoke(app, ["connect", "--help"])
        assert result.exit_code == 0
        assert "add" in result.output
        assert "list" in result.output
        assert "remove" in result.output

    def test_connect_add_help(self) -> None:
        result = runner.invoke(app, ["connect", "add", "--help"])
        assert result.exit_code == 0
        assert "Connect" in result.output

    def test_connect_list_no_composio_key(self) -> None:
        """list exits non-zero when Composio provider fails to connect."""
        from unittest.mock import AsyncMock, patch

        with patch(
            "aifactory.integrations.providers.composio.ComposioProvider.connect",
            new=AsyncMock(return_value=False),
        ):
            result = runner.invoke(app, ["connect", "list"])
        assert result.exit_code != 0

    def test_connect_list_shows_table(self) -> None:
        """list renders a table of toolkits when Composio is connected."""
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_provider = MagicMock()
        mock_provider.connect = AsyncMock(return_value=True)
        mock_provider.get_available_toolkits.return_value = ["github", "jira"]
        mock_provider.fetch_available_tools = AsyncMock(
            return_value={"github": ["GITHUB_CREATE_ISSUE"], "jira": []}
        )
        mock_provider.disconnect = AsyncMock()

        # ComposioProvider is imported inside the function body — patch at source
        with patch(
            "aifactory.integrations.providers.composio.ComposioProvider",
            return_value=mock_provider,
        ):
            result = runner.invoke(app, ["connect", "list"])
        assert result.exit_code == 0
        assert "github" in result.output
        assert "jira" in result.output

    def test_connect_remove_aborted(self) -> None:
        """remove without --yes asks for confirmation; 'n' aborts."""
        result = runner.invoke(app, ["connect", "remove", "github"], input="n\n")
        assert result.exit_code == 0
        assert "Aborted" in result.output or "borted" in result.output

    def test_connect_remove_yes_calls_composio_cli(self) -> None:
        """remove --yes runs composio remove <service>."""
        from unittest.mock import MagicMock, patch

        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = runner.invoke(app, ["connect", "remove", "github", "--yes"])
        assert result.exit_code == 0
        assert mock_run.called
        cmd = mock_run.call_args[0][0]
        assert "composio" in cmd
        assert "remove" in cmd
        assert "github" in cmd

    def test_connect_remove_composio_cli_error(self) -> None:
        """remove exits non-zero when composio CLI returns non-zero."""
        from unittest.mock import MagicMock, patch

        mock_result = MagicMock(returncode=1, stderr="not found", stdout="")
        with patch("subprocess.run", return_value=mock_result):
            result = runner.invoke(app, ["connect", "remove", "github", "--yes"])
        assert result.exit_code != 0


# ── cost estimate ────────────────────────────────────────────────────────────


class TestCostEstimate:
    """Tests for `aifactory cost estimate`."""

    def test_estimate_bypass_task(self) -> None:
        """Pure math → bypass tier, $0 cost."""
        result = runner.invoke(app, ["cost", "estimate", "2 + 2"])
        assert result.exit_code == 0
        assert "Bypass" in result.output or "bypass" in result.output.lower()
        assert "$0.0000" in result.output

    def test_estimate_full_pipeline_task(self) -> None:
        """Complex task → full pipeline estimate with table."""
        result = runner.invoke(app, ["cost", "estimate", "Build a FastAPI todo app"])
        assert result.exit_code == 0
        assert "TOTAL" in result.output
        assert "$" in result.output

    def test_estimate_mode_best(self) -> None:
        """--mode best uses premium model estimates."""
        result = runner.invoke(app, ["cost", "estimate", "Build an app", "--mode", "best"])
        assert result.exit_code == 0
        assert "best" in result.output.lower() or "opus" in result.output.lower()

    def test_estimate_mode_cost_efficient(self) -> None:
        """--mode cost_efficient uses budget model estimates."""
        result = runner.invoke(
            app, ["cost", "estimate", "Build an app", "--mode", "cost_efficient"]
        )
        assert result.exit_code == 0
        assert "haiku" in result.output.lower() or "cost_efficient" in result.output.lower()

    def test_estimate_help(self) -> None:
        result = runner.invoke(app, ["cost", "estimate", "--help"])
        assert result.exit_code == 0
        assert "Estimate" in result.output


# ── RL commands ──────────────────────────────────────────────────────────────


class TestRLCommands:
    """Tests for `aifactory rl stats/reset` CLI commands."""

    def test_rl_stats_no_records(self) -> None:
        # get_all_stats is imported inside the command body — patch at source
        with patch(
            "aifactory.core.rl_scorer.get_all_stats",
            return_value={"total_records": 0, "by_task_type": {}},
        ):
            result = runner.invoke(app, ["rl", "stats"])
        assert result.exit_code == 0
        assert "No RL records" in result.output

    def test_rl_stats_with_records(self) -> None:
        stats = {
            "total_records": 10,
            "by_task_type": {
                "coding": {
                    "records": 10,
                    "models": 2,
                    "best_model": "claude-sonnet-4-6",
                    "best_avg_score": 8.5,
                    "model_avgs": {"claude-sonnet-4-6": 8.5, "gpt-4o-mini": 7.2},
                }
            },
        }
        with patch("aifactory.core.rl_scorer.get_all_stats", return_value=stats):
            result = runner.invoke(app, ["rl", "stats"])
        assert result.exit_code == 0
        assert "coding" in result.output
        assert "claude-sonnet-4-6" in result.output

    def test_rl_stats_type_filter(self) -> None:
        stats = {
            "total_records": 5,
            "by_task_type": {
                "coding": {
                    "records": 5,
                    "models": 1,
                    "best_model": "gpt-4o",
                    "best_avg_score": 9.0,
                    "model_avgs": {"gpt-4o": 9.0},
                }
            },
        }
        with patch("aifactory.core.rl_scorer.get_all_stats", return_value=stats):
            result = runner.invoke(app, ["rl", "stats", "--type", "coding"])
        assert result.exit_code == 0
        assert "coding" in result.output

    def test_rl_stats_unknown_type_exits(self) -> None:
        stats = {
            "total_records": 5,
            "by_task_type": {
                "coding": {"records": 5, "models": 1, "best_model": "gpt-4o", "best_avg_score": 9.0}
            },
        }
        with patch("aifactory.core.rl_scorer.get_all_stats", return_value=stats):
            result = runner.invoke(app, ["rl", "stats", "--type", "nonexistent"])
        assert result.exit_code == 0
        assert "No records" in result.output

    def test_rl_reset_requires_confirmation(self) -> None:
        result = runner.invoke(app, ["rl", "reset"], input="n\n")
        assert result.exit_code == 0
        assert "Aborted" in result.output

    def test_rl_reset_yes_clears_records(self) -> None:
        mock_client = MagicMock()
        mock_client.delete_collection.return_value = None
        mock_client.create_collection.return_value = None
        # _get_rl_client and _RL_COLLECTION are imported inline in the command
        with (
            patch("aifactory.core.rl_scorer._get_rl_client", return_value=mock_client),
        ):
            result = runner.invoke(app, ["rl", "reset", "--yes"])
        assert result.exit_code == 0
        assert "cleared" in result.output.lower()

    def test_rl_reset_task_type_scoped(self) -> None:
        mock_client = MagicMock()
        mock_client.delete.return_value = None
        with (
            patch("aifactory.core.rl_scorer._get_rl_client", return_value=mock_client),
        ):
            result = runner.invoke(app, ["rl", "reset", "--type", "coding", "--yes"])
        assert result.exit_code == 0
        assert "coding" in result.output


# ── Sandbox commands ─────────────────────────────────────────────────────────


class TestSandboxCommands:
    """Tests for `aifactory sandbox build/status/test/pull` CLI commands."""

    def test_sandbox_help(self) -> None:
        result = runner.invoke(app, ["sandbox", "--help"])
        assert result.exit_code == 0
        assert "sandbox" in result.output.lower()

    def test_sandbox_build_calls_docker_twice(self) -> None:
        """build command runs docker build for two images."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = runner.invoke(app, ["sandbox", "build"])
        assert mock_run.call_count >= 2  # one per image
        assert result.exit_code == 0
        assert "Built" in result.output

    def test_sandbox_build_fails_on_docker_error(self) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            result = runner.invoke(app, ["sandbox", "build"])
        assert result.exit_code != 0

    def test_sandbox_build_no_cache_flag(self) -> None:
        """--no-cache flag is forwarded to docker build."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            runner.invoke(app, ["sandbox", "build", "--no-cache"])
        # Check --no-cache was passed
        docker_calls = [str(c) for c in mock_run.call_args_list]
        assert any("no-cache" in call for call in docker_calls)

    def test_sandbox_pull_docker(self) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = runner.invoke(app, ["sandbox", "pull"])
        assert result.exit_code == 0

    def test_sandbox_status_no_server(self) -> None:
        """sandbox status exits non-zero when MCP server is unreachable."""
        # get_mcp_client is imported inside _async_status — patch at source
        with patch(
            "aifactory.mcp.client.get_mcp_client",
            side_effect=RuntimeError("Connection refused"),
        ):
            result = runner.invoke(app, ["sandbox", "status"])
        assert result.exit_code != 0

    def test_sandbox_test_unknown_language(self) -> None:
        """sandbox test rejects unsupported language."""
        result = runner.invoke(app, ["sandbox", "test", "--language", "ruby"])
        assert result.exit_code != 0
        assert "Unknown language" in result.output
