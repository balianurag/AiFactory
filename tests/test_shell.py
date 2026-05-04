"""
Tests for cli/cmd_shell.py — interactive shell completer and helpers.

All tests are pure-Python (no subprocess, no pty).  The prompt_toolkit
Completer API is exercised by creating a Document and calling
get_completions() directly.
"""

from __future__ import annotations

from prompt_toolkit.document import Document
from rich.console import Console

from aifactory.cli.cmd_shell import (
    _COMMANDS,
    AifactoryCompleter,
    _print_help,
)


def _completions(text: str) -> list[str]:
    """Return the list of completion texts for *text*."""
    completer = AifactoryCompleter()
    doc = Document(text, cursor_position=len(text))
    return [c.text for c in completer.get_completions(doc, None)]


# =============================================================================
# Command catalogue sanity checks
# =============================================================================


class TestCommandCatalogue:
    def test_core_commands_present(self) -> None:
        for cmd in ("run", "status", "memory", "agents", "rl", "meta", "runs"):
            assert cmd in _COMMANDS, f"missing command: {cmd}"

    def test_each_command_has_help(self) -> None:
        for cmd, meta in _COMMANDS.items():
            assert "help" in meta, f"{cmd} has no 'help' key"

    def test_subcommand_entries_have_help(self) -> None:
        for cmd, meta in _COMMANDS.items():
            for sub, sub_meta in meta.get("subcommands", {}).items():
                assert "help" in sub_meta, f"{cmd} {sub} has no 'help' key"


# =============================================================================
# Completer — top-level commands
# =============================================================================


class TestCompleterTopLevel:
    def test_empty_input_returns_all_commands(self) -> None:
        results = _completions("")
        assert set(results) == set(_COMMANDS.keys())

    def test_prefix_narrows_to_matching_commands(self) -> None:
        results = _completions("r")
        assert "run" in results
        assert "resume" in results
        assert "rl" in results
        # Non-matching commands should not appear
        assert "status" not in results

    def test_exact_command_completes_to_itself(self) -> None:
        results = _completions("run")
        assert "run" in results

    def test_unknown_prefix_returns_empty(self) -> None:
        results = _completions("zzz")
        assert results == []


# =============================================================================
# Completer — subcommands
# =============================================================================


class TestCompleterSubcommands:
    def test_memory_subcommands(self) -> None:
        results = _completions("memory ")
        assert "stats" in results
        assert "search" in results
        assert "clear" in results
        assert "health" in results

    def test_memory_subcommand_prefix(self) -> None:
        results = _completions("memory s")
        assert "stats" in results
        assert "search" in results
        assert "clear" not in results

    def test_runs_subcommands(self) -> None:
        results = _completions("runs ")
        assert "list" in results
        assert "show" in results
        assert "search" in results

    def test_meta_subcommands(self) -> None:
        results = _completions("meta ")
        assert "review" in results
        assert "calibrate" in results

    def test_rl_subcommands(self) -> None:
        results = _completions("rl ")
        assert "stats" in results
        assert "reset" in results

    def test_unknown_command_returns_empty_for_subcommand(self) -> None:
        results = _completions("foobar ")
        assert results == []


# =============================================================================
# Completer — flags
# =============================================================================


class TestCompleterFlags:
    def test_run_flags(self) -> None:
        results = _completions("run task --")
        assert "--mode" in results
        assert "--verbose" in results
        assert "--no-chat" in results

    def test_run_flag_values(self) -> None:
        # After "--mode " the completer should return the mode values
        results = _completions("run task --mode ")
        assert "best" in results
        assert "balanced" in results
        assert "cost_efficient" in results

    def test_rl_stats_type_values(self) -> None:
        results = _completions("rl stats --type ")
        assert "coding" in results
        assert "math" in results
        assert "research" in results

    def test_memory_search_layer_values(self) -> None:
        results = _completions("memory search query --layer ")
        assert "agent" in results
        assert "team" in results
        assert "global" in results
        assert "all" in results

    def test_flag_prefix_filters(self) -> None:
        results = _completions("run task --mo")
        assert "--mode" in results
        assert "--no-chat" not in results


# =============================================================================
# Help renderer (smoke test — just ensure no exceptions)
# =============================================================================


class TestHelpRenderer:
    def _console(self) -> Console:
        return Console(record=True, highlight=False)

    def test_help_no_command_shows_all(self) -> None:
        con = self._console()
        _print_help(con, None)
        output = con.export_text()
        assert "run" in output
        assert "status" in output
        assert "memory" in output

    def test_help_with_command_shows_detail(self) -> None:
        con = self._console()
        _print_help(con, "memory")
        output = con.export_text()
        assert "stats" in output
        assert "search" in output

    def test_help_with_flag_command(self) -> None:
        con = self._console()
        _print_help(con, "run")
        output = con.export_text()
        assert "--mode" in output

    def test_help_unknown_command_shows_full_list(self) -> None:
        """Unknown commands fall back to the full command list."""
        con = self._console()
        _print_help(con, "nonexistent_cmd")
        output = con.export_text()
        # Full list should still be shown
        assert "run" in output
