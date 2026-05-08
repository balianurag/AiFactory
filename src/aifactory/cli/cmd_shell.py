"""
aifactory shell — Claude Code-style interactive REPL.

Launches a persistent terminal-within-terminal session with:
  • Auto-complete dropdown for all commands + subcommands + flags
  • Command history (up/down arrows, Ctrl-R reverse search)
  • Syntax highlighting (commands in cyan, flags in yellow, strings in green)
  • Inline help hints beneath the prompt
  • Multi-line input (Esc+Enter)
  • Vi or Emacs keybindings (default: Emacs)
  • Persistent conversation context across turns (history threaded automatically)

Type a command normally — no slash prefix needed:
  run "Build a FastAPI app"
  status
  memory stats
  agents list
  help
  exit
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
import uuid
from collections.abc import Iterable
from pathlib import Path

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggest, AutoSuggestFromHistory, Suggestion
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ── Paste cleaning ───────────────────────────────────────────────────────────

# Strips terminal box-drawing / ANSI cruft from text pasted out of the CLI
# output (e.g. │ ## 6. **Type Safety** │ ...) so it becomes plain markdown.
_BOX_CHARS = re.compile(r"[│╭╰╮╯─┤├┬┴┼║═╔╗╚╝╠╣╦╩╪]")
_ANSI_ESC = re.compile(r"\x1b\[[0-9;]*[mK]")
_LEADING_PIPE = re.compile(r"^\s*│\s?", re.MULTILINE)
_TRAILING_PIPE = re.compile(r"\s*│\s*$", re.MULTILINE)


def _clean_paste(text: str) -> str:
    """Strip Rich/terminal box-drawing chars from pasted CLI output.

    Turns  │ ## 6. **Type Safety**  │
    into   ## 6. **Type Safety**
    """
    text = _ANSI_ESC.sub("", text)
    text = _LEADING_PIPE.sub("", text)
    text = _TRAILING_PIPE.sub("", text)
    text = _BOX_CHARS.sub("", text)
    # Collapse blank lines created by stripping box borders
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Smart auto-suggest ───────────────────────────────────────────────────────


class _SmartAutoSuggest(AutoSuggest):
    """Ghost-text suggestions: slash commands when input starts with '/', else history."""

    def __init__(self) -> None:
        self._history_suggest = AutoSuggestFromHistory()

    def get_suggestion(self, buffer, document) -> Suggestion | None:
        text = document.text_before_cursor
        if text.startswith("/"):
            # Find the first slash command that starts with the current input
            for cmd in _SLASH_COMMANDS:
                if cmd.startswith(text) and cmd != text:
                    return Suggestion(cmd[len(text) :])
            return None
        return self._history_suggest.get_suggestion(buffer, document)


# ── Command catalogue ─────────────────────────────────────────────────────────
# Keeps the completer in sync with the actual CLI without importing the whole app.

_COMMANDS: dict[str, dict] = {
    "run": {
        "help": "Run the AI Factory pipeline",
        "args": ["<task>"],
        "flags": {
            "--mode": ["best", "balanced", "cost_efficient"],
            "-m": ["best", "balanced", "cost_efficient"],
            "--task-file": [],
            "-f": [],
            "--context": [],
            "-c": [],
            "--attachment": [],
            "-a": [],
            "--verbose": [],
            "-v": [],
            "--no-chat": [],
        },
    },
    "resume": {
        "help": "Resume a paused pipeline run",
        "args": ["<run-id>"],
        "flags": {},
    },
    "ask": {
        "help": "Ask a question about an existing run",
        "args": ["<question>"],
        "flags": {
            "--run-id": [],
        },
    },
    "status": {
        "help": "Show active and recent runs",
        "args": [],
        "flags": {
            "--limit": [],
            "-n": [],
        },
    },
    "inspect": {
        "help": "Deep-dive into a specific run",
        "args": ["<run-id>"],
        "flags": {},
    },
    "gate": {
        "help": "Test a query against the complexity gate",
        "args": ["<query>"],
        "flags": {
            "--verbose": [],
            "-v": [],
        },
    },
    "info": {
        "help": "Show config, API key status, and system info",
        "args": [],
        "flags": {
            "--json": [],
        },
    },
    "agents": {
        "help": "Agent management commands",
        "subcommands": {
            "list": {"help": "List all configured agents"},
        },
    },
    "models": {
        "help": "Model registry commands",
        "subcommands": {
            "list": {
                "help": "List all models",
                "flags": {
                    "--role": [],
                    "--mode": ["best", "balanced", "cost_efficient"],
                },
            },
            "tiers": {"help": "Show premium/standard/budget tiers"},
        },
    },
    "memory": {
        "help": "Memory layer commands",
        "subcommands": {
            "stats": {"help": "Show memory layer counts"},
            "search": {
                "help": "Semantic search across memory",
                "args": ["<query>"],
                "flags": {
                    "--layer": ["agent", "team", "global", "all"],
                    "--agent": [],
                },
            },
            "clear": {
                "help": "Clear a memory layer",
                "flags": {
                    "--layer": ["agent", "team", "global"],
                    "--agent": [],
                    "--yes": [],
                },
            },
            "health": {"help": "Check memory backend health"},
        },
    },
    "rl": {
        "help": "RL-lite adaptive model scoring",
        "subcommands": {
            "stats": {
                "help": "Show model performance",
                "flags": {
                    "--type": ["coding", "math", "research", "writing", "analysis", "general"]
                },
            },
            "reset": {
                "help": "Clear RL records",
                "flags": {
                    "--yes": [],
                    "--type": ["coding", "math", "research", "writing", "analysis", "general"],
                },
            },
        },
    },
    "runs": {
        "help": "Browse past pipeline runs",
        "subcommands": {
            "list": {
                "help": "List recent runs",
                "flags": {"--limit": [], "-n": []},
            },
            "show": {
                "help": "Show a specific run",
                "args": ["<run-id>"],
            },
            "search": {
                "help": "Semantic search past runs",
                "args": ["<query>"],
                "flags": {"--limit": [], "-n": []},
            },
        },
    },
    "meta": {
        "help": "Self-improvement + model calibration",
        "subcommands": {
            "review": {
                "help": "LLM review of last N runs",
                "flags": {"--last": [], "--output": []},
            },
            "calibrate": {
                "help": "Apply RL-derived model score patches",
                "flags": {
                    "--apply": [],
                    "--no-backup": [],
                    "--min-samples": [],
                },
            },
        },
    },
    "cost": {
        "help": "Cost tracking and reporting",
        "subcommands": {
            "today": {"help": "Cost for today's runs"},
            "week": {"help": "Cost for this week"},
            "all": {"help": "All-time cost summary"},
            "run": {"help": "Cost for a specific run", "args": ["<run-id>"]},
            "estimate": {
                "help": "Estimate cost for a task",
                "args": ["<task>"],
                "flags": {"--mode": ["best", "balanced", "cost_efficient"]},
            },
        },
    },
    "connect": {
        "help": "Integration + tool connection management",
        "subcommands": {
            "add": {"help": "Add an integration", "args": ["<name>"]},
            "list": {"help": "List connected integrations"},
            "remove": {"help": "Remove an integration", "args": ["<name>"]},
        },
    },
    "tools": {
        "help": "List available tools and integrations",
        "args": [],
        "flags": {},
    },
    "sandbox": {
        "help": "Docker MCP sandbox management",
        "subcommands": {
            "build": {"help": "Build sandbox Docker images", "flags": {"--no-cache": []}},
            "status": {"help": "Show pool health metrics"},
            "test": {
                "help": "Smoke test execution",
                "flags": {"--language": ["python", "bash", "javascript"]},
            },
            "pull": {"help": "Pull base images"},
        },
    },
    "serve": {
        "help": "Start persistent API server",
        "flags": {
            "--host": [],
            "--port": [],
            "--workers": [],
            "--reload": [],
            "--log-level": ["debug", "info", "warning", "error"],
            "--no-warmup": [],
        },
    },
    "batch": {
        "help": "Batch pipeline orchestration",
        "subcommands": {
            "run": {
                "help": "Run tasks from JSONL file",
                "args": ["<file>"],
                "flags": {
                    "--concurrency": [],
                    "--mode": ["best", "balanced", "cost_efficient"],
                    "--output": [],
                },
            },
            "status": {"help": "Check batch job status", "args": ["<batch-id>"]},
        },
    },
    "help": {
        "help": "Show help for a command",
        "args": ["[command]"],
        "flags": {},
    },
    "exit": {"help": "Exit the shell", "args": [], "flags": {}},
    "quit": {"help": "Exit the shell", "args": [], "flags": {}},
    "clear": {"help": "Clear the screen", "args": [], "flags": {}},
}

# ── Slash commands (Claude-style) ────────────────────────────────────────────

_MODE_ALIASES: dict[str, str] = {
    # user-facing aliases → internal mode name
    "efficient": "cost_efficient",
    "cost_efficient": "cost_efficient",
    "cost-efficient": "cost_efficient",
    "balanced": "balanced",
    "max": "best",
    "best": "best",
}

_MODE_DISPLAY: dict[str, str] = {
    "cost_efficient": "efficient",
    "balanced": "balanced",
    "best": "max",
}

_SLASH_COMMANDS: dict[str, str] = {
    "/mode": "Switch mode:  mode [efficient|balanced|max]  or just: mode info",
    "/memory": "Memory:  memory stats · memory search <q> · memory health · memory clear",
    "/clear": "Clear conversation context (start fresh)",
    "/compact": "Summarise context  (or: compact info — show session info)",
    "/info": "Show session info: ID, mode, turns, uptime",
    "/verbose": "Toggle verbose mode (show all agent output) — also Ctrl+E",
    "/status": "Show recent runs",
    "/help": "Show this help",
    "/exit": "Exit the shell",
}


# ── Completer ────────────────────────────────────────────────────────────────


class AifactoryCompleter(Completer):
    """Tab-complete commands, subcommands, flags, and flag values."""

    def get_completions(self, document: Document, complete_event) -> Iterable[Completion]:
        text = document.text_before_cursor.lstrip()
        trailing_space = text.endswith(" ")

        # Slash commands — complete before anything else
        if text.startswith("/"):
            parts = text.split()
            if len(parts) == 1 and not trailing_space:
                prefix = parts[0]
                for cmd, help_text in _SLASH_COMMANDS.items():
                    if cmd.startswith(prefix):
                        yield Completion(cmd, start_position=-len(prefix), display_meta=help_text)
            elif len(parts) >= 1 and parts[0] == "/mode":
                if len(parts) == 1 or (len(parts) == 2 and not trailing_space):
                    prefix = parts[1] if len(parts) == 2 else ""
                    for alias in _MODE_ALIASES:
                        if alias.startswith(prefix):
                            yield Completion(alias, start_position=-len(prefix))
            elif len(parts) >= 1 and parts[0] == "/memory":
                _MEM_SUBS = ["stats", "health", "search", "clear"]
                if len(parts) == 1 or (len(parts) == 2 and not trailing_space):
                    prefix = parts[1] if len(parts) == 2 else ""
                    for sub in _MEM_SUBS:
                        if sub.startswith(prefix):
                            yield Completion(sub, start_position=-len(prefix))
            return

        # Empty / single-word: complete top-level commands
        parts = text.split()
        if not parts or (len(parts) == 1 and not trailing_space):
            prefix = parts[0] if parts else ""
            for cmd, meta in _COMMANDS.items():
                if cmd.startswith(prefix):
                    yield Completion(
                        cmd,
                        start_position=-len(prefix),
                        display_meta=meta.get("help", ""),
                    )
            return

        root = parts[0]
        if root not in _COMMANDS:
            return

        cmd_meta = _COMMANDS[root]

        # Has subcommands?
        if "subcommands" in cmd_meta:
            if len(parts) == 1 or (len(parts) == 2 and not trailing_space):
                # Complete subcommand
                prefix = parts[1] if len(parts) == 2 else ""
                for sub, sub_meta in cmd_meta["subcommands"].items():
                    if sub.startswith(prefix):
                        yield Completion(
                            sub,
                            start_position=-len(prefix),
                            display_meta=sub_meta.get("help", ""),
                        )
                return

            # Flags for the subcommand
            sub = parts[1]
            if sub in cmd_meta["subcommands"]:
                sub_meta = cmd_meta["subcommands"][sub]
                yield from self._complete_flags(parts[2:], sub_meta, trailing_space)
            return

        # Top-level flags
        yield from self._complete_flags(parts[1:], cmd_meta, trailing_space)

    def _complete_flags(
        self, parts: list[str], meta: dict, trailing_space: bool = False
    ) -> Iterable[Completion]:
        flags: dict[str, list[str]] = meta.get("flags", {})

        if trailing_space:
            # Cursor is after a space — "last" is empty, second_last is parts[-1]
            last = ""
            second_last = parts[-1] if parts else ""
        else:
            last = parts[-1] if parts else ""
            second_last = parts[-2] if len(parts) >= 2 else ""

        # If the previous token is a flag that takes values, complete the values
        if flags.get(second_last):
            for val in flags[second_last]:
                if val.startswith(last):
                    yield Completion(val, start_position=-len(last))
            return

        # Complete flag names
        if last.startswith("-") or trailing_space:
            for flag, values in flags.items():
                if flag.startswith(last):
                    display_meta = f"values: {', '.join(values)}" if values else ""
                    yield Completion(flag, start_position=-len(last), display_meta=display_meta)


# ── Syntax highlighter ───────────────────────────────────────────────────────


class AifactoryLexer(Lexer):
    """Colour commands cyan/yellow, subcommands light-blue, flags yellow, strings green."""

    # Slash-command keywords get yellow (meta/session commands vs cyan pipeline commands)
    _SLASH_KW = frozenset(cmd.lstrip("/") for cmd in _SLASH_COMMANDS)

    def lex_document(self, document: Document):
        def get_tokens(line_number: int):
            line = document.lines[line_number]
            tokens = []
            words = line.split()
            cursor = 0

            for idx, word in enumerate(words):
                pos = line.find(word, cursor)
                if pos > cursor:
                    tokens.append(("", line[cursor:pos]))
                cursor = pos + len(word)

                bare = word.lstrip("/")
                if idx == 0:
                    if bare in self._SLASH_KW:
                        # Meta/session commands — yellow (mode, clear, compact, info…)
                        tokens.append(("class:slash-cmd", word))
                    elif word in _COMMANDS:
                        tokens.append(("class:command", word))
                    else:
                        tokens.append(("class:arg", word))
                elif idx == 1 and words[0] in _COMMANDS and "subcommands" in _COMMANDS[words[0]]:
                    subs = _COMMANDS[words[0]]["subcommands"]
                    if word in subs:
                        tokens.append(("class:subcommand", word))
                    else:
                        tokens.append(("class:arg", word))
                elif idx == 1 and words[0].lstrip("/") in self._SLASH_KW:
                    # Second word after a slash-cmd keyword: subcommand or value
                    tokens.append(("class:subcommand", word))
                elif word.startswith("--") or (word.startswith("-") and len(word) == 2):
                    tokens.append(("class:flag", word))
                elif word.startswith('"') or word.startswith("'"):
                    tokens.append(("class:string", word))
                else:
                    tokens.append(("class:arg", word))

            if cursor < len(line):
                tokens.append(("", line[cursor:]))

            return tokens

        return get_tokens


# ── Key bindings ─────────────────────────────────────────────────────────────


def _build_bindings(verbose_cell: list[bool] | None = None) -> KeyBindings:
    kb = KeyBindings()

    @kb.add("c-c")
    def _interrupt(event):
        """Ctrl-C clears the current line (doesn't exit — use 'exit')."""
        event.app.current_buffer.reset()

    @kb.add("c-l")
    def _clear(event):
        """Ctrl-L clears the screen."""
        event.app.renderer.clear()

    @kb.add("c-p")
    def _paste_clean(event):
        """Ctrl+P — paste clipboard content with box-drawing chars stripped."""
        import subprocess as _sp

        try:
            raw = _sp.run(["pbpaste"], capture_output=True, text=True).stdout
            if not raw:
                raw = _sp.run(
                    ["xclip", "-o", "-sel", "clip"], capture_output=True, text=True
                ).stdout
            if not raw:
                raw = _sp.run(
                    ["xsel", "--clipboard", "--output"], capture_output=True, text=True
                ).stdout
            cleaned = _clean_paste(raw)
            event.app.current_buffer.insert_text(cleaned)
        except Exception:
            pass

    @kb.add("c-e")
    def _toggle_verbose(event):
        """Ctrl+E — toggle verbose mode OR expand/collapse resumed conversation."""
        # If a resume toggle hook is active, fire it (expand/collapse prior conversation)
        if _RESUME_TOGGLE_HOOK[0] is not None:
            _RESUME_TOGGLE_HOOK[0]()
            _RESUME_TOGGLE_HOOK[0] = None  # one-shot: clear after first expand
            return
        # Otherwise toggle verbose pipeline output
        if verbose_cell is not None:
            verbose_cell[0] = not verbose_cell[0]
            label = "ON  (showing all agent output)" if verbose_cell[0] else "OFF (compact mode)"
            event.app.current_buffer.insert_text("")  # no-op insert to trigger redraw
            import sys as _sys

            _sys.stdout.write(f"\n  [verbose {label}]\n")
            _sys.stdout.flush()

    return kb


# ── Shell style ──────────────────────────────────────────────────────────────


_STYLE = Style.from_dict(
    {
        "prompt": "bold #00d7ff",  # cyan prompt
        "mode": "#555555",  # dim mode tag beside prompt
        "prompt.bracket": "bold #555555",
        "command": "bold #00d7ff",  # cyan — pipeline commands (run, status, memory…)
        "slash-cmd": "bold #ffd700",  # yellow — session commands (mode, clear, info…)
        "subcommand": "#87d7ff",  # lighter blue — subcommands
        "flag": "#ffd700",  # yellow for flags
        "string": "#87ff87",  # green for strings
        "arg": "#ffffff",  # white for args
        "unknown": "#ff5f5f",  # red for unknown
        "bottom-toolbar": "bg:#222222 #aaaaaa",
        "completion-menu.completion": "bg:#1c1c1c #dddddd",
        "completion-menu.completion.current": "bg:#005f87 bold #ffffff",
        "completion-menu.meta.completion": "bg:#1c1c1c #888888",
        "completion-menu.meta.completion.current": "bg:#005f87 #cccccc",
        "scrollbar.background": "bg:#1c1c1c",
        "scrollbar.button": "bg:#005f87",
    }
)


# ── Toolbar ──────────────────────────────────────────────────────────────────


def _make_toolbar(verbose_cell: list[bool]):
    def _bottom_toolbar():
        verbose_label = "<b style='color:orange'>verbose ON</b>" if verbose_cell[0] else "compact"
        return HTML(
            " <b>Tab</b> complete  "
            " <b>↑↓</b> history  "
            " <b>Ctrl-R</b> search  "
            " <b>Ctrl-L</b> clear  "
            " <b>Ctrl-P</b> paste  "
            f" <b>Ctrl-E</b> {verbose_label}  "
            " <b>exit</b> quit "
        )

    return _bottom_toolbar


# ── Help renderer ─────────────────────────────────────────────────────────────


def _print_help(console_: Console, command: str | None = None) -> None:
    if command and command in _COMMANDS:
        meta = _COMMANDS[command]
        table = Table(show_header=False, border_style="dim", pad_edge=False)
        table.add_column("key", style="bold cyan", no_wrap=True)
        table.add_column("val")

        if "subcommands" in meta:
            for sub, sub_meta in meta["subcommands"].items():
                flags_str = (
                    "  " + "  ".join(sub_meta.get("flags", {}).keys())
                    if sub_meta.get("flags")
                    else ""
                )
                table.add_row(f"  {command} {sub}", sub_meta.get("help", "") + flags_str)
        elif "flags" in meta:
            for flag, vals in meta["flags"].items():
                val_str = f"  [dim]{', '.join(vals)}[/dim]" if vals else ""
                table.add_row(f"  {flag}", val_str)

        console_.print(
            Panel(
                table,
                title=f"[bold cyan]{command}[/bold cyan] — {meta.get('help', '')}",
                border_style="cyan",
                padding=(0, 1),
            )
        )
    else:
        # Full command list
        table = Table(show_header=False, border_style="dim", pad_edge=False, expand=False)
        table.add_column("cmd", style="bold cyan", no_wrap=True, min_width=20)
        table.add_column("help", style="dim")

        for cmd, meta in _COMMANDS.items():
            if cmd in ("quit", "exit"):
                continue
            table.add_row(cmd, meta.get("help", ""))

        console_.print()
        console_.print(
            Panel(
                table,
                title="[bold cyan]AI Factory Commands[/bold cyan]",
                subtitle="[dim]Tab to complete · help <cmd> for details[/dim]",
                border_style="cyan",
                padding=(0, 1),
            )
        )
        console_.print()


# ── Banner ───────────────────────────────────────────────────────────────────

_VERSION = "v0.1"


def _print_banner(console_: Console) -> None:
    cwd = Path.cwd()
    home = Path.home()
    try:
        display_path = "~/" + str(cwd.relative_to(home))
    except ValueError:
        display_path = str(cwd)

    # Factory logo: red smoke/chimney at top, brick-orange body
    t = Text("\n")
    t.append(" ▐█▌  ", style="bold red")  # chimney — red
    t.append("AiFactory ", style="bold white")
    t.append(f"{_VERSION}\n", style="dim")
    t.append("▐███▌ ", style="bold color(130)")  # body — brick orange
    t.append(" self-assembling multi-agent system\n", style="dim")
    t.append("▐█▟█▌ ", style="bold color(130)")  # base — same brick
    t.append(f" {display_path}\n", style="dim white")

    console_.print(t)
    console_.print(
        Panel(
            Text.from_markup(
                "  [bold white]Just type your task[/bold white] [dim]→ runs the pipeline[/dim]\n"
                '  [green]"Build a FastAPI app with auth"[/green]\n'
                '  [green]"What is the capital of France?"[/green]\n\n'
                "  [bold yellow]mode efficient[/bold yellow]  [dim]cost-saving[/dim]  "
                "[dim]·[/dim]  [bold yellow]mode balanced[/bold yellow]  [dim]default[/dim]  "
                "[dim]·[/dim]  [bold yellow]mode max[/bold yellow]  [dim]best quality[/dim]\n"
                "  [bold yellow]clear[/bold yellow]   [dim]reset context[/dim]  "
                "[dim]·[/dim]  [bold yellow]compact[/bold yellow]  [dim]compress context[/dim]  "
                "[dim]·[/dim]  [bold yellow]info[/bold yellow]  [dim]session info[/dim]\n\n"
                "  [bold cyan]run[/bold cyan] [dim]<task>[/dim]   explicit run  "
                "[dim]·[/dim]  [bold cyan]memory stats[/bold cyan]  [dim]layer counts[/dim]  "
                "[dim]·[/dim]  [bold cyan]status[/bold cyan]  [dim]recent runs[/dim]\n"
                "  [bold cyan]help[/bold cyan]  [dim]all commands[/dim]  "
                "[dim]·[/dim]  [bold cyan]memory help[/bold cyan]  [dim]memory cmds[/dim]  "
                "[dim]·[/dim]  [bold cyan]exit[/bold cyan]  [dim]quit[/dim]"
            ),
            border_style="cyan",
            padding=(1, 3),
        )
    )
    console_.print()


# ── Session persistence ──────────────────────────────────────────────────────

_SESSIONS_DIR = Path.home() / ".aifactory" / "sessions"


def _session_path(session_id: str) -> Path:
    return _SESSIONS_DIR / f"{session_id}.json"


def _save_session(
    session_id: str,
    history: list[dict],
    mode: str,
    started_at: float,
    prior_route: str = "",
) -> None:
    """Atomically save session state using a temp-file + rename pattern.

    A direct write_text() call can leave a corrupt file if the process dies
    mid-write (e.g., SIGKILL, disk full).  Writing to a sibling temp file and
    then renaming guarantees the destination is always a complete, valid JSON
    file — rename is atomic on POSIX systems.
    """
    import tempfile

    _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "session_id": session_id,
        "mode": mode,
        "started_at": started_at,
        "saved_at": time.time(),
        "history": history,
        "prior_route": prior_route,
    }
    target = _session_path(session_id)
    payload = json.dumps(data, indent=2)
    # Write to a temp file in the same directory so rename is on the same filesystem
    fd, tmp_path = tempfile.mkstemp(dir=_SESSIONS_DIR, prefix=f".{session_id}-", suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(payload)
        Path(tmp_path).replace(target)  # atomic on POSIX
    except Exception:
        # Best-effort cleanup of temp file on failure; don't crash the shell
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def _load_session(session_id: str) -> dict | None:
    """Load a saved session, distinguishing missing vs. corrupt files."""
    import logging as _log

    p = _session_path(session_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        _log.getLogger(__name__).warning(
            "Session file %s is corrupt (JSONDecodeError: %s) — starting fresh session.",
            p,
            exc,
        )
        return None
    except OSError as exc:
        _log.getLogger(__name__).warning("Could not read session file %s: %s", p, exc)
        return None


def _generate_session_id() -> str:
    """Short memorable session ID: adjective-noun-hex4."""
    _ADJECTIVES = [
        "swift",
        "amber",
        "brave",
        "crisp",
        "delta",
        "eager",
        "frosty",
        "grand",
        "hazy",
        "indigo",
        "jolly",
        "keen",
        "lunar",
        "merry",
        "noble",
        "olive",
        "polar",
        "quiet",
        "rapid",
        "solar",
        "tidal",
        "urban",
        "vivid",
        "warm",
        "xenon",
        "young",
        "zesty",
    ]
    _NOUNS = [
        "agent",
        "beacon",
        "circuit",
        "dragon",
        "ember",
        "falcon",
        "ghost",
        "harbor",
        "impulse",
        "jaguar",
        "kernel",
        "lambda",
        "matrix",
        "nexus",
        "orbit",
        "pixel",
        "quartz",
        "relay",
        "spark",
        "titan",
        "ultra",
        "vertex",
        "wave",
        "xenon",
        "yield",
        "zephyr",
    ]
    import random

    adj = random.choice(_ADJECTIVES)
    noun = random.choice(_NOUNS)
    suffix = uuid.uuid4().hex[:4]
    return f"{adj}-{noun}-{suffix}"


# ── In-process task runner (keeps conversation history) ─────────────────────


def _run_task_in_process(
    task: str,
    conv_history: list[dict[str, str]],
    console_: Console,
    mode: str = "balanced",
    verbose: bool = False,
    _prior_route: list[str] | None = None,
) -> None:
    """Run a task through the pipeline in-process, threading conversation history.

    After the run completes the task + reply are appended to conv_history so
    subsequent turns have context of the full exchange.

    _prior_route is a single-element list [route_str] so it can be mutated by
    the caller to pass prior routing context across turns without rebuilding the
    call signature.
    """
    import importlib

    import aifactory.agents.code_security_processor as _csp_mod
    import aifactory.core.complexity_gate as _cg_mod

    importlib.reload(_cg_mod)
    importlib.reload(_csp_mod)
    from aifactory.core.complexity_gate import ComplexityGate

    gate = ComplexityGate()
    _last_route = _prior_route[0] if _prior_route else None
    decision = gate.evaluate(task, prior_route=_last_route)

    if decision.bypass:
        from aifactory.cli._helpers import render_result

        render_result(decision.answer, console_=console_)
        console_.print("[muted]Instant answer — 0 tokens[/muted]")
        conv_history.append({"role": "user", "content": task})
        conv_history.append({"role": "assistant", "content": decision.answer})
        # bypass doesn't change the session routing context — keep prior_route
        return

    # Pass history into pipeline; cap at last 6 turns, truncate long assistant
    # responses so a previous 50k-token codebase dump doesn't re-flood the next
    # run's context window.
    _MAX_TURN_CHARS = 800
    history_slice = [
        {
            "role": t["role"],
            "content": (
                t["content"][:_MAX_TURN_CHARS] + "…[truncated]"
                if t["role"] == "assistant" and len(t["content"]) > _MAX_TURN_CHARS
                else t["content"]
            ),
        }
        for t in conv_history[-6:]
    ]

    try:
        if decision.fast_llm:
            reply = asyncio.run(
                _run_fast_llm_with_history(task, history_slice, decision.research_context or "")
            )
            if _prior_route is not None:
                _prior_route[0] = "fast_llm"
        else:
            reply = asyncio.run(
                _run_pipeline_with_history(
                    task,
                    history_slice,
                    mode=mode,
                    verbose=verbose,
                    gate_complexity_score=decision.complexity_score,
                )
            )
            if _prior_route is not None:
                _prior_route[0] = "full"
    except KeyboardInterrupt:
        console_.print("\n[warning]Interrupted.[/warning]")
        return

    conv_history.append({"role": "user", "content": task})
    if reply:
        conv_history.append({"role": "assistant", "content": reply})


async def _run_fast_llm_with_history(
    task: str,
    history: list[dict[str, str]],
    research_context: str = "",
) -> str:
    """Single-master call that includes conversation history."""
    from dotenv import load_dotenv

    load_dotenv()

    from aifactory.cli._helpers import run_id_now
    from aifactory.core.memory_bridge import MemoryBridge
    from aifactory.core.model_selector import best_model, load_agent_cfg, load_available_models
    from aifactory.graphs.dynamic_pipeline import PipelineAgent

    models = load_available_models()
    master_model = best_model(models)
    master_cfg = load_agent_cfg("master")

    rid = run_id_now()
    bridge = None
    try:
        from aifactory.core.memory_router import MemoryRouter

        bridge = MemoryBridge(router=MemoryRouter(), run_id=rid)
    except Exception:
        pass

    master = PipelineAgent(
        "master", {**master_cfg, "model": master_model}, master_model, memory=bridge
    )

    # Seed the agent's message history with prior turns
    for turn in history:
        master._messages.append({"role": turn["role"], "content": turn["content"]})

    if research_context:
        prompt = (
            f"Using the research context below, answer the following task.\n\n"
            f"## Research Context\n{research_context}\n\n"
            f'## Task\n"{task}"'
        )
    else:
        prompt = task

    from aifactory.cli._helpers import console, render_result

    reply = await master.think(prompt, {"objective": task, "workspace": str(Path.cwd())})
    render_result(reply.strip(), console_=console)
    console.print(f"[muted]{master.total_tokens} tokens · model={master_model}[/muted]")
    return reply.strip()


async def _run_pipeline_with_history(
    task: str,
    history: list[dict[str, str]],
    mode: str = "balanced",
    verbose: bool = False,
    gate_complexity_score: float = 1.0,
) -> str:
    """Full 8-node pipeline run that threads conversation history into initial state."""
    from dotenv import load_dotenv

    load_dotenv()

    import sys
    import time

    from langgraph.checkpoint.memory import MemorySaver
    from rich.panel import Panel

    from aifactory.cli._helpers import console, format_elapsed, run_id_now
    from aifactory.graphs.dynamic_pipeline import (
        build_dynamic_pipeline,
        load_graph_config,
        write_dump,
    )

    rid = run_id_now()
    checkpointer = MemorySaver()

    _run_t0 = time.perf_counter()
    _node_t0: dict[str, float] = {}
    _worker_t0: dict[str, float] = {}

    _NODE_LABELS: dict[str, tuple[str, str]] = {
        "master_receive": ("📥", "receiving task"),
        "judge_brief": ("⚖️ ", "briefing quality criteria"),
        "analyse": ("🔍", "decomposing into tasks"),
        "researcher": ("🔬", "researching context"),
        "reanalyse": ("🔄", "refining task plan"),
        "execution_lead": ("🎯", "dispatching workers"),
        "judge_review": ("📊", "evaluating quality"),
        "master_respond": ("✍️ ", "synthesising response"),
    }

    def _elapsed_str(t0: float) -> str:
        s = time.perf_counter() - t0
        return f"{s:.1f}s" if s >= 1.0 else f"{int(s * 1000)}ms"

    _ORCHESTRATION_AGENTS = {"master", "judge", "analyser", "researcher", "execution_lead"}

    _suppress_tokens: set[str] = set()  # workers currently in FILE_READ loop

    def _stream_event(event: dict) -> None:
        etype = event.get("type", "")

        if etype == "wave_start":
            wave = event["wave"]
            total = event["total_waves"]
            n = event["task_count"]
            parallel_label = f"  {n} parallel" if event.get("parallel") else ""
            console.print(f"\n  [dim]── Wave {wave}/{total}{parallel_label} ──[/dim]")

        elif etype == "worker_start":
            worker = event["worker"]
            task_desc = event["task"]
            _worker_t0[worker] = time.perf_counter()
            task_lower = task_desc.lower()
            if any(
                k in task_lower
                for k in ("code", "implement", "build", "write code", "debug", "test")
            ):
                icon = "⚙️ "
            elif any(k in task_lower for k in ("research", "find", "search", "investigate")):
                icon = "🔬"
            elif any(k in task_lower for k in ("analyse", "analyze", "review", "audit")):
                icon = "🔍"
            elif any(k in task_lower for k in ("write", "draft", "document", "explain")):
                icon = "✍️ "
            else:
                icon = "▸ "
            console.print(
                f"    {icon} [bold]{worker}[/bold]  "
                f"[dim]{task_desc[:72]}{'…' if len(task_desc) > 72 else ''}[/dim]"
            )

        elif etype == "worker_complete":
            worker = event["worker"]
            elapsed = _elapsed_str(_worker_t0.get(worker, time.perf_counter()))
            if event.get("skipped"):
                console.print(f"    [dim]  ↳ {worker}  skipped  {elapsed}[/dim]")
            elif event.get("quality_flagged"):
                console.print(
                    f"    [yellow]  ↳ {worker}  done  {elapsed}  ⚠ quality flagged[/yellow]"
                )
            else:
                preview = event.get("result_preview", "")
                preview_short = preview[:60].replace("\n", " ") + ("…" if len(preview) > 60 else "")
                preview_str = f"  [dim italic]{preview_short}[/dim italic]" if preview_short else ""
                console.print(f"    [green]  ↳ {worker}  done  {elapsed}[/green]{preview_str}")

        elif etype == "code_analysis_start":
            path = event.get("path", "")
            short_path = path.replace(str(Path.home()), "~") if path else ""
            console.print(
                f"    [dim cyan]  ⊕ indexing codebase[/dim cyan]  [dim]{short_path}[/dim]"
            )

        elif etype == "code_analysis_hit":
            console.print(
                "    [dim cyan]  ⊕ codebase index  [italic]cache hit — reused[/italic][/dim cyan]"
            )

        elif etype == "file_read_tokens_suppress":
            _suppress_tokens.add(event.get("worker", ""))

        elif etype == "file_read_tokens_resume":
            _suppress_tokens.discard(event.get("worker", ""))

        elif etype == "file_read_start":
            path = event.get("path", "")
            short = path.replace(str(Path.home()), "~")
            console.print(f"    [dim cyan]  ⊕ reading[/dim cyan]  [dim]{short}[/dim]")

        elif etype == "file_read_done":
            path = event.get("path", "")
            short = path.replace(str(Path.home()), "~")
            lang = event.get("lang", "")
            size = event.get("size_kb", 0)
            truncated = event.get("truncated", False)
            err = event.get("error")
            if err:
                console.print(f"    [dim red]  ✗ {short}  {err}[/dim red]")
            else:
                meta = f"{lang}, {size} KB" if lang else f"{size} KB"
                trunc = "  [dim italic]truncated[/dim italic]" if truncated else ""
                console.print(
                    f"    [dim cyan]  ✓ read[/dim cyan]  [dim]{short}[/dim]"
                    f"  [dim italic]({meta})[/dim italic]{trunc}"
                )

        elif etype == "budget_warning":
            spent = event.get("spent", 0)
            cap = event.get("cap", 0)
            pct = int(spent / cap * 100) if cap > 0 else 0
            bar_filled = int(pct / 5)
            bar = "█" * bar_filled + "░" * (20 - bar_filled)
            console.print(
                f"\n  [bold yellow]⚠  Budget[/bold yellow]  "
                f"[yellow]{bar}[/yellow]  [bold yellow]{pct}%[/bold yellow]  "
                f"[dim]${spent:.4f} of ${cap:.2f}[/dim]\n"
            )

        elif etype == "qdrant_in_memory":
            console.print(
                "  [dim yellow]⚠  Memory in-memory only — set QDRANT_URL for persistence[/dim yellow]"
            )

        elif etype == "token":
            agent = event.get("agent", "")
            is_orchestration = any(agent.startswith(name) for name in _ORCHESTRATION_AGENTS)
            if agent not in _suppress_tokens and (verbose or not is_orchestration):
                sys.stdout.write(event.get("text", ""))
                sys.stdout.flush()

        elif etype == "report":
            print(event.get("text", ""))

    compiled, registry = build_dynamic_pipeline(
        mode=mode,
        checkpointer=checkpointer,
        run_id=rid,
        event_callback=_stream_event,
        gate_complexity_score=gate_complexity_score,
    )

    _mode_colours = {"best": "magenta", "balanced": "cyan", "cost_efficient": "green"}
    _mc = _mode_colours.get(mode, "white")
    console.print(
        f"  [dim]run[/dim] [bold]{rid}[/bold]  "
        f"[dim]mode[/dim] [bold {_mc}]{mode}[/bold {_mc}]  "
        f"[dim]agents[/dim] {len(list(registry.all()))}"
    )
    console.print()

    initial_state = {
        "user_query": task,
        "shared_context": {"workspace": str(Path.cwd())},
        "conversation_history": history,
    }

    thread_cfg = {"configurable": {"thread_id": rid}}
    t0 = time.perf_counter()
    result: dict = {}

    from langgraph.types import Command

    from aifactory.cli._helpers import QUIT_WORDS

    current_input = initial_state

    try:
        while True:
            async for chunk in compiled.astream(
                current_input, config=thread_cfg, stream_mode="updates"
            ):
                for node_name, update in chunk.items():
                    if node_name == "__start__":
                        _node_t0[node_name] = time.perf_counter()
                        continue
                    elapsed_since_start = _elapsed_str(_run_t0)
                    icon, label = _NODE_LABELS.get(node_name, ("▸", node_name))
                    console.print(
                        f"  {icon} [bold]{node_name}[/bold]  "
                        f"[dim]{label}  ·  +{elapsed_since_start}[/dim]"
                    )
                    if isinstance(update, dict):
                        result.update(update)

            snap = compiled.get_state(thread_cfg)
            if not snap.next:
                break

            interrupt_msg = ""
            for snap_task in snap.tasks:
                for intr in snap_task.interrupts:
                    interrupt_msg = str(intr.value)
                    break
                if interrupt_msg:
                    break

            from rich.panel import Panel

            console.print(
                Panel(
                    interrupt_msg, title="🤔 Worker Question", border_style="yellow", expand=False
                )
            )
            answer = input("[You] > ").strip()
            if not answer or answer.lower() in QUIT_WORDS:
                break
            current_input = Command(resume=answer)

    except KeyboardInterrupt:
        console.print("\n[warning]Interrupted.[/warning]")

    elapsed = time.perf_counter() - t0
    cfg = load_graph_config()
    dump_path = write_dump(rid, cfg, result, registry, elapsed, task)

    final = result.get("final_response", "") or result.get("direct_answer", "")
    console.print(
        f"\n[bold green]═══ Pipeline Complete ({format_elapsed(elapsed)}) ═══[/bold green]\n"
    )
    if final.strip():
        from aifactory.cli._helpers import render_result
        render_result(final, console_=console)

    from aifactory.cli._helpers import print_agent_table, print_task_results

    print_task_results(result.get("task_results", []))
    console.print()
    print_agent_table([a.dump() for a in registry.all()])

    judge = result.get("judge_metrics", {})
    if judge:
        console.print("\n[bold]Judge Metrics:[/bold]")
        for k, v in judge.items():
            console.print(f"  {k}: {v}")

    console.print(f"\n[muted]Run: {rid}  ·  dump: {dump_path}[/muted]\n")
    return final.strip()


# ── Resumed conversation renderer ───────────────────────────────────────────

# Conversation is shown verbatim when it fits within this char budget; above it
# we show a compact summary.  Ctrl+E toggles between the two views.
_RESUME_VERBATIM_THRESHOLD = 4_000  # total chars across all turns


def _render_conversation(
    console_: Console,
    history: list[dict],
    compact: bool,
) -> None:
    """Render the conversation history either verbatim or as a compact summary."""
    pairs: list[tuple[str, str]] = []
    it = iter(history)
    for turn in it:
        if turn["role"] == "user":
            asst = next(it, None)
            pairs.append((turn["content"], asst["content"] if asst else ""))

    if not pairs:
        return

    if compact:
        # One-line-per-turn summary table
        table = Table(show_header=True, border_style="dim", pad_edge=False, expand=False)
        table.add_column("#", style="dim", no_wrap=True, width=3)
        table.add_column("You asked", style="bold white", no_wrap=False, max_width=55)
        table.add_column("Answer preview", style="dim", no_wrap=False, max_width=55)
        for i, (q, a) in enumerate(pairs, 1):
            q_short = q[:80].replace("\n", " ") + ("…" if len(q) > 80 else "")
            a_short = a[:80].replace("\n", " ") + ("…" if len(a) > 80 else "")
            table.add_row(str(i), q_short, a_short)
        console_.print()
        console_.print(
            Panel(
                table,
                title=f"[bold cyan]Prior conversation — {len(pairs)} turn{'s' if len(pairs) != 1 else ''}[/bold cyan]  "
                "[dim](Ctrl+E to expand)[/dim]",
                border_style="dim",
                padding=(0, 1),
            )
        )
        console_.print()
    else:
        # Full verbatim replay — render each turn as the shell would have shown it
        console_.print()
        console_.print(
            f"[dim]─── Prior conversation ({len(pairs)} turn{'s' if len(pairs) != 1 else ''}) ───[/dim]"
        )
        console_.print()
        from aifactory.cli._helpers import render_result
        for i, (q, a) in enumerate(pairs, 1):
            console_.print(f"[bold cyan]You ({i}):[/bold cyan] {q}")
            render_result(a.strip() or "(no response)", console_=console_)
        console_.print("[dim]─── End of prior conversation — continuing from here ───[/dim]")
        console_.print()


def _print_resumed_conversation(
    console_: Console,
    history: list[dict],
    session_id: str,
    mode: str,
    verbose_cell: list[bool],
) -> None:
    """Print the resumed conversation and show a session header."""
    turns = len(history) // 2
    total_chars = sum(len(t.get("content", "")) for t in history)
    mode_label = _MODE_DISPLAY.get(mode, mode)

    console_.print(
        f"\n[dim]Resumed [bold cyan]{session_id}[/bold cyan]  "
        f"· {turns} prior turn{'s' if turns != 1 else ''}  "
        f"· mode: [bold]{mode_label}[/bold][/dim]"
    )

    if turns == 0:
        console_.print("[dim]No prior conversation to show.[/dim]\n")
        return

    # Decide verbatim vs compact based on total content size
    compact = total_chars > _RESUME_VERBATIM_THRESHOLD

    if compact:
        console_.print(
            "[dim]Conversation was long — showing compact summary.  "
            "[bold]Ctrl+E[/bold] to expand full history.[/dim]"
        )
        # Wire Ctrl+E so toggling verbose_cell also re-renders the history
        # We store the render state in a cell so the key binding can flip it.
        _compact_cell: list[bool] = [True]

        def _toggle_and_rerender() -> None:
            _compact_cell[0] = not _compact_cell[0]
            _render_conversation(console_, history, compact=_compact_cell[0])

        # Patch the verbose cell toggle to also re-render when used at resume time
        # (the verbose_cell is used by Ctrl+E in the key binding; we piggyback on it)
        # We can't directly patch the key binding here, so we store the toggle fn
        # as a module-level hook that _build_bindings can call.
        _RESUME_TOGGLE_HOOK[0] = _toggle_and_rerender
    else:
        _RESUME_TOGGLE_HOOK[0] = None

    _render_conversation(console_, history, compact=compact)


# Hook slot for Ctrl+E to re-render resumed conversation (set at resume time)
_RESUME_TOGGLE_HOOK: list = [None]


# ── Session info renderer ────────────────────────────────────────────────────


def _print_session_info(
    console_: Console,
    session_id: str,
    mode: str,
    history: list[dict],
    started_at: float,
) -> None:
    elapsed = time.time() - started_at
    hours, rem = divmod(int(elapsed), 3600)
    mins, secs = divmod(rem, 60)
    uptime = f"{hours}h {mins}m {secs}s" if hours else f"{mins}m {secs}s"

    turns = len(history) // 2
    context_chars = sum(len(t.get("content", "")) for t in history)

    table = Table(show_header=False, border_style="dim", pad_edge=False, expand=False)
    table.add_column("key", style="bold cyan", no_wrap=True, min_width=18)
    table.add_column("val", style="white")

    table.add_row("Session ID", f"[bold]{session_id}[/bold]")
    table.add_row("Mode", _MODE_DISPLAY.get(mode, mode))
    table.add_row("Uptime", uptime)
    table.add_row("Turns", str(turns))
    table.add_row("Context size", f"{context_chars:,} chars")
    table.add_row("Resume", f"aifactory --resume {session_id}")

    # List last 3 user queries
    user_turns = [t["content"][:80] for t in history if t["role"] == "user"]
    if user_turns:
        last = user_turns[-3:]
        for i, q in enumerate(last):
            label = "Recent queries" if i == 0 else ""
            table.add_row(label, f"[dim]{q}[/dim]")

    console_.print()
    console_.print(
        Panel(
            table, title="[bold cyan]Session Info[/bold cyan]", border_style="cyan", padding=(0, 1)
        )
    )
    console_.print()


# ── Main shell entry point ───────────────────────────────────────────────────


def shell(
    vi_mode: bool = typer.Option(False, "--vi", help="Use Vi keybindings instead of Emacs."),
    resume_id: str = typer.Option(None, "--resume", "-r", help="Resume a saved session by ID."),
) -> None:
    """🐚 Launch the interactive AI Factory shell (Claude Code style)."""
    from aifactory.cli._helpers import console as rich_console

    history_file = Path.home() / ".aifactory" / "shell_history"
    history_file.parent.mkdir(parents=True, exist_ok=True)

    # Shared mutable cell — lets Ctrl+E key binding toggle verbose before loop starts
    _verbose_cell: list[bool] = [False]

    session: PromptSession = PromptSession(
        history=FileHistory(str(history_file)),
        auto_suggest=_SmartAutoSuggest(),
        completer=AifactoryCompleter(),
        lexer=AifactoryLexer(),
        style=_STYLE,
        vi_mode=vi_mode,
        key_bindings=_build_bindings(verbose_cell=_verbose_cell),
        bottom_toolbar=_make_toolbar(_verbose_cell),
        mouse_support=False,
        complete_while_typing=True,
        complete_in_thread=True,
        multiline=False,
        wrap_lines=True,
        enable_history_search=True,
    )

    _print_banner(rich_console)

    # Resolve the aifactory executable (same env as the running interpreter)
    aifactory_bin = Path(sys.executable).parent / "aifactory"
    if not aifactory_bin.exists():
        aifactory_cmd = [sys.executable, "-m", "aifactory"]
    else:
        aifactory_cmd = [str(aifactory_bin)]

    # Session state — optionally restored from a saved session
    _session_id: str
    _conv_history: list[dict[str, str]]
    _mode: str
    _started_at: float
    # Mutable cell so _run_task_in_process can update the last route across calls
    _prior_route_cell: list[str] = [""]

    if resume_id:
        saved = _load_session(resume_id)
        if saved:
            _session_id = saved["session_id"]
            _conv_history = saved.get("history", [])
            _mode = saved.get("mode", "balanced")
            _started_at = saved.get("started_at", time.time())
            _prior_route_cell[0] = saved.get("prior_route", "")
            _print_resumed_conversation(
                rich_console, _conv_history, _session_id, _mode, _verbose_cell
            )
        else:
            rich_console.print(f"[bold red]Session not found:[/bold red] {resume_id}\n")
            _session_id = _generate_session_id()
            _conv_history = []
            _mode = "balanced"
            _started_at = time.time()
    else:
        _session_id = _generate_session_id()
        _conv_history = []
        _mode = "balanced"
        _started_at = time.time()

    # _verbose_cell[0] is read here; toggled by Ctrl+E via key binding above

    def _prompt_text() -> HTML:
        mode_tag = _MODE_DISPLAY.get(_mode, _mode)
        return HTML(
            f"<prompt>aifactory</prompt>"
            f"<mode> ({mode_tag})</mode>"
            f" <prompt.bracket>❯</prompt.bracket> "
        )

    while True:
        try:
            raw = session.prompt(_prompt_text)
        except KeyboardInterrupt:
            continue
        except EOFError:
            _save_session(_session_id, _conv_history, _mode, _started_at, _prior_route_cell[0])
            rich_console.print(
                f"\n[dim]Session saved: [bold cyan]{_session_id}[/bold cyan]  "
                f"Resume with: [bold]aifactory --resume {_session_id}[/bold][/dim]"
            )
            break

        line = raw.strip()
        if not line:
            continue

        # For command detection use only the first line of potentially multi-line input
        first_line = line.splitlines()[0].strip()

        # ── Slash commands — accept both /cmd and bare cmd ────────────
        # e.g. "/mode balanced" and "mode balanced" are identical.
        _SLASH_KEYWORDS = {cmd.lstrip("/") for cmd in _SLASH_COMMANDS}
        _first_word = first_line.split()[0].lower() if first_line.split() else ""
        _is_slash_input = first_line.startswith("/") or _first_word in _SLASH_KEYWORDS

        if _is_slash_input:
            parts = first_line.split()
            # Normalise: prepend "/" if the bare keyword was used
            slash_cmd = ("/" + parts[0].lstrip("/")).lower()

            if slash_cmd in ("/exit", "/quit"):
                _save_session(_session_id, _conv_history, _mode, _started_at, _prior_route_cell[0])
                rich_console.print(
                    f"[dim]Session saved: [bold cyan]{_session_id}[/bold cyan]  "
                    f"Resume with: [bold]aifactory --resume {_session_id}[/bold][/dim]"
                )
                break

            elif slash_cmd == "/clear":
                _conv_history.clear()
                rich_console.print("[dim]Conversation context cleared.[/dim]")

            elif slash_cmd == "/compact":
                sub = parts[1].lower() if len(parts) > 1 else ""
                if sub == "info":
                    # Show session info without compacting
                    _print_session_info(
                        rich_console, _session_id, _mode, _conv_history, _started_at
                    )
                elif not _conv_history:
                    rich_console.print("[dim]No conversation context to compact.[/dim]")
                else:
                    # Summarise context into a single system turn
                    turns = len(_conv_history) // 2
                    summary_lines = []
                    for turn in _conv_history:
                        if turn["role"] == "user":
                            summary_lines.append(f"User asked: {turn['content'][:120]}")
                        else:
                            summary_lines.append(f"Assistant answered: {turn['content'][:120]}")
                    compact = "Prior conversation summary:\n" + "\n".join(summary_lines[-10:])
                    _conv_history.clear()
                    _conv_history.append({"role": "user", "content": compact})
                    _conv_history.append(
                        {"role": "assistant", "content": "Understood, I have the context."}
                    )
                    rich_console.print(f"[dim]Compacted {turns} turns into a summary.[/dim]")

            elif slash_cmd == "/mode":
                current = _MODE_DISPLAY.get(_mode, _mode)
                if len(parts) < 2 or parts[1].lower() in ("info", "?", "status"):
                    rich_console.print(
                        f"[dim]Current mode: [bold]{current}[/bold]  "
                        f"Options: [bold]efficient[/bold] · [bold]balanced[/bold] · [bold]max[/bold][/dim]"
                    )
                else:
                    alias = parts[1].lower()
                    resolved = _MODE_ALIASES.get(alias)
                    if resolved:
                        _mode = resolved
                        display = _MODE_DISPLAY[resolved]
                        rich_console.print(f"[dim]Mode → [bold]{display}[/bold][/dim]")
                    else:
                        rich_console.print(
                            f"[dim]Unknown mode [bold red]{alias}[/bold red].  "
                            f"Current: [bold]{current}[/bold]  "
                            f"Options: efficient · balanced · max[/dim]"
                        )

            elif slash_cmd == "/info":
                _print_session_info(rich_console, _session_id, _mode, _conv_history, _started_at)

            elif slash_cmd == "/memory":
                # /memory → stats  (default)
                # /memory stats | search <q> | health | clear [--layer X] [--yes]
                mem_args = parts[1:] if len(parts) > 1 else ["stats"]
                # Strip a leading "help" shorthand → show stats as a proxy
                if mem_args[0] in ("help", "?", "/help"):
                    rich_console.print(
                        "[dim]/memory [bold]stats[/bold]          — layer counts + persistent facts\n"
                        "/memory [bold]health[/bold]         — backend connectivity check\n"
                        "/memory [bold]search[/bold] <query> — semantic search across memory\n"
                        "/memory [bold]clear[/bold] [--layer agent|team|global] [--yes][/dim]"
                    )
                else:
                    result = subprocess.run(aifactory_cmd + ["memory"] + mem_args, check=False)
                    if result.returncode not in (0, 1):
                        rich_console.print(f"[dim]exit code {result.returncode}[/dim]")

            elif slash_cmd == "/verbose":
                _verbose_cell[0] = not _verbose_cell[0]
                label = (
                    "ON  — showing all agent output"
                    if _verbose_cell[0]
                    else "OFF — compact mode (orchestration agents hidden)"
                )
                rich_console.print(f"[dim]Verbose [bold]{label}[/bold][/dim]")

            elif slash_cmd == "/status":
                result = subprocess.run(aifactory_cmd + ["status"], check=False)
                if result.returncode not in (0, 1):
                    rich_console.print(f"[dim]exit code {result.returncode}[/dim]")

            elif slash_cmd == "/help":
                table = Table(show_header=False, border_style="dim", pad_edge=False, expand=False)
                table.add_column("cmd", no_wrap=True, min_width=12)
                table.add_column("help", style="dim")
                for cmd, help_text in _SLASH_COMMANDS.items():
                    bare = cmd.lstrip("/")
                    table.add_row(f"[bold yellow]{bare}[/bold yellow]  [dim]{cmd}[/dim]", help_text)
                rich_console.print()
                rich_console.print(
                    Panel(
                        table,
                        title="[bold cyan]Commands[/bold cyan]  [dim](works with or without /)[/dim]",
                        border_style="cyan",
                        padding=(0, 1),
                    )
                )
                rich_console.print()

            else:
                rich_console.print(
                    f"[bold red]Unknown slash command:[/bold red] {slash_cmd}  "
                    f"[dim](type /help for options)[/dim]"
                )
            continue

        # Built-in commands (single-line only)
        if first_line in ("exit", "quit", "q"):
            _save_session(_session_id, _conv_history, _mode, _started_at, _prior_route_cell[0])
            rich_console.print(
                f"[dim]Session saved: [bold cyan]{_session_id}[/bold cyan]  "
                f"Resume with: [bold]aifactory --resume {_session_id}[/bold][/dim]"
            )
            break

        if first_line == "clear":
            import os

            os.system("clear" if os.name != "nt" else "cls")
            continue

        if first_line == "history reset":
            _conv_history.clear()
            rich_console.print("[dim]Conversation history cleared.[/dim]")
            continue

        # help [command]
        if first_line == "help" or first_line.startswith("help "):
            parts = first_line.split(maxsplit=1)
            cmd = parts[1].strip() if len(parts) > 1 else None
            _print_help(rich_console, cmd)
            continue

        # Fast path: if the first word isn't a known command, treat the whole raw
        # input as a task (no shlex parsing) — handles natural language with
        # apostrophes, quotes, and multi-line pasted content.
        first_word = first_line.split()[0] if first_line.split() else ""
        if first_word not in _COMMANDS:
            rich_console.print(
                "[dim]  → running as task (prefix with [bold]run[/bold] to be explicit)[/dim]"
            )
            _run_task_in_process(
                line,
                _conv_history,
                rich_console,
                mode=_mode,
                verbose=_verbose_cell[0],
                _prior_route=_prior_route_cell,
            )
            continue

        # Parse args for delegation (only runs when first word is a known command)
        # Use first_line so that multi-line input doesn't confuse shlex.
        try:
            args = shlex.split(first_line)
        except ValueError as exc:
            rich_console.print(f"[bold red]Parse error:[/bold red] {exc}")
            continue

        # Explicit `run <task>` — extract task text and run in-process with history
        if args[0] == "run":
            # Collect --mode flag if present; fall back to session _mode
            mode = _mode
            remaining = args[1:]
            task_parts: list[str] = []
            skip_next = False
            for i, tok in enumerate(remaining):
                if skip_next:
                    skip_next = False
                    continue
                if tok in ("--mode", "-m") and i + 1 < len(remaining):
                    mode = remaining[i + 1]
                    skip_next = True
                elif tok.startswith("--mode="):
                    mode = tok.split("=", 1)[1]
                elif not tok.startswith("-"):
                    task_parts.append(tok)

            task_text = " ".join(task_parts).strip()
            if not task_text:
                rich_console.print("[bold red]Provide a task.[/bold red]")
                continue
            _run_task_in_process(
                task_text,
                _conv_history,
                rich_console,
                mode=mode,
                verbose=_verbose_cell[0],
                _prior_route=_prior_route_cell,
            )
            continue

        # All other commands go through subprocess as before
        try:
            result = subprocess.run(aifactory_cmd + args, check=False)
            if result.returncode not in (0, 1):
                rich_console.print(f"[dim]exit code {result.returncode}[/dim]")
        except FileNotFoundError:
            rich_console.print(
                "[bold red]Error:[/bold red] Could not find 'aifactory' binary. "
                "Make sure the package is installed in the active venv."
            )
        except Exception as exc:
            rich_console.print(f"[bold red]Error:[/bold red] {exc}")
