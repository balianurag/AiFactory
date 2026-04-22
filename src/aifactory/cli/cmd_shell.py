"""
aifactory shell — Claude Code-style interactive REPL.

Launches a persistent terminal-within-terminal session with:
  • Auto-complete dropdown for all commands + subcommands + flags
  • Command history (up/down arrows, Ctrl-R reverse search)
  • Syntax highlighting (commands in cyan, flags in yellow, strings in green)
  • Inline help hints beneath the prompt
  • Multi-line input (Esc+Enter)
  • Vi or Emacs keybindings (default: Emacs)

Type a command normally — no slash prefix needed:
  run "Build a FastAPI app"
  status
  memory stats
  agents list
  help
  exit
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.styles import Style
from prompt_toolkit.document import Document
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import typer

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
                "flags": {"--type": ["coding", "math", "research", "writing", "analysis", "general"]},
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


# ── Completer ────────────────────────────────────────────────────────────────


class AifactoryCompleter(Completer):
    """Tab-complete commands, subcommands, flags, and flag values."""

    def get_completions(self, document: Document, complete_event) -> Iterable[Completion]:
        text = document.text_before_cursor.lstrip()
        trailing_space = text.endswith(" ")

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

    def _complete_flags(self, parts: list[str], meta: dict, trailing_space: bool = False) -> Iterable[Completion]:
        flags: dict[str, list[str]] = meta.get("flags", {})

        if trailing_space:
            # Cursor is after a space — "last" is empty, second_last is parts[-1]
            last = ""
            second_last = parts[-1] if parts else ""
        else:
            last = parts[-1] if parts else ""
            second_last = parts[-2] if len(parts) >= 2 else ""

        # If the previous token is a flag that takes values, complete the values
        if second_last in flags and flags[second_last]:
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
    """Colour commands cyan, flags yellow, quoted strings green."""

    def lex_document(self, document: Document):
        def get_tokens(line_number: int):
            line = document.lines[line_number]
            tokens = []
            i = 0
            words = line.split()
            cursor = 0

            for idx, word in enumerate(words):
                # Find word position in original line
                pos = line.find(word, cursor)
                # Spaces before this word
                if pos > cursor:
                    tokens.append(("", line[cursor:pos]))
                cursor = pos + len(word)

                if idx == 0:  # command
                    if word in _COMMANDS:
                        tokens.append(("class:command", word))
                    else:
                        # Free-form text → treat as task input (white, not red)
                        tokens.append(("class:arg", word))
                elif idx == 1 and words[0] in _COMMANDS and "subcommands" in _COMMANDS[words[0]]:
                    subs = _COMMANDS[words[0]]["subcommands"]
                    if word in subs:
                        tokens.append(("class:subcommand", word))
                    else:
                        tokens.append(("class:arg", word))
                elif word.startswith("--") or (word.startswith("-") and len(word) == 2):
                    tokens.append(("class:flag", word))
                elif word.startswith('"') or word.startswith("'"):
                    tokens.append(("class:string", word))
                else:
                    tokens.append(("class:arg", word))

            # Trailing space
            if cursor < len(line):
                tokens.append(("", line[cursor:]))

            return tokens

        return get_tokens


# ── Key bindings ─────────────────────────────────────────────────────────────


def _build_bindings() -> KeyBindings:
    kb = KeyBindings()

    @kb.add("c-c")
    def _interrupt(event):
        """Ctrl-C clears the current line (doesn't exit — use 'exit')."""
        event.app.current_buffer.reset()

    @kb.add("c-l")
    def _clear(event):
        """Ctrl-L clears the screen."""
        event.app.renderer.clear()

    return kb


# ── Shell style ──────────────────────────────────────────────────────────────


_STYLE = Style.from_dict(
    {
        "prompt": "bold #00d7ff",          # cyan prompt
        "prompt.bracket": "bold #555555",
        "command": "bold #00d7ff",         # cyan for commands
        "subcommand": "#87d7ff",           # lighter blue
        "flag": "#ffd700",                 # yellow for flags
        "string": "#87ff87",               # green for strings
        "arg": "#ffffff",                  # white for args
        "unknown": "#ff5f5f",              # red for unknown
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


def _bottom_toolbar():
    return HTML(
        " <b>Tab</b> complete  "
        " <b>↑↓</b> history  "
        " <b>Ctrl-R</b> search  "
        " <b>Ctrl-L</b> clear  "
        " <b>exit</b> quit "
    )


# ── Help renderer ─────────────────────────────────────────────────────────────


def _print_help(console_: Console, command: str | None = None) -> None:
    if command and command in _COMMANDS:
        meta = _COMMANDS[command]
        table = Table(show_header=False, border_style="dim", pad_edge=False)
        table.add_column("key", style="bold cyan", no_wrap=True)
        table.add_column("val")

        if "subcommands" in meta:
            for sub, sub_meta in meta["subcommands"].items():
                flags_str = "  " + "  ".join(sub_meta.get("flags", {}).keys()) if sub_meta.get("flags") else ""
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

_LOGO_LINES = [
    "  [bold cyan]╔═╗╦[/bold cyan][bold white]╔═╗┌─┐┌─┐┌┬┐┌─┐┬─┐┬ ┬[/bold white]",
    "  [bold cyan]╠═╣║[/bold cyan][bold white]╠╣ ├─┤│   │ │ │├┬┘└┬┘[/bold white]",
    "  [bold cyan]╩ ╩╩[/bold cyan][bold white]╚  ┴ ┴└─┘ ┴ └─┘┴└─ ┴ [/bold white]  [dim]v0.1  ·  self-assembling multi-agent system[/dim]",
]


def _print_banner(console_: Console) -> None:
    console_.print()
    for line in _LOGO_LINES:
        console_.print(line)
    console_.print()
    console_.print(
        Panel(
            Text.from_markup(
                "  [bold]Just type your task[/bold] [dim]→ runs the pipeline automatically[/dim]\n"
                '  [green]"Build a FastAPI app with auth"[/green]\n'
                '  [green]"What is the capital of France?"[/green]\n\n'
                "  [bold cyan]run[/bold cyan] [dim]<task>[/dim]    explicitly run  [bold cyan]·[/bold cyan]  "
                "[bold cyan]status[/bold cyan]       recent runs\n"
                "  [bold cyan]memory stats[/bold cyan]  layer counts  [bold cyan]·[/bold cyan]  "
                "[bold cyan]help[/bold cyan]         all commands\n"
                "  [bold cyan]exit[/bold cyan]          quit"
            ),
            border_style="cyan",
            padding=(1, 3),
        )
    )
    console_.print()


# ── Main shell entry point ───────────────────────────────────────────────────


def shell(
    vi_mode: bool = typer.Option(False, "--vi", help="Use Vi keybindings instead of Emacs."),
) -> None:
    """🐚 Launch the interactive AI Factory shell (Claude Code style)."""
    from aifactory.cli._helpers import console as rich_console

    history_file = Path.home() / ".aifactory" / "shell_history"
    history_file.parent.mkdir(parents=True, exist_ok=True)

    session: PromptSession = PromptSession(
        history=FileHistory(str(history_file)),
        auto_suggest=AutoSuggestFromHistory(),
        completer=AifactoryCompleter(),
        lexer=AifactoryLexer(),
        style=_STYLE,
        vi_mode=vi_mode,
        key_bindings=_build_bindings(),
        bottom_toolbar=_bottom_toolbar,
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
        # Fallback: run as a module
        aifactory_cmd = [sys.executable, "-m", "aifactory"]
    else:
        aifactory_cmd = [str(aifactory_bin)]

    while True:
        try:
            raw = session.prompt(
                HTML("<prompt>aifactory</prompt> <prompt.bracket>❯</prompt.bracket> "),
            )
        except KeyboardInterrupt:
            continue
        except EOFError:
            # Ctrl-D
            rich_console.print("\n[dim]Bye![/dim]")
            break

        line = raw.strip()
        if not line:
            continue

        # Built-in commands
        if line in ("exit", "quit", "q"):
            rich_console.print("[dim]Bye![/dim]")
            break

        if line == "clear":
            import os
            os.system("clear" if os.name != "nt" else "cls")
            continue

        # help [command]
        if line == "help" or line.startswith("help "):
            parts = line.split(maxsplit=1)
            cmd = parts[1].strip() if len(parts) > 1 else None
            _print_help(rich_console, cmd)
            continue

        # Parse args for delegation
        try:
            args = shlex.split(line)
        except ValueError as exc:
            rich_console.print(f"[bold red]Parse error:[/bold red] {exc}")
            continue

        # If the first word isn't a known command, treat the whole line as a task
        # (implicit `run` — just like Claude Code where you type anything to chat)
        if args and args[0] not in _COMMANDS:
            rich_console.print(
                f"[dim]  → running as task (prefix with [bold]run[/bold] to be explicit)[/dim]"
            )
            args = ["run", line]

        try:
            result = subprocess.run(
                aifactory_cmd + args,
                check=False,
            )
            if result.returncode not in (0, 1):
                # 1 = typer "show help" exit, which is normal
                rich_console.print(f"[dim]exit code {result.returncode}[/dim]")
        except FileNotFoundError:
            rich_console.print(
                "[bold red]Error:[/bold red] Could not find 'aifactory' binary. "
                "Make sure the package is installed in the active venv."
            )
        except Exception as exc:
            rich_console.print(f"[bold red]Error:[/bold red] {exc}")
