"""
AI Factory CLI — A smart, Claude-Code-style command-line interface.

Usage:
    aifactory run "Build a FastAPI todo app with auth"
    aifactory run --mode best -f spec.md
    aifactory run -m light -c "Use PostgreSQL" "Build a user service"
    aifactory run --attachment screenshot.png "What's wrong with this UI?"
    aifactory resume <task-id>
    aifactory status
    aifactory connect add github
    aifactory connect list
    aifactory connect remove github
    aifactory tools
    aifactory ask "What's the status of login feature?"
    aifactory inspect <task-id>
    aifactory agents list
    aifactory cost today
    aifactory cost estimate "Build a FastAPI app" --mode best
    aifactory gate "2 + 2"
    aifactory gate "Build a microservices app" --verbose

Module layout:
    cli/
    ├── __init__.py      ← this file (app + re-exports)
    ├── _helpers.py      ← shared console, logging, dump-loading
    ├── cmd_run.py       ← `aifactory run` + `aifactory resume` (unified entry point)
    ├── cmd_status.py    ← `aifactory status` / `aifactory inspect`
    ├── cmd_connect.py   ← `aifactory connect add/list/remove` / `aifactory tools`
    ├── cmd_ask.py       ← `aifactory ask`
    ├── cmd_agents.py    ← `aifactory agents list`
    ├── cmd_cost.py      ← `aifactory cost today/week/all/run/estimate`
    ├── cmd_info.py      ← `aifactory info` / `aifactory gate` / `aifactory models`
    └── cmd_sandbox.py   ← `aifactory sandbox build/status/test/pull`
"""

from __future__ import annotations

import typer

from aifactory.cli._helpers import console  # noqa: F401

# ── Root app ─────────────────────────────────────────────────────────────────
# Forward-declare so the callback can reference it before _register_commands()


def _launch_shell(ctx: typer.Context) -> None:  # noqa: D401
    """Default action: open the interactive shell when no subcommand is given."""
    if ctx.invoked_subcommand is None:
        from aifactory.cli.cmd_shell import shell as _shell

        _shell(vi_mode=False)
        raise typer.Exit()

app = typer.Typer(
    name="aifactory",
    help=(
        "🏭 AI Factory — Self-assembling agent system that turns ideas "
        "into production output.\n\n"
        "Quick start:\n"
        '  aifactory run "Build a FastAPI todo app with auth"\n'
        "  aifactory status\n"
        '  aifactory ask "What\'s the status of login feature?"\n\n'
        "  [bold]aifactory[/bold] (no args) → opens interactive shell"
    ),
    no_args_is_help=False,
    invoke_without_command=True,
    rich_markup_mode="rich",
)

app.callback()(_launch_shell)


# ── Register command modules ────────────────────────────────────────────────


def _register_commands() -> None:
    """Lazy import + registration to avoid heavy imports at CLI startup."""
    from aifactory.cli.cmd_agents import agents_app
    from aifactory.cli.cmd_ask import ask
    from aifactory.cli.cmd_batch import batch_app
    from aifactory.cli.cmd_connect import connect_app, tools
    from aifactory.cli.cmd_cost import cost_app
    from aifactory.cli.cmd_info import gate, info, memory_stats, models_app
    from aifactory.cli.cmd_memory import memory_app
    from aifactory.cli.cmd_meta import app as meta_app
    from aifactory.cli.cmd_rl import rl_app
    from aifactory.cli.cmd_run import resume, run
    from aifactory.cli.cmd_runs import runs_app
    from aifactory.cli.cmd_sandbox import sandbox_app
    from aifactory.cli.cmd_serve import serve
    from aifactory.cli.cmd_shell import shell
    from aifactory.cli.cmd_status import inspect_cmd, status

    # Top-level commands
    app.command("resume")(resume)
    app.command("run")(run)
    app.command("serve")(serve)
    app.command("shell")(shell)
    app.command("status")(status)
    app.command("inspect")(inspect_cmd)
    app.command("tools")(tools)
    app.command("ask")(ask)
    app.command("info")(info)
    app.command("gate")(gate)
    app.command("memory-stats")(memory_stats)  # kept for backward compat

    # Sub-apps (command groups)
    app.add_typer(agents_app, name="agents")
    app.add_typer(models_app, name="models")
    app.add_typer(cost_app, name="cost")
    app.add_typer(connect_app, name="connect")
    app.add_typer(memory_app, name="memory")
    app.add_typer(sandbox_app, name="sandbox")
    app.add_typer(
        rl_app, name="rl", help="🎯 RL-lite adaptive model performance stats and controls."
    )
    app.add_typer(meta_app, name="meta", help="🔬 Meta-agent: system self-improvement tools.")
    app.add_typer(runs_app, name="runs", help="📜 Browse and search past pipeline runs.")
    app.add_typer(batch_app, name="batch", help="⚡ Batch pipeline orchestration.")


_register_commands()
