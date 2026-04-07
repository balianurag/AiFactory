"""
AI Factory CLI — Run the agent factory from the command line.

Usage:
    aifactory run "Build a FastAPI CRUD app for a todo list"
    aifactory run --task-file task.md
    aifactory status
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="aifactory",
    help="🏭 AI Factory — Self-assembling agent system that turns ideas into production output.",
    no_args_is_help=True,
)

console = Console()


def _setup_logging(level: str = "INFO") -> None:
    """Configure rich logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True)],
    )


# =============================================================================
# Commands
# =============================================================================


@app.command()
def run(
    task: str = typer.Argument(None, help="The task/idea to execute"),
    task_file: Path = typer.Option(  # noqa: B008
        None,
        "--task-file",
        "-f",
        help="Read task from a file",
    ),
    context: str = typer.Option(
        "",
        "--context",
        "-c",
        help="Additional context",
    ),
    workers: int = typer.Option(
        3,
        "--workers",
        "-w",
        help="Number of workers per team",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Log level",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed agent logs",
    ),
) -> None:
    """🚀 Run the AI Factory on a task or idea."""
    _setup_logging("DEBUG" if verbose else log_level)

    # Resolve task input
    if task_file and task_file.exists():
        task_text = task_file.read_text().strip()
    elif task:
        task_text = task
    else:
        console.print("[red]Error: Provide a task as an argument or via --task-file[/red]")
        raise typer.Exit(1)

    # Display the task
    console.print()
    console.print(
        Panel(
            task_text,
            title="🏭 AI Factory — Task",
            border_style="blue",
            expand=False,
        )
    )
    console.print()

    # Run the pipeline
    t0 = time.perf_counter()

    try:
        from datetime import UTC, datetime

        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.types import Command

        from aifactory.graphs.dynamic_pipeline import (
            build_dynamic_pipeline,
            load_graph_config,
            write_dump,
        )

        run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        checkpointer = MemorySaver()
        compiled, registry = build_dynamic_pipeline(checkpointer=checkpointer, run_id=run_id)

        initial_state = {
            "user_query": task_text,
            "shared_context": {},
            "conversation_history": [],
        }
        thread_cfg = {"configurable": {"thread_id": run_id}}

        # Run pipeline with interrupt/resume support
        result: dict = {}
        current_input = initial_state
        while True:
            result = asyncio.run(compiled.ainvoke(current_input, config=thread_cfg))
            snap = compiled.get_state(thread_cfg)
            if not snap.next:
                break

            # Surface worker question
            interrupt_msg = ""
            for snap_task in snap.tasks:
                for intr in snap_task.interrupts:
                    interrupt_msg = str(intr.value)
                    break
                if interrupt_msg:
                    break

            console.print(
                Panel(interrupt_msg, title="Worker Question", border_style="yellow", expand=False)
            )
            answer = input("[You] > ").strip()
            if not answer or answer.lower() in {"quit", "exit", "done", "bye", "q"}:
                console.print("[dim]Exiting.[/dim]")
                elapsed = time.perf_counter() - t0
                cfg = load_graph_config()
                write_dump(run_id, cfg, result, registry, elapsed, task_text)
                raise typer.Exit(0) from None
            current_input = Command(resume=answer)

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        raise typer.Exit(130) from None
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"\n[red]Pipeline failed: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1) from e

    elapsed = time.perf_counter() - t0

    # ── Write dump ────────────────────────────────────────────────────
    cfg = load_graph_config()
    dump_path = write_dump(run_id, cfg, result, registry, elapsed, task_text)
    console.print(f"[dim]Run dump: {dump_path}[/dim]")

    # ── Display Results ───────────────────────────────────────────────
    console.print()
    final_response = result.get("final_response", "")
    if final_response.strip():
        console.print(
            Panel(
                final_response,
                title="✅ Result",
                border_style="green",
            )
        )
    else:
        console.print("[yellow]⚠ No final response generated.[/yellow]")

    # Tasks
    task_results = result.get("task_results", [])
    if task_results:
        console.print(f"\n[bold]Tasks ({len(task_results)}):[/bold]")
        for r in task_results:
            skipped = r.get("skipped", False)
            console.print(
                f"  • {r.get('task_id', '?')} [{r.get('model', '?')}]"
                f"{' (skipped)' if skipped else ''}: "
                f"{r.get('result', '')[:200]}{'...' if len(r.get('result', '')) > 200 else ''}"
            )

    # Metrics
    metrics_table = Table(title="📊 Metrics", show_header=True, border_style="dim")
    metrics_table.add_column("Metric", style="bold")
    metrics_table.add_column("Value", justify="right")

    total_tokens = sum(a.total_tokens for a in registry.all())
    metrics_table.add_row("Agents", str(len(registry.all())))
    metrics_table.add_row("Tasks Executed", str(len(task_results)))
    metrics_table.add_row("Total Tokens", f"{total_tokens:,}")
    metrics_table.add_row("Wall Time", f"{elapsed:.1f}s")

    judge_metrics = result.get("judge_metrics", {})
    for k, v in judge_metrics.items():
        metrics_table.add_row(k.replace("_", " ").title(), str(v))

    console.print(metrics_table)
    console.print()

    # ── Post-pipeline chat ───────────────────────────────────────────
    master = registry.get("master")
    shared_ctx = result.get("shared_context", {})
    history: list = list(result.get("conversation_history", []))

    console.print("[dim]Chat with Master (type quit/exit/done to end):[/dim]")
    while master is not None:
        try:
            user_input = input("\n[You] > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input or user_input.lower() in {"quit", "exit", "done", "bye", "q"}:
            break

        history.append({"role": "user", "content": user_input})
        reply = asyncio.run(
            master.think(
                f'The human replied: "{user_input}"\n\nRespond helpfully.',
                shared_ctx,
            )
        )
        history.append({"role": "assistant", "content": reply})
        console.print(Panel(reply, title="Master", border_style="green", expand=False))

    console.print("[dim]Session complete.[/dim]\n")


@app.command()
def info() -> None:
    """📋 Show AI Factory configuration and status."""
    from aifactory.config.settings import get_settings

    settings = get_settings()

    table = Table(title="🏭 AI Factory Configuration", show_header=True)
    table.add_column("Setting", style="bold")
    table.add_column("Value")

    table.add_row("Premium Model", settings.premium_model)
    table.add_row("Standard Model", settings.standard_model)
    table.add_row("Budget Model", settings.budget_model)
    table.add_row("Max Budget (USD)", f"${settings.max_budget_usd:.2f}")
    table.add_row("Max Iterations", str(settings.max_iterations))
    table.add_row("Max Subordinates/Lead", str(settings.max_subordinates_per_lead))
    table.add_row("Qdrant URL", settings.qdrant_url)
    table.add_row("Memory Promotion Interval", str(settings.memory_promotion_interval))
    table.add_row("RL Min Score for Promotion", str(settings.rl_min_score_for_promotion))

    # Check API keys
    keys = {
        "OpenAI": bool(settings.openai_api_key),
        "Anthropic": bool(settings.anthropic_api_key),
        "xAI": bool(settings.xai_api_key),
        "Google": bool(settings.google_api_key),
    }
    key_status = ", ".join(
        f"[green]{k}✓[/green]" if v else f"[red]{k}✗[/red]" for k, v in keys.items()
    )
    table.add_row("API Keys", key_status)

    console.print()
    console.print(table)
    console.print()


@app.command()
def memory_stats() -> None:
    """📊 Show memory system statistics."""
    from aifactory.core.memory_router import MemoryRouter

    try:
        router = MemoryRouter()
        stats = router.get_stats()

        table = Table(title="🧠 Memory Stats")
        table.add_column("Layer", style="bold")
        table.add_column("Entries", justify="right")

        for level, count in stats.items():
            table.add_row(level.title(), str(count))

        console.print()
        console.print(table)
        console.print()
    except Exception as e:
        console.print(f"[red]Could not connect to memory system: {e}[/red]")


# ── Sub-apps ──────────────────────────────────────────────────────────────────

models_app = typer.Typer(help="Model registry commands.", no_args_is_help=True)
app.add_typer(models_app, name="models")

agents_app = typer.Typer(help="Agent management commands.", no_args_is_help=True)
app.add_typer(agents_app, name="agents")


@models_app.command("list")
def models_list(
    role: str = typer.Option(
        None,
        "--role",
        "-r",
        help="Filter by agent role (e.g. coder, project_manager, team_lead)",
    ),
) -> None:
    """List the curated allowed models, optionally filtered by agent role."""
    from pathlib import Path

    import yaml

    yaml_path = Path(__file__).parent / "config" / "models.yaml"
    registry = yaml.safe_load(yaml_path.read_text())
    allowed = registry.get("allowed_models", {})
    metadata = registry.get("metadata", {})

    # Filter to requested role(s)
    if role:
        role_key = role.lower().replace("-", "_").replace(" ", "_")
        if role_key not in allowed:
            console.print(f"[red]Unknown role '{role}'. Available roles:[/red]")
            for r in allowed:
                console.print(f"  • {r}")
            raise typer.Exit(1)
        roles_to_show = {role_key: allowed[role_key]}
    else:
        roles_to_show = allowed

    console.print()
    for role_name, role_data in roles_to_show.items():
        models_list_data = role_data.get("models", []) if isinstance(role_data, dict) else role_data
        desc = role_data.get("description", "") if isinstance(role_data, dict) else ""

        table = Table(
            title=f"[bold cyan]{role_name.upper()}[/bold cyan]  [dim]{desc}[/dim]",
            show_header=True,
            border_style="dim",
            show_lines=False,
        )
        table.add_column("Model ID", style="dim", min_width=20)
        table.add_column("Display Name", style="bold")
        table.add_column("Provider", style="cyan")
        table.add_column("Context", justify="right")
        table.add_column("$/1M in", justify="right", style="green")
        table.add_column("$/1M out", justify="right", style="yellow")
        table.add_column("Strength", style="dim", max_width=40)

        for model_id in models_list_data:
            meta = metadata.get(model_id, {})
            display = meta.get("display", model_id.split("/")[-1])
            provider = meta.get("provider", "?")
            ctx = f"{meta.get('context_k', '?')}k"
            inp = f"${meta.get('input_per_1m', '?'):.3f}" if meta.get("input_per_1m") else "?"
            out = f"${meta.get('output_per_1m', '?'):.2f}" if meta.get("output_per_1m") else "?"
            strength = meta.get("strength", "")
            short_id = model_id.replace("openrouter/", "")
            table.add_row(short_id, display, provider, ctx, inp, out, strength)

        console.print(table)
        console.print()


@models_app.command("tiers")
def models_tiers() -> None:
    """Show the default model tier assignments (premium / standard / budget)."""
    from aifactory.config.settings import get_settings

    settings = get_settings()

    table = Table(title="Model Tier Assignments", show_header=True, border_style="dim")
    table.add_column("Tier", style="bold")
    table.add_column("Active Model")
    table.add_column("Used For")

    table.add_row(
        "[magenta]premium[/magenta]",
        settings.premium_model,
        "Master Researcher, PM, Assessors, Sr. PM",
    )
    table.add_row(
        "[cyan]standard[/cyan]",
        settings.standard_model,
        "Team Leads",
    )
    table.add_row(
        "[green]budget[/green]",
        settings.budget_model,
        "Workers (Coder, Researcher, Writer, Tester, DevOps, ...)",
    )

    console.print()
    console.print(table)
    console.print(
        "[dim]Override via env vars: AIFACTORY_PREMIUM_MODEL, "
        "AIFACTORY_STANDARD_MODEL, AIFACTORY_BUDGET_MODEL[/dim]"
    )
    console.print()


@agents_app.command("list")
def agents_list() -> None:
    """List all agent roles and their default model assignments."""
    from pathlib import Path

    import yaml

    yaml_path = Path(__file__).parent / "config" / "models.yaml"
    registry = yaml.safe_load(yaml_path.read_text())
    role_tier_map = registry.get("role_tier_map", {})
    allowed = registry.get("allowed_models", {})
    metadata = registry.get("metadata", {})

    from aifactory.config.settings import get_settings

    settings = get_settings()

    tier_defaults = {
        "premium": settings.premium_model,
        "standard": settings.standard_model,
        "budget": settings.budget_model,
    }

    table = Table(
        title="Agent Roles & Model Assignments",
        show_header=True,
        border_style="dim",
    )
    table.add_column("Role", style="bold", min_width=24)
    table.add_column("Tier", style="cyan")
    table.add_column("Default Model")
    table.add_column("Allowed Models", style="dim")

    for role_name, tier in role_tier_map.items():
        default_model_id = tier_defaults.get(tier, "?")
        fallback_name = default_model_id.split("/")[-1]
        default_display = metadata.get(default_model_id, {}).get("display", fallback_name)

        role_models = []
        if role_name in allowed:
            role_data = allowed[role_name]
            models = role_data.get("models", []) if isinstance(role_data, dict) else role_data
            role_models = [metadata.get(m, {}).get("display", m.split("/")[-1]) for m in models]
        allowed_str = ", ".join(role_models) if role_models else "[dim]tier default[/dim]"

        tier_colors = {"premium": "magenta", "standard": "cyan", "budget": "green"}
        tier_styled = f"[{tier_colors.get(tier, 'white')}]{tier}[/{tier_colors.get(tier, 'white')}]"

        table.add_row(role_name, tier_styled, default_display, allowed_str)

    console.print()
    console.print(table)
    console.print(
        "\n[dim]Agents are assembled dynamically at runtime. "
        "Use [bold]aifactory run[/bold] to spawn a team.[/dim]"
    )
    console.print()


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    app()
