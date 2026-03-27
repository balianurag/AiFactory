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
        from aifactory.graphs.single_team import SingleTeamPipeline

        pipeline = SingleTeamPipeline(num_workers=workers)
        delivery = asyncio.run(pipeline.run(task=task_text, context=context))

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        raise typer.Exit(130) from None
    except Exception as e:
        console.print(f"\n[red]Pipeline failed: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1) from e

    elapsed = time.perf_counter() - t0

    # ── Display Results ───────────────────────────────────────────────
    console.print()
    console.print(
        Panel(
            f"[bold green]{delivery.project_title}[/bold green]",
            title="✅ Project Delivered",
            border_style="green",
        )
    )

    # Summary
    console.print(f"\n[bold]Summary:[/bold] {delivery.summary}\n")

    # Quality
    if delivery.quality_assessment:
        console.print(f"[bold]Quality:[/bold] {delivery.quality_assessment}\n")

    # Deliverables
    if delivery.deliverables:
        console.print("[bold]📦 Deliverables:[/bold]")
        for artifact in delivery.deliverables:
            console.print(f"\n  [cyan]{artifact.name}[/cyan] ({artifact.content_type})")
            if artifact.summary:
                console.print(f"  {artifact.summary}")
            console.print(
                Panel(
                    artifact.content[:2000] + ("..." if len(artifact.content) > 2000 else ""),
                    border_style="dim",
                    expand=False,
                )
            )

    # Open Issues
    if delivery.open_issues:
        console.print("[bold yellow]⚠️  Open Issues:[/bold yellow]")
        for issue in delivery.open_issues:
            console.print(f"  • {issue}")
        console.print()

    # Recommendations
    if delivery.recommendations:
        console.print("[bold]💡 Recommendations:[/bold]")
        for rec in delivery.recommendations:
            console.print(f"  • {rec}")
        console.print()

    # Metrics
    metrics_table = Table(title="📊 Metrics", show_header=True, border_style="dim")
    metrics_table.add_column("Metric", style="bold")
    metrics_table.add_column("Value", justify="right")

    metrics = delivery.metrics
    metrics_table.add_row("Total Iterations", str(metrics.get("total_iterations", "?")))
    metrics_table.add_row("Total Tokens", f"{metrics.get('total_tokens', 0):,}")
    metrics_table.add_row("Teams Used", str(metrics.get("teams_used", "?")))
    metrics_table.add_row("Agents Used", str(metrics.get("agents_used", "?")))
    metrics_table.add_row("Wall Time", f"{elapsed:.1f}s")

    console.print(metrics_table)
    console.print()


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


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    app()
