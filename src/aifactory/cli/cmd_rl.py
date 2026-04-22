"""aifactory rl — RL-lite adaptive model performance commands."""

from __future__ import annotations

import typer
from rich.table import Table

from aifactory.cli._helpers import console

rl_app = typer.Typer(
    name="RL",
    help="🎯 RL-lite adaptive model performance stats and controls.",
    no_args_is_help=True,
)


@rl_app.command("stats")
def rl_stats(
    task_type: str = typer.Option(
        "",
        "--type",
        "-t",
        help="Filter by task type (coding, math, research, writing, analysis, tool_use)",
    ),
    min_samples: int = typer.Option(1, "--min", "-m", help="Minimum records to show a model"),
) -> None:
    """Show historical model performance scores by task type."""
    from aifactory.core.rl_scorer import get_all_stats

    stats = get_all_stats()
    total = stats.get("total_records", 0)

    if total == 0:
        console.print("[yellow]No RL records yet — run some tasks first to build history.[/yellow]")
        raise typer.Exit()

    console.print(f"\n[bold cyan]RL-lite Performance Records[/bold cyan]  (total: {total})\n")

    by_type = stats.get("by_task_type", {})

    # Filter by task type if requested
    if task_type:
        by_type = {k: v for k, v in by_type.items() if k == task_type}
        if not by_type:
            console.print(f"[yellow]No records for task type: {task_type}[/yellow]")
            raise typer.Exit()

    # Summary table
    summary_table = Table(show_header=True, header_style="bold cyan", box=None)
    summary_table.add_column("Task Type", style="bold", width=12)
    summary_table.add_column("Records", justify="right", width=8)
    summary_table.add_column("Models", justify="right", width=7)
    summary_table.add_column("Best Model", width=36)
    summary_table.add_column("Best Score", justify="right", width=10)

    for tt, info in sorted(by_type.items()):
        summary_table.add_row(
            tt,
            str(info["records"]),
            str(info["models"]),
            info["best_model"],
            f"{info['best_avg_score']:.1f}/10",
        )

    console.print(summary_table)

    # Per-model breakdown if a single type is selected
    if task_type and task_type in by_type:
        console.print(f"\n[bold]Model breakdown for[/bold] [cyan]{task_type}[/cyan]:\n")
        model_avgs = by_type[task_type].get("model_avgs", {})

        detail_table = Table(show_header=True, header_style="bold cyan", box=None)
        detail_table.add_column("Model", width=40)
        detail_table.add_column("Avg Score", justify="right", width=10)
        detail_table.add_column("Bar", width=20)

        for mid, avg in sorted(model_avgs.items(), key=lambda x: -x[1]):
            bar_len = int(avg * 2)  # 0-20 chars for 0-10 score
            bar = "[green]" + "█" * bar_len + "[/green]" + "░" * (20 - bar_len)
            detail_table.add_row(mid, f"{avg:.1f}/10", bar)

        console.print(detail_table)

    console.print()


@rl_app.command("reset")
def rl_reset(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
    task_type: str = typer.Option("", "--type", "-t", help="Reset only a specific task type"),
) -> None:
    """Clear RL performance records (useful for testing or fresh starts)."""
    if not yes:
        scope = f"task type '{task_type}'" if task_type else "ALL RL records"
        confirm = typer.confirm(f"Delete {scope}? This cannot be undone.")
        if not confirm:
            console.print("[yellow]Aborted.[/yellow]")
            raise typer.Exit()

    try:
        from aifactory.core.database import delete_rl_records

        deleted = delete_rl_records(task_type if task_type else None)
        if task_type:
            console.print(f"[green]Cleared {deleted} RL records for task type: {task_type}[/green]")
        else:
            console.print(f"[green]All RL records cleared ({deleted} rows).[/green]")
    except Exception as exc:
        console.print(f"[red]Reset failed: {exc}[/red]")
        raise typer.Exit(1) from exc
