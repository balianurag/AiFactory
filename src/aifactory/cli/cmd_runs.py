"""
CLI commands for browsing past pipeline runs.

  aifactory runs list [--last N]           Show recent runs
  aifactory runs show <run-id>             Full details for one run
  aifactory runs search <query>            Semantic similarity search
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

runs_app = typer.Typer(name="runs", help="📜 Browse and search past pipeline runs.")
console = Console()


@runs_app.command("list")
def list_runs(
    last: Annotated[int, typer.Option("--last", help="Number of recent runs to show")] = 20,
) -> None:
    """Show the N most recent pipeline runs."""
    from aifactory.core.run_store import get_run_store

    store = get_run_store()
    records = store.list_recent(n=last)

    if not records:
        console.print("[yellow]No run history found.[/yellow]")
        return

    table = Table(title=f"Last {len(records)} runs", show_lines=False)
    table.add_column("Run ID", style="dim", width=18)
    table.add_column("Query", width=45)
    table.add_column("Mode", width=12)
    table.add_column("Quality", width=8)
    table.add_column("Tasks", width=6)
    table.add_column("Flagged", width=7)
    table.add_column("Gate", width=10)

    for r in records:
        quality_style = (
            "green"
            if r.composite_quality >= 7
            else ("yellow" if r.composite_quality >= 4 else "red")
        )
        table.add_row(
            r.run_id[:16],
            r.query[:43] + ("…" if len(r.query) > 43 else ""),
            r.mode,
            f"[{quality_style}]{r.composite_quality:.1f}[/{quality_style}]",
            str(r.task_count),
            f"[red]{r.quality_flagged_count}[/red]" if r.quality_flagged_count else "0",
            r.gate or "full",
        )

    console.print(table)


@runs_app.command("show")
def show_run(
    run_id: Annotated[str, typer.Argument(help="Run ID to inspect")],
) -> None:
    """Show full details for a specific run."""
    from aifactory.core.run_store import get_run_store

    store = get_run_store()
    record = store.get(run_id)

    if record is None:
        console.print(f"[red]Run '{run_id}' not found.[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Run:[/bold] {record.run_id}")
    console.print(f"[bold]Time:[/bold] {record.timestamp}")
    console.print(f"[bold]Query:[/bold] {record.query}")
    console.print(
        f"[bold]Mode:[/bold] {record.mode}  |  Gate: {record.gate or 'full'}  |  Elapsed: {record.elapsed_seconds:.1f}s"  # noqa: E501
    )
    console.print(
        f"[bold]Quality:[/bold] {record.composite_quality:.1f}/10  |  Tasks: {record.task_count}  |  Flagged: {record.quality_flagged_count}"  # noqa: E501
    )

    if record.model_selections:
        console.print("\n[bold]Model selections:[/bold]")
        for role, model in record.model_selections.items():
            console.print(f"  {role}: {model}")

    if record.final_response_preview:
        console.print(f"\n[bold]Response preview:[/bold]\n{record.final_response_preview[:400]}")


@runs_app.command("search")
def search_runs(
    query: Annotated[str, typer.Argument(help="Query text to search similar runs")],
    top_k: Annotated[int, typer.Option("--top-k", help="Number of results")] = 5,
) -> None:
    """Find runs similar to the given query."""
    from aifactory.core.run_store import get_run_store

    store = get_run_store()
    results = store.search_similar(query, top_k=top_k)

    if not results:
        console.print("[yellow]No similar runs found.[/yellow]")
        return

    console.print(f"\n[bold]Runs similar to:[/bold] {query}\n")
    for i, r in enumerate(results, 1):
        console.print(f"[bold cyan]{i}.[/bold cyan] [{r.run_id[:16]}] {r.query[:80]}")
        console.print(
            f"   Quality: {r.composite_quality:.1f}  Mode: {r.mode}  Gate: {r.gate or 'full'}"
        )
        if r.final_response_preview:
            console.print(f"   [dim]{r.final_response_preview[:120]}[/dim]")
        console.print()
