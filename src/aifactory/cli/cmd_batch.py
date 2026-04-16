"""
CLI commands for batch pipeline orchestration.

  aifactory batch run tasks.jsonl [--concurrency N] [--mode balanced] [--output out.jsonl]
  aifactory batch status <batch-id>

Input JSONL format (one JSON object per line):
  {"query": "Build a FastAPI app", "context": "Use PostgreSQL", "id": "optional-id"}

Output JSONL format (written to --output or <input>.results.jsonl):
  {"id": "...", "query": "...", "status": "completed|failed", "result": ..., "elapsed_s": 1.2}
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

batch_app = typer.Typer(name="batch", help="⚡ Batch pipeline orchestration.")
console = Console()


async def _run_single(
    item: dict[str, Any],
    mode: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """Run one pipeline task under the semaphore concurrency limit."""
    async with semaphore:
        from aifactory.core.complexity_gate import evaluate_gate
        from aifactory.graphs.dynamic_pipeline import build_dynamic_pipeline, write_dump

        query = item.get("query", "")
        context = item.get("context", "")
        item_id = item.get("id") or str(uuid.uuid4())[:8]
        run_id = str(uuid.uuid4())

        t0 = time.perf_counter()
        try:
            # Complexity gate — may short-circuit without full pipeline
            gate = evaluate_gate(query)
            if gate.bypass:
                elapsed = time.perf_counter() - t0
                return {
                    "id": item_id,
                    "query": query,
                    "status": "completed",
                    "gate": gate.method,
                    "result": gate.answer or "",
                    "elapsed_s": round(elapsed, 2),
                }

            compiled, registry = build_dynamic_pipeline(mode=mode, run_id=run_id)
            initial_state = {
                "user_query": query,
                "shared_context": {"context": context},
                "conversation_history": [],
            }
            config = {"configurable": {"thread_id": run_id}}
            final_state = await compiled.ainvoke(initial_state, config=config)

            elapsed = time.perf_counter() - t0

            # Persist run dump
            try:
                from aifactory.core.model_selector import load_graph_config

                cfg = load_graph_config()
                write_dump(run_id, cfg, final_state, registry, elapsed, query)
            except Exception:
                pass

            return {
                "id": item_id,
                "query": query,
                "status": "completed",
                "gate": "full",
                "result": final_state.get("final_response", ""),
                "task_results": final_state.get("task_results", []),
                "judge_metrics": final_state.get("judge_metrics", {}),
                "elapsed_s": round(elapsed, 2),
            }

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            return {
                "id": item_id,
                "query": query,
                "status": "failed",
                "error": str(exc),
                "elapsed_s": round(elapsed, 2),
            }


@batch_app.command("run")
def batch_run(
    tasks_file: Annotated[Path, typer.Argument(help="JSONL file with tasks (one per line)")],
    concurrency: Annotated[
        int, typer.Option("--concurrency", "-n", help="Max parallel pipelines")
    ] = 3,
    mode: Annotated[
        str, typer.Option("--mode", "-m", help="Pipeline mode: best | balanced | cost_efficient")
    ] = "balanced",
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Output JSONL file")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Parse tasks but don't run")] = False,
) -> None:
    """Run multiple pipeline tasks in parallel from a JSONL file."""
    if not tasks_file.exists():
        console.print(f"[red]File not found: {tasks_file}[/red]")
        raise typer.Exit(1)

    # Parse JSONL
    items: list[dict[str, Any]] = []
    with open(tasks_file) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
                if "query" not in obj:
                    console.print(
                        f"[yellow]Line {lineno}: missing 'query' field — skipped[/yellow]"
                    )
                    continue
                items.append(obj)
            except json.JSONDecodeError as e:
                console.print(f"[yellow]Line {lineno}: invalid JSON ({e}) — skipped[/yellow]")

    if not items:
        console.print("[yellow]No valid tasks found.[/yellow]")
        raise typer.Exit(0)

    console.print(
        f"[bold]Batch:[/bold] {len(items)} tasks | concurrency={concurrency} | mode={mode}"
    )

    if dry_run:
        console.print("[dim]Dry run — tasks parsed but not executed.[/dim]")
        for i, item in enumerate(items, 1):
            console.print(f"  {i}. {item.get('query', '')[:80]}")
        return

    out_path = output or tasks_file.with_suffix(".results.jsonl")

    async def _run_all() -> list[dict[str, Any]]:
        sem = asyncio.Semaphore(concurrency)
        coros = [_run_single(item, mode, sem) for item in items]
        results: list[dict[str, Any]] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Running batch...", total=len(items))

            async def _tracked(coro: Any) -> dict[str, Any]:
                result = await coro
                progress.advance(task)
                return result

            results = await asyncio.gather(*[_tracked(c) for c in coros])

        return list(results)

    all_results = asyncio.run(_run_all())

    # Write output
    with open(out_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, default=str) + "\n")

    # Summary table
    completed = [r for r in all_results if r["status"] == "completed"]
    failed = [r for r in all_results if r["status"] == "failed"]
    elapsed_vals = [r["elapsed_s"] for r in all_results]
    avg_elapsed = sum(elapsed_vals) / len(elapsed_vals) if elapsed_vals else 0

    table = Table(title="Batch Summary", show_lines=False)
    table.add_column("ID", style="dim", width=10)
    table.add_column("Query", width=50)
    table.add_column("Status", width=10)
    table.add_column("Gate", width=10)
    table.add_column("Elapsed", width=8)

    for r in all_results:
        status_style = "green" if r["status"] == "completed" else "red"
        table.add_row(
            r["id"],
            r["query"][:48] + ("…" if len(r["query"]) > 48 else ""),
            f"[{status_style}]{r['status']}[/{status_style}]",
            r.get("gate", "-"),
            f"{r['elapsed_s']:.1f}s",
        )

    console.print(table)
    console.print(
        f"\n[bold]Results:[/bold] {len(completed)} completed, {len(failed)} failed | "
        f"avg {avg_elapsed:.1f}s | output → {out_path}"
    )

    if failed:
        console.print("\n[bold red]Failures:[/bold red]")
        for r in failed:
            console.print(f"  [{r['id']}] {r['query'][:60]} — {r.get('error', 'unknown')}")
