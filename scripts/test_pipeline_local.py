#!/usr/bin/env python3
"""
Local smoke-test for the dynamic pipeline.

Mimics agent-arch/main.py flow: builds the 8-node graph, sends a simple query,
and prints every node's output as it streams through.

Usage:
    cd AiFactory
    uv run python scripts/test_pipeline_local.py
    # or with a custom query:
    uv run python scripts/test_pipeline_local.py "Explain the Fibonacci sequence in Python"
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path

# Load .env before anything else
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Rich for pretty output (optional — falls back to print if missing)
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    rich_available = True
except ImportError:
    console = None
    rich_available = False


def rprint(*args, **kwargs):
    if rich_available:
        console.print(*args, **kwargs)
    else:
        print(*args)


async def main(query: str) -> None:
    # Configure logging so we see agent-level info
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quieten noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    rprint(f"\n[bold cyan]═══ AI Factory Dynamic Pipeline — Local Test ═══[/bold cyan]\n")
    rprint(f"[bold]Query:[/bold] {query}\n")

    # Import after dotenv so env vars are available
    from aifactory.graphs.dynamic_pipeline import build_dynamic_pipeline

    # ── Build the graph ──────────────────────────────────────────────
    t0 = time.time()
    rprint("[dim]Building pipeline...[/dim]")
    compiled, registry = build_dynamic_pipeline()  # mode from graph.yaml
    build_time = time.time() - t0

    rprint(f"[green]✓[/green] Pipeline built in {build_time:.1f}s")
    rprint(f"[green]✓[/green] Agents: {', '.join(a.name for a in registry.all())}")
    rprint(f"[green]✓[/green] Models: {', '.join(set(a.model for a in registry.all()))}")
    rprint()

    # ── Invoke the graph ─────────────────────────────────────────────
    initial_state = {
        "user_query": query,
        "shared_context": {},
        "conversation_history": [],
    }
    config = {"configurable": {"thread_id": "local-test"}}

    rprint("[bold yellow]Streaming pipeline execution...[/bold yellow]\n")
    t1 = time.time()

    # Use astream() to get node-level deltas — accumulate into full state
    final_state = dict(initial_state)

    async for chunk in compiled.astream(initial_state, config=config, stream_mode="updates"):
        # chunk is {node_name: state_update_dict}
        for node_name, update in chunk.items():
            if node_name == "__start__":
                continue
            rprint(f"  [cyan]▶ {node_name}[/cyan] done")
            if isinstance(update, dict):
                final_state.update(update)

    exec_time = time.time() - t1

    # ── Print results ────────────────────────────────────────────────
    rprint(f"\n[bold green]═══ Pipeline Complete ({exec_time:.1f}s) ═══[/bold green]\n")

    # Final response
    final_response = final_state.get("final_response", "")
    direct_answer = final_state.get("direct_answer", "")
    display = final_response or direct_answer or ""
    if rich_available:
        console.print(
            Panel(
                display if display.strip() else "[dim]No response[/dim]",
                title="[bold]Final Response[/bold]",
                border_style="green",
            )
        )
    else:
        print(f"\n--- Final Response ---\n{final_response or direct_answer}\n")

    # Tasks
    tasks = final_state.get("tasks", [])
    if tasks:
        rprint(f"\n[bold]Tasks ({len(tasks)}):[/bold]")
        for t in tasks:
            rprint(f"  • {t.get('id', '?')}: {t.get('description', '?')}")

    # Task results
    task_results = final_state.get("task_results", [])
    if task_results:
        rprint(f"\n[bold]Task Results ({len(task_results)}):[/bold]")
        for r in task_results:
            skipped = r.get("skipped", False)
            rprint(
                f"  • {r.get('task_id', '?')} [{r.get('model', '?')}]"
                f"{' (skipped)' if skipped else ''}: "
                f"{r.get('result', '')[:200]}..."
                if len(r.get("result", "")) > 200
                else f"  • {r.get('task_id', '?')} [{r.get('model', '?')}]: {r.get('result', '')}"
            )

    # Agent token summary
    rprint(f"\n[bold]Agent Summary:[/bold]")
    if rich_available:
        table = Table(show_header=True, header_style="bold")
        table.add_column("Agent")
        table.add_column("Model")
        table.add_column("Prompt", justify="right")
        table.add_column("Completion", justify="right")
        table.add_column("Total", justify="right")
        for a in registry.all():
            table.add_row(
                a.name,
                a.model,
                str(a.prompt_tokens),
                str(a.completion_tokens),
                str(a.total_tokens),
            )
        total_tokens = sum(a.total_tokens for a in registry.all())
        table.add_row(
            "[bold]TOTAL[/bold]", "", "", "", f"[bold]{total_tokens}[/bold]"
        )
        console.print(table)
    else:
        for a in registry.all():
            print(f"  {a.name}: {a.total_tokens} tokens ({a.model})")

    # Judge metrics
    judge_metrics = final_state.get("judge_metrics", {})
    if judge_metrics:
        rprint(f"\n[bold]Judge Metrics:[/bold]")
        for k, v in judge_metrics.items():
            rprint(f"  {k}: {v}")

    rprint(f"\n[dim]Total time: {exec_time:.1f}s | Build: {build_time:.1f}s[/dim]\n")


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is 2 + 2?"
    asyncio.run(main(query))
