#!/usr/bin/env python3
"""
Local smoke-test for the dynamic pipeline.

Mimics agent-arch/main.py flow: builds the 8-node graph, sends a simple query,
prints every node's output as it streams through, then enters a post-pipeline
chat loop where the master agent retains full conversation context.

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
from datetime import UTC, datetime
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


QUIT_WORDS = {"quit", "exit", "done", "bye", "q"}


async def _async_input(prompt: str) -> str:
    """Read a line from stdin without blocking the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, input, prompt)


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

    rprint("\n[bold cyan]═══ AI Factory Dynamic Pipeline — Local Test ═══[/bold cyan]\n")
    rprint(f"[bold]Query:[/bold] {query}\n")

    # Import after dotenv so env vars are available
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    from aifactory.graphs.dynamic_pipeline import (
        build_dynamic_pipeline,
        load_graph_config,
        write_dump,
    )

    # ── Build the graph ──────────────────────────────────────────────
    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    checkpointer = MemorySaver()
    t0 = time.time()
    rprint("[dim]Building pipeline...[/dim]")
    compiled, registry = build_dynamic_pipeline(checkpointer=checkpointer, run_id=run_id)
    build_time = time.time() - t0

    rprint(f"[green]✓[/green] Pipeline built in {build_time:.1f}s")
    rprint(f"[green]✓[/green] Agents: {', '.join(a.name for a in registry.all())}")
    rprint(f"[green]✓[/green] Models: {', '.join(set(a.model for a in registry.all()))}")
    rprint()

    # ── Invoke the graph (with interrupt/resume support) ─────────────
    initial_state = {
        "user_query": query,
        "shared_context": {},
        "conversation_history": [],
    }
    thread_cfg = {"configurable": {"thread_id": run_id}}

    rprint("[bold yellow]Streaming pipeline execution...[/bold yellow]\n")
    t1 = time.time()

    # Use astream() to get node-level deltas — accumulate into full state
    final_state = dict(initial_state)
    current_input = initial_state

    while True:
        async for chunk in compiled.astream(
            current_input, config=thread_cfg, stream_mode="updates"
        ):
            for node_name, update in chunk.items():
                if node_name == "__start__":
                    continue
                rprint(f"  [cyan]▶ {node_name}[/cyan] done")
                if isinstance(update, dict):
                    final_state.update(update)

        # Check for interrupt (worker QUESTION)
        snap = compiled.get_state(thread_cfg)
        if not snap.next:
            break  # pipeline complete

        interrupt_msg = ""
        for task in snap.tasks:
            for intr in task.interrupts:
                interrupt_msg = str(intr.value)
                break
            if interrupt_msg:
                break

        rprint()
        if rich_available:
            console.print(
                Panel(interrupt_msg, title="Worker Question", border_style="yellow", expand=False)
            )
        else:
            print(f"\n--- Worker Question ---\n{interrupt_msg}\n")

        answer = (await _async_input("[You] > ")).strip()
        rprint()
        if not answer or answer.lower() in QUIT_WORDS:
            rprint("[dim]Exiting.[/dim]")
            exec_time = time.time() - t1
            cfg = load_graph_config()
            write_dump(run_id, cfg, final_state, registry, exec_time, query)
            return

        current_input = Command(resume=answer)

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
    rprint("\n[bold]Agent Summary:[/bold]")
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
        table.add_row("[bold]TOTAL[/bold]", "", "", "", f"[bold]{total_tokens}[/bold]")
        console.print(table)
    else:
        for a in registry.all():
            print(f"  {a.name}: {a.total_tokens} tokens ({a.model})")

    # Judge metrics
    judge_metrics = final_state.get("judge_metrics", {})
    if judge_metrics:
        rprint("\n[bold]Judge Metrics:[/bold]")
        for k, v in judge_metrics.items():
            rprint(f"  {k}: {v}")

    rprint(f"\n[dim]Total time: {exec_time:.1f}s | Build: {build_time:.1f}s[/dim]\n")

    # ── Write run dump ───────────────────────────────────────────────
    cfg = load_graph_config()
    dump_path = write_dump(run_id, cfg, final_state, registry, exec_time, query)
    rprint(f"[dim]Run dump: {dump_path}[/dim]\n")

    # ── Post-pipeline chat loop ──────────────────────────────────────
    # Master agent retains full private context (_messages). We continue the
    # conversation by calling master.think() directly — no graph re-execution.
    master = registry.get("master")
    shared_ctx = final_state.get("shared_context", {})
    history: list[dict] = list(final_state.get("conversation_history", []))

    rprint("[dim]Chat with Master (type quit/exit/done to end):[/dim]")
    while master is not None:
        rprint()
        user_input = (await _async_input("[You] > ")).strip()
        rprint()

        if not user_input or user_input.lower() in QUIT_WORDS:
            break

        history.append({"role": "user", "content": user_input})
        reply = await master.think(
            f'The human replied: "{user_input}"\n\nRespond helpfully.',
            shared_ctx,
        )
        history.append({"role": "assistant", "content": reply})

        if rich_available:
            console.print(Panel(reply, title="Master", border_style="green", expand=False))
        else:
            print(f"Master: {reply}")

    rprint("[dim]Session complete.[/dim]\n")


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is 2 + 2?"
    asyncio.run(main(query))
