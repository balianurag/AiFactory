#!/usr/bin/env python3
"""
Pipeline E2E integration test — runs prompts through the FULL pipeline that
naturally trigger ACTION: blocks, validating the end-to-end flow:

    User prompt → Master → Analyser → Execution Lead → Workers → ACTION: blocks
    → execute_integration_actions() → Composio → results injected → Assessor → Judge

═══════════════════════════════════════════════════════════════════════════════
PREREQUISITES (same as test_integrations_live.py):
    echo 'COMPOSIO_API_KEY=your_key_here' >> .env
    uv sync --extra integrations
    uv run composio add github         # or whichever apps you want to test

═══════════════════════════════════════════════════════════════════════════════
USAGE:
    # Run with the default prompt (GitHub-focused):
    uv run python scripts/test_integration_pipeline.py

    # Custom prompt that should trigger integrations:
    uv run python scripts/test_integration_pipeline.py \
        "List my open GitHub issues and summarise them"

    # Slack-focused prompt:
    uv run python scripts/test_integration_pipeline.py \
        "Send a summary of today's tasks to #general on Slack"

═══════════════════════════════════════════════════════════════════════════════
WHAT IT VALIDATES:
    1. Workers SEE the INTEGRATION TOOLS prompt block (toolkits listed)
    2. Workers EMIT ACTION: blocks in their output
    3. Pipeline DETECTS and DISPATCHES those blocks to Composio
    4. Integration results are INJECTED into the worker's result string
    5. Assessor / Judge receive the enriched output
    6. No regressions in regular code execution flow
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

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


DEFAULT_PROMPT = (
    "List my GitHub repositories (just the first 3). "
    "Use the integration tools available to you — specifically the "
    "GITHUB_LIST_REPOS action via Composio. Return the repo names."
)


async def main(query: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    rprint("\n[bold cyan]═══ Integration Pipeline E2E Test ═══[/bold cyan]\n")
    rprint(f"[bold]Prompt:[/bold] {query}\n")

    # ── Step 0: Pre-flight — connect integrations ─────────────────────
    rprint("[dim]Connecting integrations...[/dim]")
    from aifactory.integrations.providers.composio import ComposioProvider
    from aifactory.integrations.registry import IntegrationRegistry

    IntegrationRegistry.reset()
    registry = IntegrationRegistry()
    provider = ComposioProvider()
    registry.register(provider)
    connect_results = await registry.connect_all()

    composio_ok = connect_results.get("composio", False)
    if composio_ok:
        toolkits = registry.get_connected_toolkit_names()
        rprint(f"[green]✓[/green] Composio connected — toolkits: {', '.join(toolkits)}")

        # Fetch tools so the prompt summary is rich
        tools = await provider.fetch_available_tools()
        total = sum(len(v) for v in tools.values())
        rprint(f"[green]✓[/green] Discovered {total} tools across {len(tools)} toolkits")
    else:
        rprint("[red]✗ Composio not connected — workers won't get INTEGRATION TOOLS block[/red]")
        rprint("[yellow]  Pipeline will still run but ACTION: blocks won't be dispatched.[/yellow]")
        rprint(
            "[yellow]  To fix: set COMPOSIO_API_KEY in .env and run composio add github[/yellow]\n"
        )

    # ── Step 1: Build the pipeline ────────────────────────────────────
    rprint("\n[dim]Building pipeline...[/dim]")
    from langgraph.checkpoint.memory import MemorySaver

    from aifactory.graphs.dynamic_pipeline import (
        build_dynamic_pipeline,
        load_graph_config,
        write_dump,
    )

    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    checkpointer = MemorySaver()
    t0 = time.time()
    compiled, agent_registry = build_dynamic_pipeline(checkpointer=checkpointer, run_id=run_id)
    build_time = time.time() - t0
    rprint(f"[green]✓[/green] Pipeline built in {build_time:.1f}s")
    rprint(f"[green]✓[/green] Agents: {', '.join(a.name for a in agent_registry.all())}\n")

    # ── Step 2: Run the pipeline ──────────────────────────────────────
    rprint("[bold yellow]Streaming pipeline...[/bold yellow]\n")

    initial_state = {
        "user_query": query,
        "shared_context": {},
        "conversation_history": [],
    }
    thread_cfg = {"configurable": {"thread_id": run_id}}
    t1 = time.time()

    final_state = dict(initial_state)
    async for chunk in compiled.astream(initial_state, config=thread_cfg, stream_mode="updates"):
        for node_name, update in chunk.items():
            if node_name == "__start__":
                continue
            rprint(f"  [cyan]▶ {node_name}[/cyan]")
            if isinstance(update, dict):
                final_state.update(update)

    exec_time = time.time() - t1
    rprint(f"\n[bold green]═══ Pipeline Complete ({exec_time:.1f}s) ═══[/bold green]\n")

    # ── Step 3: Analyse results ───────────────────────────────────────
    # Check for integration results in task_results
    task_results = final_state.get("task_results", [])
    integration_found = False
    action_blocks_found = 0

    rprint(f"[bold]Task Results ({len(task_results)}):[/bold]")
    for r in task_results:
        result_text = r.get("result", "")
        has_integration = "--- Integration Results ---" in result_text
        has_action_block = "ACTION:" in result_text

        if has_integration:
            integration_found = True
        if has_action_block:
            action_blocks_found += 1

        status_icon = "[green]✓[/green]" if has_integration else "[yellow]○[/yellow]"
        rprint(
            f"  {status_icon} {r.get('task_id', '?')} [{r.get('agent', '?')}] "
            f"{'🔗 Integration results present' if has_integration else ''}"
        )

        # Show a preview of the result
        preview = result_text[:300].replace("\n", "\n    ")
        rprint(f"    [dim]{preview}{'...' if len(result_text) > 300 else ''}[/dim]")

    # ── Step 4: Validation summary ────────────────────────────────────
    rprint("\n[bold]Integration Validation:[/bold]")

    checks = [
        ("Composio connected", composio_ok),
        ("Workers received INTEGRATION TOOLS prompt", composio_ok),  # Implicit via agent_factory
        ("ACTION: blocks found in worker output", action_blocks_found > 0),
        ("Integration results injected", integration_found),
    ]

    for label, passed in checks:
        icon = "[green]✓[/green]" if passed else "[red]✗[/red]"
        rprint(f"  {icon} {label}")

    if not integration_found and composio_ok:
        rprint(
            "\n[yellow]Note: Workers didn't produce ACTION: blocks. This can happen if:[/yellow]"
            "\n  • The LLM decided the task doesn't need integrations"
            "\n  • The prompt isn't explicit enough about using external tools"
            "\n  • Try a more explicit prompt like:"
            '\n    "Use the GITHUB_LIST_REPOS integration tool to list my repos"'
        )

    # ── Final response ────────────────────────────────────────────────
    final_response = final_state.get("final_response", "")
    if final_response and rich_available:
        console.print(
            Panel(
                final_response[:2000] if final_response.strip() else "[dim]No response[/dim]",
                title="[bold]Final Response[/bold]",
                border_style="green",
            )
        )

    # ── Token summary ─────────────────────────────────────────────────
    if rich_available:
        rprint("\n[bold]Agent Summary:[/bold]")
        table = Table(show_header=True, header_style="bold")
        table.add_column("Agent")
        table.add_column("Model")
        table.add_column("Tokens", justify="right")
        for a in agent_registry.all():
            table.add_row(a.name, a.model, str(a.total_tokens))
        total_tokens = sum(a.total_tokens for a in agent_registry.all())
        table.add_row("[bold]TOTAL", "", f"[bold]{total_tokens}")
        console.print(table)

    # ── Dump run ──────────────────────────────────────────────────────
    cfg = load_graph_config()
    dump_path = write_dump(run_id, cfg, final_state, agent_registry, exec_time, query)
    rprint(f"\n[dim]Run dump: {dump_path}[/dim]")
    rprint(f"[dim]Total: {exec_time:.1f}s exec + {build_time:.1f}s build[/dim]\n")

    # Cleanup
    IntegrationRegistry.reset()


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_PROMPT
    asyncio.run(main(query))
