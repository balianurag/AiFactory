"""
aifactory run — Advanced pipeline execution with mode selection.

This is the power-user command with full control over pipeline behaviour.
`new` is the friendly wrapper; `run` exposes all knobs.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import typer

from aifactory.cli._helpers import (
    QUIT_WORDS,
    console,
    format_elapsed,
    print_agent_table,
    print_task_results,
    run_id_now,
    setup_logging,
)


def run(
    task: str = typer.Argument(None, help="The task/idea to execute."),
    task_file: Path = typer.Option(  # noqa: B008
        None,
        "--task-file",
        "-f",
        help="Read task from a file instead.",
    ),
    mode: str = typer.Option(
        "balanced",
        "--mode",
        "-m",
        help="Pipeline mode: best | balanced | cost_efficient | light.",
    ),
    context: str = typer.Option(
        "",
        "--context",
        "-c",
        help="Additional context to inject into shared_context.",
    ),
    attachments: list[str] = typer.Option(  # noqa: B008
        None,
        "--attachment",
        "-a",
        help="Image URL or path to attach (repeatable). Vision-capable models will analyze it.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show debug logs."),
    no_chat: bool = typer.Option(False, "--no-chat", help="Skip post-pipeline chat."),
) -> None:
    """⚡ Run the AI Factory pipeline (advanced).

    Full control over mode, context injection, and output verbosity.

    Mode aliases:
      light         → cost_efficient (fastest, cheapest)
      balanced      → balanced (default)
      best          → best (highest quality)
      thorough      → best (alias)

    Examples:
        aifactory run --mode light "Fix typo in README"
        aifactory run --mode best -f spec.md
        aifactory run -m balanced -c "Use PostgreSQL" "Build a user service"
    """
    setup_logging("DEBUG" if verbose else "INFO")

    # Resolve mode aliases
    mode_map = {"light": "cost_efficient", "thorough": "best"}
    resolved_mode = mode_map.get(mode, mode)

    # Resolve task input
    if task_file and task_file.exists():
        task_text = task_file.read_text().strip()
    elif task:
        task_text = task
    else:
        console.print("[error]Provide a task as an argument or via --task-file.[/error]")
        raise typer.Exit(1)

    from rich.panel import Panel

    console.print()
    console.print(
        Panel(
            task_text,
            title=f"🏭 AI Factory — [bold]{resolved_mode}[/bold] mode",
            border_style="blue",
            expand=False,
        )
    )
    console.print()

    asyncio.run(
        _run_advanced(
            task_text,
            mode=resolved_mode,
            context=context,
            verbose=verbose,
            no_chat=no_chat,
            attachments=attachments or [],
        )
    )


async def _run_advanced(
    task: str,
    *,
    mode: str,
    context: str,
    verbose: bool,
    no_chat: bool,
    attachments: list[str] | None = None,
) -> None:
    """Full pipeline execution with all options."""
    from dotenv import load_dotenv

    load_dotenv()

    # ── Stage 0: Complexity gate — bypass LLM for trivial tasks ──────
    # Gate always fires; --context is forwarded into the dump for bypass runs
    # and merged into the pipeline for fast_llm / full-pipeline runs.
    from aifactory.core.complexity_gate import ComplexityGate

    gate = ComplexityGate()
    decision = gate.evaluate(task)
    if decision.bypass:
        from rich.panel import Panel

        console.print(
            f"[success]✓[/success] [bold]Instant answer[/bold] [dim]({decision.method}: {decision.reason})[/dim]"  # noqa: E501
        )
        console.print()
        console.print(Panel(decision.answer, title="✅ Result", border_style="green"))
        console.print("\n[muted]Resolved without LLM — 0 tokens, 0 cost, <1ms[/muted]")
        console.print("[muted]Complexity gate: pure math bypass[/muted]\n")

        # Save a lightweight dump so bypass runs appear in `aifactory status`
        from aifactory.core.agent_factory import AgentRegistry
        from aifactory.graphs.dynamic_pipeline import load_graph_config, write_dump

        rid = run_id_now()
        cfg = load_graph_config()
        registry = AgentRegistry()
        shared: dict = {"objective": task}
        if context:
            shared["injected_context"] = context
        result = {
            "final_response": decision.answer,
            "shared_context": shared,
            "tasks": [],
            "task_results": [],
        }
        write_dump(rid, cfg, result, registry, 0.0, task, gate="bypass")
        return

    # Fast LLM — single master call, no tools/agents
    if decision.fast_llm:
        from aifactory.cli.cmd_new import _run_fast_llm

        await _run_fast_llm(task)
        return

    # ── Full pipeline ───────────────────────────────────────────────────
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command
    from rich.panel import Panel
    from rich.status import Status

    from aifactory.graphs.dynamic_pipeline import (
        build_dynamic_pipeline,
        load_graph_config,
        write_dump,
    )

    rid = run_id_now()
    checkpointer = MemorySaver()

    with Status("[bold cyan]Building pipeline...", console=console):
        compiled, registry = build_dynamic_pipeline(
            mode=mode, checkpointer=checkpointer, run_id=rid
        )

    agents = [a.name for a in registry.all()]
    models = sorted({a.model for a in registry.all()})
    console.print(
        f"[success]✓[/success] Pipeline ready  [muted]mode={mode} agents={len(agents)} models={', '.join(models)}[/muted]"  # noqa: E501
    )
    console.print()

    shared_ctx: dict = {}
    if context:
        shared_ctx["user_context"] = context

    initial_state: dict = {
        "user_query": task,
        "shared_context": shared_ctx,
        "conversation_history": [],
    }
    if attachments:
        # Resolve local file paths to data URIs; pass URLs as-is
        import base64

        resolved: list[str] = []
        for att in attachments:
            p = Path(att)
            if p.exists() and p.is_file():
                import mimetypes

                mime = mimetypes.guess_type(str(p))[0] or "image/jpeg"
                b64 = base64.b64encode(p.read_bytes()).decode()
                resolved.append(f"data:{mime};base64,{b64}")
                console.print(f"  [dim]Attached local file: {p.name} ({mime})[/dim]")
            else:
                resolved.append(att)
                console.print(f"  [dim]Attached URL: {att[:60]}[/dim]")
        initial_state["attachments"] = resolved
    thread_cfg = {"configurable": {"thread_id": rid}}
    t0 = time.perf_counter()

    result: dict = dict(initial_state)
    current_input = initial_state

    try:
        while True:
            async for chunk in compiled.astream(
                current_input, config=thread_cfg, stream_mode="updates"
            ):
                for node_name, update in chunk.items():
                    if node_name == "__start__":
                        continue
                    console.print(f"  [info]▶ {node_name}[/info] done")
                    if isinstance(update, dict):
                        result.update(update)

            snap = compiled.get_state(thread_cfg)
            if not snap.next:
                break

            # Handle interrupt
            interrupt_msg = ""
            for snap_task in snap.tasks:
                for intr in snap_task.interrupts:
                    interrupt_msg = str(intr.value)
                    break
                if interrupt_msg:
                    break

            console.print(
                Panel(
                    interrupt_msg, title="🤔 Worker Question", border_style="yellow", expand=False
                )
            )
            answer = input("[You] > ").strip()
            console.print()

            if not answer or answer.lower() in QUIT_WORDS:
                break

            current_input = Command(resume=answer)

    except KeyboardInterrupt:
        console.print("\n[warning]Interrupted by user.[/warning]")

    elapsed = time.perf_counter() - t0

    # Dump
    cfg = load_graph_config()
    dump_path = write_dump(rid, cfg, result, registry, elapsed, task)

    # Results
    console.print(f"\n[bold green]═══ Complete ({format_elapsed(elapsed)}) ═══[/bold green]\n")

    final = result.get("final_response", "") or result.get("direct_answer", "")
    if final.strip():
        console.print(Panel(final, title="✅ Result", border_style="green"))

    print_task_results(result.get("task_results", []))
    console.print()
    print_agent_table([a.dump() for a in registry.all()])

    judge = result.get("judge_metrics", {})
    if judge:
        console.print("\n[bold]Judge Metrics:[/bold]")
        for k, v in judge.items():
            console.print(f"  {k}: {v}")

    console.print(f"\n[muted]Dump: {dump_path}[/muted]")
    console.print(f"[muted]Resume: aifactory resume {rid}[/muted]\n")

    # Post-pipeline chat
    if not no_chat:
        console.print("[muted]Chat with Master (type quit/exit to end):[/muted]")
        master = registry.get("master")
        if master:
            ctx = dict(result.get("shared_context", {}))

            # Inject run metrics so Master can answer questions about tokens/cost/time
            _agents = [a.dump() for a in registry.all()]
            _total_tok = sum(a.get("total_tokens", 0) for a in _agents)
            _agent_lines = "  ".join(
                f"{a['name']}={a.get('total_tokens', 0)}"
                for a in _agents
                if a.get("total_tokens", 0) > 0
            )
            _judge = result.get("judge_metrics", {})
            ctx["run_metrics"] = (
                f"run_id={rid} | elapsed={format_elapsed(elapsed)} | "
                f"total_tokens={_total_tok} | by_agent: {_agent_lines} | "
                f"status={_judge.get('overall_status', '?')} | "
                f"pipeline_efficiency={_judge.get('pipeline_efficiency', '?')}"
            )

            while True:
                try:
                    user_input = input("\n[You] > ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not user_input or user_input.lower() in QUIT_WORDS:
                    break
                reply = await master.think(
                    f'The human replied: "{user_input}"\n\nRespond helpfully.', ctx
                )
                console.print(Panel(reply, title="Master", border_style="green", expand=False))

        console.print("[muted]Session complete.[/muted]\n")
