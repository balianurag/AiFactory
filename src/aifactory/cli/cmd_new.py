"""
aifactory new  — Start a new pipeline run.
aifactory resume — Resume a previous run.

These are the primary entry points for running the AI Factory pipeline.
`new` is the friendly alias that starts a fresh run.
`resume` reloads a previous dump and enters the post-pipeline chat.
"""

from __future__ import annotations

import asyncio
import time

import typer

from aifactory.cli._helpers import (
    QUIT_WORDS,
    console,
    find_dump_fuzzy,
    format_elapsed,
    print_agent_table,
    print_banner,
    print_task_results,
    run_id_now,
    setup_logging,
)


def new(
    task: str = typer.Argument(..., help="Describe what you want to build or do."),
    mode: str = typer.Option(
        "balanced",
        "--mode",
        "-m",
        help="Pipeline mode: best, balanced, or cost_efficient.",
    ),
    context: str = typer.Option(
        "",
        "--context",
        "-c",
        help="Extra context injected into shared_context (e.g. 'Use PostgreSQL').",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show debug logs."),
) -> None:
    """🚀 Start a new AI Factory pipeline run.

    Examples:
        aifactory new "Build a FastAPI todo app with auth"
        aifactory new --mode best "Design a microservices architecture"
        aifactory new -m cost_efficient "Write unit tests for user model"
        aifactory new -c "Use PostgreSQL" "Build a user auth service"
    """
    setup_logging("DEBUG" if verbose else "INFO")

    print_banner(
        f"🏭 [bold]AI Factory[/bold]\n\n{task}",
        style="blue",
    )

    asyncio.run(_run_pipeline(task, mode=mode, context=context, verbose=verbose))


def resume(
    task_id: str = typer.Argument(..., help="Run ID or partial match to resume."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show debug logs."),
) -> None:
    """🔄 Resume a previous pipeline run (enter post-pipeline chat).

    Examples:
        aifactory resume 20260409_132447
        aifactory resume 20260409
        aifactory resume "todo app"
    """
    setup_logging("DEBUG" if verbose else "INFO")

    dump = find_dump_fuzzy(task_id)
    if not dump:
        console.print(f"[error]No run found matching '{task_id}'.[/error]")
        console.print("[muted]Try: aifactory status  (to list recent runs)[/muted]")
        raise typer.Exit(1)

    rid = dump.get("run_id", "?")
    request = dump.get("config", {}).get("request", "?")
    console.print(f"\n[success]✓[/success] Resuming run [bold]{rid}[/bold]: {request}\n")

    # Display the results from the dump
    final = dump.get("final_response", "")
    if final:
        print_banner(final, style="green")

    print_task_results(dump.get("task_results", []))
    print_agent_table(dump.get("agents", []))

    # Enter chat with master — skip when stdin is not interactive
    import sys

    if sys.stdin.isatty():
        console.print("\n[muted]Chat with Master (type quit/exit to end):[/muted]")
        _resume_chat(dump)
    else:
        console.print("\n[muted]Non-interactive mode — skipping chat.[/muted]\n")


# ── Fast LLM Path — single master call, no tools/agents ────────────────────


async def _run_fast_llm(task: str) -> None:
    """Answer a simple question with a single master LLM call.

    No Composio, no judge, no analyser, no workers — just the master agent.
    Typically finishes in ~1-3s instead of ~20s for the full pipeline.
    """
    from dotenv import load_dotenv
    from rich.panel import Panel

    load_dotenv()

    t0 = time.perf_counter()

    # Build just the master agent — no graph, no integrations
    from aifactory.core.memory_bridge import MemoryBridge
    from aifactory.core.model_selector import best_model, load_agent_cfg, load_available_models
    from aifactory.graphs.dynamic_pipeline import PipelineAgent

    models = load_available_models()
    master_model = best_model(models)
    master_cfg = load_agent_cfg("master")

    # Wire memory so master can recall prior context (e.g. "what's my name?")
    rid = run_id_now()
    bridge = None
    try:
        from aifactory.core.memory_router import MemoryRouter

        bridge = MemoryBridge(router=MemoryRouter(), run_id=rid)
    except Exception:
        pass  # memory failures never block the fast path
    master = PipelineAgent(
        "master", {**master_cfg, "model": master_model}, master_model, memory=bridge
    )

    console.print(
        f"[success]✓[/success] Fast path  "
        f"[muted]1 agent (master), model={master_model}, no tools[/muted]"
    )
    console.print()

    # Single LLM call — direct answer
    reply = await master.think(
        f'Answer this directly and concisely:\n\n"{task}"',
        {"objective": task},
    )

    elapsed = time.perf_counter() - t0

    console.print(f"\n[bold green]═══ Fast Answer ({format_elapsed(elapsed)}) ═══[/bold green]\n")
    console.print(Panel(reply.strip(), title="✅ Result", border_style="green"))

    # Token summary
    console.print(
        f"\n[muted]Single LLM call — {master.total_tokens} tokens, model={master_model}[/muted]"
    )
    console.print(
        "[muted]Complexity gate: simple Q&A → fast path (no tools, no multi-agent)[/muted]\n"
    )


# ── Internal Pipeline Runner ──────────────────────────────────────────────────


async def _run_pipeline(
    task: str, *, mode: str = "balanced", context: str = "", verbose: bool = False
) -> None:
    """Build and invoke the dynamic pipeline, handling interrupts."""
    from dotenv import load_dotenv

    load_dotenv()

    # ── Stage 0: Complexity gate — bypass LLM for trivial tasks ──────
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
        return

    # ── Stage 0b: Fast LLM — single master call, no tools/agents ────
    if decision.fast_llm:
        await _run_fast_llm(task)
        return

    # ── Full pipeline ───────────────────────────────────────────────────
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command
    from rich.status import Status

    from aifactory.graphs.dynamic_pipeline import (
        build_dynamic_pipeline,
        load_graph_config,
        write_dump,
    )

    rid = run_id_now()
    checkpointer = MemorySaver()

    # Redis task store — record run as "running" immediately
    _redis_run_key = f"aifactory:run:{rid}"
    try:
        import asyncio as _asyncio
        import json as _json
        from datetime import UTC
        from datetime import datetime as _dt

        from aifactory.core.redis_client import get_redis as _get_redis

        _redis = _asyncio.get_event_loop().run_until_complete(_get_redis()) if False else None
        # Use sync redis if available
        import os as _os

        _redis_url = _os.getenv("REDIS_URL", "")
        if _redis_url:
            import redis as _redis_sync

            _redis_client = _redis_sync.from_url(
                _redis_url, decode_responses=True, socket_connect_timeout=1
            )
            _redis_client.ping()
            _run_ttl = 7 * 86400
            _redis_client.setex(
                _redis_run_key,
                _run_ttl,
                _json.dumps(
                    {
                        "run_id": rid,
                        "status": "running",
                        "task": task,
                        "mode": mode,
                        "started_at": _dt.now(UTC).isoformat(),
                    }
                ),
            )
            _redis_client.zadd("aifactory:runs:all", {rid: _dt.now(UTC).timestamp()})
            _redis_client.expire("aifactory:runs:all", 30 * 86400)
        else:
            _redis_client = None
    except Exception:
        _redis_client = None

    # Phase 5: streaming event callback — shows worker progress inside execution_lead
    def _stream_event(event: dict) -> None:
        etype = event.get("type", "")
        if etype == "worker_start":
            console.print(
                f"    [dim cyan]↳ {event['worker']}[/dim cyan]  [dim]{event['task'][:70]}[/dim]"
            )
        elif etype == "worker_complete":
            flag = " [yellow]⚠ flagged[/yellow]" if event.get("quality_flagged") else ""
            skip = " [dim](skipped)[/dim]" if event.get("skipped") else ""
            console.print(f"    [green]✓[/green] {event['worker']} done{skip}{flag}")
        elif etype == "budget_warning":
            spent = event.get("spent", 0)
            cap = event.get("cap", 0)
            pct = int(spent / cap * 100) if cap > 0 else 0
            console.print(
                f"\n  [bold yellow]⚠  Budget alert:[/bold yellow] "
                f"${spent:.4f} spent ({pct}% of ${cap:.2f} cap) — "
                f"pipeline will halt at cap.\n"
            )
        elif etype == "qdrant_in_memory":
            console.print(
                "  [dim yellow]ℹ  Memory is in-memory only — won't persist between runs. "
                "Start Qdrant and set QDRANT_URL to enable persistence.[/dim yellow]"
            )
        elif etype == "token":
            # Stream individual tokens to stdout without newline
            # Use console.file.write to bypass Rich markup parsing (raw tokens)
            import sys

            sys.stdout.write(event.get("text", ""))
            sys.stdout.flush()

    # Build
    with Status("[bold cyan]Building pipeline...", console=console):
        compiled, registry = build_dynamic_pipeline(
            mode=mode,
            checkpointer=checkpointer,
            run_id=rid,
            event_callback=_stream_event,
        )

    console.print(
        f"[success]✓[/success] Pipeline built  [muted]agents: {', '.join(a.name for a in registry.all())}[/muted]"  # noqa: E501
    )
    console.print(f"[success]✓[/success] Mode: [bold]{mode}[/bold]  |  Run: [bold]{rid}[/bold]")
    console.print()

    # Invoke with streaming
    shared_ctx: dict = {}
    if context:
        shared_ctx["user_context"] = context

    initial_state = {
        "user_query": task,
        "shared_context": shared_ctx,
        "conversation_history": [],
    }
    thread_cfg = {"configurable": {"thread_id": rid}}
    t0 = time.perf_counter()

    result: dict = {}
    current_input = initial_state

    try:
        while True:
            # Stream node-by-node
            async for chunk in compiled.astream(
                current_input, config=thread_cfg, stream_mode="updates"
            ):
                for node_name, update in chunk.items():
                    if node_name == "__start__":
                        continue
                    console.print(f"  [info]▶ {node_name}[/info] done")
                    if isinstance(update, dict):
                        result.update(update)

            # Check for interrupts (worker QUESTION)
            snap = compiled.get_state(thread_cfg)
            if not snap.next:
                break

            interrupt_msg = ""
            for snap_task in snap.tasks:
                for intr in snap_task.interrupts:
                    interrupt_msg = str(intr.value)
                    break
                if interrupt_msg:
                    break

            from rich.panel import Panel

            console.print(
                Panel(
                    interrupt_msg, title="🤔 Worker Question", border_style="yellow", expand=False
                )
            )
            answer = input("[You] > ").strip()
            console.print()

            if not answer or answer.lower() in QUIT_WORDS:
                console.print("[muted]Exiting early.[/muted]")
                break

            current_input = Command(resume=answer)

    except KeyboardInterrupt:
        console.print("\n[warning]Interrupted by user.[/warning]")

    elapsed = time.perf_counter() - t0

    # Write dump
    cfg = load_graph_config()
    dump_path = write_dump(rid, cfg, result, registry, elapsed, task)

    # Redis task store — update run record to "completed"
    if _redis_client is not None:
        try:
            import json as _json2

            _total_tok = sum(a.get("total_tokens", 0) for a in [ag.dump() for ag in registry.all()])
            _final_prev = (result.get("final_response", "") or "")[:200]
            _redis_client.setex(
                _redis_run_key,
                7 * 86400,
                _json2.dumps(
                    {
                        "run_id": rid,
                        "status": "completed",
                        "task": task,
                        "mode": mode,
                        "elapsed_s": round(elapsed, 2),
                        "total_tokens": _total_tok,
                        "final_response_preview": _final_prev,
                        "judge_status": result.get("judge_metrics", {}).get("overall_status", "?"),
                    }
                ),
            )
        except Exception:
            pass

    # Display results
    console.print(
        f"\n[bold green]═══ Pipeline Complete ({format_elapsed(elapsed)}) ═══[/bold green]\n"
    )

    final = result.get("final_response", "") or result.get("direct_answer", "")
    if final.strip():
        from rich.panel import Panel

        console.print(Panel(final, title="✅ Result", border_style="green"))
    else:
        console.print("[warning]⚠ No final response generated.[/warning]")

    print_task_results(result.get("task_results", []))

    # Agent summary
    console.print()
    agent_data = [a.dump() for a in registry.all()]
    print_agent_table(agent_data)

    # Judge metrics
    judge = result.get("judge_metrics", {})
    if judge:
        console.print("\n[bold]Judge Metrics:[/bold]")
        for k, v in judge.items():
            console.print(f"  {k}: {v}")

    console.print(f"\n[muted]Run dump: {dump_path}[/muted]")
    console.print(f"[muted]Resume anytime: aifactory resume {rid}[/muted]\n")

    # Post-pipeline chat — skip when stdin is not interactive (piped input)
    import sys

    if sys.stdin.isatty():
        console.print("[muted]Chat with Master (type quit/exit to end):[/muted]")
        _post_pipeline_chat(registry, result, elapsed=elapsed, rid=rid)
    else:
        console.print(
            "[muted]Non-interactive mode — skipping chat. Use `aifactory resume` to chat later.[/muted]\n"  # noqa: E501
        )


def _post_pipeline_chat(
    registry: object,
    result: dict,
    elapsed: float = 0.0,
    rid: str = "",
) -> None:
    """Interactive chat loop with the master agent after pipeline completion."""
    from aifactory.core.agent_factory import AgentRegistry

    if not isinstance(registry, AgentRegistry):
        return

    master = registry.get("master")
    if master is None:
        return

    shared_ctx = dict(result.get("shared_context", {}))

    # Inject run metrics so Master can answer questions about tokens/cost/time
    _agents = [a.dump() for a in registry.all()]
    _total_tok = sum(a.get("total_tokens", 0) for a in _agents)
    _agent_lines = "  ".join(
        f"{a['name']}={a.get('total_tokens', 0)}" for a in _agents if a.get("total_tokens", 0) > 0
    )
    _judge = result.get("judge_metrics", {})
    shared_ctx["run_metrics"] = (
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

        reply = asyncio.run(
            master.think(
                f'The human replied: "{user_input}"\n\nRespond helpfully.',
                shared_ctx,
            )
        )
        from rich.panel import Panel

        console.print(Panel(reply, title="Master", border_style="green", expand=False))

    console.print("[muted]Session complete.[/muted]\n")


def _resume_chat(dump: dict) -> None:
    """Chat using a fresh master agent loaded from dump context."""
    try:
        from dotenv import load_dotenv

        load_dotenv()

        from aifactory.graphs.dynamic_pipeline import build_dynamic_pipeline

        _, registry = build_dynamic_pipeline()
        master = registry.get("master")
        if not master:
            console.print("[error]Could not initialise master agent.[/error]")
            return

        shared_ctx = dict(dump.get("shared_context_final", {}))

        # Inject run metrics from dump so Master can answer token/cost questions
        _agents = dump.get("agents", [])
        _total_tok = sum(a.get("total_tokens", 0) for a in _agents)
        _agent_lines = "  ".join(
            f"{a['name']}={a.get('total_tokens', 0)}"
            for a in _agents
            if a.get("total_tokens", 0) > 0
        )
        _judge = dump.get("judge_metrics", {})
        shared_ctx["run_metrics"] = (
            f"run_id={dump.get('run_id', '?')} | "
            f"elapsed={dump.get('elapsed_s', 0):.1f}s | "
            f"total_tokens={_total_tok} | by_agent: {_agent_lines} | "
            f"status={_judge.get('overall_status', '?')}"
        )

        # Prime master with the original conversation
        final_resp = dump.get("final_response", "")
        if final_resp:
            master._messages.append({"role": "assistant", "content": final_resp})

        while True:
            try:
                user_input = input("\n[You] > ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input or user_input.lower() in QUIT_WORDS:
                break

            reply = asyncio.run(
                master.think(
                    f'The human replied: "{user_input}"\n\nRespond helpfully.',
                    shared_ctx,
                )
            )
            from rich.panel import Panel

            console.print(Panel(reply, title="Master", border_style="green", expand=False))

    except Exception as e:
        console.print(f"[error]Chat failed: {e}[/error]")

    console.print("[muted]Session complete.[/muted]\n")
