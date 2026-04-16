"""
aifactory run — Unified pipeline execution.

Single command for all pipeline runs.  Supports every option:
  --task-file, --attachment (vision), --mode (with aliases), --context,
  --no-chat, streaming events, Redis task tracking, post-pipeline chat.

`aifactory new` is a thin alias that calls this with simplified UX.
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
    print_banner,
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
        help="Read task from a file instead of the argument.",
    ),
    mode: str = typer.Option(
        "balanced",
        "--mode",
        "-m",
        help="Pipeline mode: best | balanced | cost_efficient. Aliases: light, thorough.",
    ),
    context: str = typer.Option(
        "",
        "--context",
        "-c",
        help="Extra context injected into shared_context (e.g. 'Use PostgreSQL').",
    ),
    attachments: list[str] = typer.Option(  # noqa: B008
        None,
        "--attachment",
        "-a",
        help="Image URL or local path to attach (repeatable). Passed to vision-capable workers.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show debug logs."),
    no_chat: bool = typer.Option(False, "--no-chat", help="Skip post-pipeline chat."),
) -> None:
    """⚡ Run the AI Factory pipeline.

    Mode aliases:
      light / cost_efficient  → cheapest, fastest
      balanced                → default
      best / thorough         → highest quality

    Examples:
        aifactory run "Build a FastAPI todo app"
        aifactory run --mode best -f spec.md
        aifactory run -m light "Fix typo in README"
        aifactory run -c "Use PostgreSQL" "Build a user auth service"
        aifactory run --attachment screenshot.png "What's wrong with this UI?"
    """
    setup_logging("DEBUG" if verbose else "INFO")

    # Resolve mode aliases
    mode_map = {"light": "cost_efficient", "thorough": "best"}
    resolved_mode = mode_map.get(mode, mode)

    # Resolve task source
    if task_file and task_file.exists():
        task_text = task_file.read_text().strip()
    elif task:
        task_text = task
    else:
        console.print("[error]Provide a task argument or --task-file.[/error]")
        raise typer.Exit(1)

    print_banner(
        f"🏭 [bold]AI Factory[/bold] — [bold]{resolved_mode}[/bold] mode\n\n{task_text}",
        style="blue",
    )

    asyncio.run(
        _run_pipeline(
            task_text,
            mode=resolved_mode,
            context=context,
            attachments=attachments or [],
            verbose=verbose,
            no_chat=no_chat,
        )
    )


# =============================================================================
# Core pipeline runner (shared by `run` and the `new` alias)
# =============================================================================


async def _run_pipeline(
    task: str,
    *,
    mode: str = "balanced",
    context: str = "",
    attachments: list[str] | None = None,
    verbose: bool = False,
    no_chat: bool = False,
) -> None:
    """Build and invoke the dynamic pipeline end-to-end."""
    from dotenv import load_dotenv

    load_dotenv()

    # ── Stage 0: Complexity gate ──────────────────────────────────────────────
    from aifactory.core.complexity_gate import ComplexityGate

    gate = ComplexityGate()
    decision = gate.evaluate(task)

    if decision.bypass:
        from rich.panel import Panel

        console.print(
            f"[success]✓[/success] [bold]Instant answer[/bold] "
            f"[dim]({decision.method}: {decision.reason})[/dim]"
        )
        console.print()
        console.print(Panel(decision.answer, title="✅ Result", border_style="green"))
        console.print("\n[muted]Resolved without LLM — 0 tokens, 0 cost, <1ms[/muted]")
        console.print("[muted]Complexity gate: pure math bypass[/muted]\n")

        # Write a lightweight dump so bypass runs appear in `aifactory status`
        from aifactory.core.agent_factory import AgentRegistry
        from aifactory.graphs.dynamic_pipeline import load_graph_config, write_dump

        rid = run_id_now()
        cfg = load_graph_config()
        registry = AgentRegistry()
        shared: dict = {"objective": task}
        if context:
            shared["injected_context"] = context
        write_dump(
            rid,
            cfg,
            {
                "final_response": decision.answer,
                "shared_context": shared,
                "tasks": [],
                "task_results": [],
            },
            registry,
            0.0,
            task,
            gate="bypass",
        )
        return

    # ── Stage 0b: Fast LLM — single master call, no tools/agents ─────────────
    if decision.fast_llm:
        await _run_fast_llm(task)
        return

    # ── Full pipeline ─────────────────────────────────────────────────────────
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

    # Redis — record run as "running"
    _redis_client = _redis_start(rid, task, mode)

    # Streaming event callback — worker progress + token stream + budget warnings
    def _stream_event(event: dict) -> None:
        import sys

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
                "  [dim yellow]i  Memory is in-memory only — won't persist between runs. "
                "Start Qdrant and set QDRANT_URL to enable persistence.[/dim yellow]"
            )
        elif etype == "token":
            sys.stdout.write(event.get("text", ""))
            sys.stdout.flush()

    with Status("[bold cyan]Building pipeline...", console=console):
        compiled, registry = build_dynamic_pipeline(
            mode=mode,
            checkpointer=checkpointer,
            run_id=rid,
            event_callback=_stream_event,
        )

    console.print(
        f"[success]✓[/success] Pipeline built  "
        f"[muted]agents: {', '.join(a.name for a in registry.all())}[/muted]"
    )
    console.print(f"[success]✓[/success] Mode: [bold]{mode}[/bold]  |  Run: [bold]{rid}[/bold]")
    console.print()

    # Build initial state
    shared_ctx: dict = {}
    if context:
        shared_ctx["user_context"] = context

    initial_state: dict = {
        "user_query": task,
        "shared_context": shared_ctx,
        "conversation_history": [],
    }

    if attachments:
        import base64
        import mimetypes

        resolved: list[str] = []
        for att in attachments:
            p = Path(att)
            if p.exists() and p.is_file():
                mime = mimetypes.guess_type(str(p))[0] or "image/jpeg"
                b64 = base64.b64encode(p.read_bytes()).decode()
                resolved.append(f"data:{mime};base64,{b64}")
                console.print(f"  [dim]Attached: {p.name} ({mime})[/dim]")
            else:
                resolved.append(att)
                console.print(f"  [dim]Attached URL: {att[:60]}[/dim]")
        initial_state["attachments"] = resolved

    thread_cfg = {"configurable": {"thread_id": rid}}
    t0 = time.perf_counter()
    result: dict = {}
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

            # Handle worker QUESTION: interrupt
            interrupt_msg = ""
            for snap_task in snap.tasks:
                for intr in snap_task.interrupts:
                    interrupt_msg = str(intr.value)
                    break
                if interrupt_msg:
                    break

            console.print(
                Panel(
                    interrupt_msg,
                    title="🤔 Worker Question",
                    border_style="yellow",
                    expand=False,
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

    # Redis — mark run as completed
    _redis_finish(
        _redis_client,
        rid,
        task,
        mode,
        elapsed,
        sum(a.total_tokens for a in registry.all()),
        result.get("final_response", "")[:200],
        result.get("judge_metrics", {}).get("overall_status", "?"),
    )

    # Display results
    console.print(
        f"\n[bold green]═══ Pipeline Complete ({format_elapsed(elapsed)}) ═══[/bold green]\n"
    )

    final = result.get("final_response", "") or result.get("direct_answer", "")
    if final.strip():
        console.print(Panel(final, title="✅ Result", border_style="green"))
    else:
        console.print("[warning]⚠ No final response generated.[/warning]")

    print_task_results(result.get("task_results", []))
    console.print()
    print_agent_table([a.dump() for a in registry.all()])

    judge = result.get("judge_metrics", {})
    if judge:
        console.print("\n[bold]Judge Metrics:[/bold]")
        for k, v in judge.items():
            console.print(f"  {k}: {v}")

    console.print(f"\n[muted]Run dump: {dump_path}[/muted]")
    console.print(f"[muted]Resume:   aifactory resume {rid}[/muted]\n")

    # Post-pipeline chat
    if not no_chat and asyncio.get_event_loop().is_running():
        import sys

        if sys.stdin.isatty():
            console.print("[muted]Chat with Master (type quit/exit to end):[/muted]")
            await _post_pipeline_chat(registry, result, elapsed=elapsed, rid=rid)
        else:
            console.print(
                "[muted]Non-interactive — skipping chat. "
                "Use `aifactory resume` to chat later.[/muted]\n"
            )


# =============================================================================
# Fast LLM path — single master call, no agents
# =============================================================================


async def _run_fast_llm(task: str) -> None:
    """Answer a simple question with a single master LLM call (~1-3s)."""
    from dotenv import load_dotenv
    from rich.panel import Panel

    load_dotenv()

    t0 = time.perf_counter()

    from aifactory.core.memory_bridge import MemoryBridge
    from aifactory.core.model_selector import best_model, load_agent_cfg, load_available_models
    from aifactory.graphs.dynamic_pipeline import PipelineAgent

    models = load_available_models()
    master_model = best_model(models)
    master_cfg = load_agent_cfg("master")

    rid = run_id_now()
    bridge = None
    try:
        from aifactory.core.memory_router import MemoryRouter

        bridge = MemoryBridge(router=MemoryRouter(), run_id=rid)
    except Exception:
        pass

    master = PipelineAgent(
        "master", {**master_cfg, "model": master_model}, master_model, memory=bridge
    )

    console.print(
        f"[success]✓[/success] Fast path  "
        f"[muted]1 agent (master), model={master_model}, no tools[/muted]"
    )
    console.print()

    reply = await master.think(
        f'Answer this directly and concisely:\n\n"{task}"',
        {"objective": task},
    )

    elapsed = time.perf_counter() - t0
    console.print(f"\n[bold green]═══ Fast Answer ({format_elapsed(elapsed)}) ═══[/bold green]\n")
    console.print(Panel(reply.strip(), title="✅ Result", border_style="green"))
    console.print(
        f"\n[muted]Single LLM call — {master.total_tokens} tokens, model={master_model}[/muted]"
    )
    console.print(
        "[muted]Complexity gate: simple Q&A → fast path (no tools, no multi-agent)[/muted]\n"
    )


# =============================================================================
# Post-pipeline chat
# =============================================================================


async def _post_pipeline_chat(
    registry: object,
    result: dict,
    elapsed: float = 0.0,
    rid: str = "",
) -> None:
    from aifactory.core.agent_factory import AgentRegistry

    if not isinstance(registry, AgentRegistry):
        return

    master = registry.get("master")
    if master is None:
        return

    shared_ctx = dict(result.get("shared_context", {}))
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

    loop = asyncio.get_event_loop()
    while True:
        try:
            user_input = await loop.run_in_executor(None, lambda: input("\n[You] > "))
            user_input = user_input.strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input or user_input.lower() in QUIT_WORDS:
            break
        from rich.panel import Panel

        reply = await master.think(
            f'The human replied: "{user_input}"\n\nRespond helpfully.',
            shared_ctx,
        )
        console.print(Panel(reply, title="Master", border_style="green", expand=False))

    console.print("[muted]Session complete.[/muted]\n")


# =============================================================================
# Redis helpers
# =============================================================================


# =============================================================================
# Resume a previous run
# =============================================================================


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

    from aifactory.cli._helpers import (
        find_dump_fuzzy,
        print_agent_table,
        print_banner,
        print_task_results,
    )

    dump = find_dump_fuzzy(task_id)
    if not dump:
        console.print(f"[error]No run found matching '{task_id}'.[/error]")
        console.print("[muted]Try: aifactory status  (to list recent runs)[/muted]")
        raise typer.Exit(1)

    rid = dump.get("run_id", "?")
    request = dump.get("config", {}).get("request", "?")
    console.print(f"\n[success]✓[/success] Resuming run [bold]{rid}[/bold]: {request}\n")

    final = dump.get("final_response", "")
    if final:
        from aifactory.cli._helpers import print_banner

        print_banner(final, style="green")

    print_task_results(dump.get("task_results", []))
    print_agent_table(dump.get("agents", []))

    import sys

    if sys.stdin.isatty():
        console.print("\n[muted]Chat with Master (type quit/exit to end):[/muted]")
        _resume_chat(dump)
    else:
        console.print("\n[muted]Non-interactive mode — skipping chat.[/muted]\n")


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
            from rich.panel import Panel

            reply = asyncio.run(
                master.think(
                    f'The human replied: "{user_input}"\n\nRespond helpfully.',
                    shared_ctx,
                )
            )
            console.print(Panel(reply, title="Master", border_style="green", expand=False))

    except Exception as e:
        console.print(f"[error]Chat failed: {e}[/error]")

    console.print("[muted]Session complete.[/muted]\n")


# =============================================================================
# Redis helpers
# =============================================================================


def _redis_start(rid: str, task: str, mode: str) -> object | None:
    """Record a new run as 'running' in Redis. Returns the client or None."""
    import json
    import os
    from datetime import UTC, datetime

    redis_url = os.getenv("REDIS_URL", "")
    if not redis_url:
        return None
    try:
        import redis as _redis

        client = _redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=1)
        client.ping()
        ttl = 7 * 86400
        client.setex(
            f"aifactory:run:{rid}",
            ttl,
            json.dumps(
                {
                    "run_id": rid,
                    "status": "running",
                    "task": task,
                    "mode": mode,
                    "started_at": datetime.now(UTC).isoformat(),
                }
            ),
        )
        client.zadd("aifactory:runs:all", {rid: datetime.now(UTC).timestamp()})
        client.expire("aifactory:runs:all", 30 * 86400)
        return client
    except Exception:
        return None


def _redis_finish(
    client: object | None,
    rid: str,
    task: str,
    mode: str,
    elapsed: float,
    total_tokens: int,
    final_preview: str,
    judge_status: str,
) -> None:
    """Update the run record to 'completed' in Redis."""
    if client is None:
        return
    import contextlib
    import json

    with contextlib.suppress(Exception):
        client.setex(  # type: ignore[attr-defined]
            f"aifactory:run:{rid}",
            7 * 86400,
            json.dumps(
                {
                    "run_id": rid,
                    "status": "completed",
                    "task": task,
                    "mode": mode,
                    "elapsed_s": round(elapsed, 2),
                    "total_tokens": total_tokens,
                    "final_response_preview": final_preview,
                    "judge_status": judge_status,
                }
            ),
        )
