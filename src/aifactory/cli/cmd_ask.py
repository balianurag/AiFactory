"""
aifactory ask — Contextual question answering against memory and run history.
"""

from __future__ import annotations

import asyncio

import typer

from aifactory.cli._helpers import (
    console,
    find_dump_fuzzy,
    list_dumps,
    setup_logging,
)


def ask(
    question: str = typer.Argument(..., help="Your question."),
    run_id: str = typer.Option(None, "--run", "-r", help="Scope question to a specific run."),
) -> None:
    """💬 Ask a contextual question about your projects.

    Queries the AI Factory's memory system and recent run history
    to answer questions about past work, feature status, decisions, etc.

    Examples:
        aifactory ask "What's the status of the login feature?"
        aifactory ask "Which model was used for the GitLab task?"
        aifactory ask --run 20260409 "What tools were used?"
    """
    setup_logging()
    asyncio.run(_ask_question(question, run_id=run_id))


async def _ask_question(question: str, *, run_id: str | None = None) -> None:
    """Answer a question using memory + dump context."""
    from dotenv import load_dotenv
    from rich.panel import Panel

    load_dotenv()

    context_parts: list[str] = []

    # ── Gather context from run dumps ────────────────────────────────
    if run_id:
        dump = find_dump_fuzzy(run_id)
        if dump:
            context_parts.append(_dump_to_context(dump))
        else:
            console.print(
                f"[warning]No run found matching '{run_id}', searching all recent runs.[/warning]"
            )

    if not context_parts:
        # Search recent dumps for relevant context
        dumps = list_dumps(limit=10)
        for d in dumps:
            context_parts.append(_dump_to_context(d))

    # ── Try memory system ────────────────────────────────────────────
    memory_ctx = ""
    try:
        from aifactory.core.memory_router import MemoryRouter

        router = MemoryRouter()
        results = router.query(question, limit=5)
        if results:
            memory_ctx = "\n".join(
                f"- {r.payload.get('content', '')[:200]}" for r in results if r.payload
            )
            context_parts.append(f"Memory search results:\n{memory_ctx}")
    except Exception:
        pass  # Memory system might not be available

    # ── Ask the master agent ─────────────────────────────────────────
    full_context = "\n\n---\n\n".join(context_parts) if context_parts else ""

    try:
        from aifactory.graphs.dynamic_pipeline import build_dynamic_pipeline

        _, registry = build_dynamic_pipeline()
        master = registry.get("master")
        if not master:
            console.print("[error]Could not initialise master agent.[/error]")
            return

        prompt = (
            f'The user asks: "{question}"\n\n'
            f"Here is context from recent pipeline runs and memory:\n\n"
            f"{full_context}\n\n"
            f"Answer the question based on this context. Be specific and cite "
            f"run IDs or task IDs when relevant. If you don't have enough "
            f"context, say so honestly."
        )

        from rich.status import Status

        with Status("Thinking...", console=console):
            reply = await master.think(prompt, {})

        from aifactory.cli._helpers import render_result
        render_result(reply, title="💬 Answer")

    except Exception as e:
        console.print(f"[error]Ask failed: {e}[/error]")

        # Fallback: just show relevant dumps
        if context_parts:
            console.print("\n[heading]Relevant context found:[/heading]")
            for part in context_parts[:3]:
                console.print(f"  {part[:300]}...")
            console.print()


def _dump_to_context(dump: dict) -> str:
    """Convert a dump dict to a concise context string for the LLM."""
    rid = dump.get("run_id", "?")
    cfg = dump.get("config", {})
    request = cfg.get("request", "?")
    mode = cfg.get("mode", "?")

    tasks = dump.get("tasks", [])
    task_lines = [t.get("description", t.get("id", "?")) for t in tasks[:5]]

    results = dump.get("task_results", [])
    result_lines = [f"{r.get('task_id', '?')}: {r.get('result', '')[:150]}" for r in results[:5]]

    final = dump.get("final_response", "")[:300]
    judge = dump.get("judge_metrics", {})

    shared = dump.get("shared_context_final", {})
    shared_str = ", ".join(f"{k}={str(v)[:50]}" for k, v in shared.items())

    return (
        f"Run {rid} (mode={mode}): {request}\n"
        f"Tasks: {'; '.join(task_lines)}\n"
        f"Results: {'; '.join(result_lines)}\n"
        f"Final: {final}\n"
        f"Judge: {judge}\n"
        f"Context: {shared_str}"
    )
