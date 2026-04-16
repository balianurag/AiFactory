"""
aifactory status  — Show active + recent runs.
aifactory inspect — Deep dive into a specific run's memory & decisions.
"""

from __future__ import annotations

import typer
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from aifactory.cli._helpers import (
    console,
    find_dump_fuzzy,
    format_elapsed,
    list_dumps,
    print_agent_table,
    setup_logging,
)


def _format_run_datetime(run_id: str) -> str:
    """Parse YYYYMMDD_HHMMSS run_id into '09 Apr 2026, 14:03' format."""
    try:
        from datetime import datetime

        dt = datetime.strptime(run_id, "%Y%m%d_%H%M%S")
        return dt.strftime("%d %b %Y, %H:%M")
    except (ValueError, TypeError):
        return "?"


def status(
    limit: int = typer.Option(10, "--limit", "-n", help="Number of recent runs to show."),
) -> None:
    """📊 Show recent pipeline runs and their status.

    Displays a table of recent runs with timing, tokens, and result preview.

    Examples:
        aifactory status
        aifactory status -n 20
    """
    setup_logging()

    dumps = list_dumps(limit=limit)
    if not dumps:
        console.print("\n[warning]No pipeline runs found.[/warning]")
        console.print('[muted]Start one with: aifactory new "Your task here"[/muted]\n')
        raise typer.Exit(0)

    table = Table(
        title="🏭 Recent Pipeline Runs",
        show_header=True,
        header_style="bold magenta",
        border_style="dim",
        show_lines=False,
    )
    table.add_column("Run ID", style="bold cyan", min_width=16)
    table.add_column("Date & Time", style="dim")
    table.add_column("Task", max_width=50)
    table.add_column("Mode", style="magenta", justify="center")
    table.add_column("Time", justify="right", style="yellow")
    table.add_column("Tokens", justify="right")
    table.add_column("Status", justify="center")

    for d in dumps:
        rid = d.get("run_id", "?")
        request = d.get("config", {}).get("request", "")[:50]

        # Parse human-friendly date/time from run_id (YYYYMMDD_HHMMSS)
        dt_str = _format_run_datetime(rid)

        # Gate-aware mode display
        gate = d.get("gate", "")
        if gate == "bypass":
            mode = "[green]⚡ bypass[/green]"
        elif gate == "fast_llm":
            mode = "[cyan]🏃 fast[/cyan]"
        else:
            mode = d.get("config", {}).get("mode", "?")
        elapsed = d.get("elapsed_seconds", 0)
        totals = d.get("totals", {})
        total_tokens = totals.get("total_tokens", 0)

        # Determine status — gate-aware
        if gate == "bypass":
            status_str = "[green]⚡ instant[/green]"
        elif gate == "fast_llm":
            status_str = "[cyan]✓ fast[/cyan]"
        else:
            judge = d.get("judge_metrics", {})
            judge_status = judge.get("overall_status", "")
            if judge_status == "successful":
                status_str = "[green]✓ done[/green]"
            elif judge_status == "partial":
                status_str = "[yellow]◐ partial[/yellow]"
            elif "error" in str(judge_status).lower():
                status_str = "[red]✗ error[/red]"
            elif d.get("final_response", "").strip():
                status_str = "[green]✓ done[/green]"
            else:
                status_str = "[dim]? unknown[/dim]"

        table.add_row(
            rid,
            dt_str,
            request or "[dim]—[/dim]",
            mode,
            format_elapsed(elapsed),
            f"{total_tokens:,}" if total_tokens else "0",
            status_str,
        )

    console.print()
    console.print(table)
    console.print(
        f"\n[muted]Showing {len(dumps)} of {len(dumps)} runs. "
        f"Inspect with: aifactory inspect <run-id>[/muted]\n"
    )


def inspect_cmd(
    task_id: str = typer.Argument(..., help="Run ID or partial match to inspect."),
) -> None:
    """🔍 Deep dive into a pipeline run — memory, decisions, agent traces.

    Shows the full breakdown of a run including:
    • Task decomposition by the analyser
    • Per-agent token usage and model selection
    • Worker results with code/tool outputs
    • Judge assessment and recommendations
    • Shared context (SHARE protocol)

    Examples:
        aifactory inspect 20260409_132447
        aifactory inspect 20260409
        aifactory inspect "todo app"
    """
    setup_logging()

    dump = find_dump_fuzzy(task_id)
    if not dump:
        console.print(f"[error]No run found matching '{task_id}'.[/error]")
        console.print("[muted]Try: aifactory status  (to list recent runs)[/muted]")
        raise typer.Exit(1)

    rid = dump.get("run_id", "?")
    cfg = dump.get("config", {})
    request = cfg.get("request", "?")
    mode = cfg.get("mode", "?")
    elapsed = dump.get("elapsed_seconds", 0)
    dt_str = _format_run_datetime(rid)
    gate = dump.get("gate", "")

    # Gate-aware mode for header
    if gate == "bypass":
        mode_display = "[green]⚡ bypass[/green] (no LLM, no tools)"
    elif gate == "fast_llm":
        mode_display = "[cyan]🏃 fast[/cyan] (single LLM, no tools)"
    else:
        mode_display = f"[magenta]{mode}[/magenta]"

    # ── Header ───────────────────────────────────────────────────────
    console.print()
    console.print(
        Panel(
            f"[bold]{request}[/bold]\n\n"
            f"Run: [cyan]{rid}[/cyan]  |  Date: {dt_str}  |  "
            f"Mode: {mode_display}  |  "
            f"Time: {format_elapsed(elapsed)}",
            title="🔍 Run Inspection",
            border_style="cyan",
        )
    )

    # ── Pipeline Visualization (hierarchical tree) ───────────────────
    tasks = dump.get("tasks", [])
    task_results = dump.get("task_results", [])
    research_needed = dump.get("research_needed", False)
    research_summary = dump.get("research_summary", "")

    if tasks or task_results or gate:
        console.print("\n[heading]🗺  Pipeline[/heading]")
        tree = Tree("[bold cyan]Pipeline Run[/bold cyan]")

        # Gate node (bypass / fast_llm)
        if gate == "bypass":
            tree.add("[green]⚡ ComplexityGate → bypass (no LLM, instant)[/green]")
        elif gate == "fast_llm":
            tree.add("[cyan]🏃 ComplexityGate → fast_llm (single LLM call)[/cyan]")
        else:
            # Full pipeline — show node chain
            master_node = tree.add("[bold]master_receive[/bold]  [dim]→ seed objective[/dim]")
            master_node.add("[dim]→ judge_brief[/dim]")

            # Analyser
            analyse_node = tree.add("[bold]analyse[/bold]")
            if tasks:
                analyse_node.add(f"[dim]{len(tasks)} task(s) decomposed[/dim]")

            # Researcher (pre-analysis or execution)
            if research_needed and research_summary:
                res_node = tree.add("[bold]researcher[/bold]  [dim](execution research)[/dim]")
                res_node.add(f"[dim]{research_summary[:120]}[/dim]")
            elif research_needed:
                tree.add("[bold]researcher[/bold]  [dim](research was performed)[/dim]")

            # Execution lead + workers
            if tasks or task_results:
                lead_node = tree.add("[bold]execution_lead[/bold]")
                results_by_id = {r.get("task_id", ""): r for r in task_results}

                for i, t in enumerate(tasks, 1):
                    tid = t.get("id", f"task_{i}")
                    desc = t.get("description", tid)
                    r = results_by_id.get(tid, {})
                    model = r.get("model", "?")
                    skipped = r.get("skipped", False)
                    quality_flagged = r.get("quality_flagged", False)
                    result_text = r.get("result", "")
                    findings = r.get("findings", [])

                    if skipped:
                        w_label = (
                            f"[yellow]⊘  {tid}[/yellow]  [dim]{desc[:60]}[/dim]"
                            "  [yellow](skipped)[/yellow]"
                        )
                    elif quality_flagged:
                        w_label = (
                            f"[orange3]⚑  {tid}[/orange3]  [dim]{desc[:60]}[/dim]"
                            f"  [dim]({model})[/dim]  [orange3]⚠ flagged[/orange3]"
                        )
                    elif r:
                        w_label = (
                            f"[green]✓  {tid}[/green]  [dim]{desc[:60]}[/dim]  [dim]({model})[/dim]"
                        )
                    else:
                        w_label = f"[dim]○  {tid}  {desc[:60]}[/dim]"

                    w_node = lead_node.add(w_label)

                    # Result preview
                    if result_text and not skipped:
                        preview = result_text[:200].replace("\n", " ")
                        if len(result_text) > 200:
                            preview += f"… ({len(result_text)} chars)"
                        w_node.add(f"[dim]{preview}[/dim]")

                    # Findings facts
                    if findings:
                        f_node = w_node.add(f"[cyan]FINDINGS ({len(findings)})[/cyan]")
                        for fact in findings[:4]:
                            f_node.add(f"[dim]- {fact[:100]}[/dim]")

            # Judge
            jm = dump.get("judge_metrics", {})
            j_status = jm.get("overall_status", "")
            j_label = "[bold]judge_review[/bold]"
            if j_status == "successful":
                j_label += "  [green]✓ successful[/green]"
            elif j_status == "partial":
                j_label += "  [yellow]◐ partial[/yellow]"
            elif j_status:
                j_label += f"  [dim]{j_status}[/dim]"
            tree.add(j_label)

            tree.add("[bold]master_respond[/bold]  [dim]→ final synthesis[/dim]")

        console.print(tree)

    # ── Agent Token Breakdown ────────────────────────────────────────
    agents = dump.get("agents", [])
    if agents:
        console.print("\n[heading]🤖 Agent Breakdown[/heading]")
        print_agent_table(agents)

    # ── Judge Assessment ─────────────────────────────────────────────
    assessments = dump.get("agent_assessments", [])
    if assessments:
        console.print("\n[heading]⚖️ Judge Assessments[/heading]")
        assess_table = Table(show_header=True, border_style="dim")
        assess_table.add_column("Agent", style="bold")
        assess_table.add_column("Score", justify="center")
        assess_table.add_column("Feedback", max_width=60)

        for a in assessments:
            name = a.get("agent", a.get("name", "?"))
            score = a.get("score", "?")
            feedback = a.get("feedback", a.get("comment", ""))[:60]
            score_color = "green" if isinstance(score, int | float) and score >= 7 else "yellow"
            assess_table.add_row(name, f"[{score_color}]{score}[/{score_color}]", feedback)

        console.print(assess_table)

    judge_metrics = dump.get("judge_metrics", {})
    if judge_metrics:
        console.print("\n[heading]📊 Judge Metrics[/heading]")
        for k, v in judge_metrics.items():
            console.print(f"  {k}: {v}")

    # ── Shared Context (SHARE protocol) ──────────────────────────────
    shared = dump.get("shared_context_final", {})
    if shared:
        console.print("\n[heading]🔗 Shared Context (SHARE protocol)[/heading]")
        for k, v in shared.items():
            val_str = str(v)[:100]
            console.print(f"  [bold]{k}:[/bold] {val_str}")

    # ── Final Response ───────────────────────────────────────────────
    final = dump.get("final_response", "")
    if final:
        console.print()
        console.print(
            Panel(
                final[:1000] + ("..." if len(final) > 1000 else ""),
                title="✅ Final Response",
                border_style="green",
            )
        )

    console.print(f"\n[muted]Resume: aifactory resume {rid}[/muted]\n")
