"""
aifactory cost today   — Budget tracking for today's runs.
aifactory cost week    — Budget tracking for this week.
aifactory cost all     — Total cost across all runs.
aifactory cost <run>   — Cost for a specific run.
"""

from __future__ import annotations

import contextlib
from collections import defaultdict
from datetime import UTC, datetime, timedelta

import typer
from rich.panel import Panel
from rich.table import Table

from aifactory.cli._helpers import (
    console,
    find_dump_fuzzy,
    format_cost_usd,
    format_elapsed,
    list_dumps,
    setup_logging,
)

cost_app = typer.Typer(
    help="💰 Budget and cost tracking.",
    no_args_is_help=True,
)


@cost_app.command("today")
def cost_today() -> None:
    """💰 Show token usage and estimated cost for today's runs."""
    setup_logging()
    today = datetime.now(UTC).strftime("%Y%m%d")
    _show_cost_for_period("Today", date_prefix=today)


@cost_app.command("week")
def cost_week() -> None:
    """💰 Show token usage and estimated cost for this week's runs."""
    setup_logging()
    now = datetime.now(UTC)
    week_start = now - timedelta(days=now.weekday())
    # Collect all days this week
    prefixes = [(week_start + timedelta(days=i)).strftime("%Y%m%d") for i in range(7)]
    _show_cost_for_period("This Week", date_prefixes=prefixes)


@cost_app.command("all")
def cost_all() -> None:
    """💰 Show total token usage and estimated cost across all runs with daily chart."""
    setup_logging()
    _show_cost_for_period("All Time", date_prefix=None, show_chart=True)


@cost_app.command("run")
def cost_run(
    run_id: str = typer.Argument(..., help="Run ID or partial match."),
) -> None:
    """💰 Show detailed cost breakdown for a specific run."""
    setup_logging()

    dump = find_dump_fuzzy(run_id)
    if not dump:
        console.print(f"[error]No run found matching '{run_id}'.[/error]")
        raise typer.Exit(1)

    rid = dump.get("run_id", "?")
    request = dump.get("config", {}).get("request", "?")[:60]
    elapsed = dump.get("elapsed_seconds", 0)

    console.print()
    console.print(
        Panel(
            f"[bold]{request}[/bold]\n\n"
            f"Run: [cyan]{rid}[/cyan]  │  Time: [yellow]{format_elapsed(elapsed)}[/yellow]",
            title="💰 Cost Breakdown",
            border_style="green",
            expand=False,
        )
    )

    agents = dump.get("agents", [])
    if not agents:
        console.print("[warning]No agent data in this run dump.[/warning]")
        return

    table = Table(show_header=True, border_style="dim", show_lines=True)
    table.add_column("Agent", style="bold")
    table.add_column("Model", style="cyan")
    table.add_column("Prompt", justify="right", style="dim")
    table.add_column("Completion", justify="right", style="dim")
    table.add_column("Total Tokens", justify="right")
    table.add_column("Est. Cost", justify="right", style="green")

    total_tokens = 0
    total_cost = 0.0

    for a in agents:
        tokens = a.get("total_tokens", 0)
        model = a.get("model", "")
        cost_str = format_cost_usd(tokens, model)
        total_tokens += tokens

        with contextlib.suppress(ValueError):
            total_cost += float(cost_str.replace("$", ""))

        # Color tokens by volume
        tok_color = "green" if tokens < 500 else "yellow" if tokens < 5000 else "red"

        table.add_row(
            a.get("name", "?"),
            model.split("/")[-1],
            str(a.get("prompt_tokens", 0)),
            str(a.get("completion_tokens", 0)),
            f"[{tok_color}]{tokens:,}[/{tok_color}]",
            cost_str,
        )

    table.add_row(
        "[bold]TOTAL[/bold]",
        "",
        "",
        "",
        f"[bold]{total_tokens:,}[/bold]",
        f"[bold green]${total_cost:.4f}[/bold green]",
    )
    console.print(table)

    # Budget context
    try:
        from aifactory.config.settings import get_settings

        settings = get_settings()
        budget = settings.max_budget_usd
        pct = (total_cost / budget * 100) if budget > 0 else 0
        color = "green" if pct < 50 else "yellow" if pct < 80 else "red"
        console.print(
            f"\n[muted]Budget: ${budget:.2f}  │  "
            f"This run: [{color}]{pct:.1f}%[/{color}] of budget[/muted]\n"
        )
    except Exception:
        console.print()


@cost_app.command("estimate")
def cost_estimate(
    task: str = typer.Argument(..., help="Task description to estimate cost for."),
    mode: str = typer.Option(
        "balanced",
        "--mode",
        "-m",
        help="Pipeline mode: best, balanced, or cost_efficient.",
    ),
) -> None:
    """💸 Estimate token usage and cost before running a task.

    Uses the complexity gate to determine which tier would handle the task,
    then shows a breakdown of expected agent calls and estimated costs.

    Examples:
        aifactory cost estimate "2 + 2"
        aifactory cost estimate "Build a FastAPI todo app" --mode best
        aifactory cost estimate "What is the capital of France?"
    """
    setup_logging()

    from aifactory.core.complexity_gate import ComplexityGate

    gate = ComplexityGate()
    decision = gate.evaluate(task)

    # Resolve mode aliases
    mode_map = {"light": "cost_efficient", "thorough": "best"}
    mode = mode_map.get(mode, mode)

    # ── Tier-based estimates ──────────────────────────────────────────
    if decision.bypass:
        tier_label = "[green]⚡ Bypass (instant)[/green]"
        tier_desc = f"Answered locally — {decision.method or 'no LLM needed'}"
        agent_breakdown: list[tuple[str, str, int]] = []
        total_tokens = 0
        total_cost_str = "$0.0000"

    elif decision.fast_llm:
        tier_label = "[cyan]🚀 Fast LLM[/cyan]"
        tier_desc = "Single master call — no tools, no multi-agent (~1-3s)"
        # Typical simple Q&A: ~800 prompt + ~400 completion
        agent_breakdown = [("master", "claude-sonnet-4-6", 1200)]
        total_tokens = 1200
        total_cost_str = format_cost_usd(total_tokens, "claude-sonnet")

    else:
        tier_label = "[magenta]🏭 Full Pipeline[/magenta]"
        tier_desc = "Multi-agent orchestration with tools (~15-30s)"

        # Token estimates per agent per mode (prompt + completion combined)
        _estimates: dict[str, dict[str, tuple[str, int]]] = {
            "best": {
                "master_receive": ("claude-opus-4-6", 800),
                "judge_brief": ("claude-opus-4-6", 600),
                "analyse": ("claude-opus-4-6", 1500),
                "execution_lead": ("claude-opus-4-6", 2000),
                "worker (avg ×3)": ("claude-opus-4-6", 4500),
                "judge_review": ("claude-opus-4-6", 1200),
                "master_respond": ("claude-opus-4-6", 800),
            },
            "balanced": {
                "master_receive": ("claude-sonnet-4-6", 600),
                "judge_brief": ("claude-haiku-4-5", 500),
                "analyse": ("claude-sonnet-4-6", 1200),
                "execution_lead": ("claude-sonnet-4-6", 1800),
                "worker (avg ×3)": ("claude-sonnet-4-6", 3500),
                "judge_review": ("claude-haiku-4-5", 800),
                "master_respond": ("claude-sonnet-4-6", 700),
            },
            "cost_efficient": {
                "master_receive": ("claude-haiku-4-5", 400),
                "judge_brief": ("claude-haiku-4-5", 350),
                "analyse": ("claude-haiku-4-5", 900),
                "execution_lead": ("claude-haiku-4-5", 1200),
                "worker (avg ×2)": ("claude-haiku-4-5", 2000),
                "judge_review": ("claude-haiku-4-5", 600),
                "master_respond": ("claude-haiku-4-5", 500),
            },
        }
        breakdown_map = _estimates.get(mode, _estimates["balanced"])
        agent_breakdown = [(name, model, tokens) for name, (model, tokens) in breakdown_map.items()]
        total_tokens = sum(t for _, _, t in agent_breakdown)
        total_cost_val = sum(
            float(format_cost_usd(t, m).replace("$", "")) for _, m, t in agent_breakdown
        )
        total_cost_str = f"${total_cost_val:.4f}"

    # ── Display ───────────────────────────────────────────────────────
    console.print()
    console.print(
        Panel(
            f"Tier: {tier_label}\n[dim]{tier_desc}[/dim]\n\n"
            f"[bold]Gate decision:[/bold] {decision.reason}"
            + (
                f"\n[bold]Answer:[/bold] {decision.answer}"
                if decision.bypass and decision.answer
                else ""
            )
            + (
                f"\n[bold]Matched:[/bold] [dim]{', '.join(decision.matched_labels)}[/dim]"
                if decision.matched_labels
                else ""
            ),
            title=f'🎯 Complexity Gate — "{task[:60]}{"..." if len(task) > 60 else ""}"',
            border_style="blue",
            expand=False,
        )
    )
    console.print()

    if agent_breakdown:
        table = Table(
            title=f"💸 Cost Estimate — {mode} mode",
            show_header=True,
            border_style="dim",
        )
        table.add_column("Component", style="bold")
        table.add_column("Model", style="cyan")
        table.add_column("Est. Tokens", justify="right")
        table.add_column("Est. Cost", justify="right", style="green")

        for name, model, tokens in agent_breakdown:
            cost = format_cost_usd(tokens, model)
            table.add_row(name, model.split("/")[-1], f"{tokens:,}", cost)

        table.add_row(
            "[bold]TOTAL[/bold]",
            "",
            f"[bold]{total_tokens:,}[/bold]",
            f"[bold green]{total_cost_str}[/bold green]",
        )
        console.print(table)
        console.print(
            "\n[muted]⚠  Estimates based on typical runs. Actual cost varies with task"
            " complexity, response length, and dynamic model selection.[/muted]\n"
        )
    else:
        console.print(
            f"  {tier_label}  │  "
            f"Tokens: [bold]{total_tokens}[/bold]  │  "
            f"Cost: [bold green]{total_cost_str}[/bold green]\n"
        )


# ── Bar chart helper ─────────────────────────────────────────────────────────

_BAR_CHARS = "▏▎▍▌▋▊▉█"


def _render_daily_chart(dumps: list[dict], grand_cost: float) -> None:
    """Render a horizontal bar chart of daily costs using Rich."""
    # Group by date
    daily: dict[str, float] = defaultdict(float)
    daily_runs: dict[str, int] = defaultdict(int)

    for d in dumps:
        rid = d.get("run_id", "")
        date_str = rid[:8] if len(rid) >= 8 else "unknown"
        # Format as YYYY-MM-DD
        try:
            date_key = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        except (IndexError, ValueError):
            date_key = date_str

        agents = d.get("agents", [])
        run_cost = 0.0
        if agents:
            for a in agents:
                tokens = a.get("total_tokens", 0)
                model = a.get("model", "")
                cost_str = format_cost_usd(tokens, model)
                with contextlib.suppress(ValueError):
                    run_cost += float(cost_str.replace("$", ""))
        else:
            totals = d.get("totals", {})
            total_tokens = totals.get("total_tokens", 0)
            cost_str = format_cost_usd(total_tokens)
            with contextlib.suppress(ValueError):
                run_cost = float(cost_str.replace("$", ""))

        daily[date_key] += run_cost
        daily_runs[date_key] += 1

    if not daily:
        return

    # Sort by date
    sorted_days = sorted(daily.items())
    max_cost = max(daily.values()) if daily.values() else 1.0
    bar_width = 40  # max bar width in chars

    console.print("\n[heading]📊 Daily Cost Chart[/heading]\n")

    for date_key, cost in sorted_days:
        runs = daily_runs[date_key]
        # Calculate bar length (proportional)
        ratio = cost / max_cost if max_cost > 0 else 0
        full_blocks = int(ratio * bar_width)
        remainder = (ratio * bar_width) - full_blocks
        partial_idx = int(remainder * len(_BAR_CHARS))
        partial_idx = min(partial_idx, len(_BAR_CHARS) - 1)

        bar = "█" * full_blocks
        if full_blocks < bar_width and partial_idx > 0:
            bar += _BAR_CHARS[partial_idx]

        # Color the bar based on proportion of total budget
        if cost < 0.01:
            bar_color = "green"
        elif cost < 0.05:
            bar_color = "cyan"
        elif cost < 0.10:
            bar_color = "yellow"
        else:
            bar_color = "red"

        cost_label = f"${cost:.4f}"
        runs_label = f"({runs} run{'s' if runs != 1 else ''})"

        console.print(
            f"  [bold]{date_key}[/bold]  [{bar_color}]{bar}[/{bar_color}]  "
            f"[green]{cost_label}[/green]  [dim]{runs_label}[/dim]"
        )

    console.print()


# ── Period summary ───────────────────────────────────────────────────────────


def _show_cost_for_period(
    label: str,
    *,
    date_prefix: str | None = None,
    date_prefixes: list[str] | None = None,
    show_chart: bool = False,
) -> None:
    """Show aggregated cost for runs matching date filter."""
    dumps = list_dumps(limit=200)

    # Filter by date
    if date_prefix:
        dumps = [d for d in dumps if d.get("run_id", "").startswith(date_prefix)]
    elif date_prefixes:
        dumps = [d for d in dumps if any(d.get("run_id", "").startswith(p) for p in date_prefixes)]

    if not dumps:
        console.print(f"\n[warning]No runs found for {label}.[/warning]\n")
        return

    console.print(f"\n[heading]💰 Cost Summary — {label}[/heading]\n")

    table = Table(show_header=True, border_style="dim")
    table.add_column("Run ID", style="bold cyan")
    table.add_column("Task", max_width=40)
    table.add_column("Mode", style="magenta", justify="center")
    table.add_column("Tokens", justify="right")
    table.add_column("Time", justify="right", style="dim")
    table.add_column("Est. Cost", justify="right", style="green")

    grand_tokens = 0
    grand_cost = 0.0
    grand_time = 0.0

    for d in dumps:
        rid = d.get("run_id", "?")
        request = d.get("config", {}).get("request", "")[:40]
        mode = d.get("config", {}).get("mode", "?")
        elapsed = d.get("elapsed_seconds", 0)
        totals = d.get("totals", {})
        total_tokens = totals.get("total_tokens", 0)

        # Estimate cost from agents
        agents = d.get("agents", [])
        run_cost = 0.0
        if agents:
            for a in agents:
                tokens = a.get("total_tokens", 0)
                model = a.get("model", "")
                cost_str = format_cost_usd(tokens, model)
                with contextlib.suppress(ValueError):
                    run_cost += float(cost_str.replace("$", ""))
        else:
            # Fallback: rough estimate
            cost_str = format_cost_usd(total_tokens)
            with contextlib.suppress(ValueError):
                run_cost = float(cost_str.replace("$", ""))

        grand_tokens += total_tokens
        grand_cost += run_cost
        grand_time += elapsed

        # Color tokens by volume
        tok_color = "green" if total_tokens < 1000 else "yellow" if total_tokens < 10000 else "red"

        table.add_row(
            rid,
            request or "[dim]—[/dim]",
            mode,
            f"[{tok_color}]{total_tokens:,}[/{tok_color}]" if total_tokens else "—",
            format_elapsed(elapsed),
            f"${run_cost:.4f}",
        )

    table.add_row(
        f"[bold]{len(dumps)} runs[/bold]",
        "",
        "",
        f"[bold]{grand_tokens:,}[/bold]",
        f"[bold]{format_elapsed(grand_time)}[/bold]",
        f"[bold green]${grand_cost:.4f}[/bold green]",
    )

    console.print(table)

    # Daily bar chart (for "all" or "week" views)
    if show_chart and len(dumps) > 1:
        _render_daily_chart(dumps, grand_cost)

    # Budget context
    try:
        from aifactory.config.settings import get_settings

        settings = get_settings()
        budget = settings.max_budget_usd
        pct = (grand_cost / budget * 100) if budget > 0 else 0
        color = "green" if pct < 50 else "yellow" if pct < 80 else "red"
        console.print(
            f"\n[muted]Budget: ${budget:.2f}  │  "
            f"{label}: [{color}]${grand_cost:.4f} ({pct:.1f}%)[/{color}][/muted]\n"
        )
    except Exception:
        console.print()
