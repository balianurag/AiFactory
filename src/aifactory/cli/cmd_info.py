"""
aifactory info         — Show configuration and API key status.
aifactory memory-stats — Show memory system statistics.
aifactory gate         — Preview complexity gate decision for a task.
aifactory models list  — Show model registry.
aifactory models tiers — Show tier assignments.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import typer
from rich.table import Table

from aifactory.cli._helpers import console, setup_logging

models_app = typer.Typer(help="📦 Model registry commands.", no_args_is_help=True)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _resolve_tier_info(agent_name: str, role: str, role_tier_map: dict[str, str]) -> str:
    """Resolve the tier for an agent given its name, role, and the role_tier_map."""
    name_lower = agent_name.lower()

    if "worker" in name_lower:
        return "budget"
    if name_lower in role_tier_map:
        return role_tier_map[name_lower]
    for key, tier in role_tier_map.items():
        if key == role.lower():
            return tier
    if "master" in name_lower or "researcher" in name_lower:
        return "premium"
    if "judge" in name_lower or "analyser" in name_lower or "lead" in name_lower:
        return "standard"
    for key, tier in role_tier_map.items():
        if key in name_lower or key in role.lower():
            return tier
    return "budget"


def info() -> None:
    """📋 Show AI Factory configuration and API key status.

    Displays current model assignments, budget limits, memory settings,
    and which API keys are configured.

    Example:
        aifactory info
    """
    setup_logging()

    from aifactory.config.settings import get_settings

    settings = get_settings()

    table = Table(title="🏭 AI Factory Configuration", show_header=True, border_style="dim")
    table.add_column("Setting", style="bold")
    table.add_column("Value")

    table.add_row("Premium Model", settings.premium_model)
    table.add_row("Standard Model", settings.standard_model)
    table.add_row("Budget Model", settings.budget_model)
    table.add_row("Max Budget (USD)", f"${settings.max_budget_usd:.2f}")
    table.add_row("Max Iterations", str(settings.max_iterations))
    table.add_row("Max Subordinates/Lead", str(settings.max_subordinates_per_lead))
    table.add_row("Qdrant URL", settings.qdrant_url)
    table.add_row("Memory Promotion Interval", str(settings.memory_promotion_interval))
    table.add_row("Sandbox Backend", settings.sandbox_backend)

    # API key status — only show configured keys (skip missing ones for clean UX)
    keys = {
        "OpenAI": bool(settings.openai_api_key),
        "Anthropic": bool(settings.anthropic_api_key),
        "xAI": bool(settings.xai_api_key),
        "Google": bool(settings.google_api_key),
        "OpenRouter": bool(settings.openrouter_api_key),
    }
    configured = [f"[green]{k} ✓[/green]" for k, v in keys.items() if v]
    key_status = "  ".join(configured) if configured else "[red]None configured[/red]"
    table.add_row("API Keys", key_status)

    # Integrations — same: only show configured
    integrations: list[str] = []
    composio_ok = bool((settings.composio_api_key or "").strip())
    if composio_ok:
        integrations.append("[green]Composio ✓[/green]")
    e2b_ok = bool((settings.e2b_api_key or "").strip())
    if e2b_ok:
        integrations.append("[green]E2B ✓[/green]")
    int_status = "  ".join(integrations) if integrations else "[dim]None configured[/dim]"
    table.add_row("Integrations", int_status)

    console.print()
    console.print(table)
    console.print()


def memory_stats() -> None:
    """🧠 Show memory system statistics (Qdrant collections)."""
    setup_logging()

    try:
        from aifactory.core.memory_router import MemoryRouter

        router = MemoryRouter()
        stats = router.get_stats()

        table = Table(title="🧠 Memory Stats", border_style="dim")
        table.add_column("Layer", style="bold")
        table.add_column("Entries", justify="right")

        total = 0
        for level, count in stats.items():
            total += count
            table.add_row(level.title(), str(count))
        table.add_row("[bold]Total[/bold]", f"[bold]{total}[/bold]")

        console.print()
        console.print(table)
        console.print()
    except Exception as e:
        console.print(f"[error]Could not connect to memory system: {e}[/error]")
        console.print("[muted]Is Qdrant running? Check qdrant_url in settings.[/muted]")


def gate(
    task: str = typer.Argument(..., help="Task to evaluate."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show pattern scores."),
) -> None:
    """🚦 Preview the complexity gate decision for a task (dry run).

    Shows which routing tier would handle the task without executing anything:

      ⚡ bypass    — pure math or static fact (0 tokens, <1ms)
      🚀 fast_llm  — single LLM call, no tools (~1-3s)
      🏭 full      — 8-node pipeline with tools (~15-30s)

    Examples:
        aifactory gate "2 + 2"
        aifactory gate "What is the capital of France?"
        aifactory gate "Build a FastAPI todo app" --verbose
    """
    setup_logging()

    from rich.panel import Panel
    from rich.table import Table

    from aifactory.core.complexity_gate import (
        ComplexityGate,
        _score_qa,
        _score_tools,
    )

    g = ComplexityGate()
    decision = g.evaluate(task)

    if decision.bypass:
        tier_icon, tier_name, tier_color = "⚡", "BYPASS", "green"
        tier_desc = "Answered locally — no LLM needed"
    elif decision.fast_llm:
        tier_icon, tier_name, tier_color = "🚀", "FAST LLM", "cyan"
        tier_desc = "Single master LLM call — no tools, no multi-agent (~1-3s)"
    else:
        tier_icon, tier_name, tier_color = "🏭", "FULL PIPELINE", "magenta"
        tier_desc = "8-node multi-agent pipeline with tools (~15-30s)"

    body = (
        f"[bold {tier_color}]{tier_icon}  {tier_name}[/bold {tier_color}]\n\n"
        f"[dim]{tier_desc}[/dim]\n\n"
        f"[bold]Reason:[/bold]     {decision.reason}\n"
        + (f"[bold]Method:[/bold]     {decision.method}\n" if decision.method else "")
        + f"[bold]Confidence:[/bold] {decision.confidence:.2f}"
        + (
            f"\n[bold]Answer:[/bold]     {decision.answer}"
            if decision.bypass and decision.answer
            else ""
        )
    )

    console.print()
    console.print(
        Panel(
            body,
            title=f'🚦 Complexity Gate — "{task[:60]}{"..." if len(task) > 60 else ""}"',
            border_style=tier_color,
            expand=False,
        )
    )

    if decision.matched_labels:
        console.print(f"\n[bold]Matched patterns:[/bold] {', '.join(decision.matched_labels)}")

    if verbose:
        tools_score, tools_labels = _score_tools(task)
        qa_score, qa_labels = _score_qa(task)

        console.print()
        table = Table(title="Pattern Scores", show_header=True, border_style="dim")
        table.add_column("Side", style="bold")
        table.add_column("Score", justify="right")
        table.add_column("Matched Labels")

        table.add_row(
            "[red]needs_tools[/red]",
            f"{tools_score:.2f}",
            ", ".join(tools_labels) or "[dim]none[/dim]",
        )
        table.add_row(
            "[cyan]simple_qa[/cyan]",
            f"{qa_score:.2f}",
            ", ".join(qa_labels) or "[dim]none[/dim]",
        )
        console.print(table)

    console.print()


@models_app.command("list")
def models_list(
    role: str = typer.Option(None, "--role", "-r", help="Filter by agent role."),
) -> None:
    """📦 List the curated allowed models, optionally filtered by role."""
    setup_logging()

    import yaml

    yaml_path = Path(__file__).resolve().parent.parent / "config" / "models.yaml"
    registry = yaml.safe_load(yaml_path.read_text())
    allowed = registry.get("allowed_models", {})
    metadata = registry.get("metadata", {})

    if role:
        role_key = role.lower().replace("-", "_").replace(" ", "_")
        if role_key not in allowed:
            console.print(f"[error]Unknown role '{role}'. Available:[/error]")
            for r in allowed:
                console.print(f"  • {r}")
            raise typer.Exit(1)
        roles_to_show = {role_key: allowed[role_key]}
    else:
        roles_to_show = allowed

    console.print()
    for role_name, role_data in roles_to_show.items():
        models_data = role_data.get("models", []) if isinstance(role_data, dict) else role_data
        desc = role_data.get("description", "") if isinstance(role_data, dict) else ""

        table = Table(
            title=f"[bold cyan]{role_name.upper()}[/bold cyan]  [dim]{desc}[/dim]",
            show_header=True,
            border_style="dim",
        )
        table.add_column("Model ID", style="dim", min_width=20)
        table.add_column("Display", style="bold")
        table.add_column("Provider", style="cyan")
        table.add_column("Context", justify="right")
        table.add_column("$/1M in", justify="right", style="green")
        table.add_column("$/1M out", justify="right", style="yellow")
        table.add_column("Strength", style="dim", max_width=40)

        for model_id in models_data:
            meta = metadata.get(model_id, {})
            display = meta.get("display", model_id.split("/")[-1])
            provider = meta.get("provider", "?")
            ctx = f"{meta.get('context_k', '?')}k"
            inp = f"${meta.get('input_per_1m', '?'):.3f}" if meta.get("input_per_1m") else "?"
            out = f"${meta.get('output_per_1m', '?'):.2f}" if meta.get("output_per_1m") else "?"
            strength = meta.get("strength", "")
            short_id = model_id.replace("openrouter/", "")
            table.add_row(short_id, display, provider, ctx, inp, out, strength)

        console.print(table)
        console.print()


@models_app.command("tiers")
def models_tiers(
    run: str = typer.Option(None, "--run", "-r", help="Show model tiers from a specific run ID."),
) -> None:
    """📊 Show model tier assignments.

    Without --run: shows the default tier configuration.
    With --run:    shows the actual models used in that run mapped to their tiers.

    Examples:
        aifactory models tiers
        aifactory models tiers --run 20260409_100420
    """
    setup_logging()

    if run:
        _models_tiers_for_run(run)
    else:
        _models_tiers_default()


def _models_tiers_for_run(run_id: str) -> None:
    """Show actual model→tier mapping from a specific run."""
    import yaml

    from aifactory.cli._helpers import find_dump_fuzzy, format_cost_usd

    dump = find_dump_fuzzy(run_id)
    if not dump:
        console.print(f"[error]No run found matching '[bold]{run_id}[/bold]'.[/error]")
        raise typer.Exit(1)

    rid = dump.get("run_id", "?")
    agents = dump.get("agents", [])
    # Load tier map
    yaml_path = Path(__file__).resolve().parent.parent / "config" / "models.yaml"
    role_tier_map: dict[str, str] = {}
    if yaml_path.exists():
        registry = yaml.safe_load(yaml_path.read_text())
        role_tier_map = registry.get("role_tier_map", {})

    console.print(f"\n[heading]📊 Model Tiers — Run {rid}[/heading]\n")

    if not agents:
        console.print("[warning]No agent data in this run dump.[/warning]\n")
        return

    tier_colors = {"premium": "magenta", "standard": "cyan", "budget": "green"}

    # Group agents by tier
    tier_groups: dict[str, list[dict]] = {"premium": [], "standard": [], "budget": []}
    for a in agents:
        name = a.get("name", "?")
        role = a.get("role", name.split("-")[0].lower() if "-" in name else name.lower())
        tier = _resolve_tier_info(name, role, role_tier_map)
        tier_groups.setdefault(tier, []).append(a)

    table = Table(
        title=f"Model Tiers — Run [cyan]{rid}[/cyan]",
        show_header=True,
        border_style="dim",
        show_lines=True,
    )
    table.add_column("Tier", style="bold", min_width=10)
    table.add_column("Model", style="cyan")
    table.add_column("Agents", min_width=30)
    table.add_column("Total Tokens", justify="right")
    table.add_column("Est. Cost", justify="right", style="green")

    grand_tokens = 0
    grand_cost = 0.0

    for tier in ("premium", "standard", "budget"):
        group = tier_groups.get(tier, [])
        if not group:
            continue

        color = tier_colors.get(tier, "white")
        tier_styled = f"[{color}]{tier}[/{color}]"

        # Group by model within tier
        model_map: dict[str, list[dict]] = {}
        for a in group:
            model = a.get("model", "?")
            model_map.setdefault(model, []).append(a)

        for model, model_agents in model_map.items():
            agent_names = ", ".join(a.get("name", "?") for a in model_agents)
            tokens = sum(a.get("total_tokens", 0) for a in model_agents)
            grand_tokens += tokens

            cost = 0.0
            for a in model_agents:
                cost_str = format_cost_usd(a.get("total_tokens", 0), model)
                with contextlib.suppress(ValueError):
                    cost += float(cost_str.replace("$", ""))
            grand_cost += cost

            model_short = model.split("/")[-1]
            table.add_row(tier_styled, model_short, agent_names, f"{tokens:,}", f"${cost:.4f}")

    table.add_row(
        "[bold]TOTAL[/bold]",
        "",
        "",
        f"[bold]{grand_tokens:,}[/bold]",
        f"[bold green]${grand_cost:.4f}[/bold green]",
    )

    console.print(table)
    console.print()


def _models_tiers_default() -> None:
    """Show default model tier assignments."""
    from aifactory.config.settings import get_settings

    settings = get_settings()

    table = Table(title="Model Tier Assignments", show_header=True, border_style="dim")
    table.add_column("Tier", style="bold")
    table.add_column("Active Model")
    table.add_column("Used For")

    table.add_row(
        "[magenta]premium[/magenta]",
        settings.premium_model,
        "Master, Researcher, PM, Assessors",
    )
    table.add_row(
        "[cyan]standard[/cyan]",
        settings.standard_model,
        "Judge, Analyser, Execution Lead",
    )
    table.add_row(
        "[green]budget[/green]",
        settings.budget_model,
        "Workers (dynamic)",
    )

    console.print()
    console.print(table)
    console.print(
        "\n[muted]Override: AIFACTORY_PREMIUM_MODEL, AIFACTORY_STANDARD_MODEL, "
        "AIFACTORY_BUDGET_MODEL\n"
        "Use [bold]aifactory models tiers --run <run-id>[/bold] to see a specific run.[/muted]\n"
    )
