"""
aifactory agents list  — Show agent hierarchy and live status.
aifactory agents roles — Show agent roles and model assignments.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import typer
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from aifactory.cli._helpers import (
    console,
    find_dump_fuzzy,
    format_cost_usd,
    format_elapsed,
    setup_logging,
)

agents_app = typer.Typer(
    help="🤖 Agent management — list roles, hierarchy, and live status.",
    no_args_is_help=True,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _resolve_tier(agent_name: str, role: str, role_tier_map: dict[str, str]) -> str:
    """Resolve the tier for an agent given its name, role, and the role_tier_map."""
    name_lower = agent_name.lower()
    role_lower = role.lower()

    # Worker agents are always budget
    if "worker" in name_lower:
        return "budget"

    # Direct lookup by name
    if name_lower in role_tier_map:
        return role_tier_map[name_lower]

    # Direct lookup by role (lowercase)
    for key, tier in role_tier_map.items():
        if key == role_lower:
            return tier

    # Name-based heuristics for the dynamic pipeline agents
    if "master" in name_lower:
        return "premium"
    if "researcher" in name_lower:
        return "premium"
    if "judge" in name_lower:
        return "standard"
    if "analyser" in name_lower or "analyzer" in name_lower:
        return "standard"
    if "lead" in name_lower:
        return "standard"

    # Fuzzy match: check if any role_tier_map key is a substring of name or role
    for key, tier in role_tier_map.items():
        if key in name_lower or key in role_lower:
            return tier

    return "budget"


_ROLE_ICONS = {
    "master": "👑",
    "judge": "⚖️ ",
    "analyser": "🔍",
    "researcher": "🔬",
    "execution_lead": "📋",
}
_ROLE_COLORS = {
    "master": "bold magenta",
    "judge": "bold yellow",
    "analyser": "bold cyan",
    "researcher": "dim",
    "execution_lead": "bold green",
}


def _role_icon(name: str) -> str:
    lower = name.lower()
    for key, icon in _ROLE_ICONS.items():
        if key in lower:
            return icon
    return "🔧" if "worker" in lower else "•"


def _role_color(name: str) -> str:
    lower = name.lower()
    for key, color in _ROLE_COLORS.items():
        if key in lower:
            return color
    return "dim" if "worker" in lower else "white"


def _token_badge(tokens: int) -> str:
    if tokens == 0:
        return "[dim]0 tok[/dim]"
    if tokens < 500:
        return f"[green]{tokens:,} tok[/green]"
    if tokens < 5000:
        return f"[yellow]{tokens:,} tok[/yellow]"
    return f"[red]{tokens:,} tok[/red]"


# ── agents list ──────────────────────────────────────────────────────────────


@agents_app.command("list")
def agents_list(
    run: str = typer.Option(None, "--run", "-r", help="Show agents from a specific run ID."),
) -> None:
    """🤖 Show agent hierarchy as a tree with model assignments.

    Without --run: shows the default pipeline architecture.
    With --run:    shows the actual agents used in that run, with tokens & models.

    Examples:
        aifactory agents list
        aifactory agents list --run 20260409_100420
        aifactory agents list -r 20260409
    """
    setup_logging()

    if run:
        _agents_list_for_run(run)
    else:
        _agents_list_default()


def _agents_list_for_run(run_id: str) -> None:
    """Show the actual agent tree from a specific pipeline run."""
    dump = find_dump_fuzzy(run_id)
    if not dump:
        console.print(f"[error]No run found matching '[bold]{run_id}[/bold]'.[/error]")
        raise typer.Exit(1)

    rid = dump.get("run_id", "?")
    request = dump.get("config", {}).get("request", "?")[:60]
    mode = dump.get("config", {}).get("mode", "?")
    elapsed = dump.get("elapsed_seconds", 0)
    agents = dump.get("agents", [])

    # Header
    console.print()
    console.print(
        Panel(
            f"[bold]{request}[/bold]\n\n"
            f"Run: [cyan]{rid}[/cyan]  │  Mode: [magenta]{mode}[/magenta]  │  "
            f"Time: [yellow]{format_elapsed(elapsed)}[/yellow]  │  "
            f"Agents: [bold]{len(agents)}[/bold]",
            title="🤖 Agent Tree",
            border_style="cyan",
            expand=False,
        )
    )

    if not agents:
        console.print("[warning]No agent data in this run dump.[/warning]\n")
        return

    # Build tree from actual agents
    tree = Tree(f"[bold cyan]🏭 Run {rid}[/bold cyan]  [dim]({mode})[/dim]")

    # Separate core agents and workers
    core_agents = [a for a in agents if "worker" not in a.get("name", "").lower()]
    workers = [a for a in agents if "worker" in a.get("name", "").lower()]

    for a in core_agents:
        name = a.get("name", "?")
        model = a.get("model", "?")
        tokens = a.get("total_tokens", 0)
        icon = _role_icon(name)
        color = _role_color(name)
        model_short = model.split("/")[-1]

        node_label = (
            f"[{color}]{icon} {name}[/{color}]  [dim]({model_short})[/dim]  {_token_badge(tokens)}"
        )

        # If this is execution_lead, add workers as children
        if "execution_lead" in name.lower() or "lead" in name.lower():
            lead_branch = tree.add(node_label)
            for w in workers:
                wname = w.get("name", "?")
                wmodel = w.get("model", "?").split("/")[-1]
                wtokens = w.get("total_tokens", 0)
                lead_branch.add(
                    f"[dim]🔧 {wname}[/dim]  [dim]({wmodel})[/dim]  {_token_badge(wtokens)}"
                )
        else:
            tree.add(node_label)

    # If no execution_lead found, add workers at root
    if not any("lead" in a.get("name", "").lower() for a in core_agents) and workers:
        for w in workers:
            wname = w.get("name", "?")
            wmodel = w.get("model", "?").split("/")[-1]
            wtokens = w.get("total_tokens", 0)
            tree.add(f"[dim]🔧 {wname}[/dim]  [dim]({wmodel})[/dim]  {_token_badge(wtokens)}")

    console.print()
    console.print(tree)

    # Totals summary
    total_tokens = sum(a.get("total_tokens", 0) for a in agents)
    total_cost = 0.0
    for a in agents:
        cost_str = format_cost_usd(a.get("total_tokens", 0), a.get("model", ""))
        with contextlib.suppress(ValueError):
            total_cost += float(cost_str.replace("$", ""))

    console.print(
        f"\n  [bold]Total:[/bold] {total_tokens:,} tokens  │  "
        f"[green]${total_cost:.4f}[/green]  │  "
        f"{len(agents)} agents\n"
    )


def _agents_list_default() -> None:
    """Show the default pipeline agent hierarchy from config."""
    import yaml

    yaml_path = Path(__file__).resolve().parent.parent / "config" / "models.yaml"
    if not yaml_path.exists():
        console.print("[error]models.yaml not found.[/error]")
        raise typer.Exit(1)

    registry = yaml.safe_load(yaml_path.read_text())
    role_tier_map = registry.get("role_tier_map", {})
    metadata = registry.get("metadata", {})

    from aifactory.config.settings import get_settings

    settings = get_settings()
    tier_defaults = {
        "premium": settings.premium_model,
        "standard": settings.standard_model,
        "budget": settings.budget_model,
    }

    def _model_display(role: str) -> str:
        tier = role_tier_map.get(role, "budget")
        model_id = tier_defaults.get(tier, "?")
        return metadata.get(model_id, {}).get("display", model_id.split("/")[-1])

    # Build hierarchy tree
    tree = Tree(
        f"[bold cyan]🏭 AI Factory Pipeline[/bold cyan]  "
        f"[dim](mode: {settings.premium_model.split('/')[-1]} / "
        f"{settings.standard_model.split('/')[-1]} / "
        f"{settings.budget_model.split('/')[-1]})[/dim]"
    )

    tree.add(
        f"[bold magenta]👑 master[/bold magenta]  "
        f"[dim]({_model_display('master')})[/dim]  — orchestrates pipeline"
    )
    tree.add(
        f"[bold yellow]⚖️  judge[/bold yellow]  "
        f"[dim]({_model_display('judge')})[/dim]  — quality & scoring"
    )
    tree.add(
        f"[bold cyan]🔍 analyser[/bold cyan]  "
        f"[dim]({_model_display('analyser')})[/dim]  — task decomposition"
    )
    tree.add(
        f"[dim]🔬 researcher[/dim]  "
        f"[dim]({_model_display('researcher')})[/dim]  — pre-analysis research (conditional)"
    )

    lead_branch = tree.add(
        f"[bold green]📋 execution_lead[/bold green]  "
        f"[dim]({_model_display('execution_lead')})[/dim]  — dispatches workers"
    )
    lead_branch.add(
        f"[dim]🔧 Worker-task_1[/dim]  "
        f"[dim]({_model_display('worker')})[/dim]  — dynamically spawned"
    )
    lead_branch.add("[dim]🔧 Worker-task_2[/dim]  [dim]...[/dim]")
    lead_branch.add(
        f"[dim]🔧 Worker-task_N[/dim]  [dim](up to {settings.max_subordinates_per_lead} per lead)[/dim]"  # noqa: E501
    )

    console.print()
    console.print(tree)
    console.print()
    console.print(
        "[muted]Workers are spawned dynamically based on task decomposition.\n"
        "Use [bold]aifactory agents list --run <run-id>[/bold] to see a specific run's agents.[/muted]\n"  # noqa: E501
    )


@agents_app.command("roles")
def agents_roles(
    run: str = typer.Option(None, "--run", "-r", help="Show roles from a specific run ID."),
) -> None:
    """📋 Show agent roles and model assignments.

    Without --run: shows default role→tier→model mapping from config.
    With --run:    shows actual models used per agent in that run.

    Examples:
        aifactory agents roles
        aifactory agents roles --run 20260409_100420
    """
    setup_logging()

    if run:
        _agents_roles_for_run(run)
    else:
        _agents_roles_default()


def _agents_roles_for_run(run_id: str) -> None:
    """Show actual role/model mapping from a specific run."""
    import yaml

    dump = find_dump_fuzzy(run_id)
    if not dump:
        console.print(f"[error]No run found matching '[bold]{run_id}[/bold]'.[/error]")
        raise typer.Exit(1)

    rid = dump.get("run_id", "?")
    request = dump.get("config", {}).get("request", "?")[:60]
    agents = dump.get("agents", [])

    # Load tier map to determine tier for each agent
    yaml_path = Path(__file__).resolve().parent.parent / "config" / "models.yaml"
    role_tier_map: dict[str, str] = {}
    if yaml_path.exists():
        registry = yaml.safe_load(yaml_path.read_text())
        role_tier_map = registry.get("role_tier_map", {})

    console.print(f"\n[heading]🤖 Agent Roles — Run {rid}[/heading]")
    console.print(f"[muted]Task: {request}[/muted]\n")

    if not agents:
        console.print("[warning]No agent data in this run dump.[/warning]\n")
        return

    tier_colors = {"premium": "magenta", "standard": "cyan", "budget": "green"}

    table = Table(
        title=f"🤖 Agents in Run [cyan]{rid}[/cyan]",
        show_header=True,
        border_style="dim",
        show_lines=True,
    )
    table.add_column("Agent", style="bold", min_width=18)
    table.add_column("Role", style="dim")
    table.add_column("Tier", justify="center")
    table.add_column("Model", style="cyan")
    table.add_column("Tokens", justify="right")
    table.add_column("Est. Cost", justify="right", style="green")

    total_tokens = 0
    total_cost = 0.0

    for a in agents:
        name = a.get("name", "?")
        role = a.get("role", name.split("-")[0].lower() if "-" in name else name.lower())
        model = a.get("model", "?")
        tokens = a.get("total_tokens", 0)
        total_tokens += tokens

        # Resolve tier — match by name then by role_tier_map keys
        tier = _resolve_tier(name, role, role_tier_map)
        color = tier_colors.get(tier, "white")
        tier_styled = f"[{color}]{tier}[/{color}]"

        cost_str = format_cost_usd(tokens, model)
        with contextlib.suppress(ValueError):
            total_cost += float(cost_str.replace("$", ""))

        model_short = model.split("/")[-1]
        table.add_row(name, role, tier_styled, model_short, f"{tokens:,}", cost_str)

    table.add_row(
        f"[bold]{len(agents)} agents[/bold]",
        "",
        "",
        "",
        f"[bold]{total_tokens:,}[/bold]",
        f"[bold green]${total_cost:.4f}[/bold green]",
    )

    console.print(table)
    console.print()


def _agents_roles_default() -> None:
    """Show default role→tier→model mapping from config."""
    import yaml

    yaml_path = Path(__file__).resolve().parent.parent / "config" / "models.yaml"
    registry = yaml.safe_load(yaml_path.read_text())
    role_tier_map = registry.get("role_tier_map", {})
    allowed = registry.get("allowed_models", {})
    metadata = registry.get("metadata", {})

    from aifactory.config.settings import get_settings

    settings = get_settings()
    tier_defaults = {
        "premium": settings.premium_model,
        "standard": settings.standard_model,
        "budget": settings.budget_model,
    }

    table = Table(
        title="🤖 Agent Roles & Models",
        show_header=True,
        border_style="dim",
    )
    table.add_column("Role", style="bold", min_width=20)
    table.add_column("Tier", style="cyan", justify="center")
    table.add_column("Default Model")
    table.add_column("Allowed Models", style="dim", max_width=50)

    tier_colors = {"premium": "magenta", "standard": "cyan", "budget": "green"}

    for role_name, tier in role_tier_map.items():
        model_id = tier_defaults.get(tier, "?")
        display = metadata.get(model_id, {}).get("display", model_id.split("/")[-1])
        color = tier_colors.get(tier, "white")
        tier_styled = f"[{color}]{tier}[/{color}]"

        role_models = []
        if role_name in allowed:
            role_data = allowed[role_name]
            models_list = role_data.get("models", []) if isinstance(role_data, dict) else role_data
            role_models = [
                metadata.get(m, {}).get("display", m.split("/")[-1]) for m in models_list
            ]
        allowed_str = ", ".join(role_models) if role_models else "[dim]tier default[/dim]"

        table.add_row(role_name, tier_styled, display, allowed_str)

    console.print()
    console.print(table)
    console.print(
        "\n[muted]Override tiers: AIFACTORY_PREMIUM_MODEL, AIFACTORY_STANDARD_MODEL, "
        "AIFACTORY_BUDGET_MODEL\n"
        "Use [bold]aifactory agents roles --run <run-id>[/bold] to see a specific run.[/muted]\n"
    )
