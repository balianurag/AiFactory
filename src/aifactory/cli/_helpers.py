"""Shared helpers for the AI Factory CLI — console, logging, dump utilities."""

from __future__ import annotations

import json
import logging
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Suppress known noisy library warnings that are irrelevant to end users
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_core")
warnings.filterwarnings("ignore", message=".*Pydantic V1.*")

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# ── Constants ────────────────────────────────────────────────────────────────

DUMPS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "dumps"
QUIT_WORDS = frozenset({"quit", "exit", "done", "bye", "q"})

# ── Rich Console ─────────────────────────────────────────────────────────────

_theme = Theme(
    {
        "info": "dim cyan",
        "warning": "bold yellow",
        "error": "bold red",
        "success": "bold green",
        "heading": "bold cyan",
        "muted": "dim",
        "accent": "bold magenta",
    }
)

console = Console(theme=_theme)


def setup_logging(level: str = "INFO") -> None:
    """Configure rich logging for the CLI.

    Normal mode (level=INFO): only warnings and errors reach the user.
    All internal plumbing (Qdrant, PostgreSQL, pipeline nodes, LangChain,
    run dumps) is suppressed so the terminal stays clean.
    Verbose mode (level=DEBUG): everything is shown.
    """
    is_verbose = level.upper() == "DEBUG"
    root_level = logging.DEBUG if is_verbose else logging.WARNING

    logging.basicConfig(
        level=root_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True)],
        force=True,
    )

    if is_verbose:
        # In verbose mode let everything through except the noisiest libraries
        for name in ("httpx", "httpcore", "urllib3"):
            logging.getLogger(name).setLevel(logging.WARNING)
        return

    # ── Suppress all internal / infrastructure loggers in normal mode ─────────
    _silent = [
        # Noisy libraries
        "httpx",
        "httpcore",
        "litellm",
        "LiteLLM",
        "urllib3",
        "openai",
        "anthropic",
        "langchain",
        "langchain_core",
        "langsmith",
        "langgraph",
        # Internal aifactory plumbing
        "aifactory.core.qdrant_pool",
        "aifactory.core.database",
        "aifactory.core.memory_router",
        "aifactory.core.memory_bridge",
        "aifactory.core.embed_provider",
        "aifactory.core.semantic_cache",
        "aifactory.core.rl_scorer",
        "aifactory.core.run_store",
        "aifactory.core.model_calibrator",
        "aifactory.core.retry",
        "aifactory.core.rate_limiter",
        "aifactory.graphs.dynamic_pipeline",
        "aifactory.graphs.single_team",
        "aifactory.core.agent_factory",
        "aifactory.core.base_agent",
        "aifactory.core.complexity_gate",
        "aifactory.core.sandbox",
        "aifactory.core.sandbox_pool",
        "aifactory.mcp",
    ]
    for name in _silent:
        logging.getLogger(name).setLevel(logging.ERROR)


# ── Dump I/O ─────────────────────────────────────────────────────────────────


def list_dumps(limit: int = 20) -> list[dict[str, Any]]:
    """
    Load the N most-recent run dumps from dumps/runs/, newest first.

    Returns list of dicts with at least: run_id, timestamp, config, elapsed_seconds.
    """
    runs_dir = DUMPS_DIR / "runs"
    if not runs_dir.exists():
        return []

    files = sorted(runs_dir.glob("*.json"), reverse=True)[:limit]
    dumps: list[dict[str, Any]] = []
    for f in files:
        try:
            data = json.loads(f.read_text())
            dumps.append(data)
        except Exception:
            pass
    return dumps


def load_dump(run_id: str) -> dict[str, Any] | None:
    """Load a single dump by run_id (timestamp string like '20260409_132447')."""
    path = DUMPS_DIR / "runs" / f"{run_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def find_dump_fuzzy(query: str) -> dict[str, Any] | None:
    """
    Find a dump by partial run_id or by searching request text.

    Tries exact match first, then prefix match, then substring search.
    """
    runs_dir = DUMPS_DIR / "runs"
    if not runs_dir.exists():
        return None

    # Exact
    exact = load_dump(query)
    if exact:
        return exact

    # Prefix match (e.g. "20260409")
    candidates = sorted(runs_dir.glob(f"{query}*.json"), reverse=True)
    if candidates:
        try:
            return json.loads(candidates[0].read_text())
        except Exception:
            pass

    # Substring search in request text
    for f in sorted(runs_dir.glob("*.json"), reverse=True)[:50]:
        try:
            data = json.loads(f.read_text())
            request = data.get("config", {}).get("request", "")
            if query.lower() in request.lower():
                return data
        except Exception:
            continue

    return None


def run_id_now() -> str:
    """Generate a run_id from current UTC timestamp."""
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


# ── Display Helpers ──────────────────────────────────────────────────────────


def print_banner(text: str, style: str = "blue") -> None:
    """Print a styled panel banner."""
    console.print()
    console.print(
        Panel(
            Text.from_markup(text),
            border_style=style,
            expand=False,
            padding=(1, 2),
        )
    )
    console.print()


def print_agent_table(agents: list[dict[str, Any]]) -> None:
    """Print agent summary as a rich table with color-coded tokens and cost."""
    table = Table(
        show_header=True,
        header_style="bold magenta",
        border_style="dim",
        title="🤖 Agent Summary",
        show_lines=False,
    )
    table.add_column("Agent", style="bold")
    table.add_column("Model", style="cyan")
    table.add_column("Prompt", justify="right", style="dim")
    table.add_column("Completion", justify="right", style="dim")
    table.add_column("Total", justify="right")
    table.add_column("Cost", justify="right", style="dim green")

    total_tokens = 0
    total_cost = 0.0
    has_cost = False

    for a in agents:
        tokens = a.get("total_tokens", 0)
        total_tokens += tokens
        cost_usd = a.get("cost_usd")
        if cost_usd is not None:
            total_cost += cost_usd
            has_cost = True
            cost_str = f"${cost_usd:.4f}"
        else:
            cost_str = "—"
        # Color-code tokens by volume
        tok_color = "green" if tokens < 500 else "yellow" if tokens < 5000 else "red"
        model_short = a.get("model", "?").split("/")[-1]
        table.add_row(
            a.get("name", "?"),
            model_short,
            str(a.get("prompt_tokens", 0)),
            str(a.get("completion_tokens", 0)),
            f"[{tok_color}]{tokens:,}[/{tok_color}]",
            cost_str,
        )

    total_cost_str = f"[bold]${total_cost:.4f}[/bold]" if has_cost else "—"
    table.add_row(
        "[bold]TOTAL[/bold]",
        "",
        "",
        "",
        f"[bold]{total_tokens:,}[/bold]",
        total_cost_str,
    )
    console.print(table)


def print_task_results(task_results: list[dict[str, Any]], max_preview: int = 200) -> None:
    """Print task results with colored status indicators."""
    if not task_results:
        console.print("[muted]No tasks executed.[/muted]")
        return

    console.print(f"\n[bold cyan]📦 Tasks ({len(task_results)}):[/bold cyan]")
    for r in task_results:
        skipped = r.get("skipped", False)
        result = r.get("result", "")
        preview = result[:max_preview] + ("..." if len(result) > max_preview else "")

        if skipped:
            status_icon = "[yellow]⊘[/yellow]"
            status_text = " [yellow](skipped)[/yellow]"
        else:
            status_icon = "[green]✓[/green]"
            status_text = ""

        agent = r.get("agent", "")
        model = r.get("model", "")
        agent_info = ""
        if agent or model:
            parts = [p for p in [agent, model.split("/")[-1] if model else ""] if p]
            agent_info = f" [dim]({' / '.join(parts)})[/dim]"

        console.print(
            f"  {status_icon} [bold]{r.get('task_id', '?')}[/bold]{status_text}{agent_info}"
        )
        if preview.strip():
            # Indent the preview and dim it slightly
            for line in preview.split("\n")[:3]:
                if line.strip():
                    console.print(f"    [dim]{line.strip()}[/dim]")


def format_elapsed(seconds: float) -> str:
    """Format elapsed seconds nicely."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.0f}s"


def format_cost_usd(tokens: int, model: str = "") -> str:
    """Rough cost estimate based on token count and model."""
    # Very rough estimates — $/1M tokens (blended input+output)
    rates: dict[str, float] = {
        "gpt-4o": 7.5,
        "gpt-4o-mini": 0.3,
        "gpt-5.2": 10.0,
        "gemini-2.0-flash": 0.1,
        "gemini-2.5-pro": 5.0,
        "claude-sonnet": 9.0,
        "claude-haiku": 1.0,
    }
    rate = 1.0  # default
    for key, r in rates.items():
        if key in model.lower():
            rate = r
            break
    cost = (tokens / 1_000_000) * rate
    return f"${cost:.4f}"
