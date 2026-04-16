"""
aifactory memory stats   — Rich memory layer statistics.
aifactory memory search  — Semantic search across memory layers.
aifactory memory clear   — Clear agent or all memories.

These commands expose the 3-layer memory system (Agent / Team / Global)
to the CLI for inspection, debugging, and maintenance.
"""

from __future__ import annotations

import typer
from rich.table import Table

from aifactory.cli._helpers import console, setup_logging

memory_app = typer.Typer(
    help="🧠 Memory system commands.",
    no_args_is_help=True,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_router():
    """Create a MemoryRouter connected to the configured Qdrant instance."""
    from aifactory.core.memory_router import MemoryRouter

    return MemoryRouter()


# ── Commands ──────────────────────────────────────────────────────────────────


@memory_app.command("health")
def memory_health() -> None:
    """🩺 Check memory system health: Qdrant, embedding provider, collections.

    Shows connection status, embedding model in use, whether data will persist,
    and the size of each memory collection.

    Examples:
        aifactory memory health
    """
    setup_logging()
    from rich.panel import Panel

    console.print("\n[bold]🧠 Memory System Health Check[/bold]\n")
    issues: list[str] = []

    # ── Qdrant ───────────────────────────────────────────────────────
    console.print("[bold]Qdrant[/bold]")
    try:
        from aifactory.core.memory_router import MemoryRouter

        router = MemoryRouter()
        in_mem = getattr(router, "in_memory_mode", False)
        if in_mem:
            console.print(
                "  [yellow]⚠[/yellow]  Status: [yellow]in-memory[/yellow] — "
                "data will NOT persist between runs"
            )
            console.print("  [dim]Fix:[/dim] docker compose -f docker-compose.dbs.yml up -d qdrant")
            issues.append("Qdrant in-memory mode")
        else:
            from aifactory.config.settings import get_settings

            url = get_settings().qdrant_url
            console.print(f"  [green]✓[/green]  Status: persistent  [dim]({url})[/dim]")
    except Exception as exc:
        console.print(f"  [red]✗[/red]  Qdrant unavailable: {exc}")
        issues.append(f"Qdrant: {exc}")
        router = None  # type: ignore[assignment]

    # ── Embedding provider ────────────────────────────────────────────
    console.print("\n[bold]Embedding Provider[/bold]")
    try:
        from aifactory.core.embed_provider import resolve_provider

        provider, model_id, dims = resolve_provider()
        if provider == "stub":
            console.print(
                f"  [yellow]⚠[/yellow]  Provider: [yellow]stub (SHA-256)[/yellow]  "
                f"dim={dims} — NO semantic search"
            )
            console.print("  [dim]Fix: set OPENAI_API_KEY, VOYAGE_API_KEY, or COHERE_API_KEY[/dim]")
            issues.append("Stub embeddings — memory search is non-semantic")
        elif provider == "huggingface":
            console.print(
                f"  [green]✓[/green]  Provider: HuggingFace (local)  model={model_id}  dim={dims}"
            )
        else:
            console.print(f"  [green]✓[/green]  Provider: {provider}  model={model_id}  dim={dims}")
    except Exception as exc:
        console.print(f"  [red]✗[/red]  Embedding probe failed: {exc}")
        issues.append(f"Embedding: {exc}")

    # ── Embedding cache (Redis) ───────────────────────────────────────
    console.print("\n[bold]Embedding Cache (Redis)[/bold]")
    try:
        import os

        redis_url = os.getenv("REDIS_URL", "")
        if redis_url:
            import redis as _r

            rc = _r.from_url(redis_url, socket_connect_timeout=1)
            rc.ping()
            console.print(
                f"  [green]✓[/green]  Connected — embedding vectors cached to Redis  [dim]({redis_url})[/dim]"  # noqa: E501
            )
        else:
            console.print(
                "  [dim]—  REDIS_URL not set — embeddings recomputed each run (costs API calls)[/dim]"  # noqa: E501
            )
    except Exception as exc:
        console.print(f"  [yellow]⚠[/yellow]  Redis unavailable: {exc}")

    # ── Collection stats ──────────────────────────────────────────────
    console.print("\n[bold]Memory Collections[/bold]")
    if router is not None:
        try:
            stats = router.get_stats()
            for level_key, count in stats.items():
                label = {
                    "agent": "Agent (per-run STM)",
                    "team": "Team (shared 7d)",
                    "global": "Global (LTM)",
                }.get(level_key, level_key)
                icon = "[green]✓[/green]" if count >= 0 else "[red]✗[/red]"
                console.print(f"  {icon}  {label:<28} {count:>6} entries")
        except Exception as exc:
            console.print(f"  [dim]Could not read collection stats: {exc}[/dim]")

        # Semantic cache
        try:
            collections = {c.name for c in router.client.get_collections().collections}
            sc_name = "aifactory_semantic_cache"
            if sc_name in collections:
                info = router.client.get_collection(sc_name)
                console.print(
                    f"  [green]✓[/green]  Semantic cache               "
                    f"{info.points_count or 0:>6} entries"
                )
            else:
                console.print(
                    "  [dim]—   Semantic cache: not yet created (initialises on first pipeline run)[/dim]"  # noqa: E501
                )
        except Exception:
            pass
    else:
        console.print("  [dim]  (Qdrant unavailable — collection stats not available)[/dim]")

    # ── Summary ───────────────────────────────────────────────────────
    console.print()
    if issues:
        console.print(
            Panel(
                "\n".join(f"• {i}" for i in issues),
                title="⚠  Issues Found",
                border_style="yellow",
                expand=False,
            )
        )
    else:
        console.print("[success]✓[/success] Memory system fully operational.\n")


@memory_app.command("stats")
def memory_stats(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show per-collection details."),
) -> None:
    """📊 Show memory system statistics across all layers.

    Displays entry counts per layer (Agent / Team / Global) and overall totals.

    Examples:
        aifactory memory stats
        aifactory memory stats --verbose
    """
    setup_logging()

    try:
        router = _make_router()
        stats = router.get_stats()

        table = Table(
            title="🧠 Memory System — Layer Statistics",
            show_header=True,
            border_style="dim",
        )
        table.add_column("Layer", style="bold", min_width=12)
        table.add_column("Entries", justify="right")
        table.add_column("Description", style="dim")

        layer_info = {
            "agent": ("Agent (private)", "Per-agent STM — outputs, turn-by-turn observations"),
            "team": ("Team (shared)", "Shared working memory — SHARE: updates, promoted summaries"),
            "global": ("Global (LTM)", "Cross-project curated knowledge — stable facts"),
        }

        total = 0
        for level_key, count in stats.items():
            total += count
            name, desc = layer_info.get(level_key, (level_key.title(), ""))
            table.add_row(name, str(count), desc)

        table.add_row("[bold]Total[/bold]", f"[bold]{total}[/bold]", "")

        console.print()
        console.print(table)

        if verbose:
            _print_verbose_stats(router)

        console.print(
            "\n[muted]Use [bold]aifactory memory search[/bold] to query entries "
            "or [bold]aifactory memory clear[/bold] to prune.[/muted]\n"
        )
    except Exception as e:
        console.print(f"\n[error]Memory system unavailable: {e}[/error]")
        console.print("[muted]Is Qdrant running? Check QDRANT_URL in settings.[/muted]\n")
        raise typer.Exit(1) from e


def _print_verbose_stats(router) -> None:
    """Print per-collection Qdrant details for debugging."""
    from aifactory.core.schemas import MemoryLevel

    console.print("\n[bold]Collection Details[/bold]")
    for level in MemoryLevel:
        col_name = router._collection_name(level)
        try:
            info = router.client.get_collection(col_name)
            count = info.points_count or 0
            status = str(getattr(info, "status", "?"))
            vec_size = (
                info.config.params.vectors.size if hasattr(info.config.params, "vectors") else "?"
            )
            console.print(
                f"  [cyan]{col_name}[/cyan]  points={count}  status={status}  vec_dim={vec_size}"
            )
        except Exception as exc:
            console.print(f"  [warning]{col_name}[/warning]  error: {exc}")


@memory_app.command("search")
def memory_search(
    query: str = typer.Argument(..., help="Natural language search query."),
    layer: str = typer.Option(
        "all",
        "--layer",
        "-l",
        help="Layer to search: agent | team | global | all.",
    ),
    agent_id: str = typer.Option(
        None,
        "--agent",
        "-a",
        help="Filter by agent ID (e.g. 'analyser', 'Worker-task_1').",
    ),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Max results to return."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full entry content."),
) -> None:
    """🔍 Semantic search across memory layers.

    Searches the vector store using the query and returns the most relevant
    entries, optionally filtered by layer or agent.

    Examples:
        aifactory memory search "FastAPI authentication"
        aifactory memory search "user preferences" --layer team
        aifactory memory search "coding patterns" --agent Worker-task_1 --top-k 3
    """
    setup_logging()

    from aifactory.core.schemas import MemoryLevel

    # Resolve layer arg
    layer_map = {
        "agent": [MemoryLevel.AGENT],
        "team": [MemoryLevel.TEAM],
        "global": [MemoryLevel.GLOBAL],
        "all": list(MemoryLevel),
    }
    resolved_layers = layer_map.get(layer.lower())
    if resolved_layers is None:
        console.print(
            f"[error]Unknown layer '[bold]{layer}[/bold]'. Choose: agent | team | global | all[/error]"  # noqa: E501
        )
        raise typer.Exit(1)

    try:
        router = _make_router()
        entries = router.retrieve(
            query=query,
            levels=resolved_layers,
            agent_id=agent_id,
            top_k=top_k,
        )
    except Exception as e:
        console.print(f"\n[error]Search failed: {e}[/error]\n")
        raise typer.Exit(1) from e

    if not entries:
        console.print(
            f"\n[muted]No results for '[bold]{query}[/bold]' in {layer} layer(s).[/muted]\n"
        )
        return

    console.print(
        f"\n[bold]Search:[/bold] [cyan]{query}[/cyan]  "
        f"[muted]layer={layer}  results={len(entries)}[/muted]\n"
    )

    table = Table(show_header=True, border_style="dim", expand=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Layer", style="bold", width=8)
    table.add_column("Agent", style="cyan", width=18)
    table.add_column("Type", style="dim", width=14)
    table.add_column("Content", ratio=1)

    for i, entry in enumerate(entries, 1):
        content = entry.content.strip()
        if not verbose and len(content) > 120:
            content = content[:120] + "…"
        table.add_row(
            str(i),
            entry.level.value,
            entry.agent_id or "—",
            entry.entry_type or "—",
            content,
        )

    console.print(table)
    console.print(
        "\n[muted]Tip: use [bold]--verbose[/bold] to see full entry content, "
        "[bold]--layer agent[/bold] to narrow scope.[/muted]\n"
    )


@memory_app.command("clear")
def memory_clear(
    agent_id: str = typer.Option(
        None,
        "--agent",
        "-a",
        help="Clear memories for a specific agent ID only.",
    ),
    layer: str = typer.Option(
        "agent",
        "--layer",
        "-l",
        help="Layer to clear: agent | team | global | all. Defaults to 'agent'.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompt.",
    ),
) -> None:
    """🗑  Clear memory entries.

    By default clears the agent layer only (safest). Use --layer all to
    wipe everything (irreversible).

    Examples:
        aifactory memory clear --agent analyser
        aifactory memory clear --layer team --yes
        aifactory memory clear --layer all --yes
    """
    setup_logging()

    from aifactory.core.schemas import MemoryLevel

    layer_map = {
        "agent": [MemoryLevel.AGENT],
        "team": [MemoryLevel.TEAM],
        "global": [MemoryLevel.GLOBAL],
        "all": list(MemoryLevel),
    }
    resolved_layers = layer_map.get(layer.lower())
    if resolved_layers is None:
        console.print(f"[error]Unknown layer '[bold]{layer}[/bold]'.[/error]")
        raise typer.Exit(1)

    # Build a clear description for the confirmation prompt
    scope_desc = f"agent=[bold]{agent_id}[/bold]" if agent_id else "all agents"
    layers_desc = ", ".join(lv.value for lv in resolved_layers)

    if not yes:
        console.print(
            f"\n[warning]This will delete memories for {scope_desc} "
            f"in layer(s): [bold]{layers_desc}[/bold][/warning]"
        )
        confirmed = typer.confirm("Proceed?", default=False)
        if not confirmed:
            console.print("[muted]Cancelled.[/muted]\n")
            raise typer.Exit(0)

    try:
        router = _make_router()
        deleted_total = 0

        if agent_id:
            # Clear a specific agent across the requested layers
            for level in resolved_layers:
                _delete_agent_from_layer(router, agent_id, level)
            console.print(
                f"\n[success]✓[/success] Cleared memories for "
                f"[bold]{agent_id}[/bold] in {layers_desc}.\n"
            )
        else:
            # Clear all agents from the requested layers
            for level in resolved_layers:
                count = _count_layer(router, level)
                deleted_total += count
                _clear_layer(router, level)
            console.print(
                f"\n[success]✓[/success] Cleared [bold]{deleted_total}[/bold] entries "
                f"from layer(s): {layers_desc}.\n"
            )
    except Exception as e:
        console.print(f"\n[error]Clear failed: {e}[/error]\n")
        raise typer.Exit(1) from e


# ── Internal helpers ──────────────────────────────────────────────────────────


def _delete_agent_from_layer(router, agent_id: str, level) -> None:
    """Delete all entries for agent_id in a specific layer using MemoryRouter.clear_agent."""
    # MemoryRouter.clear_agent clears across ALL layers, so we use it only for
    # the "all layers" case. For single-layer, use Qdrant filter directly.
    from qdrant_client.models import FieldCondition, Filter, MatchValue
    from qdrant_client.models import models as qdrant_models

    col = router._collection_name(level)
    try:
        router.client.delete(
            collection_name=col,
            points_selector=qdrant_models.FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="agent_id", match=MatchValue(value=agent_id))]
                )
            ),
        )
        router._index_cache.pop(col, None)  # invalidate LlamaIndex cache
    except Exception as exc:
        raise RuntimeError(f"Failed to delete {agent_id} from {col}: {exc}") from exc


def _count_layer(router, level) -> int:
    """Count entries in a layer."""
    try:
        info = router.client.get_collection(router._collection_name(level))
        return info.points_count or 0
    except Exception:
        return 0


def _clear_layer(router, level) -> None:
    """Delete all points in a layer by recreating the collection."""
    from qdrant_client.models import Distance, VectorParams

    col = router._collection_name(level)
    try:
        router.client.delete_collection(col)
        router.client.create_collection(
            collection_name=col,
            vectors_config=VectorParams(size=router.dimensions, distance=Distance.COSINE),
        )
        router._index_cache.pop(col, None)
    except Exception as exc:
        raise RuntimeError(f"Failed to clear layer {level.value}: {exc}") from exc
