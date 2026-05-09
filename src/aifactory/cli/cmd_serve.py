"""
aifactory serve — Start the AI Factory HTTP/WebSocket API server.

Keeps all infrastructure connections warm:
  - Qdrant client (memory persistence)
  - Redis (task store + embedding cache + response cache)
  - MCP catalog (tool integrations)
  - LiteLLM response cache
  - Model registry (no per-request YAML reload)
  - LangGraph compiled pipelines (one per mode, cached in-process)

Cold start: ~3–5s once.  Per-request overhead: ~50ms vs ~4s for standalone CLI.

Endpoints (see /docs for OpenAPI):
  POST /api/tasks          — start a pipeline run, returns task_id immediately
  GET  /api/tasks/{id}     — poll status + partial/final results
  GET  /api/tasks          — list recent runs
  WS   /api/ws/{task_id}   — real-time event + token streaming
  GET  /api/health         — liveness check + dependency status
  GET  /api/models         — available models per role
"""

from __future__ import annotations

import typer

from aifactory.cli._helpers import console


def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-H", help="Bind host."),
    port: int = typer.Option(8080, "--port", "-p", help="Bind port."),
    workers: int = typer.Option(
        1,
        "--workers",
        "-w",
        help="Uvicorn worker processes. Use 1 for development (shared in-process state).",
    ),
    reload: bool = typer.Option(
        False,
        "--reload",
        help="Auto-reload on code changes (dev mode, single worker only).",
    ),
    log_level: str = typer.Option(
        "info",
        "--log-level",
        "-l",
        help="Uvicorn log level: debug | info | warning | error.",
    ),
    no_warmup: bool = typer.Option(
        False,
        "--no-warmup",
        help="Skip connection pre-warming (faster cold start, first request slower).",
    ),
) -> None:
    """🚀 Start the AI Factory API server.

    Keeps Qdrant, Redis, MCP catalog, and model registry warm across requests.
    Recommended for iterative development and production deployments.

    WebSocket streaming at /api/ws/{task_id} provides real-time token output.
    Interactive API docs at http://<host>:<port>/docs

    Examples:
        aifactory serve                       # dev: localhost:8080
        aifactory serve --port 9000           # custom port
        aifactory serve --workers 4           # production (multi-process)
        aifactory serve --reload --log-level debug  # hot-reload dev mode
    """
    try:
        import uvicorn
    except ImportError:
        console.print(
            "[error]uvicorn is not installed. Install it:[/error]\n  uv add uvicorn[standard]\n"
        )
        raise typer.Exit(1) from None

    if not no_warmup:
        _warmup()

    console.print(
        f"\n[bold green]AI Factory Server[/bold green]  "
        f"[cyan]http://{host}:{port}[/cyan]  "
        f"workers={workers}  log_level={log_level}"
    )
    console.print(
        "[muted]OpenAPI docs: [/muted]"
        + f"http://{'localhost' if host == '0.0.0.0' else host}:{port}/docs"
    )
    console.print("[muted]Press Ctrl+C to stop.[/muted]\n")

    uvicorn.run(
        "aifactory.api.main:app",
        host=host,
        port=port,
        workers=workers if not reload else 1,
        reload=reload,
        log_level=log_level.lower(),
    )


def _warmup() -> None:
    """Pre-warm all infrastructure connections before accepting requests."""
    from dotenv import load_dotenv

    load_dotenv()

    console.print("[dim]Warming up connections...[/dim]")
    issues: list[str] = []

    # Redis
    try:
        import os

        redis_url = os.getenv("REDIS_URL", "")
        if redis_url:
            import redis

            r = redis.from_url(redis_url, socket_connect_timeout=2)
            r.ping()
            console.print("  [green]✓[/green] Redis connected")
        else:
            console.print("  [dim]— Redis: REDIS_URL not set (task store disabled)[/dim]")
    except Exception as exc:
        issues.append(f"Redis: {exc}")
        console.print(f"  [yellow]⚠[/yellow] Redis: {exc}")

    # Qdrant
    try:
        from aifactory.core.memory_router import MemoryRouter

        router = MemoryRouter()
        mode = "in-memory" if getattr(router, "in_memory_mode", False) else "persistent"
        console.print(f"  [green]✓[/green] Qdrant connected ({mode})")
    except Exception as exc:
        issues.append(f"Qdrant: {exc}")
        console.print(f"  [yellow]⚠[/yellow] Qdrant: {exc}")

    # Embedding provider
    try:
        from aifactory.core.embed_provider import resolve_provider

        provider, model, dims = resolve_provider()
        console.print(f"  [green]✓[/green] Embedding: {model} ({dims}-dim) via {provider}")
        if provider == "stub":
            console.print(
                "    [yellow]⚠  Stub embeddings active — set API key for semantic search[/yellow]"
            )
    except Exception as exc:
        console.print(f"  [dim]— Embedding: {exc}[/dim]")

    # LiteLLM response cache
    try:
        from aifactory.graphs.dynamic_pipeline import _init_response_cache

        _init_response_cache()
        import litellm

        cache_type = getattr(litellm.cache, "type", "in-memory") if litellm.cache else "disabled"
        console.print(f"  [green]✓[/green] LiteLLM cache: {cache_type}")
    except Exception as exc:
        console.print(f"  [dim]— LiteLLM cache: {exc}[/dim]")

    # Model registry
    try:
        from aifactory.core.model_selector import load_available_models

        models = load_available_models()
        console.print(f"  [green]✓[/green] Model registry: {len(models)} models loaded")
    except Exception as exc:
        console.print(f"  [dim]— Model registry: {exc}[/dim]")

    console.print()
    if issues:
        console.print(
            f"  [yellow]Note:[/yellow] {len(issues)} connection(s) unavailable — "
            "server will start but affected features may be degraded.\n"
        )
