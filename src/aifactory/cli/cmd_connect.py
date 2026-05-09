"""
aifactory connect add <service>    — OAuth / API key flow for integrations.
aifactory connect list             — Show configured services and connection status.
aifactory connect remove <service> — Disconnect a service integration.
aifactory tools                    — List available integration tools.

NOTE: Composio has been removed. Use the MCP catalog instead.
See: aifactory serve  (MCP catalog is loaded automatically when
AIFACTORY_MCP_CATALOG_ENABLED=true and mcp_catalog.yaml is configured).
"""

from __future__ import annotations

import typer

from aifactory.cli._helpers import console, setup_logging

connect_app = typer.Typer(
    help="🔌 Manage service integrations.",
    no_args_is_help=True,
)

_MCP_NOTE = (
    "Integration management via Composio has been removed.\n"
    "Use the MCP catalog instead:\n"
    "  • Configure servers in config/integrations/mcp_catalog.yaml\n"
    "  • Enable with AIFACTORY_MCP_CATALOG_ENABLED=true\n"
    "  • Inspect the catalog: GET /api/mcp/catalog"
)


# ── connect add ──────────────────────────────────────────────────────────────


@connect_app.command("add")
def connect_add(
    service: str = typer.Argument(
        ...,
        help="Service to connect (not available — use MCP catalog).",
    ),
) -> None:
    """🔌 Connect an external service integration.

    NOTE: Composio-based connection has been removed.
    Configure services via the MCP catalog instead.
    """
    setup_logging()
    console.print(f"\n[warning]connect add is not available for '{service}'.[/warning]")
    console.print(f"[muted]{_MCP_NOTE}[/muted]\n")
    raise typer.Exit(1)


# ── connect list ─────────────────────────────────────────────────────────────


@connect_app.command("list")
def connect_list() -> None:
    """📋 Show configured services and their connection status.

    NOTE: Composio-based listing has been removed.
    Use GET /api/mcp/catalog to inspect the MCP catalog.
    """
    setup_logging()
    console.print("\n[warning]connect list is not available.[/warning]")
    console.print(f"[muted]{_MCP_NOTE}[/muted]\n")
    raise typer.Exit(1)


# ── connect remove ───────────────────────────────────────────────────────────


@connect_app.command("remove")
def connect_remove(
    service: str = typer.Argument(..., help="Service to disconnect (not available)."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """🗑️  Disconnect a service integration.

    NOTE: Composio-based removal has been removed.
    Configure services via the MCP catalog instead.
    """
    setup_logging()
    console.print(f"\n[warning]connect remove is not available for '{service}'.[/warning]")
    console.print(f"[muted]{_MCP_NOTE}[/muted]\n")
    raise typer.Exit(1)


# ── tools (top-level, kept separate) ─────────────────────────────────────────


def tools(
    service: str = typer.Option(
        None, "--service", "-s", help="Filter to a specific service/toolkit."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show tool descriptions."),
) -> None:
    """🧰 List available integration tools.

    NOTE: Composio-based tool listing has been removed.
    Use GET /api/mcp/catalog to inspect MCP catalog servers and their tools.
    """
    setup_logging()
    console.print("\n[warning]tools listing via Composio is not available.[/warning]")
    console.print(f"[muted]{_MCP_NOTE}[/muted]\n")
    raise typer.Exit(1)
