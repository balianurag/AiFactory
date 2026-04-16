"""
aifactory connect add <service>    — OAuth / API key flow for integrations.
aifactory connect list             — Show configured services and connection status.
aifactory connect remove <service> — Disconnect a service integration.
aifactory tools                    — List available integration tools.
"""

from __future__ import annotations

import asyncio
import subprocess

import typer

from aifactory.cli._helpers import console, setup_logging

connect_app = typer.Typer(
    help="🔌 Manage service integrations.",
    no_args_is_help=True,
)


# ── connect add ──────────────────────────────────────────────────────────────


@connect_app.command("add")
def connect_add(
    service: str = typer.Argument(
        ...,
        help="Service to connect: github, gitlab, jira, notion, gmail, slack, etc.",
    ),
) -> None:
    """🔌 Connect an external service integration.

    Sets up OAuth or API key authentication for the specified service
    via Composio. After connecting, the service's tools become available
    to pipeline agents.

    Examples:
        aifactory connect add github
        aifactory connect add gitlab
        aifactory connect add gmail
        aifactory connect add notion
    """
    setup_logging()
    asyncio.run(_connect_service(service))


# ── connect list ─────────────────────────────────────────────────────────────


@connect_app.command("list")
def connect_list() -> None:
    """📋 Show configured services and their connection status.

    Checks each enabled toolkit and reports whether OAuth / API key
    authentication is active (tools reachable) or not yet set up.

    Example:
        aifactory connect list
    """
    setup_logging()
    asyncio.run(_list_connections())


# ── connect remove ───────────────────────────────────────────────────────────


@connect_app.command("remove")
def connect_remove(
    service: str = typer.Argument(..., help="Service to disconnect."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """🗑️  Disconnect a service integration.

    Revokes the stored OAuth token / API key for the given service via
    the Composio CLI. The service will no longer be available to agents.

    Examples:
        aifactory connect remove github
        aifactory connect remove github --yes
    """
    setup_logging()
    asyncio.run(_remove_service(service, yes=yes))


# ── tools (top-level, kept separate) ─────────────────────────────────────────


def tools(
    service: str = typer.Option(
        None, "--service", "-s", help="Filter to a specific service/toolkit."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show tool descriptions."),
) -> None:
    """🧰 List available integration tools.

    Shows all connected Composio toolkits and their available tools.
    Use --service to filter to a specific toolkit.

    Examples:
        aifactory tools
        aifactory tools --service github
        aifactory tools -s gitlab -v
    """
    setup_logging()
    asyncio.run(_list_tools(service=service, verbose=verbose))


# ── Internals ────────────────────────────────────────────────────────────────


async def _connect_service(service: str) -> None:
    """Run the Composio connection flow for a service."""
    from rich.status import Status

    console.print(f"\n[heading]🔌 Connecting {service}...[/heading]\n")

    try:
        from aifactory.integrations.providers.composio import ComposioProvider

        provider = ComposioProvider()
        ok = await provider.connect()
        if not ok:
            console.print(
                f"[error]Failed to connect Composio provider: {provider.status.value}[/error]"
            )
            console.print("[muted]Check COMPOSIO_API_KEY in your .env file.[/muted]")
            raise typer.Exit(1)

        console.print("[success]✓[/success] Composio provider connected")

        available = provider.get_available_toolkits()
        service_lower = service.lower().replace("-", "").replace("_", "")

        matched_toolkit = None
        for tk in available:
            if tk.lower().replace("-", "").replace("_", "") == service_lower:
                matched_toolkit = tk
                break

        if not matched_toolkit:
            console.print(f"\n[warning]'{service}' is not in the enabled toolkit list.[/warning]")
            console.print(f"[muted]Available toolkits: {', '.join(available)}[/muted]")
            console.print(
                "\n[muted]To add it, update enabled_toolkits in your Composio config.[/muted]"
            )
            raise typer.Exit(1)

        with Status(f"Fetching {matched_toolkit} tools...", console=console):
            tools_map = await provider.fetch_available_tools(toolkits=[matched_toolkit])

        tool_count = len(tools_map.get(matched_toolkit, []))
        console.print(f"[success]✓[/success] {matched_toolkit}: {tool_count} tools available")

        console.print(
            f"\n[bold]Next steps for OAuth (if not already done):[/bold]\n"
            f"  1. Run: [cyan]composio add {matched_toolkit}[/cyan]\n"
            f"  2. Follow the browser OAuth flow\n"
            f"  3. Test: [cyan]aifactory tools --service {matched_toolkit}[/cyan]\n"
        )

        summary = provider.get_toolkit_tools_for_prompt(matched_toolkit)
        lines = summary.strip().split("\n")[:10]
        if lines:
            console.print(f"[bold]Sample tools ({matched_toolkit}):[/bold]")
            for line in lines:
                console.print(f"  {line.strip()}")
            if len(summary.strip().split("\n")) > 10:
                console.print(
                    f"  [muted]... and {len(summary.strip().split(chr(10))) - 10} more[/muted]"
                )

        await provider.disconnect()

    except ImportError:
        console.print("[error]composio package not installed.[/error]")
        console.print("[muted]Run: uv add composio[/muted]")
        raise typer.Exit(1) from None
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[error]Connection failed: {e}[/error]")
        raise typer.Exit(1) from e

    console.print()


async def _list_connections() -> None:
    """Check each configured toolkit and show whether it has active auth."""
    from rich.table import Table

    try:
        from aifactory.integrations.providers.composio import ComposioProvider

        provider = ComposioProvider()
        ok = await provider.connect()
        if not ok:
            console.print("[error]Could not connect to Composio — check COMPOSIO_API_KEY.[/error]")
            raise typer.Exit(1)

        toolkits = provider.get_available_toolkits()

        from rich.status import Status

        with Status("Checking service connections...", console=console):
            tools_map = await provider.fetch_available_tools(toolkits=toolkits)

        table = Table(
            title="🔌 Service Connections",
            show_header=True,
            border_style="dim",
        )
        table.add_column("Service", style="bold cyan")
        table.add_column("Status", justify="center")
        table.add_column("Tools", justify="right")
        table.add_column("To connect", style="dim")

        for tk in sorted(toolkits):
            tool_count = len(tools_map.get(tk, []))
            if tool_count > 0:
                status_text = "[green]● connected[/green]"
                count_str = str(tool_count)
            else:
                status_text = "[dim]○ not connected[/dim]"
                count_str = "—"
            table.add_row(tk, status_text, count_str, f"composio add {tk}")

        console.print()
        console.print(table)
        console.print(
            "\n[muted]Connect a service:    aifactory connect add <service>[/muted]"
            "\n[muted]Disconnect a service: aifactory connect remove <service>[/muted]\n"
        )

        await provider.disconnect()

    except ImportError:
        console.print("[error]composio package not installed.[/error]")
        console.print("[muted]Run: uv add composio[/muted]")
        raise typer.Exit(1) from None
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[error]Failed to list connections: {e}[/error]")
        raise typer.Exit(1) from e


async def _remove_service(service: str, *, yes: bool = False) -> None:
    """Revoke a service connection via the Composio CLI."""
    if not yes:
        confirmed = typer.confirm(f"Disconnect '{service}'?")
        if not confirmed:
            console.print("[muted]Aborted.[/muted]")
            return

    console.print(f"\n[heading]🗑️  Disconnecting {service}...[/heading]\n")

    try:
        result = subprocess.run(
            ["composio", "remove", service],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            console.print(f"[success]✓[/success] {service} disconnected.")
            console.print("[muted]Run 'aifactory connect list' to verify.[/muted]\n")
        else:
            output = (result.stderr or result.stdout).strip()
            console.print(f"[error]composio remove failed:[/error] {output}")
            console.print(f"\n[muted]Try manually: composio remove {service}[/muted]")
            raise typer.Exit(1)

    except FileNotFoundError:
        console.print("[error]composio CLI not found in PATH.[/error]")
        console.print(f"[muted]Try manually: composio remove {service}[/muted]")
        raise typer.Exit(1) from None
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[error]Disconnect failed: {e}[/error]")
        raise typer.Exit(1) from e


async def _list_tools(*, service: str | None = None, verbose: bool = False) -> None:
    """Fetch and display available tools."""
    from rich.table import Table

    try:
        from aifactory.integrations.providers.composio import ComposioProvider

        provider = ComposioProvider()
        ok = await provider.connect()
        if not ok:
            console.print("[error]Could not connect to Composio.[/error]")
            console.print("[muted]Check COMPOSIO_API_KEY in .env[/muted]")
            raise typer.Exit(1)

        toolkits = [service.lower()] if service else provider.get_available_toolkits()

        from rich.status import Status

        with Status("Fetching tools...", console=console):
            tools_map = await provider.fetch_available_tools(toolkits=toolkits)

        console.print()

        if verbose:
            for tk in sorted(tools_map):
                summary = provider.get_toolkit_tools_for_prompt(tk)
                if summary:
                    console.print(f"[heading]🔧 {tk.upper()}[/heading]")
                    for line in summary.strip().split("\n"):
                        console.print(f"  {line.strip()}")
                    console.print()
        else:
            table = Table(
                title="🧰 Available Integration Tools",
                show_header=True,
                border_style="dim",
            )
            table.add_column("Toolkit", style="bold cyan")
            table.add_column("Tools", justify="right")
            table.add_column("Sample Tools", max_width=60)

            total = 0
            for tk in sorted(tools_map):
                slugs = tools_map[tk]
                total += len(slugs)
                sample = ", ".join(slugs[:3])
                if len(slugs) > 3:
                    sample += f" ... (+{len(slugs) - 3})"
                table.add_row(tk, str(len(slugs)), sample)

            table.add_row("[bold]TOTAL[/bold]", f"[bold]{total}[/bold]", "")
            console.print(table)

        await provider.disconnect()
        console.print("[muted]Use --verbose for full tool descriptions.[/muted]\n")

    except ImportError:
        console.print("[error]composio package not installed.[/error]")
        console.print("[muted]Run: uv add composio[/muted]")
        raise typer.Exit(1) from None
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[error]Failed to list tools: {e}[/error]")
        raise typer.Exit(1) from e
