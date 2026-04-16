"""
CLI commands for Docker MCP sandbox management.

Commands:
    aifactory sandbox build    — build the sandbox Docker images
    aifactory sandbox status   — show pool health metrics
    aifactory sandbox test     — run a quick smoke test in a fresh container
    aifactory sandbox pull     — pull the latest sandbox base image
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

sandbox_app = typer.Typer(
    name="sandbox", help="🐳 Docker MCP sandbox management", no_args_is_help=True
)
console = Console()


def _repo_root() -> Path:
    """Return the repository root (3 levels up from this file)."""
    return Path(__file__).resolve().parents[3]


# ── build ─────────────────────────────────────────────────────────────────────


@sandbox_app.command("build")
def cmd_build(
    no_cache: bool = typer.Option(False, "--no-cache", help="Build without Docker layer cache"),
    push: bool = typer.Option(False, "--push", help="Push images to registry after build"),
) -> None:
    """Build the aifactory-sandbox and aifactory-mcp-server Docker images."""
    root = _repo_root()

    images = [
        {
            "tag": "aifactory-sandbox:latest",
            "dockerfile": "docker/sandbox/Dockerfile",
            "description": "Sandbox execution environment (Python/Node/bash)",
        },
        {
            "tag": "aifactory-mcp-server:latest",
            "dockerfile": "docker/sandbox/Dockerfile.mcp",
            "description": "MCP server (FastMCP + pool manager)",
        },
    ]

    for img in images:
        console.print(f"\n[bold cyan]Building {img['tag']}[/] — {img['description']}")
        cmd = [
            "docker",
            "build",
            "-t",
            img["tag"],
            "-f",
            img["dockerfile"],
        ]
        if no_cache:
            cmd.append("--no-cache")
        cmd.append(".")  # build context = repo root

        result = subprocess.run(cmd, cwd=str(root))
        if result.returncode != 0:
            console.print(f"[red]Build failed for {img['tag']}[/]")
            raise typer.Exit(1)

        console.print(f"[green]✓ Built {img['tag']}[/]")

        if push:
            push_result = subprocess.run(["docker", "push", img["tag"]], cwd=str(root))
            if push_result.returncode != 0:
                console.print(f"[red]Push failed for {img['tag']}[/]")
                raise typer.Exit(1)
            console.print(f"[green]✓ Pushed {img['tag']}[/]")

    console.print("\n[bold green]All images built successfully.[/]")
    console.print(
        "\nTo start the sandbox stack:\n"
        "  docker compose -f docker/docker-compose.sandbox.yml up -d\n"
        "\nTo enable Docker sandbox in AI Factory:\n"
        "  export AIFACTORY_SANDBOX_DOCKER_ENABLED=true\n"
        "  export AIFACTORY_MCP_SANDBOX_URL=http://localhost:8100/mcp\n"
        "  export MCP_AUTH_TOKEN=<your-token>"
    )


# ── pull ──────────────────────────────────────────────────────────────────────


@sandbox_app.command("pull")
def cmd_pull() -> None:
    """Pull the latest Python base image used by the sandbox Dockerfile."""
    console.print("[bold cyan]Pulling python:3.12-slim ...[/]")
    result = subprocess.run(["docker", "pull", "python:3.12-slim"])
    if result.returncode != 0:
        console.print("[red]Pull failed[/]")
        raise typer.Exit(1)
    console.print("[green]✓ Pulled python:3.12-slim[/]")

    console.print("[bold cyan]Pulling tecnativa/docker-socket-proxy ...[/]")
    result = subprocess.run(["docker", "pull", "tecnativa/docker-socket-proxy:latest"])
    if result.returncode != 0:
        console.print("[yellow]Warning: socket-proxy pull failed (optional)[/]")
    else:
        console.print("[green]✓ Pulled tecnativa/docker-socket-proxy[/]")


# ── status ────────────────────────────────────────────────────────────────────


@sandbox_app.command("status")
def cmd_status(
    url: str = typer.Option(
        "", "--url", help="MCP server base URL, e.g. http://localhost:8100 (overrides settings)"
    ),
) -> None:
    """Show Docker container pool health metrics from the MCP server."""
    asyncio.run(_async_status(url))


async def _async_status(url: str) -> None:
    import os

    if url:
        os.environ["AIFACTORY_MCP_SANDBOX_URL"] = url

    try:
        from aifactory.mcp.client import get_mcp_client
    except ImportError:
        console.print("[red]MCP client not available. Is the mcp package installed?[/]")
        raise typer.Exit(1) from None

    try:
        client = await get_mcp_client()

        # Health check
        try:
            health = await client.health()
            status_color = "green" if health.get("status") == "ok" else "yellow"
            console.print(f"\n[{status_color}]Server health: {health.get('status', 'unknown')}[/]")
        except Exception as exc:
            console.print(f"[red]Health check failed: {exc}[/]")
            raise typer.Exit(1) from exc

        # Pool status
        pool = await client.pool_status()
        await client.close()

        table = Table(title="Docker Sandbox Pool", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value")

        table.add_row("Image", str(pool.get("image", "—")))
        table.add_row("Ready containers", str(pool.get("ready", 0)))
        table.add_row("In-use containers", str(pool.get("in_use", 0)))
        table.add_row("Total containers", str(pool.get("total", 0)))
        table.add_row("Min idle target", str(pool.get("min_idle", 0)))
        table.add_row("Max pool size", str(pool.get("max_size", 0)))
        table.add_row("gVisor enabled", str(pool.get("gvisor", False)))

        console.print(table)

    except Exception as exc:
        console.print(f"[red]Error: {exc}[/]")
        raise typer.Exit(1) from exc


# ── test ──────────────────────────────────────────────────────────────────────


@sandbox_app.command("test")
def cmd_test(
    url: str = typer.Option(
        "", "--url", help="MCP server base URL, e.g. http://localhost:8100 (overrides settings)"
    ),
    language: str = typer.Option(
        "python", "--language", "-l", help="Language to test: python/bash/javascript"
    ),
) -> None:
    """Run a quick smoke test against the Docker sandbox."""
    asyncio.run(_async_test(url, language))


_TEST_SNIPPETS = {
    "python": (
        "import sys, platform\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "arr = np.arange(5)\n"
        "print(f'Python {sys.version.split()[0]} OK')\n"
        "print(f'numpy {np.__version__}, pandas {pd.__version__}')\n"
        "print(f'sum(arr)={arr.sum()}')\n"
        "print('ALL GOOD')"
    ),
    "bash": "echo 'bash OK'; python3 --version; node --version; git --version",
    "javascript": (
        "const os = require('os');\n"
        "console.log('Node.js', process.version, 'OK');\n"
        "console.log('platform:', os.platform());\n"
        "console.log('ALL GOOD');"
    ),
}


async def _async_test(url: str, language: str) -> None:
    import os

    if url:
        os.environ["AIFACTORY_MCP_SANDBOX_URL"] = url

    lang = language.lower()
    if lang not in _TEST_SNIPPETS:
        console.print(f"[red]Unknown language '{lang}'. Choose: python, bash, javascript[/]")
        raise typer.Exit(1)

    code = _TEST_SNIPPETS[lang]
    console.print(f"\n[bold cyan]Running {lang} smoke test...[/]")

    try:
        from aifactory.mcp.client import get_mcp_client
    except ImportError:
        console.print("[red]MCP client not available.[/]")
        raise typer.Exit(1) from None

    try:
        client = await get_mcp_client()
        if lang == "python":
            result = await client.execute_python(code=code, agent_id="cli-test", timeout=30)
        elif lang == "bash":
            result = await client.execute_bash(code=code, agent_id="cli-test", timeout=30)
        else:
            result = await client.execute_javascript(code=code, agent_id="cli-test", timeout=30)
        await client.close()

        if result.success:
            console.print(f"[green]✓ {lang} smoke test passed ({result.elapsed_ms:.0f}ms)[/]")
            console.print(f"\nOutput:\n{result.output}")
        else:
            console.print(f"[red]✗ {lang} smoke test FAILED (exit_code={result.exit_code})[/]")
            console.print(f"\nOutput:\n{result.output}")
            raise typer.Exit(1)

    except Exception as exc:
        console.print(f"[red]Test error: {exc}[/]")
        raise typer.Exit(1) from exc
