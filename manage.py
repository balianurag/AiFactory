#!/usr/bin/env python3
"""
AI Factory — Project Management CLI

A single entry point for all developer and operational tasks.

Usage:
    python manage.py <command> [options]
    python manage.py --help
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

ROOT = Path(__file__).parent
REQUIRED_PYTHON = (3, 13)

app = typer.Typer(
    name="manage",
    help="AI Factory project management CLI.",
    no_args_is_help=True,
)
console = Console()


def _run(cmd: list[str], *, check: bool = True) -> int:
    """Run a shell command, streaming output to the terminal."""
    result = subprocess.run(cmd, cwd=ROOT)
    if check and result.returncode != 0:
        raise typer.Exit(result.returncode)
    return result.returncode


# =============================================================================
# Setup
# =============================================================================


@app.command()
def setup(
    skip_precommit: bool = typer.Option(False, "--skip-pre-commit", help="Skip pre-commit install"),
) -> None:
    """Full environment setup: check Python, install uv, create venv, install deps, pre-commit."""
    console.print(Panel("AI Factory — Environment Setup", border_style="blue"))

    # 1. Python version check
    major, minor = sys.version_info.major, sys.version_info.minor
    console.print(f"[bold]1/4  Python version[/bold]  {major}.{minor}", end="  ")
    if (major, minor) < REQUIRED_PYTHON:
        console.print(
            f"[red]FAIL[/red] — need {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}+, "
            f"got {major}.{minor}"
        )
        console.print(
            f"[dim]Install Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]} via pyenv, "
            "mise, or https://python.org[/dim]"
        )
        raise typer.Exit(1)
    console.print("[green]OK[/green]")

    # 2. uv availability
    console.print("[bold]2/4  uv[/bold]", end="  ")
    if shutil.which("uv") is None:
        console.print("[yellow]not found — installing via pip[/yellow]")
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "uv",
                "--quiet",
                "--index-url",
                "https://pypi.org/simple/",
            ]
        )
    else:
        uv_ver = subprocess.check_output(["uv", "--version"], text=True).strip()
        console.print(f"[green]OK[/green]  ({uv_ver})")

    # 3. Create venv + install dependencies
    console.print("[bold]3/4  Installing dependencies[/bold]")
    _run(["uv", "sync", "--dev"])

    # Copy pip.conf into the venv so bare `pip install` uses PyPI, not the global config
    src_pip_conf = ROOT / "pip.conf"
    venv_pip_conf = ROOT / ".venv" / "pip.conf"
    if src_pip_conf.exists() and (ROOT / ".venv").exists():
        shutil.copy(src_pip_conf, venv_pip_conf)

    # 4. pre-commit hooks
    console.print("[bold]4/4  pre-commit hooks[/bold]", end="  ")
    if skip_precommit:
        console.print("[dim]skipped[/dim]")
    else:
        _run(["uv", "run", "pre-commit", "install"])
        console.print("[green]OK[/green]")

    console.print()
    console.print("[bold green]Setup complete.[/bold green]  Run [cyan]manage.py dev[/cyan].")


# =============================================================================
# Dev tooling
# =============================================================================


@app.command()
def install(
    dev: bool = typer.Option(True, "--dev/--no-dev", help="Include dev dependencies"),
) -> None:
    """Install all project dependencies via uv."""
    cmd = ["uv", "sync"]
    if not dev:
        cmd.append("--no-dev")
    _run(cmd)


@app.command()
def dev(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host"),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port"),
    reload: bool = typer.Option(True, "--reload/--no-reload", help="Auto-reload on code changes"),
    log_level: str = typer.Option("info", "--log-level", "-l", help="Uvicorn log level"),
) -> None:
    """Start the FastAPI development server with auto-reload."""
    cmd = [
        "uv",
        "run",
        "uvicorn",
        "aifactory.api.main:app",
        "--host",
        host,
        "--port",
        str(port),
        "--log-level",
        log_level,
    ]
    if reload:
        cmd.append("--reload")
    _run(cmd)


@app.command()
def test(
    path: str = typer.Argument("tests/", help="Test path or pattern"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    cov: bool = typer.Option(False, "--cov", help="Generate coverage report"),
    keyword: str = typer.Option(None, "-k", help="Filter tests by keyword expression"),
    fail_fast: bool = typer.Option(False, "--fail-fast", "-x", help="Stop on first failure"),
) -> None:
    """Run the test suite with pytest."""
    cmd = ["uv", "run", "pytest", path]
    if verbose:
        cmd.append("-v")
    if cov:
        cmd += ["--cov=aifactory", "--cov-report=term", "--cov-report=html:htmlcov"]
    if keyword:
        cmd += ["-k", keyword]
    if fail_fast:
        cmd.append("-x")
    _run(cmd)


@app.command()
def lint(
    fix: bool = typer.Option(False, "--fix", help="Auto-fix fixable violations"),
) -> None:
    """Run ruff linter over src/ and tests/."""
    cmd = ["uv", "run", "ruff", "check", "src/", "tests/"]
    if fix:
        cmd.append("--fix")
    _run(cmd)


@app.command()
def format(
    check: bool = typer.Option(False, "--check", help="Check only, do not write changes"),
) -> None:
    """Format source code with ruff."""
    cmd = ["uv", "run", "ruff", "format", "src/", "tests/"]
    if check:
        cmd.append("--check")
    _run(cmd)


@app.command()
def typecheck() -> None:
    """Run mypy static type checking over src/."""
    _run(["uv", "run", "mypy", "src/"])


@app.command()
def build() -> None:
    """Build the distributable package (wheel + sdist) into dist/."""
    _run(["uv", "build"])


# =============================================================================
# Application commands (delegate to aifactory CLI)
# =============================================================================


@app.command()
def run(
    task: str = typer.Argument(None, help="The task/idea to execute"),
    task_file: Path = typer.Option(None, "--task-file", "-f", help="Read task from a file"),  # noqa: B008
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed agent logs"),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Log level (DEBUG/INFO/WARNING)"),
) -> None:
    """Run the AI Factory dynamic pipeline on a task or idea."""
    cmd = ["uv", "run", "aifactory", "run"]
    if task:
        cmd.append(task)
    if task_file:
        cmd += ["--task-file", str(task_file)]
    if verbose:
        cmd.append("--verbose")
    cmd += ["--log-level", log_level]
    _run(cmd)


@app.command()
def info() -> None:
    """Show AI Factory configuration, mode, and API key status."""
    _run(["uv", "run", "aifactory", "info"])


@app.command("memory-stats")
def memory_stats() -> None:
    """Show memory system statistics (agent / team / global layers)."""
    _run(["uv", "run", "aifactory", "memory-stats"])


# ── Sub-apps ──────────────────────────────────────────────────────────────────

models_app = typer.Typer(help="Model registry commands.", no_args_is_help=True)
app.add_typer(models_app, name="models")

agents_app = typer.Typer(help="Agent management commands.", no_args_is_help=True)
app.add_typer(agents_app, name="agents")


@models_app.command("list")
def models_list(
    role: str = typer.Option(
        None, "--role", "-r", help="Filter by agent role (e.g. analyser, execution_lead)"
    ),
) -> None:
    """List models from the registry, optionally filtered by agent role."""
    cmd = ["uv", "run", "aifactory", "models", "list"]
    if role:
        cmd += ["--role", role]
    _run(cmd)


@models_app.command("tiers")
def models_tiers() -> None:
    """Show the active premium / standard / budget model tier assignments."""
    _run(["uv", "run", "aifactory", "models", "tiers"])


@agents_app.command("list")
def agents_list() -> None:
    """List all pipeline agent roles and their ability-selected models."""
    _run(["uv", "run", "aifactory", "agents", "list"])


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    app()
