"""
Code Execution Sandbox — Secure code runner for agent-generated code.

Three backends (priority order for "auto"):
  1. DOCKER — warm Docker container pool via MCP (lowest latency, most secure)
  2. E2B    — cloud micro-VM via e2b-code-interpreter (when E2B_API_KEY is set)
  3. LOCAL  — subprocess with timeout + memory limits (fallback, zero cost)

Unified async interface:
  result = await execute(code, language="python")

Backend selection:
  - "auto"   → docker (if AIFACTORY_SANDBOX_DOCKER_ENABLED=true) > e2b > local
  - "docker" → force Docker MCP pool
  - "e2b"    → force E2B cloud sandbox
  - "local"  → force local subprocess

Workers invoke code execution via the CODE: protocol in their responses:
  ```python
  print("hello")
  ```
  CODE_END

The pipeline detects these blocks and feeds them through execute().
"""

from __future__ import annotations

import ast
import contextlib
import logging
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_MEMORY_MB = 256


def _e2b_api_key() -> str:
    """Get E2B API key from env var or settings (.env file)."""
    key = os.getenv("E2B_API_KEY", "").strip()
    if key:
        return key
    try:
        from aifactory.config.settings import get_settings

        return (get_settings().e2b_api_key or "").strip()
    except Exception:
        return ""


# =============================================================================
# Result / Config
# =============================================================================


@dataclass
class ExecutionResult:
    """Result of running code in the sandbox."""

    success: bool
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    error_summary: str = ""
    backend: str = "local"  # "local" or "e2b"

    @property
    def output(self) -> str:
        """Combined stdout + stderr for display."""
        parts = []
        if self.stdout.strip():
            parts.append(self.stdout.strip())
        if self.stderr.strip():
            parts.append(f"STDERR:\n{self.stderr.strip()}")
        return "\n".join(parts) or "(no output)"


@dataclass
class SandboxConfig:
    """Configuration for the code sandbox."""

    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    memory_mb: int = DEFAULT_MEMORY_MB
    python_executable: str = field(default_factory=lambda: sys.executable)
    allowed_imports: list[str] | None = None  # None = allow all stdlib
    work_dir: str | None = None  # None = temp dir


# =============================================================================
# CODE: block extraction
# =============================================================================

# Matches code blocks in worker responses:
#   ```python\n<code>\n```
#   or CODE:python\n<code>\nCODE_END
_CODE_BLOCK_RE = re.compile(
    r"(?:```(\w+)\n(.*?)```|CODE:(\w+)\n(.*?)CODE_END)",
    re.DOTALL,
)


def extract_code_blocks(text: str) -> list[tuple[str, str]]:
    """
    Extract code blocks from agent output.

    Returns list of (language, code) tuples. Supports:
      - Markdown fenced blocks: ```python\\ncode\\n```
      - CODE: protocol: CODE:python\\ncode\\nCODE_END

    Only returns blocks for executable languages.
    """
    executable_langs = {"python", "javascript", "js", "bash", "sh", "shell"}
    blocks: list[tuple[str, str]] = []
    for m in _CODE_BLOCK_RE.finditer(text):
        lang = (m.group(1) or m.group(3) or "").lower()
        code = (m.group(2) or m.group(4) or "").strip()
        if lang in executable_langs and code:
            # Normalize language names
            if lang in ("js",):
                lang = "javascript"
            if lang in ("sh", "shell"):
                lang = "bash"
            blocks.append((lang, code))
    return blocks


# =============================================================================
# Syntax check
# =============================================================================


def check_syntax(code: str) -> tuple[bool, str]:
    """
    Fast syntax check without executing the code.

    Returns:
        (is_valid, error_message)
    """
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"


# =============================================================================
# Local executor (subprocess)
# =============================================================================


def execute_python(
    code: str,
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    memory_mb: int = DEFAULT_MEMORY_MB,
    python_executable: str | None = None,
    extra_files: dict[str, str] | None = None,
    workspace: str | None = None,
) -> ExecutionResult:
    """
    Execute Python code in an isolated subprocess.

    Args:
        code: Python source code to execute.
        timeout: Maximum execution time in seconds.
        memory_mb: Maximum memory in megabytes (enforced on Unix).
        python_executable: Path to Python interpreter (defaults to current).
        extra_files: Additional files to write alongside the main script.
                     dict of {filename: content}.
        workspace: If set, use this directory as cwd so generated files
                   land on the real filesystem instead of a temp dir.

    Returns:
        ExecutionResult with success, output, and error info.
    """
    # Step 1: Syntax check (fast, no subprocess)
    is_valid, syntax_error = check_syntax(code)
    if not is_valid:
        return ExecutionResult(
            success=False,
            exit_code=1,
            stdout="",
            stderr=syntax_error,
            error_summary=syntax_error,
        )

    py = python_executable or sys.executable

    # When workspace is provided, run directly in that directory.
    # We still need a temp file for the wrapper script itself — write it to
    # a sibling temp dir so it doesn't pollute the workspace.
    _use_workspace = workspace and Path(workspace).is_dir()

    with tempfile.TemporaryDirectory(prefix="aifactory_sandbox_") as tmpdir:
        tmppath = Path(tmpdir)

        # Write extra files if any
        if extra_files:
            dest = Path(workspace) if _use_workspace else tmppath
            for fname, fcontent in extra_files.items():
                (dest / fname).write_text(fcontent, encoding="utf-8")

        # Write main script to temp dir (keeps workspace clean of helper files)
        script_path = tmppath / "_main.py"
        script_path.write_text(code, encoding="utf-8")

        # Build the subprocess command with memory limiting wrapper
        wrapper = textwrap.dedent(f"""\
            import sys, os
            try:
                import resource
                mem_bytes = {memory_mb} * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            except (ImportError, ValueError, OSError):
                pass  # resource module not available (Windows) or limit not supported
            # Run the actual script
            exec(open({str(script_path)!r}, encoding="utf-8").read())
        """)

        wrapper_path = tmppath / "_wrapper.py"
        wrapper_path.write_text(wrapper, encoding="utf-8")

        run_cwd = str(workspace) if _use_workspace else str(tmppath)

        try:
            result = subprocess.run(
                [py, str(wrapper_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=run_cwd,
                env={
                    **dict(os.environ),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONIOENCODING": "utf-8",
                },
            )

            success = result.returncode == 0
            error_summary = ""
            if not success and result.stderr:
                lines = result.stderr.strip().splitlines()
                error_summary = lines[-1] if lines else "Unknown error"

            return ExecutionResult(
                success=success,
                exit_code=result.returncode,
                stdout=result.stdout[:10000],
                stderr=result.stderr[:10000],
                error_summary=error_summary,
                backend="local",
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=f"Execution timed out after {timeout}s",
                timed_out=True,
                error_summary=f"Timeout after {timeout}s",
                backend="local",
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                error_summary=str(e),
                backend="local",
            )


def execute_bash(
    code: str,
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    workspace: str | None = None,
) -> ExecutionResult:
    """Execute a bash script in a subprocess.

    When workspace is provided, the script runs in that directory so files
    written by the script land on the real filesystem.
    """
    import shutil

    bash_bin = shutil.which("bash") or "/bin/bash"
    run_cwd = str(workspace) if (workspace and Path(workspace).is_dir()) else None

    with tempfile.TemporaryDirectory(prefix="aifactory_bash_") as tmpdir:
        script_path = Path(tmpdir) / "_script.sh"
        script_path.write_text(code, encoding="utf-8")
        script_path.chmod(0o755)

        try:
            result = subprocess.run(
                [bash_bin, str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=run_cwd or tmpdir,
                env={**dict(os.environ), "PYTHONIOENCODING": "utf-8"},
            )
            success = result.returncode == 0
            error_summary = ""
            if not success and result.stderr:
                lines = result.stderr.strip().splitlines()
                error_summary = lines[-1] if lines else "Unknown error"
            return ExecutionResult(
                success=success,
                exit_code=result.returncode,
                stdout=result.stdout[:10000],
                stderr=result.stderr[:10000],
                error_summary=error_summary,
                backend="local",
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=f"Execution timed out after {timeout}s",
                timed_out=True,
                error_summary=f"Timeout after {timeout}s",
                backend="local",
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                error_summary=str(e),
                backend="local",
            )


def run_pytest(
    code: str,
    test_code: str | None = None,
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> ExecutionResult:
    """
    Run code + tests via pytest in a subprocess.

    If test_code is None, assumes the code itself contains test functions.

    Args:
        code: The implementation code.
        test_code: Optional separate test file content.
        timeout: Maximum execution time.

    Returns:
        ExecutionResult from pytest execution.
    """
    extra_files: dict[str, str] = {"implementation.py": code}

    if test_code:
        runner = textwrap.dedent("""\
            import subprocess, sys
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "test_impl.py", "-v", "--tb=short", "--no-header"],
                capture_output=True, text=True, timeout=25,
            )
            print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            sys.exit(result.returncode)
        """)
        extra_files["test_impl.py"] = test_code
    else:
        runner = code

    return execute_python(runner, timeout=timeout, extra_files=extra_files)


# =============================================================================
# E2B executor
# =============================================================================


async def execute_e2b(
    code: str,
    language: str = "python",
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    api_key: str | None = None,
) -> ExecutionResult:
    """
    Execute code in an E2B cloud sandbox.

    Spins up a micro-VM, runs the code, captures output, tears it down.
    Supports Python and JavaScript.

    Args:
        code: Source code to execute.
        language: "python" or "javascript".
        timeout: Maximum execution time in seconds.
        api_key: E2B API key. Falls back to E2B_API_KEY env var.

    Returns:
        ExecutionResult with output and metadata.
    """
    try:
        from e2b_code_interpreter import Sandbox
    except ImportError:
        return ExecutionResult(
            success=False,
            exit_code=-1,
            stdout="",
            stderr="e2b-code-interpreter not installed. Run: pip install e2b-code-interpreter",
            error_summary="E2B SDK not installed",
            backend="e2b",
        )

    key = api_key if api_key is not None else _e2b_api_key()
    if not key:
        return ExecutionResult(
            success=False,
            exit_code=-1,
            stdout="",
            stderr="E2B_API_KEY not set",
            error_summary="E2B_API_KEY not set",
            backend="e2b",
        )

    sandbox = None
    try:
        sandbox = Sandbox.create(api_key=key, timeout=timeout + 10)
        execution = sandbox.run_code(code, language=language, timeout=timeout)

        stdout_text = "".join(
            r.text for r in (execution.results or []) if hasattr(r, "text") and r.text
        )
        # Also capture logs (print statements)
        log_lines = []
        for log in execution.logs.stdout or []:
            log_lines.append(log)
        stdout_combined = stdout_text or "\n".join(log_lines)

        stderr_lines = []
        for log in execution.logs.stderr or []:
            stderr_lines.append(log)
        stderr_text = "\n".join(stderr_lines)

        has_error = execution.error is not None
        error_summary = ""
        if has_error:
            err = execution.error
            error_summary = f"{err.name}: {err.value}" if err else "Unknown error"
            if err and err.traceback:
                stderr_text = err.traceback + "\n" + stderr_text

        return ExecutionResult(
            success=not has_error,
            exit_code=1 if has_error else 0,
            stdout=stdout_combined[:10000],
            stderr=stderr_text[:10000],
            error_summary=error_summary,
            backend="e2b",
        )

    except TimeoutError:
        return ExecutionResult(
            success=False,
            exit_code=-1,
            stdout="",
            stderr=f"E2B execution timed out after {timeout}s",
            timed_out=True,
            error_summary=f"Timeout after {timeout}s",
            backend="e2b",
        )

    except Exception as e:
        logger.warning("E2B execution failed: %s", e)
        return ExecutionResult(
            success=False,
            exit_code=-1,
            stdout="",
            stderr=str(e),
            error_summary=str(e),
            backend="e2b",
        )

    finally:
        if sandbox is not None:
            with contextlib.suppress(Exception):
                sandbox.kill()


# =============================================================================
# Unified execute() — auto-selects backend
# =============================================================================


def _docker_enabled() -> bool:
    """Check if Docker sandbox backend is enabled."""
    env = os.getenv("AIFACTORY_SANDBOX_DOCKER_ENABLED", "").lower()
    if env in ("true", "1", "yes"):
        return True
    if env in ("false", "0", "no"):
        return False
    try:
        from aifactory.config.settings import get_settings

        return get_settings().sandbox_docker_enabled
    except Exception:
        return False


def _get_backend() -> str:
    """Determine which sandbox backend to use (docker > e2b > local)."""
    backend = os.getenv("AIFACTORY_SANDBOX_BACKEND", "auto").lower()
    if backend != "auto":
        return backend
    if _docker_enabled():
        return "docker"
    if _e2b_api_key():
        return "e2b"
    return "local"


async def execute_docker(
    code: str,
    language: str = "python",
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    agent_id: str = "unknown",
    extra_files: dict[str, str] | None = None,
) -> ExecutionResult:
    """
    Execute code via the Docker MCP sandbox pool.

    Requires AIFACTORY_SANDBOX_DOCKER_ENABLED=true and Docker daemon.
    Falls back to E2B or local on failure.

    Args:
        code: Source code to execute.
        language: "python", "bash", or "javascript".
        timeout: Maximum execution time in seconds.
        agent_id: Calling agent identifier (for audit log).
        extra_files: Additional files to create in /workspace.

    Returns:
        ExecutionResult with output and backend="docker".
    """
    try:
        from aifactory.mcp.client import mcp_client

        async with mcp_client() as client:
            if language == "python":
                result = await client.execute_python(
                    code=code,
                    agent_id=agent_id,
                    timeout=timeout,
                    extra_files=extra_files,
                )
            elif language in ("bash", "sh"):
                result = await client.execute_bash(code=code, agent_id=agent_id, timeout=timeout)
            elif language in ("javascript", "js"):
                result = await client.execute_javascript(
                    code=code, agent_id=agent_id, timeout=timeout
                )
            else:
                return ExecutionResult(
                    success=False,
                    exit_code=-1,
                    stdout="",
                    stderr=f"Docker sandbox: unsupported language '{language}'",
                    error_summary=f"Unsupported language: {language}",
                    backend="docker",
                )

        error_summary = ""
        if not result.success and result.stderr:
            lines = result.stderr.strip().splitlines()
            error_summary = lines[-1] if lines else "Unknown error"

        return ExecutionResult(
            success=result.success,
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            timed_out=result.timed_out,
            error_summary=error_summary,
            backend="docker",
        )

    except Exception as exc:
        logger.warning("Docker sandbox failed: %s — falling back", exc)
        return ExecutionResult(
            success=False,
            exit_code=-1,
            stdout="",
            stderr=str(exc),
            error_summary=str(exc),
            backend="docker",
        )


async def execute(
    code: str,
    language: str = "python",
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    backend: str | None = None,
    agent_id: str = "unknown",
    extra_files: dict[str, str] | None = None,
    workspace: str | None = None,
) -> ExecutionResult:
    """
    Execute code using the configured sandbox backend.

    Priority: docker > e2b > local (when backend="auto").

    Args:
        code: Source code to execute.
        language: "python", "javascript", or "bash".
        timeout: Maximum execution time.
        backend: Force "docker", "e2b", or "local". None = auto-detect.
        agent_id: Calling agent identifier (for Docker audit log).
        extra_files: Additional files to create in workspace (Docker only).
        workspace: Optional local directory to use as cwd (local backend only).
                   Files created by the code will persist here.

    Returns:
        ExecutionResult with output from whichever backend ran.
    """
    chosen = backend or _get_backend()

    if chosen == "docker":
        result = await execute_docker(
            code,
            language=language,
            timeout=timeout,
            agent_id=agent_id,
            extra_files=extra_files,
        )
        # If docker failed due to infra issue (not a code error), fall through
        if not result.success and result.exit_code == -1 and not result.timed_out:
            logger.warning("Docker backend error, falling back to e2b/local")
            chosen = "e2b" if _e2b_api_key() else "local"
        else:
            return result

    if chosen == "e2b":
        return await execute_e2b(code, language=language, timeout=timeout)

    # Local backend: Python and Bash
    if language in ("bash", "sh", "shell"):
        return execute_bash(code, timeout=timeout, workspace=workspace)

    if language != "python":
        return ExecutionResult(
            success=False,
            exit_code=-1,
            stdout="",
            stderr=f"Local sandbox only supports Python and Bash, got '{language}'. "
            "Set E2B_API_KEY or enable Docker for multi-language support.",
            error_summary=f"Unsupported language: {language}",
            backend="local",
        )

    return execute_python(code, timeout=timeout, workspace=workspace)


async def execute_code_blocks(
    text: str,
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    backend: str | None = None,
    agent_id: str = "unknown",
    workspace: str | None = None,
) -> list[tuple[str, str, ExecutionResult]]:
    """
    Extract and execute all code blocks from agent output.

    Args:
        text: Agent response text potentially containing code blocks.
        timeout: Per-block execution timeout.
        backend: Force backend or None for auto.
        agent_id: Calling agent identifier (for Docker audit log).
        workspace: Optional directory to use as cwd (local backend only).

    Returns:
        List of (language, code, result) tuples for each executed block.
    """
    blocks = extract_code_blocks(text)
    results: list[tuple[str, str, ExecutionResult]] = []
    for lang, code in blocks:
        result = await execute(
            code,
            language=lang,
            timeout=timeout,
            backend=backend,
            agent_id=agent_id,
            workspace=workspace,
        )
        logger.info(
            "Code execution [%s, %s]: success=%s, %d chars output",
            lang,
            result.backend,
            result.success,
            len(result.output),
        )
        results.append((lang, code, result))
    return results
