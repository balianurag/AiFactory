"""
Code Execution Sandbox — Secure subprocess runner for agent-generated code.

Runs Python code in an isolated subprocess with:
  - Timeout enforcement (default 30s)
  - Memory limit (default 256MB via resource.setrlimit on Linux/macOS)
  - Network access disabled (optional, future)
  - stdout + stderr capture
  - Exit code tracking

This is NOT a full container sandbox (E2B / Docker) — it's a lightweight
local subprocess runner for validation. Suitable for:
  - Running pytest / inline tests on generated code
  - Validating syntax (compile check)
  - Catching obvious runtime errors

For untrusted code in production, swap this for E2B or a Docker sidecar.
"""

from __future__ import annotations

import ast
import logging
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


@dataclass
class ExecutionResult:
    """Result of running code in the sandbox."""

    success: bool
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    error_summary: str = ""

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


def execute_python(
    code: str,
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    memory_mb: int = DEFAULT_MEMORY_MB,
    python_executable: str | None = None,
    extra_files: dict[str, str] | None = None,
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

    with tempfile.TemporaryDirectory(prefix="aifactory_sandbox_") as tmpdir:
        tmppath = Path(tmpdir)

        # Write extra files if any
        if extra_files:
            for fname, fcontent in extra_files.items():
                (tmppath / fname).write_text(fcontent, encoding="utf-8")

        # Write main script
        script_path = tmppath / "_main.py"
        script_path.write_text(code, encoding="utf-8")

        # Build the subprocess command with memory limiting wrapper
        # On Unix, we use a small wrapper that sets resource limits
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

        try:
            result = subprocess.run(
                [py, str(wrapper_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(tmppath),
                env={
                    **dict(__import__("os").environ),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONIOENCODING": "utf-8",
                },
            )

            success = result.returncode == 0
            error_summary = ""
            if not success and result.stderr:
                # Extract the last line of traceback as the summary
                lines = result.stderr.strip().splitlines()
                error_summary = lines[-1] if lines else "Unknown error"

            return ExecutionResult(
                success=success,
                exit_code=result.returncode,
                stdout=result.stdout[:10000],  # cap output
                stderr=result.stderr[:10000],
                error_summary=error_summary,
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=f"Execution timed out after {timeout}s",
                timed_out=True,
                error_summary=f"Timeout after {timeout}s",
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                error_summary=str(e),
            )


def run_pytest(
    code: str,
    test_code: str | None = None,
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> ExecutionResult:
    """
    Run code + tests via pytest in a subprocess.

    If test_code is None, assumes the code itself contains test functions
    (e.g., inline if __name__ == "__main__" or test_* functions).

    Args:
        code: The implementation code.
        test_code: Optional separate test file content.
        timeout: Maximum execution time.

    Returns:
        ExecutionResult from pytest execution.
    """
    extra_files: dict[str, str] = {"implementation.py": code}

    if test_code:
        # Run separate test file that imports from implementation
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
        # Run the code directly — it should contain inline tests
        runner = code

    return execute_python(runner, timeout=timeout, extra_files=extra_files)
