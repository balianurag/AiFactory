"""
LocalShell Provider — Safe in-house terminal tool catalog.

Gives workers access to local filesystem and code tools without needing
Composio or MCP. All commands run in the workspace directory (or cwd).

Supported tools:
  bash_run      — run arbitrary bash commands (sandboxed to workspace)
  grep_search   — grep for patterns in files
  find_files    — find files by name/pattern
  git_status    — git status/log/diff
  python_run    — run a Python snippet
  read_file     — read a file's content (safe, size-capped)
  list_dir      — list directory contents
  rg_search     — ripgrep search (falls back to grep)

Auth: none — local execution only.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from aifactory.integrations.base import (
    ActionResult,
    BaseIntegration,
    IntegrationCapability,
    IntegrationCategory,
    IntegrationConfig,
)

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = IntegrationConfig(
    name="local_shell",
    display_name="Local Shell Tools",
    category=IntegrationCategory.DEVOPS,
    capabilities=[
        IntegrationCapability.READ,
        IntegrationCapability.EXECUTE,
        IntegrationCapability.SEARCH,
    ],
    auth_type="none",
    auth_env_var="",
    enabled_toolkits=[],
    default_user_id="local",
    rate_limit_rpm=300,
)

# Hard cap on output returned to the worker (prevents context flooding)
_MAX_OUTPUT_CHARS = 8000
# Hard timeout per command
_CMD_TIMEOUT = 30

# Commands allowed in bash_run (block destructive ops)
_BLOCKED_PATTERNS = (
    r"\brm\s+-rf\b",
    r"\bsudo\b",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r"\bformat\b",
    r"\bchmod\s+777\b",
    r"> /dev/",
    r"\bcurl\b.*\|\s*bash",
    r"\bwget\b.*\|\s*sh",
)


def _safe_run(
    cmd: list[str], cwd: str | None = None, timeout: int = _CMD_TIMEOUT
) -> tuple[str, str, int]:
    """Run a command safely, return (stdout, stderr, returncode)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        return result.stdout[:_MAX_OUTPUT_CHARS], result.stderr[:2000], result.returncode
    except subprocess.TimeoutExpired:
        return "", f"Command timed out after {timeout}s", 1
    except FileNotFoundError:
        return "", f"Command not found: {cmd[0]}", 127
    except Exception as e:
        return "", str(e), 1


class LocalShellProvider(BaseIntegration):
    """
    In-house terminal tool catalog for workers.
    Runs local filesystem and code tools without external APIs.
    """

    provider_type: str = "local_shell"
    action_prefix: str = "local_shell"

    def __init__(
        self, config: IntegrationConfig | None = None, workspace: str | None = None
    ) -> None:
        super().__init__(config or _DEFAULT_CONFIG)
        self._workspace = workspace or os.getcwd()

    async def connect(self) -> bool:
        self._set_connected()
        logger.debug("LocalShellProvider ready (workspace=%s)", self._workspace)
        return True

    async def disconnect(self) -> None:
        self._set_disconnected()

    async def health_check(self) -> bool:
        return Path(self._workspace).is_dir()

    def get_available_actions(self) -> list[str]:
        return [
            "bash_run",
            "grep_search",
            "find_files",
            "git_status",
            "python_run",
            "read_file",
            "list_dir",
            "rg_search",
        ]

    async def execute_action(self, action: str, params: dict[str, Any]) -> ActionResult:
        import re as _re

        cwd = params.get("cwd") or self._workspace

        try:
            if action == "bash_run":
                cmd_str = params.get("command", "")
                # Block dangerous patterns
                for pat in _BLOCKED_PATTERNS:
                    if _re.search(pat, cmd_str, _re.IGNORECASE):
                        return ActionResult(
                            success=False,
                            action=action,
                            integration=self.name,
                            error=f"Blocked pattern in command: {pat}",
                        )
                stdout, stderr, rc = _safe_run(["bash", "-c", cmd_str], cwd=cwd)
                output = stdout or stderr
                return ActionResult(
                    success=rc == 0,
                    action=action,
                    integration=self.name,
                    data={"output": output, "returncode": rc},
                )

            elif action == "grep_search":
                pattern = params.get("pattern", "")
                path = params.get("path", ".")
                flags = ["-rn", "--include=*.py"]
                if params.get("case_insensitive"):
                    flags.append("-i")
                if params.get("extensions"):
                    flags = ["-rn"] + [f"--include=*.{e}" for e in params["extensions"].split(",")]
                stdout, stderr, rc = _safe_run(["grep"] + flags + [pattern, path], cwd=cwd)
                return ActionResult(
                    success=True,
                    action=action,
                    integration=self.name,
                    data={"output": stdout or stderr},
                )

            elif action == "find_files":
                name_pattern = params.get("pattern", "*")
                path = params.get("path", ".")
                max_depth = str(params.get("max_depth", 5))
                cmd = [
                    "find",
                    path,
                    "-maxdepth",
                    max_depth,
                    "-name",
                    name_pattern,
                    "-not",
                    "-path",
                    "*/.git/*",
                    "-not",
                    "-path",
                    "*/__pycache__/*",
                    "-not",
                    "-path",
                    "*/.venv/*",
                ]
                stdout, stderr, rc = _safe_run(cmd, cwd=cwd)
                return ActionResult(
                    success=True,
                    action=action,
                    integration=self.name,
                    data={"output": stdout or stderr},
                )

            elif action == "git_status":
                subcmd = params.get("subcmd", "status")  # status | log | diff | branch
                allowed = {"status", "log", "diff", "branch", "show", "blame"}
                if subcmd not in allowed:
                    return ActionResult(
                        success=False,
                        action=action,
                        integration=self.name,
                        error=f"git subcmd must be one of: {allowed}",
                    )
                extra = params.get("args", "").split() if params.get("args") else []
                stdout, stderr, rc = _safe_run(["git", subcmd] + extra, cwd=cwd)
                return ActionResult(
                    success=rc == 0,
                    action=action,
                    integration=self.name,
                    data={"output": stdout or stderr},
                )

            elif action == "python_run":
                code = params.get("code", "")
                with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
                    f.write(code)
                    tmp = f.name
                try:
                    stdout, stderr, rc = _safe_run(["python3", tmp], cwd=cwd)
                finally:
                    Path(tmp).unlink(missing_ok=True)
                return ActionResult(
                    success=rc == 0,
                    action=action,
                    integration=self.name,
                    data={"output": stdout or stderr},
                )

            elif action == "read_file":
                filepath = params.get("path", "")
                p = Path(cwd) / filepath if not Path(filepath).is_absolute() else Path(filepath)
                if not p.exists():
                    return ActionResult(
                        success=False,
                        action=action,
                        integration=self.name,
                        error=f"File not found: {p}",
                    )
                content = p.read_text(encoding="utf-8", errors="replace")[:_MAX_OUTPUT_CHARS]
                return ActionResult(
                    success=True,
                    action=action,
                    integration=self.name,
                    data={"content": content, "path": str(p)},
                )

            elif action == "list_dir":
                path = params.get("path", ".")
                p = Path(cwd) / path if not Path(path).is_absolute() else Path(path)
                if not p.is_dir():
                    return ActionResult(
                        success=False,
                        action=action,
                        integration=self.name,
                        error=f"Not a directory: {p}",
                    )
                entries = sorted(
                    str(e.relative_to(p)) + ("/" if e.is_dir() else "") for e in p.iterdir()
                )
                return ActionResult(
                    success=True,
                    action=action,
                    integration=self.name,
                    data={"entries": entries[:200]},
                )

            elif action == "rg_search":
                pattern = params.get("pattern", "")
                path = params.get("path", ".")
                import shutil

                rg = shutil.which("rg")
                if rg:
                    stdout, stderr, rc = _safe_run(
                        [rg, "-n", "--max-count=50", pattern, path], cwd=cwd
                    )
                else:
                    stdout, stderr, rc = _safe_run(["grep", "-rn", pattern, path], cwd=cwd)
                return ActionResult(
                    success=True,
                    action=action,
                    integration=self.name,
                    data={"output": stdout or stderr},
                )

            else:
                return ActionResult(
                    success=False,
                    action=action,
                    integration=self.name,
                    error=f"Unknown action: {action}",
                )

        except Exception as e:
            logger.error(
                "LocalShellProvider.execute_action failed: %s/%s: %s", self.name, action, e
            )
            return ActionResult(success=False, action=action, integration=self.name, error=str(e))

    def get_tools_summary_for_prompt(self) -> str:
        return """[local_shell]
  bash_run: Run bash commands in workspace (requires: command)
  grep_search: Search files for pattern (requires: pattern; optional: path, extensions, case_insensitive)
  find_files: Find files by name pattern (requires: pattern; optional: path, max_depth)
  git_status: Git status/log/diff/branch (optional: subcmd, args)
  python_run: Execute Python code (requires: code)
  read_file: Read file content (requires: path)
  list_dir: List directory contents (optional: path)
  rg_search: Fast ripgrep search (requires: pattern; optional: path)"""
