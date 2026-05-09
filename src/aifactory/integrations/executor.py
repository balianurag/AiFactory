"""
Integration Action Executor — Detects and dispatches ACTION: blocks in worker output.

Workers invoke integrations via a text protocol (same pattern as CODE: blocks
in sandbox.py):

    ACTION:mcp
    {"tool": "github::create_issue", "params": {"owner": "org", "repo": "r", "title": "Bug"}}
    ACTION_END

The pipeline calls execute_integration_actions() after each worker's LLM call,
detects these blocks, dispatches to the IntegrationRegistry, and appends
ActionResult output back into the worker's result string.

Code Leakage Prevention
-----------------------
_sanitize_params() is applied to every ACTION: payload before it reaches the
IntegrationRegistry (and from there, any MCP server).  Any string
parameter that looks like source code is replaced with a safe redaction notice.
Raw source code must NEVER travel over the external API boundary — only
safe abstractions (summaries, metadata, embedding IDs) may be sent.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from aifactory.integrations.base import ActionResult

logger = logging.getLogger(__name__)


# =============================================================================
# Code Leakage Prevention — param sanitizer
# =============================================================================

# Signals that strongly indicate source code rather than normal text.
# Each signal is a substring found in virtually all programming languages.
_CODE_SIGNALS: tuple[str, ...] = (
    "def ",  # Python / Ruby function
    "class ",  # Python / JS / Java class
    "import ",  # Python / JS / Java import
    "function ",  # JS / PHP / Go function
    "return ",  # universal
    "const ",  # JS / C++ / Rust
    "let ",  # JS / Rust
    "var ",  # JS / older langs
    "    #",  # indented comment (Python style)
    "    //",  # indented comment (C-style)
    "=>",  # arrow function
    "#!/",  # shebang
    "```",  # fenced code block in markdown
)

# Minimum string length before we even check for code signals (short strings
# like "import CSV" are fine to send; 300 chars is a conservative threshold).
_MIN_CODE_LEN = 300

# Number of distinct signals that must match before we treat a value as code.
_CODE_SIGNAL_THRESHOLD = 3


def _looks_like_code(value: str) -> bool:
    """
    Return True when *value* appears to contain source code.

    Heuristic: the string is long AND matches at least _CODE_SIGNAL_THRESHOLD
    distinct code signals.  This avoids false positives on short technical
    strings while catching multi-line code blobs reliably.

    >>> _looks_like_code("def foo():\\n    return 1\\nimport os\\nclass Bar:\\n    pass")
    True
    >>> _looks_like_code("Create a GitHub issue about the login bug")
    False
    """
    if len(value) < _MIN_CODE_LEN:
        return False
    hits = sum(1 for sig in _CODE_SIGNALS if sig in value)
    return hits >= _CODE_SIGNAL_THRESHOLD


def _sanitize_params(params: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """
    Walk *params* recursively and redact any value that looks like source code.

    Returns:
        (sanitized_params, redacted_keys) — the cleaned dict and a list of
        key-paths that were redacted (for logging / audit trail).

    Handles nested dicts and lists.  Non-string values are passed through unchanged.
    """
    redacted: list[str] = []

    def _clean(obj: Any, path: str) -> Any:
        if isinstance(obj, str):
            if _looks_like_code(obj):
                redacted.append(path)
                return (
                    f"[CODE REDACTED — {len(obj)} chars of source code were removed before "
                    "this payload reached the external API.  Use the CODE_ANALYSIS: protocol "
                    "or a local read_file tool to process code without leaking it.]"
                )
            return obj
        if isinstance(obj, dict):
            return {k: _clean(v, f"{path}.{k}") for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(item, f"{path}[{i}]") for i, item in enumerate(obj)]
        return obj

    cleaned = _clean(params, "params")
    return cleaned, redacted


# =============================================================================
# ACTION: block extraction (mirrors sandbox.py _CODE_BLOCK_RE)
# =============================================================================

# Matches:
#   ACTION:integration_name\n{json}\nACTION_END
#   ACTION:mcp\n{"tool": "github::create_issue", "params": {...}}\nACTION_END
_ACTION_BLOCK_RE = re.compile(
    r"ACTION:(\w+)\n(.*?)ACTION_END",
    re.DOTALL,
)


def extract_action_blocks(text: str) -> list[tuple[str, dict[str, Any]]]:
    """
    Extract ACTION: blocks from agent output.

    Returns list of (integration_name, parsed_json) tuples.

    Each JSON block must have:
      - "tool": str — the action/tool slug (e.g. "GITHUB_CREATE_ISSUE")
      - "params": dict — arguments to pass to the tool

    Malformed JSON blocks are logged and skipped.
    """
    blocks: list[tuple[str, dict[str, Any]]] = []
    for m in _ACTION_BLOCK_RE.finditer(text):
        integration_name = m.group(1).strip().lower()
        json_str = m.group(2).strip()
        try:
            parsed = json.loads(json_str)
            if not isinstance(parsed, dict):
                logger.warning(
                    "ACTION block for '%s' is not a JSON object — skipping",
                    integration_name,
                )
                continue
            if "tool" not in parsed:
                logger.warning(
                    "ACTION block for '%s' missing 'tool' key — skipping",
                    integration_name,
                )
                continue
            blocks.append((integration_name, parsed))
        except json.JSONDecodeError as e:
            logger.warning(
                "ACTION block for '%s' has invalid JSON: %s — skipping",
                integration_name,
                e,
            )
    return blocks


# =============================================================================
# Dispatch (called from dynamic_pipeline.py after each worker)
# =============================================================================


async def execute_integration_actions(
    text: str,
    registry: Any | None = None,
) -> list[tuple[str, ActionResult]]:
    """
    Detect ACTION: blocks in worker output and dispatch to the registry.

    Args:
        text: Raw worker output (may contain ACTION: blocks among other content).
        registry: IntegrationRegistry instance. If None, uses singleton.

    Returns:
        List of (tool_slug, ActionResult) tuples for each executed action.
        Empty list if no ACTION: blocks found or registry unavailable.
    """
    if registry is None:
        try:
            from aifactory.integrations.registry import IntegrationRegistry

            registry = IntegrationRegistry()
        except Exception:
            return []

    blocks = extract_action_blocks(text)
    if not blocks:
        return []

    results: list[tuple[str, ActionResult]] = []
    for integration_name, parsed in blocks:
        tool = parsed.get("tool", "unknown")
        params = parsed.get("params", {})
        if not isinstance(params, dict):
            params = {}

        # Code leakage prevention — redact source code before it leaves the process
        params, redacted_keys = _sanitize_params(params)
        if redacted_keys:
            logger.warning(
                "CODE LEAK PREVENTED [%s/%s]: redacted %d param(s) containing source code: %s. "
                "Use CODE_ANALYSIS: protocol for codebase tasks.",
                integration_name,
                tool,
                len(redacted_keys),
                redacted_keys,
            )

        logger.info(
            "Dispatching integration action: %s/%s with %d params",
            integration_name,
            tool,
            len(params),
        )

        result = await registry.execute(integration_name, tool, params)
        results.append((tool, result))

        status = "✓" if result.success else "✗"
        logger.info(
            "Integration action [%s/%s]: %s%s",
            integration_name,
            tool,
            status,
            f" — {result.error}" if result.error else "",
        )

    return results
