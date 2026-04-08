"""
Integration Action Executor — Detects and dispatches ACTION: blocks in worker output.

Workers invoke integrations via a text protocol (same pattern as CODE: blocks
in sandbox.py):

    ACTION:composio
    {"tool": "GITHUB_CREATE_ISSUE", "params": {"owner": "org", "repo": "r", "title": "Bug"}}
    ACTION_END

The pipeline calls execute_integration_actions() after each worker's LLM call,
detects these blocks, dispatches to the IntegrationRegistry, and appends
ActionResult output back into the worker's result string.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from aifactory.integrations.base import ActionResult

logger = logging.getLogger(__name__)


# =============================================================================
# ACTION: block extraction (mirrors sandbox.py _CODE_BLOCK_RE)
# =============================================================================

# Matches:
#   ACTION:integration_name\n{json}\nACTION_END
#   ACTION:composio\n{"tool": "GITHUB_CREATE_ISSUE", "params": {...}}\nACTION_END
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
