"""
Composio Provider — Wraps Composio's Direct Execution API behind BaseIntegration.

Composio provides 1000+ tools across GitHub, Jira, Slack, Notion, Gmail, etc.
This provider uses Direct Execution mode (composio.tools.execute()) — NOT
Sessions/meta-tools — because AiFactory's pipeline already handles tool
discovery and dispatch via the ACTION: text-protocol.

Auth: COMPOSIO_API_KEY env var → Composio SDK init.
Per-tool auth: Managed by Composio (run `composio add github` etc. via CLI).

Optional dependency: `composio` package. Graceful degradation when not installed
(same pattern as E2B in sandbox.py).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from aifactory.integrations.base import (
    ActionResult,
    BaseIntegration,
    IntegrationCapability,
    IntegrationCategory,
    IntegrationConfig,
    IntegrationStatus,
)

logger = logging.getLogger(__name__)


def _composio_api_key() -> str:
    """Get Composio API key from env var or settings (.env file)."""
    key = os.getenv("COMPOSIO_API_KEY", "").strip()
    if key:
        return key
    try:
        from aifactory.config.settings import get_settings

        return (get_settings().composio_api_key or "").strip()
    except Exception:
        return ""


# Default config — used when no YAML config is provided
_DEFAULT_CONFIG = IntegrationConfig(
    name="composio",
    display_name="Composio",
    category=IntegrationCategory.DEVOPS,
    capabilities=[
        IntegrationCapability.READ,
        IntegrationCapability.WRITE,
        IntegrationCapability.SEARCH,
        IntegrationCapability.EXECUTE,
    ],
    auth_type="api_key",
    auth_env_var="COMPOSIO_API_KEY",
    enabled_toolkits=[
        "github",
        "gitlab",
        "jira",
        "notion",
        "gmail",
        "linear",
        "googledrive",
        "googlecalendar",
    ],
    default_user_id="default",
    rate_limit_rpm=100,
)


class ComposioProvider(BaseIntegration):
    """
    Composio integration via Direct Execution API.

    Uses composio.tools.execute() for deterministic tool calls.
    Workers specify which tool to run via ACTION: blocks; the pipeline
    dispatches here.

    Requires: pip install composio (optional dep, guarded at import time).
    """

    def __init__(self, config: IntegrationConfig | None = None) -> None:
        super().__init__(config or _DEFAULT_CONFIG)
        self._api_key: str = ""
        self._client: Any = None  # composio.Composio instance
        self._available_tools: dict[str, list[str]] = {}  # toolkit → [tool_slugs]

    # ── Lifecycle ────────────────────────────────────────────────────

    async def connect(self) -> bool:
        """Validate API key and initialise the Composio client."""
        self.status = IntegrationStatus.CONNECTING
        self._api_key = _composio_api_key()
        if not self._api_key:
            self._set_error(f"Missing {self.config.auth_env_var} env var")
            return False

        try:
            from composio import Composio
        except ImportError:
            self._set_error("composio package not installed — run: uv add composio")
            return False

        try:
            self._client = Composio(api_key=self._api_key)
            # Validate by listing tools for the first enabled toolkit
            ok = await self.health_check()
            if ok:
                self._set_connected()
            return ok
        except Exception as e:
            self._set_error(f"Composio init failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Release the Composio client."""
        self._client = None
        self._available_tools.clear()
        self._set_disconnected()

    async def health_check(self) -> bool:
        """Verify the Composio API key is valid by listing tools."""
        if not self._client:
            return False
        try:
            # Quick validation — fetch one tool from the first enabled toolkit
            toolkits = self.config.enabled_toolkits
            if toolkits:
                tools = self._client.tools.get(
                    self.config.default_user_id,
                    toolkits=[toolkits[0]],
                    limit=1,
                )
                logger.info(
                    "Composio health check: found %d tools for toolkit '%s'",
                    len(tools) if tools else 0,
                    toolkits[0],
                )
            return True
        except Exception as e:
            logger.error("Composio health check failed: %s", e)
            return False

    # ── Actions ──────────────────────────────────────────────────────

    def get_available_actions(self) -> list[str]:
        """Return cached tool slugs across all enabled toolkits."""
        all_slugs: list[str] = []
        for slugs in self._available_tools.values():
            all_slugs.extend(slugs)
        return all_slugs

    def get_available_toolkits(self) -> list[str]:
        """Return the enabled toolkit names."""
        return list(self.config.enabled_toolkits)

    async def execute_action(self, action: str, params: dict[str, Any]) -> ActionResult:
        """
        Execute a Composio tool via Direct Execution.

        Args:
            action: Tool slug, e.g. "GITHUB_CREATE_ISSUE"
            params: Tool arguments dict, e.g. {"owner": "org", "repo": "r", "title": "Bug"}

        Returns:
            ActionResult with success/data or error.
        """
        if not self._client:
            return ActionResult(
                success=False,
                action=action,
                integration=self.name,
                error="Composio not connected",
            )

        try:
            result = self._client.tools.execute(
                action,
                params,
                user_id=self.config.default_user_id,
                dangerously_skip_version_check=True,
            )

            # Composio SDK v1.x returns a ToolExecutionResponse object
            result_data: dict[str, Any] = {}
            if hasattr(result, "model_dump"):
                result_data = result.model_dump()
            elif hasattr(result, "data"):
                result_data = (
                    result.data if isinstance(result.data, dict) else {"result": str(result.data)}
                )
            elif isinstance(result, dict):
                result_data = result
            else:
                result_data = {"result": str(result)}

            # Check for errors in the response
            is_error = False
            error_msg = ""
            if isinstance(result_data, dict):
                if result_data.get("error"):
                    is_error = True
                    error_msg = str(result_data["error"])
                elif result_data.get("status") == "error":
                    is_error = True
                    error_msg = result_data.get("message", "Unknown Composio error")

            if is_error:
                return ActionResult(
                    success=False,
                    action=action,
                    integration=self.name,
                    data=result_data,
                    error=error_msg,
                )

            # Extract URL if present (GitHub issues, PRs, etc.)
            url = ""
            if isinstance(result_data, dict):
                url = result_data.get("html_url", "") or result_data.get("url", "")

            return ActionResult(
                success=True,
                action=action,
                integration=self.name,
                data=result_data,
                url=url,
            )

        except Exception as e:
            logger.error("Composio execute_action failed: %s/%s: %s", self.name, action, e)
            return ActionResult(
                success=False,
                action=action,
                integration=self.name,
                error=str(e),
            )

    # ── Tool Discovery ───────────────────────────────────────────────

    async def fetch_available_tools(
        self, toolkits: list[str] | None = None
    ) -> dict[str, list[str]]:
        """
        Fetch available tool slugs from Composio for the given toolkits.

        Caches results on the instance. Call after connect().
        """
        if not self._client:
            return {}

        target_toolkits = toolkits or self.config.enabled_toolkits
        for tk in target_toolkits:
            if tk in self._available_tools:
                continue
            try:
                tools = self._client.tools.get(
                    self.config.default_user_id,
                    toolkits=[tk],
                    limit=100,
                )
                slugs = []
                if tools:
                    for t in tools:
                        # SDK v1.x: tools are dicts with function.name
                        slug = None
                        if isinstance(t, dict):
                            func = t.get("function", {})
                            slug = func.get("name") if isinstance(func, dict) else t.get("name")
                        else:
                            slug = getattr(t, "name", None)
                        if slug:
                            slugs.append(slug)
                self._available_tools[tk] = slugs
                logger.info("Composio: found %d tools for toolkit '%s'", len(slugs), tk)
            except Exception as e:
                logger.warning("Failed to fetch tools for toolkit '%s': %s", tk, e)
                self._available_tools[tk] = []

        return dict(self._available_tools)

    def get_tools_summary_for_prompt(self) -> str:
        """
        Build a concise summary of available tools for injection into worker prompts.

        Returns a string like:
          github: GITHUB_CREATE_ISSUE, GITHUB_LIST_ISSUES, ...
          slack: SLACK_SEND_MESSAGE, ...
        """
        if not self._available_tools:
            # Fall back to generic toolkit names
            return ", ".join(self.config.enabled_toolkits)

        lines: list[str] = []
        for tk, slugs in sorted(self._available_tools.items()):
            if slugs:
                # Show first 8 tools per toolkit to avoid prompt bloat
                display = slugs[:8]
                suffix = f" (+{len(slugs) - 8} more)" if len(slugs) > 8 else ""
                lines.append(f"  {tk}: {', '.join(display)}{suffix}")
            else:
                lines.append(f"  {tk}: (available — use ACTION: block to invoke)")
        return "\n".join(lines)
