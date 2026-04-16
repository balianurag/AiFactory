"""
MCP Provider — Connects directly to MCP servers via the Python MCP SDK.

Supports two deployment modes controlled by AIFACTORY_DEPLOY_MODE:

  local (default)
    Each enabled server is spawned as a stdio subprocess per connect() call
    (docker run / uvx / npx command defined in mcp_catalog.yaml).
    Simple to use — no infrastructure needed.  Not suitable for concurrent
    production load because each pipeline run pays the subprocess startup cost
    (500ms–2s) and cannot safely share sessions.

  cloud
    Connects to persistent HTTP MCP server services that are deployed
    separately (e.g. as Docker containers, Kubernetes pods, or managed services).
    Each server's base URL is read from the env var named in `url_env_var`.
    MCPProvider maintains an asyncio.Queue-based session pool
    (AIFACTORY_MCP_HTTP_POOL_SIZE sessions per server, default 3).
    Concurrent pipeline runs acquire a session from the pool, execute the
    tool call, and release it back — zero subprocess spawning, sub-millisecond
    acquire time after warm-up.

Auth: each server defines env_vars that map host OS env var names to subprocess
env var names.  If any required env var is missing the server is skipped
(non-fatal — logged at INFO level).

Optional dependency: `mcp` package.  Graceful degradation when not installed
(same pattern as E2B / Composio).
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import AsyncExitStack, suppress
from pathlib import Path
from typing import Any

import yaml

from aifactory.core.retry import retry_with_backoff
from aifactory.integrations.base import (
    ActionResult,
    BaseIntegration,
    IntegrationCapability,
    IntegrationCategory,
    IntegrationConfig,
    IntegrationStatus,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional MCP SDK imports — guarded for graceful degradation
# ---------------------------------------------------------------------------
try:
    from mcp import ClientSession, StdioServerParameters  # type: ignore[import]
    from mcp.client.stdio import stdio_client  # type: ignore[import]

    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False
    ClientSession = None  # type: ignore[assignment,misc]
    StdioServerParameters = None  # type: ignore[assignment,misc]
    stdio_client = None  # type: ignore[assignment]

# HTTP transport: try streamable-http first, fall back to SSE
_HTTP_CLIENT: Any = None
try:
    from mcp.client.streamable_http import streamablehttp_client  # type: ignore[import]

    _HTTP_CLIENT = streamablehttp_client
except ImportError:
    try:
        from mcp.client.sse import sse_client  # type: ignore[import]

        _HTTP_CLIENT = sse_client
    except ImportError:
        pass

# ---------------------------------------------------------------------------
# Catalog path
# ---------------------------------------------------------------------------

_CATALOG_PATH = Path(__file__).resolve().parents[2] / "config" / "integrations" / "mcp_catalog.yaml"

# Sentinel stored in _sessions to mark a server as pool-backed
_POOLED = "__pooled__"

# Default config for MCPProvider
_DEFAULT_CONFIG = IntegrationConfig(
    name="mcp_catalog",
    display_name="MCP Catalog",
    category=IntegrationCategory.DEVOPS,
    capabilities=[
        IntegrationCapability.READ,
        IntegrationCapability.WRITE,
        IntegrationCapability.EXECUTE,
    ],
    auth_type="api_key",
    auth_env_var="",
)


class MCPProvider(BaseIntegration):
    """
    Integration provider that connects directly to MCP servers.

    Reads server configs from config/integrations/mcp_catalog.yaml.
    Keeps MCP ClientSession objects alive across multiple tool calls.

    Transport strategy is driven by AIFACTORY_DEPLOY_MODE:
      - local  → stdio subprocess per server (simple, dev-friendly)
      - cloud  → HTTP session pool per server (HA, zero subprocess overhead)

    Tool slugs use the format "server::tool_name" (e.g. "github::create_issue").
    """

    provider_type: str = "mcp"
    action_prefix: str = "mcp"

    def __init__(self, config: IntegrationConfig | None = None) -> None:
        super().__init__(config or _DEFAULT_CONFIG)
        self._server_configs: dict[str, dict[str, Any]] = {}

        # server_name → ClientSession (stdio) | _POOLED (http pool)
        self._sessions: dict[str, Any] = {}

        # stdio lifecycle manager
        self._exit_stack: AsyncExitStack = AsyncExitStack()

        # cloud-mode pool structures
        # server_name → asyncio.Queue[ClientSession]
        self._http_session_pools: dict[str, asyncio.Queue] = {}
        # server_name → list[AsyncExitStack]  (one stack per pooled session)
        self._http_pool_stacks: dict[str, list[AsyncExitStack]] = {}

        # toolkit → list of {slug, description, required}
        self._tool_metadata: dict[str, list[dict[str, Any]]] = {}
        # server_name → list of tool names (raw)
        self._server_tools: dict[str, list[str]] = {}

    # ── Catalog loading ──────────────────────────────────────────────

    def _load_catalog(self) -> None:
        """Load server configs from mcp_catalog.yaml into self._server_configs."""
        try:
            if not _CATALOG_PATH.exists():
                logger.warning("mcp_catalog.yaml not found at %s", _CATALOG_PATH)
                return
            raw = yaml.safe_load(_CATALOG_PATH.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                logger.warning("mcp_catalog.yaml: expected a dict at top level")
                return
            servers = raw.get("servers", {})
            if not isinstance(servers, dict):
                logger.warning("mcp_catalog.yaml: 'servers' key must be a dict")
                return
            self._server_configs = {
                name: cfg for name, cfg in servers.items() if isinstance(cfg, dict)
            }
            logger.debug(
                "MCPProvider: loaded %d server configs from catalog",
                len(self._server_configs),
            )
        except Exception as exc:
            logger.error("MCPProvider: failed to load catalog: %s", exc)

    # ── Helpers ──────────────────────────────────────────────────────

    def _extract_required_params(self, input_schema: dict[str, Any]) -> list[str]:
        """Extract required field names from a JSON Schema object."""
        if not isinstance(input_schema, dict):
            return []
        return list(input_schema.get("required", []))

    def _build_server_env(self, server_cfg: dict[str, Any]) -> dict[str, str] | None:
        """
        Build subprocess env dict for the server.

        Returns None if any required env var is missing (skip server silently).
        The env_vars mapping: key = env var name passed to subprocess,
        value = env var name to read from host OS.
        """
        env_vars: dict[str, str] = server_cfg.get("env_vars", {})
        if not env_vars:
            return {}

        result: dict[str, str] = {}
        for subprocess_key, host_key in env_vars.items():
            value = os.environ.get(host_key, "").strip()
            if not value:
                logger.info(
                    "MCPProvider: skipping server '%s' — missing env var %s",
                    server_cfg.get("display_name", "?"),
                    host_key,
                )
                return None
            result[subprocess_key] = value
        return result

    # ── Stdio transport (local mode) ─────────────────────────────────

    async def _connect_stdio_server(
        self,
        server_name: str,
        server_cfg: dict[str, Any],
        server_env: dict[str, str],
    ) -> bool:
        """Spawn a stdio subprocess MCP server and list its tools."""
        if not _MCP_AVAILABLE:
            return False

        command: str = server_cfg.get("command", "")
        args: list[str] = server_cfg.get("args", [])

        if not command:
            logger.warning("MCPProvider: server '%s' has no command", server_name)
            return False

        try:
            params = StdioServerParameters(
                command=command,
                args=[str(a) for a in args],
                env=server_env if server_env else None,
            )
            stdio_transport = await self._exit_stack.enter_async_context(stdio_client(params))
            read_stream, write_stream = stdio_transport
            session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await session.initialize()
            self._sessions[server_name] = session
            await self._fetch_server_tools(server_name, session, server_cfg)
            logger.info(
                "MCPProvider [local]: connected '%s' (%d tools)",
                server_name,
                len(self._server_tools.get(server_name, [])),
            )
            return True
        except Exception as exc:
            logger.warning("MCPProvider: failed to connect server '%s': %s", server_name, exc)
            return False

    # ── HTTP transport — single session (legacy / direct URL in catalog) ──

    async def _connect_http_server(
        self,
        server_name: str,
        server_cfg: dict[str, Any],
    ) -> bool:
        """Connect an HTTP-transport MCP server (single session, no pool)."""
        if not _MCP_AVAILABLE or _HTTP_CLIENT is None:
            logger.warning("MCPProvider: HTTP transport not available for '%s'", server_name)
            return False

        url: str = server_cfg.get("url", "")
        if not url:
            logger.warning("MCPProvider: server '%s' has no url for HTTP transport", server_name)
            return False

        try:
            http_transport = await self._exit_stack.enter_async_context(_HTTP_CLIENT(url))
            read_stream, write_stream = http_transport
            session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await session.initialize()
            self._sessions[server_name] = session
            await self._fetch_server_tools(server_name, session, server_cfg)
            logger.info(
                "MCPProvider [http-single]: connected '%s' (%d tools)",
                server_name,
                len(self._server_tools.get(server_name, [])),
            )
            return True
        except Exception as exc:
            logger.warning("MCPProvider: failed to connect HTTP server '%s': %s", server_name, exc)
            return False

    # ── HTTP session pool (cloud mode) ───────────────────────────────

    async def _create_http_session(self, url: str) -> tuple[Any, AsyncExitStack]:
        """
        Create a single HTTP MCP session with its own lifecycle stack.

        Returns (session, stack).  Caller owns the stack and must call
        stack.aclose() when the session is retired.
        """
        if not _MCP_AVAILABLE or _HTTP_CLIENT is None:
            raise RuntimeError("MCP HTTP transport not available")

        stack = AsyncExitStack()
        try:
            http_transport = await stack.enter_async_context(_HTTP_CLIENT(url))
            read_stream, write_stream = http_transport
            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()
            return session, stack
        except Exception:
            await stack.aclose()
            raise

    async def _connect_http_server_pooled(
        self,
        server_name: str,
        server_cfg: dict[str, Any],
        url: str,
        pool_size: int,
    ) -> bool:
        """
        Create a pool of `pool_size` independent HTTP sessions for the server.

        Sessions are stored in an asyncio.Queue.  execute_action() acquires a
        session, calls the tool, then releases it back — safe for concurrent use.
        """
        if not _MCP_AVAILABLE or _HTTP_CLIENT is None:
            logger.warning(
                "MCPProvider: HTTP transport not available — cannot pool '%s'", server_name
            )
            return False

        queue: asyncio.Queue = asyncio.Queue(maxsize=pool_size)
        stacks: list[AsyncExitStack] = []

        for i in range(pool_size):
            try:
                session, stack = await self._create_http_session(url)
                queue.put_nowait(session)
                stacks.append(stack)
            except Exception as exc:
                logger.warning(
                    "MCPProvider: pool session %d/%d failed for '%s': %s",
                    i + 1,
                    pool_size,
                    server_name,
                    exc,
                )
                # Clean up already-created sessions before giving up
                for s in stacks:
                    with suppress(Exception):
                        await s.aclose()
                return False

        self._http_session_pools[server_name] = queue
        self._http_pool_stacks[server_name] = stacks
        self._sessions[server_name] = _POOLED

        # Fetch tool catalog using one session from the pool
        probe = await asyncio.wait_for(queue.get(), timeout=10)
        try:
            await self._fetch_server_tools(server_name, probe, server_cfg)
        finally:
            queue.put_nowait(probe)

        logger.info(
            "MCPProvider [cloud pool]: '%s' — %d sessions, %d tools",
            server_name,
            pool_size,
            len(self._server_tools.get(server_name, [])),
        )
        return True

    # ── Tool metadata ────────────────────────────────────────────────

    async def _fetch_server_tools(
        self,
        server_name: str,
        session: Any,
        server_cfg: dict[str, Any],
    ) -> None:
        """Fetch tool list from a connected session and populate metadata."""
        try:
            tools_response = await session.list_tools()
            tools = getattr(tools_response, "tools", tools_response) or []
            tool_names: list[str] = []
            toolkits: list[str] = server_cfg.get("toolkits", [server_name])

            for tool in tools:
                tool_name = getattr(tool, "name", None) or (
                    tool.get("name") if isinstance(tool, dict) else None
                )
                if not tool_name:
                    continue
                tool_names.append(tool_name)
                slug = f"{server_name}::{tool_name}"
                desc_raw = getattr(tool, "description", "") or (
                    tool.get("description", "") if isinstance(tool, dict) else ""
                )
                desc = str(desc_raw)[:120] if desc_raw else ""
                schema = (
                    getattr(tool, "inputSchema", {})
                    or getattr(tool, "input_schema", {})
                    or (tool.get("inputSchema", {}) if isinstance(tool, dict) else {})
                )
                required = self._extract_required_params(schema if isinstance(schema, dict) else {})

                for tk in toolkits:
                    if tk not in self._tool_metadata:
                        self._tool_metadata[tk] = []
                    self._tool_metadata[tk].append(
                        {
                            "slug": slug,
                            "description": desc,
                            "required": required[:5],
                        }
                    )

            self._server_tools[server_name] = tool_names
        except Exception as exc:
            logger.warning(
                "MCPProvider: failed to list tools for server '%s': %s", server_name, exc
            )
            self._server_tools[server_name] = []

    # ── Lifecycle ────────────────────────────────────────────────────

    async def connect(self) -> bool:
        """
        Iterate enabled servers and establish connections.

        Routing is controlled by AIFACTORY_DEPLOY_MODE:
          local  → stdio subprocess (docker/uvx/npx) — default, no URL needed
          cloud  → HTTP session pool using the URL from url_env_var

        Missing credentials/URL → skip server (non-fatal).
        At least one server must connect for the provider to report success.
        """
        if not _MCP_AVAILABLE:
            self._set_error("mcp package not installed — run: uv add mcp")
            return False

        self.status = IntegrationStatus.CONNECTING
        self._load_catalog()

        if not self._server_configs:
            self._set_error("No server configs found in mcp_catalog.yaml")
            return False

        deploy_mode = os.environ.get("AIFACTORY_DEPLOY_MODE", "local").lower()
        try:
            from aifactory.config.settings import get_settings

            _s = get_settings()
            pool_size = _s.mcp_http_pool_size
        except Exception:
            pool_size = 3

        connected_count = 0

        for server_name, server_cfg in self._server_configs.items():
            if not server_cfg.get("enabled", True):
                logger.debug("MCPProvider: server '%s' is disabled, skipping", server_name)
                continue

            if deploy_mode == "cloud":
                # ── Cloud mode: persistent HTTP service ──────────────────
                url_env_var = server_cfg.get("url_env_var", "")
                url = os.environ.get(url_env_var, "").strip() if url_env_var else ""
                if not url:
                    logger.info(
                        "MCPProvider [cloud]: no URL for '%s' (%s not set), skipping",
                        server_name,
                        url_env_var or "url_env_var not configured",
                    )
                    continue
                # Per-server pool_size override
                server_pool = int(server_cfg.get("pool_size", pool_size))
                ok = await self._connect_http_server_pooled(
                    server_name, server_cfg, url, server_pool
                )
            else:
                # ── Local mode: stdio subprocess ──────────────────────────
                transport = server_cfg.get("transport", "stdio")
                if transport == "stdio":
                    server_env = self._build_server_env(server_cfg)
                    if server_env is None:
                        continue
                    ok = await self._connect_stdio_server(server_name, server_cfg, server_env)
                elif transport in ("http", "streamable_http", "sse"):
                    ok = await self._connect_http_server(server_name, server_cfg)
                else:
                    logger.warning(
                        "MCPProvider: unknown transport '%s' for '%s'", transport, server_name
                    )
                    ok = False

            if ok:
                connected_count += 1

        if connected_count > 0:
            self._set_connected()
            logger.info(
                "MCPProvider [%s]: %d/%d servers connected",
                deploy_mode,
                connected_count,
                len(self._server_configs),
            )
            return True

        logger.info(
            "MCPProvider [%s]: no servers connected (check env vars / URLs in mcp_catalog.yaml)",
            deploy_mode,
        )
        self._set_error("No MCP servers could connect")
        return False

    async def disconnect(self) -> None:
        """Close all sessions — both stdio (exit stack) and HTTP pool stacks."""
        # Close HTTP pool stacks first
        for server_name, stacks in self._http_pool_stacks.items():
            for i, stack in enumerate(stacks):
                try:
                    await stack.aclose()
                except Exception as exc:
                    logger.debug(
                        "MCPProvider: error closing pool session %d for '%s': %s",
                        i,
                        server_name,
                        exc,
                    )

        # Close stdio stack
        try:
            await self._exit_stack.aclose()
        except Exception as exc:
            logger.debug("MCPProvider: error during stdio disconnect: %s", exc)

        # Reset all state
        self._sessions.clear()
        self._http_session_pools.clear()
        self._http_pool_stacks.clear()
        self._tool_metadata.clear()
        self._server_tools.clear()
        self._exit_stack = AsyncExitStack()
        self._set_disconnected()

    async def health_check(self) -> bool:
        """Verify at least one server responds to list_tools."""
        if not self._sessions:
            return False

        for server_name, session_or_sentinel in self._sessions.items():
            if session_or_sentinel == _POOLED:
                pool = self._http_session_pools.get(server_name)
                if not pool:
                    continue
                try:
                    session = await asyncio.wait_for(pool.get(), timeout=5)
                    try:
                        await session.list_tools()
                        return True
                    finally:
                        pool.put_nowait(session)
                except Exception as exc:
                    logger.warning(
                        "MCPProvider: health check failed for pooled '%s': %s", server_name, exc
                    )
            else:
                try:
                    await session_or_sentinel.list_tools()
                    return True
                except Exception as exc:
                    logger.warning(
                        "MCPProvider: health check failed for '%s': %s", server_name, exc
                    )
        return False

    # ── Actions ──────────────────────────────────────────────────────

    async def execute_action(self, action: str, params: dict[str, Any]) -> ActionResult:
        """
        Execute a tool via MCP.

        Args:
            action: Tool slug in "server::tool_name" format
            params: Tool arguments dict

        For pooled servers (cloud mode) a session is acquired from the pool,
        used, and returned — safe for concurrent pipeline runs.
        """
        if "::" not in action:
            return ActionResult(
                success=False,
                action=action,
                integration=self.name,
                error=(
                    f"Invalid MCP action format '{action}'. "
                    "Expected 'server::tool_name' (e.g. 'github::create_issue')."
                ),
            )

        server_name, tool_name = action.split("::", 1)

        session_or_sentinel = self._sessions.get(server_name)
        if session_or_sentinel is None:
            return ActionResult(
                success=False,
                action=action,
                integration=self.name,
                error=f"MCP server '{server_name}' is not connected",
            )

        # ── Pool acquire path (cloud mode) ────────────────────────────
        if session_or_sentinel == _POOLED:
            pool = self._http_session_pools.get(server_name)
            if pool is None:
                return ActionResult(
                    success=False,
                    action=action,
                    integration=self.name,
                    error=f"Session pool for '{server_name}' is missing",
                )
            try:
                session = await asyncio.wait_for(pool.get(), timeout=30)
            except TimeoutError:
                return ActionResult(
                    success=False,
                    action=action,
                    integration=self.name,
                    error=(
                        f"Timed out waiting for a free session from '{server_name}' pool. "
                        "Consider increasing AIFACTORY_MCP_HTTP_POOL_SIZE."
                    ),
                )
            try:
                return await self._call_tool(action, server_name, tool_name, session, params)
            finally:
                pool.put_nowait(session)

        # ── Direct session path (local / stdio mode) ──────────────────
        return await self._call_tool(action, server_name, tool_name, session_or_sentinel, params)

    async def _call_tool(
        self,
        action: str,
        server_name: str,
        tool_name: str,
        session: Any,
        params: dict[str, Any],
    ) -> ActionResult:
        """Inner tool-call logic shared by pooled and direct session paths."""

        def _is_mcp_rate_limit(exc: Exception) -> bool:
            msg = str(exc).lower()
            return "429" in msg or "rate limit" in msg or "too many requests" in msg

        try:
            try:
                _retry_attempts = (
                    __import__("aifactory.config.settings", fromlist=["get_settings"])
                    .get_settings()
                    .tool_retry_max_attempts
                )
            except Exception:
                _retry_attempts = 3

            result = await retry_with_backoff(
                lambda: session.call_tool(tool_name, params),
                max_retries=_retry_attempts,
                base_delay=1.0,
                jitter=True,
                is_rate_limit_exc=_is_mcp_rate_limit,
            )

            content = getattr(result, "content", None)
            is_error = getattr(result, "isError", False) or getattr(result, "is_error", False)

            if is_error:
                error_text = ""
                if content:
                    if isinstance(content, list):
                        texts = [
                            str(
                                getattr(c, "text", None)
                                or (c.get("text", "") if isinstance(c, dict) else str(c))
                            )
                            for c in content
                        ]
                        error_text = " ".join(t for t in texts if t)
                    else:
                        error_text = str(content)
                return ActionResult(
                    success=False,
                    action=action,
                    integration=self.name,
                    error=error_text or "MCP tool returned an error",
                )

            result_data: dict[str, Any] = {}
            result_text = ""
            if content:
                if isinstance(content, list):
                    texts = []
                    for c in content:
                        t = getattr(c, "text", None) or (
                            c.get("text", "") if isinstance(c, dict) else None
                        )
                        if t:
                            texts.append(str(t))
                    result_text = "\n".join(texts)
                else:
                    result_text = str(content)
                result_data = {"output": result_text} if result_text else {}

            url = ""
            if result_text:
                try:
                    import json

                    parsed = json.loads(result_text)
                    if isinstance(parsed, dict):
                        url = parsed.get("html_url", "") or parsed.get("url", "")
                        result_data = parsed
                except (json.JSONDecodeError, ValueError):
                    pass

            return ActionResult(
                success=True,
                action=action,
                integration=self.name,
                data=result_data,
                url=url,
            )

        except Exception as exc:
            logger.error(
                "MCPProvider: execute_action failed: %s/%s: %s", server_name, tool_name, exc
            )
            return ActionResult(
                success=False,
                action=action,
                integration=self.name,
                error=str(exc),
            )

    def get_available_actions(self) -> list[str]:
        """Return all 'server::tool_name' slugs from connected servers."""
        slugs: list[str] = []
        for server_name, tool_names in self._server_tools.items():
            for tool_name in tool_names:
                slugs.append(f"{server_name}::{tool_name}")
        return slugs

    def get_available_toolkits(self) -> list[str]:
        """Return toolkit names from connected servers."""
        toolkits: list[str] = []
        for server_name, server_cfg in self._server_configs.items():
            if server_name in self._sessions:
                toolkits.extend(server_cfg.get("toolkits", [server_name]))
        return sorted(set(toolkits))

    # ── Tool catalog (for prompt injection) ──────────────────────────

    def get_tools_summary_for_prompt(self) -> str:
        """Full catalog across all toolkits."""
        if not self._tool_metadata:
            return ""
        lines: list[str] = []
        for tk in sorted(self._tool_metadata):
            meta = self._tool_metadata[tk]
            if meta:
                lines.append(f"[{tk}]")
                for m in meta:
                    slug = m["slug"]
                    desc = m.get("description", "")
                    req = m.get("required", [])
                    req_str = f" (requires: {', '.join(req)})" if req else ""
                    if desc:
                        lines.append(f"  {slug}: {desc}{req_str}")
                    else:
                        lines.append(f"  {slug}{req_str}")
        return "\n".join(lines)

    def get_toolkit_tools_for_prompt(
        self, toolkit: str, task_keywords: list[str] | None = None
    ) -> str:
        """
        Return tool catalog for a single toolkit in 'server::tool_name: description' format.

        Optionally filtered by task_keywords. Capped at 15 tools.
        """
        meta = self._tool_metadata.get(toolkit, [])
        if not meta:
            return ""

        if task_keywords:
            kw_lower = [kw.lower() for kw in task_keywords if len(kw) > 2]
            if kw_lower:
                filtered = [
                    m
                    for m in meta
                    if any(
                        kw in m["slug"].lower() or kw in m.get("description", "").lower()
                        for kw in kw_lower
                    )
                ]
                if filtered:
                    meta = filtered

        lines: list[str] = []
        for m in meta[:15]:
            slug = m["slug"]
            desc = m.get("description", "")[:60]
            req = m.get("required", [])
            req_str = f" (requires: {', '.join(req)})" if req else ""
            lines.append(f"  {slug}: {desc}{req_str}" if desc else f"  {slug}{req_str}")
        if len(meta) > 15:
            lines.append(f"  ... and {len(meta) - 15} more tools")
        return "\n".join(lines)
