"""
MCP Sandbox Client — connects to the sandbox MCP server.

Two modes auto-detected from settings:
  1. HTTP mode  (prod)  — when AIFACTORY_MCP_SANDBOX_URL is set
                         → connects to running sandbox-mcp container
  2. stdio mode (dev)   — spawns sandbox_server.py as a subprocess
                         → zero config for local development

Usage:
    client = await get_mcp_client()
    result = await client.execute_python(code="print('hello')", agent_id="worker-1")
    await client.close()

Or as a context manager:
    async with mcp_client() as client:
        result = await client.execute_python(code=code, agent_id=agent_id)
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Result wrapper
# =============================================================================


class McpExecResult:
    """Thin wrapper around the JSON dict returned by the MCP server."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    @property
    def success(self) -> bool:
        return bool(self._data.get("success", False))

    @property
    def exit_code(self) -> int:
        return int(self._data.get("exit_code", -1))

    @property
    def stdout(self) -> str:
        return str(self._data.get("stdout", ""))

    @property
    def stderr(self) -> str:
        return str(self._data.get("stderr", ""))

    @property
    def timed_out(self) -> bool:
        return bool(self._data.get("timed_out", False))

    @property
    def elapsed_ms(self) -> float:
        return float(self._data.get("elapsed_ms", 0.0))

    @property
    def container_id(self) -> str:
        return str(self._data.get("container_id", ""))

    @property
    def output(self) -> str:
        parts = []
        if self.stdout.strip():
            parts.append(self.stdout.strip())
        if self.stderr.strip():
            parts.append(f"STDERR:\n{self.stderr.strip()}")
        return "\n".join(parts) or "(no output)"

    def __repr__(self) -> str:
        return f"McpExecResult(success={self.success}, exit_code={self.exit_code})"


# =============================================================================
# HTTP client (prod)
# =============================================================================


class HttpMcpClient:
    """
    MCP client that calls the sandbox-mcp HTTP server via the FastMCP
    Streamable HTTP transport (JSON-RPC 2.0 over SSE).

    Protocol:
      1. POST /mcp with ``initialize`` → get ``mcp-session-id`` header
      2. Subsequent requests include the session ID header
      3. Responses are SSE-wrapped: ``event: message\\ndata: {...}``
    """

    def __init__(self, base_url: str, auth_token: str = "") -> None:
        self._base_url = base_url.rstrip("/")
        self._auth_token = auth_token
        self._client: object = None  # httpx.AsyncClient, imported lazily
        self._request_id = 0
        self._session_id: str | None = None

    async def _ensure_client(self) -> None:
        if self._client is not None:
            return

        import httpx  # type: ignore[import]

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=180.0,
        )

        # MCP Streamable HTTP transport: must send ``initialize`` first
        # and capture the session ID for all subsequent requests.
        self._request_id += 1
        init_payload = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "aifactory-sandbox-client", "version": "1.0"},
            },
        }
        resp = await self._client.post("/mcp", json=init_payload)
        resp.raise_for_status()
        self._session_id = resp.headers.get("mcp-session-id")
        if self._session_id:
            self._client.headers["mcp-session-id"] = self._session_id

    @staticmethod
    def _parse_sse(raw: str) -> dict:
        """Extract the last JSON payload from an SSE response body."""
        data_lines = [
            line[len("data:") :].strip()
            for line in raw.splitlines()
            if line.startswith("data:") and line[len("data:") :].strip()
        ]
        return json.loads(data_lines[-1]) if data_lines else {}

    async def _call_tool(self, tool_name: str, arguments: dict) -> dict:
        await self._ensure_client()
        self._request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }
        response = await self._client.post("/mcp", json=payload)  # type: ignore[union-attr]
        response.raise_for_status()

        raw = response.text.strip()
        data = self._parse_sse(raw) if ("data:" in raw) else response.json()

        if "error" in data:
            raise RuntimeError(f"MCP error: {data['error']}")
        # FastMCP wraps the result in content[0].text as a JSON string
        content = data.get("result", {}).get("content", [])
        if content and isinstance(content[0], dict):
            text = content[0].get("text", "{}")
            return json.loads(text)
        return data.get("result", {})

    async def execute_python(
        self,
        *,
        code: str,
        agent_id: str = "unknown",
        timeout: int = 30,
        extra_files: dict[str, str] | None = None,
    ) -> McpExecResult:
        args: dict[str, Any] = {"code": code, "agent_id": agent_id, "timeout": timeout}
        if extra_files:
            args["extra_files"] = extra_files
        data = await self._call_tool("execute_python", args)
        return McpExecResult(data)

    async def execute_bash(
        self, *, code: str, agent_id: str = "unknown", timeout: int = 30
    ) -> McpExecResult:
        data = await self._call_tool(
            "execute_bash", {"code": code, "agent_id": agent_id, "timeout": timeout}
        )
        return McpExecResult(data)

    async def execute_javascript(
        self, *, code: str, agent_id: str = "unknown", timeout: int = 30
    ) -> McpExecResult:
        data = await self._call_tool(
            "execute_javascript", {"code": code, "agent_id": agent_id, "timeout": timeout}
        )
        return McpExecResult(data)

    async def pool_status(self) -> dict[str, Any]:
        return await self._call_tool("pool_status", {})

    async def health(self) -> dict[str, Any]:
        await self._ensure_client()
        resp = await self._client.get("/health")  # type: ignore[union-attr]
        resp.raise_for_status()
        return resp.json()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


# =============================================================================
# Stdio client (dev — in-process server)
# =============================================================================


class StdioMcpClient:
    """
    MCP client that directly calls the in-process server functions.

    Used in dev/test mode when no HTTP server is running — avoids subprocess
    overhead while keeping the same interface as HttpMcpClient.
    """

    async def execute_python(
        self,
        *,
        code: str,
        agent_id: str = "unknown",
        timeout: int = 30,
        extra_files: dict[str, str] | None = None,
    ) -> McpExecResult:
        from aifactory.mcp.sandbox_server import _run_in_pool

        data = await _run_in_pool(
            code=code,
            language="python",
            agent_id=agent_id,
            timeout=timeout,
            extra_files=extra_files,
        )
        return McpExecResult(data)

    async def execute_bash(
        self, *, code: str, agent_id: str = "unknown", timeout: int = 30
    ) -> McpExecResult:
        from aifactory.mcp.sandbox_server import _run_in_pool

        data = await _run_in_pool(
            code=code,
            language="bash",
            agent_id=agent_id,
            timeout=timeout,
        )
        return McpExecResult(data)

    async def execute_javascript(
        self, *, code: str, agent_id: str = "unknown", timeout: int = 30
    ) -> McpExecResult:
        from aifactory.mcp.sandbox_server import _run_in_pool

        data = await _run_in_pool(
            code=code,
            language="javascript",
            agent_id=agent_id,
            timeout=timeout,
        )
        return McpExecResult(data)

    async def pool_status(self) -> dict[str, Any]:
        from aifactory.core.sandbox_pool import get_pool

        pool = await get_pool()
        return await pool.pool_status()

    async def health(self) -> dict[str, Any]:
        return {"status": "ok", "mode": "stdio"}

    async def close(self) -> None:
        pass  # nothing to close


# =============================================================================
# Factory
# =============================================================================


def _get_mcp_url() -> str:
    url = os.getenv("AIFACTORY_MCP_SANDBOX_URL", "").strip()
    if url:
        return url
    try:
        from aifactory.config.settings import get_settings

        return (get_settings().sandbox_mcp_url or "").strip()
    except Exception:
        return ""


def _get_mcp_token() -> str:
    token = os.getenv("MCP_AUTH_TOKEN", "").strip()
    if token:
        return token
    try:
        from aifactory.config.settings import get_settings

        return (get_settings().sandbox_mcp_auth_token or "").strip()
    except Exception:
        return ""


async def get_mcp_client() -> HttpMcpClient | StdioMcpClient:
    """
    Return an MCP client based on settings.

    - If AIFACTORY_MCP_SANDBOX_URL is set → HttpMcpClient (connects to server)
    - Otherwise → StdioMcpClient (in-process, dev mode)
    """
    url = _get_mcp_url()
    if url:
        logger.debug("MCP client: HTTP mode → %s", url)
        return HttpMcpClient(base_url=url, auth_token=_get_mcp_token())
    logger.debug("MCP client: stdio/in-process mode")
    return StdioMcpClient()


@contextlib.asynccontextmanager
async def mcp_client():
    """Async context manager that yields an MCP client and closes it on exit."""
    client = await get_mcp_client()
    try:
        yield client
    finally:
        await client.close()
