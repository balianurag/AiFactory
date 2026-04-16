"""
AI Factory MCP Sandbox Server.

Exposes code execution tools via the Model Context Protocol (MCP).
Manages a warm Docker container pool for low-latency, secure execution.

Tools:
    execute_python      — run Python code in an isolated Docker container
    execute_bash        — run bash script in an isolated Docker container
    execute_javascript  — run Node.js script in an isolated Docker container
    pool_status         — inspect pool health metrics

Transport modes:
    stdio               — subprocess mode (dev / local)
    streamable-http     — HTTP server mode (prod, requires auth token)

Security:
    - Bearer token authentication (HTTP mode)
    - Per-agent rate limiting (max_executions_per_worker enforced by caller)
    - All execution inside Docker containers with security hardening
    - Audit log on every tool call (agent_id, language, success, elapsed_ms)

Usage (prod):
    python -m aifactory.mcp.sandbox_server --transport streamable-http --port 8100

Usage (dev):
    python -m aifactory.mcp.sandbox_server --transport stdio
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Lazy FastMCP import (not required for non-sandbox deployments)
# =============================================================================


def _get_fastmcp():
    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore[import]

        return FastMCP
    except ImportError:
        pass
    try:
        import fastmcp  # type: ignore[import]

        return fastmcp.FastMCP
    except ImportError as exc:
        raise ImportError(
            "fastmcp is not installed. Install it with: pip install 'mcp[server]' or fastmcp"
        ) from exc


# =============================================================================
# Audit logger
# =============================================================================


def _audit(
    agent_id: str,
    language: str,
    success: bool,
    exit_code: int,
    elapsed_ms: float,
    container_id: str,
) -> None:
    """Write a structured audit log line for every execution."""
    record = {
        "event": "sandbox_exec",
        "agent_id": agent_id,
        "language": language,
        "success": success,
        "exit_code": exit_code,
        "elapsed_ms": round(elapsed_ms, 1),
        "container_id": container_id[:12] if container_id else "",
        "ts": time.time(),
    }
    logger.info("AUDIT %s", json.dumps(record))


# =============================================================================
# Server builder
# =============================================================================


def build_server():
    """
    Build and return a FastMCP server instance with all sandbox tools.

    Called once at module level so the server object is importable by
    other modules without launching HTTP listeners.
    """
    fast_mcp_cls = _get_fastmcp()
    mcp = fast_mcp_cls("aifactory-sandbox")

    # ── execute_python ────────────────────────────────────────────────────────
    @mcp.tool()
    async def execute_python(
        code: str,
        agent_id: str = "unknown",
        timeout: int = 30,
        extra_files: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Execute Python code in an isolated Docker sandbox.

        Args:
            code: Python source code to execute.
            agent_id: Identifier of the calling agent (for audit logging).
            timeout: Max execution time in seconds (default 30, max 120).
            extra_files: Additional files to create in /workspace, e.g.
                         {"data.csv": "a,b\\n1,2"}.

        Returns:
            dict with: success, exit_code, stdout, stderr, elapsed_ms, container_id
        """
        return await _run_in_pool(
            code=code,
            language="python",
            agent_id=agent_id,
            timeout=min(timeout, 120),
            extra_files=extra_files,
        )

    # ── execute_bash ─────────────────────────────────────────────────────────
    @mcp.tool()
    async def execute_bash(
        code: str,
        agent_id: str = "unknown",
        timeout: int = 30,
    ) -> dict[str, Any]:
        """
        Execute a bash script in an isolated Docker sandbox.

        Args:
            code: Bash script content.
            agent_id: Identifier of the calling agent.
            timeout: Max execution time in seconds (default 30, max 120).

        Returns:
            dict with: success, exit_code, stdout, stderr, elapsed_ms, container_id
        """
        return await _run_in_pool(
            code=code,
            language="bash",
            agent_id=agent_id,
            timeout=min(timeout, 120),
        )

    # ── execute_javascript ────────────────────────────────────────────────────
    @mcp.tool()
    async def execute_javascript(
        code: str,
        agent_id: str = "unknown",
        timeout: int = 30,
    ) -> dict[str, Any]:
        """
        Execute a Node.js script in an isolated Docker sandbox.

        Args:
            code: JavaScript/Node.js source code.
            agent_id: Identifier of the calling agent.
            timeout: Max execution time in seconds (default 30, max 120).

        Returns:
            dict with: success, exit_code, stdout, stderr, elapsed_ms, container_id
        """
        return await _run_in_pool(
            code=code,
            language="javascript",
            agent_id=agent_id,
            timeout=min(timeout, 120),
        )

    # ── pool_status ───────────────────────────────────────────────────────────
    @mcp.tool()
    async def pool_status() -> dict[str, Any]:
        """
        Return current Docker container pool metrics.

        Returns:
            dict with: ready, in_use, total, min_idle, max_size, image, gvisor
        """
        from aifactory.core.sandbox_pool import get_pool

        pool = await get_pool()
        return await pool.pool_status()

    return mcp


# =============================================================================
# Shared execution helper
# =============================================================================


async def _run_in_pool(
    *,
    code: str,
    language: str,
    agent_id: str,
    timeout: int,
    extra_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Acquire a container from the pool, run code, return structured result."""
    from aifactory.core.sandbox_pool import get_pool

    t0 = time.monotonic()

    try:
        pool = await get_pool()
        async with pool.acquire() as container:
            result = await pool.execute(
                container,
                code,
                language=language,
                timeout=timeout,
                extra_files=extra_files,
            )
    except TimeoutError as exc:
        elapsed = (time.monotonic() - t0) * 1000
        _audit(agent_id, language, False, -1, elapsed, "")
        return {
            "success": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": str(exc),
            "elapsed_ms": round(elapsed, 1),
            "container_id": "",
            "error": "pool_timeout",
        }
    except Exception as exc:
        elapsed = (time.monotonic() - t0) * 1000
        logger.exception("Unexpected error in _run_in_pool: %s", exc)
        _audit(agent_id, language, False, -1, elapsed, "")
        return {
            "success": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": str(exc),
            "elapsed_ms": round(elapsed, 1),
            "container_id": "",
            "error": "internal_error",
        }

    elapsed = (time.monotonic() - t0) * 1000
    _audit(agent_id, language, result.success, result.exit_code, elapsed, result.container_id)

    return {
        "success": result.success,
        "exit_code": result.exit_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "elapsed_ms": round(elapsed, 1),
        "container_id": result.container_id[:12] if result.container_id else "",
        "timed_out": result.timed_out,
    }


# =============================================================================
# Bearer auth middleware (HTTP mode)
# =============================================================================


def _make_auth_middleware(token: str):
    """Return ASGI middleware that validates Bearer token on all requests."""
    from starlette.middleware.base import BaseHTTPMiddleware  # type: ignore[import]
    from starlette.requests import Request  # type: ignore[import]
    from starlette.responses import JSONResponse  # type: ignore[import]

    class BearerAuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            # Health endpoint is always unauthenticated
            if request.url.path in ("/health", "/"):
                return await call_next(request)

            auth_header = request.headers.get("authorization", "")
            if not auth_header.startswith("Bearer ") or auth_header[7:] != token:
                return JSONResponse(
                    {"error": "Unauthorized"},
                    status_code=401,
                )
            return await call_next(request)

    return BearerAuthMiddleware


# =============================================================================
# Health endpoint
# =============================================================================


def _add_health_route(app) -> None:
    """Attach /health GET to the ASGI app (Starlette route)."""
    from starlette.requests import Request  # type: ignore[import]
    from starlette.responses import JSONResponse  # type: ignore[import]
    from starlette.routing import Route  # type: ignore[import]

    async def health(request: Request) -> JSONResponse:
        from aifactory.core.sandbox_pool import _pool

        status = "ok"
        pool_info: dict = {}
        if _pool is not None:
            pool_info = await _pool.pool_status()
            if pool_info.get("ready", 0) + pool_info.get("in_use", 0) == 0:
                status = "degraded"
        return JSONResponse({"status": status, "pool": pool_info})

    # Starlette apps expose .routes; add only if not already present
    if hasattr(app, "routes"):
        app.routes.insert(0, Route("/health", health))


# =============================================================================
# Entry point
# =============================================================================


async def _async_main(transport: str, port: int, host: str) -> None:
    """Start pool then launch server."""
    from aifactory.core.sandbox_pool import get_pool

    # Pre-warm pool before accepting connections
    pool = await get_pool()
    logger.info("Pool ready: %s", await pool.pool_status())

    mcp = build_server()

    if transport == "stdio":
        await mcp.run_async(transport="stdio")
    else:
        # HTTP mode
        auth_token = os.getenv("MCP_AUTH_TOKEN", "").strip()
        asgi_app = mcp.streamable_http_app()

        if auth_token:
            middleware_cls = _make_auth_middleware(auth_token)
            asgi_app.add_middleware(middleware_cls)
            logger.info("Bearer auth enabled")
        else:
            logger.warning(
                "MCP_AUTH_TOKEN not set — server is UNAUTHENTICATED. "
                "Set MCP_AUTH_TOKEN or bind to localhost only."
            )

        _add_health_route(asgi_app)

        import uvicorn  # type: ignore[import]

        config = uvicorn.Config(
            asgi_app,
            host=host,
            port=port,
            log_level="info",
            access_log=True,
        )
        server = uvicorn.Server(config)
        await server.serve()


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Factory MCP Sandbox Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="streamable-http",
        help="MCP transport (default: streamable-http)",
    )
    parser.add_argument("--port", type=int, default=8100, help="HTTP port (default: 8100)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    asyncio.run(_async_main(transport=args.transport, port=args.port, host=args.host))


if __name__ == "__main__":
    main()
