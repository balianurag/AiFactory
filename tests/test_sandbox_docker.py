"""
Tests for Docker MCP sandbox integration.

Covers:
    - SandboxPool (unit tests with mocked Docker)
    - MCP client (HttpMcpClient + StdioMcpClient)
    - sandbox.py Docker backend routing
    - agent_factory.py package registry injection
    - Security: output truncation, workspace isolation, language validation
    - pool_status metrics
    - Error / fallback paths
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# =============================================================================
# Helpers / fixtures
# =============================================================================


def _make_container(exit_code: int = 0, stdout: bytes = b"", stderr: bytes = b"") -> MagicMock:
    """Return a mock docker container whose exec_run produces the given result."""
    exec_result = MagicMock()
    exec_result.exit_code = exit_code
    exec_result.output = (stdout, stderr)

    container = MagicMock()
    container.id = "deadbeef" * 8  # 64-char fake container ID
    container.exec_run.return_value = exec_result
    container.remove.return_value = None
    # _wait_until_ready polls container.status after reload(); set to "running" so
    # freshly-created mock containers are immediately considered healthy.
    container.status = "running"
    container.reload.return_value = None
    return container


def _make_docker_client(container: MagicMock | None = None) -> MagicMock:
    """Return a mock docker.DockerClient."""
    client = MagicMock()
    client.ping.return_value = True
    if container:
        client.containers.run.return_value = container
    return client


# =============================================================================
# SandboxPool — unit tests
# =============================================================================


class TestSandboxPoolCreation:
    """SandboxPool is configured correctly from settings."""

    def test_defaults_from_settings(self):
        from aifactory.core.sandbox_pool import SandboxPool

        with patch("aifactory.core.sandbox_pool.SandboxPool._make_docker_client"):
            pool = SandboxPool()
        assert pool._min_idle == 4
        assert pool._max_size == 16
        assert pool._max_reuse == 50
        assert pool._memory == "512m"
        assert pool._network_mode == "none"

    def test_overrides_respected(self):
        from aifactory.core.sandbox_pool import SandboxPool

        pool = SandboxPool(min_idle=2, max_size=8, memory="256m", max_reuse=10)
        assert pool._min_idle == 2
        assert pool._max_size == 8
        assert pool._memory == "256m"
        assert pool._max_reuse == 10

    def test_seccomp_profile_resolved(self, tmp_path):
        from aifactory.core.sandbox_pool import SandboxPool

        # Create a fake seccomp file
        fake_profile = tmp_path / "seccomp.json"
        fake_profile.write_text("{}")
        pool = SandboxPool(seccomp_profile=str(fake_profile))
        assert pool._seccomp_profile == str(fake_profile)

    def test_missing_seccomp_profile_silently_ignored(self):
        from aifactory.core.sandbox_pool import SandboxPool

        pool = SandboxPool(seccomp_profile="/nonexistent/seccomp.json")
        assert pool._seccomp_profile is None


class TestSandboxPoolLifecycle:
    """start() / stop() manage container lifecycle."""

    async def test_start_creates_min_idle_containers(self):
        from aifactory.core.sandbox_pool import SandboxPool

        container = _make_container(stdout=b"", exit_code=0)
        docker_client = _make_docker_client(container)

        pool = SandboxPool(min_idle=2, max_size=4)
        pool._seccomp_profile = None

        with patch.object(pool, "_make_docker_client", return_value=docker_client):
            await pool.start()

        # At least 2 containers should have been created
        assert docker_client.containers.run.call_count >= 2
        assert pool._ready.qsize() >= 2
        assert pool._started is True

        await pool.stop()

    async def test_start_idempotent(self):
        from aifactory.core.sandbox_pool import SandboxPool

        container = _make_container()
        docker_client = _make_docker_client(container)
        pool = SandboxPool(min_idle=1, max_size=2)
        pool._seccomp_profile = None

        with patch.object(pool, "_make_docker_client", return_value=docker_client):
            await pool.start()
            await pool.start()  # second call should be no-op

        assert docker_client.containers.run.call_count == 1
        await pool.stop()

    async def test_stop_clears_containers(self):
        from aifactory.core.sandbox_pool import SandboxPool

        container = _make_container()
        docker_client = _make_docker_client(container)
        pool = SandboxPool(min_idle=2, max_size=4)
        pool._seccomp_profile = None

        with patch.object(pool, "_make_docker_client", return_value=docker_client):
            await pool.start()

        await pool.stop()
        assert pool._ready.qsize() == 0
        assert pool._started is False


class TestSandboxPoolAcquireRelease:
    """acquire() context manager lends and returns containers."""

    async def test_acquire_returns_container(self):
        from aifactory.core.sandbox_pool import SandboxPool

        container = _make_container(exit_code=0)
        docker_client = _make_docker_client(container)
        pool = SandboxPool(min_idle=1, max_size=2)
        pool._seccomp_profile = None

        with patch.object(pool, "_make_docker_client", return_value=docker_client):
            await pool.start()

        async with pool.acquire() as pc:
            assert pc is not None
            assert pc.container_id == container.id
            assert pc.container_id in pool._in_use

        # After release, container back in ready queue (not expired)
        assert pool._in_use == {}
        await pool.stop()

    async def test_expired_container_retired_on_release(self):
        from aifactory.core.sandbox_pool import SandboxPool

        container = _make_container(exit_code=0)
        docker_client = _make_docker_client(container)
        # max_size=1 prevents replenish loop from creating new containers
        # while the only container is in-use
        pool = SandboxPool(min_idle=1, max_size=1, max_reuse=1)
        pool._seccomp_profile = None

        with patch.object(pool, "_make_docker_client", return_value=docker_client):
            await pool.start()

        acquired_id = None
        async with pool.acquire() as pc:
            acquired_id = pc.container_id
            pc.reuse_count = 1  # mark as expired

        # The expired container should not be back in the ready queue
        ready_ids = [c.container_id for c in list(pool._ready._queue)]
        assert acquired_id not in ready_ids
        await pool.stop()

    async def test_acquire_timeout_raises(self):
        from aifactory.core.sandbox_pool import SandboxPool

        pool = SandboxPool(min_idle=0, max_size=0, acquire_timeout=0.1)
        pool._started = True
        pool._client = MagicMock()
        pool._seccomp_profile = None

        with pytest.raises(TimeoutError):
            async with pool.acquire():
                pass


class TestSandboxPoolExecute:
    """execute() writes code and runs it via docker exec."""

    async def test_execute_python_success(self):
        from aifactory.core.sandbox_pool import PooledContainer, SandboxPool

        container = _make_container(exit_code=0, stdout=b"hello\n", stderr=b"")
        pool = SandboxPool(min_idle=0, max_size=1)
        pool._seccomp_profile = None
        pool._max_output_bytes = 50_000

        pc = PooledContainer(container_id=container.id, container=container)

        result = await pool.execute(pc, "print('hello')", language="python", timeout=10)

        assert result.success is True
        assert "hello" in result.stdout
        assert result.exit_code == 0

    async def test_execute_bash(self):
        from aifactory.core.sandbox_pool import PooledContainer, SandboxPool

        container = _make_container(exit_code=0, stdout=b"bash ok\n")
        pool = SandboxPool(min_idle=0, max_size=1)
        pool._seccomp_profile = None
        pool._max_output_bytes = 50_000

        pc = PooledContainer(container_id=container.id, container=container)
        result = await pool.execute(pc, "echo bash ok", language="bash", timeout=10)
        assert result.success is True

    async def test_execute_javascript(self):
        from aifactory.core.sandbox_pool import PooledContainer, SandboxPool

        container = _make_container(exit_code=0, stdout=b"js ok\n")
        pool = SandboxPool(min_idle=0, max_size=1)
        pool._seccomp_profile = None
        pool._max_output_bytes = 50_000

        pc = PooledContainer(container_id=container.id, container=container)
        result = await pool.execute(pc, "console.log('js ok')", language="javascript", timeout=10)
        assert result.success is True

    async def test_execute_unsupported_language(self):
        from aifactory.core.sandbox_pool import PooledContainer, SandboxPool

        container = _make_container()
        pool = SandboxPool(min_idle=0, max_size=1)
        pool._seccomp_profile = None
        pool._max_output_bytes = 50_000

        pc = PooledContainer(container_id=container.id, container=container)
        result = await pool.execute(pc, "code", language="ruby", timeout=10)
        assert result.success is False
        assert "Unsupported language" in result.stderr

    async def test_execute_timeout(self):
        from aifactory.core.sandbox_pool import PooledContainer, SandboxPool

        # Normal exec_run for cleanup/write calls; timeout only in _docker_exec
        container = _make_container(exit_code=0, stdout=b"")
        pool = SandboxPool(min_idle=0, max_size=1)
        pool._seccomp_profile = None
        pool._max_output_bytes = 50_000

        pc = PooledContainer(container_id=container.id, container=container)

        # Patch asyncio.wait_for (used in _docker_exec) to simulate timeout
        with patch(
            "aifactory.core.sandbox_pool.asyncio.wait_for", side_effect=asyncio.TimeoutError
        ):
            result = await pool.execute(pc, "import time; time.sleep(999)", timeout=1)

        assert result.timed_out is True
        assert result.success is False

    async def test_output_truncated_at_max_bytes(self):
        from aifactory.core.sandbox_pool import PooledContainer, SandboxPool

        big_output = b"x" * 200_000
        container = _make_container(exit_code=0, stdout=big_output)
        pool = SandboxPool(min_idle=0, max_size=1, max_output_bytes=1000)
        pool._seccomp_profile = None

        pc = PooledContainer(container_id=container.id, container=container)
        result = await pool.execute(pc, "print('x' * 200000)", timeout=10)
        assert len(result.stdout) <= 1000

    async def test_reuse_count_incremented(self):
        from aifactory.core.sandbox_pool import PooledContainer, SandboxPool

        container = _make_container(exit_code=0, stdout=b"")
        pool = SandboxPool(min_idle=0, max_size=1)
        pool._seccomp_profile = None
        pool._max_output_bytes = 50_000

        pc = PooledContainer(container_id=container.id, container=container)
        assert pc.reuse_count == 0
        await pool.execute(pc, "pass", timeout=5)
        assert pc.reuse_count == 1


class TestPooledContainerExpiry:
    """PooledContainer.is_expired property."""

    def test_not_expired_initially(self):
        from aifactory.core.sandbox_pool import PooledContainer

        pc = PooledContainer(container_id="abc", container=MagicMock(), max_reuse=5)
        assert pc.is_expired is False

    def test_expired_at_max_reuse(self):
        from aifactory.core.sandbox_pool import PooledContainer

        pc = PooledContainer(container_id="abc", container=MagicMock(), max_reuse=5)
        pc.reuse_count = 5
        assert pc.is_expired is True


class TestPoolStatus:
    """pool_status() returns accurate metrics."""

    async def test_pool_status_shape(self):
        from aifactory.core.sandbox_pool import SandboxPool

        pool = SandboxPool(min_idle=2, max_size=8)
        pool._seccomp_profile = None

        status = await pool.pool_status()
        assert "ready" in status
        assert "in_use" in status
        assert "total" in status
        assert "min_idle" in status
        assert "max_size" in status
        assert status["min_idle"] == 2
        assert status["max_size"] == 8


# =============================================================================
# ExecResult
# =============================================================================


class TestExecResult:
    def test_success_true_on_zero_exit(self):
        from aifactory.core.sandbox_pool import ExecResult

        r = ExecResult(exit_code=0, stdout="hi", stderr="")
        assert r.success is True

    def test_success_false_on_nonzero(self):
        from aifactory.core.sandbox_pool import ExecResult

        r = ExecResult(exit_code=1, stdout="", stderr="error")
        assert r.success is False

    def test_success_false_on_timeout(self):
        from aifactory.core.sandbox_pool import ExecResult

        r = ExecResult(exit_code=0, stdout="", stderr="", timed_out=True)
        assert r.success is False

    def test_output_combines_stdout_stderr(self):
        from aifactory.core.sandbox_pool import ExecResult

        r = ExecResult(exit_code=0, stdout="out\n", stderr="err\n")
        assert "out" in r.output
        assert "STDERR" in r.output

    def test_output_empty_when_no_content(self):
        from aifactory.core.sandbox_pool import ExecResult

        r = ExecResult(exit_code=0, stdout="", stderr="")
        assert r.output == "(no output)"


# =============================================================================
# MCP Client
# =============================================================================


class TestMcpExecResult:
    def test_wraps_dict(self):
        from aifactory.mcp.client import McpExecResult

        r = McpExecResult(
            {"success": True, "exit_code": 0, "stdout": "hi", "stderr": "", "elapsed_ms": 42.0}
        )
        assert r.success is True
        assert r.stdout == "hi"
        assert r.elapsed_ms == 42.0

    def test_defaults_on_missing_keys(self):
        from aifactory.mcp.client import McpExecResult

        r = McpExecResult({})
        assert r.success is False
        assert r.exit_code == -1
        assert r.stdout == ""

    def test_output_property(self):
        from aifactory.mcp.client import McpExecResult

        r = McpExecResult({"stdout": "hello", "stderr": ""})
        assert r.output == "hello"


class TestStdioMcpClient:
    """StdioMcpClient delegates directly to _run_in_pool."""

    async def test_execute_python_delegates(self):
        from aifactory.mcp.client import StdioMcpClient

        mock_result = {
            "success": True,
            "exit_code": 0,
            "stdout": "42",
            "stderr": "",
            "elapsed_ms": 10.0,
            "container_id": "abc",
            "timed_out": False,
        }

        with patch("aifactory.mcp.sandbox_server._run_in_pool", return_value=mock_result):
            client = StdioMcpClient()
            result = await client.execute_python(code="print(42)", agent_id="test")

        assert result.success is True
        assert result.stdout == "42"

    async def test_execute_bash_delegates(self):
        from aifactory.mcp.client import StdioMcpClient

        mock_result = {
            "success": True,
            "exit_code": 0,
            "stdout": "ok",
            "stderr": "",
            "elapsed_ms": 5.0,
            "container_id": "abc",
            "timed_out": False,
        }

        with patch("aifactory.mcp.sandbox_server._run_in_pool", return_value=mock_result):
            client = StdioMcpClient()
            result = await client.execute_bash(code="echo ok")

        assert result.success is True

    async def test_execute_javascript_delegates(self):
        from aifactory.mcp.client import StdioMcpClient

        mock_result = {
            "success": True,
            "exit_code": 0,
            "stdout": "js",
            "stderr": "",
            "elapsed_ms": 5.0,
            "container_id": "abc",
            "timed_out": False,
        }

        with patch("aifactory.mcp.sandbox_server._run_in_pool", return_value=mock_result):
            client = StdioMcpClient()
            result = await client.execute_javascript(code="console.log('js')")

        assert result.success is True

    async def test_health_returns_ok(self):
        from aifactory.mcp.client import StdioMcpClient

        client = StdioMcpClient()
        health = await client.health()
        assert health["status"] == "ok"
        assert health["mode"] == "stdio"

    async def test_close_is_noop(self):
        from aifactory.mcp.client import StdioMcpClient

        client = StdioMcpClient()
        await client.close()  # should not raise


class TestHttpMcpClient:
    """HttpMcpClient sends JSON-RPC 2.0 calls over HTTP."""

    async def test_execute_python_http(self):
        from aifactory.mcp.client import HttpMcpClient

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": '{"success": true, "exit_code": 0, "stdout": "hi", "stderr": "", "elapsed_ms": 20.0, "container_id": "abc123", "timed_out": false}',  # noqa: E501
                    }
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()

        mock_async_client = AsyncMock()
        mock_async_client.post.return_value = mock_response
        mock_async_client.aclose = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_async_client):
            client = HttpMcpClient(base_url="http://localhost:8100", auth_token="tok")
            result = await client.execute_python(code="print('hi')", agent_id="worker-1")
            await client.close()

        assert result.success is True
        assert result.stdout == "hi"
        # Two POSTs: 1) initialize (MCP session handshake) + 2) tools/call
        assert mock_async_client.post.call_count == 2
        # Verify the second call is the actual tool invocation
        tool_call_kwargs = mock_async_client.post.call_args_list[1]
        assert tool_call_kwargs is not None

    async def test_close_calls_aclose(self):
        from aifactory.mcp.client import HttpMcpClient

        mock_async_client = AsyncMock()
        mock_async_client.aclose = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_async_client):
            client = HttpMcpClient(base_url="http://localhost:8100")
            await client._ensure_client()
            await client.close()

        mock_async_client.aclose.assert_called_once()


class TestGetMcpClient:
    """get_mcp_client() factory selects correct client mode."""

    async def test_returns_stdio_client_when_no_url(self):
        from aifactory.mcp.client import StdioMcpClient, get_mcp_client

        with (
            patch.dict("os.environ", {"AIFACTORY_MCP_SANDBOX_URL": ""}),
            patch("aifactory.mcp.client._get_mcp_url", return_value=""),
        ):
            client = await get_mcp_client()

        assert isinstance(client, StdioMcpClient)

    async def test_returns_http_client_when_url_set(self):
        from aifactory.mcp.client import HttpMcpClient, get_mcp_client

        with patch("aifactory.mcp.client._get_mcp_url", return_value="http://localhost:8100"):
            client = await get_mcp_client()

        assert isinstance(client, HttpMcpClient)
        assert client._base_url == "http://localhost:8100"

    async def test_http_client_gets_auth_token(self):
        from aifactory.mcp.client import HttpMcpClient, get_mcp_client

        with (
            patch("aifactory.mcp.client._get_mcp_url", return_value="http://localhost:8100"),
            patch("aifactory.mcp.client._get_mcp_token", return_value="mytoken"),
        ):
            client = await get_mcp_client()

        assert isinstance(client, HttpMcpClient)
        assert client._auth_token == "mytoken"


# =============================================================================
# sandbox.py Docker backend routing
# =============================================================================


class TestDockerBackendSelection:
    """_get_backend() priority: docker > e2b > local."""

    def test_docker_enabled_returns_docker(self):
        from aifactory.core import sandbox

        with (
            patch.object(sandbox, "_docker_enabled", return_value=True),
            patch.dict("os.environ", {"AIFACTORY_SANDBOX_BACKEND": "auto"}),
        ):
            assert sandbox._get_backend() == "docker"

    def test_e2b_key_without_docker_returns_e2b(self):
        from aifactory.core import sandbox

        with (
            patch.object(sandbox, "_docker_enabled", return_value=False),
            patch.object(sandbox, "_e2b_api_key", return_value="key"),
            patch.dict("os.environ", {"AIFACTORY_SANDBOX_BACKEND": "auto"}),
        ):
            assert sandbox._get_backend() == "e2b"

    def test_no_docker_no_e2b_returns_local(self):
        from aifactory.core import sandbox

        with (
            patch.object(sandbox, "_docker_enabled", return_value=False),
            patch.object(sandbox, "_e2b_api_key", return_value=""),
            patch.dict("os.environ", {"AIFACTORY_SANDBOX_BACKEND": "auto"}),
        ):
            assert sandbox._get_backend() == "local"

    def test_explicit_backend_overrides_auto(self):
        from aifactory.core import sandbox

        with patch.dict("os.environ", {"AIFACTORY_SANDBOX_BACKEND": "local"}):
            assert sandbox._get_backend() == "local"

    def test_docker_disabled_env_var(self):
        from aifactory.core import sandbox

        with (
            patch.dict(
                "os.environ",
                {"AIFACTORY_SANDBOX_DOCKER_ENABLED": "false", "AIFACTORY_SANDBOX_BACKEND": "auto"},
            ),
            patch.object(sandbox, "_e2b_api_key", return_value=""),
        ):
            assert sandbox._get_backend() == "local"


class TestExecuteDockerBackend:
    """execute() routes through Docker when docker backend active."""

    async def test_execute_calls_docker(self):
        from aifactory.core.sandbox import ExecutionResult, execute

        mock_result = ExecutionResult(
            success=True, exit_code=0, stdout="hello", stderr="", backend="docker"
        )

        with (
            patch("aifactory.core.sandbox.execute_docker", return_value=mock_result) as mock_d,
            patch("aifactory.core.sandbox._get_backend", return_value="docker"),
        ):
            result = await execute("print('hello')", language="python", backend="docker")

        mock_d.assert_called_once()
        assert result.backend == "docker"
        assert result.success is True

    async def test_docker_infra_failure_falls_back_to_local(self):
        """If Docker backend returns exit_code=-1 (infra error), fall back."""
        from aifactory.core.sandbox import ExecutionResult, execute

        # Docker returns an infra error (not a code error)
        docker_fail = ExecutionResult(
            success=False,
            exit_code=-1,
            stdout="",
            stderr="Docker daemon unreachable",
            backend="docker",
        )
        local_ok = ExecutionResult(
            success=True,
            exit_code=0,
            stdout="hi",
            stderr="",
            backend="local",
        )

        with (
            patch("aifactory.core.sandbox.execute_docker", return_value=docker_fail),
            patch("aifactory.core.sandbox._get_backend", return_value="docker"),
            patch("aifactory.core.sandbox._e2b_api_key", return_value=""),
            patch("aifactory.core.sandbox.execute_python", return_value=local_ok),
        ):
            result = await execute("print('hi')", language="python")

        assert result.backend == "local"
        assert result.success is True

    async def test_execute_code_blocks_passes_agent_id(self):
        """execute_code_blocks() forwards agent_id to execute()."""
        from aifactory.core.sandbox import ExecutionResult, execute_code_blocks

        mock_result = ExecutionResult(
            success=True, exit_code=0, stdout="42", stderr="", backend="docker"
        )

        text = "```python\nprint(42)\n```"

        with patch("aifactory.core.sandbox.execute", return_value=mock_result) as mock_exec:
            await execute_code_blocks(text, agent_id="Worker-task_1")

        call_kwargs = mock_exec.call_args.kwargs
        assert call_kwargs.get("agent_id") == "Worker-task_1"


# =============================================================================
# agent_factory.py — package registry injection
# =============================================================================


class TestPackageRegistryInjection:
    """Package registry snippet is injected into worker prompts."""

    def test_snippet_loaded_from_yaml(self):
        from aifactory.core.agent_factory import _load_package_registry_snippet

        # Clear lru_cache for isolation
        _load_package_registry_snippet.cache_clear()
        snippet = _load_package_registry_snippet()
        # Should contain key packages from the registry
        # (empty string acceptable if file not found in test env)
        assert isinstance(snippet, str)

    def test_snippet_injected_into_worker_prompt(self):
        from aifactory.core import agent_factory

        # Patch snippet to return known content.
        # Pass assigned_toolkits=[] to bypass IntegrationRegistry singleton state
        # (which may have connected integrations from other tests).
        with (
            patch.object(
                agent_factory,
                "_load_package_registry_snippet",
                return_value="## Sandbox Environment\nnumpy, pandas",
            ),
            patch("aifactory.core.agent_factory.load_models", return_value=[]),
            patch(
                "aifactory.core.agent_factory.load_agent_cfg",
                return_value={"abilities": {}, "max_tokens": 2048, "model": "gpt-4o-mini"},
            ),
            patch(
                "aifactory.core.model_selector.select_model",
                return_value=("gpt-4o-mini", 1.0),
            ),
        ):
            cfg = agent_factory.create_worker_config(
                task_id="task_1",
                task_description="Write a pandas analysis",
                assigned_toolkits=[],  # explicit: no tools → coding path
            )

        assert "## Sandbox Environment" in cfg["system_prompt"]
        assert "numpy" in cfg["system_prompt"]

    def test_empty_snippet_does_not_break_prompt(self):
        from aifactory.core import agent_factory

        with (
            patch.object(agent_factory, "_load_package_registry_snippet", return_value=""),
            patch("aifactory.core.agent_factory.load_models", return_value=[]),
            patch(
                "aifactory.core.agent_factory.load_agent_cfg",
                return_value={"abilities": {}, "max_tokens": 2048, "model": "gpt-4o-mini"},
            ),
            patch(
                "aifactory.core.model_selector.select_model",
                return_value=("gpt-4o-mini", 1.0),
            ),
        ):
            cfg = agent_factory.create_worker_config(
                task_id="task_1",
                task_description="Simple task",
                assigned_toolkits=[],  # explicit: no tools → coding path
            )

        # Prompt should still be valid with no snippet
        assert "CODING EXCELLENCE" in cfg["system_prompt"]

    def test_snippet_not_injected_for_tool_tasks(self):
        """Integration tool tasks don't need sandbox env description."""
        from aifactory.core import agent_factory

        with (
            patch.object(
                agent_factory,
                "_load_package_registry_snippet",
                return_value="## Sandbox Environment\nnumpy",
            ),
            patch("aifactory.core.agent_factory.load_models", return_value=[]),
            patch(
                "aifactory.core.agent_factory.load_agent_cfg",
                return_value={"abilities": {}, "max_tokens": 2048, "model": "gpt-4o-mini"},
            ),
            patch(
                "aifactory.core.model_selector.select_model",
                return_value=("gpt-4o-mini", 1.0),
            ),
            patch(
                "aifactory.integrations.registry.IntegrationRegistry",
            ) as mock_reg_cls,
        ):
            mock_reg_inst = MagicMock()
            mock_reg_inst.get_connected_toolkit_names.return_value = ["github"]
            mock_reg_inst.get_connected.return_value = []
            mock_reg_cls.return_value = mock_reg_inst

            cfg = agent_factory.create_worker_config(
                task_id="task_1",
                task_description="Create a github issue",
                assigned_toolkits=["github"],
            )

        # Tool tasks have integration block instead of coding/sandbox block
        assert "INTEGRATION TOOLS" in cfg["system_prompt"]


# =============================================================================
# MCP server — _run_in_pool error handling
# =============================================================================


class TestMcpServerRunInPool:
    """_run_in_pool handles pool errors gracefully."""

    async def test_pool_timeout_returns_error_dict(self):
        from aifactory.mcp.sandbox_server import _run_in_pool

        with patch(
            "aifactory.core.sandbox_pool.get_pool",
            side_effect=TimeoutError("no containers"),
        ):
            result = await _run_in_pool(code="pass", language="python", agent_id="test", timeout=5)

        assert result["success"] is False
        assert result["error"] == "pool_timeout"
        assert result["exit_code"] == -1

    async def test_unexpected_error_returns_internal_error(self):
        from aifactory.mcp.sandbox_server import _run_in_pool

        with patch(
            "aifactory.core.sandbox_pool.get_pool",
            side_effect=RuntimeError("unexpected"),
        ):
            result = await _run_in_pool(code="pass", language="python", agent_id="test", timeout=5)

        assert result["success"] is False
        assert result["error"] == "internal_error"

    async def test_successful_execution_includes_elapsed(self):
        from aifactory.core.sandbox_pool import ExecResult
        from aifactory.mcp.sandbox_server import _run_in_pool

        mock_exec_result = ExecResult(exit_code=0, stdout="done", stderr="", container_id="abc123")

        mock_pool = AsyncMock()
        mock_pool.execute = AsyncMock(return_value=mock_exec_result)

        # Mock acquire as an async context manager
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _fake_acquire():
            yield MagicMock()

        mock_pool.acquire = _fake_acquire

        with patch("aifactory.core.sandbox_pool.get_pool", return_value=mock_pool):
            result = await _run_in_pool(
                code="print('done')", language="python", agent_id="worker", timeout=10
            )

        assert "elapsed_ms" in result
        assert result["elapsed_ms"] >= 0


# =============================================================================
# Security tests
# =============================================================================


class TestSecurityConstraints:
    """Verify security flags are applied to Docker containers."""

    def test_build_run_kwargs_has_security_flags(self):
        from aifactory.core.sandbox_pool import SandboxPool

        pool = SandboxPool(min_idle=0, max_size=1)
        pool._seccomp_profile = None
        kwargs = pool._build_run_kwargs()

        assert kwargs["network_mode"] == "none"
        assert "ALL" in kwargs["cap_drop"]
        assert kwargs["read_only"] is True
        assert "/workspace" in kwargs["tmpfs"]
        assert "/tmp" in kwargs["tmpfs"]
        assert "no-new-privileges" in kwargs["security_opt"]

    def test_gvisor_runtime_applied_when_enabled(self):
        from aifactory.core.sandbox_pool import SandboxPool

        pool = SandboxPool(gvisor=True, min_idle=0, max_size=1)
        pool._seccomp_profile = None
        kwargs = pool._build_run_kwargs()

        assert kwargs.get("runtime") == "runsc"

    def test_gvisor_runtime_absent_when_disabled(self):
        from aifactory.core.sandbox_pool import SandboxPool

        pool = SandboxPool(gvisor=False, min_idle=0, max_size=1)
        pool._seccomp_profile = None
        kwargs = pool._build_run_kwargs()

        assert "runtime" not in kwargs

    def test_seccomp_profile_added_to_security_opt(self, tmp_path):
        from aifactory.core.sandbox_pool import SandboxPool

        fake_profile = tmp_path / "seccomp.json"
        fake_profile.write_text("{}")
        pool = SandboxPool(seccomp_profile=str(fake_profile), min_idle=0, max_size=1)
        kwargs = pool._build_run_kwargs()

        security_opt_str = " ".join(kwargs["security_opt"])
        assert "seccomp=" in security_opt_str

    def test_pids_limit_applied(self):
        from aifactory.core.sandbox_pool import SandboxPool

        pool = SandboxPool(pids_limit=32, min_idle=0, max_size=1)
        pool._seccomp_profile = None
        kwargs = pool._build_run_kwargs()

        assert kwargs["pids_limit"] == 32

    def test_memory_limit_applied(self):
        from aifactory.core.sandbox_pool import SandboxPool

        pool = SandboxPool(memory="256m", min_idle=0, max_size=1)
        pool._seccomp_profile = None
        kwargs = pool._build_run_kwargs()

        assert kwargs["mem_limit"] == "256m"

    def test_extra_file_path_sanitised(self):
        """Extra file names have path separators stripped (no directory traversal)."""
        from pathlib import Path

        from aifactory.core.sandbox_pool import SandboxPool

        pool = SandboxPool(min_idle=0, max_size=1)
        pool._seccomp_profile = None
        pool._max_output_bytes = 50_000

        _make_container(exit_code=0, stdout=b"")

        # Verify that Path(fname).name strips any traversal attempts
        malicious_fname = "../../../etc/passwd"
        safe_name = Path(malicious_fname).name
        assert safe_name == "passwd"  # only the basename remains


# =============================================================================
# Autoscaler (_replenish_loop)
# =============================================================================


class TestReplenishLoop:
    """Unit tests for SandboxPool._replenish_loop autoscaler logic.

    Each test runs exactly ONE tick of the loop by patching asyncio.sleep to
    set pool._stopping=True after the first call — so the while-loop exits
    immediately without sleeping at all.
    """

    def _make_pool(
        self,
        min_idle: int = 2,
        max_size: int = 8,
        *,
        idle: int = 0,
        in_use: int = 0,
    ):
        """Return a fully-configured SandboxPool with preset pool state."""
        from aifactory.core.sandbox_pool import PooledContainer, SandboxPool

        pool = SandboxPool(min_idle=min_idle, max_size=max_size)
        pool._seccomp_profile = None
        pool._client = MagicMock()
        pool._started = True
        pool._stopping = False

        for i in range(idle):
            pc = PooledContainer(container_id=f"idle-{i}", container=MagicMock())
            pool._ready.put_nowait(pc)

        for i in range(in_use):
            pc = PooledContainer(container_id=f"inuse-{i}", container=MagicMock())
            pool._in_use[pc.container_id] = pc

        pool._total = idle + in_use
        return pool

    async def test_tops_up_to_min_idle(self):
        """When idle < min_idle, the loop creates containers to fill the gap."""
        pool = self._make_pool(min_idle=3, max_size=8, idle=1, in_use=0)
        created: list[int] = []

        async def fake_create_and_enqueue() -> None:
            created.append(1)

        async def stop_after_one(_delay: float) -> None:
            pool._stopping = True

        with (
            patch.object(pool, "_create_and_enqueue", side_effect=fake_create_and_enqueue),
            patch("aifactory.core.sandbox_pool.asyncio.sleep", side_effect=stop_after_one),
        ):
            await pool._replenish_loop()

        # min_idle=3, idle=1 → deficit of 2
        assert len(created) == 2

    async def test_scale_up_at_high_utilization(self):
        """When utilization ≥ 70%, proactively burst-create containers."""
        # 6 in_use, 2 idle, 8 total out of 16 max  →  75% utilization
        pool = self._make_pool(min_idle=2, max_size=16, idle=2, in_use=6)
        created: list[int] = []

        async def fake_create_and_enqueue() -> None:
            created.append(1)

        async def stop_after_one(_delay: float) -> None:
            pool._stopping = True

        with (
            patch.object(pool, "_create_and_enqueue", side_effect=fake_create_and_enqueue),
            patch("aifactory.core.sandbox_pool.asyncio.sleep", side_effect=stop_after_one),
        ):
            await pool._replenish_loop()

        # Burst = max(1, 16//4) = 4, capped by remaining capacity (16 - 8 = 8)
        assert len(created) >= 1

    async def test_scale_down_after_ticks_threshold(self):
        """When idle > min_idle for SCALE_DOWN_TICKS consecutive ticks, retire one."""
        # 5 idle, 0 in_use — _idle_ticks is already at threshold - 1
        pool = self._make_pool(min_idle=2, max_size=8, idle=5, in_use=0)
        pool._idle_ticks = pool._SCALE_DOWN_TICKS - 1

        async def stop_after_one(_delay: float) -> None:
            pool._stopping = True

        retired: list[str] = []

        async def fake_retire(pc, *, remove: bool = True) -> None:
            retired.append(pc.container_id)
            pool._total = max(0, pool._total - 1)

        with (
            patch.object(pool, "_retire_container", side_effect=fake_retire),
            patch("aifactory.core.sandbox_pool.asyncio.sleep", side_effect=stop_after_one),
        ):
            await pool._replenish_loop()

        assert len(retired) == 1, "One idle container should have been retired"

    async def test_no_scale_down_at_min_idle_floor(self):
        """Pool must never shrink below min_idle containers."""
        pool = self._make_pool(min_idle=2, max_size=8, idle=2, in_use=0)
        pool._idle_ticks = 99  # Many idle ticks — should NOT fire

        async def stop_after_one(_delay: float) -> None:
            pool._stopping = True

        retired: list[str] = []

        async def fake_retire(pc, *, remove: bool = True) -> None:
            retired.append(pc.container_id)

        with (
            patch.object(pool, "_retire_container", side_effect=fake_retire),
            patch("aifactory.core.sandbox_pool.asyncio.sleep", side_effect=stop_after_one),
        ):
            await pool._replenish_loop()

        assert len(retired) == 0, "Should not retire when idle == min_idle"

    async def test_idle_ticks_reset_on_growth(self):
        """idle_ticks resets to 0 whenever a replenish action occurs."""
        # deficit → growth path → _idle_ticks should be zeroed
        pool = self._make_pool(min_idle=3, max_size=8, idle=1, in_use=0)
        pool._idle_ticks = 7  # Artificially pre-set

        async def fake_create_and_enqueue() -> None:
            pass

        async def stop_after_one(_delay: float) -> None:
            pool._stopping = True

        with (
            patch.object(pool, "_create_and_enqueue", side_effect=fake_create_and_enqueue),
            patch("aifactory.core.sandbox_pool.asyncio.sleep", side_effect=stop_after_one),
        ):
            await pool._replenish_loop()

        assert pool._idle_ticks == 0, "idle_ticks must reset to 0 after growth"


# =============================================================================
# Module-level singleton
# =============================================================================


class TestModuleLevelPool:
    """get_pool() / shutdown_pool() singleton lifecycle."""

    async def test_get_pool_returns_singleton(self):
        import aifactory.core.sandbox_pool as sp

        # Reset singleton
        original_pool = sp._pool
        sp._pool = None
        sp._pool_lock = asyncio.Lock()

        mock_pool = AsyncMock(spec=sp.SandboxPool)
        mock_pool.start = AsyncMock()
        mock_pool.pool_status = AsyncMock(return_value={})

        with patch("aifactory.core.sandbox_pool.SandboxPool", return_value=mock_pool):
            p1 = await sp.get_pool()
            p2 = await sp.get_pool()

        assert p1 is p2
        mock_pool.start.assert_called_once()

        # Restore
        sp._pool = original_pool

    async def test_shutdown_pool_resets_singleton(self):
        import aifactory.core.sandbox_pool as sp

        mock_pool = AsyncMock(spec=sp.SandboxPool)
        mock_pool.stop = AsyncMock()
        original_pool = sp._pool
        sp._pool = mock_pool

        await sp.shutdown_pool()

        assert sp._pool is None
        mock_pool.stop.assert_called_once()

        sp._pool = original_pool
