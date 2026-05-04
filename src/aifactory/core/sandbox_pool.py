"""
Docker Container Pool — Warm sandbox pool for low-latency code execution.

Pre-warms N containers so every code execution hits a running container via
`docker exec` (10-50ms) rather than spinning up a fresh container per request
(800ms-2s cold start).

Security layers applied to each container:
  - --network=none               no outbound connectivity
  - --cap-drop=ALL               all Linux capabilities dropped
  - --read-only                  root FS is read-only
  - --tmpfs /workspace           ephemeral scratch space (cleared between runs)
  - --memory / --pids-limit      resource caps
  - --security-opt seccomp=...   custom syscall allowlist
  - gVisor runtime (optional)    kernel isolation via syscall interception

Usage:
    pool = SandboxPool()
    await pool.start()

    async with pool.acquire() as container:
        result = await pool.execute(container, "print('hi')", language="python")

    await pool.stop()
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import docker.models.containers

    import docker

logger = logging.getLogger(__name__)

# Workspace directory inside containers (tmpfs-mounted)
WORKSPACE_DIR = "/workspace"
# Execution timeout sentinel used in exec output polling
_POLL_INTERVAL = 0.05  # 50ms


# =============================================================================
# Data types
# =============================================================================


@dataclass
class PooledContainer:
    """A running Docker container managed by SandboxPool."""

    container_id: str
    container: object  # docker.models.containers.Container
    created_at: float = field(default_factory=time.monotonic)
    reuse_count: int = 0
    max_reuse: int = 50

    @property
    def is_expired(self) -> bool:
        return self.reuse_count >= self.max_reuse


@dataclass
class ExecResult:
    """Result from docker exec in a PooledContainer."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    container_id: str = ""

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    @property
    def output(self) -> str:
        parts = []
        if self.stdout.strip():
            parts.append(self.stdout.strip())
        if self.stderr.strip():
            parts.append(f"STDERR:\n{self.stderr.strip()}")
        return "\n".join(parts) or "(no output)"


# =============================================================================
# SandboxPool
# =============================================================================


class SandboxPool:
    """
    Async warm-container pool for Docker sandbox execution.

    Lifecycle:
        await pool.start()   — pre-warms min_idle containers
        # use pool ...
        await pool.stop()    — shuts down all containers + background task
    """

    def __init__(
        self,
        *,
        image: str | None = None,
        min_idle: int | None = None,
        max_size: int | None = None,
        max_reuse: int | None = None,
        acquire_timeout: float | None = None,
        memory: str | None = None,
        cpus: float | None = None,
        pids_limit: int | None = None,
        seccomp_profile: str | None = None,
        gvisor: bool | None = None,
        max_output_bytes: int | None = None,
        network_mode: str = "none",
    ) -> None:
        from aifactory.config.settings import get_settings

        s = get_settings()
        self._image = image or s.sandbox_docker_image
        self._min_idle = min_idle if min_idle is not None else s.sandbox_pool_min_idle
        self._max_size = max_size if max_size is not None else s.sandbox_pool_max_size
        self._max_reuse = max_reuse if max_reuse is not None else s.sandbox_pool_max_reuse
        self._acquire_timeout = (
            acquire_timeout if acquire_timeout is not None else s.sandbox_pool_acquire_timeout
        )
        self._memory = memory or s.sandbox_docker_memory
        self._cpus = cpus if cpus is not None else s.sandbox_docker_cpus
        self._pids_limit = pids_limit if pids_limit is not None else s.sandbox_docker_pids_limit
        self._gvisor = gvisor if gvisor is not None else s.sandbox_gvisor_enabled
        self._max_output_bytes = (
            max_output_bytes if max_output_bytes is not None else s.sandbox_max_output_bytes
        )
        self._network_mode = network_mode

        # Resolve seccomp profile path
        raw_profile = seccomp_profile or s.sandbox_seccomp_profile
        if raw_profile:
            p = Path(raw_profile)
            if not p.is_absolute():
                # Resolve relative to repo root.
                # settings.py lives at <root>/src/aifactory/config/settings.py → parents[3].
                # In Docker it lives at /app/aifactory/config/settings.py → parents[3] = /
                # which won't have the file; we fall back to Docker default seccomp silently.
                try:
                    from aifactory.config import settings as _s_mod

                    repo_root = Path(_s_mod.__file__).resolve().parents[3]
                    p = repo_root / raw_profile
                except (IndexError, Exception):
                    p = Path(raw_profile)  # keep as-is; existence check below handles it
            self._seccomp_profile: str | None = str(p) if p.exists() else None
            if raw_profile and not self._seccomp_profile:
                logger.warning("seccomp profile not found at %s — using Docker default", p)
        else:
            self._seccomp_profile = None

        # Pool state
        self._ready: asyncio.Queue[PooledContainer] = asyncio.Queue()
        self._in_use: dict[str, PooledContainer] = {}
        self._total = 0
        self._lock = asyncio.Lock()
        self._replenish_task: asyncio.Task | None = None
        self._started = False
        self._stopping = False
        self._client: docker.DockerClient | None = None

        # Autoscaler state — consecutive ticks of low utilisation before shrink
        self._idle_ticks: int = 0
        self._SCALE_UP_THRESHOLD: float = 0.70  # grow when 70 % of pool is busy
        self._SCALE_DOWN_THRESHOLD: float = 0.20  # shrink when < 20 % busy
        self._SCALE_DOWN_TICKS: int = 3  # require 3 × 5 s = 15 s sustained idle

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Initialise Docker client and pre-warm min_idle containers."""
        if self._started:
            return
        self._client = await asyncio.to_thread(self._make_docker_client)
        self._started = True
        # Remove orphan containers left over from a previous crashed run
        await self._cleanup_orphans()
        # Pre-warm
        await self._prefill()
        # Background replenish loop
        self._replenish_task = asyncio.create_task(
            self._replenish_loop(), name="sandbox-pool-replenish"
        )
        logger.info(
            "SandboxPool started — image=%s, min_idle=%d, max_size=%d",
            self._image,
            self._min_idle,
            self._max_size,
        )

    async def stop(self) -> None:
        """Gracefully shut down: cancel replenish task, kill all containers."""
        if not self._started:
            return
        self._stopping = True
        if self._replenish_task and not self._replenish_task.done():
            self._replenish_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._replenish_task
        # Kill all ready containers
        killed = 0
        while not self._ready.empty():
            try:
                c = self._ready.get_nowait()
                await self._retire_container(c, remove=True)
                killed += 1
            except asyncio.QueueEmpty:
                break
        # Kill in-use containers
        for c in list(self._in_use.values()):
            await self._retire_container(c, remove=True)
            killed += 1
        logger.info("SandboxPool stopped — removed %d containers", killed)
        self._started = False

    @contextlib.asynccontextmanager
    async def acquire(self) -> AsyncIterator[PooledContainer]:
        """
        Context manager that lends a PooledContainer from the pool.

        Waits up to acquire_timeout seconds for a free container.
        Automatically returns or retires the container on exit.
        """
        container = await self._acquire_container()
        try:
            yield container
        finally:
            await self._release_container(container)

    async def execute(
        self,
        container: PooledContainer,
        code: str,
        language: str = "python",
        *,
        timeout: int = 30,
        extra_files: dict[str, str] | None = None,
    ) -> ExecResult:
        """
        Execute code inside *container* via docker exec.

        Args:
            container: Acquired PooledContainer.
            code: Source code to run.
            language: "python", "bash", or "javascript"/"js".
            timeout: Per-execution timeout in seconds.
            extra_files: Additional files to write to /workspace/{name}.

        Returns:
            ExecResult with stdout, stderr, exit_code.
        """
        # Pre-flight: refresh container state from Docker and verify it is still
        # running.  This catches the TOCTOU window between _health_check in
        # _acquire_container and the first exec call in this method.
        try:
            await asyncio.to_thread(
                lambda: container.container.reload()  # type: ignore[union-attr]
            )
            status = getattr(container.container, "status", "unknown")
            if status != "running":
                return ExecResult(
                    exit_code=-1,
                    stdout="",
                    stderr=(
                        f"Container {container.container_id[:12]} is not running "
                        f"(status={status}). The sandbox pool will replenish automatically."
                    ),
                    container_id=container.container_id,
                )
        except Exception as exc:
            logger.warning(
                "Container %s pre-flight reload failed: %s", container.container_id[:12], exc
            )
            # Non-fatal — proceed and let the exec fail with its own error if needed.

        container.reuse_count += 1

        # Clear workspace from previous run
        await self._cleanup_workspace(container)

        # Write extra files
        if extra_files:
            for fname, content in extra_files.items():
                safe_name = Path(fname).name  # strip any path separators
                await self._write_file(container, f"{WORKSPACE_DIR}/{safe_name}", content)

        # Write main script
        if language == "python":
            script_path = f"{WORKSPACE_DIR}/_main.py"
            cmd = ["python3", script_path]
        elif language in ("bash", "sh"):
            script_path = f"{WORKSPACE_DIR}/_main.sh"
            cmd = ["bash", script_path]
        elif language in ("javascript", "js"):
            script_path = f"{WORKSPACE_DIR}/_main.js"
            cmd = ["node", script_path]
        else:
            return ExecResult(
                exit_code=1,
                stdout="",
                stderr=f"Unsupported language: {language}",
                container_id=container.container_id,
            )

        await self._write_file(container, script_path, code)

        # Run via docker exec
        return await self._docker_exec(container, cmd, timeout=timeout)

    async def pool_status(self) -> dict:
        """Return a snapshot of pool health metrics."""
        ready = self._ready.qsize()
        in_use = len(self._in_use)
        total = self._total
        return {
            "ready": ready,
            "in_use": in_use,
            "total": total,
            "min_idle": self._min_idle,
            "max_size": self._max_size,
            "utilization_pct": round(in_use / total * 100, 1) if total > 0 else 0.0,
            "image": self._image,
            "gvisor": self._gvisor,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _make_docker_client(self):
        """Create Docker client — runs in a thread (blocking)."""
        try:
            import docker as docker_sdk

            client = docker_sdk.from_env()
            client.ping()
            return client
        except Exception as exc:
            raise RuntimeError(
                "Docker is not available. Ensure the Docker daemon is running "
                "and the docker Python SDK is installed (pip install docker)."
            ) from exc

    def _build_run_kwargs(self) -> dict:
        """Build docker.run() keyword arguments with security flags."""
        security_opts = ["no-new-privileges"]
        if self._seccomp_profile:
            # The Docker Python SDK passes security_opt strings directly to the API.
            # The API expects the seccomp option value to be the JSON *content*,
            # not a file path (unlike the Docker CLI which reads the file itself).
            try:
                seccomp_json = Path(self._seccomp_profile).read_text(encoding="utf-8")
                security_opts.append(f"seccomp={seccomp_json}")
            except Exception as exc:
                logger.warning(
                    "Could not read seccomp profile %s: %s — using default",
                    self._seccomp_profile,
                    exc,
                )

        kwargs: dict = dict(
            image=self._image,
            command=["sleep", "infinity"],  # keep-alive; exec into it
            detach=True,
            auto_remove=False,
            network_mode=self._network_mode,
            cap_drop=["ALL"],
            read_only=True,
            tmpfs={
                # uid=1000,gid=1000 matches the sandbox user in the Dockerfile.
                # Without explicit ownership Docker mounts tmpfs as root (0), so
                # the non-root sandbox user cannot create files.
                WORKSPACE_DIR: "rw,noexec,nosuid,size=128m,uid=1000,gid=1000,mode=755",
                "/tmp": "rw,noexec,nosuid,size=64m,uid=1000,gid=1000,mode=755",
            },
            security_opt=security_opts,
            mem_limit=self._memory,
            pids_limit=self._pids_limit,
            nano_cpus=int(self._cpus * 1e9),
            labels={"managed-by": "aifactory-sandbox-pool"},
        )

        if self._gvisor:
            kwargs["runtime"] = "runsc"

        return kwargs

    async def _create_container(self) -> PooledContainer | None:
        """Spin up a new sandbox container. Returns None if Docker errors."""
        if not self._client:
            return None
        try:
            run_kwargs = self._build_run_kwargs()
            container = await asyncio.to_thread(
                lambda: self._client.containers.run(**run_kwargs)  # type: ignore[union-attr]
            )
            pc = PooledContainer(
                container_id=container.id,
                container=container,
                max_reuse=self._max_reuse,
            )
            async with self._lock:
                self._total += 1
            logger.debug("Created container %s (total=%d)", container.id[:12], self._total)

            # Wait until the container is stably running and accepting exec calls.
            # Without this, freshly created containers can fail exec with 409 due to a
            # race condition where `containers.run()` returns before the process is
            # fully ready, or the container exits immediately (exit 255) when created
            # inside a nested Docker context (e.g., MCP server container via socket proxy).
            if not await self._wait_until_ready(pc):
                logger.warning("Container %s did not become ready — retiring", container.id[:12])
                async with self._lock:
                    self._total = max(0, self._total - 1)
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(lambda: container.remove(force=True))
                return None

            return pc
        except Exception as exc:
            logger.warning("Failed to create container: %s", exc)
            return None

    async def _wait_until_ready(
        self, pc: PooledContainer, *, max_wait: float = 8.0, interval: float = 0.2
    ) -> bool:
        """
        Poll until *pc* is stably running and accepting exec calls.

        Reloads container state from the Docker daemon on each attempt so that
        containers that exit immediately (exit 255 race) are caught before they
        are handed to callers.  Returns True when ready, False on timeout.
        """
        deadline = time.monotonic() + max_wait
        attempt = 0
        while time.monotonic() < deadline:
            attempt += 1
            try:
                # Refresh container state (GET /containers/{id}/json)
                await asyncio.to_thread(
                    lambda: pc.container.reload()  # type: ignore[union-attr]
                )
                status = getattr(pc.container, "status", "unknown")
                if status != "running":
                    logger.debug(
                        "Container %s status=%s on attempt %d",
                        pc.container_id[:12],
                        status,
                        attempt,
                    )
                    await asyncio.sleep(interval)
                    continue
                # Container reports running — verify exec actually works
                if await self._health_check(pc):
                    elapsed = time.monotonic() - (deadline - max_wait)
                    logger.debug(
                        "Container %s ready in %.2fs (attempt %d)",
                        pc.container_id[:12],
                        elapsed,
                        attempt,
                    )
                    return True
            except Exception as exc:
                logger.debug("Container %s readiness probe error: %s", pc.container_id[:12], exc)
            await asyncio.sleep(interval)
        return False

    async def _retire_container(self, pc: PooledContainer, *, remove: bool = True) -> None:
        """Kill and optionally remove a container."""
        async with self._lock:
            self._in_use.pop(pc.container_id, None)
            self._total = max(0, self._total - 1)
        if not remove:
            return
        try:
            await asyncio.to_thread(
                lambda: pc.container.remove(force=True)  # type: ignore[union-attr]
            )
            logger.debug("Removed container %s", pc.container_id[:12])
        except Exception as exc:
            logger.debug("Container removal failed (may already be gone): %s", exc)

    async def _health_check(self, pc: PooledContainer) -> bool:
        """Quick liveness check — run `true` in the container."""
        try:
            result = await asyncio.to_thread(
                lambda: pc.container.exec_run(  # type: ignore[union-attr]
                    "true", stdout=False, stderr=False
                )
            )
            return result.exit_code == 0
        except Exception:
            return False

    async def _cleanup_workspace(self, pc: PooledContainer) -> None:
        """Clear /workspace between executions."""
        with contextlib.suppress(Exception):
            await asyncio.to_thread(
                lambda: pc.container.exec_run(  # type: ignore[union-attr]
                    ["sh", "-c", f"rm -rf {WORKSPACE_DIR}/* {WORKSPACE_DIR}/.[!.]*"],
                    stdout=False,
                    stderr=False,
                )
            )

    async def _write_file(self, pc: PooledContainer, path: str, content: str) -> None:
        """Write a file into the container via exec + base64 decode."""
        import base64

        encoded = base64.b64encode(content.encode()).decode()
        cmd = ["sh", "-c", f"echo {encoded} | base64 -d > {path}"]
        try:
            await asyncio.to_thread(
                lambda: pc.container.exec_run(cmd, stdout=False, stderr=False)  # type: ignore[union-attr]
            )
        except Exception as exc:
            exc_str = str(exc)
            if "409" in exc_str or "not running" in exc_str.lower():
                raise RuntimeError(
                    f"Container {pc.container_id[:12]} stopped during file write (409). "
                    "The container exited between health-check and execution — "
                    "the pool will create a fresh replacement automatically."
                ) from exc
            raise

    async def _docker_exec(
        self,
        pc: PooledContainer,
        cmd: list[str],
        *,
        timeout: int,
    ) -> ExecResult:
        """Run *cmd* inside *pc* and capture output with a timeout."""
        try:
            exec_result = await asyncio.wait_for(
                asyncio.to_thread(
                    lambda: pc.container.exec_run(  # type: ignore[union-attr]
                        cmd,
                        stdout=True,
                        stderr=True,
                        workdir=WORKSPACE_DIR,
                        demux=True,
                    )
                ),
                timeout=float(timeout),
            )
        except TimeoutError:
            return ExecResult(
                exit_code=-1,
                stdout="",
                stderr=f"Execution timed out after {timeout}s",
                timed_out=True,
                container_id=pc.container_id,
            )
        except Exception as exc:
            return ExecResult(
                exit_code=-1,
                stdout="",
                stderr=str(exc),
                container_id=pc.container_id,
            )

        exit_code = exec_result.exit_code or 0
        stdout_bytes, stderr_bytes = exec_result.output or (b"", b"")
        stdout = (stdout_bytes or b"").decode(errors="replace")[: self._max_output_bytes]
        stderr = (stderr_bytes or b"").decode(errors="replace")[: self._max_output_bytes]
        return ExecResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            container_id=pc.container_id,
        )

    async def _acquire_container(self) -> PooledContainer:
        """Wait for a ready container, respecting acquire_timeout."""
        deadline = time.monotonic() + self._acquire_timeout
        while True:
            # Try to get from ready queue (non-blocking)
            try:
                pc = self._ready.get_nowait()
                if await self._health_check(pc):
                    async with self._lock:
                        self._in_use[pc.container_id] = pc
                    return pc
                # Unhealthy — retire and try again
                await self._retire_container(pc)
            except asyncio.QueueEmpty:
                pass

            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"SandboxPool: no container available after {self._acquire_timeout}s "
                    f"(ready={self._ready.qsize()}, in_use={len(self._in_use)}, total={self._total})"  # noqa: E501
                )

            # If we have capacity, spawn a new container immediately.
            # _create_container() now calls _wait_until_ready internally, so the
            # returned container is guaranteed to be exec-ready (or None on failure).
            if self._total < self._max_size:
                pc = await self._create_container()
                if pc:
                    async with self._lock:
                        self._in_use[pc.container_id] = pc
                    return pc

            await asyncio.sleep(0.1)

    async def _release_container(self, pc: PooledContainer) -> None:
        """Return a container to the pool or retire it if expired."""
        async with self._lock:
            self._in_use.pop(pc.container_id, None)

        if pc.is_expired or self._stopping:
            await self._retire_container(pc, remove=True)
        else:
            await self._ready.put(pc)

    async def _cleanup_orphans(self) -> None:
        """Remove stale sandbox containers from previous/crashed pool runs.

        Two-phase cleanup:
        1. Remove ALL exited/stopped containers with the pool label, regardless
           of which pool instance created them.  These are debris from crashes or
           rejected seccomp profiles that would otherwise accumulate and waste
           disk space.  Without ``all=True``, ``containers.list()`` skips stopped
           containers entirely.
        2. Leave *running* containers alone — they may belong to another live pool
           instance (e.g. the MCP server's warm pool running in its own process).
           Removing them would invalidate that pool's in-memory state and cause
           immediate 409 errors on the next exec call.
        """
        if self._client is None:
            return
        try:
            # Phase 1: exited/stopped containers — always safe to remove
            stopped = await asyncio.to_thread(
                self._client.containers.list,
                all=True,
                filters={
                    "label": "managed-by=aifactory-sandbox-pool",
                    "status": "exited",
                },
            )
            # Also catch containers stuck in "created" or "dead" state
            for extra_status in ("created", "dead"):
                extras = await asyncio.to_thread(
                    self._client.containers.list,
                    all=True,
                    filters={
                        "label": "managed-by=aifactory-sandbox-pool",
                        "status": extra_status,
                    },
                )
                stopped = stopped + extras

            if stopped:
                logger.info("Removing %d stopped orphan container(s) with pool label", len(stopped))
                for c in stopped:
                    try:
                        await asyncio.to_thread(c.remove, force=True)
                    except Exception as exc:
                        logger.debug("Could not remove orphan %s: %s", c.short_id, exc)

        except Exception as exc:
            logger.warning("Orphan cleanup failed (non-fatal): %s", exc)

    async def _prefill(self) -> None:
        """Create min_idle containers upfront."""
        tasks = [asyncio.create_task(self._create_and_enqueue()) for _ in range(self._min_idle)]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _create_and_enqueue(self) -> None:
        pc = await self._create_container()
        if pc:
            await self._ready.put(pc)

    async def _replenish_loop(self) -> None:
        """
        Background task — maintains pool size dynamically.

        Every 5 s it:
          1. Tops up ready queue to min_idle (existing behaviour).
          2. Scales UP proactively when utilisation > SCALE_UP_THRESHOLD.
          3. Scales DOWN gradually when utilisation < SCALE_DOWN_THRESHOLD
             for SCALE_DOWN_TICKS consecutive ticks (default: 15 s).
        """
        while not self._stopping:
            try:
                idle = self._ready.qsize()
                in_use = len(self._in_use)
                total = self._total
                utilization = in_use / total if total > 0 else 0.0
                capacity = self._max_size - total

                # ── 1. Always maintain min_idle floor ────────────────────────
                deficit = self._min_idle - idle
                if deficit > 0 and capacity > 0:
                    to_create = min(deficit, capacity)
                    logger.debug(
                        "Replenish: idle=%d in_use=%d total=%d creating=%d",
                        idle,
                        in_use,
                        total,
                        to_create,
                    )
                    await asyncio.gather(
                        *[
                            asyncio.create_task(self._create_and_enqueue())
                            for _ in range(to_create)
                        ],
                        return_exceptions=True,
                    )
                    self._idle_ticks = 0  # reset shrink counter on any growth

                # ── 2. Scale UP — proactive burst capacity ────────────────────
                elif utilization >= self._SCALE_UP_THRESHOLD and capacity > 0:
                    # Add up to 25 % of max_size in one burst, capped by capacity
                    burst = min(max(1, self._max_size // 4), capacity)
                    logger.info(
                        "Autoscale UP: util=%.0f%% total=%d → +%d containers",
                        utilization * 100,
                        total,
                        burst,
                    )
                    await asyncio.gather(
                        *[asyncio.create_task(self._create_and_enqueue()) for _ in range(burst)],
                        return_exceptions=True,
                    )
                    self._idle_ticks = 0

                # ── 3. Scale DOWN — shed excess idle containers gradually ─────
                elif (
                    utilization < self._SCALE_DOWN_THRESHOLD
                    and idle > self._min_idle
                    and total > self._min_idle
                ):
                    self._idle_ticks += 1
                    if self._idle_ticks >= self._SCALE_DOWN_TICKS:
                        # Remove one container per tick until back at min_idle
                        try:
                            pc = self._ready.get_nowait()
                            await self._retire_container(pc, remove=True)
                            logger.info(
                                "Autoscale DOWN: util=%.0f%% total=%d → removed 1 container",
                                utilization * 100,
                                total,
                            )
                        except asyncio.QueueEmpty:
                            pass
                        self._idle_ticks = 0  # reset; re-check next tick
                else:
                    self._idle_ticks = 0

                # Emit utilisation metric every tick for observability
                logger.debug(
                    "Pool tick: idle=%d in_use=%d total=%d util=%.0f%%",
                    idle,
                    in_use,
                    total,
                    utilization * 100,
                )

            except Exception as exc:
                logger.warning("Replenish loop error: %s", exc)

            await asyncio.sleep(5)


# =============================================================================
# Module-level singleton helpers
# =============================================================================

_pool: SandboxPool | None = None
_pool_lock = asyncio.Lock()


async def get_pool() -> SandboxPool:
    """Return the module-level SandboxPool, starting it on first call."""
    global _pool
    async with _pool_lock:
        if _pool is None:
            _pool = SandboxPool()
            await _pool.start()
    return _pool


async def shutdown_pool() -> None:
    """Gracefully stop the module-level pool."""
    global _pool
    if _pool is not None:
        await _pool.stop()
        _pool = None
