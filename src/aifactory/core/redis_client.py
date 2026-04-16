"""
Redis caching + pub/sub layer for AiFactory.

Provides:
  - Task status cache (fast reads, avoids hitting PostgreSQL on every poll)
  - Pub/sub channel for real-time WebSocket fan-out (multi-worker ready)
  - TTL-based cache invalidation

Connection is lazy — if REDIS_URL is not set, all operations silently no-op
so the app still works without Redis (just hits Postgres directly).
"""

from __future__ import annotations

import contextlib
import json
import logging
from typing import Any

import redis.asyncio as aioredis

from aifactory.config.settings import get_settings

logger = logging.getLogger(__name__)

# ── Module-level client (lazy singleton) ──────────────────────────────────────

_client: aioredis.Redis | None = None
_available: bool | None = None  # None = not yet checked

# Cache TTLs
TASK_CACHE_TTL = 300  # 5 minutes for completed tasks
ACTIVE_TASK_TTL = 30  # 30 seconds for running tasks


def get_redis() -> aioredis.Redis | None:
    """Return (or create) the async Redis client. Returns None if unconfigured."""
    global _client, _available
    if _available is False:
        return None
    if _client is not None:
        return _client

    settings = get_settings()
    url = settings.redis_url
    if not url:
        logger.info("REDIS_URL not set — Redis caching disabled (Postgres-only mode)")
        _available = False
        return None

    try:
        _client = aioredis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=3,
            retry_on_timeout=True,
        )
        _available = True
        logger.info("Redis client created: %s", url.split("@")[-1] if "@" in url else url)
        return _client
    except Exception as e:
        logger.warning("Redis connection failed: %s — running without cache", e)
        _available = False
        return None


async def close_redis() -> None:
    """Close the Redis connection (call on app shutdown)."""
    global _client, _available
    if _client:
        await _client.aclose()
        _client = None
        _available = None
        logger.info("Redis connection closed")


def reset_redis() -> None:
    """Reset the client reference (for testing)."""
    global _client, _available
    _client = None
    _available = None


async def redis_health() -> bool:
    """Return True if Redis is reachable."""
    client = get_redis()
    if client is None:
        return False
    try:
        return await client.ping()
    except Exception:
        return False


# ── Task Cache ────────────────────────────────────────────────────────────────

_KEY_PREFIX = "aifactory:task:"


def _task_key(task_id: str) -> str:
    return f"{_KEY_PREFIX}{task_id}"


async def cache_task(task_id: str, data: dict[str, Any]) -> None:
    """Cache a task's current state in Redis."""
    client = get_redis()
    if client is None:
        return
    try:
        # Don't cache WebSocket subscriber objects
        safe = {k: v for k, v in data.items() if k != "subscribers"}
        ttl = TASK_CACHE_TTL if data.get("status") in ("completed", "failed") else ACTIVE_TASK_TTL
        await client.setex(_task_key(task_id), ttl, json.dumps(safe, default=str))
    except Exception as e:
        logger.debug("Redis cache_task failed: %s", e)


async def get_cached_task(task_id: str) -> dict[str, Any] | None:
    """Fetch a task from Redis cache. Returns None on miss."""
    client = get_redis()
    if client is None:
        return None
    try:
        raw = await client.get(_task_key(task_id))
        if raw:
            return json.loads(raw)
    except Exception as e:
        logger.debug("Redis get_cached_task failed: %s", e)
    return None


async def invalidate_task(task_id: str) -> None:
    """Remove a task from the cache (force next read to hit Postgres)."""
    client = get_redis()
    if client is None:
        return
    try:
        await client.delete(_task_key(task_id))
    except Exception as e:
        logger.debug("Redis invalidate_task failed: %s", e)


# ── Task List Cache ──────────────────────────────────────────────────────────

_LIST_KEY = "aifactory:task_list"


async def cache_task_list(tasks: list[dict[str, Any]], status: str | None = None) -> None:
    """Cache the task list response."""
    client = get_redis()
    if client is None:
        return
    try:
        key = f"{_LIST_KEY}:{status or 'all'}"
        await client.setex(key, ACTIVE_TASK_TTL, json.dumps(tasks, default=str))
    except Exception as e:
        logger.debug("Redis cache_task_list failed: %s", e)


async def get_cached_task_list(status: str | None = None) -> list[dict[str, Any]] | None:
    """Fetch cached task list. Returns None on miss."""
    client = get_redis()
    if client is None:
        return None
    try:
        key = f"{_LIST_KEY}:{status or 'all'}"
        raw = await client.get(key)
        if raw:
            return json.loads(raw)
    except Exception as e:
        logger.debug("Redis get_cached_task_list failed: %s", e)
    return None


async def invalidate_task_list() -> None:
    """Invalidate all task list caches (called after any task mutation)."""
    client = get_redis()
    if client is None:
        return
    try:
        keys = []
        async for key in client.scan_iter(f"{_LIST_KEY}:*"):
            keys.append(key)
        if keys:
            await client.delete(*keys)
    except Exception as e:
        logger.debug("Redis invalidate_task_list failed: %s", e)


# ── Pub/Sub — multi-replica WebSocket fan-out (D3) ───────────────────────────
# Worker events published by any API replica are broadcast to all replicas via
# Redis pub/sub, so WebSocket clients always receive events regardless of which
# replica ran the pipeline.

_PUBSUB_CHANNEL_PREFIX = "aifactory:events:"


def _event_channel(task_id: str) -> str:
    return f"{_PUBSUB_CHANNEL_PREFIX}{task_id}"


async def publish_event(task_id: str, event: dict[str, Any]) -> None:
    """Publish a pipeline event to the Redis pub/sub channel for task_id."""
    client = get_redis()
    if client is None:
        return
    try:
        await client.publish(_event_channel(task_id), json.dumps(event, default=str))
    except Exception as exc:
        logger.debug("Redis publish_event failed: %s", exc)


async def subscribe_events(task_id: str):
    """
    Async generator that yields pipeline events from Redis pub/sub.

    Yields dict events published via publish_event().
    The caller is responsible for breaking out of the loop when done.

    Usage:
        async for event in subscribe_events(task_id):
            await ws.send_json(event)
    """
    client = get_redis()
    if client is None:
        return

    pubsub = client.pubsub()
    try:
        await pubsub.subscribe(_event_channel(task_id))
        async for message in pubsub.listen():
            if message["type"] == "message":
                with contextlib.suppress(Exception):
                    yield json.loads(message["data"])
    except Exception as exc:
        logger.debug("Redis subscribe_events failed: %s", exc)
    finally:
        try:
            await pubsub.unsubscribe(_event_channel(task_id))
            await pubsub.aclose()
        except Exception:
            pass
