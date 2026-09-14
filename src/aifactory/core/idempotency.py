"""
Idempotency-Key support for run-creation endpoints.

Stripe-style semantics (RFC 9457 inspired):

    On a POST that creates a run:
      1. If `Idempotency-Key` header is absent → request is processed normally.
      2. If present and unseen → response is recorded keyed by the value.
      3. If present and seen with the *same* request body → the cached
         response (run_id, status, body snapshot) is returned with HTTP 200.
      4. If present and seen with a *different* request body → HTTP 422.

Storage:
  - Redis (preferred) when REDIS_URL is configured.  TTL = 24h.
  - In-process dict fallback when Redis is unavailable — emits a single
    warning on first use, preserves correctness for a single worker
    process (no cross-replica safety in fallback mode).

Design notes:
  - Body hash is SHA-256 of the canonicalised JSON request body so trivial
    key reordering or whitespace does not trigger a 422 collision.
  - Methods are async; the Redis client matches the rest of the API surface.
  - The module deliberately exposes a small object-oriented surface so tests
    can build a clean in-memory IdempotencyStore without monkey-patching.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# 24h TTL — matches Stripe's documented retention window.
DEFAULT_TTL_SECONDS = 24 * 60 * 60

_REDIS_KEY_PREFIX = "aifactory:idempotency:"

# Public header name — exported so routes can reference a single source of truth.
HEADER_NAME = "Idempotency-Key"


@dataclass(slots=True)
class IdempotencyRecord:
    """A stored idempotency entry."""

    key: str
    body_hash: str
    run_id: str
    status: str
    response_snapshot: dict[str, Any]
    stored_at: float

    def to_json(self) -> str:
        return json.dumps(
            {
                "key": self.key,
                "body_hash": self.body_hash,
                "run_id": self.run_id,
                "status": self.status,
                "response_snapshot": self.response_snapshot,
                "stored_at": self.stored_at,
            },
            default=str,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, raw: str) -> IdempotencyRecord:
        data = json.loads(raw)
        return cls(
            key=data["key"],
            body_hash=data["body_hash"],
            run_id=data["run_id"],
            status=data["status"],
            response_snapshot=data.get("response_snapshot") or {},
            stored_at=float(data.get("stored_at") or 0.0),
        )


class IdempotencyConflict(Exception):  # noqa: N818 — public API name; add-Error would break callers
    """Raised when a key is reused with a different request body."""

    def __init__(self, key: str, existing_body_hash: str, incoming_body_hash: str) -> None:
        super().__init__(
            f"Idempotency-Key '{key}' was previously used with a different "
            f"request body (existing={existing_body_hash[:12]}…, "
            f"new={incoming_body_hash[:12]}…)"
        )
        self.key = key
        self.existing_body_hash = existing_body_hash
        self.incoming_body_hash = incoming_body_hash


def canonical_body_hash(body: dict[str, Any]) -> str:
    """SHA-256 of a JSON body, canonicalised so key order does not matter."""
    serialised = json.dumps(body, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(serialised.encode()).hexdigest()


class IdempotencyStore:
    """
    Async-safe idempotency cache with Redis primary + in-memory fallback.

    Lazy-binds the Redis client so tests can construct the store without a
    live Redis connection.  All operations are wrapped in try/except — a
    Redis blip never breaks the API path.
    """

    def __init__(self, *, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        self.ttl_seconds = ttl_seconds
        self._memory: dict[str, IdempotencyRecord] = {}
        self._memory_lock = asyncio.Lock()
        self._warned_fallback = False

    # ── Public API ─────────────────────────────────────────────────────────

    async def get(self, key: str) -> IdempotencyRecord | None:
        """Return the stored record for *key* or None on miss / expiry."""
        if not key:
            return None

        # Redis path
        redis = await self._maybe_redis()
        if redis is not None:
            try:
                raw = await redis.get(_REDIS_KEY_PREFIX + key)
                if raw:
                    return IdempotencyRecord.from_json(raw)
            except Exception as exc:
                logger.debug("Idempotency: Redis get failed (%s) — using memory", exc)

        # Memory fallback
        async with self._memory_lock:
            record = self._memory.get(key)
            if record is None:
                return None
            if self.ttl_seconds > 0 and (time.time() - record.stored_at) > self.ttl_seconds:
                self._memory.pop(key, None)
                return None
            return record

    async def put(self, record: IdempotencyRecord) -> None:
        """Persist a record under its key."""
        if not record.key:
            return

        redis = await self._maybe_redis()
        if redis is not None:
            try:
                await redis.setex(
                    _REDIS_KEY_PREFIX + record.key,
                    self.ttl_seconds,
                    record.to_json(),
                )
                return
            except Exception as exc:
                logger.debug("Idempotency: Redis set failed (%s) — using memory", exc)

        # Memory fallback
        self._warn_fallback_once()
        async with self._memory_lock:
            self._memory[record.key] = record

    async def reserve_or_replay(
        self,
        *,
        key: str,
        body: dict[str, Any],
    ) -> IdempotencyRecord | None:
        """
        Look up an existing record for *key*.

        Returns:
            None if the key is unseen (caller should process the request and
                then store the response via store_response()).
            IdempotencyRecord  if the key is a duplicate of a prior request
                with the *same* body — caller should return it.

        Raises:
            IdempotencyConflict if the key matches a different body.
        """
        if not key:
            return None

        incoming_hash = canonical_body_hash(body)
        existing = await self.get(key)
        if existing is None:
            return None
        if existing.body_hash != incoming_hash:
            raise IdempotencyConflict(key, existing.body_hash, incoming_hash)
        return existing

    async def store_response(
        self,
        *,
        key: str,
        body: dict[str, Any],
        run_id: str,
        status: str,
        response_snapshot: dict[str, Any],
    ) -> IdempotencyRecord:
        """Persist the outcome of a fresh request under *key*."""
        record = IdempotencyRecord(
            key=key,
            body_hash=canonical_body_hash(body),
            run_id=run_id,
            status=status,
            response_snapshot=response_snapshot,
            stored_at=time.time(),
        )
        await self.put(record)
        return record

    async def clear(self) -> None:
        """Drop all in-memory entries (testing helper)."""
        async with self._memory_lock:
            self._memory.clear()

    # ── Internal helpers ──────────────────────────────────────────────────

    async def _maybe_redis(self) -> Any | None:
        """Return the shared async Redis client, or None when unavailable."""
        try:
            from aifactory.core import redis_client

            client = redis_client.get_redis()
            return client
        except Exception as exc:
            logger.debug("Idempotency: Redis client unavailable (%s)", exc)
            return None

    def _warn_fallback_once(self) -> None:
        if self._warned_fallback:
            return
        self._warned_fallback = True
        logger.warning(
            "Idempotency: Redis unavailable — using in-process memory store. "
            "Duplicate detection will NOT survive process restarts or work across replicas."
        )


# ── Module-level singleton ────────────────────────────────────────────────────

_store: IdempotencyStore | None = None


def get_idempotency_store() -> IdempotencyStore:
    """Lazy singleton — shared across the FastAPI app."""
    global _store
    if _store is None:
        _store = IdempotencyStore()
    return _store


def reset_idempotency_store() -> None:
    """Reset the singleton (testing only)."""
    global _store
    _store = None
