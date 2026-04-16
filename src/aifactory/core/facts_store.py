"""
Facts Store — JSON-backed filesystem persistence for REMEMBER: facts.

Ensures cross-run REMEMBER: facts survive process restarts even when Qdrant
is running in in-memory mode (the default for local dev without a Qdrant server).

Usage in MemoryBridge:
  - store_remember_facts() writes to BOTH Qdrant (if available) AND this file.
  - retrieve_global_context() seeds from this file when Qdrant returns nothing.

Persistence: ~/.aifactory/memories/facts.json (one JSON array of entry dicts).
Override with AIFACTORY_MEMORIES_DIR env var or by passing store_dir.

Design: all operations are fail-safe — exceptions are caught and logged, never raised.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_DIR = Path(os.environ.get("AIFACTORY_MEMORIES_DIR", "~/.aifactory/memories"))


class FactsStore:
    """
    Simple JSON-file fact store for REMEMBER: protocol persistence.

    Facts are stored as a list of dicts with keys:
      ``content``, ``from_agent``, ``run_id``, ``timestamp``.

    Deduplication: exact content-string match prevents duplicate entries.
    Fail-safe: every method catches and logs exceptions, never raises.
    """

    def __init__(self, store_dir: Path | None = None) -> None:
        self._dir = (store_dir or _DEFAULT_DIR).expanduser()
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.debug("FactsStore: could not create store dir %s: %s", self._dir, exc)
        self._path = self._dir / "facts.json"

    # ── Internal I/O ──────────────────────────────────────────────────

    def _load(self) -> list[dict[str, Any]]:
        try:
            if not self._path.exists():
                return []
            raw = self._path.read_text(encoding="utf-8")
            data = json.loads(raw) if raw.strip() else []
            # Tolerate both list and legacy dict format from experiments
            if isinstance(data, dict):
                return list(data.values())
            return data if isinstance(data, list) else []
        except Exception as exc:
            logger.debug("FactsStore: load failed: %s", exc)
            return []

    def _save(self, facts: list[dict[str, Any]]) -> None:
        try:
            self._path.write_text(json.dumps(facts, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            logger.debug("FactsStore: save failed: %s", exc)

    # ── Public API ────────────────────────────────────────────────────

    def store(self, from_agent: str, fact: str, run_id: str = "") -> None:
        """
        Persist a single REMEMBER: fact.

        Silently deduplicates by content string — the same fact is never
        stored twice even across multiple pipeline runs.

        Args:
            from_agent: Name of the worker agent that emitted the fact.
            fact:       The fact string (one REMEMBER: bullet line).
            run_id:     Pipeline run ID for provenance.
        """
        if not fact.strip():
            return
        try:
            facts = self._load()
            existing = {f.get("content", "").strip() for f in facts}
            if fact.strip() in existing:
                return
            facts.append(
                {
                    "content": fact.strip(),
                    "from_agent": from_agent,
                    "run_id": run_id,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )
            self._save(facts)
            logger.debug("FactsStore: stored fact from %s: %s", from_agent, fact[:80])
        except Exception as exc:
            logger.debug("FactsStore: store error: %s", exc)

    def load_all(self) -> list[str]:
        """
        Return all stored fact content strings.

        Used by MemoryBridge to seed Qdrant global layer and inject into
        shared context when Qdrant returns no results.
        """
        try:
            return [f["content"] for f in self._load() if f.get("content")]
        except Exception as exc:
            logger.debug("FactsStore: load_all error: %s", exc)
            return []

    def size(self) -> int:
        """Number of stored facts."""
        return len(self._load())

    def clear(self) -> None:
        """Remove all stored facts."""
        self._save([])
        logger.debug("FactsStore: cleared")
