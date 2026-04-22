"""
Facts Store — persistent storage for REMEMBER: facts.

Primary backend: PostgreSQL `facts` table (multi-process safe, survives restarts).
Fallback: JSON file at ~/.aifactory/memories/facts.json (single-process, dev).

REMEMBER: facts written by one worker in `aifactory serve --workers 4` are
immediately visible to all other workers because they share the PostgreSQL store.

Usage in MemoryBridge:
  - store_remember_facts() calls store() — writes to PG or file
  - retrieve_global_context() calls load_all() — reads from PG or file

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
    Persistent store for REMEMBER: protocol facts.

    PostgreSQL is the primary backend when DATABASE_URL is set.
    Falls back to a local JSON file for dev / offline use.

    Facts are stored with unique constraint on content — duplicates are silently
    ignored in both backends.
    """

    def __init__(self, store_dir: Path | None = None) -> None:
        self._dir = (store_dir or _DEFAULT_DIR).expanduser()
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.debug("FactsStore: could not create store dir %s: %s", self._dir, exc)
        self._path = self._dir / "facts.json"

        # PostgreSQL availability flag
        self._pg: bool = False
        try:
            from aifactory.core.database import get_sync_engine
            if get_sync_engine() is not None:
                self._pg = True
        except Exception:
            pass

    # ── Public API ────────────────────────────────────────────────────

    def store(self, from_agent: str, fact: str, run_id: str = "") -> None:
        """
        Persist a single REMEMBER: fact.

        Silently deduplicates by content — the same fact is never stored twice.

        Args:
            from_agent: Name of the worker agent that emitted the fact.
            fact:       The fact string (one REMEMBER: bullet line).
            run_id:     Pipeline run ID for provenance.
        """
        if not fact.strip():
            return

        if self._pg:
            try:
                from aifactory.core.database import upsert_fact
                upsert_fact(fact.strip(), from_agent=from_agent, run_id=run_id)
                logger.debug("FactsStore: stored fact via PG from %s: %s", from_agent, fact[:80])
                return
            except Exception as exc:
                logger.debug("FactsStore: PG write failed — falling back to file: %s", exc)

        # JSON file fallback
        self._store_file(from_agent, fact.strip(), run_id)

    def load_all(self) -> list[str]:
        """
        Return all stored fact content strings.

        Used by MemoryBridge to seed the Qdrant global layer and inject
        into shared context when Qdrant returns no results.
        """
        if self._pg:
            try:
                from aifactory.core.database import load_all_facts
                facts = load_all_facts()
                if facts:
                    return facts
                # Fall through to file if PG returned nothing (might be fresh DB)
            except Exception as exc:
                logger.debug("FactsStore: PG load failed — falling back to file: %s", exc)

        return self._load_file_all()

    def size(self) -> int:
        """Number of stored facts."""
        if self._pg:
            try:
                from aifactory.core.database import count_facts
                return count_facts()
            except Exception:
                pass
        return len(self._load_file())

    def clear(self) -> None:
        """Remove all stored facts from both backends."""
        if self._pg:
            try:
                from aifactory.core.database import clear_facts
                clear_facts()
                logger.debug("FactsStore: cleared PG facts")
            except Exception as exc:
                logger.debug("FactsStore: PG clear failed: %s", exc)
        # Always clear the local file too
        self._save_file([])
        logger.debug("FactsStore: cleared local facts file")

    # ── JSON file fallback ────────────────────────────────────────────

    def _load_file(self) -> list[dict[str, Any]]:
        try:
            if not self._path.exists():
                return []
            raw = self._path.read_text(encoding="utf-8")
            data = json.loads(raw) if raw.strip() else []
            if isinstance(data, dict):
                return list(data.values())
            return data if isinstance(data, list) else []
        except Exception as exc:
            logger.debug("FactsStore: file load failed: %s", exc)
            return []

    def _load_file_all(self) -> list[str]:
        try:
            return [f["content"] for f in self._load_file() if f.get("content")]
        except Exception as exc:
            logger.debug("FactsStore: load_all file error: %s", exc)
            return []

    def _save_file(self, facts: list[dict[str, Any]]) -> None:
        try:
            self._path.write_text(
                json.dumps(facts, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:
            logger.debug("FactsStore: file save failed: %s", exc)

    def _store_file(self, from_agent: str, fact: str, run_id: str) -> None:
        try:
            facts = self._load_file()
            existing = {f.get("content", "").strip() for f in facts}
            if fact in existing:
                return
            facts.append({
                "content": fact,
                "from_agent": from_agent,
                "run_id": run_id,
                "timestamp": datetime.now(UTC).isoformat(),
            })
            self._save_file(facts)
            logger.debug("FactsStore: stored fact via file from %s: %s", from_agent, fact[:80])
        except Exception as exc:
            logger.debug("FactsStore: file store error: %s", exc)
