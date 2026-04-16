"""
Memory Bridge — fail-safe adapter between PipelineAgent and MemoryRouter.

Connects the 3-layer memory system (Agent → Team → Global) to the production
PipelineAgent so agents accumulate knowledge across runs.

Responsibilities:
  - Retrieve relevant past memories before an agent thinks (inject as context)
  - Store agent outputs after thinking (agent layer)
  - Store SHARE: updates to team layer (visible to all agents in the run)
  - Trigger memory promotion when the turn threshold is reached

Design principles:
  - All methods are fail-safe: exceptions are caught and logged, never raised.
    A memory failure must NEVER break the pipeline.
  - Zero-cost when router=None (noop in every method). Pass router=None in tests
    that don't care about memory, or set AIFACTORY_MEMORY_PIPELINE_ENABLED=false.
  - Token-aware: injected context is capped at max_context_chars to prevent
    system prompt bloat on long-running agents.
  - run_id isolation: each pipeline run uses run_id as team_id, so agent-layer
    memories are namespaced and team-layer memories stay within the same run.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aifactory.core.memory_router import MemoryRouter
    from aifactory.core.schemas import MemoryEntry

logger = logging.getLogger(__name__)

# Default cap for injected memory context (chars, not tokens).
# ~800 chars ≈ 200 tokens — enough to be useful without inflating prompts.
_DEFAULT_MAX_CONTEXT_CHARS = 800

# Section header used in system prompt injection
_MEMORY_SECTION_HEADER = "\n\n## Relevant Memory\n"


class MemoryBridge:
    """
    Thin adapter between PipelineAgent and MemoryRouter.

    Usage in PipelineAgent:
        # Before LLM call:
        ctx = bridge.retrieve_context(agent_id="analyser", query=user_msg)
        # After LLM reply:
        bridge.store_output(agent_id="analyser", content=reply, turn_number=turn)
        bridge.maybe_promote(agent_id="analyser", summary=compact_summary)

    All methods are safe to call even when router=None — they become noops.
    """

    def __init__(
        self,
        router: MemoryRouter | None,
        run_id: str = "",
        max_context_chars: int = _DEFAULT_MAX_CONTEXT_CHARS,
    ) -> None:
        self._router = router
        self._run_id = run_id
        self._max_context_chars = max_context_chars

    @property
    def enabled(self) -> bool:
        """True when a real router is attached."""
        return self._router is not None

    # ─── Retrieval ────────────────────────────────────────────────────

    def retrieve_context(
        self,
        agent_id: str,
        query: str,
        top_k: int = 3,
    ) -> str:
        """
        Retrieve the most relevant memories for an agent and format them as
        a compact text block for injection into the system prompt.

        Searches agent-private layer first, then team layer (not global —
        too broad for per-call injection).

        Returns "" on any failure or when router=None.
        """
        if self._router is None or not query.strip():
            return ""
        try:
            from aifactory.core.schemas import MemoryLevel

            entries = self._router.retrieve(
                query=query,
                levels=[MemoryLevel.AGENT, MemoryLevel.TEAM],
                agent_id=agent_id,
                team_id=self._run_id,
                top_k=top_k,
            )
            if not entries:
                return ""
            return self._format_context(entries)
        except Exception as exc:
            logger.debug("[MemoryBridge] retrieve failed for %s: %s", agent_id, exc)
            return ""

    def _format_context(self, entries: list[MemoryEntry]) -> str:
        """
        Format retrieved memory entries as a compact block.

        Each entry is labelled with its layer so agents understand provenance.
        Total output is capped at max_context_chars.
        """
        lines: list[str] = []
        total_chars = 0

        for entry in entries:
            snippet = entry.content.strip()
            if not snippet:
                continue
            label = f"[{entry.level.value}]"
            line = f"{label} {snippet}"
            remaining = self._max_context_chars - total_chars
            if remaining <= 0:
                break
            if len(line) > remaining:
                # Only include if we can fit at least 80 meaningful chars
                if remaining > 80:
                    lines.append(line[:remaining] + "…")
                break
            lines.append(line)
            total_chars += len(line) + 1  # +1 for newline

        if not lines:
            return ""
        return _MEMORY_SECTION_HEADER + "\n".join(lines)

    # ─── Storage ──────────────────────────────────────────────────────

    def store_output(
        self,
        agent_id: str,
        content: str,
        turn_number: int = 0,
        entry_type: str = "agent_output",
    ) -> None:
        """
        Store agent output in the agent-private memory layer.

        Called after each think() call. Ignores empty content. Fail-safe.
        """
        if self._router is None or not content.strip():
            return
        try:
            from aifactory.core.schemas import MemoryEntry, MemoryLevel

            entry = MemoryEntry(
                level=MemoryLevel.AGENT,
                agent_id=agent_id,
                team_id=self._run_id,
                content=content,
                entry_type=entry_type,
                turn_number=turn_number,
                promoted=False,
            )
            self._router.store(entry)
            logger.debug(
                "[MemoryBridge] stored %s/%s output (%d chars)",
                agent_id,
                entry_type,
                len(content),
            )
        except Exception as exc:
            logger.debug("[MemoryBridge] store failed for %s: %s", agent_id, exc)

    def store_share_update(
        self,
        from_agent: str,
        key: str,
        value: str,
    ) -> None:
        """
        Store a single SHARE: key=value in the team memory layer.

        Team-layer entries are visible to all agents in the same run (same
        team_id = run_id). This makes important pipeline decisions retrievable
        by downstream agents and persists them for future runs.
        """
        if self._router is None or not value.strip():
            return
        try:
            from aifactory.core.schemas import MemoryEntry, MemoryLevel

            content = f"{key}: {value}  (shared by {from_agent})"
            entry = MemoryEntry(
                level=MemoryLevel.TEAM,
                agent_id=from_agent,
                team_id=self._run_id,
                content=content,
                entry_type="share_update",
                promoted=False,
            )
            self._router.store(entry)
            logger.debug(
                "[MemoryBridge] stored SHARE: %s=%r from %s",
                key,
                value[:60],
                from_agent,
            )
        except Exception as exc:
            logger.debug(
                "[MemoryBridge] store_share failed (%s=%s): %s",
                key,
                value[:30],
                exc,
            )

    def store_share_updates(
        self,
        from_agent: str,
        updates: dict[str, str],
    ) -> None:
        """Store all SHARE: updates from one agent to the team memory layer."""
        for key, value in updates.items():
            self.store_share_update(from_agent, key, value)

    def retrieve_global_context(
        self,
        query: str,
        top_k: int = 2,
    ) -> str:
        """
        Query the global (cross-run) memory layer for relevant knowledge.

        Called at pipeline start to seed shared context with proven solutions,
        research findings, and patterns from prior runs.  Only the global layer
        is searched — agent/team layers are scoped to the current run.

        Returns "" on any failure or when router=None.
        """
        if self._router is None or not query.strip():
            return ""
        try:
            from aifactory.core.schemas import MemoryLevel

            entries = self._router.retrieve(
                query=query,
                levels=[MemoryLevel.GLOBAL],
                top_k=top_k,
            )
            if not entries:
                return ""
            return self._format_context(entries)
        except Exception as exc:
            logger.debug("[MemoryBridge] retrieve_global failed: %s", exc)
            return ""

    def promote_to_global(
        self,
        agent_id: str,
        summary: str,
        run_quality: float = 10.0,
    ) -> None:
        """
        Promote a high-quality run summary to the global cross-run memory layer.

        Only promotes when run_quality >= settings.rl_min_score_for_promotion
        (default 6.0).  Called once at the end of a successful pipeline run to
        build long-term memory across runs.  Fail-safe — never raises.
        """
        if self._router is None or not summary.strip():
            return
        try:
            from aifactory.config.settings import get_settings as _gs

            _min_quality = _gs().rl_min_score_for_promotion
        except Exception:
            _min_quality = 6.0

        if run_quality < _min_quality:
            logger.debug(
                "[MemoryBridge] skip global promotion (quality=%.1f < threshold=%.1f)",
                run_quality,
                _min_quality,
            )
            return
        try:
            from aifactory.core.schemas import MemoryLevel

            self._router.promote(
                agent_id=agent_id,
                team_id=self._run_id,
                summary=summary,
                from_level=MemoryLevel.TEAM,
                to_level=MemoryLevel.GLOBAL,
            )
            logger.info(
                "[MemoryBridge] promoted run summary → global layer (quality=%.1f, run=%s)",
                run_quality,
                self._run_id,
            )
        except Exception as exc:
            logger.debug("[MemoryBridge] promote_to_global failed: %s", exc)

    # ─── Promotion ────────────────────────────────────────────────────

    def maybe_promote(
        self,
        agent_id: str,
        summary: str,
    ) -> None:
        """
        Promote agent-layer memories to team layer when the threshold is reached.

        The MemoryRouter tracks unpromoted entries per agent_id. When the count
        reaches memory_promotion_interval (default 10), this triggers a promotion
        where the provided summary is stored at team level and all agent-layer
        entries are marked as promoted.

        Safe to call after every think() — does nothing if below threshold.
        """
        if self._router is None or not summary.strip():
            return
        try:
            if self._router.should_promote(agent_id):
                from aifactory.core.schemas import MemoryLevel

                self._router.promote(
                    agent_id=agent_id,
                    team_id=self._run_id,
                    summary=summary,
                    from_level=MemoryLevel.AGENT,
                    to_level=MemoryLevel.TEAM,
                )
                logger.info(
                    "[MemoryBridge] promoted %s memories → team layer (run=%s)",
                    agent_id,
                    self._run_id,
                )
        except Exception as exc:
            logger.debug("[MemoryBridge] promote failed for %s: %s", agent_id, exc)
