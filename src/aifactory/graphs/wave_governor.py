"""
WaveGovernor — dynamic execution controller for the EXTENDED topology.

The WaveGovernor sits between execution_lead and judge_review in the EXTENDED
topology. After each worker wave completes it:

  1. Inspects wave results for quality signals (errors, low-confidence markers,
     REVISION: flags that remained unresolved).
  2. Decides whether to: PROCEED (move to judge_review), ESCALATE (trigger a
     higher-capability wave for failed tasks), or TERMINATE (hard-stop on
     budget/iteration limit).
  3. Enforces second-price floor from the Worker Auction (Phase W4) when auction
     results are present in shared_ctx.

Mode escalation:
  - If current mode is "cost_efficient" and quality < escalate_threshold → escalate
    to "balanced".
  - If current mode is "balanced" and quality < escalate_threshold → escalate to
    "best".
  - "best" mode: no further escalation possible → PROCEED regardless.

The node is a no-op stub in FULL topology (never added to graph); only activated
when topology == EXTENDED (complexity_score > 0.8).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Quality threshold below which mode escalation fires
_DEFAULT_ESCALATE_THRESHOLD = 0.55

# Maximum number of governor retry rounds before forcing PROCEED
_MAX_ROUNDS = 2

_MODE_ESCALATION: dict[str, str] = {
    "cost_efficient": "balanced",
    "balanced": "best",
    "best": "best",  # no-op ceiling
}


@dataclass
class WaveDecision:
    action: str  # "PROCEED" | "ESCALATE" | "TERMINATE"
    reason: str
    escalated_mode: str | None = None
    failed_task_ids: list[str] = field(default_factory=list)
    quality_score: float = 1.0


class WaveGovernor:
    """
    Evaluates wave results and decides whether to proceed, escalate, or terminate.

    Instantiated once per pipeline run in EXTENDED topology; stateful across
    multiple make_wave_governor() invocations within the same run.
    """

    def __init__(
        self,
        *,
        escalate_threshold: float = _DEFAULT_ESCALATE_THRESHOLD,
        max_rounds: int = _MAX_ROUNDS,
        event_callback: Any = None,
    ) -> None:
        self._escalate_threshold = escalate_threshold
        self._max_rounds = max_rounds
        self._event_callback = event_callback
        self._round = 0

    def evaluate(
        self,
        wave_results: list[dict[str, Any]],
        current_mode: str,
        shared_ctx: dict[str, Any],
    ) -> WaveDecision:
        """
        Assess wave results and return a WaveDecision.

        wave_results: list of result_entry dicts from execution_lead worker dispatch.
        current_mode: pipeline mode for this wave ("best" / "balanced" / "cost_efficient").
        shared_ctx: pipeline shared context (may contain auction_floor from W4).
        """
        self._round += 1

        # Hard stop: never exceed max_rounds of re-tries
        if self._round > self._max_rounds:
            logger.info("WaveGovernor: max rounds (%d) reached → PROCEED", self._max_rounds)
            return WaveDecision(
                action="PROCEED",
                reason=f"max_rounds={self._max_rounds} reached",
                quality_score=self._compute_quality(wave_results),
            )

        quality = self._compute_quality(wave_results)
        failed = self._find_failed(wave_results)

        # Apply second-price floor from auction (Phase W4 — optional)
        auction_floor = shared_ctx.get("auction_floor", 0.0)
        if isinstance(auction_floor, (int, float)) and quality < float(auction_floor):
            logger.info(
                "WaveGovernor: quality %.2f below auction floor %.2f → ESCALATE",
                quality,
                auction_floor,
            )
            return self._make_escalate(current_mode, failed, quality, "auction_floor_violation")

        if quality < self._escalate_threshold and current_mode != "best":
            logger.info(
                "WaveGovernor: quality %.2f < threshold %.2f [mode=%s] → ESCALATE",
                quality,
                self._escalate_threshold,
                current_mode,
            )
            return self._make_escalate(current_mode, failed, quality, "low_quality")

        if quality < self._escalate_threshold and current_mode == "best":
            logger.info(
                "WaveGovernor: quality %.2f < threshold but already at best mode → PROCEED",
                quality,
            )

        self._emit_event("governor_decision", {"action": "PROCEED", "quality": quality})
        return WaveDecision(
            action="PROCEED",
            reason="quality_acceptable",
            quality_score=quality,
        )

    def reset(self) -> None:
        self._round = 0

    # ── Internal ──────────────────────────────────────────────────────────────

    def _make_escalate(
        self,
        current_mode: str,
        failed: list[str],
        quality: float,
        reason: str,
    ) -> WaveDecision:
        next_mode = _MODE_ESCALATION.get(current_mode, "best")
        self._emit_event(
            "governor_decision",
            {
                "action": "ESCALATE",
                "from_mode": current_mode,
                "to_mode": next_mode,
                "quality": quality,
            },
        )
        return WaveDecision(
            action="ESCALATE",
            reason=reason,
            escalated_mode=next_mode,
            failed_task_ids=failed,
            quality_score=quality,
        )

    @staticmethod
    def _compute_quality(wave_results: list[dict[str, Any]]) -> float:
        """
        Derive a quality score 0.0-1.0 from wave_results.

        Heuristics (no LLM call — purely structural):
          - Each result with status "error" or "failed" → 0.0 contribution.
          - Results starting with "REVISION:" but not having a corrected follow-up → 0.3.
          - Normal results → 1.0 contribution.
        Score = mean of contributions.
        """
        if not wave_results:
            return 1.0
        scores = []
        for r in wave_results:
            result_text = str(r.get("result", ""))
            if r.get("status") in ("error", "failed"):
                scores.append(0.0)
            elif result_text.strip().upper().startswith("REVISION:"):
                scores.append(0.3)
            elif not result_text.strip():
                scores.append(0.2)
            else:
                scores.append(1.0)
        return sum(scores) / len(scores)

    @staticmethod
    def _find_failed(wave_results: list[dict[str, Any]]) -> list[str]:
        return [
            str(r.get("task_id", ""))
            for r in wave_results
            if r.get("status") in ("error", "failed") or not str(r.get("result", "")).strip()
        ]

    def _emit_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if callable(self._event_callback):
            self._event_callback({"type": event_type, **payload})


# ── LangGraph node factory ────────────────────────────────────────────────────


def make_wave_governor(governor: WaveGovernor, current_mode: str) -> Any:
    """
    Return a LangGraph node coroutine for the WaveGovernor.

    The node reads worker results from state, runs the governor, and writes
    the decision + (optionally) escalated mode back to shared_ctx.

    If the decision is ESCALATE, execution_lead will be re-invoked with
    the escalated mode on the next router call (Phase W5 wiring).
    """

    async def wave_governor_node(state: dict[str, Any]) -> dict[str, Any]:
        shared_ctx = state.get("shared_context", {})
        wave_results = state.get("results", [])

        decision = governor.evaluate(
            wave_results=wave_results,
            current_mode=current_mode,
            shared_ctx=shared_ctx,
        )

        updated_ctx = {
            **shared_ctx,
            "governor_action": decision.action,
            "governor_quality": str(round(decision.quality_score, 3)),
            "governor_reason": decision.reason,
        }
        if decision.escalated_mode:
            updated_ctx["escalated_mode"] = decision.escalated_mode

        logger.info(
            "WaveGovernor node: action=%s quality=%.3f mode=%s",
            decision.action,
            decision.quality_score,
            current_mode,
        )
        return {"shared_context": updated_ctx}

    return wave_governor_node


def route_after_wave_governor(state: dict[str, Any]) -> str:
    """
    Conditional edge router after wave_governor node.

    Returns "execution_lead" (re-run) on ESCALATE, "judge_review" on PROCEED/TERMINATE.
    """
    ctx = state.get("shared_context", {})
    action = ctx.get("governor_action", "PROCEED")
    if action == "ESCALATE":
        return "execution_lead"
    return "judge_review"
