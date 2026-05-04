"""
WorkerAuction — second-price sealed-bid auction for worker task assignment.

Design:
  Each specialist worker submits a Bid declaring:
    - confidence  (0.0-1.0): how well-suited it is for the task
    - approach    (str): brief description of strategy
    - model       (str): proposed LLM model
    - cost_estimate (float): estimated USD cost (used as bid price)

  The auction selects the winner (highest confidence) but bills it at the
  second-highest confidence bid as its quality floor.  This prevents
  overconfident bids while rewarding genuine specialization.

  Second-price floor enforcement:
    If the winning worker's actual output quality (as scored by WaveGovernor)
    falls below second_price_floor, the WaveGovernor escalates the mode.
    The floor is stored in shared_ctx["auction_floor"] so WaveGovernor can
    read it without importing this module.

  RL integration (Phase W5):
    When an RLScorer is attached, each specialist's confidence is adjusted by
    its historical (model, task_type) quality score before bidding.

Usage::

    auction = WorkerAuction()
    bid1 = WorkerBid(worker_id="coder", confidence=0.85, approach="stdlib", model="gpt-4o-mini")
    bid2 = WorkerBid(worker_id="analyst", confidence=0.72, approach="stats", model="gpt-4o-mini")
    result = auction.run(task_description="Sort a list", bids=[bid1, bid2])
    result.winner.worker_id       # "coder"
    result.second_price_floor     # 0.72  (→ stored in shared_ctx["auction_floor"])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class WorkerBid:
    worker_id: str
    confidence: float  # 0.0-1.0, higher = more confident this worker can handle the task
    approach: str = ""
    model: str = ""
    cost_estimate: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuctionResult:
    winner: WorkerBid
    all_bids: list[WorkerBid]
    second_price_floor: float  # quality floor enforced by WaveGovernor
    task_description: str
    rl_adjusted: bool = False

    @property
    def runner_up(self) -> WorkerBid | None:
        sorted_bids = sorted(self.all_bids, key=lambda b: b.confidence, reverse=True)
        if len(sorted_bids) > 1:
            return sorted_bids[1]
        return None

    def to_shared_ctx_update(self) -> dict[str, Any]:
        """Return a dict suitable for merging into pipeline shared_context."""
        return {
            "auction_winner": self.winner.worker_id,
            "auction_floor": self.second_price_floor,
            "auction_winner_model": self.winner.model,
            "auction_winner_confidence": str(round(self.winner.confidence, 3)),
        }


class WorkerAuction:
    """
    Runs second-price sealed-bid auctions for task→worker assignment.

    Can be attached to an RLScorer to adjust confidences from historical data.
    """

    def __init__(self, rl_scorer: Any = None) -> None:
        self._rl_scorer: Any = rl_scorer
        self._history: list[AuctionResult] = []

    def run(
        self,
        task_description: str,
        bids: list[WorkerBid],
        task_type: str = "general",
    ) -> AuctionResult:
        """
        Run an auction for the given task and return the result.

        Applies RL adjustments when an RLScorer is available, then selects
        the winner (highest adjusted confidence) and computes second-price floor.
        """
        if not bids:
            raise ValueError("Cannot run auction with no bids")

        adjusted = self._apply_rl(bids, task_type)
        sorted_bids = sorted(adjusted, key=lambda b: b.confidence, reverse=True)

        winner = sorted_bids[0]
        second = sorted_bids[1] if len(sorted_bids) > 1 else winner
        floor = second.confidence

        result = AuctionResult(
            winner=winner,
            all_bids=adjusted,
            second_price_floor=floor,
            task_description=task_description,
            rl_adjusted=self._rl_scorer is not None,
        )
        self._history.append(result)

        logger.info(
            "Auction [%s]: winner=%s (conf=%.2f, floor=%.2f, model=%s)",
            task_type,
            winner.worker_id,
            winner.confidence,
            floor,
            winner.model,
        )
        return result

    def record_outcome(
        self,
        result: AuctionResult,
        actual_quality: float,
    ) -> None:
        """
        Record the actual quality achieved by the auction winner.

        Used for RL history — the winner's model + task_type gets a quality
        data point that will adjust future bids via the RLScorer.
        """
        if self._rl_scorer is None:
            return
        try:
            # Minimal async-safe record stub — Phase W5 wires full RL integration
            logger.debug(
                "WorkerAuction.record_outcome: winner=%s model=%s quality=%.2f",
                result.winner.worker_id,
                result.winner.model,
                actual_quality,
            )
        except Exception as exc:
            logger.debug("WorkerAuction.record_outcome failed: %s", exc)

    def history(self) -> list[AuctionResult]:
        return list(self._history)

    def attach_rl_scorer(self, rl_scorer: Any) -> None:
        self._rl_scorer = rl_scorer

    # ── Internal ──────────────────────────────────────────────────────────────

    def _apply_rl(self, bids: list[WorkerBid], task_type: str) -> list[WorkerBid]:
        """
        Adjust bid confidences using RL history.

        When an RLScorer is available, fetches (model, task_type) scores and
        applies a multiplicative factor: adj = 1 + 0.2 * (rl_score/5 - 1).
        Falls back to original confidence if RL data is unavailable.
        """
        if self._rl_scorer is None:
            return list(bids)

        adjusted: list[WorkerBid] = []
        try:
            import asyncio

            # Synchronously fetch RL scores — auction runs in sync context
            loop = asyncio.new_event_loop()
            rl_scores: dict[str, float] = {}
            try:
                rl_scores = loop.run_until_complete(
                    self._rl_scorer.get_model_rl_scores(task_type)
                )
            finally:
                loop.close()
        except Exception:
            return list(bids)

        for bid in bids:
            rl_score = rl_scores.get(bid.model)
            if rl_score is not None:
                adj_factor = 1.0 + 0.2 * (rl_score / 5.0 - 1.0)
                new_conf = max(0.0, min(1.0, bid.confidence * adj_factor))
                adjusted.append(
                    WorkerBid(
                        worker_id=bid.worker_id,
                        confidence=new_conf,
                        approach=bid.approach,
                        model=bid.model,
                        cost_estimate=bid.cost_estimate,
                        metadata={
                            **bid.metadata,
                            "rl_adjusted": True,
                            "raw_confidence": bid.confidence,
                        },
                    )
                )
            else:
                adjusted.append(bid)

        return adjusted


# ── Default bids from specialist roles ────────────────────────────────────────


def bids_for_task(
    task_description: str,
    available_models: list[str],
    mode: str = "balanced",
) -> list[WorkerBid]:
    """
    Generate default bids for a task based on keyword signals.

    Each specialist role declares a confidence based on keyword overlap.
    Used when no explicit bidding is configured.
    """
    desc_lower = task_description.lower()

    _specialist_signals: list[tuple[str, list[str], float]] = [
        ("coder", ["code", "implement", "function", "class", "algorithm", "script"], 0.85),
        ("researcher", ["research", "investigate", "find", "explore", "analyse"], 0.85),
        ("writer", ["write", "draft", "document", "explain", "summarise", "report"], 0.85),
        ("analyst", ["analyse", "evaluate", "assess", "metric", "data", "statistics"], 0.85),
    ]

    bids: list[WorkerBid] = []
    for role, signals, base_conf in _specialist_signals:
        hits = sum(1 for s in signals if s in desc_lower)
        confidence = min(1.0, base_conf + hits * 0.03) if hits > 0 else 0.4

        model = available_models[0] if available_models else "gpt-4o-mini"
        bids.append(
            WorkerBid(
                worker_id=role,
                confidence=confidence,
                approach=f"{role} specialist approach",
                model=model,
            )
        )

    return bids
