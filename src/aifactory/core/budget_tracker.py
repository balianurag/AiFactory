"""
Per-run USD budget cap enforcement.

BudgetTracker accumulates cost across all PipelineAgent.think() calls in a
single run and raises BudgetExceededError before any call that would push the
total past the configured cap.

Usage in dynamic_pipeline.py:
    tracker = BudgetTracker(cap_usd=settings.max_budget_usd)
    # injected into each PipelineAgent as agent._budget_tracker
    # think() calls tracker.record() after each LLM call
    # think() calls tracker.check() before each LLM call

Disabled when cap_usd <= 0 (default settings.max_budget_usd = 5.00, so budget
enforcement is always ON unless the user explicitly sets it to 0).
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from typing import Any

from aifactory.core.model_selector import cost_usd_from_registry

logger = logging.getLogger(__name__)


class BudgetExceededError(Exception):
    """
    Raised by BudgetTracker.check() when a run has exceeded its USD cap.

    Attributes:
        spent: Total USD spent so far in this run.
        cap:   Configured cap (settings.max_budget_usd).
    """

    def __init__(self, spent: float, cap: float) -> None:
        self.spent = spent
        self.cap = cap
        super().__init__(f"Run budget exceeded: spent ${spent:.4f} of ${cap:.2f} cap")


@dataclass
class BudgetTracker:
    """
    Accumulates LLM call cost per run and enforces a hard USD cap.

    Thread-safety: single-process, single-run use only.  Parallel workers
    each call record() + check() on the same shared tracker instance — the
    check is a simple float comparison so races are benign (worst case: one
    extra call sneaks through before the cap is caught).
    """

    cap_usd: float
    warn_threshold: float = 0.80  # warn when this fraction of cap is consumed
    _total: float = field(default=0.0, init=False, repr=False)
    _warned: bool = field(default=False, init=False, repr=False)

    # Optional callback invoked once when the 80% threshold is crossed.
    # Signature: (spent: float, cap: float) -> None
    on_warn: Any = field(default=None, init=False, repr=False)

    def record(self, model_id: str, prompt_tokens: int, completion_tokens: int) -> None:
        """
        Add the cost of one LLM call to the running total.

        Uses cost_usd_from_registry() — returns 0.0 for unknown models so
        budget tracking degrades gracefully if a new model isn't in the registry.
        Fires a warning callback once when warn_threshold is crossed.
        """
        try:
            call_cost = cost_usd_from_registry(model_id, prompt_tokens, completion_tokens)
        except Exception:
            call_cost = 0.0

        self._total += call_cost
        logger.debug(
            "BudgetTracker: +$%.6f (call) → $%.6f / $%.2f cap  [%s]",
            call_cost,
            self._total,
            self.cap_usd,
            model_id,
        )

        # 80% warning — fire once per run
        if (
            not self._warned
            and self.cap_usd > 0
            and self._total >= self.cap_usd * self.warn_threshold
        ):
            self._warned = True
            logger.warning(
                "BudgetTracker: %.0f%% of $%.2f cap consumed ($%.4f spent)",
                self.warn_threshold * 100,
                self.cap_usd,
                self._total,
            )
            if callable(self.on_warn):
                with contextlib.suppress(Exception):
                    self.on_warn(self._total, self.cap_usd)

    def check(self) -> None:
        """
        Raise BudgetExceededError if the accumulated cost is at or above cap.

        No-op when cap_usd <= 0 (budget enforcement disabled).
        """
        if self.cap_usd > 0 and self._total >= self.cap_usd:
            raise BudgetExceededError(self._total, self.cap_usd)

    @property
    def total_cost_usd(self) -> float:
        """Running total USD cost for this run."""
        return self._total

    @property
    def remaining_usd(self) -> float | None:
        """Remaining budget, or None if enforcement is disabled."""
        if self.cap_usd <= 0:
            return None
        return max(0.0, self.cap_usd - self._total)
