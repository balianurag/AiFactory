"""Tests for budget_tracker.py — per-run USD cap enforcement."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aifactory.core.budget_tracker import BudgetExceededError, BudgetTracker


class TestBudgetTracker:
    def test_initial_total_is_zero(self):
        t = BudgetTracker(cap_usd=5.0)
        assert t.total_cost_usd == 0.0

    def test_record_accumulates_cost(self):
        with patch(
            "aifactory.core.budget_tracker.cost_usd_from_registry",
            return_value=0.01,
        ):
            t = BudgetTracker(cap_usd=5.0)
            t.record("gpt-4o", 100, 50)
            t.record("gpt-4o", 100, 50)
            t.record("gpt-4o", 100, 50)
            assert abs(t.total_cost_usd - 0.03) < 1e-9

    def test_check_raises_when_at_cap(self):
        with patch(
            "aifactory.core.budget_tracker.cost_usd_from_registry",
            return_value=5.0,
        ):
            t = BudgetTracker(cap_usd=5.0)
            t.record("gpt-4o", 1000, 500)  # total = 5.0, at cap
            with pytest.raises(BudgetExceededError) as exc_info:
                t.check()
            assert exc_info.value.spent == pytest.approx(5.0)
            assert exc_info.value.cap == 5.0

    def test_check_raises_when_over_cap(self):
        with patch(
            "aifactory.core.budget_tracker.cost_usd_from_registry",
            return_value=6.0,
        ):
            t = BudgetTracker(cap_usd=5.0)
            t.record("gpt-4o", 1000, 500)
            with pytest.raises(BudgetExceededError):
                t.check()

    def test_check_no_raise_under_cap(self):
        with patch(
            "aifactory.core.budget_tracker.cost_usd_from_registry",
            return_value=0.01,
        ):
            t = BudgetTracker(cap_usd=5.0)
            t.record("gpt-4o", 100, 50)
            t.check()  # should not raise

    def test_zero_cap_disables_enforcement(self):
        """cap_usd=0.0 disables enforcement regardless of spend."""
        with patch(
            "aifactory.core.budget_tracker.cost_usd_from_registry",
            return_value=999.0,
        ):
            t = BudgetTracker(cap_usd=0.0)
            t.record("gpt-4o", 1_000_000, 1_000_000)
            t.check()  # must NOT raise

    def test_remaining_usd_decreases(self):
        with patch(
            "aifactory.core.budget_tracker.cost_usd_from_registry",
            return_value=1.0,
        ):
            t = BudgetTracker(cap_usd=5.0)
            assert t.remaining_usd == pytest.approx(5.0)
            t.record("gpt-4o", 100, 50)
            assert t.remaining_usd == pytest.approx(4.0)

    def test_remaining_usd_none_when_cap_disabled(self):
        t = BudgetTracker(cap_usd=0.0)
        assert t.remaining_usd is None

    def test_remaining_usd_never_negative(self):
        with patch(
            "aifactory.core.budget_tracker.cost_usd_from_registry",
            return_value=10.0,
        ):
            t = BudgetTracker(cap_usd=5.0)
            t.record("gpt-4o", 100, 50)
            assert t.remaining_usd == 0.0  # clamped, not negative

    def test_cost_registry_import_error_does_not_raise(self):
        """If cost_usd_from_registry fails, record() is silently ignored."""
        with patch(
            "aifactory.core.budget_tracker.cost_usd_from_registry",
            side_effect=ImportError("no module"),
        ):
            t = BudgetTracker(cap_usd=5.0)
            t.record("unknown-model", 100, 50)  # must not raise
            assert t.total_cost_usd == 0.0

    def test_budget_exceeded_error_message(self):
        err = BudgetExceededError(spent=3.1234, cap=3.0)
        assert "3.1234" in str(err)
        assert "3.00" in str(err)
