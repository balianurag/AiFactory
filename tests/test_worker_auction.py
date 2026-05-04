"""Tests for WorkerAuction (Week 4 — Adaptive Mesh Pipeline)."""

from __future__ import annotations

import pytest

from aifactory.core.worker_auction import (
    AuctionResult,
    WorkerAuction,
    WorkerBid,
    bids_for_task,
)


def _bid(worker_id: str, confidence: float, model: str = "gpt-4o-mini") -> WorkerBid:
    return WorkerBid(worker_id=worker_id, confidence=confidence, model=model)


# ── WorkerBid ─────────────────────────────────────────────────────────────────


class TestWorkerBid:
    def test_defaults(self):
        bid = WorkerBid(worker_id="coder", confidence=0.8)
        assert bid.model == ""
        assert bid.cost_estimate == 0.0
        assert bid.metadata == {}

    def test_fields(self):
        bid = WorkerBid(worker_id="coder", confidence=0.75, approach="stdlib", model="gpt-4o")
        assert bid.worker_id == "coder"
        assert bid.approach == "stdlib"


# ── AuctionResult ─────────────────────────────────────────────────────────────


class TestAuctionResult:
    def _make_result(self) -> AuctionResult:
        bids = [_bid("coder", 0.85), _bid("analyst", 0.72), _bid("writer", 0.60)]
        auction = WorkerAuction()
        return auction.run("implement a sort function", bids)

    def test_winner_has_highest_confidence(self):
        result = self._make_result()
        assert result.winner.worker_id == "coder"

    def test_second_price_floor_is_runner_up(self):
        result = self._make_result()
        assert result.second_price_floor == pytest.approx(0.72)

    def test_runner_up_is_second_highest(self):
        result = self._make_result()
        assert result.runner_up is not None
        assert result.runner_up.worker_id == "analyst"

    def test_to_shared_ctx_update(self):
        result = self._make_result()
        ctx = result.to_shared_ctx_update()
        assert ctx["auction_winner"] == "coder"
        assert "auction_floor" in ctx
        assert float(ctx["auction_floor"]) == pytest.approx(0.72)

    def test_runner_up_none_for_single_bid(self):
        auction = WorkerAuction()
        result = auction.run("task", [_bid("coder", 0.8)])
        assert result.runner_up is None


# ── WorkerAuction.run ─────────────────────────────────────────────────────────


class TestWorkerAuctionRun:
    def test_raises_on_empty_bids(self):
        auction = WorkerAuction()
        with pytest.raises(ValueError, match="no bids"):
            auction.run("task", [])

    def test_single_bid_wins_with_own_floor(self):
        auction = WorkerAuction()
        result = auction.run("task", [_bid("only", 0.7)])
        assert result.winner.worker_id == "only"
        assert result.second_price_floor == pytest.approx(0.7)

    def test_two_bids_second_price_correct(self):
        auction = WorkerAuction()
        result = auction.run("task", [_bid("a", 0.9), _bid("b", 0.6)])
        assert result.winner.worker_id == "a"
        assert result.second_price_floor == pytest.approx(0.6)

    def test_history_records_results(self):
        auction = WorkerAuction()
        auction.run("task1", [_bid("coder", 0.8), _bid("writer", 0.5)])
        auction.run("task2", [_bid("coder", 0.9)])
        assert len(auction.history()) == 2

    def test_history_returns_copy(self):
        auction = WorkerAuction()
        auction.run("task", [_bid("a", 0.8)])
        h = auction.history()
        h.clear()
        assert len(auction.history()) == 1  # original unchanged

    def test_task_description_stored(self):
        auction = WorkerAuction()
        result = auction.run("implement a parser", [_bid("coder", 0.8)])
        assert "parser" in result.task_description

    def test_all_bids_in_result(self):
        bids = [_bid("a", 0.9), _bid("b", 0.7), _bid("c", 0.5)]
        auction = WorkerAuction()
        result = auction.run("task", bids)
        assert len(result.all_bids) == 3

    def test_rl_adjusted_false_without_rl(self):
        auction = WorkerAuction()
        result = auction.run("task", [_bid("a", 0.8)])
        assert result.rl_adjusted is False

    def test_attach_rl_scorer(self):
        auction = WorkerAuction()
        auction.attach_rl_scorer(object())  # stub
        assert auction._rl_scorer is not None


# ── bids_for_task ─────────────────────────────────────────────────────────────


class TestBidsForTask:
    def test_coding_signals_boost_coder(self):
        bids = bids_for_task("implement a sorting function in Python", ["gpt-4o-mini"])
        coder_bid = next(b for b in bids if b.worker_id == "coder")
        analyst_bid = next(b for b in bids if b.worker_id == "analyst")
        assert coder_bid.confidence > analyst_bid.confidence

    def test_research_signals_boost_researcher(self):
        bids = bids_for_task("research the history of AI", ["gpt-4o-mini"])
        researcher_bid = next(b for b in bids if b.worker_id == "researcher")
        coder_bid = next(b for b in bids if b.worker_id == "coder")
        assert researcher_bid.confidence > coder_bid.confidence

    def test_all_four_specialists_returned(self):
        bids = bids_for_task("write code to analyse data", ["gpt-4o-mini"])
        ids = {b.worker_id for b in bids}
        assert ids == {"coder", "researcher", "writer", "analyst"}

    def test_no_signals_gives_fallback_confidence(self):
        bids = bids_for_task("xyz irrelevant gibberish task", ["gpt-4o-mini"])
        for b in bids:
            assert b.confidence == pytest.approx(0.4)

    def test_model_assigned_from_available(self):
        bids = bids_for_task("code something", ["claude-3-opus"])
        assert all(b.model == "claude-3-opus" for b in bids)

    def test_fallback_model_when_empty(self):
        bids = bids_for_task("code something", [])
        assert all(b.model == "gpt-4o-mini" for b in bids)

    def test_confidence_bounded_at_1(self):
        bids = bids_for_task(
            "implement code function class algorithm script analyse evaluate",
            ["gpt-4o-mini"],
        )
        for b in bids:
            assert b.confidence <= 1.0
