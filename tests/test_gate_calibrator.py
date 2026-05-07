"""Tests for GateCalibrator (Week 6 — Adaptive Mesh Pipeline)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from aifactory.core.gate_calibrator import (
    _DEFAULT_EXTENDED_THRESHOLD,
    _DEFAULT_LITE_THRESHOLD,
    _EXTENDED_MAX,
    _LITE_MAX,
    GateCalibrator,
    apply_patches,
    compute_patches,
    load_thresholds,
    save_thresholds,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _run(topology: str, quality: float) -> dict:
    return {"topology": topology, "composite_quality": quality}


# ── load / save thresholds ────────────────────────────────────────────────────


class TestThresholdPersistence:
    def test_load_returns_defaults_when_file_absent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "aifactory.core.gate_calibrator._THRESHOLDS_PATH", tmp_path / "gate_thresholds.json"
        )
        thr = load_thresholds()
        assert thr["lite_threshold"] == pytest.approx(_DEFAULT_LITE_THRESHOLD)
        assert thr["extended_threshold"] == pytest.approx(_DEFAULT_EXTENDED_THRESHOLD)

    def test_save_and_reload(self, tmp_path, monkeypatch):
        p = tmp_path / "gate_thresholds.json"
        monkeypatch.setattr("aifactory.core.gate_calibrator._THRESHOLDS_PATH", p)
        save_thresholds({"lite_threshold": 0.25, "extended_threshold": 0.75})
        assert p.exists()
        thr = load_thresholds()
        assert thr["lite_threshold"] == pytest.approx(0.25)
        assert thr["extended_threshold"] == pytest.approx(0.75)

    def test_load_handles_corrupt_file(self, tmp_path, monkeypatch):
        p = tmp_path / "gate_thresholds.json"
        p.write_text("not json")
        monkeypatch.setattr("aifactory.core.gate_calibrator._THRESHOLDS_PATH", p)
        thr = load_thresholds()
        assert thr["lite_threshold"] == pytest.approx(_DEFAULT_LITE_THRESHOLD)


# ── compute_patches ───────────────────────────────────────────────────────────


class TestComputePatches:
    def _make_runs(self, topology_qualities: dict[str, list[float]]) -> list[dict]:
        runs = []
        for topo, qualities in topology_qualities.items():
            for q in qualities:
                runs.append(_run(topo, q))
        return runs

    def test_no_runs_returns_no_patches(self):
        report = compute_patches([])
        assert report["patches"]["lite_threshold"] is None
        assert report["patches"]["extended_threshold"] is None

    def test_minimal_worse_than_full_raises_lite_threshold(self):
        # MINIMAL quality=4, FULL quality=8 → gap=4 → lite threshold should rise
        runs = self._make_runs({"MINIMAL": [4.0] * 10, "FULL": [8.0] * 10})
        current = {"lite_threshold": 0.35, "extended_threshold": 0.80}
        report = compute_patches(runs, current_thresholds=current, min_samples=5)
        new_lite = report["patches"]["lite_threshold"]
        assert new_lite is not None
        assert new_lite > 0.35  # raised because MINIMAL underperforms

    def test_minimal_on_par_with_full_lowers_lite_threshold(self):
        # MINIMAL quality=8, FULL quality=7 → gap=-1 → lite threshold should fall slightly
        runs = self._make_runs({"MINIMAL": [8.0] * 10, "FULL": [7.0] * 10})
        current = {"lite_threshold": 0.35, "extended_threshold": 0.80}
        report = compute_patches(runs, current_thresholds=current, min_samples=5)
        new_lite = report["patches"]["lite_threshold"]
        assert new_lite is not None
        assert new_lite < 0.35  # lowered because MINIMAL quality is good

    def test_extended_better_than_full_lowers_extended_threshold(self):
        # EXTENDED quality=9, FULL quality=6 → gap=3 → extended threshold should fall
        runs = self._make_runs({"EXTENDED": [9.0] * 10, "FULL": [6.0] * 10})
        current = {"lite_threshold": 0.35, "extended_threshold": 0.80}
        report = compute_patches(runs, current_thresholds=current, min_samples=5)
        new_ext = report["patches"]["extended_threshold"]
        assert new_ext is not None
        assert new_ext < 0.80  # lowered to promote EXTENDED more

    def test_extended_worse_than_full_raises_extended_threshold(self):
        # EXTENDED quality=4, FULL quality=8 → gap=-4 → extended threshold should rise
        runs = self._make_runs({"EXTENDED": [4.0] * 10, "FULL": [8.0] * 10})
        current = {"lite_threshold": 0.35, "extended_threshold": 0.80}
        report = compute_patches(runs, current_thresholds=current, min_samples=5)
        new_ext = report["patches"]["extended_threshold"]
        assert new_ext is not None
        assert new_ext > 0.80  # raised to avoid EXTENDED

    def test_thresholds_stay_within_bounds(self):
        # Extreme quality gap should still respect bounds
        runs = self._make_runs({"MINIMAL": [0.0] * 20, "FULL": [10.0] * 20})
        # Start at max allowed value
        current = {"lite_threshold": _LITE_MAX, "extended_threshold": 0.80}
        report = compute_patches(runs, current_thresholds=current, min_samples=5)
        new_lite = report["patches"]["lite_threshold"]
        if new_lite is not None:
            assert new_lite <= _LITE_MAX

        runs_ext = self._make_runs(
            {"MINIMAL": [0.0] * 20, "FULL": [0.0] * 20, "EXTENDED": [0.0] * 20}
        )
        current_ext = {"lite_threshold": 0.35, "extended_threshold": _EXTENDED_MAX}
        report_ext = compute_patches(runs_ext, current_thresholds=current_ext, min_samples=5)
        new_ext = report_ext["patches"]["extended_threshold"]
        if new_ext is not None:
            assert new_ext <= _EXTENDED_MAX

    def test_min_samples_filters_insufficient_topology_data(self):
        # Only 2 MINIMAL runs — below min_samples=5, should not affect lite_threshold
        runs = self._make_runs({"MINIMAL": [2.0, 3.0], "FULL": [8.0] * 10})
        report = compute_patches(runs, min_samples=5)
        assert report["patches"]["lite_threshold"] is None

    def test_quality_by_topology_populated(self):
        runs = self._make_runs({"MINIMAL": [5.0] * 6, "FULL": [8.0] * 6})
        report = compute_patches(runs, min_samples=5)
        assert "MINIMAL" in report["quality_by_topology"]
        assert "FULL" in report["quality_by_topology"]
        assert report["quality_by_topology"]["MINIMAL"] == pytest.approx(5.0)

    def test_reasoning_included_in_report(self):
        runs = self._make_runs({"MINIMAL": [3.0] * 6, "FULL": [8.0] * 6})
        report = compute_patches(runs, min_samples=5)
        assert isinstance(report["reasoning"], str)
        assert len(report["reasoning"]) > 0

    def test_no_patches_when_quality_identical(self):
        # Identical quality → gap=0 → nudge=0 → no patch
        runs = self._make_runs({"MINIMAL": [7.0] * 10, "FULL": [7.0] * 10})
        report = compute_patches(runs, min_samples=5)
        assert report["patches"]["lite_threshold"] is None


# ── apply_patches ─────────────────────────────────────────────────────────────


class TestApplyPatches:
    def test_dry_run_does_not_write_file(self, tmp_path, monkeypatch):
        p = tmp_path / "gate_thresholds.json"
        monkeypatch.setattr("aifactory.core.gate_calibrator._THRESHOLDS_PATH", p)
        result = apply_patches({"lite_threshold": 0.40, "extended_threshold": None}, dry_run=True)
        assert not p.exists()
        assert result["lite_threshold"] == pytest.approx(0.40)

    def test_apply_writes_file(self, tmp_path, monkeypatch):
        p = tmp_path / "gate_thresholds.json"
        monkeypatch.setattr("aifactory.core.gate_calibrator._THRESHOLDS_PATH", p)
        apply_patches({"lite_threshold": 0.42, "extended_threshold": 0.78}, dry_run=False)
        assert p.exists()
        data = json.loads(p.read_text())
        assert data["lite_threshold"] == pytest.approx(0.42)
        assert data["extended_threshold"] == pytest.approx(0.78)

    def test_none_patches_leave_value_unchanged(self, tmp_path, monkeypatch):
        p = tmp_path / "gate_thresholds.json"
        monkeypatch.setattr("aifactory.core.gate_calibrator._THRESHOLDS_PATH", p)
        save_thresholds({"lite_threshold": 0.35, "extended_threshold": 0.80})
        result = apply_patches({"lite_threshold": None, "extended_threshold": None}, dry_run=True)
        assert result["lite_threshold"] == pytest.approx(0.35)
        assert result["extended_threshold"] == pytest.approx(0.80)


# ── GateCalibrator class ──────────────────────────────────────────────────────


class TestGateCalibrator:
    def test_current_thresholds_returns_defaults(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "aifactory.core.gate_calibrator._THRESHOLDS_PATH", tmp_path / "gate_thresholds.json"
        )
        cal = GateCalibrator()
        thr = cal.current_thresholds()
        assert "lite_threshold" in thr
        assert "extended_threshold" in thr

    async def test_calibrate_runs_and_returns_report(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "aifactory.core.gate_calibrator._THRESHOLDS_PATH", tmp_path / "gate_thresholds.json"
        )
        # Mock get_calibration_report to avoid needing a real RunStore
        mock_report = {
            "patches": {"lite_threshold": None, "extended_threshold": None},
            "quality_by_topology": {},
            "current": {"lite_threshold": 0.35, "extended_threshold": 0.80},
            "reasoning": "No threshold changes — not enough data.",
        }
        with patch(
            "aifactory.core.gate_calibrator.get_calibration_report",
            new_callable=AsyncMock,
            return_value=mock_report,
        ):
            cal = GateCalibrator()
            report = await cal.calibrate(dry_run=True)
            assert report == mock_report
            assert cal.last_report() == mock_report

    def test_last_report_none_before_first_calibrate(self):
        cal = GateCalibrator()
        assert cal.last_report() is None


# ── Dynamic pipeline threshold loading ───────────────────────────────────────


class TestDynamicPipelineThresholdLoading:
    def test_gate_thresholds_loaded_at_pipeline_build(self, tmp_path, monkeypatch):
        # Write custom thresholds to the calibrator path
        p = tmp_path / "gate_thresholds.json"
        p.write_text(json.dumps({"lite_threshold": 0.20, "extended_threshold": 0.90}))
        monkeypatch.setattr("aifactory.core.gate_calibrator._THRESHOLDS_PATH", p)
        # Importing and calling load_thresholds should return the custom values
        from aifactory.core.gate_calibrator import load_thresholds as _load

        thr = _load()
        assert thr["lite_threshold"] == pytest.approx(0.20)
        assert thr["extended_threshold"] == pytest.approx(0.90)
