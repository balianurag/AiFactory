"""
GateCalibrator — self-calibrating complexity_score thresholds for the ComplexityGate.

After N pipeline runs the judge_review node fires this as a background task.
The calibrator reads the run history, groups runs by the topology the gate
selected, computes mean composite_quality per topology, and nudges the
LITE_THRESHOLD / EXTENDED_THRESHOLD stored in ~/.aifactory/gate_thresholds.json
so that topology choices that led to low quality are less likely next time.

Design decisions:
  - State persisted to ~/.aifactory/gate_thresholds.json (survives restarts)
  - Thresholds capped to [LITE_MIN, LITE_MAX] and [EXTENDED_MIN, EXTENDED_MAX]
    so the calibrator can't push the system into degenerate configs
  - Nudge magnitude scales with the quality gap — tiny adjustments for small gaps
  - Dry-run mode: compute and return patches without writing them
  - Used by dynamic_pipeline judge_review via AIFACTORY_GATE_CALIBRATE_EVERY_N_RUNS
  - Also callable directly via `aifactory gate calibrate` CLI command
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ─── Paths ───────────────────────────────────────────────────────────────────

_THRESHOLDS_PATH = Path.home() / ".aifactory" / "gate_thresholds.json"

# ─── Safe bounds for thresholds ──────────────────────────────────────────────

_LITE_MIN = 0.15
_LITE_MAX = 0.55
_EXTENDED_MIN = 0.60
_EXTENDED_MAX = 0.95

# ─── Defaults (mirror dynamic_pipeline.py) ───────────────────────────────────

_DEFAULT_LITE_THRESHOLD = 0.35
_DEFAULT_EXTENDED_THRESHOLD = 0.80

# ─── Nudge magnitude per quality unit ────────────────────────────────────────
# e.g. if quality gap is 2.0 points → nudge = 0.02 threshold units

_NUDGE_RATE = 0.01


# ─── Persistence helpers ─────────────────────────────────────────────────────


def load_thresholds() -> dict[str, float]:
    """Load persisted thresholds; fall back to defaults if file is absent."""
    if _THRESHOLDS_PATH.exists():
        try:
            data = json.loads(_THRESHOLDS_PATH.read_text())
            return {
                "lite_threshold": float(data.get("lite_threshold", _DEFAULT_LITE_THRESHOLD)),
                "extended_threshold": float(
                    data.get("extended_threshold", _DEFAULT_EXTENDED_THRESHOLD)
                ),
            }
        except Exception as exc:
            logger.debug("gate_calibrator: failed to read thresholds: %s", exc)
    return {
        "lite_threshold": _DEFAULT_LITE_THRESHOLD,
        "extended_threshold": _DEFAULT_EXTENDED_THRESHOLD,
    }


def save_thresholds(thresholds: dict[str, float]) -> None:
    _THRESHOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _THRESHOLDS_PATH.write_text(json.dumps(thresholds, indent=2))
    logger.info(
        "gate_calibrator: saved thresholds lite=%.3f extended=%.3f",
        thresholds["lite_threshold"],
        thresholds["extended_threshold"],
    )


# ─── Quality aggregation ─────────────────────────────────────────────────────


def _group_quality(
    runs: list[dict[str, Any]],
) -> dict[str, list[float]]:
    """
    Group composite_quality by topology for the last N runs.

    Returns {topology_name: [quality_score, ...]} where topology is one of
    "MINIMAL", "STANDARD", "FULL", "EXTENDED".
    """
    groups: dict[str, list[float]] = {}
    for run in runs:
        topology = str(run.get("topology", "")).upper() or "FULL"
        quality = run.get("composite_quality")
        if quality is not None:
            try:
                groups.setdefault(topology, []).append(float(quality))
            except (TypeError, ValueError):
                pass
    return groups


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ─── Core calibration logic ───────────────────────────────────────────────────


def compute_patches(
    runs: list[dict[str, Any]],
    current_thresholds: dict[str, float] | None = None,
    min_samples: int = 5,
) -> dict[str, Any]:
    """
    Analyse run history and propose threshold patches.

    Returns a dict with:
      - patches: {lite_threshold: float | None, extended_threshold: float | None}
        (None means no change)
      - quality_by_topology: {topology: mean_quality}
      - reasoning: human-readable explanation
      - current: current threshold values
    """
    if current_thresholds is None:
        current_thresholds = load_thresholds()

    groups = _group_quality(runs)
    quality_by_topology = {t: _mean(q) for t, q in groups.items() if len(q) >= min_samples}

    patches: dict[str, float | None] = {
        "lite_threshold": None,
        "extended_threshold": None,
    }
    reasoning_lines: list[str] = []

    lite_thr = current_thresholds["lite_threshold"]
    ext_thr = current_thresholds["extended_threshold"]

    # ── Lite threshold calibration ──────────────────────────────────────────
    # If MINIMAL topology has poor quality vs FULL/STANDARD, raise the lite
    # threshold so fewer runs get downgraded to MINIMAL.
    # If MINIMAL quality is on par, lower the threshold slightly to save cost.
    minimal_q = quality_by_topology.get("MINIMAL")
    full_q = quality_by_topology.get("FULL") or quality_by_topology.get("STANDARD")

    if minimal_q is not None and full_q is not None:
        gap = full_q - minimal_q  # positive = FULL is better
        nudge = gap * _NUDGE_RATE
        if abs(nudge) >= 0.001:
            new_lite = max(_LITE_MIN, min(_LITE_MAX, lite_thr + nudge))
            if abs(new_lite - lite_thr) >= 0.001:
                patches["lite_threshold"] = round(new_lite, 4)
                direction = "raised" if nudge > 0 else "lowered"
                reasoning_lines.append(
                    f"lite_threshold {direction} {lite_thr:.3f}→{new_lite:.3f}: "
                    f"MINIMAL quality {minimal_q:.1f} vs FULL {full_q:.1f} "
                    f"(gap={gap:+.1f})"
                )

    # ── Extended threshold calibration ─────────────────────────────────────
    # If EXTENDED topology produces noticeably better quality than FULL,
    # lower the extended threshold so more runs get the WaveGovernor.
    # If EXTENDED quality is similar or worse, raise the threshold.
    extended_q = quality_by_topology.get("EXTENDED")
    baseline_q = full_q  # use FULL as the baseline

    if extended_q is not None and baseline_q is not None:
        gap = extended_q - baseline_q  # positive = EXTENDED is better
        nudge = -gap * _NUDGE_RATE  # lower threshold when EXTENDED wins
        if abs(nudge) >= 0.001:
            new_ext = max(_EXTENDED_MIN, min(_EXTENDED_MAX, ext_thr + nudge))
            if abs(new_ext - ext_thr) >= 0.001:
                patches["extended_threshold"] = round(new_ext, 4)
                direction = "lowered" if nudge < 0 else "raised"
                reasoning_lines.append(
                    f"extended_threshold {direction} {ext_thr:.3f}→{new_ext:.3f}: "
                    f"EXTENDED quality {extended_q:.1f} vs FULL {baseline_q:.1f} "
                    f"(gap={gap:+.1f})"
                )

    if not reasoning_lines:
        reasoning_lines.append(
            "No threshold changes — topology quality differences are within tolerance "
            "or not enough data."
        )

    return {
        "patches": patches,
        "quality_by_topology": quality_by_topology,
        "current": dict(current_thresholds),
        "reasoning": " | ".join(reasoning_lines),
    }


def apply_patches(
    patches: dict[str, float | None],
    dry_run: bool = True,
) -> dict[str, float]:
    """
    Apply calibrated threshold patches.

    Returns the resulting thresholds (written to disk when dry_run=False).
    """
    current = load_thresholds()
    updated = dict(current)

    for key, value in patches.items():
        if value is not None and key in updated:
            updated[key] = value

    if not dry_run:
        save_thresholds(updated)

    return updated


# ─── Public async entry point (called from dynamic_pipeline) ─────────────────


async def get_calibration_report(
    n_runs: int = 200,
    min_samples: int = 5,
) -> dict[str, Any]:
    """
    Load recent run history and compute calibration patches.

    Async wrapper so it can be awaited from dynamic_pipeline background tasks.
    """
    try:
        from aifactory.core.run_store import get_run_store

        store = get_run_store()
        recent = store.list_recent(n=n_runs)
        runs_data = [
            {
                "topology": getattr(r, "topology", "FULL"),
                "composite_quality": getattr(r, "composite_quality", None),
            }
            for r in recent
        ]
    except Exception as exc:
        logger.debug("gate_calibrator: failed to load run history: %s", exc)
        runs_data = []

    return compute_patches(runs_data, min_samples=min_samples)


# ─── GateCalibrator class (thin stateful wrapper for CLI use) ────────────────


class GateCalibrator:
    """
    Stateful wrapper around the functional calibration API.

    Holds the last computed report so the CLI can display it without
    re-computing, and provides apply() to write patches to disk.
    """

    def __init__(self) -> None:
        self._last_report: dict[str, Any] | None = None

    def current_thresholds(self) -> dict[str, float]:
        return load_thresholds()

    async def calibrate(
        self,
        n_runs: int = 200,
        min_samples: int = 5,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        report = await get_calibration_report(n_runs=n_runs, min_samples=min_samples)
        self._last_report = report
        if not dry_run:
            apply_patches(report["patches"], dry_run=False)
        return report

    def last_report(self) -> dict[str, Any] | None:
        return self._last_report
