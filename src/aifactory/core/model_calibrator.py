"""
Model Calibrator — RL-driven ability score self-update.

Reads historical (model_id, task_type) → quality_score data from rl_scorer.py
and computes empirical ability score patches that can be written back to
model_registry.yaml.

Task-type → ability mapping:
  coding   → coding, math
  math     → math, reasoning
  research → knowledge, reasoning
  writing  → writing, instruction_following
  analysis → reasoning, knowledge
  tool_use → tool_use, instruction_following
  general  → reasoning

Usage:
  from aifactory.core.model_calibrator import get_calibration_report, apply_patches

  report = await get_calibration_report()
  print(report["summary"])
  if report["patches"]:
      apply_patches(report["patches"], dry_run=False)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Where to find the model registry
_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "config" / "model_registry.yaml"

# Mapping from RL task type → ability names that the task type exercises
_TASK_TO_ABILITIES: dict[str, list[str]] = {
    "coding": ["coding", "math"],
    "math": ["math", "reasoning"],
    "research": ["knowledge", "reasoning"],
    "writing": ["writing", "instruction_following"],
    "analysis": ["reasoning", "knowledge"],
    "tool_use": ["tool_use", "instruction_following"],
    "general": ["reasoning"],
}

# Score scale: RL returns 0–10; model_registry.yaml uses 0–3
_RL_TO_ABILITY_SCALE = 3.0 / 10.0


async def get_calibration_report(
    min_samples: int = 5,
) -> dict[str, Any]:
    """
    Compute empirical ability score patches from RL history.

    Returns:
        {
          "scores": {model_id: {task_type: avg_rl_score}},
          "patches": {model_id: {ability: proposed_score}},
          "summary": str,
          "models_eligible": int,
        }
    """
    from aifactory.core.rl_scorer import get_model_rl_scores

    # Gather RL scores for all task types
    task_types = list(_TASK_TO_ABILITIES.keys())
    all_scores: dict[str, dict[str, float]] = {}  # model_id → {task_type: score}

    for task_type in task_types:
        try:
            type_scores = await get_model_rl_scores(task_type, min_samples=min_samples)
            for model_id, score in type_scores.items():
                all_scores.setdefault(model_id, {})[task_type] = score
        except Exception as exc:
            logger.debug("RL score fetch failed for task_type=%s: %s", task_type, exc)

    if not all_scores:
        return {
            "scores": {},
            "patches": {},
            "summary": f"No RL data available (need at least {min_samples} samples per model/task combo).",  # noqa: E501
            "models_eligible": 0,
        }

    # Load current registry
    current_registry = _load_registry()

    # Compute patches
    patches: dict[str, dict[str, float]] = {}
    for model_id, type_scores in all_scores.items():
        model_patches: dict[str, float] = {}
        for task_type, rl_score in type_scores.items():
            abilities = _TASK_TO_ABILITIES.get(task_type, [])
            empirical_score = round(rl_score * _RL_TO_ABILITY_SCALE, 2)
            for ability in abilities:
                current = _get_current_ability(current_registry, model_id, ability)
                if current is not None and abs(empirical_score - current) > 0.3:
                    # Only patch when empirical evidence is meaningfully different
                    model_patches[ability] = round(min(3.0, max(0.0, empirical_score)), 2)
        if model_patches:
            patches[model_id] = model_patches

    # Build summary
    lines = ["RL Calibration Report", "=" * 40]
    lines.append(f"Models with ≥{min_samples} samples: {len(all_scores)}")
    lines.append(f"Models with suggested patches: {len(patches)}")
    lines.append("")

    for model_id, mpatch in sorted(patches.items()):
        lines.append(f"  {model_id}:")
        for ability, new_score in mpatch.items():
            current = _get_current_ability(current_registry, model_id, ability) or 0.0
            direction = "↑" if new_score > current else "↓"
            lines.append(f"    {ability}: {current:.1f} → {new_score:.1f} {direction}")

    if not patches:
        lines.append("  No significant deviations from static scores detected.")

    return {
        "scores": all_scores,
        "patches": patches,
        "summary": "\n".join(lines),
        "models_eligible": len(all_scores),
    }


def apply_patches(
    patches: dict[str, dict[str, float]],
    dry_run: bool = True,
    backup: bool = True,
) -> dict[str, Any]:
    """
    Write ability score patches back to model_registry.yaml.

    Each model gets an `_empirical_scores` section that is loaded by
    model_selector.py alongside the static scores.  The original scores
    are never overwritten — only the empirical section is updated.

    Args:
        patches: {model_id: {ability: new_score}} from get_calibration_report()
        dry_run: If True, print changes without writing.
        backup:  Write a .bak copy before modifying.

    Returns:
        {"applied": int, "skipped": int, "path": str, "dry_run": bool}
    """
    import yaml

    if not _REGISTRY_PATH.exists():
        logger.error("model_registry.yaml not found at %s", _REGISTRY_PATH)
        return {"applied": 0, "skipped": 0, "path": str(_REGISTRY_PATH), "dry_run": dry_run}

    raw = yaml.safe_load(_REGISTRY_PATH.read_text(encoding="utf-8"))
    models_list: list[dict] = raw.get("models", [])

    applied = 0
    skipped = 0

    for model_cfg in models_list:
        model_id = model_cfg.get("id", "")
        if model_id not in patches:
            skipped += 1
            continue

        model_patches = patches[model_id]
        existing_empirical = model_cfg.get("_empirical_scores", {})
        updated = {**existing_empirical, **model_patches}

        if dry_run:
            logger.info("[DRY RUN] Would patch %s: %s", model_id, model_patches)
        else:
            model_cfg["_empirical_scores"] = updated
        applied += 1

    if not dry_run and applied > 0:
        if backup:
            backup_path = _REGISTRY_PATH.with_suffix(".yaml.bak")
            backup_path.write_text(_REGISTRY_PATH.read_text(encoding="utf-8"))
            logger.info("Registry backup written to %s", backup_path)

        _REGISTRY_PATH.write_text(
            yaml.dump(raw, default_flow_style=False, allow_unicode=True),
            encoding="utf-8",
        )
        logger.info("model_registry.yaml updated with %d model patches", applied)

        # Invalidate downstream caches so best_model() / load_models() re-read
        # the freshly patched abilities on the next call. Import locally to
        # avoid a circular import at module load time.
        from aifactory.core.model_selector import (
            clear_best_model_cache,
            load_models,
        )

        load_models.cache_clear()  # also clears best_model cache via the wrapper
        clear_best_model_cache()

    return {
        "applied": applied,
        "skipped": skipped,
        "path": str(_REGISTRY_PATH),
        "dry_run": dry_run,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_registry() -> dict[str, Any]:
    try:
        import yaml

        return yaml.safe_load(_REGISTRY_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _get_current_ability(
    registry: dict[str, Any],
    model_id: str,
    ability: str,
) -> float | None:
    for model_cfg in registry.get("models", []):
        if model_cfg.get("id") == model_id:
            abilities = model_cfg.get("abilities", {})
            val = abilities.get(ability)
            if val is not None:
                return float(val)
    return None
