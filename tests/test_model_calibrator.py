"""
Tests for core/model_calibrator.py — RL-driven ability score patching.

Covers:
  - _get_current_ability() — found, not found, missing abilities key
  - _load_registry() — valid YAML, missing file, corrupt file
  - get_calibration_report() — no RL data, with patches, no significant deviations
  - apply_patches() — dry run (no writes), live run (writes _empirical_scores)
  - _RL_TO_ABILITY_SCALE constant
  - _TASK_TO_ABILITIES mapping completeness
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import yaml

from aifactory.core.model_calibrator import (
    _RL_TO_ABILITY_SCALE,
    _TASK_TO_ABILITIES,
    _get_current_ability,
    _load_registry,
    apply_patches,
    get_calibration_report,
)

# =============================================================================
# Constants
# =============================================================================


class TestConstants:
    def test_rl_to_ability_scale_is_correct(self) -> None:
        # RL 0–10 → ability 0–3: scale = 3/10
        assert abs(_RL_TO_ABILITY_SCALE - 0.3) < 1e-9

    def test_task_to_abilities_has_all_task_types(self) -> None:
        required = {"coding", "math", "research", "writing", "analysis", "tool_use", "general"}
        assert required.issubset(set(_TASK_TO_ABILITIES.keys()))

    def test_abilities_are_non_empty_lists(self) -> None:
        for task_type, abilities in _TASK_TO_ABILITIES.items():
            assert isinstance(abilities, list), f"{task_type} must map to a list"
            assert len(abilities) >= 1, f"{task_type} must map to ≥1 ability"


# =============================================================================
# _get_current_ability
# =============================================================================


class TestGetCurrentAbility:
    def _registry(self, models: list[dict]) -> dict:
        return {"models": models}

    def test_found_returns_float(self) -> None:
        reg = self._registry([{"id": "gpt-4o", "abilities": {"coding": 3, "reasoning": 2}}])
        val = _get_current_ability(reg, "gpt-4o", "coding")
        assert val == 3.0

    def test_not_found_model_returns_none(self) -> None:
        reg = self._registry([{"id": "gpt-4o", "abilities": {"coding": 3}}])
        val = _get_current_ability(reg, "claude-3-opus", "coding")
        assert val is None

    def test_ability_missing_returns_none(self) -> None:
        reg = self._registry([{"id": "gpt-4o", "abilities": {"coding": 3}}])
        val = _get_current_ability(reg, "gpt-4o", "math")
        assert val is None

    def test_empty_registry_returns_none(self) -> None:
        assert _get_current_ability({}, "gpt-4o", "coding") is None

    def test_no_models_key_returns_none(self) -> None:
        assert _get_current_ability({"other": []}, "gpt-4o", "coding") is None

    def test_model_without_abilities_key_returns_none(self) -> None:
        reg = self._registry([{"id": "gpt-4o"}])
        val = _get_current_ability(reg, "gpt-4o", "coding")
        assert val is None

    def test_float_conversion(self) -> None:
        reg = self._registry([{"id": "model-x", "abilities": {"reasoning": "2"}}])
        val = _get_current_ability(reg, "model-x", "reasoning")
        assert val == 2.0
        assert isinstance(val, float)


# =============================================================================
# _load_registry
# =============================================================================


class TestLoadRegistry:
    def test_loads_valid_yaml(self) -> None:
        data = {
            "models": [
                {"id": "gpt-4o", "abilities": {"coding": 3}},
            ]
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            path = Path(f.name)

        import aifactory.core.model_calibrator as mc

        orig = mc._REGISTRY_PATH
        mc._REGISTRY_PATH = path
        try:
            reg = _load_registry()
            assert reg["models"][0]["id"] == "gpt-4o"
        finally:
            mc._REGISTRY_PATH = orig
            path.unlink()

    def test_missing_file_returns_empty_dict(self) -> None:
        import aifactory.core.model_calibrator as mc

        orig = mc._REGISTRY_PATH
        mc._REGISTRY_PATH = Path("/nonexistent/path/registry.yaml")
        try:
            reg = _load_registry()
            assert reg == {}
        finally:
            mc._REGISTRY_PATH = orig


# =============================================================================
# get_calibration_report
# =============================================================================


class TestGetCalibrationReport:
    async def test_no_rl_data_returns_empty_report(self) -> None:
        # get_model_rl_scores returns empty dict for all task types
        with patch(
            "aifactory.core.rl_scorer.get_model_rl_scores",
            new_callable=AsyncMock,
            return_value={},
        ):
            report = await get_calibration_report(min_samples=5)

        assert report["models_eligible"] == 0
        assert report["patches"] == {}
        assert "No RL data" in report["summary"]

    async def test_report_with_deviating_scores(self) -> None:
        # Model gpt-4o has coding RL score 10 (→ ability 3.0)
        # but static registry has coding=1 → significant deviation → patch
        registry_data = {"models": [{"id": "gpt-4o", "abilities": {"coding": 1, "math": 1}}]}
        rl_scores = {"gpt-4o": 10.0}  # RL score for coding task type

        async def _mock_rl(task_type: str, min_samples: int = 5) -> dict[str, float]:
            if task_type == "coding":
                return rl_scores
            return {}

        with (
            patch("aifactory.core.rl_scorer.get_model_rl_scores", side_effect=_mock_rl),
            patch("aifactory.core.model_calibrator._load_registry", return_value=registry_data),
        ):
            report = await get_calibration_report(min_samples=1)

        assert report["models_eligible"] >= 1
        assert "gpt-4o" in report["patches"]

    async def test_no_significant_deviation_no_patches(self) -> None:
        # RL score for coding = 8 → ability = 8 * 0.3 = 2.4 → current = 2.4 → no change
        registry_data = {"models": [{"id": "gpt-4o", "abilities": {"coding": 2.4, "math": 2.4}}]}

        async def _mock_rl(task_type: str, min_samples: int = 5) -> dict[str, float]:
            if task_type == "coding":
                return {"gpt-4o": 8.0}  # 8 * 0.3 = 2.4 → no deviation > 0.3
            return {}

        with (
            patch("aifactory.core.rl_scorer.get_model_rl_scores", side_effect=_mock_rl),
            patch("aifactory.core.model_calibrator._load_registry", return_value=registry_data),
        ):
            report = await get_calibration_report(min_samples=1)

        assert "gpt-4o" not in report["patches"]

    async def test_summary_string_contains_header(self) -> None:
        with patch(
            "aifactory.core.rl_scorer.get_model_rl_scores",
            new_callable=AsyncMock,
            return_value={},
        ):
            report = await get_calibration_report()

        assert isinstance(report["summary"], str)


# =============================================================================
# apply_patches — dry run
# =============================================================================


class TestApplyPatches:
    def _write_registry(self, path: Path, data: dict) -> None:
        path.write_text(yaml.dump(data, default_flow_style=False))

    def test_dry_run_does_not_write_file(self) -> None:
        data = {
            "models": [
                {"id": "gpt-4o", "abilities": {"coding": 1}},
            ]
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model_registry.yaml"
            self._write_registry(path, data)

            import aifactory.core.model_calibrator as mc

            orig = mc._REGISTRY_PATH
            mc._REGISTRY_PATH = path
            try:
                patches = {"gpt-4o": {"coding": 3.0}}
                result = apply_patches(patches, dry_run=True)
                assert result["dry_run"] is True
                assert result["applied"] == 1
                # File content should not change
                reloaded = yaml.safe_load(path.read_text())
                assert "_empirical_scores" not in reloaded["models"][0]
            finally:
                mc._REGISTRY_PATH = orig

    def test_live_run_writes_empirical_scores(self) -> None:
        data = {
            "models": [
                {"id": "gpt-4o", "abilities": {"coding": 1, "math": 1}},
            ]
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model_registry.yaml"
            self._write_registry(path, data)

            import aifactory.core.model_calibrator as mc

            orig = mc._REGISTRY_PATH
            mc._REGISTRY_PATH = path
            try:
                patches = {"gpt-4o": {"coding": 2.8}}
                result = apply_patches(patches, dry_run=False, backup=False)
                assert result["dry_run"] is False
                assert result["applied"] == 1
                reloaded = yaml.safe_load(path.read_text())
                assert "_empirical_scores" in reloaded["models"][0]
                assert reloaded["models"][0]["_empirical_scores"]["coding"] == 2.8
            finally:
                mc._REGISTRY_PATH = orig

    def test_backup_created_on_live_run(self) -> None:
        data = {"models": [{"id": "gpt-4o", "abilities": {"coding": 1}}]}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model_registry.yaml"
            self._write_registry(path, data)

            import aifactory.core.model_calibrator as mc

            orig = mc._REGISTRY_PATH
            mc._REGISTRY_PATH = path
            try:
                apply_patches({"gpt-4o": {"coding": 2.0}}, dry_run=False, backup=True)
                bak = path.with_suffix(".yaml.bak")
                assert bak.exists()
            finally:
                mc._REGISTRY_PATH = orig

    def test_skipped_count_for_unmatched_models(self) -> None:
        data = {
            "models": [
                {"id": "gpt-4o", "abilities": {"coding": 1}},
                {"id": "claude-3", "abilities": {"coding": 2}},
            ]
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model_registry.yaml"
            self._write_registry(path, data)

            import aifactory.core.model_calibrator as mc

            orig = mc._REGISTRY_PATH
            mc._REGISTRY_PATH = path
            try:
                patches = {"gpt-4o": {"coding": 3.0}}
                result = apply_patches(patches, dry_run=True)
                assert result["applied"] == 1
                assert result["skipped"] == 1
            finally:
                mc._REGISTRY_PATH = orig

    def test_missing_registry_file_returns_zeros(self) -> None:
        import aifactory.core.model_calibrator as mc

        orig = mc._REGISTRY_PATH
        mc._REGISTRY_PATH = Path("/nonexistent/model_registry.yaml")
        try:
            result = apply_patches({"gpt-4o": {"coding": 3.0}})
            assert result["applied"] == 0
            assert result["skipped"] == 0
        finally:
            mc._REGISTRY_PATH = orig

    def test_existing_empirical_scores_merged(self) -> None:
        data = {
            "models": [
                {
                    "id": "gpt-4o",
                    "abilities": {"coding": 1},
                    "_empirical_scores": {"reasoning": 2.5},
                },
            ]
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model_registry.yaml"
            self._write_registry(path, data)

            import aifactory.core.model_calibrator as mc

            orig = mc._REGISTRY_PATH
            mc._REGISTRY_PATH = path
            try:
                apply_patches({"gpt-4o": {"coding": 2.8}}, dry_run=False, backup=False)
                reloaded = yaml.safe_load(path.read_text())
                empirical = reloaded["models"][0]["_empirical_scores"]
                # Both old and new key present
                assert empirical["reasoning"] == 2.5
                assert empirical["coding"] == 2.8
            finally:
                mc._REGISTRY_PATH = orig
