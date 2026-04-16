"""Tests for output_validator.py — worker quality gate."""

from __future__ import annotations

from aifactory.core.output_validator import validate_worker_output


class TestValidateWorkerOutput:
    def test_empty_result_rejected(self):
        valid, reason = validate_worker_output("")
        assert not valid
        assert "empty" in reason

    def test_whitespace_only_rejected(self):
        valid, reason = validate_worker_output("   \n\t  ")
        assert not valid
        assert "empty" in reason

    def test_too_short_rejected(self):
        valid, reason = validate_worker_output("ok")
        assert not valid
        assert "short" in reason

    def test_exactly_min_chars_accepted(self):
        # 20 chars exactly — on the boundary
        valid, reason = validate_worker_output("a" * 20)
        assert valid
        assert reason == ""

    def test_refusal_i_cannot_rejected(self):
        valid, reason = validate_worker_output("I cannot help with that request.")
        assert not valid
        assert "refusal" in reason

    def test_refusal_i_am_unable_rejected(self):
        valid, reason = validate_worker_output("I am unable to complete this task.")
        assert not valid
        assert "refusal" in reason

    def test_refusal_as_an_ai_rejected(self):
        valid, reason = validate_worker_output("As an AI language model, I cannot...")
        assert not valid
        assert "refusal" in reason

    def test_refusal_mixed_case_rejected(self):
        valid, reason = validate_worker_output("I CANNOT process this request.")
        assert not valid
        assert "refusal" in reason

    def test_valid_prose_accepted(self):
        valid, reason = validate_worker_output(
            "Here is the Python function you requested:\n\n"
            "def add(a, b):\n    return a + b\n\nThis implements simple addition."
        )
        assert valid
        assert reason == ""

    def test_valid_short_answer_accepted(self):
        valid, _reason = validate_worker_output("The capital of France is Paris.")
        assert valid

    def test_non_refusal_starting_with_i_accepted(self):
        """'I have completed...' should NOT be rejected — only specific prefixes."""
        valid, _reason = validate_worker_output(
            "I have completed the analysis. The results show that..."
        )
        assert valid

    def test_task_description_does_not_affect_result(self):
        """task_description is informational only — doesn't change validation logic."""
        v1, _ = validate_worker_output("I cannot help.", "write a poem")
        v2, _ = validate_worker_output("I cannot help.", "")
        assert v1 == v2 == False  # noqa: E712
