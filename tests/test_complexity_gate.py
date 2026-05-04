"""Tests for the Complexity Gate v2 — YAML-driven, confidence-scored classifier."""

from __future__ import annotations

import pytest

from aifactory.core.complexity_gate import (
    ComplexityGate,
    GateDecision,
    _extract_math_expression,
    _is_simple_qa,
    _needs_tools,
    _normalize_natural_language,
    _score_qa,
    _score_tools,
    _try_static_fact,
    _words_to_digits,
    safe_math_eval,
)

# ═════════════════════════════════════════════════════════════════════════════
# Safe Math Evaluator
# ═════════════════════════════════════════════════════════════════════════════


class TestSafeMathEval:
    def test_basic_addition(self) -> None:
        assert safe_math_eval("2 + 2") == 4

    def test_basic_subtraction(self) -> None:
        assert safe_math_eval("10 - 3") == 7

    def test_multiplication(self) -> None:
        assert safe_math_eval("15 * 3") == 45

    def test_division(self) -> None:
        assert safe_math_eval("100 / 4") == 25

    def test_floor_division(self) -> None:
        assert safe_math_eval("7 // 2") == 3

    def test_modulo(self) -> None:
        assert safe_math_eval("10 % 3") == 1

    def test_exponentiation(self) -> None:
        assert safe_math_eval("2 ** 10") == 1024

    def test_order_of_operations(self) -> None:
        assert safe_math_eval("2 + 3 * 4") == 14

    def test_parentheses(self) -> None:
        assert safe_math_eval("(2 + 3) * 4") == 20

    def test_negative_number(self) -> None:
        assert safe_math_eval("-5 + 3") == -2

    def test_float_result(self) -> None:
        assert safe_math_eval("10 / 3") == pytest.approx(3.333, rel=1e-2)

    def test_sqrt(self) -> None:
        assert safe_math_eval("sqrt(144)") == 12

    def test_factorial(self) -> None:
        assert safe_math_eval("factorial(5)") == 120

    def test_pi_constant(self) -> None:
        result = safe_math_eval("pi")
        assert result is not None
        assert abs(result - 3.14159) < 0.001

    def test_rejects_import(self) -> None:
        assert safe_math_eval("__import__('os')") is None

    def test_rejects_exec(self) -> None:
        assert safe_math_eval("exec('print(1)')") is None

    def test_rejects_arbitrary_function(self) -> None:
        assert safe_math_eval("open('/etc/passwd')") is None

    def test_rejects_string(self) -> None:
        assert safe_math_eval("'hello'") is None

    def test_rejects_empty(self) -> None:
        assert safe_math_eval("") is None

    def test_rejects_nonsense(self) -> None:
        assert safe_math_eval("build a fastapi app") is None

    def test_large_factorial(self) -> None:
        assert safe_math_eval("factorial(10)") == 3628800

    def test_nested_expression(self) -> None:
        assert safe_math_eval("sqrt(16) + factorial(3)") == 10


# ═════════════════════════════════════════════════════════════════════════════
# Math Expression Extraction
# ═════════════════════════════════════════════════════════════════════════════


class TestExtractMathExpression:
    def test_what_is(self) -> None:
        assert _extract_math_expression("What is 2 + 2?") == "2 + 2"

    def test_whats(self) -> None:
        assert _extract_math_expression("what's 15 * 3") == "15 * 3"

    def test_calculate(self) -> None:
        assert _extract_math_expression("Calculate 100 / 4") == "100 / 4"

    def test_compute(self) -> None:
        assert _extract_math_expression("Compute sqrt(16)") == "sqrt(16)"

    def test_how_much(self) -> None:
        assert _extract_math_expression("How much is 5 + 3?") == "5 + 3"

    def test_bare_arithmetic(self) -> None:
        assert _extract_math_expression("15 * 3") == "15 * 3"

    def test_bare_function(self) -> None:
        assert _extract_math_expression("factorial(5)") == "factorial(5)"

    def test_caret_to_power(self) -> None:
        expr = _extract_math_expression("What is 2^10?")
        assert expr is not None
        assert "**" in expr

    def test_rejects_prose(self) -> None:
        assert _extract_math_expression("Build a FastAPI todo app") is None

    def test_rejects_research(self) -> None:
        assert _extract_math_expression("Research about world war 3") is None

    def test_rejects_empty(self) -> None:
        assert _extract_math_expression("") is None


# ═════════════════════════════════════════════════════════════════════════════
# Natural Language Numbers
# ═════════════════════════════════════════════════════════════════════════════


class TestNaturalLanguageNumbers:
    def test_words_to_digits_simple(self) -> None:
        assert _words_to_digits("two plus three") == "2 plus 3"

    def test_words_to_digits_compound(self) -> None:
        assert _words_to_digits("twenty three") == "23"

    def test_words_to_digits_hundred(self) -> None:
        assert _words_to_digits("five hundred") == "500"

    def test_words_to_digits_mixed(self) -> None:
        assert _words_to_digits("two + 2") == "2 + 2"

    def test_words_to_digits_preserves_non_numbers(self) -> None:
        result = _words_to_digits("build a fastapi app")
        assert "build" in result

    def test_normalize_word_plus_word(self) -> None:
        assert _normalize_natural_language("two plus three") == "2 + 3"

    def test_normalize_word_times_word(self) -> None:
        assert _normalize_natural_language("five times six") == "5 * 6"

    def test_normalize_mixed_word_digit(self) -> None:
        assert _normalize_natural_language("two + 2") == "2 + 2"

    def test_normalize_word_divided_by(self) -> None:
        result = _normalize_natural_language("one hundred divided by four")
        assert result is not None
        assert "100" in result
        assert "/" in result

    def test_normalize_sqrt_words(self) -> None:
        assert _normalize_natural_language("square root of sixteen") == "sqrt(16)"

    def test_normalize_factorial_words(self) -> None:
        assert _normalize_natural_language("factorial of five") == "factorial(5)"

    def test_normalize_with_question_prefix(self) -> None:
        result = _normalize_natural_language("What is twenty plus 3?")
        assert result is not None
        assert "20" in result
        assert "+" in result

    def test_normalize_how_much(self) -> None:
        result = _normalize_natural_language("How much is twelve times twelve?")
        assert result is not None
        assert "12" in result

    def test_normalize_rejects_prose(self) -> None:
        assert _normalize_natural_language("Build a FastAPI app") is None

    def test_normalize_rejects_empty(self) -> None:
        assert _normalize_natural_language("") is None

    def test_extract_mixed_word_digit(self) -> None:
        expr = _extract_math_expression("What is two + 2?")
        assert expr is not None
        assert safe_math_eval(expr) == 4

    def test_extract_word_plus_digit(self) -> None:
        expr = _extract_math_expression("What is twenty plus 3?")
        assert expr is not None
        assert safe_math_eval(expr) == 23

    def test_extract_all_words(self) -> None:
        expr = _extract_math_expression("What is two plus three?")
        assert expr is not None
        assert safe_math_eval(expr) == 5

    # ── Extended NL tokens ───────────────────────────────────────

    def test_dozen(self) -> None:
        result = _normalize_natural_language("a dozen times 3")
        assert result is not None
        assert safe_math_eval(result) == 36

    def test_twice(self) -> None:
        result = _normalize_natural_language("twice five")
        assert result is not None
        assert safe_math_eval(result) == 10

    def test_half_of(self) -> None:
        result = _normalize_natural_language("half of 100")
        assert result is not None
        assert safe_math_eval(result) == 50

    def test_double(self) -> None:
        result = _normalize_natural_language("double 25")
        assert result is not None
        assert safe_math_eval(result) == 50

    def test_triple(self) -> None:
        result = _normalize_natural_language("triple 7")
        assert result is not None
        assert safe_math_eval(result) == 21

    def test_couple(self) -> None:
        result = _normalize_natural_language("a couple times ten")
        assert result is not None
        assert safe_math_eval(result) == 20


# ═════════════════════════════════════════════════════════════════════════════
# Static Factual Bypass
# ═════════════════════════════════════════════════════════════════════════════


class TestStaticFacts:
    def test_capital_of_france(self) -> None:
        assert _try_static_fact("What is the capital of France?") == "Paris"

    def test_capital_of_india(self) -> None:
        assert _try_static_fact("capital of India?") == "New Delhi"

    def test_how_many_continents(self) -> None:
        assert _try_static_fact("How many continents are there?") == "7"

    def test_how_many_planets(self) -> None:
        assert _try_static_fact("How many planets?") == "8"

    def test_tallest_mountain(self) -> None:
        result = _try_static_fact("What is the tallest mountain?")
        assert result is not None
        assert "Everest" in result

    def test_no_match_prose(self) -> None:
        assert _try_static_fact("Build a FastAPI app") is None

    def test_no_match_random(self) -> None:
        assert _try_static_fact("Tell me a joke") is None

    def test_capital_case_insensitive(self) -> None:
        assert _try_static_fact("CAPITAL OF JAPAN?") == "Tokyo"


# ═════════════════════════════════════════════════════════════════════════════
# Confidence Scoring
# ═════════════════════════════════════════════════════════════════════════════


class TestConfidenceScoring:
    def test_tools_score_build(self) -> None:
        score, labels = _score_tools("Build a FastAPI todo app")
        assert score > 0.5
        assert len(labels) > 0

    def test_tools_score_github(self) -> None:
        score, _labels = _score_tools("Create a pull request on GitHub")
        assert score >= 0.9

    def test_tools_score_simple_question(self) -> None:
        score, _ = _score_tools("Capital of France?")
        assert score == 0.0

    def test_qa_score_what_is(self) -> None:
        score, labels = _score_qa("What is the capital of France?")
        assert score >= 0.7
        assert len(labels) > 0

    def test_qa_score_who(self) -> None:
        score, _ = _score_qa("Who wrote Hamlet?")
        assert score >= 0.7

    def test_qa_score_build(self) -> None:
        score, _ = _score_qa("Build a REST API")
        assert score == 0.0

    def test_multiple_qa_signals_boost(self) -> None:
        """A question matching multiple QA patterns should score higher."""
        score_single, _ = _score_qa("Why?")
        score_multi, _ = _score_qa("What is the capital of India?")
        # "What is" matches what_question; "capital of" matches factual_lookup
        assert score_multi >= score_single


# ═════════════════════════════════════════════════════════════════════════════
# Needs Tools / Is Simple QA (backward compat)
# ═════════════════════════════════════════════════════════════════════════════


class TestNeedsTools:
    def test_build_code(self) -> None:
        assert _needs_tools("Build a FastAPI todo app") is True

    def test_github_pr(self) -> None:
        assert _needs_tools("Create a pull request on GitHub") is True

    def test_email(self) -> None:
        assert _needs_tools("Send an email to the team") is True

    def test_jira_ticket(self) -> None:
        assert _needs_tools("Create a Jira ticket for this bug") is True

    def test_research(self) -> None:
        # Pure research questions now route to fast_llm (narrowed pattern)
        # Only "research + build/implement" combos need the full pipeline
        assert _needs_tools("Research AI trends and build a dashboard") is True
        assert _needs_tools("Research AI trends in 2026") is False

    def test_deploy(self) -> None:
        assert _needs_tools("Deploy the service to production") is True

    def test_simple_question_no_tools(self) -> None:
        assert _needs_tools("Capital of France?") is False

    def test_factual_no_tools(self) -> None:
        assert _needs_tools("Who wrote Romeo and Juliet?") is False

    def test_geography_no_tools(self) -> None:
        assert _needs_tools("How many continents are there?") is False


class TestIsSimpleQA:
    def test_what_is(self) -> None:
        assert _is_simple_qa("What is the capital of France?") is True

    def test_who_is(self) -> None:
        assert _is_simple_qa("Who is the president of the USA?") is True

    def test_where_is(self) -> None:
        assert _is_simple_qa("Where is the Eiffel Tower?") is True

    def test_when_was(self) -> None:
        assert _is_simple_qa("When was the moon landing?") is True

    def test_how_many(self) -> None:
        assert _is_simple_qa("How many planets are in the solar system?") is True

    def test_is_question(self) -> None:
        assert _is_simple_qa("Is Python a compiled language?") is True

    def test_define(self) -> None:
        assert _is_simple_qa("Define photosynthesis") is True

    def test_capital_of(self) -> None:
        assert _is_simple_qa("capital of India?") is True

    def test_name_the(self) -> None:
        assert _is_simple_qa("Name the tallest mountain") is True

    def test_not_simple_build(self) -> None:
        assert _is_simple_qa("Build a REST API") is False

    def test_not_simple_bare_statement(self) -> None:
        assert _is_simple_qa("Optimize the database queries") is False


# ═════════════════════════════════════════════════════════════════════════════
# Full Gate — bypass / fast_llm / full pipeline
# ═════════════════════════════════════════════════════════════════════════════


class TestComplexityGate:
    def setup_method(self) -> None:
        self.gate = ComplexityGate()

    # ── Math bypass ──────────────────────────────────────────────

    def test_bypass_simple_addition(self) -> None:
        d = self.gate.evaluate("What is 2 + 2?")
        assert d.bypass is True
        assert d.answer == "4"
        assert d.method == "safe_math"
        assert d.confidence == 1.0

    def test_bypass_multiplication(self) -> None:
        d = self.gate.evaluate("15 * 3")
        assert d.bypass is True
        assert d.answer == "45"

    def test_bypass_sqrt(self) -> None:
        d = self.gate.evaluate("what is sqrt(144)?")
        assert d.bypass is True
        assert d.answer == "12"

    def test_bypass_factorial(self) -> None:
        d = self.gate.evaluate("factorial(5)")
        assert d.bypass is True
        assert d.answer == "120"

    def test_bypass_complex_expression(self) -> None:
        d = self.gate.evaluate("What is 2 + 2 * 3?")
        assert d.bypass is True
        assert d.answer == "8"

    def test_bypass_exponentiation(self) -> None:
        d = self.gate.evaluate("2^10")
        assert d.bypass is True
        assert d.answer == "1024"

    # ── NL math bypass ───────────────────────────────────────────

    def test_bypass_word_plus_digit(self) -> None:
        d = self.gate.evaluate("What is two + 2?")
        assert d.bypass is True
        assert d.answer == "4"

    def test_bypass_all_word_numbers(self) -> None:
        d = self.gate.evaluate("What is two plus three?")
        assert d.bypass is True
        assert d.answer == "5"

    def test_bypass_word_times_word(self) -> None:
        d = self.gate.evaluate("five times six")
        assert d.bypass is True
        assert d.answer == "30"

    def test_bypass_twenty_plus_digit(self) -> None:
        d = self.gate.evaluate("What is twenty plus 3?")
        assert d.bypass is True
        assert d.answer == "23"

    def test_bypass_sqrt_words(self) -> None:
        d = self.gate.evaluate("square root of sixteen")
        assert d.bypass is True
        assert d.answer == "4"

    def test_bypass_how_much_word_times(self) -> None:
        d = self.gate.evaluate("How much is twelve times twelve?")
        assert d.bypass is True
        assert d.answer == "144"

    def test_bypass_hundred_divided(self) -> None:
        d = self.gate.evaluate("what is five hundred divided by twenty five?")
        assert d.bypass is True
        assert d.answer == "20"

    def test_bypass_factorial_words(self) -> None:
        d = self.gate.evaluate("factorial of five")
        assert d.bypass is True
        assert d.answer == "120"

    # ── Extended NL math bypass ──────────────────────────────────

    def test_bypass_twice(self) -> None:
        d = self.gate.evaluate("twice five")
        assert d.bypass is True
        assert d.answer == "10"

    def test_bypass_half_of(self) -> None:
        d = self.gate.evaluate("half of 100")
        assert d.bypass is True
        assert d.answer == "50"

    def test_bypass_dozen(self) -> None:
        d = self.gate.evaluate("a dozen times 3")
        assert d.bypass is True
        assert d.answer == "36"

    # ── Static fact bypass ───────────────────────────────────────

    def test_bypass_capital_of_france(self) -> None:
        d = self.gate.evaluate("What is the capital of France?")
        # Math extraction for "What is X" tries first but won't parse "capital of France"
        # so static fact kicks in
        assert d.bypass is True
        assert d.answer == "Paris"
        assert d.method == "static_fact"

    def test_bypass_how_many_continents(self) -> None:
        d = self.gate.evaluate("How many continents are there?")
        assert d.bypass is True
        assert d.answer == "7"

    def test_bypass_capital_of_india(self) -> None:
        d = self.gate.evaluate("capital of India?")
        assert d.bypass is True
        assert d.answer == "New Delhi"

    # ── Fast LLM ─────────────────────────────────────────────────

    def test_fast_llm_who_wrote(self) -> None:
        d = self.gate.evaluate("Who wrote Hamlet?")
        assert d.bypass is False
        assert d.fast_llm is True
        assert d.confidence >= 0.7

    def test_fast_llm_define(self) -> None:
        d = self.gate.evaluate("Define quantum entanglement")
        assert d.bypass is False
        assert d.fast_llm is True

    def test_fast_llm_is_question(self) -> None:
        d = self.gate.evaluate("Is water wet?")
        assert d.bypass is False
        assert d.fast_llm is True

    def test_fast_llm_translate(self) -> None:
        d = self.gate.evaluate("Translate hello to Spanish")
        assert d.bypass is False
        assert d.fast_llm is True

    def test_fast_llm_where_is(self) -> None:
        d = self.gate.evaluate("Where is Mount Everest?")
        assert d.bypass is False
        assert d.fast_llm is True

    def test_fast_llm_when_was(self) -> None:
        d = self.gate.evaluate("When was the French Revolution?")
        assert d.bypass is False
        assert d.fast_llm is True

    # ── Full pipeline ────────────────────────────────────────────

    def test_full_pipeline_build(self) -> None:
        d = self.gate.evaluate("Build a FastAPI todo app with auth")
        assert d.bypass is False
        assert d.fast_llm is False
        assert d.reason == "requires_llm"

    def test_full_pipeline_github(self) -> None:
        d = self.gate.evaluate("Create a pull request on GitHub")
        assert d.bypass is False
        assert d.fast_llm is False

    def test_full_pipeline_research(self) -> None:
        d = self.gate.evaluate("Research AI trends and write a report")
        assert d.bypass is False
        assert d.fast_llm is False

    def test_full_pipeline_email(self) -> None:
        d = self.gate.evaluate("Send an email to the team about the sprint")
        assert d.bypass is False
        assert d.fast_llm is False

    def test_full_pipeline_gitlab_issues(self) -> None:
        d = self.gate.evaluate("List my GitLab projects under anuragbali")
        assert d.bypass is False
        assert d.fast_llm is False

    # ── Edge cases ───────────────────────────────────────────────

    def test_empty_task(self) -> None:
        d = self.gate.evaluate("")
        assert d.bypass is False
        assert d.reason == "empty_task"

    def test_decision_is_frozen(self) -> None:
        d = self.gate.evaluate("What is 2 + 2?")
        with pytest.raises(AttributeError):
            d.bypass = False  # type: ignore[misc]

    def test_returns_gate_decision_type(self) -> None:
        d = self.gate.evaluate("hello")
        assert isinstance(d, GateDecision)

    def test_decision_has_confidence(self) -> None:
        d = self.gate.evaluate("Who is Einstein?")
        assert 0.0 <= d.confidence <= 1.0

    def test_decision_has_matched_labels(self) -> None:
        d = self.gate.evaluate("Build a FastAPI app")
        assert len(d.matched_labels) > 0

    def test_default_fast_llm_is_false(self) -> None:
        d = GateDecision(bypass=False, answer="", reason="test")
        assert d.fast_llm is False
        assert d.confidence == 0.0
        assert d.matched_labels == ()

    def test_math_bypass_trumps_fast_llm(self) -> None:
        """Math should fully bypass, not just fast_llm."""
        d = self.gate.evaluate("What is 10 + 5?")
        assert d.bypass is True
        assert d.fast_llm is False

    # ── research_cached tier ──────────────────────────────────────────

    def test_research_cached_defaults_false(self) -> None:
        """GateDecision has research_cached=False by default."""
        d = GateDecision(bypass=False, answer="", reason="test")
        assert d.research_cached is False
        assert d.research_context == ""

    def test_research_cached_gate_decision_fields(self) -> None:
        """GateDecision can carry research_cached + research_context."""
        d = GateDecision(
            bypass=False,
            answer="",
            reason="research_cached",
            method="fast_llm",
            fast_llm=True,
            research_cached=True,
            research_context="Cached summary here.",
            confidence=0.8,
        )
        assert d.fast_llm is True
        assert d.research_cached is True
        assert d.research_context == "Cached summary here."

    def test_research_cached_tier_fires_on_cache_hit(self, tmp_path) -> None:
        """When keyword scores are low and ResearchCache has a hit, gate returns research_cached."""
        from unittest.mock import MagicMock, patch

        # Build a real ResearchCache instance with a stored entry
        from aifactory.core.research_cache import ResearchCache

        rc = ResearchCache(cache_dir=tmp_path)
        rc.store(
            "aifactory internals configuration", "Cached summary about aifactory…", run_id="r1"
        )

        # Force both keyword scores to 0 so the gate reaches Stage 7.
        # ResearchCache is imported inside evaluate() as a local import, so we
        # patch it at the source module to control the instance it creates.
        rc_cls_mock = MagicMock(return_value=rc)

        with (
            patch("aifactory.core.complexity_gate._score_tools", return_value=(0.0, [])),
            patch("aifactory.core.complexity_gate._score_qa", return_value=(0.0, [])),
            patch(
                "aifactory.core.complexity_gate._sem_classify",
                return_value=("full", 0.0),
                create=True,
            ),
            patch("aifactory.core.research_cache.ResearchCache", rc_cls_mock),
        ):
            decision = self.gate.evaluate("aifactory internals configuration")

        assert isinstance(decision, GateDecision)
        assert decision.research_cached is True
        assert decision.fast_llm is True
        assert decision.reason == "research_cached"
        assert (
            "aifactory" in decision.research_context.lower() or len(decision.research_context) > 0
        )

    def test_research_cached_not_triggered_for_bypass_tasks(self) -> None:
        """Math bypass tasks are never routed through research_cached tier."""
        d = self.gate.evaluate("What is 100 * 42?")
        assert d.bypass is True
        assert d.research_cached is False


# ═════════════════════════════════════════════════════════════════════════════
# Stage -1: Session-aware continuation detection
# ═════════════════════════════════════════════════════════════════════════════


class TestSessionContinuation:
    gate = ComplexityGate()

    def test_yes_after_full_routes_to_full(self) -> None:
        """'yes' after a full-pipeline run should inherit full route."""
        d = self.gate.evaluate("yes", prior_route="full")
        assert d.bypass is False
        assert d.fast_llm is False
        assert d.reason == "session_continuation"

    def test_sure_after_full_routes_to_full(self) -> None:
        d = self.gate.evaluate("sure", prior_route="full")
        assert d.bypass is False
        assert d.fast_llm is False
        assert d.reason == "session_continuation"

    def test_continue_after_full_routes_to_full(self) -> None:
        d = self.gate.evaluate("continue", prior_route="full")
        assert d.bypass is False
        assert d.fast_llm is False
        assert d.reason == "session_continuation"

    def test_ok_after_full_routes_to_full(self) -> None:
        d = self.gate.evaluate("ok", prior_route="full")
        assert d.bypass is False
        assert d.fast_llm is False

    def test_elaborate_after_full_routes_to_full(self) -> None:
        d = self.gate.evaluate("elaborate", prior_route="full")
        assert d.bypass is False
        assert d.fast_llm is False
        assert d.reason == "session_continuation"

    def test_yes_without_prior_route_hits_fast_llm(self) -> None:
        """'yes' with no prior_route should not be forced to full pipeline."""
        d = self.gate.evaluate("yes", prior_route=None)
        # Should route to fast_llm or bypass, NOT full pipeline via continuation
        assert d.reason != "session_continuation"

    def test_yes_after_fast_llm_not_promoted(self) -> None:
        """'yes' after a fast_llm run should not be promoted to full pipeline."""
        d = self.gate.evaluate("yes", prior_route="fast_llm")
        assert d.reason != "session_continuation"

    def test_long_message_after_full_not_forced(self) -> None:
        """Messages >= 120 chars after full run are evaluated normally."""
        long_task = "yes " + "x" * 120
        d = self.gate.evaluate(long_task, prior_route="full")
        # Should NOT be caught by Stage -1 (too long)
        assert d.reason != "session_continuation"

    def test_continuation_confidence(self) -> None:
        """Continuation decisions carry 0.85 confidence."""
        d = self.gate.evaluate("go ahead", prior_route="full")
        assert d.reason == "session_continuation"
        assert d.confidence == 0.85


# ═════════════════════════════════════════════════════════════════════════════
# Stage 4b: Project-state queries → always full pipeline
# ═════════════════════════════════════════════════════════════════════════════


class TestProjectStateSignals:
    gate = ComplexityGate()

    def test_what_has_been_done_routes_full(self) -> None:
        d = self.gate.evaluate("what has been done so far?")
        assert d.bypass is False
        assert d.fast_llm is False
        assert d.reason == "project_state_query"

    def test_where_were_we_routes_full(self) -> None:
        d = self.gate.evaluate("where were we?")
        assert d.bypass is False
        assert d.fast_llm is False
        assert d.reason == "project_state_query"

    def test_what_is_left_routes_full(self) -> None:
        d = self.gate.evaluate("what's left to do?")
        assert d.bypass is False
        assert d.fast_llm is False
        assert d.reason == "project_state_query"

    def test_catch_me_up_routes_full(self) -> None:
        d = self.gate.evaluate("catch me up on the project")
        assert d.bypass is False
        assert d.fast_llm is False
        assert d.reason == "project_state_query"

    def test_progress_on_routes_full(self) -> None:
        d = self.gate.evaluate("what's the progress on the auth module?")
        assert d.bypass is False
        assert d.fast_llm is False
        assert d.reason == "project_state_query"

    def test_summarise_the_routes_full(self) -> None:
        d = self.gate.evaluate("summarise the work done today")
        assert d.bypass is False
        assert d.fast_llm is False
        assert d.reason == "project_state_query"
