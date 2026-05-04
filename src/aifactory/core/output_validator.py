"""
Worker output quality gate.

Detects clearly invalid worker responses before they enter task_results:
  - Empty / whitespace-only output
  - Responses that are shorter than MIN_CHARS
  - LLM refusal / safety phrases
  - Evasive non-answers (generic templates, "without access" disclaimers)

Called by _run_worker_task() in dynamic_pipeline.py after the self-correction
loop, before the result_entry dict is assembled.  On failure, the worker is
given one retry with an explicit "your previous response was invalid because X"
injection.  One retry max — result is accepted either way (with quality_flagged
set on the result entry) to avoid infinite loops.
"""

from __future__ import annotations

# Minimum meaningful response length in chars
MIN_CHARS: int = 20

# Phrases that indicate an LLM refusal or safety block.
# Checked case-insensitively against the START of the response.
REFUSAL_PREFIXES: tuple[str, ...] = (
    "i cannot",
    "i'm unable",
    "i am unable",
    "as an ai",
    "as a language model",
    "i don't have",
    "i do not have",
    "i can't help",
    "i cannot help",
    "i'm not able",
    "i am not able",
    "i'm sorry, but i",
    "i apologize, but i",
    "i apologize for my previous",
    "since i don't have",
    "since i do not have",
    "without access to",
    "without seeing the",
    "without viewing the",
    "i require the content",
    "i need access to",
    "i would need access",
    "i don't have direct",
    "i do not have direct",
)

# Phrases that indicate a generic/evasive response anywhere in the text.
# These are checked as substrings (not just prefixes) and weighted — if 2+
# are found together the response is flagged as a non-answer.
_EVASION_SIGNALS: tuple[str, ...] = (
    "without access to the actual codebase",
    "without access to the actual code",
    "if you provide specific files",
    "if you share the code",
    "general approach",
    "general outline",
    "common software engineering principles",
    "best practices for fastapi",
    "potential flaws to look for",
    "this is a general",
    "based on common patterns",
    "i will outline a general",
    "framework even without seeing",
    "much more detailed and accurate analysis",
)


def validate_worker_output(result: str, task_description: str = "") -> tuple[bool, str]:
    """
    Validate worker output quality.

    Returns:
        (is_valid, rejection_reason)
        is_valid=True  — output is acceptable, proceed normally.
        is_valid=False — rejection_reason explains why; caller should retry.
    """
    stripped = result.strip()

    # 1. Empty response
    if not stripped:
        return False, "empty response"

    # 2. Too short to be meaningful
    if len(stripped) < MIN_CHARS:
        return False, f"response too short ({len(stripped)} chars, minimum {MIN_CHARS})"

    lower = stripped.lower()

    # 3. Hard refusal — matches start of response
    for prefix in REFUSAL_PREFIXES:
        if lower.startswith(prefix):
            return False, f"refusal detected: '{stripped[:80].replace(chr(10), ' ')}'"

    # 4. Evasive non-answer — generic template instead of real work
    # Flag if 2+ evasion signals appear anywhere in the response
    evasion_hits = sum(1 for sig in _EVASION_SIGNALS if sig in lower)
    if evasion_hits >= 2:
        return False, (
            f"evasive non-answer detected ({evasion_hits} evasion signals) — "
            "response is a generic template rather than actual task output; "
            "use CODE_ANALYSIS: to access the codebase"
        )

    return True, ""
