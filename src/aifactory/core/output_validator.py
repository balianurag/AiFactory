"""
Worker output quality gate.

Detects clearly invalid worker responses before they enter task_results:
  - Empty / whitespace-only output
  - Responses that are shorter than MIN_CHARS
  - LLM refusal / safety phrases

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
# Checked case-insensitively against the start of the response.
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
)


def validate_worker_output(result: str, task_description: str = "") -> tuple[bool, str]:
    """
    Validate worker output quality.

    Args:
        result:           The worker's response text.
        task_description: Used only for context in log messages.

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

    # 3. LLM refusal / safety block
    lower = stripped.lower()
    for prefix in REFUSAL_PREFIXES:
        if lower.startswith(prefix):
            return False, f"refusal detected: '{stripped[:80].replace(chr(10), ' ')}'"

    return True, ""
