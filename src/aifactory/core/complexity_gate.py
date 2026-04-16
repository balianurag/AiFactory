"""
Complexity Gate v2 — YAML-driven, confidence-scored task classifier.

Three-tier routing:

  1. **Bypass** (zero-cost, <1ms):
     Pure arithmetic / NL math — no LLM needed.
     "2 + 2", "five times six", "a dozen times 3"

  2. **Static factual** (zero-cost, <1ms):
     Geography, science, and common facts answered from a built-in dict.
     "Capital of France?", "How many continents?"

  3. **Fast LLM** (single call, ~2s):
     Simple Q&A — one master LLM call, no Composio/tools/agents.
     "Who wrote Hamlet?", "Define photosynthesis"

  4. **Full pipeline** (~15-30s):
     Complex tasks needing tools, multi-agent orchestration, or research.

Patterns are loaded from config/gate_patterns.yaml — edit YAML, not code.
Each match contributes a weighted confidence score; the final decision is
based on which side (tools vs QA) accumulates more confidence.

Phase-2 placeholder: lightweight LLM classifier for uncertain cases.

Usage:
    gate = ComplexityGate()
    decision = gate.evaluate(task)
    if decision.bypass:       ...  # computed answer
    elif decision.fast_llm:   ...  # single LLM call
    else:                     ...  # full pipeline
"""

from __future__ import annotations

import ast
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ─── Config path ────────────────────────────────────────────────────────────

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_GATE_PATTERNS_PATH = _CONFIG_DIR / "gate_patterns.yaml"


# ═════════════════════════════════════════════════════════════════════════════
# GateDecision
# ═════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class GateDecision:
    """Immutable result of the complexity gate evaluation."""

    bypass: bool
    """If True, the answer is already computed — skip the full pipeline."""

    answer: str
    """The computed answer (only meaningful when bypass=True)."""

    reason: str
    """Why the gate made this decision."""

    method: str = ""
    """How the answer was computed (e.g. 'safe_math', 'static_fact')."""

    fast_llm: bool = False
    """If True, use single master LLM call — skip tools & multi-agent."""

    confidence: float = 0.0
    """Confidence score (0.0–1.0) for the routing decision."""

    matched_labels: tuple[str, ...] = ()
    """Labels of the patterns that fired (for debugging / dumps)."""


# ═════════════════════════════════════════════════════════════════════════════
# Safe Math Evaluator (AST-based)
# ═════════════════════════════════════════════════════════════════════════════

_SAFE_NODE_NAMES = {
    "Expression",
    "Constant",
    "BinOp",
    "UnaryOp",
    "Add",
    "Sub",
    "Mult",
    "Div",
    "FloorDiv",
    "Mod",
    "Pow",
    "USub",
    "UAdd",
}

_SAFE_FUNCS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "int": int,
    "float": float,
    "sqrt": math.sqrt,
    "factorial": math.factorial,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "ceil": math.ceil,
    "floor": math.floor,
    "pow": pow,
    "pi": math.pi,
    "e": math.e,
}


def _is_safe_ast(node: ast.AST) -> bool:
    """Recursively check that an AST tree only contains safe nodes."""
    name = type(node).__name__
    if name in _SAFE_NODE_NAMES:
        return all(_is_safe_ast(c) for c in ast.iter_child_nodes(node))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _SAFE_FUNCS
    ):
        return all(_is_safe_ast(a) for a in node.args)
    return bool(isinstance(node, ast.Name) and node.id in _SAFE_FUNCS)


def safe_math_eval(expr: str) -> float | int | None:
    """
    Safely evaluate a mathematical expression using AST parsing.

    Returns None if the expression is unsafe or invalid.
    """
    try:
        tree = ast.parse(expr.strip(), mode="eval")
        if not _is_safe_ast(tree):
            return None
        code = compile(tree, "<math>", "eval")
        result = eval(code, {"__builtins__": {}}, _SAFE_FUNCS)
        if isinstance(result, int | float | complex):
            if isinstance(result, float) and result == int(result) and not math.isinf(result):
                return int(result)
            return result
        return None
    except Exception:
        return None


# ═════════════════════════════════════════════════════════════════════════════
# Natural Language Math — word-to-digit conversion
# ═════════════════════════════════════════════════════════════════════════════

_WORD_TO_NUM: dict[str, int | float] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
    "thousand": 1000,
    "million": 1_000_000,
    # Extended NL tokens
    "dozen": 12,
    "couple": 2,
    "pair": 2,
    "score": 20,
}

# Multiplier words handled specially before other ops
_MULTIPLIER_WORDS: dict[str, float] = {
    "twice": 2,
    "thrice": 3,
    "double": 2,
    "triple": 3,
    "quadruple": 4,
    "half": 0.5,
    "quarter": 0.25,
    "third": 1 / 3,
}

_WORD_TO_OP: dict[str, str] = {
    "plus": "+",
    "add": "+",
    "added to": "+",
    "and": "+",
    "minus": "-",
    "subtract": "-",
    "less": "-",
    "times": "*",
    "multiply": "*",
    "multiplied by": "*",
    "divided by": "/",
    "divide": "/",
    "over": "/",
    "to the power of": "**",
    "raised to": "**",
    "power": "**",
    "squared": "** 2",
    "cubed": "** 3",
    "mod": "%",
    "modulo": "%",
}

_WORD_TO_FUNC: dict[str, str] = {
    "square root of": "sqrt",
    "the square root of": "sqrt",
    "factorial of": "factorial",
    "the factorial of": "factorial",
    "log of": "log",
    "the log of": "log",
}


def _normalize_natural_language(text: str) -> str | None:
    """Convert natural language math into a numeric expression.

    Returns None if no number words are found.
    """
    text = text.strip().lower()
    text = re.sub(
        r"^(?:what(?:'?s| is| are)|calculate|compute|evaluate|solve|find|how much is)\s+",
        "",
        text,
    )
    text = re.sub(r"[\?\.!\s]+$", "", text)
    if not text:
        return None

    # "a dozen" / "a couple" → number
    text = re.sub(r"\ba\s+(dozen|couple|pair|score)\b", r"\1", text)

    # Handle multiplier words: "twice 5" → "2 * 5", "half of 10" → "0.5 * 10"
    for word, factor in _MULTIPLIER_WORDS.items():
        pattern = rf"\b{word}\s+(?:of\s+)?(.+)"
        m = re.match(pattern, text)
        if m:
            rest = m.group(1)
            # Substitute operators in the rest too: "50 plus a dozen" → "50 + 12"
            for phrase, op in sorted(_WORD_TO_OP.items(), key=lambda x: -len(x[0])):
                rest = rest.replace(phrase, f" {op} ")
            rest = re.sub(r"\ba\s+(dozen|couple|pair|score)\b", r"\1", rest)
            rest_normalized = _words_to_digits(rest)
            rest_normalized = re.sub(r"\s+", " ", rest_normalized).strip()
            if re.search(r"\d", rest_normalized):
                return f"{factor} * {rest_normalized}"

    # Function-style: "square root of sixteen"
    for phrase, func in _WORD_TO_FUNC.items():
        if phrase in text:
            arg_text = text.split(phrase, 1)[1].strip()
            arg_normalized = _words_to_digits(arg_text)
            if arg_normalized and re.search(r"\d", arg_normalized):
                return f"{func}({arg_normalized})"

    # Replace multi-word operators (longest first)
    result = text
    for phrase, op in sorted(_WORD_TO_OP.items(), key=lambda x: -len(x[0])):
        result = result.replace(phrase, f" {op} ")

    result = _words_to_digits(result)
    result = re.sub(r"\s+", " ", result).strip()

    if re.search(r"\d", result) and re.search(r"[\+\-\*/\%\*]", result):
        return result
    return None


def _words_to_digits(text: str) -> str:
    """Replace number words with their digit equivalents.

    Handles compounds: "twenty three" → 23, "five hundred" → 500.
    """
    tokens = text.split()
    result_tokens: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i].strip(",.;:!?")
        if token in _WORD_TO_NUM:
            num = _WORD_TO_NUM[token]
            if isinstance(num, float):
                result_tokens.append(str(num))
                i += 1
                continue
            # Compound: "twenty three" → 23
            if 20 <= num < 100 and i + 1 < len(tokens):
                nxt = tokens[i + 1].strip(",.;:!?")
                if (
                    nxt in _WORD_TO_NUM
                    and isinstance(_WORD_TO_NUM[nxt], int)
                    and _WORD_TO_NUM[nxt] < 10
                ):
                    num += _WORD_TO_NUM[nxt]
                    i += 1
            elif num < 10 and i + 1 < len(tokens):
                nxt = tokens[i + 1].strip(",.;:!?")
                if nxt == "hundred":
                    num *= 100
                    i += 1
                    if i + 1 < len(tokens):
                        rest = tokens[i + 1].strip(",.;:!?")
                        if rest in _WORD_TO_NUM and isinstance(_WORD_TO_NUM[rest], int):
                            if _WORD_TO_NUM[rest] >= 20:
                                num += _WORD_TO_NUM[rest]
                                i += 1
                                if i + 1 < len(tokens):
                                    u = tokens[i + 1].strip(",.;:!?")
                                    if (
                                        u in _WORD_TO_NUM
                                        and isinstance(_WORD_TO_NUM[u], int)
                                        and _WORD_TO_NUM[u] < 10
                                    ):
                                        num += _WORD_TO_NUM[u]
                                        i += 1
                            elif _WORD_TO_NUM[rest] < 20:
                                num += _WORD_TO_NUM[rest]
                                i += 1
                elif nxt == "thousand":
                    num *= 1000
                    i += 1
            result_tokens.append(str(num))
        else:
            result_tokens.append(tokens[i])
        i += 1
    return " ".join(result_tokens)


# ═════════════════════════════════════════════════════════════════════════════
# Math Expression Extraction
# ═════════════════════════════════════════════════════════════════════════════

_EXTRACTION_PATTERNS = [
    re.compile(r"(?:what(?:'?s| is|are))\s+(.+?)[\?\.!]*$", re.I),
    re.compile(r"(?:calculate|compute|evaluate|solve|find)\s+(.+?)[\?\.!]*$", re.I),
    re.compile(r"how much is\s+(.+?)[\?\.!]*$", re.I),
    re.compile(r"^(\d[\d\s\+\-\*/\.\(\)\^%]+\d)[\?\.!]*$"),
]


def _extract_math_expression(task: str) -> str | None:
    """Try to extract a mathematical expression from a task string."""
    task = task.strip()
    if not task:
        return None

    for pattern in _EXTRACTION_PATTERNS:
        m = pattern.match(task)
        if m:
            expr = m.group(1).strip().replace("^", "**")
            has_number_words = any(w in _WORD_TO_NUM for w in re.split(r"[\s,]+", expr.lower()))
            if has_number_words:
                normalized = _normalize_natural_language(expr)
                if normalized is not None:
                    return normalized
            if re.search(r"\d", expr):
                return expr

    normalized = _normalize_natural_language(task)
    if normalized is not None:
        return normalized

    cleaned = re.sub(r"[\?\.!]+$", "", task).strip().replace("^", "**")
    if re.search(r"\d", cleaned) and len(cleaned) < 100 and safe_math_eval(cleaned) is not None:
        return cleaned
    # Bare math constants: "pi", "e"
    if cleaned.strip() in _SAFE_FUNCS and isinstance(_SAFE_FUNCS.get(cleaned.strip()), int | float):
        return cleaned.strip()
    return None


# ═════════════════════════════════════════════════════════════════════════════
# Unit Conversion Bypass — zero-cost, no LLM needed
# ═════════════════════════════════════════════════════════════════════════════

_UNIT_ALIASES: dict[str, str] = {
    # Temperature
    "celsius": "c",
    "°c": "c",
    "centigrade": "c",
    "degc": "c",
    "fahrenheit": "f",
    "°f": "f",
    "degf": "f",
    "kelvin": "k",
    "kelvins": "k",
    # Mass
    "kilogram": "kg",
    "kilograms": "kg",
    "kilo": "kg",
    "pound": "lb",
    "pounds": "lb",
    "lbs": "lb",
    "gram": "g",
    "grams": "g",
    # Distance
    "kilometer": "km",
    "kilometers": "km",
    "kilometre": "km",
    "kilometres": "km",
    "mile": "miles",
    "mi": "miles",
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",
    "centimeter": "cm",
    "centimeters": "cm",
    "centimetre": "cm",
    "centimetres": "cm",
    "inch": "in",
    "inches": "in",
    "foot": "ft",
    "feet": "ft",
}

# Linear conversions (result = value * factor)
_LINEAR_CONVERSIONS: dict[tuple[str, str], float] = {
    ("kg", "lb"): 2.20462,
    ("lb", "kg"): 0.453592,
    ("km", "miles"): 0.621371,
    ("miles", "km"): 1.60934,
    ("m", "ft"): 3.28084,
    ("ft", "m"): 0.3048,
    ("cm", "in"): 0.393701,
    ("in", "cm"): 2.54,
    ("kg", "g"): 1000.0,
    ("g", "kg"): 0.001,
    ("miles", "m"): 1609.34,
    ("m", "miles"): 0.000621371,
    ("km", "m"): 1000.0,
    ("m", "km"): 0.001,
}

# Temperature conversions (need offsets — stored as callables)
_TEMP_FNS: dict[tuple[str, str], Any] = {
    ("c", "f"): lambda x: x * 9 / 5 + 32,
    ("f", "c"): lambda x: (x - 32) * 5 / 9,
    ("c", "k"): lambda x: x + 273.15,
    ("k", "c"): lambda x: x - 273.15,
    ("f", "k"): lambda x: (x - 32) * 5 / 9 + 273.15,
    ("k", "f"): lambda x: (x - 273.15) * 9 / 5 + 32,
}

# Regex to detect "VALUE UNIT to UNIT" patterns
_UNIT_PART = (
    r"celsius|fahrenheit|kelvin|centigrade|°c|°f|°k"
    r"|kilograms?|kilos?|pounds?|lbs?|grams?"
    r"|kilometers?|kilometres?|miles?|meters?|metres?"
    r"|centimeters?|centimetres?|inches|inch|feet|foot"
    r"|kg|km|cm|in|ft|mi|lb|[cfk]"
)
_UNIT_CONVERT_RE = re.compile(
    rf"^(?:convert\s+|what(?:'?s| is)\s+)?(-?\d+(?:\.\d+)?)\s*"
    rf"({_UNIT_PART})\s+(?:to|in|into)\s+({_UNIT_PART})\s*\??$",
    re.I,
)


def _fmt_number(x: float) -> str:
    """Format number cleanly: 105.0 → '105', 37.78 → '37.78'."""
    rounded = round(x, 4)
    if rounded == int(rounded):
        return str(int(rounded))
    return str(rounded)


def _try_unit_conversion(task: str) -> str | None:
    """Compute a unit conversion without LLM.

    Examples: '100C to F', 'convert 5 kg to lb', '10 miles in km'
    Returns a formatted answer string or None if not applicable.
    """
    text = task.strip()
    # Strip leading question prefixes
    text = re.sub(r"^(?:what(?:'?s| is)\s+|how (?:many|much) (?:is|are)\s+)", "", text, flags=re.I)
    m = _UNIT_CONVERT_RE.match(text.strip())
    if not m:
        return None

    value = float(m.group(1))
    from_raw = m.group(2).lower().strip()
    to_raw = m.group(3).lower().strip()

    from_unit = _UNIT_ALIASES.get(from_raw, from_raw)
    to_unit = _UNIT_ALIASES.get(to_raw, to_raw)

    if from_unit == to_unit:
        return f"{value} {from_raw} = {value} {to_raw}"

    # Temperature
    temp_fn = _TEMP_FNS.get((from_unit, to_unit))
    if temp_fn:
        return f"{value} {from_raw} = {_fmt_number(temp_fn(value))} {to_raw}"

    # Linear
    factor = _LINEAR_CONVERSIONS.get((from_unit, to_unit))
    if factor is not None:
        return f"{value} {from_raw} = {_fmt_number(value * factor)} {to_unit}"

    return None


# ═════════════════════════════════════════════════════════════════════════════
# Static Factual Bypass — zero-cost answers from built-in knowledge
# ═════════════════════════════════════════════════════════════════════════════

# Normalised lookup: key → answer.  Keys are lowercase, stripped.
_STATIC_FACTS: dict[str, str] = {
    # ── Capitals ───────────────────────────────────────────────
    "capital of india": "New Delhi",
    "capital of france": "Paris",
    "capital of germany": "Berlin",
    "capital of japan": "Tokyo",
    "capital of china": "Beijing",
    "capital of usa": "Washington, D.C.",
    "capital of united states": "Washington, D.C.",
    "capital of uk": "London",
    "capital of united kingdom": "London",
    "capital of brazil": "Brasília",
    "capital of australia": "Canberra",
    "capital of canada": "Ottawa",
    "capital of russia": "Moscow",
    "capital of italy": "Rome",
    "capital of spain": "Madrid",
    "capital of mexico": "Mexico City",
    "capital of south korea": "Seoul",
    "capital of indonesia": "Jakarta",
    "capital of turkey": "Ankara",
    "capital of egypt": "Cairo",
    "capital of south africa": "Pretoria (executive), Cape Town (legislative), Bloemfontein (judicial)",  # noqa: E501
    "capital of pakistan": "Islamabad",
    "capital of bangladesh": "Dhaka",
    "capital of nigeria": "Abuja",
    "capital of argentina": "Buenos Aires",
    # ── Science / geography ────────────────────────────────────
    "how many continents": "7",
    "how many continents are there": "7",
    "how many oceans": "5",
    "how many oceans are there": "5",
    "how many planets": "8",
    "how many planets are there": "8",
    "how many planets in the solar system": "8",
    "speed of light": "299,792,458 m/s",
    "boiling point of water": "100 °C (212 °F)",
    "freezing point of water": "0 °C (32 °F)",
    # ── Common knowledge ───────────────────────────────────────
    "tallest mountain": "Mount Everest (8,849 m)",
    "tallest mountain in the world": "Mount Everest (8,849 m)",
    "longest river": "Nile (6,650 km) or Amazon (by volume)",
    "longest river in the world": "Nile (6,650 km) or Amazon (by volume)",
    "largest ocean": "Pacific Ocean",
    "largest country": "Russia (by area)",
    "largest country by area": "Russia",
    "largest country by population": "India",
    "smallest country": "Vatican City",
}


def _try_static_fact(task: str) -> str | None:
    """Look up a task in the static facts dictionary.

    Strips question wrappers and punctuation for a normalised match.
    """
    text = task.strip().lower()
    text = re.sub(
        r"^(?:what(?:'?s| is| are| was)?|where is|who is|tell me|name)\s+(?:the\s+)?",
        "",
        text,
    )
    text = re.sub(r"[\?\.!\s]+$", "", text).strip()
    return _STATIC_FACTS.get(text)


# ═════════════════════════════════════════════════════════════════════════════
# Compound Query Bypass — resolve multiple bypass-able parts in one shot
# ═════════════════════════════════════════════════════════════════════════════


def _try_compound_query(task: str) -> str | None:
    """Resolve compound queries that mix bypass-able parts.

    Examples:
      "capital of Germany + 15*7"  →  "capital of Germany: Berlin\\n15*7 = 105"
      "100C to F + capital of France"  →  "100 c = 212 f\\ncapital of France: Paris"

    Splits on ' + ' only when at least one part contains alphabetic text
    (pure arithmetic like '2 + 3' is handled by the math bypass tier).
    Returns None if any part cannot be resolved without an LLM.
    """
    parts = [p.strip() for p in task.split(" + ") if p.strip()]
    if len(parts) < 2:
        return None

    # Only activate when at least one part has alphabetic content
    # (pure math expressions are handled by the math bypass tier above)
    if not any(re.search(r"[a-zA-Z]", p) for p in parts):
        return None

    answers: list[str] = []
    for part in parts:
        # 1. Static fact
        fact = _try_static_fact(part)
        if fact is not None:
            answers.append(f"{part}: {fact}")
            continue

        # 2. Unit conversion
        conv = _try_unit_conversion(part)
        if conv is not None:
            answers.append(conv)
            continue

        # 3. Math expression
        expr = _extract_math_expression(part)
        if expr is not None:
            result = safe_math_eval(expr)
            if result is not None:
                answers.append(f"{part} = {result}")
                continue

        # This part requires an LLM — abort compound resolution
        return None

    return "\n".join(answers) if len(answers) >= 2 else None


# ═════════════════════════════════════════════════════════════════════════════
# YAML-driven Pattern Loader
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class _WeightedPattern:
    """A compiled regex with a confidence weight and label."""

    pattern: re.Pattern[str]
    weight: float
    label: str


def _load_patterns() -> tuple[list[_WeightedPattern], list[_WeightedPattern], float]:
    """Load patterns from gate_patterns.yaml.

    Returns (needs_tools_patterns, simple_qa_patterns, fast_llm_threshold).
    Falls back to empty lists if YAML is missing or malformed.
    """
    needs: list[_WeightedPattern] = []
    qa: list[_WeightedPattern] = []
    threshold = 0.7

    try:
        raw = yaml.safe_load(_GATE_PATTERNS_PATH.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            logger.warning("gate_patterns.yaml is not a dict — using defaults")
            return needs, qa, threshold

        threshold = float(raw.get("fast_llm_threshold", 0.7))

        for entry in raw.get("needs_tools", []):
            try:
                needs.append(
                    _WeightedPattern(
                        pattern=re.compile(entry["regex"], re.I),
                        weight=float(entry.get("weight", 0.5)),
                        label=entry.get("label", "?"),
                    )
                )
            except (re.error, KeyError) as exc:
                logger.warning("Bad needs_tools pattern %s: %s", entry, exc)

        for entry in raw.get("simple_qa", []):
            try:
                qa.append(
                    _WeightedPattern(
                        pattern=re.compile(entry["regex"], re.I),
                        weight=float(entry.get("weight", 0.5)),
                        label=entry.get("label", "?"),
                    )
                )
            except (re.error, KeyError) as exc:
                logger.warning("Bad simple_qa pattern %s: %s", entry, exc)

    except FileNotFoundError:
        logger.info("gate_patterns.yaml not found — using hardcoded fallbacks")
    except Exception as exc:
        logger.warning("Failed to load gate_patterns.yaml: %s", exc)

    return needs, qa, threshold


# Module-level load (cached — patterns don't change at runtime)
_TOOLS_PATTERNS, _QA_PATTERNS, _FAST_LLM_THRESHOLD = _load_patterns()

# ── Hardcoded fallbacks (used when YAML is empty/missing) ────────────────────
_FALLBACK_TOOLS: list[re.Pattern[str]] = [
    re.compile(r"\b(?:build|create|implement|develop|scaffold|deploy|refactor)\b", re.I),
    re.compile(r"\bwrites?(?!r|n)\b", re.I),
    re.compile(
        r"\b(?:github|gitlab|jira|notion|linear|repo(?:sitory)?|pull request|PR|issue|ticket|commit)\b",  # noqa: E501
        re.I,
    ),
    re.compile(r"\b(?:upload|download|read file|write file|save to|delete file)\b", re.I),
    re.compile(r"\b(?:send (?:an? )?email|gmail|send mail|schedule|meeting|google drive)\b", re.I),
    re.compile(
        r"\b(?:scrape|crawl|fetch from|api call|http request|endpoint|query db|query (?:the )?database|sql)\b",  # noqa: E501
        re.I,
    ),
    re.compile(r"\b(?:research|investigate|compare .+ (?:and|vs|with)|deep dive|explore)\b", re.I),
    re.compile(r"(?:^|\.\s)(?:run|execute|test|debug|install|setup|configure|migrate)\b", re.I),
    re.compile(r"\b(?:design|architect|plan|blueprint|roadmap|strategy)\b", re.I),
    re.compile(r"\b(?:and then|after that|step \d|first .+ then)\b", re.I),
]

_FALLBACK_QA: list[re.Pattern[str]] = [
    re.compile(r"^what(?:'?s| is| are| was| were| does| do| did)\b", re.I),
    re.compile(r"^who\b", re.I),
    re.compile(r"^where(?:'?s| is| are| was| were| do| does)\b", re.I),
    re.compile(r"^when(?:'?s| is| was| did| were| does| do)\b", re.I),
    re.compile(r"^which\b", re.I),
    re.compile(r"^why\b", re.I),
    re.compile(
        r"^how (?:many|much|old|tall|long|far|big|large|small|deep|fast|heavy|often)\b", re.I
    ),
    re.compile(r"^how (?:to|do (?:i|you|we)|can (?:i|you|we)|does)\b", re.I),
    re.compile(
        r"^(?:is|are|was|were|does|do|did|can|could|has|have|had|will|would|should)\b", re.I
    ),
    re.compile(r"^(?:name|list|tell me|give me|mention)\b", re.I),
    re.compile(
        r"\b(?:capital|president|prime minister|currency|language|population|flag|anthem|continent)\s+(?:of|for)\b",  # noqa: E501
        re.I,
    ),
    re.compile(r"^(?:define|explain|describe|summarize|translate|spell)\b", re.I),
    re.compile(
        r"\d+\s*(?:fahrenheit|celsius|kg|lb|km|miles?|meters?|feet|inches|cm|mm|oz|gallons?|liters?)\s+(?:in|to)\b",
        re.I,
    ),
]


# ═════════════════════════════════════════════════════════════════════════════
# Scoring helpers
# ═════════════════════════════════════════════════════════════════════════════


def _score_tools(task: str) -> tuple[float, list[str]]:
    """Score how likely the task needs tools.  Returns (confidence, labels)."""
    if _TOOLS_PATTERNS:
        matches = [(p.weight, p.label) for p in _TOOLS_PATTERNS if p.pattern.search(task)]
    else:
        matches = [(0.8, "fallback") for p in _FALLBACK_TOOLS if p.search(task)]
    if not matches:
        return 0.0, []
    weights = [w for w, _ in matches]
    labels = [lbl for _, lbl in matches]
    return min(max(weights) + 0.05 * (len(weights) - 1), 1.0), labels


def _score_qa(task: str) -> tuple[float, list[str]]:
    """Score how likely the task is simple Q&A.  Returns (confidence, labels)."""
    if _QA_PATTERNS:
        matches = [(p.weight, p.label) for p in _QA_PATTERNS if p.pattern.search(task)]
    else:
        matches = [(0.8, "fallback") for p in _FALLBACK_QA if p.search(task)]
    if not matches:
        return 0.0, []
    weights = [w for w, _ in matches]
    labels = [lbl for _, lbl in matches]
    return min(max(weights) + 0.05 * (len(weights) - 1), 1.0), labels


# Public helpers kept for backward compatibility & direct testing
def _needs_tools(task: str) -> bool:
    """Return True if the task likely needs external tools / multi-agent."""
    score, _ = _score_tools(task)
    return score > 0


def _is_simple_qa(task: str) -> bool:
    """Return True if the task is a simple factual / knowledge question."""
    score, _ = _score_qa(task)
    return score > 0


# ═════════════════════════════════════════════════════════════════════════════
# Complexity Gate
# ═════════════════════════════════════════════════════════════════════════════


class ComplexityGate:
    """
    YAML-driven, confidence-scored pre-pipeline gate.

    Routing tiers:
      1. bypass   — computed answer (math / static fact), zero cost
      2. fast_llm — single LLM call, no tools / multi-agent
      3. full     — 8-node pipeline + Composio
    """

    def evaluate(self, task: str) -> GateDecision:
        """Classify task complexity and return a routing decision."""
        task = task.strip()
        if not task:
            return GateDecision(bypass=False, answer="", reason="empty_task")

        # ── Stage 1: Pure math ──────────────────────────────────────
        expr = _extract_math_expression(task)
        if expr is not None:
            result = safe_math_eval(expr)
            if result is not None:
                return GateDecision(
                    bypass=True,
                    answer=str(result),
                    reason=f"pure_math: {expr} = {result}",
                    method="safe_math",
                    confidence=1.0,
                )

        # ── Stage 2: Unit conversion bypass ─────────────────────────
        conversion = _try_unit_conversion(task)
        if conversion is not None:
            return GateDecision(
                bypass=True,
                answer=conversion,
                reason="unit_conversion",
                method="unit_conversion",
                confidence=1.0,
            )

        # ── Stage 3: Static factual bypass ──────────────────────────
        fact = _try_static_fact(task)
        if fact is not None:
            return GateDecision(
                bypass=True,
                answer=fact,
                reason="static_fact",
                method="static_fact",
                confidence=1.0,
            )

        # ── Stage 4: Compound query bypass ───────────────────────────
        # Handles "capital of Germany + 15*7" — splits and resolves each part
        compound = _try_compound_query(task)
        if compound is not None:
            return GateDecision(
                bypass=True,
                answer=compound,
                reason="compound_bypass",
                method="compound_bypass",
                confidence=1.0,
            )

        # ── Stage 5: Confidence-scored classification ───────────────
        tools_score, tools_labels = _score_tools(task)
        qa_score, qa_labels = _score_qa(task)

        threshold = _FAST_LLM_THRESHOLD

        # Tools signal dominates → full pipeline
        if tools_score > 0 and tools_score >= qa_score:
            return GateDecision(
                bypass=False,
                answer="",
                reason="requires_llm",
                confidence=tools_score,
                matched_labels=tuple(tools_labels),
            )

        # Strong QA signal, no tools → fast path
        if qa_score >= threshold and tools_score == 0:
            return GateDecision(
                bypass=False,
                answer="",
                reason="simple_qa",
                method="fast_llm",
                fast_llm=True,
                confidence=qa_score,
                matched_labels=tuple(qa_labels),
            )

        # ── Stage 6: Semantic classifier for ambiguous queries ───────
        # When keyword signals are weak (both < threshold), use cosine
        # similarity to labelled examples as a second opinion.
        if tools_score < threshold and qa_score < threshold:
            try:
                from aifactory.core.semantic_classifier import (
                    classify as _sem_classify,
                )

                sem_label, sem_conf = _sem_classify(task, confidence_threshold=0.65)
                if sem_conf >= 0.65:
                    if sem_label == "bypass":
                        # Shouldn't happen here (stages 1-4 already handle bypass)
                        pass
                    elif sem_label == "fast_llm":
                        return GateDecision(
                            bypass=False,
                            answer="",
                            reason="simple_qa",
                            method="fast_llm",
                            fast_llm=True,
                            confidence=sem_conf,
                            matched_labels=("semantic_classifier",),
                        )
                    # sem_label == "full" → fall through to default below
            except Exception:
                pass  # semantic classifier is always optional

        # Weak QA signal OR no signal at all → full pipeline (safe default)
        return GateDecision(
            bypass=False,
            answer="",
            reason="requires_llm",
            confidence=max(tools_score, qa_score),
            matched_labels=tuple(tools_labels + qa_labels),
        )
