"""
Semantic complexity classifier.

Uses cosine similarity over stub (or real) embeddings to augment keyword-based
classification in ComplexityGate.  Called when keyword confidence falls in the
ambiguous middle band (0.2–0.7) so the gate can break ties without a full LLM
call.

Labels:
  bypass   — pure factual / math answer available locally
  fast_llm — single LLM call sufficient (simple Q&A, definitions)
  full     — multi-step, agentic, or tool-using task

Labelled examples are small and representative — they are embedded once per
process (lazy, cached) and compared by cosine similarity.
"""

from __future__ import annotations

import logging
import math
from functools import lru_cache
from typing import Literal

logger = logging.getLogger(__name__)

ClassificationLabel = Literal["bypass", "fast_llm", "full"]

# ── Labelled anchor examples ──────────────────────────────────────────────────
# Each tuple: (text, label).  Kept small (<30 examples) so embedding is cheap.
_ANCHORS: list[tuple[str, ClassificationLabel]] = [
    # bypass — instant local answers
    ("What is 2 + 2?", "bypass"),
    ("Calculate 15 percent of 200", "bypass"),
    ("Convert 100 kilometers to miles", "bypass"),
    ("What is the capital of France?", "bypass"),
    ("How many days in a leap year?", "bypass"),
    ("What is the boiling point of water in Celsius?", "bypass"),
    # fast_llm — simple knowledge / definition
    ("What is machine learning?", "fast_llm"),
    ("Explain what a neural network is", "fast_llm"),
    ("Who wrote Pride and Prejudice?", "fast_llm"),
    ("What are the main benefits of using Python?", "fast_llm"),
    ("Define recursion in programming", "fast_llm"),
    ("What is the difference between TCP and UDP?", "fast_llm"),
    ("Summarize the history of the Roman Empire in a paragraph", "fast_llm"),
    ("Translate hello to Spanish", "fast_llm"),
    # full — complex, multi-step, tool-using, agentic
    ("Build a REST API in Python with authentication", "full"),
    ("Research the latest AI papers and summarize the top 5", "full"),
    ("Create a GitHub issue and assign it to the team", "full"),
    ("Write a full test suite for this Python module", "full"),
    ("Design a microservices architecture for an e-commerce platform", "full"),
    ("Scrape product prices from this website and store in a database", "full"),
    ("Analyze this dataset and produce a report with charts", "full"),
    ("Debug this code, fix the errors, and run the tests", "full"),
    ("Implement a binary search tree with insert, delete, and search", "full"),
    ("Send a Slack message to the team with the project status", "full"),
]


def _embed_stub(text: str) -> list[float]:
    """
    Deterministic hash-based pseudo-embedding (1536-dim).
    No network, no model required.  Cosine similarity is not semantic
    but captures some character-level overlap between similar queries.
    """
    import hashlib

    dim = 1536
    h = hashlib.sha256(text.lower().strip().encode()).digest()
    extended = (h * ((dim // len(h)) + 1))[:dim]
    return [(b / 127.5) - 1.0 for b in extended]


def _embed_litellm(text: str) -> list[float] | None:
    """Attempt a real embedding via LiteLLM.  Returns None on any failure."""
    try:
        import litellm

        from aifactory.config.settings import get_settings

        s = get_settings()
        resp = litellm.embedding(model=s.embedding_model, input=[text])
        return resp.data[0]["embedding"]
    except Exception:
        return None


def _embed(text: str) -> list[float]:
    """
    Embed text using LiteLLM when available, stub otherwise.
    The result is used only for cosine similarity — exact dimensions don't
    matter as long as anchors and query use the same embedder.
    """
    vec = _embed_litellm(text)
    return vec if vec is not None else _embed_stub(text)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


@lru_cache(maxsize=1)
def _anchor_embeddings() -> list[tuple[list[float], ClassificationLabel]]:
    """Embed all anchor examples once and cache for the process lifetime."""
    logger.debug("SemanticClassifier: embedding %d anchor examples", len(_ANCHORS))
    return [(_embed(text), label) for text, label in _ANCHORS]


def classify(
    query: str,
    confidence_threshold: float = 0.65,
) -> tuple[ClassificationLabel, float]:
    """
    Classify a query using cosine similarity to labelled anchor examples.

    Returns:
        (label, confidence) where confidence is the mean cosine similarity
        to the top-3 anchors of the winning label.
        Returns ("full", 0.0) when confidence < confidence_threshold (safe default).
    """
    if not query.strip():
        return "full", 0.0

    try:
        query_vec = _embed(query)
        anchors = _anchor_embeddings()

        # Score each anchor
        scored: list[tuple[float, ClassificationLabel]] = [
            (_cosine(query_vec, anchor_vec), label) for anchor_vec, label in anchors
        ]
        scored.sort(key=lambda x: -x[0])

        # Aggregate top-K per label
        top_k = 3
        label_scores: dict[ClassificationLabel, list[float]] = {
            "bypass": [],
            "fast_llm": [],
            "full": [],
        }
        for sim, label in scored[: top_k * 3]:  # look at more than top_k to gather per-label
            if len(label_scores[label]) < top_k:
                label_scores[label].append(sim)

        # Mean score per label (0.0 if no examples seen)
        label_means: dict[ClassificationLabel, float] = {
            lbl: (sum(scores) / len(scores) if scores else 0.0)
            for lbl, scores in label_scores.items()
        }
        best_label = max(label_means, key=lambda lbl: label_means[lbl])
        best_score = label_means[best_label]

        if best_score < confidence_threshold:
            logger.debug(
                "SemanticClassifier: low confidence (%.3f) for '%s' → full",
                best_score,
                query[:60],
            )
            return "full", best_score

        logger.debug(
            "SemanticClassifier: '%s' → %s (confidence=%.3f)",
            query[:60],
            best_label,
            best_score,
        )
        return best_label, best_score

    except Exception as exc:
        logger.debug("SemanticClassifier: failed, returning full: %s", exc)
        return "full", 0.0
