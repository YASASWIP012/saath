"""Which sentences drove the score.

We remove one sentence at a time and re-score the message. A sentence's
weight is how far the high-risk probability falls without it. This is slower
than reading attention weights, but it measures the thing a counsellor
actually wants to know: "if this line were not here, would we still be
worried?"
"""

from __future__ import annotations

import re

import numpy as np

from .models import risk_score

_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")


def sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SPLIT.split(text) if p and p.strip()]
    return parts or [text.strip()]


def explain(model, text: str) -> list[dict]:
    """Per-sentence contribution to P(level >= 4), largest first in the text order."""
    parts = sentences(text)
    variants = [text] + [" ".join(p for j, p in enumerate(parts) if j != i) or "." for i in range(len(parts))]
    scores = risk_score(model.exceed_proba(variants))
    base = float(scores[0])
    return [{"text": p, "weight": float(max(base - s, 0.0))} for p, s in zip(parts, scores[1:], strict=True)]


def top_sentence(model, text: str) -> str:
    parts = explain(model, text)
    return max(parts, key=lambda d: d["weight"])["text"]


__all__ = ["explain", "np", "sentences", "top_sentence"]
