"""Ordinal severity models.

Every model exposes the same interface: ``fit(texts, y)`` and
``exceed_proba(texts)``, which returns P(level > k) for k = 0..5. Working with
cumulative probabilities keeps the ordering of the scale intact and gives the
triage score for free: P(level >= 4) is column 3.
"""

from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_union

from . import embed
from .data import HIGH_RISK, N_LEVELS


def cumulative_targets(level, jury=None, jury_weight: float = 0.0) -> np.ndarray:
    level = np.asarray(level)
    ks = np.arange(N_LEVELS - 1)
    hard = (level[:, None] > ks[None, :]).astype(np.float32)
    if jury is None or jury_weight == 0:
        return hard
    jury = np.asarray(jury, dtype=float)
    votes = (jury[:, :, None] > ks[None, None, :]).astype(float)
    valid = ~np.isnan(jury)[:, :, None]
    soft = (votes * valid).sum(1) / np.maximum(valid.sum(1), 1)
    return ((1 - jury_weight) * hard + jury_weight * soft).astype(np.float32)


def to_level_proba(exceed: np.ndarray) -> np.ndarray:
    """Turn cumulative P(y > k) into per-level probabilities."""
    # Independent binary fits can cross; force the cumulative curve downhill.
    exceed = np.minimum.accumulate(np.clip(exceed, 0, 1), axis=1)
    upper = np.hstack([np.ones((len(exceed), 1)), exceed])
    lower = np.hstack([exceed, np.zeros((len(exceed), 1))])
    p = np.clip(upper - lower, 0, None)
    return p / p.sum(axis=1, keepdims=True)


def risk_score(exceed: np.ndarray) -> np.ndarray:
    """P(level >= HIGH_RISK): the number the queue sorts on."""
    return exceed[:, HIGH_RISK - 1]


def predict_level(exceed: np.ndarray) -> np.ndarray:
    """Median of the predicted distribution, which minimises absolute error."""
    exceed = np.minimum.accumulate(np.clip(exceed, 0, 1), axis=1)
    return (exceed > 0.5).sum(axis=1)


class _Cumulative:
    """Frank & Hall ordinal decomposition over any feature extractor."""

    C = 1.0

    def _features(self, texts, fit: bool):
        raise NotImplementedError

    def fit(self, texts, y, jury=None, jury_weight: float = 0.0):
        from scipy import sparse

        X = self._features(list(texts), fit=True)
        soft = cumulative_targets(y, jury, jury_weight)
        stack = sparse.vstack if sparse.issparse(X) else np.vstack
        X2 = stack([X, X])
        self.heads_ = []
        for k in range(N_LEVELS - 1):
            # A soft target t becomes two weighted rows: positive with weight
            # t, negative with weight 1 - t. With hard labels one row of each
            # pair has weight zero and this is ordinary logistic regression.
            t = soft[:, k]
            pos = max(t.mean(), 1e-6)
            w = np.concatenate([t / pos, (1 - t) / max(1 - pos, 1e-6)]) / 2  # class balance
            lab = np.concatenate([np.ones(len(t)), np.zeros(len(t))])
            keep = w > 0
            head = LogisticRegression(C=self.C, max_iter=4000)
            head.fit(X2[keep], lab[keep], sample_weight=w[keep])
            self.heads_.append(head)
        return self

    def exceed_proba(self, texts) -> np.ndarray:
        X = self._features(list(texts), fit=False)
        return np.column_stack([h.predict_proba(X)[:, 1] for h in self.heads_])


class TfidfOrdinal(_Cumulative):
    """Word and character n-grams. Cheap, strong on explicit phrasing."""

    def __init__(self, C: float = 4.0):
        self.C = C

    def _features(self, texts, fit):
        if fit:
            self.vec_ = make_union(
                TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
            )
            return self.vec_.fit_transform(texts)
        return self.vec_.transform(texts)


class EmbedOrdinal(_Cumulative):
    """Frozen sentence encoder plus ordinal logistic heads."""

    def __init__(self, model: str = "intfloat/multilingual-e5-base", C: float = 1.0, cache: bool = True):
        self.model = model
        self.C = C
        # The disk cache speeds up cross-validation. A deployed model scores
        # live messages, which must never be written to disk.
        self.cache = cache

    def _features(self, texts, fit):
        return embed.encode(texts, self.model, use_cache=self.cache)


class Ensemble:
    """Average of cumulative probabilities across members."""

    def __init__(self, members, weights=None):
        self.members = members
        self.weights = weights

    def fit(self, texts, y, jury=None, jury_weight: float = 0.0):
        for m in self.members:
            m.fit(texts, y, jury=jury, jury_weight=jury_weight)
        return self

    def exceed_proba(self, texts):
        stack = np.stack([m.exceed_proba(texts) for m in self.members])
        w = np.ones(len(self.members)) if self.weights is None else np.asarray(self.weights)
        return np.tensordot(w / w.sum(), stack, axes=1)


class Routed:
    """English ensemble for most messages, a separate model for romanised Hindi/Telugu.

    The language detector decides which model scores each message. Both are
    trained on the same posts; only their features differ.
    """

    def __init__(self, base, alt):
        self.base = base
        self.alt = alt

    def fit(self, texts, y, **kw):
        self.base.fit(texts, y, **kw)
        self.alt.fit(texts, y, **kw)
        return self

    def exceed_proba(self, texts):
        from .language import is_romanised_indic

        texts = list(texts)
        out = self.base.exceed_proba(texts)
        mask = np.array([is_romanised_indic(t) for t in texts])
        if mask.any():
            out[mask] = self.alt.exceed_proba([t for t, m in zip(texts, mask, strict=True) if m])
        return out


# A transparent baseline any helpline could write in an afternoon. Patterns are
# grouped by the C-SSRS level they most directly signal.
_KEYWORDS = [
    (6, r"\b(took|swallowed|overdos\w*|od'?d|cut (my|myself)|tried to (kill|end)|attempt(ed)?|noose|jumped)\b"),
    (5, r"\b(plan(ned|ning)?|note|tonight|tomorrow|goodbye|wrote (a|my) letter|date)\b"),
    (4, r"\b(going to|gonna|will) (kill|end|do it)|\bready to (die|go)\b"),
    (3, r"\b(pills|rope|gun|bridge|knife|blade|paracetamol|tylenol|hang(ing)?)\b"),
    (2, r"\b(kill (my)?self|suicid\w*|end (it|my life)|take my (own )?life)\b"),
    (1, r"\b(want(ed)? to die|wish i (was|were) dead|not wake up|no reason to live|better off dead)\b"),
]


class KeywordRule:
    """Highest C-SSRS level whose pattern appears in the text."""

    def fit(self, texts, y, **_):
        return self

    def levels(self, texts) -> np.ndarray:
        out = []
        for t in texts:
            low = t.lower()
            out.append(max((lvl for lvl, pat in _KEYWORDS if re.search(pat, low)), default=0))
        return np.array(out)

    def exceed_proba(self, texts):
        lv = self.levels(texts)
        return np.column_stack([(lv > k).astype(float) for k in range(N_LEVELS - 1)])


ENCODERS = ["intfloat/multilingual-e5-base", "sentence-transformers/all-mpnet-base-v2"]


def default_model(C: float = 1.0) -> Ensemble:
    """Two frozen encoders, one multilingual, plus character and word n-grams."""
    return Ensemble([EmbedOrdinal(m, C=C) for m in ENCODERS] + [TfidfOrdinal(C=4.0)])


__all__ = [
    "EmbedOrdinal",
    "Ensemble",
    "KeywordRule",
    "TfidfOrdinal",
    "default_model",
    "predict_level",
    "risk_score",
    "to_level_proba",
]
