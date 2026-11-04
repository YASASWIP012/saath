"""Metrics, with bootstrap intervals for everything we report."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score

from .data import HIGH_RISK


def qwk(y, p) -> float:
    return float(cohen_kappa_score(y, p, weights="quadratic", labels=list(range(7))))


def high_risk_counts(y, p):
    t, h = np.asarray(y) >= HIGH_RISK, np.asarray(p) >= HIGH_RISK
    return int((t & h).sum()), int((t & ~h).sum()), int((~t & h).sum())


def summary(y, pred_level, score=None) -> dict:
    """Headline numbers for one set of predictions."""
    y, pred_level = np.asarray(y), np.asarray(pred_level)
    tp, fn, fp = high_risk_counts(y, pred_level)
    out = {
        "qwk": qwk(y, pred_level),
        "exact": float((y == pred_level).mean()),
        "within_one": float((np.abs(y - pred_level) <= 1).mean()),
        "mae": float(np.abs(y - pred_level).mean()),
        "hr_recall": tp / max(tp + fn, 1),
        "hr_precision": tp / max(tp + fp, 1),
        "hr_f1": 2 * tp / max(2 * tp + fn + fp, 1),
    }
    if score is not None:
        t = y >= HIGH_RISK
        out["hr_auroc"] = float(roc_auc_score(t, score))
        out["hr_auprc"] = float(average_precision_score(t, score))
    return out


def ece(prob, outcome, bins: int = 10) -> float:
    """Expected calibration error of a binary probability."""
    prob, outcome = np.asarray(prob), np.asarray(outcome, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)
    err = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            err += m.mean() * abs(prob[m].mean() - outcome[m].mean())
    return float(err)


def bootstrap(fn, *arrays, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Point estimate and 95% percentile interval of fn over resampled rows."""
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(a) for a in arrays]
    size = len(arrays[0])
    point = fn(*arrays)
    draws = []
    for _ in range(n):
        i = rng.integers(0, size, size)
        draws.append(fn(*(a[i] for a in arrays)))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def paired_bootstrap(fn, y, a, b, n: int = 2000, seed: int = 0) -> dict:
    """Difference fn(y, a) - fn(y, b) on the same resamples, with a p-value
    for the one-sided claim that a is better."""
    rng = np.random.default_rng(seed)
    y, a, b = map(np.asarray, (y, a, b))
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        diffs.append(fn(y[i], a[i]) - fn(y[i], b[i]))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "diff": float(fn(y, a) - fn(y, b)),
        "lo": float(lo),
        "hi": float(hi),
        "p_value": float((diffs <= 0).mean()),
    }
