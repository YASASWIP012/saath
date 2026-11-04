"""Distribution-free guarantees on top of any risk score.

Two pieces:

* ``UrgentThreshold`` picks a cut-off on P(level >= 4) so that, for a new
  high-risk person drawn from the same population as the calibration set, the
  chance of landing *below* the cut-off is at most ``alpha``. This is the
  conformal quantile on the high-risk calibration scores, and it holds in
  finite samples with no assumption about the model being right.

* ``LevelSets`` returns a contiguous range of severity levels per message
  (for example "3 to 5") that contains the true level with probability at
  least ``1 - alpha``. The width of the range is the model being honest
  about how unsure it is.
"""

from __future__ import annotations

import math

import numpy as np

from .data import HIGH_RISK, N_LEVELS


class UrgentThreshold:
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def fit(self, score, level):
        s = np.sort(np.asarray(score)[np.asarray(level) >= HIGH_RISK])
        n = len(s)
        k = math.floor(self.alpha * (n + 1))
        # k == 0 means the calibration set is too small to promise alpha;
        # the only safe answer is to flag everyone.
        self.threshold_ = -np.inf if k == 0 else float(s[k - 1])
        self.n_calib_ = n
        return self

    def urgent(self, score) -> np.ndarray:
        return np.asarray(score) >= self.threshold_


class LevelSets:
    """Central intervals of the predicted severity distribution.

    For a candidate level k, the score is the smaller of the two tail masses
    it leaves outside: min(P(Y <= k), P(Y >= k)). Levels near the middle of
    the distribution score high, levels out in a thin tail score low. The
    set keeps every level whose score clears a cut-off calibrated so the true
    level is inside with probability at least 1 - alpha. Because the score
    rises towards the median and falls away from it, the kept levels are
    always one contiguous range.
    """

    def __init__(self, alpha: float = 0.1, by_level: bool = False):
        self.alpha = alpha
        # by_level calibrates one cut-off per true level (Mondrian conformal),
        # so every level, not just the average, gets 1 - alpha coverage.
        self.by_level = by_level

    @staticmethod
    def _tail_scores(level_proba):
        p = np.asarray(level_proba)
        cdf = np.cumsum(p, axis=1)
        sf = 1 - cdf + p  # P(Y >= k)
        return np.minimum(cdf, sf)

    def fit(self, level_proba, level):
        y = np.asarray(level)
        scores = self._tail_scores(level_proba)[np.arange(len(y)), y]
        self.tau_ = self._cut(scores)
        if self.by_level:
            self.tau_by_level_ = np.array([self._cut(scores[y == v]) for v in range(level_proba.shape[1])])
        # Kept for the API; with this score the cut-off itself is tau_.
        self.qhat_ = self.tau_
        return self

    def _cut(self, scores) -> float:
        k = math.floor(self.alpha * (len(scores) + 1))
        return -np.inf if k == 0 else float(np.sort(scores)[k - 1])

    def predict(self, level_proba) -> np.ndarray:
        """(n, 2) array of [low, high] levels, inclusive."""
        scores = self._tail_scores(level_proba)
        keep = scores >= (self.tau_by_level_[None, :] if self.by_level else self.tau_)
        out = np.empty((len(keep), 2), dtype=int)
        for i, row in enumerate(keep):
            idx = np.flatnonzero(row)
            if idx.size == 0:
                idx = np.array([int(np.argmax(scores[i]))])
            # Filling any gap keeps one contiguous range and only adds coverage.
            out[i] = idx.min(), idx.max()
        return out


def covered(sets: np.ndarray, level) -> np.ndarray:
    level = np.asarray(level)
    return (sets[:, 0] <= level) & (level <= sets[:, 1])


__all__ = ["N_LEVELS", "LevelSets", "UrgentThreshold", "covered"]
