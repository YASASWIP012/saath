"""Everything the API needs to score one message, in one object."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import evidence, language
from .conformal import LevelSets, UrgentThreshold
from .data import HIGH_RISK, LEVELS
from .models import predict_level, risk_score, to_level_proba


@dataclass
class Calibration:
    urgent_threshold: float
    alpha_urgent: float
    level_qhat: float
    alpha_level: float
    # Isotonic map from raw risk to the observed high-risk rate, stored as
    # breakpoints. Ranking, the urgent cut-off and the queue all use the raw
    # score; this only makes the percentage shown to a counsellor honest.
    risk_x: list | None = None
    risk_y: list | None = None
    # Raw-risk percentiles of the calibration posts. The queue sorts on a
    # message's percentile, exactly as the replay does.
    risk_quantiles: list | None = None
    # A smaller, sharper tier inside the urgent band: the cut-off that
    # catches at least half of high-risk messages, with high precision.
    critical_threshold: float | None = None
    alpha_critical: float = 0.5

    @classmethod
    def fit(cls, exceed, level, alpha_urgent=0.05, alpha_level=0.1):
        from sklearn.isotonic import IsotonicRegression

        raw = risk_score(exceed)
        u = UrgentThreshold(alpha_urgent).fit(raw, level)
        crit = UrgentThreshold(0.5).fit(raw, level)
        s = LevelSets(alpha_level).fit(to_level_proba(exceed), level)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw, np.asarray(level) >= HIGH_RISK)
        return cls(
            u.threshold_,
            alpha_urgent,
            s.qhat_,
            alpha_level,
            [float(v) for v in iso.X_thresholds_],
            [float(v) for v in iso.y_thresholds_],
            [float(v) for v in np.quantile(raw, np.linspace(0, 1, 101))],
            crit.threshold_,
        )

    def priority(self, raw):
        if not self.risk_quantiles:
            return np.asarray(raw, dtype=float)
        return np.interp(raw, self.risk_quantiles, np.linspace(0, 1, 101))

    def calibrated(self, raw):
        if not self.risk_x:
            return np.asarray(raw, dtype=float)
        return np.interp(raw, self.risk_x, self.risk_y)

    def save(self, path):
        Path(path).write_text(json.dumps(self.__dict__, indent=2))

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


def widen_for_urgent(lo: int, hi: int, urgent: bool) -> tuple[int, int]:
    """An urgent message's range must reach the high-risk levels.

    The urgent band and the range come from separate guarantees and can
    disagree. Widening a range only adds coverage, so the range keeps its
    guarantee and the card stops contradicting itself.
    """
    return (lo, max(hi, HIGH_RISK)) if urgent else (lo, hi)


def band_for(urgent: bool, level_high: int, text: str, critical: bool = False) -> tuple[str, str]:
    """Queue band plus the reason a counsellor sees for it."""
    if urgent:
        return "urgent", "critical" if critical else "risk"
    if level_high >= HIGH_RISK:
        return "review", "range"
    if language.is_romanised_indic(text):
        # The model reads romanised Hindi and Telugu poorly, so it is not
        # allowed to call these messages routine.
        return "review", "language"
    return "routine", "risk"


class Scorer:
    def __init__(self, model, calibration: Calibration):
        self.model = model
        self.cal = calibration
        self._sets = LevelSets(calibration.alpha_level)
        self._sets.tau_ = calibration.level_qhat

    def _critical(self, raw) -> bool:
        t = self.cal.critical_threshold
        return t is not None and bool(raw >= t)

    def score(self, text: str, explain: bool = True) -> dict:
        t0 = time.perf_counter()
        exceed = self.model.exceed_proba([text])
        proba = to_level_proba(exceed)[0]
        risk = float(risk_score(exceed)[0])
        level = int(predict_level(exceed)[0])
        lo, hi = (int(v) for v in self._sets.predict(proba[None, :])[0])
        urgent = risk >= self.cal.urgent_threshold
        lo, hi = widen_for_urgent(lo, hi, urgent)
        band, reason = band_for(urgent, hi, text, self._critical(risk))
        out = {
            "level": level,
            "level_label": LEVELS[level],
            "level_range": [lo, hi],
            "proba": [round(float(p), 4) for p in proba],
            "risk": round(risk, 4),
            "risk_calibrated": round(float(self.cal.calibrated(risk)), 4),
            "priority": round(float(self.cal.priority(risk)), 4),
            "band": band,
            "band_reason": reason,
            "evidence": evidence.explain(self.model, text) if explain else [],
        }
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out

    def score_many(self, texts: list[str]) -> list[dict]:
        exceed = self.model.exceed_proba(texts)
        proba = to_level_proba(exceed)
        sets = self._sets.predict(proba)
        risk = risk_score(exceed)
        level = predict_level(exceed)
        out = []
        for i in range(len(texts)):
            urgent = bool(risk[i] >= self.cal.urgent_threshold)
            lo, hi = widen_for_urgent(int(sets[i, 0]), int(sets[i, 1]), urgent)
            band, reason = band_for(urgent, hi, texts[i], self._critical(risk[i]))
            out.append(
                {
                    "level": int(level[i]),
                    "level_range": [lo, hi],
                    "risk": round(float(risk[i]), 4),
                    "risk_calibrated": round(float(self.cal.calibrated(risk[i])), 4),
                    "priority": round(float(self.cal.priority(risk[i])), 4),
                    "band": band,
                    "band_reason": reason,
                }
            )
        return out


__all__ = ["Calibration", "Scorer", "np"]
