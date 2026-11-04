"""Nested cross-validation with exact split-conformal checks.

    uv run python scripts/nested.py

For each of the five outer folds:

1. Choose the regularisation strength C by 4-fold CV inside the training
   folds only. The outer test fold is never looked at.
2. Fit on the full training folds with that C and score the test fold.
   These out-of-fold scores are the honest accuracy estimate and feed the
   queue replay.
3. Separately, split the training folds 75/25 into "fit" and "calibrate",
   fit on "fit", set both conformal cut-offs on "calibrate" and check them
   on the test fold. The calibration posts were never seen by the model
   that scores them, so the split-conformal guarantee holds exactly.

Writes results/oof/nested.npy, results/oof/nested.json and
results/conformal.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from saath import conformal, data, metrics, models

GRID = [0.5, 1.0, 2.0, 4.0, 8.0]
OUT = ROOT / "results"


def fit_score(texts, y, tr, te, C):
    m = models.default_model(C=C).fit([texts[i] for i in tr], y[tr])
    return m.exceed_proba([texts[i] for i in te])


def choose_C(texts, y, tr):
    inner = StratifiedKFold(4, shuffle=True, random_state=1)
    best, best_q = None, -1.0
    for C in GRID:
        pred = np.zeros(len(tr), dtype=int)
        for a, b in inner.split(tr, y[tr]):
            pred[b] = models.predict_level(fit_score(texts, y, tr[a], tr[b], C))
        q = metrics.qwk(y[tr], pred)
        if q > best_q:
            best, best_q = C, q
    return best, best_q


def main():
    df = data.load()
    texts, y = df.text.tolist(), df.level.to_numpy()
    oof = np.zeros((len(y), data.N_LEVELS - 1))
    urgent = np.zeros(len(y), dtype=bool)
    critical = np.zeros(len(y), dtype=bool)
    sets = np.zeros((len(y), 2), dtype=int)
    sets_lv = np.zeros((len(y), 2), dtype=int)
    chosen = []
    t0 = time.time()

    outer = StratifiedKFold(5, shuffle=True, random_state=0)
    for k, (tr, te) in enumerate(outer.split(texts, y)):
        C, inner_q = choose_C(texts, y, tr)
        chosen.append({"fold": k, "C": C, "inner_qwk": inner_q})
        oof[te] = fit_score(texts, y, tr, te, C)

        fit, cal = train_test_split(tr, test_size=0.25, stratify=y[tr], random_state=k)
        model = models.default_model(C=C).fit([texts[i] for i in fit], y[fit])
        ex_cal = model.exceed_proba([texts[i] for i in cal])
        ex_te = model.exceed_proba([texts[i] for i in te])
        u = conformal.UrgentThreshold(0.05).fit(models.risk_score(ex_cal), y[cal])
        urgent[te] = u.urgent(models.risk_score(ex_te))
        crit = conformal.UrgentThreshold(0.5).fit(models.risk_score(ex_cal), y[cal])
        critical[te] = crit.urgent(models.risk_score(ex_te))
        ls = conformal.LevelSets(0.1).fit(models.to_level_proba(ex_cal), y[cal])
        sets[te] = ls.predict(models.to_level_proba(ex_te))
        lsm = conformal.LevelSets(0.1, by_level=True).fit(models.to_level_proba(ex_cal), y[cal])
        sets_lv[te] = lsm.predict(models.to_level_proba(ex_te))
        print(f"fold {k}: C={C} inner qwk {inner_q:.3f} ({time.time() - t0:.0f}s)", flush=True)

    (OUT / "oof").mkdir(parents=True, exist_ok=True)
    np.save(OUT / "oof" / "nested.npy", oof)
    s = metrics.summary(y, models.predict_level(oof), models.risk_score(oof))
    (OUT / "oof" / "nested.json").write_text(
        json.dumps(
            {"args": {"model": "ensemble", "nested": True, "grid": GRID}, "chosen": chosen, "summary": s}, indent=2
        )
    )

    hr = y >= data.HIGH_RISK
    cov = conformal.covered(sets, y)
    report = {
        "method": "split conformal inside each outer fold: fit on 75% of the training folds, calibrate on the "
        "other 25%, test on the outer fold",
        "urgent": {
            "alpha": 0.05,
            "miss_rate": float((~urgent[hr]).mean()),
            "miss_ci": list(metrics.bootstrap(lambda u: float((~u).mean()), urgent[hr], n=2000)[1:]),
            "urgent_share": float(urgent.mean()),
            "precision": float(hr[urgent].mean()),
            "n_high_risk": int(hr.sum()),
        },
        "critical": {
            "alpha": 0.5,
            "miss_rate": float((~critical[hr]).mean()),
            "share": float(critical.mean()),
            "precision": float(hr[critical].mean()),
        },
        "level_sets": {
            "alpha": 0.1,
            "coverage": float(cov.mean()),
            "mean_width": float((sets[:, 1] - sets[:, 0] + 1).mean()),
            "singleton_share": float((sets[:, 0] == sets[:, 1]).mean()),
            "coverage_by_level": {int(v): float(cov[y == v].mean()) for v in range(data.N_LEVELS)},
        },
        "level_sets_by_level": {
            "alpha": 0.1,
            "coverage": float(conformal.covered(sets_lv, y).mean()),
            "mean_width": float((sets_lv[:, 1] - sets_lv[:, 0] + 1).mean()),
            "singleton_share": float((sets_lv[:, 0] == sets_lv[:, 1]).mean()),
            "coverage_by_level": {
                int(v): float(conformal.covered(sets_lv, y)[y == v].mean()) for v in range(data.N_LEVELS)
            },
        },
    }
    (OUT / "conformal.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"summary": s, **report}, indent=2))


if __name__ == "__main__":
    main()
