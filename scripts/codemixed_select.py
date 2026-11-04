"""Choosing how to handle code-mixed messages, without touching the test half.

    uv run python scripts/codemixed_select.py sentence-transformers/LaBSE ...

The 56 code-mixed messages are split by row parity: even rows are a
development set for choosing an approach, odd rows are held out and scored
once at the end. Candidates, for each multilingual encoder X given:

* base: the deployed English ensemble
* +X: the ensemble with X added as a fourth member
* route:X: messages the language detector flags are scored by a model built
  on X alone; everything else goes to the base ensemble

Every candidate is also scored by 5-fold CV on the English posts, and any
that costs more than 0.01 English κ is ruled out before the dev comparison.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from saath import data, models
from saath.models import Routed

C = 4.0


def english_cv(make, texts, y):
    oof = np.zeros((len(y), 6))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(texts, y):
        oof[te] = make().fit([texts[i] for i in tr], y[tr]).exceed_proba([texts[i] for i in te])
    from saath import metrics

    return metrics.qwk(y, models.predict_level(oof)), float(roc_auc_score(y >= 4, models.risk_score(oof)))


def cm_scores(model, cm):
    from saath import metrics

    ex = model.exceed_proba(cm.text.tolist())
    hr = cm.level.to_numpy() >= data.HIGH_RISK
    return {
        "qwk": metrics.qwk(cm.level, models.predict_level(ex)),
        "auroc": float(roc_auc_score(hr, models.risk_score(ex))),
        "n": len(cm),
    }


def main():
    encoders = sys.argv[1:]
    df = data.load()
    texts, y = df.text.tolist(), df.level.to_numpy()
    cm = pd.read_csv(ROOT / "data" / "stress" / "codemixed.csv")
    dev, test = cm.iloc[::2].reset_index(drop=True), cm.iloc[1::2].reset_index(drop=True)

    makers = {"base": lambda: models.default_model(C=C)}
    for enc in encoders:
        makers[f"+{enc}"] = lambda enc=enc: models.Ensemble(
            [*models.default_model(C=C).members, models.EmbedOrdinal(enc, C=C)]
        )
        makers[f"route:{enc}"] = lambda enc=enc: Routed(models.default_model(C=C), models.EmbedOrdinal(enc, C=C))

    rows, fitted = {}, {}
    for name, make in makers.items():
        q, a = english_cv(make, texts, y)
        fitted[name] = make().fit(texts, y)
        d = cm_scores(fitted[name], dev)
        rows[name] = {"english_qwk": q, "english_auroc": a, "dev_qwk": d["qwk"], "dev_auroc": d["auroc"]}
        print(f"{name:72} english qwk {q:.3f} auroc {a:.3f} | dev qwk {d['qwk']:+.3f} auroc {d['auroc']:.3f}", flush=True)

    base_q = rows["base"]["english_qwk"]
    allowed = [k for k, v in rows.items() if v["english_qwk"] >= base_q - 0.01]
    pick = max(allowed, key=lambda k: rows[k]["dev_auroc"])
    test_rows = {k: cm_scores(fitted[k], test) for k in dict.fromkeys(["base", pick])}
    out = {
        "split": "even rows of data/stress/codemixed.csv for choosing, odd rows held out and scored once",
        "rule": "highest dev AUROC among candidates within 0.01 English κ of the base model",
        "candidates": rows,
        "picked": pick,
        "test": test_rows,
    }
    (ROOT / "results" / "codemixed_select.json").write_text(json.dumps(out, indent=2))
    print("picked", pick)
    print("held-out half:", json.dumps(test_rows))


if __name__ == "__main__":
    main()
