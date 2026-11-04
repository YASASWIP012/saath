"""Tests the deployed model was never tuned on.

    uv run python scripts/stress.py

1. Code-mixed messages: 56 short texts in romanised Hindi-English and
   Telugu-English, eight per severity level. They were written for this
   project to probe how the model handles the way many Indian texters write.
   They are synthetic and small, so we treat the result as a stress test,
   not a validation.
2. A different corpus: posts from r/SuicideWatch versus r/teenagers
   (Komati, "Suicide and Depression Detection", Kaggle, mirrored on Hugging
   Face). Labels there are only "suicide-related or
   not", so we check whether P(level >= 1) separates the two. Posts that
   also appear in our training data are removed first.
3. Latency on CPU, which is what a helpline's own server would have.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from saath import data, metrics, models
from saath.service import Calibration

ART = ROOT / "artifacts"
OUT = ROOT / "results"
EXTERNAL_URL = "https://huggingface.co/datasets/jquiros/suicide/resolve/main/Suicide_Detection.csv"


def codemixed(model, cal):
    df = pd.read_csv(ROOT / "data" / "stress" / "codemixed.csv")
    ex = model.exceed_proba(df.text.tolist())
    lv, risk = models.predict_level(ex), models.risk_score(ex)
    y = df.level.to_numpy()
    hr = y >= data.HIGH_RISK
    from saath import conformal, language
    from saath.service import band_for

    urgent = risk >= cal.urgent_threshold
    sets = conformal.LevelSets(cal.alpha_level)
    sets.tau_ = cal.level_qhat
    hi = sets.predict(models.to_level_proba(ex))[:, 1]
    bands = np.array([band_for(u, int(h), t)[0] for u, h, t in zip(urgent, hi, df.text, strict=True)])
    gated = np.array([language.is_romanised_indic(t) for t in df.text])
    out = {"n": len(df), "overall": metrics.summary(y, lv, risk)}
    out["urgent_recall"] = float(urgent[hr].mean())
    out["detected_as_romanised"] = int(gated.sum())
    out["missed_by_detector"] = df.text[~gated].tolist()
    out["high_risk_kept_out_of_routine"] = float((bands[hr] != "routine").mean())
    out["high_risk_routine_texts"] = df.text[hr & (bands == "routine")].tolist()
    out["by_language"] = {
        lang: {**metrics.summary(y[m], lv[m], risk[m]), "urgent_recall": float(urgent[m & hr].mean())}
        for lang in df.language.unique()
        for m in [df.language.to_numpy() == lang]
    }
    return out


def external(model, n=1000, seed=0):
    path = data.CACHE / "external.csv"
    if not path.exists():
        urlretrieve(EXTERNAL_URL, path)
    ext = pd.read_csv(path, usecols=["text", "class"]).dropna()
    train = set(data.load().text.str.strip().str.lower())
    ext = ext[~ext.text.str.strip().str.lower().isin(train)]
    ext = ext[ext.text.str.len().between(20, 2000)]
    ext = pd.concat([g.sample(n // 2, random_state=seed) for _, g in ext.groupby("class")])
    t0 = time.time()
    ex = model.exceed_proba(ext.text.tolist())
    any_risk = ex[:, 0]
    target = (ext["class"] == "suicide").to_numpy()
    return {
        "n": len(ext),
        "auroc_suicide_related": float(roc_auc_score(target, any_risk)),
        "auroc_ci": list(metrics.bootstrap(lambda t, s: roc_auc_score(t, s), target, any_risk, n=500)[1:]),
        "share_flagged_high_risk_in_suicide_posts": float((models.risk_score(ex)[target] >= 0.5).mean()),
        "share_flagged_high_risk_in_other_posts": float((models.risk_score(ex)[~target] >= 0.5).mean()),
        "seconds": round(time.time() - t0, 1),
    }


def latency(model, texts, reps=3):
    single = []
    for _ in range(reps):
        for t in texts[:20]:
            t0 = time.perf_counter()
            model.exceed_proba([t])
            single.append((time.perf_counter() - t0) * 1000)
    return {
        "cpu_ms_per_message": round(float(np.median(single)), 1),
        "p95_ms": round(float(np.percentile(single, 95)), 1),
    }


def main():
    import joblib

    cal = Calibration.load(ART / "calibration.json")
    model = joblib.load(ART / "model.joblib")
    cm = codemixed(model, cal)
    (OUT / "codemixed.json").write_text(json.dumps(cm, indent=2))
    print("codemixed", json.dumps(cm["overall"]), "urgent recall", cm["urgent_recall"])

    ext = external(model)
    (OUT / "external.json").write_text(json.dumps(ext, indent=2))
    print("external", ext)

    from saath import embed

    embed._encoder.cache_clear()
    embed._DEVICE = "cpu"
    lat = latency(model, data.load().text.tolist())
    (OUT / "latency.json").write_text(json.dumps(lat, indent=2))
    print("latency", lat)


if __name__ == "__main__":
    main()
