"""Fit the deployed model and write artifacts/.

    uv run python scripts/train_final.py

The model is fitted on a stratified 80% of the posts and both conformal
cut-offs are set on the other 20%, which the model never sees. That makes
the deployed guarantees exact split conformal, at the cost of training on
fewer posts. C is the value nested cross-validation picked most often.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from saath import data, models
from saath.service import Calibration

ART = ROOT / "artifacts"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration-share", type=float, default=0.2)
    args = ap.parse_args()

    nested = json.loads((ROOT / "results" / "oof" / "nested.json").read_text())
    picks = [c["C"] for c in nested["chosen"]]
    C = max(set(picks), key=picks.count)

    df = data.load()
    texts, y = df.text.tolist(), df.level.to_numpy()
    fit, cal_idx = train_test_split(np.arange(len(y)), test_size=args.calibration_share, stratify=y, random_state=0)
    model = models.default_model(C=C).fit([texts[i] for i in fit], y[fit])
    cal = Calibration.fit(model.exceed_proba([texts[i] for i in cal_idx]), y[cal_idx])
    run = {"C": C, "fit_posts": len(fit), "calibration_posts": len(cal_idx)}
    for m in model.members:
        if isinstance(m, models.EmbedOrdinal):
            m.cache = False

    ART.mkdir(exist_ok=True)
    joblib.dump(model, ART / "model.joblib")
    cal.save(ART / "calibration.json")
    (ART / "model.json").write_text(
        json.dumps({"kind": "sklearn", "trained_on": len(fit), "encoders": models.ENCODERS, "config": run}, indent=2)
    )
    print("saved", cal)


if __name__ == "__main__":
    main()
