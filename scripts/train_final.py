"""Fit the deployed model on every post and write artifacts/.

    uv run python scripts/train_final.py --oof ensemble

The conformal cut-offs are calibrated on the out-of-fold predictions of the
same configuration (cross-conformal), because a model scored on its own
training data would look far more certain than it is.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from saath import data, models
from saath.service import Calibration

ART = ROOT / "artifacts"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True, help="tag of the cross-validated run to calibrate from")
    args = ap.parse_args()

    run = json.loads((ROOT / "results" / "oof" / f"{args.oof}.json").read_text())["args"]
    df = data.load()
    y = df.level.to_numpy()

    cal = Calibration.fit(np.load(ROOT / "results" / "oof" / f"{args.oof}.npy"), y)
    model = models.default_model(C=run.get("C", 1.0)).fit(df.text.tolist(), y)
    for m in model.members:
        if isinstance(m, models.EmbedOrdinal):
            m.cache = False

    ART.mkdir(exist_ok=True)
    joblib.dump(model, ART / "model.joblib")
    cal.save(ART / "calibration.json")
    (ART / "model.json").write_text(
        json.dumps({"kind": "sklearn", "trained_on": len(y), "encoders": models.ENCODERS, "config": run}, indent=2)
    )
    print("saved", cal)


if __name__ == "__main__":
    main()
