"""Out-of-fold predictions for one model configuration.

Every configuration uses the same five stratified folds, so later analysis can
compare models post by post. Each post appears in exactly one test fold, and
each post has a different author, so no writer is ever seen in both training
and test.

    uv run python scripts/cv_predict.py tfidf
    uv run python scripts/cv_predict.py ensemble --C 2 --jury-weight 0.3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from saath import data, metrics, models

OUT = ROOT / "results" / "oof"


def build(name: str, args):
    if name == "keyword":
        return models.KeywordRule()
    if name == "tfidf":
        return models.TfidfOrdinal(C=4.0)
    if name == "ensemble":
        return models.default_model(C=args.C)
    if name.startswith("embed:"):
        return models.EmbedOrdinal(name.split(":", 1)[1], C=args.C)
    raise ValueError(f"unknown model {name!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--jury-weight", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    df = data.load()
    texts, y, jury = df.text.tolist(), df.level.to_numpy(), data.jury_matrix(df)
    oof = np.full((len(y), data.N_LEVELS - 1), np.nan)
    t0 = time.time()
    splits = StratifiedKFold(5, shuffle=True, random_state=0).split(texts, y)
    for f, (tr, te) in enumerate(splits):
        if f >= args.folds:
            break
        m = build(args.model, args)
        kw = {"jury": jury[tr], "jury_weight": args.jury_weight} if args.jury_weight else {}
        m.fit([texts[i] for i in tr], y[tr], **kw)
        oof[te] = m.exceed_proba([texts[i] for i in te])
        s = metrics.summary(y[te], models.predict_level(oof[te]), models.risk_score(oof[te]))
        print(f"fold {f}: qwk {s['qwk']:.3f} hr_auroc {s['hr_auroc']:.3f} ({time.time() - t0:.0f}s)", flush=True)

    done = ~np.isnan(oof[:, 0])
    s = metrics.summary(y[done], models.predict_level(oof[done]), models.risk_score(oof[done]))
    tag = args.tag or args.model.replace("/", "_").replace(":", "_")
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / f"{tag}.npy", oof)
    (OUT / f"{tag}.json").write_text(json.dumps({"args": vars(args), "summary": s}, indent=2))
    print(tag, {k: round(v, 3) for k, v in s.items()}, flush=True)


if __name__ == "__main__":
    main()
