"""How far the queue result can be trusted.

    uv run python scripts/queue_robustness.py --ours nested

Two analyses, written to results/queue_robustness.json:

1. Post bootstrap. The replay draws texters from a fixed pool of 1,170
   held-out posts, only 245 of them high risk. Re-running many seeds on one
   pool shrinks the interval toward zero and says nothing about that pool.
   Here we resample the pool itself, run the replay on each resample, and
   report intervals for the paired differences between orderings.

2. Sensitivity. One factor at a time away from the baseline scenario:
   high-risk prevalence, whether high-risk texters are more or less patient,
   the aging weight, the number of counsellors and mean patience. If the
   benefit only appears under one convenient setting, this shows it.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from saath import data, models, queue_sim

OUT = ROOT / "results"
BASE = queue_sim.Scenario(load=1.1)


def policies(y, ours, kw, llm):
    return {
        "First come": None,
        "Keyword": models.predict_level(kw) / 6,
        "Saath": models.risk_score(ours),
        "Frontier LLM": llm / 6,
        "Oracle": (y >= data.HIGH_RISK).astype(float),
    }


def mean_abandon(pol, y, scn, seeds):
    res = queue_sim.compare(pol, y, scn, seeds=seeds)
    return {name: float(np.nanmean(r["hr_abandon"])) for name, r in res.items()}, res


def post_bootstrap(y, pol, B, seeds, rng):
    rows = []
    for _ in range(B):
        idx = rng.integers(0, len(y), len(y))
        sub = {k: None if v is None else v[idx] for k, v in pol.items()}
        m, _ = mean_abandon(sub, y[idx], BASE, seeds)
        rows.append(m)
    names = list(pol)
    arr = {n: np.array([r[n] for r in rows]) for n in names}

    def ci(a):
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    return {
        "B": B,
        "seeds_per_resample": len(seeds),
        "abandon": {n: {"mean": float(arr[n].mean()), "ci": ci(arr[n])} for n in names},
        "diff_vs_saath": {
            n: {
                "mean": float((arr[n] - arr["Saath"]).mean()),
                "ci": ci(arr[n] - arr["Saath"]),
                "share_saath_better": float((arr[n] > arr["Saath"]).mean()),
            }
            for n in names
            if n != "Saath"
        },
        "relative_reduction_vs_fifo": {
            "mean": float((1 - arr["Saath"] / arr["First come"]).mean()),
            "ci": ci(1 - arr["Saath"] / arr["First come"]),
        },
    }


def with_prevalence(y, pol, target, rng, size=2000):
    hr = np.flatnonzero(y >= data.HIGH_RISK)
    lo = np.flatnonzero(y < data.HIGH_RISK)
    n_hr = round(target * size)
    idx = np.concatenate([rng.choice(hr, n_hr), rng.choice(lo, size - n_hr)])
    return y[idx], {k: None if v is None else v[idx] for k, v in pol.items()}


def sensitivity(y, pol, seeds, rng):
    cells = []

    def run(label, value, yy, pp, scn):
        m, res = mean_abandon(pp, yy, scn, seeds)
        low = {n: float(np.nanmean(r["low_abandon"])) for n, r in res.items()}
        cells.append(
            {
                "factor": label,
                "value": value,
                "abandon": m,
                "low_abandon": low,
                "relative_reduction": 1 - m["Saath"] / m["First come"],
                "share_of_oracle_gain": (m["First come"] - m["Saath"]) / max(m["First come"] - m["Oracle"], 1e-9),
            }
        )

    for prev in [0.05, 0.10, 0.21]:
        yy, pp = with_prevalence(y, pol, prev, rng)
        run("high-risk prevalence", prev, yy, pp, BASE)
    for f in [0.5, 1.0, 2.0]:
        run("high-risk patience multiplier", f, y, pol, replace(BASE, hr_patience=f))
    for a in [15.0, 45.0, 120.0, 1e9]:
        run("aging (minutes per unit risk)", a if a < 1e8 else "none", y, pol, replace(BASE, aging=a))
    for c in [2, 4, 8]:
        run("counsellors", c, y, pol, replace(BASE, counsellors=c))
    for p in [10.0, 20.0, 40.0]:
        run("mean patience (minutes)", p, y, pol, replace(BASE, mean_patience=p))
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default="nested")
    ap.add_argument("--B", type=int, default=200)
    args = ap.parse_args()

    df = data.load()
    y = df.level.to_numpy()
    ours = np.load(OUT / "oof" / f"{args.ours}.npy")
    kw = np.load(OUT / "oof" / "keyword.npy")
    jury = np.nan_to_num(data.jury_matrix(df), nan=0).astype(int)
    from saath import metrics

    best = int(np.argmax([metrics.qwk(y, jury[:, j]) for j in range(jury.shape[1])]))
    pol = policies(y, ours, kw, jury[:, best])
    rng = np.random.default_rng(0)

    report = {
        "baseline": BASE.__dict__,
        "post_bootstrap": post_bootstrap(y, pol, args.B, range(30), rng),
        "sensitivity": sensitivity(y, pol, range(150), rng),
    }
    (OUT / "queue_robustness.json").write_text(json.dumps(report, indent=2, default=float))
    pb = report["post_bootstrap"]
    print("relative reduction", pb["relative_reduction_vs_fifo"])
    for n, d in pb["diff_vs_saath"].items():
        print(f"  {n} - Saath: {d['mean']:+.3f} {d['ci']}  Saath better in {d['share_saath_better']:.0%}")
    for c in report["sensitivity"]:
        print(
            f"{c['factor']:32} {c['value']!s:>6}  fifo {c['abandon']['First come']:.3f}  saath {c['abandon']['Saath']:.3f}"
            f"  rel {c['relative_reduction']:.2f}  oracle-share {c['share_of_oracle_gain']:.2f}"
        )


if __name__ == "__main__":
    main()
