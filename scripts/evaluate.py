"""Build results/report.json from the out-of-fold predictions.

    uv run python scripts/evaluate.py --ours deberta_j03

Everything here uses predictions made on posts the model never trained on.
The conformal checks are cross-fitted: the cut-off for fold k is calibrated
on the other four folds' predictions and tested on fold k.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from saath import conformal, data, metrics, models, queue_sim

OOF = ROOT / "results" / "oof"
OUT = ROOT / "results"


def load_oof(tag):
    return np.load(OOF / f"{tag}.npy")


def model_row(name, y, level, score=None, ours=False):
    row = {"name": name, "ours": ours}
    for key, fn in {
        "qwk": metrics.qwk,
        "exact": lambda a, b: float((a == b).mean()),
        "hr_recall": lambda a, b: metrics.high_risk_counts(a, b)[0] / max(sum(metrics.high_risk_counts(a, b)[:2]), 1),
        "hr_precision": lambda a, b: (
            metrics.high_risk_counts(a, b)[0]
            / max(metrics.high_risk_counts(a, b)[0] + metrics.high_risk_counts(a, b)[2], 1)
        ),
    }.items():
        row[key] = metrics.bootstrap(fn, y, level, n=1000)
    if score is not None:
        from sklearn.metrics import roc_auc_score

        row["hr_auroc"] = metrics.bootstrap(lambda a, s: roc_auc_score(a >= data.HIGH_RISK, s), y, score, n=1000)
    return row


def cross_fitted_conformal(exceed, y, alpha_u=0.05, alpha_l=0.1):
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(exceed, y))
    urgent = np.zeros(len(y), dtype=bool)
    sets = np.zeros((len(y), 2), dtype=int)
    for _, te in folds:
        cal = np.setdiff1d(np.arange(len(y)), te)
        u = conformal.UrgentThreshold(alpha_u).fit(models.risk_score(exceed[cal]), y[cal])
        urgent[te] = u.urgent(models.risk_score(exceed[te]))
        ls = conformal.LevelSets(alpha_l).fit(models.to_level_proba(exceed[cal]), y[cal])
        sets[te] = ls.predict(models.to_level_proba(exceed[te]))
    hr = y >= data.HIGH_RISK
    cov = conformal.covered(sets, y)
    return {
        "urgent": {
            "alpha": alpha_u,
            "miss_rate": float((~urgent[hr]).mean()),
            "miss_ci": metrics.bootstrap(lambda u: float((~u).mean()), urgent[hr], n=1000)[1:],
            "urgent_share": float(urgent.mean()),
            "precision": float(hr[urgent].mean()),
        },
        "level_sets": {
            "alpha": alpha_l,
            "coverage": float(cov.mean()),
            "mean_width": float((sets[:, 1] - sets[:, 0] + 1).mean()),
            "singleton_share": float((sets[:, 0] == sets[:, 1]).mean()),
            "coverage_by_level": {int(k): float(cov[y == k].mean()) for k in range(data.N_LEVELS)},
        },
        "_urgent_mask": urgent,
    }


def queue_report(y, policies, loads, seeds):
    series = {name: {"mean": [], "lo": [], "hi": []} for name in policies}
    paired = []
    for load in loads:
        scn = queue_sim.Scenario(load=load)
        res = queue_sim.compare(policies, y, scn, seeds=seeds)
        for name in policies:
            a = res[name]["hr_abandon"]
            # Interval for the mean across replays, not the spread of single replays.
            half = 1.96 * np.nanstd(a, ddof=1) / np.sqrt(np.isfinite(a).sum())
            lo, hi = np.nanmean(a) - half, np.nanmean(a) + half
            series[name]["mean"].append(float(np.nanmean(a)))
            series[name]["lo"].append(float(lo))
            series[name]["hi"].append(float(hi))
        diff = res["First come"]["hr_abandon"] - res["Saath"]["hr_abandon"]
        paired.append(
            {
                "load": load,
                "fifo": float(np.nanmean(res["First come"]["hr_abandon"])),
                "saath": float(np.nanmean(res["Saath"]["hr_abandon"])),
                "keyword": float(np.nanmean(res["Keyword"]["hr_abandon"])),
                "llm": float(np.nanmean(res["Frontier LLM"]["hr_abandon"])),
                "oracle": float(np.nanmean(res["Oracle"]["hr_abandon"])),
                # Interval for the mean difference across replays (paired by seed).
                "reduction_ci": list(metrics.bootstrap(np.nanmean, diff, n=2000)[1:]),
                "relative_reduction": float(
                    1 - np.nanmean(res["Saath"]["hr_abandon"]) / np.nanmean(res["First come"]["hr_abandon"])
                ),
                "share_of_oracle_gain": float(
                    np.nanmean(diff) / np.nanmean(res["First come"]["hr_abandon"] - res["Oracle"]["hr_abandon"])
                ),
                "saath_wins_share": float(np.nanmean(diff > 0)),
                "hr_wait_fifo": float(np.nanmean(res["First come"]["hr_wait_mean"])),
                "hr_wait_saath": float(np.nanmean(res["Saath"]["hr_wait_mean"])),
                "low_abandon_fifo": float(np.nanmean(res["First come"]["low_abandon"])),
                "low_abandon_saath": float(np.nanmean(res["Saath"]["low_abandon"])),
                "all_abandon_fifo": float(np.nanmean(res["First come"]["all_abandon"])),
                "all_abandon_saath": float(np.nanmean(res["Saath"]["all_abandon"])),
            }
        )
    return series, paired


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--seeds", type=int, default=300)
    args = ap.parse_args()

    df = data.load()
    y = df.level.to_numpy()
    jury = data.jury_matrix(df)

    ours = load_oof(args.ours)
    ours_level, ours_risk = models.predict_level(ours), models.risk_score(ours)
    kw = load_oof("keyword")
    tfidf = load_oof("tfidf")

    # Frontier LLM baselines: the five zero-shot labels shipped with the data.
    # Rows where a model returned nothing count as level 0 for that model.
    llm = np.nan_to_num(jury, nan=0).astype(int)
    llm_qwk = [metrics.qwk(y, llm[:, j]) for j in range(llm.shape[1])]
    best = int(np.argmax(llm_qwk))
    vote = np.round(np.nanmedian(jury, axis=1)).astype(int)

    rows = [
        model_row("Keyword rules", y, models.predict_level(kw), models.risk_score(kw)),
        model_row("TF-IDF ordinal regression", y, models.predict_level(tfidf), models.risk_score(tfidf)),
        model_row("Saath (local ensemble)", y, ours_level, ours_risk, ours=True),
        model_row("Best single frontier LLM, zero-shot", y, llm[:, best]),
        model_row("Median of 5 frontier LLMs, zero-shot", y, vote),
    ]
    vs_best = metrics.paired_bootstrap(metrics.qwk, y, ours_level, llm[:, best], n=2000)
    vs_tfidf = metrics.paired_bootstrap(metrics.qwk, y, ours_level, models.predict_level(tfidf), n=2000)

    conf = cross_fitted_conformal(ours, y)
    urgent_mask = conf.pop("_urgent_mask")

    # Calibration of the high-risk probability.
    hr = (y >= data.HIGH_RISK).astype(int)
    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(ours_risk, edges[1:-1]), 0, 9)
    reliability = [
        {
            "bin": float(edges[b] + 0.05),
            "pred": float(ours_risk[idx == b].mean()),
            "obs": float(hr[idx == b].mean()),
            "n": int((idx == b).sum()),
        }
        for b in range(10)
        if (idx == b).any()
    ]

    # Queue replay: everyone draws from the same pool of held-out posts.
    policies = {
        "First come": None,
        "Keyword": models.predict_level(kw) / 6,
        "Saath": ours_risk,
        "Frontier LLM": llm[:, best] / 6,
        "Oracle": (y >= data.HIGH_RISK).astype(float),
    }
    loads = [0.8, 0.9, 1.0, 1.1, 1.25, 1.5]
    series, paired = queue_report(y, policies, loads, seeds=range(args.seeds))
    head = next(p for p in paired if p["load"] == 1.1)

    # Cross-fitted isotonic recalibration of the displayed probability.
    from sklearn.isotonic import IsotonicRegression

    shown = np.zeros(len(y))
    for _, te in StratifiedKFold(5, shuffle=True, random_state=0).split(ours, y):
        cal_idx = np.setdiff1d(np.arange(len(y)), te)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(ours_risk[cal_idx], hr[cal_idx])
        shown[te] = iso.predict(ours_risk[te])

    report = {
        "n_posts": len(y),
        "level_counts": {int(k): int((y == k).sum()) for k in range(data.N_LEVELS)},
        "models": rows,
        "vs_best_llm_qwk": vs_best,
        "vs_tfidf_qwk": vs_tfidf,
        "models_note": "Five-fold cross-validation; every post has a different author, so no writer appears in both "
        "training and test. Brackets are 95% bootstrap intervals. LLM labels were collected zero-shot by the "
        "dataset authors.",
        **conf,
        "urgent_band_share": float(urgent_mask.mean()),
        "reliability": reliability,
        "ece_high_risk": metrics.ece(ours_risk, hr),
        "ece_high_risk_recalibrated": metrics.ece(shown, hr),
        "queue": {
            "loads": loads,
            "series": series,
            "paired": paired,
            "headline": {"load": 1.1, "fifo": head["fifo"], "saath": head["saath"]},
            "note": f"{args.seeds} replays per load, 4 counsellors, 25-minute mean conversation, 20-minute mean "
            "patience. Shaded bands are 95% intervals for the mean. Oracle knows the true level.",
        },
    }
    for extra in ["codemixed", "external", "latency"]:
        p = OUT / f"{extra}.json"
        if p.exists():
            report[extra] = json.loads(p.read_text())
    ms = report.get("latency", {}).get("cpu_ms_per_message")
    report["latency_ms"] = None if ms is None else round(ms)

    OUT.mkdir(exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ["vs_best_llm_qwk", "urgent", "level_sets"]}, indent=2))
    for r in rows:
        print(
            r["name"],
            "qwk",
            [round(v, 3) for v in r["qwk"]],
            "rec",
            round(r["hr_recall"][0], 3),
            "prec",
            round(r["hr_precision"][0], 3),
        )
    for p in paired:
        print(p["load"], {k: round(v, 3) if isinstance(v, float) else v for k, v in p.items()})


if __name__ == "__main__":
    main()
