#!/usr/bin/env python3
"""
58_conformal_operating_point.py - the per-class table at a threshold whose false-positive rate has
                                  a guarantee, beside the published one that does not.

The gap this closes
-------------------
`src/45_negative_test_set_audit.py` established that the published `np.quantile` threshold does not
return its own nominal rate on negatives held back from calibration, and identified the conformal
threshold as the estimator that does. `src/48` and `src/49` then confirmed it on two routes and two
arms. **None of that was ever applied to the table anyone reads.** Every recovery figure in
`docs/MECHANISM_GENERALIZATION.md` still comes from `np.quantile`: `03b` line 93, and the same call
in 03e, 03f, 03h, 03j and 15e.

This script does not replace that table. It recomputes it at both estimators, on identical folds,
models and scores, so the estimator is the only thing that differs, and reports them side by side.
Replacing the published figures would break comparability with every number in that document and
with the frozen v2 panel, for a gain that is already visible in a parallel column.

The reproduction gate, which runs first
---------------------------------------
This script re-implements `03b`'s fold, so before any conformal number is printed it checks that its
own `np.quantile` column reproduces the published `lomo_results.json` per class, exactly. If the
folds have drifted apart the conformal column is measuring two things at once and the comparison is
void, so the script says so and exits non-zero. A recomputation that does not reproduce the thing it
is extending is not an extension.

What conformal cannot do, which is the point
--------------------------------------------
The conformal threshold is the k-th largest calibration score with k = floor((m+1)*alpha), so it
exists only when k >= 1. The fold hands it m = int(len(N) * 0.40) negatives:

    v2, 154 negatives -> m = 61   alpha 5%: k=3, guarantee 4.84%   alpha 1%: k=0, UNREACHABLE
    v3, 296 negatives -> m = 118  alpha 5%: k=5, guarantee 4.20%   alpha 1%: k=1, guarantee 0.84%

So on the **frozen v2 panel, the published `flagged@99` column has no conformal counterpart at all.**
`np.quantile` returns a number there and that number has no finite-sample guarantee behind it. That
is not a defect of this script; it is the resolution ceiling of the panel showing up in the column
that quotes the strictest budget.

Usage:
    python src/58_conformal_operating_point.py --panel v2
    python src/58_conformal_operating_point.py --panel v3 --tag esmc_600M
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
# 03b's constants, copied rather than imported so a change there cannot silently change what this
# script means. The reproduction gate below is what notices if they drift apart.
SEEDS = [0, 1, 2, 3, 4]
NEG_HOLDOUT_FRAC = 0.40
ALPHAS = [0.05, 0.01]


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))


def quantile_threshold(s_cal, alpha):
    """The published estimator: interpolates, and has no finite-sample guarantee."""
    return float(np.quantile(s_cal, 1 - alpha)), None, None


def conformal_threshold(s_cal, alpha):
    """k-th largest calibration score, k = floor((m+1)*alpha).

    A fresh exchangeable negative exceeds this with probability at most k/(m+1) <= alpha. When m is
    too small for k to reach 1 there is no such score, and the caller is told rather than handed a
    number that looks like a threshold.
    """
    m = len(s_cal)
    k = int(np.floor((m + 1) * alpha))
    if k < 1:
        return None, 0, None
    return float(np.sort(s_cal)[-k]), k, k / (m + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="v2", choices=["v2", "v3"])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    suf = f"_{a.tag}" if a.tag else ""
    RES = ROOT / "results" / a.panel

    P = np.load(RES / f"embeddings_positive_{a.panel}{suf}.npy")
    N = np.load(RES / f"embeddings_negative_{a.panel}{suf}.npy")
    man = json.load(open(RES / f"embedding_manifest_{a.panel}{suf}.json"))
    mech = json.load(open(ROOT / f"data/annotations/mechanism_classes_{a.panel}.json"))
    published = json.load(open(RES / f"lomo_results{suf}.json"))["leave_one_mechanism_out"]

    pos_acc = [r["acc"] for r in man["positive_rows"]]
    cls_of = {p["fasta_id"]: p["mechanism_class"] for p in mech["proteins"]}
    pos_cls = np.array([cls_of.get(x, "UNMAPPED") for x in pos_acc])

    m_cal = int(len(N) * NEG_HOLDOUT_FRAC)
    print(f"panel {a.panel}{suf}  positives={len(P)}  negatives={len(N)}  "
          f"calibration m={m_cal}  seeds={len(SEEDS)}")
    for al in ALPHAS:
        _, k, g = conformal_threshold(np.zeros(m_cal), al)
        print(f"  nominal {al:.0%}: conformal k={k}  "
              + (f"guarantee {g:.2%}" if g else "UNREACHABLE at this calibration size"))

    est = {"quantile": quantile_threshold, "conformal": conformal_threshold}
    out = {}
    for C in sorted(set(pos_cls)):
        hi = np.where(pos_cls == C)[0]
        if len(hi) == 0 or C == "UNMAPPED":
            continue
        cell = {name: {str(al): {"recovery": [], "realized_fp": []} for al in ALPHAS}
                for name in est}
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            nperm = rng.permutation(len(N))
            ncut = int(len(N) * NEG_HOLDOUT_FRAC)
            nte, ntr = nperm[:ncut], nperm[ncut:]
            tri = np.array([i for i in range(len(P)) if i not in set(hi.tolist())])
            model = clf().fit(np.vstack([P[tri], N[ntr]]),
                              np.r_[np.ones(len(tri)), np.zeros(len(ntr))])
            s_ho = model.predict_proba(P[hi])[:, 1]
            s_nte = model.predict_proba(N[nte])[:, 1]
            for name, fn in est.items():
                for al in ALPHAS:
                    t, _, _ = fn(s_nte, al)
                    if t is None:
                        continue
                    cell[name][str(al)]["recovery"].append(float((s_ho >= t).mean()))
                    # the realized rate on the same negatives the threshold was set on, so a
                    # recovery difference can be read against the budget it was bought at
                    cell[name][str(al)]["realized_fp"].append(float((s_nte >= t).mean()))
        out[C] = {"n": int(len(hi))}
        for name in est:
            for al in ALPHAS:
                r = cell[name][str(al)]
                out[C][f"{name}_{al}"] = (
                    {"recovery": float(np.mean(r["recovery"])),
                     "realized_fp": float(np.mean(r["realized_fp"]))}
                    if r["recovery"] else None)

    # ---- the reproduction gate --------------------------------------------------------------
    drift = []
    for C, v in out.items():
        pub = published.get(C)
        if not pub:
            continue
        for al, key in ((0.05, "flagged_95_mean"), (0.01, "flagged_99_mean")):
            mine = v.get(f"quantile_{al}")
            if mine is None or key not in pub:
                continue
            if abs(mine["recovery"] - pub[key]) > 1e-9:
                drift.append((C, al, mine["recovery"], pub[key]))
    print()
    if drift:
        print("XX the quantile column does NOT reproduce lomo_results.json, so the conformal")
        print("   column is not attributable to the estimator. Nothing below may be quoted.")
        for C, al, mine, pub in drift:
            print(f"   {C} at {al:.0%}: this script {mine:.6f}, published {pub:.6f}")
    else:
        print("OK the quantile column reproduces lomo_results.json exactly, so every difference")
        print("   below is attributable to the threshold rule alone")

    # ---- the table -------------------------------------------------------------------------
    print()
    hdr = (f"{'class':<34}{'n':>3}  {'quant@95':>9}{'conf@95':>9}{'Δ':>7}  "
           f"{'fp@95 q':>8}{'fp@95 c':>8}  {'quant@99':>9}{'conf@99':>9}")
    print(hdr + "\n" + "-" * len(hdr))
    def pct(cell, field="recovery", w=9, dp=0):
        if cell is None:
            return f"{'—':>{w}}"
        return f"{cell[field]:>{w}.{dp}%}"

    rows = sorted(out.items(),
                  key=lambda kv: -((kv[1].get("quantile_0.05") or {"recovery": 0})["recovery"]))
    for C, v in rows:
        q5, c5 = v.get("quantile_0.05"), v.get("conformal_0.05")
        q1, c1 = v.get("quantile_0.01"), v.get("conformal_0.01")
        delta = (f"{(c5['recovery'] - q5['recovery']) * 100:+.1f}" if (q5 and c5) else "—")
        print(f"{C:<34}{v['n']:>3}  "
              f"{pct(q5, w=8)}{pct(c5)}{delta:>7}  "
              f"{pct(q5, 'realized_fp', 8, 2)}{pct(c5, 'realized_fp', 8, 2)}  "
              f"{pct(q1, w=8)}{pct(c1)}")

    unreachable = [al for al in ALPHAS
                   if conformal_threshold(np.zeros(m_cal), al)[1] < 1]
    res = {"panel": a.panel, "tag": a.tag, "seeds": SEEDS,
           "calibration_m": m_cal, "neg_holdout_frac": NEG_HOLDOUT_FRAC,
           "conformal_k": {str(al): conformal_threshold(np.zeros(m_cal), al)[1] for al in ALPHAS},
           "conformal_guarantee": {str(al): conformal_threshold(np.zeros(m_cal), al)[2]
                                   for al in ALPHAS},
           "alphas_unreachable_by_conformal": [float(x) for x in unreachable],
           "reproduces_published_quantile": not drift,
           "drift": [{"class": c, "alpha": al, "recomputed": mi, "published": pu}
                     for c, al, mi, pu in drift],
           "per_class": out}
    dest = RES / f"conformal_operating_point{suf}.json"
    json.dump(res, open(dest, "w"), indent=2)
    print(f"\nwrote {dest}")
    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
