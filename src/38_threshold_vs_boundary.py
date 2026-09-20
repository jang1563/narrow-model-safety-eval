#!/usr/bin/env python3
"""
38_threshold_vs_boundary.py - is the big gain from growing the negative set real discrimination,
                              or a better-estimated threshold?

The result that forces this question
------------------------------------
`37` on the canonical 650M arm, with the panel's 296 negatives kept as a fixed floor, found that
ADDING pool negatives makes both unreachable classes far better rather than worse:

    phage_peptidoglycan_hydrolase   12.2% at K=0  ->  49.3% random / 47.9% nearest at K=8259
    beta_lactamase                  21.2% at K=0  ->  41.9% random / 42.1% nearest at K=8259

That is a three to fourfold improvement in the two classes this project calls unreachable, and it
contradicts §10.7.1's closing line that curating the negative set is not the repair.

🔴 Before any of that is written down, one confound has to be excluded, and it is not small. The
03b protocol holds out 40% of the negatives and sets the threshold at the 95th percentile of THAT
held-out set. At 296 negatives the held-out set is 118, so the 0.95 quantile sits between the 5th
and 6th highest score of 118, estimated from a handful of order statistics. At K=8259 the held-out
set is 3,422 and roughly 171 negatives sit above the threshold. Recovery is a nonlinear function
of the threshold, so a noisy threshold can depress mean recovery even when the estimator is
unbiased. Growing the negative set does two things at once and they have to be separated.

The decomposition, which is 03e's discipline applied to this question
--------------------------------------------------------------------
    baseline        panel negatives only, the published protocol
    boundary_only   calibration set FIXED at 118 panel negatives; TRAINING negatives grow by K
                    from the pool. Any gain here is the decision boundary getting better.
    threshold_only  TRAINING negatives FIXED at the panel's 178; the CALIBRATION set grows by K
                    from the pool. Any gain here is the threshold estimate getting more stable.
    both            both grow, which should reproduce 37's curve and is the sanity check that the
                    decomposition adds up.

PREREGISTERED, written before the run
-------------------------------------
    P1  boundary_only carries most of the gain. Then the improvement is real discrimination: a
        larger benign reference set genuinely teaches the probe to separate these classes, and
        §10.7.1's closing line is wrong and has to be corrected.

    P2  threshold_only carries most of the gain. Then the improvement is an artifact of estimating
        a tail quantile from 118 samples, the classes are no more reachable than before, and what
        this measures is the cost of a small calibration set, which is §10.8's point arriving from
        a new direction.

    Both could contribute. The reported quantity is the split, not a winner, and if the two arms
    together fall well short of `both` then the effects interact and that is reported too.

⚠️ At K=8259 the pool is exhausted, so boundary_only trains on 178 + 8259 negatives against 149
positives. Class imbalance is then 57:1. That is the realistic direction for a screen and it is
also a regime the published protocol never tested, so it is named rather than smoothed over.

Usage:
    python src/38_threshold_vs_boundary.py --arm esm2_650M
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
V3 = ROOT / "results" / "v3"
FRAC, SPEC, SEEDS = 0.40, 0.95, 30
K_GRID = [0, 500, 1500, 4000, 8259]
FAIL_AT = 0.25
CONTROL_CLASS = "virulence_associated_non_toxin"


def fit_and_score(P, tri, hi, N_train, N_cal):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
    m.fit(np.vstack([P[tri], N_train]), np.r_[np.ones(len(tri)), np.zeros(len(N_train))])
    thr = np.quantile(m.predict_proba(N_cal)[:, 1], SPEC)
    return float((m.predict_proba(P[hi])[:, 1] >= thr).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="esm2_650M")
    ap.add_argument("--seeds", type=int, default=SEEDS)
    a = ap.parse_args()
    tag = a.arm
    suf = "" if tag == "esm2_650M" else f"_{tag}"

    man = json.load(open(V3 / f"embedding_manifest_v3{suf}.json"))
    mech = json.load(open(ROOT / "data/annotations/mechanism_classes_v3.json"))
    lomo = json.load(open(V3 / f"lomo_results{suf}.json"))["leave_one_mechanism_out"]
    P = np.load(V3 / f"embeddings_positive_v3{suf}.npy")
    N_panel = np.load(V3 / f"embeddings_negative_v3{suf}.npy")
    N_pool = np.load(V3 / f"embeddings_pool_large_{tag}.npy")
    cls = {e["fasta_id"]: e["mechanism_class"] for e in mech["proteins"]}
    pcls = np.array([cls[r["acc"]] for r in man["positive_rows"]])

    failures = sorted((c for c in lomo
                       if lomo[c]["flagged_95_mean"] < FAIL_AT and c != CONTROL_CLASS),
                      key=lambda c: lomo[c]["flagged_95_mean"])
    below = {c: lomo[c]["flagged_95_mean"] for c in lomo
             if c not in failures and c != CONTROL_CLASS
             and lomo[c]["flagged_95_mean"] < 0.999}
    comparison = max(below, key=below.get)
    targets = failures + [comparison]
    ks = [k for k in K_GRID if k <= len(N_pool)]
    print(f"arm {tag}: panel {len(N_panel)}, pool {len(N_pool)}, seeds {a.seeds}")
    print(f"failures {failures}, comparison {comparison}\n")

    out = {c: {m: {} for m in ("boundary_only", "threshold_only", "both")} for c in targets}
    for c in targets:
        hi = np.where(pcls == c)[0]
        tri = np.setdiff1d(np.arange(len(P)), hi)
        for k in ks:
            acc = {"boundary_only": [], "threshold_only": [], "both": []}
            for seed in range(a.seeds):
                rng = np.random.default_rng(seed)
                perm = rng.permutation(len(N_panel))
                cut = int(len(N_panel) * FRAC)
                cal_idx, tr_idx = perm[:cut], perm[cut:]
                base_cal, base_tr = N_panel[cal_idx], N_panel[tr_idx]
                add = N_pool[rng.permutation(len(N_pool))[:k]] if k else N_pool[:0]
                # boundary_only: pool proteins join TRAINING, calibration stays the panel's 118
                acc["boundary_only"].append(
                    fit_and_score(P, tri, hi, np.vstack([base_tr, add]), base_cal))
                # threshold_only: pool proteins join CALIBRATION, training stays the panel's 178
                acc["threshold_only"].append(
                    fit_and_score(P, tri, hi, base_tr, np.vstack([base_cal, add])))
                # both: the 37 condition, pool split 40/60 the same way
                split = int(len(add) * FRAC) if k else 0
                acc["both"].append(
                    fit_and_score(P, tri, hi, np.vstack([base_tr, add[split:]]),
                                  np.vstack([base_cal, add[:split]])))
            for mode, v in acc.items():
                v = np.array(v)
                out[c][mode][str(k)] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1))}

    hdr = f"{'class':<32}{'mode':<16}" + "".join(f"{f'K={k}':>9}" for k in ks)
    print(hdr)
    print("-" * len(hdr))
    for c in targets:
        for mode in ("boundary_only", "threshold_only", "both"):
            r = out[c][mode]
            print(f"{c:<32}{mode:<16}"
                  + "".join(f"{r[str(k)]['mean'] * 100:>8.1f}%" for k in ks))
        print()

    # ---- how the gain splits -------------------------------------------------------
    print(f"{'class':<32}{'total gain':>11}{'boundary':>10}{'threshold':>11}{'share bnd':>11}")
    split = {}
    for c in targets:
        k0 = out[c]["both"][str(ks[0])]["mean"]
        kN = out[c]["both"][str(ks[-1])]["mean"]
        g_total = kN - k0
        g_bnd = out[c]["boundary_only"][str(ks[-1])]["mean"] - k0
        g_thr = out[c]["threshold_only"][str(ks[-1])]["mean"] - k0
        share = g_bnd / g_total if abs(g_total) > 1e-9 else float("nan")
        split[c] = {"total_gain_pts": g_total * 100, "boundary_gain_pts": g_bnd * 100,
                    "threshold_gain_pts": g_thr * 100, "boundary_share": share,
                    "additive_residual_pts": (g_total - g_bnd - g_thr) * 100}
        print(f"{c:<32}{g_total * 100:>+10.1f}{g_bnd * 100:>+10.1f}{g_thr * 100:>+10.1f}"
              f"{share:>11.2f}")

    fail_shares = [split[c]["boundary_share"] for c in failures]
    if all(s > 0.6 for s in fail_shares):
        verdict = ("BOUNDARY: most of the gain survives with the calibration set frozen, so a "
                   "larger benign reference set genuinely improves discrimination on these "
                   "classes and §10.7.1's closing line needs correcting")
    elif all(s < 0.4 for s in fail_shares):
        verdict = ("THRESHOLD: the gain largely disappears with the calibration set frozen, so it "
                   "was a tail-quantile estimated from 118 samples. The classes are no more "
                   "reachable and this measures the cost of a small calibration set")
    else:
        verdict = (f"MIXED: boundary share {[round(s, 2) for s in fail_shares]} for the failing "
                   "classes, so both effects contribute materially and neither reading stands "
                   "alone")
    print(f"\nverdict: {verdict}")

    dest = V3 / f"threshold_vs_boundary_{tag}.json"
    json.dump({"arm": tag, "K_grid": ks, "seeds": a.seeds,
               "failures": failures, "comparison": comparison,
               "curves": out, "gain_split": split,
               "imbalance_note": (f"at K={ks[-1]} boundary_only trains on "
                                  f"{len(N_panel) - int(len(N_panel) * FRAC) + ks[-1]} negatives "
                                  f"against {len(P) - 0} positives, a regime the published "
                                  "protocol never tested"),
               "verdict": verdict}, open(dest, "w"), indent=2)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
