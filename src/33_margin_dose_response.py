#!/usr/bin/env python3
"""
33_margin_dose_response.py - does removing more benign neighbours keep helping, or does it stop?

The question §10.7 left open
---------------------------
§10.7 removed the **K=10** training negatives nearest each held-out class and found a real,
replicated, small effect: +8.8 points for beta-lactamase and +7.1 for the phage class on v3,
every interval excluding zero, closing 8 to 11% of the distance to a recovered class. Its
standing caveat names this experiment: *"whether the effect scales with how many benign
neighbours are removed is untested, and a dose-response curve is the obvious next check: an
effect that saturates at 10 and one that grows to 50 imply different things about how much of
the failure is proximity."*

The two readings differ operationally, which is why the curve is worth a run:

  **saturates early**   a small, nameable set of benign proteins is responsible. A curator could
                        in principle find and handle them, and the failure has a local cause.
  **grows with K**      the class sits inside a dense benign region rather than beside a few
                        specific proteins. No curation of the negative set fixes it, and §10.7's
                        "closes a tenth" is a statement about the whole neighbourhood.
  **peaks then falls**  removing many negatives degrades the boundary generally, so the useful
                        range is bounded and the random control is what separates the two causes.

Design, identical to §10.7 except that K is swept
-------------------------------------------------
Removal applies to the TRAINING negatives only, so the 95th-percentile threshold stays calibrated
on an untouched split and nothing can move because the threshold moved. At each K the targeted
removal is compared against DRAWS random removals of the same size, paired within seed, and the
attributable effect is the targeted result minus that seed's own random mean.

⚠️ K=80 removes 45% of the ~178 training negatives, so the arms are not equally reliable across
the sweep. Training-negative count is reported per K, and the random control carries the
size reduction at every K rather than only at the smallest.

⚠️ Failing classes are defined by **recovery below 25%**, not by margin, for the reason recorded
in the seventh DATA_CORRECTIONS entry: selecting the test set with the predictor under test is
circular, and a first version of §10.7 did exactly that.

PREREGISTERED, written before the run
-------------------------------------
    SATURATING   the attributable effect at K=40 is within 2 points of its value at K=10 for both
                 failing classes. The cause is local and a small set of proteins carries it.
    SCALING      the effect at K=40 exceeds K=10 by more than 5 points for both. The class sits in
                 a dense benign region and curation cannot fix it.
    NON-MONOTONE anything else, including a peak inside the range, reported as the curve rather
                 than summarised.

Usage:
    python src/33_margin_dose_response.py --panel v3
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RES_ROOT = ROOT / "results"
FRAC, SPEC, SEEDS, DRAWS = 0.40, 0.95, 30, 25
K_GRID = [5, 10, 20, 40, 80]
FAIL_AT = 0.25
CONTROL_CLASS = "virulence_associated_non_toxin"


def cos(A, B):
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    return A @ B.T


def recover(P, N, hi, tri, ntr, nte):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
    m.fit(np.vstack([P[tri], N[ntr]]), np.r_[np.ones(len(tri)), np.zeros(len(ntr))])
    thr = np.quantile(m.predict_proba(N[nte])[:, 1], SPEC)
    return float((m.predict_proba(P[hi])[:, 1] >= thr).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="v3", choices=["v2", "v3"])
    pv = ap.parse_args().panel
    RES = RES_ROOT / pv

    man = json.load(open(RES / f"embedding_manifest_{pv}.json"))
    mech = json.load(open(ROOT / f"data/annotations/mechanism_classes_{pv}.json"))
    lomo = json.load(open(RES / "lomo_results.json"))["leave_one_mechanism_out"]
    sfc = json.load(open(RES / "second_failure_class.json"))
    P = np.load(RES / f"embeddings_positive_{pv}.npy")
    N = np.load(RES / f"embeddings_negative_{pv}.npy")
    cls = {e["fasta_id"]: e["mechanism_class"] for e in mech["proteins"]}
    pcls = np.array([cls[r["acc"]] for r in man["positive_rows"]])
    margins = {c: v["margin"] for c, v in sfc["classes"].items()}

    failures = sorted((c for c in margins
                       if lomo[c]["flagged_95_mean"] < FAIL_AT and c != CONTROL_CLASS),
                      key=lambda c: lomo[c]["flagged_95_mean"])
    below = {c: lomo[c]["flagged_95_mean"] for c in margins
             if c not in failures and c != CONTROL_CLASS
             and lomo[c]["flagged_95_mean"] < 0.999}
    comparison = max(below, key=below.get)
    targets = failures + [comparison]

    simPN = cos(P, N)
    n_train = len(N) - int(len(N) * FRAC)
    print(f"panel {pv}: {n_train} training negatives per fold, K swept over {K_GRID}")
    print(f"  failing classes: {failures}")
    print(f"  comparison: {comparison} at {below[comparison] * 100:.0f}%\n")

    out = {c: {} for c in targets}
    hdr = f"{'class':<32}{'K':>5}{'std':>8}{'near':>8}{'attr':>8}{'ci95':>18}{'pctile':>8}"
    print(hdr)
    print("-" * len(hdr))
    for c in targets:
        hi = np.where(pcls == c)[0]
        tri = np.setdiff1d(np.arange(len(P)), hi)
        rank = list(np.argsort(-simPN[hi].max(0)))
        for K in K_GRID:
            std, near, randm = [], [], []
            for seed in range(SEEDS):
                rng = np.random.default_rng(seed)
                perm = rng.permutation(len(N))
                cut = int(len(N) * FRAC)
                nte, ntr = perm[:cut], perm[cut:]
                tset = set(ntr.tolist())
                std.append(recover(P, N, hi, tri, ntr, nte))
                drop = [i for i in rank if i in tset][:K]
                near.append(recover(P, N, hi, tri,
                                    np.array([i for i in ntr if i not in set(drop)]), nte))
                rr = []
                for d in range(DRAWS):
                    rd = np.random.default_rng(1000 * seed + d)
                    dr = set(rd.choice(ntr, size=len(drop), replace=False).tolist())
                    rr.append(recover(P, N, hi, tri,
                                      np.array([i for i in ntr if i not in dr]), nte))
                randm.append(rr)
            std, near, randm = np.array(std), np.array(near), np.array(randm)
            attr = near - randm.mean(1)
            se = attr.std(ddof=1) / np.sqrt(len(attr))
            pct = np.array([(randm[s] < near[s]).mean() for s in range(SEEDS)])
            out[c][str(K)] = {
                "standard": float(std.mean()), "nearest": float(near.mean()),
                "random_mean": float(randm.mean()),
                "attributable_pts": float(attr.mean()) * 100,
                "ci95": [float(attr.mean() - 1.96 * se) * 100,
                         float(attr.mean() + 1.96 * se) * 100],
                "mean_within_seed_percentile": float(pct.mean()),
                "training_negatives_left": n_train - K}
            r = out[c][str(K)]
            print(f"{c:<32}{K:>5}{std.mean() * 100:>7.1f}%{near.mean() * 100:>7.1f}%"
                  f"{r['attributable_pts']:>+8.1f}"
                  f"   [{r['ci95'][0]:>+5.1f},{r['ci95'][1]:>+5.1f}]"
                  f"{pct.mean() * 100:>8.0f}", flush=True)
        print()

    # ---- shape of the curve --------------------------------------------------------
    def at(c, k):
        return out[c][str(k)]["attributable_pts"]

    deltas = {c: at(c, 40) - at(c, 10) for c in failures}
    peaks = {c: max(K_GRID, key=lambda k: at(c, k)) for c in failures}
    print("K=10 to K=40 change in the attributable effect:")
    for c in failures:
        print(f"  {c:<32}{at(c, 10):>+6.1f} -> {at(c, 40):>+6.1f} "
              f"= {deltas[c]:>+5.1f} pts, peak at K={peaks[c]}")
    print(f"comparison class {comparison}: "
          + ", ".join(f"K{k} {at(comparison, k):+.1f}" for k in K_GRID))

    saturating = all(abs(deltas[c]) <= 2.0 for c in failures)
    scaling = all(deltas[c] > 5.0 for c in failures)
    if saturating:
        verdict = ("SATURATING: the effect at K=40 is within 2 points of K=10 for every failing "
                   "class, so a small set of benign proteins carries it and the cause is local")
    elif scaling:
        verdict = ("SCALING: the effect grows by more than 5 points from K=10 to K=40 for every "
                   "failing class, so the class sits inside a dense benign region and curating "
                   "the negative set cannot fix it")
    else:
        verdict = ("NON-MONOTONE: " + "; ".join(
            f"{c} {at(c, 10):+.1f} at K=10, {at(c, 40):+.1f} at K=40, peak K={peaks[c]}"
            for c in failures) + ". Reported as the curve rather than a single number")
    print(f"\nverdict: {verdict}")

    dest = RES / "margin_dose_response.json"
    json.dump({"panel": pv, "K_grid": K_GRID, "seeds": SEEDS, "draws_per_seed": DRAWS,
               "training_negatives_per_fold": n_train,
               "failing_classes": failures, "comparison_class": comparison,
               "curves": out,
               "k10_to_k40_change_pts": deltas,
               "peak_K": peaks,
               "verdict": verdict}, open(dest, "w"), indent=2)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
