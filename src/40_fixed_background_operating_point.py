#!/usr/bin/env python3
"""
40_fixed_background_operating_point.py - the one comparison 37, 38 and 39 cannot make

Why a third framing is needed
-----------------------------
Growing the negative set changes two things that the first three scripts can only trade off:

    37  calibrates on 118 panel negatives plus a share of the added pool, so the operating point
        drifts with K and the headline gain is partly a drifting measure.
    38  freezes one factor at a time, which shows the effects interact but leaves each arm at an
        operating point that is either the panel's or the mixture's.
    39  pins everything to the panel's 118 matched negatives. That is the right measure for the
        research panel and the wrong one for a screen: 118 curated hard negatives are an
        adversarial background, not the background a deployed screen sees.

The deployment question is neither of those. It is: hold the benign background FIXED and realistic,
then ask whether training on more benign proteins catches more of an unreachable class. That needs
a calibration background that no arm trains on, which no existing script has.

Design
------
    RESERVE   2,000 pool proteins, drawn once with a fixed seed independent of the run seeds, are
              the calibration background for every arm and every K. No arm ever trains on them, so
              the operating point is literally the same number of the same proteins everywhere.
    ARMS      training negatives are the panel's 178 plus K proteins drawn from the remaining
              6,259. K=0 is the published protocol's training set judged against this background.
    REPORT    recovery at 95% specificity on the reserved background, and alongside it the false
              positive rate the same threshold incurs on the panel's 118 matched negatives.

That second column is the price list. If recovery climbs while the hard-negative false-positive
rate climbs with it, the screen is not getting better, it is getting louder on exactly the proteins
the panel was built to make it quiet on.

PREREGISTERED, written before the run
-------------------------------------
    R1  Recovery on the fixed background rises with K for the two unreachable classes. Then adding
        benign training data genuinely helps a screen, and §10.7.1's closing line needs qualifying
        for the addition direction even though its removal experiment stands.
    R2  Recovery on the fixed background is flat or falls with K. Then the pool buys nothing a
        screen can use, and 37's three to fourfold gain lives entirely in the moving operating
        point.
    R3  Either way, the hard-negative false-positive rate is reported next to it, because a gain
        bought by a looser effective boundary on matched negatives is not a gain.

⚠️ Caveat that cannot be fixed here. The pool has internal redundancy: `36` put its effective
count at about 5,203 of 8,259 by homology, a keep rate near 0.63. So the reserved background almost
certainly contains homologs of proteins some arms train on, which inflates apparent specificity at
large K. Removing them needs the full pairwise matrix the pool never had computed. The direction of
the bias is known and it flatters the large-K arms, so an R2 result is safe against it and an R1
result is not.

Usage:
    python src/40_fixed_background_operating_point.py --arm esm2_650M
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
N_RESERVE = 2000
# Fixed and independent of the run seeds, so the background is the same proteins in every arm. The
# pool file is in harvest order, which groups by family and organism, so a head or tail slice would
# be a biased background rather than a broad one.
RESERVE_SEED = 12345
FAIL_AT = 0.25
CONTROL_CLASS = "virulence_associated_non_toxin"


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

    res_perm = np.random.default_rng(RESERVE_SEED).permutation(len(N_pool))
    reserve_idx, avail_idx = res_perm[:N_RESERVE], res_perm[N_RESERVE:]
    BG, AVAIL = N_pool[reserve_idx], N_pool[avail_idx]
    ks = [0, 500, 1500, 4000, len(AVAIL)]
    print(f"arm {tag}: panel {len(N_panel)}, pool {len(N_pool)}, "
          f"reserved background {len(BG)}, available to add {len(AVAIL)}")
    print(f"K grid {ks}, seeds {a.seeds}\n")

    failures = sorted((c for c in lomo
                       if lomo[c]["flagged_95_mean"] < FAIL_AT and c != CONTROL_CLASS),
                      key=lambda c: lomo[c]["flagged_95_mean"])
    below = {c: lomo[c]["flagged_95_mean"] for c in lomo
             if c not in failures and c != CONTROL_CLASS
             and lomo[c]["flagged_95_mean"] < 0.999}
    comparison = max(below, key=below.get)
    targets = failures + [comparison]
    print(f"failures {failures}, comparison {comparison}\n")

    out = {c: {} for c in targets}
    for c in targets:
        hi = np.where(pcls == c)[0]
        tri = np.setdiff1d(np.arange(len(P)), hi)
        for k in ks:
            rec, fp_hard, fp_bg = [], [], []
            for seed in range(a.seeds):
                rng = np.random.default_rng(seed)
                perm = rng.permutation(len(N_panel))
                cut = int(len(N_panel) * FRAC)
                # The panel's held-out 40% is kept out of training for continuity with 03b, but it
                # is no longer the calibration set: it is the hard-negative price list.
                hard, base_tr = N_panel[perm[:cut]], N_panel[perm[cut:]]
                add = AVAIL[rng.permutation(len(AVAIL))[:k]] if k else AVAIL[:0]
                N_tr = np.vstack([base_tr, add])
                m = make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=5000, C=1.0))
                m.fit(np.vstack([P[tri], N_tr]),
                      np.r_[np.ones(len(tri)), np.zeros(len(N_tr))])
                s_bg = m.predict_proba(BG)[:, 1]
                thr = np.quantile(s_bg, SPEC)
                rec.append(float((m.predict_proba(P[hi])[:, 1] >= thr).mean()))
                fp_hard.append(float((m.predict_proba(hard)[:, 1] >= thr).mean()))
                fp_bg.append(float((s_bg >= thr).mean()))
            out[c][str(k)] = {
                "recovery": {"mean": float(np.mean(rec)), "sd": float(np.std(rec, ddof=1))},
                "fp_panel_hard": {"mean": float(np.mean(fp_hard)),
                                  "sd": float(np.std(fp_hard, ddof=1))},
                "fp_background": {"mean": float(np.mean(fp_bg)),
                                  "sd": float(np.std(fp_bg, ddof=1))}}

    # S1: the threshold is the 0.95 quantile of the background it is measured on, so the background
    # false-positive rate must be ~0.05 at every K. 2,000 points, granularity 1/2000.
    bad = [(c, k) for c in targets for k in ks
           if abs(out[c][str(k)]["fp_background"]["mean"] - 0.05) > 0.003]
    print("self-test S1 (background FP pinned at 0.05): "
          + ("PASS" if not bad else f"FAIL at {bad[:4]}"))
    if bad:
        raise SystemExit("self-test failure, results not written")

    hdr = f"{'class':<32}{'quantity':<18}" + "".join(f"{f'K={k}':>9}" for k in ks)
    print(f"\n{hdr}\n" + "-" * len(hdr))
    for c in targets:
        for key, label in (("recovery", "recovery"), ("fp_panel_hard", "FP on 118 hard")):
            print(f"{c:<32}{label:<18}"
                  + "".join(f"{out[c][str(k)][key]['mean'] * 100:>8.1f}%" for k in ks))
        print()

    print(f"{'class':<32}{'K=0':>8}{'best':>8}{'at K':>8}{'gain':>8}"
          f"{'hardFP K=0':>12}{'hardFP best':>13}   verdict")
    summary = {}
    for c in targets:
        r0 = out[c]["0"]["recovery"]["mean"]
        bk = max(ks, key=lambda k: out[c][str(k)]["recovery"]["mean"])
        rb = out[c][str(bk)]["recovery"]["mean"]
        h0 = out[c]["0"]["fp_panel_hard"]["mean"]
        hb = out[c][str(bk)]["fp_panel_hard"]["mean"]
        v = "RISES" if rb - r0 > 0.02 else "FLAT/FALLS"
        summary[c] = {"recovery_K0": r0, "recovery_best": rb, "best_K": bk,
                      "gain_pts": (rb - r0) * 100, "fp_hard_K0": h0, "fp_hard_best": hb,
                      "fp_hard_ratio": (hb / h0) if h0 > 0 else None, "verdict": v}
        print(f"{c:<32}{r0 * 100:>7.1f}%{rb * 100:>7.1f}%{bk:>8}{(rb - r0) * 100:>+8.1f}"
              f"{h0 * 100:>11.1f}%{hb * 100:>12.1f}%   {v}")

    fv = {summary[c]["verdict"] for c in failures}
    if fv == {"RISES"}:
        verdict = ("R1: on a fixed realistic background, recovery of every unreachable class rises "
                   "with the amount of benign training data, so the addition direction is a real "
                   "repair and §10.7.1's closing line needs qualifying. Read with the "
                   "hard-negative column and the homology caveat, both of which flatter this result")
    elif fv == {"FLAT/FALLS"}:
        verdict = ("R2: on a fixed realistic background the pool buys no unreachable class "
                   "anything, so 37's three to fourfold gain lived in the moving operating point "
                   "and §10.7.1's closing line stands for addition as well as removal")
    else:
        verdict = ("R1/R2 SPLIT: " + ", ".join(f"{c}={summary[c]['verdict']}" for c in failures)
                   + ", so the answer is class-specific")
    print(f"\nverdict: {verdict}")

    dest = V3 / f"fixed_background_operating_point_{tag}.json"
    json.dump({"arm": tag, "K_grid": ks, "seeds": a.seeds, "n_reserved_background": N_RESERVE,
               "reserve_seed": RESERVE_SEED, "failures": failures, "comparison": comparison,
               "curves": out, "summary": summary,
               "homology_caveat": ("36 put the pool's effective count at about 5203 of 8259, so the "
                                  "reserved background probably contains homologs of proteins the "
                                  "large-K arms train on. The bias flatters large K"),
               "verdict": verdict}, open(dest, "w"), indent=2)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
