#!/usr/bin/env python3
"""
03f_coverage_strictness.py - report a curve, not two operating points.

Why this replaces the 95/99 pair
--------------------------------
`03e` showed that most of the leave-one-mechanism-out movement lives in the
operating point rather than in the model. Once that is known, quoting recovery at
two fixed thresholds is the weakest possible summary: it reports two arbitrary
points on a curve whose shape is the actual finding. This script sweeps the
threshold and reports, per mechanism class, how coverage trades against strictness.

The summary statistic
---------------------
    s90   the STRICTEST specificity at which the held-out class is still recovered
          at 90% or better.
Read it as a false-positive budget. s90 = 0.98 means the class is still caught when
only 2% of unseen benign proteins are allowed through; s90 = none means the class
never reaches 90% recovery at any threshold this panel can estimate.

Estimator ceiling, stated rather than ignored
---------------------------------------------
The threshold is a quantile of the HELD-OUT negative scores, so with h held-out
negatives no specificity above roughly 1 - 1/h is estimable; asking for 0.999 from
77 negatives silently returns the maximum. The sweep therefore stops at
1 - 1/h rounded down, and the ceiling is printed with the results. This is the same
class of error as calibrating on training negatives, which this project already hit
once: a number that looks fine and means nothing.

Run after 02b. Uses cached embeddings.

Usage:
    python src/03f_coverage_strictness.py [--tag esm2_3B] [--seeds 30]
"""

import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "results" / "v2"
NEG_HOLDOUT_FRAC = 0.50  # larger than 03b's 0.40, to push the estimable ceiling up
TARGET = 0.90  # the recovery level s90 is defined against


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--seeds", type=int, default=30)
    a = ap.parse_args()
    suf = f"_{a.tag}" if a.tag else ""

    P = np.load(V2 / f"embeddings_positive_v2{suf}.npy")
    N = np.load(V2 / f"embeddings_negative_v2{suf}.npy")
    man = json.load(open(V2 / f"embedding_manifest_v2{suf}.json"))
    mech = json.load(open(ROOT / "data/annotations/mechanism_classes_v2.json"))

    pos_acc = [r["acc"] for r in man["positive_rows"]]
    cls_of = {p["fasta_id"]: p["mechanism_class"] for p in mech["proteins"]}
    pos_cls = np.array([cls_of.get(x, "UNMAPPED") for x in pos_acc])
    classes = sorted(
        {c for c in pos_cls if c in set(mech["holdout_eligible_classes"])}
        | {"virulence_associated_non_toxin"}
    )

    h = int(len(N) * NEG_HOLDOUT_FRAC)
    ceiling = np.floor((1 - 1.0 / h) * 1000) / 1000
    grid = np.array([s for s in np.round(np.arange(0.50, 0.995, 0.005), 3) if s <= ceiling])
    shown = [s for s in (0.50, 0.80, 0.90, 0.95, 0.98) if s <= ceiling]

    print(f"model={man['model']}  positives={len(P)}  negatives={len(N)}  held-out negatives={h}")
    print(
        f"estimable specificity ceiling = {ceiling:.3f} (1 - 1/{h}); "
        f"the sweep stops there rather than extrapolating\n"
    )

    # A flat stretch in a class curve means the class is not homogeneous: some
    # members are caught at every threshold and others at none. Per-member flag
    # rates at a reference specificity separate those, which the class mean hides.
    REF = 0.95
    ref_i = int(np.argmin(np.abs(grid - REF)))
    curves = {C: [] for C in classes}
    member = {C: [] for C in classes}
    fpr_check = []
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        p = rng.permutation(len(N))
        nte, ntr = p[:h], p[h:]
        for C in classes:
            hi = np.where(pos_cls == C)[0]
            if len(hi) == 0:
                continue
            tri = np.setdiff1d(np.arange(len(P)), hi)
            model = clf().fit(
                np.vstack([P[tri], N[ntr]]), np.r_[np.ones(len(tri)), np.zeros(len(ntr))]
            )
            s_ho = model.predict_proba(P[hi])[:, 1]
            s_nte = model.predict_proba(N[nte])[:, 1]
            t = np.quantile(s_nte, grid)  # (len(grid),)
            curves[C].append((s_ho[None, :] >= t[:, None]).mean(axis=1))
            member[C].append(s_ho >= t[ref_i])  # per-member, at REF
            fpr_check.append((s_nte[None, :] >= t[:, None]).mean(axis=1))

    fpr_mean = np.mean(fpr_check, axis=0)
    out = {
        "model": man["model"],
        "tag": a.tag or None,
        "seeds": a.seeds,
        "n_heldout_negatives": h,
        "specificity_ceiling": float(ceiling),
        "grid": grid.tolist(),
        "target_recovery": TARGET,
        "classes": {},
    }

    hdr = "".join(f"{f'@{s:.2f}':>9}" for s in shown)
    print(f"{'class':<32}{hdr}{'s90':>8}{'':>3}")
    print("-" * (32 + 9 * len(shown) + 11))
    for C in classes:
        if not curves[C]:
            continue
        m = np.mean(curves[C], axis=0)
        ok = np.where(m >= TARGET)[0]
        s90 = float(grid[ok.max()]) if len(ok) else None
        cells = "".join(f"{m[list(grid).index(s)]:>8.0%} " for s in shown)
        tail = f"{s90:>8.3f}" if s90 is not None else f"{'none':>8}"
        mark = "" if s90 is not None and s90 >= ceiling - 1e-9 else ""
        print(f"{C:<32}{cells}{tail}{mark:<3}")
        out["classes"][C] = {
            "n": int((pos_cls == C).sum()),
            "curve_mean": m.tolist(),
            "curve_sd": np.std(curves[C], axis=0).tolist(),
            "s90": s90,
        }
    print(
        f"\n{'  (achieved FPR, should be 1 - specificity)':<32}"
        + "".join(f"{fpr_mean[list(grid).index(s)]:>8.1%} " for s in shown)
    )
    out["achieved_fpr"] = fpr_mean.tolist()

    # ---- is the class homogeneous, or a mixture? ---------------------------
    print(f"\nper-member behaviour at specificity {REF:.2f}, mean over {a.seeds} seeds")
    print(f"{'class':<32}{'n':>4}{'always caught':>15}{'never caught':>14}{'in between':>12}")
    print("-" * 77)
    for C in classes:
        if not member[C]:
            continue
        r = np.mean(member[C], axis=0)  # per member, rate over seeds
        always = int((r >= 0.9).sum())
        never = int((r <= 0.1).sum())
        mid = len(r) - always - never
        print(f"{C:<32}{len(r):>4}{always:>15}{never:>14}{mid:>12}")
        out["classes"][C].update(
            {
                "member_flag_rate_at_ref": r.tolist(),
                "n_always_caught": always,
                "n_never_caught": never,
                "n_intermediate": mid,
            }
        )
    out["reference_specificity"] = REF
    print("  a class with members in both extreme columns is a mixture, and its")
    print("  single recovery number is an average over two different behaviours")

    p_out = V2 / f"coverage_strictness{suf}.json"
    json.dump(out, open(p_out, "w"), indent=2)
    print(f"\nwrote {p_out}")
    print("\ns90 is the strictest false-positive budget at which the class is still")
    print("recovered at 90%. 'none' means it never reaches 90% at any estimable threshold.")


if __name__ == "__main__":
    main()
