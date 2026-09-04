#!/usr/bin/env python3
"""
03e_negative_difficulty_curve.py - is the recovery drop difficulty, size, or threshold?

The problem
-----------
Between panel v2.1 (100 negatives) and v2.2 (154 negatives) the leave-one-mechanism-out
recovery of several classes collapsed: beta-lactamase from 63% to 21% at the 95%
operating point, pore-forming from 100% to 57%. Positives and model were unchanged.
Reporting that as "the negatives got harder" is a hypothesis, not a measurement, and
three different mechanisms could produce it:

  (a) threshold   the operating point is calibrated on HELD-OUT negatives. Harder
                  held-out negatives score higher, the 95th percentile moves up, and
                  fewer held-out positives clear it. Nothing about the model changed.
  (b) boundary    harder TRAINING negatives move the decision boundary itself.
  (c) size        v2.2 simply has more negatives, so the classifier sees more data and
                  the class balance shifts. This is not difficulty at all.

This script separates them with a 2x2 design over nested negative tiers.

Nested tiers, strictly T1 < T2 < T3
-----------------------------------
  T1  lab_strain            49   the original two lab strains
  T2  + pathogen cytoplasmic 100  adds same-organism, non-exported negatives
  T3  + pathogen secreted   154  adds same-organism, same-localization negatives

Arms
----
Every arm holds out exactly HELDOUT_N negatives for calibration, so the quantile
estimator has identical precision everywhere and cannot itself explain a difference.

  full        train on tier T, calibrate on tier T      what was actually reported
  matched     train on tier T subsampled to TRAIN_N, calibrate on tier T
              -> difficulty with SIZE held constant, so (c) is removed
  calib_only  train on T1 subsampled to TRAIN_N, calibrate on tier T
              -> isolates (a): identical boundary, different threshold
  train_only  train on tier T subsampled to TRAIN_N, calibrate on T1
              -> isolates (b): identical threshold source, different boundary

Reading the output: if `calib_only` reproduces the drop and `train_only` does not,
the effect is the operating point rather than the model. If both move, both mechanisms
contribute. If `matched` is flat while `full` drops, the effect was sample size.

Two tier-level difficulty measures are reported alongside, so the x-axis is a measured
quantity and not just a label:
  baseline AUROC        positives against that tier's negatives
  nn_similarity         median over positives of the maximum cosine similarity to any
                        negative in the tier. Higher means the negatives crowd the
                        positives more closely.

Run after 02b. Uses cached embeddings.

Usage:
    python src/03e_negative_difficulty_curve.py [--tag smoke150M] [--seeds 20]
"""

import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "results" / "v2"
HELDOUT_N = 20  # negatives reserved for threshold calibration, identical in every arm
TRAIN_N = 29  # training negatives in every size-matched arm = |T1| - HELDOUT_N


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))


def cv_auroc(X, y, seed=0):
    s = cross_val_score(
        clf(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed), scoring="roc_auc"
    )
    return float(s.mean()), float(s.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--seeds", type=int, default=20)
    a = ap.parse_args()
    suf = f"_{a.tag}" if a.tag else ""

    P = np.load(V2 / f"embeddings_positive_v2{suf}.npy")
    N = np.load(V2 / f"embeddings_negative_v2{suf}.npy")
    man = json.load(open(V2 / f"embedding_manifest_v2{suf}.json"))
    mech = json.load(open(ROOT / "data/annotations/mechanism_classes_v2.json"))
    panel = json.load(open(ROOT / "data/sequences/panel_v2_manifest.json"))

    pos_acc = [r["acc"] for r in man["positive_rows"]]
    neg_acc = [r["acc"] for r in man["negative_rows"]]
    assert len(pos_acc) == P.shape[0] and len(neg_acc) == N.shape[0]

    meta = {n["acc"]: n for n in panel["negatives"]}
    lab = np.array([bool(meta[x].get("lab_strain")) for x in neg_acc])
    blk = np.array([meta[x]["block"] for x in neg_acc])

    # strictly nested by construction
    t1 = np.where(lab)[0]
    t2 = np.where(lab | (blk == "cytoplasmic_housekeeping"))[0]
    t3 = np.arange(len(neg_acc))
    assert set(t1) <= set(t2) <= set(t3), "tiers are not nested"
    TIERS = [
        ("T1_lab_strain", t1),
        ("T2_plus_pathogen_cytoplasmic", t2),
        ("T3_plus_pathogen_secreted", t3),
    ]

    cls_of = {p["fasta_id"]: p["mechanism_class"] for p in mech["proteins"]}
    pos_cls = np.array([cls_of.get(x, "UNMAPPED") for x in pos_acc])
    classes = sorted(
        {c for c in pos_cls if c in set(mech["holdout_eligible_classes"])}
        | {"virulence_associated_non_toxin"}
    )

    # ---- tier-level difficulty, independent of the holdout experiment --------
    Pn = P / np.linalg.norm(P, axis=1, keepdims=True)
    Nn = N / np.linalg.norm(N, axis=1, keepdims=True)
    tier_stats = {}
    print(f"{'tier':<32}{'n':>5}{'baseline':>12}{'nn_sim':>9}")
    print("-" * 58)
    for name, idx in TIERS:
        Xt = np.vstack([P, N[idx]])
        yt = np.r_[np.ones(len(P)), np.zeros(len(idx))]
        m, s = cv_auroc(Xt, yt)
        nn = float(np.median((Pn @ Nn[idx].T).max(axis=1)))
        tier_stats[name] = {
            "n_negative": int(len(idx)),
            "baseline_auroc": [m, s],
            "nn_similarity_median": nn,
        }
        print(f"{name:<32}{len(idx):>5}{m:>9.3f}±{s:.3f}{nn:>9.3f}")

    # ---- the 2x2 decomposition ---------------------------------------------
    ARMS = ["full", "matched", "calib_only", "train_only"]
    res = {arm: {t[0]: {c: [] for c in classes} for t in TIERS} for arm in ARMS}
    fpr = {arm: {t[0]: [] for t in TIERS} for arm in ARMS}

    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        # one T1 calibration/training draw per seed, shared by the arms that need it
        t1p = rng.permutation(t1)
        t1_cal = t1p[:HELDOUT_N]
        for name, idx in TIERS:
            p = rng.permutation(idx)
            cal, pool = p[:HELDOUT_N], p[HELDOUT_N:]
            # Training must never contain a calibration negative. A negative the model
            # has already fit scores near zero, which drags the quantile down and
            # inflates recovery. The cross arms draw their training and calibration
            # sets from different tiers, so the disjointness has to be enforced
            # explicitly rather than falling out of a single permutation.
            t1_minus_cal = np.setdiff1d(t1, cal)
            tier_minus_t1cal = np.setdiff1d(idx, t1_cal)
            arms = {
                "full": (pool, cal),
                "matched": (rng.permutation(pool)[:TRAIN_N], cal),
                "calib_only": (rng.permutation(t1_minus_cal)[:TRAIN_N], cal),
                "train_only": (rng.permutation(tier_minus_t1cal)[:TRAIN_N], t1_cal),
            }
            for _arm, (_tr, _ca) in arms.items():
                assert not set(np.asarray(_tr).tolist()) & set(np.asarray(_ca).tolist()), (
                    f"calibration negative leaked into training in arm {_arm}"
                )
            for arm, (tr, ca) in arms.items():
                if len(tr) < TRAIN_N or len(ca) < HELDOUT_N:
                    continue
                for C in classes:
                    hi = np.where(pos_cls == C)[0]
                    if len(hi) == 0:
                        continue
                    tri = np.setdiff1d(np.arange(len(P)), hi)
                    Xtr = np.vstack([P[tri], N[tr]])
                    ytr = np.r_[np.ones(len(tri)), np.zeros(len(tr))]
                    model = clf().fit(Xtr, ytr)
                    s_ho = model.predict_proba(P[hi])[:, 1]
                    s_ca = model.predict_proba(N[ca])[:, 1]
                    t95 = float(np.quantile(s_ca, 0.95))
                    t99 = float(np.quantile(s_ca, 0.99))
                    res[arm][name][C].append(
                        (float((s_ho >= t95).mean()), float((s_ho >= t99).mean()))
                    )
                    fpr[arm][name].append(float((s_ca >= t95).mean()))

    # ---- report -------------------------------------------------------------
    out = {
        "model": man["model"],
        "tag": a.tag or None,
        "seeds": a.seeds,
        "heldout_n": HELDOUT_N,
        "train_n_matched": TRAIN_N,
        "tiers": tier_stats,
        "arms": {},
    }

    def paired_delta_ci(v1, v3, boot=5000, seed=0):
        """Bootstrap CI for the paired T1 -> T3 difference across seeds.

        Seeds are paired: the same random draw produced both tier results, so the
        difference is taken within a seed and resampled. No distributional assumption,
        and it degrades gracefully when a class is at 100% in both tiers."""
        d = np.asarray(v3) - np.asarray(v1)
        if len(d) == 0:
            return None, None, None
        rng = np.random.default_rng(seed)
        bs = rng.choice(d, size=(boot, len(d)), replace=True).mean(axis=1)
        return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    for arm in ARMS:
        print(
            f"\n=== arm: {arm} ===   flagged@95, mean ± sd over {a.seeds} seeds; "
            f"delta is paired T1->T3 with a 95% bootstrap CI"
        )
        hdr = "".join(f"{t[0].split('_')[0]:>14}" for t in TIERS)
        print(f"{'class':<32}{hdr}{'delta @95 [95% CI]':>26}{'delta @99':>11}")
        print("-" * (32 + 14 * len(TIERS) + 37))
        arm_out = {}
        for C in classes:
            per_tier = [np.array(res[arm][t[0]][C]) for t in TIERS]  # (seeds, 2)
            if any(len(x) == 0 for x in per_tier):
                continue
            m95 = [x[:, 0].mean() for x in per_tier]
            s95 = [x[:, 0].std() for x in per_tier]
            d95, lo, hi = paired_delta_ci(per_tier[0][:, 0], per_tier[-1][:, 0])
            d99, _, _ = paired_delta_ci(per_tier[0][:, 1], per_tier[-1][:, 1])
            sig = "" if (lo is not None and lo <= 0 <= hi) else " *"
            cells = "".join(f"{m:>8.0%}±{s:<5.0%}" for m, s in zip(m95, s95))
            print(f"{C:<32}{cells}{d95:>+11.0%} [{lo:+.0%},{hi:+.0%}]{sig:<2}{d99:>+10.0%}")
            arm_out[C] = {
                "flagged95_mean_by_tier": [float(v) for v in m95],
                "flagged95_sd_by_tier": [float(v) for v in s95],
                "flagged99_mean_by_tier": [float(x[:, 1].mean()) for x in per_tier],
                "delta95_t1_t3": d95,
                "delta95_ci": [lo, hi],
                "delta99_t1_t3": d99,
                "ci_excludes_zero": bool(not (lo <= 0 <= hi)),
            }
        f = [np.mean(fpr[arm][t[0]]) if fpr[arm][t[0]] else np.nan for t in TIERS]
        print(f"{'  (held-out FPR, target 5%)':<32}" + "".join(f"{v:>13.1%} " for v in f))
        out["arms"][arm] = {"per_class": arm_out, "heldout_fpr_by_tier": [float(v) for v in f]}
    out["note_ci"] = (
        "* marks a paired T1->T3 delta whose 95% bootstrap CI excludes zero. "
        "Seeds are paired across tiers, so the CI reflects draw-to-draw "
        "variation in the negative split, not sampling of proteins."
    )

    p = V2 / f"negative_difficulty_curve{suf}.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nwrote {p}")
    print("\nReading guide: compare `matched` against `full` to remove sample size, then")
    print("`calib_only` against `train_only` to see whether the drop lives in the operating")
    print("point or in the decision boundary.")


if __name__ == "__main__":
    main()
