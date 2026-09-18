#!/usr/bin/env python3
"""
15e_sae_feature_space_lomo.py - the seventh candidate: can a probe reach the
                                beta-lactamase information that §9.6 proved is there?

Where this comes from
---------------------
§9.6 refused the sixth explanation (SAE feature uniqueness) and left a sharper
statement in its place: 31 InterPLM features separate beta-lactamase from benign
proteins at n=6 under Bonferroni correction, so **the information is in the
representation**. The probe's 21% recovery is therefore not the representation
lacking the class; it is the probe not reaching what is demonstrably there.

That makes one question obvious. §9.1 compared four classifier heads and found the
head does not rescue beta-lactamase -- but it compared them on the RAW embedding.
Nobody has run leave-one-mechanism-out in the SAE feature space, where the
discriminative features were actually located.

🔴 Why this needs a dimension control, not just a comparison
-----------------------------------------------------------
The SAE feature matrix is 234 x 10240. Against 80 positives that is 43.8 dimensions
per sample, where the raw embedding is 5.5. A logistic probe in that regime is
dominated by regularisation and overfitting, so "SAE features recover more" would be
uninterpretable on its own: it could mean the feature space is better, or only that
10240 loosely-constrained dimensions fit the holdout differently.

Three conditions under one protocol, 03b's fold logic (40% negative holdout, threshold
at the 95th percentile of HELD-OUT negatives) at 30 seeds rather than 03b's 5. The
re-implementation was checked against the published table before being trusted: at 5
seeds `recover()` below reproduces all nine published class numbers exactly, which is
what entitles the comparison. See `src/03v_lomo_seed_stability.py`, which also shows
that raising 5 to 30 moves only beta-lactamase, from 21.4% to 15.7%.

    raw            the canonical 1280-d ESM-2 embedding, the §3 baseline
    sae_full       all 10240 SAE features
    sae_matched    SAE features reduced to 1280 dimensions, so dimensionality is
                   equal to raw and only the SPACE differs

Reduction for sae_matched keeps the 1280 features with the highest variance across
the panel, chosen on TRAINING rows only inside each fold, so the holdout class never
influences which features are kept. Doing that selection once on the full panel
would leak the held-out class into feature choice, which is exactly the kind of
mistake this file exists to avoid.

Regularisation is swept (C in 0.01, 0.1, 1, 10) rather than fixed at 1.0, because a
single C cannot be fair to spaces whose dimensionality differs by 8x.

Usage:
    python src/15e_sae_feature_space_lomo.py
"""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "results" / "v2"
FRAC, SPEC, SEEDS = 0.40, 0.95, 30
C_GRID = [0.01, 0.1, 1.0, 10.0]
MATCHED_DIM = 1280


def recover(Xp, Xn, hi, tri, C, seed, matched):
    """One LOMO fold. If matched, pick the top-variance features using TRAINING rows
    only, so the held-out class cannot influence the feature subset."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Xn))
    cut = int(len(Xn) * FRAC)
    nte, ntr = perm[:cut], perm[cut:]

    Xtr_p, Xtr_n = Xp[tri], Xn[ntr]
    if matched and Xp.shape[1] > MATCHED_DIM:
        var = np.vstack([Xtr_p, Xtr_n]).var(0)
        keep = np.argsort(var)[-MATCHED_DIM:]
    else:
        keep = np.arange(Xp.shape[1])

    m = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=5000, C=C))
    m.fit(np.vstack([Xtr_p[:, keep], Xtr_n[:, keep]]),
          np.r_[np.ones(len(tri)), np.zeros(len(ntr))])
    thr = np.quantile(m.predict_proba(Xn[np.ix_(nte, keep)])[:, 1], SPEC)
    return float((m.predict_proba(Xp[np.ix_(hi, keep)])[:, 1] >= thr).mean())


def main():
    man = json.load(open(V2 / "embedding_manifest_v2.json"))
    mech = json.load(open(ROOT / "data/annotations/mechanism_classes_v2.json"))
    cls = {e["fasta_id"]: e["mechanism_class"] for e in mech["proteins"]}
    pcls = np.array([cls[r["acc"]] for r in man["positive_rows"]])
    n_pos = len(pcls)

    raw_p = np.load(V2 / "embeddings_positive_v2.npy")
    raw_n = np.load(V2 / "embeddings_negative_v2.npy")
    F = np.load(V2 / "interplm_features_L18.npy")
    sae_p, sae_n = F[:n_pos], F[n_pos:]
    print(f"raw {raw_p.shape[1]}d, SAE {sae_p.shape[1]}d, "
          f"{n_pos} positives / {len(raw_n)} negatives")

    eligible = set(mech["holdout_eligible_classes"]) | {"virulence_associated_non_toxin"}
    classes = sorted(c for c in set(pcls) if c in eligible and (pcls == c).sum() >= 4)

    conditions = [("raw", raw_p, raw_n, False),
                  ("sae_full", sae_p, sae_n, False),
                  ("sae_matched", sae_p, sae_n, True)]

    out = {}
    for name, Xp, Xn, matched in conditions:
        out[name] = {}
        for C in C_GRID:
            per_class = {}
            for c in classes:
                hi = np.where(pcls == c)[0]
                tri = np.setdiff1d(np.arange(n_pos), hi)
                vals = [recover(Xp, Xn, hi, tri, C, s, matched) for s in range(SEEDS)]
                per_class[c] = float(np.mean(vals))
            out[name][str(C)] = {
                "per_class": per_class,
                "mean": float(np.mean(list(per_class.values()))),
                "beta_lactamase": per_class["beta_lactamase"],
            }
            print(f"  {name:<12} C={C:<6} mean {out[name][str(C)]['mean'] * 100:5.1f}%"
                  f"  beta_lactamase {per_class['beta_lactamase'] * 100:5.1f}%", flush=True)

    print(f"\n{'condition':<14}{'best C':>8}{'mean':>9}{'beta_lact':>11}")
    summary = {}
    for name in out:
        best_C = max(out[name], key=lambda c: out[name][c]["mean"])
        b = out[name][best_C]
        summary[name] = {"best_C": float(best_C), "mean": b["mean"],
                         "beta_lactamase": b["beta_lactamase"],
                         "per_class": b["per_class"]}
        print(f"{name:<14}{best_C:>8}{b['mean'] * 100:>8.1f}%"
              f"{b['beta_lactamase'] * 100:>10.1f}%")

    bl_raw = summary["raw"]["beta_lactamase"]
    bl_sae = summary["sae_matched"]["beta_lactamase"]
    verdict = ("RESCUED: beta-lactamase recovery rises in the dimension-matched SAE space"
               if bl_sae - bl_raw > 0.15 else
               "REFUSED: the feature space does not let the probe reach it either")
    print(f"\nbeta_lactamase raw {bl_raw * 100:.1f}% -> sae_matched {bl_sae * 100:.1f}%")
    print(f"verdict: {verdict}")

    dest = V2 / "sae_feature_space_lomo.json"
    json.dump({"protocol": "03b: 40% negative holdout, 95th pct of held-out negatives",
               "seeds": SEEDS, "C_grid": C_GRID, "matched_dim": MATCHED_DIM,
               "feature_selection": "top-variance, fit on training rows only, per fold",
               "summary": summary, "all": out, "verdict": verdict},
              open(dest, "w"), indent=2)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
