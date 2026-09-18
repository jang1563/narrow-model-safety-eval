#!/usr/bin/env python3
"""
03v_lomo_seed_stability.py - how much of each published LOMO number is the seed draw.

Why this exists
---------------
`03b` fixes SEEDS = [0, 1, 2, 3, 4], and every recovery percentage in
docs/MECHANISM_GENERALIZATION.md §3 is a mean over those five negative-holdout
splits. Five is few. This script re-runs the identical fold logic at 30 seeds and
reports the spread, so a reader can tell which published numbers are the protocol
and which are the draw.

It doubles as an independent check on `15e`, whose `recover()` is a separate
implementation of the same protocol. At five seeds that implementation reproduces
**all nine published class numbers exactly**, which is why §9.7 is entitled to
compare SAE feature spaces against the raw embedding using it.

🔴 What it found, and the one number it changes
----------------------------------------------
Eight of nine classes are stable: six have zero seed variance, CDI moves 2.5 points
and the virulence control 1.3. Beta-lactamase is the exception.

    class                    published (5 seeds)   30 seeds   sd    zero seeds
    beta_lactamase                         21.4%      15.7%   12.5    7 of 30
    contact_dependent_inhibition           35.0%      37.5%   22.5    1 of 30
    every other class                    unchanged within 0.5 pt

The 30-seed 95% CI for beta-lactamase is [11.2, 20.2], which does NOT contain the
published 21.4%. So the headline number for the anomalous class sits at the
optimistic edge of its own sampling distribution, and 7 of 30 negative-holdout
splits recover the class at exactly 0%.

The published figure is left in place rather than rewritten: 5 seeds is what the
preregistered 03b protocol specifies, and it reproduces. What changes is how it
should be read, which is stated wherever it appears. Nothing about the direction of
§9 moves; the true recovery being LOWER than published makes the anomaly sharper,
not weaker.

⚠️ Scope: only the 95% specificity operating point at C=1.0 on the canonical ESM-2
650M mean-pooled embedding. The @99 column and the 14 model arms are not re-run here.

Usage:
    python src/03v_lomo_seed_stability.py
"""

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "results" / "v2"
SEEDS_MAX = 30
C = 1.0
# A class with zero seed variance has a CI of zero width, so an exact-equal published
# value can fall a float epsilon outside it. Without this tolerance t3ss_effector_apparatus
# (sd 0.0, published 0.8000000000000003) is flagged as a discrepancy it is not.
CI_TOL = 1e-6

sys.path.insert(0, str(ROOT / "src"))
# 15e owns the fold logic; importing it keeps one implementation rather than two.
recover = import_module("15e_sae_feature_space_lomo").recover


def main():
    man = json.load(open(V2 / "embedding_manifest_v2.json"))
    mech = json.load(open(ROOT / "data/annotations/mechanism_classes_v2.json"))
    lomo = json.load(open(V2 / "lomo_results.json"))["leave_one_mechanism_out"]

    cls = {e["fasta_id"]: e["mechanism_class"] for e in mech["proteins"]}
    pcls = np.array([cls[r["acc"]] for r in man["positive_rows"]])
    P = np.load(V2 / "embeddings_positive_v2.npy")
    N = np.load(V2 / "embeddings_negative_v2.npy")

    eligible = set(mech["holdout_eligible_classes"]) | {"virulence_associated_non_toxin"}
    classes = sorted(c for c in set(pcls) if c in eligible and (pcls == c).sum() >= 4)

    print(f"{'class':<32}{'pub(5s)':>9}{'5 seed':>8}{'30 seed':>9}"
          f"{'sd':>7}{'zeros':>7}{'ci95':>16}")
    rows = {}
    for c in classes:
        hi = np.where(pcls == c)[0]
        tri = np.setdiff1d(np.arange(len(pcls)), hi)
        v = np.array([recover(P, N, hi, tri, C, s, False) for s in range(SEEDS_MAX)])
        se = v.std(ddof=1) / np.sqrt(SEEDS_MAX)
        lo, hh = v.mean() - 1.96 * se, v.mean() + 1.96 * se
        rows[c] = {"published_5seed": lomo[c]["flagged_95_mean"],
                   "mean_5seed": float(v[:5].mean()), "mean_30seed": float(v.mean()),
                   "sd_30seed": float(v.std(ddof=1)), "zero_seeds": int((v == 0).sum()),
                   "ci95_30seed": [float(lo), float(hh)],
                   "published_inside_ci": bool(lo - CI_TOL <= lomo[c]["flagged_95_mean"]
                                               <= hh + CI_TOL)}
        print(f"{c:<32}{lomo[c]['flagged_95_mean'] * 100:>8.1f}%"
              f"{v[:5].mean() * 100:>7.1f}%{v.mean() * 100:>8.1f}%"
              f"{v.std(ddof=1) * 100:>6.1f}{(v == 0).sum():>7}"
              f"   [{lo * 100:>5.1f},{hh * 100:>5.1f}]", flush=True)

    exact = sum(abs(rows[c]["mean_5seed"] - rows[c]["published_5seed"]) < 1e-9
                for c in classes)
    outside = [str(c) for c in classes if not rows[c]["published_inside_ci"]]
    d5 = np.array([rows[c]["mean_5seed"] for c in classes])
    d30 = np.array([rows[c]["mean_30seed"] for c in classes])

    print(f"\n5-seed reproduces published exactly: {exact}/{len(classes)} classes")
    print(f"mean over classes: 5 seed {d5.mean() * 100:.1f}%, "
          f"30 seed {d30.mean() * 100:.1f}%, delta {(d5.mean() - d30.mean()) * 100:+.1f}")
    print(f"max per-class |5s - 30s|: {np.abs(d5 - d30).max() * 100:.1f} pts")
    print(f"published value outside its own 30-seed CI: {outside or 'none'}")

    dest = V2 / "lomo_seed_stability.json"
    json.dump({"protocol": "03b fold logic, C=1.0, 95% specificity on held-out negatives",
               "seeds_compared": [5, SEEDS_MAX],
               "reimplementation_check": f"{exact}/{len(classes)} classes reproduced exactly",
               "per_class": rows,
               "mean_5seed": float(d5.mean()), "mean_30seed": float(d30.mean()),
               "published_outside_own_ci": outside},
              open(dest, "w"), indent=2)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
