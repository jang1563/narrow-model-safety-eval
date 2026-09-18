#!/usr/bin/env python3
"""
15d_interplm_feature_overlap.py - the sixth candidate for the beta-lactamase anomaly, refused.

The hypothesis
--------------
Five explanations for beta-lactamase's 21% LOMO recovery have been tested and refused
(classifier head §9.1, corpus/capacity §9.3, AMR-as-a-category §9.4, layer depth §9.5,
training-set composition §5.1). This is the sixth, and the first to look inside the
representation: if the features that distinguish beta-lactamase from benign proteins are
not SHARED with the other hazard classes, then a probe trained on those other classes has
no feature path to reach it, and the recovery failure is explained.

The unconditioned numbers look like a finding. On the full panel at InterPLM layer 18,
beta-lactamase has 279 features significant against benign (Mann-Whitney, Bonferroni over
10240), and 264 of them -- 95% -- appear in no other class's significant set. No feature is
shared by all seven other classes. Class-by-class the uniqueness fraction even tracked
recovery in the right direction.

🔴 REFUSED, because class size drives it
------------------------------------------
beta-lactamase is the largest class (n=14), and more members means more power means more
features clear Bonferroni. Subsampling every class to n=6 over 20 draws:

    class                     sig features   unique %   recovery@95
    beta_lactamase                     31        85%          21%
    clostridial_neurotoxin            472        78%         100%
    t3ss_effector_apparatus            12        50%          80%
    adp_ribosyl_ab_toxin              131        48%         100%
    superantigen_enterotoxin           59        41%         100%
    rip_rrna_glycosidase               64        25%         100%
    pore_forming_cytolysin             50        22%          69%

    unique fraction vs recovery: Spearman rho -0.217, p 0.641, n=7

beta-lactamase's significant-feature count falls from 279 to 31 once n is matched, a
ninefold drop, so most of the original signal was power. And clostridial neurotoxin is the
decisive counterexample: 78% unique, nearly beta-lactamase's 85%, and recovered at 100%.
Feature uniqueness does not predict recovery failure.

What survives is worth stating plainly: the information IS in the representation. 31
features separate beta-lactamase from benign proteins at n=6 with Bonferroni correction,
so the probe's failure is not the representation lacking the class.

contact_dependent_inhibition (n=4) is excluded from the matched comparison, which is why
n=7 rather than 8.

Input this script does not contain
----------------------------------
It reads `results/v2/interplm_features_L18.npy` (234 x 10240, 9.6 MB), which is gitignored
as a regenerable artifact, so a fresh clone does not have it. Run `15c` first; that script
writes it. This is called out because the reverse mistake -- a published result whose input
lived only outside the repository -- is logged in docs/DATA_CORRECTIONS.md for 2026-09-18.

Usage:
    python src/15c_interplm_power_check.py 1 18     # writes the feature matrices
    python src/15d_interplm_feature_overlap.py     # then this
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "results" / "v2"
K = 6           # every class subsampled to this size
DRAWS = 20      # subsamples per class
TARGETS = ["beta_lactamase", "adp_ribosyl_ab_toxin", "clostridial_neurotoxin",
           "rip_rrna_glycosidase", "superantigen_enterotoxin",
           "pore_forming_cytolysin", "t3ss_effector_apparatus",
           "contact_dependent_inhibition"]


def significant_features(F, idx, ben, alpha):
    """Features whose activation differs between these proteins and the benign set,
    Mann-Whitney with Bonferroni correction over the whole dictionary."""
    out = set()
    for j in range(F.shape[1]):
        a, b = F[idx, j], F[ben, j]
        if (a > 0).sum() + (b > 0).sum() < 5:
            continue
        if mannwhitneyu(a, b).pvalue < alpha:
            out.add(j)
    return out


def main():
    F = np.load(V2 / "interplm_features_L18.npy")
    man = json.load(open(V2 / "embedding_manifest_v2.json"))
    mech = json.load(open(ROOT / "data/annotations/mechanism_classes_v2.json"))
    lomo = json.load(open(V2 / "lomo_results.json"))["leave_one_mechanism_out"]

    cls = {e["fasta_id"]: e["mechanism_class"] for e in mech["proteins"]}
    prow = man["positive_rows"]
    classes = np.array([cls[r["acc"]] for r in prow]
                       + ["benign"] * (F.shape[0] - len(prow)))
    ben = classes == "benign"
    alpha = 0.05 / F.shape[1]
    idxs = {c: np.where(classes == c)[0] for c in TARGETS}

    rng = np.random.default_rng(0)
    sub = {c: [] for c in TARGETS}
    for c in TARGETS:
        if len(idxs[c]) < K:
            continue
        for _ in range(DRAWS):
            pick = rng.choice(idxs[c], K, replace=False)
            sub[c].append(significant_features(F, pick, ben, alpha))

    print(f"=== every class subsampled to n={K}, {DRAWS} draws ===")
    print(f"{'class':<32}{'sig feats':>10}{'unique %':>10}{'recovery':>10}")
    rows = {}
    for c in TARGETS:
        if not sub[c]:
            print(f"{c:<32}{'excluded, n<' + str(K):>30}")
            continue
        sizes = [len(s) for s in sub[c]]
        uniq = []
        for d in range(DRAWS):
            others = set.union(*[sub[k][d] for k in sub if k != c and sub[k]])
            uniq.append(len(sub[c][d] - others) / max(len(sub[c][d]), 1))
        rec = lomo[c]["flagged_95_mean"]
        rows[c] = (float(np.mean(sizes)), float(np.mean(uniq)), rec)
        print(f"{c:<32}{np.mean(sizes):>10.0f}{np.mean(uniq) * 100:>9.0f}%"
              f"{rec * 100:>9.0f}%")

    names = list(rows)
    rho, p = spearmanr([rows[c][1] for c in names], [rows[c][2] for c in names])
    print(f"\nunique fraction vs recovery: Spearman rho={rho:+.3f}, "
          f"p={p:.4f}, n={len(names)}")
    verdict = ("REFUSED: feature uniqueness does not predict recovery failure"
               if p > 0.05 else "SURVIVES the size control")
    print(f"verdict: {verdict}")

    out = {"K": K, "draws": DRAWS,
           "per_class": {c: {"sig_mean": rows[c][0], "unique_frac": rows[c][1],
                             "recovery": rows[c][2]} for c in rows},
           "spearman_unique_vs_recovery": {"rho": float(rho), "p": float(p),
                                           "n": len(names)},
           "verdict": verdict,
           "excluded_too_small": [c for c in TARGETS if not sub[c]]}
    dest = V2 / "interplm_feature_overlap.json"
    json.dump(out, open(dest, "w"), indent=2)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
