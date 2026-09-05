#!/usr/bin/env python3
"""
21_fspe_protein_level_test.py - the FSPE result at the correct unit of analysis.

Why this exists
---------------
`04_esm2_masked_prediction.py` reports a "pooled meta-analysis" that runs a
Mann-Whitney over 74 functional against 300 background residues, giving
p = 2.6e-08. That test treats every residue as an independent observation, but
residues within one protein are not independent: they share a sequence, a fold and
a model forward pass. Pooling them is pseudoreplication, and it is also not a
meta-analysis, which would combine per-protein effect sizes rather than raw
residues.

The independent unit here is the protein, n = 15. This script runs two tests at
that unit, neither of which assumes normality:

  sign test    exact binomial on how many proteins have an FSPE ratio below 1.0
  permutation  sign-flip null on the mean log ratio, 20,000 draws

The conclusion is unchanged; the confidence attached to it is not. Reporting
10^-8 for what the data support at roughly 10^-3 overstates the result by five
orders of magnitude, which is why the public documents now lead with these
numbers and keep the pooled figure as an explicitly labelled descriptive.

Output: results/fspe_protein_level_test.json
"""

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "fspe_results.json"
OUT = ROOT / "results" / "fspe_protein_level_test.json"
N_PERM = 20000


def main():
    d = json.load(open(SRC))
    pp = d["per_protein"]
    ratio = np.array([x["fspe_ratio"] for x in pp], float)
    n = len(ratio)
    k = int((ratio < 1.0).sum())

    # exact one-sided sign test: are more proteins below 1.0 than chance allows
    p_sign = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n

    # sign-flip permutation on the mean log ratio
    logr = np.log(np.clip(ratio, 1e-9, None))
    obs = float(logr.mean())
    rng = np.random.default_rng(0)
    null = (logr * rng.choice([-1, 1], size=(N_PERM, n))).mean(axis=1)
    p_perm = float((null <= obs).mean())

    pooled = d.get("pooled_meta_analysis", {})
    out = {
        "unit_of_analysis": "protein",
        "n_proteins": n,
        "n_ratio_below_1": k,
        "sign_test_one_sided_p": p_sign,
        "mean_log_ratio": obs,
        "permutation_p": p_perm,
        "n_permutations": N_PERM,
        "residue_pooled_for_comparison": {
            "p_value": pooled.get("mannwhitney_pvalue"),
            "n_functional": pooled.get("n_functional"),
            "n_nonfunctional": pooled.get("n_nonfunctional"),
            "caveat": "treats residues within a protein as independent; "
                      "pseudoreplicated, retain only as a descriptive statistic",
        },
        "reading": "the direction holds at the protein level; the residue-pooled "
                   "p-value overstates significance by about five orders of magnitude",
    }
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"proteins: {n}, ratio below 1.0: {k}")
    print(f"sign test one-sided p      : {p_sign:.4f}")
    print(f"permutation p (mean log r) : {p_perm:.4f}   observed {obs:+.3f}")
    print(f"residue-pooled, for contrast: p = {pooled.get('mannwhitney_pvalue'):.2e}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
