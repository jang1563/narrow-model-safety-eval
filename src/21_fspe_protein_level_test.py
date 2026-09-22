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

🔴 Exclusions, added 2026-09-22
-------------------------------
An entry can carry `fspe_excluded: true` in `data/annotations/functional_sites.json`,
with a reason in `_fspe_exclusion_reason`. The headline test runs on the entries that
remain, and BOTH figures are written to the artifact so the exclusion can never be a
silent one: `n_proteins`/`sign_test_one_sided_p` are the reported values and
`without_exclusions` holds what the full set would have given.

`P01552` (SEB) is the first and only exclusion. Its published offset of 0 is disproven
(two annotated "MHC-II binding interface" positions fall inside its cleaved signal
peptide) and the only admissible offset fails `src/46`'s identity rule, so no
annotation for it meets this project's standard. Independently, SEB was already
excluded from FSI because a superantigen has no discrete catalytic site, and that
objection applies here unchanged.

⚠️ This exclusion makes the headline WEAKER, 13/15 at p = 0.0037 becoming 12/14 at
p = 0.0065. That direction is the point rather than an inconvenience: see
`docs/DETECTOR_CRITERIA.md` criterion 11 on direction-blind corrections.

⚠️ `fspe_results.json` is deliberately NOT regenerated. Every per-protein ratio in it,
including SEB's 0.9556, is the record of what was computed, and recomputing 14
unaffected proteins to drop one row would move them by floating-point drift
(`docs/DATA_CORRECTIONS.md` entry seventeen) and force every published per-protein
figure to be restated for no gain. The exclusion is an analysis decision and is applied
where the analysis happens.

Output: results/fspe_protein_level_test.json
"""

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "fspe_results.json"
SITES = ROOT / "data" / "annotations" / "functional_sites.json"
OUT = ROOT / "results" / "fspe_protein_level_test.json"
N_PERM = 20000


def _tests(ratio):
    n = len(ratio)
    k = int((ratio < 1.0).sum())
    p_sign = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    logr = np.log(np.clip(ratio, 1e-9, None))
    obs = float(logr.mean())
    rng = np.random.default_rng(0)
    null = (logr * rng.choice([-1, 1], size=(N_PERM, n))).mean(axis=1)
    return {"n_proteins": n, "n_ratio_below_1": k, "sign_test_one_sided_p": p_sign,
            "mean_log_ratio": obs, "permutation_p": float((null <= obs).mean())}


def main():
    d = json.load(open(SRC))
    pp = d["per_protein"]
    sites = json.load(open(SITES))
    excluded = {a: sites[a]["functional_sites"].get("_fspe_exclusion_reason", "")
                for a in sites if not a.startswith("_")
                and sites[a]["functional_sites"].get("fspe_excluded")}

    def acc(x):
        return x.get("uniprot_id") or x.get("accession")

    kept = [x for x in pp if acc(x) not in excluded]
    dropped = [acc(x) for x in pp if acc(x) in excluded]
    # An entry flagged for exclusion that is not in the results at all is a silent no-op, so say so.
    flagged_absent = sorted(set(excluded) - {acc(x) for x in pp})

    full = _tests(np.array([x["fspe_ratio"] for x in pp], float))
    res = _tests(np.array([x["fspe_ratio"] for x in kept], float))
    n, k, p_sign, obs, p_perm = (res["n_proteins"], res["n_ratio_below_1"],
                                 res["sign_test_one_sided_p"], res["mean_log_ratio"],
                                 res["permutation_p"])

    pooled = d.get("pooled_meta_analysis", {})
    out = {
        "unit_of_analysis": "protein",
        "n_proteins": n,
        "n_ratio_below_1": k,
        "sign_test_one_sided_p": p_sign,
        "mean_log_ratio": obs,
        "permutation_p": p_perm,
        "n_permutations": N_PERM,
        "excluded": dropped,
        "exclusion_reasons": {a: excluded[a] for a in dropped},
        "excluded_ratios": {acc(x): x["fspe_ratio"] for x in pp if acc(x) in excluded},
        "flagged_but_absent_from_results": flagged_absent,
        "without_exclusions": full,
        "exclusion_weakens_headline": p_sign > full["sign_test_one_sided_p"],
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
    if flagged_absent:
        print(f"⚠️ flagged for exclusion but absent from {SRC.name}: {flagged_absent}")
    for a in dropped:
        print(f"excluded {a} (ratio {out['excluded_ratios'][a]:.4f})")
    print(f"without exclusions : {full['n_ratio_below_1']}/{full['n_proteins']}, "
          f"sign p {full['sign_test_one_sided_p']:.4f}, perm p {full['permutation_p']:.4f}")
    print(f"proteins: {n}, ratio below 1.0: {k}")
    print(f"sign test one-sided p      : {p_sign:.4f}")
    print(f"permutation p (mean log r) : {p_perm:.4f}   observed {obs:+.3f}")
    print(f"residue-pooled, for contrast: p = {pooled.get('mannwhitney_pvalue'):.2e}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
