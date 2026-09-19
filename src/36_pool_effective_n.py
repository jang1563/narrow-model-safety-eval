#!/usr/bin/env python3
"""
36_pool_effective_n.py - how many INDEPENDENT negatives does an 8,259-protein pool contain?

Why this qualifies a claim already made
---------------------------------------
§10.8's calibration arithmetic treats held-out negatives as independent order statistics: with
`h` of them the finest estimable false-positive rate is `1/h`, so 3,303 held out gives a ceiling
of 0.99970 against 118's 0.9915. That arithmetic is only right if the negatives are independent.

🔴 They are not, and the pool makes it obvious by inspection. 8,259 proteins carry **3,550
distinct protein names**, and the most common are core housekeeping enzymes repeated across
species: **GlmU 389 times, CTP synthase 378, GatB 341, RlmN 385 (two spellings), YdiU 167.**
`src/34`'s per-organism cap of 30 stops one organism dominating and does nothing about the same
enzyme arriving from 389 different organisms.

§2.1 established exactly this for the positives, under the heading "Effective n, not n": class
size overstates independence, and single-linkage clustering at normalized Smith-Waterman 0.30
gave beta-lactamase 10 independent sequences out of 14 and clostridial neurotoxin 3 out of 6. The
same correction applies to negatives, and it has not been applied to them.

What is measured, and why two estimates rather than one
------------------------------------------------------
    by_name        proteins sharing a recommended name, lowercased, counted as one. Cheap,
                   complete over all 8,259, and an upper bound on redundancy only in so far as
                   curators name orthologs alike. It misses homologs named differently.
    by_homology    single linkage at normalized Smith-Waterman <= 0.30, the panel's own rule,
                   on a random SAMPLE. All-pairs over 8,259 is 34 million alignments, so the
                   sample estimate is extrapolated and its interval is reported.

⚠️ The homology estimate is a sample, not a census, and the extrapolation assumes the sampled
subset has the same clustering density as the whole. For a pool assembled from proportional
taxonomic slices that is a weaker assumption than it would be for a hand-curated set, but it is
an assumption and the figure is reported as an estimate.

Usage:
    python src/36_pool_effective_n.py --sample 600
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from Bio.Align import PairwiseAligner, substitution_matrices

ROOT = Path(__file__).resolve().parent.parent
SEQ = ROOT / "data" / "sequences"
POOL = SEQ / "_scaled_negative_pool.json"
OUT = ROOT / "results" / "v3" / "pool_effective_n.json"
HOMOLOGY_MAX = 0.30
FRAC = 0.40


def aligner():
    a = PairwiseAligner()
    a.substitution_matrix = substitution_matrices.load("BLOSUM62")
    a.open_gap_score, a.extend_gap_score, a.mode = -11, -1, "local"
    return a


def clean(s):
    return "".join(c for c in s if c in "ACDEFGHIKLMNPQRSTVWY")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=600)
    a = ap.parse_args()
    pool = json.load(open(POOL))["proteins"]
    n = len(pool)

    # ---- by name, complete ----------------------------------------------------------
    names = Counter(x["protein_name"].strip().lower() for x in pool)
    eff_name = len(names)
    print(f"pool {n} proteins")
    print(f"  distinct names: {eff_name}  ->  redundancy factor {n / eff_name:.2f}")
    print("  most repeated names:")
    for nm, c in names.most_common(5):
        print(f"      {c:>4}  {nm[:60]}")

    # ---- by homology, on a sample ---------------------------------------------------
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=min(a.sample, n), replace=False)
    al = aligner()
    seqs = [clean(pool[i]["sequence"]) for i in idx]
    selfs = [al.score(s, s) for s in seqs]
    kept, kept_self = [], []
    for s, ss in zip(seqs, selfs):
        if ss <= 0:
            continue
        sim = max((al.score(s, t) / np.sqrt(ss * tf)
                   for t, tf in zip(kept, kept_self)), default=0.0)
        if sim <= HOMOLOGY_MAX:
            kept.append(s)
            kept_self.append(ss)
    keep_rate = len(kept) / len(seqs)
    # binomial interval on the keep rate, then scaled to the pool
    se = np.sqrt(keep_rate * (1 - keep_rate) / len(seqs))
    lo, hi = max(0.0, keep_rate - 1.96 * se), min(1.0, keep_rate + 1.96 * se)
    eff_hom = keep_rate * n
    print(f"\n  homology sample: {len(seqs)} drawn, {len(kept)} survive greedy "
          f"single-linkage at <= {HOMOLOGY_MAX}")
    print(f"  keep rate {keep_rate:.3f} [{lo:.3f}, {hi:.3f}]  ->  effective n "
          f"{eff_hom:.0f} [{lo * n:.0f}, {hi * n:.0f}]")

    # ---- what it does to the calibration claim --------------------------------------
    print("\ncalibration ceiling, held out at 40%:")
    rows = {}
    for label, eff in (("raw count", n), ("by name", eff_name),
                       ("by homology, estimated", eff_hom)):
        h = max(1, int(eff * FRAC))
        rows[label] = {"effective_n": float(eff), "held_out": h,
                       "ceiling": 1 - 1 / h, "spec_with_10_above": 1 - 10 / h}
        print(f"  {label:<24}effective {eff:>7.0f}  held out {h:>5}  "
              f"ceiling {1 - 1 / h:.5f}  with 10 above {1 - 10 / h:.4f}")
    print(f"  {'panel v3, 296 negatives':<24}effective {'296':>7}  held out   118  "
          f"ceiling {1 - 1 / 118:.5f}  with 10 above {1 - 10 / 118:.4f}")

    verdict = (f"the pool's independent count is closer to {eff_hom:.0f} than to {n}, so the "
               f"ceiling it supports is {rows['by homology, estimated']['ceiling']:.5f} rather "
               f"than {1 - 1 / int(n * FRAC):.5f}. Still an order of magnitude better than the "
               f"296-negative panel's 0.9915, and not the figure the raw count implies")
    print(f"\nverdict: {verdict}")

    json.dump({"pool_n": n, "distinct_names": eff_name,
               "name_redundancy_factor": n / eff_name,
               "most_repeated_names": names.most_common(10),
               "homology_sample_size": len(seqs), "homology_survivors": len(kept),
               "homology_keep_rate": keep_rate,
               "homology_keep_rate_ci95": [lo, hi],
               "effective_n_by_homology": eff_hom,
               "effective_n_ci95": [lo * n, hi * n],
               "calibration": rows,
               "caveat": ("the homology figure is a sample extrapolated to the pool and assumes "
                          "the sampled subset clusters like the whole; the name figure is "
                          "complete but misses homologs that curators named differently"),
               "verdict": verdict}, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
