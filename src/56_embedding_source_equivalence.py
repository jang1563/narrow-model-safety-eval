#!/usr/bin/env python3
"""
56_embedding_source_equivalence.py - are two embedding runs of the SAME panel and model
                                     interchangeable, or only nearly so?

Why this exists
---------------
The v2 ESM-C arrays were built on 2026-09-05 under esm 3.4.0 / torch 2.11.0, where
`LogitsConfig(return_mean_embedding=True)` exists and the SDK returns the pooled vector. The
environment available now is esm 3.2.1 / torch 2.5.1, where that kwarg does not exist and
`src/02e` pools the per-residue stack itself. Both are "the mean embedding" and they are not
guaranteed to be the same number: the SDK's pooling may or may not include the BOS/EOS
positions, and the two torch builds differ in kernel selection.

That matters here for one specific reason. The v3 ESM-C arm exists in order to be compared
against the v2 ESM-C arm — it is the only arm that recovers beta-lactamase, and the question is
whether it also recovers v3's second unreachable class. If the two panels are embedded through
different pooling, a difference between them is not attributable to the panel.

So this script re-embeds the SAME panel with the SAME model under the new path and compares it
to the stored arrays, row by row. It reports the comparison and does not decide anything: the
caller reads the verdict.

What "equivalent" means here
----------------------------
Reported per label set:
  - row alignment, checked from the manifests' accession lists rather than assumed
  - max absolute difference, and the same relative to the arrays' own scale
  - minimum per-row cosine similarity
  - whether a probe would see the same thing: the correlation of pairwise distances

Usage:
    python src/56_embedding_source_equivalence.py --panel v2 --a esmc_600M --b esmc_600M_mp
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load(res, pv, tag):
    suf = f"_{tag}" if tag else ""
    P = np.load(res / f"embeddings_positive_{pv}{suf}.npy")
    N = np.load(res / f"embeddings_negative_{pv}{suf}.npy")
    man = json.load(open(res / f"embedding_manifest_{pv}{suf}.json"))
    return P, N, man


# 🔴 A single tolerance was the wrong instrument, 2026-09-24. The first run of this script
# returned EQUIVALENT=False on a pair whose minimum row cosine is 0.99988 and whose pairwise
# distance correlation is 0.99996: numerically distinguishable, geometrically the same object.
# A probe fitted on either would see the same neighbourhood structure, and reporting only the
# binary would have either blocked a valid comparison or, if the tolerance had been set loosely,
# hidden a real difference. Both verdicts are reported now, with their thresholds named.
COSINE_FLOOR = 0.999      # a row that rotates less than this is the same direction
PDIST_FLOOR = 0.999       # distance structure a probe would see


def compare(A, B, name, accs=None):
    assert A.shape == B.shape, f"{name}: shape {A.shape} vs {B.shape}"
    d = np.abs(A - B)
    num = (A * B).sum(1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
    cos = num / np.where(den == 0, 1, den)
    # what a distance-based probe would actually see
    def pdist(X):
        g = X @ X.T
        n = np.diag(g)
        return np.sqrt(np.maximum(n[:, None] + n[None, :] - 2 * g, 0))
    pa, pb = pdist(A), pdist(B)
    iu = np.triu_indices(len(A), 1)
    rho = float(np.corrcoef(pa[iu], pb[iu])[0, 1]) if len(A) > 2 else float("nan")
    worst_row = int(np.argmax(d.max(1)))
    return {"rows": int(A.shape[0]), "dim": int(A.shape[1]),
            "worst_row": worst_row,
            "worst_row_acc": (accs[worst_row] if accs else None),
            "worst_row_cosine": float(cos[worst_row]),
            "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
            "scale": float(np.abs(A).mean()),
            "max_abs_diff_over_scale": float(d.max() / max(np.abs(A).mean(), 1e-12)),
            "min_row_cosine": float(cos.min()), "mean_row_cosine": float(cos.mean()),
            "pairwise_distance_corr": rho}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="v2", choices=["v2", "v3"])
    ap.add_argument("--a", required=True, help="reference tag, the stored arrays")
    ap.add_argument("--b", required=True, help="new tag, the run being checked")
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="max absolute difference treated as the same computation")
    a = ap.parse_args()

    res = ROOT / "results" / a.panel
    PA, NA, MA = load(res, a.panel, a.a)
    PB, NB, MB = load(res, a.panel, a.b)

    # row alignment is checked, not assumed: a reordered FASTA would make every number below
    # meaningless while looking fine
    for key in ("positive_rows", "negative_rows"):
        ax = [r["acc"] for r in MA[key]]
        bx = [r["acc"] for r in MB[key]]
        assert ax == bx, f"{key}: accession order differs between the two runs"

    out = {"panel": a.panel, "reference": a.a, "candidate": a.b,
           "reference_env": {k: MA.get(k) for k in ("torch", "esm_version",
                                                    "mean_embedding_source", "built")},
           "candidate_env": {k: MB.get(k) for k in ("torch", "esm_version",
                                                    "mean_embedding_source", "built")},
           "positives": compare(PA, PB, "positives",
                                [r["acc"] for r in MA["positive_rows"]]),
           "negatives": compare(NA, NB, "negatives",
                                [r["acc"] for r in MA["negative_rows"]])}
    worst = max(out["positives"]["max_abs_diff"], out["negatives"]["max_abs_diff"])
    out["tolerance"] = a.tol
    out["cosine_floor"] = COSINE_FLOOR
    out["pdist_floor"] = PDIST_FLOOR
    out["numerically_identical"] = bool(worst < a.tol)
    out["geometrically_equivalent"] = bool(
        min(out[k]["min_row_cosine"] for k in ("positives", "negatives")) >= COSINE_FLOOR
        and min(out[k]["pairwise_distance_corr"] for k in ("positives", "negatives"))
        >= PDIST_FLOOR)
    # kept for the artifacts written before the two-verdict change
    out["equivalent"] = out["numerically_identical"]
    if out["numerically_identical"]:
        out["verdict"] = ("the same computation to within tolerance: arrays from either may be "
                          "compared across panels without further qualification")
    elif out["geometrically_equivalent"]:
        out["verdict"] = (
            "NOT numerically identical but geometrically equivalent: every row keeps its "
            f"direction to better than {COSINE_FLOOR} cosine and the pairwise distance structure "
            f"a probe sees correlates above {PDIST_FLOOR}. A recovery number computed on one may "
            "be compared with one computed on the other, and the difference must be stated, "
            "because a bit-level reproduction claim is not available")
    else:
        out["verdict"] = ("the two runs DIFFER in geometry, not only in arithmetic; a "
                          "cross-panel comparison must use arrays from one source only")

    dest = res / f"embedding_source_equivalence_{a.a}_vs_{a.b}.json"
    json.dump(out, open(dest, "w"), indent=2)
    for k in ("positives", "negatives"):
        c = out[k]
        print(f"{k:10} rows={c['rows']:4} dim={c['dim']:5}  max|d|={c['max_abs_diff']:.3e}  "
              f"min cos={c['min_row_cosine']:.6f}  pdist r={c['pairwise_distance_corr']:.6f}")
    print(f"\nreference {out['reference_env']}")
    print(f"candidate {out['candidate_env']}")
    print(f"\nnumerically identical:   {out['numerically_identical']}  (tol {a.tol:g})")
    print(f"geometrically equivalent: {out['geometrically_equivalent']}  "
          f"(cos >= {COSINE_FLOOR}, pdist r >= {PDIST_FLOOR})")
    print(f"\n{out['verdict']}")
    print(f"wrote {dest}")
    return 0 if (out["numerically_identical"] or out["geometrically_equivalent"]) else 2


if __name__ == "__main__":
    sys.exit(main())
