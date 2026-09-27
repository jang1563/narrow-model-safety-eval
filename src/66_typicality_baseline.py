#!/usr/bin/env python3
"""
66_typicality_baseline.py - the label-free baseline the margin result never had.

Why this exists
---------------
`docs/DETECTOR_CRITERIA.md` criterion 14 is "report the boring baselines, including the ones that make
the model look worse", and § 9 of `docs/MECHANISM_GENERALIZATION.md` does that for the *recovery*
figures: amino-acid composition reaches AUROC 0.754, shuffled labels 0.506, lab-strain provenance
0.818. **The margin result in § 10.4 to § 10.6 never got the same treatment.** Margin is reported
against its own parts -- nearest-positive alone, nearest-negative alone, class size -- and against
nothing simpler than itself.

The obvious simpler thing was found by reading prior work rather than by introspection. "Viral
Proteins Reveal Geometry of Protein Language Models" (arXiv 2606.12609, ICML 2026 workshops) reports a
**dominant nativeness axis** in PLM embedding space, aligned with masked reconstruction perplexity,
ordering sequences from well-modelled cellular proteins through viral proteins to shuffled ones. If
embedding geometry is organised by how *typical* a sequence is, then a class sitting close to benign
proteins may be a class sitting close to protein space in general, and margin would be reading a
typicality axis rather than a hazard-specific geometry.

The baseline
------------
    typicality(protein) = cosine to the mean of the 8,259-protein benign pool
    typicality(class)   = mean over the class's members

**It uses no hazard label, no class label and no held-out structure.** One line, no fit. If it
predicted recovery as well as margin does, margin would have nothing of its own.

What this reports
-----------------
Per arm: margin against recovery, typicality against recovery, margin against typicality, and both
partial correlations. The partial is the load-bearing number: margin controlling for typicality says
whether margin carries information typicality does not.

Usage:
    python src/66_typicality_baseline.py
    python src/66_typicality_baseline.py --panel v3 --perms 20000
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "typicality_baseline.json"
POOL_TAG = {"": "esm2_650M", "_esm2_35M": "esm2_35M"}   # arms with pool embeddings


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def rank(x):
    """Tie-averaged ranks, matching src/30's `_spearman`.

    🔴 The first version used argsort(argsort(...)), which breaks ties arbitrarily. Recovery has
    real ties -- three classes sit at 100% on the canonical arm -- so that returned +0.902 for
    margin against src/30's published +0.894. Same quantity, different tie handling, and src/30's
    is the correct one. A baseline that disagrees with the number it is a baseline for, for a
    reason unrelated to the baseline, is worse than no baseline.
    """
    v = np.asarray(x, float)
    o = v.argsort()
    r = np.empty(len(v), float)
    r[o] = np.arange(1, len(v) + 1)
    for u in np.unique(v):
        m = v == u
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def partial(a, b, ctrl):
    """Spearman partial correlation of a and b controlling for ctrl, on ranks."""
    ra, rb, rc = rank(a), rank(b), rank(ctrl)
    A = np.vstack([np.ones_like(rc), rc]).T
    resid = lambda y: y - A @ np.linalg.lstsq(A, y, rcond=None)[0]   # noqa: E731
    return float(np.corrcoef(resid(ra), resid(rb))[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="v3", choices=["v2", "v3"])
    ap.add_argument("--perms", type=int, default=20000)
    a = ap.parse_args()
    RES = ROOT / "results" / a.panel

    cls = {p["fasta_id"]: p["mechanism_class"] for p in
           json.load(open(ROOT / f"data/annotations/mechanism_classes_{a.panel}.json"))["proteins"]}
    pools = sorted(RES.glob("embeddings_pool_large_*.npy"))
    if not pools:
        print(f"XX no pool embeddings in {RES}; the typicality centroid needs one")
        return 2

    rows, rng = {}, np.random.default_rng(0)
    for pool_path in pools:
        tag = pool_path.name[len("embeddings_pool_large_"):-4]
        suf = "" if tag == "esm2_650M" else f"_{tag}"
        need = [RES / f"embeddings_positive_{a.panel}{suf}.npy",
                RES / f"embeddings_negative_{a.panel}{suf}.npy",
                RES / f"embedding_manifest_{a.panel}{suf}.json",
                RES / f"lomo_results{suf}.json"]
        if not all(f.exists() for f in need):
            print(f"  skipping {tag}: missing {[f.name for f in need if not f.exists()]}")
            continue
        P, N = np.load(need[0]), np.load(need[1])
        POOL = np.load(pool_path)
        accs = [r["acc"] for r in json.load(open(need[2]))["positive_rows"]]
        lomo = json.load(open(need[3]))["leave_one_mechanism_out"]

        Pu, Nu, c = unit(P), unit(N), POOL.mean(0)
        cu = c / np.linalg.norm(c)
        pcls = np.array([cls.get(x, "?") for x in accs])

        typ, marg, rec, names = [], [], [], []
        for k in sorted(set(pcls) - {"?"}):
            if k not in lomo or lomo[k].get("flagged_95_mean") is None:
                continue
            idx = np.where(pcls == k)[0]
            other = np.where((pcls != k) & (pcls != "?"))[0]
            typ.append(float(np.mean(Pu[idx] @ cu)))
            marg.append(float(np.mean([(Pu[other] @ Pu[i]).max() - (Nu @ Pu[i]).max()
                                       for i in idx])))
            rec.append(lomo[k]["flagged_95_mean"])
            names.append(k)

        if len(names) < 4:
            continue
        r_m, r_t = spearman(marg, rec), spearman(typ, rec)
        p_m, p_t = partial(marg, rec, typ), partial(typ, rec, marg)
        null_t = [spearman(typ, rng.permutation(rec)) for _ in range(a.perms)]
        rows[tag] = {
            "n_classes": len(names), "classes": names,
            "margin_vs_recovery": r_m, "typicality_vs_recovery": r_t,
            "margin_vs_typicality": spearman(marg, typ),
            "margin_vs_recovery_typicality_controlled": p_m,
            "typicality_vs_recovery_margin_controlled": p_t,
            "typicality_perm_p": float(np.mean(np.array(null_t) <= r_t)),
            "margin_dominates": abs(p_m) > abs(p_t),
            "per_class": [{"class": n, "typicality": t, "margin": mg, "recovery_95": rc}
                          for n, t, mg, rc in zip(names, typ, marg, rec)],
        }
        print(f"{tag}: margin {r_m:+.3f}  typicality {r_t:+.3f}  "
              f"margin|typ {p_m:+.3f}  typ|margin {p_t:+.3f}  "
              f"perm p {rows[tag]['typicality_perm_p']:.4f}")

    res = {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"), "panel": a.panel, "perms": a.perms,
        "baseline": ("typicality(class) = mean cosine of its members to the mean of the "
                     "8,259-protein benign pool. No hazard label, no class label, no fit."),
        "prompted_by": ("arXiv 2606.12609, ICML 2026 workshops, which reports a dominant nativeness "
                        "axis in PLM embedding space aligned with masked reconstruction perplexity"),
        "reading": ("the partial correlations are the load-bearing numbers: margin controlling for "
                    "typicality says whether margin carries anything typicality does not"),
        "arms": rows,
    }
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
