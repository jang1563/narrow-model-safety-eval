#!/usr/bin/env python3
"""
35_negative_scaling_curve.py - does a comprehensive benign reference set make the screen WORSE
                               on the classes that live near benign?

The prediction being tested, made before the data existed
--------------------------------------------------------
§10.7.1 found the two unreachable classes sit inside a **dense** benign region rather than beside
a few specific proteins: removing the nearest 5, 10, 20, 40 and 80 training negatives helped
monotonically (+5.2 to +16.6 for the phage class, +6.0 to +20.7 for beta-lactamase), while the
recovered comparison class peaked at 20 and fell back to an interval containing zero.

If density is the mechanism, the converse must hold. **Adding** benign proteins should push those
classes down and leave the recovered classes alone. That is a direction, not a hedge, and it is
the operationally interesting case: a screening operator's instinct is that a more comprehensive
benign reference set is strictly better.

§10.8 gives the second reason to do it. The threshold is a quantile of held-out negatives, and 40%
of 296 is 118, so **every specificity above 0.9915 was extrapolation**. `src/34` harvested a pool
of 8,259 reviewed Swiss-Prot proteins under the hazard exclusions, giving 3,303 held out and a
ceiling of 0.99970. With ten negatives behind the threshold that supports a calibrated **0.997**
operating point, against 0.9915 with a single order statistic before.

PREREGISTERED, written before the run
-------------------------------------
    P1  both failing classes' recovery declines monotonically in the size of the negative set.
    P2  the recovered comparison class declines by less than either failure.

    SUPPORTED     both hold. Density is the mechanism and a comprehensive benign set is
                  actively harmful for classes that sit in it.
    REFUTED       the failures do not decline, or they decline no faster than a recovered class.
                  §10.7.1's density reading would then be wrong in the direction that matters.

⚠️ Composition is NOT held constant against the existing panel, and the difference is documented
rather than smoothed over. The 296-negative panel is taxon-matched to its positives by `02d` and
`27`; this pool is not, because a comprehensive reference set spans everything. The pool is also
**87% bacterial** (7,226 Bacteria, 424 Archaea, 339 Viruses, 270 Eukaryota) rather than the
proportional split `34` asked for: the per-organism cap of 30 bit hard on eukaryotic Swiss-Prot,
which is dominated by a handful of model organisms. So the pool happens to resemble the panel's
producer composition (74 of 80 bacterial) more closely than Swiss-Prot's. Reported as what
happened, not as what was designed.

⚠️ Within the pool, composition IS constant across sizes: every size subsamples from the same
single pool, which is `03e`'s discipline of varying size without varying the boundary.

Usage:
    python src/35_negative_scaling_curve.py --embed        # embed the pool first, hours
    python src/35_negative_scaling_curve.py                # then the curve
"""

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
V3 = ROOT / "results" / "v3"
SEQ = ROOT / "data" / "sequences"
POOL_NPY = V3 / "embeddings_pool_large.npy"
POOL_FASTA = SEQ / "benign_pool_large.fasta"
FRAC, SPEC, SEEDS = 0.40, 0.95, 30
SIZES = [296, 1000, 3000, 8259]
FAIL_AT = 0.25
CONTROL_CLASS = "virulence_associated_non_toxin"
MODEL = "facebook/esm2_t33_650M_UR50D"


def embed_pool(batch_size=8):
    """Embed the pool with the canonical arm, because that is where every published number
    lives and because the smaller arms have beta-lactamase near the floor already, so a
    DECLINE could not be measured in them."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    sys.path.insert(0, str(ROOT / "src"))
    m02 = import_module("02b_esm2_embed_v2")
    recs = m02.read_fasta(POOL_FASTA)
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if getattr(torch.backends, "mps", None)
           and torch.backends.mps.is_available() else "cpu")
    print(f"embedding {len(recs)} pool proteins with {MODEL} on {dev}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(dev).eval()
    X = m02.embed(recs, model, tok, dev, batch_size)
    np.save(POOL_NPY, X)
    json.dump({"model": MODEL, "n": int(X.shape[0]), "dim": int(X.shape[1]),
               "rows": [r[0] for r in recs]},
              open(V3 / "embedding_manifest_pool_large.json", "w"), indent=2)
    print(f"wrote {POOL_NPY} {X.shape}")


def recovery(P, N, hi, tri, seeds=SEEDS):
    out = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(N))
        cut = int(len(N) * FRAC)
        nte, ntr = perm[:cut], perm[cut:]
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
        m.fit(np.vstack([P[tri], N[ntr]]), np.r_[np.ones(len(tri)), np.zeros(len(ntr))])
        thr = np.quantile(m.predict_proba(N[nte])[:, 1], SPEC)
        out.append(float((m.predict_proba(P[hi])[:, 1] >= thr).mean()))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", action="store_true")
    a = ap.parse_args()
    if a.embed:
        embed_pool()
        return
    if not POOL_NPY.exists():
        raise SystemExit(f"{POOL_NPY} absent; run with --embed first")

    man = json.load(open(V3 / "embedding_manifest_v3.json"))
    mech = json.load(open(ROOT / "data/annotations/mechanism_classes_v3.json"))
    lomo = json.load(open(V3 / "lomo_results.json"))["leave_one_mechanism_out"]
    P = np.load(V3 / "embeddings_positive_v3.npy")
    NP_ = np.load(POOL_NPY)
    cls = {e["fasta_id"]: e["mechanism_class"] for e in mech["proteins"]}
    pcls = np.array([cls[r["acc"]] for r in man["positive_rows"]])

    failures = sorted((c for c in lomo
                       if lomo[c]["flagged_95_mean"] < FAIL_AT and c != CONTROL_CLASS),
                      key=lambda c: lomo[c]["flagged_95_mean"])
    below = {c: lomo[c]["flagged_95_mean"] for c in lomo
             if c not in failures and c != CONTROL_CLASS
             and lomo[c]["flagged_95_mean"] < 0.999}
    comparison = max(below, key=below.get)
    targets = failures + [comparison]
    print(f"pool {NP_.shape}, positives {P.shape}")
    print(f"failures {failures}, comparison {comparison}\n")

    rng = np.random.default_rng(0)
    order = rng.permutation(len(NP_))
    out = {c: {} for c in targets}
    hdr = f"{'class':<32}" + "".join(f"{f'n={s}':>12}" for s in SIZES)
    print(hdr)
    print("-" * len(hdr))
    for c in targets:
        hi = np.where(pcls == c)[0]
        tri = np.setdiff1d(np.arange(len(P)), hi)
        for s in SIZES:
            if s > len(NP_):
                continue
            v = recovery(P, NP_[order[:s]], hi, tri)
            held = int(s * FRAC)
            out[c][str(s)] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                              "held_out": held, "ceiling": 1 - 1 / held,
                              "spec_with_10_above": 1 - 10 / held}
        print(f"{c:<32}" + "".join(f"{out[c][str(s)]['mean'] * 100:>11.1f}%"
                                   for s in SIZES if str(s) in out[c]))

    sizes = [s for s in SIZES if str(s) in out[targets[0]]]
    drops = {c: out[c][str(sizes[0])]["mean"] - out[c][str(sizes[-1])]["mean"]
             for c in targets}
    mono = {c: all(out[c][str(b)]["mean"] <= out[c][str(a_)]["mean"] + 0.005
                   for a_, b in zip(sizes, sizes[1:])) for c in targets}
    print(f"\n{'class':<32}{'drop':>10}{'monotone':>10}")
    for c in targets:
        print(f"{c:<32}{drops[c] * 100:>+9.1f}{str(mono[c]):>10}")
    print(f"\ncalibration: n={sizes[-1]} holds out {int(sizes[-1] * FRAC)}, ceiling "
          f"{1 - 1 / int(sizes[-1] * FRAC):.5f}; a threshold with ten negatives above it is "
          f"specificity {1 - 10 / int(sizes[-1] * FRAC):.4f} against 0.9915 at n=296")

    p1 = all(mono[c] and drops[c] > 0 for c in failures)
    p2 = drops[comparison] < min(drops[c] for c in failures)
    if p1 and p2:
        verdict = ("SUPPORTED: both failing classes decline monotonically as the benign set "
                   "grows and the recovered class declines less, so density is the mechanism "
                   "and a comprehensive benign reference set is actively harmful for classes "
                   "that sit inside it")
    elif p1:
        verdict = ("PARTIAL: the failures decline monotonically but the comparison class "
                   f"declines by {drops[comparison] * 100:+.1f} points, not less than they do")
    else:
        verdict = ("REFUTED: the failing classes do not decline monotonically with the size of "
                   "the benign set, so §10.7.1's density reading does not predict the converse")
    print(f"\nverdict: {verdict}")

    dest = V3 / "negative_scaling_curve.json"
    json.dump({"model": MODEL, "pool_n": int(NP_.shape[0]), "sizes": sizes, "seeds": SEEDS,
               "failures": failures, "comparison": comparison,
               "pool_composition_note": ("87% bacterial: the per-organism cap bit hard on "
                                         "eukaryotic Swiss-Prot, so the pool resembles the "
                                         "panel's producer mix rather than Swiss-Prot's"),
               "curves": out, "drops_pts": {c: drops[c] * 100 for c in drops},
               "monotone": mono, "verdict": verdict}, open(dest, "w"), indent=2)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
