#!/usr/bin/env python3
"""
03q_layer_depth_sweep.py - is the last layer the right layer?

Why
---
Every embedding in this repository is pooled from the FINAL layer of ESM-2 650M.
Pooling was swept (mean, max, CLS, §9) and scale was swept (8M to 15B, §9), but
depth never was. The protein-LM literature reports mid-to-late layers beating the
final layer on functional classification, sometimes by a wide margin, because the
last layer specializes back toward the masked-token objective rather than toward
function. If that holds here, every recovery figure in this document is read off a
layer that is not the best available, and §9's "not the pooling, not the scale"
list is missing an axis it never checked.

Protocol is 03b's exactly: leave one mechanism out, train on the rest plus 60% of
the negatives, calibrate at the 95th percentile of the held-out negatives, 30
seeds. Only the layer whose hidden state is pooled differs.

⚠️ Platform check first
-----------------------
The canonical final-layer arm was embedded on the cluster; the layer arms are
embedded locally on CPU. Comparing them directly would confound depth with
platform. --verify re-embeds a sample at the final layer locally and reports the
maximum absolute deviation from the canonical rows. If that is not tiny, the
comparison is void and the final layer must be re-embedded locally too.

Usage:
    python src/03q_layer_depth_sweep.py [--layers 6,12,18,24,30] [--verify 6]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "results" / "v2"
SEEDS, FRAC, SPEC = range(30), 0.40, 0.95
MAX_LEN = 1022


def read_fasta(p):
    out, acc, seq = {}, None, []
    for line in open(p):
        if line.startswith(">"):
            if acc:
                out[acc] = "".join(seq)
            acc, seq = line[1:].split()[0], []
        elif acc:
            seq.append(line.strip())
    if acc:
        out[acc] = "".join(seq)
    return out


def verify_platform(n):
    """Re-embed n panel positives locally at the final layer and compare to the
    canonical rows. Depth cannot be compared across platforms without this."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    man = json.load(open(V2 / "embedding_manifest_v2.json"))
    pf = read_fasta(ROOT / "data/sequences/toxins_positive_v2.fasta")
    accs = [r["acc"] for r in man["positive_rows"]][:n]
    P = np.load(V2 / "embeddings_positive_v2.npy")[:n]
    mid = "facebook/esm2_t33_650M_UR50D"
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModel.from_pretrained(mid).eval()
    enc = tok([pf[a][:MAX_LEN] for a in accs], return_tensors="pt", padding=True,
              truncation=True, max_length=MAX_LEN + 2)
    with torch.no_grad():
        h = model(**enc).last_hidden_state
    m = enc["attention_mask"].unsqueeze(-1)
    local = ((h * m).sum(1) / m.sum(1)).float().numpy()
    dev = float(np.abs(local - P).max())
    rel = float(np.abs(local - P).max() / (np.abs(P).max() + 1e-12))
    print(f"platform check on {n} sequences: max abs deviation {dev:.3e}, "
          f"relative {rel:.3e}")
    return dev, rel


def lomo(P, N, pcls, classes):
    out = {}
    for C in classes:
        hi = np.where(pcls == C)[0]
        tri = np.setdiff1d(np.arange(len(P)), hi)
        r = []
        for sd in SEEDS:
            rng = np.random.default_rng(sd)
            p = rng.permutation(len(N))
            h = int(len(N) * FRAC)
            nte, ntr = p[:h], p[h:]
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
            m.fit(np.vstack([P[tri], N[ntr]]), np.r_[np.ones(len(tri)), np.zeros(len(ntr))])
            r.append(float((m.predict_proba(P[hi])[:, 1]
                            >= np.quantile(m.predict_proba(N[nte])[:, 1], SPEC)).mean()))
        out[C] = float(np.mean(r))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="6,12,18,24,30")
    ap.add_argument("--verify", type=int, default=6)
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]

    dev, rel = verify_platform(a.verify)
    if rel > 1e-3:
        print("\n\U0001F534 Platform deviation is not negligible. Depth and platform are "
              "confounded;\n   re-embed the final layer locally before comparing.")

    man = json.load(open(V2 / "embedding_manifest_v2.json"))
    mech = json.load(open(ROOT / "data/annotations/mechanism_classes_v2.json"))
    cls = {p["fasta_id"]: p["mechanism_class"] for p in mech["proteins"]}
    pcls = np.array([cls[r["acc"]] for r in man["positive_rows"]])
    classes = sorted(set(mech["holdout_eligible_classes"]) | {"virulence_associated_non_toxin"})

    arms = {"final (33)": (np.load(V2 / "embeddings_positive_v2.npy"),
                           np.load(V2 / "embeddings_negative_v2.npy"))}
    for ly in layers:
        pp = V2 / f"embeddings_positive_v2_esm2_650M_L{ly}.npy"
        if pp.exists():
            arms[f"L{ly}"] = (np.load(pp),
                              np.load(V2 / f"embeddings_negative_v2_esm2_650M_L{ly}.npy"))
    print(f"\narms: {list(arms)}")

    res = {k: lomo(P, N, pcls, classes) for k, (P, N) in arms.items()}
    order = list(res)
    print(f"\n{'class':<32}" + "".join(f"{k:>12}" for k in order))
    print("-" * (32 + 12 * len(order)))
    for C in classes:
        print(f"{C:<32}" + "".join(f"{res[k][C] * 100:>11.1f}%" for k in order))
    means = {k: float(np.mean(list(v.values()))) for k, v in res.items()}
    print("-" * (32 + 12 * len(order)))
    print(f"{'mean':<32}" + "".join(f"{means[k] * 100:>11.1f}%" for k in order))

    base = means["final (33)"]
    best = max(means, key=means.get)
    print(f"\nfinal layer {base:.1%}; best arm {best} at {means[best]:.1%} "
          f"({(means[best] - base) * 100:+.1f} points)")
    if best == "final (33)":
        print("the last layer is already the best of those tested")

    out = {"protocol": "identical to src/03b_leave_one_mechanism_out.py, 30 seeds",
           "platform_check": {"max_abs_deviation": dev, "relative": rel,
                              "valid": bool(rel <= 1e-3)},
           "means": means, "best": best, "best_minus_final": means[best] - base,
           "classes": res}
    p = V2 / "layer_depth_sweep.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
