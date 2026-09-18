#!/usr/bin/env python3
"""
03o_plm_fusion_baseline.py - does fusing two protein language models help? No.

Why this was worth running
--------------------------
VF-Fuse (Briefings in Bioinformatics 2025, bbaf481) predicts virulence factors by
fusing ESM-2 and ProtT5 along two paths and reports F1 87.15%, with a majority vote
over fifteen combination rules coming out best. This project already runs ESM-2 and
ProtT5, but only ever as SEPARATE arms. §8's ensemble result covers alignment
combined with embeddings, not one PLM combined with another, so the obvious
question "why not do what VF-Fuse does" had no answer here.

It also looked promising rather than pro forma, because the two models are
genuinely complementary per class: ProtT5 leads ESM-2 by 17.6 points on
pore-forming cytolysin and 14.2 on contact-dependent inhibition, while trailing it
by 25.0 on the virulence control and 10.5 on beta-lactamase.

The result
----------
Concatenating the two z-scored embeddings lands at 72.1% mean recovery against
ESM-2's 72.5%, so fusion costs 0.4 points. Per class it tracks the mean of its two
inputs rather than the better of them: it does not keep ProtT5's advantage on
pore-forming cytolysin, and it does not avoid ProtT5's collapse on beta-lactamase.

This extends §8's finding to a second kind of combination. Two complementary
detectors do not make a better one under a fixed false-positive budget, whether the
second detector is an alignment score or another protein language model.

Protocol is 03b's exactly: leave one mechanism out, train on the rest plus 60% of
negatives, calibrate at the 95th percentile of the held-out negatives, 30 seeds.
Reproduces §9.1's published logistic numbers (72.5% mean, 15.7% beta-lactamase) on
the ESM-2 arm, which is the check that the harness is wired correctly.

Usage:
    python src/03o_plm_fusion_baseline.py
"""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "results" / "v2"
SEEDS, FRAC, SPEC = range(30), 0.40, 0.95


def arm(tag):
    s = f"_{tag}" if tag else ""
    return (np.load(V2 / f"embeddings_positive_v2{s}.npy"),
            np.load(V2 / f"embeddings_negative_v2{s}.npy"))


def zscore(a, b):
    m, s = a.mean(0), a.std(0) + 1e-8
    return (a - m) / s, (b - m) / s


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
            s_ho = m.predict_proba(P[hi])[:, 1]
            s_nt = m.predict_proba(N[nte])[:, 1]
            r.append(float((s_ho >= np.quantile(s_nt, SPEC)).mean()))
        out[C] = float(np.mean(r))
    return out


def main():
    man = json.load(open(V2 / "embedding_manifest_v2.json"))
    mech = json.load(open(ROOT / "data/annotations/mechanism_classes_v2.json"))
    cls = {p["fasta_id"]: p["mechanism_class"] for p in mech["proteins"]}
    pcls = np.array([cls[r["acc"]] for r in man["positive_rows"]])
    classes = sorted(set(mech["holdout_eligible_classes"]) | {"virulence_associated_non_toxin"})

    E_P, E_N = arm("")
    T_P, T_N = arm("prott5_xl")
    Ez_P, Ez_N = zscore(E_P, E_N)
    Tz_P, Tz_N = zscore(T_P, T_N)
    print(f"ESM-2 {E_P.shape[1]}d, ProtT5 {T_P.shape[1]}d, concat {E_P.shape[1] + T_P.shape[1]}d")

    res = {"esm2_650M": lomo(E_P, E_N, pcls, classes),
           "prott5_xl": lomo(T_P, T_N, pcls, classes),
           "concat": lomo(np.hstack([Ez_P, Tz_P]), np.hstack([Ez_N, Tz_N]), pcls, classes)}

    print(f"\n{'class':<32}" + "".join(f"{k:>14}" for k in res))
    print("-" * 74)
    for C in classes:
        print(f"{C:<32}" + "".join(f"{res[k][C] * 100:>13.1f}%" for k in res))
    means = {k: float(np.mean(list(v.values()))) for k, v in res.items()}
    print("-" * 74)
    print(f"{'mean':<32}" + "".join(f"{means[k] * 100:>13.1f}%" for k in res))

    delta = means["concat"] - means["esm2_650M"]
    best = max(means, key=means.get)
    print(f"\nconcat minus ESM-2: {delta * 100:+.1f} points; best arm {best}")
    print("fusion does not beat the better single model" if best != "concat"
          else "fusion wins, which contradicts the recorded result")

    out = {"protocol": "identical to src/03b_leave_one_mechanism_out.py, 30 seeds",
           "reference": "VF-Fuse, Briefings in Bioinformatics 2025 bbaf481",
           "means": means, "concat_minus_esm2": delta, "best": best,
           "classes": res}
    p = V2 / "plm_fusion_baseline.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
