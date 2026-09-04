#!/usr/bin/env python3
"""
03c_ablation_baselines.py

Three cheap controls that each close an obvious challenge to the ESM probe.

  composition        20-dim amino-acid frequency. If this matches the ESM
                     probe, the learned representation is not earning its keep
                     and the task is solvable from bulk composition alone.
  length             log sequence length, 1 feature. Toxins and benign enzymes
                     could differ in size; this bounds how much of the signal
                     that explains.
  composition+length 21 features, the strongest trivial baseline.
  shuffled_labels    ESM embeddings with permuted labels. Must return ~0.5,
                     otherwise the cross-validation itself is leaking.

Run after 02b. Uses cached embeddings; takes seconds.
"""

import json
import sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "results" / "v2"
AA = "ACDEFGHIKLMNPQRSTVWY"


def read_fasta(p):
    out, acc, seq = {}, None, []
    for line in open(p):
        if line.startswith(">"):
            if acc:
                out[acc] = "".join(seq)
            acc = line[1:].split()[0]
            seq = []
        elif acc:
            seq.append(line.strip())
    if acc:
        out[acc] = "".join(seq)
    return out


def cv(X, y, seed=0):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    s = cross_val_score(
        pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed), scoring="roc_auc"
    )
    return float(s.mean()), float(s.std())


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else ""
    suf = f"_{tag}" if tag else ""
    P = np.load(V2 / f"embeddings_positive_v2{suf}.npy")
    N = np.load(V2 / f"embeddings_negative_v2{suf}.npy")
    man = json.load(open(V2 / f"embedding_manifest_v2{suf}.json"))
    pf = read_fasta(ROOT / "data/sequences/toxins_positive_v2.fasta")
    nf = read_fasta(ROOT / "data/sequences/benign_negatives_v2.fasta")

    seqs = [pf[r["acc"]] for r in man["positive_rows"]] + [
        nf[r["acc"]] for r in man["negative_rows"]
    ]
    y = np.r_[np.ones(len(P)), np.zeros(len(N))]
    X_esm = np.vstack([P, N])
    comp = np.array([[s.count(a) / max(len(s), 1) for a in AA] for s in seqs])
    leng = np.array([[np.log10(len(s))] for s in seqs])

    print(f"model={man['model']}  n={len(y)}  ({int(y.sum())} pos / {int((1 - y).sum())} neg)\n")
    rows = [
        ("ESM embedding", X_esm.shape[1], cv(X_esm, y)),
        ("composition (AA freq)", comp.shape[1], cv(comp, y)),
        ("length only", 1, cv(leng, y)),
        ("composition + length", 21, cv(np.hstack([comp, leng]), y)),
    ]
    print(f"{'features':<26}{'d':>6}{'AUROC':>10}{'sd':>8}")
    print("-" * 50)
    for nme, d, (m, s) in rows:
        print(f"{nme:<26}{d:>6}{m:>10.3f}{s:>8.3f}")

    shuf = [cv(X_esm, np.random.default_rng(k).permutation(y), seed=k)[0] for k in range(5)]
    print(
        f"\n{'shuffled labels (ESM)':<26}{X_esm.shape[1]:>6}{np.mean(shuf):>10.3f}"
        f"{np.std(shuf):>8.3f}   <- must be near 0.5"
    )

    out = {
        "model": man["model"],
        "tag": tag or None,
        "results": {n: {"d": d, "auroc": m, "sd": s} for n, d, (m, s) in rows},
        "shuffled_labels_auroc_mean": float(np.mean(shuf)),
    }
    json.dump(out, open(V2 / f"ablation_baselines{suf}.json", "w"), indent=2)
    print(f"\nwrote {V2 / f'ablation_baselines{suf}.json'}")


if __name__ == "__main__":
    main()
