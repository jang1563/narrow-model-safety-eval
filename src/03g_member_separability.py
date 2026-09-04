#!/usr/bin/env python3
"""
03g_member_separability.py - what separates an always-caught protein from a never-caught one?

Motivation
----------
`03f` found that class-level recovery is an average over per-protein behaviour that
is essentially binary: in T3SS every member is caught on all 30 seeds or on none,
same for virulence-non-toxin, and beta-lactamase has no reliably caught member at
all. If a cheap property predicts which side a protein lands on, that is a usable
triage signal. If nothing predicts it, that bounds what this approach can promise,
which is worth knowing just as much.

Candidate predictors, all computable from what is already built:
    length              sequence length
    exported            UniProt localization in the exported set
    signal_peptide      UniProt Signal feature present
    nn_pos              max cosine similarity to a training positive, i.e. a positive
                        OUTSIDE the protein's own mechanism class. This is "how close
                        is it to the toxins the probe did see"
    nn_neg              max cosine similarity to any negative
    margin              nn_pos - nn_neg. The natural candidate: does the protein sit
                        nearer the remaining toxins or nearer the benigns
    kmer_pos            max 5-mer Jaccard to a positive outside its own class. The
                        control that decides what an embedding-space result means: if
                        margin predicts and kmer_pos does not, the proximity being
                        exploited is representational rather than homological, which
                        is a different and stronger claim about the model

Two analyses, because each has a different failure mode:
    pooled     all extreme-behaviour members from every class together. More power,
               but class identity is a confound: if one whole class is always caught,
               a feature that merely tracks that class will look predictive.
    within     computed separately inside the classes that contain BOTH always- and
               never-caught members. Confound-free but very small n, so it is a
               direction check rather than a test.

Single-feature AUROC is reported with a permutation p-value, since n is small and a
parametric test would overstate confidence.

Run after 02b, 02c and 03f.

Usage:
    python src/03g_member_separability.py [--tag esm2_3B]
"""

import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "results" / "v2"
EXPORTED = {"secreted", "cell_surface", "periplasm", "membrane"}
HI, LO = 0.9, 0.1  # always-caught / never-caught thresholds on the flag rate


def auroc(x, y):
    """Rank-based AUROC of feature x against binary label y, ties averaged."""
    x = np.asarray(x, float)
    y = np.asarray(y, int)
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float)
    r[order] = np.arange(len(x), dtype=float) + 1
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def perm_p(x, y, a, n=20000, seed=0):
    if np.isnan(a):
        return float("nan")
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    null = np.array([auroc(x, rng.permutation(y)) for _ in range(n // 100)])
    return float((np.abs(null - 0.5) >= abs(a - 0.5) - 1e-12).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    tag = ap.parse_args().tag
    suf = f"_{tag}" if tag else ""

    P = np.load(V2 / f"embeddings_positive_v2{suf}.npy")
    N = np.load(V2 / f"embeddings_negative_v2{suf}.npy")
    man = json.load(open(V2 / f"embedding_manifest_v2{suf}.json"))
    mech = json.load(open(ROOT / "data/annotations/mechanism_classes_v2.json"))
    ann = json.load(open(ROOT / "data/annotations/localization_v2.json"))["proteins"]
    cov = json.load(open(V2 / f"coverage_strictness{suf}.json"))
    panel = json.load(open(ROOT / "data/sequences/panel_v2_manifest.json"))

    pos_acc = [r["acc"] for r in man["positive_rows"]]
    cls_of = {p["fasta_id"]: p["mechanism_class"] for p in mech["proteins"]}
    pos_cls = np.array([cls_of.get(x, "UNMAPPED") for x in pos_acc])
    len_of = {e["acc"]: e["len"] for e in panel["positives"]}

    Pn = P / np.linalg.norm(P, axis=1, keepdims=True)
    Nn = N / np.linalg.norm(N, axis=1, keepdims=True)
    simPP = Pn @ Pn.T
    simPN = Pn @ Nn.T

    # 5-mer Jaccard between positives, same order as the embedding rows
    seqs, cur, hdr_i = {}, [], None
    for line in open(ROOT / "data/sequences/toxins_positive_v2.fasta"):
        if line.startswith(">"):
            if hdr_i is not None:
                seqs[hdr_i] = "".join(cur)
            hdr_i, cur = line[1:].split()[0], []
        else:
            cur.append(line.strip())
    if hdr_i is not None:
        seqs[hdr_i] = "".join(cur)
    kset = [set(seqs[a][i : i + 5] for i in range(max(0, len(seqs[a]) - 4))) for a in pos_acc]
    K = np.zeros((len(P), len(P)))
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            u = len(kset[i] | kset[j])
            K[i, j] = K[j, i] = (len(kset[i] & kset[j]) / u) if u else 0.0

    rows = []
    for C, rec in cov["classes"].items():
        rates = rec.get("member_flag_rate_at_ref")
        if not rates:
            continue
        idx = np.where(pos_cls == C)[0]
        assert len(idx) == len(rates), f"member order mismatch for {C}"
        for i, rate in zip(idx, rates):
            if LO < rate < HI:
                continue  # keep only the two extremes
            a = pos_acc[i]
            other = np.setdiff1d(np.arange(len(P)), idx)  # positives outside this class
            rows.append(
                {
                    "class": C,
                    "caught": int(rate >= HI),
                    "length": len_of.get(a, np.nan),
                    "exported": int(ann[a]["localization"] in EXPORTED),
                    "signal_peptide": int(ann[a]["has_signal_peptide"]),
                    "nn_pos": float(simPP[i, other].max()),
                    "nn_neg": float(simPN[i].max()),
                    "kmer_pos": float(K[i, other].max()),
                }
            )
    for r in rows:
        r["margin"] = r["nn_pos"] - r["nn_neg"]

    feats = ["margin", "nn_pos", "kmer_pos", "nn_neg", "length", "exported", "signal_peptide"]
    y = np.array([r["caught"] for r in rows])
    print(
        f"model={man['model']}  extreme-behaviour members: "
        f"{int(y.sum())} always caught, {int((1 - y).sum())} never caught\n"
    )

    print("POOLED across classes  (⚠️ class identity is a confound here)")
    print(f"{'feature':<18}{'AUROC':>8}{'perm p':>9}   direction")
    print("-" * 56)
    out = {
        "model": man["model"],
        "tag": tag or None,
        "n_rows": len(rows),
        "n_caught": int(y.sum()),
        "pooled": {},
        "within": {},
    }
    for f in feats:
        x = np.array([r[f] for r in rows], float)
        a = auroc(x, y)
        p = perm_p(x, y, a)
        d = "higher -> caught" if a > 0.5 else "lower -> caught"
        star = " *" if p < 0.05 else ""
        print(f"{f:<18}{a:>8.3f}{p:>9.3f}   {d}{star}")
        out["pooled"][f] = {"auroc": a, "perm_p": p}

    print("\nWITHIN class  (confound-free, but n is tiny; direction check only)")
    hdr = "".join(f"{f:>12}" for f in feats)
    print(f"{'class':<32}{'n(+/-)':>9}{hdr}")
    print("-" * (32 + 9 + 12 * len(feats)))
    for C in sorted({r["class"] for r in rows}):
        sub = [r for r in rows if r["class"] == C]
        yy = np.array([r["caught"] for r in sub])
        if len(np.unique(yy)) < 2:
            continue
        cells, rec = "", {}
        for f in feats:
            a = auroc([r[f] for r in sub], yy)
            rec[f] = a
            cells += f"{a:>12.2f}"
        print(f"{C:<32}{f'{int(yy.sum())}/{int((1 - yy).sum())}':>9}{cells}")
        out["within"][C] = rec

    p_out = V2 / f"member_separability{suf}.json"
    json.dump(out, open(p_out, "w"), indent=2)
    print(f"\nwrote {p_out}")
    print("\nAUROC 0.5 means the feature carries no information about which side a")
    print("protein falls on. A feature that works pooled but not within class is")
    print("tracking class identity, not the property of interest.")


if __name__ == "__main__":
    main()
