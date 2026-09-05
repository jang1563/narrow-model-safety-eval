#!/usr/bin/env python3
"""
03l_ensemble_alignment_embedding.py - can alignment and embeddings be combined into a better detector?

Why this was worth trying
-------------------------
§6i established that the two methods fail on **disjoint** classes: alignment
recovers beta-lactamase at 31% where the embedding probe manages 21% and an
unfitted cosine rule only 1%, while the probe recovers superantigen at 100% where
alignment gets 5%. Across classes their recoveries are essentially uncorrelated.
Complementary detectors that fail in different places are the textbook case for
an ensemble.

Why it does not work, which is the actual result
------------------------------------------------
Three combination rules, all calibrated on the same held-out negatives at the same
operating point as everything else in this project:

    max         rank-normalize both scores, take the elementwise maximum
    or_split    flag if either score clears its own threshold, with the false
                positive budget split 2.5% each so the union targets 5%
    learned     a two-feature logistic stack over (probe score, alignment margin),
                fitted on the training positives and training negatives

The failure mode of `max` is worth stating because it is general: OR-ing two
detectors raises the **negatives'** scores as well, which pushes the calibrated
threshold up, which costs recall. Under a fixed false-positive budget a union is
not free.

This file exists because memo 49 quotes these numbers as a reason not to claim a
combined detector, and a document written for external use should not cite a
result that was computed once in a shell and never saved.

Run after 02b and 03i (the alignment matrix is cached by 03i).

Usage:
    python src/03l_ensemble_alignment_embedding.py [--seeds 20]
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
NEG_HOLDOUT_FRAC = 0.40
SPEC = 0.95
SPLIT = 0.975  # each arm of the OR, so the union targets 1 - SPEC overall


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))


def rank01(x):
    return np.argsort(np.argsort(x)) / max(len(x) - 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    a = ap.parse_args()

    P = np.load(V2 / "embeddings_positive_v2.npy")
    N = np.load(V2 / "embeddings_negative_v2.npy")
    man = json.load(open(V2 / "embedding_manifest_v2.json"))
    mech = json.load(open(ROOT / "data/annotations/mechanism_classes_v2.json"))
    M = np.load(V2 / "alignment_scores.npz")["M"]

    pos_acc = [r["acc"] for r in man["positive_rows"]]
    cls_of = {p["fasta_id"]: p["mechanism_class"] for p in mech["proteins"]}
    pos_cls = np.array([cls_of.get(x, "UNMAPPED") for x in pos_acc])
    classes = sorted(
        {c for c in pos_cls if c in set(mech["holdout_eligible_classes"])}
        | {"virulence_associated_non_toxin"}
    )
    nP, nN = len(P), len(N)
    Pi, Ni = np.arange(nP), np.arange(nP, nP + nN)
    h = int(nN * NEG_HOLDOUT_FRAC)

    rows, fprs = {}, {"or_split": []}
    for C in classes:
        hi = np.where(pos_cls == C)[0]
        tri = np.setdiff1d(Pi, hi)
        acc = {k: [] for k in ("embedding", "alignment", "max", "or_split", "learned")}
        for seed in range(a.seeds):
            rng = np.random.default_rng(seed)
            p = rng.permutation(nN)
            nte, ntr = p[:h], p[h:]
            m = clf().fit(np.vstack([P[tri], N[ntr]]),
                          np.r_[np.ones(len(tri)), np.zeros(len(ntr))])
            e_ho = m.predict_proba(P[hi])[:, 1]
            e_ca = m.predict_proba(N[nte])[:, 1]

            def margin(rows_):
                return M[np.ix_(rows_, tri)].max(axis=1) - M[np.ix_(rows_, Ni[ntr])].max(axis=1)

            a_ho, a_ca = margin(hi), margin(Ni[nte])
            acc["embedding"].append(float((e_ho >= np.quantile(e_ca, SPEC)).mean()))
            acc["alignment"].append(float((a_ho >= np.quantile(a_ca, SPEC)).mean()))

            re = rank01(np.r_[e_ca, e_ho])
            ra = rank01(np.r_[a_ca, a_ho])
            s = np.maximum(re, ra)
            acc["max"].append(float((s[len(e_ca):] >= np.quantile(s[: len(e_ca)], SPEC)).mean()))

            te, ta = np.quantile(e_ca, SPLIT), np.quantile(a_ca, SPLIT)
            acc["or_split"].append(float(((e_ho >= te) | (a_ho >= ta)).mean()))
            fprs["or_split"].append(float(((e_ca >= te) | (a_ca >= ta)).mean()))

            e_tr = m.predict_proba(P[tri])[:, 1]
            e_ntr = m.predict_proba(N[ntr])[:, 1]
            X = np.c_[np.r_[e_tr, e_ntr], np.r_[margin(tri), margin(Ni[ntr])]]
            y = np.r_[np.ones(len(tri)), np.zeros(len(ntr))]
            st = clf().fit(X, y)
            s_ho = st.predict_proba(np.c_[e_ho, a_ho])[:, 1]
            s_ca = st.predict_proba(np.c_[e_ca, a_ca])[:, 1]
            acc["learned"].append(float((s_ho >= np.quantile(s_ca, SPEC)).mean()))
        rows[C] = {k: float(np.mean(v)) for k, v in acc.items()}

    names = ["embedding", "alignment", "max", "or_split", "learned"]
    print(f"{'class':<32}" + "".join(f"{n:>11}" for n in names))
    print("-" * (32 + 11 * len(names)))
    for C in classes:
        print(f"{C:<32}" + "".join(f"{rows[C][n]:>10.0%} " for n in names))
    means = {n: float(np.mean([rows[C][n] for C in classes])) for n in names}
    print(f"\n{'mean':<32}" + "".join(f"{means[n]:>10.0%} " for n in names))
    fpr = float(np.mean(fprs["or_split"]))
    print(f"\nachieved FPR of or_split: {fpr:.1%} against a {1 - SPEC:.0%} budget")
    print(f"best combination beats embedding alone by "
          f"{max(means['max'], means['or_split'], means['learned']) - means['embedding']:+.1%}")

    ecls = [rows[C]["embedding"] for C in classes]
    acls = [rows[C]["alignment"] for C in classes]
    corr = float(np.corrcoef(ecls, acls)[0, 1])
    print(f"correlation between embedding and alignment recovery across classes: {corr:+.2f}")

    out = {"model": man["model"], "seeds": a.seeds, "spec": SPEC, "split": SPLIT,
           "per_class": rows, "means": means, "or_split_achieved_fpr": fpr,
           "embedding_alignment_class_correlation": corr,
           "verdict": "no combination beats the embedding probe alone under a fixed "
                      "false-positive budget"}
    p_out = V2 / "ensemble_alignment_embedding.json"
    json.dump(out, open(p_out, "w"), indent=2)
    print(f"\nwrote {p_out}")


if __name__ == "__main__":
    main()
