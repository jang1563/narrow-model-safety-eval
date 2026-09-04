#!/usr/bin/env python3
"""
03h_probe_vs_similarity.py - three controls that could overturn the earlier conclusions.

Each of these tests an alternative explanation that the existing results cannot
rule out. They are listed in order of how much damage they would do.

1. Is the probe doing anything a nearest-neighbour lookup does not?
   `03g` found that similarity to an already-seen toxin predicts whether a
   held-out protein is caught, AUROC 0.942. But the probe is a linear model over
   the same embeddings that the similarity is computed in, so that result risks
   being close to circular. The sharp version of the question is operational:
   run leave-one-mechanism-out with a pure similarity rule, score = max cosine to
   a training positive minus max cosine to a training negative, calibrated on the
   same held-out negatives. If the similarity rule matches the trained probe,
   **the probe adds nothing over lookup** and every recovery number in this
   project should be read as a nearest-neighbour result.

2. Is poor recovery just "we deleted more training data"?
   Holding out beta-lactamase removes 14 of 66 positives, 21% of the training
   positives; holding out a 3-member class removes 4.5%. Class size and recovery
   are therefore confounded by construction. This repeats the experiment with the
   training positives subsampled to a FIXED count for every class, so each class
   is trained against the same amount of data.

3. How independent are the members of a class?
   "All four clostridial neurotoxins are always caught" is four correlated
   observations if those four are homologous to each other. Within-class mean
   pairwise similarity bounds how much weight a class-level claim can carry.

Run after 02b. Uses cached embeddings.

Usage:
    python src/03h_probe_vs_similarity.py [--tag esmc_600M] [--seeds 30]
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


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))


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


def kmers(s, k=5):
    return set(s[i : i + k] for i in range(max(0, len(s) - k + 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--seeds", type=int, default=30)
    a = ap.parse_args()
    suf = f"_{a.tag}" if a.tag else ""

    P = np.load(V2 / f"embeddings_positive_v2{suf}.npy")
    N = np.load(V2 / f"embeddings_negative_v2{suf}.npy")
    man = json.load(open(V2 / f"embedding_manifest_v2{suf}.json"))
    mech = json.load(open(ROOT / "data/annotations/mechanism_classes_v2.json"))

    pos_acc = [r["acc"] for r in man["positive_rows"]]
    cls_of = {p["fasta_id"]: p["mechanism_class"] for p in mech["proteins"]}
    pos_cls = np.array([cls_of.get(x, "UNMAPPED") for x in pos_acc])
    classes = sorted(
        {c for c in pos_cls if c in set(mech["holdout_eligible_classes"])}
        | {"virulence_associated_non_toxin"}
    )

    Pn = P / np.linalg.norm(P, axis=1, keepdims=True)
    Nn = N / np.linalg.norm(N, axis=1, keepdims=True)

    # smallest training-positive count any class leaves behind, used for control 2
    fixed_train = int(len(P) - max((pos_cls == C).sum() for C in classes))
    h = int(len(N) * NEG_HOLDOUT_FRAC)
    print(f"model={man['model']}  positives={len(P)}  negatives={len(N)}")
    print(f"held-out negatives={h}  fixed training positives for control 2={fixed_train}\n")

    probe, simrule, sized = {C: [] for C in classes}, {C: [] for C in classes}, {C: [] for C in classes}
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        p = rng.permutation(len(N))
        nte, ntr = p[:h], p[h:]
        for C in classes:
            hi = np.where(pos_cls == C)[0]
            tri = np.setdiff1d(np.arange(len(P)), hi)

            # --- trained probe, the number reported everywhere else -----------
            m = clf().fit(
                np.vstack([P[tri], N[ntr]]), np.r_[np.ones(len(tri)), np.zeros(len(ntr))]
            )
            s_ho, s_ca = m.predict_proba(P[hi])[:, 1], m.predict_proba(N[nte])[:, 1]
            t = np.quantile(s_ca, SPEC)
            probe[C].append(float((s_ho >= t).mean()))

            # --- pure similarity rule, no fitting at all ----------------------
            def simscore(rows):
                return (rows @ Pn[tri].T).max(axis=1) - (rows @ Nn[ntr].T).max(axis=1)

            q_ho, q_ca = simscore(Pn[hi]), simscore(Nn[nte])
            tq = np.quantile(q_ca, SPEC)
            simrule[C].append(float((q_ho >= tq).mean()))

            # --- probe again, training positives capped at a fixed count ------
            tri_s = rng.permutation(tri)[:fixed_train]
            m2 = clf().fit(
                np.vstack([P[tri_s], N[ntr]]), np.r_[np.ones(len(tri_s)), np.zeros(len(ntr))]
            )
            s2, c2 = m2.predict_proba(P[hi])[:, 1], m2.predict_proba(N[nte])[:, 1]
            sized[C].append(float((s2 >= np.quantile(c2, SPEC)).mean()))

    # ---- within-class redundancy -------------------------------------------
    pf = read_fasta(ROOT / "data/sequences/toxins_positive_v2.fasta")
    ks = [kmers(pf[a_]) for a_ in pos_acc]

    def within(idx):
        cos, jac = [], []
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                cos.append(float(Pn[idx[i]] @ Pn[idx[j]]))
                u = len(ks[idx[i]] | ks[idx[j]])
                jac.append(len(ks[idx[i]] & ks[idx[j]]) / u if u else 0.0)
        return (float(np.mean(cos)), float(np.mean(jac)), float(np.max(jac))) if cos else (np.nan,) * 3

    print(f"{'class':<32}{'n':>3}{'probe':>9}{'sim-only':>10}{'size-fixed':>12}"
          f"{'w/in cos':>10}{'w/in 5mer':>11}{'max 5mer':>10}")
    print("-" * 97)
    out = {"model": man["model"], "tag": a.tag or None, "seeds": a.seeds,
           "fixed_train_positives": fixed_train, "classes": {}}
    for C in classes:
        idx = np.where(pos_cls == C)[0]
        pr, sr, sz = np.mean(probe[C]), np.mean(simrule[C]), np.mean(sized[C])
        wc, wj, mj = within(idx)
        print(f"{C:<32}{len(idx):>3}{pr:>8.0%} {sr:>9.0%} {sz:>11.0%}"
              f"{wc:>10.3f}{wj:>11.3f}{mj:>10.3f}")
        out["classes"][C] = {"n": int(len(idx)), "probe": pr, "similarity_only": sr,
                             "size_fixed_probe": sz, "within_cosine": wc,
                             "within_kmer_mean": wj, "within_kmer_max": mj}
    d = np.array([np.mean(probe[C]) - np.mean(simrule[C]) for C in classes])
    dz = np.array([np.mean(probe[C]) - np.mean(sized[C]) for C in classes])
    print(f"\nprobe minus similarity-only: mean {d.mean():+.1%}, "
          f"range {d.min():+.0%} to {d.max():+.0%}")
    print(f"probe minus size-fixed probe: mean {dz.mean():+.1%}, "
          f"range {dz.min():+.0%} to {dz.max():+.0%}")
    out["probe_minus_similarity_mean"] = float(d.mean())
    out["probe_minus_sizefixed_mean"] = float(dz.mean())

    p_out = V2 / f"probe_vs_similarity{suf}.json"
    json.dump(out, open(p_out, "w"), indent=2)
    print(f"\nwrote {p_out}")
    print("\nIf sim-only matches probe, the probe is a nearest-neighbour lookup.")
    print("If size-fixed matches probe, class size was not driving the differences.")
    print("High within-class similarity means that class's members are not independent.")


if __name__ == "__main__":
    main()
