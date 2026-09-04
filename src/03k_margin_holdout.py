#!/usr/bin/env python3
"""
03k_margin_holdout.py - the novel-molecule case, stated as a limit instead of a correlation.

The gap this closes
-------------------
§6f found that whether a held-out protein is caught is predicted by its
embedding-space margin to an already-seen toxin, AUROC 0.942. That is a
correlation over proteins that happened to be in each class. Every mechanism
class in this panel contains at least some members close to the training
positives, so the honest test case, a holdout whose members are ALL far from
anything seen, has never been run.

This runs it, using only the existing panel. Positives are ranked by a
leave-one-out margin, and three holdouts of identical size are compared:

    low     the k lowest-margin positives. The novel-molecule case
    random  k drawn at random, repeated over seeds. The control that makes the
            comparison mean something
    high    the k highest-margin positives. The optimistic case

If `low` collapses while `random` sits near the panel average, the ceiling is not
a property of any mechanism class but of distance to what the probe has seen, and
that is a limit that can be stated in advance rather than discovered afterwards.

⚠️ The ranking uses each protein's margin against all 65 others, while the
evaluation uses its margin against the 66 - k that remain after the holdout is
removed. Those differ slightly. They are monotonically related and the ranking is
only used to select the holdout, but it is not a circularity-free design and is
reported as such.

A second view splits each class at its own median margin, so the effect is also
visible inside the leave-one-mechanism-out framing rather than only across it.

Usage:
    python src/03k_margin_holdout.py [--tag saprot_650M] [--k 10] [--seeds 30]
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


def run_holdout(P, N, hi, seeds):
    """flagged@95 for a fixed holdout set, averaged over negative splits."""
    tri = np.setdiff1d(np.arange(len(P)), hi)
    h = int(len(N) * NEG_HOLDOUT_FRAC)
    out = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        p = rng.permutation(len(N))
        nte, ntr = p[:h], p[h:]
        m = clf().fit(np.vstack([P[tri], N[ntr]]),
                      np.r_[np.ones(len(tri)), np.zeros(len(ntr))])
        s_ho = m.predict_proba(P[hi])[:, 1]
        s_ca = m.predict_proba(N[nte])[:, 1]
        out.append(float((s_ho >= np.quantile(s_ca, SPEC)).mean()))
    return float(np.mean(out)), float(np.std(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--random_repeats", type=int, default=15)
    a = ap.parse_args()
    suf = f"_{a.tag}" if a.tag else ""

    P = np.load(V2 / f"embeddings_positive_v2{suf}.npy")
    N = np.load(V2 / f"embeddings_negative_v2{suf}.npy")
    man = json.load(open(V2 / f"embedding_manifest_v2{suf}.json"))
    mech = json.load(open(ROOT / "data/annotations/mechanism_classes_v2.json"))
    pos_acc = [r["acc"] for r in man["positive_rows"]]
    cls_of = {p["fasta_id"]: p["mechanism_class"] for p in mech["proteins"]}
    pos_cls = np.array([cls_of.get(x, "UNMAPPED") for x in pos_acc])

    Pn = P / np.linalg.norm(P, axis=1, keepdims=True)
    Nn = N / np.linalg.norm(N, axis=1, keepdims=True)
    simPP = Pn @ Pn.T
    # Margin must be measured against positives OUTSIDE the protein's own mechanism
    # class, which is how §6f defined it. Using all other positives instead makes
    # "high margin" mean "has a close same-class relative", which is confounded with
    # within-class redundancy: a first version of this script did that and produced a
    # high-margin holdout that was 4/14 beta-lactamase and 3/4 superantigen, and
    # whose members were each other's nearest neighbours 6 times out of 10.
    cross = simPP.copy()
    for C in set(pos_cls):
        idx = np.where(pos_cls == C)[0]
        cross[np.ix_(idx, idx)] = -np.inf
    margin = cross.max(axis=1) - (Pn @ Nn.T).max(axis=1)
    order = np.argsort(margin)

    print(f"model={man['model']}  {len(P)} positives / {len(N)} negatives  k={a.k}")
    print(f"leave-one-out margin: min {margin.min():.3f}  median {np.median(margin):.3f}"
          f"  max {margin.max():.3f}\n")

    low, high = order[: a.k], order[-a.k :]
    m_low, s_low = run_holdout(P, N, low, a.seeds)
    m_high, s_high = run_holdout(P, N, high, a.seeds)
    rnd = []
    for r in range(a.random_repeats):
        idx = np.random.default_rng(1000 + r).permutation(len(P))[: a.k]
        rnd.append(run_holdout(P, N, idx, a.seeds)[0])
    m_rnd, s_rnd = float(np.mean(rnd)), float(np.std(rnd))

    # The decisive control. A plain random holdout has a different class mix from
    # the low-margin one, so any gap between them could just be "we happened to
    # select the hardest classes". This draws random holdouts with the SAME class
    # counts as the low-margin set, so class composition is held fixed and margin
    # is the only thing that differs.
    import collections
    cnt = collections.Counter(pos_cls[low])
    matched = []
    for r in range(a.random_repeats):
        rng = np.random.default_rng(500 + r)
        H = []
        for C, n in cnt.items():
            H += list(rng.permutation(np.where(pos_cls == C)[0])[:n])
        matched.append(run_holdout(P, N, np.array(H), a.seeds)[0])
    m_mat, s_mat = float(np.mean(matched)), float(np.std(matched))

    print(f"{'holdout of ' + str(a.k):<26}{'flagged@95':>12}{'sd':>8}   margin of the holdout")
    print("-" * 74)
    for nm, mm, ss, idx in (("lowest margin", m_low, s_low, low),
                            ("random, class-matched", m_mat, s_mat, None),
                            ("random", m_rnd, s_rnd, None),
                            ("highest margin", m_high, s_high, high)):
        mg = f"{margin[idx].mean():+.3f}" if idx is not None else "  mixed"
        print(f"{nm:<26}{mm:>11.0%}{ss:>8.0%}   {mg}")
    print(f"\nlow minus class-matched random: {m_low - m_mat:+.0%}   "
          f"<- class composition held fixed, so this is the margin effect")
    print(f"low minus plain random: {m_low - m_rnd:+.0%}    "
          f"high minus plain random: {m_high - m_rnd:+.0%}")

    # ---- the same effect inside the class framing --------------------------
    print("\nwithin class, members split at that class's own median margin")
    print(f"{'class':<32}{'n':>3}{'low half':>11}{'high half':>12}{'gap':>8}")
    print("-" * 68)
    within = {}
    for C in sorted(set(pos_cls)):
        idx = np.where(pos_cls == C)[0]
        if len(idx) < 6:
            continue
        med = np.median(margin[idx])
        lo_i, hi_i = idx[margin[idx] <= med], idx[margin[idx] > med]
        if len(lo_i) == 0 or len(hi_i) == 0:
            continue
        # hold out the whole class, then read recovery separately per half
        tri = np.setdiff1d(np.arange(len(P)), idx)
        h = int(len(N) * NEG_HOLDOUT_FRAC)
        lo_r, hi_r = [], []
        for seed in range(a.seeds):
            rng = np.random.default_rng(seed)
            p = rng.permutation(len(N))
            nte, ntr = p[:h], p[h:]
            m = clf().fit(np.vstack([P[tri], N[ntr]]),
                          np.r_[np.ones(len(tri)), np.zeros(len(ntr))])
            t = np.quantile(m.predict_proba(N[nte])[:, 1], SPEC)
            lo_r.append((m.predict_proba(P[lo_i])[:, 1] >= t).mean())
            hi_r.append((m.predict_proba(P[hi_i])[:, 1] >= t).mean())
        lo_m, hi_m = float(np.mean(lo_r)), float(np.mean(hi_r))
        within[C] = {"n": int(len(idx)), "low_half": lo_m, "high_half": hi_m}
        print(f"{C:<32}{len(idx):>3}{lo_m:>10.0%} {hi_m:>11.0%} {hi_m - lo_m:>+7.0%}")

    out = {"model": man["model"], "tag": a.tag or None, "k": a.k, "seeds": a.seeds,
           "margin": {"min": float(margin.min()), "median": float(np.median(margin)),
                      "max": float(margin.max())},
           "holdout": {"low_margin": [m_low, s_low], "random": [m_rnd, s_rnd],
                       "random_class_matched": [m_mat, s_mat],
                       "high_margin": [m_high, s_high]},
           "low_minus_random": m_low - m_rnd,
           "low_minus_class_matched_random": m_low - m_mat,
           "within_class_median_split": within}
    p_out = V2 / f"margin_holdout{suf}.json"
    json.dump(out, open(p_out, "w"), indent=2)
    print(f"\nwrote {p_out}")


if __name__ == "__main__":
    main()
