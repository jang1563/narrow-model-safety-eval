#!/usr/bin/env python3
"""
03j_classifier_sweep.py - are the findings an artifact of using a linear head?

Every leave-one-mechanism-out number in this project comes from L2 logistic
regression on standardized mean-pooled embeddings. That is one decision rule, and
two of the project's conclusions could plausibly be properties of it rather than
of the representation:

  - beta-lactamase is never recovered above 21% on ESM-2
  - on ESM-3 the fitted probe is worse than an unfitted cosine rule

If a non-linear head rescues beta-lactamase, the failure was the decision rule.
If nothing does, the failure is upstream, in what the pooled embedding retains.

Four heads spanning very different inductive biases, all on the same folds, the
same held-out negatives and the same 95% operating point:
    logistic     the incumbent, linear in embedding space
    svm_rbf      non-linear boundary, still a margin method
    random_forest  axis-aligned partitions, no distance metric at all
    knn_5        local, no global boundary, closest thing to a lookup with a vote

Usage:
    python src/03j_classifier_sweep.py [--tag esm3_1_4B] [--seeds 20]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "results" / "v2"
NEG_HOLDOUT_FRAC = 0.40
SPEC = 0.95


def heads(seed):
    return {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        "svm_rbf": make_pipeline(StandardScaler(), SVC(probability=True, random_state=seed)),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=seed, class_weight="balanced_subsample"
        ),
        "knn_5": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--seeds", type=int, default=20)
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
    names = list(heads(0))
    h = int(len(N) * NEG_HOLDOUT_FRAC)
    print(f"model={man['model']}  {len(P)} pos / {len(N)} neg  held-out negatives={h}\n")

    rec = {n: {C: [] for C in classes} for n in names}
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        p = rng.permutation(len(N))
        nte, ntr = p[:h], p[h:]
        for C in classes:
            hi = np.where(pos_cls == C)[0]
            tri = np.setdiff1d(np.arange(len(P)), hi)
            Xtr = np.vstack([P[tri], N[ntr]])
            ytr = np.r_[np.ones(len(tri)), np.zeros(len(ntr))]
            for nm, mdl in heads(seed).items():
                m = mdl.fit(Xtr, ytr)
                s_ho = m.predict_proba(P[hi])[:, 1]
                s_ca = m.predict_proba(N[nte])[:, 1]
                rec[nm][C].append(float((s_ho >= np.quantile(s_ca, SPEC)).mean()))

    print(f"{'class':<32}{'n':>3}" + "".join(f"{n:>15}" for n in names))
    print("-" * (35 + 15 * len(names)))
    out = {"model": man["model"], "tag": a.tag or None, "seeds": a.seeds, "classes": {}}
    for C in classes:
        vals = [np.mean(rec[n][C]) for n in names]
        best = names[int(np.argmax(vals))]
        cells = "".join(f"{v:>14.0%} " for v in vals)
        print(f"{C:<32}{int((pos_cls == C).sum()):>3}{cells}  best={best}")
        out["classes"][C] = dict(zip(names, [float(v) for v in vals]))
    print(f"\n{'mean over classes':<35}" + "".join(
        f"{np.mean([np.mean(rec[n][C]) for C in classes]):>14.0%} " for n in names))
    out["mean_by_head"] = {
        n: float(np.mean([np.mean(rec[n][C]) for C in classes])) for n in names
    }
    bl = out["classes"].get("beta_lactamase", {})
    if bl:
        print(f"\nbeta-lactamase, the class no head has rescued so far: "
              f"best {max(bl.values()):.0%} by {max(bl, key=bl.get)}")

    p_out = V2 / f"classifier_sweep{suf}.json"
    json.dump(out, open(p_out, "w"), indent=2)
    print(f"\nwrote {p_out}")


if __name__ == "__main__":
    main()
