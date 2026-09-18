#!/usr/bin/env python3
"""
03r_prevalence_adjusted.py - what the numbers look like at a deployment base rate.

Why
---
This panel is 80 positives against 154 negatives, so 34.2% of it is hazardous. A
real order queue is not. Every AUROC in this repository is computed at that 1:1.9
ratio, and AUROC is insensitive to prevalence, so it carries over unchanged to a
setting where it means something very different. AUPRC has never been computed
here at all.

Precision is the quantity that moves. At a fixed operating point the true and
false positive rates are properties of the classifier, so precision at any other
prevalence follows directly:

    precision(pi) = TPR * pi / (TPR * pi + FPR * (1 - pi))

That needs no new data and no assumption beyond the operating point transferring,
which is the same assumption already made everywhere else in this document.

What this is not
----------------
Not a claim about any real screening queue's base rate, which is not public and
varies by provider. The prevalences below are round numbers spanning plausible
regimes, reported so a reader can locate their own.

Usage:
    python src/03r_prevalence_adjusted.py
"""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "results" / "v2"
PREV = [0.342, 1e-2, 1e-3, 1e-4, 1e-5]


def main():
    P = np.load(V2 / "embeddings_positive_v2.npy")
    N = np.load(V2 / "embeddings_negative_v2.npy")
    X = np.vstack([P, N])
    y = np.r_[np.ones(len(P)), np.zeros(len(N))]
    panel_prev = len(P) / len(X)
    print(f"panel: {len(P)} positive / {len(X)} total = {panel_prev:.1%} prevalence")

    aurocs, auprcs, scores = [], [], np.zeros(len(y))
    for seed in range(10):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
            m.fit(X[tr], y[tr])
            s = m.predict_proba(X[te])[:, 1]
            aurocs.append(roc_auc_score(y[te], s))
            auprcs.append(average_precision_score(y[te], s))
            scores[te] += s / 10
    base = y.mean()
    print("\npooled 5-fold, 10 repeats")
    print(f"  AUROC {np.mean(aurocs):.3f} +- {np.std(aurocs):.3f}")
    print(f"  AUPRC {np.mean(auprcs):.3f} +- {np.std(auprcs):.3f}   "
          f"(a coin at this prevalence scores {base:.3f})")
    print(f"  AUPRC lift over the base rate: {np.mean(auprcs) / base:.2f}x")

    # operating points from the pooled out-of-fold scores
    neg = scores[y == 0]
    print(f"\n{'operating point':<22}{'TPR':>7}{'FPR':>8}" +
          "".join(f"{p:>12.0e}" if p < 0.1 else f"{'panel':>12}" for p in PREV))
    print("-" * (37 + 12 * len(PREV)))
    rows = {}
    for spec in (0.95, 0.99, 0.999):
        thr = float(np.quantile(neg, spec))
        tpr = float((scores[y == 1] >= thr).mean())
        fpr = float((neg >= thr).mean())
        prec = {}
        line = f"{f'{spec:.1%} specificity':<22}{tpr:>6.0%}{fpr:>8.1%}"
        for pi in PREV:
            v = tpr * pi / (tpr * pi + fpr * (1 - pi)) if (tpr * pi + fpr * (1 - pi)) > 0 else 0.0
            prec[str(pi)] = v
            line += f"{v:>11.1%}" if v >= 0.001 else f"{v:>11.2e}"
        rows[f"spec_{spec}"] = {"threshold": thr, "tpr": tpr, "fpr": fpr, "precision": prec}
        print(line)

    print("\nreading: at the 95% specificity operating point used throughout this")
    print("document, precision collapses once the base rate is realistic. AUROC does")
    print("not show that, which is why it should not be the headline for a screening")
    print("claim.")

    out = {"panel_prevalence": panel_prev,
           "auroc_mean": float(np.mean(aurocs)), "auroc_sd": float(np.std(aurocs)),
           "auprc_mean": float(np.mean(auprcs)), "auprc_sd": float(np.std(auprcs)),
           "auprc_baseline": float(base),
           "auprc_lift": float(np.mean(auprcs) / base),
           "prevalences": PREV, "operating_points": rows}
    p = V2 / "prevalence_adjusted.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
