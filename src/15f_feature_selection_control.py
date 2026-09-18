#!/usr/bin/env python3
"""
15f_feature_selection_control.py - was §9.6/§9.7's 0.0% an artifact of how features were chosen?

The worry
---------
15e reduced InterPLM's 10240 features to 1280 by keeping the highest-variance ones,
fit on training rows only so the held-out class could not influence the choice. In
that space beta-lactamase recovers 0.0% at every regularisation strength, while six
of seven other mechanisms reach 100%.

But variance is not discriminativeness. Checking directly: of the 279 features that
separate beta-lactamase from benign on the full panel, top-variance selection keeps
only **40 on average, 14%** (range 38-44 over 10 negative splits). So the 0.0% could
be selection discarding the signal rather than a property of the space.

The control
-----------
Same protocol, same 1280 dimensions, two selection rules:

    variance         top variance across training rows              (what 15e used)
    discriminative   top |mean_pos - mean_neg| / sd across training rows

Both are fit on training rows only. The second deliberately prioritises features that
separate positives from negatives, which is the thing variance ignores.

Result
------
    selection          mean    beta_lactamase
    variance          78.1%              0.0%
    discriminative    68.4%              0.0%

Choosing features FOR discriminativeness does not rescue beta-lactamase, and makes
every other class worse (78.1% to 68.4%). So the 0.0% is not a selection artifact.

🔑 What this pins down, combined with §9.6
------------------------------------------
Features that distinguish beta-lactamase from benign proteins demonstrably exist --
279 on the full panel, 31 under a matched-n Bonferroni test. Yet a probe reaches 0%
of the class no matter how the feature subset is picked. The difference between the
two situations is the only thing left: §9.6 selected features while LOOKING AT
beta-lactamase, and leave-one-mechanism-out must select them without it.

The discriminative features are real and are not findable from the other eight
classes. After seven refused candidates that is the sharpest available statement of
what the anomaly is.

Usage:
    python src/15f_feature_selection_control.py
"""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "results" / "v2"
FRAC, SPEC, SEEDS, DIM = 0.40, 0.95, 30, 1280


def select(Xp, Xn, rule):
    """Feature subset, fit on TRAINING rows only so the held-out class cannot
    influence which features survive."""
    if rule == "variance":
        return np.argsort(np.vstack([Xp, Xn]).var(0))[-DIM:]
    if rule == "discriminative":
        sd = np.vstack([Xp, Xn]).std(0) + 1e-9
        return np.argsort(np.abs(Xp.mean(0) - Xn.mean(0)) / sd)[-DIM:]
    raise ValueError(rule)


def run(Fp, Fn, pcls, classes, rule, C=1.0):
    out = {}
    for c in classes:
        hi = np.where(pcls == c)[0]
        tri = np.setdiff1d(np.arange(len(pcls)), hi)
        vals = []
        for seed in range(SEEDS):
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(Fn))
            cut = int(len(Fn) * FRAC)
            nte, ntr = perm[:cut], perm[cut:]
            Xp, Xn = Fp[tri], Fn[ntr]
            keep = select(Xp, Xn, rule)
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=C))
            m.fit(np.vstack([Xp[:, keep], Xn[:, keep]]),
                  np.r_[np.ones(len(tri)), np.zeros(len(ntr))])
            thr = np.quantile(m.predict_proba(Fn[np.ix_(nte, keep)])[:, 1], SPEC)
            vals.append(float((m.predict_proba(Fp[np.ix_(hi, keep)])[:, 1] >= thr).mean()))
        out[c] = float(np.mean(vals))
    return out


def main():
    F = np.load(V2 / "interplm_features_L18.npy")
    man = json.load(open(V2 / "embedding_manifest_v2.json"))
    mech = json.load(open(ROOT / "data/annotations/mechanism_classes_v2.json"))
    cls = {e["fasta_id"]: e["mechanism_class"] for e in mech["proteins"]}
    pcls = np.array([cls[r["acc"]] for r in man["positive_rows"]])
    Fp, Fn = F[:len(pcls)], F[len(pcls):]

    eligible = set(mech["holdout_eligible_classes"]) | {"virulence_associated_non_toxin"}
    classes = sorted(c for c in set(pcls) if c in eligible and (pcls == c).sum() >= 4)

    results = {}
    for rule in ("variance", "discriminative"):
        out = run(Fp, Fn, pcls, classes, rule)
        results[rule] = {"per_class": out,
                         "mean": float(np.mean(list(out.values()))),
                         "beta_lactamase": out["beta_lactamase"]}
        print(f"\n=== selection: {rule} ===")
        for c, v in sorted(out.items(), key=lambda x: -x[1]):
            print(f"  {c:<32}{v * 100:>6.1f}%")
        print(f"  {'mean':<32}{results[rule]['mean'] * 100:>6.1f}%")

    bl = {r: results[r]["beta_lactamase"] for r in results}
    verdict = ("NOT a selection artifact: beta-lactamase is 0.0% under both rules"
               if all(v == 0.0 for v in bl.values())
               else "selection matters: beta-lactamase differs between rules")
    print(f"\nbeta_lactamase by rule: {bl}")
    print(f"verdict: {verdict}")

    dest = V2 / "feature_selection_control.json"
    json.dump({"protocol": "03b: 40% negative holdout, 95th pct of held-out negatives",
               "seeds": SEEDS, "dim": DIM,
               "note": "top-variance selection keeps only 40 of the 279 "
                       "beta-lactamase-discriminative features on average (14%), "
                       "which is why the discriminative rule was tested",
               "results": results, "verdict": verdict},
              open(dest, "w"), indent=2)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
