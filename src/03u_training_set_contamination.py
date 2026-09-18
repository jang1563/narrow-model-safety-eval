#!/usr/bin/env python3
"""
03u_training_set_contamination.py - what one class in the training set costs another.

What this measures
------------------
§5 says recovery is a joint property of the class, the rest of the positive set and
the operating point, shown there by changing the panel SIZE. This is the same claim
with a named cause: remove the 14 beta-lactamases from the TRAINING set, hold the
test class out as usual, and see what moves.

⚠️ EXPLORATORY, not preregistered. Found while running src/03t, which tested
whether non-animal-target hazard is a learnable category and refuted it. 03t's
per-class output showed two classes moving 20+ points in opposite directions while
the mean interaction sat at zero; this attributes those movements.

🔴 A single random removal is not a control, and assuming it was nearly produced a
wrong number here. Two draws of 14 random positives moved pore-forming recovery to
71.4% and to 80.9%, a 9.5-point spread, which is most of the effect that was about
to be attributed to beta-lactamase. The control is the DISTRIBUTION over draws, and
the question is where removing beta-lactamase falls inside it.

Design
------
One protocol throughout, 03b's: hold out 40% of the negatives, calibrate at the
95th percentile of the HELD-OUT negatives. The test class is removed from training
in every condition, so conditions differ only in which OTHER positives remain:

    standard    every positive except the test class
    minus_BL    the same, minus the 14 beta-lactamases
    random      the same, minus 14 drawn at random, repeated over many draws

Usage:
    python src/03u_training_set_contamination.py
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
FRAC, SPEC = 0.40, 0.95
FOCUS = "pore_forming_cytolysin"


def recover(P, N, tri, hi, seeds):
    per, mem = [], np.zeros(len(hi))
    for sd in range(seeds):
        rng = np.random.default_rng(sd)
        p = rng.permutation(len(N))
        cut = int(len(N) * FRAC)
        nte, ntr = p[:cut], p[cut:]
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
        m.fit(np.vstack([P[tri], N[ntr]]), np.r_[np.ones(len(tri)), np.zeros(len(ntr))])
        thr = np.quantile(m.predict_proba(N[nte])[:, 1], SPEC)
        f = m.predict_proba(P[hi])[:, 1] >= thr
        per.append(float(f.mean()))
        mem += f
    return np.array(per), mem / seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=60)
    ap.add_argument("--draws", type=int, default=25)
    ap.add_argument("--survey-seeds", type=int, default=20)
    ap.add_argument("--survey-draws", type=int, default=10)
    a = ap.parse_args()

    P = np.load(V2 / "embeddings_positive_v2.npy")
    N = np.load(V2 / "embeddings_negative_v2.npy")
    man = json.load(open(V2 / "embedding_manifest_v2.json"))
    mech = json.load(open(ROOT / "data/annotations/mechanism_classes_v2.json"))
    cls = {e["fasta_id"]: e["mechanism_class"] for e in mech["proteins"]}
    accs = [r["acc"] for r in man["positive_rows"]]
    names = [r["name"] for r in man["positive_rows"]]
    pcls = np.array([cls[x] for x in accs])
    bl = pcls == "beta_lactamase"
    print(f"panel {len(P)} positives, {int(bl.sum())} beta-lactamases")

    def masks(hi):
        held = set(hi.tolist())
        std = np.array([i for i in range(len(P)) if i not in held])
        return std, np.array([i for i in std if not bl[i]])

    def draws_of(std, k, n):
        rng = np.random.default_rng(0)
        return [rng.choice(std, k, replace=False) for _ in range(n)]

    print(f"\nsurvey, {a.survey_seeds} seeds, {a.survey_draws} random draws")
    print(f"{'class':<32}{'n':>3}{'standard':>10}{'minus_BL':>10}"
          f"{'random draws':>18}{'BL pctile':>11}")
    print("-" * 84)
    survey = {}
    for C in sorted(set(pcls)):
        hi = np.where(pcls == C)[0]
        if len(hi) < 4 or C == "beta_lactamase":
            continue
        std, mbl = masks(hi)
        s = recover(P, N, std, hi, a.survey_seeds)[0].mean()
        b = recover(P, N, mbl, hi, a.survey_seeds)[0].mean()
        rs = np.array([recover(P, N, r, hi, a.survey_seeds)[0].mean()
                       for r in draws_of(std, len(mbl), a.survey_draws)])
        pct = float((rs < b).mean())
        survey[C] = {"n": int(len(hi)), "standard": float(s), "minus_BL": float(b),
                     "random_mean": float(rs.mean()), "random_sd": float(rs.std()),
                     "random_min": float(rs.min()), "random_max": float(rs.max()),
                     "minus_BL_percentile": pct}
        print(f"{C:<32}{len(hi):>3}{s * 100:>9.1f}%{b * 100:>9.1f}%"
              f"{rs.mean() * 100:>11.1f}% +-{rs.std() * 100:>4.1f}{pct * 100:>10.0f}%")

    hi = np.where(pcls == FOCUS)[0]
    std, mbl = masks(hi)
    s_per, s_mem = recover(P, N, std, hi, a.seeds)
    b_per, b_mem = recover(P, N, mbl, hi, a.seeds)
    rdist = np.array([recover(P, N, r, hi, a.seeds)[0].mean()
                      for r in draws_of(std, len(mbl), a.draws)])
    d = b_per - s_per
    bs = [float(np.mean(np.random.default_rng(i).choice(d, len(d)))) for i in range(5000)]
    lo, hi_ci = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    pct = float((rdist < b_per.mean()).mean())
    attrib = float(b_per.mean() - rdist.mean())

    print(f"\n{FOCUS}, {a.seeds} seeds")
    print(f"  standard   {s_per.mean() * 100:>6.1f}% +- {s_per.std() * 100:.1f}")
    print(f"  minus_BL   {b_per.mean() * 100:>6.1f}% +- {b_per.std() * 100:.1f}")
    print(f"  paired minus_BL - standard: {d.mean() * 100:+.1f} points, "
          f"95% CI [{lo * 100:+.1f}, {hi_ci * 100:+.1f}], "
          f"wins {int((d > 0).sum())}/{a.seeds}")
    print(f"  random removals, {a.draws} draws: {rdist.mean() * 100:.1f}% "
          f"+- {rdist.std() * 100:.1f}, "
          f"range [{rdist.min() * 100:.1f}, {rdist.max() * 100:.1f}]")
    print(f"  minus_BL sits at the {pct * 100:.0f}th percentile of random removals")
    print(f"  attributable to beta-lactamase beyond a random removal of the same size: "
          f"{attrib * 100:+.1f} points")

    print("\nper member, flagged fraction")
    print(f"  {'member':<16}{'standard':>10}{'minus_BL':>10}")
    for k, i in enumerate(hi):
        print(f"  {names[i]:<16}{s_mem[k] * 100:>9.0f}%{b_mem[k] * 100:>9.0f}%")
    movers = [names[i] for k, i in enumerate(hi) if b_mem[k] - s_mem[k] > 0.25]
    print(f"\n  members gaining more than 25 points: {movers}")

    res = {"exploratory": True,
           "found_while": "running src/03t_animal_only_training.py, which refuted a "
                          "different hypothesis",
           "control_note": "a single random removal is not a control; two draws spanned "
                           "9.5 points on this class, so the distribution is used",
           "protocol": "03b: 40% negative holdout, threshold at the 95th percentile of "
                       "held-out negatives",
           "n_removed": int(bl.sum()),
           "seeds": a.seeds, "draws": a.draws,
           "survey_seeds": a.survey_seeds, "survey_draws": a.survey_draws,
           "survey": survey, "focus_class": FOCUS,
           "focus": {"standard": float(s_per.mean()), "standard_sd": float(s_per.std()),
                     "minus_BL": float(b_per.mean()), "minus_BL_sd": float(b_per.std()),
                     "paired_delta": float(d.mean()), "paired_ci95": [lo, hi_ci],
                     "paired_wins": int((d > 0).sum()),
                     "random_mean": float(rdist.mean()), "random_sd": float(rdist.std()),
                     "random_min": float(rdist.min()), "random_max": float(rdist.max()),
                     "minus_BL_percentile": pct,
                     "attributable_beyond_random": attrib,
                     "per_member_standard": {names[i]: float(s_mem[k])
                                             for k, i in enumerate(hi)},
                     "per_member_minus_BL": {names[i]: float(b_mem[k])
                                             for k, i in enumerate(hi)}},
           "members_gaining_over_25pts": movers}
    p = V2 / "training_set_contamination.json"
    json.dump(res, open(p, "w"), indent=2)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
