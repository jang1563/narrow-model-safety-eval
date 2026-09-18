#!/usr/bin/env python3
"""
03w_target_host_class_holdout.py - is target host a real axis, or mechanism class relabelled?

What this tests, and why it is the load-bearing question
--------------------------------------------------------
§2.4 reports that a frozen ESM-2 650M representation encodes TARGET HOST at
**AUROC 0.929 +/- 0.065** with hazard held constant, from positives alone. That number
is the entire feasibility argument for a species-stratified hazard project: it says the
discrimination task is not speculative.

`src/03s_target_host_control.py` computed it with `StratifiedKFold(5)` over 73 proteins.
StratifiedKFold shuffles PROTEINS. And in this panel target host is very nearly a
function of mechanism class:

    adp_ribosyl 7/7 animal        beta_lactamase 14/14 small-molecule
    clostridial 6/6 animal        contact_dependent_inhibition 4/4 bacteria
    pore_forming 7/7 animal
    rip 7/7 animal                (10 of 13 classes are single-target)
    superantigen 7/7 animal
    t3ss 10/10 animal

So a probe scored by per-protein CV can reach 0.929 by recognising the mechanism class
and reading the target off it. That is not a target-host detector. It is the class
detector this repository has already characterised at length, wearing a different label.

This script asks the question the pooled number cannot: **does target host generalise to
a mechanism class the probe never saw?**

Producer taxonomy is NOT the confound, and that was checked first
----------------------------------------------------------------
Design constraint 1 of the scope memo is that producer species is not target species
(6 of 7 RIPs are plant-produced and act on animal ribosomes). But `producer_kingdom` in
`data/annotations/target_host_v2.json` is `bacteria_or_virus` for **74 of 80** positives,
the other 6 being those plant-produced RIPs, which are animal-target. There is almost no
producer variance for a probe to exploit, so producer kingdom cannot account for 0.929.
Finer producer structure (genus) is not tested here; §2.3's provenance probe reaches
0.818 on hazard, so it is not nothing, and it stays an open question.

Three arms
----------
    pooled_cv       replicate 03s exactly: StratifiedKFold(5) x 10 reps over 73 proteins
    class_holdout   leave-one-mechanism-class-out. Train on every assigned protein
                    except the held-out class, then predict that class's members
    within_class    the one class carrying BOTH target kinds at n>=3: the labelled
                    virulence control. Train on all other classes, separate its
                    animal members from its non-animal ones. Class identity is
                    constant here by construction, so this is the cleanest arm and
                    also the smallest

`class_weight="balanced"` throughout, so a 0.5 probability threshold means something with
51 animal against 22 non-animal.

Metric, chosen because the majority baseline is 80%
---------------------------------------------------
Eight eligible single-target classes are animal and two are not, so "always say animal"
scores 8/10 = 80% and a raw accuracy target would be vacuous. The statistic is
**balanced class-level accuracy**: the mean of (fraction of animal classes called right,
fraction of non-animal classes called right). Majority prediction scores 0.5.

A class counts as right when a majority of its members are predicted correctly.

PREREGISTERED, written before the run
-------------------------------------
Null: permute the class -> target mapping over eligible classes, keeping class sizes
fixed, 200 draws. Effective n is the number of CLASSES, which is design constraint 4 of
the scope memo, and the reason a per-protein permutation would be wrong here.

    SUPPORTED     balanced class-level accuracy >= 0.75 AND permutation p < 0.05.
                  Target host survives class holdout and the axis is real.

    REFUTED       observed balanced accuracy at or below the permutation median.
                  0.929 was class identity and the scope memo's §2 does not stand as
                  written.

    INCONCLUSIVE  between those, reported as such with the structural count below.

No fixed threshold decides this on its own: two guards written that way on 2026-09-18
each missed their own motivating case by one step (0.50 against 0.49, 0.55 against
0.556), so the permutation interval is reported alongside whatever the point estimate is.

The structural count, which is reported either way
--------------------------------------------------
However the arms come out, the number that decides whether the project is feasible today
is **how many mechanism classes carry a non-animal target**. If it is two, then no
leave-one-class-out design can train on non-animal hazard and test on held-out
non-animal hazard, because removing one leaves a single example of the category. That is
a statement about the panel rather than about the representation, and it belongs in the
scope memo either way.

Usage:
    python src/03w_target_host_class_holdout.py
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RES_ROOT = ROOT / "results"
NONANIMAL = {"bacteria", "none_small_molecule", "plant", "other_nonanimal"}
# 200 rather than 2000: each draw refits the probe once per eligible class, so 2000 is
# hours of compute for a resolution the class-level n cannot use anyway.
PERMS = 200
MIN_CLASS_N = 2


def probe():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=5000, C=1.0,
                                            class_weight="balanced"))


def pooled_cv(X, y, reps=10):
    """03s's own metric, recomputed here so the replication is visible."""
    out = []
    for seed in range(reps):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            m = probe().fit(X[tr], y[tr])
            out.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return float(np.mean(out)), float(np.std(out))


def balanced_class_accuracy(correct_by_class, target_of_class):
    """Mean of the per-target-group hit rates, so 'always animal' scores 0.5."""
    groups = {}
    for c, ok in correct_by_class.items():
        groups.setdefault(target_of_class[c], []).append(ok)
    if len(groups) < 2:
        return float("nan"), {k: float(np.mean(v)) for k, v in groups.items()}
    return (float(np.mean([np.mean(v) for v in groups.values()])),
            {k: float(np.mean(v)) for k, v in groups.items()})


def exact_rank_test(low_group, high_group):
    """One-tailed exact Mann-Whitney on class-level values. Enumerates every assignment
    rather than approximating, because n here is single digits.

    Returns (auroc, p) where auroc is the fraction of cross-group pairs ordered as
    predicted (low_group below high_group) and p is the exact probability of an ordering
    at least that extreme under random assignment.
    """
    from itertools import combinations
    vals = list(low_group) + list(high_group)
    n1 = len(low_group)

    def stat(idx):
        lo = [vals[i] for i in idx]
        hi = [vals[i] for i in range(len(vals)) if i not in idx]
        pairs = [(a < b) + 0.5 * (a == b) for a in lo for b in hi]
        return sum(pairs) / len(pairs)

    obs = stat(tuple(range(n1)))
    allc = list(combinations(range(len(vals)), n1))
    p = sum(1 for c in allc if stat(c) >= obs) / len(allc)
    return float(obs), float(p), len(allc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="v2", choices=["v2", "v3"],
                    help="panel version. v2 is the frozen 80/154 panel every published number in docs/ rests on; v3 is 149/296, v2 plus bacteriocin, phage_peptidoglycan_hydrolase and cry_insecticidal. Switches the results directory and the annotation files together.")
    pv = ap.parse_args().panel
    V2 = RES_ROOT / pv
    man = json.load(open(V2 / f"embedding_manifest_{pv}.json"))
    th = json.load(open(ROOT / f"data/annotations/target_host_{pv}.json"))["proteins"]
    P = np.load(V2 / f"embeddings_positive_{pv}.npy")

    tgt = {e["fasta_id"]: e["target_host"] for e in th}
    mech = {e["fasta_id"]: e["mechanism_class"] for e in th}
    prod = Counter(e["producer_kingdom"] for e in th)

    rows, y, cls = [], [], []
    for i, r in enumerate(man["positive_rows"]):
        t = tgt[r["acc"]]
        if t == "animal" or t in NONANIMAL:
            rows.append(i)
            y.append(1 if t == "animal" else 0)
            cls.append(mech[r["acc"]])
    X, y, cls = P[rows], np.array(y), np.array(cls)
    print(f"producer kingdoms: {dict(prod)}")
    print(f"target-host task: {len(y)} positives, {int(y.sum())} animal, "
          f"{int((1 - y).sum())} non-animal\n")

    # ---- the structural count, independent of any model -------------------------
    per_class = {}
    for c in sorted(set(cls)):
        m = cls == c
        per_class[c] = {"n": int(m.sum()), "animal": int(y[m].sum()),
                        "nonanimal": int((1 - y[m]).sum())}
    single = {c: ("animal" if v["nonanimal"] == 0 else "non-animal")
              for c, v in per_class.items()
              if v["n"] >= MIN_CLASS_N and (v["animal"] == 0 or v["nonanimal"] == 0)}
    mixed = [c for c, v in per_class.items() if v["animal"] and v["nonanimal"]]
    n_nonanimal_classes = sum(1 for t in single.values() if t == "non-animal")
    print("=== structural count ===")
    for c, v in sorted(per_class.items(), key=lambda kv: -kv[1]["n"]):
        tag = single.get(c, "MIXED" if c in mixed else "n<2")
        print(f"  {c:<32}n={v['n']:>3}  animal={v['animal']:>3}  "
              f"non-animal={v['nonanimal']:>3}   {tag}")
    print(f"\n  single-target classes carrying non-animal: {n_nonanimal_classes}")
    print(f"  classes carrying both kinds: {mixed or 'none'}\n")

    # ---- arm 1: replicate the published pooled number ---------------------------
    auroc, sd = pooled_cv(X, y)
    print(f"=== arm 1, pooled per-protein CV ===\n  AUROC {auroc:.4f} +/- {sd:.4f}"
          f"   (03s published 0.9294 +/- 0.0650)\n")

    # ---- arm 2: leave one mechanism class out ----------------------------------
    eligible = sorted(single)
    correct, detail = {}, {}
    for c in eligible:
        te = cls == c
        m = probe().fit(X[~te], y[~te])
        pred = m.predict(X[te])
        frac = float((pred == y[te]).mean())
        correct[c] = frac > 0.5
        detail[c] = {"target": single[c], "n": int(te.sum()), "frac_correct": frac,
                     "mean_p_animal": float(m.predict_proba(X[te])[:, 1].mean())}
    bal, by_group = balanced_class_accuracy(correct, single)
    print("=== arm 2, leave-one-mechanism-class-out ===")
    print(f"  {'class':<32}{'target':>12}{'n':>4}{'correct':>9}{'p(animal)':>11}")
    for c in eligible:
        d = detail[c]
        print(f"  {c:<32}{d['target']:>12}{d['n']:>4}"
              f"{d['frac_correct'] * 100:>8.0f}%{d['mean_p_animal']:>11.3f}")
    print(f"\n  balanced class-level accuracy: {bal:.3f}   by group: "
          + ", ".join(f"{k} {v:.2f}" for k, v in by_group.items()))

    # class-level permutation null: shuffle which classes carry which target
    rng = np.random.default_rng(0)
    labels = np.array([single[c] for c in eligible])
    null = []
    for _ in range(PERMS):
        perm = dict(zip(eligible, rng.permutation(labels)))
        # Only the eligible classes' labels are permuted. Members of the mixed class and
        # of the n<2 classes keep their true labels, because they are in training under
        # arm 2 as well and the null has to permute the association being tested, not
        # the rest of the training set.
        yb = y.copy()
        for c_ in eligible:
            yb[cls == c_] = 1 if perm[c_] == "animal" else 0
        ok = {}
        for c in eligible:
            te = cls == c
            m = probe().fit(X[~te], yb[~te])
            ok[c] = float((m.predict(X[te]) == yb[te]).mean()) > 0.5
        b, _ = balanced_class_accuracy(ok, perm)
        null.append(b)
    null = np.array([v for v in null if not np.isnan(v)])
    p = float((null >= bal).mean()) if len(null) else float("nan")
    frac_perfect = float((null >= 0.999).mean())
    print(f"  class-level permutation null: {len(null)} draws, median "
          f"{np.median(null):.3f}, 95th pct {np.percentile(null, 95):.3f}, p {p:.4f}")
    # Say what the number means rather than asserting a fixed conclusion. On v2 this
    # printed "almost no power" from 8% of draws reaching a perfect score, which was true;
    # printing the same sentence at 0% would have been false, and it did once.
    verdict_power = ("has almost no power" if frac_perfect >= 0.05 else "is usable")
    print(f"  null reaches 1.000 on {frac_perfect * 100:.0f}% of draws and its 95th "
          f"percentile is {np.percentile(null, 95):.3f}, so the preregistered test "
          f"{verdict_power} at {len(eligible)} classes")

    # POST HOC, and labelled as such: added after seeing that the permutation null above
    # is uninformative. Not part of the preregistration and it does not set the verdict.
    lo = [detail[c]["mean_p_animal"] for c in eligible if single[c] == "non-animal"]
    hi = [detail[c]["mean_p_animal"] for c in eligible if single[c] == "animal"]
    cl_auroc, cl_p, n_orderings = exact_rank_test(lo, hi)
    inside = [c for c in eligible if single[c] == "animal"
              and detail[c]["mean_p_animal"] < max(lo)]
    print(f"\n  post hoc, class-level ordering of mean p(animal): AUROC {cl_auroc:.3f}, "
          f"exact one-tailed p {cl_p:.3f} over {n_orderings} orderings "
          f"({len(lo)} non-animal vs {len(hi)} animal classes)")
    print(f"  animal classes scoring BELOW the highest non-animal class: "
          f"{inside or 'none'}")

    # ---- arm 3: within the one class that carries both kinds -------------------
    within = {}
    for c in mixed:
        te = cls == c
        if y[te].sum() < 2 or (1 - y[te]).sum() < 2:
            within[c] = {"n": int(te.sum()), "skipped": "fewer than 2 per kind"}
            continue
        m = probe().fit(X[~te], y[~te])
        s = m.predict_proba(X[te])[:, 1]
        # Exact test in BOTH directions, because an AUROC of 0 is as unlikely as one of 1
        # and calling it "chance" would be wrong. n is 3 against 3, so 20 orderings exist.
        _, p_pred, n_ord = exact_rank_test(s[y[te] == 0], s[y[te] == 1])
        _, p_inv, _ = exact_rank_test(s[y[te] == 1], s[y[te] == 0])
        within[c] = {"n": int(te.sum()), "n_animal": int(y[te].sum()),
                     "n_nonanimal": int((1 - y[te]).sum()),
                     "auroc": float(roc_auc_score(y[te], s)),
                     "frac_correct": float((m.predict(X[te]) == y[te]).mean()),
                     "exact_p_predicted_direction": p_pred,
                     "exact_p_inverted_direction": p_inv,
                     "n_orderings": n_ord}
    print("\n=== arm 3, within a class carrying both kinds (class identity constant) ===")
    for c, v in within.items():
        if "skipped" in v:
            print(f"  {c:<32}n={v['n']:>3}  skipped: {v['skipped']}")
        else:
            print(f"  {c:<32}n={v['n']:>3}  {v['n_animal']} animal vs "
                  f"{v['n_nonanimal']} non-animal   AUROC {v['auroc']:.3f}  "
                  f"{v['frac_correct'] * 100:.0f}% correct")
            print(f"    exact p, predicted direction {v['exact_p_predicted_direction']:.3f}"
                  f" / inverted {v['exact_p_inverted_direction']:.3f}"
                  f"  over {v['n_orderings']} orderings")

    # ---- verdict, against the preregistered bounds -----------------------------
    if bal >= 0.75 and p < 0.05:
        verdict = ("SUPPORTED: target host survives leave-one-mechanism-class-out, "
                   f"balanced class accuracy {bal:.3f}, permutation p {p:.4f}")
    elif bal <= float(np.median(null)):
        verdict = ("REFUTED: balanced class accuracy "
                   f"{bal:.3f} is at or below the class-level permutation median "
                   f"{np.median(null):.3f}. The pooled 0.929 is not separable from "
                   "mechanism-class identity")
    else:
        verdict = (f"INCONCLUSIVE: balanced class accuracy {bal:.3f}, permutation p "
                   f"{p:.4f}, between the preregistered bounds")
    print(f"\nverdict: {verdict}")
    print(f"structural limit: non-animal target is carried by "
          f"{n_nonanimal_classes} mechanism class(es)")

    dest = V2 / "target_host_class_holdout.json"
    json.dump({"task": "target host from positives only, animal vs non-animal",
               "n": int(len(y)), "n_animal": int(y.sum()),
               "n_nonanimal": int((1 - y).sum()),
               "producer_kingdoms": dict(prod),
               "per_class": per_class, "single_target_classes": single,
               "mixed_classes": mixed,
               "n_nonanimal_single_target_classes": n_nonanimal_classes,
               "pooled_cv": {"auroc": auroc, "sd": sd, "published_03s": 0.9294},
               "class_holdout": {"per_class": detail,
                                 "balanced_class_accuracy": bal,
                                 "by_group": by_group,
                                 "perm_draws": int(len(null)),
                                 "perm_median": float(np.median(null)),
                                 "perm_p95": float(np.percentile(null, 95)),
                                 "perm_p": p,
                                 "perm_frac_at_1.000": frac_perfect,
                                 "perm_usable": bool(frac_perfect < 0.05)},
               "post_hoc_class_level_ordering": {
                   "disclosure": ("added after seeing the permutation null was "
                                  "uninformative; not preregistered, does not set "
                                  "the verdict"),
                   "auroc": cl_auroc, "exact_one_tailed_p": cl_p,
                   "n_orderings": n_orderings,
                   "nonanimal_mean_p": lo, "animal_mean_p": hi,
                   "animal_classes_below_highest_nonanimal": [str(c) for c in inside]},
               "within_class": within,
               "verdict": verdict},
              open(dest, "w"), indent=2)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
