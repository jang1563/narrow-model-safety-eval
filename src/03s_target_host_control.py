#!/usr/bin/env python3
"""
03s_target_host_control.py - the confound §2 does not control: what the protein acts on.

Why this exists
---------------
§2 measures three confounds and all three hold the protein's ORIGIN constant:
pathogen-matched (same organisms), localization (origin plus secretion), provenance
(lab strain versus pathogen, hazard label discarded). None of them holds constant
what the protein ACTS ON, and the panel is not uniform on that axis.

`data/annotations/target_host_v2.json` assigns a target per protein with a written
reason. Of 80 positives, 51 act on an animal host, 15 on a diffusing small molecule
and no host at all (14 beta-lactamases plus a teichoic-acid transferase annotated
for beta-lactam resistance), 5 on another bacterium (4 contact-dependent inhibition
systems plus colicin E2, a bacteriocin), 1 on plant cells (the Agrobacterium T-pilus
subunit), 4 on the producing organism itself, and 4 are mixed or unassigned.

The document never says which of these "hazardous" means. That is the same kind of
unstated scope as the beta-lactamase provenance gap in docs/DATA_CORRECTIONS.md:
not a wrong number, an undocumented definition.

🔑 The reason it matters is a rank pattern nobody had looked for. Ordering the eight
LOMO mechanism classes by recovery, the two whose target is not an animal are
exactly the bottom two: beta-lactamase 21% and contact-dependent inhibition 35%,
against 69-100% for the six animal-targeting classes. The exact probability of that
split by chance is 1/C(8,2) = 0.036. Class size does not explain it: clostridial
neurotoxin has the SMALLEST effective n of any class, 3, and is recovered at 100%.

⚠️ That observation is post hoc, on eight classes that are not independent draws,
and it is reported at that strength. This script measures the three things that can
be measured on the existing panel without new data.

Producer and target come apart, which is why both are recorded: six of the seven
ribosome-inactivating proteins are PLANT-produced and act on ANIMAL ribosomes.

Usage:
    python src/03s_target_host_control.py
"""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "results" / "v2"
SEEDS, FRAC, SPEC = range(30), 0.40, 0.95
NONANIMAL = {"bacteria", "none_small_molecule", "plant", "other_nonanimal"}


def cv_auroc(X, y, reps=10):
    out = []
    for seed in range(reps):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
            m.fit(X[tr], y[tr])
            out.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return float(np.mean(out)), float(np.std(out))


def main():
    P = np.load(V2 / "embeddings_positive_v2.npy")
    N = np.load(V2 / "embeddings_negative_v2.npy")
    man = json.load(open(V2 / "embedding_manifest_v2.json"))
    th = json.load(open(ROOT / "data/annotations/target_host_v2.json"))
    mech = json.load(open(ROOT / "data/annotations/mechanism_classes_v2.json"))
    tgt = {e["fasta_id"]: e["target_host"] for e in th["proteins"]}
    cls = {e["fasta_id"]: e["mechanism_class"] for e in mech["proteins"]}
    accs = [r["acc"] for r in man["positive_rows"]]
    t = np.array([tgt[a] for a in accs])
    pcls = np.array([cls[a] for a in accs])

    animal = t == "animal"
    nonani = np.isin(t, list(NONANIMAL))
    print(f"positives: {animal.sum()} animal-target, {nonani.sum()} non-animal-target, "
          f"{len(t) - animal.sum() - nonani.sum()} regulatory/mixed/unassigned")

    # 1. is target host legible at all? the analogue of §2's provenance control
    idx = np.where(animal | nonani)[0]
    a1, s1 = cv_auroc(P[idx], animal[idx].astype(float))
    print("\n1. TARGET-HOST LEGIBILITY, positives only, hazard held constant")
    print(f"   animal vs non-animal target: AUROC {a1:.3f} +- {s1:.3f}  (n={len(idx)})")
    print("   §2's provenance control, for comparison: 0.818")

    # 2. does the hazard signal depend on which target?
    print("\n2. HAZARD SEPARATION, by target, against the same 154 negatives")
    rows = {}
    for lab, mask in (("all 80 positives", np.ones(len(t), bool)),
                      ("animal-target only", animal),
                      ("non-animal-target only", nonani)):
        X = np.vstack([P[mask], N])
        y = np.r_[np.ones(mask.sum()), np.zeros(len(N))]
        m_, s_ = cv_auroc(X, y)
        rows[lab] = {"n_positive": int(mask.sum()), "auroc": m_, "sd": s_}
        print(f"   {lab:<26}n={int(mask.sum()):>3}  AUROC {m_:.3f} +- {s_:.3f}")

    # 2b. size-matched: is the non-animal gap just a smaller n?
    rng = np.random.default_rng(0)
    ai, ni = np.where(animal)[0], np.where(nonani)[0]
    subs = []
    for _ in range(30):
        pick = rng.choice(ai, len(ni), replace=False)
        X = np.vstack([P[pick], N])
        subs.append(cv_auroc(X, np.r_[np.ones(len(pick)), np.zeros(len(N))])[0])
    subs = np.array(subs)
    nonv = rows["non-animal-target only"]["auroc"]
    print(f"\n2b. SIZE MATCHED: animal subsampled to n={len(ni)}, 30 draws")
    print(f"    animal subsamples  AUROC {subs.mean():.3f} +- {subs.std():.3f}  "
          f"range [{subs.min():.3f}, {subs.max():.3f}]")
    print(f"    non-animal, n={len(ni)}   AUROC {nonv:.3f}  -> "
          f"{'below the whole animal range' if nonv < subs.min() else 'inside the animal range'}")
    sizematch = {"animal_subsample_mean": float(subs.mean()),
                 "animal_subsample_sd": float(subs.std()),
                 "animal_subsample_min": float(subs.min()),
                 "animal_subsample_max": float(subs.max()),
                 "nonanimal": nonv, "n": int(len(ni)),
                 "nonanimal_below_animal_range": bool(nonv < subs.min())}

    # 3. the rank pattern, stated at its strength
    lomo = json.load(open(V2 / "lomo_results.json"))["leave_one_mechanism_out"]
    cls_target = {}
    for C in lomo:
        if C == "virulence_associated_non_toxin":
            continue
        ts = set(t[pcls == C])
        cls_target[C] = "animal" if ts == {"animal"} else "non-animal"
    ranked = sorted(((C, lomo[C]["flagged_95_mean"], cls_target[C]) for C in cls_target),
                    key=lambda r: r[1])
    print("\n3. THE RANK PATTERN, eight mechanism classes, control excluded")
    print(f"   {'class':<32}{'recovery':>9}  target")
    for C, r, tt in ranked:
        print(f"   {C:<32}{r * 100:>8.0f}%  {tt}")
    bottom = [tt for _, _, tt in ranked[:2]]
    n_non = sum(1 for _, _, tt in ranked if tt == "non-animal")
    import math
    p_rank = 1 / math.comb(len(ranked), n_non)
    print(f"   non-animal classes in the bottom {n_non}: {bottom.count('non-animal')}/{n_non}, "
          f"exact chance probability 1/C({len(ranked)},{n_non}) = {p_rank:.3f}")
    print("   ⚠️ post hoc, eight non-independent classes; not a preregistered test")

    out = {"built": "2026-09-18",
           "counts": {k: int((t == k).sum()) for k in sorted(set(t))},
           "target_host_legibility": {"auroc": a1, "sd": s1, "n": int(len(idx)),
                                      "provenance_control_for_comparison": 0.818},
           "hazard_separation_by_target": rows,
           "size_matched": sizematch,
           "rank_pattern": {"ranked": [[C, r, tt] for C, r, tt in ranked],
                            "n_nonanimal": n_non,
                            "nonanimal_in_bottom": int(bottom.count("non-animal")),
                            "exact_p": p_rank, "post_hoc": True}}
    p = V2 / "target_host_control.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
