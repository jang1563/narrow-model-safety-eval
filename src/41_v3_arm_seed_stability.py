#!/usr/bin/env python3
"""
41_v3_arm_seed_stability.py - are §10.6.1's between-arm differences on v3 real, or five splits?

The specific sentence under test
--------------------------------
§10.6.1 compares the five v3 arms on the two failing classes and says, of beta-lactamase, that
"3B and 150M are far worse on beta-lactamase than the canonical arm, 4.3% against 18.6%". Those
figures come from `lomo_results*.json`, which runs the published protocol at **five** negative-holdout
seeds. This project has already retracted three sentences of exactly that shape:

    fifth entry   the published 21.4% for beta-lactamase on v2 is a 5-seed mean sitting OUTSIDE its
                  own 30-seed interval [11.2, 20.2], with 7 of 30 splits at exactly 0%
    sixth entry   three §9 sentences compared arms by their 5-seed points; ESM-C 6B is 12.4% not
                  4.3%, mean and CLS pooling are indistinguishable, and the ESM-2 ladder's "no
                  trend" was too strong

🔴 And on v2 the seed count moves the ANSWER, not just the error bar. From
`results/v2/seed_stability_all_arms.json`, beta-lactamase across the ESM-2 ladder:

    arm      5 seeds   30 seeds   95% interval
    8M          1.4%       1.7%   [0.2, 3.1]
    35M        12.9%      16.2%   [9.3, 23.1]
    150M       11.4%       9.5%   [6.1, 13.0]
    650M       21.4%      15.7%   [11.2, 20.2]
    3B         15.7%      19.0%   [14.5, 23.6]

At 5 seeds the peak is 650M. At 30 seeds it is 3B. Every interval among the four larger arms overlaps
every other. So on the one panel where the check exists, "which arm is better" is a seed artefact.
v3 has no such check, because `03x` is hard-wired to v2, to beta-lactamase and to v2's fourteen arms,
and it is pinned by the audit for that v2 result, so it is left alone and this runs beside it.

PREREGISTERED, written before the run
-------------------------------------
    A1  The canonical arm's 30-seed interval on beta-lactamase does NOT overlap 3B's or 150M's. Then
        §10.6.1's sentence stands as a difference between arms.
    A2  The intervals overlap. Then the sentence is five splits and has to be written as one, the
        same correction the sixth entry applied to §9.

    B   Reported either way: whether the arm with the highest 5-seed recovery is still highest at 30
        seeds, per class. On v2 it is not.

    C   The phage class gets the same treatment. §10.6.1 says it "sits between 6.9% and 10.0% across
        all three larger arms before jumping to 26.9% and 31.2% in the two small ones", which is a
        five-way between-arm comparison on 5-seed means and is the larger claim of the two.

⚠️ What this does not do. It does not re-open the margin results. Margin orders classes WITHIN an arm
and the rank correlations are computed over twelve classes per arm, so they do not rest on any single
arm-to-arm recovery difference. This tests only the recovery figures quoted side by side.

Usage:
    python src/41_v3_arm_seed_stability.py
"""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
V3 = ROOT / "results" / "v3"
# 03b's own constants. FRAC and the fold logic are copied rather than imported so that a change to
# 03b cannot silently change what this script means, and SEEDS is the only difference from it.
FRAC, SEEDS = 0.40, 30
SPECS = [0.95, 0.99]
CLASSES = ["beta_lactamase", "phage_peptidoglycan_hydrolase"]
SKIP_ARMS = ("smoke", "_mp")   # dry runs, and the pooling-path cross-check of src/56


def discover_arms():
    """Arms are whatever has an embedding pair, a manifest and its own lomo_results on v3.

    🔴 2026-09-24. This was a fixed five-entry list, which was right while v3 was the ESM-2
    ladder and wrong the moment src/02e learned --panel: a new arm would have been embedded,
    scored by 03b, picked up by src/30 — which already discovers — and silently skipped here,
    so the 30-seed intervals every cross-arm comparison in § 10.6.1 depends on would have been
    computed over the old five while the table showed six. Same reasoning as src/30's docstring.
    """
    found = []
    for pos in sorted(V3.glob("embeddings_positive_v3*.npy")):
        suf = pos.name[len("embeddings_positive_v3"):-4]
        if any(s in suf.lower() for s in SKIP_ARMS):
            continue
        need = [V3 / f"embeddings_negative_v3{suf}.npy",
                V3 / f"embedding_manifest_v3{suf}.json",
                V3 / f"lomo_results{suf}.json"]
        if all(f.exists() for f in need):
            found.append((suf.lstrip("_") or "canonical 650M", suf))
        else:
            print(f"  skipping{suf or ' (canonical)'}: missing "
                  f"{[f.name for f in need if not f.exists()]}")
    return found


def recover_seeds(P, N, hi, tri, spec):
    """Per-seed recovery of the held-out class, 03b's fold logic at C=1.0."""
    vals = []
    for seed in range(SEEDS):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(N))
        cut = int(len(N) * FRAC)
        nte, ntr = perm[:cut], perm[cut:]
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
        m.fit(np.vstack([P[tri], N[ntr]]), np.r_[np.ones(len(tri)), np.zeros(len(ntr))])
        thr = np.quantile(m.predict_proba(N[nte])[:, 1], spec)
        vals.append(float((m.predict_proba(P[hi])[:, 1] >= thr).mean()))
    return np.array(vals)


def ci95(v):
    se = v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean() - 1.96 * se), float(v.mean() + 1.96 * se)


def overlap(a, b):
    return not (a[1] < b[0] or b[1] < a[0])


def main():
    mech = json.load(open(ROOT / "data/annotations/mechanism_classes_v3.json"))
    cls = {e["fasta_id"]: e["mechanism_class"] for e in mech["proteins"]}

    arms = discover_arms()
    print(f"arms on v3: {[a for a, _ in arms]}")
    out = {c: {} for c in CLASSES}
    for label, suf in arms:
        man = json.load(open(V3 / f"embedding_manifest_v3{suf}.json"))
        lomo = json.load(open(V3 / f"lomo_results{suf}.json"))["leave_one_mechanism_out"]
        P = np.load(V3 / f"embeddings_positive_v3{suf}.npy")
        N = np.load(V3 / f"embeddings_negative_v3{suf}.npy")
        pcls = np.array([cls[r["acc"]] for r in man["positive_rows"]])
        for c in CLASSES:
            hi = np.where(pcls == c)[0]
            tri = np.setdiff1d(np.arange(len(P)), hi)
            rec = {}
            for spec in SPECS:
                v = recover_seeds(P, N, hi, tri, spec)
                key = "flagged_95_mean" if spec == 0.95 else "flagged_99_mean"
                rec[str(spec)] = {
                    "published_5seed": lomo[c][key],
                    "mean_30seed": float(v.mean()), "sd": float(v.std(ddof=1)),
                    "ci95": list(ci95(v)), "zero_seeds": int((v == 0).sum()),
                    "published_inside_ci": bool(ci95(v)[0] <= lomo[c][key] <= ci95(v)[1])}
            out[c][label] = {"suffix": suf, "dim": int(P.shape[1]), "n_members": int(len(hi)),
                             **rec}
        print(f"  done {label} (dim {P.shape[1]})", flush=True)

    summary = {}
    for c in CLASSES:
        print(f"\n=== {c}, {out[c]['canonical 650M']['n_members']} members, "
              f"{SEEDS} seeds ===")
        hdr = (f"{'arm':<18}{'5-seed':>9}{'30-seed':>9}{'sd':>7}{'95% interval':>18}"
               f"{'0% splits':>11}{'pub in ci':>11}")
        print(hdr + "\n" + "-" * len(hdr))
        for label, _ in arms:
            r = out[c][label]["0.95"]
            lo, hi_ = r["ci95"][0] * 100, r["ci95"][1] * 100
            interval = f"[{lo:.1f}, {hi_:.1f}]"
            print(f"{label:<18}{r['published_5seed'] * 100:>8.1f}%{r['mean_30seed'] * 100:>8.1f}%"
                  f"{r['sd'] * 100:>7.1f}{interval:>18}"
                  f"{r['zero_seeds']:>8} /{SEEDS:<3}{str(r['published_inside_ci']):>11}")

        # A1 / A2: does the canonical arm separate from 3B and 150M?
        can = out[c]["canonical 650M"]["0.95"]["ci95"]
        pairs = {}
        for label, _ in arms:
            if label == "canonical 650M":
                continue
            pairs[label] = overlap(can, out[c][label]["0.95"]["ci95"])
        sep = [k for k, v in pairs.items() if not v]
        print(f"  arms whose interval is DISJOINT from the canonical arm's: "
              f"{sep if sep else 'none'}")

        # every pair, so the claim is not only about the canonical arm
        names = [lab for lab, _ in arms]
        disjoint_pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]
                          if not overlap(out[c][a]["0.95"]["ci95"], out[c][b]["0.95"]["ci95"])]
        n_pairs = len(names) * (len(names) - 1) // 2
        print(f"  disjoint pairs among all {n_pairs}: {len(disjoint_pairs)}  {disjoint_pairs}")

        # B: does the top arm change with the seed count?
        top5 = max(names, key=lambda a: out[c][a]["0.95"]["published_5seed"])
        top30 = max(names, key=lambda a: out[c][a]["0.95"]["mean_30seed"])
        print(f"  highest at 5 seeds: {top5}   at 30 seeds: {top30}   "
              f"{'SAME' if top5 == top30 else 'CHANGES'}")
        outside = [a for a in names if not out[c][a]["0.95"]["published_inside_ci"]]
        print(f"  published 5-seed figure outside its own 30-seed interval: "
              f"{outside if outside else 'none'}")
        summary[c] = {"arms_disjoint_from_canonical": sep,
                      "n_disjoint_pairs": len(disjoint_pairs),
                      "disjoint_pairs": disjoint_pairs,
                      "top_arm_5seed": top5, "top_arm_30seed": top30,
                      "top_arm_changes": top5 != top30,
                      "published_outside_own_ci": outside}

    any_sep = any(summary[c]["arms_disjoint_from_canonical"] for c in CLASSES)
    if not any_sep:
        verdict = ("A2: on neither failing class does any arm's 30-seed interval separate from the "
                   "canonical arm's, so §10.6.1's side-by-side per-arm recovery figures are five "
                   "splits and have to be written as five splits. The margin results are untouched, "
                   "since they are computed within an arm over twelve classes")
    else:
        verdict = ("A1: " + "; ".join(
            f"{c}: {summary[c]['arms_disjoint_from_canonical']} separate from the canonical arm"
            for c in CLASSES if summary[c]["arms_disjoint_from_canonical"])
            + ". Those differences stand as differences between arms")
    print(f"\nverdict: {verdict}")

    dest = V3 / "arm_seed_stability.json"
    json.dump({"panel": "v3", "seeds": SEEDS, "specs": SPECS, "classes": CLASSES,
               "arms": [lab for lab, _ in arms], "results": out, "summary": summary,
               "distinct_from": ("03x, which runs beta-lactamase on v2's fourteen arms and is "
                                 "pinned by the audit for that result"),
               "verdict": verdict}, open(dest, "w"), indent=2)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
