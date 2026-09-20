#!/usr/bin/env python3
"""
03x_seed_stability_all_arms.py - does the beta-lactamase column of §9 survive 30 seeds?

Why this exists
---------------
`src/03v_lomo_seed_stability.py` found that `03b`'s five negative-holdout seeds are enough
for eight of nine classes on ESM-2 650M and not enough for beta-lactamase, whose published
21.4% sits outside its own 30-seed interval [11.2, 20.2]. DATA_CORRECTIONS left the
fourteen model arms and the @99 column as a standing caveat rather than a correction.

That caveat covers a headline. §9 states that **ESM-C 600M recovers 51.4% of
beta-lactamase, above alignment's 29.5%, and is the only arm that clears alignment**, and
`src/22_claims_audit.py` pins it to three public documents. A claim that one arm out of
fourteen beats a baseline is exactly the shape of claim a 5-seed mean can invent. If
ESM-C 600M's interval overlaps alignment, or if some other arm's interval reaches it, the
sentence is wrong as written.

So this re-runs the beta-lactamase hold-out on **every cached arm at 30 seeds**, at both
operating points, with a bootstrap-free paired interval from the seed distribution itself.

What it does not do
-------------------
⚠️ Only beta-lactamase, and only its LOMO recovery. The other eight classes are checked at
30 seeds on ESM-2 650M by `03v` and are stable there, and re-running all nine on all
fourteen arms buys resolution the class-level n cannot use. If a future claim leans on
another class in another arm, it needs its own run.

⚠️ Alignment's 29.5% is a fixed reference from `results/v2/alignment_baseline.json`, not a
seeded quantity, so the comparison is one interval against a point.

Usage:
    python src/03x_seed_stability_all_arms.py
"""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "results" / "v2"
FRAC, SEEDS = 0.40, 30
SPECS = [0.95, 0.99]
CLASS = "beta_lactamase"
# The 14 arms of the §9 table. The canonical run has no suffix; esm2_650M_mean is its
# duplicate under an explicit name, kept because §9's audit checks the two agree.
ARMS = ["", "_esm2_8M", "_esm2_35M", "_esm2_150M", "_esm2_3B", "_esm2_650M_max",
        "_esm2_650M_cls", "_esmc_300M", "_esmc_600M", "_esmc_6B", "_esm3_1_4B",
        "_prott5_xl", "_saprot_650M", "_esm2_650M_mean"]


def recover_seeds(P, N, hi, tri, spec):
    """Per-seed recovery of the held-out class, 03b's fold logic at C=1.0."""
    vals = []
    for seed in range(SEEDS):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(N))
        cut = int(len(N) * FRAC)
        nte, ntr = perm[:cut], perm[cut:]
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
        m.fit(np.vstack([P[tri], N[ntr]]),
              np.r_[np.ones(len(tri)), np.zeros(len(ntr))])
        thr = np.quantile(m.predict_proba(N[nte])[:, 1], spec)
        vals.append(float((m.predict_proba(P[hi])[:, 1] >= thr).mean()))
    return np.array(vals)


def ci95(v):
    se = v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean() - 1.96 * se), float(v.mean() + 1.96 * se)


def _spearman_rho(x, y):
    """Rank correlation without scipy. The release-surface CI job installs numpy only,
    and importing scipy into an audited path has already broken CI once: the audit did it
    for one Spearman call, passed locally and failed with ModuleNotFoundError, fixed in
    405d2a7. Written up in docs/DATA_CORRECTIONS.md, 2026-09-18 fourth entry, which was
    itself two days late and empty while this comment already cited it."""
    def rank(v):
        v = np.asarray(v, float)
        order = v.argsort()
        r = np.empty(len(v), float)
        r[order] = np.arange(1, len(v) + 1)
        for u in np.unique(v):
            m = v == u
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def overlap(a, b):
    """Do two 95% intervals overlap? Used instead of comparing point estimates, because
    three of §9's sentences compare 5-seed points whose intervals turn out to cross."""
    return not (a[1] < b[0] or b[1] < a[0])


def main():
    mech = json.load(open(ROOT / "data/annotations/mechanism_classes_v2.json"))
    cls = {e["fasta_id"]: e["mechanism_class"] for e in mech["proteins"]}
    align = json.load(open(V2 / "alignment_baseline.json"))["classes"][CLASS]
    align_rec = align["alignment_recovery"]
    print(f"alignment recovery for {CLASS}: {align_rec * 100:.1f}%  (fixed reference)\n")

    out, missing = {}, []
    hdr = (f"{'arm':<22}{'pub@95':>8}{'30s@95':>8}{'sd':>6}{'ci95@95':>16}"
           f"{'>align?':>9}{'pub@99':>8}{'30s@99':>8}")
    print(hdr)
    print("-" * len(hdr))
    for suf in ARMS:
        pf, nf = (V2 / f"embeddings_positive_v2{suf}.npy",
                  V2 / f"embeddings_negative_v2{suf}.npy")
        mf, lf = (V2 / f"embedding_manifest_v2{suf}.json",
                  V2 / f"lomo_results{suf}.json")
        if not all(p.exists() for p in (pf, nf, mf, lf)):
            missing.append(suf or "canonical")
            continue
        P, N = np.load(pf), np.load(nf)
        man = json.load(open(mf))
        pcls = np.array([cls[r["acc"]] for r in man["positive_rows"]])
        lomo = json.load(open(lf))["leave_one_mechanism_out"][CLASS]
        hi = np.where(pcls == CLASS)[0]
        tri = np.setdiff1d(np.arange(len(pcls)), hi)

        rec = {}
        for spec in SPECS:
            v = recover_seeds(P, N, hi, tri, spec)
            lo, hh = ci95(v)
            key = "flagged_95_mean" if spec == 0.95 else "flagged_99_mean"
            rec[spec] = {"published_5seed": lomo[key], "mean_30seed": float(v.mean()),
                         "sd": float(v.std(ddof=1)), "ci95": [lo, hh],
                         "zero_seeds": int((v == 0).sum()),
                         "published_inside_ci": bool(lo - 1e-6 <= lomo[key] <= hh + 1e-6)}
        a = rec[0.95]
        above = ("yes" if a["ci95"][0] > align_rec else
                 "overlap" if a["ci95"][1] > align_rec else "no")
        rec["beats_alignment_at_95"] = above
        out[suf or "canonical"] = rec
        print(f"{(suf or 'canonical'):<22}{rec[0.95]['published_5seed'] * 100:>7.1f}%"
              f"{a['mean_30seed'] * 100:>7.1f}%{a['sd'] * 100:>6.1f}"
              f"   [{a['ci95'][0] * 100:>5.1f},{a['ci95'][1] * 100:>5.1f}]{above:>9}"
              f"{rec[0.99]['published_5seed'] * 100:>7.1f}%"
              f"{rec[0.99]['mean_30seed'] * 100:>7.1f}%", flush=True)

    # ---- what the numbers do to the published sentences -------------------------
    clears = [a for a, r in out.items() if r["beats_alignment_at_95"] == "yes"]
    overlaps = [a for a, r in out.items() if r["beats_alignment_at_95"] == "overlap"]
    drifted = [a for a, r in out.items() if not r[0.95]["published_inside_ci"]]
    esmc = out.get("_esmc_600M", {})

    print(f"\narms whose 95% CI lies entirely above alignment: {clears or 'none'}")
    print(f"arms whose CI overlaps alignment: {overlaps or 'none'}")
    print(f"arms whose published @95 value falls outside its own 30-seed CI: "
          f"{drifted or 'none'}")
    if missing:
        print(f"arms skipped for missing artifacts: {missing}")

    if esmc:
        e = esmc[0.95]
        verdict = (
            "HOLDS: ESM-C 600M stays above alignment at 30 seeds and is still the only "
            f"arm whose interval clears it ({e['mean_30seed'] * 100:.1f}%, "
            f"CI [{e['ci95'][0] * 100:.1f}, {e['ci95'][1] * 100:.1f}])"
            if clears == ["_esmc_600M"] and not overlaps else
            "NEEDS REWORDING: the 'only arm above alignment' sentence does not survive "
            f"30 seeds. clears={clears}, overlaps={overlaps}")
    else:
        verdict = "ESM-C 600M artifacts absent, nothing to say about the headline"
    print(f"\nverdict: {verdict}")

    # ---- the three §9 sentences that compare arms by their 5-seed points -----------
    def at95(a):
        return out[a][0.95]

    checks = {}

    # 1. "Scale is not the fix": the ESM-2 mean-pool ladder, 8M to 3B.
    ladder = ["_esm2_8M", "_esm2_35M", "_esm2_150M", "canonical", "_esm2_3B"]
    params = [8, 35, 150, 650, 3000]
    if all(a in out for a in ladder):
        vals = [at95(a)["mean_30seed"] for a in ladder]
        rho = _spearman_rho(params, vals)
        checks["esm2_ladder"] = {
            "arms": ladder, "params_M": params,
            "published_5seed": [at95(a)["published_5seed"] for a in ladder],
            "mean_30seed": vals, "spearman_rho_vs_params": rho,
            "note": ("a rank correlation on five points is not a trend test, it is "
                     "reported so the ladder's shape is visible at 30 seeds")}
        print("\nESM-2 ladder at 30 seeds: "
              + ", ".join(f"{a.replace('_esm2_', '').replace('canonical', '650M')} "
                          f"{v * 100:.1f}%" for a, v in zip(ladder, vals))
              + f"   Spearman vs parameters {rho:+.2f}")

    # 2. "6B recovers less than 300M": compare intervals, not points.
    if "_esmc_300M" in out and "_esmc_6B" in out and "_esmc_600M" in out:
        c3, c6, cb = at95("_esmc_300M"), at95("_esmc_6B"), at95("_esmc_600M")
        checks["esmc_trio"] = {
            "300M": c3["mean_30seed"], "600M": cb["mean_30seed"],
            "6B": c6["mean_30seed"],
            "6B_below_300M_point": c6["mean_30seed"] < c3["mean_30seed"],
            "6B_vs_300M_intervals_overlap": overlap(c6["ci95"], c3["ci95"]),
            "6B_vs_600M_intervals_overlap": overlap(c6["ci95"], cb["ci95"]),
            "600M_over_6B_ratio": (cb["mean_30seed"] / c6["mean_30seed"]
                                   if c6["mean_30seed"] else None)}
        print(f"ESM-C trio at 30 seeds: 300M {c3['mean_30seed'] * 100:.1f}%, "
              f"600M {cb['mean_30seed'] * 100:.1f}%, 6B {c6['mean_30seed'] * 100:.1f}%"
              f"   6B-vs-300M intervals overlap: "
              f"{checks['esmc_trio']['6B_vs_300M_intervals_overlap']}"
              f"   600M/6B ratio {checks['esmc_trio']['600M_over_6B_ratio']:.1f}x")

    # 3. "Pooling is not the fix": mean against CLS against max.
    if all(a in out for a in ("canonical", "_esm2_650M_cls", "_esm2_650M_max")):
        mn, cl, mx = (at95("canonical"), at95("_esm2_650M_cls"),
                      at95("_esm2_650M_max"))
        checks["pooling"] = {
            "mean": mn["mean_30seed"], "cls": cl["mean_30seed"],
            "max": mx["mean_30seed"],
            "cls_above_mean_point": cl["mean_30seed"] > mn["mean_30seed"],
            "cls_vs_mean_intervals_overlap": overlap(cl["ci95"], mn["ci95"]),
            "max_vs_mean_intervals_overlap": overlap(mx["ci95"], mn["ci95"])}
        print(f"pooling at 30 seeds: mean {mn['mean_30seed'] * 100:.1f}%, "
              f"CLS {cl['mean_30seed'] * 100:.1f}%, max {mx['mean_30seed'] * 100:.1f}%"
              f"   CLS-vs-mean overlap: "
              f"{checks['pooling']['cls_vs_mean_intervals_overlap']}")

    dest = V2 / "seed_stability_all_arms.json"
    json.dump({"class": CLASS, "seeds": SEEDS, "specs": SPECS,
               "protocol": "03b fold logic at C=1.0, 40% negative holdout",
               "alignment_recovery": align_rec,
               "arms": {a: {str(k): v for k, v in r.items()} for a, r in out.items()},
               "arms_clearing_alignment": clears, "arms_overlapping": overlaps,
               "published_outside_own_ci": drifted, "skipped": missing,
               "sentence_checks": checks,
               "ci_note": ("intervals are normal-approximation on a bounded quantity, so "
                           "a lower bound can print below zero; read those as truncated "
                           "at zero"),
               "verdict": verdict},
              open(dest, "w"), indent=2)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
