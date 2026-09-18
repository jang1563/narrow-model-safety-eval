#!/usr/bin/env python3
"""
03p_pretraining_exposure.py - does pretraining exposure predict recovery? Not measurably.

The question
------------
§9.3 records that leave-one-mechanism-out removes a class from the PROBE's training
set and not from the foundation model's, and that every class here sits in the
public databases ESM-2 was trained on. That is written as a caveat with no number.
This is the attempt to attach one WITHOUT needing new sequences, after the external
holdout route failed twice (see 03n).

If the caveat drives the results, families ESM-2 saw more of during pretraining
should be recovered better. UniRef50 cluster size is the available proxy for how
much of a family was in the pretraining corpus, since ESM-2 was trained on UniRef50
2021_04 and one cluster contributes roughly one representative plus its neighbours.

🔴 The pooled answer is significant, and it is pseudoreplicated
--------------------------------------------------------------
Pooling all 72 members gives Spearman rho -0.399, p 0.0005: MORE pretraining
exposure, WORSE recovery, which is the opposite of what the caveat predicts and
looks like a finding.

It does not survive the controls, and all three say the same thing:

    within class, recovery-varying classes only   mean rho -0.139, no class p < 0.05
    after removing class means (residuals)        rho -0.136, p 0.2554
    excluding beta-lactamase                      rho -0.199, p 0.1334
    at the class level, which is the real n       rho -0.331, p 0.3846, n = 9

The pooled correlation is carried by between-class differences, largely by
beta-lactamase's 14 members. Recovery is dominated by class structure, so the
effective sample size is the NINE classes, not the 72 members. This project has
already published one pseudoreplicated p-value across four surfaces
(docs/DATA_CORRECTIONS.md); this entry exists so the same mistake is not made twice
with the same shape.

What survives is descriptive, and it is one class, not a trend: beta-lactamase is
the most heavily represented family in the panel (median cluster 224, max 2535) and
the worst recovered (21%). The counterexample sits in the same table: T3SS effectors
have the largest median cluster of all (317) and are recovered at 80%.

Verdict: no measurable relationship between pretraining exposure and recovery at
the resolution nine classes can support. The caveat in §9.3 remains logically true
and empirically untested, and the honest statement of what would settle it is a
pretraining run with a family held out, which is out of scope here.

Usage:
    python src/03p_pretraining_exposure.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "results" / "v2"


def main():
    U = json.load(open(V2 / "uniref50_cluster_sizes.json"))
    lomo = json.load(open(V2 / "lomo_results.json"))["leave_one_mechanism_out"]
    name2acc = {v["name"]: k for k, v in U.items()}

    rows = []
    for C, r in lomo.items():
        per = np.array([s["probs"] for s in r["per_seed"]])
        thr = np.array([s["t95"] for s in r["per_seed"]])[:, None]
        rec = (per >= thr).mean(0)
        for m, v in zip(r["members"], rec):
            a = name2acc.get(m)
            sz = U.get(a, {}).get("size") if a else None
            if sz:
                rows.append({"member": m, "cls": C, "rec": float(v), "size": int(sz)})
    sz = np.array([r["size"] for r in rows])
    rc = np.array([r["rec"] for r in rows])
    print(f"members with a cluster size: {len(rows)}")

    pooled_rho, pooled_p = spearmanr(np.log10(sz), rc)
    caught = rc >= 0.5
    mw_u, mw_p = mannwhitneyu(sz[caught], sz[~caught])
    print(f"\npooled           rho {pooled_rho:+.3f}  p {pooled_p:.4f}  n {len(rows)}")
    print(f"  caught {int(caught.sum())} median cluster {int(np.median(sz[caught]))}, "
          f"missed {int((~caught).sum())} median {int(np.median(sz[~caught]))}, "
          f"Mann-Whitney p {mw_p:.4f}")

    within = {}
    for C in sorted({r["cls"] for r in rows}):
        g = [r for r in rows if r["cls"] == C]
        y = np.array([r["rec"] for r in g])
        if len(g) < 4 or y.std() == 0:
            continue
        rho, p = spearmanr(np.log10([r["size"] for r in g]), y)
        within[C] = {"n": len(g), "rho": float(rho), "p": float(p)}
    print(f"\nwithin class, {len(within)} classes with recovery variance:")
    for C, v in within.items():
        print(f"  {C:<32}n {v['n']:>3}  rho {v['rho']:+.3f}  p {v['p']:.3f}")
    mean_within = float(np.mean([v["rho"] for v in within.values()]))
    print(f"  mean within-class rho {mean_within:+.3f}, "
          f"any p < 0.05: {any(v['p'] < 0.05 for v in within.values())}")

    mu_r, mu_s = defaultdict(list), defaultdict(list)
    for r in rows:
        mu_r[r["cls"]].append(r["rec"])
        mu_s[r["cls"]].append(np.log10(r["size"]))
    mr = {k: np.mean(v) for k, v in mu_r.items()}
    ms = {k: np.mean(v) for k, v in mu_s.items()}
    res_rho, res_p = spearmanr([np.log10(r["size"]) - ms[r["cls"]] for r in rows],
                               [r["rec"] - mr[r["cls"]] for r in rows])
    g = [r for r in rows if r["cls"] != "beta_lactamase"]
    nobl_rho, nobl_p = spearmanr(np.log10([r["size"] for r in g]), [r["rec"] for r in g])

    cl = {C: {"recovery": lomo[C]["flagged_95_mean"],
              "median_cluster": float(np.median([r["size"] for r in rows if r["cls"] == C])),
              "n": sum(1 for r in rows if r["cls"] == C)}
          for C in sorted({r["cls"] for r in rows})}
    cls_rho, cls_p = spearmanr(np.log10([v["median_cluster"] for v in cl.values()]),
                               [v["recovery"] for v in cl.values()])

    print(f"\nresiduals after class means   rho {res_rho:+.3f}  p {res_p:.4f}")
    print(f"excluding beta-lactamase      rho {nobl_rho:+.3f}  p {nobl_p:.4f}  n {len(g)}")
    print(f"class level, the real n       rho {cls_rho:+.3f}  p {cls_p:.4f}  n {len(cl)}")

    print(f"\n{'class':<32}{'n':>3}{'recovery':>10}{'median cluster':>16}")
    print("-" * 62)
    for C, v in sorted(cl.items(), key=lambda kv: -kv[1]["median_cluster"]):
        print(f"{C:<32}{v['n']:>3}{v['recovery'] * 100:>9.0f}%{int(v['median_cluster']):>15}")

    survives = res_p < 0.05 and nobl_p < 0.05 and cls_p < 0.05
    verdict = ("relationship survives the class controls" if survives else
               "NO measurable relationship: the pooled correlation is pseudoreplicated, "
               "carried by between-class differences; effective n is 9 classes")
    print(f"\nverdict: {verdict}")

    out = {"proxy": "UniRef50 cluster size (ESM-2 pretrained on UniRef50 2021_04)",
           "n_members": len(rows),
           "pooled": {"rho": float(pooled_rho), "p": float(pooled_p)},
           "caught_median_cluster": int(np.median(sz[caught])),
           "missed_median_cluster": int(np.median(sz[~caught])),
           "mannwhitney_p": float(mw_p),
           "within_class": within, "mean_within_rho": mean_within,
           "residual": {"rho": float(res_rho), "p": float(res_p)},
           "excluding_beta_lactamase": {"rho": float(nobl_rho), "p": float(nobl_p)},
           "class_level": {"rho": float(cls_rho), "p": float(cls_p), "n": len(cl)},
           "classes": cl, "survives_controls": bool(survives), "verdict": verdict}
    p = V2 / "pretraining_exposure.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
