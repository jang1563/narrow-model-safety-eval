#!/usr/bin/env python3
"""
30_margin_across_arms.py - is the class-level margin mechanism a property of the geometry, or
                           of one representation?

What is at stake
----------------
§10.4 says the two unreachable classes are the two lowest-margin classes and that margin tracks
recovery at rho +0.894. §10.5 says that ordering transfers to mechanisms the model was not fit
on. Both are computed on **one arm**, ESM-2 650M mean-pooled, and both are phrased as claims
about geometry: a class fails when its members sit closer to a benign protein than to any hazard
the probe trained on.

A claim about geometry has to survive a change of representation. If margin only locates the
failure in ESM-2 650M mean, then §10.4 is a fact about that model and the geometric language is
overreach. Panel v2 has all fourteen arms embedded and scored, so this is answerable from cache.

⚠️ v2, not v3. v3 has only the canonical arm embedded, so v2's **single** failure class is what
is being located here rather than v3's pair. That makes the per-arm test weaker (one class out
of nine, chance 1/9) and it is the only version available without re-embedding thirteen arms.

⚠️ §9.1's related result is about a different quantity and should not be read as this one. The
audit already pins "the internal margin effect holds on mean-pooled arms, not on CLS or max",
which is `03k_margin_holdout`: whether low-margin MEMBERS are recovered less, 12 of 14 arms over
a 25-point threshold. This asks whether class-mean margin ranks the failing CLASS lowest.

PREREGISTERED, written before the run
-------------------------------------
    P1  a majority of the fourteen arms rank `beta_lactamase` as the lowest-margin class of the
        nine. Chance per arm is 1/9.

    P2  a majority of arms have a positive margin-against-recovery rank correlation with
        permutation p < 0.05.

    SUPPORTED        both hold. The mechanism is representation-general and §10.4's geometric
                     phrasing is earned.
    ARM-SPECIFIC     neither holds outside the mean-pooled ESM-2 arms. §10.4 must be restated
                     as a property of those representations.
    PARTIAL          one holds, reported as such with the per-arm table.

⚠️ The two arms §9 already singles out, CLS and max pooling, are expected to behave differently
and are reported separately rather than excluded, because excluding them after seeing the result
would be the failure this file exists to avoid.

Usage:
    python src/30_margin_across_arms.py
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "results" / "v2"
PERMS = 20000
FAILURE = "beta_lactamase"
ARMS = ["", "_esm2_8M", "_esm2_35M", "_esm2_150M", "_esm2_3B", "_esm2_650M_max",
        "_esm2_650M_cls", "_esmc_300M", "_esmc_600M", "_esmc_6B", "_esm3_1_4B",
        "_prott5_xl", "_saprot_650M", "_esm2_650M_mean"]
NON_MEAN = {"_esm2_650M_max", "_esm2_650M_cls"}


def cos(A, B):
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    return A @ B.T


def _spearman(x, y):
    def rank(v):
        v = np.asarray(v, float)
        o = v.argsort()
        r = np.empty(len(v), float)
        r[o] = np.arange(1, len(v) + 1)
        for u in np.unique(v):
            m = v == u
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def main():
    mech = json.load(open(ROOT / "data/annotations/mechanism_classes_v2.json"))
    cls = {e["fasta_id"]: e["mechanism_class"] for e in mech["proteins"]}
    rng = np.random.default_rng(0)

    rows = {}
    hdr = (f"{'arm':<22}{'dim':>6}{'rho':>8}{'perm p':>9}{'lowest-margin class':>32}"
           f"{'hit':>5}")
    print(hdr)
    print("-" * len(hdr))
    for suf in ARMS:
        P = np.load(V2 / f"embeddings_positive_v2{suf}.npy")
        N = np.load(V2 / f"embeddings_negative_v2{suf}.npy")
        man = json.load(open(V2 / f"embedding_manifest_v2{suf}.json"))
        lomo = json.load(open(V2 / f"lomo_results{suf}.json"))["leave_one_mechanism_out"]
        pcls = np.array([cls[r["acc"]] for r in man["positive_rows"]])
        simPP, simPN = cos(P, P), cos(P, N)
        np.fill_diagonal(simPP, -np.inf)

        classes = [c for c in sorted(set(pcls)) if c in lomo]
        m, rec = [], []
        for c in classes:
            idx = np.where(pcls == c)[0]
            other = np.setdiff1d(np.arange(len(P)), idx)
            m.append(float((simPP[np.ix_(idx, other)].max(1) - simPN[idx].max(1)).mean()))
            rec.append(lomo[c]["flagged_95_mean"])
        rho = _spearman(m, rec)
        null = np.array([_spearman(m, rng.permutation(rec)) for _ in range(PERMS)])
        p = float((null >= rho).mean())
        lowest = classes[int(np.argmin(m))]
        hit = lowest == FAILURE
        name = suf.lstrip("_") or "canonical"
        rows[name] = {"suffix": suf, "dim": int(P.shape[1]), "rho": rho, "perm_p": p,
                      "lowest_margin_class": lowest, "locates_failure": bool(hit),
                      "n_classes": len(classes), "mean_pooled": suf not in NON_MEAN,
                      "failure_margin": m[classes.index(FAILURE)]}
        print(f"{name:<22}{P.shape[1]:>6}{rho:>+8.3f}{p:>9.4f}{lowest:>32}"
              f"{('yes' if hit else 'no'):>5}", flush=True)

    hits = [k for k, v in rows.items() if v["locates_failure"]]
    sig = [k for k, v in rows.items() if v["rho"] > 0 and v["perm_p"] < 0.05]
    mean_arms = [k for k, v in rows.items() if v["mean_pooled"]]
    non_mean = [k for k, v in rows.items() if not v["mean_pooled"]]
    n = len(rows)

    print(f"\nP1  arms ranking {FAILURE} lowest: {len(hits)}/{n}  (chance 1/9 per arm)")
    print(f"P2  arms with a positive rho at p<0.05: {len(sig)}/{n}")
    print(f"    mean-pooled arms: {sum(rows[k]['locates_failure'] for k in mean_arms)}"
          f"/{len(mean_arms)} locate it, "
          f"{sum(rows[k]['rho'] > 0 and rows[k]['perm_p'] < 0.05 for k in mean_arms)}"
          f"/{len(mean_arms)} significant")
    print(f"    CLS and max pooling: "
          f"{sum(rows[k]['locates_failure'] for k in non_mean)}/{len(non_mean)} locate it, "
          f"{[(k, round(rows[k]['rho'], 3), round(rows[k]['perm_p'], 4)) for k in non_mean]}")
    neg = [k for k, v in rows.items() if v["failure_margin"] < 0]
    print(f"    arms where {FAILURE}'s margin is negative: {len(neg)}/{n}")

    p1, p2 = len(hits) > n / 2, len(sig) > n / 2
    if p1 and p2:
        verdict = (f"SUPPORTED: {len(hits)}/{n} arms rank {FAILURE} as the lowest-margin class "
                   f"and {len(sig)}/{n} have a significant positive rank correlation, so the "
                   f"mechanism is representation-general rather than a property of one model")
    elif p1 or p2:
        verdict = (f"PARTIAL: {len(hits)}/{n} arms locate the failure and {len(sig)}/{n} have a "
                   f"significant rho, so one preregistered majority holds and the other does not")
    else:
        verdict = (f"ARM-SPECIFIC: only {len(hits)}/{n} arms locate the failure and {len(sig)}/{n} "
                   f"have a significant rho. §10.4 is a property of particular representations "
                   f"and its geometric phrasing has to be narrowed")
    print(f"\nverdict: {verdict}")

    dest = V2 / "margin_across_arms.json"
    json.dump({"panel": "v2", "failure_class": FAILURE, "perms": PERMS,
               "note": ("v2 has one failure class, so per-arm chance is 1/9. v3's pair cannot "
                        "be used here: only the canonical arm is embedded for v3"),
               "distinct_from": ("03k_margin_holdout, which tests whether low-margin MEMBERS "
                                 "are recovered less and is already pinned across 14 arms"),
               "arms": rows,
               "P1": {"arms_locating_failure": sorted(hits), "n_arms": n,
                      "majority": bool(p1)},
               "P2": {"arms_significant": sorted(sig), "majority": bool(p2)},
               "arms_with_negative_failure_margin": sorted(neg),
               "verdict": verdict}, open(dest, "w"), indent=2)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
