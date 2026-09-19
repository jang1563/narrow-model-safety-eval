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

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES_ROOT = ROOT / "results"
PERMS = 20000
# Which classes count as failures is taken from the panel's own recovery, not from margin,
# for the reason in DATA_CORRECTIONS' seventh entry: selecting the test set with the predictor
# under test is circular. v2 has one class below FAIL_AT, v3 has two, so k differs by panel and
# the per-arm chance of hitting the bottom-k by accident is reported rather than assumed.
FAIL_AT = 0.25
CONTROL_CLASS = "virulence_associated_non_toxin"
NON_MEAN = {"_esm2_650M_max", "_esm2_650M_cls"}


# Dry-run artifacts live beside the real ones and must not be counted as model arms. The
# smoke arm is a 150M sanity run; globbing for arms picked it up and turned the published
# 12-of-14 into 12-of-15 until it was excluded, which the claims audit caught.
SKIP_ARMS = ("smoke",)


def discover_arms(res, pv):
    """Arms are whatever has BOTH an embedding pair and its own lomo_results on this panel,
    excluding dry-run artifacts. A fixed list was fine while only v2 existed; v3 is embedded
    incrementally, so the list is read off the filesystem and what is missing is printed
    rather than silently skipped."""
    found, missing = [], []
    for pos in sorted(res.glob(f"embeddings_positive_{pv}*.npy")):
        suf = pos.name[len(f"embeddings_positive_{pv}"):-4]
        if any(s in suf.lower() for s in SKIP_ARMS):
            continue
        need = [res / f"embeddings_negative_{pv}{suf}.npy",
                res / f"embedding_manifest_{pv}{suf}.json",
                res / f"lomo_results{suf}.json"]
        (found if all(x.exists() for x in need) else missing).append(suf)
    return found, missing


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="v2", choices=["v2", "v3"])
    pv = ap.parse_args().panel
    RES = RES_ROOT / pv
    mech = json.load(open(ROOT / f"data/annotations/mechanism_classes_{pv}.json"))
    cls = {e["fasta_id"]: e["mechanism_class"] for e in mech["proteins"]}
    rng = np.random.default_rng(0)

    ARMS, missing = discover_arms(RES, pv)
    base = json.load(open(RES / "lomo_results.json"))["leave_one_mechanism_out"]
    failures = sorted((c for c in base
                       if base[c]["flagged_95_mean"] < FAIL_AT and c != CONTROL_CLASS),
                      key=lambda c: base[c]["flagged_95_mean"])
    k = len(failures)
    print(f"panel {pv}: {len(ARMS)} arms with a complete set, "
          f"{len(missing)} embedded but unscored {missing if missing else ''}")
    print(f"failure classes on the canonical arm (recovery < {FAIL_AT:.0%}): {failures}\n")

    rows = {}
    hdr = (f"{'arm':<22}{'dim':>6}{'rho':>8}{'perm p':>9}{'lowest-margin class':>32}"
           f"{'hit':>5}")
    print(hdr)
    print("-" * len(hdr))
    for suf in ARMS:
        P = np.load(RES / f"embeddings_positive_{pv}{suf}.npy")
        N = np.load(RES / f"embeddings_negative_{pv}{suf}.npy")
        man = json.load(open(RES / f"embedding_manifest_{pv}{suf}.json"))
        lomo = json.load(open(RES / f"lomo_results{suf}.json"))["leave_one_mechanism_out"]
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
        order = [classes[i] for i in np.argsort(m)]
        lowest = order[0]
        hit = set(order[:k]) == set(failures)
        name = suf.lstrip("_") or "canonical"
        rows[name] = {"suffix": suf, "dim": int(P.shape[1]), "rho": rho, "perm_p": p,
                      "lowest_margin_class": lowest, "locates_failure": bool(hit),
                      "bottom_k": order[:k],
                      "n_classes": len(classes), "mean_pooled": suf not in NON_MEAN,
                      "failure_margins": {c: m[classes.index(c)] for c in failures}}
        print(f"{name:<22}{P.shape[1]:>6}{rho:>+8.3f}{p:>9.4f}{lowest:>32}"
              f"{('yes' if hit else 'no'):>5}", flush=True)

    # 🔴 Loop variables are named `a` for arm throughout. A first version used `k`, which is
    # already the number of failure classes, so `rows[k]` indexed the arm dictionary with an
    # integer. It would have raised KeyError rather than producing a wrong number, but the
    # shadowing is the kind that does produce wrong numbers when the types happen to line up.
    hits = [a for a, v in rows.items() if v["locates_failure"]]
    sig = [a for a, v in rows.items() if v["rho"] > 0 and v["perm_p"] < 0.05]
    mean_arms = [a for a, v in rows.items() if v["mean_pooled"]]
    non_mean = [a for a, v in rows.items() if not v["mean_pooled"]]
    n = len(rows)
    n_cls = max(v["n_classes"] for v in rows.values())
    chance = math.comb(n_cls, k)

    print(f"\nP1  arms putting the {k} failure class(es) in the bottom {k} by margin: "
          f"{len(hits)}/{n}   chance 1/{chance} per arm")
    print(f"P2  arms with a positive rho at p<0.05: {len(sig)}/{n}")
    print(f"    mean-pooled arms: {sum(rows[a]['locates_failure'] for a in mean_arms)}"
          f"/{len(mean_arms)} locate them, "
          f"{sum(rows[a]['rho'] > 0 and rows[a]['perm_p'] < 0.05 for a in mean_arms)}"
          f"/{len(mean_arms)} significant")
    if non_mean:
        print(f"    CLS and max pooling: "
              f"{sum(rows[a]['locates_failure'] for a in non_mean)}/{len(non_mean)} locate them, "
              f"{[(a, round(rows[a]['rho'], 3), round(rows[a]['perm_p'], 4)) for a in non_mean]}")
    neg = [a for a, v in rows.items()
           if all(x < 0 for x in v["failure_margins"].values())]
    print(f"    arms where EVERY failure class has a negative margin: {len(neg)}/{n}")

    p1, p2 = len(hits) > n / 2, len(sig) > n / 2
    if p1 and p2:
        verdict = (f"SUPPORTED: {len(hits)}/{n} arms put the failure class(es) at the bottom "
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

    dest = RES / "margin_across_arms.json"
    json.dump({"panel": pv, "failure_classes": failures, "k": k, "perms": PERMS,
               "arms_embedded_but_unscored": missing,
               "note": ("v2 has one failure class, so per-arm chance is 1/9. v3's pair cannot "
                        "be used here: only the canonical arm is embedded for v3"),
               "distinct_from": ("03k_margin_holdout, which tests whether low-margin MEMBERS "
                                 "are recovered less and is already pinned across 14 arms"),
               "arms": rows,
               "P1": {"arms_locating_failure": sorted(hits), "n_arms": n,
                      "majority": bool(p1)},
               "P2": {"arms_significant": sorted(sig), "majority": bool(p2)},
               "arms_with_all_failure_margins_negative": sorted(neg),
               "chance_per_arm": 1 / chance,
               "verdict": verdict}, open(dest, "w"), indent=2)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
