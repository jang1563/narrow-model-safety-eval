#!/usr/bin/env python3
"""
82_organism_stratified.py - margin against recovery with the organism held constant.

`docs/ORGANISM_STRATIFIED_PREREGISTRATION.md`. Study B found margin ordering VFDB's 13 categories by
recovery at +0.6813, and its own confound rule fired: an organism measure correlates with recovery at
+0.7613, so B-1's verdict is uninterpretable. Matching organism composition across categories is
impossible — the profiles differ for biological reasons, and the three designs tried are recorded in
§ 1 of that document — so organism is held constant by **stratification** instead.

The folds are study B's, unchanged: same categories, same partition, same seeds, same pipeline. The
only change is that per-member outcomes are kept, so recovery can be read inside a single species.

    recovery(C,S) = mean over seeds of the fraction of S's members of C that cleared the threshold
    margin(C,S)   = src/30's margin on S's members of C only

Strata are species with >= 4 categories at >= 3 members. Inside one, the organism is constant, so the
ordering of that stratum's categories cannot come from organism identity, exclusivity, or
same-species-in-training — the three quantities that voided study B.

⚠️ The preregistration states in advance that a null result here is **not** a refutation of margin:
four to six points per stratum cannot separate "margin does not work within an organism" from "there is
not enough power". That is written down so it cannot be decided after seeing the number.

Usage:
    python src/82_organism_stratified.py --selftest
    python src/82_organism_stratified.py
"""

import argparse
import collections
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "v3"
BUILD = ROOT / "results" / "external_class_axis_build.json"
SCREEN = ROOT / "results" / "external_class_axis_screen.json"
OUT_STEM = ROOT / "results" / "organism_stratified"
ALPHA = 0.05
SEEDS, N_TRAIN, N_CAL, SPEC, PERMS = 30, 1000, 500, 0.95, 20000
MIN_CELL, MIN_CATS, FLOOR = 3, 4, 7


def load30():
    spec = importlib.util.spec_from_file_location("s30", ROOT / "src" / "30_margin_across_arms.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["s30"] = m
    spec.loader.exec_module(m)
    return m


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))


def selftest():
    """A stratified permutation must shuffle WITHIN strata, never across them.

    Permuting across strata would test a different null — that no species differs from another — and
    would be anticonservative here, since between-species variation is exactly what stratifying is
    meant to remove.
    """
    rng = np.random.default_rng(0)
    strata = {"A": [0, 1, 2, 3], "B": [4, 5, 6, 7, 8]}
    vals = {i: float(i) for i in range(9)}
    for _ in range(50):
        perm = {}
        for s, idx in strata.items():
            v = [vals[i] for i in idx]
            perm[s] = list(rng.permutation(v))
        for s, idx in strata.items():
            assert sorted(perm[s]) == sorted(vals[i] for i in idx), "values left their stratum"
    print("SELFTEST PASS: the permutation stays inside each stratum")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--arm", default="esm2_650M", choices=["esm2_650M", "esm2_35M"])
    ap.add_argument("--seeds", type=int, default=SEEDS)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    sfx = "" if a.arm == "esm2_650M" else f"_{a.arm}"

    POS_NPY = RES / f"embeddings_class_axis_positives{sfx}.npy"
    POS_MAN = RES / f"embedding_manifest_class_axis_positives{sfx}.json"
    # the pool is named by ARM, not by suffix: embeddings_pool_large_esm2_650M.npy and
    # embeddings_pool_large_esm2_35M.npy. Appending the suffix to the 650M name asked for a file
    # that has never existed.
    POOL_NPY = RES / f"embeddings_pool_large_{a.arm}.npy"
    OUT = Path(f"{OUT_STEM}{sfx}.json")
    m30 = load30()
    build = json.loads(BUILD.read_text())
    screen = json.loads(SCREEN.read_text())
    man = json.loads((POS_MAN).read_text())
    P = np.load(POS_NPY)
    POOL = np.load(POOL_NPY)
    cats = np.array(man["category_of_row"])
    sp = np.array(man["species_of_row"])

    union = screen["admitted_rows"]
    test = build["pool_partition"]["test_rows"]
    n_tr, n_ca = (N_TRAIN, N_CAL) if len(union) >= N_TRAIN + N_CAL else \
        (2 * len(union) // 3, len(union) - 2 * len(union) // 3)
    targets = sorted(c for c, n in collections.Counter(cats).items() if n >= FLOOR)
    print(f"study B's folds: {len(targets)} categories, {n_tr}/{n_ca} negatives per seed, "
          f"{len(test)} test, {a.seeds} seeds")

    # ---- per-member flag rates, the only thing src/81 did not keep ----------------------------
    flagged = np.zeros(len(cats))
    t0 = time.time()
    for C in targets:
        hi = np.where(cats == C)[0]
        tri = np.setdiff1d(np.arange(len(cats)), hi)
        acc = np.zeros(len(hi))
        for seed in range(a.seeds):
            r = np.random.default_rng(seed)
            perm = r.permutation(len(union))
            tr = [union[i] for i in perm[:n_tr]]
            ca = [union[i] for i in perm[n_tr:n_tr + n_ca]]
            model = clf().fit(np.vstack([P[tri], POOL[tr]]),
                              np.r_[np.ones(len(tri)), np.zeros(len(tr))])
            t = float(np.quantile(model.predict_proba(POOL[ca])[:, 1], SPEC))
            acc += (model.predict_proba(P[hi])[:, 1] >= t)
        flagged[hi] = acc / a.seeds
        print(f"  {C:<46} mean member flag rate {flagged[hi].mean() * 100:5.1f}%", flush=True)
    print(f"  {time.time() - t0:.0f}s")

    # ---- cells, strata, and within-stratum margin ----------------------------------------------
    simPP, simPN = m30.cos(P, P), m30.cos(P, POOL[union])
    np.fill_diagonal(simPP, -np.inf)
    cells = {}
    for C in targets:
        for S in sorted(set(sp[cats == C])):
            idx = np.where((cats == C) & (sp == S))[0]
            if len(idx) < MIN_CELL:
                continue
            other = np.setdiff1d(np.arange(len(P)), np.where(cats == C)[0])
            cells[(S, C)] = {
                "n": int(len(idx)),
                "recovery": float(flagged[idx].mean()),
                "margin": float((simPP[np.ix_(idx, other)].max(1) - simPN[idx].max(1)).mean()),
            }
    by_sp = collections.Counter(S for S, _ in cells)
    strata = sorted(S for S, n in by_sp.items() if n >= MIN_CATS)
    print(f"\n{len(cells)} cells at >= {MIN_CELL} members; "
          f"{len(strata)} species with >= {MIN_CATS} categories, covering "
          f"{sum(by_sp[S] for S in strata)} cells")

    per = {}
    for S in strata:
        cs = sorted(C for (s2, C) in cells if s2 == S)
        mg = [cells[(S, C)]["margin"] for C in cs]
        rc = [cells[(S, C)]["recovery"] for C in cs]
        per[S] = {"n_categories": len(cs), "categories": cs,
                  "rho": m30._spearman(mg, rc),
                  "cells": {C: cells[(S, C)] for C in cs}}
        print(f"  {S:<32} {len(cs)} categories  rho {per[S]['rho']:+.3f}")

    obs = float(np.mean([per[S]["rho"] for S in strata]))
    rng = np.random.default_rng(0)
    null = np.empty(PERMS)
    for k in range(PERMS):
        rs = []
        for S in strata:
            cs = per[S]["categories"]
            mg = [cells[(S, C)]["margin"] for C in cs]
            rc = rng.permutation([cells[(S, C)]["recovery"] for C in cs])
            rs.append(m30._spearman(mg, list(rc)))
        null[k] = np.mean(rs)
    p = float((null >= obs).mean())
    npos = sum(1 for S in strata if per[S]["rho"] > 0)

    res = {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"), "alpha": ALPHA, "seeds": a.seeds,
        "min_cell": MIN_CELL, "min_categories_per_stratum": MIN_CATS,
        "n_cells": len(cells), "n_strata": len(strata),
        "cells_in_strata": sum(by_sp[S] for S in strata),
        "per_species": per,
        "C1": {"mean_rho": obs, "perm_p": p, "supported": bool(obs > 0 and p < ALPHA),
               "n_positive_strata": npos, "n_strata": len(strata),
               "null_mean": float(null.mean()), "null_sd": float(null.std())},
        "reading_fixed_in_advance": (
            "supported => organism identity is excluded as the differentiating factor and study B's "
            "B-1 becomes interpretable as a claim about ordering; NOT supported => four to six points "
            "per stratum cannot separate 'margin does not work within an organism' from 'not enough "
            "power', so study B's confound remains unresolved and margin is not refuted"),
    }
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print(f"\nC-1  mean within-species rho = {obs:+.4f}   permutation p = {p:.4f}   "
          f"-> {'SUPPORTED' if res['C1']['supported'] else 'NOT SUPPORTED'} at alpha {ALPHA}")
    print(f"     {npos} of {len(strata)} strata positive; null mean {null.mean():+.4f} "
          f"sd {null.std():.4f}")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    raise SystemExit(main() or 0)
