#!/usr/bin/env python3
"""Does any training-free quantity predict HOW MUCH the subtraction is worth?

⚠️ EXPLORATORY. No preregistration, no bands, and nothing here is a result.

🔑 Study AV established prospectively that R = rho(nearest_positive, nearest_negative)
ranks the SIGN of the gain from subtracting the negative term -- rho(R, gain) = -0.6610 over
24 cell-settings, against a band frozen before four of the eight encoders existed here. 🔴 Its
own stated gap: R says nothing about MAGNITUDE, and the single sealed miss was the smallest
|gain| of the four scored.

🔴 This file sweeps NINE candidates against 24 points. With nine candidates the best will
look good by chance, so the output names the selection problem rather than reporting a
winner: the expected maximum |rho| under the null is estimated by permutation, and any
candidate that does not clear it is noise. 🔒 A candidate that does clear it is a HYPOTHESIS
and the only thing that would make it a result is a sealed test on cells not yet embedded --
the procedure study AV used.

🔒 Every candidate is computable from the frozen embedding and the class partition alone.
Nothing here is fitted and nothing uses recovery.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ("results/generality_grid.json", "results/image_axis_repair.json",
           "results/training_free_rule.json")
OUT = ROOT / "results/magnitude_candidates.json"
PERMS = 20000


def load(mod):
    spec = importlib.util.spec_from_file_location(mod[:4], ROOT / "src" / mod)
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod[:4]] = m
    spec.loader.exec_module(m)
    return m


def candidates(cv, m30):
    """Nine training-free summaries of a cell's geometry, from per-class numbers only."""
    ks = sorted(cv["margin"])
    mar = np.array([cv["margin"][k] for k in ks])
    hp = np.array([cv["nearest_positive"][k] for k in ks])
    hn = np.array([cv["nearest_negative"][k] for k in ks])
    R = m30._spearman(hp.tolist(), hn.tolist())
    sd_hp, sd_hn = float(hp.std(ddof=1)), float(hn.std(ddof=1))
    return {
        # the known sign predictor, for reference
        "R_rho_hp_hn": R,
        "abs_R": abs(R) if np.isfinite(R) else float("nan"),
        # how close the negatives sit, and how much that varies
        "mean_nearest_negative": float(hn.mean()),
        "sd_nearest_negative": sd_hn,
        "sd_ratio_neg_over_pos": (sd_hn / sd_hp) if sd_hp > 0 else float("nan"),
        # interleaving: classes whose mean margin is at or below zero
        "frac_classes_negative_margin": float((mar <= 0).mean()),
        # overall separation and its spread
        "mean_margin": float(mar.mean()),
        "sd_margin": float(mar.std(ddof=1)),
        # 🔑 "the negative term varies AND is independent of the positive one" in one number
        "sd_neg_times_one_minus_R": (sd_hn * (1.0 - R)) if np.isfinite(R) else float("nan"),
    }


def collect(m30):
    rows = []
    for f in SOURCES:
        p = ROOT / f
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for cell, cv in d.get("cells", {}).items():
            if "margin" not in cv:
                continue
            c = candidates(cv, m30)
            for s in ("fixed", "swept"):
                sv = cv["settings"][s]
                gain = sv["AT1_rho"] - sv["rivals"]["nearest_positive"]
                rows.append({"cell": cell, "setting": s, "source": f,
                             "gain": gain, "abs_gain": abs(gain), **c})
    return rows


def null_max_abs_rho(n, k, m30, seed=0, perms=PERMS):
    """Expected max |rho| over k independent candidates against n points, by permutation.

    🔑 This is the number that decides whether a sweep found anything. Candidates here are
    NOT independent -- several are built from the same two terms -- so the true threshold is
    lower than this and the estimate is CONSERVATIVE in the direction that makes a false
    positive harder, which is the direction to err in.
    """
    rng = np.random.default_rng(seed)
    base = np.arange(n, dtype=float)
    out = []
    for _ in range(perms):
        best = 0.0
        for _ in range(k):
            r = m30._spearman(base.tolist(), rng.permutation(base).tolist())
            best = max(best, abs(r))
        out.append(best)
    a = np.array(out)
    return {"n": n, "k": k, "mean": float(a.mean()),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99))}


def selftest():
    ok = True
    m30 = load("30_margin_across_arms.py")
    # a planted cell: margin is its two halves subtracted, and the summaries must agree
    cv = {"margin": {"a": 0.1, "b": -0.2, "c": 0.3},
          "nearest_positive": {"a": 0.5, "b": 0.4, "c": 0.7},
          "nearest_negative": {"a": 0.4, "b": 0.6, "c": 0.4}}
    c = candidates(cv, m30)
    if abs(c["frac_classes_negative_margin"] - 1 / 3) > 1e-12:
        print(f"  XX the negative-margin fraction is wrong: {c}")
        ok = False
    if abs(c["mean_margin"] - (0.1 - 0.2 + 0.3) / 3) > 1e-12:
        print(f"  XX mean_margin is wrong: {c['mean_margin']}")
        ok = False
    if len(c) != 9:
        print(f"  XX {len(c)} candidates, the docstring says nine")
        ok = False
    # 🔒 the null threshold must RISE with the number of candidates, or the correction is not
    # correcting for anything
    a = null_max_abs_rho(24, 1, m30, perms=400)
    b = null_max_abs_rho(24, 9, m30, perms=400)
    if not b["p95"] > a["p95"]:
        print(f"  XX the null max |rho| did not rise with k: {a['p95']} -> {b['p95']}")
        ok = False
    # and it must be substantial at n=24: a single candidate's own p95 is already large
    if a["p95"] < 0.3:
        print(f"  XX a single candidate's null p95 at n=24 is implausibly low: {a['p95']}")
        ok = False
    rows = collect(m30)
    if rows and len({r["cell"] for r in rows}) < 8:
        print(f"  XX only {len({r['cell'] for r in rows})} cells collected, expected 8+")
        ok = False
    print("selftest PASS" if ok else "selftest FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    t0 = time.time()
    m30 = load("30_margin_across_arms.py")
    rows = collect(m30)
    names = [k for k in rows[0] if k not in
             ("cell", "setting", "source", "gain", "abs_gain")]
    res = {}
    for k in names:
        x = [r[k] for r in rows]
        fin = [i for i, v in enumerate(x) if np.isfinite(v)]
        xs = [x[i] for i in fin]
        res[k] = {
            "n": len(fin),
            "rho_vs_signed_gain": m30._spearman(xs, [rows[i]["gain"] for i in fin]),
            "rho_vs_abs_gain": m30._spearman(xs, [rows[i]["abs_gain"] for i in fin]),
        }
    null = null_max_abs_rho(len(rows), len(names), m30)
    best_mag = max(res, key=lambda k: abs(res[k]["rho_vs_abs_gain"]))
    best_sign = max(res, key=lambda k: abs(res[k]["rho_vs_signed_gain"]))
    out = {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "study": "exploratory -- does any training-free quantity predict the MAGNITUDE "
                 "of the gain from subtracting?",
        "exploratory": ("no preregistration and no bands. Nine candidates against "
                        f"{len(rows)} points, so the best will look good by chance; the "
                        "permutation threshold below is what decides whether anything was "
                        "found, and a candidate that clears it is a HYPOTHESIS requiring a "
                        "sealed test on cells not yet embedded"),
        "n_points": len(rows), "n_cells": len({r["cell"] for r in rows}),
        "n_candidates": len(names),
        "null_max_abs_rho": null,
        "candidates": res,
        "best_for_magnitude": {"name": best_mag,
                               "rho": res[best_mag]["rho_vs_abs_gain"],
                               "clears_null_p95": bool(
                                   abs(res[best_mag]["rho_vs_abs_gain"]) > null["p95"])},
        "best_for_sign": {"name": best_sign,
                          "rho": res[best_sign]["rho_vs_signed_gain"],
                          "clears_null_p95": bool(
                              abs(res[best_sign]["rho_vs_signed_gain"]) > null["p95"])},
        "rows": rows,
    }
    out["reading"] = (
        "a candidate whose |rho| does not exceed the permutation p95 for NINE candidates at "
        f"n={len(rows)} ({null['p95']:.4f}) is indistinguishable from noise in this sweep, "
        "however large it looks on its own")
    OUT.write_text(json.dumps(out, indent=2) + "\n")

    print(f"  {len(rows)} cell-settings from {out['n_cells']} cells, "
          f"{len(names)} candidates")
    print(f"  permutation max |rho| for 9 candidates at n={len(rows)}: "
          f"mean {null['mean']:.4f}  p95 {null['p95']:.4f}  p99 {null['p99']:.4f}\n")
    print(f"  {'candidate':34s} {'vs signed gain':>15s} {'vs |gain|':>12s}")
    for k in sorted(res, key=lambda k: -abs(res[k]["rho_vs_abs_gain"])):
        s, m = res[k]["rho_vs_signed_gain"], res[k]["rho_vs_abs_gain"]
        flag = "  <- clears p95" if abs(m) > null["p95"] else ""
        print(f"  {k:34s} {s:+15.4f} {m:+12.4f}{flag}")
    print(f"\n  best for MAGNITUDE: {best_mag} at {res[best_mag]['rho_vs_abs_gain']:+.4f}"
          f"  (clears p95: {out['best_for_magnitude']['clears_null_p95']})")
    print(f"  best for SIGN:      {best_sign} at "
          f"{res[best_sign]['rho_vs_signed_gain']:+.4f}")
    print(f"\nwrote {OUT.relative_to(ROOT)} in {round(time.time() - t0, 1)}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
