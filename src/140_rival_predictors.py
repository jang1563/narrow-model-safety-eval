#!/usr/bin/env python3
"""Is max-cosine special, or would any neighbourhood quantity do?

⚠️ EXPLORATORY. No preregistration, no bands, and nothing here can change study AT's
verdict. `docs/GENERALITY_GRID_PREREGISTRATION.md` froze AT-2's rivals as class size and the
two halves of margin's own definition; this adds predictors AFTER those were fixed, so they
are reported as a descriptive sweep and labelled as one.

🔑 The question it answers is the sharpest one an outside reader has about the result: margin
is a MAX over cosines, and a max is a strong choice. If a mean, a centroid or a cohesion
measure orders the classes just as well, then what the result is about is "the held-out class
sits near the negatives", not margin in particular -- which is a weaker and more honest claim.
If the max version wins, that is worth knowing too.

🔒 No refitting. Recoveries come from study AT's artifact; only the predictors are new.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GRID = ROOT / "results/generality_grid.json"
OUT = ROOT / "results/rival_predictors.json"


def load(mod):
    spec = importlib.util.spec_from_file_location(mod[:4], ROOT / "src" / mod)
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod[:4]] = m
    spec.loader.exec_module(m)
    return m


def predictors(P, NEGU, y, m30):
    """Every predictor is a dict class -> float, computed on the frozen vectors alone.

    🔒 `max_minus_max` is study AT's margin, recomputed here so the sweep contains the
    quantity it is a sweep over -- if it disagrees with AT's own number the comparison is
    void, and the selftest checks that.
    """
    cats = sorted(set(y.tolist()))
    n = len(P)
    mx_o, mn_o = np.empty(n), np.empty(n)
    mx_n, mn_n = np.empty(n), np.empty(n)
    CH = 1024
    for i in range(0, n, CH):
        sl = slice(i, min(i + CH, n))
        spp = m30.cos(P[sl], P)
        for r in range(spp.shape[0]):
            spp[r, i + r] = -np.inf
        same = y[sl][:, None] == y[None, :]
        spp[same] = -np.inf
        mx_o[sl] = spp.max(1)
        # the MEAN over out-of-class positives needs the masked entries excluded, not -inf
        msk = np.where(np.isfinite(spp), spp, np.nan)
        mn_o[sl] = np.nanmean(msk, axis=1)
        spn = m30.cos(P[sl], NEGU)
        mx_n[sl] = spn.max(1)
        mn_n[sl] = spn.mean(1)

    # class and negative centroids, cosine to them
    cen = {c: P[y == c].mean(0) for c in cats}
    ncen = NEGU.mean(0)
    if len(cats) < 2:
        raise SystemExit(f"{len(cats)} class(es): every predictor here is a contrast "
                         "against the other classes and needs at least two")

    def cos1(A, b):
        b = b / (np.linalg.norm(b) + 1e-12)
        An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
        return An @ b

    out = {
        # study AT's margin
        "max_minus_max": {c: float((mx_o[y == c] - mx_n[y == c]).mean()) for c in cats},
        # the same shape with MEANS instead of maxima -- does the max matter?
        "mean_minus_mean": {c: float((mn_o[y == c] - mn_n[y == c]).mean()) for c in cats},
        # mixed, to separate which half the max matters in
        "max_pos_mean_neg": {c: float((mx_o[y == c] - mn_n[y == c]).mean()) for c in cats},
        "mean_pos_max_neg": {c: float((mn_o[y == c] - mx_n[y == c]).mean()) for c in cats},
        # the centroid version: the quantity a sparse-MoE router approximates
        "centroid_margin": {
            c: (max(float(cos1(cen[c][None, :], cen[o])[0])
                    for o in cats if o != c)
                - float(cos1(cen[c][None, :], ncen)[0])) for c in cats},
        # cohesion: how tight is the class itself, with no reference to anything else
        "within_class_cohesion": {
            c: float(cos1(P[y == c], cen[c]).mean()) for c in cats},
        # distance to the global positive centroid -- a one-term rival
        "cos_to_global_positive_centroid": {
            c: float(cos1(P[y == c], P.mean(0)).mean()) for c in cats},
        "class_size": {c: float((y == c).sum()) for c in cats},
    }
    return out


def mechanism(grid, m30):
    """When does margin beat its own positive half, and why?

    ⚠️ EXPLORATORY, no bands. Reads only the stored rivals, so it touches no embedding.

    🔑 The question: `margin = nearest_positive - nearest_negative`. If the negative term
    carries no signal then margin is the positive term plus noise, and it should LOSE. So the
    gain from subtracting should track how much signal the negative term has.

    Returns the per-cell table and the Spearman of |rho(nearest_negative)| against
    (rho(margin) - rho(nearest_positive)) across every cell x setting.
    """
    rows = []
    for cell, cv in grid.get("cells", {}).items():
        for setting, sv in cv["settings"].items():
            r = sv.get("rivals", {})
            if "nearest_positive" not in r or "nearest_negative" not in r:
                continue
            mg, hp, hn = sv["AT1_rho"], r["nearest_positive"], r["nearest_negative"]
            if not all(np.isfinite(x) for x in (mg, hp, hn)):
                continue
            rows.append({
                "cell": cell, "setting": setting,
                "rho_margin": mg, "rho_nearest_positive": hp, "rho_nearest_negative": hn,
                "abs_rho_negative": abs(hn),
                "gain_from_subtracting": mg - hp,
                "margin_wins": bool(mg > hp),
                "margin_clears": bool(mg >= 0.60),
                "nearest_positive_clears": bool(hp >= 0.60),
            })
    out = {"rows": rows, "n": len(rows)}
    if len(rows) >= 4:
        out["rho_of_absneg_vs_gain"] = m30._spearman(
            [r["abs_rho_negative"] for r in rows],
            [r["gain_from_subtracting"] for r in rows])
    out["margin_clears_count"] = sum(1 for r in rows if r["margin_clears"])
    out["nearest_positive_clears_count"] = sum(
        1 for r in rows if r["nearest_positive_clears"])
    out["cells_where_one_term_is_enough"] = sorted(
        {f"{r['cell']}/{r['setting']}" for r in rows
         if r["nearest_positive_clears"] and not r["margin_clears"]})
    out["reading"] = (
        "if nearest_positive clears in strictly more cell-settings than margin, the general "
        "claim is about a held-out class's distance to its nearest TRAINED class, and the "
        "subtraction is a protein-fold refinement rather than part of the general result")
    return out


def selftest():
    ok = True
    m30 = load("30_margin_across_arms.py")
    rng = np.random.default_rng(0)
    A = rng.normal(size=(40, 16)) + 5.0
    B = rng.normal(size=(40, 16)) + 5.0
    Cc = rng.normal(size=(40, 16)) - 5.0
    NEG = rng.normal(size=(200, 16)) - 5.0
    P = np.vstack([A, B, Cc])
    y = np.array(["a"] * 40 + ["b"] * 40 + ["c"] * 40)
    # 🔒 the mechanism table, driven both ways
    # ⚠️ The swept cell must have margin BELOW the 0.60 floor and nearest_positive above it,
    # or "one term is enough" has no instance. The first version used margin 0.7, which
    # clears, so the expected list was empty and the assertion was simply wrong -- caught by
    # the gate, not by me, because my own check ran `timeout 300 python3 ...` and macOS has
    # no `timeout`, so the selftest never executed and the rc=0 came from `echo`.
    fake = {"cells": {"a": {"settings": {
        "fixed": {"AT1_rho": 0.9, "rivals": {"nearest_positive": 0.6,
                                             "nearest_negative": -0.5}},
        "swept": {"AT1_rho": 0.5, "rivals": {"nearest_positive": 0.8,
                                             "nearest_negative": -0.05}}}}}}
    mk = mechanism(fake, m30)
    if mk["n"] != 2:
        print(f"  XX mechanism() read {mk['n']} rows from 2")
        ok = False
    if not mk["rows"][0]["margin_wins"] or mk["rows"][1]["margin_wins"]:
        print(f"  XX margin_wins is wrong: {[r['margin_wins'] for r in mk['rows']]}")
        ok = False
    if mk["cells_where_one_term_is_enough"] != ["a/swept"]:
        print(f"  XX the one-term-is-enough cells are wrong: "
              f"{mk['cells_where_one_term_is_enough']}")
        ok = False
    # a NaN rival must be skipped, not propagated
    fake2 = {"cells": {"b": {"settings": {"fixed": {
        "AT1_rho": 0.9, "rivals": {"nearest_positive": float("nan"),
                                   "nearest_negative": -0.5}}}}}}
    if mechanism(fake2, m30)["n"] != 0:
        print("  XX mechanism() kept a row with a NaN rival")
        ok = False

    pr = predictors(P, NEG, y, m30)
    if not all(len(v) == 3 for v in pr.values()):
        print(f"  XX a predictor does not cover every class: "
              f"{[(k, len(v)) for k, v in pr.items() if len(v) != 3]}")
        ok = False
    # 🔒 the planted geometry must order on the margin-shaped predictors
    for k in ("max_minus_max", "mean_minus_mean", "max_pos_mean_neg", "mean_pos_max_neg"):
        if not (pr[k]["a"] > pr[k]["c"] and pr[k]["b"] > pr[k]["c"]):
            print(f"  XX {k} does not order the planted geometry: {pr[k]}")
            ok = False
    # ⚠️ and the recomputed margin must AGREE with study AT's, or the sweep is comparing a
    # different quantity to AT's verdict. Checked against the grid when it exists.
    if GRID.exists():
        # ⚠️ Only the CHEAPEST cell. This check recomputes every pairwise cosine, and running
        # it on all of them -- 12,989 x 12,989 at 1,280 dims for the protein cell -- made the
        # selftest take minutes. One cell proves the code path; the full comparison runs in
        # `main()`, where it is a hard failure on every cell.
        g = json.loads(GRID.read_text())
        cells = g.get("cells", {})
        if cells:
            cheapest = min(cells, key=lambda k: cells[k]["n_positives"] * cells[k]["dim"])
            try:
                mine = _cell_margin(cheapest, m30)
                v = cells[cheapest]
                d = max(abs(mine[c] - v["margin"][c]) for c in v["margin"])
                if d > 1e-9:
                    print(f"  XX {cheapest}: recomputed margin differs from AT's "
                          f"by {d:.3g}")
                    ok = False
            except (FileNotFoundError, KeyError) as e:
                print(f"  -- margin agreement unchecked for {cheapest}: {e}")
    if not pr["class_size"] == {"a": 40.0, "b": 40.0, "c": 40.0}:
        print(f"  XX class_size is wrong: {pr['class_size']}")
        ok = False
    print("selftest PASS" if ok else "selftest FAIL")
    return 0 if ok else 1


def _cell_margin(cell, m30):
    g139 = load("139_generality_grid.py")
    P, N, y = g139.get_data(cell)
    X = np.vstack([P, N])
    neg = np.arange(len(P), len(P) + len(N))
    perm = np.random.default_rng(12345).permutation(len(neg))
    union = neg[perm[:2 * len(neg) // 3]]
    return predictors(P, X[union], y, m30)["max_minus_max"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not GRID.exists():
        raise SystemExit(f"{GRID.name} does not exist; study AT has not produced a cell yet")

    t0 = time.time()
    m30 = load("30_margin_across_arms.py")
    g139 = load("139_generality_grid.py")
    grid = json.loads(GRID.read_text())
    out = {"built": time.strftime("%Y-%m-%d %H:%M:%S"),
           "study": "exploratory rival-predictor sweep over study AT's cells",
           "exploratory": ("no preregistration and no bands. AT-2's rivals were frozen as "
                           "class size and margin's two halves; these were added after, so "
                           "nothing here can change AT's verdict"),
           "question": ("margin is a MAX over cosines. If a mean, a centroid or a cohesion "
                        "measure orders the classes as well, the result is about a held-out "
                        "class sitting near the negatives and not about margin in particular"),
           "cells": {}}
    out["mechanism"] = mechanism(grid, m30)

    for cell, cv in grid.get("cells", {}).items():
        try:
            P, N, y = g139.get_data(cell)
        except FileNotFoundError as e:
            print(f"  -- {cell} skipped: {e}", flush=True)
            continue
        X = np.vstack([P, N])
        neg = np.arange(len(P), len(P) + len(N))
        perm = np.random.default_rng(12345).permutation(len(neg))
        union = neg[perm[:2 * len(neg) // 3]]
        pr = predictors(P, X[union], y, m30)
        # 🔒 the recomputed margin must match AT's to the last place, or this is not a sweep
        # over AT's quantity
        d = max(abs(pr["max_minus_max"][c] - cv["margin"][c]) for c in cv["margin"])
        if d > 1e-9:
            raise SystemExit(f"{cell}: recomputed margin differs from AT's by {d:.3g}")
        row = {"margin_matches_AT": True, "settings": {}}
        for setting, sv in cv["settings"].items():
            rec = sv["recovery"]
            ks = sorted(rec)
            rv = [rec[c]["mean"] for c in ks]
            rhos = {k: m30._spearman([pr[k][c] for c in ks], rv) for k in pr}
            best = max(rhos, key=lambda k: abs(rhos[k]))
            row["settings"][setting] = {
                "rho": rhos,
                "margin_rho": rhos["max_minus_max"],
                "best_predictor": best,
                "margin_is_best": best == "max_minus_max",
                "margin_beats_mean_version": (
                    abs(rhos["max_minus_max"]) > abs(rhos["mean_minus_mean"])),
                "margin_beats_centroid": (
                    abs(rhos["max_minus_max"]) > abs(rhos["centroid_margin"])),
            }
            print(f"  {cell:<24} {setting:<6} margin {rhos['max_minus_max']:+.4f}  "
                  f"best {best} {rhos[best]:+.4f}", flush=True)
        out["cells"][cell] = row

    # a summary that does not hide a loss
    wins = {s: sum(1 for v in out["cells"].values()
                   if v["settings"].get(s, {}).get("margin_is_best"))
            for s in ("fixed", "swept")}
    out["summary"] = {
        "cells": len(out["cells"]),
        "margin_is_best_predictor": wins,
        "margin_beats_mean_version": {
            s: sum(1 for v in out["cells"].values()
                   if v["settings"].get(s, {}).get("margin_beats_mean_version"))
            for s in ("fixed", "swept")},
        "margin_beats_centroid": {
            s: sum(1 for v in out["cells"].values()
                   if v["settings"].get(s, {}).get("margin_beats_centroid"))
            for s in ("fixed", "swept")},
        "reading": ("if margin is NOT the best predictor in most cells, the claim is about a "
                    "held-out class sitting near the negatives and the paper should say that "
                    "instead of naming a formula"),
    }
    out["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nmargin best in {wins} of {len(out['cells'])} cells (fixed/swept)")
    print(f"wrote {OUT.relative_to(ROOT)} in {out['seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
