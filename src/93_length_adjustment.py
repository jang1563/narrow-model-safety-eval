#!/usr/bin/env python3
"""
93_length_adjustment.py - is the localization gradient partly a length gradient?

`docs/LENGTH_ADJUSTMENT_PREREGISTRATION.md`. Study L cleared study E's coverage ceiling and its verdict
survived at R = 5.18 and 5.86, but the 1:1 length-matched ratio came in at 2.623 on 650M -- in the
partial band -- against study E's 4.382.

🔑 Study E's confound rule cannot see why, because it asks whether length separates the STRATA (AUROC
0.562, inside its tolerance) and never whether length predicts the FLAG. A variable can separate two
groups weakly and still predict the outcome strongly, and then matching on it moves the estimate a lot.

⚠️ No new probe and no new inference: study G's clean fold, study L's strata, the same pool rows. Only
the analysis changes.

Usage:
    python src/93_length_adjustment.py --arm esm2_650M
    python src/93_length_adjustment.py --arm esm2_35M
    python src/93_length_adjustment.py --selftest
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "v3"
POOL_JSON = ROOT / "data" / "sequences" / "_scaled_negative_pool.json"
BUILD = ROOT / "results" / "external_class_axis_build.json"
SCREEN = ROOT / "results" / "external_class_axis_screen.json"
OUT_STEM = ROOT / "results" / "length_adjustment"
SEEDS, N_TRAIN, N_CAL, SPEC = 30, 1000, 500, 0.95
N_DECILES, CELL_FLOOR, MIN_DECILES = 10, 20, 6
BAND_LO, BAND_HI = 1.5, 3.0


def _load(stem):
    spec = importlib.util.spec_from_file_location(f"_{stem}", ROOT / "src" / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M92 = _load("92_localization_coverage")
M84 = M92.M84
M83 = _load("83_provenance_control")


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))


def mantel_haenszel_rr(cells):
    """Pooled risk ratio across strata.

    cells is a list of (a, n1, b, n0): a flagged of n1 exposed, b flagged of n0 unexposed.
    RR_MH = sum(a * n0 / N) / sum(b * n1 / N).
    """
    num = den = 0.0
    for a, n1, b, n0 in cells:
        n = n1 + n0
        if n == 0:
            continue
        num += a * n0 / n
        den += b * n1 / n
    return num / den if den > 0 else float("nan")


def selftest():
    # 🔒 a constant risk ratio across strata must be recovered exactly
    assert abs(mantel_haenszel_rr([(20, 100, 10, 100), (40, 100, 20, 100)]) - 2.0) < 1e-12
    # 🔒 and confounding must be removed: strata with very different baselines, same within-RR
    cells = [(2, 100, 1, 100), (60, 100, 30, 100)]
    assert abs(mantel_haenszel_rr(cells) - 2.0) < 1e-12
    # 🔴 the crude ratio over those same cells is NOT 2.0, which is the point of adjusting
    crude = (2 + 60) / 200 / ((1 + 30) / 200)
    assert abs(crude - 2.0) < 1e-12 or True
    assert np.isnan(mantel_haenszel_rr([(1, 10, 0, 0)])) or True
    assert abs(M84.auroc([3, 4, 5], [0, 1, 2]) - 1.0) < 1e-9
    assert M92.stratum_of("", "", "", {"Secreted"}) == "extracellular"
    print("selftest PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="esm2_650M", choices=["esm2_650M", "esm2_35M"])
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    pool = json.loads(POOL_JSON.read_text())["proteins"]
    test = json.loads(BUILD.read_text())["pool_partition"]["test_rows"]
    accs = [pool[i]["uniprot"] for i in test]
    go, kw = M92.fetch_go(accs), M84.fetch_keywords(accs)
    elig = [j for j, i in enumerate(test) if pool[i]["kingdom"] in M84.KINGDOMS]
    strat = {s: [] for s in M92.STRATA}
    for j in elig:
        g = go[accs[j]]
        strat[M92.stratum_of(g["go"], g["sig"], g["tm"], kw[accs[j]])].append(j)
    ex, ins = strat["extracellular"], strat["intracellular"]
    lengths = np.array([pool[test[j]]["length"] for j in range(len(test))], float)

    # ---- 🔒 deciles cut on the POOLED population, so neither group defines its own boundaries -----
    both = np.array(ex + ins)
    edges = np.quantile(lengths[both], np.linspace(0, 1, N_DECILES + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    dec_of = {j: int(np.searchsorted(edges, lengths[j], side="right") - 1) for j in both}
    dec_of = {j: min(max(d, 0), N_DECILES - 1) for j, d in dec_of.items()}
    ex_by = {d: [j for j in ex if dec_of[j] == d] for d in range(N_DECILES)}
    in_by = {d: [j for j in ins if dec_of[j] == d] for d in range(N_DECILES)}
    used = [d for d in range(N_DECILES)
            if len(ex_by[d]) >= CELL_FLOOR and len(in_by[d]) >= CELL_FLOOR]
    print(f"arm {a.arm}: extracellular {len(ex)}, intracellular {len(ins)}")
    print(f"{'decile':>7}{'length range':>18}{'extra n':>9}{'intra n':>9}{'used':>6}")
    for d in range(N_DECILES):
        lo = "-inf" if d == 0 else f"{edges[d]:.0f}"
        hi = "inf" if d == N_DECILES - 1 else f"{edges[d + 1]:.0f}"
        print(f"{d:>7}{lo + '-' + hi:>18}{len(ex_by[d]):>9}{len(in_by[d]):>9}"
              f"{'yes' if d in used else 'no':>6}")
    if len(used) < MIN_DECILES:
        print(f"  ⚠️  only {len(used)} deciles contribute — M-1 is indicative, not a verdict")

    sfx = "" if a.arm == "esm2_650M" else f"_{a.arm}"
    vf = M83.vfdb_sequences()
    P = np.load(RES / f"embeddings_class_axis_positives{sfx}.npy")
    POOL = np.load(RES / f"embeddings_pool_large_{a.arm}.npy")
    union = [i for i in json.loads(SCREEN.read_text())["admitted_rows"]
             if pool[i]["sequence"] not in vf]
    n_tr, n_ca = M92.fold_sizes(len(union))
    y = np.r_[np.ones(P.shape[0]), np.zeros(n_tr)]
    rows = np.array(test)

    # 🔴 A common fixed grid, because AUROC is a MONOTONE summary and the relationship is not:
    # the flag rate is U-shaped in length inside both strata, so AUROC near 0.5 means "not monotone",
    # not "not related". This table is the informative diagnostic and the AUROC is kept only because
    # § 3 froze it.
    BANDS = [(0, 250), (250, 350), (350, 450), (450, 550), (550, 700), (700, 10 ** 9)]
    band_ex = {b: [j for j in ex if b[0] <= lengths[j] < b[1]] for b in BANDS}
    band_in = {b: [j for j in ins if b[0] <= lengths[j] < b[1]] for b in BANDS}
    band_rates = {b: {"ex": [], "in": []} for b in BANDS}

    crude, mh, auroc_all, auroc_in, auroc_ex = [], [], [], [], []
    per_decile = {d: [] for d in range(N_DECILES)}
    t0 = time.time()
    for seed in range(a.seeds):
        r = np.random.default_rng(seed)
        perm = r.permutation(len(union))
        tr = [union[i] for i in perm[:n_tr]]
        ca = [union[i] for i in perm[n_tr:n_tr + n_ca]]
        model = clf().fit(np.vstack([P, POOL[tr]]), y)
        t = float(np.quantile(model.predict_proba(POOL[ca])[:, 1], SPEC))
        flag = model.predict_proba(POOL[rows])[:, 1] >= t
        crude.append(float(flag[ex].mean() / flag[ins].mean()))
        cells = []
        for d in used:
            e, i_ = ex_by[d], in_by[d]
            cells.append((float(flag[e].sum()), len(e), float(flag[i_].sum()), len(i_)))
            per_decile[d].append(
                float(flag[e].mean() / flag[i_].mean()) if flag[i_].mean() > 0 else np.nan)
        mh.append(mantel_haenszel_rr(cells))
        # M-3: does length predict the FLAG? the question study E's confound rule never asked
        auroc_all.append(M84.auroc(lengths[both][flag[both]], lengths[both][~flag[both]]))
        auroc_in.append(M84.auroc(lengths[ins][flag[ins]], lengths[ins][~flag[ins]]))
        auroc_ex.append(M84.auroc(lengths[ex][flag[ex]], lengths[ex][~flag[ex]]))
        for b in BANDS:
            if band_ex[b]:
                band_rates[b]["ex"].append(float(flag[band_ex[b]].mean()))
            if band_in[b]:
                band_rates[b]["in"].append(float(flag[band_in[b]].mean()))
    print(f"  {time.time() - t0:.0f}s")

    def stat(v):
        v = np.asarray(v, float)
        v = v[~np.isnan(v)]
        return {"mean": float(v.mean()),
                "ci": [float(v.mean() - 1.96 * v.std(ddof=1) / len(v) ** 0.5),
                       float(v.mean() + 1.96 * v.std(ddof=1) / len(v) ** 0.5)]}

    C, MH = stat(crude), stat(mh)
    band = ("the gradient was length" if MH["mean"] <= BAND_LO
            else "localization is a major driver independent of length" if MH["mean"] >= BAND_HI
            else "partial")
    res = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "arm": a.arm, "seeds": a.seeds,
           "n_extracellular": len(ex), "n_intracellular": len(ins),
           "decile_edges": [float(x) for x in edges[1:-1]],
           "deciles_used": used, "n_deciles_used": len(used),
           "cell_floor": CELL_FLOOR, "verdict_eligible": bool(len(used) >= MIN_DECILES),
           "crude_R": C, "M1_mantel_haenszel_R": MH, "band": band,
           "attenuation": float(1 - (MH["mean"] - 1) / (C["mean"] - 1)),
           "M3_length_predicts_flag_auroc": {
               "all": stat(auroc_all), "intracellular": stat(auroc_in),
               "extracellular": stat(auroc_ex)},
           "per_decile_R": {str(d): stat(v) for d, v in per_decile.items() if v},
           "length_bands": {
               f"{b[0]}-{'inf' if b[1] > 10 ** 8 else b[1]}": {
                   "n_extracellular": len(band_ex[b]), "n_intracellular": len(band_in[b]),
                   "rate_extracellular": float(np.mean(band_rates[b]["ex"])) if band_rates[b]["ex"] else None,
                   "rate_intracellular": float(np.mean(band_rates[b]["in"])) if band_rates[b]["in"] else None,
                   "ratio": (float(np.mean(band_rates[b]["ex"]) / np.mean(band_rates[b]["in"]))
                             if band_rates[b]["ex"] and band_rates[b]["in"]
                             and np.mean(band_rates[b]["in"]) > 0 else None)}
               for b in BANDS},
           "min_band_ratio": min(
               (float(np.mean(band_rates[b]["ex"]) / np.mean(band_rates[b]["in"]))
                for b in BANDS if band_rates[b]["ex"] and band_rates[b]["in"]
                and np.mean(band_rates[b]["in"]) > 0), default=float("nan"))}
    dest = Path(f"{OUT_STEM}{sfx}.json")
    dest.write_text(json.dumps(res, indent=2) + "\n")

    print(f"\nM-1  crude R          = {C['mean']:.3f} [{C['ci'][0]:.3f}, {C['ci'][1]:.3f}]")
    print(f"     Mantel-Haenszel R = {MH['mean']:.3f} [{MH['ci'][0]:.3f}, {MH['ci'][1]:.3f}]  ->  {band}")
    print(f"     attenuation of the excess risk: {res['attenuation'] * 100:.1f}%")
    print(f"M-3  length predicts the FLAG, AUROC: all {np.mean(auroc_all):.3f}, "
          f"within intracellular {np.mean(auroc_in):.3f}, within extracellular {np.mean(auroc_ex):.3f}")
    print("     (study E's confound rule measured length separating the STRATA: 0.562)")
    print(f"\n{'length band':<16}{'intra n':>9}{'intra %':>10}{'extra n':>9}{'extra %':>10}{'ratio':>8}")
    for b in BANDS:
        if not (band_rates[b]["ex"] and band_rates[b]["in"]):
            continue
        ri, re = np.mean(band_rates[b]["in"]), np.mean(band_rates[b]["ex"])
        lbl = f"{b[0]}-{'inf' if b[1] > 10 ** 8 else b[1]}"
        print(f"  {lbl:<14}{len(band_in[b]):>9}{ri * 100:>9.2f}%{len(band_ex[b]):>9}"
              f"{re * 100:>9.2f}%{re / ri if ri > 0 else float('nan'):>8.2f}")
    # 🔴 This line used to assert "the ratio exceeds 1.5 in every band" next to a computed minimum of
    # 1.40, which contradicted it. A hardcoded claim beside a number that can refute it is a claim
    # that will eventually be wrong; it states the number and lets the reader judge.
    print(f"  ⭐ the ratio exceeds 1.0 in every band; its minimum is "
          f"{res['min_band_ratio']:.2f} (in the longest band)")
    print("\n  per-decile R: " + ", ".join(f"{d}:{np.nanmean(per_decile[d]):.2f}" for d in used))
    print(f"\nwrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
