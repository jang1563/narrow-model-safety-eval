#!/usr/bin/env python3
"""
104_length_band_sensitivity.py - is the length-U an artifact of where the bands were drawn?

`docs/LENGTH_BAND_SENSITIVITY_PREREGISTRATION.md`. Studies M, N, O and P all measure the length effect
through fixed residue cuts chosen once in study M and reused four times without a sensitivity check.

🔑 Perturbing the cuts is the weak version. The strong version removes them: a U-shape in length IS a
positive quadratic coefficient in a regression on log L and (log L)^2, which contains no bands. If that
coefficient is positive on every arm the U is not a cut-point artifact, and if its magnitude falls with
model size then study P's trend survives without bands too.

⚠️ No new probe and no new inference: study G's clean fold, study L's strata, all four arms.

Usage:
    python src/104_length_band_sensitivity.py
    python src/104_length_band_sensitivity.py --selftest
"""

import argparse
import importlib.util
import json
import time
from itertools import permutations
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
OUT = ROOT / "results" / "length_band_sensitivity.json"
SEEDS, SPEC, BAND_FLOOR = 30, 0.95, 100
ARMS = [("esm2_8M", 8e6), ("esm2_35M", 35e6), ("esm2_150M", 150e6), ("esm2_650M", 650e6)]


def _load(stem):
    spec = importlib.util.spec_from_file_location(f"_{stem}", ROOT / "src" / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M92 = _load("92_localization_coverage")
M84 = M92.M84
M83 = _load("83_provenance_control")
M97 = _load("97_size_scaling")


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))


def quad_fit(x, y):
    """Least squares of y on [1, x, x^2]; returns the quadratic coefficient.

    🔒 Plain least squares, on the flag as well as the score, because the quantity of interest is the
    SIGN of curvature and an unpenalised fit cannot shrink it toward zero the way a regularised one
    could. x is expected centred.
    """
    A = np.column_stack([np.ones_like(x), x, x * x])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(beta[2])


def selftest():
    x = np.linspace(-1, 1, 200)
    # 🔒 a known U must give a positive coefficient and a known inverted U a negative one
    assert quad_fit(x, 3 * x ** 2 - 1) > 2.9
    assert quad_fit(x, -3 * x ** 2 + 1) < -2.9
    # 🔒 and a straight line must give ~0, or the fit is inventing curvature
    assert abs(quad_fit(x, 2 * x + 1)) < 1e-9
    assert M97.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    for arm, _ in ARMS:
        sfx = "" if arm == "esm2_650M" else f"_{arm}"
        for f in (RES / f"embeddings_class_axis_positives{sfx}.npy",
                  RES / f"embeddings_pool_large_{arm}.npy"):
            assert f.exists(), f"{arm}: {f.name} missing"
    print(f"selftest PASS (quadratic fit recovers sign and zero; {len(ARMS)} arms present)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    pool = json.loads(POOL_JSON.read_text())["proteins"]
    test = json.loads(BUILD.read_text())["pool_partition"]["test_rows"]
    accs = [pool[i]["uniprot"] for i in test]
    kw, go = M84.fetch_keywords(accs), M92.fetch_go(accs)
    elig = [j for j, i in enumerate(test) if pool[i]["kingdom"] in M84.KINGDOMS]
    strat = {s: [] for s in M92.STRATA}
    for j in elig:
        g = go[accs[j]]
        strat[M92.stratum_of(g["go"], g["sig"], g["tm"], kw[accs[j]])].append(j)
    ins = np.array(strat["intracellular"])
    lengths = np.array([pool[test[j]]["length"] for j in range(len(test))], float)
    x = np.log10(lengths[ins])
    x = x - x.mean()
    print(f"intracellular stratum: {len(ins)} proteins, log10 length centred "
          f"[{x.min():.2f}, {x.max():.2f}]")

    vf = M83.vfdb_sequences()
    union = [i for i in json.loads(SCREEN.read_text())["admitted_rows"]
             if pool[i]["sequence"] not in vf]
    n_tr, n_ca = M92.fold_sizes(len(union))
    rows = np.array(test)

    # ---- U-3's band grid, defined before anything is fitted --------------------------------------
    grids = {"as published": ((0, 250), (450, 550))}
    for d in (-100, -50, 50, 100):
        grids[f"shift {d:+d}"] = ((0, 250 + d), (450 + d, 550 + d))
    q = np.quantile(lengths[ins], [0.2, 0.4, 0.6])
    grids["quintiles"] = ((0, q[0]), (q[1], q[2]))

    out = {}
    for arm, params in ARMS:
        sfx = "" if arm == "esm2_650M" else f"_{arm}"
        P = np.load(RES / f"embeddings_class_axis_positives{sfx}.npy")
        POOL = np.load(RES / f"embeddings_pool_large_{arm}.npy")
        y = np.r_[np.ones(P.shape[0]), np.zeros(n_tr)]
        qf, qs, uidx = [], [], {k: [] for k in grids}
        t0 = time.time()
        for seed in range(a.seeds):
            r = np.random.default_rng(seed)
            perm = r.permutation(len(union))
            tr = [union[i] for i in perm[:n_tr]]
            ca = [union[i] for i in perm[n_tr:n_tr + n_ca]]
            pipe = clf().fit(np.vstack([P, POOL[tr]]), y)
            sc = pipe.named_steps["standardscaler"]
            w = pipe.named_steps["logisticregression"].coef_[0]
            Zt = sc.transform(POOL[rows])
            score = Zt @ w
            t = float(np.quantile(sc.transform(POOL[ca]) @ w, SPEC))
            flag = (score >= t).astype(float)
            qf.append(quad_fit(x, flag[ins]))
            qs.append(quad_fit(x, score[ins]))
            for name, ((lo1, hi1), (lo2, hi2)) in grids.items():
                s_ = [k for k, j in enumerate(ins) if lo1 <= lengths[j] < hi1]
                m_ = [k for k, j in enumerate(ins) if lo2 <= lengths[j] < hi2]
                if len(s_) < BAND_FLOOR or len(m_) < BAND_FLOOR:
                    uidx[name].append(np.nan)
                    continue
                fi = flag[ins]
                uidx[name].append(float(fi[s_].mean() / fi[m_].mean())
                                  if fi[m_].mean() > 0 else np.nan)
        out[arm] = {
            "params": params, "dim": int(POOL.shape[1]),
            "quad_flag": float(np.mean(qf)),
            "quad_flag_ci": [float(np.mean(qf) - 1.96 * np.std(qf, ddof=1) / a.seeds ** 0.5),
                             float(np.mean(qf) + 1.96 * np.std(qf, ddof=1) / a.seeds ** 0.5)],
            "quad_score": float(np.mean(qs)),
            "quad_score_ci": [float(np.mean(qs) - 1.96 * np.std(qs, ddof=1) / a.seeds ** 0.5),
                              float(np.mean(qs) + 1.96 * np.std(qs, ddof=1) / a.seeds ** 0.5)],
            "U_index_by_grid": {k: (float(np.nanmean(v)) if not np.all(np.isnan(v)) else None)
                                for k, v in uidx.items()},
            "seconds": round(time.time() - t0)}
        print(f"  {arm:<11} dim {POOL.shape[1]:>5}  quad(score) {out[arm]['quad_score']:>+9.3f}  "
              f"quad(flag) {out[arm]['quad_flag']:>+8.4f}  "
              f"U as published {out[arm]['U_index_by_grid']['as published']:.2f}")

    arms_o = [k for k, _ in ARMS]
    lp = [np.log10(out[k]["params"]) for k in arms_o]
    mag = [abs(out[k]["quad_score"]) for k in arms_o]
    rho = M97.spearman(lp, mag)
    pp = (sum(1 for z in permutations(mag) if M97.spearman(lp, z) <= rho + 1e-12)
          / len(list(permutations(mag)))) if rho < 0 else None
    all_pos = all(out[k]["quad_score"] > 0 for k in arms_o)
    pos_650 = out["esm2_650M"]["quad_score"] > 0
    u1 = ("the U is not a cut-point artifact" if all_pos
          else "partial — holds where measured, not generally" if pos_650
          else "the U as reported is an artifact of the bands")
    u2 = ("study P's size trend survives without bands" if rho <= -1.0
          else "consistent, not strict" if rho <= -0.6 else "the trend was a band artifact")
    grid_vals = [v for v in out["esm2_650M"]["U_index_by_grid"].values() if v is not None]
    dup = len({round(out[k]["quad_score"], 3) for k in arms_o}) != len(arms_o)

    res = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "seeds": a.seeds,
           "n_intracellular": int(len(ins)), "per_arm": out,
           "U1_all_positive": bool(all_pos), "U1_650M_positive": bool(pos_650),
           "U1_verdict": u1, "U2_rho": rho, "U2_perm_p": pp, "U2_verdict": u2,
           "U3_650M_range": [min(grid_vals), max(grid_vals)],
           "U3_650M_min_above_3": bool(min(grid_vals) > 3.0),
           "U3_grids": list(grids), "duplicate_quad": bool(dup)}
    OUT.write_text(json.dumps(res, indent=2) + "\n")

    print(f"\nU-1  quadratic on the score positive on all four arms: {all_pos}  ->  {u1}")
    print(f"U-2  rho = {rho:+.2f}" + (f", one-tailed p = {pp:.4f}" if pp else "") + f"  ->  {u2}")
    print(f"\nU-3  650M U-index across {len(grids)} band grids:")
    for k, v in out["esm2_650M"]["U_index_by_grid"].items():
        print(f"     {k:<14}{'n/a' if v is None else f'{v:8.2f}'}")
    print(f"     range [{min(grid_vals):.2f}, {max(grid_vals):.2f}]; minimum "
          f"{'above' if min(grid_vals) > 3.0 else 'BELOW'} 3.0")
    if dup:
        print("\n🔴 two arms share a quadratic coefficient — an embedding was fitted twice")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
