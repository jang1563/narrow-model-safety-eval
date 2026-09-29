#!/usr/bin/env python3
"""
96_score_decomposition.py - is the length-U a magnitude effect or a direction effect?

`docs/SCORE_DECOMPOSITION_PREREGISTRATION.md`. Study M found the flag rate is U-shaped in protein
length (intracellular U-index 8.38); study N eliminated the special-token displacement, which is real
but orthogonal to the probe; and proximity to the class-axis positives was checked and refuted before
this study was frozen -- the maximum cosine to a positive is LOWEST in the band where the flag rate is
highest.

The probe is linear on standardized features, so the score factors exactly:

    score = ||z|| * cos(z, w) * ||w||

and the two factors do not move together with length. This holds each at its pooled median in turn and
asks which one reproduces the U.

🔒 The threshold is recalibrated under each counterfactual, at the same 95th percentile of the same
calibration rows, so every version keeps a 5% nominal rate and the U-indices are comparable.

Usage:
    python src/96_score_decomposition.py --arm esm2_650M
    python src/96_score_decomposition.py --arm esm2_35M
    python src/96_score_decomposition.py --selftest
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
OUT_STEM = ROOT / "results" / "score_decomposition"
SEEDS, SPEC = 30, 0.95
SHORT, MID = (0, 250), (450, 550)
BAND_FLOOR, HI, LO = 100, 4.0, 2.0


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


def factors(Z, w):
    """||z||, cos(z, w) and the score, for standardized rows Z and coefficient vector w."""
    zn = np.linalg.norm(Z, axis=1)
    wn = np.linalg.norm(w)
    score = Z @ w
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = np.where(zn > 0, score / (zn * wn), 0.0)
    return zn, cos, score


def selftest():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(200, 8))
    w = rng.normal(size=8)
    zn, cos, score = factors(Z, w)
    # 🔒 the factorisation must be exact, or every counterfactual below is meaningless
    assert np.allclose(zn * cos * np.linalg.norm(w), score, atol=1e-10), "score must factor exactly"
    # 🔒 holding a factor at its median must change the score unless the factor is constant
    cf = np.median(zn) * cos * np.linalg.norm(w)
    assert not np.allclose(cf, score), "a counterfactual that changes nothing is a bug"
    # and holding BOTH reproduces a constant
    both = np.median(zn) * np.median(cos) * np.linalg.norm(w)
    assert np.isscalar(both) or both.ndim == 0
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
    ins = np.array(strat["intracellular"])
    lengths = np.array([pool[test[j]]["length"] for j in range(len(test))], float)
    short = ins[(lengths[ins] >= SHORT[0]) & (lengths[ins] < SHORT[1])]
    mid = ins[(lengths[ins] >= MID[0]) & (lengths[ins] < MID[1])]
    print(f"arm {a.arm}: intracellular short {len(short)}, mid {len(mid)}")
    floors_met = min(len(short), len(mid)) >= BAND_FLOOR

    sfx = "" if a.arm == "esm2_650M" else f"_{a.arm}"
    P = np.load(RES / f"embeddings_class_axis_positives{sfx}.npy")
    POOL = np.load(RES / f"embeddings_pool_large_{a.arm}.npy")
    vf = M83.vfdb_sequences()
    union = [i for i in json.loads(SCREEN.read_text())["admitted_rows"]
             if pool[i]["sequence"] not in vf]
    n_tr, n_ca = M92.fold_sizes(len(union))
    y = np.r_[np.ones(P.shape[0]), np.zeros(n_tr)]
    rows = np.array(test)

    keys = ("observed", "direction_only", "magnitude_only")
    acc = {k: {"short": [], "mid": []} for k in keys}
    comp = {"zn_short": [], "zn_mid": [], "cos_short": [], "cos_mid": []}
    t0 = time.time()
    for seed in range(a.seeds):
        r = np.random.default_rng(seed)
        perm = r.permutation(len(union))
        tr = [union[i] for i in perm[:n_tr]]
        ca = [union[i] for i in perm[n_tr:n_tr + n_ca]]
        pipe = clf().fit(np.vstack([P, POOL[tr]]), y)
        sc = pipe.named_steps["standardscaler"]
        w = pipe.named_steps["logisticregression"].coef_[0]
        wn = np.linalg.norm(w)
        Zt, Zc = sc.transform(POOL[rows]), sc.transform(POOL[ca])
        znt, cost, st = factors(Zt, w)
        znc, cosc, scc = factors(Zc, w)
        # 🔒 medians are taken over the ELIGIBLE population, once, and used for both test and calib
        zbar, cbar = float(np.median(znt[elig])), float(np.median(cost[elig]))
        variants = {
            "observed": (st, scc),
            "direction_only": (zbar * cost * wn, zbar * cosc * wn),
            "magnitude_only": (znt * cbar * wn, znc * cbar * wn),
        }
        for k, (s_test, s_cal) in variants.items():
            t = float(np.quantile(s_cal, SPEC))     # recalibrated, same 5% nominal rate
            f = s_test >= t
            acc[k]["short"].append(float(f[short].mean()))
            acc[k]["mid"].append(float(f[mid].mean()))
        comp["zn_short"].append(float(znt[short].mean()))
        comp["zn_mid"].append(float(znt[mid].mean()))
        comp["cos_short"].append(float(cost[short].mean()))
        comp["cos_mid"].append(float(cost[mid].mean()))
    print(f"  {time.time() - t0:.0f}s")

    out = {}
    for k in keys:
        s_, m_ = float(np.mean(acc[k]["short"])), float(np.mean(acc[k]["mid"]))
        out[k] = {"short": s_, "mid": m_, "U_index": s_ / m_ if m_ > 0 else float("nan")}
    d, m = out["direction_only"]["U_index"], out["magnitude_only"]["U_index"]
    verdict = ("the U is directional" if d >= HI and m < LO
               else "the U is a magnitude effect" if m >= HI and d < LO
               else "both contribute" if min(d, m) >= 2.5
               else "uninterpretable — the decomposition reproduces neither arm")
    res = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "arm": a.arm, "seeds": a.seeds,
           "n_short": len(short), "n_mid": len(mid), "floors_met": bool(floors_met),
           "variants": out, "verdict": verdict,
           "components": {k: float(np.mean(v)) for k, v in comp.items()},
           "reproduces_observed": bool(abs(d - out["observed"]["U_index"]) > 0.01
                                       and abs(m - out["observed"]["U_index"]) > 0.01)}
    dest = Path(f"{OUT_STEM}{sfx}.json")
    dest.write_text(json.dumps(res, indent=2) + "\n")

    print(f"\n{'variant':<20}{'short':>9}{'mid':>9}{'U-index':>10}")
    for k in keys:
        print(f"  {k:<18}{out[k]['short'] * 100:>8.2f}%{out[k]['mid'] * 100:>8.2f}%"
              f"{out[k]['U_index']:>10.2f}")
    print(f"\n  components: ||z|| short {np.mean(comp['zn_short']):.2f} vs mid "
          f"{np.mean(comp['zn_mid']):.2f};  cos short {np.mean(comp['cos_short']):.4f} vs mid "
          f"{np.mean(comp['cos_mid']):.4f}")
    print(f"\nO-1  {verdict}")
    if not res["reproduces_observed"]:
        print("  🔴 a counterfactual returned the observed U-index — the factor was not replaced")
    print(f"\nwrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
