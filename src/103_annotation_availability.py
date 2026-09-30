#!/usr/bin/env python3
"""
103_annotation_availability.py - why are unannotated proteins flagged at half the rate?

`docs/ANNOTATION_AVAILABILITY_PREREGISTRATION.md`. Study E froze an availability check -- the
unannotated stratum's flag rate over the annotated rate must sit in [0.67, 1.5] -- and it has failed on
every version run: 0.41 and 0.32 under study E's keyword strata, 0.48 and 0.52 under study L's broader
ones. Doubling the annotated fraction did not fix it. It has been reported four times and explained
none, and it sits underneath every localization number here.

🔑 The declared diagnostic points at length: the share over 700 residues is 14.7% among annotated
proteins and 7.3% among unannotated, a twofold difference, and study M established the flag rate is
U-shaped in length. Organism curation depth is eliminated -- 12 proteins per organism against 11.

T-1 adjusts for length by Mantel-Haenszel over deciles. T-2 asks whether the residual is magnitude or
direction, by study O's decomposition. T-3 tests an uncomfortable hypothesis: that a protein UniProt has
localized is one that has been studied, so the probe may be reading family recognisability.

⚠️ No new probe and no new inference.

Usage:
    python src/103_annotation_availability.py --arm esm2_650M
    python src/103_annotation_availability.py --arm esm2_35M
    python src/103_annotation_availability.py --selftest
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
OUT_STEM = ROOT / "results" / "annotation_availability"
SEEDS, SPEC = 30, 0.95
N_DECILES, CELL_FLOOR, MIN_DECILES, KNN = 10, 20, 6, 10
BAND_OK, BAND_PART = 0.67, 0.50


def _load(stem):
    spec = importlib.util.spec_from_file_location(f"_{stem}", ROOT / "src" / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M92 = _load("92_localization_coverage")
M84 = M92.M84
M83 = _load("83_provenance_control")
M93 = _load("93_length_adjustment")
M96 = _load("96_score_decomposition")


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))


def stat(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return {"mean": float(v.mean()),
            "ci": [float(v.mean() - 1.96 * v.std(ddof=1) / len(v) ** 0.5),
                   float(v.mean() + 1.96 * v.std(ddof=1) / len(v) ** 0.5)]}


def selftest():
    # 🔒 Mantel-Haenszel is imported from src/93, not reimplemented, so the two studies pool the same way
    assert abs(M93.mantel_haenszel_rr([(20, 100, 10, 100), (40, 100, 20, 100)]) - 2.0) < 1e-12
    # 🔴 The first version of this assertion ended in `or True`, which cannot fail. The real
    # property to check is that src/96's factorisation is exact, because T-2's counterfactuals are
    # meaningless otherwise.
    rng = np.random.default_rng(0)
    Z, w = rng.normal(size=(50, 6)), rng.normal(size=6)
    zn, cos, score = M96.factors(Z, w)
    assert np.allclose(zn * cos * np.linalg.norm(w), score, atol=1e-10), \
        "src/96's factorisation must be exact or T-2 means nothing"
    assert not np.allclose(np.median(zn) * cos * np.linalg.norm(w), score), \
        "holding the magnitude at its median must change the score"
    assert M92.stratum_of("", "", "", set()) == "unannotated"
    assert M92.stratum_of("", "", "", {"Cytoplasm"}) == "intracellular"
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
    kw, go = M84.fetch_keywords(accs), M92.fetch_go(accs)
    elig = [j for j, i in enumerate(test) if pool[i]["kingdom"] in M84.KINGDOMS]
    ann, una = [], []
    for j in elig:
        g = go[accs[j]]
        (una if M92.stratum_of(g["go"], g["sig"], g["tm"], kw[accs[j]]) == "unannotated"
         else ann).append(j)
    lengths = np.array([pool[test[j]]["length"] for j in range(len(test))], float)
    print(f"arm {a.arm}: annotated {len(ann)}, unannotated {len(una)}")

    both = np.array(ann + una)
    edges = np.quantile(lengths[both], np.linspace(0, 1, N_DECILES + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    dec = {j: min(max(int(np.searchsorted(edges, lengths[j], side="right") - 1), 0),
                  N_DECILES - 1) for j in both}
    a_by = {d: [j for j in ann if dec[j] == d] for d in range(N_DECILES)}
    u_by = {d: [j for j in una if dec[j] == d] for d in range(N_DECILES)}
    used = [d for d in range(N_DECILES)
            if len(a_by[d]) >= CELL_FLOOR and len(u_by[d]) >= CELL_FLOOR]
    print(f"  deciles contributing: {len(used)}/{N_DECILES}")

    sfx = "" if a.arm == "esm2_650M" else f"_{a.arm}"
    P = np.load(RES / f"embeddings_class_axis_positives{sfx}.npy")
    POOL = np.load(RES / f"embeddings_pool_large_{a.arm}.npy")
    vf = M83.vfdb_sequences()
    union = [i for i in json.loads(SCREEN.read_text())["admitted_rows"]
             if pool[i]["sequence"] not in vf]
    n_tr, n_ca = M92.fold_sizes(len(union))
    y = np.r_[np.ones(P.shape[0]), np.zeros(n_tr)]
    rows = np.array(test)

    crude, mh = [], []
    cf = {k: {"ann": [], "una": []} for k in ("observed", "direction_only", "magnitude_only")}
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
        znt, cost, st = M96.factors(Zt, w)
        znc, cosc, scc = M96.factors(Zc, w)
        zbar, cbar = float(np.median(znt[elig])), float(np.median(cost[elig]))
        for name, (s_test, s_cal) in {
                "observed": (st, scc),
                "direction_only": (zbar * cost * wn, zbar * cosc * wn),
                "magnitude_only": (znt * cbar * wn, znc * cbar * wn)}.items():
            t = float(np.quantile(s_cal, SPEC))
            f = s_test >= t
            cf[name]["ann"].append(float(f[ann].mean()))
            cf[name]["una"].append(float(f[una].mean()))
            if name == "observed":
                crude.append(float(f[una].mean() / f[ann].mean()) if f[ann].mean() > 0 else np.nan)
                mh.append(M93.mantel_haenszel_rr(
                    [(float(f[u_by[d]].sum()), len(u_by[d]),
                      float(f[a_by[d]].sum()), len(a_by[d])) for d in used]))
    print(f"  {time.time() - t0:.0f}s")

    # ---- T-3: embedding-space density, no probe involved ------------------------------------------
    X = POOL[rows]
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    sub = Xn[both]
    sims = sub @ sub.T
    np.fill_diagonal(sims, -1.0)
    knn = np.sort(sims, axis=1)[:, -KNN:].mean(1)
    is_ann = np.isin(both, np.array(ann))
    dens = {"annotated": float(knn[is_ann].mean()), "unannotated": float(knn[~is_ann].mean())}

    C, MH = stat(crude), stat(mh)
    band = ("length composition explains it" if MH["mean"] >= BAND_OK
            else "length explains part of it" if MH["mean"] >= BAND_PART
            else "length explains essentially none of it")
    ratios = {k: float(np.mean(v["una"]) / np.mean(v["ann"])) if np.mean(v["ann"]) > 0 else float("nan")
              for k, v in cf.items()}
    res = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "arm": a.arm, "seeds": a.seeds,
           "n_annotated": len(ann), "n_unannotated": len(una),
           "n_deciles_used": len(used), "verdict_eligible": bool(len(used) >= MIN_DECILES),
           "crude_ratio": C, "T1_mantel_haenszel": MH, "band": band,
           "explained_fraction": float((MH["mean"] - C["mean"]) / (1.0 - C["mean"])),
           "T2_counterfactual_ratios": ratios,
           "T2_rates": {k: {"ann": float(np.mean(v["ann"])), "una": float(np.mean(v["una"]))}
                        for k, v in cf.items()},
           "T3_knn_cosine": dens, "T3_delta": dens["annotated"] - dens["unannotated"],
           "knn": KNN,
           "stratified": bool(abs(MH["mean"] - C["mean"]) > 1e-3)}
    dest = Path(f"{OUT_STEM}{sfx}.json")
    dest.write_text(json.dumps(res, indent=2) + "\n")

    print(f"\nT-1  crude unannotated/annotated = {C['mean']:.3f} "
          f"[{C['ci'][0]:.3f}, {C['ci'][1]:.3f}]")
    print(f"     length-adjusted (Mantel-Haenszel) = {MH['mean']:.3f} "
          f"[{MH['ci'][0]:.3f}, {MH['ci'][1]:.3f}]  ->  {band}")
    print(f"     that closes {res['explained_fraction'] * 100:.0f}% of the distance to parity")
    print(f"T-2  counterfactual ratios: observed {ratios['observed']:.3f}, "
          f"direction-only {ratios['direction_only']:.3f}, "
          f"magnitude-only {ratios['magnitude_only']:.3f}")
    print(f"T-3  mean cosine to {KNN} nearest pool neighbours: annotated "
          f"{dens['annotated']:.4f}, unannotated {dens['unannotated']:.4f} "
          f"(delta {res['T3_delta']:+.4f})")
    if not res["stratified"]:
        print("     🔴 the adjusted ratio equals the crude one — the stratification did not happen")
    print(f"\nwrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
