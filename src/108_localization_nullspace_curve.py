#!/usr/bin/env python3
"""
108_localization_nullspace_curve.py - Y-5: drop the stopping rule, report the curve

`docs/LOCALIZATION_NULLSPACE_PREREGISTRATION.md` § 10. Study Y's gate stopped at the first iteration
where a noisy inner-holdout AUROC dipped below 0.55 -- a minimum over noise, therefore optimistic. It was
met on all four runs while localization stayed separable at 0.778-0.890 on rows the axis never saw, and
Y-1's R stayed at 4.3-4.7.

🔒 Y-5 removes the stopping rule, which is the thing that failed. The iterative procedure runs to the
largest grid value, giving a NESTED direction sequence, so every prefix k is read off one pass and nothing
is selected. At each k: unseen-row localization AUROC, R, and CA with rho. k* is the smallest grid k whose
unseen-row AUROC is <= 0.55 on BOTH folds, and Y-1's and Y-2's frozen bands apply there at 30 seeds.

🔴 If no grid k reaches 0.55, that is the result -- the localization axis is not linearly removable within
d/4 directions -- and it is reported as a finding, not as a failed run.

⚠️ Removing a direction from an embedding is a measurement technique. Nothing here is a mitigation.

Usage:
    python src/108_localization_nullspace_curve.py --arm esm2_650M
    python src/108_localization_nullspace_curve.py --arm esm2_35M
    python src/108_localization_nullspace_curve.py --selftest
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
OUT_STEM = ROOT / "results" / "localization_nullspace_curve"

# 🔒 frozen in § 10
K_GRID = (0, 2, 5, 10, 20, 40, 80, 160)
DIM_CAP = 0.25
CURVE_SEEDS, FINAL_SEEDS = 10, 30


def _load(stem):
    spec = importlib.util.spec_from_file_location(f"_{stem}", ROOT / "src" / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M107 = _load("107_localization_nullspace")
M99, M92, M84, M83, M91 = M107.M99, M107.M92, M107.M84, M107.M83, M107.M91
GATE, Y1_REMOVED, Y1_PARTLY = M107.GATE_AUROC, M107.Y1_REMOVED, M107.Y1_PARTLY
Y2_SURVIVES, Y2_PARTIAL, Y3_CLEAN = M107.Y2_SURVIVES, M107.Y2_PARTIAL, M107.Y3_CLEAN
FOLD_SEED, SPEC, FLOOR = M107.FOLD_SEED, M107.SPEC, M107.FLOOR
inner_split, loc_auroc, project_out = M107.inner_split, M107.loc_auroc, M107.project_out
random_basis, rho = M107.random_basis, M107.rho


def direction_sequence(X, lab, tr, k_max):
    """The nested direction sequence, run to k_max with NO stopping rule.

    🔴 The stopping rule is the thing study Y's gate failed on, so it is absent here by design: every
    prefix of the returned array is the k-direction projection, and k is chosen afterwards against a
    measurement made on rows this never sees.
    """
    Z = X.copy()
    rows = []
    for _ in range(k_max):
        w = LogisticRegression(max_iter=5000, C=1.0).fit(Z[tr], lab[tr]).coef_[0].astype(float)
        for u in rows:
            w -= (w @ u) * u
        nw = np.linalg.norm(w)
        if nw < 1e-9:
            break
        u = w / nw
        rows.append(u)
        Z = Z - np.outer(Z @ u, u)
    return np.array(rows) if rows else np.zeros((0, X.shape[1]))


def selftest():
    rng = np.random.default_rng(0)
    d, n, kk = 30, 600, 6
    lab = (rng.random(n) < 0.5).astype(int)
    axis = rng.normal(size=(4, d))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    X = rng.normal(size=(n, d)) + np.outer(lab * 3.0, axis.sum(0))
    tr, ho = inner_split(n, 1)
    U = direction_sequence(X, lab, tr, kk)
    assert U.shape == (kk, d), U.shape
    assert np.allclose(U @ U.T, np.eye(kk), atol=1e-8), "rows must be orthonormal"
    # 🔒 the prefix property is what makes one pass serve the whole grid
    U3 = direction_sequence(X, lab, tr, 3)
    assert np.allclose(U3, U[:3], atol=1e-10), "a shorter run must be a prefix of a longer one"
    # 🔒 and the curve must be monotone-ish downward on a planted axis: more removal, less AUROC
    aus = [loc_auroc(project_out(X, U[:k]), lab, tr, ho) for k in (0, 2, 4, 6)]
    assert aus[0] > 0.95 and aus[-1] < aus[0] - 0.3, aus
    # 🔴 no stopping rule means k_max directions come back even with nothing to find -- the caller's
    # read-off rule is what handles that, and this asserts the helper does not silently stop early
    Un = direction_sequence(rng.normal(size=(n, d)), lab, tr, kk)
    assert Un.shape[0] == kk, Un.shape
    print("selftest PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="esm2_650M", choices=["esm2_650M", "esm2_35M"])
    ap.add_argument("--labels", default="coverage", choices=["keyword", "coverage"])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    RES = M107.RES
    pool = json.loads(M107.POOL_JSON.read_text())["proteins"]
    test = json.loads(M107.BUILD.read_text())["pool_partition"]["test_rows"]
    accs = [pool[i]["uniprot"] for i in test]
    kw = M84.fetch_keywords(accs)
    vf_seqs, vfdb_species = M83.vfdb_sequences(), M83.vfdb_species()

    bcell = {"extracellular": [], "intracellular": []}
    for k_, i in enumerate(test):
        sp = " ".join(pool[i]["organism"].replace("(", "").split()[:2])
        if pool[i]["kingdom"] != "Bacteria" or sp not in vfdb_species:
            continue
        if pool[i]["sequence"] in vf_seqs:
            continue
        st = M84.stratum_of(kw[pool[i]["uniprot"]])
        if st in bcell:
            bcell[st].append(k_)
    if a.labels == "keyword":
        axis_cell = bcell
    else:
        go = M92.fetch_go(accs)
        axis_cell = {"extracellular": [], "intracellular": []}
        for k_, i in enumerate(test):
            if pool[i]["kingdom"] not in M84.KINGDOMS or pool[i]["sequence"] in vf_seqs:
                continue
            g = go[accs[k_]]
            st = M92.stratum_of(g["go"], g["sig"], g["tm"], kw[accs[k_]])
            if st in axis_cell:
                axis_cell[st].append(k_)

    admitted = json.loads(M107.NEGSET.read_text())["admitted"]
    len_of = {r["vfg"]: r["len"] for r in admitted}
    row_of = {r["vfg"]: i for i, r in enumerate(admitted)}
    noncirc = M83.vfdb_noncircular_mask()
    up_len, acc_of = M91.uniprot_lengths(), M91.vfg_accessions()
    keep = {}
    for vfg, info in json.loads(M107.MAPPED.read_text())["chosen"].items():
        if not noncirc[row_of[vfg]]:
            continue
        ul = up_len.get((acc_of[vfg], info["uniprot"]))
        if ul is None or str(len_of[vfg]) != str(ul):
            continue
        keep[vfg] = info
    vrows = {"all": [row_of[v] for v, i in keep.items() if i["stratum"] == "extracellular"],
             "non_hazardous": [row_of[v] for v, i in keep.items()
                               if i["stratum"] == "extracellular" and i["category"] in M107.NON_HAZ],
             "partly_hazardous": [row_of[v] for v, i in keep.items()
                                  if i["stratum"] == "extracellular"
                                  and i["category"] in M107.PARTLY],
             "motility": [row_of[v] for v, i in keep.items()
                          if i["stratum"] == "extracellular" and i["category"] == "Motility"]}

    sfx = "" if a.arm == "esm2_650M" else f"_{a.arm}"
    P = np.load(RES / f"embeddings_class_axis_positives{sfx}.npy")
    POOL = np.load(RES / f"embeddings_pool_large_{a.arm}.npy")
    VFDB = np.load(RES / f"embeddings_vfdb_neg_{a.arm}.npy")
    union = [i for i in json.loads(M107.SCREEN.read_text())["admitted_rows"]
             if pool[i]["sequence"] not in vf_seqs]
    n_tr, n_ca = M92.fold_sizes(len(union))
    y = np.r_[np.ones(P.shape[0]), np.zeros(n_tr)]
    T = POOL[np.array(test)]
    d = T.shape[1]
    grid = [k for k in K_GRID if k <= int(DIM_CAP * d)]
    print(f"arm {a.arm} ({d} dims), axis labels {a.labels} "
          f"{len(axis_cell['extracellular'])}/{len(axis_cell['intracellular'])}, grid {grid}")

    lab_rows = np.array(axis_cell["extracellular"] + axis_cell["intracellular"])
    lab = np.r_[np.ones(len(axis_cell["extracellular"])),
                np.zeros(len(axis_cell["intracellular"]))].astype(int)
    order = np.random.default_rng(FOLD_SEED).permutation(len(lab_rows))
    halves = [order[:len(order) // 2], order[len(order) // 2:]]
    key_ex, key_in = set(bcell["extracellular"]), set(bcell["intracellular"])

    def measure(B, seeds, m_ex, m_in):
        """Flag rates under a projection. Returns (R, CA per group) averaged over seeds."""
        Pp, POOLp, Tp, Vp = (project_out(x, B) for x in (P, POOL, T, VFDB))
        Rs, CAs = [], {g: [] for g in vrows}
        for seed in range(seeds):
            r = np.random.default_rng(seed)
            perm = r.permutation(len(union))
            tr = [union[i] for i in perm[:n_tr]]
            ca = [union[i] for i in perm[n_tr:n_tr + n_ca]]
            mdl = M99.clf().fit(np.vstack([Pp, POOLp[tr]]), y)
            t = float(np.quantile(mdl.predict_proba(POOLp[ca])[:, 1], SPEC))
            fb = mdl.predict_proba(Tp)[:, 1] >= t
            fv = mdl.predict_proba(Vp)[:, 1] >= t
            be, bi = float(fb[m_ex].mean()), float(fb[m_in].mean())
            Rs.append(be / bi if bi > 0 else np.nan)
            for g, rows in vrows.items():
                CAs[g].append(float(fv[rows].mean()) / be if be > 0 else np.nan)
        return Rs, CAs

    t0 = time.time()
    per_fold = []
    for f in (0, 1):
        fit_i = halves[f]
        Xf, yf = T[lab_rows[fit_i]], lab[fit_i]
        itr, _ = inner_split(len(yf), FOLD_SEED + f)
        U = direction_sequence(Xf, yf, itr, max(grid))
        fitted = set(lab_rows[fit_i].tolist())
        m_ex = np.array(sorted(key_ex - fitted))
        m_in = np.array(sorted(key_in - fitted))
        d_rows = np.r_[m_ex, m_in]
        d_lab = np.r_[np.ones(len(m_ex)), np.zeros(len(m_in))].astype(int)
        dtr, dho = inner_split(len(d_lab), FOLD_SEED + 200 + f)
        rows = {}
        for k in grid:
            B = U[:k]
            au = loc_auroc(project_out(T[d_rows], B), d_lab, dtr, dho)
            Rs, CAs = measure(B, CURVE_SEEDS, m_ex, m_in)
            rows[k] = {"auroc": au, "R": float(np.nanmean(Rs)),
                       "CA": {g: float(np.nanmean(v)) for g, v in CAs.items()}}
            print(f"  fold {f} k={k:>3}: unseen AUROC {au:.4f}  R {rows[k]['R']:.3f}  "
                  f"CA {rows[k]['CA']['all']:.3f}   {time.time() - t0:.0f}s")
        per_fold.append({"k_available": int(U.shape[0]), "rows": rows, "U": U,
                         "m_ex": m_ex, "m_in": m_in, "dtr": dtr, "dho": dho,
                         "d_rows": d_rows, "d_lab": d_lab})

    # ---- the frozen read-off rule ----------------------------------------------------------------
    kstar = next((k for k in grid
                  if all(pf["rows"][k]["auroc"] <= GATE for pf in per_fold)), None)
    ca_orig = M107.CA_ORIGINAL[a.arm]
    res = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "arm": a.arm, "dim": int(d),
           "labels": a.labels, "grid": grid, "curve_seeds": CURVE_SEEDS,
           "n_axis_labels": {k_: len(v) for k_, v in axis_cell.items()},
           "n_vfdb_extracellular": {g: len(v) for g, v in vrows.items()},
           "floors_met": {g: len(v) >= FLOOR for g, v in vrows.items()},
           "CA_original": ca_orig,
           "curve": [{"k": k,
                      "auroc": [round(pf["rows"][k]["auroc"], 4) for pf in per_fold],
                      "R": [round(pf["rows"][k]["R"], 4) for pf in per_fold],
                      "CA": [round(pf["rows"][k]["CA"]["all"], 4) for pf in per_fold],
                      "rho": [round(rho(pf["rows"][k]["CA"]["all"], ca_orig), 4)
                              for pf in per_fold]} for k in grid],
           "k_star": kstar,
           "removable_within_cap": kstar is not None,
           "min_auroc_reached": round(min(max(pf["rows"][k]["auroc"] for pf in per_fold)
                                          for k in grid), 4)}

    if kstar is not None:
        Rs, CAs, Rr, CAr = [], {g: [] for g in vrows}, [], {g: [] for g in vrows}
        for pf in per_fold:
            r1, c1 = measure(pf["U"][:kstar], FINAL_SEEDS, pf["m_ex"], pf["m_in"])
            Br = random_basis(d, kstar, np.random.default_rng(FOLD_SEED + 300))
            r2, c2 = measure(Br, FINAL_SEEDS, pf["m_ex"], pf["m_in"])
            Rs += r1
            Rr += r2
            for g in vrows:
                CAs[g] += c1[g]
                CAr[g] += c2[g]
        S = M99.stat
        res["final"] = {
            "seeds": FINAL_SEEDS,
            "Y1_R": S(Rs), "Y1_R_control": S(Rr),
            "Y2_CA": S(CAs["all"]), "Y2_rho": round(rho(S(CAs["all"])["mean"], ca_orig), 4),
            "Y3_CA": S(CAr["all"]), "Y3_rho": round(rho(S(CAr["all"])["mean"], ca_orig), 4),
            "Y4_rho": {g: round(rho(S(CAs[g])["mean"], ca_orig), 4)
                       for g in ("non_hazardous", "partly_hazardous", "motility")}}
        fin = res["final"]
        fin["Y4_delta"] = round(fin["Y4_rho"]["non_hazardous"]
                                - fin["Y4_rho"]["partly_hazardous"], 4)
        fin["Y1_band"] = ("the gradient is removed" if fin["Y1_R"]["mean"] <= Y1_REMOVED
                          else "partly removed" if fin["Y1_R"]["mean"] <= Y1_PARTLY
                          else "the projection did not remove the gradient")
        fin["Y2_band"] = ("membership survives localization removal"
                          if fin["Y2_rho"] >= Y2_SURVIVES
                          else "partial" if fin["Y2_rho"] >= Y2_PARTIAL
                          else "the membership contrast was largely the localization axis")
        fin["Y3_control_clean"] = bool(fin["Y3_rho"] >= Y3_CLEAN)
        fin["verdict_eligible"] = bool(fin["Y3_control_clean"])

    dest = Path(f"{OUT_STEM}{sfx}.json")
    dest.write_text(json.dumps({k: v for k, v in res.items()}, indent=2) + "\n")

    print(f"\n{'k':>5}{'unseen AUROC':>26}{'R':>18}{'rho':>18}")
    print("-" * 67)
    for row in res["curve"]:
        print(f"{row['k']:>5}   {row['auroc'][0]:.4f} / {row['auroc'][1]:.4f}   "
              f"  {row['R'][0]:6.3f} / {row['R'][1]:6.3f}     "
              f"{row['rho'][0]:6.3f} / {row['rho'][1]:6.3f}")
    if kstar is None:
        print(f"\n🔴 no grid k reaches {GATE} on both folds (best {res['min_auroc_reached']:.4f}) "
              f"-> the localization axis is NOT linearly removable within {int(DIM_CAP * d)} directions")
    else:
        fin = res["final"]
        print(f"\nk* = {kstar}  ({kstar / d * 100:.1f}% of {d} dims), 30 seeds")
        print(f"Y-1  R = {fin['Y1_R']['mean']:.3f} "
              f"[{fin['Y1_R']['ci'][0]:.3f}, {fin['Y1_R']['ci'][1]:.3f}]  "
              f"(control {fin['Y1_R_control']['mean']:.3f})  ->  {fin['Y1_band']}")
        print(f"Y-2  CA = {fin['Y2_CA']['mean']:.3f} against {ca_orig}, rho = {fin['Y2_rho']:.3f}"
              f"  ->  {fin['Y2_band']}")
        print(f"Y-3  control rho = {fin['Y3_rho']:.3f}  ->  "
              f"{'clean' if fin['Y3_control_clean'] else '🔴 CONFOUNDED'}")
        print(f"Y-4  non-haz {fin['Y4_rho']['non_hazardous']:.3f}, partly "
              f"{fin['Y4_rho']['partly_hazardous']:.3f}, motility "
              f"{fin['Y4_rho']['motility']:.3f}  (delta {fin['Y4_delta']:+.3f})")
    print(f"\nwrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
