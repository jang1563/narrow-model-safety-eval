#!/usr/bin/env python3
"""
109_leace_erasure.py - Z: erase localization with a guarantee, and measure the tail separately

`docs/LEACE_ERASURE_PREREGISTRATION.md`. Study Y ran INLP and it did not remove localization -- 160 of
1,280 directions left a refitted classifier at AUROC 0.698 and R above 3.35 -- and entry 78 retired the
conclusion, because INLP's under-erasure is published and LEACE (Belrose et al., NeurIPS 2023) provably
prevents every linear classifier from recovering a binary concept, in closed form.

🔒 For a binary concept the eraser is rank-1 and oblique: r(x) = x - A (x - mu) with
A = S^(1/2) u u^T S^(-1/2) and u proportional to S^(-1/2) Sxz. No iterations, NO STOPPING RULE -- the
thing that broke study Y's gate.

🔑 The point of the study is that LEACE equalizes class-conditional MEANS while the probe's flag is a
95th-percentile TAIL event. Those are different objects (entry 76), and study X showed dispersion drives
tails at equal means. So R and the localization AUROC are reported side by side at every step.

⚠️ A provable erasure that leaves R high is evidence AGAINST erasure as a fix. Nothing here is a
mitigation.

Usage:
    python src/109_leace_erasure.py --arm esm2_650M
    python src/109_leace_erasure.py --arm esm2_35M
    python src/109_leace_erasure.py --selftest
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_STEM = ROOT / "results" / "leace_erasure"

# 🔒 frozen in § 3 - § 7
Z0_TOL = 0.02
Z0_GEN, Z0_MARGINAL = 0.55, 0.60
EIG_FLOOR = 1e-10


def _load(stem):
    spec = importlib.util.spec_from_file_location(f"_{stem}", ROOT / "src" / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M108 = _load("108_localization_nullspace_curve")
M107, M99, M92, M84, M83, M91 = M108.M107, M108.M99, M108.M92, M108.M84, M108.M83, M108.M91
inner_split, loc_auroc, rho = M107.inner_split, M107.loc_auroc, M107.rho
Y1_REMOVED, Y1_PARTLY = M107.Y1_REMOVED, M107.Y1_PARTLY
Y2_SURVIVES, Y2_PARTIAL, Y3_CLEAN, Y4_SAME = (M107.Y2_SURVIVES, M107.Y2_PARTIAL,
                                              M107.Y3_CLEAN, M107.Y4_SAME)
FOLD_SEED, SPEC, FLOOR, SEEDS = M107.FOLD_SEED, M107.SPEC, M107.FLOOR, M99.SEEDS


def sym_powers(X):
    """(S^(1/2), S^(-1/2)) of the covariance of X, on its numerically supported subspace."""
    Xc = X - X.mean(0)
    S = (Xc.T @ Xc) / (len(Xc) - 1)
    w, V = np.linalg.eigh(S)
    keep = w > w.max() * EIG_FLOOR
    w, V = w[keep], V[:, keep]
    return (V * np.sqrt(w)) @ V.T, (V / np.sqrt(w)) @ V.T


def leace(X_cov, X_lab, lab, direction=None, whiten=True):
    """The rank-1 LEACE eraser. Returns (A, mu) for r(x) = x - (x - mu) @ A.T.

    🔒 `X_cov` supplies the covariance and carries NO labels, so it may be the whole pool; `X_lab` and
    `lab` supply the concept direction and come from the fitting fold only. `direction` overrides the
    concept direction, which is how Z-3's control gets the same construction with a random axis.
    """
    Sh, Sih = sym_powers(X_cov)
    mu = X_cov.mean(0)
    if direction is None:
        z = lab.astype(float)
        Xc = X_lab - X_lab.mean(0)
        sxz = (Xc.T @ (z - z.mean())) / (len(z) - 1)
    else:
        sxz = np.asarray(direction, float)
    u = Sih @ sxz if whiten else sxz
    n = np.linalg.norm(u)
    if n < 1e-12:
        return np.zeros((X_cov.shape[1],) * 2), mu
    u = u / n
    # 🔒 without whitening this reduces to the orthogonal rank-1 projection on the mean difference,
    # which is the naive ablation and carries no guarantee -- it is here to separate "the guarantee
    # does not transfer" from "whitening amplified the noise".
    return (Sh @ np.outer(u, u) @ Sih) if whiten else np.outer(u, u), mu


def erase(X, A, mu):
    return X - (X - mu) @ A.T


def residual_corr(X, lab):
    """|corr| between the label and the most favourable linear score of X.

    🔴 This is the in-fold half of Z-0 and it must not be a single arbitrary direction: the LEACE
    guarantee is that NO linear score correlates, so the check maximizes over linear scores. With
    centred X and centred z the maximizing score is X @ Sxz, whose correlation with z is the multiple
    correlation; on the supported subspace that is what this returns.
    """
    z = lab.astype(float)
    z = z - z.mean()
    Xc = X - X.mean(0)
    s = Xc @ (Xc.T @ z)
    sd = s.std()
    if sd < 1e-12:
        return 0.0
    return float(abs(np.corrcoef(s, z)[0, 1]))


def selftest():
    rng = np.random.default_rng(0)
    d, n = 30, 900
    lab = (rng.random(n) < 0.4).astype(int)
    ax = rng.normal(size=d)
    ax /= np.linalg.norm(ax)
    X = rng.normal(size=(n, d)) @ np.diag(1 + rng.random(d)) + np.outer(lab * 6.0, ax)
    tr, ho = inner_split(n, 1)
    assert loc_auroc(X, lab, tr, ho) > 0.95, "the planted axis must be findable first"
    # 🔒 same-sample LEACE is EXACT: no linear score may correlate with the label
    A, mu = leace(X, X, lab)
    Xe = erase(X, A, mu)
    assert residual_corr(Xe, lab) < 1e-8, residual_corr(Xe, lab)
    assert loc_auroc(Xe, lab, tr, ho) < 0.60, loc_auroc(Xe, lab, tr, ho)
    # 🔒 rank 1 and oblique: A is idempotent but NOT symmetric, which is the whole point of LEACE
    assert np.allclose(A @ A, A, atol=1e-8), "the eraser must be a projection"
    assert abs(np.trace(A) - 1.0) < 1e-8, f"rank must be 1, trace is {np.trace(A)}"
    assert not np.allclose(A, A.T, atol=1e-6), "an ORTHOGONAL projection would mean whitening did nothing"
    # 🔴 the control must NOT erase: a random direction of the same construction leaves the axis
    Ar, mur = leace(X, X, lab, direction=rng.normal(size=d))
    assert loc_auroc(erase(X, Ar, mur), lab, tr, ho) > 0.90, "the control erased the concept"
    # 🔴 and equal means must NOT imply an equal tail: build a case where they dissociate, because
    # that dissociation is Z-1's entire hypothesis and a helper that hid it would hide the result
    Xd = rng.normal(size=(n, d)) * np.where(lab == 1, 3.0, 1.0)[:, None]
    Ad, mud = leace(Xd, Xd, lab)
    Xde = erase(Xd, Ad, mud)
    assert residual_corr(Xde, lab) < 1e-8
    score = Xde @ rng.normal(size=d)
    t = np.quantile(score, 0.95)
    r = (score[lab == 1] >= t).mean() / max((score[lab == 0] >= t).mean(), 1e-9)
    assert r > 3.0, f"a dispersion-only difference must survive mean erasure in the tail, got {r}"
    print("selftest PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="esm2_650M", choices=["esm2_650M", "esm2_35M"])
    ap.add_argument("--labels", default="coverage", choices=["keyword", "coverage"])
    ap.add_argument("--cov", default="pool", choices=["pool", "fold"])
    # 🔴 Added after the first run as an append-only amendment. Z-0's in-fold half was exact (3e-06)
    # and its out-of-fold half failed (0.81/0.88 against a control of 0.90/0.98). `--no-whiten` erases
    # the RAW mean-difference direction instead, which is the same rank-1 ablation without the
    # Sigma^(-1/2) that amplifies estimation error in low-variance directions.
    ap.add_argument("--no-whiten", action="store_true")
    ap.add_argument("--seeds", type=int, default=SEEDS)
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
    cats = {"all": None, "non_hazardous": M107.NON_HAZ, "partly_hazardous": M107.PARTLY,
            "motility": {"Motility"}}
    vrows = {g: [row_of[v] for v, i in keep.items() if i["stratum"] == "extracellular"
                 and (c is None or i["category"] in c)] for g, c in cats.items()}

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
    print(f"arm {a.arm} ({d} dims), axis labels {a.labels} "
          f"{len(axis_cell['extracellular'])}/{len(axis_cell['intracellular'])}, cov {a.cov}")

    lab_rows = np.array(axis_cell["extracellular"] + axis_cell["intracellular"])
    lab = np.r_[np.ones(len(axis_cell["extracellular"])),
                np.zeros(len(axis_cell["intracellular"]))].astype(int)
    order = np.random.default_rng(FOLD_SEED).permutation(len(lab_rows))
    halves = [order[:len(order) // 2], order[len(order) // 2:]]
    key_ex, key_in = set(bcell["extracellular"]), set(bcell["intracellular"])

    def rates(A, mu, m_ex, m_in, seeds):
        Pe, POOLe, Te, Ve = (erase(x, A, mu) for x in (P, POOL, T, VFDB))
        R, CA = [], {g: [] for g in vrows}
        for seed in range(seeds):
            r = np.random.default_rng(seed)
            perm = r.permutation(len(union))
            tr = [union[i] for i in perm[:n_tr]]
            ca = [union[i] for i in perm[n_tr:n_tr + n_ca]]
            mdl = M99.clf().fit(np.vstack([Pe, POOLe[tr]]), y)
            t = float(np.quantile(mdl.predict_proba(POOLe[ca])[:, 1], SPEC))
            fb = mdl.predict_proba(Te)[:, 1] >= t
            fv = mdl.predict_proba(Ve)[:, 1] >= t
            be, bi = float(fb[m_ex].mean()), float(fb[m_in].mean())
            R.append(be / bi if bi > 0 else np.nan)
            for g, rr in vrows.items():
                CA[g].append(float(fv[rr].mean()) / be if be > 0 else np.nan)
        return R, CA

    arms = {k: {"R": [], "CA": {g: [] for g in vrows}, "gen": [], "infold": [], "outfold": []}
            for k in ("leace", "control")}
    t0 = time.time()
    for f in (0, 1):
        fit_i = halves[f]
        Xf, yf = T[lab_rows[fit_i]], lab[fit_i]
        Xcov = POOL if a.cov == "pool" else Xf
        fitted = set(lab_rows[fit_i].tolist())
        m_ex = np.array(sorted(key_ex - fitted))
        m_in = np.array(sorted(key_in - fitted))
        d_rows = np.r_[m_ex, m_in]
        d_lab = np.r_[np.ones(len(m_ex)), np.zeros(len(m_in))].astype(int)
        dtr, dho = inner_split(len(d_lab), FOLD_SEED + 200 + f)
        rng = np.random.default_rng(FOLD_SEED + 400 + f)
        for name, direction in (("leace", None), ("control", rng.normal(size=d))):
            A, mu = leace(Xcov, Xf, yf, direction=direction, whiten=not a.no_whiten)
            arms[name]["infold"].append(residual_corr(erase(Xf, A, mu), yf))
            # 🔴 the amendment's decisive check: the SAME residual correlation on rows the direction
            # was not fitted on. Near zero with a high AUROC would mean the separation is not
            # mean-based at all; clearly non-zero means the mean difference simply did not transfer.
            arms[name]["outfold"].append(residual_corr(erase(T[d_rows], A, mu), d_lab))
            arms[name]["gen"].append(loc_auroc(erase(T[d_rows], A, mu), d_lab, dtr, dho))
            R, CA = rates(A, mu, m_ex, m_in, a.seeds)
            arms[name]["R"] += R
            for g in vrows:
                arms[name]["CA"][g] += CA[g]
        print(f"  fold {f}: in-fold |corr| {arms['leace']['infold'][-1]:.2e} "
              f"(control {arms['control']['infold'][-1]:.3f}), unseen AUROC "
              f"{arms['leace']['gen'][-1]:.4f} (control {arms['control']['gen'][-1]:.4f}), "
              f"{time.time() - t0:.0f}s")

    S = M99.stat
    ca_orig = M107.CA_ORIGINAL[a.arm]
    L, C = arms["leace"], arms["control"]
    gen = max(L["gen"])
    res = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "arm": a.arm, "dim": int(d),
           "labels": a.labels, "cov": a.cov, "seeds": a.seeds,
           "n_axis_labels": {k_: len(v) for k_, v in axis_cell.items()},
           "n_vfdb_extracellular": {g: len(v) for g, v in vrows.items()},
           "floors_met": {g: len(v) >= FLOOR for g, v in vrows.items()},
           "CA_original": ca_orig,
           "Z0_infold_corr": [round(x, 8) for x in L["infold"]],
           "Z0_infold_ok": bool(max(L["infold"]) <= Z0_TOL),
           "Z0_infold_control": [round(x, 4) for x in C["infold"]],
           "whiten": not a.no_whiten,
           "Z0_outfold_corr": [round(x, 4) for x in L["outfold"]],
           "Z0_outfold_control": [round(x, 4) for x in C["outfold"]],
           "Z0_unseen_auroc": [round(x, 4) for x in L["gen"]],
           "Z0_unseen_control": [round(x, 4) for x in C["gen"]],
           "Z0_generalized": bool(gen <= Z0_GEN),
           "Z0_band": ("the erasure generalizes" if gen <= Z0_GEN
                       else "marginal" if gen <= Z0_MARGINAL
                       else "the erasure does not generalize either"),
           "Z1_R": S(L["R"]), "Z1_R_control": S(C["R"]),
           "Z2_CA": S(L["CA"]["all"]), "Z3_CA": S(C["CA"]["all"])}
    res["Z1_band"] = ("the gradient is removed" if res["Z1_R"]["mean"] <= Y1_REMOVED
                      else "partly removed" if res["Z1_R"]["mean"] <= Y1_PARTLY
                      else "the gradient survives a provable mean-erasure")
    res["Z2_rho"] = round(rho(res["Z2_CA"]["mean"], ca_orig), 4)
    res["Z3_rho"] = round(rho(res["Z3_CA"]["mean"], ca_orig), 4)
    res["Z2_band"] = ("membership survives localization erasure" if res["Z2_rho"] >= Y2_SURVIVES
                      else "partial" if res["Z2_rho"] >= Y2_PARTIAL
                      else "the membership contrast was largely the localization axis")
    res["Z3_control_clean"] = bool(res["Z3_rho"] >= Y3_CLEAN)
    res["Z4_rho"] = {g: round(rho(S(L["CA"][g])["mean"], ca_orig), 4)
                     for g in ("non_hazardous", "partly_hazardous", "motility")}
    res["Z4_delta"] = round(res["Z4_rho"]["non_hazardous"] - res["Z4_rho"]["partly_hazardous"], 4)
    res["Z4_band"] = ("no more hazard-shaped than what went in" if abs(res["Z4_delta"]) <= Y4_SAME
                      else "erasure preferentially removes the non-hazardous signal"
                      if res["Z4_delta"] <= -Y4_SAME
                      else "the non-hazardous signal is the more robust one")
    # 🔒 Y-2 is answered only if the erasure held on BOTH halves of Z-0 and the control is clean
    res["answers_Y2"] = bool(res["Z0_infold_ok"] and res["Z0_generalized"]
                             and res["Z3_control_clean"])
    tag = ("" if a.cov == "pool" else "_foldcov") + ("" if not a.no_whiten else "_nowhiten")
    dest = Path(f"{OUT_STEM}{sfx}{tag}.json")
    dest.write_text(json.dumps(res, indent=2) + "\n")

    print(f"\nZ-0  in-fold |corr| {res['Z0_infold_corr']} (tol {Z0_TOL}) -> "
          f"{'OK' if res['Z0_infold_ok'] else '🔴 MISSED'}"
          f"   control {res['Z0_infold_control']}")
    print(f"     out-of-fold |corr| {res['Z0_outfold_corr']} "
          f"(control {res['Z0_outfold_control']})")
    print(f"     unseen-row AUROC {res['Z0_unseen_auroc']} (control {res['Z0_unseen_control']})"
          f"  ->  {res['Z0_band']}")
    print(f"Z-1  R = {res['Z1_R']['mean']:.3f} [{res['Z1_R']['ci'][0]:.3f}, "
          f"{res['Z1_R']['ci'][1]:.3f}]  (control {res['Z1_R_control']['mean']:.3f})"
          f"  ->  {res['Z1_band']}")
    print(f"Z-2  CA = {res['Z2_CA']['mean']:.3f} against {ca_orig}, rho = {res['Z2_rho']:.3f}"
          f"  ->  {res['Z2_band']}")
    print(f"Z-3  control rho = {res['Z3_rho']:.3f}  ->  "
          f"{'clean' if res['Z3_control_clean'] else '🔴 CONFOUNDED'}")
    print(f"Z-4  non-haz {res['Z4_rho']['non_hazardous']:.3f}, partly "
          f"{res['Z4_rho']['partly_hazardous']:.3f}, motility "
          f"{res['Z4_rho']['motility']:.3f} (n=28, indicative)  delta {res['Z4_delta']:+.3f}")
    print(f"     -> {res['Z4_band']}")
    print(f"\n{'🔒 Y-2 is answered' if res['answers_Y2'] else '🔴 Y-2 stays open'}")
    print(f"wrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
