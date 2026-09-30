#!/usr/bin/env python3
"""
107_localization_nullspace.py - project the localization axis out, and see what is left

`docs/LOCALIZATION_NULLSPACE_PREREGISTRATION.md`. Nine studies decomposed the same score: localization
(R = 5.07), length (a U), provenance (2.7x), two metadata fields (AUROC 0.8096), and a positive finding
that is VFDB REGISTRATION rather than hazard (Motility 99.05%). None of them asks whether the membership
contrast survives removing the localization axis, or whether what survives is any more hazard-shaped.

Iterative nullspace projection, because localization is not a one-dimensional property and a rank-1
ablation does not remove it. Directions are fitted on benign extracellular-vs-intracellular labels under
study E's KEYWORD strata (study K's and study Q's rule, so 2.418 and 2.410 stay comparable), cross-fitted
50/50 so the rows that fit the directions are never the rows that measure them. VFDB never enters
direction fitting.

🔒 Y-0 is a GATE, not a result: if post-projection localization AUROC exceeds 0.55, the method did not do
its job and everything downstream is indicative. Y-3 is the equal-rank random-direction control, without
which a low Y-2 is indistinguishable from ablation damage.

⚠️ Removing a direction from an embedding is a measurement technique. Nothing here is a mitigation and
nothing here is a repaired probe.

Usage:
    python src/107_localization_nullspace.py --arm esm2_650M
    python src/107_localization_nullspace.py --arm esm2_35M
    python src/107_localization_nullspace.py --selftest
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "v3"
POOL_JSON = ROOT / "data" / "sequences" / "_scaled_negative_pool.json"
NEGSET = ROOT / "results" / "vfdb_negative_set.json"
MAPPED = ROOT / "results" / "vfdb_uniprot_map.json"
BUILD = ROOT / "results" / "external_class_axis_build.json"
SCREEN = ROOT / "results" / "external_class_axis_screen.json"
OUT_STEM = ROOT / "results" / "localization_nullspace"

# 🔒 frozen in § 2 - § 7 of the preregistration
FOLD_SEED, INNER_FRAC = 20260930, 0.75
GATE_AUROC, K_MAX = 0.55, 20
Y1_REMOVED, Y1_PARTLY = 1.5, 3.0
Y2_SURVIVES, Y2_PARTIAL = 0.70, 0.30
Y3_CLEAN = 0.85
Y4_SAME = 0.20
# study K's and study Q's unprojected contrasts, for rho's denominator
CA_ORIGINAL = {"esm2_650M": 2.418, "esm2_35M": 2.617}


def _load(stem):
    spec = importlib.util.spec_from_file_location(f"_{stem}", ROOT / "src" / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M99 = _load("99_hazard_vs_membership")
M92, M84, M83, M91 = M99.M92, M99.M84, M99.M83, M99.M91
SEEDS, SPEC, FLOOR = M99.SEEDS, M99.SPEC, M99.FLOOR
NON_HAZ, PARTLY = M99.NON_HAZARDOUS, M99.PARTLY_HAZARDOUS


def inner_split(n, seed, inner=INNER_FRAC):
    """The fit/holdout split inside a direction-fitting fold, from a seed so treatment and control
    are measured on the SAME rows. 🔴 Without this the control's gate number would be a different
    quantity from the treatment's and the two would still print under one heading."""
    perm = np.random.default_rng(seed).permutation(n)
    cut = int(round(inner * n))
    return perm[:cut], perm[cut:]


def loc_auroc(Z, lab, tr, ho):
    """Held-out AUROC of a freshly fitted localization classifier. One definition, three callers:
    before projection, after the treatment's projection, and after the control's."""
    m = LogisticRegression(max_iter=5000, C=1.0).fit(Z[tr], lab[tr])
    s = Z[ho] @ m.coef_[0]
    return float(M84.auroc(s[lab[ho] == 1], s[lab[ho] == 0]))


def inlp(X, lab, tr, ho, k_max=K_MAX, gate=GATE_AUROC):
    """Iterative nullspace projection of a binary axis. Returns (U, k, auroc_before, auroc_after).

    `U` is (k, d) with orthonormal rows. Every AUROC comes from `loc_auroc` on the holdout of the fold
    this is called on, so the measurement fold stays untouched.
    """
    Z = X.copy()
    rows, before = [], None
    for _ in range(k_max + 1):
        au = loc_auroc(Z, lab, tr, ho)
        if before is None:
            before = au
        if au <= gate or len(rows) == k_max:
            break
        w = LogisticRegression(max_iter=5000, C=1.0).fit(Z[tr], lab[tr]).coef_[0].astype(float)
        for u in rows:                                    # re-orthogonalize against what is gone
            w -= (w @ u) * u
        nw = np.linalg.norm(w)
        if nw < 1e-9:
            break
        u = w / nw
        rows.append(u)
        Z = Z - np.outer(Z @ u, u)
    U = np.array(rows) if rows else np.zeros((0, X.shape[1]))
    return U, len(rows), float(before), float(au)


def random_basis(d, k, rng):
    """k orthonormal rows in R^d, so the control removes exactly the rank the treatment removed."""
    if k == 0:
        return np.zeros((0, d))
    q, _ = np.linalg.qr(rng.normal(size=(d, k)))
    return q.T[:k]


def project_out(X, U):
    return X if U.shape[0] == 0 else X - (X @ U.T) @ U


def selftest():
    rng = np.random.default_rng(0)
    d, n = 40, 800
    # 🔒 a planted 3-dimensional axis must be removable, and removal must kill a held-out AUROC
    lab = (rng.random(n) < 0.5).astype(int)
    axis = rng.normal(size=(3, d))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    X = rng.normal(size=(n, d)) + np.outer(lab * 3.0, axis.sum(0))
    tr, ho = inner_split(n, 1)
    U, k, before, after = inlp(X, lab, tr, ho)
    assert before > 0.9 and after <= GATE_AUROC and 1 <= k <= K_MAX, (before, after, k)
    # 🔒 the same split must reproduce `before` exactly from the standalone helper, or the gate's
    # "before" and the control's "after" are not the same quantity
    assert abs(loc_auroc(X, lab, tr, ho) - before) < 1e-12
    assert np.allclose(U @ U.T, np.eye(k), atol=1e-8), "the removed rows must be orthonormal"
    Xp = project_out(X, U)
    assert np.allclose(Xp @ U.T, 0, atol=1e-8), "projection must leave nothing along U"
    # 🔴 an equal-rank RANDOM ablation must NOT kill it -- that is what makes Y-3 informative, and a
    # projection helper that damaged everything equally would pass the treatment and fail here
    Ur = random_basis(d, k, np.random.default_rng(2))
    au_r = loc_auroc(project_out(X, Ur), lab, tr, ho)
    assert au_r > 0.9, f"random ablation of rank {k} should not remove a planted axis: {au_r}"
    # 🔒 no axis to find: k must stay small rather than grinding to K_MAX on noise
    _, k0, b0, _ = inlp(rng.normal(size=(n, d)), lab, tr, ho)
    assert b0 <= GATE_AUROC and k0 == 0, (b0, k0)
    # 🔒 rho arithmetic, stated once so the report and the bands cannot disagree
    assert abs(rho(2.418, 2.418) - 1.0) < 1e-12 and abs(rho(1.0, 2.418)) < 1e-12
    assert abs(rho(1.709, 2.418) - 0.5) < 1e-3
    print("selftest PASS")


def rho(ca_proj, ca_orig):
    """Fraction of the membership contrast retained. 1.0 = untouched, 0.0 = gone."""
    return (ca_proj - 1.0) / (ca_orig - 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="esm2_650M", choices=["esm2_650M", "esm2_35M"])
    ap.add_argument("--seeds", type=int, default=SEEDS)
    # 🔴 Added after the first run as an append-only amendment. `keyword` is what § 2 froze; it gives
    # only 171 extracellular rows, so directions are fitted on ~128 points in 1,280 dimensions and
    # the projection did not generalize. `coverage` uses study L's GO-augmented strata for the AXIS
    # only -- every measured rate stays on the keyword strata, so 2.418 remains the comparator.
    ap.add_argument("--labels", default="keyword", choices=["keyword", "coverage"])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    pool = json.loads(POOL_JSON.read_text())["proteins"]
    test = json.loads(BUILD.read_text())["pool_partition"]["test_rows"]
    accs = [pool[i]["uniprot"] for i in test]
    kw = M84.fetch_keywords(accs)
    vf_seqs, vfdb_species = M83.vfdb_sequences(), M83.vfdb_species()

    # ---- benign strata, study E's keyword rule, study K's organism and contamination filters ------
    bcell = {"extracellular": [], "intracellular": []}
    for k_, i in enumerate(test):
        sp = " ".join(pool[i]["organism"].replace("(", "").split()[:2])
        if pool[i]["kingdom"] != "Bacteria" or sp not in vfdb_species:
            continue
        if pool[i]["sequence"] in vf_seqs:
            continue
        s = M84.stratum_of(kw[pool[i]["uniprot"]])
        if s in bcell:
            bcell[s].append(k_)                                  # index into `test`, not into `pool`
    print(f"arm {a.arm}: benign extracellular {len(bcell['extracellular'])}, "
          f"intracellular {len(bcell['intracellular'])}")

    # 🔒 the label set for the AXIS. Measurement always uses `bcell` above.
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
        print(f"  axis labels (study L strata): extracellular {len(axis_cell['extracellular'])}, "
              f"intracellular {len(axis_cell['intracellular'])}")

    # ---- VFDB side, study K's three exclusions, study Q's category groups ------------------------
    admitted = json.loads(NEGSET.read_text())["admitted"]
    len_of = {r["vfg"]: r["len"] for r in admitted}
    row_of = {r["vfg"]: i for i, r in enumerate(admitted)}
    noncirc = M83.vfdb_noncircular_mask()
    up_len, acc_of = M91.uniprot_lengths(), M91.vfg_accessions()
    keep = {}
    for vfg, info in json.loads(MAPPED.read_text())["chosen"].items():
        if not noncirc[row_of[vfg]]:
            continue
        ul = up_len.get((acc_of[vfg], info["uniprot"]))
        if ul is None or str(len_of[vfg]) != str(ul):
            continue
        keep[vfg] = info
    vgrp = {"all": None, "non_hazardous": NON_HAZ, "partly_hazardous": PARTLY,
            "motility": {"Motility"}}
    vrows = {g: [row_of[v] for v, i in keep.items()
                 if i["stratum"] == "extracellular" and (cats is None or i["category"] in cats)]
             for g, cats in vgrp.items()}
    print("  VFDB extracellular after study K's exclusions: "
          + ", ".join(f"{g} {len(v)}" for g, v in vrows.items()))

    sfx = "" if a.arm == "esm2_650M" else f"_{a.arm}"
    P = np.load(RES / f"embeddings_class_axis_positives{sfx}.npy")
    POOL = np.load(RES / f"embeddings_pool_large_{a.arm}.npy")
    VFDB = np.load(RES / f"embeddings_vfdb_neg_{a.arm}.npy")
    union = [i for i in json.loads(SCREEN.read_text())["admitted_rows"]
             if pool[i]["sequence"] not in vf_seqs]
    n_tr, n_ca = M92.fold_sizes(len(union))
    y = np.r_[np.ones(P.shape[0]), np.zeros(n_tr)]
    T = POOL[np.array(test)]
    d = T.shape[1]

    # ---- the frozen 50/50 direction/measurement split -------------------------------------------
    lab_rows = np.array(axis_cell["extracellular"] + axis_cell["intracellular"])
    lab = np.r_[np.ones(len(axis_cell["extracellular"])),
                np.zeros(len(axis_cell["intracellular"]))].astype(int)
    fr = np.random.default_rng(FOLD_SEED)
    order = fr.permutation(len(lab_rows))
    halves = [order[:len(order) // 2], order[len(order) // 2:]]
    # 🔴 The measurement rows are always the KEYWORD strata, and a row used to fit the axis must never
    # be measured. With the wider axis labels the two sets overlap, so the measured rows are the
    # keyword rows that are NOT in the fitting half -- computed per fold below.
    key_ex = set(bcell["extracellular"])
    key_in = set(bcell["intracellular"])

    out = {c: {g: [] for g in list(vgrp) + ["R", "benign_extra", "benign_intra", "gate_after",
                                            "gate_measfold"]}
           for c in ("treatment", "control")}
    ks, gate_before = [], []
    t0 = time.time()
    for f in (0, 1):
        fit_i = halves[f]      # the other half is implicit: measurement = keyword rows minus fit
        Xf, yf = T[lab_rows[fit_i]], lab[fit_i]
        itr, iho = inner_split(len(yf), FOLD_SEED + f)
        U, k, au_b, au_a = inlp(Xf, yf, itr, iho)
        Ur = random_basis(d, k, np.random.default_rng(FOLD_SEED + 100 + f))
        ks.append(k)
        gate_before.append(au_b)
        fitted = set(lab_rows[fit_i].tolist())
        m_ex = np.array(sorted(key_ex - fitted))
        m_in = np.array(sorted(key_in - fitted))
        # 🔒 the measurement-fold diagnostic the frozen § 3 lacked: can a classifier fitted on UNSEEN
        # rows still separate the strata after projection? The gate in § 3 is measured on a holdout of
        # the FITTING fold, which a direction fitted to 128 points in 1,280 dimensions can satisfy by
        # overfitting. Nothing is tuned on this number; it decides only whether Y-2 is interpretable.
        d_rows = np.r_[m_ex, m_in]
        d_lab = np.r_[np.ones(len(m_ex)), np.zeros(len(m_in))].astype(int)
        dtr, dho = inner_split(len(d_lab), FOLD_SEED + 200 + f)
        for cname, B in (("treatment", U), ("control", Ur)):
            Pp, POOLp, Tp, Vp = (project_out(x, B) for x in (P, POOL, T, VFDB))
            # 🔒 one definition of the gate for both arms: refit on the projected fit fold, score its
            # holdout. The treatment's own number is reused and asserted identical.
            au = loc_auroc(project_out(Xf, B), yf, itr, iho)
            assert cname != "treatment" or abs(au - au_a) < 1e-9, (au, au_a)
            out[cname]["gate_after"].append(au)
            out[cname]["gate_measfold"].append(loc_auroc(Tp[d_rows], d_lab, dtr, dho))
            for seed in range(a.seeds):
                r = np.random.default_rng(seed)
                perm = r.permutation(len(union))
                tr = [union[i] for i in perm[:n_tr]]
                ca = [union[i] for i in perm[n_tr:n_tr + n_ca]]
                mdl = M99.clf().fit(np.vstack([Pp, POOLp[tr]]), y)
                t = float(np.quantile(mdl.predict_proba(POOLp[ca])[:, 1], SPEC))
                fb = mdl.predict_proba(Tp)[:, 1] >= t
                fv = mdl.predict_proba(Vp)[:, 1] >= t
                be, bi = float(fb[m_ex].mean()), float(fb[m_in].mean())
                out[cname]["benign_extra"].append(be)
                out[cname]["benign_intra"].append(bi)
                out[cname]["R"].append(be / bi if bi > 0 else np.nan)
                for g, rows in vrows.items():
                    out[cname][g].append(float(fv[rows].mean()) / be if be > 0 and rows else np.nan)
        print(f"  fold {f}: k={k}, localization AUROC {au_b:.4f} -> {au_a:.4f}, "
              f"{time.time() - t0:.0f}s")

    S = M99.stat
    res = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "arm": a.arm, "seeds": a.seeds,
           "dim": int(d), "k_per_fold": ks, "k_fraction_of_dim": round(max(ks) / d, 4),
           "n_benign": {k_: len(v) for k_, v in bcell.items()},
           "n_vfdb_extracellular": {g: len(v) for g, v in vrows.items()},
           "floors_met": {g: len(v) >= FLOOR for g, v in vrows.items()},
           "CA_original": CA_ORIGINAL[a.arm],
           "labels": a.labels,
           "n_axis_labels": {k_: len(v) for k_, v in axis_cell.items()},
           "Y0_auroc_before": S(gate_before), "Y0_auroc_after": S(out["treatment"]["gate_after"]),
           "Y0_gate_met": bool(max(out["treatment"]["gate_after"]) <= GATE_AUROC),
           # 🔴 the amendment's diagnostic: the same question asked on rows the axis never saw
           "Y0b_auroc_measfold": S(out["treatment"]["gate_measfold"]),
           "Y0b_auroc_measfold_control": S(out["control"]["gate_measfold"]),
           "Y0b_generalized": bool(max(out["treatment"]["gate_measfold"]) <= GATE_AUROC),
           "Y1_R_projected": S(out["treatment"]["R"]),
           "Y1_R_control": S(out["control"]["R"])}
    for tag, cname in (("Y2", "treatment"), ("Y3", "control")):
        ca = S(out[cname]["all"])
        res[f"{tag}_CA"] = ca
        res[f"{tag}_rho"] = round(rho(ca["mean"], res["CA_original"]), 4)
    res["Y4_rho"] = {g: round(rho(S(out["treatment"][g])["mean"], res["CA_original"]), 4)
                     for g in ("non_hazardous", "partly_hazardous", "motility")}
    res["Y4_delta"] = round(res["Y4_rho"]["non_hazardous"] - res["Y4_rho"]["partly_hazardous"], 4)
    res["Y1_band"] = ("the gradient is removed" if res["Y1_R_projected"]["mean"] <= Y1_REMOVED
                      else "partly removed" if res["Y1_R_projected"]["mean"] <= Y1_PARTLY
                      else "the projection did not remove the gradient")
    res["Y2_band"] = ("membership survives localization removal" if res["Y2_rho"] >= Y2_SURVIVES
                      else "partial" if res["Y2_rho"] >= Y2_PARTIAL
                      else "the membership contrast was largely the localization axis")
    res["Y3_control_clean"] = bool(res["Y3_rho"] >= Y3_CLEAN)
    res["Y4_band"] = ("no more hazard-shaped than what went in" if abs(res["Y4_delta"]) <= Y4_SAME
                      else "projection preferentially removes the non-hazardous signal"
                      if res["Y4_delta"] <= -Y4_SAME
                      else "the non-hazardous signal is the more robust one")
    # 🔒 the verdict is a verdict only if the gate held AND the control is clean
    # 🔴 and it is a verdict only if the projection GENERALIZED: a gate met by overfitting the fitting
    # fold tells us nothing about the rows Y-1 and Y-2 are measured on.
    res["verdict_eligible"] = bool(res["Y0_gate_met"] and res["Y0b_generalized"]
                                   and res["Y3_control_clean"])
    # 🔒 the amendment's run gets its own file: the frozen keyword run stays the record of § 2
    dest = Path(f"{OUT_STEM}{sfx}{'' if a.labels == 'keyword' else '_coverage_axis'}.json")
    dest.write_text(json.dumps(res, indent=2) + "\n")

    print(f"\nY-0  localization AUROC {res['Y0_auroc_before']['mean']:.4f} -> "
          f"{res['Y0_auroc_after']['mean']:.4f}   gate {'MET' if res['Y0_gate_met'] else '🔴 FAILED'}"
          f"   (k = {ks}, {res['k_fraction_of_dim'] * 100:.1f}% of {d} dims)")
    print(f"Y-0b on rows the axis never saw: {res['Y0b_auroc_measfold']['mean']:.4f} "
          f"[{res['Y0b_auroc_measfold']['ci'][0]:.4f}, {res['Y0b_auroc_measfold']['ci'][1]:.4f}]"
          f"  (control {res['Y0b_auroc_measfold_control']['mean']:.4f})  ->  "
          f"{'generalized' if res['Y0b_generalized'] else '🔴 DID NOT GENERALIZE'}")
    print(f"Y-1  R projected {res['Y1_R_projected']['mean']:.3f} "
          f"[{res['Y1_R_projected']['ci'][0]:.3f}, {res['Y1_R_projected']['ci'][1]:.3f}]  "
          f"(random-direction control {res['Y1_R_control']['mean']:.3f})  ->  {res['Y1_band']}")
    print(f"Y-2  CA projected {res['Y2_CA']['mean']:.3f} against {res['CA_original']} "
          f"unprojected, rho = {res['Y2_rho']:.3f}  ->  {res['Y2_band']}")
    print(f"Y-3  control CA {res['Y3_CA']['mean']:.3f}, rho = {res['Y3_rho']:.3f}  ->  "
          f"{'clean' if res['Y3_control_clean'] else '🔴 CONFOUNDED by capacity loss'}")
    print(f"Y-4  rho non-hazardous {res['Y4_rho']['non_hazardous']:.3f}, partly-hazardous "
          f"{res['Y4_rho']['partly_hazardous']:.3f}, motility "
          f"{res['Y4_rho']['motility']:.3f}  (delta {res['Y4_delta']:+.3f})")
    print(f"     -> {res['Y4_band']}")
    if not res["verdict_eligible"]:
        print("\n🔴 indicative only: the gate or the control did not hold")
    print(f"\nwrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
