#!/usr/bin/env python3
"""Study AP: can the estimation gap study AO located be repaired by a modelling choice?

`docs/ESTIMATION_REPAIR_PREREGISTRATION.md`, frozen before this file existed.

AO showed the deficit is estimation, not information. Estimation deficits sometimes yield to a
modelling choice, so this tries two: regularization strength, swept over C, and dimensionality,
PCA to k=50.

🔴 The rule that makes this a study and not a search: for each held-out category, C is selected
on a random MEMBER split of the other thirteen categories plus train negatives, and only then
scored on the held-out category. No number from the scored category reaches the selection. The
PCA basis is fitted on the negative union only, so no held-out positive informs it.
"""
import argparse
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results/v3"
LOCO = ROOT / "results/deepvic_loco.json"
COMP = ROOT / "results/composition_arm.json"
OUT = ROOT / "results/estimation_repair.json"
SEEDS, INNER_SEEDS, N_TRAIN, N_CAL, SPEC, FLOOR = 30, 5, 1000, 500, 0.95, 7
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
PCA_K = 50
BAND_REPAIRED, BAND_NOT = 0.03, 0.005
BAND_GAP_REPAIRED, BAND_GAP_NOT = 0.50, 0.10
ARMS = {"esm2_650M": "", "esm2_35M": "_esm2_35M"}


def load(mod):
    spec = importlib.util.spec_from_file_location(mod[:4], ROOT / "src" / mod)
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod[:4]] = m
    spec.loader.exec_module(m)
    return m


def clf(creg):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=5000, C=creg))


def fold_once(X, fit_pos, fit_neg, score_rows, cal_rows, creg):
    """Fit, threshold at SPEC on cal_rows, return the flag rate on score_rows."""
    mdl = clf(creg).fit(np.vstack([X[fit_pos], X[fit_neg]]),
                        np.r_[np.ones(len(fit_pos)), np.zeros(len(fit_neg))])
    t = float(np.quantile(mdl.predict_proba(X[cal_rows])[:, 1], SPEC))
    return float((mdl.predict_proba(X[score_rows])[:, 1] >= t).mean())


def select_C(X, trained_pos, union, rng_seed):
    """Choose C on a member split of the TRAINED categories only.

    🔴 The held-out category is not in `trained_pos`, so nothing from the category about to be
    scored can reach this choice. That is the whole difference between this and a search.
    """
    best, best_score = C_GRID[2], -1.0
    for creg in C_GRID:
        got = []
        for s in range(INNER_SEEDS):
            r = np.random.default_rng(10_000 + 97 * rng_seed + s)
            pp = r.permutation(len(trained_pos))
            n_val = max(1, len(trained_pos) // 5)
            val, fit = trained_pos[pp[:n_val]], trained_pos[pp[n_val:]]
            pn = r.permutation(len(union))
            tr, ca = union[pn[:N_TRAIN]], union[pn[N_TRAIN:N_TRAIN + N_CAL]]
            got.append(fold_once(X, fit, tr, val, ca, creg))
        m = float(np.mean(got))
        if m > best_score:
            best, best_score = creg, m
    return best, best_score


def verdict(d):
    if not np.isfinite(d):
        return "unmeasured"
    if d >= BAND_REPAIRED:
        return "repaired"
    if d <= BAND_NOT:
        return "not repaired"
    return "between bands"


def selftest():
    ok = True
    if (BAND_REPAIRED, BAND_NOT) != (0.03, 0.005):
        print(f"  XX bands are {(BAND_REPAIRED, BAND_NOT)}, frozen at (0.03, 0.005)")
        ok = False
    if (BAND_GAP_REPAIRED, BAND_GAP_NOT) != (0.50, 0.10):
        print(f"  XX gap bands are {(BAND_GAP_REPAIRED, BAND_GAP_NOT)}, frozen at (0.5, 0.1)")
        ok = False
    if C_GRID != (0.01, 0.1, 1.0, 10.0, 100.0) or PCA_K != 50:
        print(f"  XX the grid or k changed: {C_GRID}, {PCA_K}")
        ok = False
    for d, want in ((0.05, "repaired"), (0.002, "not repaired"), (0.015, "between bands"),
                    (-0.02, "not repaired")):
        if verdict(d) != want:
            print(f"  XX verdict({d}) is '{verdict(d)}', expected '{want}'")
            ok = False
    if verdict(float("nan")) != "unmeasured":
        print("  XX a nan difference is not reported as unmeasured")
        ok = False

    # 🔒 select_C must be blind to the held-out category. Planted: a feature that separates
    # ONLY the held-out members. If selection saw it, the chosen C would depend on it.
    # 🔴 The synthetic union was 600 rows against N_TRAIN=1000 and N_CAL=500, so the
    # calibrate slice came back EMPTY and StandardScaler raised. The fold's own sizes are what
    # the synthetic data has to accommodate; it is sized from them rather than guessed.
    rng = np.random.default_rng(0)
    n_union = N_TRAIN + N_CAL + 500
    held = np.arange(0, 60)
    trained = np.arange(60, 300)
    union = np.arange(300, 300 + n_union)
    N = 300 + n_union
    X = rng.normal(size=(N, 6))
    X[held, 0] += 50.0          # visible only in the held-out block
    c1, _ = select_C(X, trained, union, 0)
    X2 = X.copy()
    X2[held, 0] -= 100.0        # flip it; selection must be unchanged
    c2, _ = select_C(X2, trained, union, 0)
    if c1 != c2:
        print(f"  XX select_C changed with the held-out block ({c1} -> {c2}); it is not blind")
        ok = False
    # and it must actually respond to the TRAINED block, or it is selecting nothing
    X3 = X.copy()
    X3[trained, 1] += 40.0
    c3, _ = select_C(X3, trained, union, 0)
    if c3 == c1 and len(set(C_GRID)) > 1:
        # not strictly required, but a selector that never moves is worth naming
        print(f"  ⚠ select_C returned {c1} both with and without a planted trained signal")

    # fold_once must separate a planted signal and sit near the nominal rate on noise
    y_rows = np.arange(0, 60)
    r1 = fold_once(X, trained, union[:N_TRAIN], y_rows,
                   union[N_TRAIN:N_TRAIN + N_CAL], 1.0)
    Xn = rng.normal(size=(N, 6))
    r0 = fold_once(Xn, trained, union[:N_TRAIN], y_rows,
                   union[N_TRAIN:N_TRAIN + N_CAL], 1.0)
    if not (0.0 <= r1 <= 1.0 and 0.0 <= r0 <= 1.0):
        print("  XX fold_once returned a rate outside [0, 1]")
        ok = False
    if r0 > 0.40:
        print(f"  XX fold_once on noise gave {r0:.3f}, far above the 5% nominal")
        ok = False

    # the baselines the bands were set from must still be there
    for f in (LOCO, COMP):
        if not f.exists():
            print(f"  XX {f.name} missing")
            ok = False
    if ok:
        L, Cj = json.loads(LOCO.read_text()), json.loads(COMP.read_text())
        for arm, w in (("esm2_650M", 0.414), ("esm2_35M", 0.418)):
            r = L["arms"][arm]["full"]["recovery"]
            if abs(float(np.mean([r[c]["mean"] for c in r])) - w) > 0.002:
                print(f"  XX {arm}'s fixed-C baseline moved from {w}")
                ok = False
        exo = {x["category"]: x["recovery"]
               for x in Cj["arms"]["composition"]["ranking"]}["Exoenzyme"]
        if abs(exo - 0.623) < 0.002:
            pass
        else:
            print(f"  XX composition's Exoenzyme recovery is {exo:.4f}, bands used 0.623")
            ok = False
    print("selftest PASS" if ok else "selftest FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seeds", type=int, default=SEEDS)
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    t0 = time.time()
    m124 = load("124_deepvic_embed.py")
    recs, meta = m124.ordered()
    cls = np.array(meta["class_of_row"])
    lab = np.array(meta["label_of_row"])
    pos, neg = np.where(lab == 1)[0], np.where(lab == 0)[0]
    perm = np.random.default_rng(12345).permutation(len(neg))
    cut = 2 * len(neg) // 3
    union, test = neg[perm[:cut]], neg[perm[cut:]]
    targets = sorted(c for c, n in Counter(cls[pos]).items() if n >= FLOOR)
    L = json.loads(LOCO.read_text())
    Cj = json.loads(COMP.read_text())
    comp = {x["category"]: x["recovery"]
            for x in Cj["arms"]["composition"]["ranking"]}

    out = {"built": time.strftime("%Y-%m-%d %H:%M:%S"),
           "study": "AP -- can the estimation gap be repaired by a modelling choice?",
           "preregistration": "docs/ESTIMATION_REPAIR_PREREGISTRATION.md",
           "selection_rule": ("C is chosen on a member split of the TRAINED categories plus "
                              "train negatives; no number from the scored category reaches "
                              "the selection"),
           "pca_basis": "fitted on the negative union only, so no held-out positive informs it",
           "C_grid": list(C_GRID), "pca_k": PCA_K,
           "bands": {"repaired_floor": BAND_REPAIRED, "not_ceiling": BAND_NOT,
                     "gap_repaired_floor": BAND_GAP_REPAIRED,
                     "gap_not_ceiling": BAND_GAP_NOT},
           "seeds": a.seeds, "inner_seeds": INNER_SEEDS, "arms": {}}

    for arm, sfx in ARMS.items():
        E = np.load(RES / f"embeddings_deepvic{sfx}.npy").astype(np.float64)
        pca = PCA(n_components=PCA_K, random_state=0).fit(E[union])
        Ep = pca.transform(E)
        print(f"  {arm}: PCA {E.shape[1]} -> {PCA_K}, explained "
              f"{pca.explained_variance_ratio_.sum():.1%}", flush=True)
        base = {c: L["arms"][arm]["full"]["recovery"][c]["mean"] for c in targets}

        res = {"swept": {}, "pca": {}, "chosen_C": {}}
        for ci, cat in enumerate(targets):
            hi = pos[cls[pos] == cat]
            tri = pos[cls[pos] != cat]
            chosen, inner = select_C(E, tri, union, ci)
            res["chosen_C"][cat] = {"C": chosen, "inner_score": inner}
            for tag, X in (("swept", E), ("pca", Ep)):
                creg = chosen if tag == "swept" else 1.0
                rr, ff = [], []
                for seed in range(a.seeds):
                    r = np.random.default_rng(seed)
                    p = r.permutation(len(union))
                    tr, ca = union[p[:N_TRAIN]], union[p[N_TRAIN:N_TRAIN + N_CAL]]
                    rr.append(fold_once(X, tri, tr, hi, ca, creg))
                    # 🔒 the test partition is scored so the fold can be shown to be
                    # calibrated like studies AI, AJ, AK, AM and AO -- which is what makes
                    # these recoveries comparable to the fixed-C baseline at all. A swept C
                    # could move the realized FPR off nominal, and that would invalidate the
                    # comparison silently.
                    ff.append(fold_once(X, tri, tr, test, ca, creg))
                res[tag][cat] = {"n": int(len(hi)), "mean": float(np.mean(rr)),
                                 "sd": float(np.std(rr, ddof=1)),
                                 "test_fpr": float(np.mean(ff))}
            print(f"    {cat:<46}C={chosen:<6g} swept {res['swept'][cat]['mean']*100:6.2f}%  "
                  f"pca {res['pca'][cat]['mean']*100:6.2f}%  "
                  f"(fixed-C {base[cat]*100:5.1f}%, composition {comp[cat]*100:5.1f}%)",
                  flush=True)

        m_base = float(np.mean([base[c] for c in targets]))
        m_sw = float(np.mean([res["swept"][c]["mean"] for c in targets]))
        m_pca = float(np.mean([res["pca"][c]["mean"] for c in targets]))
        gap = comp["Exoenzyme"] - base["Exoenzyme"]
        closed_sw = (res["swept"]["Exoenzyme"]["mean"] - base["Exoenzyme"]) / gap
        closed_pca = (res["pca"]["Exoenzyme"]["mean"] - base["Exoenzyme"]) / gap
        out["arms"][arm] = {
            "pca_explained": float(pca.explained_variance_ratio_.sum()),
            "chosen_C": res["chosen_C"],
            "distinct_C_chosen": sorted({v["C"] for v in res["chosen_C"].values()}),
            "swept": res["swept"], "pca": res["pca"],
            "mean_fixed_C": m_base, "mean_swept": m_sw, "mean_pca": m_pca,
            "AP1_swept_minus_fixed": m_sw - m_base,
            "AP1_pca_minus_fixed": m_pca - m_base,
            "exoenzyme": {"composition": comp["Exoenzyme"], "fixed_C": base["Exoenzyme"],
                          "swept": res["swept"]["Exoenzyme"]["mean"],
                          "pca": res["pca"]["Exoenzyme"]["mean"], "gap": gap},
            "AP2_gap_closed_swept": closed_sw,
            "AP2_gap_closed_pca": closed_pca,
            "test_fpr_swept": {"min": float(min(res["swept"][c]["test_fpr"]
                                                for c in targets)),
                               "max": float(max(res["swept"][c]["test_fpr"]
                                                for c in targets))},
            "test_fpr_pca": {"min": float(min(res["pca"][c]["test_fpr"]
                                              for c in targets)),
                             "max": float(max(res["pca"][c]["test_fpr"]
                                              for c in targets))},
            "verdict_swept": verdict(m_sw - m_base),
            "verdict_pca": verdict(m_pca - m_base),
        }
        v = out["arms"][arm]
        print(f"    mean fixed-C {m_base*100:.2f}%  swept {m_sw*100:.2f}% "
              f"({v['AP1_swept_minus_fixed']*100:+.2f}pp -> {v['verdict_swept']})  "
              f"pca {m_pca*100:.2f}% ({v['AP1_pca_minus_fixed']*100:+.2f}pp -> "
              f"{v['verdict_pca']})", flush=True)
        print(f"    Exoenzyme gap {gap*100:.1f}pp: swept closes {closed_sw:.1%}, "
              f"pca closes {closed_pca:.1%}  | C chosen: {v['distinct_C_chosen']}", flush=True)
        print(f"    out-of-sample FPR: swept {v['test_fpr_swept']['min']*100:.2f}-"
              f"{v['test_fpr_swept']['max']*100:.2f}%  pca "
              f"{v['test_fpr_pca']['min']*100:.2f}-{v['test_fpr_pca']['max']*100:.2f}%",
              flush=True)

    A, B = out["arms"]["esm2_650M"], out["arms"]["esm2_35M"]
    out["second_arm_rule"] = {
        "swept_agree": A["verdict_swept"] == B["verdict_swept"],
        "verdict_swept": (A["verdict_swept"] if A["verdict_swept"] == B["verdict_swept"]
                          else "SPLIT -- indicative only"),
        "pca_agree": A["verdict_pca"] == B["verdict_pca"],
        "verdict_pca": (A["verdict_pca"] if A["verdict_pca"] == B["verdict_pca"]
                        else "SPLIT -- indicative only")}
    out["predictions"] = {
        "P1_swept_not_repaired_both_arms": {
            "kind": "derived",
            "held": bool(A["AP1_swept_minus_fixed"] <= BAND_NOT
                         and B["AP1_swept_minus_fixed"] <= BAND_NOT),
            "values": {k: out["arms"][k]["AP1_swept_minus_fixed"] for k in ARMS}},
        "P2_gap_closed_under_10pct_both_arms": {
            "kind": "derived",
            "held": bool(A["AP2_gap_closed_swept"] <= BAND_GAP_NOT
                         and B["AP2_gap_closed_swept"] <= BAND_GAP_NOT),
            "values": {k: out["arms"][k]["AP2_gap_closed_swept"] for k in ARMS}},
        "P3_pca_beats_sweep_on_exoenzyme": {
            "kind": "reasoned",
            "held": bool(A["AP2_gap_closed_pca"] > A["AP2_gap_closed_swept"]
                         and B["AP2_gap_closed_pca"] > B["AP2_gap_closed_swept"]),
            "values": {k: [out["arms"][k]["AP2_gap_closed_swept"],
                           out["arms"][k]["AP2_gap_closed_pca"]] for k in ARMS}},
    }
    out["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nsecond arm: {json.dumps(out['second_arm_rule'])}")
    for k, v in out["predictions"].items():
        print(f"  {k:<40} {v['kind']:<9} held={v['held']}")
    print(f"wrote {OUT.relative_to(ROOT)} in {out['seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
