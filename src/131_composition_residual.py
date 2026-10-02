#!/usr/bin/env python3
"""Study AN: is the representation's increment compositional, or information of its own?

`docs/COMPOSITION_RESIDUAL_PREREGISTRATION.md`, frozen before this file existed.

Studies AJ, AK and AM each found composition doing something the embedding was credited with,
and none of them tested whether the embedding's +7-point increment IS composition re-encoded.
Studies Y and Z answered the same shape of question about localization by removing the axis
and re-measuring. This removes the composition subspace:

    X_resid = X - C (CtC)^-1 Ct X

with C the 21 features of studies AI-1, AJ and AM. The residual correlation is reported as
evidence rather than assumed, as study Z did.

⚠️ The projection is fitted on the NEGATIVE UNION only -- no labels, no held-out positives --
so it cannot leak the held-out category. Weaker than per-fold refitting, and stated.
⚠️ A linear projection cannot remove non-linearly encoded composition, and the probe is
linear, so this measures what a LINEAR probe can still use. Section 3 of the preregistration
says so in advance.
"""
import argparse
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results/v3"
LOCO = ROOT / "results/deepvic_loco.json"
COMP = ROOT / "results/composition_arm.json"
OUT = ROOT / "results/composition_residual.json"
AA = "ACDEFGHIKLMNPQRSTVWY"
SEEDS, N_TRAIN, N_CAL, SPEC, FLOOR = 30, 1000, 500, 0.95, 7
BAND_NOT_COMP, BAND_IS_COMP = 0.38, 0.35
BAND_WON_NOT, BAND_WON_IS = 10, 7
ARMS = {"esm2_650M": "", "esm2_35M": "_esm2_35M"}


def load(mod):
    spec = importlib.util.spec_from_file_location(mod[:4], ROOT / "src" / mod)
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod[:4]] = m
    spec.loader.exec_module(m)
    return m


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))


def comp_features(seqs):
    n = np.array([len(s) for s in seqs], float)
    loglen = np.log(np.maximum(n, 1.0))[:, None]
    c = np.array([[s.count(x) for x in AA] for s in seqs], float)
    c /= np.maximum(n[:, None], 1.0)
    return np.hstack([loglen, c])


def residualize(X, C, fit_rows):
    """Remove everything linearly predictable from C, with the map fitted on fit_rows.

    🔒 An intercept column is included, so the mean is removed too; without it the
    projection would leave a constant component of the composition signal behind.
    """
    A = np.hstack([np.ones((C.shape[0], 1)), C])
    Af = A[fit_rows]
    W, *_ = np.linalg.lstsq(Af, X[fit_rows], rcond=None)
    return X - A @ W


def max_abs_corr(X, C):
    """The largest |correlation| between any composition feature and any residual column."""
    def z(M):
        # 🔴 This guarded `s == 0` only. After an exact residualization the columns are zero
        # to about 1e-16, not exactly zero, so dividing by that std turned numerical noise
        # into correlations of 0.98 -- the function reported a PERFECT erasure as no erasure
        # at all. The guard is now relative to the matrix's own scale, and a degenerate
        # column contributes zero rather than noise.
        M = M - M.mean(axis=0, keepdims=True)
        s = M.std(axis=0, keepdims=True)
        tol = 1e-10 * max(float(np.abs(M).max()), 1.0)
        dead = s <= tol
        s = np.where(dead, 1.0, s)
        Z = M / s
        Z[:, dead[0]] = 0.0
        return Z
    zx, zc = z(X), z(C)
    return float(np.abs(zc.T @ zx).max() / X.shape[0])


def selftest():
    ok = True
    # residualize must annihilate an exactly-linear dependence and keep an orthogonal one
    rng = np.random.default_rng(0)
    C = rng.normal(size=(400, 3))
    lin = C @ rng.normal(size=(3, 5))
    orth = rng.normal(size=(400, 5))
    rows = np.arange(400)
    r_lin = residualize(lin, C, rows)
    if np.abs(r_lin).max() > 1e-8:
        print(f"  XX a purely linear dependence survived at {np.abs(r_lin).max():.2e}")
        ok = False
    r_or = residualize(orth, C, rows)
    if np.linalg.norm(r_or) < 0.5 * np.linalg.norm(orth):
        print("  XX an orthogonal signal was destroyed by the projection")
        ok = False
    # 🔒 a CONSTANT offset must be removed too, which is what the intercept column is for
    if np.abs(residualize(np.ones((400, 2)), C, rows)).max() > 1e-8:
        print("  XX a constant column survived the projection; the intercept is missing")
        ok = False
    # 🔴 The first version of this check asked whether the residual OUTSIDE fit_rows was
    # large, and a projection fitted on all rows leaves it large too -- so it passed under
    # both. The discriminating construction gives X two DIFFERENT exact linear maps, one on
    # fit_rows and one off them: fitting on fit_rows only drives the in-fit residual to zero,
    # while fitting on everything compromises and leaves it non-zero.
    half = np.arange(200)
    M1, M2 = rng.normal(size=(3, 4)), rng.normal(size=(3, 4)) * 7.0
    X2 = np.vstack([C[:200] @ M1, C[200:] @ M2])
    r2 = residualize(X2, C, half)
    if np.abs(r2[half]).max() > 1e-8:
        print(f"  XX the in-fit residual is {np.abs(r2[half]).max():.2e}, not zero -- the "
              "projection did not fit on fit_rows alone")
        ok = False
    if np.abs(r2[200:]).max() < 1.0:
        print("  XX the out-of-fit dependence was removed, so fit_rows was ignored")
        ok = False

    # max_abs_corr must be ~0 after residualizing and large before
    before = max_abs_corr(lin, C)
    after = max_abs_corr(r_lin, C)
    if not (before > 0.1 and after < 1e-6):
        print(f"  XX correlation before {before:.3e} after {after:.3e}")
        ok = False

    # the comparison targets must exist and still hold the figures the bands were set from
    for f in (LOCO, COMP):
        if not f.exists():
            print(f"  XX {f.name} missing")
            ok = False
    if ok:
        L, Cj = json.loads(LOCO.read_text()), json.loads(COMP.read_text())
        for arm, w in (("esm2_650M", 0.414), ("esm2_35M", 0.418)):
            r = L["arms"][arm]["full"]["recovery"]
            got = float(np.mean([r[c]["mean"] for c in r]))
            if abs(got - w) > 0.002:
                print(f"  XX {arm} mean recovery {got:.4f}, bands were set from {w}")
                ok = False
        if abs(Cj["arms"]["composition"]["mean_recovery"] - 0.342) > 0.002:
            print("  XX composition's mean recovery is no longer 0.342")
            ok = False
    # the bands must discriminate in all three directions
    def verdict(m, won):
        if m >= BAND_NOT_COMP and won >= BAND_WON_NOT:
            return "not compositional"
        if m <= BAND_IS_COMP and won <= BAND_WON_IS:
            return "is compositional"
        return "partly"
    # 🔒 the band CONSTANTS, which the cases below do not pin: with the floor moved to 0.34
    # every one of them still returned the same verdict, so relaxing it went undetected.
    if (BAND_NOT_COMP, BAND_IS_COMP) != (0.38, 0.35):
        print(f"  XX bands are {(BAND_NOT_COMP, BAND_IS_COMP)}, frozen at (0.38, 0.35)")
        ok = False
    if (BAND_WON_NOT, BAND_WON_IS) != (10, 7):
        print(f"  XX count bands are {(BAND_WON_NOT, BAND_WON_IS)}, frozen at (10, 7)")
        ok = False
    for m, w, want in ((0.40, 12, "not compositional"), (0.33, 5, "is compositional"),
                       (0.365, 9, "partly"), (0.40, 5, "partly"),
                       # ⭐ the case that discriminates the floor itself: 0.36 is below 0.38
                       # and above 0.34, so it is "partly" only while the floor is the frozen one
                       (0.36, 12, "partly")):
        if verdict(m, w) != want:
            print(f"  XX verdict({m}, {w}) is '{verdict(m, w)}', expected '{want}'")
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

    C = comp_features([s for _, s in recs])
    L = json.loads(LOCO.read_text())
    Cj = json.loads(COMP.read_text())
    comp_rec = {r["category"]: r["recovery"]
                for r in Cj["arms"]["composition"]["ranking"]}

    out = {"built": time.strftime("%Y-%m-%d %H:%M:%S"),
           "study": "AN -- is the representation's increment compositional?",
           "preregistration": "docs/COMPOSITION_RESIDUAL_PREREGISTRATION.md",
           "projection_fitted_on": ("the negative union only -- no labels and no held-out "
                                    "positives, so it cannot leak the held-out category"),
           "linear_scope": ("a linear projection cannot remove non-linearly encoded "
                            "composition, and the probe is linear, so this measures what a "
                            "LINEAR probe can still use"),
           "bands": {"not_compositional_floor": BAND_NOT_COMP,
                     "is_compositional_ceiling": BAND_IS_COMP,
                     "won_not_floor": BAND_WON_NOT, "won_is_ceiling": BAND_WON_IS},
           "seeds": a.seeds, "arms": {}}

    for arm, sfx in ARMS.items():
        X = np.load(RES / f"embeddings_deepvic{sfx}.npy").astype(np.float64)
        before = max_abs_corr(X, C)
        Xr = residualize(X, C, union)
        after = max_abs_corr(Xr, C)
        print(f"  {arm}: residual |corr| {before:.3e} -> {after:.3e}", flush=True)

        rec, fpr = {}, []
        for cat in targets:
            hi = pos[cls[pos] == cat]
            tri = pos[cls[pos] != cat]
            rr, ff = [], []
            for seed in range(a.seeds):
                r = np.random.default_rng(seed)
                p = r.permutation(len(union))
                tr, ca = union[p[:N_TRAIN]], union[p[N_TRAIN:N_TRAIN + N_CAL]]
                mdl = clf().fit(np.vstack([Xr[tri], Xr[tr]]),
                                np.r_[np.ones(len(tri)), np.zeros(len(tr))])
                t = float(np.quantile(mdl.predict_proba(Xr[ca])[:, 1], SPEC))
                rr.append(float((mdl.predict_proba(Xr[hi])[:, 1] >= t).mean()))
                ff.append(float((mdl.predict_proba(Xr[test])[:, 1] >= t).mean()))
            rec[cat] = {"n": int(len(hi)), "mean": float(np.mean(rr)),
                        "sd": float(np.std(rr, ddof=1))}
            fpr.append(float(np.mean(ff)))
            print(f"    {cat:<46}{rec[cat]['mean']*100:6.2f}%  "
                  f"(embedding {L['arms'][arm]['full']['recovery'][cat]['mean']*100:5.1f}%, "
                  f"composition {comp_rec[cat]*100:5.1f}%)", flush=True)

        emb = {c: L["arms"][arm]["full"]["recovery"][c]["mean"] for c in targets}
        m_res = float(np.mean([rec[c]["mean"] for c in targets]))
        m_emb = float(np.mean([emb[c] for c in targets]))
        m_comp = float(np.mean([comp_rec[c] for c in targets]))
        won = sum(1 for c in targets if rec[c]["mean"] > comp_rec[c])
        won_emb = sum(1 for c in targets if emb[c] > comp_rec[c])
        inc_before = m_emb - m_comp
        inc_after = m_res - m_comp
        out["arms"][arm] = {
            "residual_corr_before": before, "residual_corr_after": after,
            "recovery": rec,
            "mean_recovery_residual": m_res,
            "mean_recovery_embedding": m_emb,
            "mean_recovery_composition": m_comp,
            "increment_before": inc_before, "increment_after": inc_after,
            "fraction_of_increment_surviving": (inc_after / inc_before
                                                if inc_before > 0 else float("nan")),
            "categories_won_residual": won,
            "categories_won_embedding": won_emb,
            "test_fpr_min": float(min(fpr)), "test_fpr_max": float(max(fpr)),
            "verdict": ("not compositional"
                        if m_res >= BAND_NOT_COMP and won >= BAND_WON_NOT
                        else "is compositional"
                        if m_res <= BAND_IS_COMP and won <= BAND_WON_IS else "partly"),
        }
        v = out["arms"][arm]
        print(f"    mean {m_res*100:.2f}% (embedding {m_emb*100:.2f}%, composition "
              f"{m_comp*100:.2f}%)  increment {inc_before*100:+.2f} -> {inc_after*100:+.2f}pp "
              f"({v['fraction_of_increment_surviving']:.1%} survives)  won {won}/14 "
              f"(was {won_emb}) -> {v['verdict']}", flush=True)

    A, B = out["arms"]["esm2_650M"], out["arms"]["esm2_35M"]
    out["second_arm_rule"] = {"arms_agree": A["verdict"] == B["verdict"],
                              "verdict": A["verdict"] if A["verdict"] == B["verdict"]
                              else "SPLIT -- indicative only"}
    out["predictions"] = {
        "P1_mean_between_bands": {
            "kind": "derived",
            "held": bool(all(BAND_IS_COMP < out["arms"][k]["mean_recovery_residual"]
                             < BAND_NOT_COMP for k in ARMS)),
            "values": {k: out["arms"][k]["mean_recovery_residual"] for k in ARMS}},
        "P2_won_at_least_10_on_one_arm": {
            "kind": "derived",
            "held": bool(any(out["arms"][k]["categories_won_residual"] >= BAND_WON_NOT
                             for k in ARMS)),
            "values": {k: out["arms"][k]["categories_won_residual"] for k in ARMS}},
        "P3_residual_corr_below_1e-6": {
            "kind": "reasoned",
            "held": bool(all(out["arms"][k]["residual_corr_after"] < 1e-6 for k in ARMS)),
            "values": {k: out["arms"][k]["residual_corr_after"] for k in ARMS}},
    }
    out["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nsecond arm: {out['second_arm_rule']}")
    for k, v in out["predictions"].items():
        print(f"  {k:<34} {v['kind']:<9} held={v['held']}")
    print(f"wrote {OUT.relative_to(ROOT)} in {out['seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
