#!/usr/bin/env python3
"""Study AM: can the screen ORDER a toxin above a length-matched non-toxin, at 828 pairs?

`docs/WITHIN_FOLD_ORDERING_PREREGISTRATION.md`, frozen before this file existed.

Study AH answered this on 31 UniProt-keyword pairs: AUROC 0.4599 and 0.5247, both inside a
0.40-0.60 band. It is the ONE hazard claim section 6 still makes, and 31 pairs is the limit of
what 30.6% reviewed coverage allowed. DeepVIC's axis has 1,126 Exotoxin and 1,121 Motility,
and study AI-3 already built 828 length-matched pairs -- but AI-3 measured the RECOVERY
DIFFERENCE at a fixed threshold, which is a different quantity and did not replicate.

This measures the ordering quantity on those pairs: both groups held out of the SAME fit,
scored by the SAME model, AUROC rather than recovery, 30 seeds, both arms, plus a composition
arm on the identical pairs because study AJ showed 21 numbers reproduce the 35M arm here.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results/v3"
OUT = ROOT / "results/within_fold_ordering.json"
SEEDS, N_TRAIN, N_CAL = 30, 1000, 500
BOOT = 20000
BAND_LO, BAND_HI, BAND_ORDER = 0.40, 0.60, 0.70
AA = "ACDEFGHIKLMNPQRSTVWY"
ARMS = {"esm2_650M": "", "esm2_35M": "_esm2_35M"}


def load(mod):
    spec = importlib.util.spec_from_file_location(mod[:4], ROOT / "src" / mod)
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod[:4]] = m
    spec.loader.exec_module(m)
    return m


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))


def auroc(pos, neg):
    a = np.concatenate([np.asarray(pos, float), np.asarray(neg, float)])
    o = a.argsort()
    r = np.empty(len(a), float)
    r[o] = np.arange(1, len(a) + 1)
    for u in np.unique(a):
        m = a == u
        if m.sum() > 1:
            r[m] = r[m].mean()
    n1 = len(pos)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(neg)))


def hier_boot_auroc(SH, SM, seed=0, b=BOOT):
    """Resample a SEED, then pairs with replacement inside it.

    🔴 The first version averaged the per-seed score vectors and bootstrapped pairs on the
    average. That brackets the AUROC OF THE MEAN SCORES, which is not the statistic reported
    beside it -- the mean of per-seed AUROCs -- and on the 35M arm the point estimate fell
    OUTSIDE its own interval (0.5047 against [0.5395, 0.5899]). Seed-averaging denoises, so
    the averaged-score AUROC sits further from 0.5 than the average of noisy per-seed ones,
    and the gap grows with seed spread: composition's sd is 0.012 and it agreed, 35M's is
    0.069 and it did not. Resampling the seed first brackets the right statistic.
    """
    rng = np.random.default_rng(seed)
    SH, SM = np.asarray(SH, float), np.asarray(SM, float)
    n_seed, n_pair = SH.shape
    d = np.empty(b)
    for i in range(b):
        j = rng.integers(n_seed)
        k = rng.integers(0, n_pair, n_pair)
        d[i] = auroc(SH[j][k], SM[j][k])
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def verdict(a):
    if not np.isfinite(a):
        return "unmeasured"
    if a >= BAND_ORDER:
        return "can order"
    if BAND_LO <= a <= BAND_HI:
        return "cannot order"
    return "between bands"


def composition_features(seqs):
    n = np.array([len(s) for s in seqs], float)
    loglen = np.log(np.maximum(n, 1.0))[:, None]
    comp = np.array([[s.count(c) for c in AA] for s in seqs], float)
    comp /= np.maximum(n[:, None], 1.0)
    return np.hstack([loglen, comp])


def selftest():
    ok = True
    m124, m125 = load("124_deepvic_embed.py"), load("125_deepvic_loco.py")
    recs, meta = m124.ordered()
    LEN = {h: len(s) for h, s in recs}
    cls = np.array(meta["class_of_row"])
    ids = np.array(meta["id_of_row"])
    pos = np.where(np.array(meta["label_of_row"]) == 1)[0]
    exo = [ids[r] for r in pos[cls[pos] == "Exotoxin"]]
    mot = [ids[r] for r in pos[cls[pos] == "Motility"]]
    if (len(exo), len(mot)) != (1126, 1121):
        print(f"  XX {len(exo)}/{len(mot)} Exotoxin/Motility, expected 1126/1121")
        ok = False
    pairs, dropped = m125.match_pairs(exo, mot, LEN)
    # 🔒 the pairing must be study AI-3's, or this measures a different population
    if len(pairs) != 828:
        print(f"  XX {len(pairs)} pairs, AI-3 built 828")
        ok = False
    if dropped != 298:
        print(f"  XX {dropped} dropped, AI-3 dropped 298")
        ok = False
    if len({b for _, b in pairs}) != len(pairs):
        print("  XX a Motility member was reused across pairs")
        ok = False

    # auroc: hand-computable cases, and the tie convention
    if auroc([3, 4, 5], [0, 1, 2]) != 1.0 or auroc([0, 1, 2], [3, 4, 5]) != 0.0:
        print("  XX auroc fails on a perfect separation")
        ok = False
    if abs(auroc([1, 2], [1, 2]) - 0.5) > 1e-12:
        print("  XX auroc on identical samples != 0.5")
        ok = False
    # the bands must discriminate in every direction they define
    for v, want in ((0.50, "cannot order"), (0.75, "can order"),
                    (0.65, "between bands"), (0.20, "between bands")):
        if verdict(v) != want:
            print(f"  XX AUROC {v} is not '{want}'")
            ok = False
    if verdict(float("nan")) != "unmeasured":
        print("  XX a nan AUROC is not reported as unmeasured")
        ok = False
    # the paired bootstrap must separate a planted effect and not invent one
    rng = np.random.default_rng(0)
    lo, hi = hier_boot_auroc(rng.normal(3, 1, (5, 300)), rng.normal(0, 1, (5, 300)))
    if lo < 0.9:
        print(f"  XX hier_boot on a planted effect gave [{lo:.3f}, {hi:.3f}]")
        ok = False
    lo, hi = hier_boot_auroc(rng.normal(0, 1, (5, 300)), rng.normal(0, 1, (5, 300)))
    if not (lo < 0.5 < hi):
        print(f"  XX hier_boot on noise gave [{lo:.3f}, {hi:.3f}], must span 0.5")
        ok = False
    # 🔒 and the interval must BRACKET the statistic reported beside it. A planted
    # seed-to-seed spread is the case the first version failed on.
    SH = np.vstack([rng.normal(m, 1, 300) for m in (0.0, 0.6, 1.2, 1.8, 2.4)])
    SM = rng.normal(0, 1, (5, 300))
    per = np.array([auroc(SH[i], SM[i]) for i in range(5)])
    lo, hi = hier_boot_auroc(SH, SM)
    if not (lo <= per.mean() <= hi):
        print(f"  XX the interval [{lo:.4f}, {hi:.4f}] does not bracket the mean of "
              f"per-seed AUROCs ({per.mean():.4f})")
        ok = False
    # composition features must be the 21 of studies AI-1 and AJ
    f = composition_features(["AAAA", "CCCC"])
    if f.shape[1] != 21 or np.allclose(f[0, 1:], f[1, 1:]):
        print(f"  XX composition features are {f.shape[1]} wide or do not discriminate")
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
    m124, m125 = load("124_deepvic_embed.py"), load("125_deepvic_loco.py")
    recs, meta = m124.ordered()
    LEN = {h: len(s) for h, s in recs}
    ids = np.array(meta["id_of_row"])
    cls = np.array(meta["class_of_row"])
    lab = np.array(meta["label_of_row"])
    pos, neg = np.where(lab == 1)[0], np.where(lab == 0)[0]
    row_of = {v: i for i, v in enumerate(ids)}

    exo = [ids[r] for r in pos[cls[pos] == "Exotoxin"]]
    mot = [ids[r] for r in pos[cls[pos] == "Motility"]]
    pairs, dropped = m125.match_pairs(exo, mot, LEN)
    ih = np.array([row_of[h] for h, _ in pairs])
    im = np.array([row_of[m] for _, m in pairs])
    print(f"{len(pairs)} pairs ({dropped} dropped), "
          f"paired medians exo {int(np.median([LEN[h] for h, _ in pairs]))} / "
          f"mot {int(np.median([LEN[m] for _, m in pairs]))}", flush=True)

    perm = np.random.default_rng(12345).permutation(len(neg))
    cut = 2 * len(neg) // 3
    union, test = neg[perm[:cut]], neg[perm[cut:]]

    Xcomp = composition_features([s for _, s in recs])
    feats = {arm: None for arm in ARMS}
    out = {"built": time.strftime("%Y-%m-%d %H:%M:%S"),
           "study": "AM -- within-fold ordering of toxins against matched non-toxins",
           "preregistration": "docs/WITHIN_FOLD_ORDERING_PREREGISTRATION.md",
           "n_pairs": len(pairs), "n_dropped": dropped, "seeds": a.seeds,
           "bands": {"cannot_order": [BAND_LO, BAND_HI], "can_order_floor": BAND_ORDER},
           "ah_reference": {"n_pairs": 31, "auroc_650M": 0.4599, "auroc_35M": 0.5247},
           "arms": {}}

    for tag in ("esm2_650M", "esm2_35M", "composition"):
        if tag == "composition":
            X = Xcomp
        else:
            X = np.load(RES / f"embeddings_deepvic{ARMS[tag]}.npy")
        feats[tag] = int(X.shape[1])
        for variant in ("permissive", "strict"):
            held = set(ih.tolist()) | set(im.tolist())
            if variant == "strict":
                held |= {int(r) for r in pos[(cls[pos] == "Exotoxin")
                                             | (cls[pos] == "Motility")]}
            tri = np.setdiff1d(pos, np.array(sorted(held)))
            # 🔒 the test partition is scored too, not because AM needs an FPR but because
            # 4.7-5.3% on it is the evidence that this fold is calibrated like studies AI, AJ
            # and AK -- which is what makes their recovery figures comparable to these.
            au, rec_h, rec_m, fpr = [], [], [], []
            sh_acc, sm_acc = [], []
            for seed in range(a.seeds):
                r = np.random.default_rng(seed)
                p = r.permutation(len(union))
                tr, ca = union[p[:N_TRAIN]], union[p[N_TRAIN:N_TRAIN + N_CAL]]
                model = clf().fit(np.vstack([X[tri], X[tr]]),
                                  np.r_[np.ones(len(tri)), np.zeros(len(tr))])
                sh = model.predict_proba(X[ih])[:, 1]
                sm = model.predict_proba(X[im])[:, 1]
                t = float(np.quantile(model.predict_proba(X[ca])[:, 1], 0.95))
                au.append(auroc(sh, sm))
                rec_h.append(float((sh >= t).mean()))
                rec_m.append(float((sm >= t).mean()))
                fpr.append(float((model.predict_proba(X[test])[:, 1] >= t).mean()))
                sh_acc.append(sh)
                sm_acc.append(sm)
            SH, SM = np.array(sh_acc), np.array(sm_acc)
            lo, hi = hier_boot_auroc(SH, SM)
            amean = float(np.mean(au))
            # ⚠️ reported alongside, because the two differ when seed spread is large and the
            # difference is the defect the first version of this file hid
            a_mean_scores = auroc(SH.mean(axis=0), SM.mean(axis=0))
            out["arms"].setdefault(tag, {"n_features": feats[tag]})[variant] = {
                "n_held_out_positives": int(len(held)),
                "n_fitted_positives": int(len(tri)),
                "auroc": amean, "auroc_sd": float(np.std(au, ddof=1)),
                "auroc_of_mean_scores": a_mean_scores,
                "auroc_per_seed_min": float(np.min(au)),
                "auroc_per_seed_max": float(np.max(au)),
                "auroc_hier_ci": [lo, hi],
                "auroc_hier_ci_is": ("a PREDICTION interval for one run -- one seed drawn, "
                                     "then pairs resampled inside it, which is src/110's "
                                     "idiom. A deployment uses one fit, so this is the "
                                     "honest object, but it is NOT a confidence interval "
                                     "for the 30-seed mean and must not be read as one."),
                "auroc_mean_ci": [amean - 1.96 * float(np.std(au, ddof=1)) / np.sqrt(len(au)),
                                  amean + 1.96 * float(np.std(au, ddof=1)) / np.sqrt(len(au))],
                "auroc_mean_ci_is": ("the seed-mean's own interval, sd/sqrt(30), which omits "
                                     "pair sampling entirely and is reported only so the two "
                                     "objects are not conflated"),
                "mean_ci_excludes_half": bool(
                    abs(amean - 0.5) > 1.96 * float(np.std(au, ddof=1)) / np.sqrt(len(au))),
                "ci_brackets_point": bool(lo <= amean <= hi),
                "excludes_half": bool(lo > 0.5 or hi < 0.5),
                "recovery_hazard": float(np.mean(rec_h)),
                "recovery_matched": float(np.mean(rec_m)),
                "test_fpr": float(np.mean(fpr)),
                "verdict": verdict(amean),
                "distance_from_half": abs(amean - 0.5),
            }
            v = out["arms"][tag][variant]
            print(f"  {tag:<11} {variant:<11} AUROC {amean:.4f} (sd {v['auroc_sd']:.4f}, "
                  f"mean-scores {a_mean_scores:.4f}) "
                  f"[{lo:.4f}, {hi:.4f}] brackets={v['ci_brackets_point']} "
                  f"excl0.5={v['excludes_half']}  "
                  f"rec {v['recovery_hazard']*100:.1f}/{v['recovery_matched']*100:.1f}%  "
                  f"FPR {v['test_fpr']*100:.2f}%  "
                  f"-> {v['verdict']}", flush=True)

    A = out["arms"]["esm2_650M"]["strict"]
    B = out["arms"]["esm2_35M"]["strict"]
    C = out["arms"]["composition"]["strict"]
    out["second_arm_rule"] = {
        "arms_agree": A["verdict"] == B["verdict"],
        "verdict": A["verdict"] if A["verdict"] == B["verdict"] else "SPLIT -- indicative only",
    }
    out["predictions"] = {
        "P1_both_arms_cannot_order": {
            "kind": "derived",
            "held": bool(A["verdict"] == "cannot order" and B["verdict"] == "cannot order"),
            "values": {"650M": A["auroc"], "35M": B["auroc"]}},
        "P2_composition_further_from_half_than_650M": {
            "kind": "derived",
            "held": bool(C["distance_from_half"] > A["distance_from_half"]),
            "values": {"composition": C["auroc"], "650M": A["auroc"]}},
        "P3_bootstrap_excludes_half_on_one_arm": {
            "kind": "reasoned",
            "held": bool(A["excludes_half"] or B["excludes_half"]),
            "values": {"650M": A["auroc_hier_ci"], "35M": B["auroc_hier_ci"]}},
    }
    out["vs_AH"] = {
        "ah_auroc": [0.4599, 0.5247], "am_auroc": [A["auroc"], B["auroc"]],
        "both_in_same_band_as_AH": bool(A["verdict"] == "cannot order"
                                        and B["verdict"] == "cannot order"),
        "pairs_ratio": round(len(pairs) / 31, 1),
    }
    out["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nsecond arm: {out['second_arm_rule']}")
    for k, v in out["predictions"].items():
        print(f"  {k:<44} {v['kind']:<9} held={v['held']}")
    print(f"wrote {OUT.relative_to(ROOT)} in {out['seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
