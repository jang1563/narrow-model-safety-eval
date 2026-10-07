#!/usr/bin/env python3
"""Study AR: do the verdicts survive a second draw of negatives?

`docs/SECOND_DRAW_PREREGISTRATION.md`, frozen before this file existed.

All seven DeepVIC studies reproduce default_rng(12345) bit-for-bit, so the whole arm rests on
one draw of 33,456 negatives. This re-runs src/133's fold on default_rng(67890) at both
classifier settings -- fixed C=1.0 and nested-selected C -- which re-tests AI-2's margin
ordering and AP's tuning repair in one pass, and measures how far the LEVELS move.

🔒 src/133's select_C and clf are imported, not reimplemented; the only thing that changes is
the partition seed.
"""
import argparse
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results/v3"
LOCO = ROOT / "results/deepvic_loco.json"
AP = ROOT / "results/estimation_repair.json"
OUT = ROOT / "results/second_draw.json"
SEED_ORIGINAL, SEED_NEW = 12345, 67890
SEEDS, FLOOR = 30, 7
BAND_RHO_OK, BAND_RHO_NOT = 0.60, 0.20
BAND_GAIN_OK, BAND_GAIN_NOT = 0.03, 0.005
ARMS = {"esm2_650M": "", "esm2_35M": "_esm2_35M"}


def load(mod):
    spec = importlib.util.spec_from_file_location(mod[:4], ROOT / "src" / mod)
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod[:4]] = m
    spec.loader.exec_module(m)
    return m


def partition(neg, seed):
    perm = np.random.default_rng(seed).permutation(len(neg))
    cut = 2 * len(neg) // 3
    return neg[perm[:cut]], neg[perm[cut:]]


def selftest():
    ok = True
    m133 = load("133_estimation_repair.py")
    m124 = load("124_deepvic_embed.py")
    if (BAND_RHO_OK, BAND_RHO_NOT) != (0.60, 0.20):
        print(f"  XX rho bands are {(BAND_RHO_OK, BAND_RHO_NOT)}, frozen at (0.60, 0.20)")
        ok = False
    if (BAND_GAIN_OK, BAND_GAIN_NOT) != (0.03, 0.005):
        print(f"  XX gain bands are {(BAND_GAIN_OK, BAND_GAIN_NOT)}, frozen at (0.03, 0.005)")
        ok = False
    if (SEED_ORIGINAL, SEED_NEW) != (12345, 67890):
        print(f"  XX seeds are {(SEED_ORIGINAL, SEED_NEW)}, frozen at (12345, 67890)")
        ok = False
    for fn in ("select_C", "clf", "C_GRID", "N_TRAIN", "N_CAL", "SPEC"):
        if not hasattr(m133, fn):
            print(f"  XX src/133 exposes no {fn}; this study must not reimplement it")
            ok = False

    _, meta = m124.ordered()
    neg = np.where(np.array(meta["label_of_row"]) == 0)[0]
    u0, t0 = partition(neg, SEED_ORIGINAL)
    u1, t1 = partition(neg, SEED_NEW)
    # 🔒 the original partition must be reproduced exactly, or the comparison is against
    # something other than what the seven studies used
    ref = np.random.default_rng(SEED_ORIGINAL).permutation(len(neg))
    cut = 2 * len(neg) // 3
    if not (np.array_equal(u0, neg[ref[:cut]]) and np.array_equal(t0, neg[ref[cut:]])):
        print("  XX the original partition is not reproduced")
        ok = False
    # 🔒 and the new one must be genuinely different, while covering the same negatives
    if np.array_equal(np.sort(u0), np.sort(u1)):
        print("  XX the second draw gives the same union as the first; the seed did nothing")
        ok = False
    if not np.array_equal(np.sort(np.concatenate([u1, t1])), np.sort(neg)):
        print("  XX the second partition does not cover the same negative set")
        ok = False
    if set(u1) & set(t1):
        print("  XX the second partition's union and test overlap")
        ok = False
    frac = len(set(u0) & set(u1)) / len(u0)
    if not (0.55 < frac < 0.78):
        print(f"  XX the two unions share {frac:.1%}; two thirds of a shuffle should share "
              "about two thirds, so one of the draws is not a shuffle")
        ok = False
    print(f"  (the two unions share {frac:.1%} of their rows, as two independent "
          f"two-thirds draws should)")

    for f in (LOCO, AP):
        if not f.exists():
            print(f"  XX {f.name} missing; the first draw's figures are the comparison")
            ok = False
    if ok:
        d = json.loads(AP.read_text())
        for arm, w in (("esm2_650M", 0.5539), ("esm2_35M", 0.4691)):
            if abs(d["arms"][arm]["mean_swept"] - w) > 0.0005:
                print(f"  XX AP's tuned mean for {arm} moved from {w}")
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
    m133, m124 = load("133_estimation_repair.py"), load("124_deepvic_embed.py")
    m30 = load("30_margin_across_arms.py")
    recs, meta = m124.ordered()
    cls = np.array(meta["class_of_row"])
    lab = np.array(meta["label_of_row"])
    pos, neg = np.where(lab == 1)[0], np.where(lab == 0)[0]
    union, test = partition(neg, SEED_NEW)
    targets = sorted(c for c, n in Counter(cls[pos]).items() if n >= FLOOR)
    L = json.loads(LOCO.read_text())
    apd = json.loads(AP.read_text())

    out = {"built": time.strftime("%Y-%m-%d %H:%M:%S"),
           "study": "AR -- do the verdicts survive a second draw of negatives?",
           "preregistration": "docs/SECOND_DRAW_PREREGISTRATION.md",
           "seed_original": SEED_ORIGINAL, "seed_new": SEED_NEW,
           "imported_fold": ("src/133's select_C and clf, imported rather than "
                             "reimplemented; only the partition seed changes"),
           "not_retested": ("study AM's within-fold ordering AUROC and study AQ's fair "
                            "comparison keep the original partition"),
           "bands": {"rho_ok": BAND_RHO_OK, "rho_not": BAND_RHO_NOT,
                     "gain_ok": BAND_GAIN_OK, "gain_not": BAND_GAIN_NOT},
           "seeds": a.seeds, "arms": {}}

    for arm, sfx in ARMS.items():
        E = np.load(RES / f"embeddings_deepvic{sfx}.npy").astype(np.float64)
        fix, swp, chosen, fpr = {}, {}, {}, []
        for ci, cat in enumerate(targets):
            hi = pos[cls[pos] == cat]
            tri = pos[cls[pos] != cat]
            creg, _ = m133.select_C(E, tri, union, ci)
            chosen[cat] = creg
            for tag, cv, store in (("fixed", 1.0, fix), ("swept", creg, swp)):
                rr, ff = [], []
                for seed in range(a.seeds):
                    r = np.random.default_rng(seed)
                    p = r.permutation(len(union))
                    tr = union[p[:m133.N_TRAIN]]
                    ca = union[p[m133.N_TRAIN:m133.N_TRAIN + m133.N_CAL]]
                    mdl = m133.clf(cv).fit(np.vstack([E[tri], E[tr]]),
                                           np.r_[np.ones(len(tri)), np.zeros(len(tr))])
                    t = float(np.quantile(mdl.predict_proba(E[ca])[:, 1], m133.SPEC))
                    rr.append(float((mdl.predict_proba(E[hi])[:, 1] >= t).mean()))
                    ff.append(float((mdl.predict_proba(E[test])[:, 1] >= t).mean()))
                store[cat] = {"mean": float(np.mean(rr)), "sd": float(np.std(rr, ddof=1)),
                              "n": int(len(hi))}
                fpr.append(float(np.mean(ff)))
            print(f"    {cat:<46}fixed {fix[cat]['mean']*100:6.2f}%  "
                  f"swept {swp[cat]['mean']*100:6.2f}% (C={creg:g})  "
                  f"[draw 1: {L['arms'][arm]['full']['recovery'][cat]['mean']*100:5.1f}% / "
                  f"{apd['arms'][arm]['swept'][cat]['mean']*100:5.1f}%]", flush=True)

        mar = [L["arms"][arm]["full"]["margin"][c] for c in targets]
        rho_fix = m30._spearman(mar, [fix[c]["mean"] for c in targets])
        rho_swp = m30._spearman(mar, [swp[c]["mean"] for c in targets])
        m_fix = float(np.mean([fix[c]["mean"] for c in targets]))
        m_swp = float(np.mean([swp[c]["mean"] for c in targets]))
        d1_fix = {c: L["arms"][arm]["full"]["recovery"][c]["mean"] for c in targets}
        d1_swp = {c: apd["arms"][arm]["swept"][c]["mean"] for c in targets}
        shift_fix = float(np.mean([abs(fix[c]["mean"] - d1_fix[c]) for c in targets]))
        shift_swp = float(np.mean([abs(swp[c]["mean"] - d1_swp[c]) for c in targets]))
        out["arms"][arm] = {
            "fixed": fix, "swept": swp, "chosen_C": chosen,
            "distinct_C": sorted(set(chosen.values())),
            "mean_fixed": m_fix, "mean_swept": m_swp,
            "mean_fixed_draw1": float(np.mean([d1_fix[c] for c in targets])),
            "mean_swept_draw1": float(np.mean([d1_swp[c] for c in targets])),
            "AR1_rho_fixed": rho_fix, "AR1_rho_swept": rho_swp,
            "AR1_rho_fixed_draw1": m30._spearman(mar, [d1_fix[c] for c in targets]),
            "AR2_tuning_gain": m_swp - m_fix,
            "AR2_tuning_gain_draw1": (apd["arms"][arm]["mean_swept"]
                                      - apd["arms"][arm]["mean_fixed_C"]),
            "AR3_mean_abs_shift_fixed": shift_fix,
            "AR3_mean_abs_shift_swept": shift_swp,
            "AR3_max_abs_shift_fixed": float(max(abs(fix[c]["mean"] - d1_fix[c])
                                                 for c in targets)),
            "AR3_largest_shift_category": max(targets,
                                              key=lambda c: abs(fix[c]["mean"] - d1_fix[c])),
            "test_fpr": {"min": float(min(fpr)), "max": float(max(fpr))},
            "verdict_AR1": ("survives" if min(rho_fix, rho_swp) >= BAND_RHO_OK
                            else "does not" if min(rho_fix, rho_swp) <= BAND_RHO_NOT
                            else "between bands"),
            "verdict_AR2": ("reproduces" if (m_swp - m_fix) >= BAND_GAIN_OK
                            else "does not" if (m_swp - m_fix) <= BAND_GAIN_NOT
                            else "between bands"),
        }
        v = out["arms"][arm]
        print(f"  {arm}: rho fixed {rho_fix:+.4f} swept {rho_swp:+.4f} "
              f"(draw 1 fixed {v['AR1_rho_fixed_draw1']:+.4f}) -> {v['verdict_AR1']}",
              flush=True)
        print(f"    mean fixed {m_fix*100:.2f}% (draw 1 {v['mean_fixed_draw1']*100:.2f}%)  "
              f"swept {m_swp*100:.2f}% (draw 1 {v['mean_swept_draw1']*100:.2f}%)  "
              f"gain {v['AR2_tuning_gain']*100:+.2f}pp (draw 1 "
              f"{v['AR2_tuning_gain_draw1']*100:+.2f}pp) -> {v['verdict_AR2']}", flush=True)
        print(f"    AR-3 mean |shift| fixed {shift_fix*100:.2f}pp swept {shift_swp*100:.2f}pp"
              f"  max {v['AR3_max_abs_shift_fixed']*100:.2f}pp "
              f"({v['AR3_largest_shift_category']})  C {v['distinct_C']}  "
              f"FPR {v['test_fpr']['min']*100:.2f}-{v['test_fpr']['max']*100:.2f}%", flush=True)

    A, B = out["arms"]["esm2_650M"], out["arms"]["esm2_35M"]
    out["second_arm_rule"] = {
        "ar1_agree": A["verdict_AR1"] == B["verdict_AR1"],
        "verdict_ar1": (A["verdict_AR1"] if A["verdict_AR1"] == B["verdict_AR1"]
                        else "SPLIT -- indicative only"),
        "ar2_agree": A["verdict_AR2"] == B["verdict_AR2"],
        "verdict_ar2": (A["verdict_AR2"] if A["verdict_AR2"] == B["verdict_AR2"]
                        else "SPLIT -- indicative only")}
    out["predictions"] = {
        "P1_rho_clears_0.60_in_all_four_cells": {
            "kind": "derived",
            "held": bool(all(out["arms"][k][f"AR1_rho_{t}"] >= BAND_RHO_OK
                             for k in ARMS for t in ("fixed", "swept"))),
            "values": {f"{k}/{t}": out["arms"][k][f"AR1_rho_{t}"]
                       for k in ARMS for t in ("fixed", "swept")}},
        "P2_tuning_gain_reproduces": {
            "kind": "derived",
            "held": bool(all(out["arms"][k]["AR2_tuning_gain"] >= BAND_GAIN_OK
                             for k in ARMS)),
            "values": {k: out["arms"][k]["AR2_tuning_gain"] for k in ARMS}},
        "P3_levels_shift_over_3pp": {
            "kind": "reasoned",
            "held": bool(any(out["arms"][k]["AR3_mean_abs_shift_fixed"] > 0.03
                             for k in ARMS)),
            "values": {k: out["arms"][k]["AR3_mean_abs_shift_fixed"] for k in ARMS}},
    }
    out["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nsecond arm: {json.dumps(out['second_arm_rule'])}")
    for k, v in out["predictions"].items():
        print(f"  {k:<42} {v['kind']:<9} held={v['held']}")
    print(f"wrote {OUT.relative_to(ROOT)} in {out['seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else main())
