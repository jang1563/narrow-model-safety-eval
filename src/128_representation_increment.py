#!/usr/bin/env python3
"""Study AK: what does the representation buy, category by category, at matched FPR?

`docs/REPRESENTATION_INCREMENT_PREREGISTRATION.md`, frozen before this file existed.

No fitting. Every number is read from `results/deepvic_loco.json` (the two embedding arms)
and `results/composition_arm.json` (the model-free arms), which were produced by the same
fold with the same negative partition and calibrated to the same 95th percentile -- so their
recovery rates are comparable without adjustment, and the out-of-sample FPRs recorded there
are the evidence for that rather than an assumption.

⚠️ The per-category interval is UNPAIRED. The upstream artifacts record each category's mean
and across-seed SD but not the per-seed vectors, and seeds are shared between the folds, so a
paired interval would be tighter and is not available. The frozen bands are on the COUNT of
categories won and the MEAN increment, neither of which uses the interval.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LOCO = ROOT / "results/deepvic_loco.json"
COMP = ROOT / "results/composition_arm.json"
OUT = ROOT / "results/representation_increment.json"
ARMS = ("esm2_650M", "esm2_35M")
SEEDS = 30
BOOT = 20000
# frozen in section 4
BAND_WON_EARNS, BAND_WON_NOT = 11, 7
BAND_MEAN_EARNS, BAND_MEAN_NOT = 0.10, 0.03


def boot_mean(x, seed=0):
    """Category-level bootstrap on the mean increment. The unit is the category."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    d = x[rng.integers(0, x.size, size=(BOOT, x.size))].mean(axis=1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def verdict(won, mean):
    earns = won >= BAND_WON_EARNS and mean >= BAND_MEAN_EARNS
    notlb = won <= BAND_WON_NOT and mean <= BAND_MEAN_NOT
    if earns:
        return "earns its place"
    if notlb:
        return "not load-bearing"
    return "adds something and not much"


def selftest():
    ok = True
    for f in (LOCO, COMP):
        if not f.exists():
            print(f"  XX {f.name} missing")
            ok = False
    if not ok:
        print("selftest FAIL")
        return 1
    L, C = json.loads(LOCO.read_text()), json.loads(COMP.read_text())

    # 🔒 the comparison is only legitimate if the folds were calibrated alike. The upstream
    # FPRs are the evidence; if any pass drifts off nominal the recoveries are not comparable.
    for arm in ARMS:
        f = L["arms"][arm]["full"]["test_fpr"]
        lo, hi = min(v["mean"] for v in f.values()), max(v["mean"] for v in f.values())
        if not (0.045 < lo and hi < 0.055):
            print(f"  XX {arm} test FPR spans [{lo:.4f}, {hi:.4f}], not within 4.5-5.5%")
            ok = False
    # and the two artifacts must describe the SAME fourteen categories, or the subtraction
    # would silently pair different things
    ecat = set(L["arms"]["esm2_650M"]["full"]["recovery"])
    ccat = {r["category"] for r in C["arms"]["composition"]["ranking"]}
    if ecat != ccat or len(ecat) != 14:
        print(f"  XX category sets differ or are not 14: {len(ecat)} vs {len(ccat)}")
        ok = False
    # the disclosed means must still be what the preregistration disclosed
    want = {"esm2_650M": 0.414, "esm2_35M": 0.418}
    for arm, w in want.items():
        got = float(np.mean([L["arms"][arm]["full"]["recovery"][c]["mean"] for c in ecat]))
        if abs(got - w) > 0.002:
            print(f"  XX {arm} mean recovery {got:.4f}, disclosed {w}")
            ok = False
    if abs(C["arms"]["composition"]["mean_recovery"] - 0.342) > 0.002:
        print(f"  XX composition mean {C['arms']['composition']['mean_recovery']:.4f}, "
              "disclosed 0.342")
        ok = False

    # the bands must discriminate in all three directions
    for w, m, want_v in ((13, 0.15, "earns its place"), (5, 0.01, "not load-bearing"),
                         (9, 0.07, "adds something and not much"),
                         (13, 0.01, "adds something and not much")):
        if verdict(w, m) != want_v:
            print(f"  XX verdict({w}, {m}) is '{verdict(w, m)}', expected '{want_v}'")
            ok = False
    # boot_mean must widen with spread and collapse without it
    lo, hi = boot_mean([0.07] * 14)
    if not (abs(lo - 0.07) < 1e-9 and abs(hi - 0.07) < 1e-9):
        print(f"  XX boot_mean on a constant gave [{lo}, {hi}]")
        ok = False
    lo, hi = boot_mean(list(np.linspace(-0.3, 0.3, 14)))
    if not (lo < 0 < hi and hi - lo > 0.1):
        print(f"  XX boot_mean on a spread gave [{lo:.3f}, {hi:.3f}]")
        ok = False
    print("selftest PASS" if ok else "selftest FAIL")
    return 0 if ok else 1


def main():
    L, C = json.loads(LOCO.read_text()), json.loads(COMP.read_text())
    cats = sorted(L["arms"]["esm2_650M"]["full"]["recovery"])
    comp = {r["category"]: r["recovery"] for r in C["arms"]["composition"]["ranking"]}
    compsd = {c: C["arms"]["composition"]["recovery"][c]["sd"] for c in cats}
    lenr = {r["category"]: r["recovery"] for r in C["arms"]["length_only"]["ranking"]}

    out = {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "study": "AK -- the representation's per-category increment at matched FPR",
        "preregistration": "docs/REPRESENTATION_INCREMENT_PREREGISTRATION.md",
        "nothing_fitted": "every number is read from src/125 and src/127; nothing is fitted",
        "interval_disclosure": (
            "the per-category interval is UNPAIRED: the upstream artifacts record means and "
            "across-seed SDs but not per-seed vectors, and seeds are shared between folds, so "
            "a paired interval would be tighter and is unavailable. The frozen bands are on "
            "the count and the mean and do not use it."),
        "fpr_evidence": {
            arm: {"min": min(v["mean"] for v in
                             L["arms"][arm]["full"]["test_fpr"].values()),
                  "max": max(v["mean"] for v in
                             L["arms"][arm]["full"]["test_fpr"].values())}
            for arm in ARMS},
        "bands": {"won_earns_floor": BAND_WON_EARNS, "won_not_ceiling": BAND_WON_NOT,
                  "mean_earns_floor": BAND_MEAN_EARNS,
                  "mean_not_ceiling": BAND_MEAN_NOT},
        "arms": {},
    }

    for arm in ARMS:
        rec = L["arms"][arm]["full"]["recovery"]
        inc = {c: rec[c]["mean"] - comp[c] for c in cats}
        se = {c: float(np.sqrt(rec[c]["sd"] ** 2 + compsd[c] ** 2) / np.sqrt(SEEDS))
              for c in cats}
        won = [c for c in cats if inc[c] > 0]
        lost = [c for c in cats if inc[c] <= 0]
        pos = [inc[c] for c in won]
        lo, hi = boot_mean([inc[c] for c in cats])
        mean_all = float(np.mean([inc[c] for c in cats]))
        out["arms"][arm] = {
            "per_category": [
                {"category": c, "n": rec[c]["n"],
                 "recovery_embedding": rec[c]["mean"], "recovery_composition": comp[c],
                 "increment": inc[c],
                 "unpaired_ci": [inc[c] - 1.96 * se[c], inc[c] + 1.96 * se[c]],
                 "unpaired_excludes_zero": bool(abs(inc[c]) > 1.96 * se[c]),
                 "increment_vs_length_only": rec[c]["mean"] - lenr[c]}
                for c in sorted(cats, key=lambda c: -inc[c])],
            "AK1_categories_won": len(won),
            "AK1_categories_lost": len(lost),
            "loss_set": sorted(lost),
            "AK3_mean_increment_all": mean_all,
            "AK3_mean_increment_ci_category_bootstrap": [lo, hi],
            "AK3_mean_increment_where_positive": float(np.mean(pos)) if pos else 0.0,
            "AK3_largest_gain": max(cats, key=lambda c: inc[c]),
            "AK3_largest_loss": min(cats, key=lambda c: inc[c]),
            "AK4_mean_increment_vs_length_only": float(
                np.mean([rec[c]["mean"] - lenr[c] for c in cats])),
            "n_unpaired_intervals_excluding_zero": sum(
                1 for c in cats if abs(inc[c]) > 1.96 * se[c]),
            "verdict": verdict(len(won), mean_all),
        }
        v = out["arms"][arm]
        print(f"  {arm}: won {v['AK1_categories_won']}/14  mean increment "
              f"{mean_all * 100:+.2f}pp [{lo * 100:+.2f}, {hi * 100:+.2f}]  "
              f"where positive {v['AK3_mean_increment_where_positive'] * 100:+.2f}pp")
        print(f"       vs length alone {v['AK4_mean_increment_vs_length_only'] * 100:+.2f}pp  "
              f"-> {v['verdict']}")
        print(f"       loss set: {v['loss_set']}")

    A, B = out["arms"]["esm2_650M"], out["arms"]["esm2_35M"]
    out["AK2_loss_sets_identical"] = A["loss_set"] == B["loss_set"]
    out["AK2_loss_set_intersection"] = sorted(set(A["loss_set"]) & set(B["loss_set"]))
    out["second_arm_rule"] = {
        "arms_agree_on_verdict": A["verdict"] == B["verdict"],
        "verdict": (A["verdict"] if A["verdict"] == B["verdict"]
                    else "SPLIT -- indicative only"),
    }
    out["predictions"] = {
        "P1_mean_between_bands": {
            "kind": "derived",
            "held": bool(all(BAND_MEAN_NOT < out["arms"][a]["AK3_mean_increment_all"]
                             < BAND_MEAN_EARNS for a in ARMS)),
            "values": {a: out["arms"][a]["AK3_mean_increment_all"] for a in ARMS}},
        "P2_exotoxin_in_loss_set": {
            "kind": "derived",
            "held": bool(all("Exotoxin" in out["arms"][a]["loss_set"] for a in ARMS)),
            "values": {a: out["arms"][a]["loss_set"] for a in ARMS}},
        "P3_loss_sets_identical_and_small": {
            "kind": "reasoned",
            "held": bool(out["AK2_loss_sets_identical"] and len(A["loss_set"]) <= 3),
            "values": {"identical": out["AK2_loss_sets_identical"],
                       "sizes": [len(A["loss_set"]), len(B["loss_set"])]}},
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nsecond arm: {out['second_arm_rule']}")
    print(f"loss sets identical: {out['AK2_loss_sets_identical']}  "
          f"intersection {out['AK2_loss_set_intersection']}")
    for k, v in out["predictions"].items():
        print(f"  {k:<36} {v['kind']:<9} held={v['held']}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else main())
