#!/usr/bin/env python3
"""Study AI: apply the frozen bands and the second-arm rule, and score the predictions.

`docs/DEEPVIC_BENCHMARK_PREREGISTRATION.md` § 4-5 and amendment 1.

`src/125` writes per-arm, per-variant numbers. This file does the thing the preregistration
actually committed to: cross the arms, apply the bands, apply amendment 1's reading rule, and
score P3-P6 as held or missed. It fits nothing and measures nothing new -- every number comes
from `results/deepvic_loco.json` or `results/deepvic_metadata_baseline.json`.

🔴 It exists as a separate file because the verdict is not a number in a table; it is a
conjunction over four passes, and writing that conjunction by hand into a paper sentence is
how entry 101 happened.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LOCO = ROOT / "results/deepvic_loco.json"
BASE = ROOT / "results/deepvic_metadata_baseline.json"
OUT = ROOT / "results/deepvic_verdicts.json"
ARMS = ("esm2_650M", "esm2_35M")
VARIANTS = ("full", "shared_removed")
CIRC_PP = 2.0   # section 2's circularity tolerance, in points


def both(d, f):
    """f applied to every (arm, variant) pass, as a flat dict."""
    return {f"{a}/{v}": f(d["arms"][a][v]) for a in ARMS for v in VARIANTS}


def agree(vals):
    return len(set(vals)) == 1


def selftest():
    ok = True
    if agree(["x", "y"]) or not agree(["x", "x"]):
        print("  XX agree() is wrong")
        ok = False
    # both() must visit all four passes, and must not silently skip a missing one
    d = {"arms": {a: {v: {"k": f"{a}{v}"} for v in VARIANTS} for a in ARMS}}
    got = both(d, lambda p: p["k"])
    if len(got) != 4 or set(got) != {f"{a}/{v}" for a in ARMS for v in VARIANTS}:
        print(f"  XX both() returned {got}")
        ok = False
    try:
        both({"arms": {"esm2_650M": {"full": {}}}}, lambda p: 1)
        print("  XX both() tolerated a missing pass instead of raising")
        ok = False
    except KeyError:
        pass

    # the artifacts this file reads must exist and carry the frozen bands, or the
    # verdicts would be computed against defaults that live nowhere
    if not LOCO.exists() or not BASE.exists():
        print("  XX an input artifact is missing")
        ok = False
    else:
        b = json.loads(LOCO.read_text())["bands"]
        want = {"rho_replicate_floor": 0.6, "rho_fail_ceiling": 0.2, "rho_invert": -0.4,
                "ai3_replicate_floor": 0.0, "ai3_hazard_ceiling": -0.2}
        if b != want:
            print(f"  XX the bands in the artifact are {b}, frozen were {want}")
            ok = False
        L = json.loads(LOCO.read_text())
        for a in ARMS:
            for v in VARIANTS:
                p = L["arms"][a][v]
                for k in ("AI2", "AI3a_all", "AI3b_untruncated", "AI3c_length_matched"):
                    if k not in p:
                        print(f"  XX {a}/{v} has no {k}")
                        ok = False
    print("selftest PASS" if ok else "selftest FAIL")
    return 0 if ok else 1


def main():
    L = json.loads(LOCO.read_text())
    M = json.loads(BASE.read_text())
    bands = L["bands"]

    # ---- AI-2 ----------------------------------------------------------------------
    rho = both(L, lambda p: p["AI2"]["rho"])
    pv = both(L, lambda p: p["AI2"]["perm_p"])
    vd = both(L, lambda p: p["AI2"]["verdict"])
    sig = both(L, lambda p: p["AI2"]["significant"])
    ctrl = both(L, lambda p: p["AI2"]["margin_beats_both_controls"])
    rtr = both(L, lambda p: p["AI2"]["rho_truncation_vs_recovery"])
    bimo = both(L, lambda p: p["AI2"]["bimodality"])
    ai2 = {
        "rho": rho, "perm_p": pv, "verdict_per_pass": vd,
        "significant_per_pass": sig,
        "margin_beats_both_controls_per_pass": ctrl,
        "rho_truncation_vs_recovery": rtr,
        "controls": both(L, lambda p: {
            "nearest_positive_alone": p["AI2"]["rho_nearest_positive_alone"],
            "nearest_negative_alone": p["AI2"]["rho_nearest_negative_alone"]}),
        "all_four_in_replication_band": all(v == "replicates" for v in vd.values()),
        "all_four_significant": all(sig.values()),
        "all_four_beat_both_controls": all(ctrl.values()),
        "arms_agree": agree([vd[f"{a}/full"] for a in ARMS]),
        "min_rho": min(rho.values()), "max_perm_p": max(pv.values()),
        # 🔴 truncation rate is a rival explanation for the ordering, so it is reported
        # next to margin rather than left implicit. On 35M it reaches +0.365.
        "max_rho_truncation": max(rtr.values()),
        "margin_exceeds_truncation_rho_everywhere": all(
            rho[k] > rtr[k] for k in rho),
        # ⚠️ the preregistration also asked whether recovery is BIMODAL as on our panel
        "bimodality": {
            k: {"gap_over_spread": v["gap_over_spread"],
                "gap_after_rank": v["gap_after_rank"], "spread": v["spread"]}
            for k, v in bimo.items()},
        "bimodal_gap_location_agrees_across_arms": agree(
            [bimo[f"{a}/full"]["gap_after_rank"] for a in ARMS]),
        "max_gap_over_spread": max(v["gap_over_spread"] for v in bimo.values()),
    }
    ai2["verdict"] = (
        "REPLICATES on both arms" if (ai2["all_four_in_replication_band"]
                                      and ai2["all_four_significant"]
                                      and ai2["arms_agree"])
        else "SPLIT or below band -- indicative only")

    # ---- AI-3, three variants, amendment 1's reading rule --------------------------
    ai3 = {}
    for tag, key in (("a_all", "AI3a_all"),
                     ("b_untruncated", "AI3b_untruncated"),
                     ("c_length_matched", "AI3c_length_matched")):
        diff = both(L, lambda p, k=key: p[k]["diff"])
        ver = both(L, lambda p, k=key: p[k]["verdict"])
        ci = both(L, lambda p, k=key: p[k]["ci"])
        ai3[tag] = {
            "diff": diff, "ci": ci, "verdict_per_pass": ver,
            "excludes_zero_per_pass": both(L, lambda p, k=key: p[k]["excludes_zero"]),
            "arms_agree_on_verdict": agree([ver[f"{a}/full"] for a in ARMS]),
            "arms_agree_on_sign": agree(
                [bool(diff[f"{a}/full"] > 0) for a in ARMS]),
            "all_four_in_replication_band": all(
                v == "replicates the narrowing" for v in ver.values()),
        }
    primary = ai3["c_length_matched"]
    ai3["reading_rule"] = L["arms"]["esm2_650M"]["full"][
        "AI3c_length_matched"]["reading_rule"]
    ai3["primary_is_c"] = True
    ai3["verdict"] = (
        "REPLICATES on both arms" if primary["all_four_in_replication_band"]
        else "DOES NOT REPLICATE -- amendment 1's reading rule is not met")
    # 🔴 where the disagreement lives: a per-class arm difference, not a global one
    A, B = L["arms"]["esm2_650M"]["full"], L["arms"]["esm2_35M"]["full"]
    cls = sorted(A["recovery"])
    da = {c: A["recovery"][c]["mean"] - B["recovery"][c]["mean"] for c in cls}
    ai3["arm_difference_localization"] = {
        "mean_recovery_650M": float(np.mean([A["recovery"][c]["mean"] for c in cls])),
        "mean_recovery_35M": float(np.mean([B["recovery"][c]["mean"] for c in cls])),
        "n_classes_where_35M_higher": int(sum(1 for c in cls if da[c] < 0)),
        "n_classes": len(cls),
        "exotoxin_arm_gap_pp": da["Exotoxin"] * 100,
        "motility_arm_gap_pp": da["Motility"] * 100,
        "largest_abs_arm_gap_class": max(cls, key=lambda c: abs(da[c])),
        "exotoxin_seed_sd": {"650M": A["recovery"]["Exotoxin"]["sd"],
                             "35M": B["recovery"]["Exotoxin"]["sd"]},
        "exotoxin_zero_seeds": {"650M": A["recovery"]["Exotoxin"]["zero_seeds"],
                                "35M": B["recovery"]["Exotoxin"]["zero_seeds"]},
        "note": ("the arms have near-identical mean recovery and split 7/7 on direction, "
                 "so AI-3's disagreement is not a globally weaker arm; it is Exotoxin "
                 "specifically, and not seed instability either"),
    }

    # ---- AI-4, the circularity check across every figure --------------------------
    moves = {
        "ai2_rho_pp": {a: (L["arms"][a]["shared_removed"]["AI2"]["rho"]
                           - L["arms"][a]["full"]["AI2"]["rho"]) * 100 for a in ARMS},
        "ai3c_diff_pp": {a: (L["arms"][a]["shared_removed"]["AI3c_length_matched"]["diff"]
                             - L["arms"][a]["full"]["AI3c_length_matched"]["diff"]) * 100
                         for a in ARMS},
        "ai1_length_pp": M["ai4_shared_removal_moves"]["length_only_pp"],
        "ai1_composition_pp": M["ai4_shared_removal_moves"]["composition_pp"],
    }
    allmoves = (list(moves["ai2_rho_pp"].values()) + list(moves["ai3c_diff_pp"].values())
                + [moves["ai1_length_pp"], moves["ai1_composition_pp"]])
    # 🔴 Section 2 froze "moves no figure by more than 2pp" and did NOT say what unit a
    # Spearman's "pp" is. Multiplying rho by 100 is the literal reading and it FAILS, on the
    # 35M arm, by 0.64. Rho is not measured in percentage points, so the literal reading is
    # arguably a category error -- but it is the frozen one, and reinterpreting a band after
    # seeing the number is what the bands exist to prevent. Both readings are reported, the
    # literal one is the verdict, and the ambiguity is recorded as a drafting defect.
    ppmoves = (list(moves["ai3c_diff_pp"].values())
               + [moves["ai1_length_pp"], moves["ai1_composition_pp"]])
    ai4 = {"moves_pp": moves, "max_abs_move_pp": float(max(abs(x) for x in allmoves)),
           "tolerance_pp": CIRC_PP,
           "all_under_tolerance": bool(max(abs(x) for x in allmoves) < CIRC_PP),
           "which_figure_exceeded": max(
               [("ai2_rho/" + a, moves["ai2_rho_pp"][a]) for a in ARMS]
               + [("ai3c/" + a, moves["ai3c_diff_pp"][a]) for a in ARMS]
               + [("ai1_length", moves["ai1_length_pp"]),
                  ("ai1_composition", moves["ai1_composition_pp"])],
               key=lambda kv: abs(kv[1]))[0],
           "rate_figures_only": {
               "max_abs_move_pp": float(max(abs(x) for x in ppmoves)),
               "all_under_tolerance": bool(max(abs(x) for x in ppmoves) < CIRC_PP),
               "note": ("recovery differences and AUROCs, which ARE in percentage points. "
                        "Every one moves under 2pp. The only figure over the band is the "
                        "Spearman, read as rho x 100.")},
           "spearman_moves_rho_units": {a: moves["ai2_rho_pp"][a] / 100 for a in ARMS},
           "band_ambiguity": ("section 2 wrote the tolerance in pp without saying whether a "
                              "correlation counts in those units; the literal reading is "
                              "used for the verdict and the decomposition is reported"),
           "n_shared": M["n_shared_with_our_panel"]}

    res = {
        "built": "2026-10-01",
        "study": "AI verdicts -- frozen bands, the second-arm rule, and the prediction score",
        "preregistration": L["preregistration"],
        "nothing_fitted": "every number is read from src/123 and src/125; nothing is fitted here",
        "bands": bands,
        "AI2": ai2, "AI3": ai3, "AI4": ai4,
        "predictions": {
            "P3_AI2_clears_0.60_both_arms": {
                "kind": "derived",
                "held": bool(ai2["all_four_in_replication_band"]
                             and ai2["all_four_significant"]),
                "values": ai2["rho"]},
            "P4_AI3_nonnegative_both_arms": {
                "kind": "derived",
                "held": primary["all_four_in_replication_band"],
                "values": primary["diff"]},
            "P5_shared_removal_under_2pp": {
                "kind": "reasoned", "held": ai4["all_under_tolerance"],
                "values": moves},
            "P6_AI3a_exceeds_AI3c": {
                "kind": "derived",
                "held": all(L["arms"][a][v]["AI3_P6_a_exceeds_c"]
                            for a in ARMS for v in VARIANTS),
                "values": both(L, lambda p: p["AI3_P6_a_exceeds_c"])},
        },
    }
    res["score"] = {
        "derived_held": sum(1 for k, v in res["predictions"].items()
                            if v["kind"] == "derived" and v["held"]),
        "derived_total": sum(1 for v in res["predictions"].values()
                             if v["kind"] == "derived"),
        "reasoned_held": sum(1 for v in res["predictions"].values()
                             if v["kind"] == "reasoned" and v["held"]),
        "reasoned_total": sum(1 for v in res["predictions"].values()
                              if v["kind"] == "reasoned"),
    }
    OUT.write_text(json.dumps(res, indent=2) + "\n")

    print(f"AI-2  {ai2['verdict']}")
    print(f"      rho {[round(ai2['rho'][k], 4) for k in ai2['rho']]}  "
          f"max p {ai2['max_perm_p']:.5f}  beats both controls everywhere "
          f"{ai2['all_four_beat_both_controls']}")
    print(f"      truncation rho up to {ai2['max_rho_truncation']:+.4f}; margin exceeds it "
          f"everywhere {ai2['margin_exceeds_truncation_rho_everywhere']}")
    print(f"      bimodality gap/spread <= {ai2['max_gap_over_spread']:.3f}, gap location "
          f"agrees across arms {ai2['bimodal_gap_location_agrees_across_arms']}")
    print(f"AI-3  {ai3['verdict']}")
    for tag in ("a_all", "b_untruncated", "c_length_matched"):
        v = ai3[tag]
        print(f"      {tag:<18} 650M {v['diff']['esm2_650M/full'] * 100:+6.2f}pp  "
              f"35M {v['diff']['esm2_35M/full'] * 100:+6.2f}pp  "
              f"sign agrees {v['arms_agree_on_sign']}")
    lo = ai3["arm_difference_localization"]
    print(f"      split is Exotoxin ({lo['exotoxin_arm_gap_pp']:+.1f}pp between arms) not "
          f"global (means {lo['mean_recovery_650M'] * 100:.1f}% / "
          f"{lo['mean_recovery_35M'] * 100:.1f}%, {lo['n_classes_where_35M_higher']}/"
          f"{lo['n_classes']} classes higher on 35M)")
    print(f"AI-4  max move {ai4['max_abs_move_pp']:.3f}pp over {ai4['n_shared']} shared, "
          f"under {CIRC_PP}pp: {ai4['all_under_tolerance']}")
    for k, v in res["predictions"].items():
        print(f"  {k:<34} {v['kind']:<9} held={v['held']}")
    print(f"  derived {res['score']['derived_held']}/{res['score']['derived_total']}, "
          f"reasoned {res['score']['reasoned_held']}/{res['score']['reasoned_total']}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else main())
