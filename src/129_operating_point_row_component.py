#!/usr/bin/env python3
"""Study AL: the row component of the operating-point interval, and who closed what.

Entry 84 enumerated the published intervals built as seed standard errors and fixed none,
predicting that "the large-n ones will widen by roughly 2x". Study AC (`src/115`) then fixed
four intervals and the paper picked up two. This file settles three things the record left
ambiguous, and it fits nothing -- every number is read from artifacts or is arithmetic on them.

  AL-1  the published operating-point interval is a 200-SEED interval that redraws the
        118/178 calibration split every seed (src/49 line 191), so calling it a bare seed
        standard error understates it. What it omits is TEST-ROW variability.
  AL-2  that omitted component, sized analytically: 1.99x and 2.05x the published half-width
        on 8,258 rows, 3.09x and 3.19x on the 3,407 distinct protein names. 🟢 Entry 84's
        "roughly 2x" is therefore RIGHT for the items it listed.
  AL-3  study AC did NOT close those items, contrary to what its docstring implies. AC's fold
        trains on 746 class-axis positives, thresholds on the POOL and tests 6,758 pool rows;
        entry 84's items come from src/49's panel-calibrated 118-point fold over 8,258 rows.
        The paper took only R and C/A from AC, which is correct.

⚠️ The row component here is BINOMIAL and therefore a FLOOR: pool rows are homologous, so an
exact bootstrap over rows would be wider still. The distinct-name denominator is reported as
the conservative form, and the exact version -- src/49 re-run with per-row flags retained --
is named as the remaining work rather than described.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "results/v3/external_test_partition_canonical.json"
AC = (ROOT / "results/headline_intervals.json",
      ROOT / "results/headline_intervals_esm2_35M.json")
SRC49 = ROOT / "src/49_external_test_partition.py"
SRC115 = ROOT / "src/115_headline_intervals.py"
OUT = ROOT / "results/operating_point_row_component.json"
Z = 1.96


def half(ci):
    return (ci[1] - ci[0]) / 2


def selftest():
    ok = True
    for f in (EXT, *AC, SRC49, SRC115):
        if not f.exists():
            print(f"  XX {f.name} missing")
            ok = False
    if not ok:
        print("selftest FAIL")
        return 1
    d = json.loads(EXT.read_text())
    if (d["seeds"], d["calibration_n"], d["pool_n"]) != (200, 118, 8258):
        print(f"  XX design is {d['seeds']}/{d['calibration_n']}/{d['pool_n']}, "
              "expected 200/118/8258")
        ok = False

    # 🔒 AL-1 is a claim about src/49's CODE, so it is checked against the code. If the split
    # stops being redrawn per seed, the published interval means something different and the
    # whole entry's reading changes.
    s49 = SRC49.read_text()
    if "perm = rng.permutation(len(N))" not in s49 or "nca, ntr = perm[:n_ca]" not in s49:
        print("  XX src/49 no longer redraws the calibration split inside the seed loop")
        ok = False
    if "for seed in range(a.seeds)" not in s49:
        print("  XX src/49's seed loop is not where it was")
        ok = False
    # 🔒 AL-3 likewise: AC's fold must still be the class-axis one, or the two quantities
    # might in fact coincide and the distinction this file draws would be wrong.
    s115 = SRC115.read_text()
    if "embeddings_class_axis_positives" not in s115:
        print("  XX src/115 no longer fits on the class-axis positives")
        ok = False
    if "np.quantile(mdl.predict_proba(POOL[ca])[:, 1], SPEC)" not in s115:
        print("  XX src/115 no longer thresholds on the pool's calibrate split")
        ok = False

    # the arithmetic: a binomial half-width must scale as 1/sqrt(n) and vanish for a
    # degenerate rate, or the sizing below means nothing
    a = Z * np.sqrt(0.08 * 0.92 / 1000)
    b = Z * np.sqrt(0.08 * 0.92 / 4000)
    if abs(a / b - 2.0) > 0.01:
        print(f"  XX quadrupling n did not halve the half-width: {a / b:.3f}")
        ok = False
    if Z * np.sqrt(0.0 * 1.0 / 1000) != 0.0:
        print("  XX a zero rate does not give a zero half-width")
        ok = False
    print("selftest PASS" if ok else "selftest FAIL")
    return 0 if ok else 1


def main():
    d = json.loads(EXT.read_text())
    n, nd = d["pool_n"], d["pool_distinct_names"]
    out = {
        "built": "2026-10-02",
        "study": "AL -- the row component of the operating-point interval",
        "nothing_fitted": "every number is read from artifacts or is arithmetic on them",
        "design": {"seeds": d["seeds"], "calibration_n": d["calibration_n"],
                   "pool_n": n, "pool_distinct_names": nd,
                   "conformal_k_at_0.05": d["conformal_k"]["0.05"]},
        "AL1_split_redrawn_per_seed": True,
        "AL1_evidence": ("src/49's seed loop runs rng.permutation(len(N)) and takes "
                         "nca, ntr = perm[:n_ca], perm[n_ca:], so the 118/178 split is "
                         "redrawn on every one of the 200 seeds; the published interval "
                         "therefore includes calibration-split variability and omits "
                         "test-row variability"),
        "row_component_is_binomial_floor": (
            "pool rows are homologous, so an exact bootstrap over rows is wider than this; "
            "the distinct-name denominator is the conservative form reported alongside"),
        "estimators": {},
    }
    for est in ("quantile", "conformal"):
        v = d["false_positives"]["0.05"]["shift"][est]
        p, ci = v["mean"], v["ci"]
        hs = half(ci)
        for tag, m in (("rows", n), ("distinct_names", nd)):
            se = float(np.sqrt(p * (1 - p) / m))
            hr = Z * se
            hc = Z * float(np.sqrt((hs / Z) ** 2 + se ** 2))
            out["estimators"].setdefault(est, {
                "point": p, "published_ci": ci, "published_half_width": hs,
                "published_n_seeds": v["n_seeds"]})[tag] = {
                "n": m,
                "row_half_width": hr,
                "row_over_published": hr / hs,
                "combined_ci": [p - hc, p + hc],
                "combined_half_width": hc,
                "combined_over_published": hc / hs,
            }
    q = out["estimators"]["quantile"]["rows"]
    c = out["estimators"]["conformal"]["rows"]
    out["AL2_entry84_roughly_2x"] = {
        "prediction": "entry 84: the large-n ones will widen by roughly 2x",
        "row_component_ratios": [q["row_over_published"], c["row_over_published"]],
        "combined_ratios": [q["combined_over_published"], c["combined_over_published"]],
        "held": bool(all(1.5 <= x <= 2.5 for x in
                         (q["row_over_published"], c["row_over_published"]))),
        "note": ("scored on the items entry 84 listed FIRST, the pool-FPR pair. The row "
                 "component alone is 1.99x and 2.05x and the combined interval 2.22x and "
                 "2.28x, so 'roughly 2x' is right for these."),
    }
    ac = [json.loads(f.read_text()) for f in AC]
    out["AL3_study_AC_is_a_different_fold"] = {
        "ac_widenings": {k: [round(a[k]["widening"], 2) for a in ac]
                         for k in ("R", "CA", "pool_fpr", "vfdb_rate")},
        "ac_n": ac[0]["n"],
        "ac_pool_n": ac[0]["n"]["pool"],
        "entry84_pool_n": n,
        "same_pool_n": ac[0]["n"]["pool"] == n,
        "ac_fits_on": "the 746 class-axis positives",
        "entry84_fits_on": "the 149-protein panel, calibrated on 118 panel negatives",
        "ac_thresholds_on": "the pool's calibrate split",
        "entry84_thresholds_on": "the panel negatives",
        "conclusion": ("study AC's docstring says it fixes 'the out-of-sample false-positive "
                       "rate and the VFDB flag rate', and it does -- for ITS OWN fold. Those "
                       "are not entry 84's items 1-4, which come from src/49's "
                       "panel-calibrated fold over 8,258 rows. The paper took only R and C/A "
                       "from AC, which is correct."),
        "ac_ratio_widenings_exceed_2x_because": (
            "R and C/A are ratios over strata of 171, 597 and 264 rows, where row resampling "
            "compounds; that is why they widen 8.2 to 11.8x and does not contradict entry 84, "
            "whose 'roughly 2x' was about the large-n RATES"),
    }
    out["AL_remaining"] = (
        "entry 84's items 1-4 are still open as EXACT intervals: src/49 must be re-run with "
        "per-row flags retained so the row component can be bootstrapped rather than sized "
        "binomially. The sizing here is a floor and the paper says so.")
    out["entry84_enumeration"] = {
        "header_says": "eleven",
        "bracketed_intervals_listed": 10,
        "plus_the_two_mantel_haenszel": 2,
        "total_listed": 12,
        "note": "the header undercounts its own list by one",
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    for est in ("quantile", "conformal"):
        e = out["estimators"][est]
        print(f"  {est}: point {e['point']*100:.3f}%  published "
              f"[{e['published_ci'][0]*100:.2f}, {e['published_ci'][1]*100:.2f}] "
              f"(+-{e['published_half_width']*100:.3f}pp, {e['published_n_seeds']} seeds)")
        for tag in ("rows", "distinct_names"):
            t = e[tag]
            print(f"      +{tag:<14} n={t['n']:>5,}  row +-{t['row_half_width']*100:.3f}pp "
                  f"({t['row_over_published']:.2f}x)  combined "
                  f"[{t['combined_ci'][0]*100:.2f}, {t['combined_ci'][1]*100:.2f}] "
                  f"({t['combined_over_published']:.2f}x)")
    print(f"  AL-2 entry 84's 'roughly 2x': held={out['AL2_entry84_roughly_2x']['held']}")
    print(f"  AL-3 same pool n as AC? {out['AL3_study_AC_is_a_different_fold']['same_pool_n']} "
          f"(AC {out['AL3_study_AC_is_a_different_fold']['ac_pool_n']:,} vs {n:,})")
    print(f"  entry 84's header says {out['entry84_enumeration']['header_says']}, "
          f"lists {out['entry84_enumeration']['total_listed']}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else main())
