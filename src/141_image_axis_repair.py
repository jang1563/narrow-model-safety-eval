#!/usr/bin/env python3
"""Study AU: the image axis had no room to move.

`docs/IMAGE_AXIS_REPAIR_PREREGISTRATION.md`, frozen before this file existed.

🔒 ONE change from study AT: the held-out unit is the CIFAR-100 superclass, not the fine
class. Same embeddings -- the caches study AT built, reused byte for byte -- same animal vs
non-animal split, same two encoders, src/125's fold unchanged, both classifier settings.

🔑 A held-out fine class leaves forty-nine siblings in training and is recovered at 96%, so
there is no variance for any quantity to predict. Holding out `large_carnivores` removes
bear, leopard, lion, tiger and wolf together.

⚠️ This does NOT touch study AT's verdicts. Amendment 3 of AT declined to reclassify the
fine-class cells and they stand as they fell.

🔒 AU-0a is TWO-SIDED, which AT-0a was not: mean recovery must be inside [20%, 90%]. Above
90% the fold is at the ceiling and rho is uninterpretable whatever it is -- the defect
EXTERNAL_VALIDATION made in the other direction and AT inherited.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/image_axis_repair.json"
CELLS = ("image_sc/clip_vit_b32", "image_sc/dinov2_vits14")
BAND_OK, BAND_NOT = 0.60, 0.20
ALPHA = 0.05 / 3
AU0_RECOVERY = (0.20, 0.90)
AU0_FPR = (0.045, 0.056)
AU3_MEAN_CEILING, AU3_SD_FLOOR, AU3_SD_CEILING = 0.90, 0.10, 0.06


def load(mod):
    spec = importlib.util.spec_from_file_location(mod[:4], ROOT / "src" / mod)
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod[:4]] = m
    spec.loader.exec_module(m)
    return m


def au0(sv):
    """AU-0: the two-sided recovery precondition and the matched operating point."""
    mr = sv["mean_recovery"]
    return {
        "AU0a_recovery_in_band": bool(AU0_RECOVERY[0] < mr < AU0_RECOVERY[1]),
        "AU0a_at_ceiling": bool(mr >= AU0_RECOVERY[1]),
        "AU0a_near_null": bool(mr <= AU0_RECOVERY[0]),
        "AU0b_fpr_in_band": bool(AU0_FPR[0] <= sv["test_fpr"]["min"]
                                 and sv["test_fpr"]["max"] <= AU0_FPR[1]),
    }


def spread(sv):
    v = np.array([r["mean"] for r in sv["recovery"].values()])
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
            "min": float(v.min()), "max": float(v.max()),
            "n_at_99pct": int((v >= 0.99).sum()), "n": int(len(v))}


def verdict(cells, at_cells=None):
    """AU-1, AU-2 and AU-3, read together. AU-3 gates AU-1 via AU-0a."""
    interp = {k: v for k, v in cells.items()
              if all(v["settings"][s]["AU0a_recovery_in_band"]
                     and v["settings"][s]["AU0b_fpr_in_band"]
                     for s in ("fixed", "swept"))}
    cs = [(k, s) for k in cells for s in ("fixed", "swept")]
    clears = sum(1 for k, s in cs if k in interp
                 and cells[k]["settings"][s]["AT1_rho"] >= BAND_OK)
    fails = sorted(f"{k}/{s}" for k, s in cs if k in interp
                   and cells[k]["settings"][s]["AT1_rho"] <= BAND_NOT)
    beats = sum(1 for k, s in cs if cells[k]["settings"][s]["AT2_beats_all_rivals"])
    sds = {f"{k}/{s}": cells[k]["settings"][s]["spread"]["sd"] for k, s in cs}
    means = {f"{k}/{s}": cells[k]["settings"][s]["spread"]["mean"] for k, s in cs}
    au3 = (bool(means) and all(m < AU3_MEAN_CEILING for m in means.values())
           and all(d > AU3_SD_FLOOR for d in sds.values()))
    au3_fail = bool(sds) and any(d <= AU3_SD_CEILING for d in sds.values())
    out = {
        "cells_run": len(cells), "cells_expected": len(CELLS),
        "complete": len(cells) == len(CELLS),
        "AU0_interpretable": sorted(interp),
        "AU0_uninterpretable": sorted(set(cells) - set(interp)),
        "AU0_at_ceiling": sorted(f"{k}/{s}" for k, s in cs
                                 if cells[k]["settings"][s]["AU0a_at_ceiling"]),
        "AU1_clears": clears, "AU1_denominator": 2 * len(interp),
        "AU1_fails": fails,
        "AU2_beats_rivals": beats, "AU2_denominator": len(cs),
        "AU3_spread_opened": au3, "AU3_failed": au3_fail,
        "AU3_sd": sds, "AU3_mean": means,
    }
    if not out["complete"]:
        out["verdict"] = (f"INCOMPLETE -- {len(cells)} of {len(CELLS)} cells; AU-1 needs "
                          "both encoders because one encoder is one arm")
        return out
    if not au3:
        out["verdict"] = ("AXIS NOT REPAIRED -- AU-3 did not open the spread, so AU-1 is "
                          "uninterpretable by AU-0a rather than informative")
    elif out["AU1_denominator"] == 0:
        out["verdict"] = "UNINTERPRETABLE -- every cell failed AU-0"
    elif clears == out["AU1_denominator"]:
        out["verdict"] = ("REPAIRED and margin orders it -- study AT's image result was "
                          "about the axis, not about images")
    elif fails:
        out["verdict"] = ("REPAIRED but margin does NOT order it -- the variance is real "
                          "and the quantity is genuinely weaker on images")
    else:
        out["verdict"] = "REPAIRED, AU-1 between bands"
    return out


def predictions(cells, at_path=ROOT / "results/generality_grid.json"):
    if len(cells) != len(CELLS):
        return None
    cs = [(k, s) for k in cells for s in ("fixed", "swept")]
    sds = [cells[k]["settings"][s]["spread"]["sd"] for k, s in cs]
    means = [cells[k]["settings"][s]["spread"]["mean"] for k, s in cs]
    at = json.loads(at_path.read_text())["cells"] if at_path.exists() else {}
    at_sd = {}
    for k in ("image/clip_vit_b32", "image/dinov2_vits14"):
        if k in at:
            for s in ("fixed", "swept"):
                v = np.array([r["mean"]
                              for r in at[k]["settings"][s]["recovery"].values()])
                at_sd[f"{k}/{s}"] = float(v.std(ddof=1))
    beats = {f"{k}/{s}": bool(cells[k]["settings"][s]["AT2_beats_all_rivals"])
             for k, s in cs}
    rho = {f"{k}/{s}": cells[k]["settings"][s]["AT1_rho"] for k, s in cs}
    return {
        "P1_spread_opens_up": {
            "kind": "derived",
            "held": bool(all(m < AU3_MEAN_CEILING for m in means)
                         and all(d > AU3_SD_FLOOR for d in sds)),
            "values": {"mean_recovery": means, "sd": sds,
                       "study_AT_fine_class_sd": at_sd}},
        "P2_margin_beats_nearest_positive_4_of_4": {
            "kind": "derived",
            "held": bool(all(beats.values())),
            "values": beats},
        "P3_rho_clears_on_both_encoders": {
            "kind": "reasoned",
            "held": bool(all(x >= BAND_OK for x in rho.values())),
            "values": rho},
        "P4_second_draw_moves_more_than_fine_class": _p4(),
    }


def _p4(path=ROOT / "results/image_axis_repair_draw67890.json"):
    """P4: does the ten-unit axis move MORE between negative draws than the fifty-unit one?

    🔴 It does not, and the prediction was derived from unit COUNT alone. Study AT's
    fifty-class image cells moved 0.0029-0.0473; this study's ten superclasses move
    0.0000-0.0242. 🔑 A superclass holds 2,500 members against a fine class's 500, so the
    per-unit recovery estimate is five times better resolved -- stability depends on unit
    SIZE as well as unit count, and P4 read only the count.
    """
    at = [0.0029, 0.0010, 0.0306, 0.0473]
    if not path.exists():
        return {"kind": "derived", "held": None,
                "note": "unscored until the second-draw run lands",
                "values": {"study_AT_shifts": at}}
    d = json.loads(path.read_text())
    mine = [abs(x["rho_shift"]) for v in d["cells"].values()
            for x in v["settings"].values() if "rho_shift" in x]
    return {"kind": "derived",
            "held": bool(mine) and max(mine) > max(at),
            "values": {"study_AU_shifts": sorted(mine),
                       "study_AT_fine_class_shifts": at,
                       "max_AU": max(mine) if mine else None, "max_AT": max(at),
                       "units": {"AU": "10 superclasses x 2,500 members",
                                 "AT": "50 fine classes x 500 members"}}}


def selftest():
    ok = True
    g = load("139_generality_grid.py")
    if sorted(CELLS) != sorted(g.REPAIR):
        print(f"  XX this study's cells {sorted(CELLS)} are not src/139's "
              f"{sorted(g.REPAIR)}")
        ok = False
    if AU0_RECOVERY != (0.20, 0.90):
        print(f"  XX AU-0a is {AU0_RECOVERY}, frozen at (0.20, 0.90)")
        ok = False
    # 🔒 the CEILING is the whole point of AU-0a, so it must actually fire
    hi = {"mean_recovery": 0.96, "test_fpr": {"min": 0.049, "max": 0.051}}
    r = au0(hi)
    if r["AU0a_recovery_in_band"] or not r["AU0a_at_ceiling"]:
        print(f"  XX a 96% mean recovery passed AU-0a: {r}")
        ok = False
    lo = {"mean_recovery": 0.05, "test_fpr": {"min": 0.049, "max": 0.051}}
    r = au0(lo)
    if r["AU0a_recovery_in_band"] or not r["AU0a_near_null"]:
        print(f"  XX a 5% mean recovery passed AU-0a: {r}")
        ok = False
    ok_mid = au0({"mean_recovery": 0.55, "test_fpr": {"min": 0.049, "max": 0.051}})
    if not (ok_mid["AU0a_recovery_in_band"] and ok_mid["AU0b_fpr_in_band"]):
        print(f"  XX a 55% mean recovery at 5% FPR failed AU-0a: {ok_mid}")
        ok = False
    if au0({"mean_recovery": 0.55, "test_fpr": {"min": 0.041, "max": 0.044}})[
            "AU0b_fpr_in_band"]:
        print("  XX an out-of-band FPR passed AU-0b")
        ok = False

    def cell(rho, mean, sd, beats=True, fpr=(0.049, 0.051)):
        rec = {f"c{i}": {"mean": float(x)} for i, x in
               enumerate(np.linspace(mean - sd * 1.5, mean + sd * 1.5, 10))}
        sv = {"AT1_rho": rho, "recovery": rec, "mean_recovery": mean,
              "test_fpr": {"min": fpr[0], "max": fpr[1]},
              "AT2_beats_all_rivals": beats}
        sv["spread"] = spread(sv)
        sv.update(au0(sv))
        return {"settings": {"fixed": dict(sv), "swept": dict(sv)}}

    # repaired and ordered
    good = {k: cell(0.8, 0.55, 0.18) for k in CELLS}
    v = verdict(good)
    if "REPAIRED and margin orders it" not in v["verdict"] or v["AU1_clears"] != 4:
        print(f"  XX a clearing repaired grid read {v['verdict']!r} / {v['AU1_clears']}")
        ok = False
    # repaired but margin fails -- the preregistration's strongest outcome
    bad = {k: cell(0.1, 0.55, 0.18) for k in CELLS}
    v = verdict(bad)
    if "does NOT order it" not in v["verdict"] or v["AU1_fails"] == []:
        print(f"  XX a repaired-but-unordered grid read {v['verdict']!r}")
        ok = False
    # 🔒 still at the ceiling: AU-1 must NOT be read, whatever rho is
    sat = {k: cell(0.9, 0.96, 0.05) for k in CELLS}
    v = verdict(sat)
    if "AXIS NOT REPAIRED" not in v["verdict"]:
        print(f"  XX a saturated grid returned an AU-1 reading: {v['verdict']!r}")
        ok = False
    if v["AU1_denominator"] != 0:
        print(f"  XX a saturated grid kept cells in AU-1's denominator: {v}")
        ok = False
    # one encoder only must refuse
    v = verdict({CELLS[0]: good[CELLS[0]]})
    if not v["verdict"].startswith("INCOMPLETE"):
        print(f"  XX one encoder returned a verdict: {v['verdict']!r}")
        ok = False
    # predictions, both ways
    pr = predictions(good)
    if not (pr["P1_spread_opens_up"]["held"] and pr["P3_rho_clears_on_both_encoders"]["held"]):
        print(f"  XX P1/P3 did not hold on a clearing repaired grid: {pr}")
        ok = False
    pr = predictions(sat)
    if pr["P1_spread_opens_up"]["held"]:
        print("  XX P1 held on a saturated grid")
        ok = False
    pr = predictions({k: cell(0.8, 0.55, 0.18, beats=False) for k in CELLS})
    if pr["P2_margin_beats_nearest_positive_4_of_4"]["held"]:
        print("  XX P2 held with margin beaten everywhere")
        ok = False
    if predictions({CELLS[0]: good[CELLS[0]]}) is not None:
        print("  XX predictions were scored on one encoder")
        ok = False
    # 🔒 P4 both ways, and unscored when the stability run has not landed
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "d.json"
        if _p4(f)["held"] is not None:
            print("  XX P4 was scored with no second-draw artifact")
            ok = False
        f.write_text(json.dumps({"cells": {"c": {"settings": {
            "fixed": {"rho_shift": 0.2}, "swept": {"rho_shift": -0.01}}}}}))
        if not _p4(f)["held"]:
            print("  XX P4 failed with a 0.20 shift against a 0.0473 maximum")
            ok = False
        f.write_text(json.dumps({"cells": {"c": {"settings": {
            "fixed": {"rho_shift": 0.001}, "swept": {"rho_shift": -0.002}}}}}))
        if _p4(f)["held"]:
            print("  XX P4 held with shifts an order of magnitude smaller")
            ok = False
    print("selftest PASS" if ok else "selftest FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=30)
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    t0 = time.time()
    g = load("139_generality_grid.py")
    m30, m133 = load("30_margin_across_arms.py"), load("133_estimation_repair.py")
    out = json.loads(OUT.read_text()) if OUT.exists() else {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "study": "AU -- the image axis had no room to move",
        "preregistration": "docs/IMAGE_AXIS_REPAIR_PREREGISTRATION.md",
        "one_change": ("the held-out unit is the CIFAR-100 SUPERCLASS, not the fine class; "
                       "same embeddings, same split, same fold, same encoders"),
        "does_not_touch": ("study AT's verdicts stand as they fell; amendment 3 of AT "
                           "declined to reclassify the fine-class cells"),
        "bands": {"ok": BAND_OK, "not": BAND_NOT, "alpha": ALPHA,
                  "AU0a_recovery": list(AU0_RECOVERY), "AU0b_fpr": list(AU0_FPR),
                  "AU3_mean_ceiling": AU3_MEAN_CEILING,
                  "AU3_sd_floor": AU3_SD_FLOOR, "AU3_sd_ceiling": AU3_SD_CEILING},
        "seeds": a.seeds, "cells": {}}

    for cell in CELLS:
        if cell in out["cells"]:
            continue
        res = g.score_cell(cell, a.seeds, m30, m133, a.jobs)
        for s in ("fixed", "swept"):
            sv = res["settings"][s]
            sv["spread"] = spread(sv)
            sv.update(au0(sv))
            print(f"    {s:6s} rho {sv['AT1_rho']:+.4f}  mean {sv['spread']['mean']*100:5.2f}%"
                  f"  sd {sv['spread']['sd']*100:5.2f}pp  "
                  f"[{sv['spread']['min']*100:.1f}-{sv['spread']['max']*100:.1f}]  "
                  f"AU-0a {'ok' if sv['AU0a_recovery_in_band'] else 'FAIL'}"
                  f"{' (ceiling)' if sv['AU0a_at_ceiling'] else ''}"
                  f"  rivals {'beaten' if sv['AT2_beats_all_rivals'] else 'NOT beaten'}",
                  flush=True)
        out["cells"][cell] = res
        out["verdict_block"] = verdict(out["cells"])
        OUT.write_text(json.dumps(out, indent=2) + "\n")

    out["verdict_block"] = verdict(out["cells"])
    pr = predictions(out["cells"])
    if pr is not None:
        out["predictions"] = pr
    out["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    v = out["verdict_block"]
    print(f"\nAU-1 {v['AU1_clears']}/{v['AU1_denominator']}   "
          f"AU-2 {v['AU2_beats_rivals']}/{v['AU2_denominator']}   "
          f"AU-3 spread opened: {v['AU3_spread_opened']}")
    print(f"verdict: {v['verdict']}")
    print(f"wrote {OUT.relative_to(ROOT)} in {out['seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
