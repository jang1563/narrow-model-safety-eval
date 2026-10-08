#!/usr/bin/env python3
"""Study AV: does a training-free quantity predict which form of the statistic to use?

`docs/TRAINING_FREE_RULE_PREREGISTRATION.md`, frozen before any new embedding existed.

    R = rho(nearest_positive, nearest_negative) over a cell's held-out classes

🔒 R needs NO training: both per-class numbers come off the frozen embedding and the class
partition. The rule is

    R < +0.15  -> the gain from subtracting is POSITIVE at both settings
    R > +0.35  -> it is NEGATIVE at both settings
    otherwise  -> ABSTAIN, recorded as an abstention and not as a hit or a miss

⚠️ Both thresholds were fitted on the sixteen cell-settings of studies AT and AU, which is
why this scores cells embedded afterwards.

🔒 The sealed record is amendment 1 of the preregistration plus
`results/training_free_rule_sealed.json`, written by `src/139 --embed-only`, which runs no
fold. This file REFUSES to score a cell whose R was not sealed first.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SEALED = ROOT / "results/training_free_rule_sealed.json"
FOLDS = ROOT / "results/training_free_rule.json"
OUT = ROOT / "results/training_free_rule_scored.json"
PREREG = ROOT / "docs/TRAINING_FREE_RULE_PREREGISTRATION.md"
PRIOR = (ROOT / "results/generality_grid.json", ROOT / "results/image_axis_repair.json")
CELLS = ("text/distilroberta", "text/minilm_l3", "text/bge_base", "image/vit_b16_sup")
LO, HI = 0.15, 0.35
AV1_OK, AV1_NOT = 6, 4
AV2_OK, AV2_NOT = 1, 3
AV3_OK, AV3_NOT = -0.60, -0.20


def load(mod):
    spec = importlib.util.spec_from_file_location(mod[:4], ROOT / "src" / mod)
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod[:4]] = m
    spec.loader.exec_module(m)
    return m


def rule(R):
    """The frozen rule. Returns 'positive gain', 'negative gain' or 'ABSTAIN'."""
    return ("positive gain" if R < LO else "negative gain" if R > HI else "ABSTAIN")


def sealed_ok():
    """Is the sealed record present, and does the preregistration carry its amendment?

    🔴 § 3 of the preregistration says the study is VOID without it. A rule tested against a
    prediction written after the fold is not tested at all, and the only thing standing
    between those two cases is this check.
    """
    if not SEALED.exists():
        return {"ok": False, "why": f"{SEALED.name} does not exist; run --embed-only first"}
    d = json.loads(SEALED.read_text())
    if not d.get("no_fold_ran"):
        return {"ok": False, "why": "the sealed record does not assert that no fold ran"}
    missing = [c for c in CELLS if c not in d.get("cells", {})]
    if missing:
        return {"ok": False, "why": f"R was never sealed for {missing}"}
    txt = PREREG.read_text() if PREREG.exists() else ""
    if "## Amendment 1" not in txt:
        return {"ok": False,
                "why": "the preregistration carries no amendment 1, so the predictions were "
                       "never sealed in the document; section 3 makes the study void"}
    # 🔒 every sealed R must appear in the amendment, so the document and the artifact agree
    absent = [c for c in CELLS
              if f"{d['cells'][c]['R']:+.4f}" not in txt and f"{d['cells'][c]['R']:.4f}" not in txt]
    if absent:
        return {"ok": False, "why": f"amendment 1 does not state the sealed R for {absent}"}
    return {"ok": True, "why": "sealed record present and stated in amendment 1",
            "sealed": {c: d["cells"][c]["R"] for c in CELLS}}


def prior_rows(m30):
    """The sixteen cell-settings of studies AT and AU, for AV-3."""
    rows = []
    for f in PRIOR:
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        for cell, cv in d.get("cells", {}).items():
            ks = sorted(cv["margin"])
            R = m30._spearman([cv["nearest_positive"][k] for k in ks],
                              [cv["nearest_negative"][k] for k in ks])
            for s in ("fixed", "swept"):
                sv = cv["settings"][s]
                rows.append({"cell": cell, "setting": s, "R": R, "new": False,
                             "gain": sv["AT1_rho"] - sv["rivals"]["nearest_positive"]})
    return rows


def selftest():
    ok = True
    m30 = load("30_margin_across_arms.py")
    if (LO, HI) != (0.15, 0.35):
        print(f"  XX the rule's band is {(LO, HI)}, frozen at (0.15, 0.35)")
        ok = False
    for R, want in ((-0.9, "positive gain"), (0.0, "positive gain"),
                    (0.1499, "positive gain"), (0.15, "ABSTAIN"), (0.25, "ABSTAIN"),
                    (0.35, "ABSTAIN"), (0.3501, "negative gain"), (0.9, "negative gain")):
        if rule(R) != want:
            print(f"  XX rule({R}) = {rule(R)!r}, expected {want!r}")
            ok = False
    # 🔒 the rule must reproduce its own fitting set: 6/6 below LO, 4/4 above HI
    rows = prior_rows(m30)
    if rows:
        below = [r for r in rows if r["R"] < LO]
        above = [r for r in rows if r["R"] > HI]
        mid = [r for r in rows if LO <= r["R"] <= HI]
        bad_b = [r for r in below if r["gain"] <= 0]
        bad_a = [r for r in above if r["gain"] >= 0]
        if bad_b or bad_a:
            print(f"  XX the rule misses on its own fitting set: "
                  f"{len(bad_b)} below, {len(bad_a)} above")
            ok = False
        if len(mid) == 0:
            print("  XX no prior cell-setting falls in the abstention band, so the band was "
                  "not calibrated on anything")
            ok = False
        print(f"     prior set: {len(below)} below / {len(mid)} abstain / {len(above)} above")
    # 🔴 the sealing check must FAIL when the record is absent, and that is the check that
    # separates a sealed prediction from a retrodiction
    global SEALED
    keep = SEALED
    try:
        SEALED = ROOT / "results/__nonexistent__.json"
        if sealed_ok()["ok"]:
            print("  XX sealed_ok() passed with no sealed record")
            ok = False
    finally:
        SEALED = keep
    print("selftest PASS" if ok else "selftest FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    t0 = time.time()
    m30 = load("30_margin_across_arms.py")
    seal = sealed_ok()
    if not seal["ok"]:
        raise SystemExit(f"VOID: {seal['why']}")
    sealed = json.loads(SEALED.read_text())["cells"]
    if not FOLDS.exists():
        raise SystemExit(f"{FOLDS.name} does not exist; run the folds for {list(CELLS)} "
                         f"with `src/139 --out {FOLDS.relative_to(ROOT)} --cell ...`")
    folds = json.loads(FOLDS.read_text())["cells"]
    missing = [c for c in CELLS if c not in folds]
    if missing:
        raise SystemExit(f"folds missing for {missing}")

    rows, per_cell = [], {}
    for cell in CELLS:
        R = sealed[cell]["R"]
        pred = rule(R)
        hits, got = [], {}
        for s in ("fixed", "swept"):
            sv = folds[cell]["settings"][s]
            gain = sv["AT1_rho"] - sv["rivals"]["nearest_positive"]
            actual = "positive gain" if gain > 0 else "negative gain"
            hit = None if pred == "ABSTAIN" else bool(pred == actual)
            hits.append(hit)
            got[s] = {"gain": gain, "actual": actual, "hit": hit,
                      "rho_margin": sv["AT1_rho"],
                      "rho_nearest_positive": sv["rivals"]["nearest_positive"],
                      "rho_nearest_negative": sv["rivals"]["nearest_negative"],
                      "mean_recovery": sv["mean_recovery"]}
            rows.append({"cell": cell, "setting": s, "R": R, "new": True, "gain": gain})
        per_cell[cell] = {"R": R, "sealed_prediction": pred, "settings": got,
                          "both_settings_hit": (None if pred == "ABSTAIN"
                                                else bool(all(hits))),
                          "dim": sealed[cell]["dim"]}

    scored = [h for v in per_cell.values() for h in
              (x["hit"] for x in v["settings"].values()) if h is not None]
    abstained = sorted(c for c, v in per_cell.items()
                       if v["sealed_prediction"] == "ABSTAIN")
    allrows = prior_rows(m30) + rows
    av3 = m30._spearman([r["R"] for r in allrows], [r["gain"] for r in allrows])
    av1_hits = sum(1 for h in scored if h)
    out = {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "study": "AV -- a training-free rule for whether the negative term belongs",
        "preregistration": "docs/TRAINING_FREE_RULE_PREREGISTRATION.md",
        "sealing": seal,
        "rule": {"positive_below": LO, "negative_above": HI},
        "cells": per_cell,
        "AV1_hits": av1_hits, "AV1_scored": len(scored),
        "AV1_verdict": ("the rule has content" if av1_hits >= AV1_OK
                        else "it does not" if av1_hits <= AV1_NOT else "between bands"),
        "AV1_per_cell": {c: v["both_settings_hit"] for c, v in per_cell.items()},
        "AV2_abstained": abstained, "AV2_n_abstained": len(abstained),
        "AV2_verdict": ("acceptable" if len(abstained) <= AV2_OK
                        else "the band swallows the claim" if len(abstained) >= AV2_NOT
                        else "between bands"),
        "AV3_rho_all_cells": av3, "AV3_n": len(allrows),
        "AV3_verdict": ("holds" if av3 <= AV3_OK
                        else "does not" if av3 >= AV3_NOT else "between bands"),
        # ⚠️ the two settings of a cell share an embedding and a fold, so the per-cell count
        # is the independent one and both are reported
        "independence_note": ("the two settings of a cell are not independent; AV1_per_cell "
                              "is the count over independent cells and AV1_hits is over "
                              "cell-settings"),
    }
    out["verdict"] = (
        f"{out['AV1_verdict']} (AV-1 {av1_hits}/{len(scored)}), "
        f"abstention {out['AV2_verdict']} ({len(abstained)}/4), "
        f"AV-3 {out['AV3_verdict']} ({av3:+.4f})")
    out["predictions"] = {
        "P1_AV1_at_least_6_of_8": {
            "kind": "derived", "held": bool(av1_hits >= AV1_OK),
            "values": {"hits": av1_hits, "scored": len(scored)}},
        "P2_supervised_image_abstains_or_predicts_negative": {
            "kind": "reasoned",
            "held": bool(per_cell["image/vit_b16_sup"]["sealed_prediction"]
                         in ("ABSTAIN", "negative gain")),
            "values": {"R": per_cell["image/vit_b16_sup"]["R"],
                       "prediction": per_cell["image/vit_b16_sup"]["sealed_prediction"]}},
        "P3_three_layer_encoder_has_lowest_recovery": {
            "kind": "derived",
            "held": bool(min(
                CELLS, key=lambda c: np.mean(
                    [per_cell[c]["settings"][s]["mean_recovery"]
                     for s in ("fixed", "swept")])) == "text/minilm_l3"),
            "values": {c: float(np.mean([per_cell[c]["settings"][s]["mean_recovery"]
                                         for s in ("fixed", "swept")]))
                       for c in CELLS}},
    }
    out["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    for c, v in per_cell.items():
        print(f"  {c:22s} R {v['R']:+.4f} -> {v['sealed_prediction']:14s} "
              f"gain {v['settings']['fixed']['gain']:+.4f}/"
              f"{v['settings']['swept']['gain']:+.4f}  "
              f"{'HIT' if v['both_settings_hit'] else 'abstained' if v['both_settings_hit'] is None else 'MISS'}")
    print(f"\n{out['verdict']}")
    print(f"wrote {OUT.relative_to(ROOT)} in {out['seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
