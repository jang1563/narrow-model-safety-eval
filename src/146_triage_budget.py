#!/usr/bin/env python3
"""Study AX: does margin triage an evaluation budget?

`docs/TRIAGE_BUDGET_PREREGISTRATION.md`, frozen before this file existed.

🔑 Section 4 of the paper calls margin "a triage signal for deciding which mechanism families
need their own validation, and not as a deployment score." Study AW tested the OTHER use --
building a mixture -- and found no advantage. This tests the one the paper asserts.

⚠️ Not redundant with rho. A rank correlation of +0.6 says the ordering beats chance; it does
not say whether picking the two worst classes out of fourteen works. The decision a reader
faces is a BUDGET, and the decision-relevant number is recall at that budget.

🔒 No fitting. Every recovery and margin already exists in the artifacts of studies AT, AU
and AV: 24 cell-settings over three modalities and eight frozen encoders.
"""
import argparse
import importlib.util
import json
import sys
import time
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ("results/generality_grid.json", "results/image_axis_repair.json",
           "results/training_free_rule.json")
OUT = ROOT / "results/triage_budget.json"
BUDGETS = (1, 2, 3)
AX1_OK, AX1_NOT = 2.0, 1.2
AX2_OK, AX2_NOT = 18, 12


def load(mod):
    spec = importlib.util.spec_from_file_location(mod[:4], ROOT / "src" / mod)
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod[:4]] = m
    spec.loader.exec_module(m)
    return m


def hypergeom_p(hits, m, k):
    """P(recall@m >= hits) when m classes are drawn at random from k.

    🔒 Exact, not a normal approximation: m is 1 to 3 and k is 10 to 50, where the
    approximation is poor and the exact tail is a few terms.
    """
    tot = 0.0
    for h in range(hits, m + 1):
        tot += comb(m, h) * comb(k - m, m - h) / comb(k, m)
    return float(tot)


def score_cell(margin, recovery, budgets):
    """recall@m for margin's pick against the true worst-m, and the recovery a reader gets."""
    ks = sorted(margin)
    k = len(ks)
    by_margin = sorted(ks, key=lambda c: margin[c])
    by_recovery = sorted(ks, key=lambda c: recovery[c])
    out = {}
    for m in list(budgets) + [max(1, -(-k // 5))]:
        if m > k:
            continue
        picked, truth = set(by_margin[:m]), set(by_recovery[:m])
        hits = len(picked & truth)
        chance = m / k
        # 🔑 the number a reader spending the budget actually experiences: how bad are the
        # classes you chose to validate, against m chosen at random
        mean_picked = float(np.mean([recovery[c] for c in by_margin[:m]]))
        mean_all = float(np.mean([recovery[c] for c in ks]))
        out[f"m{m}"] = {
            "m": m, "k": k, "hits": hits, "recall": hits / m,
            "chance": chance, "ratio": (hits / m) / chance if chance > 0 else None,
            "beats_chance": bool(hits / m > chance),
            "p_hypergeom": hypergeom_p(hits, m, k),
            "mean_recovery_picked": mean_picked,
            "mean_recovery_all": mean_all,
            "recovery_lift": mean_picked - mean_all,
            "picked": by_margin[:m], "truth": by_recovery[:m],
        }
    return out


def collect():
    rows = []
    for f in SOURCES:
        p = ROOT / f
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for cell, cv in d.get("cells", {}).items():
            if "margin" not in cv:
                continue
            for s in ("fixed", "swept"):
                rec = {c: x["mean"] for c, x in cv["settings"][s]["recovery"].items()}
                mar = {c: cv["margin"][c] for c in rec}
                rows.append({"cell": cell, "setting": s, "source": f,
                             "modality": cell.split("/")[0].replace("_sc", ""),
                             "k": len(rec), "rho": cv["settings"][s]["AT1_rho"],
                             "budgets": score_cell(mar, rec, BUDGETS)})
    return rows


def selftest():
    ok = True
    # 🔒 a perfectly ordered cell must give recall 1.0 at every budget
    mar = {f"c{i}": float(i) for i in range(10)}
    rec = {f"c{i}": float(i) for i in range(10)}
    r = score_cell(mar, rec, (1, 2, 3))
    if not all(r[f"m{m}"]["recall"] == 1.0 for m in (1, 2, 3)):
        print(f"  XX a perfectly ordered cell did not give recall 1.0: {r}")
        ok = False
    # 🔒 and a REVERSED ordering must give recall 0 at small budgets
    rev = {f"c{i}": float(9 - i) for i in range(10)}
    r = score_cell(mar, rev, (1, 2, 3))
    if r["m1"]["recall"] != 0.0 or r["m2"]["recall"] != 0.0:
        print(f"  XX a reversed cell did not give recall 0: {r['m1']}, {r['m2']}")
        ok = False
    # 🔒 the hypergeometric tail, against values computable by hand
    if abs(hypergeom_p(2, 2, 4) - 1 / 6) > 1e-12:
        print(f"  XX P(2 of 2 from 4) = {hypergeom_p(2, 2, 4)}, expected 1/6")
        ok = False
    if abs(hypergeom_p(0, 2, 4) - 1.0) > 1e-12:
        print(f"  XX P(>=0) is not 1: {hypergeom_p(0, 2, 4)}")
        ok = False
    if abs(hypergeom_p(1, 1, 10) - 0.1) > 1e-12:
        print(f"  XX P(1 of 1 from 10) = {hypergeom_p(1, 1, 10)}, expected 0.1")
        ok = False
    # 🔒 the 20% budget must be ceil(k/5)
    r = score_cell({f"c{i}": float(i) for i in range(13)},
                   {f"c{i}": float(i) for i in range(13)}, ())
    if "m3" not in r:
        print(f"  XX the 20% budget for k=13 is not 3: {list(r)}")
        ok = False
    # 🔒 recovery_lift must be NEGATIVE when triage works -- the picked classes are the bad
    # ones, so their mean recovery sits below the cell mean. A positive lift means the pick
    # is anti-correlated, and getting this sign backwards would invert the whole reading.
    r = score_cell(mar, rec, (2,))
    if r["m2"]["recovery_lift"] >= 0:
        print(f"  XX recovery_lift is not negative on a perfect pick: {r['m2']}")
        ok = False
    rows = collect()
    # 🔴 This asserted `== 24`, the population that existed when study AX was computed. It
    # went stale the moment study AT's ProtT5 cell landed and made it 26 -- a hard-coded
    # count against a population that other studies can grow, which is the sixth such number
    # in this project. 🔒 The artifact records which cells were in, and the check now asserts
    # that every cell carrying a margin contributed BOTH its settings.
    if rows:
        import json as _j
        n_cells = 0
        for f in SOURCES:
            q = ROOT / f
            if q.exists():
                n_cells += sum(1 for cv in _j.loads(q.read_text()).get("cells", {}).values()
                               if "margin" in cv)
        if len(rows) != 2 * n_cells:
            print(f"  XX collected {len(rows)} cell-settings from {n_cells} cells; "
                  "every cell must contribute both settings")
            ok = False
        if len(rows) < 24:
            print(f"  XX only {len(rows)} cell-settings; the study was computed on 24")
            ok = False
    print("selftest PASS" if ok else "selftest FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    t0 = time.time()
    rows = collect()
    r2 = [x["budgets"]["m2"] for x in rows if "m2" in x["budgets"]]
    ratios = [x["ratio"] for x in r2 if x["ratio"] is not None]
    beats = sum(1 for x in r2 if x["beats_chance"])
    by_mod = {}
    for x in rows:
        if "m2" in x["budgets"]:
            by_mod.setdefault(x["modality"], []).append(x["budgets"]["m2"]["recall"])
    mod_recall = {k: float(np.mean(v)) for k, v in by_mod.items()}
    r1 = [x["budgets"]["m1"] for x in rows if "m1" in x["budgets"]]
    ratio1 = float(np.mean([x["ratio"] for x in r1]))
    ax1 = float(np.mean(ratios))
    out = {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "study": "AX -- does margin triage an evaluation budget?",
        "preregistration": "docs/TRIAGE_BUDGET_PREREGISTRATION.md",
        "no_fitting": ("every recovery and margin already existed in the artifacts of "
                       "studies AT, AU and AV; nothing was trained for this"),
        "bands": {"AX1_ok": AX1_OK, "AX1_not": AX1_NOT,
                  "AX2_ok": AX2_OK, "AX2_not": AX2_NOT},
        "n_cell_settings": len(rows),
        "cells_included": sorted({f"{r['cell']}/{r['setting']}" for r in rows}),
        "population_note": ("study AX was first computed on 24 cell-settings from 12 cells; "
                            "study AT's ProtT5 cell later made it 26. Both figures are "
                            "reported and the verdict is recomputed on the full population, "
                            "because the added cell existed after the bands were frozen and "
                            "excluding it would be a choice about which data to use"),
        "AX1_mean_ratio_at_m2": ax1,
        "AX1_verdict": ("triage works" if ax1 >= AX1_OK
                        else "it does not" if ax1 <= AX1_NOT else "between bands"),
        "AX2_beats_chance": beats, "AX2_denominator": len(r2),
        "AX2_verdict": ("triage works" if beats >= AX2_OK
                        else "it does not" if beats <= AX2_NOT else "between bands"),
        "mean_recall_at_m2_by_modality": mod_recall,
        "mean_ratio_at_m1": ratio1,
        "mean_recall_at_m2": float(np.mean([x["recall"] for x in r2])),
        "mean_recovery_lift_at_m2": float(np.mean([x["recovery_lift"] for x in r2])),
        "ratio_note": ("chance varies with k (2/14, 2/13, 2/10, 2/50), so AX-1 is the mean "
                       "of per-cell ratios and not a ratio of pooled counts"),
        "rows": rows,
    }
    out["predictions"] = {
        "P1_AX1_clears_2x_chance": {
            "kind": "derived", "held": bool(ax1 >= AX1_OK),
            "values": {"mean_ratio": ax1, "band": AX1_OK}},
        "P2_image_cells_have_lowest_recall": {
            "kind": "derived",
            "held": bool(mod_recall and min(mod_recall, key=mod_recall.get) == "image"),
            "values": mod_recall},
        "P3_recall_at_1_worse_than_at_2": {
            "kind": "reasoned", "held": bool(ratio1 < ax1),
            "values": {"ratio_at_m1": ratio1, "ratio_at_m2": ax1}},
    }
    out["verdict"] = (f"AX-1 {out['AX1_verdict']} (mean ratio {ax1:.2f}x), "
                      f"AX-2 {out['AX2_verdict']} ({beats}/{len(r2)})")
    out["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, indent=2) + "\n")

    print(f"  {len(rows)} cell-settings, no fitting\n")
    print(f"  {'cell':24s} {'set':6s} {'k':>3s} {'rho':>7s} "
          f"{'rec@2':>6s} {'chance':>7s} {'ratio':>6s}")
    for x in sorted(rows, key=lambda r: -r["budgets"]["m2"]["ratio"]):
        b = x["budgets"]["m2"]
        print(f"  {x['cell']:24s} {x['setting']:6s} {x['k']:3d} {x['rho']:+7.3f} "
              f"{b['recall']:6.2f} {b['chance']:7.3f} {b['ratio']:6.2f}")
    print(f"\n  AX-1 mean ratio at m=2: {ax1:.2f}x  (band {AX1_OK}x / {AX1_NOT}x)")
    print(f"  AX-2 beats chance: {beats}/{len(r2)}  (band {AX2_OK} / {AX2_NOT})")
    print(f"  mean recall@2 {out['mean_recall_at_m2']:.3f}   "
          f"recovery lift {out['mean_recovery_lift_at_m2']*100:+.2f}pp")
    print(f"  by modality: {' '.join(f'{k} {v:.3f}' for k, v in mod_recall.items())}")
    print(f"\nverdict: {out['verdict']}")
    print(f"wrote {OUT.relative_to(ROOT)} in {out['seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
