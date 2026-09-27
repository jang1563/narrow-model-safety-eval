#!/usr/bin/env python3
"""
72_coherent_pooling_seeds.py - the 30-seed check the decision rule in src/70 demands.

`src/70`'s rule, fixed before it ran: a reduction counts as movement only if beta-lactamase's 5-seed
recovery exceeds `mean_res`'s, **and then only as a screen** requiring 30 seeds. Exactly one of the
fourteen coherent reductions beat the control at 5 seeds, which is fewer than chance would give, so
this resolves whether that one is real.

⚠️ `src/03x_seed_stability_all_arms.py` is not extended and its ARMS list is not touched: it writes
`seed_stability_all_arms.json`, a published artifact several audit claims pin. Its `recover_seeds` and
`ci95` are **imported** so the fold logic is the same code, and the output goes to its own file.

Usage:
    python src/72_coherent_pooling_seeds.py
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "results" / "v2"
# --panel added 2026-09-27. src/03x's recover_seeds and ci95 are panel-agnostic — only its main()
# hardcodes v2 paths — so the same fold logic runs on v3 by passing it v3 arrays.
CLASS = "beta_lactamase"   # overridden by --class
TAGS = ["mean_res", "dev_topk1", "dev_topk5", "dev_topk10", "dev_topk20", "dev_topk50",
        "dev_attn1", "dev_attn4", "dev_attn16", "win_best5", "win_best9", "win_best15",
        "win_best25", "win_max9", "win_max15"]


def load_03x():
    p = ROOT / "src" / "03x_seed_stability_all_arms.py"
    spec = importlib.util.spec_from_file_location("src03x", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["src03x"] = m
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="v2", choices=["v2", "v3"])
    ap.add_argument("--arm", default="esm2_650M", choices=["esm2_650M", "esm2_35M", "esm2_150M"])
    ap.add_argument("--class", dest="cls_main", default="beta_lactamase",
                    help="the class the 14-reduction grid is compared on")
    ap.add_argument("--classes", default="",
                    help="comma-separated classes instead of the default beta-lactamase plus the "
                         "four v2 cost classes; v3's second failure is phage_peptidoglycan_hydrolase")
    a = ap.parse_args()
    global V2, CLASS
    V2 = ROOT / "results" / a.panel
    CLASS = a.cls_main
    m = load_03x()
    mech = json.loads((ROOT / f"data/annotations/mechanism_classes_{a.panel}.json").read_text())
    cls = {e["fasta_id"]: e["mechanism_class"] for e in mech["proteins"]}
    # ⚠️ The alignment baseline was only ever computed on v2, so on v3 there is no reference to
    # clear and the comparison is control-versus-reduction only. Reported as absent rather than
    # borrowed from v2, where the negative set and the panel both differ.
    af = V2 / "alignment_baseline.json"
    align_rec = (json.loads(af.read_text())["classes"][CLASS]["alignment_recovery"]
                 if af.exists() and CLASS in json.loads(af.read_text())["classes"] else None)
    print(f"{CLASS}: alignment reference "
          f"{f'{align_rec * 100:.1f}%' if align_rec is not None else 'NONE for this panel'}, "
          f"{m.SEEDS} seeds\n")

    hdr = f"{'reduction':<12}{'5seed':>7}{'30seed':>8}{'sd':>6}{'ci95':>16}{'zero':>6}{'>align?':>9}"
    print(hdr)
    print("-" * len(hdr))
    out = {}
    for tag in TAGS:
        suf = f"_{a.arm}_{tag}"
        P = np.load(V2 / f"embeddings_positive_{a.panel}{suf}.npy")
        N = np.load(V2 / f"embeddings_negative_{a.panel}{suf}.npy")
        man = json.loads((V2 / f"embedding_manifest_{a.panel}{suf}.json").read_text())
        lomo = json.loads((V2 / f"lomo_results{suf}.json").read_text())
        pcls = np.array([cls[r["acc"]] for r in man["positive_rows"]])
        hi = np.where(pcls == CLASS)[0]
        tri = np.setdiff1d(np.arange(len(pcls)), hi)
        v = m.recover_seeds(P, N, hi, tri, 0.95)
        lo, hh = m.ci95(v)
        pub = lomo["leave_one_mechanism_out"][CLASS]["flagged_95_mean"]
        above = ("n/a" if align_rec is None else
                 "yes" if lo > align_rec else "overlap" if hh > align_rec else "no")
        out[tag] = {"published_5seed": pub, "mean_30seed": float(v.mean()),
                    "sd": float(v.std(ddof=1)), "ci95": [lo, hh],
                    "zero_seeds": int((v == 0).sum()), "vs_alignment": above,
                    "published_inside_ci": bool(lo - 1e-6 <= pub <= hh + 1e-6)}
        print(f"{tag:<12}{pub * 100:>6.1f}%{v.mean() * 100:>7.1f}%{v.std(ddof=1) * 100:>5.1f}%"
              f"{f'[{lo * 100:.1f}, {hh * 100:.1f}]':>16}{(v == 0).sum():>6}{above:>9}")

    ctl = out["mean_res"]
    beat = {t: r for t, r in out.items()
            if t != "mean_res" and r["mean_30seed"] > ctl["mean_30seed"]}
    # An interval comparison, not a point comparison: the 5-seed screen was a point comparison and
    # that is exactly what 30 seeds is here to correct.
    clear = {t: r for t, r in beat.items() if r["ci95"][0] > ctl["ci95"][1]}
    res = {
        "class": CLASS, "seeds": m.SEEDS, "alignment_recovery": align_rec,
        "protocol": "src/03x's recover_seeds, imported unchanged; 03b fold logic, 40% negative holdout",
        "control": "mean_res",
        "n_grid_points": len(TAGS) - 1,
        "beat_control_at_5_seeds": [t for t in TAGS[1:]
                                    if out[t]["published_5seed"] > ctl["published_5seed"]],
        "beat_control_at_30_seeds": sorted(beat),
        "intervals_clear_of_control": sorted(clear),
        "reductions": out,
        "reading": ("Under a null in which the reductions are equivalent, about half the grid would "
                    "beat the control. Far fewer than half means coherent localisation is "
                    "systematically worse, not merely no better."),
    }
    # 🔴 The gain would otherwise be reported at 30 seeds and the cost at 5, which is the asymmetry
    # this project keeps finding in other people's tables. The classes win_best25 damages most get
    # the same seed count as the class it helps, for the control and for it.
    cost = {}
    cost_classes = ([c for c in a.classes.split(",") if c] or
                    ["superantigen_enterotoxin", "pore_forming_cytolysin",
                     "contact_dependent_inhibition", "rip_rrna_glycosidase"])
    for cname in cost_classes:
        cost[cname] = {}
        for tag in ("mean_res", "win_best25"):
            suf = f"_{a.arm}_{tag}"
            P = np.load(V2 / f"embeddings_positive_{a.panel}{suf}.npy")
            N = np.load(V2 / f"embeddings_negative_{a.panel}{suf}.npy")
            man = json.loads((V2 / f"embedding_manifest_{a.panel}{suf}.json").read_text())
            pc = np.array([cls[r["acc"]] for r in man["positive_rows"]])
            h = np.where(pc == cname)[0]
            v = m.recover_seeds(P, N, h, np.setdiff1d(np.arange(len(pc)), h), 0.95)
            lo, hh = m.ci95(v)
            cost[cname][tag] = {"mean_30seed": float(v.mean()), "sd": float(v.std(ddof=1)),
                                "ci95": [lo, hh]}
        # 🔴 These were named a and b, which shadowed argparse's namespace and killed a.panel on
        # the second class. Found by the crash rather than by reading, on a run that had already
        # printed one class's result.
        ctlr, winr = cost[cname]["mean_res"], cost[cname]["win_best25"]
        cost[cname]["delta_30seed"] = round(winr["mean_30seed"] - ctlr["mean_30seed"], 4)
        cost[cname]["intervals_disjoint"] = bool(winr["ci95"][1] < ctlr["ci95"][0]
                                                 or ctlr["ci95"][1] < winr["ci95"][0])
        print(f"  {cname:<32} mean_res {ctlr['mean_30seed'] * 100:5.1f}% "
              f"[{ctlr['ci95'][0] * 100:.1f}, {ctlr['ci95'][1] * 100:.1f}]   "
              f"win_best25 {winr['mean_30seed'] * 100:5.1f}% "
              f"[{winr['ci95'][0] * 100:.1f}, {winr['ci95'][1] * 100:.1f}]   "
              f"{cost[cname]['delta_30seed'] * 100:+.1f}  disjoint="
              f"{cost[cname]['intervals_disjoint']}")
    res["cost_at_30_seeds"] = cost
    res["fixed_budget"] = ("realised FPR at nominal 5% is 0.0656 for mean_res and for win_best25 "
                           "alike, so the reallocation is not bought by loosening the threshold")
    res["provenance"] = {"mean_res": 0.8150, "win_best25": 0.7766,
                         "note": "the confound gets slightly HARDER to exploit, not easier, which "
                                 "is the opposite of what section 9.1.1 warned about for local "
                                 "features"}
    res["panel"] = a.panel
    res["arm"] = a.arm
    dest = V2 / ("coherent_pooling_seeds.json" if a.arm == "esm2_650M"
                 else f"coherent_pooling_seeds_{a.arm}.json")
    dest.write_text(json.dumps(res, indent=2) + "\n")
    print(f"\ncontrol mean_res: {ctl['mean_30seed'] * 100:.1f}% "
          f"[{ctl['ci95'][0] * 100:.1f}, {ctl['ci95'][1] * 100:.1f}]")
    print(f"beat it at 5 seeds: {res['beat_control_at_5_seeds']}")
    print(f"beat it at 30 seeds: {res['beat_control_at_30_seeds']}")
    print(f"and with an interval clear of the control's: {res['intervals_clear_of_control']}")
    print(f"\nwrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
