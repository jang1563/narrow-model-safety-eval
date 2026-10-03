#!/usr/bin/env python3
"""Study AQ: the comparison with a tuned classifier on BOTH sides.

`docs/FAIR_COMPARISON_PREREGISTRATION.md`, frozen before this file existed.

Study AP tuned C for the embedding and left composition at 1.0, so AK's increment and AO-2
became a tuned model against an untuned one. This applies src/133's selection rule to each
feature set separately -- composition, embedding, concatenated -- so every side is fitted the
same way. The embedding's tuned figures already exist and are reused rather than recomputed.

🔴 The rule is src/133's, unchanged: for each held-out category, C is chosen on a member split
of the other thirteen plus train negatives, and no number from the scored category reaches any
selection.
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
AP = ROOT / "results/estimation_repair.json"
COMP = ROOT / "results/composition_arm.json"
LOCO = ROOT / "results/deepvic_loco.json"
OUT = ROOT / "results/fair_comparison.json"
SEEDS, FLOOR = 30, 7
BAND_SURVIVES, BAND_NOT = 0.06, 0.02
BAND_AO2_NULL, BAND_AO2_ADDS = 0.02, 0.05
ARMS = {"esm2_650M": "", "esm2_35M": "_esm2_35M"}


def load(mod):
    spec = importlib.util.spec_from_file_location(mod[:4], ROOT / "src" / mod)
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod[:4]] = m
    spec.loader.exec_module(m)
    return m


def selftest():
    ok = True
    m133 = load("133_estimation_repair.py")
    if (BAND_SURVIVES, BAND_NOT) != (0.06, 0.02):
        print(f"  XX bands are {(BAND_SURVIVES, BAND_NOT)}, frozen at (0.06, 0.02)")
        ok = False
    if (BAND_AO2_NULL, BAND_AO2_ADDS) != (0.02, 0.05):
        print(f"  XX AO-2 bands are {(BAND_AO2_NULL, BAND_AO2_ADDS)}, frozen at (0.02, 0.05)")
        ok = False
    # 🔒 the selection rule and the grid must be src/133's, not a local copy that could drift
    if m133.C_GRID != (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0):
        print(f"  XX src/133's grid is {m133.C_GRID}, not the frozen one")
        ok = False
    for fn in ("select_C", "fold_once"):
        if not hasattr(m133, fn):
            print(f"  XX src/133 exposes no {fn}; this study must not reimplement it")
            ok = False
    if m133.INNER_SEEDS != 5 or m133.SPEC != 0.95:
        print(f"  XX src/133's inner seeds or spec changed: "
              f"{m133.INNER_SEEDS}, {m133.SPEC}")
        ok = False
    # 🔒 the embedding's tuned figures must be present to be reused, and must be the ones
    # entry 117 recorded -- otherwise this study would silently compare against something else
    if not AP.exists():
        print("  XX results/estimation_repair.json missing; AP's tuned figures are reused")
        ok = False
    else:
        d = json.loads(AP.read_text())
        for arm, w in (("esm2_650M", 0.5539), ("esm2_35M", 0.4691)):
            got = d["arms"][arm]["mean_swept"]
            if abs(got - w) > 0.0005:
                print(f"  XX AP's tuned mean for {arm} is {got:.4f}, entry 117 recorded {w}")
                ok = False
        if d["arms"]["esm2_650M"]["AP3_selected_at_edge"]:
            print("  XX AP's selection is at the grid edge; its figures are a lower bound")
            ok = False
    if not COMP.exists() or not LOCO.exists():
        print("  XX a baseline artifact is missing")
        ok = False
    else:
        c = json.loads(COMP.read_text())["arms"]["composition"]["mean_recovery"]
        if abs(c - 0.342) > 0.002:
            print(f"  XX composition's fixed-C mean is {c:.4f}, bands used 0.342")
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
    m133 = load("133_estimation_repair.py")
    m124 = load("124_deepvic_embed.py")
    recs, meta = m124.ordered()
    cls = np.array(meta["class_of_row"])
    lab = np.array(meta["label_of_row"])
    pos, neg = np.where(lab == 1)[0], np.where(lab == 0)[0]
    perm = np.random.default_rng(12345).permutation(len(neg))
    cut = 2 * len(neg) // 3
    union, test = neg[perm[:cut]], neg[perm[cut:]]
    targets = sorted(c for c, n in Counter(cls[pos]).items() if n >= FLOOR)

    C = load("132_nested_model.py").comp_features([s for _, s in recs])
    apd = json.loads(AP.read_text())
    cfix = {x["category"]: x["recovery"]
            for x in json.loads(COMP.read_text())["arms"]["composition"]["ranking"]}

    out = {"built": time.strftime("%Y-%m-%d %H:%M:%S"),
           "study": "AQ -- the comparison with a tuned classifier on both sides",
           "preregistration": "docs/FAIR_COMPARISON_PREREGISTRATION.md",
           "selection_rule": ("src/133's select_C, imported rather than reimplemented: chosen "
                              "on a member split of the other thirteen categories, and no "
                              "number from the scored category reaches any selection"),
           "grid": list(m133.C_GRID), "seeds": a.seeds,
           "bands": {"survives_floor": BAND_SURVIVES, "not_ceiling": BAND_NOT,
                     "ao2_null": BAND_AO2_NULL, "ao2_adds": BAND_AO2_ADDS},
           "arms": {}}

    for arm, sfx in ARMS.items():
        E = np.load(RES / f"embeddings_deepvic{sfx}.npy").astype(np.float64)
        CE = np.hstack([C, E])
        sets = {"composition": C, "concatenated": CE}
        res = {k: {} for k in sets}
        chosen = {k: {} for k in sets}
        for ci, cat in enumerate(targets):
            hi = pos[cls[pos] == cat]
            tri = pos[cls[pos] != cat]
            for tag, X in sets.items():
                creg, _ = m133.select_C(X, tri, union, ci)
                chosen[tag][cat] = creg
                rr, ff = [], []
                for seed in range(a.seeds):
                    r = np.random.default_rng(seed)
                    p = r.permutation(len(union))
                    tr = union[p[:m133.N_TRAIN]]
                    ca = union[p[m133.N_TRAIN:m133.N_TRAIN + m133.N_CAL]]
                    # 🔒 one fit, both scores. Calling src/133's fold_once twice would refit
                    # the same model to score the held-out category and then the test
                    # negatives -- the same fold, paid for twice.
                    mdl = m133.clf(creg).fit(
                        np.vstack([X[tri], X[tr]]),
                        np.r_[np.ones(len(tri)), np.zeros(len(tr))])
                    t = float(np.quantile(
                        mdl.predict_proba(X[ca])[:, 1], m133.SPEC))
                    rr.append(float((mdl.predict_proba(X[hi])[:, 1] >= t).mean()))
                    ff.append(float((mdl.predict_proba(X[test])[:, 1] >= t).mean()))
                res[tag][cat] = {"n": int(len(hi)), "mean": float(np.mean(rr)),
                                 "sd": float(np.std(rr, ddof=1)),
                                 "test_fpr": float(np.mean(ff)), "C": creg}
            print(f"    {cat:<46}comp {res['composition'][cat]['mean']*100:6.2f}% "
                  f"(C={chosen['composition'][cat]:g})  "
                  f"C+E {res['concatenated'][cat]['mean']*100:6.2f}% "
                  f"(C={chosen['concatenated'][cat]:g})", flush=True)

        m_c = float(np.mean([res["composition"][c]["mean"] for c in targets]))
        m_ce = float(np.mean([res["concatenated"][c]["mean"] for c in targets]))
        m_e = apd["arms"][arm]["mean_swept"]
        c_fixed = float(np.mean([cfix[c] for c in targets]))
        aq1, aq2 = m_e - m_c, m_ce - m_e
        out["arms"][arm] = {
            "composition": res["composition"], "concatenated": res["concatenated"],
            "mean_tuned_composition": m_c, "mean_tuned_embedding": m_e,
            "mean_tuned_concatenated": m_ce,
            "mean_fixed_composition": c_fixed,
            "composition_gain_from_tuning": m_c - c_fixed,
            "embedding_gain_from_tuning": m_e - apd["arms"][arm]["mean_fixed_C"],
            "AQ1_tuned_increment": aq1,
            "AQ1_untuned_increment": apd["arms"][arm]["mean_fixed_C"] - c_fixed,
            "AQ2_tuned_ao2": aq2,
            "distinct_C_composition": sorted(set(chosen["composition"].values())),
            "distinct_C_concatenated": sorted(set(chosen["concatenated"].values())),
            "test_fpr": {"min": float(min(res[t][c]["test_fpr"]
                                          for t in sets for c in targets)),
                         "max": float(max(res[t][c]["test_fpr"]
                                          for t in sets for c in targets))},
            "verdict_AQ1": ("the increment survives a fair comparison"
                            if aq1 >= BAND_SURVIVES else
                            "the increment does not survive" if aq1 <= BAND_NOT
                            else "between bands"),
            "verdict_AQ2": ("AO-2's null survives tuning" if abs(aq2) <= BAND_AO2_NULL
                            else "composition adds something once both are tuned"
                            if aq2 >= BAND_AO2_ADDS else "between bands"),
        }
        v = out["arms"][arm]
        print(f"  {arm}: tuned comp {m_c*100:.2f}% (from {c_fixed*100:.2f}%, "
              f"{v['composition_gain_from_tuning']*100:+.2f}pp)  tuned emb {m_e*100:.2f}% "
              f"({v['embedding_gain_from_tuning']*100:+.2f}pp)  tuned C+E {m_ce*100:.2f}%",
              flush=True)
        print(f"    AQ-1 {aq1*100:+.2f}pp (untuned was "
              f"{v['AQ1_untuned_increment']*100:+.2f}pp) -> {v['verdict_AQ1']}", flush=True)
        print(f"    AQ-2 {aq2*100:+.2f}pp -> {v['verdict_AQ2']}   "
              f"C for composition {v['distinct_C_composition']}  "
              f"FPR {v['test_fpr']['min']*100:.2f}-{v['test_fpr']['max']*100:.2f}%", flush=True)

    A, B = out["arms"]["esm2_650M"], out["arms"]["esm2_35M"]
    out["second_arm_rule"] = {
        "aq1_agree": A["verdict_AQ1"] == B["verdict_AQ1"],
        "verdict_aq1": (A["verdict_AQ1"] if A["verdict_AQ1"] == B["verdict_AQ1"]
                        else "SPLIT -- indicative only"),
        "aq2_agree": A["verdict_AQ2"] == B["verdict_AQ2"],
        "verdict_aq2": (A["verdict_AQ2"] if A["verdict_AQ2"] == B["verdict_AQ2"]
                        else "SPLIT -- indicative only")}
    out["predictions"] = {
        "P1_composition_gains_under_5pp": {
            "kind": "derived",
            "held": bool(all(out["arms"][k]["composition_gain_from_tuning"] < 0.05
                             for k in ARMS)),
            "values": {k: out["arms"][k]["composition_gain_from_tuning"] for k in ARMS}},
        "P2_tuned_increment_exceeds_untuned_on_650M": {
            "kind": "derived",
            "held": bool(A["AQ1_tuned_increment"] > A["AQ1_untuned_increment"]),
            "values": {"tuned": A["AQ1_tuned_increment"],
                       "untuned": A["AQ1_untuned_increment"]}},
        "P3_ao2_stays_within_2pp": {
            "kind": "derived",
            "held": bool(all(abs(out["arms"][k]["AQ2_tuned_ao2"]) <= 0.02 for k in ARMS)),
            "values": {k: out["arms"][k]["AQ2_tuned_ao2"] for k in ARMS}},
        "P4_composition_C_larger_than_embedding_C": {
            "kind": "reasoned",
            "held": bool(all(min(out["arms"][k]["distinct_C_composition"]) > 0.01
                             for k in ARMS)),
            "values": {k: out["arms"][k]["distinct_C_composition"] for k in ARMS}},
    }
    out["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nsecond arm: {json.dumps(out['second_arm_rule'])}")
    for k, v in out["predictions"].items():
        print(f"  {k:<44} {v['kind']:<9} held={v['held']}")
    print(f"wrote {OUT.relative_to(ROOT)} in {out['seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else main())
