#!/usr/bin/env python3
"""Study AO: what does the embedding add GIVEN composition, and what does composition add?

`docs/NESTED_MODEL_PREREGISTRATION.md`, frozen before this file existed.

Study AN tried to answer this by erasing the composition subspace and could not: a complete
erasure must be fitted on the rows it is applied to, and fitting across both classes absorbs
the between-class difference. A nested comparison avoids the trade entirely -- fit both
feature sets together and compare against each alone.

    C    21 features              34.25%          study AJ
    E    1,280 / 480              41.43 / 41.78%  study AI-2
    C+E  1,301 / 501              here

`src/125`'s fold, negative partition reproduced bit-for-bit, 30 seeds, both arms. No erasure,
no projection, no leak: every feature comes from the protein's own sequence.
"""
import argparse
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results/v3"
LOCO = ROOT / "results/deepvic_loco.json"
COMP = ROOT / "results/composition_arm.json"
OUT = ROOT / "results/nested_model.json"
AA = "ACDEFGHIKLMNPQRSTVWY"
SEEDS, N_TRAIN, N_CAL, SPEC, FLOOR = 30, 1000, 500, 0.95, 7
BAND_ADDS, BAND_NOT = 0.06, 0.02
ARMS = {"esm2_650M": "", "esm2_35M": "_esm2_35M"}


def load(mod):
    spec = importlib.util.spec_from_file_location(mod[:4], ROOT / "src" / mod)
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod[:4]] = m
    spec.loader.exec_module(m)
    return m


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))


def comp_features(seqs):
    n = np.array([len(s) for s in seqs], float)
    loglen = np.log(np.maximum(n, 1.0))[:, None]
    c = np.array([[s.count(x) for x in AA] for s in seqs], float)
    c /= np.maximum(n[:, None], 1.0)
    return np.hstack([loglen, c])


def verdict(d):
    if not np.isfinite(d):
        return "unmeasured"
    if d >= BAND_ADDS:
        return "adds information composition lacks"
    if d <= BAND_NOT:
        return "adds nothing over composition"
    return "between bands"


def selftest():
    ok = True
    # the bands must discriminate in all three directions, and their constants must be frozen
    if (BAND_ADDS, BAND_NOT) != (0.06, 0.02):
        print(f"  XX bands are {(BAND_ADDS, BAND_NOT)}, frozen at (0.06, 0.02)")
        ok = False
    for d, want in ((0.08, "adds information composition lacks"),
                    (0.01, "adds nothing over composition"),
                    (0.04, "between bands")):
        if verdict(d) != want:
            print(f"  XX verdict({d}) is '{verdict(d)}', expected '{want}'")
            ok = False
    if verdict(float("nan")) != "unmeasured":
        print("  XX a nan difference is not reported as unmeasured")
        ok = False

    # 🔒 concatenation must put composition FIRST and keep both blocks intact, or AO-2 would
    # be comparing a different model than the one described
    m124 = load("124_deepvic_embed.py")
    recs, meta = m124.ordered()
    C = comp_features([s for _, s in recs][:50])
    X = np.arange(50 * 4, dtype=float).reshape(50, 4)
    CE = np.hstack([C, X])
    if CE.shape[1] != C.shape[1] + X.shape[1]:
        print("  XX the concatenation lost columns")
        ok = False
    if not (np.allclose(CE[:, :21], C) and np.allclose(CE[:, 21:], X)):
        print("  XX the concatenated blocks are not the two feature sets")
        ok = False
    if C.shape[1] != 21:
        print(f"  XX composition is {C.shape[1]} wide, expected 21")
        ok = False

    # the comparison targets must exist and hold the figures the bands were set from
    for f in (LOCO, COMP):
        if not f.exists():
            print(f"  XX {f.name} missing")
            ok = False
    if ok:
        L, Cj = json.loads(LOCO.read_text()), json.loads(COMP.read_text())
        if abs(Cj["arms"]["composition"]["mean_recovery"] - 0.342) > 0.002:
            print("  XX composition's mean recovery is no longer 0.342")
            ok = False
        for arm, w in (("esm2_650M", 0.414), ("esm2_35M", 0.418)):
            r = L["arms"][arm]["full"]["recovery"]
            got = float(np.mean([r[c]["mean"] for c in r]))
            if abs(got - w) > 0.002:
                print(f"  XX {arm} mean recovery {got:.4f}, bands set from {w}")
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
    m124 = load("124_deepvic_embed.py")
    recs, meta = m124.ordered()
    cls = np.array(meta["class_of_row"])
    lab = np.array(meta["label_of_row"])
    pos, neg = np.where(lab == 1)[0], np.where(lab == 0)[0]
    perm = np.random.default_rng(12345).permutation(len(neg))
    cut = 2 * len(neg) // 3
    union, test = neg[perm[:cut]], neg[perm[cut:]]
    targets = sorted(c for c, n in Counter(cls[pos]).items() if n >= FLOOR)

    C = comp_features([s for _, s in recs])
    L = json.loads(LOCO.read_text())
    Cj = json.loads(COMP.read_text())
    comp_rec = {r["category"]: r["recovery"]
                for r in Cj["arms"]["composition"]["ranking"]}

    out = {"built": time.strftime("%Y-%m-%d %H:%M:%S"),
           "study": "AO -- what the embedding adds given composition, and the reverse",
           "preregistration": "docs/NESTED_MODEL_PREREGISTRATION.md",
           "design": ("no erasure, no projection, no leak: the two feature sets are "
                      "concatenated and compared against each alone, on src/125's fold"),
           "linear_scope": ("this compares what a LINEAR probe extracts, as every result in "
                            "the paper does, and says nothing about non-linear structure"),
           "bands": {"adds_floor": BAND_ADDS, "not_ceiling": BAND_NOT},
           "seeds": a.seeds, "n_composition_features": int(C.shape[1]), "arms": {}}

    for arm, sfx in ARMS.items():
        E = np.load(RES / f"embeddings_deepvic{sfx}.npy").astype(np.float64)
        CE = np.hstack([C, E])
        print(f"  {arm}: C+E is {CE.shape[1]} features "
              f"({C.shape[1]} + {E.shape[1]})", flush=True)
        rec, fpr = {}, []
        for cat in targets:
            hi = pos[cls[pos] == cat]
            tri = pos[cls[pos] != cat]
            rr, ff = [], []
            for seed in range(a.seeds):
                r = np.random.default_rng(seed)
                p = r.permutation(len(union))
                tr, ca = union[p[:N_TRAIN]], union[p[N_TRAIN:N_TRAIN + N_CAL]]
                mdl = clf().fit(np.vstack([CE[tri], CE[tr]]),
                                np.r_[np.ones(len(tri)), np.zeros(len(tr))])
                t = float(np.quantile(mdl.predict_proba(CE[ca])[:, 1], SPEC))
                rr.append(float((mdl.predict_proba(CE[hi])[:, 1] >= t).mean()))
                ff.append(float((mdl.predict_proba(CE[test])[:, 1] >= t).mean()))
            rec[cat] = {"n": int(len(hi)), "mean": float(np.mean(rr)),
                        "sd": float(np.std(rr, ddof=1))}
            fpr.append(float(np.mean(ff)))
            print(f"    {cat:<46}{rec[cat]['mean']*100:6.2f}%  "
                  f"(E {L['arms'][arm]['full']['recovery'][cat]['mean']*100:5.1f}%, "
                  f"C {comp_rec[cat]*100:5.1f}%)", flush=True)

        emb = {c: L["arms"][arm]["full"]["recovery"][c]["mean"] for c in targets}
        m_ce = float(np.mean([rec[c]["mean"] for c in targets]))
        m_e = float(np.mean([emb[c] for c in targets]))
        m_c = float(np.mean([comp_rec[c] for c in targets]))
        ao1, ao2 = m_ce - m_c, m_ce - m_e
        per = {c: {"CE": rec[c]["mean"], "E": emb[c], "C": comp_rec[c],
                   "CE_minus_E": rec[c]["mean"] - emb[c],
                   "CE_minus_C": rec[c]["mean"] - comp_rec[c], "n": rec[c]["n"]}
               for c in targets}
        out["arms"][arm] = {
            "n_features": int(CE.shape[1]),
            "recovery": rec, "per_category": per,
            "mean_CE": m_ce, "mean_E": m_e, "mean_C": m_c,
            "AO1_CE_minus_C": ao1, "AO2_CE_minus_E": ao2,
            "AO2_positive": bool(ao2 > 0),
            "categories_where_CE_beats_E": sum(1 for c in targets
                                               if per[c]["CE_minus_E"] > 0),
            "largest_CE_over_E": max(targets, key=lambda c: per[c]["CE_minus_E"]),
            "largest_CE_over_E_pp": max(per[c]["CE_minus_E"] for c in targets) * 100,
            "test_fpr_min": float(min(fpr)), "test_fpr_max": float(max(fpr)),
            "verdict": verdict(ao1),
        }
        v = out["arms"][arm]
        print(f"    mean C+E {m_ce*100:.2f}%  (E {m_e*100:.2f}%, C {m_c*100:.2f}%)  "
              f"AO-1 {ao1*100:+.2f}pp -> {v['verdict']}   AO-2 {ao2*100:+.2f}pp  "
              f"C+E beats E on {v['categories_where_CE_beats_E']}/14, biggest "
              f"{v['largest_CE_over_E']} {v['largest_CE_over_E_pp']:+.2f}pp", flush=True)

    A, B = out["arms"]["esm2_650M"], out["arms"]["esm2_35M"]
    out["second_arm_rule"] = {"arms_agree": A["verdict"] == B["verdict"],
                              "verdict": A["verdict"] if A["verdict"] == B["verdict"]
                              else "SPLIT -- indicative only"}
    out["predictions"] = {
        "P1_AO1_clears_6pp_both_arms": {
            "kind": "derived",
            "held": bool(A["AO1_CE_minus_C"] >= BAND_ADDS
                         and B["AO1_CE_minus_C"] >= BAND_ADDS),
            "values": {k: out["arms"][k]["AO1_CE_minus_C"] for k in ARMS}},
        "P2_AO2_positive_both_arms": {
            "kind": "derived",
            "held": bool(A["AO2_positive"] and B["AO2_positive"]),
            "values": {k: out["arms"][k]["AO2_CE_minus_E"] for k in ARMS}},
        "P3_AO2_over_1pp_on_one_arm": {
            "kind": "reasoned",
            "held": bool(A["AO2_CE_minus_E"] > 0.01 or B["AO2_CE_minus_E"] > 0.01),
            "values": {k: out["arms"][k]["AO2_CE_minus_E"] for k in ARMS},
            "largest_gain_category": {k: out["arms"][k]["largest_CE_over_E"]
                                      for k in ARMS}},
    }
    out["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nsecond arm: {out['second_arm_rule']}")
    for k, v in out["predictions"].items():
        print(f"  {k:<32} {v['kind']:<9} held={v['held']}")
    print(f"wrote {OUT.relative_to(ROOT)} in {out['seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else main())
