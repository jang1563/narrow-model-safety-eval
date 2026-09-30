#!/usr/bin/env python3
"""
117_spent_rule.py - AE: spend the reviewed-only rule, pool both benign draws, settle C/A

`docs/SPENT_RULE_PREREGISTRATION.md`. The coverage ledger put 54.4% of the VFDB loss in one rule --
reviewed-only -- whose protected fields (length, organism) are present on unreviewed entries too. Study AC
left C/A partial on an honest interval straddling 2.0.

🔒 Both expansions come from data already on disk: `vfdb_negatives_idmap.tsv` holds 3,601 unreviewed rows
with Keywords on 96.9% and Length on 100%, and the independent 11,301-protein pool is embedded.

🔴 The two benign draws DISAGREE about C/A -- 2.81 on pool 1, 1.59 on pool 2 -- so pooling moves the point
estimate and does not only narrow the interval. The numerator and denominator effects are reported
separately so a reader can see which expansion moved what.

⚠️ The unreviewed rows' strata come from automatic annotation, which is a confound this study INTRODUCES.
The reviewed-only subset is printed beside the expanded one at every step.

Usage:
    python src/117_spent_rule.py --arm esm2_650M
    python src/117_spent_rule.py --arm esm2_35M
    python src/117_spent_rule.py --selftest
"""

import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_STEM = ROOT / "results" / "spent_rule"
IDMAP = ROOT / "data" / "external" / "uniprot_idmap" / "vfdb_negatives_idmap.tsv"
IND_POOL = ROOT / "data" / "sequences" / "_independent_pool.json"
B = 20000
LO, HI = 2.5, 97.5
CA_SUPPORTED, CA_PARTIAL, CA_RETRACT = 2.0, 1.5, 1.2


def _load(stem):
    spec = importlib.util.spec_from_file_location(f"_{stem}", ROOT / "src" / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M115 = _load("115_headline_intervals")
M107, M99, M92, M84, M83, M91 = (M115.M107, M115.M99, M115.M92, M115.M84, M115.M83, M115.M91)
SPEC, SEEDS = M107.SPEC, M99.SEEDS


def idmap_rows():
    """Every mapping row, reviewed and not. 🔒 One entry per accession, by the frozen rule in src/90:
    reviewed preferred, then lexicographically smallest, so the expansion changes only WHICH rows are
    eligible and not how a winner is chosen among them."""
    by = {}
    with open(IDMAP) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            by.setdefault(r["From"], []).append(r)
    out = {}
    for acc, rows in by.items():
        rev = [r for r in rows if r["Reviewed"] == "reviewed"]
        pick = sorted(rev or rows, key=lambda r: r["Entry"])[0]
        out[acc] = pick
    return out


def boot_ratio(num, den, rng, b=B):
    out = []
    for _ in range(b):
        s = rng.integers(num.shape[0])
        a = num[s][rng.integers(0, num.shape[1], num.shape[1])].mean()
        d = den[s][rng.integers(0, den.shape[1], den.shape[1])].mean()
        if d > 0:
            out.append(a / d)
    return np.array(out)


def pct(v):
    return [float(np.percentile(v, LO)), float(np.percentile(v, HI))]


def verdict(lo):
    return ("supported" if lo > CA_SUPPORTED else "partial" if lo > CA_PARTIAL
            else "not supported" if lo > CA_RETRACT else "retracted")


def selftest():
    m = idmap_rows()
    # 🔒 the picker must return one entry per accession and prefer reviewed, or the expansion is not a
    # clean superset of the reviewed-only population
    assert len(m) == 3261, f"{len(m)} accessions mapped, expected 3261"
    nrev = sum(1 for r in m.values() if r["Reviewed"] == "reviewed")
    assert nrev == 966, f"{nrev} reviewed winners, expected 966"
    assert all(r.get("Length") for r in m.values()), "Length is required for the identity check"
    kw = sum(1 for r in m.values() if r.get("Keywords"))
    assert kw / len(m) > 0.95, f"only {kw/len(m):.1%} carry keywords"
    # 🔴 and the verdict bands must be ordered, or a lower bound could read as two things
    assert verdict(2.5) == "supported" and verdict(1.8) == "partial"
    assert verdict(1.3) == "not supported" and verdict(1.0) == "retracted"
    rng = np.random.default_rng(0)
    A = np.array([[rng.random() < 0.9 for _ in range(300)] for _ in range(20)])
    Bm = np.array([[rng.random() < 0.3 for _ in range(200)] for _ in range(20)])
    r = boot_ratio(A, Bm, np.random.default_rng(1), 3000)
    assert 2.5 < r.mean() < 3.5, r.mean()
    print("selftest PASS")


def strat_of(row):
    return M84.stratum_of(set((row.get("Keywords") or "").split(";")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="esm2_650M", choices=["esm2_650M", "esm2_35M"])
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    pool = json.loads(M107.POOL_JSON.read_text())["proteins"]
    test = json.loads(M107.BUILD.read_text())["pool_partition"]["test_rows"]
    accs = [pool[i]["uniprot"] for i in test]
    kw = M84.fetch_keywords(accs)
    vf, sp_ok = M83.vfdb_sequences(), M83.vfdb_species()

    # ---- benign cell, pool 1: study K's filters, unchanged --------------------------------------
    b1 = []
    for k_, i in enumerate(test):
        sp = " ".join(pool[i]["organism"].replace("(", "").split()[:2])
        if pool[i]["kingdom"] != "Bacteria" or sp not in sp_ok or pool[i]["sequence"] in vf:
            continue
        if M84.stratum_of(kw[pool[i]["uniprot"]]) == "extracellular":
            b1.append(k_)

    # ---- benign cell, pool 2: the same filters on the independent draw --------------------------
    ind = json.loads(IND_POOL.read_text())
    ind = ind["proteins"] if isinstance(ind, dict) else ind
    i_accs = [p["uniprot"] for p in ind]
    i_kw = M84.fetch_keywords(i_accs)
    b2 = []
    for k_, p in enumerate(ind):
        sp = " ".join(p["organism"].replace("(", "").split()[:2])
        if p["kingdom"] != "Bacteria" or sp not in sp_ok or p["sequence"] in vf:
            continue
        if M84.stratum_of(i_kw[p["uniprot"]]) == "extracellular":
            b2.append(k_)

    # ---- VFDB numerator, reviewed-only and expanded --------------------------------------------
    admitted = json.loads(M107.NEGSET.read_text())["admitted"]
    len_of = {r["vfg"]: r["len"] for r in admitted}
    row_of = {r["vfg"]: i for i, r in enumerate(admitted)}
    noncirc = M83.vfdb_noncircular_mask()
    acc_of = M91.vfg_accessions()
    m = idmap_rows()
    vrev, vall, dropped_len = [], [], 0
    for vfg, i in row_of.items():
        if not noncirc[i]:
            continue
        acc = acc_of.get(vfg)
        row = m.get(acc)
        if row is None:
            continue
        if str(len_of[vfg]) != str(row["Length"]):
            dropped_len += 1
            continue
        if strat_of(row) != "extracellular":
            continue
        vall.append(i)
        if row["Reviewed"] == "reviewed":
            vrev.append(i)
    print(f"arm {a.arm}")
    print(f"  benign extracellular: pool1 {len(b1)}, pool2 {len(b2)}, pooled {len(b1) + len(b2)}")
    print(f"  VFDB extracellular:   reviewed-only {len(vrev)}, expanded {len(vall)}"
          f"   (length-mismatch drops {dropped_len})")

    sfx = "" if a.arm == "esm2_650M" else f"_{a.arm}"
    P = np.load(M107.RES / f"embeddings_class_axis_positives{sfx}.npy")
    POOL = np.load(M107.RES / f"embeddings_pool_large_{a.arm}.npy")
    IND = np.load(M107.RES / f"embeddings_pool_independent_{a.arm}.npy")
    VFDB = np.load(M107.RES / f"embeddings_vfdb_neg_{a.arm}.npy")
    union = [i for i in json.loads(M107.SCREEN.read_text())["admitted_rows"]
             if pool[i]["sequence"] not in vf]
    n_tr, n_ca = M92.fold_sizes(len(union))
    y = np.r_[np.ones(P.shape[0]), np.zeros(n_tr)]
    T = POOL[np.array(test)]

    f_b1, f_b2, f_vr, f_va = [], [], [], []
    t0 = time.time()
    for seed in range(a.seeds):
        r = np.random.default_rng(seed)
        perm = r.permutation(len(union))
        tr = [union[i] for i in perm[:n_tr]]
        ca = [union[i] for i in perm[n_tr:n_tr + n_ca]]
        mdl = M92.clf().fit(np.vstack([P, POOL[tr]]), y)
        t = float(np.quantile(mdl.predict_proba(POOL[ca])[:, 1], SPEC))
        f_b1.append(mdl.predict_proba(T[np.array(b1)])[:, 1] >= t)
        f_b2.append(mdl.predict_proba(IND[np.array(b2)])[:, 1] >= t)
        sv = mdl.predict_proba(VFDB)[:, 1] >= t
        f_vr.append(sv[np.array(vrev)])
        f_va.append(sv[np.array(vall)])
    A_b1, A_b2 = np.array(f_b1), np.array(f_b2)
    A_vr, A_va = np.array(f_vr), np.array(f_va)
    A_bp = np.concatenate([A_b1, A_b2], axis=1)
    print(f"  {time.time() - t0:.0f}s")

    rng = np.random.default_rng(0)
    cells = {
        "as published (reviewed VFDB / pool1 benign)": (A_vr, A_b1),
        "numerator only (expanded VFDB / pool1)": (A_va, A_b1),
        "denominator only (reviewed VFDB / pooled)": (A_vr, A_bp),
        "both (expanded VFDB / pooled benign)": (A_va, A_bp),
    }
    S = M99.stat
    res = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "arm": a.arm, "seeds": a.seeds, "B": B,
           "n": {"benign_pool1": len(b1), "benign_pool2": len(b2),
                 "benign_pooled": len(b1) + len(b2),
                 "vfdb_reviewed": len(vrev), "vfdb_expanded": len(vall),
                 "length_mismatch_drops": dropped_len,
                 "accessions_mapped": len(m)},
           "cells": {}}
    print(f"\n{'cell':<44}{'point':>8}{'seed interval':>22}{'row + seed (honest)':>26}{'verdict':>16}")
    print("-" * 118)
    for name, (num, den) in cells.items():
        seed_v = np.array([num[i].mean() / den[i].mean() for i in range(num.shape[0])
                           if den[i].mean() > 0])
        row_v = boot_ratio(num, den, rng)
        s_ = S(seed_v)
        lo, hi = pct(row_v)
        res["cells"][name] = {"point": round(float(seed_v.mean()), 4),
                              "seed_ci": [round(x, 4) for x in s_["ci"]],
                              "row_ci": [round(lo, 4), round(hi, 4)],
                              "verdict": verdict(lo),
                              "n_num": int(num.shape[1]), "n_den": int(den.shape[1])}
        c = res["cells"][name]
        print(f"{name:<44}{c['point']:>8.3f}   [{c['seed_ci'][0]:6.3f},{c['seed_ci'][1]:6.3f}]"
              f"     [{c['row_ci'][0]:6.3f},{c['row_ci'][1]:6.3f}]{c['verdict']:>16}")
    both = res["cells"]["both (expanded VFDB / pooled benign)"]
    pub = res["cells"]["as published (reviewed VFDB / pool1 benign)"]
    res["point_fell"] = bool(both["point"] < pub["point"])
    res["interval_narrowed"] = bool((both["row_ci"][1] - both["row_ci"][0])
                                    < (pub["row_ci"][1] - pub["row_ci"][0]))
    res["narrowing_factor"] = round((pub["row_ci"][1] - pub["row_ci"][0])
                                    / (both["row_ci"][1] - both["row_ci"][0]), 2)
    res["final_verdict"] = both["verdict"]
    res["above_retraction_threshold"] = bool(both["row_ci"][0] > CA_RETRACT)
    dest = Path(f"{OUT_STEM}{sfx}.json")
    dest.write_text(json.dumps(res, indent=2) + "\n")
    print(f"\npoint estimate fell: {res['point_fell']}   interval narrowed "
          f"{res['narrowing_factor']}x   final verdict: {res['final_verdict']}")
    print(f"above the frozen retraction threshold of {CA_RETRACT}: "
          f"{res['above_retraction_threshold']}")
    print(f"\nwrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
