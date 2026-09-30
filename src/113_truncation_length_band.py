#!/usr/bin/env python3
"""
113_truncation_length_band.py - AA: is the length-U's right arm a truncation artifact?

`docs/TRUNCATION_BAND_PREREGISTRATION.md`. ESM-2 truncates at 1,022 residues. In the benign pool 293 of
8,259 proteins exceed that and are embedded from their first 1,022 residues -- 27.5% of the 700+ band,
which is the band the length-U rises in. 🔴 None of the three length preregistrations mentions truncation.

🔒 The test splits the 700+ band at the truncation boundary. The arithmetic bound is frozen in § 2:
truncation can only carry the rise if proteins over 1,022 residues are flagged at about 40%.

⚠️ Recorded separately: src/74 EXCLUDES VFDB proteins outside [50, 1022] while src/02b TRUNCATES pool
proteins above it, so the two sides of a pool-versus-VFDB comparison treat long proteins differently.

Usage:
    python src/113_truncation_length_band.py --arm esm2_650M
    python src/113_truncation_length_band.py --arm esm2_35M
    python src/113_truncation_length_band.py --selftest
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_STEM = ROOT / "results" / "truncation_length_band"
MAX_LEN = 1022                      # 🔒 src/02b's limit, not restated as a new constant
BANDS = ((0, 250), (250, 700), (700, MAX_LEN), (MAX_LEN, 10 ** 9))
INTACT_STRONG, INTACT_PARTLY = 8.0, 4.0
TRUNC_ARTIFACT = 30.0


def _load(stem):
    spec = importlib.util.spec_from_file_location(f"_{stem}", ROOT / "src" / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M107 = _load("107_localization_nullspace")
M99, M92, M84, M83 = M107.M99, M107.M92, M107.M84, M107.M83
SPEC, SEEDS = M107.SPEC, M99.SEEDS


def band_of(n):
    for i, (lo, hi) in enumerate(BANDS):
        if lo < n <= hi or (i == 0 and n <= hi):
            return i
    return len(BANDS) - 1


def selftest():
    # 🔒 the boundary must land where src/02b's limit is, and the two top bands must be disjoint
    assert band_of(1) == 0 and band_of(250) == 0 and band_of(251) == 1
    assert band_of(700) == 1 and band_of(701) == 2
    assert band_of(MAX_LEN) == 2, "1022 itself is INTACT: src/02b keeps s[:1022]"
    assert band_of(MAX_LEN + 1) == 3 and band_of(9999) == 3
    # 🔴 the frozen arithmetic bound, recomputed here so the document and the code cannot disagree
    frac_trunc = 0.275
    mid, band = 1.69, 12.18
    r = (band - (1 - frac_trunc) * mid) / frac_trunc
    assert 39.0 < r < 41.0, f"the bound in the preregistration is 39.8%, got {r:.1f}"
    print("selftest PASS")


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
    go = M92.fetch_go(accs)
    vf = M83.vfdb_sequences()
    elig, strat, lens, strat_L = [], {}, {}, {}
    for k_, i in enumerate(test):
        if pool[i]["kingdom"] not in M84.KINGDOMS or pool[i]["sequence"] in vf:
            continue
        elig.append(k_)
        strat[k_] = M84.stratum_of(kw[pool[i]["uniprot"]])
        g = go[accs[k_]]
        strat_L[k_] = M92.stratum_of(g["go"], g["sig"], g["tm"], kw[accs[k_]])
        lens[k_] = pool[i]["length"]
    # 🔴 Study M's published length table uses study L's GO-augmented strata (n_intracellular = 2739),
    # not the keyword rule. Its 700+ intracellular rate is 12.18% where the keyword rule gives about 2%,
    # so the band is decomposed by whether the two rules AGREE.
    long_L = [k for k in elig if lens[k] > 700 and strat_L[k] == "intracellular"]
    agree = [k for k in long_L if strat[k] == "intracellular"]
    added = [k for k in long_L if strat[k] != "intracellular"]
    cells = {"Lintra700_all": long_L, "Lintra700_agree": agree, "Lintra700_added": added}
    cells = {(k, -1): v for k, v in cells.items()}
    for b in range(len(BANDS)):
        cells[("all", b)] = [k for k in elig if band_of(lens[k]) == b]
        for s in ("extracellular", "intracellular"):
            cells[(s, b)] = [k for k in cells[("all", b)] if strat[k] == s]
    print(f"arm {a.arm}: eligible {len(elig)}")
    for b, (lo, hi) in enumerate(BANDS):
        lab = f"{lo}-{hi}" if hi < 10 ** 9 else f">{lo} (TRUNCATED)"
        print(f"  band {lab:<22} all {len(cells[('all', b)]):>5}  "
              f"extra {len(cells[('extracellular', b)]):>4}  intra {len(cells[('intracellular', b)]):>5}")

    sfx = "" if a.arm == "esm2_650M" else f"_{a.arm}"
    P = np.load(M107.RES / f"embeddings_class_axis_positives{sfx}.npy")
    POOL = np.load(M107.RES / f"embeddings_pool_large_{a.arm}.npy")
    union = [i for i in json.loads(M107.SCREEN.read_text())["admitted_rows"]
             if pool[i]["sequence"] not in vf]
    n_tr, n_ca = M92.fold_sizes(len(union))
    y = np.r_[np.ones(P.shape[0]), np.zeros(n_tr)]
    T = POOL[np.array(test)]

    acc = {k: [] for k in cells}
    t0 = time.time()
    for seed in range(a.seeds):
        r = np.random.default_rng(seed)
        perm = r.permutation(len(union))
        tr = [union[i] for i in perm[:n_tr]]
        ca = [union[i] for i in perm[n_tr:n_tr + n_ca]]
        mdl = M92.clf().fit(np.vstack([P, POOL[tr]]), y)
        t = float(np.quantile(mdl.predict_proba(POOL[ca])[:, 1], SPEC))
        f = mdl.predict_proba(T)[:, 1] >= t
        for k, rows in cells.items():
            acc[k].append(float(f[rows].mean()) if rows else np.nan)
    print(f"  {time.time() - t0:.0f}s")

    S = M99.stat
    rates = {f"{s}|{b}": S(v) for (s, b), v in acc.items() if not np.all(np.isnan(v))}
    intact = rates["all|2"]["mean"] * 100
    trunc = rates["all|3"]["mean"] * 100
    mid = rates["all|1"]["mean"] * 100
    band = ("the rise is not truncation; it is in the intact proteins" if intact >= INTACT_STRONG
            else "partly truncation" if intact >= INTACT_PARTLY
            else "the right arm is a truncation artifact" if trunc > TRUNC_ARTIFACT
            else "neither: the right arm is not present in this cut")
    res = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "arm": a.arm, "seeds": a.seeds,
           "max_len": MAX_LEN, "bands": [list(b) for b in BANDS],
           "cell_n": {f"{s}|{b}": len(v) for (s, b), v in cells.items()},
           "n_truncated": len(cells[("all", 3)]),
           "truncated_share_of_700plus": round(
               len(cells[("all", 3)]) / max(len(cells[("all", 2)]) + len(cells[("all", 3)]), 1), 4),
           "rates_pct": {k: round(v["mean"] * 100, 3) for k, v in rates.items()},
           "rates_ci_pct": {k: [round(x * 100, 3) for x in v["ci"]] for k, v in rates.items()},
           # ⭐ the decomposition of study M's 700+ intracellular band by stratum-rule agreement
           "L700_intra": {k: round(rates[f"{k}|-1"]["mean"] * 100, 3)
                          for k in ("Lintra700_all", "Lintra700_agree", "Lintra700_added")},
           "L700_intra_n": {k: len(cells[(k, -1)])
                            for k in ("Lintra700_all", "Lintra700_agree", "Lintra700_added")},
           "added_all_keyword_unannotated": bool(
               all(strat[k] == "unannotated" for k in added)),
           "mid_band_pct": round(mid, 3), "intact_700_1022_pct": round(intact, 3),
           "truncated_pct": round(trunc, 3),
           "frozen_bound_pct": 39.8, "truncated_below_bound": bool(trunc < 39.8),
           "intact_above_mid": bool(intact > mid), "band": band}
    dest = Path(f"{OUT_STEM}{sfx}.json")
    dest.write_text(json.dumps(res, indent=2) + "\n")

    print(f"\n{'band':<24}{'n':>6}{'flag rate':>22}")
    print("-" * 54)
    for b, (lo, hi) in enumerate(BANDS):
        lab = f"{lo}-{hi}" if hi < 10 ** 9 else f">{lo} TRUNCATED"
        k = f"all|{b}"
        v = rates.get(k)
        if v:
            print(f"{lab:<24}{len(cells[('all', b)]):>6}{v['mean'] * 100:>10.2f}%"
                  f"  [{v['ci'][0] * 100:.2f}, {v['ci'][1] * 100:.2f}]")
    print("\nstudy M's 700+ intracellular band (study L's rule), decomposed:")
    for k in ("Lintra700_all", "Lintra700_agree", "Lintra700_added"):
        v = rates[f"{k}|-1"]
        print(f"  {k:<22}n={len(cells[(k, -1)]):>4}  {v['mean'] * 100:6.2f}%  "
              f"[{v['ci'][0] * 100:.2f}, {v['ci'][1] * 100:.2f}]")
    print(f"  every added protein is keyword-UNANNOTATED: {res['added_all_keyword_unannotated']}")
    print(f"\nintact 700-1022 = {intact:.2f}%  vs mid band {mid:.2f}%  vs truncated {trunc:.2f}%")
    print(f"frozen bound: truncation would need {res['frozen_bound_pct']}%  -> "
          f"{'below it' if res['truncated_below_bound'] else '🔴 AT OR ABOVE it'}")
    print(f"-> {band}")
    print(f"\nwrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
