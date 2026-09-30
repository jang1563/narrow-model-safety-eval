#!/usr/bin/env python3
"""
115_headline_intervals.py - AC: honest intervals for the contrasts the paper argues from

`docs/HEADLINE_INTERVALS_PREREGISTRATION.md`. `stat()` reports mean +/- 1.96*SE over probe SEEDS on a
FIXED row set, which locates a seed mean rather than bounding the quantity (entry 81). Entry 84 enumerated
eleven published intervals built that way and fixed none. This fixes the four that carry the argument:
localization R, registration C/A, the out-of-sample false-positive rate and the VFDB flag rate.

🔒 A hierarchical bootstrap -- resample a seed, then rows WITH replacement inside each stratum -- reported
beside the seed-only interval so the gap is visible. The point estimates do not move; only the intervals
do, and the question is whether a verdict moves with them.

⚠️ C/A is the one at risk: its benign denominator is about 171 rows, and the same contrast already fell to
1.59 and 1.63 on the independent pool.

Usage:
    python src/115_headline_intervals.py --arm esm2_650M
    python src/115_headline_intervals.py --arm esm2_35M
    python src/115_headline_intervals.py --selftest
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_STEM = ROOT / "results" / "headline_intervals"
B = 20000
LO, HI = 2.5, 97.5
R_BAND = 3.0
CA_SUPPORTED, CA_PARTIAL = 2.0, 1.5


def _load(stem):
    spec = importlib.util.spec_from_file_location(f"_{stem}", ROOT / "src" / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M114 = _load("114_conformal_contrasts")
M107, M99, M92, M84, M83, M91 = M114.M107, M114.M99, M114.M92, M114.M84, M114.M83, M114.M91
SPEC, SEEDS = M107.SPEC, M99.SEEDS


def pct(v):
    return [float(np.percentile(v, LO)), float(np.percentile(v, HI))]


def boot_ratio(num, den, rng, b=B):
    """Resample a seed, then rows within each stratum, and take the ratio of the two rates.

    🔒 `num` and `den` are (seeds x rows) boolean matrices for the two strata. The seed is drawn once per
    replicate so the probe's variation and the rows' variation compose rather than being averaged apart.
    """
    out = []
    for _ in range(b):
        s = rng.integers(num.shape[0])
        a = num[s][rng.integers(0, num.shape[1], num.shape[1])].mean()
        d = den[s][rng.integers(0, den.shape[1], den.shape[1])].mean()
        if d > 0:
            out.append(a / d)
    return np.array(out)


def boot_rate(mat, rng, b=B):
    out = []
    for _ in range(b):
        s = rng.integers(mat.shape[0])
        out.append(mat[s][rng.integers(0, mat.shape[1], mat.shape[1])].mean())
    return np.array(out)


def selftest():
    rng = np.random.default_rng(0)
    # 🔒 with FEW rows the row-inclusive interval must dominate; with MANY it must converge on the
    # seed one. Rows are fixed across seeds and only a threshold moves, as in the real design.
    def design(n_a, n_b, seeds=30):
        sa, sb = rng.normal(1.2, 1.0, n_a), rng.normal(-0.6, 1.0, n_b)
        A = np.array([(sa >= 1.0 + rng.normal(0, 0.02)) for _ in range(seeds)])
        Bm = np.array([(sb >= 1.0 + rng.normal(0, 0.02)) for _ in range(seeds)])
        return A, Bm
    widths = {}
    for n in (120, 4000):
        A, Bm = design(n, n * 3)
        seed_r = np.array([A[i].mean() / Bm[i].mean() for i in range(A.shape[0])
                           if Bm[i].mean() > 0])
        row_r = boot_ratio(A, Bm, np.random.default_rng(1), 4000)
        widths[n] = (np.percentile(seed_r, HI) - np.percentile(seed_r, LO),
                     np.percentile(row_r, HI) - np.percentile(row_r, LO))
    assert widths[120][1] > 2.0 * widths[120][0], widths[120]
    assert widths[4000][1] < 0.5 * widths[120][1], (widths[120][1], widths[4000][1])
    # 🔴 and the point estimate must NOT move: the bootstrap is about the interval only
    A, Bm = design(400, 1200)
    pt = np.mean([A[i].mean() / Bm[i].mean() for i in range(A.shape[0]) if Bm[i].mean() > 0])
    bt = boot_ratio(A, Bm, np.random.default_rng(2), 4000).mean()
    assert abs(pt - bt) / pt < 0.15, (pt, bt)
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
    vf, sp_ok = M83.vfdb_sequences(), M83.vfdb_species()
    bcell = {"extracellular": [], "intracellular": []}
    for k_, i in enumerate(test):
        sp = " ".join(pool[i]["organism"].replace("(", "").split()[:2])
        if pool[i]["kingdom"] != "Bacteria" or sp not in sp_ok or pool[i]["sequence"] in vf:
            continue
        st = M84.stratum_of(kw[pool[i]["uniprot"]])
        if st in bcell:
            bcell[st].append(k_)

    admitted = json.loads(M107.NEGSET.read_text())["admitted"]
    len_of = {r["vfg"]: r["len"] for r in admitted}
    row_of = {r["vfg"]: i for i, r in enumerate(admitted)}
    noncirc = M83.vfdb_noncircular_mask()
    up_len, acc_of = M91.uniprot_lengths(), M91.vfg_accessions()
    vrows = []
    for vfg, info in json.loads(M107.MAPPED.read_text())["chosen"].items():
        if not noncirc[row_of[vfg]] or info["stratum"] != "extracellular":
            continue
        ul = up_len.get((acc_of[vfg], info["uniprot"]))
        if ul is not None and str(len_of[vfg]) == str(ul):
            vrows.append(row_of[vfg])
    print(f"arm {a.arm}: benign extra {len(bcell['extracellular'])}, "
          f"intra {len(bcell['intracellular'])}, VFDB extra {len(vrows)}")

    sfx = "" if a.arm == "esm2_650M" else f"_{a.arm}"
    P = np.load(M107.RES / f"embeddings_class_axis_positives{sfx}.npy")
    POOL = np.load(M107.RES / f"embeddings_pool_large_{a.arm}.npy")
    VFDB = np.load(M107.RES / f"embeddings_vfdb_neg_{a.arm}.npy")
    union = [i for i in json.loads(M107.SCREEN.read_text())["admitted_rows"]
             if pool[i]["sequence"] not in vf]
    n_tr, n_ca = M92.fold_sizes(len(union))
    y = np.r_[np.ones(P.shape[0]), np.zeros(n_tr)]
    T = POOL[np.array(test)]
    ex, inn = np.array(bcell["extracellular"]), np.array(bcell["intracellular"])

    fl_ex, fl_in, fl_v, fl_pool = [], [], [], []
    t0 = time.time()
    for seed in range(a.seeds):
        r = np.random.default_rng(seed)
        perm = r.permutation(len(union))
        tr = [union[i] for i in perm[:n_tr]]
        ca = [union[i] for i in perm[n_tr:n_tr + n_ca]]
        mdl = M92.clf().fit(np.vstack([P, POOL[tr]]), y)
        t = float(np.quantile(mdl.predict_proba(POOL[ca])[:, 1], SPEC))
        f = mdl.predict_proba(T)[:, 1] >= t
        fl_ex.append(f[ex])
        fl_in.append(f[inn])
        fl_v.append(mdl.predict_proba(VFDB)[:, 1] >= t)
        fl_pool.append(f)
    A_ex, A_in = np.array(fl_ex), np.array(fl_in)
    A_v, A_pool = np.array(fl_v), np.array(fl_pool)
    print(f"  {time.time() - t0:.0f}s")

    rng = np.random.default_rng(0)
    S = M99.stat
    seed_R = np.array([A_ex[i].mean() / A_in[i].mean() for i in range(A_ex.shape[0])
                       if A_in[i].mean() > 0])
    seed_CA = np.array([A_v[i][vrows].mean() / A_ex[i].mean() for i in range(A_v.shape[0])
                        if A_ex[i].mean() > 0])
    row_R = boot_ratio(A_ex, A_in, rng)
    row_CA = boot_ratio(A_v[:, vrows], A_ex, rng)
    row_pool = boot_rate(A_pool, rng)
    row_vfdb = boot_rate(A_v, rng)

    res = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "arm": a.arm, "seeds": a.seeds, "B": B,
           "n": {"benign_extracellular": len(ex), "benign_intracellular": len(inn),
                 "vfdb_extracellular": len(vrows), "pool": A_pool.shape[1],
                 "vfdb_all": A_v.shape[1]}}
    for name, seed_vals, row_vals in (("R", seed_R, row_R), ("CA", seed_CA, row_CA)):
        s_ = S(seed_vals)
        res[name] = {"point": round(float(np.mean(seed_vals)), 4),
                     "seed_ci": [round(x, 4) for x in s_["ci"]],
                     "row_ci": [round(x, 4) for x in pct(row_vals)],
                     "widening": round((pct(row_vals)[1] - pct(row_vals)[0])
                                       / (s_["ci"][1] - s_["ci"][0]), 2)}
    for name, mat, row_vals in (("pool_fpr", A_pool, row_pool), ("vfdb_rate", A_v, row_vfdb)):
        per = np.array([m.mean() for m in mat])
        s_ = S(per)
        res[name] = {"point": round(float(per.mean()) * 100, 4),
                     "seed_ci": [round(x * 100, 4) for x in s_["ci"]],
                     "row_ci": [round(x * 100, 4) for x in pct(row_vals)],
                     "widening": round((pct(row_vals)[1] - pct(row_vals)[0])
                                       / (s_["ci"][1] - s_["ci"][0]), 2)}
    res["R_clears_band"] = bool(res["R"]["row_ci"][0] > R_BAND)
    res["CA_clears_supported"] = bool(res["CA"]["row_ci"][0] > CA_SUPPORTED)
    res["CA_clears_partial"] = bool(res["CA"]["row_ci"][0] > CA_PARTIAL)
    res["CA_verdict"] = ("supported" if res["CA_clears_supported"]
                         else "partial" if res["CA_clears_partial"] else "not supported")
    dest = Path(f"{OUT_STEM}{sfx}.json")
    dest.write_text(json.dumps(res, indent=2) + "\n")

    print(f"\n{'quantity':<14}{'point':>9}{'seed interval':>24}{'row + seed (honest)':>28}{'x':>7}")
    print("-" * 84)
    for k in ("R", "CA", "pool_fpr", "vfdb_rate"):
        v = res[k]
        print(f"{k:<14}{v['point']:>9.3f}   [{v['seed_ci'][0]:8.3f},{v['seed_ci'][1]:8.3f}]"
              f"     [{v['row_ci'][0]:8.3f},{v['row_ci'][1]:8.3f}]{v['widening']:>7.2f}")
    print(f"\nR clears its 3.0 band on the honest interval: {res['R_clears_band']}")
    print(f"C/A verdict on the honest interval: {res['CA_verdict']}"
          f"  (supported needs > 2.0, partial > 1.5)")
    print(f"\nwrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
