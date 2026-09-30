#!/usr/bin/env python3
"""
114_conformal_contrasts.py - AB: do the headline contrasts survive the better operating point?

`docs/CONFORMAL_CONTRAST_PREREGISTRATION.md`. 48 files threshold with `np.quantile(s_cal, 0.95)`, which
src/48's docstring says "misses nominal from below", and the conformal estimator it calls "validated and
never used" has never been applied to a contrast -- only to the out-of-sample FPR table.

🔒 Both thresholds, same calibration fold, same seeds: R (study E's strata) and C/A (study K's exclusions),
plus the realized false-positive rate of each threshold so the estimator is visible and not assumed.

⚠️ Not a claim that conformal is correct for this design: panel and pool are non-exchangeable with the
calibration set. The question is whether the contrasts care which estimator is used.

Usage:
    python src/114_conformal_contrasts.py --arm esm2_650M
    python src/114_conformal_contrasts.py --arm esm2_35M
    python src/114_conformal_contrasts.py --selftest
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_STEM = ROOT / "results" / "conformal_contrasts"
ALPHA = 0.05
SMALL, MILD = 5.0, 15.0


def _load(stem):
    spec = importlib.util.spec_from_file_location(f"_{stem}", ROOT / "src" / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M107 = _load("107_localization_nullspace")
M99, M92, M84, M83, M91 = M107.M99, M107.M92, M107.M84, M107.M83, M107.M91
SEEDS, SPEC = M99.SEEDS, M107.SPEC


def conformal_threshold(s_cal, alpha=ALPHA):
    """k-th largest calibration score, k = floor((m+1)*alpha) -- src/48's rule, not a new one.

    🔒 A fresh exchangeable negative exceeds this with probability at most k/(m+1) <= alpha. `np.quantile`
    interpolates between order statistics and lands lower, which is why it overshoots the budget.
    """
    m = len(s_cal)
    k = int(np.floor((m + 1) * alpha))
    if k < 1:
        return None, 0, None
    return float(np.sort(s_cal)[-k]), k, k / (m + 1)


def selftest():
    # 🔒 the conformal threshold must be the HIGHER of the two at both calibration sizes in § 1, or the
    # preregistration's order-statistics table is wrong
    rng = np.random.default_rng(0)
    for m, want_k in ((118, 5), (490, 24)):
        s = rng.normal(size=m)
        tc, k, bound = conformal_threshold(s)
        tq = float(np.quantile(s, SPEC))
        assert k == want_k, f"m={m}: k={k}, expected {want_k}"
        assert bound <= ALPHA, (m, bound)
        assert tc > tq, f"m={m}: conformal {tc:.4f} must exceed quantile {tq:.4f}"
    # 🔴 and it must refuse rather than invent a threshold when m is too small for k to reach 1
    t, k, b = conformal_threshold(rng.normal(size=10), alpha=0.01)
    assert t is None and k == 0 and b is None, (t, k, b)
    # 🔒 a ratio of two rates that both move the same way must absorb most of a threshold shift: check
    # the arithmetic direction the prediction rests on, on synthetic scores
    a_s, b_s = rng.normal(1.4, 1.0, 4000), rng.normal(-0.3, 1.0, 4000)
    cal = rng.normal(0.0, 1.0, 490)
    tc, _, _ = conformal_threshold(cal)
    tq = float(np.quantile(cal, SPEC))
    ra_q, rb_q = (a_s >= tq).mean(), (b_s >= tq).mean()
    ra_c, rb_c = (a_s >= tc).mean(), (b_s >= tc).mean()
    assert ra_c < ra_q and rb_c < rb_q, "both rates must fall under the stricter threshold"
    rel = abs((ra_c / rb_c) - (ra_q / rb_q)) / (ra_q / rb_q) * 100
    assert rel < 25.0, f"the ratio should absorb most of it, moved {rel:.1f}%"
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
        if ul is None or str(len_of[vfg]) != str(ul):
            continue
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

    out = {e: {"R": [], "CA": [], "fpr": [], "thr": []} for e in ("quantile", "conformal")}
    ks = set()
    t0 = time.time()
    for seed in range(a.seeds):
        r = np.random.default_rng(seed)
        perm = r.permutation(len(union))
        tr = [union[i] for i in perm[:n_tr]]
        ca = [union[i] for i in perm[n_tr:n_tr + n_ca]]
        mdl = M92.clf().fit(np.vstack([P, POOL[tr]]), y)
        s_cal = mdl.predict_proba(POOL[ca])[:, 1]
        s_t = mdl.predict_proba(T)[:, 1]
        s_v = mdl.predict_proba(VFDB)[:, 1]
        tc, k, _ = conformal_threshold(s_cal)
        ks.add(k)
        for name, thr in (("quantile", float(np.quantile(s_cal, SPEC))), ("conformal", tc)):
            f = s_t >= thr
            be, bi = float(f[ex].mean()), float(f[inn].mean())
            out[name]["R"].append(be / bi if bi > 0 else np.nan)
            out[name]["CA"].append(float((s_v[vrows] >= thr).mean()) / be if be > 0 else np.nan)
            out[name]["fpr"].append(float((s_cal >= thr).mean()))
            out[name]["thr"].append(thr)
    print(f"  {time.time() - t0:.0f}s   calibration n = {n_ca}, conformal k = {sorted(ks)}")

    S = M99.stat
    res = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "arm": a.arm, "seeds": a.seeds,
           "alpha": ALPHA, "calibration_n": n_ca, "conformal_k": sorted(ks),
           "n_benign": {k_: len(v) for k_, v in bcell.items()}, "n_vfdb_extra": len(vrows)}
    for e in ("quantile", "conformal"):
        res[e] = {m: {"mean": round(S(out[e][m])["mean"], 5),
                      "ci": [round(x, 5) for x in S(out[e][m])["ci"]]}
                  for m in ("R", "CA", "fpr", "thr")}
    rel_R = abs(res["conformal"]["R"]["mean"] - res["quantile"]["R"]["mean"]) \
        / res["quantile"]["R"]["mean"] * 100
    rel_CA = abs(res["conformal"]["CA"]["mean"] - res["quantile"]["CA"]["mean"]) \
        / res["quantile"]["CA"]["mean"] * 100
    res["rel_change_R_pct"] = round(rel_R, 3)
    res["rel_change_CA_pct"] = round(rel_CA, 3)
    res["conformal_threshold_higher"] = bool(
        res["conformal"]["thr"]["mean"] > res["quantile"]["thr"]["mean"])
    res["conformal_fpr_lower"] = bool(res["conformal"]["fpr"]["mean"] < res["quantile"]["fpr"]["mean"])
    res["band"] = ("the contrasts do not depend on the estimator" if rel_R < SMALL
                   else "mildly dependent" if rel_R <= MILD
                   else "the headline contrasts are estimator-dependent")
    dest = Path(f"{OUT_STEM}{sfx}.json")
    dest.write_text(json.dumps(res, indent=2) + "\n")

    print(f"\n{'':<14}{'threshold':>12}{'cal FPR':>10}{'R':>12}{'C/A':>12}")
    print("-" * 60)
    for e in ("quantile", "conformal"):
        v = res[e]
        print(f"{e:<14}{v['thr']['mean']:>12.5f}{v['fpr']['mean'] * 100:>9.2f}%"
              f"{v['R']['mean']:>12.3f}{v['CA']['mean']:>12.3f}")
    print(f"\nrelative change: R {rel_R:.2f}%   C/A {rel_CA:.2f}%")
    print(f"conformal threshold higher: {res['conformal_threshold_higher']}   "
          f"conformal FPR lower: {res['conformal_fpr_lower']}")
    print(f"-> {res['band']}")
    print(f"\nwrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
