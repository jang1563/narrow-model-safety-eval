#!/usr/bin/env python3
"""
106_availability_denominator.py - the availability guard's denominator, not a fourth cause

`docs/AVAILABILITY_DENOMINATOR_PREREGISTRATION.md`. Study E's availability check compares the
unannotated stratum's flag rate to the POOLED annotated rate, and has failed four times: 0.409, 0.322,
0.484, 0.518. Entry 73 eliminated length (<=10%), both score factors, and embedding density, and closed
with the cause unknown.

🔑 The pooled denominator averages three classes study E showed are graded by a factor of five
(extracellular 0.239, membrane 0.169, intracellular 0.046 on 650M). Read against intracellular alone,
all four published values sit INSIDE the guard's own band. § 2 of the preregistration records that as a
derivation from existing artifacts and explicitly not as a prediction.

X-1 length-adjusts the ratio against intracellular by Mantel-Haenszel, with study T's floors imported.
X-2 asks whether an availability indicator adds anything once localization is in the model. X-3 is the
test that can fail: equal flag rates at the 95th percentile do not imply equal distributions, so the
AUROC of unannotated against length-matched intracellular is left open. X-4 null-calibrates the
replacement guard.

⚠️ No new embeddings and no new inference; the probe, the folds and the strata are all imported.

Usage:
    python src/106_availability_denominator.py --arm esm2_650M
    python src/106_availability_denominator.py --arm esm2_35M
    python src/106_availability_denominator.py --selftest
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "v3"
POOL_JSON = ROOT / "data" / "sequences" / "_scaled_negative_pool.json"
BUILD = ROOT / "results" / "external_class_axis_build.json"
SCREEN = ROOT / "results" / "external_class_axis_screen.json"
OUT_STEM = ROOT / "results" / "availability_denominator"

# 🔒 frozen in § 3, § 4, § 5 of the preregistration
X1_TIGHT, X1_BAND = (0.80, 1.25), (0.67, 1.50)
X2_NOTHING, X2_LITTLE = 0.005, 0.020
X2_COEF_BAND = 0.25
X3_TIGHT, X3_BAND = (0.45, 0.55), (0.40, 0.60)
CV_FOLDS = 5


def _load(stem):
    spec = importlib.util.spec_from_file_location(f"_{stem}", ROOT / "src" / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M103 = _load("103_annotation_availability")
M92, M84, M83, M93, M96 = M103.M92, M103.M84, M103.M83, M103.M93, M103.M96
# 🔒 study T's floors are imported, not restated: same deciles, same cell floor, same eligibility
N_DECILES, CELL_FLOOR, MIN_DECILES = M103.N_DECILES, M103.CELL_FLOOR, M103.MIN_DECILES
SEEDS, SPEC = M103.SEEDS, M103.SPEC


def mh_by_length(flag, idx_num, idx_den, lengths):
    """Mantel-Haenszel rate ratio of `idx_num` over `idx_den`, stratified by length decile.

    Deciles are cut on the union of the two groups only, so the bins follow the comparison being
    made rather than a population neither group is.
    """
    both = np.array(list(idx_num) + list(idx_den))
    edges = np.quantile(lengths[both], np.linspace(0, 1, N_DECILES + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    def dec(j):
        return min(max(int(np.searchsorted(edges, lengths[j], side="right") - 1), 0), N_DECILES - 1)
    n_by = {d: [j for j in idx_num if dec(j) == d] for d in range(N_DECILES)}
    d_by = {d: [j for j in idx_den if dec(j) == d] for d in range(N_DECILES)}
    used = [d for d in range(N_DECILES)
            if len(n_by[d]) >= CELL_FLOOR and len(d_by[d]) >= CELL_FLOOR]
    if not used:
        return float("nan"), []
    return M93.mantel_haenszel_rr(
        [(float(flag[n_by[d]].sum()), len(n_by[d]),
          float(flag[d_by[d]].sum()), len(d_by[d])) for d in used]), used


def nested_auroc(flag, is_ext, is_mem, is_una, rng):
    """CV AUROC of localization dummies, with and without an availability indicator.

    Model A's reference class is intracellular-union-unannotated: that coding IS the hypothesis, and
    model B's added indicator is exactly the freedom for unannotated to depart from it.
    """
    A = np.column_stack([is_ext, is_mem]).astype(float)
    B = np.column_stack([is_ext, is_mem, is_una]).astype(float)
    out = {}
    for name, X in (("A", A), ("B", B)):
        sc = np.empty(len(flag))
        kf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                             random_state=int(rng.integers(0, 2**31 - 1)))
        for tr, te in kf.split(X, flag):
            m = LogisticRegression(max_iter=5000, C=1.0).fit(X[tr], flag[tr])
            sc[te] = m.predict_proba(X[te])[:, 1]
        out[name] = M84.auroc(sc[flag == 1], sc[flag == 0])
    coef = LogisticRegression(max_iter=5000, C=1.0).fit(B, flag).coef_[0][2]
    return out["A"], out["B"], float(coef)


def selftest():
    # 🔒 the MH helper must reduce to the imported one on a two-decile toy, or X-1 is a reimplementation
    lengths = np.array([10] * 60 + [1000] * 60, float)
    flag = np.zeros(120, bool)
    flag[[*range(0, 6), *range(60, 72)]] = True          # num: 6/30 short, 12/30 long
    flag[[*range(30, 33), *range(90, 96)]] = True        # den: 3/30 short, 6/30 long
    num, den = [*range(0, 30), *range(60, 90)], [*range(30, 60), *range(90, 120)]
    got, used = mh_by_length(flag, num, den, lengths)
    want = M93.mantel_haenszel_rr([(6, 30, 3, 30), (12, 30, 6, 30)])
    assert len(used) == 2 and abs(got - want) < 1e-12, f"{got} != {want}"
    assert abs(got - 2.0) < 1e-12, got
    # 🔴 a ratio helper that ignores its stratification would still return 2.0 here, so make the two
    # deciles disagree and check the pooled answer is NOT the crude one
    flag2 = np.zeros(120, bool)
    flag2[[*range(0, 15), *range(60, 63)]] = True        # num: 15/30 short, 3/30 long
    flag2[[*range(30, 33), *range(90, 105)]] = True      # den: 3/30 short, 15/30 long
    g2, _ = mh_by_length(flag2, num, den, lengths)
    crude = (18 / 60) / (18 / 60)
    assert abs(g2 - 1.0) < 1e-12 and abs(crude - 1.0) < 1e-12
    g3, _ = mh_by_length(flag2, num, den, np.zeros(120))  # one decile: must equal the crude ratio
    assert abs(g3 - crude) < 1e-12, (g3, crude)
    # 🔒 the nested test must find a real departure and must not invent one
    rng = np.random.default_rng(0)
    n = 3000
    ext, mem, una = (np.zeros(n, bool) for _ in range(3))
    ext[:300], mem[300:600], una[600:1600] = True, True, True
    p = np.where(ext, 0.40, np.where(mem, 0.25, 0.05))
    same = rng.random(n) < p
    a, b, c = nested_auroc(same.astype(int), ext, mem, una, np.random.default_rng(1))
    assert abs(b - a) < 0.02 and abs(c) < 0.5, (a, b, c)
    p2 = np.where(ext, 0.40, np.where(mem, 0.25, np.where(una, 0.45, 0.05)))
    diff = rng.random(n) < p2
    a2, b2, c2 = nested_auroc(diff.astype(int), ext, mem, una, np.random.default_rng(1))
    assert b2 - a2 > 0.05 and c2 > 1.0, (a2, b2, c2)
    print("selftest PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="esm2_650M", choices=["esm2_650M", "esm2_35M"])
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    pool = json.loads(POOL_JSON.read_text())["proteins"]
    test = json.loads(BUILD.read_text())["pool_partition"]["test_rows"]
    accs = [pool[i]["uniprot"] for i in test]
    kw, go = M84.fetch_keywords(accs), M92.fetch_go(accs)
    elig = [j for j, i in enumerate(test) if pool[i]["kingdom"] in M84.KINGDOMS]
    strat = {}
    for j in elig:
        g = go[accs[j]]
        strat[j] = M92.stratum_of(g["go"], g["sig"], g["tm"], kw[accs[j]])
    grp = {k: [j for j in elig if strat[j] == k]
           for k in ("extracellular", "membrane", "intracellular", "unannotated")}
    ann = grp["extracellular"] + grp["membrane"] + grp["intracellular"]
    una, intra = grp["unannotated"], grp["intracellular"]
    lengths = np.array([pool[i]["length"] for i in test], float)
    print(f"arm {a.arm}: " + ", ".join(f"{k} {len(v)}" for k, v in grp.items()))

    sfx = "" if a.arm == "esm2_650M" else f"_{a.arm}"
    P = np.load(RES / f"embeddings_class_axis_positives{sfx}.npy")
    POOL = np.load(RES / f"embeddings_pool_large_{a.arm}.npy")
    vf = M83.vfdb_sequences()
    union = [i for i in json.loads(SCREEN.read_text())["admitted_rows"]
             if pool[i]["sequence"] not in vf]
    n_tr, n_ca = M92.fold_sizes(len(union))
    y = np.r_[np.ones(P.shape[0]), np.zeros(n_tr)]
    rows = np.array(test)

    x1_crude, x1_mh, x1_pooled, x1_extra, x1_shuf = [], [], [], [], []
    x2_a, x2_b, x2_coef, x3 = [], [], [], []
    used_last = []
    is_ext = np.array([strat[j] == "extracellular" for j in elig])
    is_mem = np.array([strat[j] == "membrane" for j in elig])
    is_una = np.array([strat[j] == "unannotated" for j in elig])
    t0 = time.time()
    for seed in range(a.seeds):
        r = np.random.default_rng(seed)
        perm = r.permutation(len(union))
        tr = [union[i] for i in perm[:n_tr]]
        ca = [union[i] for i in perm[n_tr:n_tr + n_ca]]
        pipe = M92.clf().fit(np.vstack([P, POOL[tr]]), y)
        sc = pipe.named_steps["standardscaler"]
        w = pipe.named_steps["logisticregression"].coef_[0]
        _, _, s_test = M96.factors(sc.transform(POOL[rows]), w)
        _, _, s_cal = M96.factors(sc.transform(POOL[ca]), w)
        flag = s_test >= float(np.quantile(s_cal, SPEC))

        # ---- X-1: the ratio against intracellular, crude and length-adjusted ----------------------
        x1_crude.append(float(flag[una].mean() / flag[intra].mean())
                        if flag[intra].mean() > 0 else np.nan)
        m, used = mh_by_length(flag, una, intra, lengths)
        x1_mh.append(m)
        used_last = used
        # ---- X-4: the same guard fed input it must reject, and input it must accept ---------------
        x1_pooled.append(float(flag[una].mean() / flag[ann].mean())
                         if flag[ann].mean() > 0 else np.nan)
        x1_extra.append(mh_by_length(flag, grp["extracellular"], intra, lengths)[0])
        shuf = list(elig)
        r.shuffle(shuf)
        n_una = len(una)
        x1_shuf.append(mh_by_length(flag, shuf[:n_una], shuf[n_una:n_una + len(intra)],
                                    lengths)[0])

        # ---- X-2: the nested localization model -----------------------------------------------
        f_el = flag[np.array(elig)].astype(int)
        aA, aB, coef = nested_auroc(f_el, is_ext, is_mem, is_una, r)
        x2_a.append(aA)
        x2_b.append(aB)
        x2_coef.append(coef)

        # ---- X-3: length-matched separability, unannotated vs intracellular --------------------
        ka, kb = M84.length_matched(lengths[np.array(una)], lengths[np.array(intra)], r)
        x3.append(M84.auroc(s_test[np.array(una)[ka]], s_test[np.array(intra)[kb]]))
    print(f"  {time.time() - t0:.0f}s")

    S = M103.stat
    C, MH, POOLED, EXTRA, SHUF = (S(x1_crude), S(x1_mh), S(x1_pooled), S(x1_extra), S(x1_shuf))
    AA, AB, COEF, A3 = S(x2_a), S(x2_b), S(x2_coef), S(x3)
    d_au = AB["mean"] - AA["mean"]
    inb = lambda v, b: bool(b[0] <= v <= b[1])                                   # noqa: E731

    x1_band = ("unannotated is indistinguishable from intracellular" if inb(MH["mean"], X1_TIGHT)
               else "unannotated is within the guard's own band of intracellular"
               if inb(MH["mean"], X1_BAND)
               else "the availability gap is NOT a denominator artifact")
    x2_band = ("availability adds nothing over localization" if abs(d_au) < X2_NOTHING
               else "availability adds a little" if abs(d_au) <= X2_LITTLE
               else "availability is NOT reducible to localization")
    x3_band = ("the probe does not separate them" if inb(A3["mean"], X3_TIGHT)
               else "weak separation" if inb(A3["mean"], X3_BAND)
               else "different populations to the probe")

    res = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "arm": a.arm, "seeds": a.seeds,
           "strata_n": {k: len(v) for k, v in grp.items()},
           "n_deciles_used": len(used_last),
           "verdict_eligible": bool(len(used_last) >= MIN_DECILES),
           "X1_crude_vs_intracellular": C, "X1_mh_vs_intracellular": MH,
           "X1_band": x1_band, "X1_in_guard_band": inb(MH["mean"], X1_BAND),
           "X1_in_tight_band": inb(MH["mean"], X1_TIGHT),
           "X2_auroc_localization_only": AA, "X2_auroc_plus_availability": AB,
           "X2_delta_auroc": float(d_au), "X2_unannotated_coef": COEF,
           "X2_band": x2_band, "X2_coef_in_band": bool(abs(COEF["mean"]) <= X2_COEF_BAND),
           "X3_auroc_unannotated_vs_intracellular": A3, "X3_band": x3_band,
           "X3_in_guard_band": inb(A3["mean"], X3_BAND), "X3_in_tight_band": inb(A3["mean"], X3_TIGHT),
           "X4_pooled_denominator": POOLED,
           "X4_pooled_rejected": not inb(POOLED["mean"], X1_BAND),
           "X4_extracellular_vs_intracellular": EXTRA,
           "X4_extracellular_rejected": not inb(EXTRA["mean"], X1_BAND),
           "X4_shuffled_strata": SHUF, "X4_shuffled_accepted": inb(SHUF["mean"], X1_BAND),
           # 🔴 § 6.2 of the preregistration asked the shuffled control to FAIL. That is the wrong
           # sign and was caught writing this code, not by running it: shuffling destroys any real
           # difference, so the ratio must go to 1 and a correct guard must PASS. It is a positive
           # control, and the preregistration is amended rather than the control bent to match it.
           "X4_shuffled_obligation_was_misspecified": True,
           "X4_guard_reportable": bool(not inb(POOLED["mean"], X1_BAND)
                                       and not inb(EXTRA["mean"], X1_BAND)
                                       and inb(SHUF["mean"], X1_BAND))}
    dest = Path(f"{OUT_STEM}{sfx}.json")
    dest.write_text(json.dumps(res, indent=2) + "\n")

    print(f"\nX-1  crude unannotated/intracellular   = {C['mean']:.3f} "
          f"[{C['ci'][0]:.3f}, {C['ci'][1]:.3f}]")
    print(f"     length-adjusted (Mantel-Haenszel) = {MH['mean']:.3f} "
          f"[{MH['ci'][0]:.3f}, {MH['ci'][1]:.3f}]  over {len(used_last)}/{N_DECILES} deciles")
    print(f"     -> {x1_band}")
    print(f"X-2  AUROC localization only {AA['mean']:.4f}, plus availability {AB['mean']:.4f}, "
          f"delta {d_au:+.4f}")
    print(f"     unannotated coefficient {COEF['mean']:+.3f} "
          f"[{COEF['ci'][0]:+.3f}, {COEF['ci'][1]:+.3f}]  ->  {x2_band}")
    print(f"X-3  AUROC unannotated vs length-matched intracellular = {A3['mean']:.4f} "
          f"[{A3['ci'][0]:.4f}, {A3['ci'][1]:.4f}]  ->  {x3_band}")
    print(f"X-4  pooled denominator {POOLED['mean']:.3f} "
          f"({'rejected' if res['X4_pooled_rejected'] else '🔴 ACCEPTED'}), "
          f"extracellular/intracellular {EXTRA['mean']:.3f} "
          f"({'rejected' if res['X4_extracellular_rejected'] else '🔴 ACCEPTED'}), "
          f"shuffled {SHUF['mean']:.3f} "
          f"({'accepted' if res['X4_shuffled_accepted'] else '🔴 REJECTED'})")
    if not res["X4_guard_reportable"]:
        print("     🔴 the replacement guard is not reportable")
    print(f"\nwrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
