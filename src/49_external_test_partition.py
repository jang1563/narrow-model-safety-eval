#!/usr/bin/env python3
"""
49_external_test_partition.py - keep the published 118-point calibration split AND get a test set,
                                by sourcing the test negatives from outside the panel.

Why this is not just src/48 again
---------------------------------
`src/48` carved a test partition out of the panel's own 296 negatives. That works and it costs two
things at once: calibration halves from 118 to 59, so its recovery numbers stop being comparable to
the published table, and the conformal threshold becomes unreachable at a nominal 1% because
floor((m+1)*alpha) is zero at m=59. Both costs come from where the test set was taken, not from the
panel.

Sourcing the test negatives from the 8,259-protein benign pool removes both. Calibration stays at the
published 118, so recovery IS comparable, and at m=118 conformal is reachable at both budgets with
genuine conservatism: k=5 giving 5/119 = 4.20% at a nominal 5%, and k=1 giving 1/119 = 0.84% at 1%.

The catch, which is the actual finding
--------------------------------------
The conformal guarantee needs the calibration and test negatives to be EXCHANGEABLE. Panel negatives
are three curated blocks; the pool is Swiss-Prot, 87% bacterial, with a name redundancy factor of
2.33. They are not exchangeable, so the guarantee is voided BY DESIGN. That is the point: a deployed
screen calibrates on negatives it curated and then meets whatever arrives, so the panel-to-pool
number is the realistic one, and the gap between it and nominal is the cost of distribution shift
rather than a defect in the estimator.

To keep shift and estimator apart, both arms are run:

    calibrate on PANEL negatives (118) -> test on POOL          the deployment-shift arm
    calibrate on POOL (118)            -> test on the rest of POOL   the exchangeable control

If conformal holds in the control and misses in the shift arm, the miss is shift. If it misses in
both, the implementation is wrong. The control exists so that conclusion is available.

Also reported: false positives after collapsing the pool to distinct names, because 8,259 raw
proteins carry only 3,550 distinct names and a rate computed over duplicates is a rate over an
effective sample that is 2.33 times smaller than it looks.

Limits, stated because they bound what may be concluded
-------------------------------------------------------
⚠️ SINGLE ARM, and which arm is the bigger caveat. Pool embeddings existed only for esm2_35M. By
criterion 7 in `docs/DETECTOR_CRITERIA.md` one arm is not a result, so this is PROVISIONAL. Worse,
`src/35_negative_scaling_curve.py` already recorded why this particular arm is weak for per-class
work: on esm2_35M beta-lactamase recovery is **already floored at 1.4%**, so the arm has headroom in
only ONE of the two failing classes. It can show a decline in phage and cannot show one in
beta-lactamase, which makes the per-class table here half a test. The false-positive decomposition
does not depend on class headroom and is the durable part. The canonical 650M arm has headroom in
both and is the run that settles the per-class question.

⚠️ Q8X739 is dropped. It is the one pool protein at 0.871 identity to a labelled panel positive, and
this repository's own rule is that anything training or calibrating on the pool drops it.

Usage:
    python src/49_external_test_partition.py --seeds 200
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
lomo = importlib.import_module("03b_leave_one_mechanism_out")

ALPHAS = (0.05, 0.01)
NEG_HOLDOUT_FRAC = 0.40  # identical to src/03b, so calibration is the published 118
CONTAMINANT = "Q8X739"
TAG = "esm2_35M"


def conformal_threshold(s_cal, alpha):
    m = len(s_cal)
    k = int(np.floor((m + 1) * alpha))
    if k < 1:
        return None, 0, None
    return float(np.sort(s_cal)[-k]), k, k / (m + 1)


def _acc(row):
    p = row.split("|")
    return p[1] if len(p) > 1 else row


def _name(row):
    p = row.split("|")
    return p[2].split("_")[0] if len(p) > 2 else _acc(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=200)
    a = ap.parse_args()
    RES = ROOT / "results/v3"

    P = np.load(RES / f"embeddings_positive_v3_{TAG}.npy")
    N = np.load(RES / f"embeddings_negative_v3_{TAG}.npy")
    POOL = np.load(RES / f"embeddings_pool_large_{TAG}.npy")
    man = json.load(open(RES / f"embedding_manifest_v3_{TAG}.json"))
    pman = json.load(open(RES / f"embedding_manifest_pool_large_{TAG}.json"))
    mech = json.load(open(ROOT / "data/annotations/mechanism_classes_v3.json"))

    pool_acc = [_acc(r) for r in pman["rows"]]
    pool_name = [_name(r) for r in pman["rows"]]
    keep = np.array([i for i, x in enumerate(pool_acc) if x != CONTAMINANT])
    assert len(keep) == len(pool_acc) - 1, f"{CONTAMINANT} not found in the pool"
    POOL, pool_name = POOL[keep], [pool_name[i] for i in keep]
    # one representative per distinct name, for the effective-n version of the rate
    first_of_name, ded = {}, []
    for i, nm in enumerate(pool_name):
        if nm not in first_of_name:
            first_of_name[nm] = i
            ded.append(i)
    ded = np.array(ded)

    pos_acc = [r["acc"] for r in man["positive_rows"]]
    cls_of = {p["fasta_id"]: p["mechanism_class"] for p in mech["proteins"]}
    pos_cls = np.array([cls_of.get(x, "UNMAPPED") for x in pos_acc])
    holdout = set(mech["holdout_eligible_classes"])
    targets = sorted({c for c in pos_cls if c in holdout} | {"virulence_associated_non_toxin"})

    n_ca = int(len(N) * NEG_HOLDOUT_FRAC)
    print(f"model={man['model']}  positives={len(P)}  panel_negatives={len(N)}")
    print(f"pool test set: {len(POOL)} proteins ({CONTAMINANT} dropped), "
          f"{len(ded)} distinct names")
    print(f"calibration stays at the published {n_ca}; seeds={a.seeds}")
    for alpha in ALPHAS:
        _, k, g = conformal_threshold(np.zeros(n_ca), alpha)
        print(f"  conformal at nominal {alpha:.0%}: k={k}  guarantee={g:.4f}"
              if k else f"  conformal at nominal {alpha:.0%}: UNREACHABLE")
    print()

    res = {}
    fp = {alpha: {arm: {est: {} for est in ("quantile", "conformal")}
                  for arm in ("shift", "control", "shift_dedup")} for alpha in ALPHAS}

    for C in targets:
        hi = np.where(pos_cls == C)[0]
        if not len(hi):
            continue
        rec = {"n": int(len(hi)), **{f"alpha_{al}": {"quantile": [], "conformal": []}
                                    for al in ALPHAS}}
        for seed in range(a.seeds):
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(N))
            nca, ntr = perm[:n_ca], perm[n_ca:]
            tri = np.array([i for i in range(len(P)) if i not in set(hi.tolist())])
            model = lomo.clf().fit(np.vstack([P[tri], N[ntr]]),
                                  np.r_[np.ones(len(tri)), np.zeros(len(ntr))])
            s_ho = model.predict_proba(P[hi])[:, 1]
            s_ca = model.predict_proba(N[nca])[:, 1]
            s_pool = model.predict_proba(POOL)[:, 1]
            # exchangeable control: calibrate on pool, test on the rest of the pool
            pperm = rng.permutation(len(POOL))
            pca, pte = pperm[:n_ca], pperm[n_ca:]

            for alpha in ALPHAS:
                tq = float(np.quantile(s_ca, 1 - alpha))
                tc, _k, _g = conformal_threshold(s_ca, alpha)
                rec[f"alpha_{alpha}"]["quantile"].append(float((s_ho >= tq).mean()))
                fp[alpha]["shift"]["quantile"].setdefault(seed, []).append(
                    float((s_pool >= tq).mean()))
                fp[alpha]["shift_dedup"]["quantile"].setdefault(seed, []).append(
                    float((s_pool[ded] >= tq).mean()))
                cq = float(np.quantile(s_pool[pca], 1 - alpha))
                fp[alpha]["control"]["quantile"].setdefault(seed, []).append(
                    float((s_pool[pte] >= cq).mean()))
                if tc is not None:
                    rec[f"alpha_{alpha}"]["conformal"].append(float((s_ho >= tc).mean()))
                    fp[alpha]["shift"]["conformal"].setdefault(seed, []).append(
                        float((s_pool >= tc).mean()))
                    fp[alpha]["shift_dedup"]["conformal"].setdefault(seed, []).append(
                        float((s_pool[ded] >= tc).mean()))
                    cc, _, _ = conformal_threshold(s_pool[pca], alpha)
                    fp[alpha]["control"]["conformal"].setdefault(seed, []).append(
                        float((s_pool[pte] >= cc).mean()))
        for alpha in ALPHAS:
            d = rec[f"alpha_{alpha}"]
            d["quantile_mean"] = float(np.mean(d["quantile"]))
            d["conformal_mean"] = float(np.mean(d["conformal"])) if d["conformal"] else None
            d["delta_pts"] = (None if d["conformal_mean"] is None
                              else 100 * (d["conformal_mean"] - d["quantile_mean"]))
            d.pop("quantile"), d.pop("conformal")
        res[C] = rec

    print(f"{'class':<36}{'n':>3}  {'quant@95':>9} {'conf@95':>9} {'delta':>7}   "
          f"{'quant@99':>9} {'conf@99':>9}")
    print("-" * 96)
    for C, r in sorted(res.items(), key=lambda kv: kv[1]["alpha_0.05"]["quantile_mean"]):
        a5, a1 = r["alpha_0.05"], r["alpha_0.01"]
        f = lambda v: "n/a" if v is None else f"{v:.1%}"  # noqa: E731
        d5 = "n/a" if a5["delta_pts"] is None else f"{a5['delta_pts']:+.1f}"
        print(f"{C:<36}{r['n']:>3}  {a5['quantile_mean']:>9.1%} {f(a5['conformal_mean']):>9} "
              f"{d5:>7}   {a1['quantile_mean']:>9.1%} {f(a1['conformal_mean']):>9}")

    def summ(alpha, arm, est):
        per = [float(np.mean(v)) for v in fp[alpha][arm][est].values()]
        if not per:
            return None
        arr = np.array(per)
        se = float(arr.std(ddof=1) / np.sqrt(len(arr)))
        return {"mean": float(arr.mean()), "ci": [arr.mean() - 1.96 * se, arr.mean() + 1.96 * se],
                "n_seeds": len(arr), "overshoot_pts": float(100 * (arr.mean() - alpha)),
                "excludes_nominal": bool(arr.mean() - 1.96 * se > alpha)}

    out_fp = {}
    print("\nfalse positives on negatives no threshold saw:")
    for alpha in ALPHAS:
        out_fp[str(alpha)] = {}
        for arm, lbl in (("control", "pool->pool  (exchangeable)"),
                         ("shift", "panel->pool (deployment)"),
                         ("shift_dedup", "panel->pool, distinct names")):
            out_fp[str(alpha)][arm] = {}
            for est in ("quantile", "conformal"):
                s = summ(alpha, arm, est)
                out_fp[str(alpha)][arm][est] = s
                if s is None:
                    print(f"  nominal {alpha:.0%}  {lbl:<28} {est:<9} unreachable")
                else:
                    print(f"  nominal {alpha:.0%}  {lbl:<28} {est:<9} {s['mean']:.2%} "
                          f"[{s['ci'][0]:.2%}, {s['ci'][1]:.2%}]  "
                          f"{'EXCEEDS' if s['excludes_nominal'] else 'covers'}")

    ctrl_ok = all(out_fp[str(al)]["control"]["conformal"] is not None
                  and not out_fp[str(al)]["control"]["conformal"]["excludes_nominal"]
                  for al in ALPHAS)
    shift_bad = [al for al in ALPHAS
                 if out_fp[str(al)]["shift"]["conformal"] is not None
                 and out_fp[str(al)]["shift"]["conformal"]["excludes_nominal"]]
    verdict = (
        f"conformal {'HOLDS' if ctrl_ok else 'FAILS'} in the exchangeable pool-to-pool control, so "
        f"the implementation is {'sound' if ctrl_ok else 'suspect'}; under panel-to-pool distribution "
        f"shift it exceeds nominal at {[f'{x:.0%}' for x in shift_bad] or 'no budget'}. "
        "Calibration stayed at the published 118, so the per-class figures above are directly "
        "comparable to the published table, unlike src/48's. SINGLE ARM, provisional.")
    print(f"\nverdict: {verdict}")

    dest = RES / f"external_test_partition_{TAG}.json"
    json.dump({"model": man["model"], "arm": TAG, "seeds": a.seeds,
               "calibration_n": n_ca, "pool_n": int(len(POOL)),
               "pool_distinct_names": int(len(ded)), "contaminant_dropped": CONTAMINANT,
               "conformal_k": {str(al): conformal_threshold(np.zeros(n_ca), al)[1]
                               for al in ALPHAS},
               "conformal_guarantee": {str(al): conformal_threshold(np.zeros(n_ca), al)[2]
                                       for al in ALPHAS},
               "per_class": res, "false_positives": out_fp,
               "conformal_holds_in_control": ctrl_ok,
               "single_arm_provisional": True,
               "verdict": verdict}, open(dest, "w"), indent=2)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
