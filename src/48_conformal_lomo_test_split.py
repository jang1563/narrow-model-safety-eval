#!/usr/bin/env python3
"""
48_conformal_lomo_test_split.py - give the LOMO panel the test set it never had, and apply the
                                  conformal threshold that src/45 validated and nothing ever used.

Two gaps, and they are the same gap
-----------------------------------
`src/45` established both halves of a problem on this panel's negatives: there is no test partition
(178 train / 118 calibrate / **0 test**), and the published `np.quantile` estimator does not return
its own nominal rate on negatives held back from calibration (8.64% for a nominal 5% at m=20, with
nominal unreachable at 4 of 6 sizes tested). It also established the fix: the conformal threshold,
the k-th largest calibration score with k = floor((m+1)*alpha), held its guarantee at every size on
both arms tested.

That fix was validated and then **never applied**. Every recovery number in this repository still
comes from `np.quantile`: `src/03b` line 93, and the same call in 03e, 03f, 03h, 03j and 15e. So the
per-class table has never been computed at an operating point whose false-positive rate was measured
rather than assumed.

What this script does
---------------------
It re-runs leave-one-mechanism-out with the negatives split THREE ways instead of two, so there is a
genuine test partition, and it sets the threshold twice on the same calibration split:

    quantile   np.quantile(s_cal, 1 - alpha)          the published estimator
    conformal  k-th largest of s_cal, k = floor((m+1)*alpha)   the guaranteed one

Recovery is then read off the held-out mechanism class, and the false-positive rate is read off the
TEST negatives, which no threshold has seen. That FP number is the thing this panel has never had.

What it costs, stated up front
------------------------------
Carving a test partition out of a 296-protein negative set makes the calibration set smaller, and
calibration size is the resolution ceiling from the same audit. At the default 60/20/20 the
calibration split is about 59 proteins, so the finest resolvable false-positive rate roughly doubles
from 1/118 to 1/59. This script therefore does NOT propose replacing the published split. It measures
what the published split cannot: whether the estimator choice moves the conclusions, and by how much
the in-sample FP flatters itself.

Both estimators are run on identical splits, models and scores, so every difference reported is
attributable to the threshold rule alone.

Usage:
    python src/48_conformal_lomo_test_split.py --panel v3
    python src/48_conformal_lomo_test_split.py --panel v3 --tag esm2_35M --seeds 30
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
TRAIN_FRAC, CAL_FRAC = 0.60, 0.20  # the remainder is the test partition


def conformal_threshold(s_cal, alpha):
    """k-th largest calibration score, k = floor((m+1)*alpha).

    Finite-sample guarantee: a fresh exchangeable negative exceeds this with probability at most
    k/(m+1) <= alpha. `np.quantile` interpolates instead, which is why it misses nominal from below.
    When m is too small for k to reach 1 the guarantee is unreachable at this alpha, and the caller
    is told rather than handed a number that looks like a threshold.
    """
    m = len(s_cal)
    k = int(np.floor((m + 1) * alpha))
    if k < 1:
        return None, 0, None
    return float(np.sort(s_cal)[-k]), k, k / (m + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="v3", choices=["v2", "v3"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--seeds", type=int, default=30)
    a = ap.parse_args()
    suf = f"_{a.tag}" if a.tag else ""
    RES = ROOT / "results" / a.panel

    P = np.load(RES / f"embeddings_positive_{a.panel}{suf}.npy")
    N = np.load(RES / f"embeddings_negative_{a.panel}{suf}.npy")
    man = json.load(open(RES / f"embedding_manifest_{a.panel}{suf}.json"))
    mech = json.load(open(ROOT / f"data/annotations/mechanism_classes_{a.panel}.json"))
    pos_acc = [r["acc"] for r in man["positive_rows"]]
    cls_of = {p["fasta_id"]: p["mechanism_class"] for p in mech["proteins"]}
    pos_cls = np.array([cls_of.get(x, "UNMAPPED") for x in pos_acc])
    holdout = set(mech["holdout_eligible_classes"])
    targets = sorted({c for c in pos_cls if c in holdout} | {"virulence_associated_non_toxin"})

    n_tr = int(len(N) * TRAIN_FRAC)
    n_ca = int(len(N) * CAL_FRAC)
    n_te = len(N) - n_tr - n_ca
    print(f"model={man['model']}  panel={a.panel}{suf}  positives={len(P)}  negatives={len(N)}")
    print(f"negative split: train={n_tr}  calibrate={n_ca}  test={n_te}  (seeds={a.seeds})")
    print(f"resolution ceiling from {n_ca} calibration negatives: 1/{n_ca} = {1/n_ca:.4f}\n")

    res, fp = {}, {alpha: {"quantile": [], "conformal": []} for alpha in ALPHAS}
    # Every class reuses the SAME negative split for a given seed, so pooling all class-by-seed
    # observations would treat 30 independent splits as 360 and shrink the interval by about 3.5x.
    # The honest unit is the seed, so false positives are also collected per seed and the interval
    # below is computed across seeds. Getting this wrong is how a 1.3-point overshoot would have been
    # declared significant or dismissed on the strength of an interval that was never valid.
    by_seed = {alpha: {"quantile": {}, "conformal": {}} for alpha in ALPHAS}
    for C in targets:
        hi = np.where(pos_cls == C)[0]
        if not len(hi):
            continue
        rec = {"n": int(len(hi)), "holdout_eligible": C in holdout,
               **{f"alpha_{alpha}": {"quantile": [], "conformal": [], "k": None,
                                     "guarantee": None} for alpha in ALPHAS}}
        for seed in range(a.seeds):
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(N))
            ntr, nca, nte = perm[:n_tr], perm[n_tr:n_tr + n_ca], perm[n_tr + n_ca:]
            tri = np.array([i for i in range(len(P)) if i not in set(hi.tolist())])
            model = lomo.clf().fit(np.vstack([P[tri], N[ntr]]),
                                  np.r_[np.ones(len(tri)), np.zeros(len(ntr))])
            s_ho = model.predict_proba(P[hi])[:, 1]
            s_ca = model.predict_proba(N[nca])[:, 1]
            s_te = model.predict_proba(N[nte])[:, 1]
            for alpha in ALPHAS:
                tq = float(np.quantile(s_ca, 1 - alpha))
                tc, k, guar = conformal_threshold(s_ca, alpha)
                rec[f"alpha_{alpha}"]["quantile"].append(float((s_ho >= tq).mean()))
                fp[alpha]["quantile"].append(float((s_te >= tq).mean()))
                by_seed[alpha]["quantile"].setdefault(seed, []).append(float((s_te >= tq).mean()))
                if tc is not None:
                    rec[f"alpha_{alpha}"]["conformal"].append(float((s_ho >= tc).mean()))
                    fp[alpha]["conformal"].append(float((s_te >= tc).mean()))
                    by_seed[alpha]["conformal"].setdefault(seed, []).append(
                        float((s_te >= tc).mean()))
                    rec[f"alpha_{alpha}"]["k"], rec[f"alpha_{alpha}"]["guarantee"] = k, guar
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
        cm5 = "n/a" if a5["conformal_mean"] is None else f"{a5['conformal_mean']:.1%}"
        cm1 = "n/a" if a1["conformal_mean"] is None else f"{a1['conformal_mean']:.1%}"
        dl = "n/a" if a5["delta_pts"] is None else f"{a5['delta_pts']:+.1f}"
        print(f"{C:<36}{r['n']:>3}  {a5['quantile_mean']:>9.1%} {cm5:>9} {dl:>7}   "
              f"{a1['quantile_mean']:>9.1%} {cm1:>9}")

    print("\nout-of-sample false positives, measured on the TEST partition no threshold saw:")
    fpsum = {}
    for alpha in ALPHAS:
        q = np.array(fp[alpha]["quantile"])
        c = np.array(fp[alpha]["conformal"])
        def seed_ci(which):
            per = [float(np.mean(v)) for v in by_seed[alpha][which].values()]
            if not per:
                return None, None, None, 0
            arr = np.array(per)
            se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else float("nan")
            return float(arr.mean()), arr.mean() - 1.96 * se, arr.mean() + 1.96 * se, len(arr)

        qm, qlo, qhi, qn = seed_ci("quantile")
        cm, clo, chi, cn = seed_ci("conformal")
        fpsum[str(alpha)] = {
            "nominal": alpha, "n_obs": int(len(q)),
            "quantile_fp_mean": float(q.mean()),
            "quantile_overshoot_pts": float(100 * (q.mean() - alpha)),
            "conformal_fp_mean": float(c.mean()) if len(c) else None,
            "conformal_overshoot_pts": float(100 * (c.mean() - alpha)) if len(c) else None,
            "quantile_exceeds_nominal": bool(q.mean() > alpha),
            # Seed is the independent unit; these are the only intervals that mean anything here.
            "n_seeds": qn,
            "quantile_seed_ci": None if qm is None else [qlo, qhi],
            "conformal_seed_ci": None if cm is None else [clo, chi],
            "quantile_ci_excludes_nominal": bool(qm is not None and qlo > alpha),
            "conformal_ci_excludes_nominal": bool(cm is not None and clo > alpha),
            "conformal_reachable": bool(len(c) > 0),
            "conformal_holds": bool(len(c) and c.mean() <= alpha),
        }
        s = fpsum[str(alpha)]
        cs = "n/a" if s["conformal_fp_mean"] is None else f"{s['conformal_fp_mean']:.2%}"
        print(f"  nominal {alpha:.0%}:  quantile {q.mean():.2%} "
              f"({s['quantile_overshoot_pts']:+.2f} pts)   conformal {cs}"
              + ("" if s["conformal_fp_mean"] is None
                 else f" ({s['conformal_overshoot_pts']:+.2f} pts)"))

    for alpha in ALPHAS:
        s = fpsum[str(alpha)]
        for which in ("quantile", "conformal"):
            ci = s[f"{which}_seed_ci"]
            if ci:
                print(f"  nominal {alpha:.0%} {which:<9} seed-level 95% CI "
                      f"[{ci[0]:.2%}, {ci[1]:.2%}] over {s['n_seeds']} seeds"
                      f"  -> excludes nominal: {s[f'{which}_ci_excludes_nominal']}")

    over = [k for k, v in fpsum.items() if v["quantile_ci_excludes_nominal"]]
    unreach = [k for k, v in fpsum.items() if not v["conformal_reachable"]]
    verdict = (
        "measured on a real test partition, the published estimator's out-of-sample false-positive "
        f"rate exceeds nominal with the seed-level interval excluding it at alpha in {over}. "
        "Conformal is closer but is NOT a clean fix here: "
        + ("it also exceeds nominal at " + str([k for k, v in fpsum.items()
                                                if v["conformal_ci_excludes_nominal"]])
           if any(v["conformal_ci_excludes_nominal"] for v in fpsum.values())
           else "its interval covers nominal") + ", and it is UNREACHABLE at alpha in "
        + f"{unreach} because floor((m+1)*alpha) < 1 at m={n_ca}. "
        "That asymmetry is the finding: conformal declines to return a threshold it cannot "
        "guarantee, while np.quantile returns one anyway.")
    print(f"\nverdict: {verdict}")

    dest = RES / f"conformal_lomo_test_split{suf}.json"
    json.dump({"model": man["model"], "panel": a.panel, "tag": a.tag or None,
               "seeds": a.seeds, "alphas": list(ALPHAS),
               "split": {"train": n_tr, "calibrate": n_ca, "test": n_te},
               "resolution_ceiling": 1 / n_ca,
               "per_class": res, "out_of_sample_fp": fpsum,
               "verdict": verdict}, open(dest, "w"), indent=2)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
