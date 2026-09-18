#!/usr/bin/env python3
"""
32_deployment_operating_points.py - what this screen would do in a queue, and what the panel
                                    cannot tell you about that.

What is already covered, and what is not
----------------------------------------
`03f_coverage_strictness.py` gives the per-class coverage-against-strictness curve and its s90
summary, and handles the estimator ceiling. `03r_prevalence_adjusted.py` gives precision at five
deployment prevalences. Neither answers the two questions an operator asks first:

  1. **How many items land in the review queue?** Precision says what fraction of alerts are
     real. Volume says whether anyone can read them. A 95%-specificity screen on a 10,000-item
     queue raises 500 false alerts whatever its precision is, and that number decides staffing.

  2. **Does the margin triage from §10.4 to §10.7 survive at a deployable operating point?**
     Margin's ordering of mechanism classes was measured at 95% specificity, which is a
     laboratory setting. If the ordering scrambles as the false-positive budget tightens, margin
     is a signal about a threshold nobody would deploy at.

And one the panel answers by refusing to:

  3. **Can 296 negatives validate a deployment false-positive budget?** The threshold is a
     quantile of the HELD-OUT negatives, and 40% of 296 is 118, so the finest resolution
     available is one negative in 118, about 0.85%. Every specificity above roughly 99.2% is
     extrapolation, and `03r`'s `spec_0.999` row already shows what that looks like: it returns
     an FPR of 0.0065, which is the ceiling rather than the requested 0.001.

PREREGISTERED, written before the run
-------------------------------------
    P1  margin's ordering of classes by catch rate holds at the strictest ESTIMABLE specificity:
        Spearman(margin, catch rate) > 0.5 with permutation p < 0.05 there, not only at 0.95.

    SURVIVES     P1 holds. The triage signal is usable at the tightest budget this panel can
                 calibrate.
    LAX-ONLY     P1 fails. Margin orders classes at 95% specificity and stops doing so as the
                 budget tightens, which has to be said wherever the triage is recommended.

⚠️ Volume figures are arithmetic on the measured TPR and FPR, not a simulation. They assume the
queue is drawn from the same distribution as the panel's negatives, which no real order queue is.
They are a scale check, not a forecast.

Usage:
    python src/32_deployment_operating_points.py --panel v3
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RES_ROOT = ROOT / "results"
FRAC, SEEDS = 0.40, 30
PREVALENCES = [1e-2, 1e-3, 1e-4]
QUEUE = 10_000
# a quantile estimated from fewer than this many negatives above it is one order statistic,
# which is not a calibrated threshold. Used to report what validating a budget would cost.
STABLE_TAIL = 10


def _spearman(x, y):
    def rank(v):
        v = np.asarray(v, float)
        o = v.argsort()
        r = np.empty(len(v), float)
        r[o] = np.arange(1, len(v) + 1)
        for u in np.unique(v):
            m = v == u
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="v3", choices=["v2", "v3"])
    pv = ap.parse_args().panel
    RES = RES_ROOT / pv

    man = json.load(open(RES / f"embedding_manifest_{pv}.json"))
    mech = json.load(open(ROOT / f"data/annotations/mechanism_classes_{pv}.json"))
    sfc = json.load(open(RES / "second_failure_class.json"))
    P = np.load(RES / f"embeddings_positive_{pv}.npy")
    N = np.load(RES / f"embeddings_negative_{pv}.npy")
    cls = {e["fasta_id"]: e["mechanism_class"] for e in mech["proteins"]}
    pcls = np.array([cls[r["acc"]] for r in man["positive_rows"]])
    margins = {c: v["margin"] for c, v in sfc["classes"].items()}
    classes = sorted(margins)

    held = int(len(N) * FRAC)
    ceiling = 1.0 - 1.0 / held
    specs = [s for s in (0.90, 0.95, 0.98, 0.99) if s <= ceiling]
    strictest = max(specs)
    print(f"panel {pv}: {len(P)} positives, {len(N)} negatives, {held} held out per fold")
    print(f"finest estimable resolution: 1 negative in {held} = {100 / held:.2f}% FPR, "
          f"so specificity above {ceiling:.4f} is extrapolation")
    print(f"operating points evaluated: {specs}\n")

    # ---- per class, per operating point -------------------------------------------
    catch = {s: {} for s in specs}
    fpr_ach = {s: [] for s in specs}
    pooled_tpr = {s: [] for s in specs}
    for c in classes:
        hi = np.where(pcls == c)[0]
        tri = np.setdiff1d(np.arange(len(P)), hi)
        per = {s: [] for s in specs}
        for seed in range(SEEDS):
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(N))
            nte, ntr = perm[:held], perm[held:]
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=5000, C=1.0))
            m.fit(np.vstack([P[tri], N[ntr]]),
                  np.r_[np.ones(len(tri)), np.zeros(len(ntr))])
            sn = m.predict_proba(N[nte])[:, 1]
            sp = m.predict_proba(P[hi])[:, 1]
            for s in specs:
                thr = np.quantile(sn, s)
                per[s].append(float((sp >= thr).mean()))
                fpr_ach[s].append(float((sn >= thr).mean()))
        for s in specs:
            catch[s][c] = float(np.mean(per[s]))
            pooled_tpr[s].extend(per[s])

    print(f"{'class':<32}{'margin':>9}" + "".join(f"{f'@{int(s * 100)}':>7}" for s in specs))
    print("-" * (41 + 7 * len(specs)))
    for c in sorted(classes, key=lambda x: margins[x]):
        print(f"{c:<32}{margins[c]:>9.4f}"
              + "".join(f"{catch[s][c] * 100:>6.0f}%" for s in specs))

    # ---- P1: does the margin ordering survive the tightening budget? ---------------
    mg = [margins[c] for c in classes]
    rng_p = np.random.default_rng(0)
    triage = {}
    for s in specs:
        y = [catch[s][c] for c in classes]
        rho = _spearman(mg, y)
        null = np.array([_spearman(mg, rng_p.permutation(y)) for _ in range(20000)])
        triage[s] = {"rho": rho, "perm_p": float((null >= rho).mean())}
    print("\nmargin against catch rate, as the false-positive budget tightens:")
    for s in specs:
        print(f"  specificity {s:.2f}   rho {triage[s]['rho']:+.3f}   "
              f"p {triage[s]['perm_p']:.4f}")
    p1 = triage[strictest]["rho"] > 0.5 and triage[strictest]["perm_p"] < 0.05

    # ---- alert volume, the number that decides staffing ----------------------------
    print(f"\nreview queue per {QUEUE:,} sequences screened, panel-mean TPR:")
    hdr = (f"{'spec':>6}{'TPR':>7}{'FPR':>7}" +
           "".join(f"{f'alerts@{p:g}':>14}" for p in PREVALENCES))
    print(hdr)
    print("-" * len(hdr))
    volume = {}
    for s in specs:
        tpr = float(np.mean(pooled_tpr[s]))
        fpr = float(np.mean(fpr_ach[s]))
        row = {}
        for prev in PREVALENCES:
            tp = prev * QUEUE * tpr
            fp = (1 - prev) * QUEUE * fpr
            row[str(prev)] = {"true_alerts": tp, "false_alerts": fp,
                              "total_alerts": tp + fp,
                              "precision": tp / (tp + fp) if tp + fp else None,
                              "missed_hazards": prev * QUEUE * (1 - tpr)}
        volume[s] = {"tpr": tpr, "fpr": fpr, "per_prevalence": row}
        print(f"{s:>6.2f}{tpr * 100:>6.0f}%{fpr * 100:>6.1f}%"
              + "".join(f"{row[str(p)]['total_alerts']:>9.0f} ({row[str(p)]['precision'] * 100:>2.0f}%)"
                        for p in PREVALENCES))

    # ---- what it would cost to validate a real budget -----------------------------
    print(f"\nnegatives needed to calibrate a threshold with {STABLE_TAIL} above it:")
    need = {}
    for s in (0.99, 0.999, 0.9999):
        n_total = math.ceil(STABLE_TAIL / (1 - s))
        need[str(s)] = {"negatives_in_calibration_split": n_total,
                        "panel_negatives_required": math.ceil(n_total / FRAC)}
        print(f"  specificity {s:<7} needs {n_total:>7,} held-out negatives, "
              f"so a panel of {math.ceil(n_total / FRAC):>7,}")
    print(f"  this panel has {len(N)} negatives, {held} held out")

    verdict = (f"SURVIVES: margin still orders the classes at the strictest estimable "
               f"specificity {strictest:.2f} (rho {triage[strictest]['rho']:+.3f}, "
               f"p {triage[strictest]['perm_p']:.4f}), so the triage is not an artifact of a "
               f"lax threshold"
               if p1 else
               f"LAX-ONLY: at specificity {strictest:.2f} the margin ordering gives rho "
               f"{triage[strictest]['rho']:+.3f} at p {triage[strictest]['perm_p']:.4f}, so the "
               f"triage signal weakens as the budget tightens and must be quoted with its "
               f"operating point")
    print(f"\nverdict: {verdict}")

    dest = RES / "deployment_operating_points.json"
    json.dump({"panel": pv, "n_negatives": len(N), "held_out_per_fold": held,
               "finest_estimable_fpr": 1.0 / held, "estimable_ceiling": ceiling,
               "specificities": specs, "strictest_estimable": strictest,
               "catch_by_spec": {str(s): catch[s] for s in specs},
               "margins": margins,
               "triage_by_spec": {str(s): triage[s] for s in specs},
               "P1_triage_survives": bool(p1),
               "queue_size": QUEUE, "volume": {str(s): volume[s] for s in specs},
               "negatives_needed": need,
               "caveat": ("volume is arithmetic on measured TPR and FPR and assumes the queue is "
                          "drawn like the panel's negatives, which no real order queue is; it is "
                          "a scale check rather than a forecast"),
               "verdict": verdict}, open(dest, "w"), indent=2)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
