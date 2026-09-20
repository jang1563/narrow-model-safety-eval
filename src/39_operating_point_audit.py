#!/usr/bin/env python3
"""
39_operating_point_audit.py - does the negative-set gain survive at the ORIGINAL operating point?

What 38 found, and why it is not yet an answer
---------------------------------------------
38 split the gain from growing the negative set into a decision-boundary arm and a
threshold-estimation arm. On the canonical 650M arm at K=8259:

    class                           K=0     boundary_only   threshold_only   both
    phage_peptidoglycan_hydrolase   12.2%          32.5%             0.9%    51.7%
    beta_lactamase                  21.2%           7.1%            11.2%    39.5%
    rip_rrna_glycosidase (healthy)  94.8%          56.7%            89.0%    90.0%

The additive residual is +30.4 points for the phage class and +42.3 for beta-lactamase, so the
two single-factor arms together fall far short of `both` and 38's `share bnd` statistic (0.51 and
-0.77) carries almost no information. 38 preregistered that case: "if the two arms together fall
well short of `both` then the effects interact and that is reported too."

The interaction has a candidate identity that 38 cannot test
-----------------------------------------------------------
Recovery at a fixed specificity depends only on where the held-out positives rank against the
CALIBRATION negatives. Of 38's three arms, only `boundary_only` leaves that population alone: its
threshold is the 0.95 quantile of the panel's own 118 held-out negatives, the same population the
published baseline uses. In `both` at K=8259 the calibration set is 118 panel negatives plus 3,303
pool proteins, so 96.5% of it is pool. Those proteins were also in training, the model scores them
low, and the 0.95 quantile of a mostly-pool calibration set is a much lower bar than the 0.95
quantile of 118 curated hard negatives.

If that is what happened, `both`'s curve compares operating points rather than classifiers, and
95% specificity against a broad benign background is a weaker requirement than the panel's 95%
against matched negatives. That is a measurement, not an argument, so it gets measured.

PREREGISTERED, written before the run
-------------------------------------
    Q1  Take `both`'s MODEL at each K and set its threshold on the panel's 118 held-out negatives
        instead of on its own calibration set. Call that recovery_at_panel_op.
          Q1a  recovery_at_panel_op stays above the K=0 baseline. Then a real discrimination gain
               exists and §10.7.1's closing line needs correcting for the classes where it holds.
          Q1b  recovery_at_panel_op falls to or below the baseline. Then the whole of 37's three
               to fourfold improvement is an operating-point change, §10.7.1's closing line
               survives, and 37's headline has to be withdrawn.

    Q2  What specificity against the panel's hard negatives does `both` actually run at? Reported
        as fp_panel_hard, the fraction of the panel's 118 held-out negatives above the threshold.
        Nominal is 0.05. Anything materially above it means 37's K>0 points were never at the
        same false-positive budget as its K=0 point.

Three self-tests, any of which failing invalidates the run
---------------------------------------------------------
    S1  boundary_only's fp_panel_hard is ~0.05 at every K, because its calibration set IS those
        118 proteins. Deviation beyond quantile granularity means the audit is measuring the wrong
        array.
    S2  boundary_only's recovery_at_panel_op equals its recovery exactly, same reason.
    S3  threshold_only's model never changes with K, so its recovery_at_panel_op is flat in K and
        equals the K=0 baseline. A trend there means the rng stream drifted from 38's.
    Additionally every `recovery` reproduces 38's artifact to 1e-9, which pins the fit order.

Usage:
    python src/39_operating_point_audit.py --arm esm2_650M
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
V3 = ROOT / "results" / "v3"
# 🔴 Every constant below must match 38 exactly. The reproduction check in S4 is the only thing
# standing between this script and a silently different experiment.
FRAC, SPEC, SEEDS = 0.40, 0.95, 30
K_GRID = [0, 500, 1500, 4000, 8259]
FAIL_AT = 0.25
CONTROL_CLASS = "virulence_associated_non_toxin"
MODES = ("boundary_only", "threshold_only", "both")


def fit_and_audit(P, tri, hi, N_train, N_cal, panel_cal):
    """One fit, scored three ways: the mode's own recovery, the same model's recovery at the
    panel's operating point, and the false-positive rate the mode's threshold incurs on the
    panel's hard negatives."""
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
    m.fit(np.vstack([P[tri], N_train]), np.r_[np.ones(len(tri)), np.zeros(len(N_train))])
    s_pos = m.predict_proba(P[hi])[:, 1]
    s_panel = m.predict_proba(panel_cal)[:, 1]
    thr_mode = np.quantile(m.predict_proba(N_cal)[:, 1], SPEC)
    thr_panel = np.quantile(s_panel, SPEC)
    return (float((s_pos >= thr_mode).mean()),
            float((s_pos >= thr_panel).mean()),
            float((s_panel >= thr_mode).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="esm2_650M")
    ap.add_argument("--seeds", type=int, default=SEEDS)
    a = ap.parse_args()
    tag = a.arm
    suf = "" if tag == "esm2_650M" else f"_{tag}"

    man = json.load(open(V3 / f"embedding_manifest_v3{suf}.json"))
    mech = json.load(open(ROOT / "data/annotations/mechanism_classes_v3.json"))
    lomo = json.load(open(V3 / f"lomo_results{suf}.json"))["leave_one_mechanism_out"]
    P = np.load(V3 / f"embeddings_positive_v3{suf}.npy")
    N_panel = np.load(V3 / f"embeddings_negative_v3{suf}.npy")
    N_pool = np.load(V3 / f"embeddings_pool_large_{tag}.npy")
    cls = {e["fasta_id"]: e["mechanism_class"] for e in mech["proteins"]}
    pcls = np.array([cls[r["acc"]] for r in man["positive_rows"]])

    prior = V3 / f"threshold_vs_boundary_{tag}.json"
    ref = json.load(open(prior))["curves"] if prior.exists() else None
    print(f"reproduction reference: {'38 artifact' if ref else 'NONE, S4 will be skipped'}")

    failures = sorted((c for c in lomo
                       if lomo[c]["flagged_95_mean"] < FAIL_AT and c != CONTROL_CLASS),
                      key=lambda c: lomo[c]["flagged_95_mean"])
    below = {c: lomo[c]["flagged_95_mean"] for c in lomo
             if c not in failures and c != CONTROL_CLASS
             and lomo[c]["flagged_95_mean"] < 0.999}
    comparison = max(below, key=below.get)
    targets = failures + [comparison]
    ks = [k for k in K_GRID if k <= len(N_pool)]
    print(f"arm {tag}: panel {len(N_panel)}, pool {len(N_pool)}, seeds {a.seeds}")
    print(f"failures {failures}, comparison {comparison}\n")

    out = {c: {m: {} for m in MODES} for c in targets}
    for c in targets:
        hi = np.where(pcls == c)[0]
        tri = np.setdiff1d(np.arange(len(P)), hi)
        for k in ks:
            acc = {m: {"rec": [], "rec_panel_op": [], "fp_panel": []} for m in MODES}
            for seed in range(a.seeds):
                # Identical rng stream to 38: one permutation of the panel, then one of the pool.
                rng = np.random.default_rng(seed)
                perm = rng.permutation(len(N_panel))
                cut = int(len(N_panel) * FRAC)
                cal_idx, tr_idx = perm[:cut], perm[cut:]
                base_cal, base_tr = N_panel[cal_idx], N_panel[tr_idx]
                add = N_pool[rng.permutation(len(N_pool))[:k]] if k else N_pool[:0]
                split = int(len(add) * FRAC) if k else 0
                plan = {
                    "boundary_only": (np.vstack([base_tr, add]), base_cal),
                    "threshold_only": (base_tr, np.vstack([base_cal, add])),
                    "both": (np.vstack([base_tr, add[split:]]),
                             np.vstack([base_cal, add[:split]])),
                }
                for mode in MODES:
                    n_tr, n_cal = plan[mode]
                    r, rp, fp = fit_and_audit(P, tri, hi, n_tr, n_cal, base_cal)
                    acc[mode]["rec"].append(r)
                    acc[mode]["rec_panel_op"].append(rp)
                    acc[mode]["fp_panel"].append(fp)
            for mode in MODES:
                out[c][mode][str(k)] = {
                    key: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))}
                    for key, v in acc[mode].items()}

    # ---- self-tests ----------------------------------------------------------------
    fails = []
    for c in targets:
        for k in ks:
            b = out[c]["boundary_only"][str(k)]
            # S1: the 0.95 quantile of 118 points sits between order statistics, so the realised
            # rate is granular at 1/118 = 0.0085. Allow two steps.
            if abs(b["fp_panel"]["mean"] - 0.05) > 0.018:
                fails.append(f"S1 {c} K={k}: fp_panel {b['fp_panel']['mean']:.4f} not ~0.05")
            if abs(b["rec"]["mean"] - b["rec_panel_op"]["mean"]) > 1e-9:
                fails.append(f"S2 {c} K={k}: rec {b['rec']['mean']} != "
                             f"rec_panel_op {b['rec_panel_op']['mean']}")
        t0 = out[c]["threshold_only"][str(ks[0])]["rec_panel_op"]["mean"]
        for k in ks:
            tk = out[c]["threshold_only"][str(k)]["rec_panel_op"]["mean"]
            if abs(tk - t0) > 1e-9:
                fails.append(f"S3 {c} K={k}: threshold_only model changed, {tk} != {t0}")
        if ref:
            for mode in MODES:
                for k in ks:
                    got = out[c][mode][str(k)]["rec"]["mean"]
                    want = ref[c][mode][str(k)]["mean"]
                    if abs(got - want) > 1e-9:
                        fails.append(f"S4 {c} {mode} K={k}: {got} != 38's {want}")
    print("self-tests: " + ("ALL PASS" if not fails else f"{len(fails)} FAILURE(S)"))
    for f in fails[:12]:
        print(f"  {f}")
    if fails:
        raise SystemExit("self-test failure, results not written")

    # ---- the two preregistered questions -------------------------------------------
    hdr = (f"{'class':<32}{'mode':<16}{'quantity':<16}"
           + "".join(f"{f'K={k}':>9}" for k in ks))
    print(f"\n{hdr}\n" + "-" * len(hdr))
    for c in targets:
        for mode in MODES:
            for key, label in (("rec", "recovery"), ("rec_panel_op", "rec @panel op"),
                               ("fp_panel", "FP on 118 hard")):
                r = out[c][mode]
                print(f"{c:<32}{mode:<16}{label:<16}"
                      + "".join(f"{r[str(k)][key]['mean'] * 100:>8.1f}%" for k in ks))
        print()

    print(f"{'class':<32}{'base':>8}{'both rec':>10}{'both @op':>10}"
          f"{'gain @op':>10}{'both FP':>9}{'verdict':>10}")
    summary = {}
    for c in targets:
        base = out[c]["both"][str(ks[0])]["rec"]["mean"]
        bk = out[c]["both"][str(ks[-1])]
        best_k = max(ks, key=lambda k: out[c]["both"][str(k)]["rec_panel_op"]["mean"])
        best_op = out[c]["both"][str(best_k)]["rec_panel_op"]["mean"]
        gain_op = best_op - base
        # ⚠️ best_op is a max over five K values on the same seeds, a post-hoc selection. It is
        # used deliberately: it is the most generous test the data allows of "adding negatives
        # improves the model", so an ARTEFACT verdict that survives it is safe, while a SURVIVES
        # verdict has to be reported with the K it was selected at.
        v = "SURVIVES" if gain_op > 0.02 else "ARTEFACT"
        summary[c] = {"baseline": base, "both_recovery_maxK": bk["rec"]["mean"],
                      "both_recovery_at_panel_op_maxK": bk["rec_panel_op"]["mean"],
                      "both_recovery_at_panel_op_best_over_K": best_op,
                      "best_K_post_hoc": best_k,
                      "gain_at_panel_op_pts": gain_op * 100,
                      "both_fp_on_panel_hard_maxK": bk["fp_panel"]["mean"],
                      "verdict": v}
        print(f"{c:<32}{base * 100:>7.1f}%{bk['rec']['mean'] * 100:>9.1f}%"
              f"{bk['rec_panel_op']['mean'] * 100:>9.1f}%{gain_op * 100:>+9.1f}"
              f"{bk['fp_panel']['mean'] * 100:>8.1f}%{v:>10}   (best K={best_k})")

    fail_verdicts = {summary[c]["verdict"] for c in failures}
    if fail_verdicts == {"SURVIVES"}:
        verdict = ("Q1a: the gain survives at the panel's own operating point for every failing "
                   "class, so a real discrimination gain exists")
    elif fail_verdicts == {"ARTEFACT"}:
        verdict = ("Q1b: no failing class beats its baseline once the threshold is set on the "
                   "panel's 118 hard negatives, so 37's improvement is an operating-point change "
                   "and §10.7.1's closing line survives")
    else:
        verdict = ("Q1 SPLIT: " + ", ".join(f"{c}={summary[c]['verdict']}" for c in failures)
                   + ", so the answer is class-specific and cannot be stated once for the panel")
    print(f"\nverdict: {verdict}")

    dest = V3 / f"operating_point_audit_{tag}.json"
    json.dump({"arm": tag, "K_grid": ks, "seeds": a.seeds, "failures": failures,
               "comparison": comparison, "curves": out, "summary": summary,
               "self_tests": "all pass", "verdict": verdict}, open(dest, "w"), indent=2)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
