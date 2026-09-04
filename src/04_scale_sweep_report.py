#!/usr/bin/env python3
"""
04_scale_sweep_report.py - cross the model-scale axis with the negative-set axis.

Two questions this answers, both raised by the 2026-09-04 results and neither
answerable from the numbers that existed before it:

  Q1  Does the T3SS model-scale effect survive on the harder panel? The original
      38-point 150M-to-650M swing was measured on the 91-negative panel, and the
      150M column was never recomputed after the panel changed, so it was never a
      like-for-like comparison. Every scale here is embedded from the SAME panel.

  Q2  Are the two fragility axes really near-orthogonal? On 650M, T3SS moves with
      model scale and not with the negative set, while beta-lactamase and
      pore-forming do the opposite. If that is a property of the evaluation rather
      than a coincidence of one model, the per-class scale sensitivity and the
      per-class negative-set sensitivity should stay uncorrelated across classes.

Inputs, one pair per model tag:
    results/v2/lomo_results{_tag}.json                 recovery on the full panel
    results/v2/negative_difficulty_curve{_tag}.json    delta T1->T3 per arm

Sensitivity definitions, both non-negative and on the same scale so they can be
compared directly:
    scale_sensitivity     range of flagged@95 across model scales, per class
    negset_sensitivity    |delta T1->T3| in the `full` arm, per class, at the
                          largest model, since that is the configuration reported

Orthogonality is tested with a Spearman correlation between the two, plus an exact
permutation p-value over class labels. Spearman rather than Pearson because the
sensitivities are bounded and a couple of classes sit at zero.

Usage:
    python src/04_scale_sweep_report.py --tags esm2_8M esm2_35M esm2_150M "" esm2_3B \\
                                        --labels 8M 35M 150M 650M 3B
"""

import argparse
import itertools
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "results" / "v2"


def load(tag, stem):
    suf = f"_{tag}" if tag else ""
    p = V2 / f"{stem}{suf}.json"
    return json.load(open(p)) if p.exists() else None


def rankdata(x):
    """Average ranks, ties shared. Avoids a scipy dependency."""
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float)
    r[order] = np.arange(len(x), dtype=float)
    x = np.asarray(x, float)
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = math.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / d) if d else float("nan")


def perm_p(a, b, rho, n_max=200000):
    """Exact over permutations when the class count is small, else sampled."""
    n = len(a)
    if math.factorial(n) <= n_max:
        perms = itertools.permutations(range(n))
        rhos = np.array([spearman(a, [b[i] for i in p]) for p in perms])
    else:
        rng = np.random.default_rng(0)
        rhos = np.array([spearman(a, rng.permutation(b)) for _ in range(20000)])
    return float((np.abs(rhos) >= abs(rho) - 1e-12).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tags",
        nargs="+",
        required=True,
        help='model tags in increasing size; use "" for the untagged run',
    )
    ap.add_argument("--labels", nargs="+", required=True)
    a = ap.parse_args()
    assert len(a.tags) == len(a.labels), "--tags and --labels must be the same length"

    lomo = {lab: load(t, "lomo_results") for t, lab in zip(a.tags, a.labels)}
    curve = {lab: load(t, "negative_difficulty_curve") for t, lab in zip(a.tags, a.labels)}
    have = [lab for lab in a.labels if lomo[lab]]
    missing = [lab for lab in a.labels if not lomo[lab]]
    if missing:
        print(f"missing lomo results, skipped: {missing}\n")
    if not have:
        raise SystemExit("no results found")

    classes = sorted(set.intersection(*[set(lomo[lab]["leave_one_mechanism_out"]) for lab in have]))

    # ---- Q1: recovery against model scale, panel fixed ----------------------
    # The seed-to-seed spread inside one scale is the yardstick for whether a
    # difference between scales means anything. Without it a non-monotonic curve
    # cannot be told apart from noise.
    def seed_sd(lab, C):
        rec = lomo[lab]["leave_one_mechanism_out"][C].get("per_seed") or []
        v = [r["flagged_95"] for r in rec]
        return float(np.std(v)) if v else float("nan")

    print("Q1  flagged@95 on the full panel, same panel at every scale, mean ± sd over seeds")
    print(
        f"{'class':<30}{''.join(f'{lab:>14}' for lab in have)}{'range':>8}{'pooled sd':>11}{'':>3}"
    )
    print("-" * (30 + 14 * len(have) + 22))
    scale_sens, monotonic = {}, {}
    for C in classes:
        v = [lomo[lab]["leave_one_mechanism_out"][C]["flagged_95_mean"] for lab in have]
        sd = [seed_sd(lab, C) for lab in have]
        pooled = float(np.nanmean(sd))
        rng_ = max(v) - min(v)
        scale_sens[C] = rng_
        # a dip is called real only if it is larger than twice the pooled seed sd
        dips = [
            i
            for i in range(1, len(v) - 1)
            if v[i] < v[i - 1] - 2 * pooled and v[i] < v[i + 1] - 2 * pooled
        ]
        tail = v[-1] < v[-2] - 2 * pooled
        monotonic[C] = {
            "range": rng_,
            "pooled_seed_sd": pooled,
            "real_dip_at": [have[i] for i in dips],
            "drops_at_largest": bool(tail),
        }
        mark = "" if not (dips or tail) else " ^"
        print(
            f"{C:<30}"
            + "".join(f"{x:>8.0%}±{s:<5.0%}" for x, s in zip(v, sd))
            + f"{rng_:>7.0%}{pooled:>10.0%}{mark:<3}"
        )
    print("  ^ marks a non-monotonicity larger than twice the pooled seed sd, so not noise")
    n_neg = {lab: lomo[lab]["n_negative"] for lab in have}
    print(f"\n  negatives per run: {n_neg}  (identical means like-for-like)")

    # ---- Q2: is negative-set sensitivity independent of scale sensitivity ----
    per_scale_negset = {}
    for lab in have:
        if not curve[lab]:
            continue
        pc = curve[lab]["arms"]["full"]["per_class"]
        per_scale_negset[lab] = {C: abs(pc[C]["delta95_t1_t3"]) for C in classes if C in pc}
    ortho = {}
    if per_scale_negset:
        # Test the independence claim at EVERY scale that has a decomposition, not
        # just at one reference model. A claim that only holds for the model it was
        # noticed on is a coincidence, not a property of the evaluation.
        print("\nQ2  per-class sensitivity to each axis (larger = more fragile)")
        cols = list(per_scale_negset)
        print(f"{'class':<30}{'scale':>8}" + "".join(f"{'negset@' + lab:>13}" for lab in cols))
        print("-" * (30 + 8 + 13 * len(cols)))
        cs = [C for C in classes if all(C in per_scale_negset[lab] for lab in cols)]
        for C in cs:
            print(
                f"{C:<30}{scale_sens[C]:>7.0%}"
                + "".join(f"{per_scale_negset[lab][C]:>12.0%} " for lab in cols)
            )
        x = [scale_sens[C] for C in cs]
        print("\n  Spearman correlation between the two sensitivities, per model:")
        for lab in cols:
            y = [per_scale_negset[lab][C] for C in cs]
            r = spearman(x, y)
            pp = perm_p(x, y, r)
            ortho[lab] = {"spearman_rho": r, "permutation_p": pp}
            flag = "independent" if pp > 0.05 else "NOT independent"
            print(f"    {lab:<8} rho = {r:+.3f}   p = {pp:.3f}   -> {flag}")
        ps = [ortho[lab]["permutation_p"] for lab in cols]
        rho = float(np.mean([ortho[lab]["spearman_rho"] for lab in cols]))
        p = float(min(ps))
        print(
            f"\n  n = {len(cs)} classes. Smallest p across models = {p:.3f}; mean rho = {rho:+.3f}."
        )
        print(
            "  -> "
            + (
                "independence holds at every scale tested"
                if p > 0.05
                else "at least one scale shows the axes are not independent"
            )
        )
    else:
        rho = p = None
        print("\nQ2  skipped: no negative_difficulty_curve found for any model")

    out = {
        "labels": have,
        "n_negative_per_run": n_neg,
        "flagged95_by_scale": {
            C: [lomo[lab]["leave_one_mechanism_out"][C]["flagged_95_mean"] for lab in have]
            for C in classes
        },
        "scale_sensitivity": scale_sens,
        "scale_monotonicity": monotonic,
        "negset_sensitivity_by_scale": per_scale_negset,
        "orthogonality_per_model": ortho,
        "orthogonality": {
            "mean_spearman_rho": rho,
            "min_permutation_p": p,
            "models_tested": list(per_scale_negset),
        },
    }
    p_out = V2 / "scale_sweep_report.json"
    json.dump(out, open(p_out, "w"), indent=2)
    print(f"\nwrote {p_out}")


if __name__ == "__main__":
    main()
