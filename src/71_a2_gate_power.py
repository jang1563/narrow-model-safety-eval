#!/usr/bin/env python3
"""
71_a2_gate_power.py - is the P5 gate's AUROC half a test now that n_benign is 60?

The fifth amendment of `docs/MUTATION_EXTENSION_PREREGISTRATION.md` retired that half at
n_benign = 4, showing its null standard deviation was 0.167 so a +/- 0.05 tolerance failed a clean
pipeline 77.3% of the time. It also predicted the repair: the tolerance "becomes meaningful somewhere
above n_benign of roughly 30". Study A2 delivered 60, so that prediction is now testable.

⚠️ This script decides nothing about dFSPE-M. It asks only whether the gate statistic can
discriminate at the sample sizes actually in hand, and what its null looks like as n_benign grows
with the panel fixed at the 15 proteins that have functional-site annotations.

Usage:
    python src/71_a2_gate_power.py
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
N_PERM = 20000
TOL = 0.05


def auroc(pos, neg):
    if not pos or not neg:
        return None
    a = np.concatenate([np.asarray(pos, float), np.asarray(neg, float)])
    order = a.argsort()
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(1, len(a) + 1)
    for u in np.unique(a):
        m = a == u
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    n1 = len(pos)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(neg)))


def null_of_group_auroc(values, n1, rng, n_perm=N_PERM):
    """Permute WHICH proteins are labelled panel. The fifth amendment's construction, unchanged."""
    v = np.asarray(values, float)
    out = np.empty(n_perm)
    for i in range(n_perm):
        q = rng.permutation(v)
        out[i] = auroc(list(q[:n1]), list(q[n1:]))
    return out


def main():
    rng = np.random.default_rng(0)
    res = {"n_perm": N_PERM, "tolerance": TOL}

    for name, path in (("frozen4", "results/fspe_m.json"), ("a2", "results/fspe_m_a2.json")):
        f = ROOT / path
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        sh = {x["acc"]: x["dfspe_m"] for x in d["shuffled"] if x["dfspe_m"] is not None}
        pan = [r["acc"] for r in d["panel"] if r["acc"] in sh]
        con = [c["acc"] for c in d["controls"] if c["acc"] in sh]
        vals = [sh[a] for a in pan] + [sh[a] for a in con]
        obs = d["P5_gate"]["shuffled_auroc"]
        null = null_of_group_auroc(vals, len(pan), rng)
        within = float(np.mean(np.abs(null - 0.5) <= TOL))
        p2s = float(np.mean(np.abs(null - 0.5) >= abs(obs - 0.5)))
        res[name] = {
            "n_panel": len(pan), "n_control": len(con),
            "observed_shuffled_auroc": obs,
            "null_mean": round(float(null.mean()), 4), "null_sd": round(float(null.std()), 4),
            "p_within_tolerance": round(within, 4),
            "clean_pipeline_fails_this_gate": round(1 - within, 4),
            "sd_from_half": round(abs(obs - 0.5) / float(null.std()), 2),
            "two_sided_p_for_observed": round(p2s, 4),
            "is_a_test": bool(within >= 0.80),
        }
        print(f"{name}: n {len(pan)} vs {len(con)}, observed {obs:.4f}, "
              f"null sd {null.std():.4f}, a clean pipeline fails {1 - within:.1%} of the time, "
              f"two-sided p {p2s:.4f}")

    # 🔴 The fifth amendment blamed n_benign. AUROC precision is set by the SMALLER group, and the
    # panel cannot grow: 15 proteins carry functional-site annotations and that is the whole set. So
    # sweep n_benign with n_panel fixed and show where the tolerance can possibly get to.
    d = json.loads((ROOT / "results/fspe_m_a2.json").read_text())
    sh = [x["dfspe_m"] for x in d["shuffled"] if x["dfspe_m"] is not None]
    spread = float(np.std(sh))
    sweep = {}
    for nb in (4, 10, 30, 60, 120, 500, 5000):
        v = rng.normal(0.0, spread, 14 + nb)          # a clean pipeline: no group difference at all
        null = null_of_group_auroc(v, 14, rng, n_perm=4000)
        within = float(np.mean(np.abs(null - 0.5) <= TOL))
        sweep[nb] = {"null_sd": round(float(null.std()), 4),
                     "p_within_tolerance": round(within, 4),
                     "is_a_test": bool(within >= 0.80)}
        print(f"  n_panel=14, n_benign={nb:5d}: null sd {null.std():.4f}, "
              f"clean pipeline passes {within:.1%}")
    res["sweep_n_benign_at_fixed_panel_14"] = sweep
    res["ceiling_note"] = (
        "An AUROC's precision is set by the smaller group. The panel contributes 14 to 15 scorable "
        "proteins and cannot grow: functional_sites.json has 16 entries, 15 with catalytic residues. "
        "So the null sd floors out no matter how many controls are added, and the fifth amendment's "
        "prediction that the tolerance becomes meaningful above n_benign of roughly 30 attributes "
        "the problem to the wrong sample size.")

    out = ROOT / "results" / "a2_gate_power.json"
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
