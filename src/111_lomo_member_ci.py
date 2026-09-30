#!/usr/bin/env python3
"""
111_lomo_member_ci.py - the published LOMO intervals locate seed means, not recovery rates

A review check, not a new study. `[11.0, 19.9]` on beta-lactamase recovery is `mean +/- 1.96*sd/sqrt(30)`
-- a standard error of the SEED mean. On a 14-member class its half-width of 4.5 points is 0.62 of one
protein, and the seed spread it comes from is [-8.9, 39.9].

🔒 This recomputes every holdout-eligible class's recovery with the member level included, straight from
the stored per-seed decisions, and reports both intervals side by side.

⚠️ The bimodal finding does not depend on it: 100% and [0, 36] do not overlap. What does not survive is
the apparent precision of any single low number.

Usage:
    python src/111_lomo_member_ci.py
    python src/111_lomo_member_ci.py --selftest
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LOMO = ROOT / "results" / "v3" / "lomo_results.json"
POOLED = ROOT / "results" / "v2" / "supervised_pooling_v2_30seeds.json"
OUT = ROOT / "results" / "lomo_member_ci.json"
B = 20000
LO, HI = 2.5, 97.5


def member_boot(per_seed, rng, b=B):
    """Resample a seed, then that seed's members with replacement, using its own threshold."""
    out = []
    for _ in range(b):
        s = per_seed[rng.integers(len(per_seed))]
        pr = np.asarray(s["probs"], float)
        idx = rng.integers(0, len(pr), len(pr))
        out.append(float((pr[idx] >= s["t95"]).mean()))
    return np.array(out)


def se_interval(v):
    v = np.asarray(v, float)
    h = 1.96 * v.std(ddof=1) / len(v) ** 0.5
    return float(v.mean()), [float(v.mean() - h), float(v.mean() + h)]


def selftest():
    rng = np.random.default_rng(0)
    # 🔒 a class whose members are FIXED and whose seeds barely differ: the seed SE must be far
    # narrower than the member interval, and the member interval must straddle multiples of 1/n
    n, seeds = 14, 30
    pr_true = np.r_[np.full(2, 0.9), np.full(n - 2, 0.1)]      # 2 of 14 recoverable
    per_seed = [{"probs": list(pr_true + rng.normal(0, 0.01, n)), "t95": 0.5} for _ in range(seeds)]
    fl = np.array([(np.asarray(s["probs"]) >= s["t95"]).mean() for s in per_seed])
    _, se = se_interval(fl)
    mb = member_boot(per_seed, np.random.default_rng(1), 4000)
    w_se, w_mb = se[1] - se[0], np.percentile(mb, HI) - np.percentile(mb, LO)
    assert w_mb > 5 * w_se, (w_se, w_mb)
    assert w_se < 1.0 / n, f"the seed SE must be narrower than one member here: {w_se} vs {1/n}"
    # 🔴 and the check must NOT fire when the class is large: the same construction at n=400 should
    # bring the member interval close to the seed one, or this is measuring nothing about class size
    n2 = 400
    pr2 = np.r_[np.full(57, 0.9), np.full(n2 - 57, 0.1)]
    ps2 = [{"probs": list(pr2 + rng.normal(0, 0.01, n2)), "t95": 0.5} for _ in range(seeds)]
    mb2 = member_boot(ps2, np.random.default_rng(1), 4000)
    w2 = np.percentile(mb2, HI) - np.percentile(mb2, LO)
    assert w2 < 0.4 * w_mb, (w_mb, w2)
    print("selftest PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        return selftest()

    d = json.loads(LOMO.read_text())["leave_one_mechanism_out"]
    rng = np.random.default_rng(0)
    rows = []
    for name, c in sorted(d.items()):
        ps = c.get("per_seed") or []
        if not ps or "probs" not in ps[0]:
            continue
        n = len(ps[0]["probs"])
        fl = np.array([(np.asarray(s["probs"], float) >= s["t95"]).mean() for s in ps])
        mean, se = se_interval(fl)
        mb = member_boot(ps, rng)
        rows.append({"class": name, "n": n, "seeds": len(ps),
                     "recovery": round(mean * 100, 2),
                     "seed_se_ci": [round(se[0] * 100, 1), round(se[1] * 100, 1)],
                     "member_ci": [round(float(np.percentile(mb, LO)) * 100, 1),
                                   round(float(np.percentile(mb, HI)) * 100, 1)],
                     "one_member_pp": round(100 / n, 1),
                     "se_halfwidth_in_members": round((se[1] - se[0]) / 2 / (100 / n) * 100, 2)})

    # the published 30-seed figure, which is the one the abstract quotes
    pooled = json.loads(POOLED.read_text())
    b = pooled["by_k"]["0"]["beta_lactamase"]
    m, sd, ns = b["flagged_95_mean"], b["flagged_95_sd"], pooled.get("seeds", 30)
    h = 1.96 * sd / ns ** 0.5
    n_bl = next((r["n"] for r in rows if r["class"] == "beta_lactamase"), 14)
    bl = {"recovery": round(m * 100, 2), "seeds": ns,
          "published_ci": [round((m - h) * 100, 1), round((m + h) * 100, 1)],
          "seed_spread": [round((m - 1.96 * sd) * 100, 1), round((m + 1.96 * sd) * 100, 1)],
          "halfwidth_pp": round(h * 100, 2), "one_member_pp": round(100 / n_bl, 1),
          "halfwidth_in_members": round(h * 100 / (100 / n_bl), 2),
          "member_ci_at_this_rate": [
              round(float(np.percentile([(rng.random(n_bl) < m).mean() for _ in range(B)], LO)) * 100, 1),
              round(float(np.percentile([(rng.random(n_bl) < m).mean() for _ in range(B)], HI)) * 100, 1)]}

    disjoint = [r for r in rows if r["member_ci"][1] < 100.0 and r["recovery"] < 50]
    res = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "B": B, "classes": rows,
           "published_30seed_beta_lactamase": bl,
           "n_classes": len(rows),
           "n_low_classes": len(disjoint),
           # 🔒 the finding that does not depend on any of this
           "bimodality_survives": bool(
               any(r["member_ci"][0] >= 99.0 for r in rows)
               and any(r["member_ci"][1] <= 60.0 for r in rows))}
    OUT.write_text(json.dumps(res, indent=2) + "\n")

    print(f"{'class':<34}{'n':>4}{'recov':>8}{'seed SE':>16}{'member':>16}{'SE in members':>15}")
    print("-" * 93)
    for r in rows:
        print(f"{r['class']:<34}{r['n']:>4}{r['recovery']:>7.1f}%"
              f"   [{r['seed_se_ci'][0]:>5.1f},{r['seed_se_ci'][1]:>6.1f}]"
              f"   [{r['member_ci'][0]:>5.1f},{r['member_ci'][1]:>6.1f}]"
              f"{r['se_halfwidth_in_members']:>13.2f}")
    print(f"\npublished 30-seed beta-lactamase: {bl['recovery']}%  {bl['published_ci']}"
          f"   half-width {bl['halfwidth_pp']}pp = {bl['halfwidth_in_members']} members")
    print(f"  the seed spread it comes from: {bl['seed_spread']}")
    print(f"  member resampling at that rate: {bl['member_ci_at_this_rate']}")
    print(f"\nbimodality survives member-level intervals: {res['bimodality_survives']}")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
