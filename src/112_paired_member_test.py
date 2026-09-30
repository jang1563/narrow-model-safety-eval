#!/usr/bin/env python3
"""
112_paired_member_test.py - does the one repair that worked survive a member-level test?

`docs/MECHANISM_GENERALIZATION.md` reports the `win_best25` pooling reduction moving beta-lactamase from
15.5% to 35.0% and the phage class from 12.3% to 27.3%, and calls the intervals "disjoint". 🔴 Those
intervals are `mean +/- 1.96*sd/sqrt(30)` -- seed standard errors on classes of 14 and 32 members, which
entry 81 showed can be narrower than one protein.

🔒 The right test is PAIRED: the same members under two pooling schemes, so resampling is over members and
the comparison is within-protein. That is both the honest analysis and the more powerful one, and it can
come out either way.

⚠️ Unpaired member intervals are reported alongside, because that is what a reader would construct from the
published table, and the gap between the two is the point.

Usage:
    python src/112_paired_member_test.py
    python src/112_paired_member_test.py --selftest
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "paired_member_test.json"
B = 20000
LO, HI = 2.5, 97.5

# 🔒 the two comparisons the document calls disjoint, with the artifacts that hold per-member decisions
CASES = (("beta_lactamase", "v2", "results/v2/lomo_results_esm2_650M_mean_res.json",
          "results/v2/lomo_results_esm2_650M_win_best25.json", 15.48, 35.00,
          [11.0, 19.9], [31.6, 38.4]),
         ("phage_peptidoglycan_hydrolase", "v3", "results/v3/lomo_results_esm2_650M_mean_res.json",
          "results/v3/lomo_results_esm2_650M_win_best25.json", 12.29, 27.29,
          [10.0, 14.5], [23.3, 31.3]))


def flag_matrix(path, cls):
    """(seeds x members) boolean flags at each seed's own 95% threshold, plus the member names."""
    c = json.loads(Path(ROOT, path).read_text())["leave_one_mechanism_out"][cls]
    M = np.array([[p >= s["t95"] for p in s["probs"]] for s in c["per_seed"]], bool)
    return M, c["members"]


def paired_boot(A, Bm, rng, b=B):
    """Resample MEMBERS once per draw and score both arms on the same members.

    🔒 Paired by construction: `idx` indexes the shared member axis, so a member that is hard under both
    schemes cannot inflate the difference, and one that is easy under both cannot hide it.
    """
    n = A.shape[1]
    d, ra, rb = [], [], []
    for _ in range(b):
        s = rng.integers(A.shape[0])
        idx = rng.integers(0, n, n)
        a, bb = A[s, idx].mean(), Bm[s, idx].mean()
        ra.append(a)
        rb.append(bb)
        d.append(bb - a)
    return np.array(ra), np.array(rb), np.array(d)


def pct(v, scale=100.0):
    return [float(np.percentile(v, LO) * scale), float(np.percentile(v, HI) * scale)]


def selftest():
    n, seeds = 14, 5
    # 🔒 A real paired effect on few members must be detectable BECAUSE it is paired: build arms that
    # differ on 4 of 14 members and agree elsewhere, and check the paired interval excludes 0 while the
    # unpaired member intervals overlap. That gap is the whole reason for this script.
    base = np.zeros((seeds, n), bool)
    base[:, :2] = True
    lift = base.copy()
    lift[:, 2:6] = True
    _, _, d = paired_boot(base, lift, np.random.default_rng(1), 4000)
    assert np.percentile(d, LO) > 0, f"a paired lift of 4/14 must exclude zero: {np.percentile(d, LO)}"
    ra, rb, _ = paired_boot(base, lift, np.random.default_rng(1), 4000)
    ia, ib = pct(ra), pct(rb)
    assert ia[1] > ib[0], f"the unpaired member intervals should overlap here: {ia} {ib}"
    # 🔴 and a null must NOT be called: identical arms give a paired interval containing zero
    _, _, d0 = paired_boot(base, base.copy(), np.random.default_rng(2), 4000)
    assert np.percentile(d0, LO) <= 0 <= np.percentile(d0, HI), "identical arms must not differ"
    assert float(np.abs(d0).max()) == 0.0, "pairing must make identical arms differ by exactly zero"
    print("selftest PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        return selftest()

    rng = np.random.default_rng(0)
    rows = []
    for cls, panel, fa, fb, pub_a, pub_b, ci_a, ci_b in CASES:
        A, ma = flag_matrix(fa, cls)
        Bm, mb = flag_matrix(fb, cls)
        assert ma == mb, f"{cls}: member order differs between arms"
        ra, rb, d = paired_boot(A, Bm, rng)
        seed_a, seed_b = A.mean(1) * 100, Bm.mean(1) * 100
        # 🔒 A count that does NOT depend on how many seeds were averaged: per seed, how many members
        # the reduction gains and how many it loses. The whole paired effect lives in the difference of
        # these two, and with n = 14 a gain of two members is 14.3 points -- so the arithmetic, not the
        # seed count, is what limits resolution.
        gain = (Bm & ~A).sum(1)
        loss = (A & ~Bm).sum(1)
        disc = (gain + loss)
        rows.append({
            "class": cls, "panel": panel, "n_members": A.shape[1], "seeds_available": A.shape[0],
            "published_30seed": [pub_a, pub_b],
            "published_seed_se_ci": [ci_a, ci_b],
            "published_disjoint": bool(ci_a[1] < ci_b[0]),
            "five_seed_means": [round(float(seed_a.mean()), 2), round(float(seed_b.mean()), 2)],
            "member_ci_control": [round(x, 1) for x in pct(ra)],
            "member_ci_reduction": [round(x, 1) for x in pct(rb)],
            "member_intervals_overlap": bool(pct(ra)[1] >= pct(rb)[0]),
            "paired_delta": round(float(d.mean() * 100), 2),
            "paired_delta_ci": [round(x, 1) for x in pct(d)],
            "paired_excludes_zero": bool(np.percentile(d, LO) > 0),
            # per-seed member gains / losses / discordant, as counts
            "gain_per_seed": gain.tolist(), "loss_per_seed": loss.tolist(),
            "net_members_per_seed": (gain - loss).tolist(),
            "median_net_members": float(np.median(gain - loss)),
            "one_member_pp": round(100 / A.shape[1], 1),
            "max_discordant": int(disc.max()),
        })
    res = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "B": B, "cases": rows,
           "verdict": ("the repair survives a paired member-level test"
                       if all(r["paired_excludes_zero"] for r in rows)
                       else "the repair does not survive a paired member-level test on every case")}
    OUT.write_text(json.dumps(res, indent=2) + "\n")

    for r in rows:
        print(f"=== {r['class']} ({r['panel']}, n={r['n_members']}, "
              f"{r['seeds_available']} seeds of per-member data)")
        print(f"  published 30-seed        {r['published_30seed'][0]}% -> {r['published_30seed'][1]}%"
              f"   seed-SE {r['published_seed_se_ci'][0]} vs {r['published_seed_se_ci'][1]}"
              f"   disjoint={r['published_disjoint']}")
        print(f"  member intervals         {r['member_ci_control']} vs {r['member_ci_reduction']}"
              f"   overlap={r['member_intervals_overlap']}")
        print(f"  PAIRED delta             {r['paired_delta']:+.2f}pp  {r['paired_delta_ci']}"
              f"   excludes zero={r['paired_excludes_zero']}")
        print(f"  members gained per seed  {r['gain_per_seed']}   lost {r['loss_per_seed']}"
              f"   net median {r['median_net_members']:+.1f} of {r['n_members']}"
              f"   (one member = {r['one_member_pp']}pp)")
    print(f"\n{res['verdict']}")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
