#!/usr/bin/env python3
"""Study AG -- does the curator's own class axis order categories by HAZARD?

Studies B/B-35M (src/81) held each of VFDB's 13 eligible categories out in turn and
measured recovery.  Study Q (src/74) asked the hazard question on *my* four-category
grouping and found parity.  This script asks the hazard question on the CURATOR's
labels, over a population that does contain exotoxins (n=102), using the SAME frozen
grouping study Q declared -- so no new judgment of mine enters.

Written 2026-10-01 after a 650M-only ranking was drafted into the paper and the second
arm reversed its top slot.  The grouping is frozen (docs/HAZARD_VS_MEMBERSHIP_
PREREGISTRATION.md section 1.1); the CONTRAST is a retrodiction -- the author had seen
both arms' per-category means before choosing these estimators.  Disclosed, not sealed.
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ARMS = {
    "650M": ROOT / "results/external_class_axis_lomo.json",
    "35M": ROOT / "results/external_class_axis_lomo_esm2_35M.json",
}
BUILD = ROOT / "results/external_class_axis_build.json"
OUT = ROOT / "results/class_axis_hazard_order.json"

# Frozen in docs/HAZARD_VS_MEMBERSHIP_PREREGISTRATION.md section 1.1, before any of
# these numbers existed.  Copied verbatim; not re-derived here.
NON_HAZARDOUS = (
    "Motility",
    "Nutritional/Metabolic factor",
    "Regulation",
    "Stress survival",
)
FROZEN_SOURCE = "docs/HAZARD_VS_MEMBERSHIP_PREREGISTRATION.md section 1.1"


def mann_whitney_p(a, b):
    """Exact two-sided Mann-Whitney p by enumeration.  No ties expected (floats)."""
    a, b = list(a), list(b)
    n, m = len(a), len(b)
    pooled = a + b
    obs = sum(1 for x in a for y in b if x > y) + 0.5 * sum(
        1 for x in a for y in b if x == y
    )
    idx = range(n + m)
    counts = 0
    total = 0
    for pick in combinations(idx, n):
        ga = [pooled[i] for i in pick]
        gb = [pooled[i] for i in idx if i not in pick]
        u = sum(1 for x in ga for y in gb if x > y) + 0.5 * sum(
            1 for x in ga for y in gb if x == y
        )
        total += 1
        # two-sided: as or more extreme than observed, measured from the null centre
        if abs(u - n * m / 2.0) >= abs(obs - n * m / 2.0) - 1e-12:
            counts += 1
    return obs, counts / total


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    d = float(np.sqrt((rx**2).sum() * (ry**2).sum()))
    return float((rx * ry).sum() / d) if d > 0 else 0.0


def boot_gap(haz, non, B=20000, seed=0):
    """Category-level bootstrap on the group gap.

    The unit is the CATEGORY, because the question is about categories.  With 9 and 4
    units this is a wide, honest interval -- not a seed SE, which would be 20x too
    narrow (entry 83's lesson).
    """
    rng = np.random.default_rng(seed)
    haz, non = np.asarray(haz, float), np.asarray(non, float)
    draws = np.empty(B)
    for i in range(B):
        h = rng.choice(haz, size=haz.size, replace=True)
        v = rng.choice(non, size=non.size, replace=True)
        draws[i] = h.mean() - v.mean()
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def covariates():
    """Per-category median length, member count and species count, from the build."""
    pos = json.loads(BUILD.read_text())["positives"]
    by = {}
    for r in pos:
        by.setdefault(r["cat"], []).append(r)
    return {
        c: {
            "n": len(rs),
            "median_len": float(np.median([r["len"] for r in rs])),
            "n_species": len({r["species"] for r in rs}),
        }
        for c, rs in by.items()
    }


def selftest():
    ok = True

    # mann_whitney_p against a hand-computable case: [1,2,3] vs [4,5,6] is the most
    # extreme split of six values, so p = 2 / C(6,3) = 2/20 = 0.1 exactly.
    u, p = mann_whitney_p([1, 2, 3], [4, 5, 6])
    if not (u == 0 and abs(p - 0.1) < 1e-12):
        print(f"  XX mann_whitney_p extreme case: U={u} p={p} (want 0, 0.1)")
        ok = False
    # and the null: identical groups must not be significant
    _, p_null = mann_whitney_p([1, 2, 3], [1.5, 2.5, 3.5])
    if p_null < 0.3:
        print(f"  XX mann_whitney_p overlapping case p={p_null} (want >= 0.3)")
        ok = False

    if abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) > 1e-12:
        print("  XX spearman monotone != +1")
        ok = False
    if abs(spearman([1, 2, 3, 4], [40, 30, 20, 10]) + 1.0) > 1e-12:
        print("  XX spearman reversed != -1")
        ok = False

    # boot_gap: a planted separation must exclude 0, and no separation must include it.
    lo, hi = boot_gap([0.9] * 9, [0.2] * 4)
    if not (lo > 0.6 and hi < 0.8):
        print(f"  XX boot_gap planted gap [{lo:.3f}, {hi:.3f}] (want inside 0.6-0.8)")
        ok = False
    lo, hi = boot_gap([0.5, 0.6, 0.4] * 3, [0.5, 0.6, 0.4, 0.5])
    if not (lo < 0 < hi):
        print(f"  XX boot_gap null [{lo:.3f}, {hi:.3f}] must span 0")
        ok = False

    # the frozen names must exist in BOTH arms, or the grouping is silently empty
    for arm, path in ARMS.items():
        rec = json.loads(path.read_text())["recovery"]
        miss = [c for c in NON_HAZARDOUS if c not in rec]
        if miss:
            print(f"  XX {arm}: frozen non-hazardous names absent: {miss}")
            ok = False
        if len(rec) != 13:
            print(f"  XX {arm}: {len(rec)} categories, expected 13")
            ok = False

    print("selftest PASS" if ok else "selftest FAIL")
    return 0 if ok else 1


def main():
    cov = covariates()
    out = {
        "built": "2026-10-01",
        "study": "AG -- the curator's class axis as a hazard ordering",
        "frozen_grouping_source": FROZEN_SOURCE,
        "non_hazardous": list(NON_HAZARDOUS),
        "disclosure": (
            "the GROUPING is frozen and pre-dates these numbers; the CONTRAST is a "
            "retrodiction -- the author had seen both arms' per-category means before "
            "choosing these estimators. Not a sealed prediction."
        ),
        "unit_of_interval": "the category (9 vs 4), not the probe seed",
        "arms": {},
    }

    for arm, path in ARMS.items():
        d = json.loads(path.read_text())
        rec = {k: v["mean"] for k, v in d["recovery"].items()}
        ns = {k: v["n"] for k, v in d["recovery"].items()}
        order = sorted(rec, key=rec.get, reverse=True)

        haz_cats = [c for c in rec if c not in NON_HAZARDOUS]
        non_cats = [c for c in rec if c in NON_HAZARDOUS]
        haz = [rec[c] for c in haz_cats]
        non = [rec[c] for c in non_cats]

        wh = sum(rec[c] * ns[c] for c in haz_cats) / sum(ns[c] for c in haz_cats)
        wn = sum(rec[c] * ns[c] for c in non_cats) / sum(ns[c] for c in non_cats)

        u, p = mann_whitney_p(haz, non)
        lo, hi = boot_gap(haz, non)
        cats = list(rec)
        out["arms"][arm] = {
            "seeds": d["seeds"],
            "n_positives": d["n_positives"],
            "n_held_out": len(d["held_out"]),
            "ranking": [
                {"rank": i, "category": c, "recovery": rec[c], "n": ns[c],
                 "margin": d["margin"][c], "hazardous": c not in NON_HAZARDOUS}
                for i, c in enumerate(order, 1)
            ],
            "top_category": order[0],
            "top_is_hazardous": order[0] not in NON_HAZARDOUS,
            "bottom_category": order[-1],
            "exotoxin_rank": order.index("Exotoxin") + 1,
            "motility_rank": order.index("Motility") + 1,
            "group_mean_hazardous": float(np.mean(haz)),
            "group_mean_non_hazardous": float(np.mean(non)),
            "gap_unweighted": float(np.mean(haz) - np.mean(non)),
            "gap_member_weighted": float(wh - wn),
            "gap_ci_category_bootstrap": [lo, hi],
            "gap_excludes_zero": bool(lo > 0 or hi < 0),
            "mann_whitney_U": u,
            "mann_whitney_p_exact": p,
            "exotoxin_minus_best_non_hazardous": float(
                rec["Exotoxin"] - max(rec[c] for c in non_cats)
            ),
            "confounds": {
                "spearman_n_vs_recovery": spearman(
                    [ns[c] for c in cats], [rec[c] for c in cats]
                ),
                "spearman_median_len_vs_recovery": spearman(
                    [cov[c]["median_len"] for c in cats], [rec[c] for c in cats]
                ),
                "spearman_n_species_vs_recovery": spearman(
                    [cov[c]["n_species"] for c in cats], [rec[c] for c in cats]
                ),
            },
            "B1_margin_predicts_recovery": {
                "rho": d["B1"]["rho"], "perm_p": d["B1"]["perm_p"],
                "supported": d["B1"]["supported"],
            },
        }

    a, b = out["arms"]["650M"], out["arms"]["35M"]
    out["second_arm_rule"] = {
        "arms_agree_on_top_category": a["top_category"] == b["top_category"],
        "arms_agree_top_is_hazardous": a["top_is_hazardous"] == b["top_is_hazardous"],
        "arms_agree_on_bottom_category": a["bottom_category"] == b["bottom_category"],
        "arms_agree_gap_sign": (a["gap_unweighted"] > 0) == (b["gap_unweighted"] > 0),
        "both_gaps_exclude_zero": a["gap_excludes_zero"] and b["gap_excludes_zero"],
        "both_mw_significant": a["mann_whitney_p_exact"] < 0.05
        and b["mann_whitney_p_exact"] < 0.05,
    }
    OUT.write_text(json.dumps(out, indent=2))

    for arm in ("650M", "35M"):
        v = out["arms"][arm]
        print(f"=== {arm}  seeds={v['seeds']}")
        for r in v["ranking"]:
            mark = "HAZ" if r["hazardous"] else " . "
            print(f"  {r['rank']:2d} {mark} {r['category']:<32}"
                  f"{r['recovery']*100:6.2f}%  n={r['n']}")
        print(f"  top={v['top_category']} (hazardous={v['top_is_hazardous']})  "
              f"Exotoxin rank {v['exotoxin_rank']}  Motility rank {v['motility_rank']}")
        print(f"  gap unweighted {v['gap_unweighted']*100:+.2f}pp  "
              f"CI [{v['gap_ci_category_bootstrap'][0]*100:+.2f}, "
              f"{v['gap_ci_category_bootstrap'][1]*100:+.2f}]  "
              f"excludes0={v['gap_excludes_zero']}")
        print(f"  gap member-weighted {v['gap_member_weighted']*100:+.2f}pp  "
              f"MW U={v['mann_whitney_U']} p={v['mann_whitney_p_exact']:.4f}")
        print(f"  Exotoxin - best non-hazardous "
              f"{v['exotoxin_minus_best_non_hazardous']*100:+.2f}pp")
        print(f"  confounds {v['confounds']}")
    print("second arm:", json.dumps(out["second_arm_rule"]))
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else main())
