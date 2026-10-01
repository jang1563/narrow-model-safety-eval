#!/usr/bin/env python3
"""Study AH -- a MATCHED, toxin-inclusive contrast along a THIRD-PARTY hazard axis.

`docs/EXTERNAL_HAZARD_AXIS_PREREGISTRATION.md`, frozen before this file was written.

Study Q used the author's hazard split.  Study AG used VFDB's categories and found the
top slot arm-dependent, the group gap confounded with length and species diversity, and
no matched comparator.  This uses UniProt curators' Swiss-Prot keywords, pairs each
hazard-positive to a length-matched hazard-negative, and holds both groups out in ONE
fold so a single model scores both.

The fold is src/81's, unchanged.  The only new things are the axis, the matching, and the
strict variant that also holds out the whole VFDB Exotoxin category.
"""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results/v3"
BUILD = ROOT / "results/external_class_axis_build.json"
SCREEN = ROOT / "results/external_class_axis_screen.json"
FAS = ROOT / "data/external/vfdb/VFDB_setA_pro.fas"
IDMAP = ROOT / "data/external/uniprot_idmap/class_axis_idmap.tsv"
OUT_STEM = ROOT / "results/external_hazard_axis_lomo"

HAZARD_KW = ("Toxin", "Hemolysis", "Cytolysis", "Enterotoxin", "Neurotoxin")
SEEDS, N_TRAIN, N_CAL, SPEC = 30, 1000, 500, 0.95
BOOT = 20000
HDR = re.compile(r"^>(VFG\d+)\(gb\|([^)]+)\)")
# frozen in section 4 of the preregistration
BAND_AUROC_REG = (0.40, 0.60)
BAND_AUROC_HAZ = 0.70
BAND_DREC_REG = (-0.10, 0.10)
BAND_DREC_HAZ = 0.20
MATCH_DROP_RATIO = 2.0
MATCH_SPECIES_RATIO = 1.5


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))


def auroc(pos, neg):
    if not len(pos) or not len(neg):
        return None
    a = np.concatenate([np.asarray(pos, float), np.asarray(neg, float)])
    o = a.argsort()
    r = np.empty(len(a), float)
    r[o] = np.arange(1, len(a) + 1)
    for u in np.unique(a):
        m = a == u
        if m.sum() > 1:
            r[m] = r[m].mean()
    n1 = len(pos)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(neg)))


def axis():
    """VFG id -> 'hazard' | 'non_hazard' | 'unlabelled', from UniProt curators' keywords.

    🔴 Restricted to the 746 class-axis REPRESENTATIVES. A first version labelled every one of
    the 4,755 setA records, which left 4,527 unlabelled against the 518 the preregistration
    froze -- the frozen-coverage selftest caught it on its first run.
    """
    reps = {r["vfg"] for r in json.loads(BUILD.read_text())["positives"]}
    acc = {}
    with FAS.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith(">"):
                m = HDR.match(line)
                if m and m.group(1) in reps:
                    acc[m.group(1)] = m.group(2)
    with IDMAP.open(encoding="utf-8", errors="replace") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    by = {}
    for r in rows:
        by.setdefault(r["From"], []).append(r)

    lab, ent = {}, {}
    for vfg, a in acc.items():
        rv = [r for r in by.get(a, []) if r["Reviewed"] == "reviewed"]
        if not rv:
            lab[vfg] = "unlabelled"
            continue
        # 🔴 prefer a hazard-carrying entry when an accession maps to several, so a
        # multi-mapping cannot hide a Toxin keyword behind an isoform that lacks it
        pick = next(
            (r for r in rv if set(r["Keywords"].split(";")) & set(HAZARD_KW)), rv[0]
        )
        kws = set(pick["Keywords"].split(";")) if pick["Keywords"] else set()
        lab[vfg] = "hazard" if kws & set(HAZARD_KW) else "non_hazard"
        ent[vfg] = pick
    return lab, ent


def match_pairs(haz, non, length, species):
    """Pair each hazard-positive to the nearest-length hazard-negative, no replacement.

    Same species preferred when one is available inside MATCH_SPECIES_RATIO; pairs above
    MATCH_DROP_RATIO are dropped and counted.  Deterministic: hazard-positives are taken
    in ascending length so the result does not depend on dict order.
    """
    pool, pairs, dropped = list(non), [], []
    for h in sorted(haz, key=lambda v: length[v]):
        if not pool:
            dropped.append((h, "pool exhausted"))
            continue
        same = [c for c in pool if species[c] == species[h]
                and max(length[c], length[h]) / max(min(length[c], length[h]), 1)
                <= MATCH_SPECIES_RATIO]
        cands = same or pool
        pick = min(cands, key=lambda c: abs(length[c] - length[h]))
        ratio = max(length[pick], length[h]) / max(min(length[pick], length[h]), 1)
        if ratio > MATCH_DROP_RATIO:
            dropped.append((h, f"best ratio {ratio:.2f}"))
            continue
        pool.remove(pick)
        pairs.append((h, pick, ratio, species[pick] == species[h]))
    return pairs, dropped


def paired_boot(d, seed=0):
    """Bootstrap over PAIRS -- the unit the matching created."""
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    if not d.size:
        return (float("nan"), float("nan"))
    draws = d[rng.integers(0, d.size, size=(BOOT, d.size))].mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def verdict(val, reg, haz_floor):
    if val is None or not np.isfinite(val):
        return "unmeasured"
    if haz_floor is not None and val >= haz_floor:
        return "hazard band"
    if reg[0] <= val <= reg[1]:
        return "registration band"
    return "between bands"


def selftest():
    ok = True
    lab, ent = axis()
    n = {k: sum(1 for v in lab.values() if v == k)
         for k in ("hazard", "non_hazard", "unlabelled")}
    # 🔒 the coverage frozen in the preregistration, so a re-fetch that changes the axis fails here
    if n != {"hazard": 31, "non_hazard": 197, "unlabelled": 518}:
        print(f"  XX axis counts {n}, preregistration froze 31/197/518")
        ok = False

    # matching must be a bijection on the pairs it returns, and must respect its own ratio
    length = {f"A{i}": 100 + i for i in range(20)}
    species = {k: "sp" for k in length}
    haz, non = [f"A{i}" for i in range(5)], [f"A{i}" for i in range(5, 20)]
    pairs, dropped = match_pairs(haz, non, length, species)
    if len(pairs) != 5 or dropped:
        print(f"  XX synthetic matching gave {len(pairs)} pairs, {len(dropped)} dropped")
        ok = False
    if len({p[1] for p in pairs}) != len(pairs):
        print("  XX matching reused a negative")
        ok = False
    # a negative pool that is all far too long must be DROPPED, not silently matched
    length2 = {"h": 100, "n1": 10000, "n2": 20000}
    species2 = {k: "sp" for k in length2}
    p2, d2 = match_pairs(["h"], ["n1", "n2"], length2, species2)
    if p2 or len(d2) != 1:
        print(f"  XX a 100x length mismatch was not dropped: {p2}")
        ok = False

    # auroc: a perfect separation is 1.0, a reversal 0.0, identical 0.5
    if auroc([3, 4, 5], [0, 1, 2]) != 1.0 or auroc([0, 1, 2], [3, 4, 5]) != 0.0:
        print("  XX auroc fails on a perfect separation")
        ok = False
    if abs(auroc([1, 2], [1, 2]) - 0.5) > 1e-12:
        print("  XX auroc on identical samples != 0.5")
        ok = False

    # the bands must actually discriminate, in all three directions
    if verdict(0.50, BAND_AUROC_REG, BAND_AUROC_HAZ) != "registration band":
        print("  XX 0.50 is not in the registration band")
        ok = False
    if verdict(0.75, BAND_AUROC_REG, BAND_AUROC_HAZ) != "hazard band":
        print("  XX 0.75 is not in the hazard band")
        ok = False
    if verdict(0.65, BAND_AUROC_REG, BAND_AUROC_HAZ) != "between bands":
        print("  XX 0.65 is not between the bands")
        ok = False

    # paired_boot must widen with noise and collapse without it
    lo, hi = paired_boot([0.2] * 31)
    if not (abs(lo - 0.2) < 1e-9 and abs(hi - 0.2) < 1e-9):
        print(f"  XX paired_boot on a constant gave [{lo}, {hi}]")
        ok = False
    lo, hi = paired_boot(list(np.linspace(-1, 1, 31)))
    if not (lo < 0 < hi and hi - lo > 0.3):
        print(f"  XX paired_boot on a spread gave [{lo:.3f}, {hi:.3f}]")
        ok = False

    print("selftest PASS" if ok else "selftest FAIL")
    return 0 if ok else 1


def run_arm(arm, seeds):
    sfx = "" if arm == "esm2_650M" else f"_{arm}"
    P = np.load(RES / f"embeddings_class_axis_positives{sfx}.npy")
    man = json.loads((RES / f"embedding_manifest_class_axis_positives{sfx}.json").read_text())
    POOL = np.load(RES / f"embeddings_pool_large_{arm}.npy")
    build = json.loads(BUILD.read_text())
    screen = json.loads(SCREEN.read_text())
    union, test = screen["admitted_rows"], build["pool_partition"]["test_rows"]
    if set(union) & set(test):
        raise SystemExit("the screened union overlaps the test partition")

    rows = man["rows"]
    row_of = {v: i for i, v in enumerate(rows)}
    cats = {r["vfg"]: r["cat"] for r in build["positives"]}
    length = {r["vfg"]: r["len"] for r in build["positives"]}
    species = dict(zip(rows, man["species_of_row"]))

    lab, _ = axis()
    haz = [v for v in rows if lab.get(v) == "hazard"]
    non = [v for v in rows if lab.get(v) == "non_hazard"]
    pairs, dropped = match_pairs(haz, non, length, species)
    hz = [p[0] for p in pairs]
    mt = [p[1] for p in pairs]
    print(f"  {arm}: {len(haz)} hazard, {len(non)} non-hazard, {len(pairs)} pairs "
          f"({sum(1 for p in pairs if p[3])} same-species), {len(dropped)} dropped")

    n_tr, n_ca = N_TRAIN, N_CAL
    out = {}
    for variant in ("permissive", "strict"):
        held = set(hz) | set(mt)
        if variant == "strict":
            held |= {v for v in rows if cats.get(v) == "Exotoxin"}
        hi = np.array(sorted(row_of[v] for v in held))
        tri = np.setdiff1d(np.arange(len(rows)), hi)
        ih = np.array([row_of[v] for v in hz])
        im = np.array([row_of[v] for v in mt])

        rec_h, rec_m, au, fpr, dpair = [], [], [], [], []
        for seed in range(seeds):
            r = np.random.default_rng(seed)
            perm = r.permutation(len(union))
            tr = [union[i] for i in perm[:n_tr]]
            ca = [union[i] for i in perm[n_tr:n_tr + n_ca]]
            model = clf().fit(np.vstack([P[tri], POOL[tr]]),
                              np.r_[np.ones(len(tri)), np.zeros(len(tr))])
            t = float(np.quantile(model.predict_proba(POOL[ca])[:, 1], SPEC))
            sh = model.predict_proba(P[ih])[:, 1]
            sm = model.predict_proba(P[im])[:, 1]
            fh, fm = (sh >= t).astype(float), (sm >= t).astype(float)
            rec_h.append(float(fh.mean()))
            rec_m.append(float(fm.mean()))
            au.append(auroc(sh, sm))
            fpr.append(float((model.predict_proba(POOL[test])[:, 1] >= t).mean()))
            dpair.append(fh - fm)
        dmean = np.mean(np.array(dpair), axis=0)          # per-pair, averaged over seeds
        lo, hi_ = paired_boot(dmean)
        rec_all_non = None
        # the unmatched figure P1 is about: all 197 hazard-negatives, same fold
        ia = np.array([row_of[v] for v in non])
        un = []
        for seed in range(seeds):
            r = np.random.default_rng(seed)
            perm = r.permutation(len(union))
            tr = [union[i] for i in perm[:n_tr]]
            ca = [union[i] for i in perm[n_tr:n_tr + n_ca]]
            model = clf().fit(np.vstack([P[tri], POOL[tr]]),
                              np.r_[np.ones(len(tri)), np.zeros(len(tr))])
            t = float(np.quantile(model.predict_proba(POOL[ca])[:, 1], SPEC))
            un.append(float((model.predict_proba(P[ia])[:, 1] >= t).mean()))
        rec_all_non = float(np.mean(un))

        # ⚠️ Did the length matching shift CATEGORY mix? If the matched 31 over-weighted
        # low-recovery categories, Drecovery would be inflated and the registration reading
        # would be getting help it has not earned. Expected recovery under study AG's own
        # per-category means answers it without fitting anything.
        ag = json.loads(
            (ROOT / "results/external_class_axis_lomo.json").read_text())["recovery"]

        def expect(vs):
            return float(np.mean([ag[cats[v]]["mean"] for v in vs]))

        out[variant] = {
            "composition_check": {
                "expected_recovery_matched": expect(mt),
                "expected_recovery_all_non_hazard": expect(non),
                "expected_recovery_hazard": expect(hz),
                "matched_minus_all_non": expect(mt) - expect(non),
                "note": ("expected recovery from study AG's 650M per-category means, "
                         "composition only -- nothing is fitted. A matched set whose mix "
                         "matched the full negative set cannot be inflating Drecovery."),
            },
            "hazard_by_vfdb_category": {
                c: sum(1 for v in hz if cats[v] == c)
                for c in sorted({cats[v] for v in hz})
            },
            "matched_by_vfdb_category": {
                c: sum(1 for v in mt if cats[v] == c)
                for c in sorted({cats[v] for v in mt})
            },
            "match_quality": {
                "mean_length_ratio": float(np.mean([q[2] for q in pairs])),
                "max_length_ratio": float(max(q[2] for q in pairs)),
                "mean_len_hazard": float(np.mean([length[v] for v in hz])),
                "mean_len_matched": float(np.mean([length[v] for v in mt])),
            },
            "n_pairs": len(pairs),
            "n_same_species_pairs": sum(1 for p in pairs if p[3]),
            "n_dropped": len(dropped),
            "n_held_out_positives": int(len(hi)),
            "n_fitted_positives": int(len(tri)),
            "recovery_hazard": float(np.mean(rec_h)),
            "recovery_matched": float(np.mean(rec_m)),
            "recovery_all_non_hazard": rec_all_non,
            "d_recovery_matched": float(np.mean(rec_h) - np.mean(rec_m)),
            "d_recovery_unmatched": float(np.mean(rec_h) - rec_all_non),
            "d_recovery_pair_ci": [lo, hi_],
            "auroc_hazard_vs_matched": float(np.mean(au)),
            "auroc_sd": float(np.std(au, ddof=1)),
            "test_fpr": float(np.mean(fpr)),
            "verdict_auroc": verdict(float(np.mean(au)), BAND_AUROC_REG, BAND_AUROC_HAZ),
            "verdict_d_recovery": verdict(
                float(np.mean(rec_h) - np.mean(rec_m)), BAND_DREC_REG, BAND_DREC_HAZ),
        }
        v = out[variant]
        print(f"    {variant:<11} rec haz {v['recovery_hazard']*100:5.1f}%  "
              f"matched {v['recovery_matched']*100:5.1f}%  all-non "
              f"{v['recovery_all_non_hazard']*100:5.1f}%")
        print(f"                Drec matched {v['d_recovery_matched']*100:+6.2f}pp "
              f"[{lo*100:+.2f}, {hi_*100:+.2f}]  unmatched "
              f"{v['d_recovery_unmatched']*100:+6.2f}pp")
        print(f"                AUROC {v['auroc_hazard_vs_matched']:.4f} "
              f"(sd {v['auroc_sd']:.4f})  FPR {v['test_fpr']*100:.2f}%  "
              f"-> {v['verdict_auroc']} / {v['verdict_d_recovery']}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seeds", type=int, default=SEEDS)
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    t0 = time.time()
    arms = {}
    for arm in ("esm2_650M", "esm2_35M"):
        arms[arm] = run_arm(arm, a.seeds)
    A, B = arms["esm2_650M"], arms["esm2_35M"]

    def both(key, field):
        return A[key][field], B[key][field]

    res = {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "study": "AH -- matched toxin-inclusive contrast on a third-party hazard axis",
        "preregistration": "docs/EXTERNAL_HAZARD_AXIS_PREREGISTRATION.md",
        "axis": "UniProt Swiss-Prot keywords: " + ", ".join(HAZARD_KW),
        "seeds": a.seeds,
        "bands": {"auroc_registration": list(BAND_AUROC_REG),
                  "auroc_hazard_floor": BAND_AUROC_HAZ,
                  "d_recovery_registration": list(BAND_DREC_REG),
                  "d_recovery_hazard_floor": BAND_DREC_HAZ},
        "arms": arms,
        "second_arm_rule": {
            "auroc_same_band_strict":
                A["strict"]["verdict_auroc"] == B["strict"]["verdict_auroc"],
            "d_recovery_same_band_strict":
                A["strict"]["verdict_d_recovery"] == B["strict"]["verdict_d_recovery"],
            "auroc_same_band_permissive":
                A["permissive"]["verdict_auroc"] == B["permissive"]["verdict_auroc"],
        },
        "predictions": {
            "P1_unmatched_positive_over_10pp": {
                "kind": "derived",
                "values": [A["strict"]["d_recovery_unmatched"],
                           B["strict"]["d_recovery_unmatched"]],
                "held": bool(A["strict"]["d_recovery_unmatched"] > 0.10
                             and B["strict"]["d_recovery_unmatched"] > 0.10)},
            "P2_matched_inside_registration_band": {
                "kind": "derived",
                "values": [A["strict"]["d_recovery_matched"],
                           B["strict"]["d_recovery_matched"],
                           A["strict"]["auroc_hazard_vs_matched"],
                           B["strict"]["auroc_hazard_vs_matched"]],
                "held": bool(A["strict"]["verdict_d_recovery"] == "registration band"
                             and B["strict"]["verdict_d_recovery"] == "registration band"
                             and A["strict"]["verdict_auroc"] == "registration band"
                             and B["strict"]["verdict_auroc"] == "registration band")},
            "P3_strict_auroc_below_permissive": {
                "kind": "reasoned",
                "values": list(both("strict", "auroc_hazard_vs_matched"))
                + list(both("permissive", "auroc_hazard_vs_matched")),
                "held": bool(
                    A["strict"]["auroc_hazard_vs_matched"]
                    < A["permissive"]["auroc_hazard_vs_matched"]
                    and B["strict"]["auroc_hazard_vs_matched"]
                    < B["permissive"]["auroc_hazard_vs_matched"])},
        },
        "seconds": round(time.time() - t0, 1),
    }
    out = Path(f"{OUT_STEM}.json")
    out.write_text(json.dumps(res, indent=2) + "\n")
    print("\nsecond arm:", json.dumps(res["second_arm_rule"]))
    for k, v in res["predictions"].items():
        print(f"  {k:<42} {v['kind']:<9} held={v['held']}")
    print(f"wrote {out.relative_to(ROOT)} in {res['seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
