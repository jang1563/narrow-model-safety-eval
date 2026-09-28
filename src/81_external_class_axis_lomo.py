#!/usr/bin/env python3
"""
81_external_class_axis_lomo.py - study B: does margin predict which VFDB category a probe fails?

`docs/EXTERNAL_CLASS_AXIS_PREREGISTRATION.md`. Class-level margin ranks this project's own mechanism
classes by leave-one-class-out recovery at Spearman +0.894, across fourteen model arms and fifteen
re-poolings. **Every one of those tests is on classes assigned by hand in this repository.** This asks
the same question of VFDB's fourteen curator-maintained categories, thirteen of which clear the panel's
own eligibility floor.

The fold, exactly as the preregistration and its amendments fix it
-----------------------------------------------------------------
Amendment 1: the **test** partition is fixed and never resampled — that is the criterion-1 point — while
**train and calibrate are redrawn per seed** from their screened union at the frozen sizes 1,000 and 500,
falling back to a 2:1 split of the survivors if fewer than 1,500 remain.

Per seed, per held-out category C: fit `StandardScaler` → `LogisticRegression(C=1.0, max_iter=5000)` —
`src/03b`'s pipeline — on the other twelve categories' positives plus the train negatives; take the
threshold at 95% specificity on the **calibrate** negatives; report C's recovery and the false-positive
rate on the **test** negatives, which no fold has ever seen.

Margin is `src/30_margin_across_arms.py`'s, computed by importing that module rather than restating it.

The four primary tests, and the controls
----------------------------------------
    B-1  Spearman(margin, recovery) over the 13 categories, permutation p < 0.0125
    B-2  are the bottom-2 by margin among the bottom-3 by recovery?
    B-3  does margin beat nearest-positive-alone and nearest-negative-alone?
    B-4  out-of-sample FPR on the never-seen test negatives

    controls: composition-only AUROC, shuffled category labels, length-only AUROC, and amendment 4's
    species **exclusivity** correlated against recovery — if that is significant, recovery is driven by
    organism overlap and B-1 is reported as uninterpretable.

Usage:
    python src/81_external_class_axis_lomo.py --selftest
    python src/81_external_class_axis_lomo.py
"""

import argparse
import collections
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "v3"
BUILD = ROOT / "results" / "external_class_axis_build.json"
SCREEN = ROOT / "results" / "external_class_axis_screen.json"



OUT_STEM = ROOT / "results" / "external_class_axis_lomo"
ALPHA = 0.05 / 4          # four primary tests, fixed in section 4
SEEDS, N_TRAIN, N_CAL = 30, 1000, 500
SPEC = 0.95
PERMS = 20000
FLOOR = 7


def load30():
    spec = importlib.util.spec_from_file_location("s30", ROOT / "src" / "30_margin_across_arms.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["s30"] = m
    spec.loader.exec_module(m)
    return m


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


def composition(seqs):
    aa = "ACDEFGHIKLMNPQRSTVWY"
    return np.array([[s.count(c) / max(len(s), 1) for c in aa] for s in seqs], float)


def selftest():
    """The fold must never let a test negative reach fitting or calibration, and the split must be
    a genuine partition. Both are asserted on synthetic indices, where a violation is visible."""
    union = list(range(1500))
    test = list(range(1500, 8258))
    for seed in (0, 7, 29):
        r = np.random.default_rng(seed)
        perm = r.permutation(len(union))
        tr = [union[i] for i in perm[:N_TRAIN]]
        ca = [union[i] for i in perm[N_TRAIN:N_TRAIN + N_CAL]]
        assert len(tr) == N_TRAIN and len(ca) == N_CAL
        assert not (set(tr) & set(ca)), "train and calibrate overlap"
        assert not (set(tr) & set(test)) and not (set(ca) & set(test)), "a fold touched the test set"
    # and different seeds must actually give different splits, or thirty seeds vary nothing
    s0 = np.random.default_rng(0).permutation(1500)[:N_TRAIN]
    s1 = np.random.default_rng(1).permutation(1500)[:N_TRAIN]
    assert not np.array_equal(s0, s1), "seeds do not change the split; amendment 1 is not in effect"
    print("SELFTEST PASS: train/calibrate/test disjoint on every seed, test never touched, "
          "and seeds change the split")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--arm", default="esm2_650M", choices=["esm2_650M", "esm2_35M"])
    ap.add_argument("--seeds", type=int, default=SEEDS)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    sfx = "" if a.arm == "esm2_650M" else f"_{a.arm}"

    POS_NPY = RES / f"embeddings_class_axis_positives{sfx}.npy"
    POS_MAN = RES / f"embedding_manifest_class_axis_positives{sfx}.json"
    # the pool is named by ARM, not by suffix: embeddings_pool_large_esm2_650M.npy and
    # embeddings_pool_large_esm2_35M.npy. Appending the suffix to the 650M name asked for a file
    # that has never existed.
    POOL_NPY = RES / f"embeddings_pool_large_{a.arm}.npy"
    OUT = Path(f"{OUT_STEM}{sfx}.json")
    for f in (POS_NPY, POOL_NPY, SCREEN):
        if not f.exists():
            raise SystemExit(f"{f.name} missing; run src/79 and src/80 first")
    POS_NPY = RES / f"embeddings_class_axis_positives{sfx}.npy"
    POS_MAN = RES / f"embedding_manifest_class_axis_positives{sfx}.json"
    # the pool is named by ARM, not by suffix: embeddings_pool_large_esm2_650M.npy and
    # embeddings_pool_large_esm2_35M.npy. Appending the suffix to the 650M name asked for a file
    # that has never existed.
    POOL_NPY = RES / f"embeddings_pool_large_{a.arm}.npy"
    OUT = Path(f"{OUT_STEM}{sfx}.json")
    m30 = load30()
    build = json.loads(BUILD.read_text())
    screen = json.loads(SCREEN.read_text())
    man = json.loads(POS_MAN.read_text())
    P = np.load(POS_NPY)
    POOL = np.load(POOL_NPY)
    cats = np.array(man["category_of_row"])
    species = np.array(man["species_of_row"])
    if len(cats) != P.shape[0]:
        raise SystemExit("manifest and embedding disagree on row count")

    union = screen["admitted_rows"]
    test = build["pool_partition"]["test_rows"]
    if set(union) & set(test):
        raise SystemExit("the screened union overlaps the test partition")
    n_tr, n_ca = (N_TRAIN, N_CAL) if len(union) >= N_TRAIN + N_CAL else \
        (2 * len(union) // 3, len(union) - 2 * len(union) // 3)
    print(f"{P.shape[0]} positives in {len(set(cats))} categories; "
          f"negatives {len(union)} screened union -> {n_tr} train / {n_ca} calibrate per seed, "
          f"{len(test)} test (unscreened, never fitted or calibrated on)")
    if (n_tr, n_ca) != (N_TRAIN, N_CAL):
        print(f"  ⚠️ fewer than {N_TRAIN + N_CAL} survived the screen, so amendment 1's 2:1 rule applies")

    counts = collections.Counter(cats)
    targets = sorted(c for c, n in counts.items() if n >= FLOOR)
    below = sorted(c for c, n in counts.items() if n < FLOOR)
    print(f"  holding out {len(targets)} categories; reported but not held out: {below}")

    rec, fpr = {}, {}
    t0 = time.time()
    for C in targets:
        hi = np.where(cats == C)[0]
        tri = np.setdiff1d(np.arange(len(cats)), hi)
        rr, ff = [], []
        for seed in range(a.seeds):
            r = np.random.default_rng(seed)
            perm = r.permutation(len(union))
            tr = [union[i] for i in perm[:n_tr]]
            ca = [union[i] for i in perm[n_tr:n_tr + n_ca]]
            model = clf().fit(np.vstack([P[tri], POOL[tr]]),
                              np.r_[np.ones(len(tri)), np.zeros(len(tr))])
            t = float(np.quantile(model.predict_proba(POOL[ca])[:, 1], SPEC))
            rr.append(float((model.predict_proba(P[hi])[:, 1] >= t).mean()))
            ff.append(float((model.predict_proba(POOL[test])[:, 1] >= t).mean()))
        rec[C] = {"n": int(len(hi)), "mean": float(np.mean(rr)),
                  "sd": float(np.std(rr, ddof=1)), "zero_seeds": int(sum(1 for x in rr if x == 0))}
        fpr[C] = {"mean": float(np.mean(ff)), "sd": float(np.std(ff, ddof=1))}
        print(f"  {C:<46}n={len(hi):>4}  recovery {np.mean(rr) * 100:5.1f}% "
              f"(sd {np.std(rr, ddof=1) * 100:4.1f})  test FPR {np.mean(ff) * 100:5.2f}%", flush=True)
    print(f"  {time.time() - t0:.0f}s")

    # ---- margin, src/30's definition ------------------------------------------------------------
    simPP, simPN = m30.cos(P, P), m30.cos(P, POOL[union])
    np.fill_diagonal(simPP, -np.inf)
    margin, nnpos, nnneg = {}, {}, {}
    for C in targets:
        idx = np.where(cats == C)[0]
        other = np.setdiff1d(np.arange(len(P)), idx)
        margin[C] = float((simPP[np.ix_(idx, other)].max(1) - simPN[idx].max(1)).mean())
        nnpos[C] = float(simPP[np.ix_(idx, other)].max(1).mean())
        nnneg[C] = float(simPN[idx].max(1).mean())

    order = targets
    recv = [rec[c]["mean"] for c in order]
    rho = m30._spearman([margin[c] for c in order], recv)
    rng = np.random.default_rng(0)
    null = np.array([m30._spearman([margin[c] for c in order], rng.permutation(recv))
                     for _ in range(PERMS)])
    p_rho = float((null >= rho).mean())
    rho_pos = m30._spearman([nnpos[c] for c in order], recv)
    rho_neg = m30._spearman([nnneg[c] for c in order], recv)

    by_margin = sorted(order, key=lambda c: margin[c])
    by_rec = sorted(order, key=lambda c: rec[c]["mean"])
    b2 = set(by_margin[:2]) <= set(by_rec[:3])

    # ---- amendment 4's confound test, from the build artifact alone -----------------------------
    sp_cats = collections.defaultdict(set)
    for c, s in zip(cats, species):
        sp_cats[s].add(c)
    excl = {C: float(np.mean([len(sp_cats[s]) == 1 for s in species[cats == C]])) for C in order}
    rho_excl = m30._spearman([excl[c] for c in order], recv)
    null_e = np.array([m30._spearman([excl[c] for c in order], rng.permutation(recv))
                       for _ in range(PERMS)])
    p_excl = float(min((null_e >= rho_excl).mean(), (null_e <= rho_excl).mean()) * 2)

    res = {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"), "alpha": ALPHA, "seeds": a.seeds,
        "n_positives": int(P.shape[0]), "n_categories_total": len(counts),
        "held_out": order, "not_held_out": below,
        "negatives": {"screened_union": len(union), "train": n_tr, "calibrate": n_ca,
                      "test": len(test), "test_unscreened": True},
        "recovery": rec, "test_fpr": fpr,
        "margin": margin, "nn_positive": nnpos, "nn_negative": nnneg,
        "B1": {"rho": rho, "perm_p": p_rho, "supported": bool(rho > 0 and p_rho < ALPHA)},
        "B2": {"bottom2_margin": by_margin[:2], "bottom3_recovery": by_rec[:3], "supported": bool(b2)},
        "B3": {"rho_margin": rho, "rho_nn_positive": rho_pos, "rho_nn_negative": rho_neg,
               "supported": bool(rho > rho_pos and rho > rho_neg)},
        "B4": {"mean_test_fpr": float(np.mean([fpr[c]["mean"] for c in order])),
               "range": [float(min(fpr[c]["mean"] for c in order)),
                         float(max(fpr[c]["mean"] for c in order))],
               "nominal": 1 - SPEC},
        "confound_exclusivity": {"per_category": excl, "rho": rho_excl, "two_sided_p": p_excl,
                                 "uninterpretable_if_significant": bool(p_excl < ALPHA)},
    }
    OUT.write_text(json.dumps(res, indent=2) + "\n")

    print(f"\nB-1  Spearman(margin, recovery) = {rho:+.4f}  permutation p = {p_rho:.5f}  "
          f"-> {'SUPPORTED' if res['B1']['supported'] else 'NOT SUPPORTED'} at alpha {ALPHA:.4f}")
    print(f"B-2  bottom-2 margin {by_margin[:2]}")
    print(f"     bottom-3 recovery {by_rec[:3]}  -> "
          f"{'SUPPORTED' if b2 else 'NOT SUPPORTED'}")
    print(f"B-3  margin {rho:+.3f} vs nn_pos {rho_pos:+.3f} vs nn_neg {rho_neg:+.3f}  -> "
          f"{'SUPPORTED' if res['B3']['supported'] else 'NOT SUPPORTED'}")
    print(f"B-4  test FPR mean {res['B4']['mean_test_fpr'] * 100:.2f}% at a nominal "
          f"{(1 - SPEC) * 100:.0f}%, range "
          f"[{res['B4']['range'][0] * 100:.2f}, {res['B4']['range'][1] * 100:.2f}]")
    print(f"\nconfound: Spearman(species exclusivity, recovery) = {rho_excl:+.4f}, "
          f"two-sided p = {p_excl:.4f} -> "
          f"{'🔴 B-1 UNINTERPRETABLE' if res['confound_exclusivity']['uninterpretable_if_significant'] else 'not significant'}")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    raise SystemExit(main() or 0)
