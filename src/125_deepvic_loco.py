#!/usr/bin/env python3
"""Study AI-2/AI-3/AI-4: leave-one-category-out on DeepVIC's published class axis.

`docs/DEEPVIC_BENCHMARK_PREREGISTRATION.md` § 3, bands § 4, predictions P3/P4/P5 § 5.

DeepVIC carries a 14-category axis over 12,989 annotated VFs and runs no leave-one-category-out
evaluation.  This runs `src/81`'s fold on it: fit on the other thirteen classes plus train
negatives, threshold at the 95th percentile of the calibrate negatives, report the held-out
class's recovery and the false-positive rate on test negatives no fold has seen.

  AI-2  Spearman(margin, recovery) over the 14 classes, and whether recovery is bimodal
  AI-3  recovery(Motility) - recovery(Exotoxin), two classes within 0.5% in size (1,121 / 1,126)
  AI-4  every figure again with the 239 representatives shared with this project removed

Margin is `src/30`'s, imported rather than restated, exactly as `src/81` does it.
"""
import argparse
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results/v3"
OUT_STEM = ROOT / "results/deepvic_loco"
SEEDS, N_TRAIN, N_CAL, SPEC = 30, 1000, 500, 0.95
PERMS, BOOT = 20000, 20000
FLOOR = 7
ALPHA = 0.05 / 4
# frozen in section 4
BAND_RHO_REPLICATE, BAND_RHO_FAIL, BAND_RHO_INVERT = 0.60, 0.20, -0.40
BAND_AI3_REPLICATE, BAND_AI3_HAZARD = 0.0, -0.20
ARMS = {"esm2_650M": "", "esm2_35M": "_esm2_35M"}


def load30():
    spec = importlib.util.spec_from_file_location(
        "s30", ROOT / "src" / "30_margin_across_arms.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["s30"] = m
    spec.loader.exec_module(m)
    return m


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))


# 🔴 This file carried its own Spearman for one revision. It used argsort(argsort(x)), which
# breaks ties arbitrarily, and it disagreed with src/30's on the first tied vector the selftest
# fed it: -0.600 against -0.789. src/30's uses average ranks and is the one every other result
# here was computed with, so there is no local definition any more -- `_rho` IS src/30's.
def _rho(x, y):
    return load30()._spearman(x, y)


def perm_p(x, y, perms=PERMS, seed=0):
    """Two-sided permutation p for Spearman. The observed statistic is included."""
    rng = np.random.default_rng(seed)
    m30 = load30()
    obs = abs(m30._spearman(x, y))
    y = np.asarray(y, float)
    hits = 1
    for _ in range(perms):
        if abs(m30._spearman(x, rng.permutation(y))) >= obs - 1e-12:
            hits += 1
    return hits / (perms + 1)


def bimodality(rec):
    """Gap statistic: the largest jump in the sorted recovery vector, and where it falls.

    🔴 Not a test. A number that says how far the panel is from evenly spread, reported next
    to the sorted values so a reader can see the shape rather than trust a label.
    """
    v = np.sort(np.asarray(rec, float))
    gaps = np.diff(v)
    k = int(gaps.argmax())
    return {"sorted": [float(x) for x in v],
            "largest_gap": float(gaps[k]),
            "gap_after_rank": k + 1,
            "spread": float(v[-1] - v[0]),
            "gap_over_spread": float(gaps[k] / (v[-1] - v[0])) if v[-1] > v[0] else 0.0}


def boot_diff(a, b, seed=0):
    """Protein-level bootstrap on a difference of two flag rates."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = np.empty(BOOT)
    for i in range(BOOT):
        d[i] = (a[rng.integers(0, a.size, a.size)].mean()
                - b[rng.integers(0, b.size, b.size)].mean())
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def verdict_rho(r):
    # 🔴 src/30's _spearman is np.corrcoef over average ranks, so a zero-variance input makes
    # it nan. Every comparison against nan is False, so without this line a nan would fall
    # through to "between bands" and be reported as a measured in-between result.
    if not np.isfinite(r):
        return "unmeasured"
    if r <= BAND_RHO_INVERT:
        return "inverted"
    if r >= BAND_RHO_REPLICATE:
        return "replicates"
    if r <= BAND_RHO_FAIL:
        return "fails to replicate"
    return "between bands"


def verdict_ai3(d):
    if not np.isfinite(d):
        return "unmeasured"
    if d <= BAND_AI3_HAZARD:
        return "hazard-selective"
    if d >= BAND_AI3_REPLICATE:
        return "replicates the narrowing"
    return "between bands"


def selftest():
    ok = True
    # 🔒 rho is src/30's everywhere in this file. These assert that the imported definition
    # behaves, and that it handles TIES by average rank -- the property a local argsort-based
    # helper got wrong here and which the untied cases below cannot detect.
    if abs(_rho([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) > 1e-12:
        print("  XX rho monotone != +1")
        ok = False
    if abs(_rho([1, 2, 3, 4], [40, 30, 20, 10]) + 1.0) > 1e-12:
        print("  XX rho reversed != -1")
        ok = False
    # ⚠️ With x fully tied there is no variance and src/30's np.corrcoef returns NAN, not 0.
    # 🔴 The first version of this check read `abs(_rho(...)) > 1e-12`, which a nan satisfies
    # vacuously -- abs(nan) > 1e-12 is False -- so it passed while measuring nothing. The real
    # behaviour is asserted, and so is the fact that a nan cannot reach a band.
    tied = _rho([1, 1, 1, 1], [4, 3, 2, 1])
    if not np.isnan(tied):
        print(f"  XX rho on a fully tied x is {tied}, expected nan from np.corrcoef")
        ok = False
    if verdict_rho(float("nan")) != "unmeasured":
        print(f"  XX a nan rho reports as '{verdict_rho(float('nan'))}', not 'unmeasured'")
        ok = False
    if verdict_ai3(float("nan")) != "unmeasured":
        print("  XX a nan AI-3 difference is not reported as 'unmeasured'")
        ok = False
    if abs(_rho([3, 1, 4, 1, 5], [2, 7, 1, 8, 2]) + 0.7894736842105264) > 1e-9:
        print(f"  XX rho on a tied pair is {_rho([3, 1, 4, 1, 5], [2, 7, 1, 8, 2])}")
        ok = False
    # perm_p must be small for a real relationship and large for none
    x = list(range(14))
    if perm_p(x, x, perms=2000) > 0.01:
        print("  XX perm_p on a perfect relationship is not small")
        ok = False
    rng = np.random.default_rng(1)
    if perm_p(x, list(rng.permutation(14)), perms=2000) < 0.05:
        print("  XX perm_p on a shuffled relationship is small")
        ok = False
    # bimodality must find a planted gap and must not invent one
    b = bimodality([0.1, 0.12, 0.15, 0.80, 0.82, 0.85])
    if b["gap_after_rank"] != 3 or b["gap_over_spread"] < 0.8:
        print(f"  XX bimodality missed a planted gap: {b}")
        ok = False
    b = bimodality(list(np.linspace(0, 1, 14)))
    if b["gap_over_spread"] > 0.15:
        print(f"  XX bimodality invents a gap in an even spread: {b}")
        ok = False
    # boot_diff must exclude zero on a planted difference and include it otherwise
    lo, hi = boot_diff([1.0] * 200, [0.0] * 200)
    if not (lo > 0.9):
        print(f"  XX boot_diff planted [{lo:.3f}, {hi:.3f}]")
        ok = False
    lo, hi = boot_diff([1.0, 0.0] * 100, [1.0, 0.0] * 100)
    if not (lo < 0 < hi):
        print(f"  XX boot_diff null [{lo:.3f}, {hi:.3f}] must span 0")
        ok = False
    # the bands must discriminate in every direction they define
    for v, want in ((0.7, "replicates"), (0.1, "fails to replicate"),
                    (-0.5, "inverted"), (0.4, "between bands")):
        if verdict_rho(v) != want:
            print(f"  XX rho {v} is not '{want}'")
            ok = False
    for v, want in ((0.05, "replicates the narrowing"), (-0.30, "hazard-selective"),
                    (-0.10, "between bands")):
        if verdict_ai3(v) != want:
            print(f"  XX AI-3 {v} is not '{want}'")
            ok = False
    print("selftest PASS" if ok else "selftest FAIL")
    return 0 if ok else 1


def run_arm(arm, seeds, drop_shared):
    sfx = ARMS[arm]
    E = np.load(RES / f"embeddings_deepvic{sfx}.npy")
    man = json.loads((RES / f"embedding_manifest_deepvic{sfx}.json").read_text())
    lab = np.array(man["label_of_row"])
    cls = np.array(man["class_of_row"])
    shared = np.array(man["shared_with_our_panel"])
    if E.shape[0] != len(lab):
        raise SystemExit("embedding and manifest disagree on row count")

    keep = np.ones(len(lab), bool)
    if drop_shared:
        keep &= ~shared
    pos = np.where((lab == 1) & keep)[0]
    neg = np.where(lab == 0)[0]

    # negatives: a seeded 1,000 / 500 draw per seed from the first two thirds, with the
    # remainder held as a fixed test partition no fold ever touches
    rng0 = np.random.default_rng(12345)
    perm = rng0.permutation(len(neg))
    cut = 2 * len(neg) // 3
    union, test = neg[perm[:cut]], neg[perm[cut:]]

    counts = Counter(cls[pos])
    targets = sorted(c for c, n in counts.items() if n >= FLOOR)
    below = sorted(c for c, n in counts.items() if n < FLOOR)

    rec, fpr, flags = {}, {}, {}
    for C in targets:
        hi = pos[cls[pos] == C]
        tri = pos[cls[pos] != C]
        rr, ff, fl = [], [], []
        for seed in range(seeds):
            r = np.random.default_rng(seed)
            p = r.permutation(len(union))
            tr, ca = union[p[:N_TRAIN]], union[p[N_TRAIN:N_TRAIN + N_CAL]]
            model = clf().fit(np.vstack([E[tri], E[tr]]),
                              np.r_[np.ones(len(tri)), np.zeros(len(tr))])
            t = float(np.quantile(model.predict_proba(E[ca])[:, 1], SPEC))
            f = (model.predict_proba(E[hi])[:, 1] >= t).astype(float)
            fl.append(f)
            rr.append(float(f.mean()))
            ff.append(float((model.predict_proba(E[test])[:, 1] >= t).mean()))
        rec[C] = {"n": int(len(hi)), "mean": float(np.mean(rr)),
                  "sd": float(np.std(rr, ddof=1)),
                  "zero_seeds": int(sum(1 for x in rr if x == 0))}
        fpr[C] = {"mean": float(np.mean(ff)), "sd": float(np.std(ff, ddof=1))}
        flags[C] = np.mean(np.array(fl), axis=0)
        print(f"    {C:<46}n={len(hi):>5}  rec {np.mean(rr) * 100:5.1f}% "
              f"(sd {np.std(rr, ddof=1) * 100:4.1f})  FPR {np.mean(ff) * 100:5.2f}%",
              flush=True)

    # ---- margin, src/81's lines 201-208 verbatim in definition, chunked in execution ----
    # 🔴 src/81 forms cos(P, P) and cos(P, POOL[union]) whole. Here P is 12,989 rows and the
    # union is ~22,300, so those are 1.4 GB and 2.3 GB in float64. The DEFINITION is unchanged
    # -- max similarity to a non-class positive minus max similarity to a union negative,
    # averaged over the class -- and only the evaluation is done in row chunks.
    m30 = load30()
    P, NEGU = E[pos], E[union]
    pcls = cls[pos]
    maxpos_other = np.empty(len(pos))
    maxneg = np.empty(len(pos))
    CH = 512
    for i in range(0, len(pos), CH):
        sl = slice(i, min(i + CH, len(pos)))
        spp = m30.cos(P[sl], P)
        # self-similarity and same-class members are excluded, as src/81's fill_diagonal
        # plus its np.ix_(idx, other) indexing do jointly
        for r in range(spp.shape[0]):
            spp[r, i + r] = -np.inf
        spp[pcls[sl][:, None] == pcls[None, :]] = -np.inf
        maxpos_other[sl] = spp.max(1)
        maxneg[sl] = m30.cos(P[sl], NEGU).max(1)
    margin, nnpos, nnneg = {}, {}, {}
    for C in targets:
        m = pcls == C
        margin[C] = float((maxpos_other[m] - maxneg[m]).mean())
        nnpos[C] = float(maxpos_other[m].mean())
        nnneg[C] = float(maxneg[m].mean())

    order = targets
    recv = [rec[c]["mean"] for c in order]
    rho = m30._spearman([margin[c] for c in order], recv)
    p = perm_p([margin[c] for c in order], recv)
    # 🔒 the two controls src/81 reports: nearest-positive alone and nearest-negative alone
    rho_pos = m30._spearman([nnpos[c] for c in order], recv)
    rho_neg = m30._spearman([nnneg[c] for c in order], recv)

    out = {
        "arm": arm, "drop_shared": drop_shared, "seeds": seeds,
        "n_positives_used": int(len(pos)), "n_negatives": int(len(neg)),
        "n_union": int(len(union)), "n_test": int(len(test)),
        "n_targets": len(targets), "below_floor": below,
        "recovery": rec, "test_fpr": fpr, "margin": margin,
        "nearest_positive": nnpos, "nearest_negative": nnneg,
        "AI2": {"rho": rho, "perm_p": p, "alpha": ALPHA,
                "rho_nearest_positive_alone": rho_pos,
                "rho_nearest_negative_alone": rho_neg,
                "margin_beats_both_controls": bool(abs(rho) > abs(rho_pos)
                                                   and abs(rho) > abs(rho_neg)),
                "significant": bool(p < ALPHA), "verdict": verdict_rho(rho),
                "bimodality": bimodality([rec[c]["mean"] for c in order])},
    }
    if "Motility" in rec and "Exotoxin" in rec:
        d = rec["Motility"]["mean"] - rec["Exotoxin"]["mean"]
        lo, hi_ = boot_diff(flags["Motility"], flags["Exotoxin"])
        out["AI3"] = {"n_motility": rec["Motility"]["n"],
                      "n_exotoxin": rec["Exotoxin"]["n"],
                      "rec_motility": rec["Motility"]["mean"],
                      "rec_exotoxin": rec["Exotoxin"]["mean"],
                      "diff": d, "ci": [lo, hi_],
                      "excludes_zero": bool(lo > 0 or hi_ < 0),
                      "verdict": verdict_ai3(d)}
        print(f"    AI-3 Motility {rec['Motility']['mean'] * 100:.1f}% - Exotoxin "
              f"{rec['Exotoxin']['mean'] * 100:.1f}% = {d * 100:+.2f}pp "
              f"[{lo * 100:+.2f}, {hi_ * 100:+.2f}] -> {out['AI3']['verdict']}", flush=True)
    print(f"    AI-2 rho {rho:+.4f} p {p:.5f} -> {out['AI2']['verdict']}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--arms", default="esm2_35M")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    t0 = time.time()
    res = {"built": time.strftime("%Y-%m-%d %H:%M:%S"),
           "study": "AI-2/AI-3/AI-4 -- LOCO on DeepVIC's published class axis",
           "preregistration": "docs/DEEPVIC_BENCHMARK_PREREGISTRATION.md",
           "bands": {"rho_replicate_floor": BAND_RHO_REPLICATE,
                     "rho_fail_ceiling": BAND_RHO_FAIL,
                     "rho_invert": BAND_RHO_INVERT,
                     "ai3_replicate_floor": BAND_AI3_REPLICATE,
                     "ai3_hazard_ceiling": BAND_AI3_HAZARD},
           "arms": {}}
    for arm in a.arms.split(","):
        res["arms"][arm] = {}
        for drop in (False, True):
            key = "shared_removed" if drop else "full"
            print(f"  {arm} / {key}", flush=True)
            res["arms"][arm][key] = run_arm(arm, a.seeds, drop)
    res["seconds"] = round(time.time() - t0, 1)
    out = Path(f"{OUT_STEM}.json")
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(f"wrote {out.relative_to(ROOT)} in {res['seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
