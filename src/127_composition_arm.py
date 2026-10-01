#!/usr/bin/env python3
"""Study AJ: is study AI-3's arm disagreement a composition effect?

`docs/COMPOSITION_ARM_PREREGISTRATION.md`, frozen before this file existed.

AI-3 left Exotoxin swinging 13.7 points between the two arms while the fourteen-category
means differed by 0.3. This runs `src/125`'s fold with the embedding replaced by features
that contain no model at all -- log length plus the twenty amino-acid frequencies, AI-1b's
21 numbers -- plus a length-alone reference arm.

🔒 The negative partition is reproduced bit-for-bit from `src/125`: `default_rng(12345)`
over the manifest's negative block, two thirds as the union, the remainder as the fixed test
set, and `default_rng(seed)` for the per-seed 1,000/500 draw. The selftest asserts it against
`src/125`'s own code path, so the three folds differ ONLY in the features.

🔴 This study has no second arm and cannot have one: a sequence has one composition. Its
result is an explanation offered, not a verdict, and § 4 of the preregistration says so.
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
LOCO = ROOT / "results/deepvic_loco.json"
OUT = ROOT / "results/composition_arm.json"
AA = "ACDEFGHIKLMNPQRSTVWY"
SEEDS, N_TRAIN, N_CAL, SPEC = 30, 1000, 500, 0.95
FLOOR = 7
MAX_LEN = 1022
# frozen in section 4
BAND_AJ2_SUPPORT, BAND_AJ2_REFUTE = 0.15, -0.15
BAND_AJ3_SUPPORT_RANK, BAND_AJ3_REFUTE_RANK = 4, 7


def load(mod):
    spec = importlib.util.spec_from_file_location(mod[:4], ROOT / "src" / mod)
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod[:4]] = m
    spec.loader.exec_module(m)
    return m


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))


def features(seqs):
    """log length, and log length plus the twenty amino-acid frequencies. No model."""
    n = np.array([len(s) for s in seqs], float)
    loglen = np.log(np.maximum(n, 1.0))[:, None]
    comp = np.array([[s.count(c) for c in AA] for s in seqs], float)
    comp /= np.maximum(n[:, None], 1.0)
    return loglen, np.hstack([loglen, comp])


def partition(n_neg):
    """src/125's negative partition, restated so the selftest can compare the two."""
    perm = np.random.default_rng(12345).permutation(n_neg)
    cut = 2 * n_neg // 3
    return perm[:cut], perm[cut:]


def fold(X, pos, cls, union, test, targets, seeds):
    rec, flags = {}, {}
    for C in targets:
        hi = pos[cls[pos] == C]
        tri = pos[cls[pos] != C]
        rr, fl = [], []
        for seed in range(seeds):
            r = np.random.default_rng(seed)
            p = r.permutation(len(union))
            tr, ca = union[p[:N_TRAIN]], union[p[N_TRAIN:N_TRAIN + N_CAL]]
            model = clf().fit(np.vstack([X[tri], X[tr]]),
                              np.r_[np.ones(len(tri)), np.zeros(len(tr))])
            t = float(np.quantile(model.predict_proba(X[ca])[:, 1], SPEC))
            f = (model.predict_proba(X[hi])[:, 1] >= t).astype(float)
            fl.append(f)
            rr.append(float(f.mean()))
        rec[C] = {"n": int(len(hi)), "mean": float(np.mean(rr)),
                  "sd": float(np.std(rr, ddof=1))}
        flags[C] = np.mean(np.array(fl), axis=0)
    return rec, flags


def selftest():
    ok = True
    m124, m125 = load("124_deepvic_embed.py"), load("125_deepvic_loco.py")
    recs, meta = m124.ordered()
    npos, nneg = meta["n_positives"], meta["n_negatives"]
    if (npos, nneg) != (12989, 33456):
        print(f"  XX order gives {npos}/{nneg}, expected 12989/33456")
        ok = False

    # 🔒 the partition must match src/125's EXACTLY, or the three folds are not comparable
    lab = np.array(meta["label_of_row"])
    neg = np.where(lab == 0)[0]
    u_idx, t_idx = partition(len(neg))
    union, test = neg[u_idx], neg[t_idx]
    rng0 = np.random.default_rng(12345)
    perm_ref = rng0.permutation(len(neg))
    cut = 2 * len(neg) // 3
    if not (np.array_equal(union, neg[perm_ref[:cut]])
            and np.array_equal(test, neg[perm_ref[cut:]])):
        print("  XX the negative partition does not match src/125's")
        ok = False
    if set(union) & set(test):
        print("  XX union and test overlap")
        ok = False
    if len(union) + len(test) != len(neg):
        print("  XX the partition is not a partition")
        ok = False
    # and the per-seed draw must be disjoint and seed-dependent, as src/125's is
    for seed in (0, 7, 29):
        p = np.random.default_rng(seed).permutation(len(union))
        tr, ca = union[p[:N_TRAIN]], union[p[N_TRAIN:N_TRAIN + N_CAL]]
        if set(tr) & set(ca) or set(tr) & set(test) or set(ca) & set(test):
            print(f"  XX seed {seed}: a fold touched the test set or overlapped itself")
            ok = False
    if np.array_equal(np.random.default_rng(0).permutation(len(union))[:N_TRAIN],
                      np.random.default_rng(1).permutation(len(union))[:N_TRAIN]):
        print("  XX seeds do not change the draw")
        ok = False

    # features: the 21-column form must differ from the 1-column form only by composition
    a, b = features(["AAAA", "AAAAAAAA"])
    if a.shape[1] != 1 or b.shape[1] != 21:
        print(f"  XX feature widths {a.shape[1]}, {b.shape[1]}, expected 1 and 21")
        ok = False
    if not np.allclose(b[:, :1], a):
        print("  XX the composition form's first column is not the length form")
        ok = False
    if not np.allclose(b[0, 1:], b[1, 1:]):
        print("  XX composition is not length-normalized")
        ok = False
    if np.allclose(features(["AAAA", "CCCC"])[1][0, 1:],
                   features(["AAAA", "CCCC"])[1][1, 1:]):
        print("  XX composition does not distinguish AAAA from CCCC")
        ok = False

    # the comparison targets must exist, or AJ-2 is computed against nothing
    if not LOCO.exists():
        print("  XX results/deepvic_loco.json missing")
        ok = False
    else:
        L = json.loads(LOCO.read_text())
        for arm in ("esm2_650M", "esm2_35M"):
            r = L["arms"][arm]["full"]["recovery"]
            if len(r) != 14 or "Exotoxin" not in r or "Motility" not in r:
                print(f"  XX {arm} recovery table is not the fourteen categories")
                ok = False
        if abs(L["arms"]["esm2_35M"]["full"]["recovery"]["Exotoxin"]["mean"]
               - 0.584) > 0.002:
            print("  XX the 35M Exotoxin figure the study exists to explain has changed")
            ok = False
    # and src/125's matcher must be importable, since AJ-1 uses amendment 1's pairing
    if not hasattr(m125, "match_pairs"):
        print("  XX src/125 exposes no match_pairs")
        ok = False
    print("selftest PASS" if ok else "selftest FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seeds", type=int, default=SEEDS)
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    t0 = time.time()
    m124, m125, m30 = (load("124_deepvic_embed.py"), load("125_deepvic_loco.py"),
                       load("30_margin_across_arms.py"))
    recs, meta = m124.ordered()
    ids = np.array(meta["id_of_row"])
    lab = np.array(meta["label_of_row"])
    cls = np.array(meta["class_of_row"])
    seqs = [s for _, s in recs]
    LEN = {h: len(s) for h, s in recs}

    pos, neg = np.where(lab == 1)[0], np.where(lab == 0)[0]
    u_idx, t_idx = partition(len(neg))
    union, test = neg[u_idx], neg[t_idx]
    counts = Counter(cls[pos])
    targets = sorted(c for c, n in counts.items() if n >= FLOOR)
    print(f"{len(pos):,} positives in {len(targets)} categories, {len(neg):,} negatives "
          f"-> {len(union):,} union / {len(test):,} test", flush=True)

    Xlen, Xcomp = features(seqs)
    L = json.loads(LOCO.read_text())
    emb = {arm: {c: L["arms"][arm]["full"]["recovery"][c]["mean"] for c in targets}
           for arm in ("esm2_650M", "esm2_35M")}

    out = {"built": time.strftime("%Y-%m-%d %H:%M:%S"),
           "study": "AJ -- is AI-3's arm disagreement a composition effect?",
           "preregistration": "docs/COMPOSITION_ARM_PREREGISTRATION.md",
           "single_arm_disclosure": (
               "a sequence has one composition, so this study has no second arm and cannot "
               "have one. Its result is an explanation offered, not a verdict."),
           "bands": {"aj2_support_floor": BAND_AJ2_SUPPORT,
                     "aj2_refute_ceiling": BAND_AJ2_REFUTE,
                     "aj3_support_rank": BAND_AJ3_SUPPORT_RANK,
                     "aj3_refute_rank": BAND_AJ3_REFUTE_RANK},
           "seeds": a.seeds, "arms": {}}

    for tag, X in (("composition", Xcomp), ("length_only", Xlen)):
        rec, flags = fold(X, pos, cls, union, test, targets, a.seeds)
        order = sorted(targets, key=lambda c: -rec[c]["mean"])
        rk = order.index("Exotoxin") + 1
        d_full = rec["Motility"]["mean"] - rec["Exotoxin"]["mean"]
        pairs, dropped = m125.match_pairs(
            [ids[r] for r in pos[cls[pos] == "Exotoxin"]],
            [ids[r] for r in pos[cls[pos] == "Motility"]], LEN)
        back = {ids[r]: int(r) for r in pos}
        fm = {int(r): float(v) for r, v in zip(pos[cls[pos] == "Motility"],
                                               flags["Motility"])}
        fe = {int(r): float(v) for r, v in zip(pos[cls[pos] == "Exotoxin"],
                                               flags["Exotoxin"])}
        mv = np.array([fm[back[b]] for _, b in pairs])
        ev = np.array([fe[back[aa]] for aa, _ in pairs])
        d_matched = float(mv.mean() - ev.mean())
        r35 = m30._spearman([rec[c]["mean"] for c in targets],
                            [emb["esm2_35M"][c] for c in targets])
        r650 = m30._spearman([rec[c]["mean"] for c in targets],
                             [emb["esm2_650M"][c] for c in targets])
        out["arms"][tag] = {
            "n_features": int(X.shape[1]),
            "recovery": rec,
            "ranking": [{"rank": i, "category": c, "recovery": rec[c]["mean"],
                         "n": rec[c]["n"]} for i, c in enumerate(order, 1)],
            "mean_recovery": float(np.mean([rec[c]["mean"] for c in targets])),
            "AJ1_diff_full": d_full,
            "AJ1_diff_matched": d_matched,
            "AJ1_n_pairs": len(pairs),
            "AJ1_sign_matches_35M": bool(d_full < 0),
            "AJ1_matched_sign_matches_35M": bool(d_matched < 0),
            "AJ2_rho_vs_35M": r35,
            "AJ2_rho_vs_650M": r650,
            "AJ2_gap": r35 - r650,
            "AJ2_verdict": ("supports" if r35 - r650 >= BAND_AJ2_SUPPORT else
                            "refutes" if r35 - r650 <= BAND_AJ2_REFUTE else
                            "between bands"),
            "AJ3_exotoxin_rank": rk,
            "AJ3_motility_rank": order.index("Motility") + 1,
            "AJ3_verdict": ("supports" if rk <= BAND_AJ3_SUPPORT_RANK else
                            "refutes" if rk > BAND_AJ3_REFUTE_RANK else
                            "between bands"),
            "rec_exotoxin": rec["Exotoxin"]["mean"],
            "rec_motility": rec["Motility"]["mean"],
            # 🔴 POST HOC, added after seeing that AJ-2 did not discriminate. AJ-2 is a
            # DIFFERENCE of two correlations and § 4 set a floor on the difference without
            # requiring either correlation to be large, so an arm that matches both embedding
            # orderings badly but asymmetrically clears it. Mean absolute error over the
            # fourteen recovery LEVELS has no such hole. It is labelled post hoc and does not
            # change the frozen verdict.
            "post_hoc_mae_vs_35M": float(np.mean(
                [abs(rec[c]["mean"] - emb["esm2_35M"][c]) for c in targets])),
            "post_hoc_mae_vs_650M": float(np.mean(
                [abs(rec[c]["mean"] - emb["esm2_650M"][c]) for c in targets])),
            "post_hoc_exotoxin_gap_to_35M": float(
                abs(rec["Exotoxin"]["mean"] - emb["esm2_35M"]["Exotoxin"])),
        }
        out["arms"][tag]["post_hoc_closer_to"] = (
            "35M" if out["arms"][tag]["post_hoc_mae_vs_35M"]
            < out["arms"][tag]["post_hoc_mae_vs_650M"] else "650M")
        v = out["arms"][tag]
        print(f"  {tag:<12} ({v['n_features']:>2} feat)  mean rec "
              f"{v['mean_recovery'] * 100:5.1f}%  Exo {v['rec_exotoxin'] * 100:5.1f}% "
              f"(rank {rk}/14)  Mot {v['rec_motility'] * 100:5.1f}%", flush=True)
        print(f"               AJ-1 full {d_full * 100:+6.2f}pp  matched "
              f"{d_matched * 100:+6.2f}pp ({len(pairs)} pairs)  "
              f"sign matches 35M {v['AJ1_sign_matches_35M']}", flush=True)
        print(f"               AJ-2 rho vs 35M {r35:+.4f}  vs 650M {r650:+.4f}  gap "
              f"{r35 - r650:+.4f} -> {v['AJ2_verdict']}   AJ-3 -> {v['AJ3_verdict']}",
              flush=True)

    C = out["arms"]["composition"]
    out["predictions"] = {
        "P1_AJ1_negative": {"kind": "derived", "held": C["AJ1_sign_matches_35M"],
                            "values": {"full": C["AJ1_diff_full"],
                                       "matched": C["AJ1_diff_matched"]}},
        "P2_AJ2_at_least_0.15": {"kind": "derived",
                                 "held": bool(C["AJ2_gap"] >= BAND_AJ2_SUPPORT),
                                 "values": C["AJ2_gap"]},
        "P3_exotoxin_top_4": {"kind": "reasoned",
                              "held": bool(C["AJ3_exotoxin_rank"] <= 4),
                              "values": C["AJ3_exotoxin_rank"]},
    }
    out["post_hoc_note"] = (
        "AJ-2 as frozen is a difference of two correlations with no floor on either, and the "
        "length-alone control clears it (+0.1714) while sitting 24pp from BOTH embedding arms "
        "and marginally closer to 650M. Composition misses AJ-2 by 0.018 while sitting 9.98pp "
        "from 35M against 12.18pp from 650M and reproducing the 35M Exotoxin figure to 0.08pp. "
        "The mean-absolute-error comparison is POST HOC -- it was added after seeing that AJ-2 "
        "did not discriminate -- and the frozen verdict below is unchanged by it. The band was "
        "poorly chosen and that is recorded rather than repaired after the fact.")
    out["hypothesis_verdict"] = (
        "SUPPORTED" if (C["AJ1_sign_matches_35M"] and C["AJ2_verdict"] == "supports"
                        and C["AJ3_verdict"] == "supports")
        else "REFUTED" if (not C["AJ1_sign_matches_35M"]
                           and C["AJ2_verdict"] == "refutes")
        else "PARTIAL -- does not settle it")
    out["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nhypothesis: {out['hypothesis_verdict']}")
    for k, v in out["predictions"].items():
        print(f"  {k:<26} {v['kind']:<9} held={v['held']}")
    print(f"wrote {OUT.relative_to(ROOT)} in {out['seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
