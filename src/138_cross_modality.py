#!/usr/bin/env python3
"""Study AS: does the margin result hold outside proteins?

`docs/CROSS_MODALITY_PREREGISTRATION.md`, frozen before this file existed.

    margin(C) = mean over members of C of [ max cos to a positive outside C
                                            - max cos to any negative ]

Nothing in that formula mentions proteins. This runs src/125's fold on 20 Newsgroups with a
frozen sentence encoder: 20 classes held out one at a time, negatives from the same corpus's
other split, both the fixed C=1.0 and src/133's nested-selected C.

🔒 margin is src/30's `cos` and `_spearman`, imported. The fold's sizes and threshold rule are
src/133's constants, imported. Nothing about the method is restated here -- only the data.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "results/v3/text20ng_minilm.npz"
OUT = ROOT / "results/cross_modality.json"
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
SEEDS, FLOOR = 30, 7
PERMS = 20000
BAND_OK, BAND_NOT = 0.60, 0.20
ALPHA = 0.05 / 2


def load(mod):
    spec = importlib.util.spec_from_file_location(mod[:4], ROOT / "src" / mod)
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod[:4]] = m
    spec.loader.exec_module(m)
    return m


POS_TOP = ("comp", "rec", "sci")
NEG_TOP = ("alt", "misc", "soc", "talk")


def build():
    """Embed 20 Newsgroups once and cache, using the corpus's own hierarchy.

    🔴 The first version took positives from the train split and negatives from the test
    split, which made them the SAME KIND OF THING -- the probe was separating train-split
    posts from test-split posts, a null task, and the held-out recoveries came out at 0.8 to
    1.4%, below the 5% nominal. Amendment 1 records it.

    🔒 The fixed design mirrors the protein fold: a genuine positive/negative axis, with
    sub-classes inside the positive side playing the role of mechanism categories. Positives
    are comp/rec/sci (13 sub-classes); negatives are alt/misc/soc/talk (7 groups). Both splits
    of the corpus are pooled, since the split is no longer doing any work.
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.datasets import fetch_20newsgroups
    rm = ("headers", "footers", "quotes")
    docs, names = [], None
    for sub in ("train", "test"):
        d = fetch_20newsgroups(subset=sub, remove=rm, random_state=0)
        names = d.target_names
        docs += [(t, names[y]) for t, y in zip(d.data, d.target)
                 if len(t.strip()) >= 40]
    pos = [(t, g) for t, g in docs if g.split(".")[0] in POS_TOP]
    neg = [(t, g) for t, g in docs if g.split(".")[0] in NEG_TOP]
    pos_txt = [t for t, _ in pos]
    pos_y = [g for _, g in pos]
    neg_txt = [t for t, _ in neg]
    print(f"  {len(pos_txt)} positives in {len(set(pos_y))} classes "
          f"({'/'.join(POS_TOP)}), {len(neg_txt)} negatives "
          f"({'/'.join(NEG_TOP)})", flush=True)
    enc = SentenceTransformer(ENCODER)
    P = enc.encode(pos_txt, batch_size=256, show_progress_bar=False,
                   convert_to_numpy=True, normalize_embeddings=False)
    N = enc.encode(neg_txt, batch_size=256, show_progress_bar=False,
                   convert_to_numpy=True, normalize_embeddings=False)
    np.savez_compressed(CACHE, P=P.astype(np.float32), N=N.astype(np.float32),
                        y=np.array(pos_y))
    print(f"  cached {CACHE.name}: P {P.shape} N {N.shape}", flush=True)


def selftest():
    ok = True
    m30, m133 = load("30_margin_across_arms.py"), load("133_estimation_repair.py")
    for fn in ("cos", "_spearman"):
        if not hasattr(m30, fn):
            print(f"  XX src/30 exposes no {fn}")
            ok = False
    for fn in ("select_C", "clf", "N_TRAIN", "N_CAL", "SPEC", "C_GRID"):
        if not hasattr(m133, fn):
            print(f"  XX src/133 exposes no {fn}")
            ok = False
    if (BAND_OK, BAND_NOT) != (0.60, 0.20):
        print(f"  XX bands are {(BAND_OK, BAND_NOT)}, frozen at (0.60, 0.20)")
        ok = False
    # 🔒 the band floor must be AI-2's, so the two domains are judged by one criterion
    L = ROOT / "results/deepvic_loco.json"
    if L.exists():
        b = json.loads(L.read_text())["bands"]["rho_replicate_floor"]
        if b != BAND_OK:
            print(f"  XX AI-2's floor is {b}, this study uses {BAND_OK}")
            ok = False
    # the margin definition must behave: a class far from the negatives and near the other
    # positives must score above one that is the reverse
    rng = np.random.default_rng(0)
    A = rng.normal(size=(40, 16)) + 5.0      # near the other positives
    B = rng.normal(size=(40, 16)) + 5.0
    Cc = rng.normal(size=(40, 16)) - 5.0     # off on its own, near the negatives
    NEG = rng.normal(size=(200, 16)) - 5.0
    Pm = np.vstack([A, B, Cc])
    ycls = np.array(["a"] * 40 + ["b"] * 40 + ["c"] * 40)
    mar = margins(Pm, NEG, ycls, m30)
    if not (mar["a"] > mar["c"] and mar["b"] > mar["c"]):
        print(f"  XX margin does not order a planted geometry: {mar}")
        ok = False
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        if z["P"].shape[1] != 384:
            print(f"  XX cached embedding dim {z['P'].shape[1]}, MiniLM is 384")
            ok = False
        ncl = len(set(z["y"].tolist()))
        if ncl != 13:
            print(f"  XX cached data has {ncl} classes, expected 13 (comp/rec/sci)")
            ok = False
        tops = {c.split(".")[0] for c in set(z["y"].tolist())}
        if tops - set(POS_TOP):
            print(f"  XX positive classes include non-positive tops: {tops}")
            ok = False
    print("selftest PASS" if ok else "selftest FAIL")
    return 0 if ok else 1


def margins(P, N, y, m30):
    """src/81's margin, in src/30's cos, evaluated in row chunks."""
    cats = sorted(set(y.tolist()))
    mx_other = np.empty(len(P))
    mx_neg = np.empty(len(P))
    CH = 1024
    for i in range(0, len(P), CH):
        sl = slice(i, min(i + CH, len(P)))
        spp = m30.cos(P[sl], P)
        for r in range(spp.shape[0]):
            spp[r, i + r] = -np.inf
        spp[y[sl][:, None] == y[None, :]] = -np.inf
        mx_other[sl] = spp.max(1)
        mx_neg[sl] = m30.cos(P[sl], N).max(1)
    return {c: float((mx_other[y == c] - mx_neg[y == c]).mean()) for c in cats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--seeds", type=int, default=SEEDS)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.build or not CACHE.exists():
        build()
        if a.build:
            return 0

    t0 = time.time()
    m30, m133 = load("30_margin_across_arms.py"), load("133_estimation_repair.py")
    z = np.load(CACHE, allow_pickle=True)
    P, N, y = z["P"].astype(np.float64), z["N"].astype(np.float64), z["y"]
    X = np.vstack([P, N])
    pos = np.arange(len(P))
    neg = np.arange(len(P), len(P) + len(N))
    perm = np.random.default_rng(12345).permutation(len(neg))
    cut = 2 * len(neg) // 3
    union, test = neg[perm[:cut]], neg[perm[cut:]]
    cats = sorted(set(y.tolist()))
    print(f"{len(P)} positives in {len(cats)} classes, {len(N)} negatives -> "
          f"{len(union)} union / {len(test)} test", flush=True)

    mar = margins(P, N, y, m30)
    out = {"built": time.strftime("%Y-%m-%d %H:%M:%S"),
           "study": "AS -- does the margin result hold outside proteins?",
           "preregistration": "docs/CROSS_MODALITY_PREREGISTRATION.md",
           "domain": "text", "dataset": "20 Newsgroups, headers/footers/quotes removed",
           "encoder": ENCODER, "dim": int(P.shape[1]),
           "single_arm": ("one text encoder is one arm, so this is single-arm indicative "
                          "however it falls; a second encoder is the named follow-up"),
           "n_positives": int(len(P)), "n_negatives": int(len(N)),
           "n_classes": len(cats), "seeds": a.seeds,
           "bands": {"ok": BAND_OK, "not": BAND_NOT, "alpha": ALPHA},
           "margin": mar, "settings": {}}

    for setting in ("fixed", "swept"):
        rec, fpr, chosen = {}, [], {}
        for ci, cat in enumerate(cats):
            hi = pos[y == cat]
            tri = pos[y != cat]
            if len(hi) < FLOOR:
                continue
            cv = 1.0
            if setting == "swept":
                cv, _ = m133.select_C(X, tri, union, ci)
            chosen[cat] = cv
            rr, ff = [], []
            for seed in range(a.seeds):
                r = np.random.default_rng(seed)
                p = r.permutation(len(union))
                tr = union[p[:m133.N_TRAIN]]
                ca = union[p[m133.N_TRAIN:m133.N_TRAIN + m133.N_CAL]]
                mdl = m133.clf(cv).fit(np.vstack([X[tri], X[tr]]),
                                       np.r_[np.ones(len(tri)), np.zeros(len(tr))])
                t = float(np.quantile(mdl.predict_proba(X[ca])[:, 1], m133.SPEC))
                rr.append(float((mdl.predict_proba(X[hi])[:, 1] >= t).mean()))
                ff.append(float((mdl.predict_proba(X[test])[:, 1] >= t).mean()))
            rec[cat] = {"n": int(len(hi)), "mean": float(np.mean(rr)),
                        "sd": float(np.std(rr, ddof=1))}
            fpr.append(float(np.mean(ff)))
            print(f"    {setting:<6} {cat:<28}{rec[cat]['mean']*100:6.2f}%  "
                  f"n={len(hi):<5} C={cv:g}  margin {mar[cat]:+.5f}", flush=True)

        ks = sorted(rec)
        rho = m30._spearman([mar[c] for c in ks], [rec[c]["mean"] for c in ks])
        rng = np.random.default_rng(0)
        rv = np.array([rec[c]["mean"] for c in ks])
        null = np.array([abs(m30._spearman([mar[c] for c in ks], rng.permutation(rv)))
                         for _ in range(PERMS)])
        pval = float((1 + (null >= abs(rho) - 1e-12).sum()) / (PERMS + 1))
        mean_rec = float(np.mean([rec[c]["mean"] for c in ks]))
        out["settings"][setting] = {
            "AS0_mean_recovery_over_20pct": bool(mean_rec > 0.20),
            "recovery": rec, "chosen_C": chosen,
            "distinct_C": sorted(set(chosen.values())),
            "mean_recovery": float(np.mean([rec[c]["mean"] for c in ks])),
            "AS1_rho": rho, "AS2_perm_p": pval,
            "significant": bool(pval < ALPHA),
            "n_classes_scored": len(ks),
            "test_fpr": {"min": float(min(fpr)), "max": float(max(fpr))},
            "verdict": ("generalizes" if rho >= BAND_OK
                        else "does not" if rho <= BAND_NOT else "between bands"),
        }
        v = out["settings"][setting]
        print(f"  {setting}: rho {rho:+.4f}  p {pval:.5f}  mean "
              f"{v['mean_recovery']*100:.2f}%  FPR {v['test_fpr']['min']*100:.2f}-"
              f"{v['test_fpr']['max']*100:.2f}%  -> {v['verdict']}", flush=True)

    F, S = out["settings"]["fixed"], out["settings"]["swept"]
    out["tuning_gain"] = S["mean_recovery"] - F["mean_recovery"]
    # 🔒 amendment 1's precondition gates the whole reading: below 20% mean recovery the fold
    # is close to null and a rank correlation over thirteen near-zero numbers is noise.
    out["AS0_precondition_met"] = F["AS0_mean_recovery_over_20pct"]
    out["verdict"] = (
        "UNINTERPRETABLE -- the fold is close to null (AS-0)"
        if not F["AS0_mean_recovery_over_20pct"]
        else "generalizes" if F["verdict"] == S["verdict"] == "generalizes"
        else "does not" if "does not" in (F["verdict"], S["verdict"])
        else "between bands")
    out["predictions"] = {
        "P1_rho_clears_0.60_both_settings": {
            "kind": "derived",
            "held": bool(F["AS1_rho"] >= BAND_OK and S["AS1_rho"] >= BAND_OK),
            "values": {"fixed": F["AS1_rho"], "swept": S["AS1_rho"]}},
        "P2_tuning_gain_smaller_than_esm2": {
            "kind": "derived", "held": bool(out["tuning_gain"] < 0.1396),
            "values": {"text": out["tuning_gain"], "esm2_650M": 0.1396}},
        "P3_withdrawn_by_amendment_1": {
            "kind": "withdrawn",
            "held": None,
            "note": ("the confusable-pair reasoning was about all twenty newsgroups, and both "
                     "members of the religion pair are now on the NEGATIVE side, so the "
                     "prediction has no referent. Withdrawn, not rewritten; counts as "
                     "neither held nor missed."),
            "values": sorted(F["recovery"], key=lambda c: F["recovery"][c]["mean"])[:3]},
        "P4_lowest_margin_classes_are_the_sibling_pairs": {
            "kind": "derived",
            "held": bool(any(c in ("comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware",
                                   "rec.sport.baseball", "rec.sport.hockey")
                             for c in sorted(out["margin"], key=out["margin"].get)[:3])),
            "values": {"lowest_margin": sorted(out["margin"],
                                               key=out["margin"].get)[:3]}},
    }
    out["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nverdict: {out['verdict']}   tuning gain {out['tuning_gain']*100:+.2f}pp")
    # 🔴 This read `P3_failures_are_the_confusable_pairs` until 2026-10-06 -- the name the
    # prediction had BEFORE amendment 1 withdrew it. The rename reached the registry and the
    # artifact and missed this one consumer, so the run raised KeyError AFTER writing the
    # artifact: the numbers were complete and the exit code was 1. Nothing downstream read the
    # exit code, which is how a crashed run looked like a finished one.
    low = sorted(F["recovery"], key=lambda c: F["recovery"][c]["mean"])[:3]
    print(f"bottom three by recovery: {low}")
    print(f"wrote {OUT.relative_to(ROOT)} in {out['seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
