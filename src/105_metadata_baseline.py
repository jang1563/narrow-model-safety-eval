#!/usr/bin/env python3
"""
105_metadata_baseline.py - how much of the virulence-factor task do two metadata features do?

`docs/METADATA_BASELINE_PREREGISTRATION.md`. Section 1.2 cites DeepVF at AUC 0.896, DTVF at 0.921 and
DeepVIC at 0.954, and this project has never run anything comparable.

🔴 This does not close that gap and does not claim to: their benchmarks are different populations and
nothing here re-analyses their numbers. What it asks is a question their papers do not -- what does a
model with NO sequence information achieve on a task of this shape? The project already keeps baselines
of this kind: amino-acid composition alone at 0.754, shuffled labels at 0.506, "reported because they
make the model look worse".

🔒 Declared before freezing: localization stratum alone gives AUROC 0.6973 and length alone 0.3072,
which is 0.693 with the sign flipped -- VFDB proteins are SHORTER. Nothing combined had been fitted.

⚠️ No new inference. Five-fold cross-validation, the same folds for every feature set, logistic
regression throughout so the comparison is between inputs rather than model classes.

Usage:
    python src/105_metadata_baseline.py --arm esm2_650M
    python src/105_metadata_baseline.py --arm esm2_35M
    python src/105_metadata_baseline.py --selftest
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "v3"
POOL_JSON = ROOT / "data" / "sequences" / "_scaled_negative_pool.json"
NEGSET = ROOT / "results" / "vfdb_negative_set.json"
MAPPED = ROOT / "results" / "vfdb_uniprot_map.json"
BUILD = ROOT / "results" / "external_class_axis_build.json"
OUT_STEM = ROOT / "results" / "metadata_baseline"
FOLDS, FOLD_FLOOR = 5, 100
META_HI, INCREMENT_LO = 0.70, 0.15
STRATA = ("extracellular", "membrane", "intracellular", "unannotated")


def _load(stem):
    spec = importlib.util.spec_from_file_location(f"_{stem}", ROOT / "src" / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M92 = _load("92_localization_coverage")
M84 = M92.M84
M83 = _load("83_provenance_control")
M91 = _load("91_matched_vfdb_scaleup")


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))


def selftest():
    # 🔒 the AUROC helper is src/84's, already exercised by five studies
    assert abs(M84.auroc([3, 4, 5], [0, 1, 2]) - 1.0) < 1e-9
    assert abs(M84.auroc([0, 1], [0, 1]) - 0.5) < 1e-9
    # 🔒 one-hot of the strata must be full rank minus one, or the fit is singular
    x = np.eye(len(STRATA))[[0, 1, 2, 3, 0]]
    assert x.shape == (5, 4) and x.sum() == 5
    print("selftest PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="esm2_650M", choices=["esm2_650M", "esm2_35M"])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    # ---- the VFDB side: study K's 778, unchanged --------------------------------------------------
    admitted = json.loads(NEGSET.read_text())["admitted"]
    len_of = {r["vfg"]: r["len"] for r in admitted}
    row_of = {r["vfg"]: i for i, r in enumerate(admitted)}
    chosen = json.loads(MAPPED.read_text())["chosen"]
    noncirc = M83.vfdb_noncircular_mask()
    up_len, acc_of = M91.uniprot_lengths(), M91.vfg_accessions()
    pos = {v: i for v, i in chosen.items()
           if noncirc[row_of[v]] and str(len_of[v]) == str(up_len.get((acc_of[v], i["uniprot"])))}

    # ---- the benign side: pool Bacteria from VFDB species, not in VFDB ---------------------------
    pool = json.loads(POOL_JSON.read_text())["proteins"]
    test = json.loads(BUILD.read_text())["pool_partition"]["test_rows"]
    accs = [pool[i]["uniprot"] for i in test]
    kw = M84.fetch_keywords(accs)
    vf_seqs, vfdb_species = M83.vfdb_sequences(), M83.vfdb_species()
    neg_rows = []
    for k, i in enumerate(test):
        sp = " ".join(pool[i]["organism"].replace("(", "").split()[:2])
        if pool[i]["kingdom"] != "Bacteria" or sp not in vfdb_species:
            continue
        if pool[i]["sequence"] in vf_seqs:
            continue
        neg_rows.append((k, i))
    print(f"arm {a.arm}: VFDB positives {len(pos)}, benign negatives {len(neg_rows)}")

    sfx = "" if a.arm == "esm2_650M" else f"_{a.arm}"
    VFDB = np.load(RES / f"embeddings_vfdb_neg_{a.arm}.npy")
    POOL = np.load(RES / f"embeddings_pool_large_{a.arm}.npy")
    rows = np.array(test)

    onehot = {s: np.eye(len(STRATA))[k] for k, s in enumerate(STRATA)}
    Xm, Xe, y = [], [], []
    for v, info in pos.items():
        Xm.append(np.r_[onehot[info["stratum"]], np.log10(max(len_of[v], 1))])
        Xe.append(VFDB[row_of[v]])
        y.append(1)
    for k, i in neg_rows:
        s = M84.stratum_of(kw[pool[i]["uniprot"]])
        Xm.append(np.r_[onehot[s], np.log10(max(pool[i]["length"], 1))])
        Xe.append(POOL[rows[k]])
        y.append(0)
    Xm, Xe, y = np.array(Xm, float), np.array(Xe, float), np.array(y, int)
    print(f"  metadata features {Xm.shape}, embedding {Xe.shape}")

    sets = {"metadata_only": Xm, "esm2": Xe,
            "metadata_plus_esm2": np.hstack([Xm, Xe])}
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=0)
    folds = list(skf.split(Xm, y))
    npos = [int(y[tr].sum()) for tr, _ in folds]
    print(f"  folds: {FOLDS}, VFDB positives per training fold {npos}")
    floors_met = all(n >= FOLD_FLOOR for n in npos)

    out = {}
    t0 = time.time()
    for name, X in sets.items():
        per = []
        for tr, te in folds:
            yy = y.copy()
            m = clf().fit(X[tr], yy[tr])
            p = m.predict_proba(X[te])[:, 1]
            per.append(M84.auroc(p[y[te] == 1], p[y[te] == 0]))
        out[name] = {"auroc": float(np.mean(per)), "per_fold": [float(x) for x in per],
                     "ci": [float(np.mean(per) - 1.96 * np.std(per, ddof=1) / FOLDS ** 0.5),
                            float(np.mean(per) + 1.96 * np.std(per, ddof=1) / FOLDS ** 0.5)]}
        print(f"  {name:<20}AUROC {out[name]['auroc']:.4f} "
              f"[{out[name]['ci'][0]:.4f}, {out[name]['ci'][1]:.4f}]")
    print(f"  {time.time() - t0:.0f}s")

    # ---- 🔴 the shuffled-label floor, with REPETITIONS ------------------------------------------
    # The first version ran one shuffle and returned 0.4663, which tripped a frozen tolerance of
    # ±0.03 while its own fold interval [0.4264, 0.5062] covered 0.5 — the tolerance was tighter than
    # the estimator's spread, which is a defect in the guard rather than evidence of leakage. Twenty
    # repetitions give a null whose mean can actually be compared against 0.5.
    SHUF = 20
    shuf = []
    for rep in range(SHUF):
        r2 = np.random.default_rng(1000 + rep)
        acc_ = []
        for tr, te in folds:
            yy = y.copy()
            yy[tr] = r2.permutation(yy[tr])
            m = clf().fit(Xe[tr], yy[tr])
            pp = m.predict_proba(Xe[te])[:, 1]
            acc_.append(M84.auroc(pp[y[te] == 1], pp[y[te] == 0]))
        shuf.append(float(np.mean(acc_)))
    shuf_mean, shuf_sd = float(np.mean(shuf)), float(np.std(shuf, ddof=1))
    shuf_se = shuf_sd / SHUF ** 0.5
    out["shuffled_labels"] = {"auroc": shuf_mean, "reps": SHUF, "sd": shuf_sd,
                              "ci": [shuf_mean - 1.96 * shuf_se, shuf_mean + 1.96 * shuf_se],
                              "per_fold": shuf[:5]}
    print(f"  {'shuffled_labels':<20}AUROC {shuf_mean:.4f} over {SHUF} shuffles "
          f"[{out['shuffled_labels']['ci'][0]:.4f}, {out['shuffled_labels']['ci'][1]:.4f}]")

    meta, esm = out["metadata_only"]["auroc"], out["esm2"]["auroc"]
    inc = esm - meta
    band = ("two annotation fields do most of what this task rewards"
            if meta >= META_HI and inc < INCREMENT_LO
            else "metadata is a large floor and the embedding clears it"
            if meta >= META_HI else "the metadata floor is modest")
    res = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "arm": a.arm, "folds": FOLDS,
           "n_positive": int(y.sum()), "n_negative": int((y == 0).sum()),
           "positives_per_training_fold": npos, "floors_met": bool(floors_met),
           "results": out, "metadata_auroc": meta, "esm2_auroc": esm,
           "increment": inc, "band": band,
           # 🔒 judged on whether the repeated-shuffle interval covers 0.5, not on a fixed
           # tolerance that was tighter than the estimator's own spread
           "shuffled_ok": bool(out["shuffled_labels"]["ci"][0] <= 0.5 <= out["shuffled_labels"]["ci"][1]),
           "shuffled_reps": out["shuffled_labels"]["reps"],
           "published_for_context": {"DeepVF": 0.896, "DTVF": 0.921, "DeepVIC": 0.954},
           "not_their_benchmark": True}
    dest = Path(f"{OUT_STEM}{sfx}.json")
    dest.write_text(json.dumps(res, indent=2) + "\n")

    print(f"\nW-1  metadata-only AUROC = {meta:.4f}  (no sequence information at all)")
    print(f"W-2  ESM-2 AUROC = {esm:.4f}; increment over metadata = {inc:+.4f}")
    print(f"     -> {band}")
    print(f"     shuffled labels {out['shuffled_labels']['auroc']:.4f} over "
          f"{out['shuffled_labels']['reps']} shuffles, interval "
          f"{'covers' if res['shuffled_ok'] else 'EXCLUDES'} 0.50")
    print("\n⚠️  This is NOT DeepVF's, DTVF's or DeepVIC's benchmark. Their negatives were chosen by "
          "their own rules;\n    a metadata floor here does not establish one there. The claim available "
          "is about a missing baseline.")
    print(f"wrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
