#!/usr/bin/env python3
"""
101_independent_pool_replication.py - do the headline findings hold on a pool they have never seen?

`docs/INDEPENDENT_POOL_PREREGISTRATION.md`. Eleven analysis scripts share one union and one test
partition, so studies E through Q are re-analyses of one fit on one population. This refits the whole
procedure on pool 2 -- same query, same filters, same per-organism cap, disjoint accessions -- and
re-measures the two headline quantities.

S-1 is the adverse finding: the localization ratio R, study L's strata.
S-2 is the positive finding: C/A, study K's 778 VFDB proteins over pool 2's benign extracellular cell.

🔒 One declared procedural difference. Pool 1's 1,500 partition rows were screened against the 746
class-axis positives by pairwise alignment (src/79), which rejected 32 of them. Running that screen on
pool 2 is 1.1M alignments. Instead this measures what the screen was worth on POOL 1 -- refitting there
with all 1,500 rows instead of the 1,468 admitted -- and reports it beside the pool-2 result, so the
difference is quantified rather than assumed away.

Usage:
    python src/101_independent_pool_replication.py --arm esm2_35M
    python src/101_independent_pool_replication.py --selftest
"""

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "v3"
SEQ = ROOT / "data" / "sequences"
POOL1 = SEQ / "_scaled_negative_pool.json"
POOL2 = SEQ / "_independent_pool.json"
BUILD = ROOT / "results" / "external_class_axis_build.json"
SCREEN = ROOT / "results" / "external_class_axis_screen.json"
MAPPED = ROOT / "results" / "vfdb_uniprot_map.json"
NEGSET = ROOT / "results" / "vfdb_negative_set.json"
OUT_STEM = ROOT / "results" / "independent_pool_replication"
SEEDS, SPEC = 30, 0.95
N_TRAIN, N_CAL = 1000, 500
FLOOR = 300
R_LO, R_HI, CA_LO, CA_HI = 1.5, 3.0, 1.2, 2.0


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


def fold_sizes(n):
    return (N_TRAIN, N_CAL) if n >= N_TRAIN + N_CAL else (2 * n // 3, n - 2 * n // 3)


def partition(proteins):
    """Pool 1's rule, verbatim: sha256(accession) ascending, 1,000 train / 500 calibrate / rest test."""
    order = sorted(range(len(proteins)),
                   key=lambda i: hashlib.sha256(proteins[i]["uniprot"].encode()).hexdigest())
    return order[:N_TRAIN + N_CAL], order[N_TRAIN + N_CAL:]


def stat(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return {"mean": float(v.mean()),
            "ci": [float(v.mean() - 1.96 * v.std(ddof=1) / len(v) ** 0.5),
                   float(v.mean() + 1.96 * v.std(ddof=1) / len(v) ** 0.5)]}


def measure(P, POOL, VFDB, union, test_idx, strat, vfdb_rows, seeds):
    """Refit `seeds` folds and return the strata rates, R, and C/A against the VFDB cell."""
    n_tr, n_ca = fold_sizes(len(union))
    y = np.r_[np.ones(P.shape[0]), np.zeros(n_tr)]
    rows = np.array(test_idx)
    acc = {k: [] for k in ("extracellular", "intracellular", "vfdb_extra")}
    for seed in range(seeds):
        r = np.random.default_rng(seed)
        perm = r.permutation(len(union))
        tr = [union[i] for i in perm[:n_tr]]
        ca = [union[i] for i in perm[n_tr:n_tr + n_ca]]
        m = clf().fit(np.vstack([P, POOL[tr]]), y)
        t = float(np.quantile(m.predict_proba(POOL[ca])[:, 1], SPEC))
        flag = m.predict_proba(POOL[rows])[:, 1] >= t
        for s in ("extracellular", "intracellular"):
            acc[s].append(float(flag[strat[s]].mean()) if strat[s] else np.nan)
        if VFDB is not None and len(vfdb_rows):
            fv = m.predict_proba(VFDB)[:, 1] >= t
            acc["vfdb_extra"].append(float(fv[vfdb_rows].mean()))
    out = {k: stat(v) for k, v in acc.items() if v and not np.all(np.isnan(v))}
    ex = np.array(acc["extracellular"], float)
    inn = np.array(acc["intracellular"], float)
    out["R"] = stat(np.where(inn > 0, ex / inn, np.nan))
    if acc["vfdb_extra"]:
        out["CA"] = stat(np.array(acc["vfdb_extra"], float) / np.where(ex > 0, ex, np.nan))
    return out, (n_tr, n_ca)


def selftest():
    p1 = json.loads(POOL1.read_text())["proteins"]
    part, test = partition(p1)
    pub = json.loads(BUILD.read_text())["pool_partition"]
    # 🔒 the partition rule must reproduce pool 1's published split, minus its dropped contaminant
    pub_part = set(pub["train_rows"]) | set(pub["calibrate_rows"])
    overlap = len(set(part) & pub_part) / len(pub_part)
    assert overlap > 0.99, f"partition rule reproduces only {overlap:.3f} of pool 1's split"
    assert fold_sizes(1500) == (1000, 500) and fold_sizes(1450) == (966, 484)
    print(f"selftest PASS (partition rule reproduces {overlap * 100:.1f}% of pool 1's split)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="esm2_35M", choices=["esm2_35M", "esm2_650M"])
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    sfx = "" if a.arm == "esm2_650M" else f"_{a.arm}"
    vf_seqs = M83.vfdb_sequences()
    P = np.load(RES / f"embeddings_class_axis_positives{sfx}.npy")

    # ---- the VFDB side: study K's 778, unchanged ------------------------------------------------
    admitted = json.loads(NEGSET.read_text())["admitted"]
    len_of = {r["vfg"]: r["len"] for r in admitted}
    row_of = {r["vfg"]: i for i, r in enumerate(admitted)}
    chosen = json.loads(MAPPED.read_text())["chosen"]
    noncirc = M83.vfdb_noncircular_mask()
    up_len, acc_of = M91.uniprot_lengths(), M91.vfg_accessions()
    vfdb_extra = [row_of[v] for v, i in chosen.items()
                  if noncirc[row_of[v]] and i["stratum"] == "extracellular"
                  and str(len_of[v]) == str(up_len.get((acc_of[v], i["uniprot"])))]
    VFDB = np.load(RES / f"embeddings_vfdb_neg_{a.arm}.npy")
    print(f"arm {a.arm}: VFDB extracellular cell {len(vfdb_extra)}")

    results = {}
    for label, pool_path, emb in (
            ("pool2", POOL2, RES / f"embeddings_pool_independent_{a.arm}.npy"),
            ("pool1_unscreened", POOL1, RES / f"embeddings_pool_large_{a.arm}.npy")):
        if not emb.exists():
            print(f"  ⚠️  {label}: {emb.name} missing, skipped")
            continue
        proteins = json.loads(pool_path.read_text())["proteins"]
        POOL = np.load(emb)
        if POOL.shape[0] != len(proteins):
            raise SystemExit(f"{label}: {POOL.shape[0]} rows for {len(proteins)} proteins")
        accs = [x["uniprot"] for x in proteins]
        kw, go = M84.fetch_keywords(accs), M92.fetch_go(accs)
        part, test_idx = partition(proteins)
        # contaminants out of both the fitting union and the evaluated strata, as study G's clean
        # condition does
        union = [i for i in part if proteins[i]["sequence"] not in vf_seqs]
        strat = {s: [] for s in M92.STRATA}
        for k, i in enumerate(test_idx):
            if proteins[i]["kingdom"] not in M84.KINGDOMS or proteins[i]["sequence"] in vf_seqs:
                continue
            g = go[accs[i]]
            strat[M92.stratum_of(g["go"], g["sig"], g["tm"], kw[accs[i]])].append(k)
        t0 = time.time()
        out, fold = measure(P, POOL, VFDB, union, test_idx, strat, vfdb_extra, a.seeds)
        n_ex, n_in = len(strat["extracellular"]), len(strat["intracellular"])
        results[label] = {"n_pool": len(proteins), "n_union": len(union), "fold": list(fold),
                          "n_test": len(test_idx), "strata_n": {s: len(v) for s, v in strat.items()},
                          "floors_met": bool(min(n_ex, n_in) >= FLOOR), **out,
                          "seconds": round(time.time() - t0)}
        print(f"  {label:<18} pool {len(proteins):>6}  union {len(union):>5} fold {fold}  "
              f"extra {n_ex:>5} intra {n_in:>5}  R {out['R']['mean']:>6.3f}"
              + (f"  C/A {out['CA']['mean']:.3f}" if "CA" in out else ""))

    if "pool2" not in results:
        raise SystemExit("pool 2 embedding absent; nothing to report")
    p2 = results["pool2"]
    band_R = ("study E's verdict replicates" if p2["R"]["mean"] >= R_HI
              else "does not replicate" if p2["R"]["mean"] <= R_LO else "partial")
    band_CA = ("the registration contrast replicates" if p2["CA"]["mean"] >= CA_HI
               else "does not replicate" if p2["CA"]["mean"] <= CA_LO else "partial") \
        if "CA" in p2 else "not measured"
    res = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "arm": a.arm, "seeds": a.seeds,
           "single_arm_indicative": a.arm != "esm2_650M",
           "results": results, "S1_band": band_R, "S2_band": band_CA,
           "screen_worth": (results["pool1_unscreened"]["R"]["mean"]
                            if "pool1_unscreened" in results else None)}
    dest = Path(f"{OUT_STEM}_{a.arm}.json")
    dest.write_text(json.dumps(res, indent=2) + "\n")

    print(f"\nS-1  pool 2 R = {p2['R']['mean']:.3f} "
          f"[{p2['R']['ci'][0]:.3f}, {p2['R']['ci'][1]:.3f}]  ->  {band_R}")
    if "CA" in p2:
        print(f"S-2  pool 2 C/A = {p2['CA']['mean']:.3f} "
              f"[{p2['CA']['ci'][0]:.3f}, {p2['CA']['ci'][1]:.3f}]  ->  {band_CA}")
    print(f"     floors {'met' if p2['floors_met'] else 'NOT met'} "
          f"(extra {p2['strata_n']['extracellular']}, intra {p2['strata_n']['intracellular']}, "
          f"floor {FLOOR})")
    if "pool1_unscreened" in results:
        print(f"     the similarity screen, measured: pool 1 unscreened gives R = "
              f"{results['pool1_unscreened']['R']['mean']:.3f}")
    print(f"\nwrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
