#!/usr/bin/env python3
"""
86_decontamination_sensitivity.py - do studies E and F survive a pool with no VFDB proteins in it?

`docs/DECONTAMINATION_PREREGISTRATION.md`. Entry 54 found 133 exact VFDB sequences in the benign pool
and entry 55 corrected their distribution; both said this run was owed. Two effects oppose each other
and this separates them: contaminants in the CALIBRATION set raise the threshold (so removing them
raises every rate), and contaminants inside an EVALUATED stratum are near-certain flags (so removing
them lowers that stratum's rate). Study E never excluded them and its numerator is nearly five times
more contaminated than its denominator.

⚠️ src/84 and src/85 are deliberately not modified -- they are the record of the preregistered
analyses. This script reads the same inputs and reports both conditions side by side.

Usage:
    python src/86_decontamination_sensitivity.py --arm esm2_650M
    python src/86_decontamination_sensitivity.py --arm esm2_35M
    python src/86_decontamination_sensitivity.py --selftest
"""

import argparse
import importlib.util
import json
import re
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "v3"
POOL_JSON = ROOT / "data" / "sequences" / "_scaled_negative_pool.json"
BUILD = ROOT / "results" / "external_class_axis_build.json"
SCREEN = ROOT / "results" / "external_class_axis_screen.json"
OUT_STEM = ROOT / "results" / "decontamination_sensitivity"
SEEDS, N_TRAIN, N_CAL, SPEC = 30, 1000, 500, 0.95
VFDB_RATE = 0.7348585427532796
SPECIES_RE = re.compile(r"\[([A-Z][a-z]+ [a-z]+)")
FLOOR_CELL, FLOOR_STRATUM, R_FLOOR, RR_LO, RR_HI = 150, 300, 3.0, 0.67, 1.5


def _load(stem):
    spec = importlib.util.spec_from_file_location(f"_{stem}", ROOT / "src" / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M84 = _load("84_localization_control")
M83 = _load("83_provenance_control")


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))


def fold_sizes(n):
    """src/83's rule, which is the one all of these studies are supposed to share."""
    return (N_TRAIN, N_CAL) if n >= N_TRAIN + N_CAL else (2 * n // 3, n - 2 * n // 3)


def run(P, POOL, union, groups, seeds):
    """Fit `seeds` folds drawn from `union` and return per-seed flag rates for each named group."""
    n_tr, n_ca = fold_sizes(len(union))
    y = np.r_[np.ones(P.shape[0]), np.zeros(n_tr)]
    out = {k: [] for k in groups}
    thr = []
    for seed in range(seeds):
        r = np.random.default_rng(seed)
        perm = r.permutation(len(union))
        tr = [union[i] for i in perm[:n_tr]]
        ca = [union[i] for i in perm[n_tr:n_tr + n_ca]]
        model = clf().fit(np.vstack([P, POOL[tr]]), y)
        t = float(np.quantile(model.predict_proba(POOL[ca])[:, 1], SPEC))
        thr.append(t)
        for k, rows in groups.items():
            out[k].append(float((model.predict_proba(POOL[rows])[:, 1] >= t).mean())
                          if len(rows) else float("nan"))
    return out, thr, (n_tr, n_ca)


def selftest():
    assert fold_sizes(1468) == (978, 490), "src/83's fallback must be reproduced exactly"
    assert fold_sizes(1450) == (966, 484), "the decontaminated set takes the same branch"
    assert fold_sizes(2000) == (N_TRAIN, N_CAL), "a large enough set takes the fixed branch"
    assert M84.stratum_of({"Secreted"}) == "extracellular"
    assert len(M83.vfdb_sequences()) > 10000
    print("selftest PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="esm2_650M", choices=["esm2_650M", "esm2_35M"])
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    pool = json.loads(POOL_JSON.read_text())["proteins"]
    test = json.loads(BUILD.read_text())["pool_partition"]["test_rows"]
    accs = [pool[i]["uniprot"] for i in test]
    kw = M84.fetch_keywords(accs)
    vf_seqs = M83.vfdb_sequences()
    vfdb_species = set()
    for name in ("VFDB_setA_pro.fas", "VFDB_setB_pro.fas"):
        for line in (ROOT / "data/external/vfdb" / name).read_text(errors="replace").splitlines():
            if line.startswith(">"):
                m = SPECIES_RE.search(line)
                if m:
                    vfdb_species.add(m.group(1))

    # ---- the two conditions differ only in whether contaminants are present --------------------
    union_dirty = json.loads(SCREEN.read_text())["admitted_rows"]
    union_clean = [i for i in union_dirty if pool[i]["sequence"] not in vf_seqs]

    def build_groups(clean):
        """Study E's strata and study F's cells, optionally with contaminants removed."""
        g = {k: [] for k in ("extracellular", "membrane", "intracellular", "unannotated", "all")}
        for k in ("pathogen|extracellular", "pathogen|intracellular",
                  "benign_species|extracellular", "benign_species|intracellular"):
            g[k] = []
        for jj, i in enumerate(test):
            if pool[i]["kingdom"] not in M84.KINGDOMS:
                continue
            contaminated = pool[i]["sequence"] in vf_seqs
            if clean and contaminated:
                continue
            s = M84.stratum_of(kw[accs[jj]])
            g[s].append(jj)
            g["all"].append(jj)
            if s in ("extracellular", "intracellular"):
                # study F ALWAYS excluded contaminants from its cells, in both conditions
                if not contaminated:
                    sp = " ".join(pool[i]["organism"].replace("(", "").split()[:2])
                    g[f"{'pathogen' if sp in vfdb_species else 'benign_species'}|{s}"].append(jj)
        return {k: np.array(v, int) for k, v in g.items()}

    sfx = "" if a.arm == "esm2_650M" else f"_{a.arm}"
    P = np.load(RES / f"embeddings_class_axis_positives{sfx}.npy")
    POOL = np.load(RES / f"embeddings_pool_large_{a.arm}.npy")
    rowmap = np.array(test)

    res = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "arm": a.arm, "seeds": a.seeds,
           "n_contaminants_removed_from_union": len(union_dirty) - len(union_clean),
           "conditions": {}}
    t0 = time.time()
    for cond, union, clean in (("dirty", union_dirty, False), ("clean", union_clean, True)):
        groups = {k: rowmap[v] for k, v in build_groups(clean).items()}
        rates, thr, (n_tr, n_ca) = run(P, POOL, union, groups, a.seeds)
        extra = np.array(rates["extracellular"], float)
        intra = np.array(rates["intracellular"], float)
        rr = ((np.array(rates["pathogen|extracellular"], float)
               / np.array(rates["pathogen|intracellular"], float))
              / (np.array(rates["benign_species|extracellular"], float)
                 / np.array(rates["benign_species|intracellular"], float)))
        pool_rate = float(np.mean(rates["all"]))
        res["conditions"][cond] = {
            "n_union": len(union), "fold": [n_tr, n_ca],
            "group_n": {k: int(len(v)) for k, v in groups.items()},
            "threshold": float(np.mean(thr)),
            "rates": {k: float(np.mean(v)) for k, v in rates.items()},
            "R": float(np.mean(extra / intra)),
            "R_ci": [float(np.mean(extra / intra) - 1.96 * np.std(extra / intra, ddof=1) / a.seeds ** 0.5),
                     float(np.mean(extra / intra) + 1.96 * np.std(extra / intra, ddof=1) / a.seeds ** 0.5)],
            "RR": float(np.mean(rr)),
            "RR_ci": [float(np.mean(rr) - 1.96 * np.std(rr, ddof=1) / a.seeds ** 0.5),
                      float(np.mean(rr) + 1.96 * np.std(rr, ddof=1) / a.seeds ** 0.5)],
            "joint_share": float((np.mean(rates["pathogen|extracellular"]) - pool_rate)
                                 / (VFDB_RATE - pool_rate)),
        }
    print(f"  {time.time() - t0:.0f}s")

    d, c = res["conditions"]["dirty"], res["conditions"]["clean"]
    res["verdict"] = {
        "G1_R_holds": c["R"] >= R_FLOOR,
        "G2_RR_holds": RR_LO <= c["RR"] <= RR_HI,
        "G3_joint_within_10pp": abs(c["joint_share"] - d["joint_share"]) <= 0.10,
        "threshold_fell": c["threshold"] < d["threshold"],
        "floors_met": (min(c["group_n"][k] for k in c["group_n"] if "|" in k) >= FLOOR_CELL
                       and min(c["group_n"]["extracellular"],
                               c["group_n"]["intracellular"]) >= FLOOR_STRATUM),
    }
    dest = Path(f"{OUT_STEM}{sfx}.json")
    dest.write_text(json.dumps(res, indent=2) + "\n")

    print(f"\narm {a.arm}: {res['n_contaminants_removed_from_union']} contaminants removed from the "
          f"admitted set; fold {d['fold']} -> {c['fold']}")
    print(f"\n{'quantity':<30}{'dirty':>12}{'clean':>12}{'delta':>11}")
    print("-" * 65)
    for k, f in (("threshold", "{:.4f}"), ("extracellular", "{:.4f}"), ("intracellular", "{:.4f}"),
                 ("all (pool)", "{:.4f}")):
        kk = "all" if k == "all (pool)" else k
        dv = d["threshold"] if k == "threshold" else d["rates"][kk]
        cv = c["threshold"] if k == "threshold" else c["rates"][kk]
        print(f"  {k:<28}{f.format(dv):>12}{f.format(cv):>12}{cv - dv:>+11.4f}")
    for k in ("R", "RR", "joint_share"):
        print(f"  {k:<28}{d[k]:>12.3f}{c[k]:>12.3f}{c[k] - d[k]:>+11.3f}")
    v = res["verdict"]
    print(f"\nG-1  clean R = {c['R']:.3f} [{c['R_ci'][0]:.3f}, {c['R_ci'][1]:.3f}]  -> "
          f"{'HOLDS' if v['G1_R_holds'] else 'FAILS'} (floor {R_FLOOR})")
    print(f"G-2  clean RR = {c['RR']:.3f} [{c['RR_ci'][0]:.3f}, {c['RR_ci'][1]:.3f}]  -> "
          f"{'HOLDS' if v['G2_RR_holds'] else 'FAILS'} (band [{RR_LO}, {RR_HI}])")
    print(f"G-3  joint share {d['joint_share'] * 100:.1f}% -> {c['joint_share'] * 100:.1f}%  -> "
          f"{'HOLDS' if v['G3_joint_within_10pp'] else 'FAILS'}")
    print(f"     threshold {'fell' if v['threshold_fell'] else 'ROSE'}, "
          f"floors {'met' if v['floors_met'] else 'NOT met'}")
    print(f"\nwrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
