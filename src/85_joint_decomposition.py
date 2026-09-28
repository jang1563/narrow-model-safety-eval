#!/usr/bin/env python3
"""
85_joint_decomposition.py - do provenance and localization overlap, and what do they buy together?

`docs/JOINT_DECOMPOSITION_PREREGISTRATION.md`. Study D measured pathogen origin at 22.7% of the
benign-to-VFDB gap and study E measured localization at 22.1%, and three documents now say the two
may not be added because pathogen-derived proteins are themselves enriched for secretion. This runs
the 2x2 that settles it, and measures the joint share directly instead of inferring it.

⚠️ No new probe and no new inference, for the third study running: study B's fold exactly as src/83
and src/84 use it, on proteins already embedded on both arms. Both factors are annotations.

Usage:
    python src/85_joint_decomposition.py --arm esm2_650M
    python src/85_joint_decomposition.py --arm esm2_35M
    python src/85_joint_decomposition.py --selftest
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
OUT_STEM = ROOT / "results" / "joint_decomposition"
SEEDS, N_TRAIN, N_CAL, SPEC = 30, 1000, 500, 0.95
VFDB_RATE, STUDY_D_RATE = 0.7348585427532796, 0.2207547169811321
FLOOR, BAND_LO, BAND_HI = 150, 0.67, 1.5
SPECIES_RE = re.compile(r"\[([A-Z][a-z]+ [a-z]+)")


def _load(stem):
    """Import a numerically-prefixed sibling script.

    🔒 The localization strata rule and the VFDB sequence reader are FROZEN in src/84 and src/83.
    Copying either into this file would let the two drift apart silently, and a frozen rule that
    exists in two places is not frozen. Imported instead.
    """
    spec = importlib.util.spec_from_file_location(f"_{stem}", ROOT / "src" / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M84 = _load("84_localization_control")
M83 = _load("83_provenance_control")


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))


def ratio_of_ratios(a, b, c, d):
    """(a/b) / (c/d), the 2x2 interaction on a multiplicative scale.

    a = pathogen & extracellular, b = pathogen & intracellular,
    c = benign-species & extracellular, d = benign-species & intracellular.
    """
    if b == 0 or d == 0 or c == 0:
        return float("nan")
    return (a / b) / (c / d)


def selftest():
    assert M84.stratum_of({"Secreted"}) == "extracellular", "imported strata rule changed"
    assert M84.stratum_of({"Cytoplasm"}) == "intracellular", "imported strata rule changed"
    assert abs(ratio_of_ratios(0.4, 0.1, 0.2, 0.05) - 1.0) < 1e-12, "independent case must give 1"
    assert ratio_of_ratios(0.3, 0.1, 0.2, 0.02) < 1.0, "sub-multiplicative case must be below 1"
    assert ratio_of_ratios(0.4, 0.05, 0.2, 0.05) > 1.0, "super-multiplicative case must exceed 1"
    assert np.isnan(ratio_of_ratios(0.4, 0.0, 0.2, 0.05)), "a zero denominator must not divide"
    vf = M83.vfdb_sequences()
    assert len(vf) > 10000, f"VFDB sequence set looks wrong: {len(vf)}"
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

    # ---- the two frozen factors ------------------------------------------------------------------
    vfdb_species = set()
    for name in ("VFDB_setA_pro.fas", "VFDB_setB_pro.fas"):
        for line in (ROOT / "data/external/vfdb" / name).read_text(errors="replace").splitlines():
            if line.startswith(">"):
                m = SPECIES_RE.search(line)
                if m:
                    vfdb_species.add(m.group(1))
    vf_seqs = M83.vfdb_sequences()

    cells = {(p, s): [] for p in ("pathogen", "benign_species")
             for s in ("extracellular", "intracellular")}
    elig, dropped = [], []
    for jj, i in enumerate(test):
        if pool[i]["kingdom"] not in M84.KINGDOMS:
            continue
        elig.append(jj)
        strat = M84.stratum_of(kw[accs[jj]])
        if strat not in ("extracellular", "intracellular"):
            continue
        if pool[i]["sequence"] in vf_seqs:            # a positive sitting in a negative set
            dropped.append(pool[i]["uniprot"])
            continue
        sp = " ".join(pool[i]["organism"].replace("(", "").split()[:2])
        cells[("pathogen" if sp in vfdb_species else "benign_species", strat)].append(jj)

    print(f"arm {a.arm}: {len(vfdb_species)} VFDB species; {len(elig)} eligible pool proteins")
    print(f"  excluded {len(dropped)} exact VFDB sequence matches — positives in a negative set")
    for k, v in cells.items():
        print(f"  {k[0]:<16}{k[1]:<16}{len(v):>6}")
    floor_met = min(len(v) for v in cells.values()) >= FLOOR
    if not floor_met:
        print(f"  ⚠️  floor of {FLOOR} NOT met — F-1 is indicative, not a verdict")

    pathogen_all = cells[("pathogen", "extracellular")] + cells[("pathogen", "intracellular")]

    # ---- study B's fold, unchanged ---------------------------------------------------------------
    sfx = "" if a.arm == "esm2_650M" else f"_{a.arm}"
    P = np.load(RES / f"embeddings_class_axis_positives{sfx}.npy")
    POOL = np.load(RES / f"embeddings_pool_large_{a.arm}.npy")
    union = json.loads(SCREEN.read_text())["admitted_rows"]
    # 🔴 src/83 takes a fallback branch here and src/84 and src/85 did not, so for one commit the
    # three studies used different folds while all three asserted they used the same one. The screen
    # rejects 32 of the 1,500 partition rows for similarity to positives, leaving 1,468 -- FEWER than
    # N_TRAIN + N_CAL -- so a fixed 1000/500 slice silently calibrated on 468. Entry 55.
    n_tr, n_ca = (N_TRAIN, N_CAL) if len(union) >= N_TRAIN + N_CAL else \
        (2 * len(union) // 3, len(union) - 2 * len(union) // 3)
    y = np.r_[np.ones(P.shape[0]), np.zeros(n_tr)]

    # 🔴 The exclusion above prompted the count nobody had run. Recorded here because this is the
    # script that found it: the benign pool is not VFDB-free, and 12 of the contaminants are in the
    # 500 proteins that set the 95% threshold, where only 25 sit above the cut. The bias is
    # conservative -- contaminants score high, push the threshold up, and every flag rate reported
    # anywhere in this repository is too LOW -- but direction is not magnitude.
    part = json.loads(BUILD.read_text())["pool_partition"]
    contam = {k.replace("_rows", ""): sum(1 for i in part[k] if pool[i]["sequence"] in vf_seqs)
              for k in ("train_rows", "calibrate_rows", "test_rows")}
    contam["whole_pool"] = sum(1 for q in pool if q["sequence"] in vf_seqs)
    contam["n"] = {"train": len(part["train_rows"]), "calibrate": len(part["calibrate_rows"]),
                   "test": len(part["test_rows"]), "whole_pool": len(pool)}
    # 🔴 train/calibrate above are the BUILD partition, which is not what gets fitted: the rows are
    # drawn per seed from the screen's admitted set, so no contaminant is fixed in either. What
    # matters is how many survive the screen. Entry 55.
    trca = set(part["train_rows"]) | set(part["calibrate_rows"])
    adm = set(union)
    contam["partition_rows"] = sum(1 for i in trca if pool[i]["sequence"] in vf_seqs)
    contam["admitted"] = sum(1 for i in adm if pool[i]["sequence"] in vf_seqs)
    contam["screened_out"] = contam["partition_rows"] - contam["admitted"]
    contam["n"]["partition_rows"], contam["n"]["admitted"] = len(trca), len(adm)
    # 🟢 the screen was built to drop pool rows too similar to the positives, and the positives ARE
    # VFDB representatives, so it removes contaminants far above its base rate -- an incidental
    # safeguard nobody designed, quantified here rather than assumed.
    contam["screen_enrichment"] = round(
        (contam["screened_out"] / max(contam["partition_rows"], 1))
        / ((len(trca) - len(adm)) / len(trca)), 2)
    contam["expected_in_calibration_per_seed"] = round(contam["admitted"] * n_ca / len(adm), 1)
    print(f"  pool contamination by exact VFDB sequence: whole pool {contam['whole_pool']}/{len(pool)}, "
          f"test {contam['test']}/{contam['n']['test']}, partition rows "
          f"{contam['partition_rows']}/{len(trca)} of which the screen removed "
          f"{contam['screened_out']} ({contam['screen_enrichment']}x its base rate), leaving "
          f"{contam['admitted']}/{len(adm)} fitted and about "
          f"{contam['expected_in_calibration_per_seed']} in each seed's {n_ca}-protein calibration")

    rows = np.array(test)

    per = {k: [] for k in cells}
    per["pool_all"], per["pathogen_all"] = [], []
    rr = []
    t0 = time.time()
    for seed in range(a.seeds):
        r = np.random.default_rng(seed)
        perm = r.permutation(len(union))
        tr = [union[i] for i in perm[:n_tr]]
        ca = [union[i] for i in perm[n_tr:n_tr + n_ca]]
        model = clf().fit(np.vstack([P, POOL[tr]]), y)
        t = float(np.quantile(model.predict_proba(POOL[ca])[:, 1], SPEC))
        flag = model.predict_proba(POOL[rows])[:, 1] >= t
        for k, v in cells.items():
            per[k].append(float(flag[v].mean()))
        per["pool_all"].append(float(flag[elig].mean()))
        per["pathogen_all"].append(float(flag[pathogen_all].mean()))
        rr.append(ratio_of_ratios(per[("pathogen", "extracellular")][-1],
                                  per[("pathogen", "intracellular")][-1],
                                  per[("benign_species", "extracellular")][-1],
                                  per[("benign_species", "intracellular")][-1]))
    print(f"  {time.time() - t0:.0f}s")

    def stat(v):
        v = np.asarray(v, float)
        return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                "ci": [float(v.mean() - 1.96 * v.std(ddof=1) / len(v) ** 0.5),
                       float(v.mean() + 1.96 * v.std(ddof=1) / len(v) ** 0.5)]}

    RR = stat(rr)
    band = ("multiplicatively independent" if BAND_LO <= RR["mean"] <= BAND_HI
            else "sub-multiplicative — they overlap" if RR["mean"] < BAND_LO
            else "super-multiplicative")
    pool_rate = float(np.mean(per["pool_all"]))
    joint = float((np.mean(per[("pathogen", "extracellular")]) - pool_rate)
                  / (VFDB_RATE - pool_rate))
    f5 = float(np.mean(per[("pathogen", "intracellular")])
               / np.mean(per[("benign_species", "intracellular")]))

    res = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "arm": a.arm, "seeds": a.seeds,
           "n_vfdb_species": len(vfdb_species), "n_eligible": len(elig),
           "n_excluded_as_vfdb": len(dropped), "excluded_as_vfdb": dropped,
           "cell_n": {f"{p}|{s}": len(v) for (p, s), v in cells.items()},
           "floor": FLOOR, "floor_met": bool(floor_met),
           "rates": {("|".join(k) if isinstance(k, tuple) else k): stat(v)
                     for k, v in per.items()},
           "F1_ratio_of_ratios": RR, "band": band,
           "F3_joint_share": joint,
           "F4_pathogen_pool_rate": float(np.mean(per["pathogen_all"])),
           "F4_study_d_rate": STUDY_D_RATE,
           "F4_bound_holds": bool(np.mean(per["pathogen_all"]) < STUDY_D_RATE),
           "F5_provenance_within_intracellular": f5,
           "pool_contamination": contam}
    dest = Path(f"{OUT_STEM}{sfx}.json")
    dest.write_text(json.dumps(res, indent=2) + "\n")

    print(f"\n{'cell':<34}{'n':>7}{'flag rate at a nominal 5%':>28}")
    print("-" * 70)
    for k in (("pathogen", "extracellular"), ("pathogen", "intracellular"),
              ("benign_species", "extracellular"), ("benign_species", "intracellular")):
        s = res["rates"][f"{k[0]}|{k[1]}"]
        print(f"  {k[0] + ' x ' + k[1]:<32}{len(cells[k]):>7}{s['mean'] * 100:>16.2f}% "
              f"[{s['ci'][0] * 100:.2f}, {s['ci'][1] * 100:.2f}]")
    print(f"\nF-1  RR = {RR['mean']:.3f} [{RR['ci'][0]:.3f}, {RR['ci'][1]:.3f}]  ->  {band}")
    print(f"F-3  jointly they span {joint * 100:.1f}% of the pool->VFDB gap "
          f"(provenance alone 22.7%, localization alone 22.1%; the SUM is forbidden)")
    print(f"F-4  pool pathogen-species rate {np.mean(per['pathogen_all']) * 100:.2f}% vs study D's "
          f"{STUDY_D_RATE * 100:.2f}% -> bound {'HOLDS' if res['F4_bound_holds'] else 'FAILS'}")
    print(f"F-5  provenance within intracellular only: {f5:.3f}x")
    print(f"\nwrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
