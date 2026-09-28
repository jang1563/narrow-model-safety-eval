#!/usr/bin/env python3
"""
87_matched_vfdb.py - with provenance AND localization held fixed, does VFDB membership still matter?

`docs/MATCHED_VFDB_PREREGISTRATION.md`. Studies D, E, F and G measured pathogen origin at 22.7% of the
benign-to-VFDB gap, localization at 22.1%, and both together at a measured 46.0%. The majority belongs
to neither and every document says what it does belong to is unknown. This is the first test that can
separate hazard from some further property the curated sets share.

The instrument is the contamination study G removed: 133 pool proteins are exact VFDB sequence matches,
so they are VFDB virulence factors AND Swiss-Prot entries carrying UniProt localization -- the pairing
no other population here has. All are held out under G's clean fold and embedded on both arms.

🔴 The bridge is selected AGAINST the hypothesis: the pool query excludes Virulence, Toxin, Cytolysis,
Hemolysis, Bacteriocin and Bacteriolytic enzyme, so every bridge protein is a VFDB virulence factor
UniProt declines to call virulent. Elevation is strong evidence; a null is weak evidence of absence.

Usage:
    python src/87_matched_vfdb.py --arm esm2_650M
    python src/87_matched_vfdb.py --arm esm2_35M
    python src/87_matched_vfdb.py --selftest
"""

import argparse
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
POOL_JSON = ROOT / "data" / "sequences" / "_scaled_negative_pool.json"
BUILD = ROOT / "results" / "external_class_axis_build.json"
SCREEN = ROOT / "results" / "external_class_axis_screen.json"
OUT_STEM = ROOT / "results" / "matched_vfdb"
SEEDS, N_TRAIN, N_CAL, SPEC = 30, 1000, 500, 0.95
FLOOR, BAND_LO, BAND_HI = 25, 1.2, 2.0


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
    return (N_TRAIN, N_CAL) if n >= N_TRAIN + N_CAL else (2 * n // 3, n - 2 * n // 3)


def stat(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if not len(v):
        return {"mean": float("nan"), "ci": [float("nan"), float("nan")]}
    return {"mean": float(v.mean()),
            "ci": [float(v.mean() - 1.96 * v.std(ddof=1) / len(v) ** 0.5),
                   float(v.mean() + 1.96 * v.std(ddof=1) / len(v) ** 0.5)]}


def selftest():
    assert fold_sizes(1450) == (966, 484), "study G's clean fold must be reproduced"
    assert M84.stratum_of({"Secreted"}) == "extracellular"
    assert M84.stratum_of({"Cytoplasm"}) == "intracellular"
    s = stat([2.0, 2.0, 2.0])
    assert abs(s["mean"] - 2.0) < 1e-12 and abs(s["ci"][0] - 2.0) < 1e-12
    assert np.isnan(stat([float("nan")])["mean"]), "an all-nan cell must not fabricate a mean"
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
    vf_seqs = M83.vfdb_sequences()
    vfdb_species = M83.vfdb_species()

    # ---- 🔒 the positive-side check, which has never been run --------------------------------------
    sfx = "" if a.arm == "esm2_650M" else f"_{a.arm}"
    # 🔴 A circularity check that can be skipped is not a check. The first draft of this script
    # pointed at a filename that does not exist and would have printed a warning and carried on,
    # which is the seventh instance of that pattern in this repository. It exits instead.
    pos_fasta = ROOT / "data" / "sequences" / "vfdb_class_axis_positives.fasta"
    if not pos_fasta.exists():
        raise SystemExit(f"{pos_fasta.relative_to(ROOT)} not found, so the circularity check cannot "
                         f"run; refusing to report a matched contrast without it")
    pos_seqs = {s for _, s in M83.read_fasta(pos_fasta)}
    if len(pos_seqs) < 700:
        raise SystemExit(f"positive FASTA holds {len(pos_seqs)} distinct sequences, expected ~746")
    bridge = [i for i in range(len(pool)) if pool[i]["sequence"] in vf_seqs]
    circular = [pool[i]["uniprot"] for i in bridge if pool[i]["sequence"] in pos_seqs]
    bridge = [i for i in bridge if pool[i]["sequence"] not in pos_seqs]
    print(f"arm {a.arm}: bridge {len(bridge) + len(circular)} proteins, {len(circular)} excluded as "
          f"class-axis positives{' (' + ', '.join(circular[:4]) + ')' if circular else ''}")

    # ---- populations, all Bacteria from VFDB species ---------------------------------------------
    accs_test = [pool[i]["uniprot"] for i in test]
    kw = M84.fetch_keywords(accs_test)
    kw.update(M84.fetch_keywords([pool[i]["uniprot"] for i in bridge]))

    def eligible(i):
        """Held-fixed provenance, for the BENIGN side only.

        🔴 This was applied to the bridge as well in the first run, which is wrong: a protein whose
        sequence matches VFDB exactly is pathogen-derived by construction, and the species test
        dropped nine of them because UniProt has renamed the genus -- Mycoplasmoides pneumoniae,
        Mycobacteroides abscessus, Klebsiella aerogenes -- while VFDB still uses the old name. A
        taxonomic synonym is not evidence about provenance. Amendment 1.
        """
        sp = " ".join(pool[i]["organism"].replace("(", "").split()[:2])
        return pool[i]["kingdom"] == "Bacteria" and sp in vfdb_species

    def eligible_bridge(i):
        """The bridge only has to be bacterial; VFDB membership already fixes provenance."""
        return pool[i]["kingdom"] == "Bacteria"

    groups = {k: [] for k in ("A_benign_extra", "B_benign_intra", "C_vfdb_extra", "D_vfdb_intra",
                              "vfdb_membrane", "vfdb_unannotated",
                              "C_testonly", "D_testonly")}
    testset = set(test)
    for i in test:
        if not eligible(i) or pool[i]["sequence"] in vf_seqs:
            continue
        s = M84.stratum_of(kw[pool[i]["uniprot"]])
        if s == "extracellular":
            groups["A_benign_extra"].append(i)
        elif s == "intracellular":
            groups["B_benign_intra"].append(i)
    for i in bridge:
        if not eligible_bridge(i):
            continue
        s = M84.stratum_of(kw[pool[i]["uniprot"]])
        if s == "extracellular":
            groups["C_vfdb_extra"].append(i)
            if i in testset:
                groups["C_testonly"].append(i)
        elif s == "intracellular":
            groups["D_vfdb_intra"].append(i)
            if i in testset:
                groups["D_testonly"].append(i)
        elif s == "membrane":
            groups["vfdb_membrane"].append(i)
        else:
            groups["vfdb_unannotated"].append(i)
    for k, v in groups.items():
        print(f"  {k:<20}{len(v):>6}")
    floor_met = min(len(groups["C_vfdb_extra"]), len(groups["D_vfdb_intra"])) >= FLOOR

    # ---- study G's clean fold, unchanged ----------------------------------------------------------
    P = np.load(RES / f"embeddings_class_axis_positives{sfx}.npy")
    POOL = np.load(RES / f"embeddings_pool_large_{a.arm}.npy")
    VFDB = np.load(RES / "embeddings_vfdb_neg_esm2_650M.npy") if a.arm == "esm2_650M" else None
    union = [i for i in json.loads(SCREEN.read_text())["admitted_rows"]
             if pool[i]["sequence"] not in vf_seqs]
    n_tr, n_ca = fold_sizes(len(union))
    y = np.r_[np.ones(P.shape[0]), np.zeros(n_tr)]
    print(f"  clean fold: {len(union)} admitted -> train {n_tr}, calibrate {n_ca}")

    per = {k: [] for k in groups}
    per["vfdb_full"] = []
    t0 = time.time()
    for seed in range(a.seeds):
        r = np.random.default_rng(seed)
        perm = r.permutation(len(union))
        tr = [union[i] for i in perm[:n_tr]]
        ca = [union[i] for i in perm[n_tr:n_tr + n_ca]]
        model = clf().fit(np.vstack([P, POOL[tr]]), y)
        t = float(np.quantile(model.predict_proba(POOL[ca])[:, 1], SPEC))
        for k, rows in groups.items():
            per[k].append(float((model.predict_proba(POOL[rows])[:, 1] >= t).mean())
                          if rows else float("nan"))
        if VFDB is not None:
            per["vfdb_full"].append(float((model.predict_proba(VFDB)[:, 1] >= t).mean()))
    print(f"  {time.time() - t0:.0f}s")

    def ratio(num, den):
        n, d = np.array(per[num], float), np.array(per[den], float)
        return stat(np.where(d > 0, n / d, np.nan))

    CA, DB = ratio("C_vfdb_extra", "A_benign_extra"), ratio("D_vfdb_intra", "B_benign_intra")
    band = ("VFDB membership matters beyond both confounds" if CA["mean"] >= BAND_HI
            else "membership adds little" if CA["mean"] <= BAND_LO else "partial")
    res = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "arm": a.arm, "seeds": a.seeds,
           "n_excluded_as_positive": len(circular), "excluded_as_positive": circular,
           "circularity_check_ran": True, "n_positive_sequences": len(pos_seqs),
           "group_n": {k: len(v) for k, v in groups.items()},
           "floor": FLOOR, "floor_met": bool(floor_met),
           "clean_fold": [n_tr, n_ca], "n_union": len(union),
           "rates": {k: stat(v) for k, v in per.items() if len(v)},
           "H1_CA": CA, "H1_DB": DB, "band": band,
           "H1_CA_testonly": ratio("C_testonly", "A_benign_extra"),
           "H1_DB_testonly": ratio("D_testonly", "B_benign_intra"),
           "renamed_genus_dropped_from_benign_side": sorted(
               {" ".join(pool[i]["organism"].replace("(", "").split()[:2]) for i in bridge
                if not eligible(i)})}
    if per["vfdb_full"]:
        A, C, full = (np.mean(per["A_benign_extra"]), np.mean(per["C_vfdb_extra"]),
                      np.mean(per["vfdb_full"]))
        # 🔑 how much of the distance from the MATCHED benign cell to the full VFDB rate does
        # membership alone close? This is the residual that studies D-G could not attribute.
        res["H5_membership_closes"] = float((C - A) / (full - A))
    dest = Path(f"{OUT_STEM}{sfx}.json")
    dest.write_text(json.dumps(res, indent=2) + "\n")

    print(f"\n{'population':<24}{'n':>6}{'flag rate at a nominal 5%':>30}")
    print("-" * 62)
    for k in ("A_benign_extra", "C_vfdb_extra", "B_benign_intra", "D_vfdb_intra",
              "vfdb_membrane", "vfdb_unannotated", "vfdb_full"):
        if k in res["rates"] and not np.isnan(res["rates"][k]["mean"]):
            s = res["rates"][k]
            n = len(groups[k]) if k in groups else 4218
            print(f"  {k:<22}{n:>6}{s['mean'] * 100:>16.2f}% [{s['ci'][0] * 100:.2f}, {s['ci'][1] * 100:.2f}]")
    print(f"\nH-1  extracellular C/A = {CA['mean']:.3f} [{CA['ci'][0]:.3f}, {CA['ci'][1]:.3f}]  ->  {band}")
    print(f"     intracellular D/B = {DB['mean']:.3f} [{DB['ci'][0]:.3f}, {DB['ci'][1]:.3f}]")
    print(f"     test-partition only: C/A = {res['H1_CA_testonly']['mean']:.3f}, "
          f"D/B = {res['H1_DB_testonly']['mean']:.3f}")
    print(f"     floor of {FLOOR} {'met' if floor_met else 'NOT met'}")
    print(f"\nwrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
