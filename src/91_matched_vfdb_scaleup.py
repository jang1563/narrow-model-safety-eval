#!/usr/bin/env python3
"""
91_matched_vfdb_scaleup.py - study H's matched contrast, at seven times the size and without its selection.

`docs/MATCHED_VFDB_SCALEUP_PREREGISTRATION.md`. Study H held pathogen origin and extracellular
localization fixed and varied only VFDB membership, and found membership covers 99.1% of the remaining
distance. It carried two limits: its VFDB side was the pool∩VFDB bridge, so the pool's build query had
already removed every member UniProt calls virulent, and it was small -- 38 extracellular and 24
intracellular, one under its own floor.

src/90 gives the 4,218 admitted negatives a Swiss-Prot identity through their RefSeq accessions,
including the 300 virulence-annotated members the bridge could not contain. This runs the same contrast
on 264 and 134.

⚠️ Study G's clean fold, unchanged. The only new inference anywhere is the 35M embedding of the 4,218,
whose gate reproduces the published 35M panel negatives exactly.

Usage:
    python src/91_matched_vfdb_scaleup.py --arm esm2_650M
    python src/91_matched_vfdb_scaleup.py --arm esm2_35M
    python src/91_matched_vfdb_scaleup.py --selftest
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
NEGSET = ROOT / "results" / "vfdb_negative_set.json"
MAPPED = ROOT / "results" / "vfdb_uniprot_map.json"
IDMAP = ROOT / "data" / "external" / "uniprot_idmap" / "vfdb_negatives_idmap.tsv"
BUILD = ROOT / "results" / "external_class_axis_build.json"
SCREEN = ROOT / "results" / "external_class_axis_screen.json"
OUT_STEM = ROOT / "results" / "matched_vfdb_scaleup"
SEEDS, N_TRAIN, N_CAL, SPEC = 30, 1000, 500, 0.95
FLOOR, BAND_LO, BAND_HI = 100, 1.2, 2.0
HDR = re.compile(r">(VFG\d+)\((\w+)\|([\w.]+)\)")


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


def uniprot_lengths():
    """(source accession, UniProt entry) -> length, from src/90's cached mapping."""
    out = {}
    for line in IDMAP.read_text().splitlines()[1:]:
        f = line.split("\t")
        if len(f) >= 6:
            out[(f[0], f[1])] = f[5]
    return out


def vfg_accessions():
    out = {}
    for name in ("VFDB_setA_pro.fas", "VFDB_setB_pro.fas"):
        for line in (ROOT / "data/external/vfdb" / name).read_text(errors="replace").splitlines():
            if line.startswith(">"):
                m = HDR.match(line)
                if m:
                    out.setdefault(m.group(1), m.group(3))
    return out


def selftest():
    assert fold_sizes(1450) == (966, 484), "study G's clean fold must be reproduced"
    assert M84.stratum_of({"Secreted"}) == "extracellular"
    assert M84.stratum_of({"Cytoplasm"}) == "intracellular"
    assert HDR.match(">VFG000002(gb|WP_010930159) x").group(3) == "WP_010930159"
    m = M83.vfdb_noncircular_mask()
    assert m.shape[0] == 4218 and int((~m).sum()) == 542, "the circularity mask must be entry 61's"
    assert np.isnan(stat([float("nan")])["mean"])
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
    vfdb_species = M83.vfdb_species()

    admitted = [r["vfg"] for r in json.loads(NEGSET.read_text())["admitted"]]
    len_of = {r["vfg"]: r["len"] for r in json.loads(NEGSET.read_text())["admitted"]}
    chosen = json.loads(MAPPED.read_text())["chosen"]
    noncirc = M83.vfdb_noncircular_mask()
    row_of = {v: i for i, v in enumerate(admitted)}
    up_len, acc_of = uniprot_lengths(), vfg_accessions()

    # ---- the three frozen exclusions -------------------------------------------------------------
    keep, drop_circ, drop_len = {}, 0, []
    for vfg, info in chosen.items():
        if not noncirc[row_of[vfg]]:
            drop_circ += 1
            continue
        ul = up_len.get((acc_of[vfg], info["uniprot"]))
        if ul is None or str(len_of[vfg]) != str(ul):
            drop_len.append({"vfg": vfg, "uniprot": info["uniprot"],
                             "vfdb_len": len_of[vfg], "uniprot_len": ul})
            continue
        keep[vfg] = info
    print(f"arm {a.arm}: {len(chosen)} mapped-reviewed -> {len(keep)} after exclusions "
          f"({drop_circ} circular, {len(drop_len)} length-mismatched)")

    vcell = {"extracellular": [], "intracellular": [], "membrane": [], "unannotated": []}
    virulence, other = [], []
    for vfg, info in keep.items():
        vcell[info["stratum"]].append(row_of[vfg])
        (virulence if "Virulence" in info["keywords"] else other).append(row_of[vfg])

    bcell = {"extracellular": [], "intracellular": []}
    for i in test:
        sp = " ".join(pool[i]["organism"].replace("(", "").split()[:2])
        if pool[i]["kingdom"] != "Bacteria" or sp not in vfdb_species:
            continue
        if pool[i]["sequence"] in vf_seqs:
            continue
        s = M84.stratum_of(kw[pool[i]["uniprot"]])
        if s in bcell:
            bcell[s].append(i)
    for s in ("extracellular", "intracellular"):
        print(f"  {s:<16} VFDB {len(vcell[s]):>5}   benign {len(bcell[s]):>5}")
    floor_met = min(len(vcell["extracellular"]), len(vcell["intracellular"])) >= FLOOR

    # ---- study G's clean fold, unchanged ---------------------------------------------------------
    sfx = "" if a.arm == "esm2_650M" else f"_{a.arm}"
    P = np.load(RES / f"embeddings_class_axis_positives{sfx}.npy")
    POOL = np.load(RES / f"embeddings_pool_large_{a.arm}.npy")
    VFDB = np.load(RES / f"embeddings_vfdb_neg_{a.arm}.npy")
    if VFDB.shape[0] != len(admitted):
        raise SystemExit(f"VFDB embedding has {VFDB.shape[0]} rows, artifact has {len(admitted)}")
    union = [i for i in json.loads(SCREEN.read_text())["admitted_rows"]
             if pool[i]["sequence"] not in vf_seqs]
    n_tr, n_ca = fold_sizes(len(union))
    y = np.r_[np.ones(P.shape[0]), np.zeros(n_tr)]
    print(f"  clean fold: {len(union)} admitted -> train {n_tr}, calibrate {n_ca}")

    per = {f"V_{s}": [] for s in vcell}
    per.update({f"B_{s}": [] for s in bcell})
    per["V_virulence"], per["V_other"], per["V_reference"] = [], [], []
    t0 = time.time()
    for seed in range(a.seeds):
        r = np.random.default_rng(seed)
        perm = r.permutation(len(union))
        tr = [union[i] for i in perm[:n_tr]]
        ca = [union[i] for i in perm[n_tr:n_tr + n_ca]]
        model = clf().fit(np.vstack([P, POOL[tr]]), y)
        t = float(np.quantile(model.predict_proba(POOL[ca])[:, 1], SPEC))
        fv = model.predict_proba(VFDB)[:, 1] >= t
        fb = model.predict_proba(POOL[np.array(test)])[:, 1] >= t
        idx = {i: k for k, i in enumerate(test)}
        for s, rows in vcell.items():
            per[f"V_{s}"].append(float(fv[rows].mean()) if rows else float("nan"))
        for s, rows in bcell.items():
            per[f"B_{s}"].append(float(fb[[idx[i] for i in rows]].mean()) if rows else float("nan"))
        per["V_virulence"].append(float(fv[virulence].mean()) if virulence else float("nan"))
        per["V_other"].append(float(fv[other].mean()) if other else float("nan"))
        per["V_reference"].append(float(fv[noncirc].mean()))
    print(f"  {time.time() - t0:.0f}s")

    def ratio(n, d):
        x, z = np.array(per[n], float), np.array(per[d], float)
        return stat(np.where(z > 0, x / z, np.nan))

    CA, DB = ratio("V_extracellular", "B_extracellular"), ratio("V_intracellular", "B_intracellular")
    band = ("VFDB membership matters beyond both confounds" if CA["mean"] >= BAND_HI
            else "membership adds little" if CA["mean"] <= BAND_LO else "partial")
    A, C, ref = (np.mean(per["B_extracellular"]), np.mean(per["V_extracellular"]),
                 np.mean(per["V_reference"]))
    res = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "arm": a.arm, "seeds": a.seeds,
           "n_mapped_reviewed": len(chosen), "n_kept": len(keep),
           "n_dropped_circular": drop_circ, "n_dropped_length": len(drop_len),
           "length_mismatched": drop_len,
           "cell_n": {**{f"V_{s}": len(v) for s, v in vcell.items()},
                      **{f"B_{s}": len(v) for s, v in bcell.items()}},
           "n_virulence_keyword": len(virulence), "n_other_keyword": len(other),
           "floor": FLOOR, "floor_met": bool(floor_met), "clean_fold": [n_tr, n_ca],
           "rates": {k: stat(v) for k, v in per.items()},
           "K1_CA": CA, "K1b_DB": DB, "band": band,
           "K3_virulence_minus_other": float(np.mean(per["V_virulence"]) - np.mean(per["V_other"])),
           "K4_membership_closes": float((C - A) / (ref - A))}
    dest = Path(f"{OUT_STEM}{sfx}.json")
    dest.write_text(json.dumps(res, indent=2) + "\n")

    print(f"\n{'population':<26}{'n':>6}{'flag rate at a nominal 5%':>30}")
    print("-" * 64)
    for k in ("B_extracellular", "V_extracellular", "B_intracellular", "V_intracellular",
              "V_membrane", "V_unannotated", "V_virulence", "V_other", "V_reference"):
        s = res["rates"][k]
        n = res["cell_n"].get(k, {"V_virulence": len(virulence), "V_other": len(other),
                                  "V_reference": int(noncirc.sum())}.get(k, 0))
        print(f"  {k:<24}{n:>6}{s['mean'] * 100:>16.2f}% [{s['ci'][0] * 100:.2f}, {s['ci'][1] * 100:.2f}]")
    print(f"\nK-1  extracellular C/A = {CA['mean']:.3f} [{CA['ci'][0]:.3f}, {CA['ci'][1]:.3f}]  ->  {band}")
    print(f"K-1b intracellular D/B = {DB['mean']:.3f} [{DB['ci'][0]:.3f}, {DB['ci'][1]:.3f}]  "
          f"(floor {'met' if floor_met else 'NOT met'})")
    print(f"K-3  Virulence-keyword members minus the rest: {res['K3_virulence_minus_other'] * 100:+.2f} pp")
    print(f"K-4  membership closes {res['K4_membership_closes'] * 100:.1f}% of the distance to the "
          f"{ref * 100:.2f}% reference")
    print(f"\nwrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
