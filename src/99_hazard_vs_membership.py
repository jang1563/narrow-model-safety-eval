#!/usr/bin/env python3
"""
99_hazard_vs_membership.py - does the probe respond to hazard, or to VFDB membership?

`docs/HAZARD_VS_MEMBERSHIP_PREREGISTRATION.md`. Study K found VFDB membership raises the flag rate
2.418x and 2.617x with provenance and localization held fixed -- the project's main positive result,
and the one most at risk of overreach, because VFDB membership is a CURATOR label. VFDB contains
Motility: flagellar and chemotaxis proteins, surface structures with no toxic function. Study B already
put Motility's recovery at 0.8481, the highest of all 13 categories and above Exotoxin's 0.7944, and
nothing followed it up.

🔴 Study K's population excludes exotoxins by construction -- src/74 built it from "VFDB setA,
non-Exotoxin records" -- so its result is properly "non-toxin virulence-factor membership matters", and
the contrast available here is partly-hazardous against non-hazardous rather than toxin against
non-toxin. Both are declared in the preregistration.

⚠️ No new probe and no new inference: study G's clean fold, study L's strata, study K's three
exclusions, the same 30 seeds. Only the VFDB side is split.

Usage:
    python src/99_hazard_vs_membership.py --arm esm2_650M
    python src/99_hazard_vs_membership.py --arm esm2_35M
    python src/99_hazard_vs_membership.py --selftest
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
NEGSET = ROOT / "results" / "vfdb_negative_set.json"
MAPPED = ROOT / "results" / "vfdb_uniprot_map.json"
IDMAP = ROOT / "data" / "external" / "uniprot_idmap" / "vfdb_negatives_idmap.tsv"
BUILD = ROOT / "results" / "external_class_axis_build.json"
SCREEN = ROOT / "results" / "external_class_axis_screen.json"
OUT_STEM = ROOT / "results" / "hazard_vs_membership"
SEEDS, SPEC, FLOOR = 30, 0.95, 30
BAND_LO, BAND_HI = 1.2, 2.0

# 🔒 Frozen in § 1.1 of the preregistration. VFDB does not label categories by hazard, so this is a
# domain judgment and the reasoning is written there rather than assumed here.
NON_HAZARDOUS = {"Motility", "Nutritional/Metabolic factor", "Regulation", "Stress survival"}
PARTLY_HAZARDOUS = {"Effector delivery system", "Immune modulation", "Invasion", "Enzyme"}
# Adherence is the third largest and is deliberately in NEITHER group: forcing the largest ambiguous
# class into either arm would decide the result by that choice alone.
AMBIGUOUS = {"Adherence", "Biofilm", "Others", "Post-translational modification",
             "Antimicrobial activity/Competitive advantage"}


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


def stat(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if not len(v):
        return {"mean": float("nan"), "ci": [float("nan"), float("nan")]}
    return {"mean": float(v.mean()),
            "ci": [float(v.mean() - 1.96 * v.std(ddof=1) / len(v) ** 0.5),
                   float(v.mean() + 1.96 * v.std(ddof=1) / len(v) ** 0.5)]}


def selftest():
    assert not (NON_HAZARDOUS & PARTLY_HAZARDOUS), "the two groups must be disjoint"
    assert not (NON_HAZARDOUS & AMBIGUOUS) and not (PARTLY_HAZARDOUS & AMBIGUOUS)
    cats = set(json.loads(NEGSET.read_text())["categories"])
    covered = NON_HAZARDOUS | PARTLY_HAZARDOUS | AMBIGUOUS
    missing = cats - covered
    assert not missing, f"unassigned VFDB categories would be silently dropped: {missing}"
    assert "Exotoxin" not in cats, "src/74 excluded exotoxins; the preregistration depends on that"
    assert "Motility" in NON_HAZARDOUS
    print(f"selftest PASS ({len(cats)} VFDB categories, all assigned; no Exotoxin present)")


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
    # 🔒 study K classified its benign side with src/84's KEYWORD rule, not study L's GO-augmented
    # one, and the preregistration requires this study's bands to be directly comparable to study K's
    # 2.418. So the keyword rule is used here too and no GO fetch is needed.
    kw = M84.fetch_keywords(accs)
    vf_seqs = M83.vfdb_sequences()
    vfdb_species = M83.vfdb_species()

    admitted = json.loads(NEGSET.read_text())["admitted"]
    len_of = {r["vfg"]: r["len"] for r in admitted}
    row_of = {r["vfg"]: i for i, r in enumerate(admitted)}
    chosen = json.loads(MAPPED.read_text())["chosen"]
    noncirc = M83.vfdb_noncircular_mask()
    up_len, acc_of = M91.uniprot_lengths(), M91.vfg_accessions()

    # ---- study K's three exclusions, unchanged ---------------------------------------------------
    keep = {}
    for vfg, info in chosen.items():
        if not noncirc[row_of[vfg]]:
            continue
        ul = up_len.get((acc_of[vfg], info["uniprot"]))
        if ul is None or str(len_of[vfg]) != str(ul):
            continue
        keep[vfg] = info
    print(f"arm {a.arm}: {len(keep)} VFDB proteins after study K's exclusions")

    groups = {"non_hazardous": NON_HAZARDOUS, "partly_hazardous": PARTLY_HAZARDOUS,
              "ambiguous": AMBIGUOUS, "motility": {"Motility"}}
    cells = {}
    for gname, cats in groups.items():
        for s in ("extracellular", "intracellular"):
            cells[(gname, s)] = [row_of[v] for v, i in keep.items()
                                 if i["category"] in cats and i["stratum"] == s]
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
    print(f"{'group':<20}{'extra':>8}{'intra':>8}")
    for g in groups:
        print(f"  {g:<18}{len(cells[(g, 'extracellular')]):>8}{len(cells[(g, 'intracellular')]):>8}")
    print(f"  {'benign':<18}{len(bcell['extracellular']):>8}{len(bcell['intracellular']):>8}")

    sfx = "" if a.arm == "esm2_650M" else f"_{a.arm}"
    P = np.load(RES / f"embeddings_class_axis_positives{sfx}.npy")
    POOL = np.load(RES / f"embeddings_pool_large_{a.arm}.npy")
    VFDB = np.load(RES / f"embeddings_vfdb_neg_{a.arm}.npy")
    union = [i for i in json.loads(SCREEN.read_text())["admitted_rows"]
             if pool[i]["sequence"] not in vf_seqs]
    n_tr, n_ca = M92.fold_sizes(len(union))
    y = np.r_[np.ones(P.shape[0]), np.zeros(n_tr)]

    per = {f"{g}|{s}": [] for g in groups for s in ("extracellular", "intracellular")}
    per.update({f"benign|{s}": [] for s in bcell})
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
        for (g, s), rows in cells.items():
            per[f"{g}|{s}"].append(float(fv[rows].mean()) if rows else float("nan"))
        for s, rows in bcell.items():
            per[f"benign|{s}"].append(float(fb[[idx[i] for i in rows]].mean()))
    print(f"  {time.time() - t0:.0f}s")

    def ratio(n, d):
        x, z = np.array(per[n], float), np.array(per[d], float)
        return stat(np.where(z > 0, x / z, np.nan))

    Q1 = ratio("non_hazardous|extracellular", "benign|extracellular")
    Q1i = ratio("non_hazardous|intracellular", "benign|intracellular")
    Q3 = ratio("motility|extracellular", "benign|extracellular")
    PH = ratio("partly_hazardous|extracellular", "benign|extracellular")
    Q4 = {s: ratio(f"non_hazardous|{s}", f"partly_hazardous|{s}")
          for s in ("extracellular", "intracellular")}
    band = ("the probe responds to membership, not hazard" if Q1["mean"] >= BAND_HI
            else "the non-hazardous categories are not elevated" if Q1["mean"] <= BAND_LO
            else "partial")
    floors = {f"{g}|{s}": len(v) >= FLOOR for (g, s), v in cells.items()}

    res = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "arm": a.arm, "seeds": a.seeds,
           "n_kept": len(keep), "floor": FLOOR,
           "cell_n": {f"{g}|{s}": len(v) for (g, s), v in cells.items()}
                     | {f"benign|{s}": len(v) for s, v in bcell.items()},
           "floors_met": floors,
           "rates": {k: stat(v) for k, v in per.items()},
           "Q1_non_hazardous_CA": Q1, "Q1_intracellular_DB": Q1i,
           "Q3_motility_CA": Q3, "partly_hazardous_CA": PH,
           "Q4_non_over_partly": Q4, "band": band,
           "study_K_CA": {"650M": 2.418, "35M": 2.617},
           "split_happened": bool(abs(np.mean(per["non_hazardous|extracellular"])
                                      - np.mean(per["partly_hazardous|extracellular"])) > 1e-4)}
    dest = Path(f"{OUT_STEM}{sfx}.json")
    dest.write_text(json.dumps(res, indent=2) + "\n")

    print(f"\n{'population':<34}{'n':>6}{'flag rate':>24}")
    print("-" * 66)
    for k in ("benign|extracellular", "non_hazardous|extracellular", "motility|extracellular",
              "partly_hazardous|extracellular", "ambiguous|extracellular",
              "benign|intracellular", "non_hazardous|intracellular",
              "partly_hazardous|intracellular"):
        s_ = res["rates"][k]
        print(f"  {k:<32}{res['cell_n'][k]:>6}{s_['mean'] * 100:>14.2f}% "
              f"[{s_['ci'][0] * 100:.2f}, {s_['ci'][1] * 100:.2f}]")
    print(f"\nQ-1  non-hazardous C/A = {Q1['mean']:.3f} [{Q1['ci'][0]:.3f}, {Q1['ci'][1]:.3f}]"
          f"   (study K, all categories: {res['study_K_CA'][a.arm.replace('esm2_', '')]})")
    print(f"     -> {band}")
    print(f"     partly-hazardous C/A = {PH['mean']:.3f}; intracellular D/B = {Q1i['mean']:.3f}")
    print(f"Q-3  Motility alone C/A = {Q3['mean']:.3f} [{Q3['ci'][0]:.3f}, {Q3['ci'][1]:.3f}]"
          f"  (n={res['cell_n']['motility|extracellular']})")
    print(f"Q-4  non-hazardous / partly-hazardous: extra {Q4['extracellular']['mean']:.3f}, "
          f"intra {Q4['intracellular']['mean']:.3f}")
    short = [k for k, ok in floors.items() if not ok and k.split("|")[0] != "ambiguous"]
    if short:
        print(f"     ⚠️  below the floor of {FLOOR}, indicative only: {', '.join(short)}")
    if not res["split_happened"]:
        print("     🔴 the two groups returned the same rate — the split did not happen")
    print(f"\nwrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
