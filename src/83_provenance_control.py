#!/usr/bin/env python3
"""
83_provenance_control.py - is it virulence, or is it just pathogen origin?

`docs/PROVENANCE_CONTROL_PREREGISTRATION.md`. A1 found the probe flags **40.84%** of non-toxin
virulence factors at a nominal 5% against **5.98%** on the Swiss-Prot pool, and left two readings
undecided: the probe responds to virulence, or it responds to pathogen origin. The control that
separates them is **pathogen-derived proteins that are not virulence factors**, and the repository
already has 106 of them — v3 negatives from the 20 species that also contribute study B
representatives, with any exact VFDB sequence match excluded.

⚠️ No new probe. Study B's fold is used unchanged and a third population is scored at the same
thresholds, so the only thing differing between the three populations is what the proteins are.

Usage:
    python src/83_provenance_control.py --arm esm2_650M
    python src/83_provenance_control.py --arm esm2_35M
"""

import argparse
import collections
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
SEQ = ROOT / "data" / "sequences"
BUILD = ROOT / "results" / "external_class_axis_build.json"
SCREEN = ROOT / "results" / "external_class_axis_screen.json"
OUT_STEM = ROOT / "results" / "provenance_control"
SEEDS, N_TRAIN, N_CAL, SPEC = 30, 1000, 500, 0.95
OS_RE = re.compile(r"OS=([A-Z][a-z]+ [a-z]+)")


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))


def read_fasta(path, headers=False):
    out, h, b = [], None, []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith(">"):
            if h is not None:
                out.append((h, "".join(b)))
            h, b = (line if headers else line[1:].split()[0]), []
        elif h is not None:
            b.append(line.strip())
    if h is not None:
        out.append((h, "".join(b)))
    return out


def vfdb_sequences():
    seqs, cur = set(), []
    for name in ("VFDB_setA_pro.fas", "VFDB_setB_pro.fas"):
        for line in (ROOT / "data/external/vfdb" / name).read_text(errors="replace").splitlines():
            if line.startswith(">"):
                if cur:
                    seqs.add("".join(cur))
                cur = []
            else:
                cur.append(line.strip())
        if cur:
            seqs.add("".join(cur))
            cur = []
    return seqs


def vfdb_species():
    r"""The organism of every VFDB record, at species resolution.

    \U0001f534 A VFDB header carries TWO bracket groups -- `[VF name (VFxxxx) - Category (VFCxxxx)]`
    and then `[Organism strain]` -- and src/85 through src/88 each had their own copy of
    `re.search(r"\[([A-Z][a-z]+ [a-z]+)")`, which takes the FIRST match. Any VF name shaped
    "Capitalized lowercase" therefore won a header outright: "Accessory secretion", "Acid phosphatase",
    "Adhesive fimbriae". 12.0% of records captured a non-organism, the set held 133 such strings and
    was MISSING 11 real pathogens -- Helicobacter bilis, H. canis, H. cinaedi, H. pullorum among them --
    whose proteins were then scored benign. Entry 58.

    The organism is the LAST bracket group. Defined here, once, because four scripts sharing a
    regex by copy is how they came to share a bug.
    """
    out = set()
    for name in ("VFDB_setA_pro.fas", "VFDB_setB_pro.fas"):
        for line in (ROOT / "data/external/vfdb" / name).read_text(errors="replace").splitlines():
            if line.startswith(">"):
                groups = re.findall(r"\[([^\]]*)\]", line)
                if groups:
                    out.add(" ".join(groups[-1].split()[:2]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="esm2_650M", choices=["esm2_650M", "esm2_35M"])
    ap.add_argument("--seeds", type=int, default=SEEDS)
    a = ap.parse_args()
    sfx = "" if a.arm == "esm2_650M" else f"_{a.arm}"

    P = np.load(RES / f"embeddings_class_axis_positives{sfx}.npy")
    POOL = np.load(RES / f"embeddings_pool_large_{a.arm}.npy")
    NEG = np.load(RES / f"embeddings_negative_v3{sfx}.npy")
    VFDB = np.load(RES / "embeddings_vfdb_neg_esm2_650M.npy") if a.arm == "esm2_650M" else None
    man = json.loads((RES / f"embedding_manifest_class_axis_positives{sfx}.json").read_text())
    build = json.loads(BUILD.read_text())
    screen = json.loads(SCREEN.read_text())

    # ---- the control population ----------------------------------------------------------------
    negs = read_fasta(SEQ / "benign_negatives_v3.fasta", headers=True)
    if len(negs) != NEG.shape[0]:
        raise SystemExit(f"negatives FASTA has {len(negs)} rows, embedding has {NEG.shape[0]}")
    pos_species = set(man["species_of_row"])
    vf = vfdb_sequences()
    keep, dropped_vfdb, orgs = [], [], []
    for i, (h, s) in enumerate(negs):
        m = OS_RE.search(h)
        sp = m.group(1) if m else "?"
        if sp not in pos_species:
            continue
        if s in vf:
            dropped_vfdb.append(h.split()[0])
            continue
        keep.append(i)
        orgs.append(sp)
    print(f"arm {a.arm}: control population {len(keep)} pathogen-derived non-VFDB proteins "
          f"from {len(set(orgs))} species")
    print(f"  excluded {len(dropped_vfdb)} that match a VFDB sequence exactly — positives in a "
          f"negative set")

    union = screen["admitted_rows"]
    test = build["pool_partition"]["test_rows"]
    n_tr, n_ca = (N_TRAIN, N_CAL) if len(union) >= N_TRAIN + N_CAL else \
        (2 * len(union) // 3, len(union) - 2 * len(union) // 3)

    # ---- study B's fold, unchanged, scoring three populations ----------------------------------
    rates = {k: [] for k in ("pool_test", "pathogen_nonvf", "vfdb")}
    t0 = time.time()
    y = np.r_[np.ones(P.shape[0]), np.zeros(n_tr)]
    for seed in range(a.seeds):
        r = np.random.default_rng(seed)
        perm = r.permutation(len(union))
        tr = [union[i] for i in perm[:n_tr]]
        ca = [union[i] for i in perm[n_tr:n_tr + n_ca]]
        model = clf().fit(np.vstack([P, POOL[tr]]), y)
        t = float(np.quantile(model.predict_proba(POOL[ca])[:, 1], SPEC))
        rates["pool_test"].append(float((model.predict_proba(POOL[test])[:, 1] >= t).mean()))
        rates["pathogen_nonvf"].append(float((model.predict_proba(NEG[keep])[:, 1] >= t).mean()))
        if VFDB is not None:
            rates["vfdb"].append(float((model.predict_proba(VFDB)[:, 1] >= t).mean()))
    print(f"  {time.time() - t0:.0f}s")

    def stat(v):
        return {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                "ci": [float(np.mean(v) - 1.96 * np.std(v, ddof=1) / len(v) ** 0.5),
                       float(np.mean(v) + 1.96 * np.std(v, ddof=1) / len(v) ** 0.5)]}

    res = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "arm": a.arm, "seeds": a.seeds,
           "nominal": 1 - SPEC, "n_control": len(keep),
           "control_species": len(set(orgs)),
           "excluded_as_vfdb": dropped_vfdb,
           "species_counts": dict(collections.Counter(orgs).most_common()),
           "rates": {k: stat(v) for k, v in rates.items() if v}}
    dest = Path(f"{OUT_STEM}{sfx}.json")
    dest.write_text(json.dumps(res, indent=2) + "\n")

    print(f"\n{'population':<40}{'n':>7}{'flag rate at a nominal 5%':>30}")
    print("-" * 78)
    for k, lbl, n in (("pool_test", "Swiss-Prot pool, test partition", len(test)),
                      ("pathogen_nonvf", "pathogen-derived, NOT in VFDB", len(keep)),
                      ("vfdb", "VFDB non-toxin virulence factors", 4218)):
        if k in res["rates"]:
            s = res["rates"][k]
            print(f"  {lbl:<38}{n:>7}{s['mean'] * 100:>18.2f}% "
                  f"[{s['ci'][0] * 100:.2f}, {s['ci'][1] * 100:.2f}]")
    print(f"\nwrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
