#!/usr/bin/env python3
"""
60_negative_test_partition.py - freeze a standing test partition for the negatives, so the panel
                                has one instead of being told it needs one.

The failure this addresses
--------------------------
`docs/DETECTOR_CRITERIA.md` scores criterion 1 as this project's worst, and criterion 1 is the split:
the panel's negatives divide 178 train / 118 calibrate / **0 test**, so every false-positive rate
ever published here was measured on negatives the pipeline had already seen. `src/48` and `src/49`
measured what an honest rate looks like, on two routes, and the scorecard still reads "measured, not
repaired" because both were analysis-time draws. Nothing in the repository *is* a test partition.

This makes one and freezes it.

Why the pool rather than a carve-out of the panel
-------------------------------------------------
`src/48` carved the test set out of the panel's own 296 negatives and it cost two things at once:
calibration halved from 118 to 59, which broke comparability with every published figure, and the
conformal threshold went out of reach at a nominal 1% because `floor((m+1)*alpha)` is zero at m = 59.
Both costs come from *where* the test set was taken. Taking it from the 8,259-protein benign pool
leaves calibration at the published 118 and keeps conformal reachable at both budgets.

The admission rule, stated because criterion 18 says the curator is a trusted party
-----------------------------------------------------------------------------------
A pool protein is admitted to the test partition when all of:

 1. it is in `data/sequences/benign_pool_large.fasta`, the reviewed Swiss-Prot set `src/34` built
    under its own length and keyword rule,
 2. its accession is in neither the panel's positive nor its negative set,
 3. its maximum normalized local-alignment similarity to any panel positive is **below the maximum
    the panel's own negatives reach under the same screen**, which is 0.282 and is read from
    `results/v3/pool_homology_against_panel.json` rather than hardcoded.

⚠️ Rule 3 is deliberately **tighter than the 0.30 rule** that governs positives against positives.
`src/27`'s policy is that negatives are not screened against positives at all, because
mechanism-matched benign proteins are wanted as hard negatives, and `src/42` follows it. That policy
is right for the panel's curated negatives and wrong for a test partition: a test negative more
similar to a hazard than any of the panel's own negatives is a different kind of object from the ones
the threshold was set on, and admitting it would let the measured false-positive rate be argued about.
Bounding at the panel's own maximum means the test set is no closer to the hazards than the
calibration set already is.

**The partition is test-only.** It must never be trained on and never calibrated on. Criterion 18's
attack works by supplying negatives that the fit sees; a set that only ever appears at evaluation
time cannot suppress a class's recovery, it can only change the rate that is reported. That
asymmetry is the reason this is the safe half of the repair and the reason the file says so.

What this does not fix
----------------------
Calibration resolution. It stays at 118 points, so the finest false-positive rate the *threshold* can
express is unchanged; what improves is the resolution of the rate that is *measured*, from 1/296 to
1/3,407 by distinct name. And the conformal guarantee is **voided by design** here: the panel's
negatives are three curated blocks and the pool is Swiss-Prot, 87% bacterial, with a name redundancy
factor of 2.33, so the two are not exchangeable. `src/49` measured that cost rather than assuming it
away, and the honest reading of a panel-to-pool rate is as the deployment-shift number.

Usage:
    python src/60_negative_test_partition.py
    python src/60_negative_test_partition.py --verify     # check the frozen file still holds
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POOL_FASTA = ROOT / "data" / "sequences" / "benign_pool_large.fasta"
SCREEN = ROOT / "results" / "v3" / "pool_homology_against_panel.json"
MANIFEST = ROOT / "data" / "sequences" / "panel_v3_manifest.json"
OUT = ROOT / "data" / "sequences" / "negative_test_partition_v3.json"


def acc_of(header):
    p = header.split("|")
    return p[1] if len(p) > 1 else header


def read_pool():
    """Accession -> (header, name). The sequence itself is not needed or stored here."""
    out = {}
    for line in open(POOL_FASTA):
        if line.startswith(">"):
            h = line[1:].strip()
            out[acc_of(h)] = (h.split()[0], h.split(" ", 1)[1].split(" OS=")[0]
                              if " " in h else "")
    return out


def build():
    screen = json.load(open(SCREEN))
    if "pool_max_similarity" not in screen:
        print("XX the screen artifact has no per-protein maxima. Re-run "
              "src/42_pool_homology_against_panel.py, which dumps them as of 2026-09-27; without "
              "them only the 0.30 bound can be applied and rule 3 needs 0.282.")
        return None

    bound = screen["panel_negative_reference"]["max"]
    sim = {r["pool_acc"]: r for r in screen["pool_max_similarity"]}
    pool = read_pool()
    man = json.load(open(MANIFEST))
    panel_neg = {x["acc"].split("|")[1] if "|" in x["acc"] else x["acc"] for x in man["negatives"]}
    panel_pos = {x["acc"].split("|")[1] if "|" in x["acc"] else x["acc"] for x in man["positives"]}

    missing = [a for a in pool if a not in sim]
    if missing:
        print(f"XX {len(missing)} pool proteins have no screened similarity, e.g. {missing[:3]}. "
              f"The screen and the FASTA disagree, so the rule cannot be applied to all of them.")
        return None

    admitted, rejected = [], []
    for a, (hdr, name) in sorted(pool.items()):
        s = sim[a]["similarity"]
        if a in panel_pos:
            rejected.append({"acc": a, "reason": "in the panel's positive set", "similarity": s})
        elif a in panel_neg:
            rejected.append({"acc": a, "reason": "in the panel's negative set", "similarity": s})
        elif s >= bound:
            rejected.append({"acc": a, "reason": f"similarity {s:.4f} >= the panel negatives' own "
                                                 f"maximum {bound:.4f}",
                             "similarity": s, "nearest_positive": sim[a]["positive"],
                             "nearest_positive_class": sim[a]["positive_class"]})
        else:
            admitted.append({"acc": a, "name": name, "max_similarity_to_panel_positive": s})

    # 🔴 Name grouping is src/36's, lowercased description, 2026-09-27. Three figures for "distinct
    # names in this pool" already existed and they measure different things: src/36 gets 3,550 by
    # lowercased description, src/49 gets 3,407 by gene symbol (`sp|ACC|GENE_ORGANISM` -> GENE),
    # and the first version of this function got 3,552 by NOT lowercasing. The first two are both
    # valid effective sizes of different quantities and must not be conflated; the third was a bug.
    names = {}
    for r in admitted:
        names.setdefault(r["name"].strip().lower(), []).append(r["acc"])
    genes = {}
    for r in admitted:
        h = pool[r["acc"]][0].split("|")
        genes.setdefault(h[2].split("_")[0] if len(h) > 2 else r["acc"], []).append(r["acc"])
    digest = hashlib.sha256(
        "\n".join(sorted(r["acc"] for r in admitted)).encode()).hexdigest()

    return {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "role": "TEST ONLY. Never trained on, never calibrated on. See the module docstring.",
        "panel": "v3",
        "source": "data/sequences/benign_pool_large.fasta",
        "rule": {
            "1": "in the reviewed Swiss-Prot pool built by src/34",
            "2": "accession in neither the panel's positive nor its negative set",
            "3": f"max normalized local-alignment similarity to any panel positive < {bound:.6f}, "
                 f"the maximum the panel's own negatives reach under the same screen",
            "screen": screen["aligner"],
            "bound_source": "results/v3/pool_homology_against_panel.json "
                            "panel_negative_reference.max",
            "why_tighter_than_0.30": "src/27's policy leaves negatives unscreened against positives "
                                     "so that mechanism-matched benign proteins can serve as hard "
                                     "negatives. That is right for curated calibration negatives "
                                     "and wrong for a test partition, where a member closer to a "
                                     "hazard than any calibration negative is a different object.",
        },
        "counts": {"pool": len(pool), "admitted": len(admitted), "rejected": len(rejected),
                   "distinct_names": len(names), "distinct_gene_symbols": len(genes)},
        "name_grouping": (
            "distinct_names is src/36's definition, the lowercased protein description, which gives "
            "3550 over the undeduplicated pool. distinct_gene_symbols is src/49's, the gene prefix "
            "of sp|ACC|GENE_ORGANISM, which gives 3407. They are effective sizes of different "
            "quantities and are both reported so neither is mistaken for the other."),
        "resolution": {
            "finest_rate_by_count": 1 / len(admitted),
            "finest_rate_by_distinct_name": 1 / len(names),
            "finest_rate_by_gene_symbol": 1 / len(genes),
            "panel_negatives_for_comparison": len(panel_neg),
            "panel_finest_rate": 1 / len(panel_neg),
        },
        "exchangeability": (
            "VOIDED BY DESIGN. The panel's negatives are three curated blocks; this partition is "
            "Swiss-Prot, 87% bacterial, name redundancy 2.33. The conformal guarantee assumes "
            "exchangeability and does not hold across that shift, which is the deployment "
            "condition rather than a defect. src/49 measures the cost: at a nominal 5% the "
            "realized rate is 7.87% on the canonical arm and 9.64% over distinct names."),
        "does_not_fix": (
            "Calibration resolution, which stays at 118 points. What improves is the resolution of "
            "the rate that is measured, not of the threshold that is set."),
        "sha256_of_sorted_accessions": digest,
        "admitted": admitted,
        "rejected": rejected,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="rebuild and check the frozen file still matches, without writing")
    a = ap.parse_args()

    res = build()
    if res is None:
        return 2

    c, r = res["counts"], res["resolution"]
    print(f"pool {c['pool']}  admitted {c['admitted']}  rejected {c['rejected']}  "
          f"distinct names {c['distinct_names']}")
    print(f"rule 3 bound: {res['rule']['3']}")
    for row in res["rejected"]:
        print(f"  rejected {row['acc']}: {row['reason']}")
    print(f"\nfinest measurable rate: 1/{c['admitted']} by count, "
          f"1/{c['distinct_names']} by distinct name, 1/{c['distinct_gene_symbols']} by gene "
          f"symbol, against the panel's 1/{r['panel_negatives_for_comparison']}")
    print(f"sha256 {res['sha256_of_sorted_accessions'][:16]}")

    if a.verify:
        if not OUT.exists():
            print(f"XX {OUT.name} does not exist yet; run without --verify to create it")
            return 2
        old = json.load(open(OUT))
        same = old["sha256_of_sorted_accessions"] == res["sha256_of_sorted_accessions"]
        print(f"\nfrozen file {'MATCHES' if same else 'DIFFERS'}: stored "
              f"{old['sha256_of_sorted_accessions'][:16]}, rebuilt "
              f"{res['sha256_of_sorted_accessions'][:16]}")
        return 0 if same else 2

    json.dump(res, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
