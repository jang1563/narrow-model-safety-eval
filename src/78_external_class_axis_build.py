#!/usr/bin/env python3
"""
78_external_class_axis_build.py - build study B's positives and the pool partition it needs.

Study B
-------
`docs/EXTERNAL_CLASS_AXIS_PREREGISTRATION.md`. Class-level **margin** ranks mechanism classes by
leave-one-class-out recovery at Spearman +0.894 across fourteen model arms and fifteen re-poolings — and
every one of those tests is on a panel whose classes were assigned by hand in this repository. This study
asks the same question of **VFDB's fourteen curator-maintained categories**, which DeepVIC uses over
12,989 VFs while running no leave-one-category-out evaluation of any kind.

This script builds and partitions. It fits nothing and measures nothing.

The two rules that were frozen, applied here unchanged
------------------------------------------------------
**Representatives.** VFDB lists orthologs of one virulence factor across organisms under a shared
`VF####` id; setA's redundancy factor is 6.37. One representative per group: **the longest sequence,
ties broken by the lowest VFG id.** Fixed in the preregistration and not tuned.

**Partition.** The 8,259-protein benign pool, contaminant `Q8X739` dropped, ordered by
`sha256(accession)` ascending and cut into **1,000 train / 500 calibrate / the rest test**. 🟢 This is
the partition criterion 1 of `docs/DETECTOR_CRITERIA.md` records as absent from the main panel
(178/118/**0**), so every false-positive figure in this study is out of sample by construction.

⚠️ **The test partition is deliberately left unscreened** — a deployed screen does not get to remove the
proteins it will meet, and screening it would make the rate optimistic in the direction § 2.6.1 warns
about. Only train and calibrate are screened, by `src/79`.

Usage:
    python src/78_external_class_axis_build.py
"""

import collections
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VFDB = ROOT / "data" / "external" / "vfdb" / "VFDB_setA_pro.fas"
POOL_FASTA = ROOT / "data" / "sequences" / "benign_pool_large.fasta"
POOL_MAN = ROOT / "results" / "v3" / "embedding_manifest_pool_large_esm2_650M.json"
OUT_FASTA = ROOT / "data" / "sequences" / "vfdb_class_axis_positives.fasta"
OUT = ROOT / "results" / "external_class_axis_build.json"
CONTAMINANT = "Q8X739"
N_TRAIN, N_CAL = 1000, 500
FLOOR = 7            # the panel's own holdout-eligibility floor, from mechanism_classes_v3.json

HEADER = re.compile(
    r"^>(?P<vfg>VFG\d+)(?:\([^)]*\))?\s*(?:\((?P<gene>[^)]*)\)\s*)?(?P<name>.*?)\s*"
    r"\[[^\[\]]*?\((?P<vf>VF\d+)\)\s*-\s*(?P<cat>[^\[\]]*?)\(VFC\d+\)\]\s*\[(?P<org>[^\[\]]+)\]\s*$")


def read_fasta(path, keep_header=False):
    out, hid, buf = [], None, []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith(">"):
            if hid is not None:
                out.append((hid, "".join(buf)))
            hid, buf = (line if keep_header else line[1:].split()[0]), []
        elif hid is not None:
            buf.append(line.strip())
    if hid is not None:
        out.append((hid, "".join(buf)))
    return out


def main():
    # ---- positives: one representative per VF group -------------------------------------------
    rows, bad = [], 0
    for h, s in read_fasta(VFDB, keep_header=True):
        m = HEADER.match(h)
        if not m:
            bad += 1
            continue
        d = m.groupdict()
        d["cat"], d["org"] = d["cat"].strip(), d["org"].strip()
        d["species"] = " ".join(d["org"].split()[:2])
        d["seq"] = s
        rows.append(d)
    if bad:
        raise SystemExit(f"{bad} unparsed VFDB headers; the representative rule cannot be applied")

    best = {}
    for r in rows:
        k = best.get(r["vf"])
        if k is None or len(r["seq"]) > len(k["seq"]) or \
                (len(r["seq"]) == len(k["seq"]) and r["vfg"] < k["vfg"]):
            best[r["vf"]] = r
    rep = sorted(best.values(), key=lambda r: r["vfg"])
    cats = collections.Counter(r["cat"] for r in rep)
    eligible = sorted(c for c, n in cats.items() if n >= FLOOR)
    below = sorted(c for c, n in cats.items() if n < FLOOR)

    with OUT_FASTA.open("w") as fh:
        for r in rep:
            fh.write(f">{r['vfg']} {r['cat']} | {r['vf']} | {r['species']}\n")
            for i in range(0, len(r["seq"]), 60):
                fh.write(r["seq"][i:i + 60] + "\n")

    # ---- the pool partition --------------------------------------------------------------------
    pman = json.loads(POOL_MAN.read_text())
    ids = [x if isinstance(x, str) else x.get("acc", x.get("id")) for x in pman["rows"]]
    pool_seq = dict(read_fasta(POOL_FASTA))
    if [k for k in pool_seq] != ids:
        raise SystemExit("pool FASTA order does not match the published manifest")

    def acc(x):
        p = x.split("|")
        return p[1] if len(p) > 2 else x

    rows_keep = [(i, x) for i, x in enumerate(ids) if acc(x) != CONTAMINANT]
    if len(rows_keep) != len(ids) - 1:
        raise SystemExit(f"{CONTAMINANT} not found exactly once in the pool")
    order = sorted(rows_keep, key=lambda t: hashlib.sha256(acc(t[1]).encode()).hexdigest())
    train = [i for i, _ in order[:N_TRAIN]]
    calib = [i for i, _ in order[N_TRAIN:N_TRAIN + N_CAL]]
    test = [i for i, _ in order[N_TRAIN + N_CAL:]]
    assert len(set(train) & set(calib)) == 0 and len(set(train) & set(test)) == 0
    assert len(train) + len(calib) + len(test) == len(ids) - 1

    OUT.write_text(json.dumps({
        "built": "2026-09-28",
        "purpose": "study B of docs/EXTERNAL_CLASS_AXIS_PREREGISTRATION.md. Builds and partitions; "
                   "fits nothing, measures nothing.",
        "representative_rule": "longest sequence per VF#### group, ties by lowest VFG id",
        "n_records": len(rows), "n_representatives": len(rep),
        "redundancy_factor": round(len(rows) / len(rep), 3),
        "eligibility_floor": FLOOR,
        "categories": dict(cats.most_common()),
        "eligible_categories": eligible, "below_floor": below,
        "n_in_eligible": sum(cats[c] for c in eligible),
        "n_species": len(set(r["species"] for r in rep)),
        "positives": [{"vfg": r["vfg"], "vf": r["vf"], "cat": r["cat"],
                       "species": r["species"], "len": len(r["seq"])} for r in rep],
        "pool_partition": {
            "rule": "sha256(accession) ascending over the pool minus the contaminant",
            "contaminant_dropped": CONTAMINANT,
            "n_train": len(train), "n_calibrate": len(calib), "n_test": len(test),
            "train_rows": train, "calibrate_rows": calib, "test_rows": test,
            "test_is_unscreened": True,
            "note": "row indices into embeddings_pool_large_esm2_650M.npy as published",
        },
    }, indent=2) + "\n")

    print(f"positives: {len(rows)} records -> {len(rep)} representatives "
          f"({len(rows) / len(rep):.2f}x redundancy), {len(set(r['species'] for r in rep))} species")
    print(f"  {len(eligible)} categories at or above the floor of {FLOOR}, "
          f"holding {sum(cats[c] for c in eligible)}")
    print(f"  below the floor, reported and not held out: {below} "
          f"({[cats[c] for c in below]})")
    for c in eligible:
        print(f"    {c:<46}{cats[c]:>5}")
    pp = json.loads(OUT.read_text())["pool_partition"]
    print(f"\npool: {len(ids)} - 1 contaminant -> {pp['n_train']} train / "
          f"{pp['n_calibrate']} calibrate / {pp['n_test']} test (unscreened by design)")
    print(f"wrote {OUT_FASTA.relative_to(ROOT)} and {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
