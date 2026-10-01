#!/usr/bin/env python3
"""Study AI, build step: partition DeepVIC's published benchmark and price the overlap.

`docs/DEEPVIC_BENCHMARK_PREREGISTRATION.md` (written after this step, which fits nothing).

§ 1.2 of the paper states the comparison problem: DeepVIC is the largest virulence-factor
classifier in the literature, it carries a published 14-category class axis over 12,989
annotated VFs, and it **runs no leave-one-category-out evaluation**.  Its repository ships
the sequences, the class labels and the trained weights, so the class-level question this
project exists to ask can be asked on someone else's benchmark with someone else's labels.

This step only partitions and measures composition.  🔴 It also prices the one thing that
could invalidate the whole arm: how much of DeepVIC's positive set is the VFDB material this
project's own panels were drawn from.  An overlap that large would make a probe trained here
and tested there circular, and the number has to exist before any result does.
"""
import csv
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DV = ROOT / "data/external/deepvic"
FAA = DV / "DLDB_33456.faa"
META = DV / "metadata.txt"
WEIGHTS = (DV / "binary.keras", DV / "multiclass.keras")
OURS = ROOT / "results/external_class_axis_build.json"
VFDB_FAS = ROOT / "data/external/vfdb/VFDB_setA_pro.fas"
OUT = ROOT / "results/deepvic_build.json"
CLASS_RE = re.compile(r"\s*\(VFC\d+\)\s*$")
VFG_RE = re.compile(r"VFG\d+")


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_fasta(p):
    """id -> (header, length). The id is the first whitespace-delimited field."""
    out, key, hdr, n = {}, None, None, 0
    with p.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith(">"):
                if key is not None:
                    out[key] = (hdr, n)
                hdr = line[1:].rstrip("\n")
                key = re.split(r"[\t ]", hdr, maxsplit=1)[0].strip()
                n = 0
            else:
                n += len(line.strip())
    if key is not None:
        out[key] = (hdr, n)
    return out


def meta():
    with META.open(encoding="utf-8", errors="replace") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def selftest():
    ok = True
    for f in (FAA, META, *WEIGHTS):
        if not f.exists():
            print(f"  XX {f.name} missing")
            ok = False
    if not ok:
        print("selftest FAIL")
        return 1

    md = meta()
    if len(md) != 33456:
        print(f"  XX metadata has {len(md)} rows, DeepVIC states 33,456")
        ok = False
    fa = read_fasta(FAA)
    if len(fa) != 66912:
        print(f"  XX fasta has {len(fa)} records, expected 66,912 (33,456 x 2)")
        ok = False
    # 🔴 the positive/negative split must come from the metadata join, not from the id
    # prefix: if every DLDB_* id were assumed positive, a metadata row that is absent from
    # the fasta would be silently counted. Both directions are checked.
    ids = {r["DLDB header/Internal Reference"] for r in md}
    pos = [k for k in fa if k in ids]
    if len(pos) != len(md):
        print(f"  XX {len(pos)} fasta records join the metadata, {len(md)} rows exist")
        ok = False
    if len(fa) - len(pos) != len(md):
        print(f"  XX {len(fa) - len(pos)} negatives against {len(md)} positives")
        ok = False

    # the class column must parse: 14 real classes plus NA, and NA must be the majority
    cl = Counter(CLASS_RE.sub("", r["VF class"]).strip() for r in md)
    if len(cl) != 15 or "NA" not in cl:
        print(f"  XX {len(cl)} distinct VF classes, expected 14 + NA: {sorted(cl)}")
        ok = False
    if cl.get("NA", 0) != 20467:
        print(f"  XX NA count {cl.get('NA')}, expected 20,467")
        ok = False
    # and the regex must actually strip a code, or every class name keeps its VFC suffix
    if CLASS_RE.sub("", "Exotoxin (VFC0235)").strip() != "Exotoxin":
        print("  XX the VFC code is not being stripped")
        ok = False
    if CLASS_RE.sub("", "Exotoxin") != "Exotoxin":
        print("  XX stripping damages a name that carries no code")
        ok = False

    # a length of zero would mean the fasta parser never accumulated residues
    if min(n for _, n in fa.values()) < 10:
        print("  XX a fasta record came out under 10 residues")
        ok = False

    print("selftest PASS" if ok else "selftest FAIL")
    return 0 if ok else 1


def main():
    md = meta()
    fa = read_fasta(FAA)
    ids = {r["DLDB header/Internal Reference"]: r for r in md}
    pos = [k for k in fa if k in ids]
    neg = [k for k in fa if k not in ids]

    cls = {k: CLASS_RE.sub("", ids[k]["VF class"]).strip() for k in pos}
    labelled = [k for k in pos if cls[k] != "NA"]
    counts = Counter(cls[k] for k in labelled)

    # 🔴 the circularity price: how much of DeepVIC's positive set is the VFDB material this
    # project's panels were drawn from, and how much is OUR OWN 746 representatives.
    dv_vfg = {}
    for k in pos:
        m = VFG_RE.search(ids[k]["Original Database header"] or "")
        if m:
            dv_vfg[k] = m.group(0)
    ours = json.loads(OURS.read_text())
    our_reps = {r["vfg"] for r in ours["positives"]}
    setA = set(VFG_RE.findall(VFDB_FAS.read_text(encoding="utf-8", errors="replace")))
    shared_reps = {v for v in dv_vfg.values() if v in our_reps}
    shared_setA = {v for v in dv_vfg.values() if v in setA}

    res = {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "study": "AI build -- DeepVIC's published benchmark, partitioned and priced",
        "source": "https://github.com/ansontwk/DeepVIC (src/data, src/models)",
        "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)}
                  for p in (FAA, META, *WEIGHTS)},
        "n_fasta": len(fa),
        "n_positives": len(pos),
        "n_negatives": len(neg),
        "n_labelled_positives": len(labelled),
        "n_class_NA": len(pos) - len(labelled),
        "source_databases": dict(Counter(ids[k]["Database"] for k in pos).most_common()),
        "classes": dict(counts.most_common()),
        "n_classes": len(counts),
        "length": {
            "positives_median": sorted(fa[k][1] for k in pos)[len(pos) // 2],
            "negatives_median": sorted(fa[k][1] for k in neg)[len(neg) // 2],
            "labelled_median": sorted(fa[k][1] for k in labelled)[len(labelled) // 2],
        },
        "overlap_with_this_project": {
            "deepvic_positives_carrying_a_VFG_id": len(dv_vfg),
            "distinct_VFG_ids_in_deepvic": len(set(dv_vfg.values())),
            "in_VFDB_setA": len(shared_setA),
            "in_our_746_representatives": len(shared_reps),
            "frac_of_our_746": len(shared_reps) / len(our_reps),
            "frac_of_deepvic_positives": len(shared_reps) / len(pos),
            "note": ("the circularity price. A probe fitted on this project's VFDB panel and "
                     "tested on DeepVIC shares exactly these proteins; the preregistration "
                     "has to say what is done about them before any result exists."),
        },
    }
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print(f"positives {len(pos):,}  negatives {len(neg):,}  labelled {len(labelled):,} "
          f"in {len(counts)} classes")
    print(f"source dbs: {res['source_databases']}")
    print(f"median length pos {res['length']['positives_median']} / "
          f"neg {res['length']['negatives_median']} / "
          f"labelled {res['length']['labelled_median']}")
    o = res["overlap_with_this_project"]
    print(f"VFG ids in DeepVIC: {o['distinct_VFG_ids_in_deepvic']:,}  "
          f"of which in setA {o['in_VFDB_setA']:,}  "
          f"in OUR 746 reps {o['in_our_746_representatives']} "
          f"({o['frac_of_our_746']:.1%} of ours, "
          f"{o['frac_of_deepvic_positives']:.2%} of theirs)")
    for c, n in counts.most_common():
        print(f"    {c:<48}{n:>7,}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else main())
