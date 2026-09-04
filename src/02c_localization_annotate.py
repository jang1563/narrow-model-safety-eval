#!/usr/bin/env python3
"""
02c_localization_annotate.py - subcellular localization for the whole v2 panel.

Why this exists
---------------
The 2026-09-04 rerun showed that with organism held constant, ESM-2 separates
secreted from cytoplasmic negatives at AUROC 1.000. Every pathogen-derived
negative is cytoplasmic while the positives are largely secreted or exported, so
`pathogen_matched` controls organism but not localization and `secreted_only`
controls localization but not organism. No control holds both.

Worse, the positives carried NO localization annotation at all, so the size of
that confound could not even be measured. This script annotates both sides from
UniProt so the confound can be quantified before anything is done about it.

Nothing here is derived from the panel's own labels: localization comes from
UniProt, the hazard label comes from the panel, and they are only crossed
afterwards.

Derivation rule, applied in this order to the SUBCELLULAR LOCATION comment.
The order matters and is deliberate: a protein annotated both "Secreted" and
"Cytoplasm" is counted as secreted, because the exported pool is the one that
matters for the confound.

    secreted     any location containing "Secreted"
    cell_surface any location containing "Cell wall", "Cell surface" or "Fimbrium"
    membrane     any location containing "Membrane"
    periplasm    any location containing "Periplasm"
    cytoplasmic  any location containing "Cytoplasm" or "Cytosol"
    other        a location comment exists but matches none of the above
    unknown      no SUBCELLULAR LOCATION comment

`has_signal_peptide` is recorded separately from the Signal feature. It is the
more objective secretion indicator of the two and is what the confound analysis
should prefer, since it does not depend on curator phrasing.

Inputs : data/sequences/{toxins_positive_v2,benign_negatives_v2}.fasta
Output : data/annotations/localization_v2.json

Usage:
    python src/02c_localization_annotate.py
"""

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "annotations" / "localization_v2.json"
API = "https://rest.uniprot.org/uniprotkb/accessions"
BATCH = 50

# (label, substrings). First match in list order wins.
RULES = [
    ("secreted", ("secreted",)),
    ("cell_surface", ("cell wall", "cell surface", "fimbrium")),
    ("membrane", ("membrane",)),
    ("periplasm", ("periplasm",)),
    ("cytoplasmic", ("cytoplasm", "cytosol")),
]


def fasta_ids(path):
    return [ln[1:].split()[0] for ln in open(path) if ln.startswith(">")]


def fetch(accs):
    """Return {accession: {"locations": [...], "has_signal_peptide": bool}}."""
    out = {}
    for i in range(0, len(accs), BATCH):
        chunk = accs[i : i + BATCH]
        url = (
            API
            + "?"
            + urllib.parse.urlencode(
                {
                    "accessions": ",".join(chunk),
                    "fields": "accession,cc_subcellular_location,ft_signal",
                    "format": "json",
                }
            )
        )
        with urllib.request.urlopen(url, timeout=60) as r:
            data = json.load(r)
        for rec in data.get("results", []):
            locs = []
            for c in rec.get("comments", []):
                if c.get("commentType") == "SUBCELLULAR LOCATION":
                    for sl in c.get("subcellularLocations", []):
                        v = (sl.get("location") or {}).get("value")
                        if v:
                            locs.append(v)
            out[rec["primaryAccession"]] = {
                "locations": locs,
                "has_signal_peptide": any(
                    f.get("type") == "Signal" for f in rec.get("features", [])
                ),
            }
        print(f"  fetched {min(i + BATCH, len(accs))}/{len(accs)}")
        time.sleep(0.3)
    return out


def derive(locations):
    low = " | ".join(locations).lower()
    if not locations:
        return "unknown"
    for label, keys in RULES:
        if any(k in low for k in keys):
            return label
    return "other"


def main():
    sides = {
        "positives": fasta_ids(ROOT / "data/sequences/toxins_positive_v2.fasta"),
        "negatives": fasta_ids(ROOT / "data/sequences/benign_negatives_v2.fasta"),
    }
    acc_of = {fid: fid.split("|")[1] for ids in sides.values() for fid in ids}
    print(f"annotating {len(acc_of)} proteins from UniProt")
    raw = fetch(sorted(set(acc_of.values())))

    ann = {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "UniProtKB cc_subcellular_location + ft_signal",
        "rule_order": [r[0] for r in RULES],
        "note": "localization is fetched independently of the panel's hazard label; "
        "the two are crossed only in analysis",
        "proteins": {},
    }
    missing = []
    for side, ids in sides.items():
        for fid in ids:
            a = acc_of[fid]
            r = raw.get(a)
            if r is None:
                missing.append(a)
                r = {"locations": [], "has_signal_peptide": False}
            ann["proteins"][fid] = {
                "acc": a,
                "side": side,
                "locations": r["locations"],
                "has_signal_peptide": r["has_signal_peptide"],
                "localization": derive(r["locations"]),
            }
    ann["not_found_in_uniprot"] = missing

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(ann, open(OUT, "w"), indent=2)

    # counts only; no identifiers printed
    import collections

    print()
    for side in sides:
        c = collections.Counter(
            v["localization"] for v in ann["proteins"].values() if v["side"] == side
        )
        sig = sum(
            1 for v in ann["proteins"].values() if v["side"] == side and v["has_signal_peptide"]
        )
        print(f"{side:<10} n={len(sides[side]):<4} signal_peptide={sig:<4} {dict(c)}")
    if missing:
        print(f"\nnot found in UniProt: {len(missing)}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
