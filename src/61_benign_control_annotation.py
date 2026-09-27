#!/usr/bin/env python3
"""
61_benign_control_annotation.py - make the benign controls usable as sequences, which is what P2 of
                                  the mutation preregistration needs and does not have.

The gap
-------
`docs/MUTATION_EXTENSION_PREREGISTRATION.md` P2 tests whether FSPE-M measures hazard rather than
evolutionary constraint, by running the same computation on "the astacin, thermolysin, saporin-6 and
lysozyme controls already in the repository". § 4 calls P2 "the test most likely to fail, and the one
that decides whether this axis is worth building on". The 2026-09-27 prerequisite audit found it
blocked three ways:

  1. **Thermolysin is not in the repository at all.** Three of the four named controls are in
     `functional_sites.json` under `_benign_controls`; thermolysin is absent.
  2. **None of them has a sequence.** They entered as PDB structures for the FSI controls, so
     `data/sequences/` has no FASTA for them and masked prediction has nothing to mask.
  3. **Their positions carry `use_pdb_numbering: true`.** So each needs an identity-verified offset
     onto its UniProt sequence before it can be masked, which is the hazard entry sixteen of
     `DATA_CORRECTIONS.md` is about, now on the control side.

This script closes all three, and refuses rather than guesses.

The acceptance rule, the same one the panel had to meet
------------------------------------------------------
`src/46` requires that an offset be the **unique** integer placing **every** annotated residue
identity correctly. That rule is applied here unchanged. A control whose offset is not unique, or
where any identity fails, is reported and **excluded**, because the alternative is a control scored
at the wrong positions — which is exactly how the panel's own numbering defect survived for months.

Expected identities come from the annotation text for the three existing controls, which name the
residue (`His92`, `Glu93`, `Tyr72`, `Glu35`). For thermolysin, which has no annotation to read, they
come from **UniProt's own Active site and Binding site features**, so the positions are not asserted
from memory. That is the rule `src/52` applied when it checked the panel against UniProt.

Outputs
-------
  data/sequences/benign_controls.fasta          the sequences, canonical UniProt
  data/annotations/benign_control_sites.json    verified positions, offsets, and every rejection

Neither is consumed by anything yet. P2 is the consumer and is not run here: this script supplies an
input and takes no measurement, so it cannot be tuned by looking at a result.

Usage:
    python src/61_benign_control_annotation.py
    python src/61_benign_control_annotation.py --fetch    # refresh the UniProt cache
"""

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "uniprot_cache"
SITES = ROOT / "data" / "annotations" / "functional_sites.json"
OUT_FASTA = ROOT / "data" / "sequences" / "benign_controls.fasta"
OUT_JSON = ROOT / "data" / "annotations" / "benign_control_sites.json"

# three-letter to one-letter, for reading identities out of the annotation text
AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
       "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
       "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}

# The fourth control P2 names and the repository does not have. Its positions are NOT listed here:
# they are read from UniProt's own Active site and Binding site features below, because asserting
# catalytic positions from memory is what this project's standing rule forbids.
THERMOLYSIN = {"uniprot": "P00800", "pdb_id": "1LNF", "chain": "E",
               "name": "Thermolysin",
               "organism": "Bacillus thermoproteolyticus",
               "is_control_for": "P0DPI1",
               "mechanism_match": "HExxH zinc metalloprotease, thermolysin-like fold. Named by P2 "
                                  "of the mutation preregistration as a control for BoNT-A and "
                                  "absent from _benign_controls until 2026-09-27.",
               "bsl_level": 1}
MAX_OFFSET = 60          # signal peptides and propeptides in this size class

# 🔴 Ligand selection, 2026-09-27. The first run took every single-position Active site AND Binding
# site feature, which gave thermolysin 18 positions against astacin's 4 and lysozyme's 2. Reading
# them showed why: 13 of the 18 are structural **Ca(2+)** sites. Thermolysin is a calcium-stabilised
# thermophilic protease and UniProt annotates all four of its calcium sites, none of which is
# catalytic. Including them would have made one control's "functional sites" a mixture of catalysis
# and structural metal binding while the other three are catalysis only, and P2 compares MEANS
# across controls, so the mixture would have moved the comparison without anyone choosing that.
#
# The rule kept: every Active site, plus Binding sites whose ligand is the CATALYTIC metal. For a
# zinc metalloprotease that is Zn(2+); calcium and everything else are excluded and logged.
CATALYTIC_LIGANDS = ("Zn(2+)", "Mn(2+)", "Mg(2+)", "Fe(2+)", "Fe(3+)", "Cu(2+)", "Ni(2+)")
EXCLUDED_LIGAND_NOTE = ("structural or non-catalytic ligand; Ca(2+) in particular is a "
                        "thermostability site in this fold, not part of the active site")


def uniprot(acc, fetch=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / f"{acc}.json"
    if fetch or not dest.exists():
        url = f"https://rest.uniprot.org/uniprotkb/{acc}.json"
        with urllib.request.urlopen(url, timeout=60) as r:   # noqa: S310
            dest.write_bytes(r.read())
        time.sleep(0.3)
    return json.loads(dest.read_text())


def expected_from_text(residue_annotations):
    """{position: one-letter} parsed from annotations that name the residue, e.g. 'His92 — zinc'."""
    out = {}
    for pos, text in (residue_annotations or {}).items():
        m = re.match(r"\s*([A-Za-z]{3})\s*(\d+)", text)
        if m and m.group(1).upper() in AA3 and int(m.group(2)) == int(pos):
            out[int(pos)] = AA3[m.group(1).upper()]
    return out


def expected_from_uniprot(rec):
    """({position: one-letter}, [excluded]) from UniProt's own single-position features.

    Active sites always; Binding sites only for a catalytic metal. See CATALYTIC_LIGANDS.
    """
    seq = rec["sequence"]["value"]
    out, dropped = {}, []
    for f in rec.get("features", []):
        loc = f["location"]
        if loc["start"]["value"] != loc["end"]["value"]:
            continue
        p = loc["start"]["value"]
        if not 1 <= p <= len(seq):
            continue
        lig = (f.get("ligand") or {}).get("name")
        if f["type"] == "Active site":
            out[p] = seq[p - 1]
        elif f["type"] == "Binding site":
            if lig in CATALYTIC_LIGANDS:
                out[p] = seq[p - 1]
            elif p not in out:
                dropped.append({"position": p, "residue": seq[p - 1], "ligand": lig,
                                "reason": EXCLUDED_LIGAND_NOTE})
    # a position can be both, and a dropped one must not shadow a kept one
    dropped = [d for d in dropped if d["position"] not in out]
    return out, {d["position"]: d for d in dropped}


def unique_offset(seq, expected):
    """Offsets in [0, MAX_OFFSET] placing EVERY expected identity correctly. src/46's rule."""
    hits = []
    for off in range(0, MAX_OFFSET + 1):
        if all(1 <= p + off <= len(seq) and seq[p + off - 1] == aa for p, aa in expected.items()):
            hits.append(off)
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="refresh the UniProt cache")
    a = ap.parse_args()

    controls = dict(json.load(open(SITES))["_benign_controls"])
    controls.pop("_description", None)
    controls["1LNF"] = THERMOLYSIN

    verified, rejected, fasta = {}, [], []
    for key in sorted(controls):
        c = controls[key]
        acc = c["uniprot"]
        rec = uniprot(acc, a.fetch)
        seq = rec["sequence"]["value"]
        src = "annotation text"
        dropped = {}
        expected = expected_from_text((c.get("functional_sites") or {}).get("residue_annotations"))
        if not expected:
            expected, dropped = expected_from_uniprot(rec)
            src = "UniProt Active site features plus catalytic-metal Binding sites"

        print(f"--- {key} {c['name']} ({acc}, {len(seq)} aa)")
        if not expected:
            rejected.append({"key": key, "uniprot": acc,
                             "reason": "no expected residue identities available from either the "
                                       "annotation text or UniProt's own features"})
            print("    XX no expected identities, excluded")
            continue
        print(f"    expected from {src}: "
              f"{ {p: aa for p, aa in sorted(expected.items())} }")

        offs = unique_offset(seq, expected)
        if len(offs) != 1:
            rejected.append({"key": key, "uniprot": acc, "expected": expected,
                             "offsets_found": offs,
                             "reason": ("no offset places every identity correctly"
                                        if not offs else
                                        f"offset is not unique: {offs}. src/46's rule requires "
                                        f"exactly one, because two frames that both fit cannot be "
                                        f"distinguished from the identities alone")})
            print(f"    XX offset not unique ({offs or 'none found'}), excluded")
            continue

        off = offs[0]
        positions = sorted(p + off for p in expected)
        verified[key] = {
            "name": c["name"], "uniprot": acc, "pdb_id": c.get("pdb_id"),
            "chain": c.get("chain"), "organism": c.get("organism"),
            "is_control_for": c.get("is_control_for"),
            "mechanism_match": c.get("mechanism_match"),
            "sequence_length": len(seq),
            "identity_source": src,
            "excluded_non_catalytic_ligand_sites": sorted(dropped.values(),
                                                         key=lambda d: d["position"]),
            "annotated_positions": sorted(expected),
            "offset": off,
            "positions": positions,
            "residues": "".join(seq[p - 1] for p in positions),
            "catalytic_residues": positions,   # sequence coordinates, for a masking consumer
        }
        fasta.append((f"sp|{acc}|{key}_CONTROL {c['name']} OS={c.get('organism', '')}", seq))
        print(f"    OK offset +{off} unique, positions {positions} = "
              f"{verified[key]['residues']}")

    OUT_FASTA.write_text("".join(
        f">{h}\n" + "\n".join(s[i:i + 60] for i in range(0, len(s), 60)) + "\n"
        for h, s in fasta))
    json.dump({
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": ("sequence-coordinate catalytic positions for the benign controls P2 of "
                    "docs/MUTATION_EXTENSION_PREREGISTRATION.md names. Supplies an input; takes no "
                    "measurement."),
        "acceptance_rule": ("src/46's: the offset must be the unique integer in [0, "
                            f"{MAX_OFFSET}] placing EVERY expected residue identity correctly. A "
                            "control failing it is excluded, not rescored."),
        "identity_sources": ("the annotation text where it names residues (His92, Glu93, ...), "
                             "otherwise UniProt's own Active site features plus Binding sites whose "
                             "ligand is a catalytic metal. Positions are never asserted from "
                             "memory, and non-catalytic ligand sites are excluded and logged "
                             "per control rather than swept in."),
        "catalytic_ligands": list(CATALYTIC_LIGANDS),
        "n_verified": len(verified), "n_rejected": len(rejected),
        "verified": verified, "rejected": rejected,
    }, open(OUT_JSON, "w"), indent=2)

    print(f"\nverified {len(verified)}, rejected {len(rejected)}")
    for r in rejected:
        print(f"  {r['key']} ({r['uniprot']}): {r['reason']}")
    print(f"wrote {OUT_FASTA.name} and {OUT_JSON.name}")
    return 0 if verified else 2


if __name__ == "__main__":
    sys.exit(main())
