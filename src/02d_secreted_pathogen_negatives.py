#!/usr/bin/env python3
"""
02d_secreted_pathogen_negatives.py - fill the empty cell in the doubly controlled test.

Why this exists
---------------
`03d` showed that hazard separation survives with localization held constant, and
that the doubly controlled cell (organism AND localization both fixed) passes at
0.997 on the signal-peptide axis. But it passes on ONE side only: every one of the
50 pathogen-derived negatives lacks a signal peptide, so the exported half of that
test has n = 0 and cannot be run at all.

This script fetches the missing side: benign SECRETED proteins from the same
pathogen organisms that supply the positives. With those in the panel, the
comparison "positives against secreted benign proteins from the same pathogens"
becomes possible, which is the strongest available control.

Selection discipline
--------------------
The risk here is the opposite of the usual one. A secreted protein from a pathogen
is more likely to be a virulence factor than a cytoplasmic one, so a careless fetch
would put hazardous proteins into the NEGATIVE set and quietly destroy the labels.
Filters are therefore layered, and every rejection is recorded with its reason:

  1. query-side   reviewed only, signal peptide required, virulence/toxin/cytolysis/
                  hemolysis keywords excluded in the query itself
  2. species      the queried genus and species must appear in the RETURNED organism
                  string. Applied from the first pass, not retroactively. The earlier
                  pathogen-derived batch skipped this and shipped one mislabelled
                  entry that had to be found and corrected afterwards
  3. keyword      re-checked in Python, because query-side exclusion is not proof
  4. name         protein-name blocklist for virulence vocabulary the keywords miss
  5. dedup        accession and sequence sha256 against everything already in the panel
  6. cap          at most PER_ORG per organism, so one well-annotated organism cannot
                  dominate the block

Run in two stages. The default is a dry run that writes the staging file and prints
counts; nothing touches the panel until --integrate is passed.

    python src/02d_secreted_pathogen_negatives.py
    python src/02d_secreted_pathogen_negatives.py --integrate

Inputs : data/sequences/panel_v2_manifest.json, benign_negatives_v2.fasta
Outputs: data/sequences/_secreted_pathogen_fetch.json   (staging, always)
         data/sequences/benign_negatives_v2.fasta       (only with --integrate)
         data/sequences/panel_v2_manifest.json          (only with --integrate)
"""

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEQ = ROOT / "data" / "sequences"
PANEL = SEQ / "panel_v2_manifest.json"
NEG_FASTA = SEQ / "benign_negatives_v2.fasta"
STAGE = SEQ / "_secreted_pathogen_fetch.json"
API = "https://rest.uniprot.org/uniprotkb/search"

PER_ORG = 2
PER_ORG_FETCH = 40  # candidates to consider before filtering
BLOCK_KEYWORDS = {
    "KW-0843": "Virulence",
    "KW-0800": "Toxin",
    "KW-0204": "Cytolysis",
    "KW-0354": "Hemolysis",
}
NAME_BLOCK = (
    "toxin",
    "hemolysin",
    "haemolysin",
    "cytolysin",
    "leukocidin",
    "leucocidin",
    "adhesin",
    "invasin",
    "virulence",
    "lethal factor",
    "edema factor",
    "oedema factor",
    "protective antigen",
    "aerolysin",
    "listeriolysin",
    "streptolysin",
    "phospholipase",
    "coagulase",
    "intimin",
    "internalin",
    "hemagglutinin",
    "haemagglutinin",
)

# Terms belonging to the POSITIVE mechanism classes. A candidate matching any of
# these is rejected regardless of its keywords, because a negative drawn from a
# positive's own family is a label contradiction, not a benign control.
#
# This list exists because the first pass did not have it and the consequences were
# concrete: the UniProt Virulence and Toxin keyword exclusions passed six
# beta-lactamases (a positive class with 14 members) and two Mono-ADP-ribosyl-
# transferase C3 entries, C3 being an actual bacterial toxin. Query-side keyword
# exclusion is necessary and demonstrably not sufficient.
#
# Over-blocking is the intended bias. Secreted proteases and nucleases from a
# pathogen are genuinely ambiguous, since secreted_protease and nuclease_dnase_rnase
# are themselves positive classes, so they are excluded rather than argued about.
CLASS_BLOCK = (
    "lactamase",
    "beta-lactam",
    "penicillinase",
    "cephalosporinase",
    "carbapenemase",
    "adp-ribosyl",
    "ribosyltransferase",
    "nuclease",
    "dnase",
    "rnase",
    "deoxyribonuclease",
    "ribonuclease",
    "protease",
    "peptidase",
    "proteinase",
    "glycosidase",
    "ribosome-inactivating",
    "rrna n-glycosidase",
    "pore-forming",
    "perfringolysin",
    "pneumolysin",
    "superantigen",
    "enterotoxin",
    "type iii secretion",
    "t3ss",
    "secretion system",
    "effector",
    "neurotoxin",
    "contact-dependent",
)
KW_NAME_BLOCK = ("virulence", "toxin", "cytolysis", "hemolysis", "haemolysis")


def H(s):
    return hashlib.sha256(s.encode()).hexdigest()


def genus_species(org):
    """First two tokens of the organism string, which is the species-level name."""
    return " ".join(org.split()[:2])


def query(org):
    excl = " ".join(f"NOT keyword:{k}" for k in BLOCK_KEYWORDS)
    q = f'organism_name:"{genus_species(org)}" AND reviewed:true AND ft_signal:* {excl}'
    url = (
        API
        + "?"
        + urllib.parse.urlencode(
            {
                "query": q,
                "format": "json",
                "size": PER_ORG_FETCH,
                "fields": "accession,id,organism_name,protein_name,keyword,length,sequence,"
                "ft_signal,cc_subcellular_location",
            }
        )
    )
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r).get("results", [])


def protein_name(rec):
    d = rec.get("proteinDescription", {})
    for key in ("recommendedName", "submissionNames"):
        v = d.get(key)
        if isinstance(v, dict) and v.get("fullName"):
            return v["fullName"]["value"]
        if isinstance(v, list) and v and v[0].get("fullName"):
            return v[0]["fullName"]["value"]
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--integrate",
        action="store_true",
        help="write the accepted entries into the panel; default is a dry run",
    )
    args = ap.parse_args()

    panel = json.load(open(PANEL))
    have_acc = {e["acc"].split("|")[1] for e in panel["positives"] + panel["negatives"]}
    have_sha = {e["sha256"] for e in panel["positives"] + panel["negatives"]}
    pos_orgs = {e["organism"] for e in panel["positives"]}
    pdc_orgs = sorted(
        {e["organism"] for e in panel["negatives"] if e.get("pathogen_derived_control")}
    )
    # Prefer organisms that already anchor a pathogen-derived negative, then the rest.
    # Deduplicate by genus and species: the panel stores strain-level organism strings,
    # and several of them collapse to the same species. Without this the per-organism
    # cap is applied once per strain string rather than once per species, which is how
    # a cap of 2 first produced 90 accepted entries across 27 species.
    seen_gs, targets = set(), []
    for org in pdc_orgs + sorted(pos_orgs - set(pdc_orgs)):
        gs = genus_species(org)
        if gs not in seen_gs:
            seen_gs.add(gs)
            targets.append(org)

    accepted, rejected, per_org = [], [], {}
    print(f"querying {len(targets)} species, cap {PER_ORG} accepted each\n")
    for org in targets:
        gs = genus_species(org)
        try:
            recs = query(org)
        except Exception as e:
            rejected.append({"organism": gs, "reason": f"query_error:{type(e).__name__}"})
            continue
        kept = 0
        for rec in recs:
            if kept >= PER_ORG:
                break
            acc = rec["primaryAccession"]
            got_org = rec.get("organism", {}).get("scientificName", "")
            seq = rec.get("sequence", {}).get("value", "")
            name = protein_name(rec)
            kws = [k.get("name", "") for k in rec.get("keywords", [])]
            has_sig = any(f.get("type") == "Signal" for f in rec.get("features", []))
            why = None
            if gs.lower() not in got_org.lower():
                why = "species_mismatch"  # filter 2
            elif any(b in k.lower() for k in kws for b in KW_NAME_BLOCK):
                why = "blocked_keyword"  # filter 3
            elif any(b in name.lower() for b in NAME_BLOCK):
                why = "blocked_protein_name"  # filter 4
            elif any(b in name.lower() for b in CLASS_BLOCK):
                why = "collides_with_positive_class"  # filter 4b
            elif not has_sig:
                why = "no_signal_peptide"
            elif acc in have_acc:
                why = "duplicate_accession"  # filter 5
            elif H(seq) in have_sha:
                why = "duplicate_sequence"
            elif not seq:
                why = "empty_sequence"
            if why:
                rejected.append({"organism": gs, "acc": acc, "reason": why})
                continue
            accepted.append(
                {
                    "acc": f"sp|{acc}|{rec.get('uniProtkbId', acc)}",
                    "uniprot": acc,
                    "name": rec.get("uniProtkbId", acc),
                    "protein_name": name,
                    "organism": got_org,
                    "len": len(seq),
                    "sha256": H(seq),
                    "sequence": seq,
                    "keywords": kws,
                    "queried_species": gs,
                }
            )
            have_acc.add(acc)
            have_sha.add(H(seq))
            kept += 1
        per_org[gs] = kept
        time.sleep(0.25)

    import collections

    rc = collections.Counter(r["reason"] for r in rejected)
    stage = {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "secreted benign negatives from pathogen organisms, to fill the "
        "exported half of the doubly controlled localization test",
        "filters": {
            "per_org_cap": PER_ORG,
            "query_excluded_keywords": BLOCK_KEYWORDS,
            "name_blocklist": list(NAME_BLOCK),
        },
        "n_accepted": len(accepted),
        "rejections": rc,
        "organisms_with_at_least_one": sum(1 for v in per_org.values() if v),
        "accepted": accepted,
        "rejected": rejected,
    }
    json.dump(stage, open(STAGE, "w"), indent=2)

    print(f"accepted            {len(accepted)}")
    print(f"organisms covered   {stage['organisms_with_at_least_one']} of {len(targets)}")
    print(f"rejections          {dict(rc)}")
    print(f"\nstaging file: {STAGE}")

    if not args.integrate:
        print("\nDRY RUN. Nothing written to the panel. Re-run with --integrate to apply.")
        return

    with open(NEG_FASTA, "a") as f:
        for e in accepted:
            f.write(f">{e['acc']} {e['protein_name']} OS={e['organism']}\n")
            for i in range(0, len(e["sequence"]), 60):
                f.write(e["sequence"][i : i + 60] + "\n")
    for e in accepted:
        panel["negatives"].append(
            {
                "acc": e["acc"],
                "name": e["name"],
                "organism": e["organism"],
                "len": e["len"],
                "sha256": e["sha256"],
                "truncated_1022": e["len"] > 1022,
                "block": "secreted_pathogen",
                "lab_strain": False,
                "pathogen_derived_control": True,
            }
        )
    panel["counts"]["negative_final"] = len(panel["negatives"])
    panel["counts"]["negative_blocks"] = dict(
        collections.Counter(e["block"] for e in panel["negatives"])
    )
    panel["counts"]["negative_pathogen_derived"] = sum(
        1 for e in panel["negatives"] if e.get("pathogen_derived_control")
    )
    note = (
        f"{time.strftime('%Y-%m-%d')}: added {len(accepted)} secreted benign proteins from "
        f"pathogen organisms (block secreted_pathogen) so the exported half of the doubly "
        f"controlled localization test is no longer empty. Species verification applied on "
        f"the FIRST pass this time. Rejections: {dict(rc)}."
    )
    panel.setdefault("maintenance_notes", []).append(note)
    json.dump(panel, open(PANEL, "w"), indent=2)
    print(
        f"\nINTEGRATED. negatives now {panel['counts']['negative_final']}, "
        f"blocks {panel['counts']['negative_blocks']}"
    )
    print("Re-run 02b, 02c, 03b, 03c, 03d.")


if __name__ == "__main__":
    main()
