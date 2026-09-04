#!/usr/bin/env python3
"""
05b_external_negatives.py - negatives for the held-out panel, using 02d's filters unchanged.

Preregistration rule 3, quoted: "For each positive's organism, benign proteins
fetched by 02d's existing filters unchanged, including filter 4b, excluding every
internal panel accession."

The filter constants are imported from `02d_secreted_pathogen_negatives` rather than
retyped, so they cannot drift from the versions that were frozen. Filter 4b is the
positive-class vocabulary screen that was added after UniProt's own Virulence and
Toxin keyword exclusions let six beta-lactamases and two Mono-ADP-ribosyltransferase
C3 entries through.

Note, recorded rather than fixed: the preregistration does not ask for the 0.30
homology screen on the negative side, and it is not applied here. Only the positives
are homology-screened against the internal panel.

Outputs: data/sequences/external_negatives.fasta and the negatives section appended
to data/sequences/external_panel_manifest.json
"""

import hashlib
import importlib.util
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEQ = ROOT / "data" / "sequences"
API = "https://rest.uniprot.org/uniprotkb/search"
PER_ORG = 3
PER_ORG_FETCH = 40


def _load_02d():
    p = ROOT / "src" / "02d_secreted_pathogen_negatives.py"
    spec = importlib.util.spec_from_file_location("mod02d", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def H(s):
    return hashlib.sha256(s.encode()).hexdigest()


def read_fasta_ids(p):
    return [line[1:].split()[0] for line in open(p) if line.startswith(">")]


def main():
    d02 = _load_02d()
    print(f"filters imported from 02d: {len(d02.CLASS_BLOCK)} class terms, "
          f"{len(d02.NAME_BLOCK)} name terms, {len(d02.BLOCK_KEYWORDS)} query keywords")

    internal = json.load(open(SEQ / "panel_v2_manifest.json"))
    ext = json.load(open(SEQ / "external_panel_manifest.json"))
    have_acc = {e["acc"].split("|")[1] for e in internal["positives"] + internal["negatives"]}
    have_acc |= {p["uniprot"] for p in ext["positives"]}
    have_sha = {e["sha256"] for e in internal["positives"] + internal["negatives"]}
    have_sha |= {p["sha256"] for p in ext["positives"]}

    species = sorted({p["species"] for p in ext["positives"]})
    print(f"{len(species)} organisms from the external positives\n")

    accepted, rejected = [], []
    for org in species:
        excl = " ".join(f"NOT keyword:{k}" for k in d02.BLOCK_KEYWORDS)
        q = f'organism_name:"{org}" AND reviewed:true AND ft_signal:* {excl}'
        url = API + "?" + urllib.parse.urlencode({
            "query": q, "format": "json", "size": PER_ORG_FETCH,
            "fields": "accession,id,organism_name,protein_name,keyword,length,sequence,ft_signal"})
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                recs = json.load(r).get("results", [])
        except Exception as e:
            rejected.append({"organism": org, "reason": f"query_error:{type(e).__name__}"})
            continue
        kept = 0
        for rec in recs:
            if kept >= PER_ORG:
                break
            acc = rec["primaryAccession"]
            got = rec.get("organism", {}).get("scientificName", "")
            seq = rec.get("sequence", {}).get("value", "")
            name = d02.protein_name(rec)
            kws = [k.get("name", "") for k in rec.get("keywords", [])]
            has_sig = any(f.get("type") == "Signal" for f in rec.get("features", []))
            why = None
            if org.lower() not in got.lower():
                why = "species_mismatch"
            elif any(b in k.lower() for k in kws for b in d02.KW_NAME_BLOCK):
                why = "blocked_keyword"
            elif any(b in name.lower() for b in d02.NAME_BLOCK):
                why = "blocked_protein_name"
            elif any(b in name.lower() for b in d02.CLASS_BLOCK):
                why = "collides_with_positive_class"
            elif not has_sig:
                why = "no_signal_peptide"
            elif acc in have_acc:
                why = "duplicate_accession"
            elif H(seq) in have_sha:
                why = "duplicate_sequence"
            elif not seq:
                why = "empty_sequence"
            if why:
                rejected.append({"organism": org, "acc": acc, "reason": why})
                continue
            accepted.append({"acc": f"sp|{acc}|{rec.get('uniProtkbId', acc)}",
                             "uniprot": acc, "name": rec.get("uniProtkbId", acc),
                             "protein_name": name, "organism": got, "species": org,
                             "len": len(seq), "sha256": H(seq), "sequence": seq})
            have_acc.add(acc)
            have_sha.add(H(seq))
            kept += 1
        time.sleep(0.2)

    import collections
    rc = collections.Counter(r["reason"] for r in rejected)
    print(f"accepted {len(accepted)} negatives from "
          f"{len({a['species'] for a in accepted})} organisms")
    print(f"rejections: {dict(rc)}")

    with open(SEQ / "external_negatives.fasta", "w") as f:
        for e in accepted:
            f.write(f">{e['acc']} {e['protein_name']} OS={e['organism']}\n")
            for i in range(0, len(e["sequence"]), 60):
                f.write(e["sequence"][i : i + 60] + "\n")
    ext["negatives"] = [{k: v for k, v in e.items() if k != "sequence"} for e in accepted]
    ext["negative_rejections"] = dict(rc)
    ext["counts"] = {"positives": len(ext["positives"]), "negatives": len(accepted)}
    json.dump(ext, open(SEQ / "external_panel_manifest.json", "w"), indent=2)
    print(f"\npanel: {len(ext['positives'])} positives / {len(accepted)} negatives")
    print(f"wrote {SEQ / 'external_negatives.fasta'}")
    print(f"positive fasta ids: {len(read_fasta_ids(SEQ / 'external_positives.fasta'))}")


if __name__ == "__main__":
    main()
