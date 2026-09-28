#!/usr/bin/env python3
"""
89_fallback_name_check.py - was entry 59's declared limitation real?

Entry 59 declared that 823 pool proteins still match on a superseded name, because their organism
string reads "Enterobacter agglomerans (Erwinia herbicola) (Pantoea agglomerans)" and src/88's lineage
fallback takes the first two words -- so fixing it would move MORE proteins and 5.05% was a lower
bound on that basis. The follow-up study meant to fix it refuted its own premise on the first check.

🔑 UniProt's taxonomy keeps the same primary name the protein record displays: taxid 549's
scientificName IS "Enterobacter agglomerans". And src/88's resolver maps VFDB's current-name entry
onto that same primary. Both sides were already in one vocabulary.

This checks that exhaustively rather than by example, and measures what the real residual is.

Usage:
    python src/89_fallback_name_check.py
    python src/89_fallback_name_check.py --selftest
"""

import argparse
import importlib.util
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POOL_JSON = ROOT / "data" / "sequences" / "_scaled_negative_pool.json"
BUILD = ROOT / "results" / "external_class_axis_build.json"
RESOLVED = ROOT / "data" / "external" / "uniprot_taxonomy" / "vfdb_species_resolved.json"
OUT = ROOT / "results" / "fallback_name_check.json"


def _load(stem):
    spec = importlib.util.spec_from_file_location(f"_{stem}", ROOT / "src" / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M88 = _load("88_taxid_provenance")
M84 = _load("84_localization_control")
M83 = _load("83_provenance_control")


def scientific_names(taxids, batch=100, pause=0.3):
    """UniProt taxonomy scientificName for each taxid, batched."""
    out = {}
    for k in range(0, len(taxids), batch):
        chunk = taxids[k:k + batch]
        u = urllib.parse.urlencode({"query": " OR ".join(f"tax_id:{t}" for t in chunk),
                                    "format": "tsv", "fields": "id,scientific_name", "size": 500})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(
                        f"https://rest.uniprot.org/taxonomy/search?{u}", timeout=120) as r:
                    for line in r.read().decode().splitlines()[1:]:
                        f = line.split("\t")
                        if len(f) > 1:
                            out[f[0]] = f[1]
                break
            except Exception:                                          # noqa: BLE001
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        time.sleep(pause)
    return out


def first_two(s):
    return " ".join(s.replace("(", "").split()[:2])


def selftest():
    assert first_two("Enterobacter agglomerans (Erwinia herbicola) (Pantoea agglomerans)") \
        == "Enterobacter agglomerans"
    assert first_two("Escherichia coli (strain K12)") == "Escherichia coli"
    assert first_two("Mycoplasmoides pneumoniae") == "Mycoplasmoides pneumoniae"
    print("selftest PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        return selftest()

    pool = json.loads(POOL_JSON.read_text())["proteins"]
    test = json.loads(BUILD.read_text())["pool_partition"]["test_rows"]
    accs = [pool[i]["uniprot"] for i in test]
    lin = M88.fetch_lineage(accs)
    kw = M84.fetch_keywords(accs)
    vf_seqs = M83.vfdb_sequences()

    # ---- the claim: fallback name == UniProt scientificName, for every parenthetical case ---------
    fb = [a for a in accs if M88.canonical_species(lin[a])[1]]
    paren = [a for a in fb if "(" in lin[a]["organism"] and " sp." not in lin[a]["organism"]]
    tids = sorted({lin[a]["taxid"] for a in paren if lin[a]["taxid"]})
    sci = scientific_names(tids)
    agree, disagree, unresolved_taxid, examples = 0, 0, 0, []
    for a in paren:
        s = sci.get(lin[a]["taxid"])
        if not s:
            unresolved_taxid += 1
            continue
        if first_two(lin[a]["organism"]) == first_two(s):
            agree += 1
        else:
            disagree += 1
            if len(examples) < 5:
                examples.append([first_two(lin[a]["organism"]), first_two(s)])

    # ---- the real residual: unresolved VFDB names that actually have pool proteins ----------------
    res_map = json.loads(RESOLVED.read_text())
    current = M83.vfdb_species()
    unresolved = sorted(n for n in current if res_map.get(n, {}).get("how") == "UNRESOLVED")
    canon_of = {a: M88.canonical_species(lin[a])[0] for a in accs}
    elig = [jj for jj, i in enumerate(test)
            if pool[i]["kingdom"] in M84.KINGDOMS and pool[i]["sequence"] not in vf_seqs]
    in_cells = [jj for jj in elig
                if M84.stratum_of(kw[accs[jj]]) in ("extracellular", "intracellular")]
    # 🔴 A shared species epithet is NOT a rename. The first version of this matched
    # "Arcanobacterium pyogenes" to Streptococcus pyogenes (55 proteins) and "Mycobacterium bovis" to
    # Moraxella bovis -- unrelated organisms that happen to share an epithet. Two filters make the
    # match sound: a protein whose species is ALREADY classified pathogen cannot be a
    # misclassification, and a candidate has to be confirmed against the documented renames below.
    pathogen_names = {v["canonical"] for v in res_map.values() if v.get("canonical")}
    # Documented genus renames, each a case where UniProt moved the species and dropped the old name
    # from its synonym list, so no resolver can recover it. Listed explicitly rather than inferred.
    RENAMES = {"Clostridium difficile": "Clostridioides difficile",
               "Borrelia bavariensis": "Borreliella bavariensis"}
    residual, epithet_only = {}, {}
    for jj in elig:
        nm = canon_of[accs[jj]]
        if nm in pathogen_names:                      # already pathogen: cannot be misclassified
            continue
        for u in unresolved:
            if RENAMES.get(u) == nm:
                residual.setdefault(f"{u} -> {nm}", []).append(jj)
            elif nm != u and nm.split()[-1:] == u.split()[-1:] and nm.split()[0] != u.split()[0]:
                epithet_only.setdefault(f"{u} !~ {nm}", []).append(jj)
    n_residual = sum(len(v) for v in residual.values())
    n_residual_cells = sum(1 for v in residual.values() for jj in v if jj in set(in_cells))

    res = {"built": time.strftime("%Y-%m-%d %H:%M:%S"),
           "n_fallback": len(fb), "n_parenthetical": len(paren), "n_taxids": len(tids),
           "n_taxids_resolved": len(sci),
           "agree": agree, "disagree": disagree, "unresolved_taxid": unresolved_taxid,
           "disagreement_examples": examples,
           "entry_59_claimed_misclassified": 823,
           "n_unresolved_vfdb_names": len(unresolved), "unresolved_vfdb_names": unresolved,
           "residual_by_name": {k: len(v) for k, v in sorted(residual.items())},
           "rejected_epithet_only": {k: len(v) for k, v in sorted(epithet_only.items())},
           "n_residual": n_residual, "n_eligible": len(elig),
           "residual_frac": n_residual / len(elig),
           "n_residual_in_cells": n_residual_cells, "n_in_cells": len(in_cells)}
    OUT.write_text(json.dumps(res, indent=2) + "\n")

    print(f"parenthetical-synonym fallback proteins: {len(paren)} over {len(tids)} taxids "
          f"({len(sci)} resolved)")
    print(f"  fallback name == UniProt scientificName : {agree}")
    print(f"  disagree                                : {disagree}")
    print(f"  taxid unresolved                        : {unresolved_taxid}")
    print(f"\n=> entry 59 declared 823 of these misclassified; the measured number is {disagree}.")
    if epithet_only:
        print("\n  rejected as coincidental shared epithets, not renames:")
        for k, v in sorted(epithet_only.items()):
            print(f"    {k:<52}{len(v):>4}")
    print(f"\nreal residual, from the {len(unresolved)} unresolved VFDB names:")
    for k, v in sorted(residual.items()):
        print(f"  {k:<52}{len(v):>4}")
    print(f"  {'TOTAL':<52}{n_residual:>4} = {n_residual / len(elig) * 100:.2f}% of "
          f"{len(elig)} eligible; {n_residual_cells} of {len(in_cells)} reach the 2x2")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
