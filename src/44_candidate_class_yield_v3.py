#!/usr/bin/env python3
"""
44_candidate_class_yield_v3.py - can a candidate mechanism class clear the panel's admission rule
                                 against v3, and is each survivor actually hazard-annotated?

Why this exists beside 26
------------------------
`26` measures supply: how many members of a candidate family survive normalized Smith-Waterman at
≤ 0.30 against each other and against the panel's positives. It screens against **v2's 80** and it
deliberately reports no hazard annotation at all, because, in its own words, "Supply is not a label."

That separation is right, and it leaves the decision unmakeable from `26`'s output alone. Whether a
candidate class can be built depends on two numbers that `26` gives one of:

    1. how many independent members survive the 0.30 rule against the CURRENT panel, now 149 positives
    2. how many of those carry a hazard keyword, and which one

The second is what settles the borderline cases, and it is the number that decides whether a family
clears the **n = 4** eligibility floor. So this reports supply per entry with its keywords attached,
and it still admits nothing.

The counting rule, stated because it is the whole question
--------------------------------------------------------
The panel's negative harvests (`02d`, `27`, `34`) exclude a benign candidate if it carries **any** of
Toxin, Virulence, Cytolysis or Hemolysis. Read as a hazard criterion, that is a four-keyword OR. The
strictest reading is Toxin **and** Virulence together. Those two readings can straddle the n=4 floor
for the same family, so both counts are reported and neither is called the answer.

What this does NOT do
---------------------
⚠️ It admits nothing. No file under `data/` changes, no annotation moves, the panel is untouched.

⚠️ Greedy acceptance in UniProt result order, matching `23` and `26`. A different order can change
which representative of a tight cluster is kept, and can change the count.

⚠️ A dated snapshot of live UniProt queries, so it is deliberately **not** pinned in
`src/22_claims_audit.py`, for the reason `26` gives: an audit claim on these counts would fail CI on
somebody's curation work rather than on a defect.

⚠️ Hazard keywords are UniProt's judgement, not this panel's. §2's provenance control exists because
the annotation and the hazard are not the same thing, and a keyword count is evidence for a labelling
decision rather than the decision.

🔴 **It does not check that a survivor's MECHANISM matches the candidate class, and this has already
produced a wrong verdict.** Run on a deamidase query that was an activity-OR-name disjunction, it
returned BUILDABLE at 4 members carrying Toxin and Virulence. Reading the six survivors' own function
lines cut that to three: `P9WNU9` deamidates Pup, a *bacterial* protein, not a host target; `Q6TKR8`
AvrRxo1's stated reaction is `NAD(+) + ATP = 3'-NADP(+) + ADP`, so it is an NAD kinase matched on a text
mention; and `P17452` has **no catalytic activity stated at all** and entered on the protein name. The
same reading excluded `O06350` LipF from the phospholipase C candidate, whose own function line says it
has no activity on triacylglycerides.

**So the counts this prints are an upper bound.** Define the candidate by an EC number or a stated
reaction rather than by a name, then read every survivor's `FUNCTION` and `CATALYTIC ACTIVITY` before
believing the verdict. A keyword is not a mechanism.

Usage:
    python src/44_candidate_class_yield_v3.py --name phospholipase_c --query "ec:3.1.4.3" \\
        --target animal --producer bacteria
"""

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from Bio.Align import PairwiseAligner, substitution_matrices

ROOT = Path(__file__).resolve().parent.parent
V3 = ROOT / "results" / "v3"
API = "https://rest.uniprot.org/uniprotkb/search"
HOMOLOGY_MAX = 0.30
ELIGIBILITY_FLOOR = 4      # contact_dependent_inhibition is eligible at n=4, so this is the floor
LEN_MIN, LEN_MAX = 100, 1400
HAZARD = ("Toxin", "Virulence", "Cytolysis", "Hemolysis")
AA = "ACDEFGHIKLMNPQRSTVWY"


def clean(s):
    return "".join(c for c in s if c in AA)


def aligner():
    a = PairwiseAligner()
    a.substitution_matrix = substitution_matrices.load("BLOSUM62")
    a.open_gap_score, a.extend_gap_score, a.mode = -11, -1, "local"
    return a


def fetch(query, taxon):
    q = f"({query}) AND reviewed:true AND length:[{LEN_MIN} TO {LEN_MAX}]"
    if taxon:
        q += f" AND (taxonomy_name:{taxon})"
    url = API + "?" + urllib.parse.urlencode(
        {"query": q, "format": "json", "size": 200,
         "fields": "accession,id,protein_name,organism_name,length,sequence,keyword"})
    with urllib.request.urlopen(
            urllib.request.Request(url, headers={"Accept": "application/json"}), timeout=120) as r:
        return json.loads(r.read()).get("results", []), q


def read_positives():
    out, key, buf = {}, None, []
    for line in open(ROOT / "data/sequences/toxins_positive_v3.fasta"):
        if line.startswith(">"):
            if key:
                out[key] = "".join(buf)
            key, buf = line[1:].split()[0], []
        else:
            buf.append(line.strip())
    if key:
        out[key] = "".join(buf)
    return {k: clean(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--query", required=True, help="UniProt query, without the reviewed/length terms")
    ap.add_argument("--taxon", default="Bacteria", help="taxonomy_name filter, or empty for none")
    ap.add_argument("--target", default="?", help="target host, target_host_v2.json vocabulary")
    ap.add_argument("--producer", default="?", help="producer kingdom, for the §2.4.1 balance")
    a = ap.parse_args()

    res, full_q = fetch(a.query, a.taxon)
    al = aligner()
    items = []
    for it in res:
        s = clean(it["sequence"]["value"])
        if not s:
            continue
        pd = it.get("proteinDescription", {})
        items.append({
            "acc": it["primaryAccession"],
            "name": pd.get("recommendedName", {}).get("fullName", {}).get("value", ""),
            "alt": [x.get("fullName", {}).get("value", "")
                    for x in pd.get("alternativeNames", [])][:3],
            "organism": it.get("organism", {}).get("scientificName", ""),
            "length": it["sequence"]["length"],
            "keywords": [k["name"] for k in it.get("keywords", [])],
            "seq": s, "self": al.score(s, s)})
    print(f"query: {full_q}")
    print(f"reviewed entries in the length window: {len(items)}\n")

    # ---- within-family independence, greedy in result order -------------------------
    keep = []
    for x in items:
        if all(al.score(x["seq"], y["seq"]) / np.sqrt(x["self"] * y["self"]) <= HOMOLOGY_MAX
               for y in keep):
            keep.append(x)
    print(f"independent at <= {HOMOLOGY_MAX}: {len(keep)} of {len(items)}\n")

    # ---- novelty against v3's 149 positives ----------------------------------------
    P = read_positives()
    Pself = {k: al.score(v, v) for k, v in P.items()}
    rows = []
    for x in keep:
        best, bk = 0.0, None
        for pk, pv in P.items():
            v = al.score(x["seq"], pv) / np.sqrt(x["self"] * Pself[pk])
            if v > best:
                best, bk = v, pk
        haz = [k for k in HAZARD if k in x["keywords"]]
        rows.append({"acc": x["acc"], "name": x["name"], "alt": x["alt"],
                     "organism": x["organism"], "length": x["length"],
                     "max_vs_panel": float(best),
                     "closest_positive": bk.split("|")[1] if bk else None,
                     # bool() is not decoration: `best` is a numpy float64, so the comparison
                     # returns numpy.bool_, which json.dump refuses with a message that names
                     # "bool" and sounds impossible.
                     "novel": bool(best <= HOMOLOGY_MAX),
                     "hazard_keywords": haz,
                     "strict": "Toxin" in haz and "Virulence" in haz,
                     "any_hazard": bool(haz)})

    hdr = f"{'acc':<12}{'name':<40}{'organism':<26}{'len':>5}{'vs panel':>10}  hazard"
    print(hdr + "\n" + "-" * (len(hdr) + 22))
    for r in sorted(rows, key=lambda r: (not r["strict"], not r["any_hazard"], r["acc"])):
        haz = ",".join(k[:4] for k in r["hazard_keywords"]) or "NONE"
        flag = "" if r["novel"] else f"  <-- overlaps {r['closest_positive']}"
        print(f"{r['acc']:<12}{r['name'][:40]:<40}{r['organism'][:26]:<26}"
              f"{r['length']:>5}{r['max_vs_panel']:>10.3f}  {haz}{flag}")

    novel = [r for r in rows if r["novel"]]
    strict = [r for r in novel if r["strict"]]
    anyh = [r for r in novel if r["any_hazard"]]
    nonh = [r for r in novel if not r["any_hazard"]]
    print(f"\nnovel against v3's {len(P)} positives: {len(novel)} of {len(keep)}")
    print(f"  of those, carrying ANY of {HAZARD}: {len(anyh)}")
    print(f"  of those, carrying Toxin AND Virulence: {len(strict)}")
    if nonh:
        print(f"  ⚠️ carrying NO hazard keyword, exclude before counting: "
              f"{[r['acc'] for r in nonh]}")

    floor = ELIGIBILITY_FLOOR
    if len(strict) >= floor:
        verdict = (f"BUILDABLE under either reading: {len(strict)} novel members carry Toxin and "
                   f"Virulence together, at or above the n={floor} floor")
    elif len(anyh) >= floor:
        verdict = (f"BUILDABLE ONLY under the four-keyword OR: {len(anyh)} novel members carry any "
                   f"hazard keyword against {len(strict)} carrying Toxin and Virulence, so the "
                   f"n={floor} floor is cleared by the loose reading and not the strict one. Which "
                   "reading applies is a labelling decision, not a retrieval result")
    else:
        verdict = (f"NOT BUILDABLE: {len(anyh)} novel hazard-annotated members under the loosest "
                   f"reading, below the n={floor} floor")
    print(f"\nverdict: {verdict}")
    if a.producer and a.target:
        print(f"note: target host {a.target}, producer {a.producer}. A class whose producer kingdom "
              f"matches the panel's existing majority does not disturb §2.4.1's balance; one whose "
              f"target host is animal does not help its non-animal deficit either.")

    dest = V3 / f"candidate_class_yield_{a.name}.json"
    json.dump({"name": a.name, "query": full_q, "taxon": a.taxon,
               "target_host": a.target, "producer_kingdom": a.producer,
               "homology_max": HOMOLOGY_MAX, "eligibility_floor": floor,
               "panel_positives_screened_against": len(P),
               "reviewed_in_window": len(items), "independent": len(keep),
               "novel": len(novel), "novel_any_hazard": len(anyh),
               "novel_strict_hazard": len(strict),
               "novel_no_hazard": [r["acc"] for r in nonh],
               "members": rows, "admits_nothing": True,
               "not_pinned_reason": ("live UniProt snapshot; an audit claim on these counts would "
                                     "fail CI on curation work rather than on a defect, the same "
                                     "reason 26 is unpinned"),
               "verdict": verdict}, open(dest, "w"), indent=2)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
