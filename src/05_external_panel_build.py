#!/usr/bin/env python3
"""
05_external_panel_build.py - build the held-out panel specified in the preregistration.

This executes, once, the query written into
`docs/EXTERNAL_VALIDATION_PREREGISTRATION.md` Amendment 1 before it was run. Nothing
here may be tuned in response to what it returns; if the panel comes out small or
awkward, that is the panel.

Rules, copied from the preregistration rather than re-derived:
  positives   UniProt reviewed, keyword KW-0800 (Toxin), bacterial, length 100-1022,
              at most 3 per species, excluding every internal panel accession and
              anything above 0.30 normalized Smith-Waterman similarity to an internal
              panel member
  classes     assigned only from the UniProt protein name using 02d's CLASS_BLOCK
              vocabulary; no match means dropped, not placed in a catch-all
  negatives   02d's existing filters unchanged, including filter 4b, per positive
              organism, excluding every internal accession

The 0.30 homology screen is the expensive step and is the one that makes this a real
holdout rather than a reshuffle, so it runs against the full internal panel.

Outputs: data/sequences/external_{positives,negatives}.fasta
         data/sequences/external_panel_manifest.json
         data/annotations/external_mechanism_classes.json
"""

import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from Bio.Align import PairwiseAligner, substitution_matrices

ROOT = Path(__file__).resolve().parent.parent
SEQ = ROOT / "data" / "sequences"
ANN = ROOT / "data" / "annotations"
API = "https://rest.uniprot.org/uniprotkb/search"
HOMOLOGY_MAX = 0.30
PER_SPECIES = 3
LEN_MIN, LEN_MAX = 100, 1022

# 02d's vocabulary, reused verbatim as the class assigner
CLASS_TERMS = {
    "beta_lactamase": ("lactamase", "penicillinase", "cephalosporinase", "carbapenemase"),
    "adp_ribosyl_ab_toxin": ("adp-ribosyl", "ribosyltransferase"),
    "nuclease_dnase_rnase": ("nuclease", "dnase", "rnase"),
    "secreted_protease": ("protease", "peptidase", "proteinase"),
    "rip_rrna_glycosidase": ("ribosome-inactivating", "rrna n-glycosidase"),
    "pore_forming_cytolysin": ("pore-forming", "cytolysin", "hemolysin", "haemolysin",
                               "aerolysin", "leukocidin", "perfringolysin", "listeriolysin",
                               "streptolysin", "pneumolysin"),
    "superantigen_enterotoxin": ("superantigen", "enterotoxin"),
    "t3ss_effector_apparatus": ("type iii secretion", "t3ss", "secretion system", "effector"),
    "clostridial_neurotoxin": ("neurotoxin",),
    "contact_dependent_inhibition": ("contact-dependent",),
    "phospholipase": ("phospholipase",),
}


def H(s):
    return hashlib.sha256(s.encode()).hexdigest()


def read_fasta(p):
    out, acc, seq = {}, None, []
    for line in open(p):
        if line.startswith(">"):
            if acc:
                out[acc] = "".join(seq)
            acc, seq = line[1:].split()[0], []
        elif acc:
            seq.append(line.strip())
    if acc:
        out[acc] = "".join(seq)
    return out


def protein_name(rec):
    d = rec.get("proteinDescription", {})
    for key in ("recommendedName", "submissionNames"):
        v = d.get(key)
        if isinstance(v, dict) and v.get("fullName"):
            return v["fullName"]["value"]
        if isinstance(v, list) and v and v[0].get("fullName"):
            return v[0]["fullName"]["value"]
    return ""


def assign_class(name):
    low = name.lower()
    for cls, terms in CLASS_TERMS.items():
        if any(t in low for t in terms):
            return cls
    return None


def query(q, size=500, fields=None):
    f = fields or ("accession,id,organism_name,protein_name,keyword,length,sequence")
    url = API + "?" + urllib.parse.urlencode(
        {"query": q, "format": "json", "size": size, "fields": f})
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.load(r).get("results", [])


def aligner():
    a = PairwiseAligner()
    a.substitution_matrix = substitution_matrices.load("BLOSUM62")
    a.open_gap_score, a.extend_gap_score, a.mode = -11, -1, "local"
    return a


def main():
    internal = json.load(open(SEQ / "panel_v2_manifest.json"))
    int_acc = {e["acc"].split("|")[1] for e in internal["positives"] + internal["negatives"]}
    int_seqs = list(read_fasta(SEQ / "toxins_positive_v2.fasta").values()) + \
        list(read_fasta(SEQ / "benign_negatives_v2.fasta").values())
    print(f"internal panel: {len(int_acc)} accessions to exclude")

    q = (f"reviewed:true AND keyword:KW-0800 AND taxonomy_id:2 "
         f"AND length:[{LEN_MIN} TO {LEN_MAX}]")
    print(f"query: {q}")
    recs = query(q)
    print(f"UniProt returned {len(recs)}")

    # class assignment and the cheap filters first
    cand, per_sp = [], {}
    for r in recs:
        acc = r["primaryAccession"]
        if acc in int_acc:
            continue
        seq = r.get("sequence", {}).get("value", "")
        name = protein_name(r)
        cls = assign_class(name)
        org = r.get("organism", {}).get("scientificName", "")
        sp = " ".join(org.split()[:2])
        if not seq or cls is None:
            continue
        if per_sp.get(sp, 0) >= PER_SPECIES:
            continue
        per_sp[sp] = per_sp.get(sp, 0) + 1
        cand.append({"acc": f"sp|{acc}|{r.get('uniProtkbId', acc)}", "uniprot": acc,
                     "name": r.get("uniProtkbId", acc), "protein_name": name,
                     "organism": org, "species": sp, "len": len(seq),
                     "sha256": H(seq), "sequence": seq, "mechanism_class": cls})
    print(f"after class assignment, accession exclusion and {PER_SPECIES}/species cap: {len(cand)}")

    # the homology screen against the whole internal panel
    al = aligner()
    clean = lambda s: "".join(c for c in s if c in "ACDEFGHIKLMNPQRSTVWY")  # noqa: E731
    int_clean = [clean(s) for s in int_seqs]
    int_self = np.array([al.score(s, s) for s in int_clean], float)
    kept, dropped = [], 0
    t0 = time.time()
    for i, c in enumerate(cand):
        s = clean(c["sequence"])
        ss = al.score(s, s)
        m = max(al.score(s, t) / np.sqrt(ss * sf) for t, sf in zip(int_clean, int_self))
        c["max_similarity_to_internal"] = float(m)
        if m > HOMOLOGY_MAX:
            dropped += 1
        else:
            kept.append(c)
        if (i + 1) % 25 == 0:
            print(f"    screened {i + 1}/{len(cand)}  {time.time() - t0:.0f}s", flush=True)
    print(f"homology screen at {HOMOLOGY_MAX}: kept {len(kept)}, dropped {dropped}")

    import collections
    cc = collections.Counter(c["mechanism_class"] for c in kept)
    print(f"\nclasses: {dict(cc)}")

    SEQ.mkdir(parents=True, exist_ok=True)
    with open(SEQ / "external_positives.fasta", "w") as f:
        for c in kept:
            f.write(f">{c['acc']} {c['protein_name']} OS={c['organism']}\n")
            for j in range(0, len(c["sequence"]), 60):
                f.write(c["sequence"][j : j + 60] + "\n")
    json.dump({"built": time.strftime("%Y-%m-%d %H:%M:%S"),
               "query": q, "homology_max": HOMOLOGY_MAX, "per_species": PER_SPECIES,
               "n_returned": len(recs), "n_candidates": len(cand),
               "n_kept": len(kept), "n_dropped_homology": dropped,
               "classes": dict(cc),
               "positives": [{k: v for k, v in c.items() if k != "sequence"} for c in kept]},
              open(SEQ / "external_panel_manifest.json", "w"), indent=2)
    ANN.mkdir(parents=True, exist_ok=True)
    json.dump({"built": time.strftime("%Y-%m-%d %H:%M:%S"),
               "note": "classes assigned only from the UniProt protein name, per the "
                       "preregistration; no model output was consulted",
               "holdout_eligible_classes": [c for c, n in cc.items() if n >= 3],
               "proteins": [{"fasta_id": c["acc"], "mechanism_class": c["mechanism_class"]}
                            for c in kept]},
              open(ANN / "external_mechanism_classes.json", "w"), indent=2)
    print(f"\nwrote {SEQ / 'external_positives.fasta'} and the two manifests")
    print("negatives are fetched separately by 05b using 02d's filters unchanged")


if __name__ == "__main__":
    main()
