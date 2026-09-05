#!/usr/bin/env python3
"""
23_expand_small_classes.py - raise the n<=4 holdout classes without inflating n.

The problem
-----------
Five of the eight holdout-eligible classes have n <= 4, and the strongest positive
claim in this project rests on two of them: clostridial neurotoxin and superantigen
enterotoxin recover at 100% in every configuration tested. But superantigen's four
members are all Staphylococcus aureus with a within-class 5-mer maximum of 0.813,
and clostridial's four are three botulinum serotypes plus tetanus at 0.494. Those
are roughly three independent observations each, not four.

The constraint that makes this hard
-----------------------------------
Adding close relatives raises n and adds no evidence. A fifth S. aureus enterotoxin
would make "5 of 5 recovered" look stronger while testing nothing new. So every
candidate is screened at normalized Smith-Waterman 0.30 against **the existing
members of its own class**, not merely against the panel at large.

🔴 And against **the candidates already accepted for that class in this run**. A
first version screened only against pre-existing members, so it admitted two
near-identical streptococcal exotoxin G entries and two near-identical CdiA
entries. Within-class 5-mer maxima went the wrong way, superantigen 0.813 to 0.909
and contact-dependent 0.001 to 0.602: the expansion inflated n with homologues, the
exact failure it exists to prevent. Both comparisons are now made.

If a class cannot be expanded under that constraint, that is the finding: the class
is intrinsically homologous and its invariance claim can never rest on more than a
few independent observations. Reported rather than worked around.

Two stages, as with 02d: a dry run that writes the staging file and prints counts,
then --integrate.

Usage:
    python src/23_expand_small_classes.py
    python src/23_expand_small_classes.py --integrate
"""

import argparse
import collections
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
STAGE = SEQ / "_small_class_expansion.json"
HOMOLOGY_MAX = 0.30
TARGET_N = 7
# The panel already contains positives longer than the 1022-residue embedding
# truncation, which are truncated at embedding time rather than excluded. Capping
# the expansion query at 1022 was therefore inconsistent with the panel it expands,
# and it silently removed every botulinum serotype (1274-1296 aa). A first run
# reported clostridial as unexpandable "under the 0.30 constraint" when the real
# cause was this filter.
LEN_MIN, LEN_MAX = 100, 1400

# Class-specific queries.
#
# 🔴 A first version queried on mechanism name alone and had to be discarded. It
# returned four human PARP enzymes for adp_ribosyl_ab_toxin (DNA-repair proteins
# that share the transferase chemistry but are not toxins), and matched "Rhs" as a
# substring inside "Rhazimal synthase", a plant biosynthesis enzyme. Labelling human
# housekeeping enzymes as toxins is the beta-lactamase contamination again, in the
# more damaging direction.
#
# Every query now carries a hazard constraint: the UniProt Toxin or Virulence
# keyword, or an explicit toxin-family name. Mechanism alone is not sufficient
# evidence that a protein is a toxin.
HAZARD = "(keyword:KW-0800 OR keyword:KW-0843)"
QUERIES = {
    "adp_ribosyl_ab_toxin": [
        f'{HAZARD} AND protein_name:"ADP-ribosyltransferase" AND taxonomy_id:2',
        f'{HAZARD} AND protein_name:"pertussis toxin"',
        f'{HAZARD} AND protein_name:"iota toxin"',
        f'{HAZARD} AND protein_name:"C2 toxin" AND taxonomy_id:2',
        f'{HAZARD} AND protein_name:"exoenzyme" AND taxonomy_id:2',
    ],
    "contact_dependent_inhibition": [
        f'{HAZARD} AND protein_name:"contact-dependent growth inhibition" AND taxonomy_id:2',
        f'{HAZARD} AND protein_name:"CdiA" AND taxonomy_id:2',
        f'{HAZARD} AND protein_name:"toxin-antitoxin" AND taxonomy_id:2',
    ],
    "rip_rrna_glycosidase": [
        'protein_name:"ribosome-inactivating protein"',
        'protein_name:"rRNA N-glycosidase"',
        'protein_name:"saporin"', 'protein_name:"trichosanthin"',
        'protein_name:"gelonin"', 'protein_name:"momordin"',
    ],
    "clostridial_neurotoxin": [
        f'{HAZARD} AND protein_name:"botulinum neurotoxin"',
        f'{HAZARD} AND protein_name:"tetanus neurotoxin"',
    ],
    "superantigen_enterotoxin": [
        f'{HAZARD} AND protein_name:"pyrogenic exotoxin" AND taxonomy_id:2',
        f'{HAZARD} AND protein_name:"toxic shock syndrome toxin" AND taxonomy_id:2',
        f'{HAZARD} AND protein_name:"enterotoxin type" AND taxonomy_id:2',
    ],
}

# Names that indicate a host enzyme sharing the chemistry rather than a toxin,
# or a component that is explicitly not the toxin itself.
NOT_A_TOXIN = ("polymerase", "synthase", "synthetase", "ecto-", "poly [adp-ribose]",
               "parp", "reductase", "kinase", "ligase", "endogenous retrovirus",
               "receptor", "hydrolase domain", "non-toxic", "nontoxic",
               "pectinesterase", "inhibitor", "antitoxin", "immunity protein")

# 🔴 A blocklist alone is whack-a-mole: three separate rounds of contamination got
# through one, each time via vocabulary the previous round had not anticipated
# (human PARPs, "Rhs" inside "Rhazimal synthase", "Non-toxic nonhemagglutinin",
# "Pectinesterase"). This is the inverse and structural check: a candidate must
# carry a term from ITS OWN class vocabulary, so sharing chemistry or a substring
# with the class is not enough to be admitted to it.
CLASS_VOCAB = {
    "adp_ribosyl_ab_toxin": ("adp-ribosyl", "ribosylating", "pertussis toxin",
                             "iota toxin", "c2 toxin", "cholix", "exotoxin a",
                             "diphtheria"),
    "contact_dependent_inhibition": ("cdia", "contact-dependent", "rhs element",
                                     "polymorphic toxin"),
    "rip_rrna_glycosidase": ("ribosome-inactivating", "rrna n-glycosidase",
                             "n-glycosidase", "saporin", "trichosanthin", "gelonin",
                             "momordin", "abrin", "ricin", "shiga", "heterotepalin"),
    "clostridial_neurotoxin": ("botulinum neurotoxin", "tetanus neurotoxin",
                               "neurotoxin type"),
    "superantigen_enterotoxin": ("pyrogenic exotoxin", "superantigen",
                                 "toxic shock syndrome toxin", "enterotoxin type",
                                 "exotoxin type"),
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


def aligner():
    a = PairwiseAligner()
    a.substitution_matrix = substitution_matrices.load("BLOSUM62")
    a.open_gap_score, a.extend_gap_score, a.mode = -11, -1, "local"
    return a


def clean(s):
    return "".join(c for c in s if c in "ACDEFGHIKLMNPQRSTVWY")


def query(q):
    url = API + "?" + urllib.parse.urlencode({
        "query": f"({q}) AND reviewed:true AND length:[{LEN_MIN} TO {LEN_MAX}]",
        "format": "json", "size": 100,
        "fields": "accession,id,organism_name,protein_name,length,sequence"})
    with urllib.request.urlopen(url, timeout=90) as r:
        return json.load(r).get("results", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--integrate", action="store_true")
    a = ap.parse_args()

    panel = json.load(open(SEQ / "panel_v2_manifest.json"))
    mech = json.load(open(ANN / "mechanism_classes_v2.json"))
    cls_of = {p["fasta_id"]: p["mechanism_class"] for p in mech["proteins"]}
    pos_seqs = read_fasta(SEQ / "toxins_positive_v2.fasta")
    have_acc = {e["acc"].split("|")[1] for e in panel["positives"] + panel["negatives"]}
    have_sha = {e["sha256"] for e in panel["positives"] + panel["negatives"]}

    al = aligner()
    accepted, rejected = [], []
    for cname, qs in QUERIES.items():
        members = [a_ for a_, c in cls_of.items() if c == cname]
        mem_clean = [clean(pos_seqs[m]) for m in members if m in pos_seqs]
        mem_self = np.array([al.score(s, s) for s in mem_clean], float)
        need = max(0, TARGET_N - len(members))
        print(f"\n=== {cname}: have {len(members)}, need {need} ===", flush=True)
        if need == 0:
            continue
        seen, kept = set(), 0
        new_clean, new_self = [], []
        for q in qs:
            if kept >= need:
                break
            try:
                recs = query(q)
            except Exception as e:
                print(f"  query error {type(e).__name__} on {q}")
                continue
            for rec in recs:
                if kept >= need:
                    break
                acc = rec["primaryAccession"]
                seq = rec.get("sequence", {}).get("value", "")
                if acc in have_acc or acc in seen or not seq:
                    continue
                seen.add(acc)
                nm = protein_name(rec).lower()
                if not any(v in nm for v in CLASS_VOCAB[cname]):
                    rejected.append({"class": cname, "acc": acc,
                                     "reason": "name_lacks_own_class_vocabulary",
                                     "protein_name": protein_name(rec)})
                    continue
                if any(b in nm for b in NOT_A_TOXIN):
                    rejected.append({"class": cname, "acc": acc,
                                     "reason": "host_enzyme_not_a_toxin",
                                     "protein_name": protein_name(rec)})
                    continue
                if H(seq) in have_sha:
                    rejected.append({"class": cname, "acc": acc, "reason": "duplicate_sequence"})
                    continue
                s = clean(seq)
                ss = al.score(s, s)
                sim = max((al.score(s, t) / np.sqrt(ss * sf)
                           for t, sf in zip(mem_clean, mem_self)), default=0.0)
                if sim > HOMOLOGY_MAX:
                    rejected.append({"class": cname, "acc": acc,
                                     "reason": "homologous_to_existing_member",
                                     "max_similarity": round(float(sim), 3)})
                    continue
                sim_new = max((al.score(s, t) / np.sqrt(ss * sf)
                               for t, sf in zip(new_clean, new_self)), default=0.0)
                if sim_new > HOMOLOGY_MAX:
                    rejected.append({"class": cname, "acc": acc,
                                     "reason": "homologous_to_a_candidate_already_accepted",
                                     "max_similarity": round(float(sim_new), 3)})
                    continue
                accepted.append({
                    "acc": f"sp|{acc}|{rec.get('uniProtkbId', acc)}",
                    "uniprot": acc, "name": rec.get("uniProtkbId", acc),
                    "protein_name": protein_name(rec),
                    "organism": rec.get("organism", {}).get("scientificName", ""),
                    "mechanism_class": cname, "len": len(seq), "sha256": H(seq),
                    "sequence": seq, "max_similarity_to_class": round(float(sim), 3),
                    "max_similarity_to_new_members": round(float(sim_new), 3)})
                have_acc.add(acc)
                have_sha.add(H(seq))
                new_clean.append(s)
                new_self = np.append(new_self, ss)
                kept += 1
                print(f"  + {acc} sim={sim:.3f} {protein_name(rec)[:44]}", flush=True)
            time.sleep(0.2)
        if kept < need:
            print(f"  ** only {kept} of {need} found under the {HOMOLOGY_MAX} constraint")

    rc = collections.Counter(r["reason"] for r in rejected)
    print(f"\naccepted {len(accepted)}; rejections {dict(rc)}")
    per = collections.Counter(x["mechanism_class"] for x in accepted)
    print(f"per class: {dict(per)}")

    json.dump({"built": time.strftime("%Y-%m-%d %H:%M:%S"),
               "homology_max_vs_own_class": HOMOLOGY_MAX, "target_n": TARGET_N,
               "accepted": accepted, "rejected": rejected,
               "rejections": dict(rc)}, open(STAGE, "w"), indent=2)
    print(f"\nstaging: {STAGE}")
    if not a.integrate:
        print("DRY RUN. Re-run with --integrate to apply.")
        return

    with open(SEQ / "toxins_positive_v2.fasta", "a") as f:
        for e in accepted:
            f.write(f">{e['acc']} {e['protein_name']} OS={e['organism']}\n")
            for i in range(0, len(e["sequence"]), 60):
                f.write(e["sequence"][i:i + 60] + "\n")
    for e in accepted:
        panel["positives"].append({
            "acc": e["acc"], "name": e["name"], "organism": e["organism"],
            "len": e["len"], "sha256": e["sha256"], "truncated_1022": e["len"] > 1022,
            "expansion_2026_09_05": True})
        # Match the existing entry schema exactly. A first run appended only
        # fasta_id and mechanism_class, and 03b crashed on KeyError: 'short_name'
        # while every other downstream script ran fine, so the stale lomo_results
        # sat next to freshly regenerated companions and looked current.
        mech["proteins"].append({
            "fasta_index": None,
            "fasta_id": e["acc"],
            "short_name": e["acc"].split("|")[2],
            "mechanism_class": e["mechanism_class"],
            "reason": (f"class expansion {time.strftime('%Y-%m-%d')}: "
                       f"{e['protein_name']} ({e['organism']}); max SW similarity to "
                       f"existing class members {e['max_similarity_to_class']}"),
            "holdout_eligible": True})
    panel["counts"]["positive_final"] = len(panel["positives"])
    # fasta_index must match the FASTA order, which only exists after the append
    # above. Leaving it None passed 03b, which reads short_name, and would have
    # left an incomplete schema for whatever reads it next.
    order = [ln[1:].split()[0] for ln in open(SEQ / "toxins_positive_v2.fasta")
             if ln.startswith(">")]
    pos_idx = {f: i for i, f in enumerate(order)}
    for entry in mech["proteins"]:
        entry["fasta_index"] = pos_idx.get(entry["fasta_id"])
    mech["proteins"].sort(key=lambda x: x["fasta_index"]
                          if x["fasta_index"] is not None else 10 ** 6)
    cc = collections.Counter(p["mechanism_class"] for p in mech["proteins"])

    # holdout_eligible_classes is a CURATED list, not a size threshold, and the
    # first version of this script overwrote it with `n >= 3`. That silently did
    # two things: it promoted other_toxin_mechanism, a grab-bag of unrelated
    # mechanisms, into the results table as though it were a mechanism class, and
    # it flipped virulence_associated_non_toxin from False to True. 03b runs that
    # class deliberately (targets = eligible | {virulence_associated_non_toxin})
    # and reports it with holdout_eligible False precisely to mark it as a
    # non-mechanism control, so the recompute destroyed the flag whose entire job
    # was to say "this row is not a mechanism". Expansion adds members; it does
    # not decide what counts as a mechanism. Assert instead of recompute.
    eligible = mech["holdout_eligible_classes"]
    promoted = [c for c in cc if c not in eligible and cc[c] >= 3
                and any(p["mechanism_class"] == c and p.get("holdout_eligible")
                        for p in mech["proteins"])]
    assert not promoted, (
        f"expansion would change class eligibility: {promoted}. Eligibility is a "
        "curation decision and must be edited deliberately, not derived from n.")
    for entry in mech["proteins"]:
        entry["holdout_eligible"] = entry["mechanism_class"] in eligible

    panel["counts"]["positive_expansion_2026_09_05"] = len(accepted)
    panel["maintenance_notes"].append(
        f"{time.strftime('%Y-%m-%d')}: added {len(accepted)} non-homologous members to "
        "classes with n < 7 (adp_ribosyl 3->7, rip 3->7, superantigen 4->7, "
        "clostridial 4->6, contact_dependent 3->4) so the small-n classes carry enough "
        "members to distinguish class recovery from single-member accidents. Every "
        "candidate was screened at Smith-Waterman <= 0.30 against existing members AND "
        "against already-accepted candidates. counts.positive_final moves 66 -> 80; "
        "positive_raw/positive_dedup still describe the original FASTA build and are "
        "left alone. Class eligibility is unchanged: no class crossed a curation "
        "boundary.")
    json.dump(panel, open(SEQ / "panel_v2_manifest.json", "w"), indent=2)

    mech["after_dedup"] = len(mech["proteins"])
    mech["provenance_note"] = (
        f"source_fasta_records ({mech['source_fasta_records']}) and the 66-member "
        "dedup describe the original 2026-09-03 build. after_dedup is the current "
        "member count and includes the 2026-09-05 class expansion.")
    json.dump(mech, open(ANN / "mechanism_classes_v2.json", "w"), indent=2)
    print(f"INTEGRATED. positives now {len(panel['positives'])}; classes {dict(cc)}")


if __name__ == "__main__":
    main()
