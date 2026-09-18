#!/usr/bin/env python3
"""
27_expand_nonanimal_classes.py - add three bacterial-producer, non-animal-target classes.

Why these three and not the five that are available
---------------------------------------------------
§2.4.1 established the blocker: non-animal target is carried by **two** mechanism classes, so
leave-one-class-out cannot both train and test on non-animal hazard, and the class-level
permutation test in `03w` had no power at nine classes (8% of its null draws reached a perfect
score). `src/26_panel_growth_yield.py` measured five admissible non-animal candidates. This
script admits three of them:

    bacteriocin                     target bacteria   producers E. coli and lactic acid bacteria
    phage_peptidoglycan_hydrolase   target bacteria   producers bacteriophages
    cry_insecticidal       target insect       producer Bacillus thuringiensis

🔴 The first dry run of this script admitted 30 "bacteriocins" of which **eight were not
bacteriocins**: a receptor on the target cell, a secretion ATPase, and six immunity proteins,
which are what the producer uses to survive its own colicin. `protein_name:colicin` matched all
of them. A hazard panel whose positive class contains the antidote is not a small error, so the
queries below select on the UniProt keyword instead, which was checked case by case to separate
the agent from the accessory. Three archaeal halocins were dropped at the same time, for a
different reason: `producer_kingdom` in the annotation has no archaeal value and §2.4.1's
producer argument depends on that coding being honest.

Two are deliberately held back:

  * `plant_target_avirulence`, the single richest candidate at 32, is crop-targeting. The scope
    memo for the species-stratified project states that the release decision for crop-target
    hazard belongs at the start of that project rather than to whoever happens to be adding
    sequences. Held for that decision.
  * `chitinase_antifungal` at 28 fails a different test. Most reviewed chitinases are plant
    defence enzymes, so admitting them as hazard positives is a labelling claim this panel's
    definition does not obviously support. Supply is not a label.

All three admitted classes keep the producer balance the panel already has, which matters:
§2.4.1's conclusion that producer taxonomy does not explain target-host legibility rests on 74
of 80 positives sharing a producer kingdom. Venom classes would have destroyed that.

The admission rule, unchanged from 23 and 26
--------------------------------------------
Normalized Smith-Waterman `score / sqrt(self_i * self_j)` at **<= 0.30**, BLOSUM62, gaps
-11/-1, local, reviewed only, length 100 to 1400. Every positive candidate is screened against:

  1. the 80 existing panel positives, so a "new class" is not a relabelled old one
  2. the candidates already accepted for its own class in this run, which is the screen that
     stops a conserved-fold family from inflating n with homologues
  3. the 154 existing negatives. A hit there is a **label conflict** rather than a duplicate,
     so it is rejected and reported under its own reason rather than folded into the others.

Matched negatives, and why the per-organism cap differs from 02d's
-----------------------------------------------------------------
🔴 The panel's negatives contain **no phage proteins at all** and no *Bacillus thuringiensis*.
Adding cry toxins and endolysins without matched negatives would make the producing organism
itself the signal, which is the failure mode §2.2 exists to prevent and which
`docs/DATA_CORRECTIONS.md` records twice: length-matched-only negatives gave AUROC 0.424,
below chance, and genus-matched gave 0.556 with an interval containing 0.5.

So each new class brings organism-matched negatives, following `02d`'s filters exactly: query
side excludes the Virulence, Toxin, Cytolysis and Hemolysis keywords, then the keyword check is
repeated in Python, then a protein-name blocklist, then the positive-class-term blocklist, then
accession and sequence dedup against the whole panel.

⚠️ `02d` caps at 2 per organism so that one well-annotated organism cannot dominate the panel.
That cap cannot apply here. Bt-matched negatives must come from Bt, which is one species, so a
cap of 2 would defeat the matching this block exists to provide. Caps are therefore set per
group and stated: phages span many species so 2 holds, LAB gets 4, and Bt is uncapped because
the organism IS the match. The resulting share, roughly 40 of about 300 negatives from Bt, is
close to the 33 of 154 the panel already carries from B. subtilis.

⚠️ Negatives are NOT homology-screened against the positives, matching `02d`. Mechanism-matched
benign proteins are wanted, not excluded, and that is what `benign_homologs.fasta` is for. The
hazard filters are the protection. Max similarity to the new positives is recorded per negative
so the choice stays visible.

⚠️ What this does not do. New members get no 3Di structure: `foldseek` is fetched as a Linux
binary by `02h_saprot_prepare.py` and the existing strings were produced on the HPC. New entries
take the `no_structure` mask, which the coverage audit accepts, and the **SaProt arm is therefore
not evaluated on these three classes** until structures are fetched. Every other arm is
unaffected.

Two stages, as with 02d and 23. The default is a dry run that writes the staging file and prints
counts. Nothing touches the panel until --integrate.

Usage:
    python src/27_expand_nonanimal_classes.py
    python src/27_expand_nonanimal_classes.py --integrate
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
STAGE = SEQ / "_nonanimal_class_expansion.json"
API = "https://rest.uniprot.org/uniprotkb/search"
HOMOLOGY_MAX = 0.30
LEN_MIN, LEN_MAX = 100, 1400
FETCH = 120

# (class, target_host, producer_kingdom, uniprot query, cap on accepted positives)
POS_CLASSES = [
    # Keyword, not protein name. A first run used `(keyword:KW-0078) OR (protein_name:colicin)`
    # and the name half admitted eight entries that are not bacteriocins: the colicin I
    # RECEPTOR (P17315, a protein on the target cell), the colicin V secretion ATPase
    # (P22520, machinery), and six colicin IMMUNITY proteins (P02985, P08701, P08702,
    # P09182, P18002, P22426), which protect the producer from its own colicin and are
    # therefore the antidote. Checked directly: Colicin-E1/N/B/A all carry KW-0078 and
    # Colicin-M immunity protein does not, so the keyword separates agent from accessory
    # and the name does not.
    ("bacteriocin", "bacteria", "bacteria_or_virus",
     "(keyword:KW-0078) AND (taxonomy_name:Bacteria)", 40),
    # KW-0081 is "Bacteriolytic enzyme", restricted to viral producers so this is the phage
    # lysis class rather than every bacteriolytic hydrolase in Swiss-Prot (244 of those).
    # It covers lysozymes, amidases and endopeptidases that `protein_name:endolysin` missed.
    # Named for the mechanism rather than for endolysins specifically. KW-0081 on viral
    # producers returns both end-of-cycle endolysins and virion-associated hydrolases that
    # digest peptidoglycan locally during injection: the T4 baseplate spike Gp5 (P16009),
    # the T5 tape measure protein pb2 (Q6QGE7), phi29 morphogenesis protein 1 (P15132) and
    # T7 gp16 (P03726). All four hydrolyse bacterial cell wall, so the catalytic mechanism
    # is shared and they belong in a mechanism class. Calling that class "endolysin" would
    # have been wrong, since a baseplate spike is not an endolysin, so the class is named
    # for what its members do.
    ("phage_peptidoglycan_hydrolase", "bacteria", "bacteria_or_virus",
     "(keyword:KW-0081) AND (taxonomy_name:Viruses)", 40),
    # Unchanged: all 22 admitted entries carried both Toxin and Virulence keywords, were
    # Bacillus thuringiensis, and were named "Pesticidal crystal protein".
    ("cry_insecticidal", "insect", "bacteria_or_virus",
     '(family:"delta endotoxin family")', 40),
]

# Positive-side name blocklist. A hazard positive has to be the agent. These terms mark
# accessory parts of the same system: the antidote, the delivery machinery, the receptor on
# the target cell, the membrane pore that releases the real agent.
POS_NAME_BLOCK = ("immunity", "receptor", "secretion", "processing", "uptake",
                  "tolerance", "transport", "permease", "holin", "sensitivity",
                  "translocation", "import",
                  # P04481, "Uncharacterized protein in cib 5'region", carries the
                  # Bacteriocin keyword because it sits in the colicin Ib operon. An
                  # uncharacterized ORF is not evidence of a hazardous activity, so
                  # uncharacterized and putative entries stay out of a positive class.
                  "uncharacterized", "putative")

# Producers outside the annotation's `producer_kingdom` vocabulary. target_host_v2.json
# records only bacteria_or_virus and plant, and §2.4.1's producer analysis rests on that
# coding, so archaeal producers are excluded here rather than silently miscoded as
# bacterial. Three halocins (P83716, Q48236, Q9HHA8) and one Methanobacterium-phage
# pseudomurein endoisopeptidase (Q77WJ4) are dropped for this reason, not for quality.
PRODUCER_BLOCK = ("halo", "archae", "methano", "sulfolobus", "thermococcus",
                  "pyrococcus", "haloferax", "halobacterium")

# (group, for which class, uniprot query, per-organism cap, target count)
NEG_GROUPS = [
    ("bt_benign", "cry_insecticidal",
     '(taxonomy_name:"Bacillus thuringiensis")', 999, 42),
    ("phage_benign", "phage_peptidoglycan_hydrolase",
     "(taxonomy_name:Caudoviricetes)", 2, 44),
    ("bacteriocin_producer_benign", "bacteriocin",
     "(taxonomy_name:Lactococcus) OR (taxonomy_name:Lactobacillus) "
     "OR (taxonomy_name:Pediococcus) OR (taxonomy_name:Enterococcus)", 4, 56),
]

# 02d's four, plus the two keywords that DEFINE the classes being added. Without
# KW-0078 and KW-0081 here, a bacteriocin or a phage lysin can be fetched as a benign
# control of its own producer: a first run put O03979, "Lysin" from Pneumococcus phage
# Dp-1, into the endolysin positives AND into the phage negatives, the same sequence on
# both sides of the label. The name blocklist could not catch it, because the name is
# just "Lysin" and bare "lysin" is deliberately not blocked (it substring-matches
# "Lysin motif", see 26). Selecting on the keyword catches it by construction.
BLOCK_KEYWORDS = {"KW-0843": "Virulence", "KW-0800": "Toxin",
                  "KW-0204": "Cytolysis", "KW-0354": "Hemolysis",
                  "KW-0078": "Bacteriocin", "KW-0081": "Bacteriolytic enzyme"}
KW_NAME_BLOCK = ("virulence", "toxin", "cytolysis", "hemolysis", "haemolysis")
# 02d's blocklists, plus the three terms the new positive classes introduce.
NAME_BLOCK = (
    "toxin", "hemolysin", "haemolysin", "cytolysin", "leukocidin", "leucocidin",
    "adhesin", "invasin", "virulence", "lethal factor", "edema factor",
    "oedema factor", "protective antigen", "aerolysin", "listeriolysin",
    "streptolysin", "phospholipase", "coagulase", "intimin", "internalin",
    "hemagglutinin", "haemagglutinin",
)
CLASS_BLOCK = (
    "lactamase", "beta-lactam", "penicillinase", "cephalosporinase", "carbapenemase",
    "adp-ribosyl", "ribosyltransferase", "nuclease", "dnase", "rnase",
    "deoxyribonuclease", "ribonuclease", "protease", "peptidase", "proteinase",
    "glycosidase", "ribosome-inactivating", "rrna n-glycosidase", "pore-forming",
    "perfringolysin", "pneumolysin", "superantigen", "enterotoxin",
    "type iii secretion", "t3ss", "secretion system", "effector", "neurotoxin",
    "contact-dependent",
    # introduced by this expansion: a negative must not be a member of one of the
    # three classes being added, and "lysin" is left out on purpose because it
    # substring-matches "Lysin motif" (see 26's docstring).
    "bacteriocin", "colicin", "microcin", "endolysin", "lysozyme", "muramidase",
    "amidase", "holin", "delta-endotoxin", "crystal protein", "insecticidal",
)


def H(s):
    return hashlib.sha256(s.encode()).hexdigest()


def aligner():
    a = PairwiseAligner()
    a.substitution_matrix = substitution_matrices.load("BLOSUM62")
    a.open_gap_score, a.extend_gap_score, a.mode = -11, -1, "local"
    return a


def clean(s):
    return "".join(c for c in s if c in "ACDEFGHIKLMNPQRSTVWY")


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
    if d.get("recommendedName", {}).get("fullName", {}).get("value"):
        return d["recommendedName"]["fullName"]["value"]
    for v in (d.get("submissionNames") or []) + (d.get("alternativeNames") or []):
        if v.get("fullName", {}).get("value"):
            return v["fullName"]["value"]
    return ""


def query(q, size=FETCH, extra=""):
    url = API + "?" + urllib.parse.urlencode({
        "query": f"({q}) AND reviewed:true AND length:[{LEN_MIN} TO {LEN_MAX}]{extra}",
        "format": "json", "size": size,
        "fields": ("accession,id,organism_name,protein_name,length,sequence,"
                   "keyword,ft_signal,cc_subcellular_location")})
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.load(r).get("results", [])


# localization vocabulary, copied from 02c so both files derive the label the same way.
# First match in list order wins.
LOC_RULES = [
    ("secreted", ("secreted",)),
    ("cell_surface", ("cell wall", "cell surface", "fimbrium")),
    ("membrane", ("membrane",)),
    ("periplasm", ("periplasm",)),
    ("cytoplasmic", ("cytoplasm", "cytosol")),
]

CLASS_REASON = {
    "bacteriocin": ("ribosomally synthesised antibacterial protein, UniProt Bacteriocin "
                    "keyword, lethal to bacteria closely related to the producer"),
    "phage_peptidoglycan_hydrolase": ("phage-encoded enzyme that hydrolyses bacterial "
                                      "peptidoglycan, UniProt Bacteriolytic enzyme keyword; "
                                      "includes end-of-cycle endolysins and "
                                      "virion-associated hydrolases"),
    "cry_insecticidal": ("Bacillus thuringiensis delta-endotoxin, pore-forming in the "
                         "insect midgut epithelium after protease activation"),
}
TARGET_REASON = {
    "bacteria": "kills or lyses another BACTERIUM; no animal host is involved",
    "insect": "acts on INSECT midgut epithelium; no vertebrate target",
}


def derive_localization(locations):
    if not locations:
        return "unknown"
    low = " | ".join(locations).lower()
    for label, keys in LOC_RULES:
        if any(k in low for k in keys):
            return label
    return "other"


def uniprot_localization(accs):
    """Fetch subcellular location and signal-peptide presence, in batches, the same
    fields 02c uses. Returns {accession: {locations, has_signal_peptide}}."""
    out = {}
    for i in range(0, len(accs), 80):
        chunk = accs[i:i + 80]
        url = API + "?" + urllib.parse.urlencode({
            "query": " OR ".join(f"accession:{a}" for a in chunk),
            "format": "json", "size": len(chunk),
            "fields": "accession,ft_signal,cc_subcellular_location"})
        with urllib.request.urlopen(url, timeout=120) as r:
            for rec in json.load(r).get("results", []):
                locs = []
                for c in rec.get("comments", []):
                    if c.get("commentType") == "SUBCELLULAR LOCATION":
                        for sl in c.get("subcellularLocations", []):
                            v = sl.get("location", {}).get("value")
                            if v:
                                locs.append(v)
                sig = any(f.get("type") == "Signal"
                          for f in rec.get("features", []))
                out[rec["primaryAccession"]] = {"locations": locs,
                                                "has_signal_peptide": sig}
        time.sleep(0.2)
    return out


def integrate(stage):
    """Write the staged expansion as panel **v3**, leaving every v2 file untouched.

    🔴 The first version of this function appended to `toxins_positive_v2.fasta` in place.
    That is wrong, and the reason is the whole point of this repository. Every number in
    docs/MECHANISM_GENERALIZATION.md, all fourteen arms in §9, §2.4.1, §9.7 and every pin in
    src/22_claims_audit.py is computed on v2's 80 positives and 154 negatives. Growing those
    files in place makes the published document unreproducible from its own committed inputs,
    which is the failure mode docs/DATA_CORRECTIONS.md records again and again.

    So v2 is frozen and this writes a parallel v3 set: two FASTA files, a manifest, and all
    four annotation files. Every file the coverage audit checks is written in the same pass,
    because the 2026-09-05 expansion updated the FASTA and the mechanism-class file but not
    localization, and 03d and 03g then raised KeyError on the first new member after the
    sweep had already spent GPU hours.

    v3 is v2 plus the new members, in that order, so v2's row indices are a prefix of v3's
    and a v2 embedding is the first 80 (or 154) rows of the v3 one. That is asserted rather
    than assumed, by 28's verification step."""
    pos, neg = stage["accepted_positives"], stage["accepted_negatives"]
    today = time.strftime("%Y-%m-%d")

    print("\nfetching localization for the new members")
    loc_raw = uniprot_localization([x["uniprot"] for x in pos + neg])
    print(f"  got {len(loc_raw)} of {len(pos) + len(neg)}")

    # ---- FASTA: v2 copied verbatim, then the new members, so v2 rows stay a prefix ----
    for src, dst, group in (("toxins_positive_v2.fasta", "toxins_positive_v3.fasta", pos),
                            ("benign_negatives_v2.fasta", "benign_negatives_v3.fasta", neg)):
        body = open(SEQ / src).read()
        if not body.endswith("\n"):
            body += "\n"
        with open(SEQ / dst, "w") as fh:
            fh.write(body)
            for x in group:
                fh.write(f">{x['acc']} {x['protein_name']} OS={x['organism']}\n")
                for i in range(0, len(x["sequence"]), 60):
                    fh.write(x["sequence"][i:i + 60] + "\n")
        print(f"wrote {dst}")

    # ---- panel manifest ----
    man = json.load(open(SEQ / "panel_v2_manifest.json"))
    pf = SEQ / "panel_v3_manifest.json"
    for x in pos:
        man["positives"].append({
            "acc": x["acc"], "name": x["name"], "organism": x["organism"],
            "lab_strain": False, "len": x["length"],
            "truncated_1022": bool(x["length"] and x["length"] > 1022),
            "sha256": x["sha256"]})
    for x in neg:
        man["negatives"].append({
            "acc": x["acc"], "name": x["name"], "organism": x["organism"],
            "lab_strain": False, "len": x["length"],
            "truncated_1022": bool(x["length"] and x["length"] > 1022),
            "sha256": x["sha256"],
            "block": "organism_matched", "pathogen_derived_control": False})
    man["panel"] = "v3"
    man["supersedes"] = "panel_v2_manifest.json, which stays frozen: every published number "\
                        "in docs/ is computed on v2 and must remain reproducible from it"
    man["counts"]["positive_final"] = len(man["positives"])
    man["counts"]["negative_final"] = len(man["negatives"])
    man.setdefault("maintenance_notes", [])
    man["maintenance_notes"].append(
        f"{today}: added {len(pos)} positives in 3 non-animal-target mechanism classes "
        f"and {len(neg)} organism-matched negatives, by src/27_expand_nonanimal_classes.py. "
        "A third negative block, organism_matched, joins secreted_cellwall and "
        "cytoplasmic_housekeeping: it holds benign proteins from the NEW positives' own "
        "producers, which the panel previously had none of for phages or B. thuringiensis.")
    json.dump(man, open(pf, "w"), indent=2)
    print(f"panel manifest: {len(man['positives'])} positives, "
          f"{len(man['negatives'])} negatives")

    # ---- mechanism classes ----
    mech = json.load(open(ANN / "mechanism_classes_v2.json"))
    mf = ANN / "mechanism_classes_v3.json"
    start = max(e["fasta_index"] for e in mech["proteins"]) + 1
    for i, x in enumerate(pos):
        mech["proteins"].append({
            "fasta_index": start + i, "fasta_id": x["acc"],
            "short_name": x["name"], "mechanism_class": x["mechanism_class"],
            "reason": CLASS_REASON[x["mechanism_class"]], "holdout_eligible": True})
    for c in ("bacteriocin", "phage_peptidoglycan_hydrolase", "cry_insecticidal"):
        if c not in mech["holdout_eligible_classes"]:
            mech["holdout_eligible_classes"].append(c)
    mech["holdout_eligible_classes"] = sorted(mech["holdout_eligible_classes"])
    mech.setdefault("class_notes", {})
    mech["class_notes"]["expansion_2026_09_18"] = (
        "Three non-animal-target classes added to make leave-one-class-out able to train "
        "AND test on non-animal hazard. Before this, non-animal target was carried by two "
        "classes, so holding one out left a single example of the category, which is why "
        "MECHANISM_GENERALIZATION §2.4.1's class-level test had no power. "
        "phage_peptidoglycan_hydrolase is named for the mechanism rather than for "
        "endolysins, because UniProt's Bacteriolytic enzyme keyword on viral producers also "
        "returns virion-associated hydrolases (T4 Gp5, T5 pb2, phi29 morphogenesis protein "
        "1, T7 gp16) that digest peptidoglycan during injection rather than at lysis.")
    json.dump(mech, open(mf, "w"), indent=2)
    print(f"mechanism classes: {len(mech['proteins'])} positives, "
          f"{len(mech['holdout_eligible_classes'])} eligible classes")

    # ---- target host ----
    th = json.load(open(ANN / "target_host_v2.json"))
    tf = ANN / "target_host_v3.json"
    for x in pos:
        th["proteins"].append({
            "fasta_id": x["acc"], "short_name": x["name"],
            "mechanism_class": x["mechanism_class"],
            "target_host": x["target_host"],
            "reason": TARGET_REASON[x["target_host"]],
            "producer_kingdom": x["producer_kingdom"]})
    th["categories"].setdefault("insect", "insect host, no vertebrate target")
    th.setdefault("maintenance_notes", []).append(
        f"{today}: +{len(pos)} positives from src/27. Producer kingdom stays "
        "bacteria_or_virus for all of them, deliberately: §2.4.1's finding that producer "
        "taxonomy does not explain target-host legibility rests on the panel's producer "
        "balance, and three archaeal halocins were rejected rather than miscoded.")
    json.dump(th, open(tf, "w"), indent=2)
    print(f"target host: {len(th['proteins'])} positives")

    # ---- localization ----
    loc = json.load(open(ANN / "localization_v2.json"))
    lf = ANN / "localization_v3.json"
    missing = []
    for side, group in (("positives", pos), ("negatives", neg)):
        for x in group:
            r = loc_raw.get(x["uniprot"])
            if r is None:
                missing.append(x["uniprot"])
                r = {"locations": [], "has_signal_peptide": False}
            loc["proteins"][x["acc"]] = {
                "acc": x["uniprot"], "side": side, "locations": r["locations"],
                "has_signal_peptide": r["has_signal_peptide"],
                "localization": derive_localization(r["locations"])}
    loc["not_found_in_uniprot"] = sorted(set(loc.get("not_found_in_uniprot", [])) | set(missing))
    json.dump(loc, open(lf, "w"), indent=2)
    print(f"localization: {len(loc['proteins'])} entries, "
          f"{len(missing)} without a UniProt record")

    # ---- 3Di, masked ----
    st = json.load(open(ANN / "structure_3di_v2.json"))
    sf = ANN / "structure_3di_v3.json"
    for side, group in (("positive", pos), ("negative", neg)):
        for x in group:
            n = min(int(x["length"] or 0), 1022)
            st["proteins"][x["acc"]] = {
                "acc": x["uniprot"], "side": side, "len": x["length"],
                "threedi": "#" * n, "status": "no_structure"}
    st.setdefault("maintenance_notes", []).append(
        f"{today}: {len(pos) + len(neg)} members from src/27 carry the no_structure mask. "
        "foldseek is fetched as a Linux binary by 02h_saprot_prepare.py and the existing "
        "3Di strings were produced on the HPC, so the SaProt arm is NOT evaluated on the "
        "three new classes until structures are fetched. Every other arm is unaffected.")
    json.dump(st, open(sf, "w"), indent=2)
    print(f"3Di: {len(st['proteins'])} entries, new ones masked as no_structure")

    print("\nv2 is untouched. v3 needs its own embeddings before any analysis:")
    print("  python src/02b_esm2_embed_v2.py --panel v3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--integrate", action="store_true")
    ap.add_argument("--from-stage", action="store_true",
                    help="skip the UniProt harvest and build v3 from the existing staging "
                         "file, so the panel is built from exactly the records that were "
                         "reviewed rather than from a fresh query that may differ")
    a = ap.parse_args()
    if a.from_stage:
        if not STAGE.exists():
            raise SystemExit(f"no staging file at {STAGE}; run the dry run first")
        stage = json.load(open(STAGE))
        bad = [r for r in stage["rejected_positives"]
               if r["reason"].startswith("LABEL_CONFLICT")]
        if bad:
            raise SystemExit(f"refusing: {len(bad)} label conflicts in the staging file")
        worst = max((x["max_sim_new_positive"] for x in stage["accepted_negatives"]),
                    default=0.0)
        if worst > HOMOLOGY_MAX:
            raise SystemExit(f"refusing: a staged negative sits at {worst:.3f} similarity "
                             "to a new positive")
        print(f"building v3 from {STAGE} ({stage['built']}): "
              f"{len(stage['accepted_positives'])} positives, "
              f"{len(stage['accepted_negatives'])} negatives")
        integrate(stage)
        return
    al = aligner()

    pos_f = read_fasta(SEQ / "toxins_positive_v2.fasta")
    neg_f = read_fasta(SEQ / "benign_negatives_v2.fasta")
    panel_hashes = {H(s) for s in list(pos_f.values()) + list(neg_f.values())}
    panel_accs = {k.split("|")[1] if "|" in k else k
                  for k in list(pos_f) + list(neg_f)}

    pos_clean = [clean(s) for s in pos_f.values()]
    pos_self = [al.score(s, s) for s in pos_clean]
    neg_clean = [clean(s) for s in neg_f.values()]
    neg_self = [al.score(s, s) for s in neg_clean]
    print(f"panel: {len(pos_f)} positives, {len(neg_f)} negatives\n")

    # ---------------- positives ----------------
    accepted, rejected = [], []
    for cname, target, producer, q, cap in POS_CLASSES:
        recs = query(q)
        kept_clean, kept_self, kept = [], [], 0
        for rec in recs:
            if kept >= cap:
                break
            acc = rec["primaryAccession"]
            s = rec.get("sequence", {}).get("value", "")
            if not s or acc in panel_accs or H(s) in panel_hashes:
                rejected.append({"class": cname, "acc": acc,
                                 "reason": "already_in_panel_or_empty"})
                continue
            pname = protein_name(rec)
            org = rec.get("organism", {}).get("scientificName", "")
            if any(b in pname.lower() for b in POS_NAME_BLOCK):
                rejected.append({"class": cname, "acc": acc,
                                 "reason": "accessory_not_the_agent",
                                 "protein_name": pname})
                continue
            if any(b in org.lower() for b in PRODUCER_BLOCK):
                rejected.append({"class": cname, "acc": acc,
                                 "reason": "producer_outside_annotation_vocabulary",
                                 "organism": org})
                continue
            c = clean(s)
            ss = al.score(c, c)
            if ss <= 0:
                continue
            sim_pos = max((al.score(c, t) / np.sqrt(ss * tf)
                           for t, tf in zip(pos_clean, pos_self)), default=0.0)
            if sim_pos > HOMOLOGY_MAX:
                rejected.append({"class": cname, "acc": acc,
                                 "reason": "homologous_to_existing_positive",
                                 "max_similarity": round(float(sim_pos), 3)})
                continue
            sim_neg = max((al.score(c, t) / np.sqrt(ss * tf)
                           for t, tf in zip(neg_clean, neg_self)), default=0.0)
            if sim_neg > HOMOLOGY_MAX:
                rejected.append({"class": cname, "acc": acc,
                                 "reason": "LABEL_CONFLICT_homologous_to_a_negative",
                                 "max_similarity": round(float(sim_neg), 3)})
                continue
            sim_new = max((al.score(c, t) / np.sqrt(ss * tf)
                           for t, tf in zip(kept_clean, kept_self)), default=0.0)
            if sim_new > HOMOLOGY_MAX:
                rejected.append({"class": cname, "acc": acc,
                                 "reason": "homologous_to_a_candidate_already_accepted",
                                 "max_similarity": round(float(sim_new), 3)})
                continue
            kws = [k.get("name", "") for k in rec.get("keywords", [])]
            accepted.append({
                "acc": f"sp|{acc}|{rec.get('uniProtkbId', acc)}",
                "uniprot": acc, "name": rec.get("uniProtkbId", acc),
                "protein_name": pname, "organism": org,
                "length": rec.get("sequence", {}).get("length"),
                "sequence": s, "sha256": H(s)[:16],
                "mechanism_class": cname, "target_host": target,
                "producer_kingdom": producer,
                "keywords": kws,
                "max_sim_existing_positive": round(float(sim_pos), 3)})
            kept_clean.append(c)
            kept_self.append(ss)
            kept += 1
        print(f"  {cname:<24}{len(recs):>4} fetched -> {kept:>3} accepted")
        time.sleep(0.3)

    rc = collections.Counter(r["reason"] for r in rejected)
    print(f"\npositives accepted {len(accepted)}; rejections {dict(rc)}")
    conflicts = [r for r in rejected if r["reason"].startswith("LABEL_CONFLICT")]
    if conflicts:
        print(f"🔴 {len(conflicts)} label conflicts with the existing negative set: "
              f"{[(c['acc'], c['max_similarity']) for c in conflicts[:5]]}")

    # ---------------- matched negatives ----------------
    # 🔴 Register the accepted positives BEFORE harvesting negatives. Omitting this is how
    # O03979 ended up on both sides: the dedup sets were seeded from the existing panel
    # only, so a sequence admitted as a positive in this same run was still invisible to
    # the negative harvest.
    for x in accepted:
        panel_accs.add(x["uniprot"])
        panel_hashes.add(H(x["sequence"]))

    new_pos_clean = [clean(x["sequence"]) for x in accepted]
    new_pos_self = [al.score(s, s) for s in new_pos_clean]
    excl = " ".join(f" NOT keyword:{k}" for k in BLOCK_KEYWORDS)
    neg_accepted, neg_rejected = [], []
    for gname, forclass, q, per_org, target_n in NEG_GROUPS:
        recs = query(q, size=min(400, target_n * 8), extra=excl)
        per = collections.Counter()
        kept = 0
        for rec in recs:
            if kept >= target_n:
                break
            acc = rec["primaryAccession"]
            s = rec.get("sequence", {}).get("value", "")
            org = rec.get("organism", {}).get("scientificName", "")
            name = protein_name(rec)
            kws = [k.get("name", "") for k in rec.get("keywords", [])]
            why = None
            if not s or acc in panel_accs or H(s) in panel_hashes:
                why = "already_in_panel_or_empty"
            elif per[org] >= per_org:
                why = "per_organism_cap"
            elif any(b in k.lower() for k in kws for b in KW_NAME_BLOCK):
                why = "blocked_keyword"
            elif any(b in name.lower() for b in NAME_BLOCK):
                why = "blocked_name"
            elif any(b in name.lower() for b in CLASS_BLOCK):
                why = "blocked_positive_class_term"
            if why:
                neg_rejected.append({"group": gname, "acc": acc, "reason": why})
                continue
            c = clean(s)
            ss = al.score(c, c)
            sim = max((al.score(c, t) / np.sqrt(ss * tf)
                       for t, tf in zip(new_pos_clean, new_pos_self)), default=0.0)
            neg_accepted.append({
                "acc": f"sp|{acc}|{rec.get('uniProtkbId', acc)}",
                "uniprot": acc, "name": rec.get("uniProtkbId", acc),
                "protein_name": name, "organism": org,
                "length": rec.get("sequence", {}).get("length"),
                "sequence": s, "sha256": H(s)[:16],
                "group": gname, "matched_to_class": forclass,
                "keywords": kws,
                "max_sim_new_positive": round(float(sim), 3)})
            per[org] += 1
            panel_accs.add(acc)
            panel_hashes.add(H(s))
            kept += 1
        print(f"  {gname:<28}{len(recs):>4} fetched -> {kept:>3} accepted "
              f"(target {target_n}, {len(per)} organisms)")
        time.sleep(0.3)

    nrc = collections.Counter(r["reason"] for r in neg_rejected)
    print(f"\nnegatives accepted {len(neg_accepted)}; rejections {dict(nrc)}")
    sims = [x["max_sim_new_positive"] for x in neg_accepted]
    if sims:
        print(f"negative-to-new-positive similarity: max {max(sims):.3f}, "
              f"median {float(np.median(sims)):.3f}, "
              f"above 0.30: {sum(s > HOMOLOGY_MAX for s in sims)}")

    pc = collections.Counter(x["mechanism_class"] for x in accepted)
    print(f"\nwould become: positives {len(pos_f)} -> {len(pos_f) + len(accepted)}, "
          f"negatives {len(neg_f)} -> {len(neg_f) + len(neg_accepted)}, "
          f"prevalence {len(pos_f) / (len(pos_f) + len(neg_f)):.1%} -> "
          f"{(len(pos_f) + len(accepted)) / (len(pos_f) + len(accepted) + len(neg_f) + len(neg_accepted)):.1%}")
    print(f"per new class: {dict(pc)}")
    print("non-animal-target mechanism classes: 2 -> "
          f"{2 + len([c for c in pc if c])}")

    json.dump({"built": time.strftime("%Y-%m-%d"),
               "rule": ("normalized Smith-Waterman score/sqrt(self_i*self_j) <= 0.30, "
                        "BLOSUM62, gaps -11/-1, local, reviewed, length 100-1400"),
               "positive_classes": [
                   {"class": c, "target_host": t, "producer_kingdom": p, "query": q}
                   for c, t, p, q, _ in POS_CLASSES],
               "negative_groups": [
                   {"group": g, "matched_to_class": f, "query": q,
                    "per_organism_cap": cap, "target": n}
                   for g, f, q, cap, n in NEG_GROUPS],
               "accepted_positives": accepted,
               "accepted_negatives": neg_accepted,
               "rejected_positives": rejected,
               "rejected_negatives": neg_rejected,
               "held_back": {
                   "plant_target_avirulence": ("32 admissible, crop-targeting; release "
                                               "decision belongs to the species-stratified "
                                               "project, not to this expansion"),
                   "chitinase_antifungal": ("28 admissible, but most reviewed chitinases are "
                                            "plant defence enzymes, so admitting them as "
                                            "hazard positives is a labelling claim this "
                                            "panel does not support")},
               "structure_note": ("new members take the no_structure 3Di mask; foldseek is a "
                                  "Linux binary run on the HPC, so the SaProt arm is not "
                                  "evaluated on these classes until structures are fetched"),
               }, open(STAGE, "w"), indent=2)
    payload = json.load(open(STAGE))
    print(f"\nwrote staging file {STAGE}")
    if not a.integrate:
        print("dry run: nothing in data/ or results/ was modified. "
              "Re-run with --integrate to apply.")
        return
    if any(r["reason"].startswith("LABEL_CONFLICT") for r in rejected):
        raise SystemExit("refusing to integrate: label conflicts with the negative set")
    sims = [x["max_sim_new_positive"] for x in neg_accepted]
    if sims and max(sims) > HOMOLOGY_MAX:
        raise SystemExit(f"refusing to integrate: a negative sits at "
                         f"{max(sims):.3f} similarity to a new positive")
    integrate(payload)


if __name__ == "__main__":
    main()
