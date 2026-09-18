#!/usr/bin/env python3
"""
26_panel_growth_yield.py - how much can the panel actually grow, measured rather than guessed.

The question this answers
-------------------------
§2.4.1 ended with a prerequisite: non-animal target is carried by **two** mechanism classes, so
no leave-one-class-out design can both train and test on non-animal hazard. Fixing that needs
new CLASSES, and the useful question is not how many reviewed entries UniProt holds for a
candidate family. It is how many survive the panel's own admission rule.

That rule, from `src/23_expand_small_classes.py`: normalized Smith-Waterman
`score / sqrt(self_i * self_j)` at **<= 0.30**, BLOSUM62, gaps -11/-1, local, reviewed entries
only, length 100 to 1400. A candidate must clear it against every member already accepted for
its class in this run, which is what stops a family of close homologues from inflating n.

Raw supply and admissible supply differ by an order of magnitude for families defined by a
conserved fold, which is exactly what these candidates are. So this script reports both.

What it does NOT do
-------------------
⚠️ It admits nothing. Nothing is written to `data/`, no annotation file changes, and the panel
is untouched. The output is a yield table for deciding whether an expansion is worth running.

⚠️ Candidates are also screened against the existing 80 positives, because a "new class" whose
members are homologous to something already in the panel is not a new class. That screen runs
only on members that already survived the within-class screen, to keep the alignment count down.

⚠️ **Supply is not a label.** These counts say a family has enough independent sequences to
form a class. They do not say its members are hazardous under this panel's definition. Chitinase
is the clearest case: most reviewed entries are plant defence enzymes, and admitting them as
hazard positives would be a labelling decision, not a retrieval result. Same question applies to
bacteriocins. Whether a candidate belongs in a hazard panel is settled by reading the entries,
not by this table.

⚠️ **The artifact is a dated snapshot of live UniProt queries**, so counts move as Swiss-Prot is
curated. It is deliberately NOT pinned in `src/22_claims_audit.py`: an audit claim asserting
exact counts here would fail CI on someone else's curation work rather than on a defect.

⚠️ Greedy acceptance in UniProt's default result order, the same as `23`. A different order can
shift which representatives are kept and can change the count slightly. The number is a
realistic estimate, not an exact maximum: the true maximum is a maximum-independent-set problem
that this does not solve.

⚠️ Length floor 100 excludes most venom peptides, whose mature chains run 10 to 40 residues.
The `conotoxin_short` row exists to measure that cost rather than leave it implicit.

Usage:
    python src/26_panel_growth_yield.py
    python src/26_panel_growth_yield.py --only cry_insecticidal,bacteriocin
"""

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from Bio.Align import PairwiseAligner, substitution_matrices

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "results" / "v2"
API = "https://rest.uniprot.org/uniprotkb/search"
HOMOLOGY_MAX = 0.30
FETCH = 100          # one UniProt page; the families here do not exceed it after filtering
LEN_MIN, LEN_MAX = 100, 1400

# target_host follows data/annotations/target_host_v2.json's vocabulary, so a yield here maps
# directly onto the count §2.4.1 says is the blocker.
CANDIDATES = [
    ("cry_insecticidal", "insect", '(family:"delta endotoxin family")'),
    # bacteriocin and colicin are ONE class, not two: run separately they returned 34 and
    # 36 reviewed entries sharing 21 accessions, union 49. Colicins ARE bacteriocins, and
    # counting them as two classes would have overstated the reachable class count by one.
    ("bacteriocin_incl_colicin", "bacteria",
     "(keyword:KW-0078) OR (protein_name:colicin)"),
    # `protein_name:lysin` substring-matches "Lysin motif", which pulled in plant LysM
    # receptors such as Q8H8C7, a rice chitin-elicitor-binding protein that lyses nothing.
    # Restricted to endolysin, which is the phage cell-wall hydrolase meant here.
    ("endolysin", "bacteria", "(protein_name:endolysin)"),
    ("plant_target_avirulence", "plant", "(protein_name:avirulence)"),
    ("chitinase_antifungal", "other_nonanimal", "(ec:3.2.1.14)"),
    # animal-target, for comparison: these would raise the class count without touching
    # the non-animal deficit, and they change the producer balance (see §2.4.1)
    ("conotoxin", "animal", "(taxonomy_name:Conus) AND (keyword:KW-0800)"),
    ("scorpion_channel_toxin", "animal",
     '(family:"long (4 C-C) scorpion toxin superfamily")'),
    ("venom_phospholipase_a2", "animal",
     '(family:"phospholipase A2 family") AND (keyword:KW-0800)'),
    ("fungal_toxin", "animal", "(taxonomy_name:Fungi) AND (keyword:KW-0800)"),
]
SHORT_PROBE = ("conotoxin_short", "animal",
               "(taxonomy_name:Conus) AND (keyword:KW-0800)", 10, 99)


def aligner():
    a = PairwiseAligner()
    a.substitution_matrix = substitution_matrices.load("BLOSUM62")
    a.open_gap_score, a.extend_gap_score, a.mode = -11, -1, "local"
    return a


def clean(s):
    return "".join(c for c in s if c in "ACDEFGHIKLMNPQRSTVWY")


def fetch(q, lo=LEN_MIN, hi=LEN_MAX):
    url = API + "?" + urllib.parse.urlencode({
        "query": f"({q}) AND reviewed:true AND length:[{lo} TO {hi}]",
        "format": "json", "size": FETCH,
        "fields": "accession,id,organism_name,length,sequence"})
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.loads(r.read()).get("results", [])


def count_total(q, lo=LEN_MIN, hi=LEN_MAX):
    url = API + "?" + urllib.parse.urlencode({
        "query": f"({q}) AND reviewed:true AND length:[{lo} TO {hi}]", "size": 0})
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("x-total-results", 0))


def greedy_independent(seqs, al):
    """Accept in order, keeping only sequences <= HOMOLOGY_MAX against everything already kept.
    Same rule as 23, and the same limitation: order-dependent, so a realistic count rather
    than the true maximum independent set."""
    kept, kept_self = [], []
    for s in seqs:
        c = clean(s)
        if not c:
            continue
        ss = al.score(c, c)
        if ss <= 0:
            continue
        sim = max((al.score(c, t) / np.sqrt(ss * tf)
                   for t, tf in zip(kept, kept_self)), default=0.0)
        if sim <= HOMOLOGY_MAX:
            kept.append(c)
            kept_self.append(ss)
    return kept, kept_self


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated candidate names")
    a = ap.parse_args()
    want = set(a.only.split(",")) if a.only else None

    al = aligner()

    # the panel this has to be non-homologous to
    pos = {}
    acc = None
    for line in open(ROOT / "data/sequences/toxins_positive_v2.fasta"):
        if line.startswith(">"):
            acc = line[1:].split()[0]
            pos[acc] = []
        elif acc:
            pos[acc].append(line.strip())
    panel = [clean("".join(v)) for v in pos.values()]
    panel_self = [al.score(s, s) for s in panel]
    print(f"existing panel: {len(panel)} positives to screen against\n")

    rows, jobs = {}, list(CANDIDATES) + [SHORT_PROBE[:3]]
    lens = {SHORT_PROBE[0]: (SHORT_PROBE[3], SHORT_PROBE[4])}
    hdr = f"{'candidate class':<32}{'target':>16}{'reviewed':>10}{'fetched':>9}{'indep':>7}{'new':>6}"
    print(hdr)
    print("-" * len(hdr))
    for name, target, q in jobs:
        if want and name not in want:
            continue
        lo, hi = lens.get(name, (LEN_MIN, LEN_MAX))
        try:
            total = count_total(q, lo, hi)
            recs = fetch(q, lo, hi)
        except Exception as e:
            print(f"{name:<32}{target:>16}{'query failed: ' + type(e).__name__:>25}")
            rows[name] = {"error": type(e).__name__}
            continue
        seqs = [r["sequence"]["value"] for r in recs if r.get("sequence")]
        kept, kept_self = greedy_independent(seqs, al)

        # of the independent members, how many are also unlike everything in the panel?
        novel = []
        for s, ss in zip(kept, kept_self):
            sim = max((al.score(s, t) / np.sqrt(ss * tf)
                       for t, tf in zip(panel, panel_self)), default=0.0)
            if sim <= HOMOLOGY_MAX:
                novel.append(s)
        rows[name] = {"target_host": target, "query": q,
                      "length_window": [lo, hi],
                      "reviewed_total": total, "fetched": len(seqs),
                      "independent_at_0.30": len(kept),
                      "novel_vs_panel": len(novel),
                      "collapse_ratio": (round(len(kept) / len(seqs), 3)
                                         if seqs else None)}
        print(f"{name:<32}{target:>16}{total:>10}{len(seqs):>9}"
              f"{len(kept):>7}{len(novel):>6}", flush=True)
        time.sleep(0.3)

    ok = {k: v for k, v in rows.items() if "error" not in v}
    nonanimal = [k for k, v in ok.items()
                 if v["target_host"] != "animal" and v["novel_vs_panel"] >= 4]
    animal = [k for k, v in ok.items()
              if v["target_host"] == "animal" and v["novel_vs_panel"] >= 4]
    print(f"\nnon-animal candidate classes reaching n>=4 after screening: "
          f"{len(nonanimal)} {nonanimal}")
    print(f"animal candidate classes reaching n>=4: {len(animal)} {animal}")
    print(f"non-animal classes today: 2 (beta_lactamase, contact_dependent_inhibition)"
          f" -> {2 + len(nonanimal)} if all of the above were admitted")
    if ok:
        worst = min(ok.items(), key=lambda kv: kv[1]["collapse_ratio"] or 1)
        print(f"heaviest collapse under the 0.30 rule: {worst[0]}, "
              f"{worst[1]['collapse_ratio']:.2f} of fetched survive")

    dest = V2 / "panel_growth_yield.json"
    json.dump({"rule": ("normalized Smith-Waterman score/sqrt(self_i*self_j) <= 0.30, "
                        "BLOSUM62, gaps -11/-1, local, reviewed only"),
               "homology_max": HOMOLOGY_MAX, "fetch_cap": FETCH,
               "length_window_default": [LEN_MIN, LEN_MAX],
               "panel_positives_screened_against": len(panel),
               "candidates": rows,
               "nonanimal_reaching_4": nonanimal, "animal_reaching_4": animal,
               "nonanimal_classes_today": 2,
               "note": ("greedy acceptance in UniProt result order, so counts are realistic "
                        "rather than maximal; nothing here is admitted to the panel")},
              open(dest, "w"), indent=2)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
