#!/usr/bin/env python3
"""
68_benign_enzyme_set.py - build the matched benign enzyme set study A2 of
                          docs/NEGATIVE_EXPANSION_PREREGISTRATION.md freezes.

What it is for
--------------
`data/annotations/benign_control_sites.json` holds **four** verified controls, and
`docs/MUTATION_EXTENSION_PREREGISTRATION.md` records that three of its four AUROC halves are
unusable at that n, putting the threshold at "somewhere above n_benign of roughly 30". This script
builds that set and **takes no measurement**, exactly as `src/61` does: it supplies an input, so it
cannot be tuned by looking at a result.

⚠️ The preregistration predicts the study this enables will **fail** — at n = 4 the benign controls
already exceed the panel on dFSPE-M (+5.26 mean, astacin +8.42, permutation p = 0.678) and the
mutation axis is read as positional constraint rather than hazard. Nothing here is arranged to avoid
that outcome.

Selection, frozen by the preregistration
---------------------------------------
Candidates are reviewed Swiss-Prot entries carrying `Active site` features, in the panel's own length
window, excluded in the preregistration's order. Two things worth stating plainly:

* **The sample is a deterministic pseudo-random draw from the whole eligible pool**, ordered by
  `sha256(accession)`, not the first N by accession. Accession order is not neutral — `P0xxxx` entries
  are older and better characterised — and taking the head of it would have been a selection choice
  dressed as an ordering. The hash is reproducible and has no relationship to any outcome.
* **Rule 2 cannot be applied as written.** The preregistration says "any accession appearing in VFDB
  setA or setB"; VFDB identifies its records by VFG ids and GenBank/RefSeq accessions, so there is no
  UniProt accession to join on. Applied here as an **exact sequence match** against setB instead, and
  recorded as amendment 1 of the preregistration rather than quietly reinterpreted.

The identity check
------------------
`src/61`'s rule is reused unchanged: the offset must be the **unique** integer in [0, 60] placing
every expected residue identity correctly. ⚠️ Here the positions and the sequence come from the same
UniProt record, so offset 0 is correct by construction and the rule is not testing coordinate
provenance as it does for the PDB-numbered panel. What it still tests is **degeneracy**: if more than
one offset places every residue, the catalytic set is too uninformative to pin a position (all serines,
say), and a control whose sites cannot be pinned is excluded rather than scored.

Outputs
-------
  data/sequences/benign_enzymes.fasta            canonical UniProt sequences
  data/annotations/benign_enzyme_sites.json      verified positions, per candidate
  results/benign_enzyme_set.json                 every admission and every rejection, with reasons

Usage:
    python src/68_benign_enzyme_set.py --refresh      # re-stream the candidate index from UniProt
    python src/68_benign_enzyme_set.py                # build from the cached index
"""

import argparse
import hashlib
import importlib.util
import json
import re
import statistics
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANN = ROOT / "data" / "annotations"
SEQ = ROOT / "data" / "sequences"
INDEX = ROOT / "data" / "external" / "uniprot" / "benign_enzyme_candidates.tsv"
SCREEN = ROOT / "results" / "v3" / "pool_homology_against_panel.json"
VFDB_SETB = ROOT / "data" / "external" / "vfdb" / "VFDB_setB_pro.fas"
OUT_FASTA = SEQ / "benign_enzymes.fasta"
OUT_ANN = ANN / "benign_enzyme_sites.json"
OUT_RES = ROOT / "results" / "benign_enzyme_set.json"

MIN_ACTIVE_SITES = 3      # preregistration exclusion 5: dFSPE-M averages over the catalytic set
LENGTH_FACTOR = 2         # "median length within a factor of 2" of the FSPE panel
TARGET_N = 30             # the preregistration's floor
MAX_CONTROLS = 60         # a disclosed compute cap, applied in hash order, not by any outcome

# Exclusion 3, applied to the protein name and the keyword list. Deliberately broad: a false
# exclusion costs one candidate out of 66,275, a false admission puts a hazard protein in the
# benign control set.
HAZARD_TERMS = ("toxin", "toxic", "virulence", "hemolysin", "haemolysin", "cytolysin", "leukocidin",
                "enterotoxin", "neurotoxin", "aerolysin", "lethal factor", "edema factor",
                "anthrax", "botulinum", "tetanus", "diphtheria", "shiga", "pertussis", "ricin",
                "abrin", "saporin", "gelonin", "pathogenesis", "invasin", "adhesin", "internalin")


def load_src61():
    """Reuse src/61's helpers. The filename starts with a digit, so it needs a spec import."""
    p = ROOT / "src" / "61_benign_control_annotation.py"
    spec = importlib.util.spec_from_file_location("src61", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["src61"] = m
    spec.loader.exec_module(m)
    return m


def panel_length_window():
    """The FSPE panel's own length window, from the cached UniProt records."""
    sites = json.loads((ANN / "functional_sites.json").read_text())
    accs = [k for k in sites if not k.startswith("_")]
    lens = []
    for a in accs:
        f = ROOT / "data" / "uniprot_cache" / f"{a}.json"
        if f.exists():
            lens.append(json.loads(f.read_text())["sequence"]["length"])
    if len(lens) != len(accs):
        raise SystemExit(f"only {len(lens)} of {len(accs)} panel records are cached; "
                         "the window would be computed on a subset")
    med = statistics.median(lens)
    return int(med / LENGTH_FACTOR), int(med * LENGTH_FACTOR), med, len(lens)


def stream_index(lo, hi):
    q = (f"(reviewed:true) AND (ft_act_site:*) AND (length:[{lo} TO {hi}]) "
         "NOT (keyword:KW-0800) NOT (keyword:KW-0843)")
    url = ("https://rest.uniprot.org/uniprotkb/stream?query=" + urllib.parse.quote(q)
           + "&format=tsv&fields=accession,protein_name,length,ec,keyword,ft_act_site,organism_name")
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=600) as r:      # noqa: S310
        INDEX.write_bytes(r.read())
    return q


def read_index():
    rows, header = [], None
    for line in INDEX.read_text(errors="replace").splitlines():
        parts = line.split("\t")
        if header is None:
            header = parts
            continue
        rows.append(dict(zip(header, parts)))
    return rows


def n_active_sites(cell):
    return len(re.findall(r"ACT_SITE\s+(\d+)", cell or ""))


def panel_accessions():
    """Every accession in the v2 and v3 panels, positives and negatives."""
    accs = set()
    for name in ("toxins_positive.fasta", "toxins_positive_v2.fasta", "toxins_positive_v3.fasta",
                 "benign_negatives_v2.fasta", "benign_negatives_v3.fasta", "benign_homologs.fasta",
                 "benign_controls.fasta", "external_positives.fasta", "external_negatives.fasta",
                 "safeprotein_positives.fasta", "safeprotein_negatives.fasta"):
        f = SEQ / name
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            if line.startswith(">"):
                # sp|P02879|RICI_RICCO, or a bare accession
                accs.update(re.findall(r"\b([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9])\b",
                                       line))
    return accs


def vfdb_sequences():
    if not VFDB_SETB.exists():
        return None
    seqs, cur = set(), []
    for line in VFDB_SETB.read_text(errors="replace").splitlines():
        if line.startswith(">"):
            if cur:
                seqs.add("".join(cur))
            cur = []
        else:
            cur.append(line.strip())
    if cur:
        seqs.add("".join(cur))
    return seqs


def aligner():
    """The project's screen, unchanged: local BLOSUM62 -11/-1, score / sqrt(self_i * self_j)."""
    from Bio.Align import PairwiseAligner, substitution_matrices
    a = PairwiseAligner()
    a.mode = "local"
    a.substitution_matrix = substitution_matrices.load("BLOSUM62")
    a.open_gap_score = -11
    a.extend_gap_score = -1
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-stream the candidate index")
    args = ap.parse_args()

    s61 = load_src61()
    lo, hi, med, n_panel = panel_length_window()
    print(f"panel: {n_panel} FSPE proteins, median length {med}, window [{lo}, {hi}]")

    query = None
    if args.refresh or not INDEX.exists():
        print("streaming the candidate index from UniProt ...")
        query = stream_index(lo, hi)
    rows = read_index()
    print(f"candidate index: {len(rows)} reviewed entries with an Active site in the window")

    bound = json.loads(SCREEN.read_text())["panel_negative_reference"]["max"]
    panel = panel_accessions()
    vfdb = vfdb_sequences()
    if vfdb is None:
        raise SystemExit(f"{VFDB_SETB} missing; run python src/67_vfdb_ingest.py --download")
    print(f"exclusion bound: similarity < {bound:.6f} | panel accessions: {len(panel)} | "
          f"VFDB setB sequences: {len(vfdb)}")

    # positives to screen against: the v3 panel
    pos = {}
    cur_id, cur = None, []
    for line in (SEQ / "toxins_positive_v3.fasta").read_text().splitlines():
        if line.startswith(">"):
            if cur_id:
                pos[cur_id] = "".join(cur)
            cur_id, cur = line[1:].split()[0], []
        else:
            cur.append(line.strip())
    if cur_id:
        pos[cur_id] = "".join(cur)
    print(f"screening against {len(pos)} v3 positives")

    # Deterministic pseudo-random order over the whole eligible pool.
    rows.sort(key=lambda r: hashlib.sha256(r["Entry"].encode()).hexdigest())

    al = aligner()
    selfscore = {k: al.score(v, v) for k, v in pos.items()}
    admitted, rejected = [], []
    examined = 0

    for r in rows:
        if len(admitted) >= MAX_CONTROLS:
            break
        acc, name = r["Entry"], r["Protein names"]
        kw = r.get("Keywords", "")
        blob = f"{name} {kw}".lower()

        if acc in panel:                                            # rule 1
            rejected.append({"acc": acc, "rule": 1, "reason": "in the v2 or v3 panel"})
            continue
        if any(t in blob for t in HAZARD_TERMS):                    # rule 3
            hit = next(t for t in HAZARD_TERMS if t in blob)
            rejected.append({"acc": acc, "rule": 3, "reason": f"hazard term {hit!r} in name or keywords"})
            continue
        if n_active_sites(r.get("Active site", "")) < MIN_ACTIVE_SITES:   # rule 5
            rejected.append({"acc": acc, "rule": 5,
                             "reason": f"{n_active_sites(r.get('Active site',''))} Active site features, "
                                       f"fewer than {MIN_ACTIVE_SITES}"})
            continue

        examined += 1
        rec = s61.uniprot(acc)                                      # cached
        seq = rec["sequence"]["value"]
        expected, dropped = s61.expected_from_uniprot(rec)
        # Active site only here: the preregistration says Active site features, and src/61's
        # catalytic-metal Binding sites would reintroduce the structural-Ca(2+) problem entry
        # sixteen of DATA_CORRECTIONS.md records.
        act = {p: seq[p - 1] for p in
               [f["location"]["start"]["value"] for f in rec.get("features", [])
                if f["type"] == "Active site"
                and f["location"]["start"]["value"] == f["location"]["end"]["value"]]
               if 1 <= p <= len(seq)}
        if len(act) < MIN_ACTIVE_SITES:
            rejected.append({"acc": acc, "rule": 5,
                             "reason": f"{len(act)} single-position Active sites in the full record"})
            continue
        if seq in vfdb:                                             # rule 2, as amended
            rejected.append({"acc": acc, "rule": 2, "reason": "exact sequence match in VFDB setB"})
            continue

        offs = s61.unique_offset(seq, act)                          # the identity rule
        if offs != [0]:
            rejected.append({"acc": acc, "rule": "identity",
                             "reason": f"offsets placing every residue: {offs} — "
                                       f"{'degenerate, cannot be pinned' if len(offs) > 1 else 'none'}"})
            continue

        sim = max((al.score(seq, p) / (selfscore[k] * al.score(seq, seq)) ** 0.5)
                  for k, p in pos.items())
        if sim >= bound:                                            # rule 4
            rejected.append({"acc": acc, "rule": 4,
                             "reason": f"similarity {sim:.4f} >= the panel negatives' own "
                                       f"maximum {bound:.4f}"})
            continue

        admitted.append({
            "acc": acc, "name": name, "organism": r.get("Organism", ""),
            "ec": r.get("EC number", ""), "length": len(seq),
            "catalytic_residues": sorted(act), "residues": {str(p): a for p, a in sorted(act.items())},
            "offset": 0, "max_similarity_to_panel_positive": round(sim, 6),
            "excluded_ligand_sites": len(dropped),
        })
        print(f"  + {acc}  {len(act)} sites  len {len(seq):4d}  sim {sim:.3f}  {name[:56]}")

    ec1 = {}
    for a in admitted:
        d = (a["ec"].split(".")[0] or "?").strip()
        ec1[d] = ec1.get(d, 0) + 1

    out = {
        "built": "2026-09-27",
        "purpose": "study A2 of docs/NEGATIVE_EXPANSION_PREREGISTRATION.md: the matched benign "
                   "enzyme set. Supplies an input; takes no measurement.",
        "query": query or "cached index; rerun with --refresh to record the query",
        "panel_window": {"median_length": med, "lo": lo, "hi": hi, "n_panel": n_panel},
        "candidate_pool": len(rows),
        "order": "sha256(accession) ascending — a reproducible draw from the whole pool, not the "
                 "head of accession order",
        "compute_cap": MAX_CONTROLS,
        "similarity_bound": bound,
        "n_admitted": len(admitted),
        "target_n": TARGET_N,
        "meets_target": len(admitted) >= TARGET_N,
        "n_examined_after_cheap_rules": examined,
        "ec_first_digit": dict(sorted(ec1.items())),
        "length": {"min": min((a["length"] for a in admitted), default=None),
                   "median": statistics.median([a["length"] for a in admitted]) if admitted else None,
                   "max": max((a["length"] for a in admitted), default=None)},
        "sites_per_control": sorted(len(a["catalytic_residues"]) for a in admitted),
        "rejected_counts": {},
        "admitted": admitted,
        "rejected": rejected,
    }
    for r in rejected:
        k = str(r["rule"])
        out["rejected_counts"][k] = out["rejected_counts"].get(k, 0) + 1

    OUT_RES.write_text(json.dumps(out, indent=2) + "\n")
    OUT_ANN.write_text(json.dumps({
        "built": out["built"], "purpose": out["purpose"],
        "acceptance_rule": "src/46's: the offset must be the unique integer in [0, 60] placing EVERY "
                           "expected residue identity correctly. Here positions and sequence come "
                           "from the same UniProt record, so this tests degeneracy rather than "
                           "coordinate provenance.",
        "site_rule": "UniProt single-position Active site features only. Binding sites are excluded "
                     "even for catalytic metals, unlike src/61, to avoid the structural-Ca(2+) "
                     "mixture recorded in entry sixteen of docs/DATA_CORRECTIONS.md.",
        "n_verified": len(admitted),
        "proteins": {a["acc"]: a for a in admitted},
    }, indent=2) + "\n")
    with OUT_FASTA.open("w") as fh:
        for a in admitted:
            rec = s61.uniprot(a["acc"])
            s = rec["sequence"]["value"]
            fh.write(f">{a['acc']} {a['name']}\n")
            for i in range(0, len(s), 60):
                fh.write(s[i:i + 60] + "\n")

    print(f"\nadmitted {len(admitted)} (target {TARGET_N}): "
          f"{'MEETS the preregistered floor' if len(admitted) >= TARGET_N else 'BELOW the floor'}")
    print(f"rejections by rule: {out['rejected_counts']}")
    print(f"EC first digit: {out['ec_first_digit']}")
    print(f"sites per control: {out['sites_per_control']}")
    print(f"wrote {OUT_FASTA.relative_to(ROOT)}, {OUT_ANN.relative_to(ROOT)}, "
          f"{OUT_RES.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
