#!/usr/bin/env python3
"""
74_vfdb_negative_set.py - build study A1's hard negative set: VFDB virulence factors that are not toxins.

What A1 is
----------
`docs/NEGATIVE_EXPANSION_PREREGISTRATION.md`, study A1. Criterion 1 of `docs/DETECTOR_CRITERIA.md` is
the project's worst: the panel's negatives split 178 train / 118 calibrate / **0 test**, so every
false-positive figure was measured on negatives the pipeline had already seen. § 2.6.1 measured the
out-of-sample rate against the 8,259-protein benign pool: **7.87%** at a nominal 5% with `np.quantile`
and **5.98%** conformal.

**VFDB's non-toxin virulence factors are the adversarial version of that test.** They are
pathogen-produced and largely secreted, which are the two features the provenance control (AUROC 0.818)
and the localization controls already show this representation reads. A pool of cytosolic housekeeping
proteins is an easy negative set; these are not. A1-2 and A1-3 predict the rate gets **worse** than
7.87% and 5.98%.

This script only builds and screens the set. It computes no rate: `src/49` does that, with the same
calibration split and both estimators, so A1 changes the test negatives and nothing else.

The two exclusions the preregistration froze, and the one it could not state
---------------------------------------------------------------------------
Rule 1, panel membership. ⚠️ **VFDB identifies records by VFG ids and GenBank/RefSeq accessions, so
there is no UniProt accession to join on** — the same obstacle amendment 1 recorded for A2. Applied
here at the sequence level: exact match against the panel, and rule 4 catches near matches anyway.

Rule 4, the similarity bound: maximum normalized local-alignment similarity to any panel positive must
be **below the panel negatives' own maximum**, 0.282, read from `results/v3/pool_homology_against_panel.json`
rather than recomputed. Same aligner as `02d`, `27` and `42`: `Bio.Align.PairwiseAligner` local,
BLOSUM62, gap −11/−1, score normalized by sqrt(self_i * self_j).

⚠️ This screen is the expensive part — every candidate against all 149 v3 positives — and it is run in
full rather than prefiltered, because a cheaper prefilter would be a different rule than the frozen one.

Usage:
    python src/74_vfdb_negative_set.py
"""

import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VFDB = ROOT / "data" / "external" / "vfdb" / "VFDB_setA_pro.fas"
SCREEN = ROOT / "results" / "v3" / "pool_homology_against_panel.json"
POS = ROOT / "data" / "sequences" / "toxins_positive_v3.fasta"
OUT_FASTA = ROOT / "data" / "sequences" / "vfdb_negatives.fasta"
OUT_RES = ROOT / "results" / "vfdb_negative_set.json"
STANDARD = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LEN, MAX_LEN = 50, 1022      # 1022 is src/02b's truncation limit

HEADER = re.compile(
    r"^>(?P<vfg>VFG\d+)(?:\([^)]*\))?\s*(?:\([^)]*\)\s*)?.*?"
    r"\[[^\[\]]*?\(VF\d+\)\s*-\s*(?P<cat>[^\[\]]*?)\(VFC\d+\)\]\s*\[(?P<org>[^\[\]]+)\]\s*$")


def read_fasta(path):
    out, hid, buf = [], None, []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith(">"):
            if hid:
                out.append((hid, "".join(buf)))
            hid, buf = line, []
        elif hid:
            buf.append(line.strip())
    if hid:
        out.append((hid, "".join(buf)))
    return out


def main():
    bound = json.loads(SCREEN.read_text())["panel_negative_reference"]["max"]
    pos = {h.split()[0].lstrip(">"): s for h, s in read_fasta(POS)}
    panel_seqs = set(pos.values())
    print(f"similarity bound {bound:.6f} | {len(pos)} v3 positives to screen against")

    # ---- category filter and basic hygiene --------------------------------------------------
    cand, rejected = {}, []
    for h, s in read_fasta(VFDB):
        m = HEADER.match(h)
        if not m:
            raise SystemExit(f"unparsed VFDB header:\n{h}")
        d = m.groupdict()
        cat = d["cat"].strip()
        vfg = d["vfg"]
        if cat == "Exotoxin":
            rejected.append({"vfg": vfg, "rule": "category", "reason": "Exotoxin is A1's POSITIVE "
                                                                      "class, not a negative"})
            continue
        if not (MIN_LEN <= len(s) <= MAX_LEN):
            rejected.append({"vfg": vfg, "rule": "length", "reason": f"length {len(s)} outside "
                                                                     f"[{MIN_LEN}, {MAX_LEN}]"})
            continue
        if set(s) - STANDARD:
            rejected.append({"vfg": vfg, "rule": "alphabet",
                             "reason": f"non-standard residues {sorted(set(s) - STANDARD)}"})
            continue
        if s in panel_seqs:                                   # rule 1, at the sequence level
            rejected.append({"vfg": vfg, "rule": 1, "reason": "exact sequence match to a panel protein"})
            continue
        if s in cand:                                         # VFDB carries the same protein per strain
            rejected.append({"vfg": vfg, "rule": "duplicate",
                             "reason": f"identical sequence already kept as {cand[s]['vfg']}"})
            continue
        cand[s] = {"vfg": vfg, "category": cat, "organism": d["org"].strip(), "len": len(s)}
    print(f"after category, length, alphabet and duplicate filters: {len(cand)} candidates "
          f"({len(rejected)} rejected)")

    # ---- rule 4, the similarity screen ------------------------------------------------------
    from Bio.Align import PairwiseAligner, substitution_matrices
    al = PairwiseAligner()
    al.mode = "local"
    al.substitution_matrix = substitution_matrices.load("BLOSUM62")
    al.open_gap_score, al.extend_gap_score = -11, -1
    selfpos = {k: al.score(v, v) for k, v in pos.items()}

    admitted, t0 = [], time.time()
    for i, (s, meta) in enumerate(sorted(cand.items(), key=lambda kv: kv[1]["vfg"])):
        ss = al.score(s, s)
        best, who = 0.0, None
        for k, v in pos.items():
            sim = al.score(s, v) / (selfpos[k] * ss) ** 0.5
            if sim > best:
                best, who = sim, k
        if best >= bound:
            rejected.append({"vfg": meta["vfg"], "rule": 4,
                             "reason": f"similarity {best:.4f} >= the panel negatives' own maximum "
                                       f"{bound:.4f}", "nearest_positive": who})
        else:
            admitted.append({**meta, "max_similarity_to_panel_positive": round(best, 6),
                             "nearest_positive": who, "seq": s})
        if (i + 1) % 200 == 0 or i + 1 == len(cand):
            el = time.time() - t0
            print(f"  screened {i + 1}/{len(cand)}  {el:.0f}s  "
                  f"(eta {el / (i + 1) * (len(cand) - i - 1) / 60:.0f} min)  "
                  f"admitted {len(admitted)}", flush=True)

    import collections
    cats = collections.Counter(a["category"] for a in admitted)
    rc = collections.Counter(str(r["rule"]) for r in rejected)
    OUT_RES.write_text(json.dumps({
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "study A1 of docs/NEGATIVE_EXPANSION_PREREGISTRATION.md: hard negatives. Supplies "
                   "an input; computes no rate. src/49 measures it.",
        "source": "VFDB setA, non-Exotoxin records",
        "similarity_bound": bound, "aligner": "Bio.Align.PairwiseAligner local BLOSUM62 -11/-1, "
                                              "score/sqrt(self_i*self_j), as 02d/27/42",
        "n_admitted": len(admitted), "n_rejected": len(rejected),
        "rejected_counts": dict(sorted(rc.items())),
        "categories": dict(cats.most_common()),
        "max_similarity_reached": max(a["max_similarity_to_panel_positive"] for a in admitted),
        "length": {"min": min(a["len"] for a in admitted), "max": max(a["len"] for a in admitted)},
        "admitted": [{k: v for k, v in a.items() if k != "seq"} for a in admitted],
        "rejected": rejected,
    }, indent=2) + "\n")
    with OUT_FASTA.open("w") as fh:
        for a in admitted:
            fh.write(f">{a['vfg']} {a['category']} [{a['organism']}]\n")
            for j in range(0, len(a["seq"]), 60):
                fh.write(a["seq"][j:j + 60] + "\n")
    print(f"\nadmitted {len(admitted)} hard negatives, max similarity reached "
          f"{max(a['max_similarity_to_panel_positive'] for a in admitted):.4f}")
    print(f"rejections by rule: {dict(sorted(rc.items()))}")
    print(f"categories: {dict(cats.most_common(6))}")
    print(f"wrote {OUT_FASTA.relative_to(ROOT)} and {OUT_RES.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
