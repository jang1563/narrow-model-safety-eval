#!/usr/bin/env python3
"""
42_pool_homology_against_panel.py - does the 8,259-protein benign pool contain homologs of the
                                    panel's POSITIVES, admitted as negatives?

Why this has to be asked after §10.9 rather than before
------------------------------------------------------
§2's panel is built on one admission rule: a candidate is admitted only if its normalized
Smith-Waterman similarity to every existing member is **≤ 0.30**. §2.1 applies the same threshold to
count effective n. Every homology claim in this project runs through that number.

`34` did not apply it to the pool. It dedups against both panels on **accession and sequence hash**,
which catches only exact duplicates, and it filters by UniProt keyword and by protein name. Nothing
in it screens a pool candidate against a panel positive by sequence.

🔴 The name filter demonstrably leaks, which is what prompted this. `CLASS_BLOCK` blocks "endolysin",
"lysozyme", "muramidase", "amidase" and "holin", so the phage class's canonical names are covered.
It does not block "peptidoglycan hydrolase", "autolysin" or "peptidoglycan", and the pool contains
**seven Staphylococcus "Bifunctional autolysin" entries plus "Peptidoglycan hydrolase PcsB"**. The
autolysins are bifunctional amidase/glucosaminidases, so the very domain `CLASS_BLOCK` names is
present under a protein name that does not contain the word. A name filter sees names.

Beta-lactamase, by contrast, is covered: `CLASS_BLOCK` carries "lactamase", "beta-lactam",
"penicillinase", "cephalosporinase" and "carbapenemase", and the pool contains **zero** entries whose
name mentions any of them.

So the two failing classes were filtered with different effective stringency, and §10.9.1's central
result is that they respond to the pool in opposite directions. That coincidence has to be checked by
sequence rather than by name.

PREREGISTERED, written before the run
-------------------------------------
    H1  No pool protein exceeds 0.30 normalized SW against any panel positive. Then the pool
        respects the panel's own admission rule despite never having been screened by it, §10.9's
        experiment is clean on this axis, and the eight cell-wall entries are functional analogues
        rather than sequence homologs.

    H2  Some do. Then the pool contains homologs of positives admitted as negatives, §10.9's result
        must be reported with that count, and the boundary arm has to be re-run without them before
        the +23.5 for the phage class can stand.

    Reported either way, per class: the number of pool proteins above 0.30, above 0.50, and the
    single highest similarity with both names, because one homolog at 0.9 and forty at 0.31 are
    different problems.

⚠️ This is a census, not a sample: every one of the 8,259 pool proteins is compared with every one of
the 149 positives, 1.23 million local alignments. `36` sampled because all-pairs within the pool is
34 million; this direction is 3% of that and needs no extrapolation.

🔑 The reference that makes any of it interpretable. §2's 0.30 rule governs POSITIVES against
positives. `27` states plainly that negatives are **not** homology-screened against the positives,
because mechanism-matched benign proteins are wanted as hard negatives and `benign_homologs.fasta`
exists for exactly that. So the pool matching that policy is not by itself a defect. What decides
whether a pool number is an outlier is what the panel's **own** 296 negatives reach under the same
policy, so the run measures that too: 44,104 more alignments, reported beside the pool's.

Usage:
    python src/42_pool_homology_against_panel.py
    python src/42_pool_homology_against_panel.py --classes beta_lactamase phage_peptidoglycan_hydrolase
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from Bio.Align import PairwiseAligner, substitution_matrices

ROOT = Path(__file__).resolve().parent.parent
SEQ = ROOT / "data" / "sequences"
V3 = ROOT / "results" / "v3"
# §2's threshold, and the stricter one, so the answer does not depend on one cut.
THRESH = 0.30
REPORT_CUTS = (0.30, 0.40, 0.50, 0.70)
AA = "ACDEFGHIKLMNPQRSTVWY"


def read_fasta(path):
    # desc is initialised here rather than only inside the header branch. At runtime it is always
    # bound whenever `key` is truthy, because the two are assigned together, but static analysis
    # cannot see that and ruff fails CI on it (F821). Initialising costs nothing and the invariant
    # is worth stating: the record written on a header line is the PREVIOUS one, so `desc` is
    # deliberately one header behind `buf`.
    out, key, desc, buf = {}, None, "", []
    for line in open(path):
        if line.startswith(">"):
            if key:
                out[key] = ("".join(buf), desc)
            head = line[1:].rstrip()
            key, _, desc = head.partition(" ")
            buf = []
        else:
            buf.append(line.strip())
    if key:
        out[key] = ("".join(buf), desc)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="*", default=None,
                    help="restrict to these mechanism classes; default is every positive")
    a = ap.parse_args()

    pool = read_fasta(SEQ / "benign_pool_large.fasta")
    pos = read_fasta(SEQ / "toxins_positive_v3.fasta")
    mech = json.load(open(ROOT / "data/annotations/mechanism_classes_v3.json"))
    cls = {e["fasta_id"]: e["mechanism_class"] for e in mech["proteins"]}
    keys = [k for k in pos if (a.classes is None or cls.get(k) in a.classes)]
    unmapped = [k for k in keys if k not in cls]
    if unmapped:
        raise SystemExit(f"{len(unmapped)} positives have no class, e.g. {unmapped[:3]}")

    al = PairwiseAligner()
    al.substitution_matrix = substitution_matrices.load("BLOSUM62")
    al.open_gap_score, al.extend_gap_score, al.mode = -11, -1, "local"

    def clean(s):
        return "".join(c for c in s if c in AA)

    pseq = {k: clean(pos[k][0]) for k in keys}
    pself = {k: al.score(v, v) for k, v in pseq.items()}
    print(f"pool {len(pool)}, positives screened {len(keys)} "
          f"across {len(set(cls[k] for k in keys))} classes")
    print(f"{len(pool) * len(keys):,} local alignments, BLOSUM62, gap -11/-1, "
          f"normalized score/sqrt(self_i*self_j)\n")

    best = {}          # pool acc -> (max sim, positive key)
    per_class = {c: [] for c in set(cls[k] for k in keys)}
    t0 = time.time()
    for n, (pk, (pv, pdesc)) in enumerate(pool.items(), 1):
        s = clean(pv)
        if not s:
            continue
        sself = al.score(s, s)
        top, top_k = 0.0, None
        by_class = {}
        for k in keys:
            v = al.score(s, pseq[k]) / np.sqrt(sself * pself[k])
            c = cls[k]
            if v > by_class.get(c, 0.0):
                by_class[c] = v
            if v > top:
                top, top_k = v, k
        best[pk] = (float(top), top_k, pdesc.split(" OS=")[0])
        for c, v in by_class.items():
            per_class[c].append(float(v))
        if n % 1000 == 0:
            el = time.time() - t0
            print(f"  {n}/{len(pool)}  {el:.0f}s elapsed, "
                  f"{el / n * (len(pool) - n):.0f}s left", flush=True)

    # ---- the answer ---------------------------------------------------------------
    print(f"\n{'class':<34}" + "".join(f"{f'>{c:.2f}':>9}" for c in REPORT_CUTS)
          + f"{'max':>8}{'n pos':>7}")
    print("-" * (34 + 9 * len(REPORT_CUTS) + 15))
    summary = {}
    for c in sorted(per_class):
        v = np.array(per_class[c])
        counts = {f"{cut:.2f}": int((v > cut).sum()) for cut in REPORT_CUTS}
        summary[c] = {"n_positives": sum(1 for k in keys if cls[k] == c),
                      "counts_above": counts, "max": float(v.max()),
                      "mean": float(v.mean())}
        print(f"{c:<34}" + "".join(f"{counts[f'{cut:.2f}']:>9}" for cut in REPORT_CUTS)
              + f"{v.max():>8.3f}{summary[c]['n_positives']:>7}")

    over = sorted((v for v in best.items() if v[1][0] > THRESH),
                  key=lambda kv: -kv[1][0])
    print(f"\npool proteins above the panel's own {THRESH} admission threshold: "
          f"{len(over)} of {len(best)}")
    for pk, (sim, tk, name) in over[:25]:
        print(f"  {sim:.3f}  {pk.split('|')[1]:<11}{name[:44]:<46} -> "
              f"{cls[tk]} / {tk.split('|')[1]}")
    if len(over) > 25:
        print(f"  ... {len(over) - 25} more")

    # ---- the reference the pool's numbers have to be read against --------------------
    # 🔴 §2's 0.30 rule governs POSITIVES against positives. `27` states plainly that negatives are
    # NOT homology-screened against the positives, because mechanism-matched benign proteins are
    # wanted as hard negatives and `benign_homologs.fasta` exists for exactly that. So "the pool
    # was never screened" is not by itself a defect: it matches the panel's own policy. What decides
    # whether a pool number is an outlier is what the panel's own negatives reach under the same
    # policy, and that has to be measured rather than assumed.
    neg = read_fasta(SEQ / "benign_negatives_v3.fasta")
    print(f"\nreference: the panel's own {len(neg)} negatives against the same positives, "
          f"{len(neg) * len(keys):,} alignments")
    nb = []
    for nk, (nv, nd) in neg.items():
        sq = clean(nv)
        if not sq:
            continue
        ss = al.score(sq, sq)
        top, tk = 0.0, None
        for k in keys:
            v = al.score(sq, pseq[k]) / np.sqrt(ss * pself[k])
            if v > top:
                top, tk = v, k
        nb.append((float(top), nk.split("|")[1], nd.split(" OS=")[0], tk))
    nb.sort(reverse=True)
    n_over = [r for r in nb if r[0] > THRESH]
    print(f"  panel negatives above {THRESH}: {len(n_over)} of {len(nb)}; "
          f"highest {nb[0][0]:.3f} ({nb[0][1]} {nb[0][2][:40]} -> {cls[nb[0][3]]})")
    for sim, acc, nm, tk in nb[1:4]:
        print(f"  then {sim:.3f}  {acc:<11}{nm[:40]:<42} -> {cls[tk]}")
    panel_ref = {"n_negatives": len(nb), "n_above_threshold": len(n_over),
                 "max": nb[0][0], "max_negative": nb[0][1], "max_name": nb[0][2],
                 "max_positive_class": cls[nb[0][3]],
                 "top5": [{"similarity": s, "negative": a, "name": n,
                           "positive_class": cls[t]} for s, a, n, t in nb[:5]]}
    if over:
        ratio = over[0][1][0] / nb[0][0]
        print(f"  so the pool's highest, {over[0][1][0]:.3f}, is {ratio:.1f}x the highest the panel's "
              f"own negative set reaches under the same policy")
        panel_ref["pool_max_over_panel_max"] = float(ratio)

    verdict = ("H1: no pool protein reaches the panel's 0.30 admission threshold against any "
               "positive, so the pool respects a rule it was never screened by and the cell-wall "
               "entries are functional analogues rather than sequence homologs"
               if not over else
               f"H2: {len(over)} pool proteins exceed 0.30 against a panel positive, so the pool "
               f"contains sequence homologs of positives admitted as negatives. The highest is "
               f"{over[0][1][0]:.3f}, against {nb[0][0]:.3f} for the panel's own negative set under "
               f"the same no-screen policy. Any script that trains on the pool has to drop them")
    print(f"\nverdict: {verdict}")

    dest = V3 / "pool_homology_against_panel.json"
    json.dump({"pool_n": len(pool), "positives_screened": len(keys),
               "classes": sorted(per_class), "threshold": THRESH,
               "alignments": len(pool) * len(keys),
               "per_class": summary, "panel_negative_reference": panel_ref,
               "above_threshold": [{"pool_acc": pk.split("|")[1], "pool_name": nm,
                                    "similarity": sim, "positive": tk.split("|")[1],
                                    "positive_class": cls[tk]}
                                   for pk, (sim, tk, nm) in over],
               "aligner": ("Bio.Align.PairwiseAligner local BLOSUM62 gap -11/-1, "
                           "normalized score/sqrt(self_i*self_j), the same screen as 02d and 27"),
               "not_a_sample": ("every pool protein against every screened positive; 36 sampled "
                                "because all-pairs WITHIN the pool is 34 million, this is 3% of that"),
               "policy_note": ("§2's 0.30 rule governs positives against positives. 27 states that "
                               "negatives are NOT screened against positives, because mechanism-matched "
                               "benign proteins are wanted as hard negatives. So the pool matching that "
                               "policy is not itself a defect; what makes a pool number an outlier is "
                               "what the panel's own negatives reach under the same policy"),
               "verdict": verdict}, open(dest, "w"), indent=2)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
