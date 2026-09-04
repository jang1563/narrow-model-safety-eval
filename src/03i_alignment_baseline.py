#!/usr/bin/env python3
"""
03i_alignment_baseline.py - alignment as both the homology control and the operational baseline.

Two open items from the 2026-09-04 review, closed by the same computation.

1. The homology control was 5-mer Jaccard, which is crude. Remote homologues can
   share almost no 5-mers, so "the probe generalizes representationally, not
   homologically" rested on a proxy that could miss exactly the cases that matter.
   This recomputes it with local Smith-Waterman under BLOSUM62 and BLAST-like gap
   penalties, normalized by self-score so long and short proteins are comparable.

2. There was no operational baseline. The first question a reviewer asks is what
   plain alignment recovers, since that is what deployed screening actually does.
   This runs the identical leave-one-mechanism-out protocol with an alignment
   score in place of the model: score = best alignment to a training positive
   minus best alignment to a training negative, thresholded on the same held-out
   negatives. If alignment matches the probe on a class, the model adds nothing
   there.

Three rules are then directly comparable on one panel, one protocol:
    probe        logistic regression on embeddings          (03b)
    embedding    unfitted cosine nearest-neighbour margin   (03h)
    alignment    unfitted Smith-Waterman margin             (here)

Biopython's PairwiseAligner is used because no blast, diamond or mmseqs binary is
available in this environment. Scores are equivalent for ranking purposes; the
speed difference does not matter at 220 sequences.

Usage:
    python src/03i_alignment_baseline.py [--tag esmc_600M] [--seeds 30]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from Bio.Align import PairwiseAligner, substitution_matrices

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "results" / "v2"
CACHE = V2 / "alignment_scores.npz"
NEG_HOLDOUT_FRAC = 0.40
SPEC = 0.95


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


def build_aligner():
    a = PairwiseAligner()
    a.substitution_matrix = substitution_matrices.load("BLOSUM62")
    a.open_gap_score = -11          # BLAST protein defaults
    a.extend_gap_score = -1
    a.mode = "local"
    return a


def score_matrix(seqs):
    """Normalized local-alignment similarity, S(i,j) / sqrt(S(i,i) S(j,j))."""
    al = build_aligner()
    n = len(seqs)
    clean = ["".join(c for c in s if c in "ACDEFGHIKLMNPQRSTVWY") for s in seqs]
    self_s = np.array([al.score(s, s) for s in clean], float)
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            v = al.score(clean[i], clean[j]) / np.sqrt(self_s[i] * self_s[j])
            M[i, j] = M[j, i] = v
        if (i + 1) % 25 == 0 or i + 1 == n:
            print(f"    aligned {i + 1}/{n} rows", flush=True)
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--seeds", type=int, default=30)
    a = ap.parse_args()
    suf = f"_{a.tag}" if a.tag else ""

    man = json.load(open(V2 / f"embedding_manifest_v2{suf}.json"))
    mech = json.load(open(ROOT / "data/annotations/mechanism_classes_v2.json"))
    pf = read_fasta(ROOT / "data/sequences/toxins_positive_v2.fasta")
    nf = read_fasta(ROOT / "data/sequences/benign_negatives_v2.fasta")
    pos_acc = [r["acc"] for r in man["positive_rows"]]
    neg_acc = [r["acc"] for r in man["negative_rows"]]
    nP, nN = len(pos_acc), len(neg_acc)

    if CACHE.exists():
        M = np.load(CACHE)["M"]
        print(f"loaded cached alignment matrix {M.shape} from {CACHE}")
        assert M.shape[0] == nP + nN, "cached matrix does not match the current panel"
    else:
        print(f"aligning {nP + nN} sequences pairwise, BLOSUM62 local, gap -11/-1")
        M = score_matrix([pf[x] for x in pos_acc] + [nf[x] for x in neg_acc])
        np.savez_compressed(CACHE, M=M)
        print(f"cached to {CACHE}")

    cls_of = {p["fasta_id"]: p["mechanism_class"] for p in mech["proteins"]}
    pos_cls = np.array([cls_of.get(x, "UNMAPPED") for x in pos_acc])
    classes = sorted(
        {c for c in pos_cls if c in set(mech["holdout_eligible_classes"])}
        | {"virulence_associated_non_toxin"}
    )
    Pi = np.arange(nP)
    Ni = np.arange(nP, nP + nN)

    # ---- 1. homology control, alignment version ----------------------------
    print("\n1. HOMOLOGY CONTROL: best alignment from a held-out member to a")
    print("   training positive OUTSIDE its class (normalized SW similarity)")
    print(f"\n   {'class':<32}{'max':>8}{'median':>9}{'>0.30':>8}")
    print("   " + "-" * 55)
    homol, out_cls = [], {}
    for C in classes:
        hi = np.where(pos_cls == C)[0]
        oth = np.setdiff1d(Pi, hi)
        v = M[np.ix_(hi, oth)].max(axis=1)
        homol.extend(v.tolist())
        out_cls[C] = {"align_max": float(v.max()), "align_median": float(np.median(v)),
                      "n_above_0.30": int((v > 0.30).sum()), "n": int(len(hi))}
        print(f"   {C:<32}{v.max():>8.3f}{np.median(v):>9.3f}{int((v > 0.30).sum()):>8}")
    homol = np.array(homol)
    pn = M[np.ix_(Pi, Ni)].max(axis=1)
    print(f"\n   all held-out members: max {homol.max():.3f}, median {np.median(homol):.3f}, "
          f"{(homol > 0.30).sum()} of {len(homol)} above 0.30")
    print(f"   positive vs any negative: max {pn.max():.3f}, median {np.median(pn):.3f}")

    # ---- 2. alignment as the classifier ------------------------------------
    h = int(nN * NEG_HOLDOUT_FRAC)
    align_rec = {C: [] for C in classes}
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        p = rng.permutation(nN)
        nte, ntr = Ni[p[:h]], Ni[p[h:]]
        for C in classes:
            hi = np.where(pos_cls == C)[0]
            tri = np.setdiff1d(Pi, hi)

            def marg(rows):
                return M[np.ix_(rows, tri)].max(axis=1) - M[np.ix_(rows, ntr)].max(axis=1)

            s_ho, s_ca = marg(hi), marg(nte)
            align_rec[C].append(float((s_ho >= np.quantile(s_ca, SPEC)).mean()))

    lomo = json.load(open(V2 / f"lomo_results{suf}.json"))["leave_one_mechanism_out"]
    try:
        pvs = json.load(open(V2 / f"probe_vs_similarity{suf}.json"))["classes"]
    except FileNotFoundError:
        pvs = {}
    print(f"\n2. OPERATIONAL BASELINE: flagged@95, {a.seeds} seeds, identical protocol")
    print(f"\n   {'class':<32}{'n':>3}{'probe':>8}{'embedding':>11}{'alignment':>11}"
          f"{'probe-align':>13}")
    print("   " + "-" * 78)
    for C in classes:
        al = np.mean(align_rec[C])
        pr = lomo[C]["flagged_95_mean"]
        em = pvs.get(C, {}).get("similarity_only")
        emb = f"{em:>10.0%} " if em is not None else f"{'n/a':>11}"
        print(f"   {C:<32}{out_cls[C]['n']:>3}{pr:>7.0%} {emb}{al:>10.0%} {pr - al:>+12.0%}")
        out_cls[C].update({"alignment_recovery": al, "probe_recovery": pr,
                           "embedding_recovery": em})
    d = np.array([lomo[C]["flagged_95_mean"] - np.mean(align_rec[C]) for C in classes])
    print(f"\n   probe minus alignment: mean {d.mean():+.1%}, "
          f"range {d.min():+.0%} to {d.max():+.0%}")

    out = {"model": man["model"], "tag": a.tag or None, "seeds": a.seeds,
           "aligner": "Bio.Align.PairwiseAligner local BLOSUM62 gap -11/-1, "
                      "normalized by self-score",
           "homology_all_max": float(homol.max()),
           "homology_all_median": float(np.median(homol)),
           "homology_n_above_0.30": int((homol > 0.30).sum()),
           "positive_vs_negative_max": float(pn.max()),
           "probe_minus_alignment_mean": float(d.mean()),
           "classes": out_cls}
    p_out = V2 / f"alignment_baseline{suf}.json"
    json.dump(out, open(p_out, "w"), indent=2)
    print(f"\nwrote {p_out}")


if __name__ == "__main__":
    main()
