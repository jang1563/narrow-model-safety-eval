#!/usr/bin/env python3
"""
03k_profile_hmm_baseline.py - HMMER as the operational baseline, alongside 03i.

Why this exists
---------------
docs/MECHANISM_GENERALIZATION.md §7 opens by saying deployed screens are
"alignment- OR PROFILE-based", and 03i only supplied the alignment half:
Smith-Waterman, BLOSUM62, gap -11/-1. The profile half was never run, so
"did you beat HMMER" had no answer in the repository. This supplies it.

Design: a strict swap-in for 03i
--------------------------------
03i builds one 234x234 normalized-similarity matrix M and then does everything
downstream from M -- the leave-one-mechanism-out loop, the margin
(best score to a training positive minus best score to a training negative),
the threshold at the 95th percentile of held-out-negative margins, 30 seeds.
This script rebuilds ONLY M, from HMMER bitscores, and reuses that protocol
unchanged. Protocol, seeds, holdout fraction and operating point are identical,
so the alignment and profile numbers are directly comparable.

Bitscores are not normalized to [0,1] the way 03i normalizes SW by self-score.
That does not matter here: the decision rule is a DIFFERENCE of two maxima on
one scale, and the threshold is calibrated from held-out negatives on that same
scale. Self-hits never enter, because a class's held-out rows and its training
columns are disjoint by construction.

⚠️ Expected result, written before running
------------------------------------------
This is predicted to land CLOSE TO 03i's alignment numbers rather than above
them, and the reason is structural. alignment_baseline.json records
homology_n_above_0.30 = 0: the panel is homology-screened, so a held-out class
has no homologue in the training set at all. A profile method gains its
sensitivity by recruiting homologues, and within a 234-sequence homology-screened
panel there is nothing to recruit. So this closes the "did you try HMMER"
question honestly, and the number it produces is not expected to be interesting
on its own.

The variant that WOULD be interesting needs an external database: build the
profile by searching the held-out query against Swiss-Prot, then score the panel
with that enriched profile. That gives the homology baseline the same access to
public sequence data that the foundation model had in pretraining (§9.3), which
is the fair comparison. It is not implemented here because it needs a ~90MB
download and a decision about whether the held-out family's own Swiss-Prot
entries are excluded. Recorded as the follow-up rather than silently skipped.

Usage:
    python src/03k_profile_hmm_baseline.py [--method phmmer|jackhmmer] [--seeds 30]
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "results" / "v2"
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


def bitscore_matrix(accs, seqs, method="phmmer"):
    """All-vs-all HMMER bitscores. --max disables the heuristic filters so weak
    hits are still reported, which is what a baseline should be given."""
    idx = {a: i for i, a in enumerate(accs)}
    n = len(accs)
    M = np.zeros((n, n), float)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fa, tbl = td / "panel.fa", td / "hits.tbl"
        with open(fa, "w") as fh:
            for a in accs:
                fh.write(f">{a}\n{seqs[a]}\n")
        cmd = [method, "--max", "-E", "1000", "--domE", "1000",
               "--tblout", str(tbl), "-o", "/dev/null", str(fa), str(fa)]
        if method == "jackhmmer":
            # --max here too: without it jackhmmer ran with default filters, came
            # back 1.3% dense against phmmer's 6.4%, and the resulting all-zero
            # margins made every class read 100%. Same filters or no comparison.
            cmd = [method, "--max", "-N", "3", "-E", "1000", "--domE", "1000",
                   "--tblout", str(tbl), "-o", "/dev/null", str(fa), str(fa)]
        print("running:", " ".join(cmd[:3]), "...", flush=True)
        subprocess.run(cmd, check=True)
        nrow = 0
        for line in open(tbl):
            if line.startswith("#"):
                continue
            f = line.split()
            if len(f) < 6:
                continue
            t, q, sc = f[0], f[2], float(f[5])
            if t in idx and q in idx:
                i, j = idx[q], idx[t]
                if sc > M[i, j]:
                    M[i, j] = sc
                nrow += 1
        print(f"  parsed {nrow} hit rows, matrix filled {100 * (M > 0).mean():.1f}%")
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="phmmer", choices=["phmmer", "jackhmmer"])
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    suf = f"_{a.tag}" if a.tag else ""

    man = json.load(open(V2 / f"embedding_manifest_v2{suf}.json"))
    mech = json.load(open(ROOT / "data/annotations/mechanism_classes_v2.json"))
    pf = read_fasta(ROOT / "data/sequences/toxins_positive_v2.fasta")
    nf = read_fasta(ROOT / "data/sequences/benign_negatives_v2.fasta")
    pos_acc = [r["acc"] for r in man["positive_rows"]]
    neg_acc = [r["acc"] for r in man["negative_rows"]]
    nP, nN = len(pos_acc), len(neg_acc)
    accs = pos_acc + neg_acc
    seqs = {**pf, **nf}
    missing = [x for x in accs if x not in seqs]
    assert not missing, f"sequences missing from FASTA: {missing[:3]}"

    cache = V2 / f"profile_hmm_scores_{a.method}.npz"
    if cache.exists():
        M = np.load(cache)["M"]
        print(f"loaded cached {a.method} matrix {M.shape}")
        assert M.shape[0] == nP + nN, "cached matrix does not match the current panel"
    else:
        print(f"scoring {nP + nN} sequences all-vs-all with {a.method}")
        M = bitscore_matrix(accs, seqs, a.method)
        np.savez_compressed(cache, M=M)
        print(f"cached to {cache}")

    cls_of = {p["fasta_id"]: p["mechanism_class"] for p in mech["proteins"]}
    pos_cls = np.array([cls_of.get(x, "UNMAPPED") for x in pos_acc])
    classes = sorted(
        {c for c in pos_cls if c in set(mech["holdout_eligible_classes"])}
        | {"virulence_associated_non_toxin"}
    )
    Pi = np.arange(nP)
    Ni = np.arange(nP, nP + nN)

    # identical protocol to 03i, only M differs
    h = int(nN * NEG_HOLDOUT_FRAC)
    rec = {C: [] for C in classes}
    tie = {}
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
            thr = np.quantile(s_ca, SPEC)
            rec[C].append(float((s_ho >= thr).mean()))
            if seed == 0:
                tie[C] = {"threshold": float(thr),
                          "heldout_margin_zero": float((s_ho == 0).mean()),
                          "calib_margin_zero": float((s_ca == 0).mean())}

    lomo = json.load(open(V2 / f"lomo_results{suf}.json"))["leave_one_mechanism_out"]
    align = json.load(open(V2 / f"alignment_baseline{suf}.json"))["classes"]

    print(f"\nflagged@95, {a.seeds} seeds, protocol identical to 03i")
    print(f"\n{'class':<32}{'n':>3}{'probe':>8}{'SW align':>10}{'profile':>9}"
          f"{'prof-SW':>9}{'probe-prof':>12}")
    print("-" * 83)
    out = {}
    for C in classes:
        pr = lomo[C]["flagged_95_mean"]
        al = align[C]["alignment_recovery"]
        hm = float(np.mean(rec[C]))
        n = int((pos_cls == C).sum())
        out[C] = {"n": n, "profile_recovery": hm, "alignment_recovery": al,
                  "probe_recovery": pr}
        print(f"{C:<32}{n:>3}{pr:>7.0%}{al:>10.0%}{hm:>9.0%}{hm - al:>+9.0%}"
              f"{pr - hm:>+12.0%}")
    dp = np.array([out[C]["probe_recovery"] - out[C]["profile_recovery"] for C in classes])
    da = np.array([out[C]["profile_recovery"] - out[C]["alignment_recovery"] for C in classes])
    print(f"\nprofile minus SW alignment: mean {da.mean():+.1%}, "
          f"range {da.min():+.0%} to {da.max():+.0%}")
    print(f"probe minus profile:        mean {dp.mean():+.1%}, "
          f"range {dp.min():+.0%} to {dp.max():+.0%}")
    beat = [C for C in classes if out[C]["profile_recovery"] > out[C]["probe_recovery"]]
    print(f"classes where the profile baseline BEATS the probe: {beat or 'none'}")

    # ---- power guard -------------------------------------------------------
    # A sparse bitscore matrix makes most margins exactly 0. The calibration
    # margins then sit at 0 too, the threshold becomes 0, and `margin >= 0`
    # passes nearly everything, so every class reads 100%. That is absence of
    # power, not sensitivity, and this project has already been burned once by a
    # panel that returned 100% on every arm (see
    # docs/EXTERNAL_VALIDATION_PREREGISTRATION.md). Refuse to report it.
    # The trigger is threshold == 0 on its own. A 95th-percentile calibration
    # margin of exactly 0 already means the operating point carries no
    # information, whatever the tie fraction is. An earlier version of this guard
    # also required calib_margin_zero > 0.50 and would have MISSED the real case
    # that motivated it: jackhmmer without --max measured 0.49.
    degenerate = sorted(C for C in classes if tie[C]["threshold"] == 0.0)
    if degenerate:
        print("\n\U0001F534 DEGENERATE, NOT A RESULT. Threshold is 0 and over half the "
              "calibration\n   margins are 0 for: " + ", ".join(degenerate))
        print("   Every such class reads high because ties at zero pass `>= 0`, not "
              "because\n   the baseline is sensitive. Matrix density "
              f"{100 * (M > 0).mean():.1f}%. Do not quote these numbers.")

    res = {"method": a.method, "seeds": a.seeds, "model": man["model"],
           "matrix_density": float((M > 0).mean()),
           "tie_diagnostics_seed0": tie,
           "degenerate_classes": degenerate,
           "usable": not degenerate,
           "scorer": f"HMMER 3.4 {a.method}, --max, E<=1000, full-sequence bitscore",
           "protocol": "identical to src/03i_alignment_baseline.py; only the score matrix differs",
           "profile_minus_alignment_mean": float(da.mean()),
           "probe_minus_profile_mean": float(dp.mean()),
           "classes_profile_beats_probe": beat,
           "classes": out}
    p_out = V2 / f"profile_hmm_baseline_{a.method}{suf}.json"
    json.dump(res, open(p_out, "w"), indent=2)
    print(f"\nwrote {p_out}")


if __name__ == "__main__":
    main()
