#!/usr/bin/env python3
"""
37_negative_supplement_from_pool.py - the corrected converse of §10.7.1, and why 35's design
                                      was not that converse.

What 35 actually tested, found after reading its own result
-------------------------------------------------------------
`35`'s n=296 point is a RANDOM 296-protein subsample of the 8,259-protein pool, not the panel's
real 296 taxon-matched negatives. On esm2_35M that random baseline gives beta-lactamase **9.0%**
recovery; the real matched panel gives **1.4%** on the same arm. Composition, not size, explains
most of that gap: the panel's negatives are chosen to resemble the positives' producing organisms
(`02d`, `27`), the pool is not. So `35`'s curve does not answer "what happens if I add benign
proteins to the existing panel." It answers "what happens if I replace the panel's negatives with
a same-size-or-larger random draw," which is a different question, and its REFUTED verdict
(recovery rose from 9.0% to ~18-20% as the random draw grew) describes that different question.

The actual converse of §10.7.1
-------------------------------
§10.7.1 removed the K training negatives NEAREST the held-out class and found recovery rose
monotonically to K=80 (55% of training negatives), while a recovered class peaked at K=20 and fell
back. The converse of removing near confusors is ADDING near confusors, not adding an arbitrary
random sample most of which are far from the failing class. This script does both, so the
distinction is measured rather than asserted:

    supplement_random    the panel's real 296 negatives, PLUS a random K-sample from the pool
    supplement_nearest    the panel's real 296 negatives, PLUS the K pool proteins with the
                          HIGHEST similarity to the failing class's own members

Both keep the panel's 296 as a fixed floor, so K=0 reproduces the real panel number exactly. That
is the check that `35` could not offer, because it had no fixed floor.

PREREGISTERED, written before this run (supplement_random was already seen in 35's result and is
reported as a confound-corrected re-measurement, not a blind prediction; supplement_nearest is the
new test)
-------------------------------------------------------------------------------------------------
    P1  supplement_nearest drives the failing class's recovery DOWN, monotonically or close to it,
        the direct converse of §10.7.1's removal curve.
    P2  supplement_random does not decline monotonically, or declines much less than
        supplement_nearest for the same K, which would show the effect is about proximity, not
        volume.

    SUPPORTED     both hold. §10.7.1's local mechanism is confirmed from the other direction and
                  bulk negative growth is a different, largely orthogonal effect.
    REFUTED       supplement_nearest does not decline. The K=80 dose-response in §10.7.1 would
                  then not extrapolate to larger K, and the mechanism needs restating as bounded.

🔴 Run on esm2_35M, 2026-09-19. Two findings before the preregistered question could even be
answered.

**The K=0 sanity check "failed" and that is correct, not a bug.** K=0 gave 6.9% against the
stored LOMO value of 1.4%. `03b`'s SEEDS is `[0, 1, 2, 3, 4]`; this script's `recover()` uses 30.
Recomputed at 5 seeds, the match is exact to 15 decimal places (0.014285714285714285 both times).
This is `docs/DATA_CORRECTIONS.md`'s fifth entry, now seen on a second arm: beta-lactamase's
per-seed values on esm2_35M run 0%, 7%, 0%, 0%, 0%, 21%, 0%, 0%, 14%, 0% for the first ten
seeds, a 21-point swing from a class of 14 members where one member is 7.1 points.

**esm2_35M cannot answer this question for beta-lactamase, and the reason is the same
instability.** The K sweep (random: 6.9/8.3/10.0/10.7/14.3%, nearest: 6.9/2.1/0.5/1.7/14.8%) is
not monotone in either mode and swings by more between adjacent K values than the effect being
measured could plausibly be. The one usable check is structural rather than substantive: random
and nearest converge at K=8259 (14.3% vs 14.8%), which they must once "top-K nearest of the whole
pool" and "K random from the whole pool" are both K=all, and that convergence is what confirms
the two code paths are consistent rather than confirming anything about density.

**And phage_peptidoglycan_hydrolase, the class with more members and more headroom on the
canonical arm, is not tested here at all.** On esm2_35M it recovers 26.9%, which clears this
script's own `FAIL_AT = 0.25` and is therefore not selected as a target, even though it is the
class this experiment most wants to watch.

Both problems point at the same fix: this test needs the **canonical 650M arm**, where both
failing classes have real headroom (18.6% and 10.0%) and where the seed-stability work already
done (`03v`, `03x`) has the class's noise floor characterized. Queued in
`slurm/negative_scaling_650M.sh`.

Usage:
    python src/37_negative_supplement_from_pool.py --arm esm2_35M   # ran, inconclusive; see above
    python src/37_negative_supplement_from_pool.py --arm esm2_650M  # the version that can answer it
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
V3 = ROOT / "results" / "v3"
FRAC, SPEC, SEEDS = 0.40, 0.95, 30
K_GRID = [0, 500, 1500, 4000, 8259]
FAIL_AT = 0.25
CONTROL_CLASS = "virulence_associated_non_toxin"


def cos(A, B):
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    return A @ B.T


def recover(P, N, hi, tri, seeds=SEEDS):
    out = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(N))
        cut = int(len(N) * FRAC)
        nte, ntr = perm[:cut], perm[cut:]
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
        m.fit(np.vstack([P[tri], N[ntr]]), np.r_[np.ones(len(tri)), np.zeros(len(ntr))])
        thr = np.quantile(m.predict_proba(N[nte])[:, 1], SPEC)
        out.append(float((m.predict_proba(P[hi])[:, 1] >= thr).mean()))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="esm2_35M")
    a = ap.parse_args()
    tag = a.arm
    suf = "" if tag == "esm2_650M" else f"_{tag}"

    man = json.load(open(V3 / f"embedding_manifest_v3{suf}.json"))
    mech = json.load(open(ROOT / "data/annotations/mechanism_classes_v3.json"))
    lomo = json.load(open(V3 / f"lomo_results{suf}.json"))["leave_one_mechanism_out"]
    P = np.load(V3 / f"embeddings_positive_v3{suf}.npy")
    N_panel = np.load(V3 / f"embeddings_negative_v3{suf}.npy")
    N_pool = np.load(V3 / f"embeddings_pool_large_{tag}.npy")
    cls = {e["fasta_id"]: e["mechanism_class"] for e in mech["proteins"]}
    pcls = np.array([cls[r["acc"]] for r in man["positive_rows"]])

    failures = sorted((c for c in lomo
                       if lomo[c]["flagged_95_mean"] < FAIL_AT and c != CONTROL_CLASS),
                      key=lambda c: lomo[c]["flagged_95_mean"])
    below = {c: lomo[c]["flagged_95_mean"] for c in lomo
             if c not in failures and c != CONTROL_CLASS
             and lomo[c]["flagged_95_mean"] < 0.999}
    comparison = max(below, key=below.get)
    targets = failures + [comparison]
    print(f"arm {tag}: panel negatives {len(N_panel)}, pool {len(N_pool)}")
    print(f"failures {failures}, comparison {comparison}")
    print("K=0 must reproduce the real panel number exactly (sanity check)\n")

    simPN = cos(P, N_pool)
    ks = [k for k in K_GRID if k <= len(N_pool)]
    out = {c: {"random": {}, "nearest": {}} for c in targets}
    for c in targets:
        hi = np.where(pcls == c)[0]
        tri = np.setdiff1d(np.arange(len(P)), hi)
        nearest_rank = np.argsort(-simPN[hi].max(0))
        rng = np.random.default_rng(0)
        random_perm = rng.permutation(len(N_pool))
        for mode, order in (("random", random_perm), ("nearest", nearest_rank)):
            for k in ks:
                N = np.vstack([N_panel, N_pool[order[:k]]]) if k else N_panel
                v = recover(P, N, hi, tri)
                out[c][mode][str(k)] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1))}

    hdr = f"{'class':<32}{'mode':<10}" + "".join(f"{f'K={k}':>10}" for k in ks)
    print(hdr)
    print("-" * len(hdr))
    for c in targets:
        for mode in ("random", "nearest"):
            row = out[c][mode]
            print(f"{c:<32}{mode:<10}"
                  + "".join(f"{row[str(k)]['mean'] * 100:>9.1f}%" for k in ks))

    # sanity: K=0 must equal the stored LOMO number for both modes (same computation)
    k0_check = {c: abs(out[c]["random"]["0"]["mean"] - lomo[c]["flagged_95_mean"]) < 0.02
                for c in targets}
    print(f"\nK=0 matches stored LOMO (within seed noise): {k0_check}")

    def mono_decline(c, mode):
        v = [out[c][mode][str(k)]["mean"] for k in ks]
        return all(b <= a_ + 0.01 for a_, b in zip(v, v[1:])), v[0] - v[-1]

    print(f"\n{'class':<32}{'mode':<10}{'monotone decline':<18}{'net drop pts':>12}")
    verdicts = {}
    for c in failures:
        for mode in ("random", "nearest"):
            mono, drop = mono_decline(c, mode)
            verdicts[(c, mode)] = (mono, drop)
            print(f"{c:<32}{mode:<10}{str(mono):<18}{drop * 100:>+11.1f}")

    p1 = all(verdicts[(c, "nearest")][1] > 0 for c in failures)
    p2 = all(verdicts[(c, "nearest")][1] >= verdicts[(c, "random")][1] for c in failures)
    if p1 and p2:
        verdict = ("SUPPORTED: adding the nearest pool negatives drives every failing class down "
                   "while random addition does not decline as much, confirming §10.7.1's local "
                   "proximity mechanism from the other direction")
    elif p1:
        verdict = ("PARTIAL: nearest-supplement declines but random-supplement declines "
                   "comparably or more, so the effect is not clearly about proximity specifically")
    else:
        verdict = ("REFUTED: adding the nearest pool negatives does not drive recovery down, so "
                   "§10.7.1's K=80 curve does not extrapolate to a much larger candidate pool")
    print(f"\nverdict: {verdict}")

    dest = V3 / f"negative_supplement_curve_{tag}.json"
    json.dump({"arm": tag, "K_grid": ks, "failures": failures, "comparison": comparison,
               "k0_matches_stored_lomo": k0_check, "curves": out,
               "note_on_35": ("35's n=296 point was a random pool subsample, not the panel's "
                             "real matched negatives, so its curve answers a different question "
                             "than 'add to the existing panel'. This script fixes that by keeping "
                             "the panel's 296 as a floor at every K."),
               "verdict": verdict}, open(dest, "w"), indent=2)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
