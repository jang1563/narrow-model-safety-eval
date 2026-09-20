#!/usr/bin/env python3
"""
43_what_predicts_the_response.py - §10.9.1 leaves one question open: WHY do the two failing classes
                                   respond to a larger benign set in opposite directions?

The open question, stated exactly
---------------------------------
At a false-positive rate held at 5.1% on the panel's own 118 matched negatives, adding pool proteins
to the training set moves three classes three ways:

    phage_peptidoglycan_hydrolase   12.2% -> 35.7%   +23.5, monotone across four doses
    beta_lactamase                  21.2% ->  7.1%   -14.0, monotone down
    rip_rrna_glycosidase            94.8% -> 56.7%   -38.1

§10.9 already refutes the obvious explanation by sequence: `42`'s census finds **zero** pool proteins
above §2's 0.30 admission threshold against either failing class, maxima 0.115 and 0.105. And margin,
which is what §10.4 to §10.7 use to locate the failures, cannot predict the sign: it puts both failing
classes at the bottom together, -0.0082 and -0.0055, and says nothing about which of them a larger
benign set will help.

So the sign is measured, real and unexplained. This runs the boundary arm on **all twelve** classes and
asks what, if anything, predicts it.

The predictors, and why each is in
---------------------------------
    margin              §10.4's predictor. Preregistered to FAIL here, because it locates the failures
                        together. Included so the failure is on the record rather than assumed.
    nn_pool             how crowded the class is by the pool: mean over members of the highest cosine
                        to any pool protein. The mechanistic candidate. If the pool crowds a class, its
                        members should be pushed to the benign side as the pool enters training.
    nn_pool - nn_neg    the same thing relative to how crowded the class already was by the panel's own
                        negatives, which is the quantity §10.4's margin is built from.
    n                   class size. §10.4 found it runs the wrong way against recovery at -0.564, so it
                        is in as the known-confounded predictor.
    baseline            recovery at K=0.

🔴 baseline is the ceiling confound and it has to be read first. A class at 94.8% can only fall and a
class at 12.2% has room, so any predictor correlated with baseline will look like it predicts the
response when it is only predicting headroom. The contrast that matters survives this: beta-lactamase
starts at 21.2% with plenty of room and still falls 14 points, while the phage class starts at 12.2%
and rises 23. Both have headroom and they go opposite ways, so headroom cannot be the whole story. The
reported quantity is therefore each predictor's correlation AND its partial correlation with baseline
removed.

PREREGISTERED, written before the run
-------------------------------------
    R1  nn_pool, or nn_pool - nn_neg, correlates with the response with a permutation p < 0.05 and
        survives partialling out baseline. Then the per-class sign has a measurable cause.
        ⚠️ The DIRECTION is not preregistered, because both are mechanistically sensible and they say
        opposite things. A NEGATIVE rho means a larger benign set hurts the classes it crowds, which
        is the crowding story. A POSITIVE rho means it helps them most, which would mean the pool is
        doing something other than pushing nearby positives to the benign side. The script reports
        the sign rather than naming one, after a three-seed smoke run came back at +0.515 against the
        expectation that prompted the experiment.
    R2  Nothing survives. Then the per-class sign is real and unexplained at this panel's resolution,
        which bounds §10.9.1's finding to a description rather than a mechanism, and says what a
        larger panel would be for.
    R3  Reported either way: margin's correlation with the response, which is preregistered near zero.

⚠️ n = 12 classes. This is the same resolution every margin claim in §10.4 to §10.6 runs at, and it is
low. A null here is weak evidence of absence, and the write-up must say so. `03p`'s entry in
DATA_CORRECTIONS is the precedent: a pooled correlation over 72 members looked significant and was
pseudoreplication of nine classes.

Usage:
    PROJECT_DIR=... PYTHON_BIN=... sbatch slurm/response_predictors.sh
    python src/43_what_predicts_the_response.py --arm esm2_650M
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
PERMS = 20000


def spearman(x, y):
    def rank(v):
        v = np.asarray(v, float)
        r = np.empty(len(v), float)
        r[v.argsort()] = np.arange(1, len(v) + 1)
        for u in np.unique(v):
            m = v == u
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def perm_p(x, y, rng, perms=PERMS):
    """Two-sided permutation p on the class labels, the same null 28, 29 and 30 use."""
    obs = abs(spearman(x, y))
    x = np.asarray(x, float)
    hits = sum(1 for _ in range(perms) if abs(spearman(rng.permutation(x), y)) >= obs)
    return (hits + 1) / (perms + 1)


def partial_out(v, ctrl):
    """Residual of v after a least-squares fit on ctrl, so a correlation can be reported with the
    ceiling confound removed rather than only flagged."""
    v, ctrl = np.asarray(v, float), np.asarray(ctrl, float)
    A = np.c_[np.ones(len(ctrl)), ctrl]
    return v - A @ np.linalg.lstsq(A, v, rcond=None)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="esm2_650M")
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--keep-pool-homologs", action="store_true",
                    help="keep pool proteins that 42 found above the panel's 0.30 admission "
                         "threshold; the default drops them")
    a = ap.parse_args()
    tag = a.arm
    suf = "" if tag == "esm2_650M" else f"_{tag}"

    man = json.load(open(V3 / f"embedding_manifest_v3{suf}.json"))
    mech = json.load(open(ROOT / "data/annotations/mechanism_classes_v3.json"))
    lomo = json.load(open(V3 / f"lomo_results{suf}.json"))["leave_one_mechanism_out"]
    P = np.load(V3 / f"embeddings_positive_v3{suf}.npy")
    N = np.load(V3 / f"embeddings_negative_v3{suf}.npy")
    POOL = np.load(V3 / f"embeddings_pool_large_{tag}.npy")
    # 🔴 Drop the pool proteins 42's census found above §2's 0.30 admission threshold. On v3 that is
    # one: Q8X739 PHOQ_ECO57 at 0.871 against the panel's D0ZV89 PHOQ_SALT1, which is a POSITIVE in
    # virulence_associated_non_toxin. Leaving it in would be actively dangerous for this experiment
    # rather than merely untidy. That class's pool proximity is extreme by construction, and letting a
    # 0.871 homolog of one of its own members enter training as a negative would push that member to
    # the benign side at high K. The result would be a strongly negative response in the
    # highest-proximity class, which is exactly the correlation R1 predicts, manufactured.
    dropped_pool = []
    hom_path = V3 / "pool_homology_against_panel.json"
    if not a.keep_pool_homologs:
        if not hom_path.exists():
            raise SystemExit(f"{hom_path} missing; run src/42 first or pass --keep-pool-homologs")
        hom = json.load(open(hom_path))
        bad = {e["pool_acc"] for e in hom["above_threshold"]}
        pool_man = json.load(open(V3 / f"embedding_manifest_pool_large_{tag}.json"))
        keep = [i for i, r in enumerate(pool_man["rows"]) if r.split("|")[1] not in bad]
        dropped_pool = [r for r in pool_man["rows"] if r.split("|")[1] in bad]
        if len(pool_man["rows"]) != len(POOL):
            raise SystemExit("pool manifest and embedding row counts disagree")
        POOL = POOL[np.array(keep)]
        print(f"dropped {len(dropped_pool)} pool protein(s) above the 0.30 admission threshold: "
              f"{[r.split('|')[1] for r in dropped_pool]}")
    else:
        print("KEEPING pool homologs, so any pool-proximity correlation is confounded by them")
    cls = {e["fasta_id"]: e["mechanism_class"] for e in mech["proteins"]}
    pcls = np.array([cls[r["acc"]] for r in man["positive_rows"]])
    # 🔴 The class set is the one `28` and `30` use: a class is in the analysis only if LOMO produced
    # a recovery figure for it. The annotation file carries SIXTEEN classes on v3, four of which LOMO
    # never holds out, and one of those four is `other_toxin_mechanism`, a catch-all that is not a
    # mechanism at all. A first version of this script took every class in the annotation and put that
    # catch-all into a mechanism-class correlation. Matching `28` and `30` keeps every §10.4 to §10.6
    # claim and this one at the same twelve.
    # str() because pcls is a numpy array, so its elements are np.str_ and print as
    # np.str_('...') in every list repr. The JSON is unaffected, np.str_ being a str subclass.
    classes = [str(c) for c in sorted(set(pcls)) if c in lomo]
    dropped = sorted(str(c) for c in set(pcls) - set(classes))
    print(f"classes in LOMO: {len(classes)}; dropped as not held out: {dropped}")
    ks = [k for k in K_GRID if k < len(POOL)] + [len(POOL)]
    print(f"arm {tag}: {len(P)} positives in {len(classes)} classes, {len(N)} panel negatives, "
          f"pool {len(POOL)}")
    print(f"K grid {ks}, {a.seeds} seeds, boundary_only arm only "
          f"(calibration frozen at the panel's held-out negatives)\n")

    # ---- the response, boundary_only for every class -------------------------------
    resp = {}
    for c in classes:
        hi = np.where(pcls == c)[0]
        tri = np.setdiff1d(np.arange(len(P)), hi)
        curve = {}
        for k in ks:
            vals = []
            for seed in range(a.seeds):
                rng = np.random.default_rng(seed)
                perm = rng.permutation(len(N))
                cut = int(len(N) * FRAC)
                cal, tr = N[perm[:cut]], N[perm[cut:]]
                add = POOL[rng.permutation(len(POOL))[:k]] if k else POOL[:0]
                n_tr = np.vstack([tr, add])
                m = make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=5000, C=1.0))
                m.fit(np.vstack([P[tri], n_tr]),
                      np.r_[np.ones(len(tri)), np.zeros(len(n_tr))])
                thr = np.quantile(m.predict_proba(cal)[:, 1], SPEC)
                vals.append(float((m.predict_proba(P[hi])[:, 1] >= thr).mean()))
            curve[str(k)] = {"mean": float(np.mean(vals)),
                             "sd": float(np.std(vals, ddof=1))}
        base = curve["0"]["mean"]
        top = curve[str(ks[-1])]["mean"]
        best_k = max(ks, key=lambda x: curve[str(x)]["mean"])
        resp[c] = {"curve": curve, "baseline": base, "at_top_K": top,
                   "response_top_pts": (top - base) * 100,
                   "response_best_pts": (curve[str(best_k)]["mean"] - base) * 100,
                   "best_K": best_k, "n": int((pcls == c).sum())}
        print(f"  {c:<34}{base * 100:>6.1f}% -> {top * 100:>6.1f}%  "
              f"({(top - base) * 100:+6.1f})", flush=True)

    # ---- the predictors -------------------------------------------------------------
    def unit(X):
        return X / np.linalg.norm(X, axis=1, keepdims=True)
    Pu, Nu, POOLu = unit(P), unit(N), unit(POOL)
    pred = {}
    for c in classes:
        hi = np.where(pcls == c)[0]
        other = np.where(pcls != c)[0]
        nn_pos = float(np.mean((Pu[hi] @ Pu[other].T).max(axis=1)))
        nn_neg = float(np.mean((Pu[hi] @ Nu.T).max(axis=1)))
        nn_pool = float(np.mean((Pu[hi] @ POOLu.T).max(axis=1)))
        pred[c] = {"margin": nn_pos - nn_neg, "nn_pos": nn_pos, "nn_neg": nn_neg,
                   "nn_pool": nn_pool, "pool_minus_neg": nn_pool - nn_neg,
                   "n": resp[c]["n"], "baseline": resp[c]["baseline"]}

    y = [resp[c]["response_top_pts"] for c in classes]
    base_v = [resp[c]["baseline"] for c in classes]
    rng = np.random.default_rng(0)
    NAMES = ["margin", "nn_pool", "pool_minus_neg", "nn_neg", "n", "baseline"]
    print(f"\n{'predictor':<18}{'rho':>8}{'perm p':>10}{'rho | baseline':>16}{'perm p':>10}")
    print("-" * 62)
    stats = {}
    y_resid = partial_out(y, base_v)
    for nm in NAMES:
        x = [pred[c][nm] for c in classes]
        r, p = spearman(x, y), perm_p(x, y, rng)
        if nm == "baseline":
            rp, pp = float("nan"), float("nan")
        else:
            xr = partial_out(x, base_v)
            rp, pp = spearman(xr, y_resid), perm_p(xr, y_resid, rng)
        stats[nm] = {"rho": r, "perm_p": p, "rho_partial_baseline": rp, "perm_p_partial": pp}
        print(f"{nm:<18}{r:>+8.3f}{p:>10.4f}"
              + (f"{rp:>+16.3f}{pp:>10.4f}" if rp == rp else f"{'-':>16}{'-':>10}"))

    survivors = [nm for nm in NAMES if nm != "baseline"
                 and stats[nm]["perm_p"] < 0.05 and stats[nm]["perm_p_partial"] < 0.05]
    pool_survives = any(nm in survivors for nm in ("nn_pool", "pool_minus_neg"))
    if pool_survives:
        # 🔴 Report the SIGN, do not assume it. R1 was written expecting a NEGATIVE rho, that a larger
        # benign set hurts the classes it crowds. A three-seed smoke run came back at +0.515, the other
        # way, and a verdict that names a direction the test never checked would have published that
        # expectation as a finding.
        d = {nm: stats[nm]["rho_partial_baseline"] for nm in survivors
             if nm in ("nn_pool", "pool_minus_neg")}
        sign = ("NEGATIVE, so a larger benign set hurts the classes it crowds, which is R1 as written"
                if all(v < 0 for v in d.values()) else
                "POSITIVE, the OPPOSITE of R1 as written: a larger benign set helps the classes it "
                "crowds most, so whatever the pool does it is not simply pushing nearby positives to "
                "the benign side" if all(v > 0 for v in d.values()) else
                "MIXED in sign across the pool terms, so no direction can be stated")
        verdict = ("R1: " + ", ".join(survivors) + " predict the response and survive partialling out "
                   f"baseline. The pool-proximity sign is {sign}")
    elif survivors:
        verdict = ("R1 PARTIAL: " + ", ".join(survivors) + " survive but no pool-proximity term "
                   "does, so something predicts the sign and it is not how crowded the class is")
    else:
        verdict = (f"R2: nothing survives at n={len(classes)}, so the per-class sign is real and "
                   "unexplained at this panel's resolution. §10.9.1 stays a description rather than "
                   f"a mechanism, and a null on {len(classes)} classes is weak evidence of absence")
    print(f"\nmargin's rho against the response: {stats['margin']['rho']:+.3f} "
          f"(p {stats['margin']['perm_p']:.4f}), preregistered near zero")
    print(f"verdict: {verdict}")

    sfx = "_withhomologs" if a.keep_pool_homologs else ""
    dest = V3 / f"response_predictors_{tag}{sfx}.json"
    json.dump({"arm": tag, "K_grid": ks, "seeds": a.seeds, "perms": PERMS,
               "n_classes": len(classes), "classes": classes, "dropped_not_in_lomo": dropped,
               "pool_n_used": int(len(POOL)),
               "pool_homologs_dropped": [r.split("|")[1] for r in dropped_pool],
               "keep_pool_homologs": bool(a.keep_pool_homologs),
               "response": resp, "predictors": pred,
               "stats": stats, "survivors": survivors,
               "underpowered_note": ("n = 12 classes, the same resolution as §10.4 to §10.6. A null "
                                     "is weak evidence of absence; see 03p's entry in "
                                     "DATA_CORRECTIONS for what pooling members instead would do"),
               "verdict": verdict}, open(dest, "w"), indent=2)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
