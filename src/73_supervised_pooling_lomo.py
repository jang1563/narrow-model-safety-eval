#!/usr/bin/env python3
"""
73_supervised_pooling_lomo.py - the supervised residue ranking src/70 declared and refused to fake.

What this is
------------
§ 9.1.2 found that one **label-free** coherent reduction (`win_best25`) moves beta-lactamase past
alignment while costing as much elsewhere, and that twelve of fourteen label-free reductions are worse
than mean pooling. The reduction it could not test is the one a CNN would actually learn: **residues
ranked by a supervised direction**. `src/70` declares it and raises rather than doing it as a flag,
because a supervised direction has to be fitted **inside each fold** or it leaks the held-out class.

Why this file exists instead of another tag
-------------------------------------------
Every other pooling arm is a fixed feature matrix, so `src/03b` reads it unchanged. A fold-internal
direction cannot be: the features differ per (held-out class, seed). So the fold loop is reproduced
here — and, because reproducing a protocol is how numbers drift, it is **gated**: with the ranking
ignored (`k = 0`, all residues, which is exactly the mean) this must reproduce
`lomo_results_esm2_650M_mean_res.json` to the decimal. If it does not, nothing else printed is
comparable to anything published.

What is fitted where, and what never sees what
----------------------------------------------
For each held-out class C and seed s, following `src/03b` exactly — `default_rng(s)`, 40% negative
holdout, `StandardScaler` then `LogisticRegression(C=1.0, max_iter=5000)`:

    1. direction  w  fitted on MEAN-pooled  P[not C] + N[train]      <- never sees C, never sees N[test]
    2. residue score  r_i = (w / sigma) . h_i                        <- scaler undone, offsets drop out
    3. every protein re-pooled as the mean of its top-k residues by r
    4. a second model fitted on the re-pooled  P[not C] + N[train]
    5. C scored, threshold from the re-pooled N[test]

⚠️ So the calibration negatives never influence the pooling, which is the leak an
artifact-per-class shortcut would have carried. This does not repair § 2.6's missing test partition;
it only avoids adding a second one.

Usage:
    python src/73_supervised_pooling_lomo.py --selftest
    python src/73_supervised_pooling_lomo.py --panel v2
    python src/73_supervised_pooling_lomo.py --panel v2 --seeds 30
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
NEG_HOLDOUT_FRAC = 0.40          # src/03b's
TOPK = (0, 1, 5, 10, 25, 50, 100)   # 0 means every residue, i.e. the mean, i.e. the gate


def clf():
    """src/03b's pipeline, unchanged."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))


def direction(model):
    """The residue-ranking direction in RAW embedding space.

    The pipeline standardises before the logistic fit, so its decision function is
    w . (x - mu) / sigma + b. Ranking residues by that is ranking by (w / sigma) . h up to a constant,
    and the constant cannot change an ordering.
    """
    sc, lr = model.named_steps["standardscaler"], model.named_steps["logisticregression"]
    return lr.coef_[0] / sc.scale_


def pool_topk(stack, rows, d, k):
    """[n_proteins, dim]: each protein reduced to the mean of its k highest-scoring residues."""
    out = np.empty((len(rows), stack.shape[1]), np.float32)
    for i, r in enumerate(rows):
        X = np.asarray(stack[r["offset"]:r["offset"] + r["n_res"]], dtype=np.float32)
        if k <= 0 or k >= X.shape[0]:
            out[i] = X.mean(0)
        else:
            sc = X @ d
            out[i] = X[np.argpartition(-sc, k - 1)[:k]].mean(0)
    return out


def run(stack, prows, nrows, pos_cls, targets, ks, seeds, mean_pos, mean_neg):
    """{k: {class: {...}}} following src/03b's fold logic, with the direction refitted per fold."""
    res = {k: {} for k in ks}
    for C in targets:
        hi = np.where(pos_cls == C)[0]
        if len(hi) == 0:
            continue
        tri = np.setdiff1d(np.arange(len(prows)), hi)
        per = {k: {"flagged_95": [], "flagged_99": [], "fpr95": []} for k in ks}
        for seed in seeds:
            rng = np.random.default_rng(seed)
            nperm = rng.permutation(len(nrows))
            ncut = int(len(nrows) * NEG_HOLDOUT_FRAC)
            nte, ntr = nperm[:ncut], nperm[ncut:]
            # step 1: the direction, from MEAN-pooled training data only
            m0 = clf().fit(np.vstack([mean_pos[tri], mean_neg[ntr]]),
                           np.r_[np.ones(len(tri)), np.zeros(len(ntr))])
            d = direction(m0)
            for k in ks:
                P = pool_topk(stack, prows, d, k)
                N = pool_topk(stack, nrows, d, k)
                m = clf().fit(np.vstack([P[tri], N[ntr]]),
                              np.r_[np.ones(len(tri)), np.zeros(len(ntr))])
                s_ho = m.predict_proba(P[hi])[:, 1]
                s_nte = m.predict_proba(N[nte])[:, 1]
                t95, t99 = np.quantile(s_nte, 0.95), np.quantile(s_nte, 0.99)
                per[k]["flagged_95"].append(float((s_ho >= t95).mean()))
                per[k]["flagged_99"].append(float((s_ho >= t99).mean()))
                per[k]["fpr95"].append(float((s_nte >= t95).mean()))
        for k in ks:
            v = per[k]
            res[k][C] = {"n": int(len(hi)),
                         "flagged_95_mean": float(np.mean(v["flagged_95"])),
                         "flagged_95_sd": float(np.std(v["flagged_95"], ddof=1))
                         if len(seeds) > 1 else 0.0,
                         "flagged_99_mean": float(np.mean(v["flagged_99"])),
                         "fpr_heldout_95_mean": float(np.mean(v["fpr95"])),
                         "zero_seeds": int(sum(1 for x in v["flagged_95"] if x == 0))}
        print(f"  {C:<34}" + "  ".join(f"k={k}:{res[k][C]['flagged_95_mean'] * 100:5.1f}%"
                                       for k in ks), flush=True)
    return res


def selftest():
    """The direction must be the ranking the fitted model uses, and top-k must pick those residues."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 40)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    m = clf().fit(X, y)
    d = direction(m)
    # ranking by d must agree with ranking by the pipeline's own decision function
    a = np.argsort(-(X @ d))
    b = np.argsort(-m.decision_function(X))
    assert (a == b).all(), "direction() does not reproduce the pipeline's ordering"
    stack = X.copy()
    rows = [{"offset": 0, "n_res": 60}]
    assert np.allclose(pool_topk(stack, rows, d, 0)[0], X.mean(0)), "k=0 is not the mean"
    assert np.allclose(pool_topk(stack, rows, d, 1000)[0], X.mean(0)), "k>n is not the mean"
    top5 = pool_topk(stack, rows, d, 5)[0]
    assert np.allclose(top5, X[np.argsort(-(X @ d))[:5]].mean(0)), "top-k picks the wrong residues"
    print("SELFTEST PASS: direction reproduces the pipeline ordering, k=0 and k>n are the mean, "
          "top-k selects by that ordering")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="v2", choices=["v2", "v3"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    out = ROOT / "results" / a.panel
    meta = json.loads((out / f"residue_stack_index_{a.panel}.json").read_text())
    stack = np.load(out / f"residue_stack_{a.panel}.npy", mmap_mode="r")
    prows, nrows = meta["rows"]["positive"], meta["rows"]["negative"]
    mech = json.loads((ROOT / f"data/annotations/mechanism_classes_{a.panel}.json").read_text())
    cls = {e["fasta_id"]: e["mechanism_class"] for e in mech["proteins"]}
    pos_cls = np.array([cls[r["id"]] for r in prows])
    # the classes src/03b reports: those it deems holdout-eligible, as its own artifact recorded them
    # src/03b writes lomo_results<suffix>.json inside the panel directory, so the name does not
    # carry the panel; v2 and v3 differ by directory only.
    ctl_name = "lomo_results_esm2_650M_mean_res.json"
    ctl = json.loads((out / ctl_name).read_text())["leave_one_mechanism_out"]
    targets = sorted(ctl)
    seeds = list(range(a.seeds))
    print(f"panel {a.panel}: {len(prows)} positives, {len(nrows)} negatives, "
          f"{len(targets)} classes, {a.seeds} seeds, k in {TOPK}\n")

    # mean-pooled features for step 1, straight off the stack
    mean_pos = pool_topk(stack, prows, np.zeros(stack.shape[1], np.float32), 0)
    mean_neg = pool_topk(stack, nrows, np.zeros(stack.shape[1], np.float32), 0)

    t0 = time.time()
    res = run(stack, prows, nrows, pos_cls, targets, TOPK, seeds, mean_pos, mean_neg)
    print(f"\n{time.time() - t0:.0f}s")

    # ---- the gate: k = 0 must reproduce the published mean_res run ---------------------------
    gate, worst = {}, 0.0
    for C in targets:
        pub = ctl[C]["flagged_95_mean"]
        got = res[0][C]["flagged_95_mean"]
        gate[C] = {"published": pub, "k0": got, "delta": round(got - pub, 6)}
        worst = max(worst, abs(got - pub))
    print(f"GATE, k=0 against {ctl_name}: worst per-class delta {worst:.2e}")
    if a.seeds == 5 and worst > 1e-9:
        for C, v in gate.items():
            if abs(v["delta"]) > 1e-9:
                print(f"   {C}: published {v['published']:.4f} vs k=0 {v['k0']:.4f}")
        raise SystemExit("GATE FAILED: the reproduced fold loop does not match src/03b at k=0, so "
                         "nothing measured with it is comparable to a published number.")

    dest = out / f"supervised_pooling_{a.panel}{'' if a.seeds == 5 else f'_{a.seeds}seeds'}.json"
    dest.write_text(json.dumps({
        "built": time.strftime("%Y-%m-%d %H:%M:%S"), "panel": a.panel, "seeds": a.seeds,
        "topk": list(TOPK), "gate_worst_delta": worst, "gate": gate,
        "leakage": ("the direction is fitted on mean-pooled positives excluding the held-out class "
                    "plus the TRAINING negatives only, so it sees neither the held-out class nor "
                    "the calibration negatives"),
        "by_k": {str(k): res[k] for k in TOPK},
    }, indent=2) + "\n")
    print(f"wrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    raise SystemExit(main() or 0)
