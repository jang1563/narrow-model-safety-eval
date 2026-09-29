#!/usr/bin/env python3
"""
97_size_scaling.py - are the arm differences a function of model size?

`docs/SIZE_SCALING_PREREGISTRATION.md`. ESM-2 35M has behaved differently from 650M in the same
direction in study after study -- the length U-index is 18.66 against 8.38, the localization R is 5.86
against 5.18, study K's C/A is 2.617 against 2.418 -- and every one of those studies recorded the gap
as unexplained. Two points cannot tell a size trend from an arm quirk. This adds esm2_8M and
esm2_150M, giving four arms over about 80x in parameters.

⚠️ Nothing changes but the embedding: the same pool rows, the same study-L strata, the same study-G
clean fold, the same 30 seeds, the same length bands. Cell sizes are identical at every arm.

🔒 The gate is post-hoc and declared as such: src/35's pool embedding has none of its own, so this
re-embeds the 296 panel negatives through the same path at each arm and compares against the published
array for that arm. An arm that fails is dropped.

Usage:
    python src/97_size_scaling.py --gate      # verify the new arms' pool embeddings
    python src/97_size_scaling.py
    python src/97_size_scaling.py --selftest
"""

import argparse
import importlib.util
import json
import sys
import time
from itertools import permutations
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "v3"
SEQ = ROOT / "data" / "sequences"
POOL_JSON = SEQ / "_scaled_negative_pool.json"
BUILD = ROOT / "results" / "external_class_axis_build.json"
SCREEN = ROOT / "results" / "external_class_axis_screen.json"
OUT = ROOT / "results" / "size_scaling.json"
SEEDS, SPEC, TOL = 30, 0.95, 5e-5
SHORT, MID = (0, 250), (450, 550)
RHO_STRICT, RHO_WEAK = -1.0, -0.6
# 🔒 nominal parameter counts; only the ORDER matters for a Spearman, and the order is unambiguous
ARMS = [("esm2_8M", "facebook/esm2_t6_8M_UR50D", 8e6),
        ("esm2_35M", "facebook/esm2_t12_35M_UR50D", 35e6),
        ("esm2_150M", "facebook/esm2_t30_150M_UR50D", 150e6),
        ("esm2_650M", "facebook/esm2_t33_650M_UR50D", 650e6)]


def _load(stem):
    spec = importlib.util.spec_from_file_location(f"_{stem}", ROOT / "src" / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M92 = _load("92_localization_coverage")
M84 = M92.M84
M83 = _load("83_provenance_control")


def paths(arm):
    sfx = "" if arm == "esm2_650M" else f"_{arm}"
    return (RES / f"embeddings_class_axis_positives{sfx}.npy",
            RES / f"embeddings_pool_large_{arm}.npy")


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))


def spearman(x, y):
    def rank(v):
        order = np.argsort(np.argsort(np.asarray(v, float)))
        return order.astype(float)
    a, b = rank(x), rank(y)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def perm_p(x, y, rho):
    """Exact one-tailed permutation p for a negative association, over all orderings of y."""
    ys = list(permutations(y))
    hits = sum(1 for z in ys if spearman(x, z) <= rho + 1e-12)
    return hits / len(ys)


def gate_arm(arm, model_id, batch_size=8):
    """Re-embed the 296 panel negatives through src/35's path and compare to the published array."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    sys.path.insert(0, str(ROOT / "src"))
    m02 = _load("02b_esm2_embed_v2")
    sfx = "" if arm == "esm2_650M" else f"_{arm}"
    pub_path = RES / f"embeddings_negative_v3{sfx}.npy"
    if not pub_path.exists():
        return None, f"no published panel-negative array at {arm}"
    recs = m02.read_fasta(SEQ / "benign_negatives_v3.fasta")
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if getattr(torch.backends, "mps", None)
           and torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(dev).eval()
    got = m02.embed(recs, model, tok, dev, batch_size)
    pub = np.load(pub_path)
    if got.shape != pub.shape:
        return None, f"shape {got.shape} against published {pub.shape}"
    return float(np.abs(got - pub).max()), None


def selftest():
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert spearman([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
    assert abs(spearman([1, 2, 3, 4], [4, 3, 1, 2]) - (-0.8)) < 1e-12
    # 🔒 the best one-tailed p available at n = 4 is 1/24, and the document says so in advance
    assert abs(perm_p([1, 2, 3, 4], [4, 3, 2, 1], -1.0) - 1 / 24) < 1e-12
    for arm, _, _ in ARMS:
        pos, pool = paths(arm)
        print(f"  {arm:<12} positives {'ok' if pos.exists() else 'MISSING':<8} "
              f"pool {'ok' if pool.exists() else 'MISSING'}")
    print("selftest PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    if a.gate:
        gates = {}
        for arm, mid, _ in ARMS:
            if arm in ("esm2_650M", "esm2_35M"):
                continue                       # these pool arrays predate this study and are in use
            print(f"gating {arm} ...", flush=True)
            d, err = gate_arm(arm, mid)
            gates[arm] = {"max_abs_delta": d, "error": err,
                          "passes": bool(d is not None and d <= TOL)}
            shown = f"delta {d:.3e}" if d is not None else str(err)
            print(f"  {arm}: {shown} -> {'PASS' if gates[arm]['passes'] else 'FAIL'}")
        (RES / "size_scaling_gates.json").write_text(json.dumps(gates, indent=2) + "\n")
        print(f"wrote {(RES / 'size_scaling_gates.json').relative_to(ROOT)}")
        return

    gate_file = RES / "size_scaling_gates.json"
    gates = json.loads(gate_file.read_text()) if gate_file.exists() else {}
    if not gates:
        raise SystemExit("run --gate first; the new arms' pool embeddings are ungated otherwise")

    pool = json.loads(POOL_JSON.read_text())["proteins"]
    test = json.loads(BUILD.read_text())["pool_partition"]["test_rows"]
    accs = [pool[i]["uniprot"] for i in test]
    go, kw = M92.fetch_go(accs), M84.fetch_keywords(accs)
    elig = [j for j, i in enumerate(test) if pool[i]["kingdom"] in M84.KINGDOMS]
    strat = {s: [] for s in M92.STRATA}
    for j in elig:
        g = go[accs[j]]
        strat[M92.stratum_of(g["go"], g["sig"], g["tm"], kw[accs[j]])].append(j)
    ex, ins = strat["extracellular"], strat["intracellular"]
    lengths = np.array([pool[test[j]]["length"] for j in range(len(test))], float)
    short = [j for j in ins if SHORT[0] <= lengths[j] < SHORT[1]]
    mid = [j for j in ins if MID[0] <= lengths[j] < MID[1]]
    vf = M83.vfdb_sequences()
    union = [i for i in json.loads(SCREEN.read_text())["admitted_rows"]
             if pool[i]["sequence"] not in vf]
    n_tr, n_ca = M92.fold_sizes(len(union))
    rows = np.array(test)
    print(f"cells identical at every arm: short {len(short)}, mid {len(mid)}, "
          f"extra {len(ex)}, intra {len(ins)}")

    used, dropped = [], []
    for arm, _, params in ARMS:
        pos_p, pool_p = paths(arm)
        if not (pos_p.exists() and pool_p.exists()):
            dropped.append((arm, "embedding missing"))
            continue
        g = gates.get(arm)
        if g is not None and not g["passes"]:
            dropped.append((arm, f"gate failed ({g.get('error') or g['max_abs_delta']})"))
            continue
        used.append((arm, params, pos_p, pool_p))
    for arm, why in dropped:
        print(f"  ⚠️  dropped {arm}: {why}")
    if len(used) < 3:
        raise SystemExit(f"only {len(used)} arms survive; P-1 is not computed")

    out = {}
    for arm, params, pos_p, pool_p in used:
        P, POOL = np.load(pos_p), np.load(pool_p)
        y = np.r_[np.ones(P.shape[0]), np.zeros(n_tr)]
        acc = {k: [] for k in ("short", "mid", "ex", "in")}
        t0 = time.time()
        for seed in range(a.seeds):
            r = np.random.default_rng(seed)
            perm = r.permutation(len(union))
            tr = [union[i] for i in perm[:n_tr]]
            ca = [union[i] for i in perm[n_tr:n_tr + n_ca]]
            model = clf().fit(np.vstack([P, POOL[tr]]), y)
            t = float(np.quantile(model.predict_proba(POOL[ca])[:, 1], SPEC))
            flag = model.predict_proba(POOL[rows])[:, 1] >= t
            for k, v in (("short", short), ("mid", mid), ("ex", ex), ("in", ins)):
                acc[k].append(float(flag[v].mean()))
        m = {k: float(np.mean(v)) for k, v in acc.items()}
        out[arm] = {"params": params, "dim": int(POOL.shape[1]), "rates": m,
                    "U_index": m["short"] / m["mid"] if m["mid"] > 0 else float("nan"),
                    "R": m["ex"] / m["in"] if m["in"] > 0 else float("nan"),
                    "seconds": round(time.time() - t0)}
        print(f"  {arm:<12} dim {POOL.shape[1]:>5}  U {out[arm]['U_index']:>7.2f}  "
              f"R {out[arm]['R']:>6.2f}   short {m['short'] * 100:5.2f}% mid {m['mid'] * 100:5.2f}%")

    arms_o = [a_ for a_, _, _, _ in used]
    lp = [np.log10(out[a_]["params"]) for a_ in arms_o]
    verdicts = {}
    for key in ("U_index", "R"):
        v = [out[a_][key] for a_ in arms_o]
        rho = spearman(lp, v)
        p = perm_p(lp, v, rho) if rho < 0 else None
        band = ("strictly monotone: a size trend" if rho <= RHO_STRICT
                else "consistent with a size trend, not strict" if rho <= RHO_WEAK
                else "the opposite trend" if rho >= 0.6 else "no size trend")
        verdicts[key] = {"rho": rho, "perm_p": p, "band": band, "values": v}
        print(f"\nP-1 {key}: rho = {rho:+.2f}"
              + (f", one-tailed permutation p = {p:.4f}" if p is not None else "")
              + f"  ->  {band}")

    # 🔒 the bug guard: two arms scoring identically would mean one embedding was used twice
    us = [round(out[a_]["U_index"], 2) for a_ in arms_o]
    dup = len(us) != len(set(us))
    res = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "seeds": a.seeds,
           "cells": {"short": len(short), "mid": len(mid), "extra": len(ex), "intra": len(ins)},
           "arms_used": arms_o, "arms_dropped": dict(dropped), "gates": gates,
           "per_arm": out, "P1": verdicts,
           "best_available_p_at_n4": 1 / 24, "duplicate_U_index": bool(dup)}
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    if dup:
        print("\n🔴 two arms returned the same U-index — an embedding was scored twice")
    print(f"\n⚠️  n = {len(arms_o)}: the best one-tailed p available is {1 / 24:.4f}, "
          f"so this is suggestive by construction.")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
