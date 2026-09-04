#!/usr/bin/env python3
"""
05c_external_score.py - score the held-out panel once, with the frozen pipeline.

This is the confirmatory run described in `docs/EXTERNAL_VALIDATION_PREREGISTRATION.md`.
Three numbers decide it, and nothing here may be tuned after seeing them:

    low-margin holdout of 10        predicted <= 40%
    class-matched random holdout    predicted >= 60%
    gap between them                predicted >= 25 points

To make "the frozen pipeline" literal rather than a claim, the scoring function is
**imported** from `03k_margin_holdout` rather than reimplemented. The margin
definition is short enough that it is repeated here, so this script asserts that its
copy reproduces 03k's published internal numbers (22% low, 84% class-matched, 100%
high on ESM-2 650M) before it is allowed to touch the external panel. If that
assertion fails, the copy has drifted and the external result is not comparable.

Embedding also reuses the frozen settings: ESM-2 650M, mean pooling, MAX_LEN 1022,
imported from 02b.

Usage:
    python src/05c_external_score.py            # verify, embed, score
    python src/05c_external_score.py --verify-only
"""

import argparse
import collections
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SEQ = ROOT / "data" / "sequences"
V2 = ROOT / "results" / "v2"
OUT = ROOT / "results" / "external"
K = 10
SEEDS = 30
REPEATS = 15


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), ROOT / "src" / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def cross_class_margin(P, N, pos_cls):
    """Copy of 03k's margin. Verified against 03k's published numbers before use."""
    Pn = P / np.linalg.norm(P, axis=1, keepdims=True)
    Nn = N / np.linalg.norm(N, axis=1, keepdims=True)
    cross = Pn @ Pn.T
    for C in set(pos_cls):
        idx = np.where(pos_cls == C)[0]
        cross[np.ix_(idx, idx)] = -np.inf
    return cross.max(axis=1) - (Pn @ Nn.T).max(axis=1)


def three_numbers(P, N, pos_cls, run_holdout, k=K, seeds=SEEDS, repeats=REPEATS):
    margin = cross_class_margin(P, N, pos_cls)
    order = np.argsort(margin)
    low, high = order[:k], order[-k:]
    m_low = run_holdout(P, N, low, seeds)[0]
    m_high = run_holdout(P, N, high, seeds)[0]
    cnt = collections.Counter(pos_cls[low])
    matched = []
    for r in range(repeats):
        rng = np.random.default_rng(500 + r)
        H = []
        for C, n in cnt.items():
            H += list(rng.permutation(np.where(pos_cls == C)[0])[:n])
        matched.append(run_holdout(P, N, np.array(H), seeds)[0])
    return {"low": m_low, "matched": float(np.mean(matched)), "high": m_high,
            "gap": float(np.mean(matched)) - m_low,
            "low_class_counts": {str(c): int(n) for c, n in cnt.items()},
            "margin_low_mean": float(margin[low].mean()),
            "margin_high_mean": float(margin[high].mean())}


def classes_of(manifest_rows, class_map):
    return np.array([class_map.get(r["acc"], "UNMAPPED") for r in manifest_rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true")
    # Attempt 2 uses a different panel prefix. The default keeps attempt 1
    # reproducible byte for byte; the frozen gate below runs either way.
    ap.add_argument("--prefix", default="external")
    a = ap.parse_args()
    m03k = load("03k_margin_holdout.py")

    # ---- gate: reproduce the internal numbers with this copy of the margin ----
    P = np.load(V2 / "embeddings_positive_v2.npy")
    N = np.load(V2 / "embeddings_negative_v2.npy")
    man = json.load(open(V2 / "embedding_manifest_v2.json"))
    mech = json.load(open(ROOT / "data/annotations/mechanism_classes_v2.json"))
    cmap = {p["fasta_id"]: p["mechanism_class"] for p in mech["proteins"]}
    internal = three_numbers(P, N, classes_of(man["positive_rows"], cmap), m03k.run_holdout,
                             seeds=20, repeats=10)
    print("gate: reproducing the published internal result with this margin copy")
    print(f"  low {internal['low']:.0%}  class-matched {internal['matched']:.0%}  "
          f"high {internal['high']:.0%}  gap {internal['gap']:+.0%}")
    assert abs(internal["low"] - 0.22) < 0.05, "internal low-margin number did not reproduce"
    assert abs(internal["matched"] - 0.84) < 0.06, "internal class-matched number did not reproduce"
    assert internal["high"] > 0.95, "internal high-margin number did not reproduce"
    print("  ✅ matches the published 22 / 84 / 100; the copy has not drifted\n")
    if a.verify_only:
        return

    # ---- embed the external panel with the frozen settings -------------------
    OUT.mkdir(parents=True, exist_ok=True)
    pe = OUT / f"embeddings_positive_{a.prefix}.npy"
    ne = OUT / f"embeddings_negative_{a.prefix}.npy"
    if pe.exists() and ne.exists():
        Pe, Ne = np.load(pe), np.load(ne)
        print(f"reusing cached external embeddings {Pe.shape} / {Ne.shape}")
    else:
        import torch
        from transformers import AutoModel, AutoTokenizer
        m02b = load("02b_esm2_embed_v2.py")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "facebook/esm2_t33_650M_UR50D"
        print(f"embedding the external panel: {model_name} on {dev}, "
              f"MAX_LEN={m02b.MAX_LEN}, mean pooling")
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModel.from_pretrained(model_name).to(dev).eval()
        pos = m02b.read_fasta(SEQ / f"{a.prefix}_positives.fasta")
        neg = m02b.read_fasta(SEQ / f"{a.prefix}_negatives.fasta")
        Pe = m02b.embed(pos, mdl, tok, dev, 2)
        Ne = m02b.embed(neg, mdl, tok, dev, 2)
        np.save(pe, Pe)
        np.save(ne, Ne)

    m02b = load("02b_esm2_embed_v2.py")
    pos_ids = [r[0] for r in m02b.read_fasta(SEQ / f"{a.prefix}_positives.fasta")]
    cls_file = ("external_mechanism_classes.json" if a.prefix == "external"
                else f"{a.prefix}_mechanism_classes.json")
    ext_cls_json = json.load(open(ROOT / "data/annotations" / cls_file))
    ext_map = {p["fasta_id"]: p["mechanism_class"] for p in ext_cls_json["proteins"]}
    ext_cls = np.array([ext_map.get(i, "UNMAPPED") for i in pos_ids])
    assert len(ext_cls) == Pe.shape[0], "external class vector does not match embedding rows"
    print(f"external panel: {Pe.shape[0]} positives / {Ne.shape[0]} negatives, "
          f"classes {dict(collections.Counter(ext_cls))}\n")

    res = three_numbers(Pe, Ne, ext_cls, m03k.run_holdout)
    print("CONFIRMATORY RESULT, external panel, ESM-2 650M")
    print(f"  low-margin holdout of {K}      {res['low']:.0%}   (predicted <= 40%)   "
          f"{'PASS' if res['low'] <= 0.40 else 'FAIL'}")
    print(f"  class-matched random          {res['matched']:.0%}   (predicted >= 60%)   "
          f"{'PASS' if res['matched'] >= 0.60 else 'FAIL'}")
    print(f"  gap                           {res['gap']:+.0%}   (predicted >= 25 pts) "
          f"{'PASS' if res['gap'] >= 0.25 else 'FAIL'}")
    print(f"  high-margin holdout           {res['high']:.0%}   (not preregistered)")
    print(f"\n  low-margin class mix: {res['low_class_counts']}")

    verdict = ("SUPPORTED" if (res["low"] <= 0.40 and res["matched"] >= 0.60
                               and res["gap"] >= 0.25)
               else "INCONCLUSIVE" if res["matched"] < 0.60 else "NOT SUPPORTED")
    print(f"\n  preregistered verdict: {verdict}")
    json.dump({"internal_gate": internal, "external": res, "verdict": verdict,
               "k": K, "seeds": SEEDS, "repeats": REPEATS,
               "model": "facebook/esm2_t33_650M_UR50D"},
              open(OUT / f"validation_{a.prefix}.json", "w"), indent=2)
    print(f"\nwrote {OUT / f'validation_{a.prefix}.json'}")


if __name__ == "__main__":
    main()
