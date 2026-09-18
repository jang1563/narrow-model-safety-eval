#!/usr/bin/env python3
"""
15c_interplm_power_check.py - can InterPLM's SAE features separate hazard at all?

Run before any claim about individual features, because a feature-level story about
beta-lactamase is worthless if the feature space cannot tell hazard from benign in the
first place. Encodes the full 234-protein v2 panel through InterPLM's pre-trained SAE
(src/15b loads it; no `interplm` package needed) and fits the same logistic probe used
everywhere else in this project.

Result, 2026-09-18:

    layer 1   AUROC 0.753 +- 0.063   214 features alive of 10240, 127 active per protein
    layer 18  AUROC 0.910 +- 0.044   10225 alive,               4351 active per protein
    raw ESM-2 embedding (§2.3)  0.973

So layer 18's SAE features do carry hazard signal, 6 points below the raw embedding, and
the premise holds. Two things this also settled, both of which changed the plan:

  * Layer 1 is NOT usable as a reconstruction-fidelity control. Its SAE reconstructs
    almost perfectly (0.6% error, see 15b) but only 214 of 10240 features ever fire and
    AUROC is 0.753, so "same structure at a faithful layer" cannot be tested there.
  * Mean-pooling SAE features over a protein destroys the sparsity that makes an SAE
    interpretable: 146 features active per RESIDUE becomes 4351 per protein, 42% of the
    dictionary. Most of that is tiny residue: only 44 features per protein exceed the
    99th percentile of activation values, whose median is 0.0003. Per-protein means are
    fine for a classifier and misleading for "which feature means what".

Usage:
    python src/15c_interplm_power_check.py 1 18
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, EsmModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
load_interplm_weights = import_module("15b_interplm_sae").load_interplm_weights

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "results" / "v2"
MAX_LEN = 1022
ESM = "facebook/esm2_t33_650M_UR50D"
RAW_EMBEDDING_AUROC = 0.973   # docs/MECHANISM_GENERALIZATION.md §2.3, for comparison


def read_fasta(path):
    out, acc, seq = {}, None, []
    for line in open(path):
        if line.startswith(">"):
            if acc:
                out[acc] = "".join(seq)
            acc, seq = line[1:].split()[0], []
        elif acc:
            seq.append(line.strip())
    if acc:
        out[acc] = "".join(seq)
    return out




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("layers", nargs="*", type=int, default=[18])
    a = ap.parse_args()
    layers = a.layers or [18]

    tok = AutoTokenizer.from_pretrained(ESM)
    esm = EsmModel.from_pretrained(ESM).eval()

    man = json.load(open(V2 / "embedding_manifest_v2.json"))
    mech = json.load(open(ROOT / "data/annotations/mechanism_classes_v2.json"))
    cls = {e["fasta_id"]: e["mechanism_class"] for e in mech["proteins"]}
    pos_fa = read_fasta(ROOT / "data/sequences/toxins_positive_v2.fasta")
    neg_fa = read_fasta(ROOT / "data/sequences/benign_negatives_v2.fasta")

    seqs = ([(cls[r["acc"]], pos_fa[r["acc"]], 1) for r in man["positive_rows"]]
            + [("benign", neg_fa[r["acc"]], 0) for r in man["negative_rows"]])
    n_pos = sum(s[2] for s in seqs)
    print(f"{len(seqs)} sequences: {n_pos} pos / {len(seqs) - n_pos} neg", flush=True)

    saes = {L: load_interplm_weights(L) for L in layers}
    feats = {L: [] for L in layers}
    for i, (_, seq, _) in enumerate(seqs):
        with torch.no_grad():
            hs = esm(**tok(seq[:MAX_LEN], return_tensors="pt"),
                     output_hidden_states=True).hidden_states
        for L in layers:
            h = hs[L][0, 1:-1, :].float()
            W_enc, b_enc, b_pre = saes[L]
            with torch.no_grad():
                f = torch.relu((h - b_pre) @ W_enc.T + b_enc)
            feats[L].append(f.mean(0).numpy())
        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{len(seqs)}", flush=True)

    y = np.array([s[2] for s in seqs])
    out = {"raw_embedding_auroc_for_comparison": RAW_EMBEDDING_AUROC, "layers": {}}
    for L in layers:
        F = np.vstack(feats[L])
        np.save(V2 / f"interplm_features_L{L}.npy", F)
        aurocs = []
        for seed in range(5):
            for tr, te in StratifiedKFold(5, shuffle=True,
                                          random_state=seed).split(F, y):
                m = make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=5000, C=1.0)).fit(F[tr], y[tr])
                aurocs.append(roc_auc_score(y[te], m.predict_proba(F[te])[:, 1]))
        alive = int((F > 0).any(0).sum())
        per_protein = float((F > 0).sum(1).mean())
        out["layers"][L] = {"auroc": float(np.mean(aurocs)), "sd": float(np.std(aurocs)),
                            "features_alive": alive, "n_features": int(F.shape[1]),
                            "active_per_protein": per_protein, "shape": list(F.shape)}
        print(f"\nlayer {L}: hazard AUROC {np.mean(aurocs):.3f} +- {np.std(aurocs):.3f}"
              f"  ({alive} of {F.shape[1]} features alive, "
              f"{per_protein:.0f} active per protein)", flush=True)

    print(f"\nraw ESM-2 embedding, for comparison: {RAW_EMBEDDING_AUROC}")
    dest = V2 / "interplm_power_check.json"
    json.dump(out, open(dest, "w"), indent=2)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
