#!/usr/bin/env python3
"""
02h_esm2_layer_sweep.py - embed the panel at several ESM-2 depths, not just the last.

Why
---
Every embedding in this project comes from the FINAL layer of ESM-2 650M. The
pooling axis was swept (mean, max, CLS) and the scale axis was swept (8M to 15B),
but depth never was. The protein-LM literature reports that mid-to-late layers can
beat the final layer for functional classification by a wide margin, because the
last layer specializes back toward the masked-token objective. If that holds here,
every recovery number in this repository is being read off a suboptimal layer.

One forward pass yields all hidden states, so this costs a single pass over the
234-sequence panel rather than one pass per layer.

Pooling is byte-for-byte 02b's: attention-mask weighted mean over every unmasked
token, BOS/EOS included, so a layer arm differs from the canonical arm only in
which hidden state is pooled.

Usage:
    python src/02h_esm2_layer_sweep.py [--layers 6,12,18,24,30,33]
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "results" / "v2"
MAX_LEN = 1022


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="6,12,18,24,30")
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]

    import torch
    from transformers import AutoModel, AutoTokenizer
    mid = "facebook/esm2_t33_650M_UR50D"
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModel.from_pretrained(mid).eval()

    man = json.load(open(V2 / "embedding_manifest_v2.json"))
    pf = read_fasta(ROOT / "data/sequences/toxins_positive_v2.fasta")
    nf = read_fasta(ROOT / "data/sequences/benign_negatives_v2.fasta")
    pos = [r["acc"] for r in man["positive_rows"]]
    neg = [r["acc"] for r in man["negative_rows"]]
    seqs = [pf[x] for x in pos] + [nf[x] for x in neg]
    print(f"{len(seqs)} sequences, layers {layers}")

    acc = {ly: [] for ly in layers}
    for i in range(0, len(seqs), a.batch):
        chunk = [s[:MAX_LEN] for s in seqs[i:i + a.batch]]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_LEN + 2)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        m = enc["attention_mask"].unsqueeze(-1)
        for ly in layers:
            h = hs[ly]
            acc[ly].append(((h * m).sum(1) / m.sum(1)).float().numpy())
        if (i + a.batch) % 40 < a.batch:
            print(f"  {min(i + a.batch, len(seqs))}/{len(seqs)}", flush=True)

    nP = len(pos)
    for ly in layers:
        X = np.vstack(acc[ly])
        np.save(V2 / f"embeddings_positive_v2_esm2_650M_L{ly}.npy", X[:nP])
        np.save(V2 / f"embeddings_negative_v2_esm2_650M_L{ly}.npy", X[nP:])
        print(f"  layer {ly}: {X.shape} saved")
    print("done")


if __name__ == "__main__":
    main()
