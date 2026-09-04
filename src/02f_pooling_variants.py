#!/usr/bin/env python3
"""
02f_pooling_variants.py - does mean pooling throw away the class it cannot catch?

Motivation
----------
§6i produced the sharpest clue in the project. Beta-lactamase is the one class
where plain Smith-Waterman alignment beats the ESM probe, 31% against 21%, and
where an embedding cosine rule collapses to 1%. Beta-lactamases are a
sequence-conserved family with a conserved active site, so a residue-level method
has real signal there. Every embedding in this project is MEAN pooled over
residues, which is exactly the operation that would average such a site away.

This embeds the panel once and writes three pooled views from the same forward
pass, so the comparison costs one run rather than three:

    mean   the incumbent, average over residues
    max    per-dimension maximum, which preserves a strong local signal that
           occupies few residues
    cls    the <cls> token, which the model is free to use as a summary

If max pooling recovers beta-lactamase, the failure was the pooling operation and
not the representation, and every recovery number in this log is understating what
the model retains. If none of the three helps, mean pooling is exonerated and the
information is genuinely absent from the residue stack the probe sees.

Outputs use the standard tagged names so 03b, 03e-03j read them with --tag.

Usage:
    python src/02f_pooling_variants.py --model facebook/esm2_t33_650M_UR50D \\
        --tag_prefix esm2_650M
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "v2"
MAX_LEN = 1022


def read_fasta(path):
    recs, acc, seq = [], None, []
    for line in open(path):
        if line.startswith(">"):
            if acc:
                recs.append((acc, "".join(seq)))
            acc, seq = line[1:].split()[0], []
        elif acc:
            seq.append(line.strip())
    if acc:
        recs.append((acc, "".join(seq)))
    return recs


def embed(recs, model, tok, dev, bs):
    """Return {pooling: array}. One forward pass, three reductions."""
    acc = {"mean": [], "max": [], "cls": []}
    t0 = time.time()
    for i in range(0, len(recs), bs):
        chunk = [r[1][:MAX_LEN] for r in recs[i : i + bs]]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_LEN + 2).to(dev)
        with torch.no_grad():
            hs = model(**enc).last_hidden_state          # [B, L, d]
        m = enc["attention_mask"].bool()
        # drop the special tokens at both ends of each real sequence
        for b in range(hs.shape[0]):
            idx = m[b].nonzero().squeeze(-1)
            body = hs[b, idx[1:-1]]                       # residues only
            acc["mean"].append(body.mean(0).float().cpu().numpy())
            acc["max"].append(body.max(0).values.float().cpu().numpy())
            acc["cls"].append(hs[b, idx[0]].float().cpu().numpy())
        if (i + bs) % 50 < bs:
            print(f"    {min(i + bs, len(recs))}/{len(recs)}  {time.time() - t0:.0f}s",
                  flush=True)
    return {k: np.vstack(v) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--tag_prefix", default="esm2_650M")
    ap.add_argument("--batch_size", type=int, default=4)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    pos = read_fasta(ROOT / "data/sequences/toxins_positive_v2.fasta")
    neg = read_fasta(ROOT / "data/sequences/benign_negatives_v2.fasta")
    print(f"model={a.model}  device={dev}  {len(pos)} pos / {len(neg)} neg", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model).to(dev).eval()
    print("--- positives ---", flush=True)
    Pp = embed(pos, model, tok, dev, a.batch_size)
    print("--- negatives ---", flush=True)
    Nn = embed(neg, model, tok, dev, a.batch_size)

    for pool in ("mean", "max", "cls"):
        tag = f"{a.tag_prefix}_{pool}"
        P, N = Pp[pool], Nn[pool]
        assert P.shape[0] == len(pos) and N.shape[0] == len(neg), "row count mismatch"
        np.save(OUT / f"embeddings_positive_v2_{tag}.npy", P)
        np.save(OUT / f"embeddings_negative_v2_{tag}.npy", N)
        man = {
            "model": a.model, "device": dev, "dry_run_tag": tag, "pooling": pool,
            "built": time.strftime("%Y-%m-%d %H:%M:%S"),
            "embedding_dim": int(P.shape[1]), "max_len": MAX_LEN,
            "note": "all three poolings come from one forward pass over the same panel",
            "positive_rows": [
                {"row": i, "acc": r[0], "name": r[0].split("|")[2], "len": len(r[1]),
                 "sha256": hashlib.sha256(r[1].encode()).hexdigest()[:16]}
                for i, r in enumerate(pos)],
            "negative_rows": [
                {"row": i, "acc": r[0], "name": r[0].split("|")[2], "len": len(r[1]),
                 "sha256": hashlib.sha256(r[1].encode()).hexdigest()[:16]}
                for i, r in enumerate(neg)],
        }
        json.dump(man, open(OUT / f"embedding_manifest_v2_{tag}.json", "w"), indent=2)
        print(f"wrote {pool}: {P.shape} / {N.shape}  tag={tag}")


if __name__ == "__main__":
    main()
