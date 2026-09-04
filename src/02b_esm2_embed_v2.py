#!/usr/bin/env python3
"""
02b_esm2_embed_v2.py - ESM-2 embeddings for the cleaned v2 panel.

Why this exists
---------------
The April 2026 arrays (results/embeddings_*.npy, 60 rows each) were never
recomputed after the May 21 panel expansion, while embedding_ids.json and
separability_results.json WERE regenerated. Row i of the old arrays therefore
does not correspond to positive_ids[i]. That silent drift is possible because
03_esm2_separability.py derives labels from array LENGTHS, so nothing fails.

This script writes a manifest in which every row carries its own accession, so
the mapping cannot drift again. It writes to NEW filenames; the April arrays
are left untouched.

Inputs : data/sequences/toxins_positive_v2.fasta   (68)
         data/sequences/benign_negatives_v2.fasta  (50)
Outputs: results/v2/embeddings_positive_v2.npy
         results/v2/embeddings_negative_v2.npy
         results/v2/embedding_manifest_v2.json     (row-aligned, with sha256)

Usage:
    python src/02b_esm2_embed_v2.py --model facebook/esm2_t33_650M_UR50D
    python src/02b_esm2_embed_v2.py --model facebook/esm2_t30_150M_UR50D --dry-run-tag smoke150M
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
    recs, acc, desc, seq = [], None, "", []
    for line in open(path):
        if line.startswith(">"):
            if acc:
                recs.append((acc, desc, "".join(seq)))
            h = line[1:].strip()
            acc = h.split()[0]
            desc = h.split(" ", 1)[1] if " " in h else ""
            seq = []
        elif acc:
            seq.append(line.strip())
    if acc:
        recs.append((acc, desc, "".join(seq)))
    return recs


def embed(recs, model, tok, device, batch_size):
    out, n_trunc = [], 0
    t0 = time.time()
    for i in range(0, len(recs), batch_size):
        batch = recs[i : i + batch_size]
        seqs = []
        for _, _, s in batch:
            if len(s) > MAX_LEN:
                n_trunc += 1
            seqs.append(s[:MAX_LEN])
        enc = tok(seqs, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN + 2)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            h = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1)
        out.append(((h * mask).sum(1) / mask.sum(1)).float().cpu().numpy())
        print(f"  [{min(i + batch_size, len(recs))}/{len(recs)}] {time.time() - t0:.0f}s", end="\r")
    print()
    if n_trunc:
        print(f"  {n_trunc} sequence(s) truncated to {MAX_LEN}")
    return np.vstack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--tag",
        default=None,
        help="Suffix outputs, e.g. esm2_150M. Use to keep several models' "
        "arrays side by side; downstream scripts take the same --tag.",
    )
    ap.add_argument(
        "--dry-run-tag",
        default=None,
        help="Deprecated alias for --tag, kept so older commands still run.",
    )
    a = ap.parse_args()
    a.dry_run_tag = a.tag or a.dry_run_tag

    dev = a.device or (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else "cpu"
    )
    tag = f"_{a.dry_run_tag}" if a.dry_run_tag else ""
    OUT.mkdir(parents=True, exist_ok=True)

    pos = read_fasta(ROOT / "data/sequences/toxins_positive_v2.fasta")
    neg = read_fasta(ROOT / "data/sequences/benign_negatives_v2.fasta")
    print(f"model={a.model}  device={dev}  positives={len(pos)}  negatives={len(neg)}")
    assert len({r[0] for r in pos}) == len(pos), "duplicate accession in positives"
    assert len({r[0] for r in neg}) == len(neg), "duplicate accession in negatives"
    assert not ({r[0] for r in pos} & {r[0] for r in neg}), "accession present in BOTH label sets"

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model).to(dev).eval()

    print("--- positives ---")
    P = embed(pos, model, tok, dev, a.batch_size)
    print("--- negatives ---")
    N = embed(neg, model, tok, dev, a.batch_size)
    assert P.shape[0] == len(pos) and N.shape[0] == len(neg), "row count mismatch"

    np.save(OUT / f"embeddings_positive_v2{tag}.npy", P)
    np.save(OUT / f"embeddings_negative_v2{tag}.npy", N)

    man = {
        "model": a.model,
        "device": dev,
        "dry_run_tag": a.dry_run_tag,
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "embedding_dim": int(P.shape[1]),
        "note": "row i of each array corresponds to entry i of the matching list below",
        "positive_rows": [
            {
                "row": i,
                "acc": r[0],
                "name": r[0].split("|")[2],
                "len": len(r[2]),
                "sha256": hashlib.sha256(r[2].encode()).hexdigest()[:16],
            }
            for i, r in enumerate(pos)
        ],
        "negative_rows": [
            {
                "row": i,
                "acc": r[0],
                "name": r[0].split("|")[2],
                "len": len(r[2]),
                "sha256": hashlib.sha256(r[2].encode()).hexdigest()[:16],
            }
            for i, r in enumerate(neg)
        ],
    }
    json.dump(man, open(OUT / f"embedding_manifest_v2{tag}.json", "w"), indent=2)
    print(f"wrote {P.shape} and {N.shape} to {OUT}")


if __name__ == "__main__":
    main()
