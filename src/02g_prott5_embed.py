#!/usr/bin/env python3
"""
02g_prott5_embed.py - the first model in this project outside the ESM lineage.

Why
---
ESM-2, ESM-C and ESM-3 are all EvolutionaryScale models. Every cross-model claim
so far, including "scaling redistributes fragility rather than removing it" and
"the newest architecture is the worst on beta-lactamase", is therefore a claim
about one lineage. ProtT5 is a T5 encoder-decoder trained by Rostlab on UniRef50
with a span-corruption objective, so it differs in architecture, objective and
training corpus at once. If the class-level pattern survives it, the pattern is
about the task; if it does not, everything above is lineage-specific.

Model: Rostlab/prot_t5_xl_half_uniref50-enc, the encoder-only fp16 release, 1024-d.

ProtT5 preprocessing differs from ESM and getting it wrong silently degrades the
embeddings rather than erroring: residues must be space separated, and the rare
letters U, Z, O and B must be mapped to X because they are not in the vocabulary.
The tokenizer appends </s>, which is dropped before pooling so the pooled vector
covers residues only, matching 02b and 02e.

Truncation is the same 1022 used everywhere else, so the model is the only thing
that changes.

Usage:
    python src/02g_prott5_embed.py --tag prott5_xl
"""

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "v2"
MAX_LEN = 1022
MODEL = "Rostlab/prot_t5_xl_half_uniref50-enc"


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
    out, t0 = [], time.time()
    for i in range(0, len(recs), bs):
        raw = [r[1][:MAX_LEN] for r in recs[i : i + bs]]
        prepped = [" ".join(re.sub(r"[UZOB]", "X", s)) for s in raw]
        enc = tok(prepped, add_special_tokens=True, padding="longest",
                  return_tensors="pt").to(dev)
        with torch.no_grad():
            hs = model(**enc).last_hidden_state
        for b, s in enumerate(raw):
            # exactly len(s) residue positions, then </s>
            out.append(hs[b, : len(s)].mean(0).float().cpu().numpy())
        if (i + bs) % 50 < bs:
            print(f"    {min(i + bs, len(recs))}/{len(recs)}  {time.time() - t0:.0f}s",
                  flush=True)
    return np.vstack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="prott5_xl")
    ap.add_argument("--batch_size", type=int, default=4)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    pos = read_fasta(ROOT / "data/sequences/toxins_positive_v2.fasta")
    neg = read_fasta(ROOT / "data/sequences/benign_negatives_v2.fasta")
    print(f"model={MODEL}  device={dev}  {len(pos)} pos / {len(neg)} neg", flush=True)

    tok = T5Tokenizer.from_pretrained(MODEL, do_lower_case=False, legacy=True)
    model = T5EncoderModel.from_pretrained(MODEL).to(dev).eval()
    if dev == "cuda":
        model = model.half()

    print("--- positives ---", flush=True)
    P = embed(pos, model, tok, dev, a.batch_size)
    print("--- negatives ---", flush=True)
    N = embed(neg, model, tok, dev, a.batch_size)
    assert P.shape[0] == len(pos) and N.shape[0] == len(neg), "row count mismatch"
    assert P.shape[1] == N.shape[1], "embedding dim mismatch"

    np.save(OUT / f"embeddings_positive_v2_{a.tag}.npy", P)
    np.save(OUT / f"embeddings_negative_v2_{a.tag}.npy", N)
    man = {
        "model": MODEL, "device": dev, "dry_run_tag": a.tag, "max_len": MAX_LEN,
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "embedding_dim": int(P.shape[1]),
        "note": "mean pooled over residue positions only, </s> excluded; "
                "U/Z/O/B mapped to X per the ProtT5 vocabulary",
        "positive_rows": [
            {"row": i, "acc": r[0], "name": r[0].split("|")[2], "len": len(r[1]),
             "sha256": hashlib.sha256(r[1].encode()).hexdigest()[:16]}
            for i, r in enumerate(pos)],
        "negative_rows": [
            {"row": i, "acc": r[0], "name": r[0].split("|")[2], "len": len(r[1]),
             "sha256": hashlib.sha256(r[1].encode()).hexdigest()[:16]}
            for i, r in enumerate(neg)],
    }
    json.dump(man, open(OUT / f"embedding_manifest_v2_{a.tag}.json", "w"), indent=2)
    print(f"wrote {P.shape} and {N.shape}")


if __name__ == "__main__":
    main()
