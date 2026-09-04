#!/usr/bin/env python3
"""
02i_saprot_embed.py - the first structure-aware representation in this project.

Why this model, and why now
---------------------------
Beta-lactamase is the one class that has resisted everything. Plain Smith-Waterman
alignment beats every embedding method on it (§6i, 31% against 21%), no classifier
head rescues it (§6j, best 25% of four), and it is worst on every ESM-2 size, at
0% on ESM-3. It is also a family defined by a conserved fold and active site. That
combination points at information the sequence models are discarding rather than
information that is absent.

SaProt reads a structure-aware alphabet: every position is the amino acid followed
by its Foldseek 3Di letter, so residue "M" with 3Di "d" is the token "Md". If a
structural view recovers beta-lactamase, the ceiling was the input representation.
If it does not, the class is genuinely hard for this whole family of methods, which
is a stronger negative result than any of the individual failures.

Structures come from 02h: AlphaFold DB v6, 218 of 220 panel proteins matched with
zero length mismatches. The two without structures carry "#" at every position,
which SaProt treats as unknown structure, making them sequence-only rather than
dropping them and changing the panel.

The token count is asserted against the sequence length. SaProt's vocabulary is
2-character tokens, so a tokenizer that silently falls back to per-character
splitting would halve the effective sequence and quietly produce garbage
embeddings rather than an error.

Usage:
    python src/02i_saprot_embed.py --tag saprot_650M
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
TRI = ROOT / "data" / "annotations" / "structure_3di_v2.json"
MAX_LEN = 1022
MODEL = "westlake-repl/SaProt_650M_AF2"


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


def sa_string(seq, threedi):
    """Interleave into SaProt's 2-character alphabet, truncated like every other run."""
    s, t = seq[:MAX_LEN], threedi[:MAX_LEN]
    assert len(s) == len(t), "sequence and 3Di lengths differ after truncation"
    return "".join(a + b.lower() for a, b in zip(s, t))


def embed(recs, tri, model, tok, dev, bs):
    out, t0 = [], time.time()
    for i in range(0, len(recs), bs):
        chunk = recs[i : i + bs]
        texts, lens = [], []
        for fid, seq in chunk:
            td = tri[fid]["threedi"]
            texts.append(sa_string(seq, td))
            lens.append(min(len(seq), MAX_LEN))
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_LEN + 2).to(dev)
        with torch.no_grad():
            hs = model(**enc).last_hidden_state
        m = enc["attention_mask"].bool()
        for b, n in enumerate(lens):
            idx = m[b].nonzero().squeeze(-1)
            body = hs[b, idx[1:-1]]                      # residues only
            assert body.shape[0] == n, (
                f"token count {body.shape[0]} != residue count {n}; the tokenizer is "
                "not using SaProt's 2-character vocabulary")
            out.append(body.mean(0).float().cpu().numpy())
        if (i + bs) % 50 < bs:
            print(f"    {min(i + bs, len(recs))}/{len(recs)}  {time.time() - t0:.0f}s",
                  flush=True)
    return np.vstack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="saprot_650M")
    ap.add_argument("--batch_size", type=int, default=4)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    tri = json.load(open(TRI))
    print(f"3Di annotation: {tri['stats']}")
    pos = read_fasta(ROOT / "data/sequences/toxins_positive_v2.fasta")
    neg = read_fasta(ROOT / "data/sequences/benign_negatives_v2.fasta")
    ann = tri["proteins"]
    print(f"model={MODEL}  device={dev}  {len(pos)} pos / {len(neg)} neg", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(dev).eval()

    print("--- positives ---", flush=True)
    P = embed(pos, ann, model, tok, dev, a.batch_size)
    print("--- negatives ---", flush=True)
    N = embed(neg, ann, model, tok, dev, a.batch_size)
    assert P.shape[0] == len(pos) and N.shape[0] == len(neg), "row count mismatch"

    np.save(OUT / f"embeddings_positive_v2_{a.tag}.npy", P)
    np.save(OUT / f"embeddings_negative_v2_{a.tag}.npy", N)
    n_struct = sum(1 for f, _ in pos + neg if ann[f]["status"] == "ok")
    man = {
        "model": MODEL, "device": dev, "dry_run_tag": a.tag, "max_len": MAX_LEN,
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "embedding_dim": int(P.shape[1]),
        "structures_used": n_struct, "structures_masked": len(pos) + len(neg) - n_struct,
        "note": "amino acid plus Foldseek 3Di per position, AlphaFold DB v6; "
                "proteins without a structure carry '#' and are effectively sequence-only",
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
    print(f"wrote {P.shape} and {N.shape}, {n_struct} with real structure")


if __name__ == "__main__":
    main()
