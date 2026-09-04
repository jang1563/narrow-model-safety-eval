#!/usr/bin/env python3
"""
02e_esm3_esmc_embed.py - embed the v2 panel with ESM-3 and ESM-C.

Why a separate script from 02b
------------------------------
ESM-2 is a HuggingFace `AutoModel`; ESM-3 and ESM-C are served through
EvolutionaryScale's own SDK, so the loading and pooling calls differ. Everything
else is deliberately identical to 02b: same FASTA inputs, same 1022-residue
truncation, same row-aligned manifest schema. The truncation matters most. ESM-3
and ESM-C accept longer inputs than ESM-2 did, so letting them see full-length
sequences would confound the model comparison with an input-length change, and
six sequences in this panel are over the limit.

Reproducibility notes worth recording with any result from this script:
  - installing `esm` 3.4.0 downgrades torch 2.14.0 to 2.11.0, because the package
    pins it. The ESM-2 arrays were produced under 2.14.0 and are not recomputed
    here; they are cached .npy files and the analysis scripts do not use torch.
  - without Transformer Engine and xformers the SDK falls back to pure-PyTorch
    LayerNorm and attention. The library warns that this shifts the unnormalized
    residual stream, and that after the final LayerNorm the difference is a few
    ULP. Mean-pooled embeddings are taken after that LayerNorm.

Outputs use the same names as 02b with a tag, so 03b/03e/03f/03g and 04 read them
unchanged via --tag.

Usage:
    python src/02e_esm3_esmc_embed.py --model esmc_300m --tag esmc_300M
    python src/02e_esm3_esmc_embed.py --model esmc_600m --tag esmc_600M
    python src/02e_esm3_esmc_embed.py --model esm3_sm_open_v1 --tag esm3_1_4B
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "v2"
MAX_LEN = 1022  # identical to 02b, so the model axis is the only thing changing


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


def load(model_name, device):
    from esm.models.esmc import ESMC
    from esm.models.esm3 import ESM3

    if model_name.startswith("esmc"):
        return ESMC.from_pretrained(model_name).to(device).eval()
    return ESM3.from_pretrained(model_name).to(device).eval()


def embed(recs, client, device):
    from esm.sdk.api import ESMProtein, LogitsConfig

    cfg = LogitsConfig(sequence=True, return_embeddings=True, return_mean_embedding=True)
    out, n_trunc, t0 = [], 0, time.time()
    for i, (acc, seq) in enumerate(recs):
        s = seq[:MAX_LEN]
        if len(seq) > MAX_LEN:
            n_trunc += 1
        with torch.no_grad():
            tensor = client.encode(ESMProtein(sequence=s))
            res = client.logits(tensor, cfg)
        e = getattr(res, "mean_embedding", None)
        if e is None:  # fall back to pooling the per-residue stack
            emb = res.embeddings
            emb = emb[0] if emb.dim() == 3 else emb
            e = emb[1:-1].mean(0)  # drop BOS/EOS
        out.append(e.float().squeeze().cpu().numpy())
        if (i + 1) % 25 == 0 or i + 1 == len(recs):
            print(f"    {i + 1}/{len(recs)}  {time.time() - t0:.0f}s", flush=True)
    if n_trunc:
        print(f"    truncated to {MAX_LEN}: {n_trunc}")
    return np.vstack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="esmc_300m | esmc_600m | esm3_sm_open_v1")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    pos = read_fasta(ROOT / "data/sequences/toxins_positive_v2.fasta")
    neg = read_fasta(ROOT / "data/sequences/benign_negatives_v2.fasta")
    print(
        f"model={a.model}  device={dev}  torch={torch.__version__}  "
        f"positives={len(pos)}  negatives={len(neg)}",
        flush=True,
    )
    assert not ({r[0] for r in pos} & {r[0] for r in neg}), "accession in BOTH label sets"

    client = load(a.model, dev)
    print("--- positives ---", flush=True)
    P = embed(pos, client, dev)
    print("--- negatives ---", flush=True)
    N = embed(neg, client, dev)
    assert P.shape[0] == len(pos) and N.shape[0] == len(neg), "row count mismatch"
    assert P.shape[1] == N.shape[1], "embedding dim mismatch between label sets"

    np.save(OUT / f"embeddings_positive_v2_{a.tag}.npy", P)
    np.save(OUT / f"embeddings_negative_v2_{a.tag}.npy", N)
    man = {
        "model": a.model,
        "device": dev,
        "dry_run_tag": a.tag,
        "torch": torch.__version__,
        "max_len": MAX_LEN,
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "embedding_dim": int(P.shape[1]),
        "note": "row i of each array corresponds to entry i of the matching list below; "
        "truncation and pooling match 02b so the model is the only variable",
        "positive_rows": [
            {
                "row": i,
                "acc": r[0],
                "name": r[0].split("|")[2],
                "len": len(r[1]),
                "sha256": hashlib.sha256(r[1].encode()).hexdigest()[:16],
            }
            for i, r in enumerate(pos)
        ],
        "negative_rows": [
            {
                "row": i,
                "acc": r[0],
                "name": r[0].split("|")[2],
                "len": len(r[1]),
                "sha256": hashlib.sha256(r[1].encode()).hexdigest()[:16],
            }
            for i, r in enumerate(neg)
        ],
    }
    json.dump(man, open(OUT / f"embedding_manifest_v2_{a.tag}.json", "w"), indent=2)
    print(f"wrote {P.shape} and {N.shape} to {OUT}")


if __name__ == "__main__":
    main()
