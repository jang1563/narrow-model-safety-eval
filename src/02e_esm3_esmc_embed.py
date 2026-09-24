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

The panel is a runtime choice, added 2026-09-24
-----------------------------------------------
This script was hardcoded to v2 while `02b` had already been given `--panel`, so v3 was
embeddable with the ESM-2 ladder and not with ESM-C, ESM-3, ProtT5 or SaProt. That is why
every v3 result in this repository is an ESM-2 arm, and why the one arm that recovers
beta-lactamase, ESM-C 600M, had never been tested against v3's second unreachable class.
A capability gap in a loader silently scoped a scientific claim.

`--panel` switches the input FASTA and the output directory together, exactly as in 02b, so
a v3 embedding can never land on a v2 filename.

Usage:
    python src/02e_esm3_esmc_embed.py --model esmc_600m --tag esmc_600M --panel v2
    python src/02e_esm3_esmc_embed.py --model esmc_600m --tag esmc_600M --panel v3
    python src/02e_esm3_esmc_embed.py --model esm3_sm_open_v1 --tag esm3_1_4B --panel v3
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
RES_ROOT = ROOT / "results"          # the panel picks the subdirectory at runtime
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


def logits_config():
    """LogitsConfig, built for whichever SDK version is installed.

    🔴 2026-09-24. `return_mean_embedding` does not exist in esm 3.2.1 and the kwarg is rejected
    at construction, so this script raised TypeError on the first v3 job (Cayuga 3397705) in an
    environment where `src/14` runs fine. The v2 ESM-C arrays in this repository were built on
    2026-09-05 under torch 2.11.0+cu130, which the docstring above ties to esm 3.4.0; the current
    environment is esm 3.2.1 / torch 2.5.1+cu121.

    ⚠️ Falling back is not free and must not be done silently. Without the kwarg the mean comes
    from this script's own pooling of the per-residue stack rather than from the SDK, and if the
    two differ then a v3 arm embedded here is not comparable to the v2 arms it would be compared
    against — which is the entire purpose of embedding v3 with this model. The fallback is
    therefore RECORDED in the manifest (`mean_embedding_source`), and the v2 panel is re-embedded
    under the fallback and checked against the stored arrays before any v3 number is read.
    """
    from esm.sdk.api import LogitsConfig
    try:
        return LogitsConfig(sequence=True, return_embeddings=True,
                            return_mean_embedding=True), "sdk"
    except TypeError:
        return LogitsConfig(sequence=True, return_embeddings=True), "manual_pool"


def embed(recs, client, device):
    from esm.sdk.api import ESMProtein

    cfg, _ = logits_config()
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
    ap.add_argument(
        "--panel",
        default="v2",
        choices=["v2", "v3"],
        help="Panel version. v2 is the frozen 80/154 panel every published number rests "
        "on; v3 is 149/296. Switches the input FASTA and the output directory together, "
        "so a v3 embedding cannot land on a v2 filename.",
    )
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pv = a.panel
    OUT = RES_ROOT / pv
    OUT.mkdir(parents=True, exist_ok=True)
    pos = read_fasta(ROOT / f"data/sequences/toxins_positive_{pv}.fasta")
    neg = read_fasta(ROOT / f"data/sequences/benign_negatives_{pv}.fasta")
    print(
        f"panel={pv}  model={a.model}  device={dev}  torch={torch.__version__}  "
        f"positives={len(pos)}  negatives={len(neg)}",
        flush=True,
    )
    # 02b has carried these two since the v2 build and this script never did, so a duplicated
    # accession would have produced a row-misaligned array rather than an error.
    assert len({r[0] for r in pos}) == len(pos), "duplicate accession in positives"
    assert len({r[0] for r in neg}) == len(neg), "duplicate accession in negatives"
    assert not ({r[0] for r in pos} & {r[0] for r in neg}), "accession in BOTH label sets"

    _, mean_source = logits_config()
    print(f"mean embedding source: {mean_source}", flush=True)
    client = load(a.model, dev)
    print("--- positives ---", flush=True)
    P = embed(pos, client, dev)
    print("--- negatives ---", flush=True)
    N = embed(neg, client, dev)
    assert P.shape[0] == len(pos) and N.shape[0] == len(neg), "row count mismatch"
    assert P.shape[1] == N.shape[1], "embedding dim mismatch between label sets"

    np.save(OUT / f"embeddings_positive_{pv}_{a.tag}.npy", P)
    np.save(OUT / f"embeddings_negative_{pv}_{a.tag}.npy", N)
    man = {
        "model": a.model,
        "panel": pv,
        "device": dev,
        # "sdk" when LogitsConfig(return_mean_embedding=True) is available, "manual_pool" when
        # this script pools the per-residue stack itself. A comparison across arms is only valid
        # within one source, or after the two have been shown to agree.
        "mean_embedding_source": mean_source,
        "esm_version": __import__("esm").__version__,
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
    json.dump(man, open(OUT / f"embedding_manifest_{pv}_{a.tag}.json", "w"), indent=2)
    print(f"wrote {P.shape} and {N.shape} to {OUT}")


if __name__ == "__main__":
    main()
