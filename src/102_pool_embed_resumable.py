#!/usr/bin/env python3
"""
102_pool_embed_resumable.py - embed a pool FASTA with checkpointing and a gate.

`src/35`'s embed path calls `src/02b`'s `embed`, which holds everything in memory and writes once at
the end. That is fine for the 35M arm, which takes half an hour. 🔴 The 650M arm of the independent
pool is about five hours, and this machine has already slept mid-run once: a crash there loses the
whole run with nothing to resume from.

🔒 `src/77`'s `embed` already does shard checkpoint and resume, and `src/80` gates it against the
published panel-negative arrays and passes, so it is the same pooling -- src/02b's include-specials
mean -- with the durability this needs. This is a thin driver over it.

🔒 The gate runs FIRST and on every invocation: the 296 panel negatives are re-embedded through this
path and compared against the published array for the arm. Nothing is written if it fails.

Usage:
    python src/102_pool_embed_resumable.py --fasta data/sequences/benign_pool_independent.fasta \\
        --arm esm2_650M --out-tag pool_independent_esm2_650M
    python src/102_pool_embed_resumable.py --selftest
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "v3"
SEQ = ROOT / "data" / "sequences"
GATE_FASTA = SEQ / "benign_negatives_v3.fasta"
TOL = 5e-5
ARMS = {"esm2_650M": ("facebook/esm2_t33_650M_UR50D", ""),
        "esm2_150M": ("facebook/esm2_t30_150M_UR50D", "_esm2_150M"),
        "esm2_35M": ("facebook/esm2_t12_35M_UR50D", "_esm2_35M"),
        "esm2_8M": ("facebook/esm2_t6_8M_UR50D", "_esm2_8M")}


def _load(stem):
    spec = importlib.util.spec_from_file_location(f"_{stem}", ROOT / "src" / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M77 = _load("77_vfdb_negative_embed")


def selftest():
    assert hasattr(M77, "embed") and hasattr(M77, "read_fasta")
    assert M77.SHARD > 0 and M77.MAX_LEN == 1022
    for arm, (_, sfx) in ARMS.items():
        pub = RES / f"embeddings_negative_v3{sfx}.npy"
        assert pub.exists(), f"{arm}: gate target {pub.name} missing"
    print(f"selftest PASS ({len(ARMS)} arms, all gate targets present, shard {M77.SHARD})")


def main():
    ap = argparse.ArgumentParser()
    # 🔴 not `required`, or --selftest cannot run without naming a FASTA it never reads. Checked below.
    ap.add_argument("--fasta")
    ap.add_argument("--arm", default="esm2_650M", choices=sorted(ARMS))
    ap.add_argument("--out-tag")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.fasta or not a.out_tag:
        raise SystemExit("--fasta and --out-tag are required unless --selftest")

    import torch
    from transformers import AutoModel, AutoTokenizer

    model_id, sfx = ARMS[a.arm]
    ck = RES / f"{a.out_tag}_ckpt"
    if a.restart and ck.exists():
        for f in sorted(ck.glob("shard_*.npz")):
            f.unlink()
        print("discarded existing shards")
    dev = a.device or ("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"model={model_id}  arm={a.arm}  device={dev}  checkpoint={ck.name}")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(dev).eval()

    # ---- 🔒 the gate, first and every time ------------------------------------------------------
    gate_recs = M77.read_fasta(GATE_FASTA)
    pub = np.load(RES / f"embeddings_negative_v3{sfx}.npy")
    if len(gate_recs) != pub.shape[0]:
        raise SystemExit(f"gate FASTA has {len(gate_recs)} rows, published has {pub.shape[0]}")
    got = M77.embed(gate_recs, tok, model, dev, a.batch_size, "gate")
    delta = float(np.abs(got - pub).max())
    print(f"gate: max abs delta against embeddings_negative_v3{sfx}.npy = {delta:.3e} "
          f"(tolerance {TOL:.0e})")
    if delta > TOL:
        raise SystemExit("GATE FAILED: this pooling is not the published one. Nothing written.")

    recs = M77.read_fasta(Path(a.fasta))
    print(f"\n{len(recs)} proteins from {Path(a.fasta).name}")
    arr = M77.embed(recs, tok, model, dev, a.batch_size, a.out_tag, ck=ck)
    np.save(RES / f"embeddings_{a.out_tag}.npy", arr)
    (RES / f"embedding_manifest_{a.out_tag}.json").write_text(json.dumps({
        "model": model_id, "arm": a.arm, "tag": a.out_tag,
        "n": int(arr.shape[0]), "dim": int(arr.shape[1]),
        "source_fasta": str(Path(a.fasta)),
        "pooling": "src/02b's include-specials mean, via src/77's checkpointed embed",
        "gate": {"panel_negatives_vs_published": delta, "tolerance": TOL},
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rows": [h for h, _ in recs]}, indent=2) + "\n")
    print(f"\nwrote embeddings_{a.out_tag}.npy {arr.shape}")


if __name__ == "__main__":
    main()
