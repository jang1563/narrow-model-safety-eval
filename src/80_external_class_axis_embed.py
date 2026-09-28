#!/usr/bin/env python3
"""
80_external_class_axis_embed.py - embed study B's 746 VFDB representatives.

Pooling is `src/02b`'s include-specials mean, because study B's negatives are the **published pool
embedding** and a positive set pooled differently would not be comparable to it. `src/77`'s `embed` is
imported rather than reimplemented, so the pooling is literally the same function, and its gate is run
again here: the 296 panel negatives must reproduce `embeddings_negative_v3.npy` through this path.

Usage:
    python src/80_external_class_axis_embed.py
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "v3"
SEQ = ROOT / "data" / "sequences"
POS = SEQ / "vfdb_class_axis_positives.fasta"
BUILD = ROOT / "results" / "external_class_axis_build.json"
# 🔑 --arm added 2026-09-28. § 10.6's standing rule is that a geometric claim runs across
# representations, and § 9.1.4 records three findings that looked clean on 650M and vanished on 35M.
# Study B and the organism-stratified follow-up are both one arm, which their own write-ups say.
ARMS = {"esm2_650M": ("facebook/esm2_t33_650M_UR50D", "", ""),
        "esm2_35M": ("facebook/esm2_t12_35M_UR50D", "_esm2_35M", "_esm2_35M")}
TOL = 5e-5


def load77():
    spec = importlib.util.spec_from_file_location("s77", ROOT / "src" / "77_vfdb_negative_embed.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["s77"] = m
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="esm2_650M", choices=sorted(ARMS))
    a = ap.parse_args()
    MODEL, gate_sfx, out_sfx = ARMS[a.arm]
    m77 = load77()
    dev = ("cuda" if torch.cuda.is_available() else
           "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"model={MODEL}  device={dev}  pooling=src/02b's include-specials mean (via src/77)")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(dev).eval()

    # the gate compares against THIS arm's published panel negatives
    gate_pub = RES / f"embeddings_negative_v3{gate_sfx}.npy"
    gate_recs = m77.read_fasta(m77.GATE_FASTA)
    pub = np.load(gate_pub)
    got = m77.embed(gate_recs, tok, model, dev, 4, "gate")
    delta = float(np.abs(got - pub).max())
    print(f"gate: max abs delta against {gate_pub.name} = {delta:.3e}  (tolerance {TOL:.0e})")
    if delta > TOL:
        raise SystemExit("GATE FAILED: this pooling is not src/02b's, so the positives would not be "
                         "comparable to the published pool embedding. Nothing written.")

    recs = m77.read_fasta(POS)
    build = json.loads(BUILD.read_text())
    want = [p["vfg"] for p in build["positives"]]
    if [h for h, _ in recs] != want:
        raise SystemExit("positives FASTA order does not match results/external_class_axis_build.json")
    print(f"\n{len(recs)} representatives, order matches the build artifact")
    # 🔴 2026-09-28: a stale copy of this script, left alive by a pkill whose pattern silently
    # matched nothing, wrote a shard from the PREVIOUS representative rule; a rerun would have
    # resumed after it and combined 250 old-rule vectors with 496 new-rule ones, with no error
    # anywhere. Shards now carry the build's own fingerprint and are refused if it disagrees.
    ck = RES / f"class_axis_pos_ckpt{out_sfx}"
    stamp = ck / "build.sha256"
    fp = hashlib.sha256(BUILD.read_bytes()).hexdigest()
    if ck.exists():
        prev = stamp.read_text().strip() if stamp.exists() else "(none)"
        if prev != fp:
            n = len(list(ck.glob("shard_*.npz")))
            raise SystemExit(
                f"{ck.name} holds {n} shard(s) from a different build "
                f"({prev[:12]} vs {fp[:12]}). Delete the directory and rerun — resuming would mix "
                "two positive sets and nothing downstream would notice.")
    ck.mkdir(parents=True, exist_ok=True)
    stamp.write_text(fp + "\n")
    arr = m77.embed(recs, tok, model, dev, 4, "positives", ck=ck)

    np.save(RES / f"embeddings_class_axis_positives{out_sfx}.npy", arr)
    (RES / f"embedding_manifest_class_axis_positives{out_sfx}.json").write_text(json.dumps({
        "model": MODEL, "arm": a.arm, "tag": f"class_axis_positives{out_sfx}",
        "n": int(arr.shape[0]), "dim": int(arr.shape[1]),
        "pooling": "src/02b's: mean over the full attention mask, <cls> and <eos> included",
        "gate": {"panel_negatives_vs_published": delta, "tolerance": TOL},
        "rows": [h for h, _ in recs],
        "category_of_row": [p["cat"] for p in build["positives"]],
        "vf_of_row": [p["vf"] for p in build["positives"]],
        "species_of_row": [p["species"] for p in build["positives"]],
    }, indent=2) + "\n")
    print(f"\nwrote embeddings_class_axis_positives{out_sfx}.npy {arr.shape}")


if __name__ == "__main__":
    main()
