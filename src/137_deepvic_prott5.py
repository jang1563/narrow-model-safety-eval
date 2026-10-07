#!/usr/bin/env python3
"""Embed DeepVIC's benchmark with ProtT5 -- a second model FAMILY, not a second size.

The DeepVIC arm's "two arms" are ESM-2 650M and 35M: one family at two sizes, trained on the
same data. Every verdict there that the two agree on is therefore weaker evidence than it looks.
ProtT5 (`Rostlab/prot_t5_xl_half_uniref50-enc`) is a different architecture trained on a
different corpus, and the repository already holds a ProtT5 embedding of the PANEL, so the
pooling gate has a target.

🔴 ProtT5's pooling is NOT ESM-2's. src/02g means over residue positions only, excluding
</s>; src/02b includes the specials. The two are not interchangeable, so `embed` is imported
from src/02g and the gate compares against `embeddings_negative_v3_prott5_xl.npy`.

Row order is src/124's `ordered()`, so this array is row-aligned with the ESM-2 ones.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results/v3"
BUILD = ROOT / "results/deepvic_build.json"
GATE = RES / "embeddings_negative_v3_prott5_xl.npy"
TOL = 5e-05
MODEL = "Rostlab/prot_t5_xl_half_uniref50-enc"


def load(mod):
    spec = importlib.util.spec_from_file_location(mod[:4], ROOT / "src" / mod)
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod[:4]] = m
    spec.loader.exec_module(m)
    return m


def selftest():
    ok = True
    m02g, m124 = load("02g_prott5_embed.py"), load("124_deepvic_embed.py")
    if not hasattr(m02g, "embed"):
        print("  XX src/02g exposes no embed; this file must not reimplement the pooling")
        ok = False
    if m02g.MODEL != MODEL:
        print(f"  XX src/02g's model is {m02g.MODEL}, not {MODEL}")
        ok = False
    if m02g.MAX_LEN != 1022:
        print(f"  XX src/02g truncates at {m02g.MAX_LEN}, not 1022")
        ok = False
    # 🔒 the gate target must exist and be a ProtT5 array, not an ESM-2 one
    if not GATE.exists():
        print(f"  XX {GATE.name} missing; there is nothing to gate the pooling against")
        ok = False
    else:
        a = np.load(GATE)
        if a.shape != (296, 1024):
            print(f"  XX gate array is {a.shape}, expected (296, 1024) for ProtT5")
            ok = False
    recs, meta = m124.ordered()
    if len(recs) != 46445 or meta["n_positives"] != 12989:
        print(f"  XX order gives {len(recs)} rows / {meta['n_positives']} positives")
        ok = False
    print("selftest PASS" if ok else "selftest FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--no-half", action="store_true",
                    help="keep the checkpoint's mixed fp16/fp32 dtypes instead of matching "
                         "the gate reference's uniform fp16; for diagnosing the gate only")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    m02g, m124 = load("02g_prott5_embed.py"), load("124_deepvic_embed.py")
    dev = ("cuda" if torch.cuda.is_available() else
           "mps" if torch.backends.mps.is_available() else "cpu")
    devname = torch.cuda.get_device_name(0) if dev == "cuda" else dev
    print(f"model={MODEL}  device={dev} ({devname})  pooling=src/02g's residue-only mean",
          flush=True)
    tok = T5Tokenizer.from_pretrained(MODEL, do_lower_case=False, legacy=True)
    model = T5EncoderModel.from_pretrained(MODEL).to(dev).eval()
    # 🔴 This file loaded the model WITHOUT halving while its gate compares against
    # `embeddings_negative_v3_prott5_xl.npy`, which `src/02g.main()` produced from a
    # `.half()`-ed model. So the precisions never matched and the gate could not pass on any
    # device -- it had simply never run, because the cluster job stayed queued. Found locally
    # at a delta of 2.441e-04 against a 5e-05 tolerance.
    # 🔒 The fix is to reproduce the REFERENCE's dtype, not to relax the tolerance. Transformers
    # 5.x keeps per-tensor checkpoint dtypes, so this model arrives as an fp16/fp32 MIX; the
    # reference is uniformly fp16.
    if not a.no_half:
        model = model.half()
    dts = sorted({str(q.dtype).replace("torch.", "") for q in model.parameters()})
    print(f"  param dtypes after load: {dts}", flush=True)

    gate_recs = m02g.read_fasta(
        ROOT / "data/sequences/benign_negatives_v3.fasta")
    got = m02g.embed(gate_recs, model, tok, dev, a.batch)
    delta = float(np.abs(got - np.load(GATE)).max())
    print(f"gate: max abs delta vs {GATE.name} = {delta:.3e} (tol {TOL:.0e})", flush=True)
    if delta > TOL:
        raise SystemExit("GATE FAILED: this is not src/02g's pooling. Nothing written.")

    recs, meta = m124.ordered()
    ck = RES / "deepvic_ckpt_prott5_xl"
    fp = hashlib.sha256(BUILD.read_bytes() + MODEL.encode() + devname.encode()).hexdigest()
    stamp = ck / "build.sha256"
    if ck.exists() and stamp.exists() and stamp.read_text().strip() != fp:
        raise SystemExit(f"{ck.name} holds shards from a different build/model/device; "
                         "delete it and rerun")
    ck.mkdir(parents=True, exist_ok=True)
    stamp.write_text(fp + "\n")

    # 🔒 src/02g's embed has no shard/resume, so the work is chunked here and each chunk is
    # written as it completes -- a two-hour job on 46,445 sequences should not be all-or-nothing
    CH = 2000
    done = sorted(ck.glob("chunk_*.npy"))
    arrs = [np.load(f) for f in done]
    start = sum(x.shape[0] for x in arrs)
    if start:
        print(f"  resuming after {start} sequences", flush=True)
    for i in range(start, len(recs), CH):
        part = m02g.embed(recs[i:i + CH], model, tok, dev, a.batch)
        np.save(ck / f"chunk_{i:06d}.npy", part)
        arrs.append(part)
        print(f"  {min(i + CH, len(recs))}/{len(recs)}", flush=True)
    arr = np.vstack(arrs)
    if arr.shape[0] != len(recs):
        raise SystemExit(f"reassembled {arr.shape[0]} rows for {len(recs)} sequences")

    np.save(RES / "embeddings_deepvic_prott5_xl.npy", arr)
    (RES / "embedding_manifest_deepvic_prott5_xl.json").write_text(json.dumps({
        "model": MODEL, "arm": "prott5_xl", "tag": "deepvic_prott5_xl",
        "n": int(arr.shape[0]), "dim": int(arr.shape[1]), "device": devname,
        # ⚠️ src/02g halves the model only on CUDA, so a CUDA run is fp16 and an MPS or CPU
        # run is fp32. Recorded because precision is not cosmetic here: on 2026-10-07 the
        # ESM-2 35M margin ordering moved by exactly one adjacent rank swap -- rho 0.8857 to
        # 0.8901, 12/2730 -- between a float32 and a float64 cosine computation.
        "half_precision": dev == "cuda",
        "precision_note": ("fp16 on CUDA, fp32 otherwise; a margin ordering can differ by "
                           "an adjacent rank swap between precisions"),
        "pooling": "src/02g's: mean over residue positions only, </s> excluded",
        "pooling_differs_from_esm2": ("src/02b includes the specials; the two poolings are "
                                      "not interchangeable and the gate is ProtT5's own"),
        "gate": {"panel_negatives_vs_published": delta, "tolerance": TOL},
        "row_aligned_with": "results/v3/embeddings_deepvic.npy, via src/124's ordered()",
        **{k: meta[k] for k in ("id_of_row", "label_of_row", "class_of_row",
                                "vfg_of_row", "shared_with_our_panel",
                                "n_positives", "n_negatives")},
    }, indent=2) + "\n")
    print(f"wrote embeddings_deepvic_prott5_xl.npy {arr.shape}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
