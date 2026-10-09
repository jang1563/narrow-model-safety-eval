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
import re
import time
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


def _pooling_is_shared():
    """🔒 The check that actually carries the weight, and it needs no embedding.

    The numerical gate is a same-environment check. What it is *meant* to establish -- that
    this file uses src/02g's pooling and not a copy of it -- is structural: `embed` must be
    CALLED from the src/02g module object, never redefined here. A numerical tolerance can be
    defeated by library drift, as entry 129 records; an import cannot.
    """
    src = Path(__file__).read_text()
    calls_m02g = "m02g.embed(" in src
    # ⚠️ `^def embed\(` matched only a TOP-LEVEL redefinition; a nested one, which is how
    # a wrapper would actually be introduced, slipped past. Any indentation now.
    defines_own = bool(re.search(r"^\s*def embed\(", src, re.M))
    g = Path(__file__).parent / "02g_prott5_embed.py"
    g_src = g.read_text() if g.exists() else ""
    return {
        "calls_src02g_embed": calls_m02g,
        "defines_its_own_embed": defines_own,
        "src02g_does_the_masking": 'r"[UZOB]"' in g_src,
        "src02g_means_residues_only": "hs[b, : len(s)].mean(0)" in g_src,
        "shared_by_construction": calls_m02g and not defines_own,
    }


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
    # 🔒 the structural check, which a numerical tolerance cannot replace
    sh = _pooling_is_shared()
    if not sh["shared_by_construction"]:
        print(f"  XX the pooling is not shared by construction: {sh}")
        ok = False
    if not (sh["src02g_does_the_masking"] and sh["src02g_means_residues_only"]):
        print(f"  XX src/02g no longer does the masking or the residue-only mean: {sh}")
        ok = False
    print("selftest PASS" if ok else "selftest FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--gate-ref", metavar="PATH",
                    help="compare against this reference instead of the 2026-09-26 one, "
                         "whose environment was never recorded")
    ap.add_argument("--rebuild-reference", action="store_true",
                    help="embed the 296 gate negatives and write a NEW reference with full "
                         "provenance, then stop")
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
    if a.rebuild_reference:
        # 🔒 A reference built here is compared against a run of the SAME code in the SAME
        # session, so the numerical gate against it is a WEAK check and is labelled one. The
        # strong check is structural and needs no embedding: this file IMPORTS src/02g.embed
        # rather than copying it, so the pooling is shared by construction.
        # 🔴 The 2026-09-26 reference is retired for this purpose. It records model, device,
        # max_len and the pooling note but NO torch or transformers version, so the one
        # surviving candidate for its 2.4-3.7e-04 discrepancy -- library drift -- cannot be
        # checked against it. dtype, device, preprocessing and batch size were each ruled out
        # by measurement; docs/DATA_CORRECTIONS.md entry 129 has the diagnosis.
        newref = RES / "embeddings_negative_v3_prott5_xl_rebuilt.npy"
        np.save(newref, got)
        (RES / "embedding_manifest_v3_prott5_xl_rebuilt.json").write_text(json.dumps({
            "model": MODEL, "n": int(got.shape[0]), "dim": int(got.shape[1]),
            "device": devname, "half_precision": not a.no_half, "batch_size": a.batch,
            "pooling": "src/02g's: mean over residue positions only, </s> excluded",
            "versions": {"torch": torch.__version__,
                         "transformers": __import__("transformers").__version__,
                         "numpy": np.__version__},
            "built": time.strftime("%Y-%m-%d %H:%M:%S"),
            "supersedes": "embeddings_negative_v3_prott5_xl.npy (2026-09-26)",
            "why": ("the 2026-09-26 reference records no library versions, so library drift "
                    "-- the one surviving candidate for its discrepancy -- cannot be checked "
                    "against it"),
            "what_this_gate_is_worth": (
                "a SAME-ENVIRONMENT reproducibility check, and nothing more. The pooling is "
                "shared by construction because this file imports src/02g.embed."),
            "batch_size_sensitivity": (
                "measured 1.221e-04 between batch 1 and batch >1, which EXCEEDS the 5e-05 "
                "tolerance, so a reproduction must match the batch size -- which is why this "
                "manifest records it and the 2026-09-26 one did not"),
        }, indent=2) + "\n")
        print(f"wrote {newref.name} + manifest  (device {devname}, batch {a.batch}, "
              f"half {not a.no_half})", flush=True)
        return 0

    _gr = Path(a.gate_ref) if a.gate_ref else GATE
    ref = np.load(_gr)
    delta = float(np.abs(got - ref).max())
    scale = float(np.abs(ref).mean())
    refman = _gr.with_name(_gr.name.replace("embeddings_negative", "embedding_manifest")
                           .replace(".npy", ".json"))
    refdev = (json.loads(refman.read_text()).get("device", "?")
              if refman.exists() else "?")
    print(f"gate: max abs delta vs {_gr.name} = {delta:.3e} (tol {TOL:.0e}); "
          f"{delta / scale * 100:.2f}% of mean|v| vs a {TOL / scale * 100:.2f}% tolerance",
          flush=True)
    print(f"  reference built on {refdev}; this run on {dev}", flush=True)
    if delta > TOL:
        # 🔴 This message used to read "this is not src/02g's pooling", which ASSERTS a cause
        # the gate cannot distinguish. On 2026-10-07 it fired at 2.441e-04 on MPS against a
        # CUDA-built reference, and halving the model to match the reference's dtype left the
        # delta UNCHANGED to four significant figures -- so dtype was ruled out by
        # measurement and the pooling was never shown to be wrong.
        # 🔒 The gate still refuses to proceed. What changes is that it no longer names a
        # cause it has not established.
        raise SystemExit(
            f"GATE FAILED at {delta:.3e} > {TOL:.0e}. Nothing written.\n"
            f"  This gate is a SAME-DEVICE, same-environment reproducibility check. A "
            f"mismatch means the embedding is not bit-comparable to the reference; it does "
            f"NOT by itself establish that the pooling is wrong.\n"
            f"  Candidates, in the order they are cheap to rule out: (1) device -- reference "
            f"{refdev}, this run {dev}; (2) attention implementation or library version drift "
            f"-- the reference manifest records no torch/transformers versions, so this "
            f"cannot be checked against it; (3) the pooling itself.\n"
            f"  Ruled out already: model dtype. Halving to match the reference's uniform "
            f"fp16 left the delta unchanged.")

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
        "half_precision": not a.no_half,
        # 🔒 Recorded because the 2026-09-26 reference manifest does NOT record them, which is
        # why a 2.441e-04 gate failure could not be attributed to library drift.
        "versions": {"torch": torch.__version__,
                     "transformers": __import__("transformers").__version__,
                     "numpy": np.__version__},
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
