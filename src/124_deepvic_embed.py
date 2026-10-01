#!/usr/bin/env python3
"""Study AI, embed step: ESM-2 over DeepVIC's benchmark, both arms.

`docs/DEEPVIC_BENCHMARK_PREREGISTRATION.md` § 3 (AI-2, AI-3, AI-4).

Mirrors `src/80` exactly, including the two things that make its output comparable to
everything else here:

  1. the pooling is `src/02b`'s include-specials mean, imported from `src/77` rather than
     reimplemented, and the **gate** re-embeds the 296 panel negatives and requires them to
     reproduce `embeddings_negative_v3*.npy` to 5e-05 before anything is written;
  2. the checkpoint directory carries the **build artifact's fingerprint**, so a resume
     cannot mix two different positive sets -- the failure src/80's comment records.

The row order is fixed here and recorded in the manifest: DeepVIC's 12,989 class-labelled
positives in metadata order, then all 33,456 negatives in FASTA order.  No homology screen is
applied to DeepVIC's negatives, because they are DeepVIC's own construction and evaluating on
their benchmark means using their partition; that difference from `src/79` is stated, not fixed.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results/v3"
DV = ROOT / "data/external/deepvic"
FAA = DV / "DLDB_33456.faa"
META = DV / "metadata.txt"
BUILD = ROOT / "results/deepvic_build.json"
OURS = ROOT / "results/external_class_axis_build.json"
SEQ_OUT = ROOT / "data/sequences/deepvic_ordered.fasta"
TOL = 5e-05
CLASS_RE = re.compile(r"\s*\(VFC\d+\)\s*$")
VFG_RE = re.compile(r"VFG\d+")
ARMS = {
    "esm2_650M": ("facebook/esm2_t33_650M_UR50D", "", ""),
    "esm2_35M": ("facebook/esm2_t12_35M_UR50D", "_esm2_35M", "_esm2_35M"),
}


def load77():
    spec = importlib.util.spec_from_file_location(
        "s77", ROOT / "src" / "77_vfdb_negative_embed.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["s77"] = m
    spec.loader.exec_module(m)
    return m


def read_fasta_seqs(p):
    out, key, buf = {}, None, []
    with p.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith(">"):
                if key is not None:
                    out[key] = "".join(buf)
                key = re.split(r"[\t ]", line[1:].rstrip("\n"), maxsplit=1)[0].strip()
                buf = []
            else:
                buf.append(line.strip())
    if key is not None:
        out[key] = "".join(buf)
    return out


def ordered():
    """The fixed row order, plus the per-row labels the studies need."""
    fa = read_fasta_seqs(FAA)
    with META.open(encoding="utf-8", errors="replace") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    our_reps = {q["vfg"] for q in json.loads(OURS.read_text())["positives"]}

    pos, cls, vfg, shared = [], [], [], []
    for r in rows:
        k = r["DLDB header/Internal Reference"]
        c = CLASS_RE.sub("", r["VF class"]).strip()
        if c == "NA" or k not in fa:
            continue
        m = VFG_RE.search(r["Original Database header"] or "")
        pos.append(k)
        cls.append(c)
        vfg.append(m.group(0) if m else "")
        shared.append(bool(m and m.group(0) in our_reps))
    posset = {r["DLDB header/Internal Reference"] for r in rows}
    neg = [k for k in fa if k not in posset]

    recs = [(k, fa[k]) for k in pos] + [(k, fa[k]) for k in neg]
    meta = {
        "id_of_row": pos + neg,
        "label_of_row": [1] * len(pos) + [0] * len(neg),
        "class_of_row": cls + ["NEGATIVE"] * len(neg),
        "vfg_of_row": vfg + [""] * len(neg),
        "shared_with_our_panel": shared + [False] * len(neg),
        "n_positives": len(pos),
        "n_negatives": len(neg),
    }
    return recs, meta


def selftest():
    ok = True
    recs, meta = ordered()
    b = json.loads(BUILD.read_text())
    if meta["n_positives"] != b["n_labelled_positives"]:
        print(f"  XX {meta['n_positives']} labelled positives, build says "
              f"{b['n_labelled_positives']}")
        ok = False
    if meta["n_negatives"] != b["n_negatives"]:
        print(f"  XX {meta['n_negatives']} negatives, build says {b['n_negatives']}")
        ok = False
    if len(recs) != meta["n_positives"] + meta["n_negatives"]:
        print("  XX record count does not equal positives plus negatives")
        ok = False
    # every per-row list must be the same length as the record list, or a downstream
    # zip() would silently truncate the labels
    for k in ("id_of_row", "label_of_row", "class_of_row", "vfg_of_row",
              "shared_with_our_panel"):
        if len(meta[k]) != len(recs):
            print(f"  XX {k} has {len(meta[k])} entries for {len(recs)} records")
            ok = False
    # no NA may survive into the positive block, and no positive may be labelled NEGATIVE
    npos = meta["n_positives"]
    if "NA" in set(meta["class_of_row"][:npos]):
        print("  XX an NA class reached the positive block")
        ok = False
    if set(meta["class_of_row"][npos:]) != {"NEGATIVE"}:
        print("  XX the negative block is not uniformly labelled")
        ok = False
    if set(meta["label_of_row"][:npos]) != {1} or set(meta["label_of_row"][npos:]) != {0}:
        print("  XX the label vector does not match the block structure")
        ok = False
    if len(set(meta["id_of_row"])) != len(recs):
        print("  XX a row id is repeated")
        ok = False
    # the shared flag must find the 239 the build priced
    nsh = sum(meta["shared_with_our_panel"])
    exp = b["overlap_with_this_project"]["in_our_746_representatives"]
    if nsh > exp:
        print(f"  XX {nsh} shared rows, build priced at most {exp}")
        ok = False
    if nsh == 0:
        print("  XX no shared rows found; the VFG match is not working")
        ok = False
    print(f"  ({meta['n_positives']:,} labelled positives, {meta['n_negatives']:,} negatives, "
          f"{nsh} shared with our 746)")
    print("selftest PASS" if ok else "selftest FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--arm", default="esm2_35M", choices=sorted(ARMS))
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    MODEL, gate_sfx, out_sfx = ARMS[a.arm]
    m77 = load77()
    dev = ("cuda" if torch.cuda.is_available() else
           "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"model={MODEL}  device={dev}  pooling=src/02b's via src/77", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(dev).eval()

    gate_pub = RES / f"embeddings_negative_v3{gate_sfx}.npy"
    got = m77.embed(m77.read_fasta(m77.GATE_FASTA), tok, model, dev, a.batch, "gate")
    delta = float(np.abs(got - np.load(gate_pub)).max())
    print(f"gate: max abs delta vs {gate_pub.name} = {delta:.3e} (tol {TOL:.0e})", flush=True)
    if delta > TOL:
        raise SystemExit("GATE FAILED: pooling is not src/02b's. Nothing written.")

    recs, meta = ordered()
    SEQ_OUT.parent.mkdir(parents=True, exist_ok=True)
    SEQ_OUT.write_text("".join(f">{h}\n{s}\n" for h, s in recs))
    print(f"{len(recs):,} sequences in fixed order -> {SEQ_OUT.name}", flush=True)

    ck = RES / f"deepvic_ckpt{out_sfx}"
    fp = hashlib.sha256(BUILD.read_bytes()).hexdigest()
    stamp = ck / "build.sha256"
    if ck.exists() and stamp.exists() and stamp.read_text().strip() != fp:
        raise SystemExit(
            f"{ck.name} holds shards from a different build; delete it and rerun -- "
            "resuming would mix two sets and nothing downstream would notice.")
    ck.mkdir(parents=True, exist_ok=True)
    stamp.write_text(fp + "\n")
    arr = m77.embed(recs, tok, model, dev, a.batch, "deepvic", ck=ck)

    np.save(RES / f"embeddings_deepvic{out_sfx}.npy", arr)
    (RES / f"embedding_manifest_deepvic{out_sfx}.json").write_text(json.dumps({
        "model": MODEL, "arm": a.arm, "tag": f"deepvic{out_sfx}",
        "n": int(arr.shape[0]), "dim": int(arr.shape[1]),
        "pooling": "src/02b's: mean over the full attention mask, <cls> and <eos> included",
        "gate": {"panel_negatives_vs_published": delta, "tolerance": TOL},
        "no_homology_screen": ("DeepVIC's negatives are used as published; src/79's screen is "
                               "not applied, because evaluating on their benchmark means using "
                               "their partition"),
        **meta,
    }, indent=2) + "\n")
    print(f"wrote embeddings_deepvic{out_sfx}.npy {arr.shape}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
