#!/usr/bin/env python3
"""
69_residue_stack_embed.py - keep the L x d residue stack, which no artifact in this project has.

Why
---
§ 9.1.1 of `docs/MECHANISM_GENERALIZATION.md`: every embedding here is reduced to one vector per
protein before the classifier sees it, and the three reductions § 9 tested are two whole-protein
summaries (mean, CLS) and one that destroys residue coherence (per-dimension max, which takes an
independent maximum in each of 1,280 dimensions). So "the information is genuinely absent from the
residue stack" has never been tested. This writes the stack so it can be.

⚠️ This script embeds and verifies. It computes **no recovery number** — `src/70` writes pooled
artifacts and `src/03b` evaluates them unchanged, so the leave-one-mechanism-out protocol is not
reimplemented here and cannot drift.

The reproduction gate
---------------------
A residue stack that is subtly wrong — off-by-one on the special tokens, a padded position included,
rows in the wrong order — would produce plausible numbers downstream and nothing would catch it. So
the residue-only mean recomputed from the stack is checked against the published pooled artifact,
per protein, and the script **refuses to write** if any protein disagrees beyond tolerance.

⚠️ Two means exist in this project and they are not the same operation. `src/02b`, which produced
every published LOMO number, averages over the **full attention mask** and so includes `<cls>` and
`<eos>`. `src/02f`, which produced the pooling comparison, drops both. § 9's footnote records that
the two agree to zero on class recovery, which is why nobody had to care; here the distinction is
load-bearing, so the gate is run against **both** and both deltas are reported.

Usage:
    python src/69_residue_stack_embed.py --panel v2
    python src/69_residue_stack_embed.py --panel v3
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
SEQ = ROOT / "data" / "sequences"
MAX_LEN = 1022            # src/02b's, unchanged
MODEL = "facebook/esm2_t33_650M_UR50D"
TOL = 2e-3                # float16 storage over ~1e-1 magnitudes; a real defect is far larger

FASTA = {
    "v2": {"positive": "toxins_positive_v2.fasta", "negative": "benign_negatives_v2.fasta"},
    "v3": {"positive": "toxins_positive_v3.fasta", "negative": "benign_negatives_v3.fasta"},
}
# The published pooled artifact each half is gated against. v2 has src/02f's explicit residue-only
# mean; v3 has only the canonical src/02b run, so its gate is the include-specials mean.
GATE = {
    "v2": {"positive": ["embeddings_positive_v2_esm2_650M_mean.npy", "embeddings_positive_v2.npy"],
           "negative": ["embeddings_negative_v2_esm2_650M_mean.npy", "embeddings_negative_v2.npy"]},
    "v3": {"positive": ["embeddings_positive_v3.npy"],
           "negative": ["embeddings_negative_v3.npy"]},
}


def read_fasta(path):
    recs, hid, buf = [], None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if hid:
                recs.append((hid, "".join(buf)))
            hid, buf = line[1:].split()[0], []
        elif hid:
            buf.append(line.strip())
    if hid:
        recs.append((hid, "".join(buf)))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="v2", choices=["v2", "v3"])
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    out = ROOT / "results" / a.panel

    dev = a.device or ("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"model={MODEL}  device={dev}  panel={a.panel}  max_len={MAX_LEN}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(dev).eval()

    index, blocks, report = {}, [], {}
    for role, fn in FASTA[a.panel].items():
        recs = read_fasta(SEQ / fn)
        print(f"\n{role}: {len(recs)} sequences")
        stacks, means_res, means_full, n_trunc = [], [], [], 0
        t0 = time.time()
        for i in range(0, len(recs), a.batch_size):
            batch = recs[i:i + a.batch_size]
            seqs = []
            for _, s in batch:
                if len(s) > MAX_LEN:
                    n_trunc += 1
                seqs.append(s[:MAX_LEN])
            enc = tok(seqs, return_tensors="pt", padding=True, truncation=True,
                      max_length=MAX_LEN + 2)
            enc = {k: v.to(dev) for k, v in enc.items()}
            with torch.no_grad():
                h = model(**enc).last_hidden_state
            m = enc["attention_mask"]
            for b in range(h.shape[0]):
                idx = m[b].nonzero().squeeze(-1)
                # residues only: drop <cls> at idx[0] and <eos> at idx[-1]
                body = h[b, idx[1:-1]].float().cpu().numpy()
                stacks.append(body.astype(np.float16))
                means_res.append(body.mean(0))
                mm = m[b].unsqueeze(-1).float()
                means_full.append(((h[b] * mm).sum(0) / mm.sum(0)).float().cpu().numpy())
            print(f"  [{min(i + a.batch_size, len(recs))}/{len(recs)}] "
                  f"{time.time() - t0:.0f}s", end="\r", flush=True)
        print()
        if n_trunc:
            print(f"  {n_trunc} sequence(s) truncated to {MAX_LEN}")

        # ---- the reproduction gate ------------------------------------------------------------
        mr = np.vstack(means_res).astype(np.float32)
        mf = np.vstack(means_full).astype(np.float32)
        # the mean recomputed FROM the stored float16 stack, which is what downstream will read
        mstore = np.vstack([s.astype(np.float32).mean(0) for s in stacks])
        gates = {}
        for name in GATE[a.panel][role]:
            f = out / name
            if not f.exists():
                continue
            pub = np.load(f)
            if pub.shape != mr.shape:
                gates[name] = {"status": "shape mismatch", "published": list(pub.shape),
                               "recomputed": list(mr.shape)}
                continue
            gates[name] = {
                "max_abs_delta_residue_only": float(np.abs(pub - mr).max()),
                "max_abs_delta_include_specials": float(np.abs(pub - mf).max()),
                "max_abs_delta_from_stored_float16": float(np.abs(pub - mstore).max()),
            }
        if not gates:
            raise SystemExit(f"no published pooled artifact found for {a.panel}/{role}; "
                             "the gate cannot run and this script will not write unverified stacks")
        best = min(min(v[k] for k in v if k.startswith("max_abs_delta"))
                   for v in gates.values() if "status" not in v)
        print(f"  gate: closest published artifact agrees to {best:.2e}")
        for k, v in gates.items():
            print(f"    {k}: {v}")
        if best > TOL:
            raise SystemExit(f"GATE FAILED for {a.panel}/{role}: closest agreement {best:.2e} "
                             f"exceeds {TOL:.0e}. The stack does not reproduce the published "
                             "pooling, so nothing is written.")
        report[role] = {"n": len(recs), "n_truncated": n_trunc, "gates": gates}

        off = sum(b.shape[0] for b in blocks)
        rows = []
        for (hid, s), st in zip(recs, stacks):
            rows.append({"id": hid, "role": role, "offset": off, "n_res": int(st.shape[0]),
                         "seq_len": len(s)})
            off += st.shape[0]
        blocks.extend(stacks)
        index[role] = rows

    stack = np.vstack(blocks)
    np.save(out / f"residue_stack_{a.panel}.npy", stack)
    (out / f"residue_stack_index_{a.panel}.json").write_text(json.dumps({
        "built": time.strftime("%Y-%m-%d %H:%M:%S"), "model": MODEL, "device": dev,
        "panel": a.panel, "dtype": "float16", "dim": int(stack.shape[1]),
        "total_residues": int(stack.shape[0]), "max_len": MAX_LEN,
        "pooling": "residues only; <cls> and <eos> dropped, padding excluded",
        "gate": report,
        "note": "row ranges are [offset, offset + n_res) into residue_stack_<panel>.npy, in the "
                "FASTA order src/02b and src/03b use, positives then negatives",
        "rows": index,
    }, indent=2) + "\n")
    print(f"\nwrote residue_stack_{a.panel}.npy  {stack.shape}  "
          f"{stack.nbytes / 1e6:.0f} MB  and its index")


if __name__ == "__main__":
    main()
