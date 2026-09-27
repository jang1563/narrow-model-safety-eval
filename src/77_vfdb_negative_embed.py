#!/usr/bin/env python3
"""
77_vfdb_negative_embed.py - embed study A1's hard negatives, so src/49 can measure the rate on them.

What A1 is testing
------------------
Criterion 1 of `docs/DETECTOR_CRITERIA.md` is the project's worst: the panel's negatives split
178 train / 118 calibrate / **0 test**. § 2.6.1 measures the out-of-sample rate against the
8,259-protein benign pool — **7.867%** at a nominal 5% with `np.quantile`, **5.978%** conformal — and
`docs/NEGATIVE_EXPANSION_PREREGISTRATION.md` study A1 predicts that **VFDB's non-toxin virulence
factors do worse**, because they are pathogen-produced and largely secreted, the two features § 2.3's
provenance probe already reads at AUROC 0.818. A pool of cytosolic housekeeping proteins is an easy
negative set; these are not.

Pooling, and why it is the include-specials mean
-----------------------------------------------
🔴 The comparison is against the **canonical** arm's figures, and `src/02b` produced those by averaging
over the full attention mask — `<cls>` and `<eos>` included (§ 9.1.4 records the two means and that they
differ by up to 0.06 per dimension). So this script reproduces `src/02b`'s pooling exactly. Using the
residue-only mean here would compare an A1 rate against a differently-pooled baseline.

The gate
--------
VFDB has no published embedding to check against, so the forward pass is gated on something that does:
the **panel negatives**, embedded through this same code path and required to reproduce
`embeddings_negative_v3.npy`. If they do, the pooling and the pass are `src/02b`'s; if not, nothing is
written. ⚠️ This gates the *method*, not the VFDB rows — their order is asserted against the FASTA and
recorded in the manifest, which is then the only source of truth for it.

Checkpointing follows `src/76`: shards of 250, resume after the last complete one, the partial tail
never written, and the row count asserted on reassembly.

Usage:
    python src/77_vfdb_negative_embed.py
    python src/77_vfdb_negative_embed.py --restart
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "v3"
SEQ = ROOT / "data" / "sequences"
VFDB_FASTA = SEQ / "vfdb_negatives.fasta"
VFDB_META = ROOT / "results" / "vfdb_negative_set.json"
GATE_FASTA = SEQ / "benign_negatives_v3.fasta"
GATE_PUB = RES / "embeddings_negative_v3.npy"
MODEL = "facebook/esm2_t33_650M_UR50D"
MAX_LEN = 1022
TOL = 5e-5
SHARD = 250


def read_fasta(path):
    out, hid, buf = [], None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if hid:
                out.append((hid, "".join(buf)))
            hid, buf = line[1:].split()[0], []
        elif hid:
            buf.append(line.strip())
    if hid:
        out.append((hid, "".join(buf)))
    return out


def embed(recs, tok, model, dev, bs, label, ck=None):
    """src/02b's pooling exactly: mean over the FULL attention mask, specials included."""
    have = 0
    if ck is not None:
        ck.mkdir(parents=True, exist_ok=True)
        while (ck / f"shard_{have:05d}.npz").exists():
            have += 1
        if have:
            print(f"  resuming after {have} shard(s) = {have * SHARD} sequences")
    start = have * SHARD
    pend, next_shard, n_trunc, t0 = [], have, 0, time.time()
    for i in range(start, len(recs), bs):
        seqs = []
        for _, s in recs[i:i + bs]:
            if len(s) > MAX_LEN:
                n_trunc += 1
            seqs.append(s[:MAX_LEN])
        enc = tok(seqs, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN + 2)
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1)
        pend.extend(((h * m).sum(1) / m.sum(1)).float().cpu().numpy())
        while ck is not None and len(pend) >= SHARD:
            np.savez(ck / f"shard_{next_shard:05d}.npz",
                     v=np.vstack(pend[:SHARD]).astype(np.float32))
            del pend[:SHARD]
            next_shard += 1
        if (i + bs) % 500 < bs or i + bs >= len(recs):
            el = time.time() - t0
            done = min(i + bs, len(recs))
            print(f"  {label} {done}/{len(recs)}  {el / 60:.1f} min  "
                  f"(eta {el / max(done - start, 1) * (len(recs) - done) / 60:.0f} min)", flush=True)
    out = []
    if ck is not None:
        for k in range(next_shard):
            with np.load(ck / f"shard_{k:05d}.npz") as z:
                out.append(z["v"])
    if pend:
        out.append(np.vstack(pend).astype(np.float32))
    arr = np.vstack(out).astype(np.float32)
    if arr.shape[0] != len(recs):
        raise SystemExit(f"{label}: reassembled {arr.shape[0]} rows for {len(recs)} sequences")
    if n_trunc:
        print(f"  {label}: {n_trunc} sequence(s) truncated to {MAX_LEN}")
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--restart", action="store_true")
    a = ap.parse_args()
    ck = RES / "vfdb_neg_ckpt"
    if a.restart and ck.exists():
        for f in sorted(ck.glob("shard_*.npz")):
            f.unlink()
        print("discarded existing shards")

    dev = a.device or ("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"model={MODEL}  device={dev}  pooling=src/02b's include-specials mean")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(dev).eval()

    # ---- the gate, first: this code path must reproduce the published panel negatives -----------
    gate_recs = read_fasta(GATE_FASTA)
    pub = np.load(GATE_PUB)
    if len(gate_recs) != pub.shape[0]:
        raise SystemExit(f"gate FASTA has {len(gate_recs)} rows, published array has {pub.shape[0]}")
    print(f"\ngate: re-embedding {len(gate_recs)} panel negatives through this path")
    got = embed(gate_recs, tok, model, dev, a.batch_size, "gate")
    delta = float(np.abs(got - pub).max())
    print(f"gate: max abs delta against {GATE_PUB.name} = {delta:.3e}  (tolerance {TOL:.0e})")
    if delta > TOL:
        raise SystemExit("GATE FAILED: this pooling is not src/02b's, so an A1 rate computed with it "
                         "would not be comparable to section 2.6.1's. Nothing written.")

    # ---- the VFDB negatives --------------------------------------------------------------------
    recs = read_fasta(VFDB_FASTA)
    meta = json.loads(VFDB_META.read_text())
    vf_of = {x["vfg"]: x["vf"] for x in meta["admitted"]}
    if [h for h, _ in recs] != [x["vfg"] for x in meta["admitted"]]:
        raise SystemExit("vfdb_negatives.fasta order does not match results/vfdb_negative_set.json")
    print(f"\n{len(recs)} VFDB hard negatives, order matches the artifact")
    arr = embed(recs, tok, model, dev, a.batch_size, "vfdb", ck=ck)

    np.save(RES / "embeddings_vfdb_neg_esm2_650M.npy", arr)
    # one representative per VF group, in first-seen order: amendment 5's effective-n subset
    seen, rep_idx = set(), []
    for i, (h, _) in enumerate(recs):
        if vf_of[h] not in seen:
            seen.add(vf_of[h])
            rep_idx.append(i)
    (RES / "embedding_manifest_vfdb_neg_esm2_650M.json").write_text(json.dumps({
        "model": MODEL, "tag": "vfdb_neg_esm2_650M", "n": int(arr.shape[0]),
        "dim": int(arr.shape[1]),
        "pooling": "src/02b's: mean over the full attention mask, <cls> and <eos> included",
        "gate": {"panel_negatives_vs_published": delta, "tolerance": TOL,
                 "note": "VFDB has no published embedding, so the gate is that this code path "
                         "reproduces embeddings_negative_v3.npy"},
        "rows": [h for h, _ in recs],
        "vf_of_row": [vf_of[h] for h, _ in recs],
        "representative_rows": rep_idx,
        "n_distinct_vf": len(rep_idx),
        "redundancy_factor": round(len(recs) / len(rep_idx), 3),
    }, indent=2) + "\n")
    print(f"\nwrote embeddings_vfdb_neg_esm2_650M.npy {arr.shape}")
    print(f"  {len(rep_idx)} distinct VF groups, redundancy {len(recs) / len(rep_idx):.3f}")


if __name__ == "__main__":
    main()
