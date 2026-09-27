#!/usr/bin/env python3
"""
76_pool_reduction_embed.py - the pooled 8,259-protein benign set under src/70's reductions, so the
                             out-of-sample false-positive rate can be measured for them.

Why this exists
---------------
🔴 Entry forty-three: § 9.1.2 and § 9.1.3 claimed § 8's fixed-budget objection was answered because the
realised FPR was "0.0656 for both". That figure is 4/61 by construction — `03b` thresholds on the same
61 negatives it then measures — so it is identical for every reduction and for random scores, and it
answers nothing. § 8's mechanism is that capacity raising the **negatives'** scores costs threshold
headroom, and the only quantity that tests it is an **out-of-sample** rate. § 2.6.1 measures exactly
that against this pool: **7.87%** at a nominal 5% with `np.quantile` and **5.98%** conformal.

So this embeds the pool once and writes it under each reduction, which is what `src/49` needs to give
`win_best25` the same treatment the canonical arm already got. **It computes no rate** — `src/49` does.

⚠️ A residue stack for 8,259 proteins would be several gigabytes, so the reductions are computed **inside
the forward pass** and only the pooled vectors are kept. `src/70`'s `reduce_all` is imported rather than
reimplemented, so a reduction here is the same function that produced the panel's.

The gate
--------
Row order and forward pass are checked against the published pool embedding, which `src/02*` produced
with an **include-specials** mean (see § 9.1.4's note on the two means). This script therefore computes
that mean too and requires it to reproduce the published array per protein. If it does not, the rows are
misaligned or the pass differs, and every rate computed downstream would be measuring that instead.

Checkpoint and resume
---------------------
🔴 The first version accumulated every vector in memory and wrote once at the end, so four hours of
forward passes depended on the process surviving. `MEMORY.md` already carries the lesson that ignores —
"per-call JSONL checkpoint+resume", recorded after a long harness was lost the same way. Fixed
2026-09-27, at the cost of redoing the 84 minutes the unprotected run had completed.

Shards of 250 proteins go to `results/v3/pool_reduction_ckpt/`, each holding every requested tag plus
the include-specials mean used by the gate. A rerun counts the complete shards and resumes after them,
so a kill costs at most 249 proteins. ⚠️ The trailing partial shard is deliberately NOT written: if it
were, `start = n_shards * SHARD` would skip proteins on resume. And nothing is combined into a final
array until the gate passes, so an interrupted run leaves no artifact that could be read as finished.

Usage:
    python src/76_pool_reduction_embed.py                       # resumes if shards exist
    python src/76_pool_reduction_embed.py --tags mean_res win_best25
    python src/76_pool_reduction_embed.py --restart             # discard shards and start over
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "v3"
FASTA = ROOT / "data" / "sequences" / "benign_pool_large.fasta"
PUB = RES / "embeddings_pool_large_esm2_650M.npy"
PUBMAN = RES / "embedding_manifest_pool_large_esm2_650M.json"
MODEL = "facebook/esm2_t33_650M_UR50D"
MAX_LEN = 1022
TOL = 5e-5
SHARD = 250      # proteins per checkpoint shard
# the reductions worth the pass: the control, the one reduction that gained, and its two neighbours
TAGS = ("mean_res", "win_best25", "win_best9", "win_max9")


def load70():
    spec = importlib.util.spec_from_file_location("s70", ROOT / "src" / "70_coherent_pooling.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["s70"] = m
    spec.loader.exec_module(m)
    return m


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=list(TAGS))
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--restart", action="store_true", help="discard checkpoint shards and start over")
    a = ap.parse_args()

    m70 = load70()
    pub = np.load(PUB)
    man = json.loads(PUBMAN.read_text())
    ids = [r if isinstance(r, str) else r.get("acc", r.get("id")) for r in man["rows"]]
    recs = read_fasta(FASTA)
    # 🔴 Row order is the whole correctness question here, so it is asserted against the published
    # manifest rather than assumed from the FASTA.
    if [h for h, _ in recs] != ids:
        raise SystemExit(f"FASTA order does not match the published manifest: "
                         f"{len(recs)} vs {len(ids)} rows, first mismatch at "
                         f"{next((i for i, (h, _) in enumerate(recs) if h != ids[i]), None)}")
    print(f"{len(recs)} pool proteins, order matches the published manifest")

    dev = a.device or ("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"model={MODEL}  device={dev}  tags={a.tags}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(dev).eval()

    ck = RES / "pool_reduction_ckpt"
    ck.mkdir(exist_ok=True)
    if a.restart:
        for f in sorted(ck.glob("shard_*.npz")):
            f.unlink()
        print("discarded existing shards")
    # A shard counts only if it holds every tag this run asks for, and only up to the first gap, so
    # widening --tags or losing a middle shard re-runs from there rather than silently skipping.
    have = 0
    while True:
        f = ck / f"shard_{have:05d}.npz"
        if not f.exists():
            break
        with np.load(f) as z:
            if not all(k in z for k in (*a.tags, "_incl")):
                break
        have += 1
    start = have * SHARD
    if start:
        print(f"resuming after {have} shard(s) = {start} proteins")

    pend = {t: [] for t in a.tags}
    pend_incl = []
    next_shard = have

    def flush():
        nonlocal next_shard
        np.savez(ck / f"shard_{next_shard:05d}.npz",
                 **{t: np.vstack(pend[t]).astype(np.float32) for t in a.tags},
                 _incl=np.vstack(pend_incl).astype(np.float32))
        next_shard += 1
        for t in a.tags:
            pend[t].clear()
        pend_incl.clear()

    n_trunc, t0 = 0, time.time()
    for i in range(start, len(recs), a.batch_size):
        batch = recs[i:i + a.batch_size]
        seqs = []
        for _, s in batch:
            if len(s) > MAX_LEN:
                n_trunc += 1
            seqs.append(s[:MAX_LEN])
        enc = tok(seqs, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN + 2)
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            h = model(**enc).last_hidden_state
        msk = enc["attention_mask"]
        for b in range(h.shape[0]):
            idx = msk[b].nonzero().squeeze(-1)
            body = h[b, idx[1:-1]].float().cpu().numpy()
            red = m70.reduce_all(body)
            for t in a.tags:
                pend[t].append(red[t])
            mm = msk[b].unsqueeze(-1).float()
            pend_incl.append(((h[b] * mm).sum(0) / mm.sum(0)).float().cpu().numpy())
            if len(pend_incl) == SHARD:
                flush()
        if (i + a.batch_size) % 200 < a.batch_size or i + a.batch_size >= len(recs):
            el = time.time() - t0
            done = min(i + a.batch_size, len(recs))
            print(f"  {done}/{len(recs)}  {el / 60:.1f} min  "
                  f"(eta {el / done * (len(recs) - done) / 60:.0f} min)", flush=True)
    print(f"  {n_trunc} sequence(s) truncated to {MAX_LEN}")

    # ---- reassemble from the shards plus the unwritten tail -----------------------------------
    acc = {t: [] for t in a.tags}
    incl = []
    for k in range(next_shard):
        with np.load(ck / f"shard_{k:05d}.npz") as z:
            for t in a.tags:
                acc[t].append(z[t])
            incl.append(z["_incl"])
    if pend_incl:
        for t in a.tags:
            acc[t].append(np.vstack(pend[t]).astype(np.float32))
        incl.append(np.vstack(pend_incl).astype(np.float32))
    acc = {t: np.vstack(v) for t, v in acc.items()}
    # 🔴 A shard arithmetic error would show up here as a row count, not as a wrong number later.
    if acc[a.tags[0]].shape[0] != len(recs):
        raise SystemExit(f"reassembled {acc[a.tags[0]].shape[0]} rows for {len(recs)} proteins: the "
                         "shard arithmetic is wrong and nothing is written")

    # ---- the gate -----------------------------------------------------------------------------
    got = np.vstack(incl).astype(np.float32)
    delta = float(np.abs(got - pub).max())
    print(f"\nGATE: include-specials mean against the published pool embedding, "
          f"max abs delta {delta:.3e}")
    if delta > TOL:
        raise SystemExit(f"GATE FAILED: {delta:.3e} exceeds {TOL:.0e}. The rows are misaligned or the "
                         "forward pass differs, so nothing is written.")

    for t in a.tags:
        arr = acc[t].astype(np.float32)
        np.save(RES / f"embeddings_pool_large_esm2_650M_{t}.npy", arr)
        (RES / f"embedding_manifest_pool_large_esm2_650M_{t}.json").write_text(json.dumps({
            "model": MODEL, "tag": f"esm2_650M_{t}", "n": int(arr.shape[0]),
            "dim": int(arr.shape[1]),
            "pooling": f"src/70's '{t}', residues only, computed inside the forward pass",
            "gate": {"include_specials_mean_vs_published": delta, "tolerance": TOL},
            "rows": ids,
        }, indent=2) + "\n")
        print(f"  wrote embeddings_pool_large_esm2_650M_{t}.npy  {arr.shape}")
    print(f"\ntotal {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
