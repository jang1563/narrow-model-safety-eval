#!/usr/bin/env python3
"""
94_class_axis_positives_mean_res.py - the 746 class-axis positives, pooled over residues only.

Study M found the flag rate is U-shaped in protein length, and the short arm has a mechanical candidate:
every embedding in this project is `src/02b`'s mean over the FULL attention mask, `<cls>` and `<eos>`
included, so for a short protein the two special tokens are a large fraction of the average. The
algebra is exact --

    m_full - m_res = [(cls + eos) - 2 m_res] / (L + 2)

-- and the pool's two published arrays confirm it: the distance correlates with 1/(L+2) at r = 0.977
and d x (L+2) is near-constant at 9.5 (CV 0.105).

Testing whether that shift DRIVES the U needs the probe refitted under residue-only pooling, and the
pool already has a mean_res array while the 746 class-axis positives do not. This writes it.

🔒 The gate is the strongest available: the include-specials mean is computed in the SAME pass and must
reproduce the published `embeddings_class_axis_positives.npy` -- same population, same order.

Usage:
    python src/94_class_axis_positives_mean_res.py
    python src/94_class_axis_positives_mean_res.py --selftest
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
FASTA = ROOT / "data" / "sequences" / "vfdb_class_axis_positives.fasta"
PUBLISHED = RES / "embeddings_class_axis_positives.npy"
MANIFEST = RES / "embedding_manifest_class_axis_positives.json"
MODEL = "facebook/esm2_t33_650M_UR50D"
MAX_LEN, TOL = 1022, 5e-5


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


def both_means(seqs, tok, model, dev):
    """Return (include-specials mean, residues-only mean) for one batch, from one forward pass."""
    enc = tok(seqs, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN + 2)
    enc = {k: v.to(dev) for k, v in enc.items()}
    with torch.no_grad():
        h = model(**enc).last_hidden_state
    m = enc["attention_mask"]
    full = (h * m.unsqueeze(-1)).sum(1) / m.unsqueeze(-1).sum(1)
    res = m.clone()
    res[:, 0] = 0                                   # <cls>
    last = m.sum(1) - 1
    res[torch.arange(res.shape[0], device=res.device), last] = 0   # <eos>
    rm = res.unsqueeze(-1)
    only = (h * rm).sum(1) / rm.sum(1)
    return full.float().cpu().numpy(), only.float().cpu().numpy()


def selftest():
    recs = read_fasta(FASTA)
    man = json.loads(MANIFEST.read_text())
    assert len(recs) == man["n"], f"FASTA has {len(recs)}, manifest says {man['n']}"
    assert [h for h, _ in recs] == man["rows"], "FASTA order must match the published manifest"
    pub = np.load(PUBLISHED, mmap_mode="r")
    assert pub.shape[0] == len(recs)
    print(f"selftest PASS ({len(recs)} positives, order matches the published manifest)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    recs = read_fasta(FASTA)
    man = json.loads(MANIFEST.read_text())
    if [h for h, _ in recs] != man["rows"]:
        raise SystemExit("FASTA order does not match the published manifest")
    dev = a.device or ("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"model={MODEL}  device={dev}  {len(recs)} positives, both poolings in one pass")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(dev).eval()

    fulls, onlys, t0 = [], [], time.time()
    for i in range(0, len(recs), a.batch_size):
        seqs = [s[:MAX_LEN] for _, s in recs[i:i + a.batch_size]]
        f, o = both_means(seqs, tok, model, dev)
        fulls.append(f)
        onlys.append(o)
        if (i + a.batch_size) % 100 < a.batch_size or i + a.batch_size >= len(recs):
            el = time.time() - t0
            done = min(i + a.batch_size, len(recs))
            print(f"  {done}/{len(recs)}  {el / 60:.1f} min "
                  f"(eta {el / max(done, 1) * (len(recs) - done) / 60:.0f} min)", flush=True)
    full = np.vstack(fulls).astype(np.float32)
    only = np.vstack(onlys).astype(np.float32)

    # ---- 🔒 the gate: same population, same order, published array ------------------------------
    pub = np.load(PUBLISHED)
    delta = float(np.abs(full - pub).max())
    print(f"\ngate: max abs delta of the include-specials mean against "
          f"{PUBLISHED.name} = {delta:.3e}  (tolerance {TOL:.0e})")
    if delta > TOL:
        raise SystemExit("GATE FAILED: this forward pass does not reproduce the published positives, "
                         "so a residue-only array from it would not be comparable. Nothing written.")

    # 🔑 cross-artifact check: the pool's two arrays give d x (L+2) near 9.5; these should agree
    lens = np.array([min(len(s), MAX_LEN) for _, s in recs], float)
    k = np.linalg.norm(full - only, axis=1) * (lens + 2)
    print(f"      d x (L+2) on the positives: median {np.median(k):.2f} "
          f"(the pool's published pair gives 9.51)")

    np.save(RES / "embeddings_class_axis_positives_mean_res.npy", only)
    (RES / "embedding_manifest_class_axis_positives_mean_res.json").write_text(json.dumps({
        "model": MODEL, "tag": "class_axis_positives_mean_res",
        "n": int(only.shape[0]), "dim": int(only.shape[1]),
        "pooling": "residues only: the attention mask with <cls> and <eos> removed",
        "gate": {"include_specials_vs_published": delta, "tolerance": TOL},
        "d_times_L_plus_2_median": float(np.median(k)),
        "rows": man["rows"]}, indent=2) + "\n")
    print(f"\nwrote embeddings_class_axis_positives_mean_res.npy {only.shape}")


if __name__ == "__main__":
    main()
