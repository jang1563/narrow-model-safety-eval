#!/usr/bin/env python3
"""
02i_saprot_embed.py - the first structure-aware representation in this project.

Why this model, and why now
---------------------------
Beta-lactamase is the one class that has resisted everything. Plain Smith-Waterman
alignment beats every embedding method on it (§6i, 31% against 21%), no classifier
head rescues it (§6j, best 25% of four), and it is worst on every ESM-2 size, at
0% on ESM-3. It is also a family defined by a conserved fold and active site. That
combination points at information the sequence models are discarding rather than
information that is absent.

SaProt reads a structure-aware alphabet: every position is the amino acid followed
by its Foldseek 3Di letter, so residue "M" with 3Di "d" is the token "Md". If a
structural view recovers beta-lactamase, the ceiling was the input representation.
If it does not, the class is genuinely hard for this whole family of methods, which
is a stronger negative result than any of the individual failures.

Structures come from 02h: AlphaFold DB v6, 218 of 220 panel proteins matched with
zero length mismatches. The two without structures carry "#" at every position,
which SaProt treats as unknown structure, making them sequence-only rather than
dropping them and changing the panel.

The token count is asserted against the sequence length. SaProt's vocabulary is
2-character tokens, so a tokenizer that silently falls back to per-character
splitting would halve the effective sequence and quietly produce garbage
embeddings rather than an error.

The panel is a runtime choice, added 2026-09-27, and v3 is NOT runnable yet
-------------------------------------------------------------------------
This script was hardcoded to v2 while 02b already had --panel, the same gap 02e had. --panel is
added here for the same reason, but unlike ProtT5 this arm cannot be run on v3 today, and the
reason is worth stating rather than discovering:

`structure_3di_v3.json` carries real Foldseek 3Di strings for the 231 v2 members and the
`no_structure` mask for the **214 members v3 added**, because src/27 built the file by inheriting
v2's entries and masking the new ones. Its own maintenance note says so. SaProt reads an amino acid
AND a structure token at every position, so a masked member is scored sequence-only: a v3 run today
would report a SaProt arm whose entire phage class had no structure, and "SaProt does not reach
phage" would be a statement about missing structures rather than about the model.

That is now enforced rather than remembered. The masked fraction is computed from the annotation
itself and the run refuses above 5%, which is 48.1% on v3 today. The prerequisite is fetching
AlphaFold structures for those 214 and running them through Foldseek, which `02h_saprot_prepare.py`
does and which needs the Linux binary on the HPC.

⚠️ The fraction is computed, not read from `tri["stats"]`, because that field in the v3 file was
inherited from v2 and says `no_structure: 3` against a real 214.

Usage:
    python src/02i_saprot_embed.py --tag saprot_650M
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
RES_ROOT = ROOT / "results"          # the panel picks the subdirectory at runtime
ANN = ROOT / "data" / "annotations"  # and the 3Di file, which is per panel
MAX_LEN = 1022
# A held-out class needs most of its members to carry real structure before a SaProt recovery figure
# for it means anything. Below this, the class is named unreportable for the arm.
MIN_CLASS_COVERAGE = 0.80
MODEL = "westlake-repl/SaProt_650M_AF2"


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


def sa_string(seq, threedi):
    """Interleave into SaProt's 2-character alphabet, truncated like every other run."""
    s, t = seq[:MAX_LEN], threedi[:MAX_LEN]
    assert len(s) == len(t), "sequence and 3Di lengths differ after truncation"
    return "".join(a + b.lower() for a, b in zip(s, t))


def embed(recs, tri, model, tok, dev, bs):
    out, t0 = [], time.time()
    for i in range(0, len(recs), bs):
        chunk = recs[i : i + bs]
        texts, lens = [], []
        for fid, seq in chunk:
            td = tri[fid]["threedi"]
            texts.append(sa_string(seq, td))
            lens.append(min(len(seq), MAX_LEN))
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_LEN + 2).to(dev)
        with torch.no_grad():
            hs = model(**enc).last_hidden_state
        m = enc["attention_mask"].bool()
        for b, n in enumerate(lens):
            idx = m[b].nonzero().squeeze(-1)
            body = hs[b, idx[1:-1]]                      # residues only
            assert body.shape[0] == n, (
                f"token count {body.shape[0]} != residue count {n}; the tokenizer is "
                "not using SaProt's 2-character vocabulary")
            out.append(body.mean(0).float().cpu().numpy())
        if (i + bs) % 50 < bs:
            print(f"    {min(i + bs, len(recs))}/{len(recs)}  {time.time() - t0:.0f}s",
                  flush=True)
    return np.vstack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="saprot_650M")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument(
        "--panel",
        default="v2",
        choices=["v2", "v3"],
        help="Panel version. v2 is the frozen 80/154 panel every published number rests "
        "on; v3 is 149/296. Switches the input FASTA, the 3Di annotation and the output "
        "directory together, so a v3 embedding cannot land on a v2 filename.",
    )
    ap.add_argument(
        "--allow-masked",
        type=float,
        default=0.05,
        help="Maximum fraction of panel members allowed to carry the no_structure mask. "
        "SaProt reads amino acid AND structure per position, so a masked member is scored "
        "sequence-only and is not really a SaProt observation. Above this fraction the run "
        "refuses; see the docstring.",
    )
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pv = a.panel
    OUT = RES_ROOT / pv
    OUT.mkdir(parents=True, exist_ok=True)
    tri = json.load(open(ANN / f"structure_3di_{pv}.json"))
    pos = read_fasta(ROOT / f"data/sequences/toxins_positive_{pv}.fasta")
    neg = read_fasta(ROOT / f"data/sequences/benign_negatives_{pv}.fasta")
    ann = tri["proteins"]
    print(f"panel={pv}  model={MODEL}  device={dev}  {len(pos)} pos / {len(neg)} neg", flush=True)
    assert len({r[0] for r in pos}) == len(pos), "duplicate accession in positives"
    assert len({r[0] for r in neg}) == len(neg), "duplicate accession in negatives"
    assert not ({r[0] for r in pos} & {r[0] for r in neg}), "accession in BOTH label sets"

    # 🔴 The masked fraction is COMPUTED here, not read from tri["stats"], 2026-09-27. The stats
    # block in structure_3di_v3.json was inherited from v2 by src/27 and says no_structure: 3
    # while the real distribution is 214 of 445. A run that trusted that field would have
    # reported a SaProt arm whose entire phage class was scored without structure.
    masked = [f for f, _ in pos + neg if ann.get(f, {}).get("status") != "ok"]
    frac = len(masked) / max(len(pos) + len(neg), 1)
    print(f"3Di coverage: {len(pos) + len(neg) - len(masked)} with structure, "
          f"{len(masked)} masked ({frac:.1%})")

    # 🔴 Per-class coverage, 2026-09-27. An aggregate masked fraction hides the only thing that
    # decides whether an arm is interpretable: WHICH classes are missing. On v3 the aggregate is
    # 18% and phage_peptidoglycan_hydrolase is 0 of 32 -- every member of the class § 10.6.2 is
    # about has no AlphaFold model, so no threshold on the aggregate could have caught that. The
    # per-class table goes in the manifest, and a class below MIN_CLASS_COVERAGE is named as
    # unreportable for this arm rather than left to be scored and quoted.
    cls_cov = {}
    mp = ROOT / "data" / "annotations" / f"mechanism_classes_{pv}.json"
    if mp.exists():
        cls_of = {x["fasta_id"]: x["mechanism_class"]
                  for x in json.load(open(mp))["proteins"]}
        for f, _ in pos:
            c = cls_of.get(f)
            if c is None:
                continue
            ok_bad = cls_cov.setdefault(c, [0, 0])
            ok_bad[0 if ann.get(f, {}).get("status") == "ok" else 1] += 1
        print("per-class 3Di coverage:")
        for c in sorted(cls_cov, key=lambda k: cls_cov[k][0] / sum(cls_cov[k])):
            ok_n, bad_n = cls_cov[c]
            flag = "  <-- UNREPORTABLE for this arm" \
                if ok_n / (ok_n + bad_n) < MIN_CLASS_COVERAGE else ""
            print(f"    {c:<34}{ok_n:>4} ok {bad_n:>4} masked "
                  f"{100 * ok_n / (ok_n + bad_n):>5.0f}%{flag}")
    unreportable = sorted(c for c, (o, b) in cls_cov.items()
                          if o / (o + b) < MIN_CLASS_COVERAGE)

    if frac > a.allow_masked:
        print(f"XX {frac:.1%} of this panel carries the no_structure mask, above the "
              f"{a.allow_masked:.0%} limit. SaProt's structure channel would be absent for those "
              f"members, so a per-class result on them would be a statement about missing "
              f"structures rather than about the model. Fetch the structures first "
              f"(src/02h_saprot_prepare.py, which needs Foldseek on Linux), or pass "
              f"--allow-masked to override deliberately.")
        return 2

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(dev).eval()

    print("--- positives ---", flush=True)
    P = embed(pos, ann, model, tok, dev, a.batch_size)
    print("--- negatives ---", flush=True)
    N = embed(neg, ann, model, tok, dev, a.batch_size)
    assert P.shape[0] == len(pos) and N.shape[0] == len(neg), "row count mismatch"

    np.save(OUT / f"embeddings_positive_{pv}_{a.tag}.npy", P)
    np.save(OUT / f"embeddings_negative_{pv}_{a.tag}.npy", N)
    n_struct = sum(1 for f, _ in pos + neg if ann[f]["status"] == "ok")
    man = {
        "model": MODEL, "panel": pv, "device": dev, "dry_run_tag": a.tag,
        "max_len": MAX_LEN,
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "embedding_dim": int(P.shape[1]),
        "structures_used": n_struct, "structures_masked": len(pos) + len(neg) - n_struct,
        "per_class_coverage": {c: {"ok": o, "masked": b, "fraction": o / (o + b)}
                               for c, (o, b) in sorted(cls_cov.items())},
        "min_class_coverage": MIN_CLASS_COVERAGE,
        "unreportable_classes": unreportable,
        "unreportable_note": ("a held-out class below min_class_coverage has too few real "
                              "structures for a SaProt recovery figure to be about the model; "
                              "any per-class number for it must not be quoted from this arm"),
        "note": "amino acid plus Foldseek 3Di per position, AlphaFold DB v6; "
                "proteins without a structure carry '#' and are effectively sequence-only",
        "positive_rows": [
            {"row": i, "acc": r[0], "name": r[0].split("|")[2], "len": len(r[1]),
             "sha256": hashlib.sha256(r[1].encode()).hexdigest()[:16]}
            for i, r in enumerate(pos)],
        "negative_rows": [
            {"row": i, "acc": r[0], "name": r[0].split("|")[2], "len": len(r[1]),
             "sha256": hashlib.sha256(r[1].encode()).hexdigest()[:16]}
            for i, r in enumerate(neg)],
    }
    json.dump(man, open(OUT / f"embedding_manifest_{pv}_{a.tag}.json", "w"), indent=2)
    print(f"wrote {P.shape} and {N.shape}, {n_struct} with real structure")
    if unreportable:
        print(f"⚠️  UNREPORTABLE classes for this arm: {unreportable}")
        print("   Their members are mostly or wholly mask, so a recovery figure for them would be")
        print("   a statement about AlphaFold coverage rather than about SaProt.")


if __name__ == "__main__":
    sys.exit(main())
