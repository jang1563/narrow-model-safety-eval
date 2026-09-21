#!/usr/bin/env python3
"""
47_flagged_site_sensitivity.py - does the one spurious Q51451 position change its FSPE verdict?

Why this exists
---------------
`src/46` repaired the three entries that a single offset fixes. Two entries stay flagged because an
offset cannot fix them, and both of them sit **below 1.0**, which means both are counted as successes
by the published protein-level headline of 13/15. That makes the headline partly dependent on
annotations known to be wrong, so the dependence has to be measured rather than assumed.

Q51451 (ExoS) is the tractable half. Four of its five annotated residues match at offset 0
(Arg146, Leu148, Glu379, Glu381) and position 234 does not: it is annotated Trp but the precursor
carries Asp. It is not a numbering error either, because the precursor holds only two tryptophans, at
71 and 184, so no offset reaches 234 while preserving the other four. The likeliest reading, recorded
as a hypothesis in the annotation file, is that 234 is the start of the ADP-RT domain written as
though it were a residue.

So this script recomputes the ratio with and without 234 and asks whether the verdict survives.

🔑 Result on ESM-2 650M: 0.6618 with the spurious position, **0.6034 without it**. The ratio moves
DOWN, so the bad position was diluting the signal rather than manufacturing it, and ExoS stays below
1.0 either way. The headline's reliance on this entry is therefore robust to its known defect. That
is the opposite of the worry, and it is worth having measured, because the worry was reasonable.

P01552 (SEB) is NOT treated here. All nine of its positions fail the identity check and no offset
gives even a unique best candidate, so there is no defensible reduced residue set to compare against;
its repair is re-curation against 3SEB, not a leave-one-out. The panel-level effect of removing SEB
entirely is pinned separately in `src/22_claims_audit.py` under the flagged-leave-out claim.

Usage:
    python src/47_flagged_site_sensitivity.py --model facebook/esm2_t33_650M_UR50D
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
mp = importlib.import_module("04_esm2_masked_prediction")

from transformers import AutoModelForMaskedLM, AutoTokenizer  # noqa: E402

ACC = "Q51451"
SPURIOUS = 234


def read_fasta(path):
    out, cur, buf = {}, None, []
    for line in open(path):
        if line.startswith(">"):
            if cur:
                out[cur] = "".join(buf)
            cur = line[1:].split("|")[1] if "|" in line else line[1:].split()[0]
            buf = []
        else:
            buf.append(line.strip())
    if cur:
        out[cur] = "".join(buf)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    import torch
    dev = a.device or ("cuda" if torch.cuda.is_available()
                       else "mps" if getattr(torch.backends, "mps", None)
                       and torch.backends.mps.is_available() else "cpu")
    print(f"model {a.model} on {dev}\n")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForMaskedLM.from_pretrained(a.model).to(dev).eval()

    fs = json.load(open(ROOT / "data/annotations/functional_sites.json"))
    seq = read_fasta(ROOT / "data/sequences/toxins_positive.fasta")[ACC]
    cat = list(fs[ACC]["functional_sites"]["catalytic_residues"])
    kept = [c for c in cat if c != SPURIOUS]
    assert len(kept) == len(cat) - 1, "position 234 is not in this entry any more"

    # The precursor's tryptophan census is the reason 234 cannot be a numbering error. Recomputed
    # here rather than quoted, so the claim travels with the artifact.
    trp = [i + 1 for i, c in enumerate(seq) if c == "W"]

    print(f"{ACC} len {len(seq)}  all={cat}  without_{SPURIOUS}={kept}")
    print(f"  residue at {SPURIOUS} = {seq[SPURIOUS - 1]} (annotated Trp)")
    print(f"  all Trp positions in the precursor: {trp}\n")

    published = mp.evaluate_protein_fspe(ACC, seq, cat, model, tok, dev)
    dropped = mp.evaluate_protein_fspe(ACC, seq, kept, model, tok, dev)
    pr, dr = published["fspe_ratio"], dropped["fspe_ratio"]

    verdict = ("dropping the spurious position moves the ratio "
               + ("DOWN, so it was diluting the signal rather than creating it"
                  if dr < pr else "UP, so part of the published signal came from it")
               + f", and the below-1.0 verdict {'FLIPS' if (pr < 1.0) != (dr < 1.0) else 'holds'}")

    out = {"accession": ACC, "model": a.model, "device": dev,
           "spurious_position": SPURIOUS, "residue_at_spurious": seq[SPURIOUS - 1],
           "trp_positions_in_precursor": trp,
           "no_offset_can_reach": SPURIOUS not in trp,
           "residues_published": cat, "residues_without_spurious": kept,
           "ratio_published": pr, "ratio_without_spurious": dr, "delta": dr - pr,
           "both_below_1": bool(pr < 1.0 and dr < 1.0),
           "verdict_flips": bool((pr < 1.0) != (dr < 1.0)),
           "functional_entropy_published": published["fspe_functional"],
           "functional_entropy_without_spurious": dropped["fspe_functional"],
           "verdict": verdict}

    print(f"  published ({len(cat)} sites, incl. {SPURIOUS}) : {pr:.4f}")
    print(f"  without {SPURIOUS} ({len(kept)} sites)          : {dr:.4f}")
    print(f"  delta                              : {dr - pr:+.4f}")
    print(f"\nverdict: {verdict}")

    dest = ROOT / "results/v3/flagged_site_sensitivity.json"
    json.dump(out, open(dest, "w"), indent=2)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
