#!/usr/bin/env python3
"""
63_fspe_m_p3.py - P3 of the mutation-axis preregistration: are annotated activity-abolishing
                  substitutions disfavoured by the model?

P3, quoted from section 4 and not re-derived
--------------------------------------------
    For each tier 2 loss-of-function substitution, the model's probability for the substituted
    residue at that position, against the distribution of all other substitutions at the same
    position, gives a percentile. Supported if the median percentile is below 25 with a sign test at
    p < 0.0083. Not supported at or above 25. Ceiling: if the median is below 5, check for annotation
    circularity before claiming anything.

The percentile is computed over the **19 non-wild-type standard residues** at that position, so it
answers "of the substitutions available here, how lowly does the model rank the one that is known to
abolish activity". Low means disfavoured, which is the hypothesis.

Why this still runs after P2 failed
-----------------------------------
The sixth amendment records P2 as not supported on its own ceiling, so the axis is not a hazard
metric. P3 is not a hazard test: it asks whether the model's substitution preferences line up with
annotated loss of function, which is a variant-effect question and is exactly the Meier et al. claim
this project cites. Running it completes the preregistered record rather than extending a closed
claim, and its ceiling is the interesting part.

⚠️ The effective n, from the twenty-eighth entry of docs/DATA_CORRECTIONS.md and the second amendment:
the tier 2 set is 34 substitutions across 5 proteins and **29 of the 34 are one protein**, anthrax
protective antigen. A median over 34 substitutions that are 85% one protein is a statement about that
protein with four anecdotes attached, and section 3 rule 4 requires that be said in the same sentence
as the count. It is said in the output and in the artifact.

Usage:
    python src/63_fspe_m_p3.py
"""

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from utils import load_positive_sequences, truncate_sequence  # noqa: E402

TIER2 = ROOT / "results" / "tier2_mutagenesis_set.json"
OUT = ROOT / "results" / "fspe_m_p3.json"
ALPHA = 0.05 / 6
STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"


def load_mod04():
    spec = importlib.util.spec_from_file_location(
        "mod04", ROOT / "src" / "04_esm2_masked_prediction.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sign_test_below(vals, cut):
    """One-sided exact sign test: how many are below `cut`."""
    n = len(vals)
    k = sum(1 for v in vals if v < cut)
    p = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return n, k, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    mod04 = load_mod04()
    model_name = a.model or mod04.DEFAULT_MODEL
    dev = a.device or ("cuda" if torch.cuda.is_available() else
                       "mps" if getattr(torch.backends, "mps", None)
                       and torch.backends.mps.is_available() else "cpu")
    tier2 = json.load(open(TIER2))
    loss = [r for r in tier2["kept"] if r["verdict"] == "loss"]
    print(f"model={model_name}  device={dev}  alpha={ALPHA:.4f}")
    print(f"tier 2 loss features {len(loss)} -> {tier2['loss_substitutions']} substitutions "
          f"across {len(tier2['loss_proteins'])} proteins, "
          f"largest share {tier2['largest_protein_share']:.0%}")

    tok = AutoTokenizer.from_pretrained(model_name)
    mlm = AutoModelForMaskedLM.from_pretrained(model_name).to(dev).eval()

    seqs = {}
    for sid, _desc, seq in load_positive_sequences():
        p = sid.split("|")
        seqs[p[1] if len(p) > 1 else sid] = seq

    rows, cache = [], {}
    for r in sorted(loss, key=lambda x: (x["acc"], x["pos"])):
        acc, pos1, wt = r["acc"], r["pos"], r["wt"]
        if acc not in seqs:
            continue
        seq = truncate_sequence(seqs[acc], mod04.MAX_SEQ_LEN)
        if not (1 <= pos1 <= len(seq)) or seq[pos1 - 1] != wt:
            rows.append({"acc": acc, "pos": pos1, "wt": wt, "alt": None,
                         "skipped": "wild-type residue does not match the sequence at this position"})
            continue
        key = (acc, pos1)
        if key not in cache:
            out = mod04.predict_masked_position(seq, pos1 - 1, mlm, tok, dev)
            cache[key] = out["aa_probs"] if out and "aa_probs" in out else None
            print(f"  {acc} {wt}{pos1}  {len(cache)} positions scored", flush=True)
        probs = cache[key]
        if not probs:
            continue
        # the 19 substitutions available at this position, wild type excluded
        alts = {aa: p for aa, p in probs.items() if aa != wt}
        for alt in (r["alts"] or []):
            if alt not in alts:
                continue
            below = sum(1 for aa, p in alts.items() if p < alts[alt])
            pct = 100.0 * below / (len(alts) - 1) if len(alts) > 1 else None
            rows.append({"acc": acc, "pos": pos1, "wt": wt, "alt": alt,
                         "p_alt": alts[alt], "p_wt": probs[wt],
                         "rank_among_19": below + 1, "percentile": pct,
                         "text": r["text"][:120]})

    scored = [x for x in rows if x.get("percentile") is not None]
    pct = [x["percentile"] for x in scored]
    med = float(np.median(pct)) if pct else None
    n, k, p = sign_test_below(pct, 25.0) if pct else (0, 0, float("nan"))
    per_prot = {}
    for x in scored:
        per_prot.setdefault(x["acc"], []).append(x["percentile"])
    dominant = max((len(v) for v in per_prot.values()), default=0) / max(len(scored), 1)

    supported = bool(med is not None and med < 25.0 and p < ALPHA)
    ceiling = bool(med is not None and med < 5.0)
    res = {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"), "model": model_name, "device": dev,
        "statement": ("percentile of the annotated loss-of-function substitution among the 19 "
                      "non-wild-type standard residues at its own position; low means disfavoured"),
        "threshold": "supported if the median is below 25 with a sign test at p < 0.0083",
        "alpha": ALPHA,
        "n_scored": len(scored), "median_percentile": med,
        "sign_test": {"n": n, "k_below_25": k, "p": p},
        "supported": supported,
        "ceiling_triggered": ceiling,
        "ceiling_rule": ("section 4: if the median is below 5, check for annotation circularity "
                         "before claiming anything -- positions annotated because a mutation "
                         "destroyed function are positions the literature has already marked as "
                         "constrained, and a training corpus that has read that literature is not "
                         "an independent witness"),
        "per_protein_counts": {k2: len(v) for k2, v in sorted(per_prot.items())},
        "per_protein_median": {k2: float(np.median(v)) for k2, v in sorted(per_prot.items())},
        "largest_protein_share": dominant,
        "effective_n_caveat": (f"{dominant:.0%} of the scored substitutions come from one protein, "
                               f"so effective n is close to 1 and the median is a statement about "
                               f"that protein with the others as anecdotes"),
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)

    print()
    print(f"{'acc':<9}{'sub':<10}{'p(alt)':>10}{'p(wt)':>10}{'rank/19':>9}{'pctile':>8}")
    print("-" * 56)
    for x in scored:
        print(f"{x['acc']:<9}{x['wt']}{x['pos']}{x['alt']:<5}{x['p_alt']:>10.2e}"
              f"{x['p_wt']:>10.2e}{x['rank_among_19']:>9}{x['percentile']:>8.1f}")
    print()
    print(f"n scored {len(scored)}  median percentile "
          f"{'—' if med is None else f'{med:.1f}'}  "
          f"below 25: {k}/{n}, sign p {p:.4f}")
    print(f"P3 {'SUPPORTED' if supported else 'NOT SUPPORTED'} at alpha {ALPHA:.4f}")
    if ceiling:
        print("CEILING TRIGGERED: median below 5. Section 4 requires the annotation-circularity")
        print("  check before anything is claimed from this.")
    print(f"per protein: {res['per_protein_counts']}")
    print(f"⚠️  {res['effective_n_caveat']}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
