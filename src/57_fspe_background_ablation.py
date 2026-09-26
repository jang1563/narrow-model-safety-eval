#!/usr/bin/env python3
"""
57_fspe_background_ablation.py - the FSPE background three ways, because the documents describe a
                                 background the code does not build.

The defect this measures
------------------------
Three documents state that FSPE's background excludes the two flanking positions on each side of
every functional site:

    docs/ARCHITECTURE.md line 73
    docs/EVALUATION_REPORT.md line 70
    docs/MUTATION_EXTENSION_PREREGISTRATION.md line 100

`src/04_esm2_masked_prediction.py` samples its background from `all_positions - func_positions` and
nothing else. **No flanking exclusion exists anywhere in the repository.** Reconstructing the sample
deterministically (`RandomState(42)`, 20 draws) shows 9 of the 300 sampled background positions,
3.00%, lie within +/-2 of a functional site, across 6 of the 15 panel proteins.

That matters twice. Every published FSPE ratio, including the 12/14 protein-level headline, used the
background the code builds and not the one the documents describe. And
`docs/MUTATION_EXTENSION_PREREGISTRATION.md` section 2.1 fixes FSPE-M's background as "exactly as
FSPE defines it", so which of the two it means has to be settled before FSPE-M is computed rather
than after.

What this script does
---------------------
One forward pass per distinct position, then three reductions over three background sets:

    published            what src/04 actually samples: RandomState(42), 20 draws, no exclusion
    drop_flanking        that same sample minus its flanking members. Isolates the contamination:
                         the draw is identical, so a difference cannot be a different random sample
    resample_excluded    the documented metric, built properly: same seed, 20 draws from candidates
                         that exclude functional positions AND their +/-2 neighbours

`drop_flanking` and `resample_excluded` answer different questions and both are reported. The first
says what the contamination cost. The second says what the documented metric would have produced.

The reproduction gate, which runs first
---------------------------------------
The `published` arm must reproduce `results/fspe_results.json` to stored precision, per protein, on
`fspe_functional`, `fspe_nonfunctional` and `fspe_ratio`. If it does not, the entropies have moved
for some reason unrelated to the background and nothing else here is interpretable. The script says
so and exits 2. This is also step 3 of the preregistration's run order, which requires FSPE to
reproduce before the mutation axis is touched at all.

Cost
----
15 proteins, 3-9 functional positions each plus up to 40 distinct background positions, so roughly
600-700 masked forward passes with ESM-2 650M. About 15-30 minutes on an M-series GPU, less on CUDA.
No sequence is written to the output: the artifact carries positions, entropies and ratios only.

Usage:
    python src/57_fspe_background_ablation.py
    python src/57_fspe_background_ablation.py --device cpu       # if mps misbehaves
    python src/57_fspe_background_ablation.py --limit 3          # smoke test, 3 proteins
"""

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from utils import (  # noqa: E402
    load_functional_sites,
    load_positive_sequences,
    sequence_functional_positions,
    truncate_sequence,
)

OUT = ROOT / "results" / "fspe_background_ablation.json"
PUBLISHED = ROOT / "results" / "fspe_results.json"
FLANK = 2          # the exclusion radius the three documents describe
N_BG = 20          # src/04's n_nonfunctional
SEED = 42          # src/04's seed
N_PERM = 20000     # src/21's permutation count, so the protein-level tests are comparable


def load_mod04():
    """Reuse src/04's own masking code, so the entropy is literally the same computation.

    Re-implementing it would make the reproduction gate meaningless: it would compare this script
    against itself rather than against the pipeline that produced the published numbers.
    """
    p = ROOT / "src" / "04_esm2_masked_prediction.py"
    spec = importlib.util.spec_from_file_location("mod04", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tests(ratio):
    """src/21's protein-level tests, copied verbatim so the arms are comparable to the headline."""
    n = len(ratio)
    k = int((ratio < 1.0).sum())
    p_sign = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    logr = np.log(np.clip(ratio, 1e-9, None))
    obs = float(logr.mean())
    rng = np.random.default_rng(0)
    null = (logr * rng.choice([-1, 1], size=(N_PERM, n))).mean(axis=1)
    return {"n_proteins": n, "n_ratio_below_1": k, "sign_test_one_sided_p": p_sign,
            "mean_log_ratio": obs, "permutation_p": float((null <= obs).mean())}


def background_sets(seq_len, func0):
    """The three background position sets, 0-indexed.

    `published` reproduces src/04 exactly: same RandomState, same candidate construction, same
    sorted() call. Any deviation here would show up in the reproduction gate as an entropy
    mismatch, which is the point of having the gate.
    """
    all_pos = set(range(seq_len))
    cand_pub = list(all_pos - set(func0))
    rng = np.random.RandomState(SEED)
    pub = sorted(rng.choice(cand_pub, min(N_BG, len(cand_pub)), replace=False))

    flank = {f + d for f in func0 for d in range(-FLANK, FLANK + 1) if d != 0}
    dropped = [int(p) for p in pub if p in flank]
    drop = [int(p) for p in pub if p not in flank]

    cand_excl = sorted(all_pos - set(func0) - flank)
    rng2 = np.random.RandomState(SEED)
    resample = sorted(rng2.choice(cand_excl, min(N_BG, len(cand_excl)), replace=False))

    return ({"published": [int(x) for x in pub],
             "drop_flanking": drop,
             "resample_excluded": [int(x) for x in resample]},
            dropped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="default: src/04's DEFAULT_MODEL")
    ap.add_argument("--device", default=None, help="cuda | mps | cpu; default: auto")
    ap.add_argument("--limit", type=int, default=0, help="first N proteins only, for a smoke test")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    mod04 = load_mod04()
    model_name = a.model or mod04.DEFAULT_MODEL
    dev = a.device or ("cuda" if torch.cuda.is_available() else
                       "mps" if getattr(torch.backends, "mps", None)
                       and torch.backends.mps.is_available() else "cpu")
    print(f"model={model_name}  device={dev}  flank=+/-{FLANK}  n_bg={N_BG}  seed={SEED}")

    tok = AutoTokenizer.from_pretrained(model_name)
    mlm = AutoModelForMaskedLM.from_pretrained(model_name).to(dev).eval()

    sites_all = load_functional_sites()
    seqs = {}
    for sid, seq, _ in load_positive_sequences():
        parts = sid.split("|")
        seqs[parts[1] if len(parts) >= 2 else sid] = seq

    pub_prev = {x.get("uniprot_id") or x.get("accession"): x
                for x in json.load(open(PUBLISHED))["per_protein"]}

    per_protein, drift = [], []
    accs = [k for k in sites_all if not k.startswith("_")]
    if a.limit:
        accs = accs[:a.limit]

    for acc in accs:
        info = sites_all[acc]
        if acc not in seqs:
            print(f"--- {acc}: not in the positive set, skipped")
            continue
        sites = info["functional_sites"]
        if not (sites.get("catalytic_residues") or []):
            print(f"--- {acc}: no catalytic_residues, skipped as src/04 does")
            continue
        seq = truncate_sequence(seqs[acc], mod04.MAX_SEQ_LEN)
        resolved = sequence_functional_positions(acc, seqs[acc], sites, verbose=False)
        func0 = sorted({p - 1 for p in resolved["positions"] if 0 <= p - 1 < len(seq)})
        if not func0:
            print(f"--- {acc}: no valid functional positions, skipped")
            continue

        bg, dropped = background_sets(len(seq), func0)
        need = sorted(set(func0) | {p for s in bg.values() for p in s})
        print(f"--- {acc}: {len(func0)} functional, {len(need)} distinct positions, "
              f"{len(dropped)} flanking in the published sample")

        ent = {}
        for i, pos in enumerate(need):
            r = mod04.predict_masked_position(seq, pos, mlm, tok, dev)
            if r is not None:
                ent[pos] = r["entropy"]
            if (i + 1) % 20 == 0 or i + 1 == len(need):
                print(f"      {i + 1}/{len(need)}", flush=True)

        fmean = float(np.mean([ent[p] for p in func0 if p in ent]))
        row = {"uniprot_id": acc, "sequence_length": len(seq),
               "n_functional_sites": len(func0),
               "precursor_offset": resolved["offset"],
               "numbering_flagged": resolved["flagged"],
               "flanking_in_published_sample": dropped,
               "fspe_functional": fmean, "arms": {}}
        for name, positions in bg.items():
            vals = [ent[p] for p in positions if p in ent]
            bmean = float(np.mean(vals)) if vals else None
            row["arms"][name] = {
                "n_background": len(vals),
                "fspe_nonfunctional": bmean,
                "fspe_ratio": (fmean / bmean) if bmean else None,
            }
        per_protein.append(row)

        # ---- the reproduction gate, per protein --------------------------------------------
        prev = pub_prev.get(acc)
        if prev:
            mine = row["arms"]["published"]
            for key, got in (("fspe_functional", fmean),
                             ("fspe_nonfunctional", mine["fspe_nonfunctional"]),
                             ("fspe_ratio", mine["fspe_ratio"])):
                want = prev.get(key)
                if want is None or got is None:
                    continue
                if abs(got - want) > 1e-6:
                    drift.append({"acc": acc, "field": key, "recomputed": got, "published": want,
                                  "delta": got - want})

    # ---- protein-level tests per arm ------------------------------------------------------
    arms = {}
    for name in ("published", "drop_flanking", "resample_excluded"):
        r = np.array([x["arms"][name]["fspe_ratio"] for x in per_protein
                      if x["arms"][name]["fspe_ratio"] is not None], float)
        arms[name] = _tests(r) if len(r) else None

    res = {"model": model_name, "device": dev, "flank_radius": FLANK, "n_background": N_BG,
           "seed": SEED, "n_permutations": N_PERM,
           "reproduces_published": not drift, "drift": drift,
           "note": ("the published arm is what src/04 samples today; resample_excluded is what "
                    "ARCHITECTURE.md line 73, EVALUATION_REPORT.md line 70 and the mutation "
                    "preregistration section 2.1 describe; drop_flanking isolates the "
                    "contamination by keeping the same draw"),
           "protein_level": arms, "per_protein": per_protein}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)

    print()
    if drift:
        print(f"XX the published arm does NOT reproduce results/fspe_results.json: "
              f"{len(drift)} field(s) moved")
        for d in drift[:8]:
            print(f"   {d['acc']} {d['field']}: {d['recomputed']:.6f} vs {d['published']:.6f} "
                  f"(delta {d['delta']:+.2e})")
        print("   Nothing below is interpretable and the mutation axis stops here, per the")
        print("   preregistration's section 7 step 3.")
    else:
        print("OK the published arm reproduces results/fspe_results.json to 1e-6 on every field,")
        print("   so each arm below differs from it only in which background positions were used")

    print()
    hdr = f"{'protein':<10}{'flank':>6}{'published':>11}{'drop':>11}{'resample':>11}{'Δ resample':>12}"
    print(hdr + "\n" + "-" * len(hdr))
    for x in per_protein:
        p_, d_, r_ = (x["arms"][k]["fspe_ratio"] for k in
                      ("published", "drop_flanking", "resample_excluded"))
        dd = f"{r_ - p_:+.4f}" if (p_ and r_) else "—"
        print(f"{x['uniprot_id']:<10}{len(x['flanking_in_published_sample']):>6}"
              f"{p_:>11.4f}{d_:>11.4f}{r_:>11.4f}{dd:>12}")

    print()
    hdr2 = f"{'arm':<20}{'n':>4}{'below 1':>9}{'sign p':>10}{'perm p':>9}{'mean log':>10}"
    print(hdr2 + "\n" + "-" * len(hdr2))
    for name, s in arms.items():
        if s:
            print(f"{name:<20}{s['n_proteins']:>4}{s['n_ratio_below_1']:>9}"
                  f"{s['sign_test_one_sided_p']:>10.4f}{s['permutation_p']:>9.5f}"
                  f"{s['mean_log_ratio']:>10.4f}")
    print(f"\nwrote {OUT}")
    print("\nNote: these are the FULL 15-protein sets. The published 12/14 headline applies")
    print("src/21's SEB exclusion on top, so run src/21 against each arm before comparing to it.")
    return 2 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
