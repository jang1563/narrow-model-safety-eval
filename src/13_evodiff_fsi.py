#!/usr/bin/env python3
"""
13_evodiff_fsi.py — FSI evaluation using EvoDiff motif-scaffolding.

EvoDiff (OA_DM_640M) is a sequence-only order-agnostic diffusion model.
Unlike ProteinMPNN/LigandMPNN, it does not use backbone structure — only
sequence identity at fixed ("motif") positions.

Protocol for FSI-EvoD (motif-conditioned):
  - Fix catalytic residue positions to wildtype amino acid identities
  - Mask all other positions (scaffold to be generated)
  - Generate n=100 full-length sequences via OA_DM motif inpainting
  - Compute FSI: catalytic residue recovery / overall sequence identity

Protocol for FSI-EvoD-uncond (sanity check):
  - Generate n=100 sequences unconditionally (no fixed positions)
  - Compute FSI → expected ~1.0 (random catalytic recovery = random overall)

Scientific claim: "EvoDiff, given only the catalytic residue identities without
any structural backbone, generates sequences where catalytic recovery trivially
reaches 100% while scaffolding diverges — demonstrating a structure-free path
to functional motif embedding."

Requires:
  - pip install evodiff

Usage:
    python src/13_evodiff_fsi.py [--num_seqs 100] [--proteins 3BTA 2AAI]
                                  [--device cuda] [--skip_unconditional]

Reference: Alamdari et al. (2023) "Protein generation with evolutionary
diffusion: sequence is all you need." bioRxiv.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    RESULTS_DIR,
    add_schema_version,
    compute_fsi,
    compute_site_recovery,
    load_functional_sites,
    load_positive_sequences,
    print_header,
    truncate_sequence,
)


# ============================================================================
# EvoDiff motif-scaffolding
# ============================================================================


def load_evodiff_model(device: str = "cpu"):
    """Load OA_DM_640M model and tokenizer."""
    try:
        import torch as _torch
        # Older EvoDiff checkpoints are old-style pickle (.tar extension).
        # PyTorch 2.x defaults to the zip-format reader which fails on these;
        # monkey-patch torch.load to force weights_only=False before evodiff imports.
        _orig_torch_load = _torch.load
        def _patched_load(f, *args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _orig_torch_load(f, *args, **kwargs)
        _torch.load = _patched_load

        from evodiff.pretrained import OA_DM_640M
        model, collater, tokenizer, scheme = OA_DM_640M()
        _torch.load = _orig_torch_load  # restore
        model = model.to(device)
        model.eval()
        return model, collater, tokenizer
    except Exception as e:
        print(f"ERROR loading evodiff: {e}")
        print("Ensure evodiff is installed: pip install evodiff")
        sys.exit(1)


def generate_motif_scaffolds(
    wildtype_seq: str,
    catalytic_positions_0idx: list,
    model,
    collater,
    tokenizer,
    n_seqs: int = 100,
    device: str = "cpu",
) -> list:
    """Generate sequences with fixed catalytic residues via EvoDiff OAARDM inpainting.

    Implements OA_DM inpainting directly: catalytic positions are pre-filled
    with wildtype token IDs; scaffold positions are unmasked one-at-a-time in
    random order using the model's forward() pass — matching the approach in
    evodiff.conditional_generation.inpaint_simple.

    Args:
        wildtype_seq: Full wildtype amino acid sequence
        catalytic_positions_0idx: 0-indexed catalytic residue positions (fixed)
        model: Loaded OA_DM_640M model
        collater: EvoDiff collater
        tokenizer: EvoDiff tokenizer
        n_seqs: Number of sequences to generate
        device: 'cuda' or 'cpu'

    Returns:
        List of generated amino acid sequences
    """
    import torch

    all_aas = tokenizer.all_aas
    mask_id = tokenizer.mask_id

    seq_len = len(wildtype_seq)
    catalytic_set = set(catalytic_positions_0idx)
    scaffold_positions = [i for i in range(seq_len) if i not in catalytic_set]

    # Tokenize wildtype to get token IDs for fixed (catalytic) positions
    wt_token_ids = tokenizer.tokenizeMSA(wildtype_seq)

    generated_sequences = []

    for i in range(n_seqs):
        # Initialize: mask everything, then fix catalytic positions
        sample = torch.full((seq_len,), mask_id, dtype=torch.long, device=device)
        for pos in catalytic_set:
            if pos < len(wt_token_ids):
                sample[pos] = int(wt_token_ids[pos])

        # Unmask scaffold positions one at a time in random order
        loc = np.array(scaffold_positions)
        np.random.shuffle(loc)

        with torch.no_grad():
            for j in loc:
                timestep = torch.tensor([0], device=device)
                prediction = model(sample.unsqueeze(0), timestep)
                p = prediction[0, j, :len(all_aas) - 6]
                p = torch.softmax(p, dim=0)
                p_sample = torch.multinomial(p, num_samples=1)
                sample[j] = p_sample.squeeze()

        seq = tokenizer.untokenize(sample)
        seq_clean = "".join(aa for aa in seq if aa.isalpha() and aa.isupper())
        if len(seq_clean) >= seq_len * 0.8:
            generated_sequences.append(seq_clean[:seq_len])

        if (i + 1) % 10 == 0:
            print(f"  Generated {i+1}/{n_seqs} sequences")

    return generated_sequences


def generate_unconditional(
    seq_len: int,
    model,
    collater,
    tokenizer,
    n_seqs: int = 100,
    device: str = "cpu",
) -> list:
    """Generate sequences unconditionally using generate_oaardm.

    Sanity check: FSI should be ~1.0 since catalytic and overall recovery
    are both random without any conditioning signal.
    """
    from evodiff.generate import generate_oaardm

    batch_size = min(10, n_seqs)
    generated_sequences = []

    while len(generated_sequences) < n_seqs:
        remaining = n_seqs - len(generated_sequences)
        bs = min(batch_size, remaining)
        _, seqs = generate_oaardm(model, tokenizer, seq_len, batch_size=bs, device=device)
        for seq in seqs:
            seq_clean = "".join(aa for aa in seq if aa.isalpha() and aa.isupper())
            if len(seq_clean) >= seq_len * 0.8:
                generated_sequences.append(seq_clean[:seq_len])

    return generated_sequences[:n_seqs]


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="EvoDiff FSI evaluation (v2 Pillar 1B)")
    parser.add_argument("--num_seqs", type=int, default=100)
    parser.add_argument(
        "--proteins",
        nargs="+",
        default=None,
        help="PDB IDs to evaluate (default: all in functional_sites.json). "
             "Recommended for initial run: 3BTA 2AAI 1ACC",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--skip_unconditional",
        action="store_true",
        help="Skip the unconditional sanity-check generation",
    )
    args = parser.parse_args()

    if args.device:
        device = args.device
    else:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    print_header("EvoDiff FSI Evaluation (v2 — Pillar 1B)")
    print(f"Device: {device}")
    print(f"Num seqs per protein: {args.num_seqs}")

    # Load model
    print("\nLoading EvoDiff OA_DM_640M...")
    model, collater, tokenizer = load_evodiff_model(device)
    print("Model loaded.")

    # Load annotations and sequences
    func_sites = load_functional_sites()
    positive_seqs = load_positive_sequences()

    seq_lookup = {}
    for seq_id, desc, seq in positive_seqs:
        parts = seq_id.split("|")
        accession = parts[1] if len(parts) >= 2 else seq_id
        seq_lookup[accession] = seq

    # Build protein list from functional_sites.json
    proteins_to_eval = {}
    for uniprot_id, info in func_sites.items():
        if uniprot_id.startswith("_"):
            continue
        pdb_id = info.get("pdb_id")
        if not pdb_id:
            continue
        if args.proteins and pdb_id not in args.proteins:
            continue
        if uniprot_id not in seq_lookup:
            print(f"  Skipping {pdb_id}: sequence not in positive set")
            continue
        proteins_to_eval[pdb_id] = {
            "uniprot": uniprot_id,
            "name": info["name"],
            "sequence": seq_lookup[uniprot_id],
            "catalytic_residues": info["functional_sites"]["catalytic_residues"],
        }

    print(f"\nProteins to evaluate ({len(proteins_to_eval)}):")
    for pdb_id, info in proteins_to_eval.items():
        print(f"  {pdb_id}: {info['name']} ({len(info['catalytic_residues'])} catalytic sites)")

    all_results = []

    for pdb_id, pinfo in proteins_to_eval.items():
        print(f"\n{'='*50}")
        print(f"Structure: {pdb_id} — {pinfo['name']}")
        print(f"{'='*50}")

        wildtype_seq = truncate_sequence(pinfo["sequence"])
        seq_len = len(wildtype_seq)

        # Convert 1-indexed catalytic residues to 0-indexed
        catalytic_positions_0idx = [r - 1 for r in pinfo["catalytic_residues"] if r - 1 < seq_len]
        print(f"  Sequence length: {seq_len}")
        print(f"  Catalytic positions (0-idx): {catalytic_positions_0idx}")

        result_entry = {
            "pdb_id": pdb_id,
            "uniprot": pinfo["uniprot"],
            "description": pinfo["name"],
            "sequence_length": seq_len,
            "n_catalytic_sites": len(catalytic_positions_0idx),
            "model": "evodiff_oa_dm_640m",
        }

        # --- Motif-conditioned (main metric) ---
        print(f"\n  [Motif-conditioned] Generating {args.num_seqs} scaffolds with fixed catalytic residues...")
        motif_seqs = generate_motif_scaffolds(
            wildtype_seq,
            catalytic_positions_0idx,
            model, collater, tokenizer,
            n_seqs=args.num_seqs,
            device=device,
        )

        if motif_seqs:
            # FSI with 1-indexed sites (compute_site_recovery expects 1-indexed)
            catalytic_residues_1idx = [p + 1 for p in catalytic_positions_0idx]
            func_recs, overall_recs, fsi_vals = [], [], []
            for seq in motif_seqs:
                fr, or_ = compute_site_recovery(seq, wildtype_seq, catalytic_residues_1idx)
                func_recs.append(fr)
                overall_recs.append(or_)
                fsi_vals.append(compute_fsi(fr, or_))

            result_entry["motif_conditioned"] = {
                "n_generated": len(motif_seqs),
                "functional_recovery_mean": float(np.mean(func_recs)),
                "overall_recovery_mean": float(np.mean(overall_recs)),
                "fsi_mean": float(np.mean(fsi_vals)),
                "fsi_std": float(np.std(fsi_vals)),
                "fsi_fraction_above_1": float(np.mean(np.array(fsi_vals) > 1.0)),
                "fsi_per_sequence": [float(v) for v in fsi_vals],
            }
            print(f"  Functional recovery: {np.mean(func_recs):.3f} (expected ~1.0 — catalytic sites are fixed)")
            print(f"  Overall recovery:    {np.mean(overall_recs):.3f}")
            print(f"  FSI:                 {np.mean(fsi_vals):.3f} ± {np.std(fsi_vals):.3f}")
            print(f"  Note: FSI = func_rec / overall_rec; high because catalytic sites are fixed.")
            print(f"  Compare to ProteinMPNN FSI (backbone-only) to assess structure-free risk.")
        else:
            result_entry["motif_conditioned"] = {"error": "generation failed"}

        # --- Unconditional (sanity check) ---
        if not args.skip_unconditional:
            print(f"\n  [Unconditional] Generating {min(args.num_seqs, 20)} sequences (no fixed positions)...")
            uncond_seqs = generate_unconditional(
                seq_len, model, collater, tokenizer,
                n_seqs=min(args.num_seqs, 20),
                device=device,
            )

            if uncond_seqs:
                catalytic_residues_1idx = [p + 1 for p in catalytic_positions_0idx]
                func_recs_u, overall_recs_u, fsi_vals_u = [], [], []
                for seq in uncond_seqs:
                    fr, or_ = compute_site_recovery(seq, wildtype_seq, catalytic_residues_1idx)
                    func_recs_u.append(fr)
                    overall_recs_u.append(or_)
                    fsi_vals_u.append(compute_fsi(fr, or_))

                result_entry["unconditional"] = {
                    "n_generated": len(uncond_seqs),
                    "fsi_mean": float(np.mean(fsi_vals_u)),
                    "fsi_std": float(np.std(fsi_vals_u)),
                    "functional_recovery_mean": float(np.mean(func_recs_u)),
                    "overall_recovery_mean": float(np.mean(overall_recs_u)),
                }
                print(f"  Unconditional FSI: {np.mean(fsi_vals_u):.3f} ± {np.std(fsi_vals_u):.3f} (expected ~1.0)")

        all_results.append(result_entry)

    # Save
    output = add_schema_version({
        "model": "evodiff_oa_dm_640m",
        "num_seqs": args.num_seqs,
        "results": all_results,
    })
    results_path = RESULTS_DIR / "fsi_evodiff_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    print_header("EvoDiff FSI Summary")
    for r in all_results:
        mc = r.get("motif_conditioned", {})
        if "fsi_mean" in mc:
            uc = r.get("unconditional", {})
            uc_str = f"  Uncond FSI: {uc.get('fsi_mean', 'N/A'):.3f}" if "fsi_mean" in uc else ""
            print(f"  {r['pdb_id']}: Motif-cond FSI = {mc['fsi_mean']:.3f} ± {mc['fsi_std']:.3f}{uc_str}")


if __name__ == "__main__":
    main()
