#!/usr/bin/env python3
"""
06_proteinmpnn_redesign.py — Evaluate ProteinMPNN's functional site recovery
                             on toxin backbone structures.

Given toxin PDB structures (backbone only), run ProteinMPNN to design new
sequences and measure whether functional (catalytic/binding) residues are
recovered at a higher rate than non-functional residues.

This computes the Functional Specificity Index (FSI).

Requires:
  - ProteinMPNN cloned at ../ProteinMPNN/ (or set --proteinmpnn_dir)
  - PDB files in data/structures/
  - Functional site annotations in data/annotations/functional_sites.json

Usage:
    python src/06_proteinmpnn_redesign.py [--proteinmpnn_dir /path/to/ProteinMPNN]
                                          [--num_seqs 100]
                                          [--temperature 0.1]
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    FIGURES_DIR,
    RESULTS_DIR,
    STRUCT_DIR,
    compute_fsi,
    compute_sequence_identity,
    compute_site_recovery,
    load_functional_sites,
    print_header,
)

# ============================================================================
# ProteinMPNN wrapper
# ============================================================================


def run_proteinmpnn(
    pdb_path: str,
    output_dir: str,
    proteinmpnn_dir: str,
    chain_id: str = "A",
    num_seqs: int = 100,
    temperature: float = 0.1,
) -> list:
    """Run ProteinMPNN on a PDB file and return designed sequences.

    Args:
        pdb_path: Path to input PDB file
        output_dir: Directory for ProteinMPNN output
        proteinmpnn_dir: Path to cloned ProteinMPNN repository
        chain_id: Chain to design
        num_seqs: Number of sequences to generate
        temperature: Sampling temperature

    Returns:
        List of designed sequences (strings)
    """
    os.makedirs(output_dir, exist_ok=True)
    pdb_name = Path(pdb_path).stem

    # Run ProteinMPNN (single PDB workflow)
    cmd = [
        sys.executable,
        os.path.join(proteinmpnn_dir, "protein_mpnn_run.py"),
        "--pdb_path", pdb_path,
        "--pdb_path_chains", chain_id,
        "--out_folder", output_dir,
        "--num_seq_per_target", str(num_seqs),
        "--sampling_temp", str(temperature),
        "--batch_size", "1",
    ]

    print(f"  Running ProteinMPNN: {' '.join(cmd[-8:])}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print("  ERROR: ProteinMPNN failed")
        print(f"  stderr: {result.stderr[:500]}")
        return []

    # Parse output FASTA
    output_fasta = os.path.join(output_dir, "seqs", f"{pdb_name}.fa")
    if not os.path.exists(output_fasta):
        # Try alternative output paths
        seqs_dir = os.path.join(output_dir, "seqs")
        if os.path.isdir(seqs_dir):
            fastas = [f for f in os.listdir(seqs_dir) if f.endswith(".fa")]
            if fastas:
                output_fasta = os.path.join(seqs_dir, fastas[0])
            else:
                print(f"  No output FASTA found in {seqs_dir}")
                return []

    sequences = []
    current_seq = []
    with open(output_fasta) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_seq:
                    sequences.append("".join(current_seq))
                current_seq = []
            else:
                current_seq.append(line)
    if current_seq:
        sequences.append("".join(current_seq))

    # First sequence is usually the wild-type; skip it
    if len(sequences) > 1:
        return sequences[1:]
    return sequences


def extract_wildtype_sequence(pdb_path: str, chain_id: str = "A") -> tuple:
    """Extract wild-type sequence from PDB file.

    Simple extraction from ATOM records (CA atoms).

    Returns:
        (sequence, pdb_resnums): sequence string and list of PDB residue numbers
        in order, so that pdb_resnums[i] is the PDB residue number for
        sequence position i.
    """
    aa_map = {
        "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
        "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
        "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
        "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    }

    residues = {}
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                chain = line[21].strip()
                if chain == chain_id:
                    resnum = int(line[22:26].strip())
                    resname = line[17:20].strip()
                    if resname in aa_map:
                        residues[resnum] = aa_map[resname]

    if not residues:
        return "", []

    # Build sequence from sorted residue numbers
    sorted_nums = sorted(residues.keys())
    sequence = "".join(residues[n] for n in sorted_nums)
    return sequence, sorted_nums


_AA3_TO_1 = {
    "Ala": "A", "Cys": "C", "Asp": "D", "Glu": "E", "Phe": "F", "Gly": "G",
    "His": "H", "Ile": "I", "Lys": "K", "Leu": "L", "Met": "M", "Asn": "N",
    "Pro": "P", "Gln": "Q", "Arg": "R", "Ser": "S", "Thr": "T", "Val": "V",
    "Trp": "W", "Tyr": "Y",
}


def expected_aa_from_annotation(annotation: str):
    """Parse the expected one-letter amino acid from a residue annotation.

    Annotations conventionally start with a 3-letter+number token, e.g.
    "His223 — zinc-binding ...". Returns the one-letter code, or None when
    the annotation does not begin with a residue identity (e.g. region notes).
    """
    if not annotation:
        return None
    m = re.match(r"\s*([A-Z][a-z]{2})\d", annotation)
    return _AA3_TO_1.get(m.group(1)) if m else None


def map_uniprot_to_pdb_positions(
    uniprot_positions: list,
    pdb_resnums: list,
    pdb_seq: str = None,
    expected_aas: list = None,
) -> list:
    """Map UniProt 1-indexed positions to 0-indexed sequence positions in PDB.

    UniProt numbering and PDB numbering may differ. This function maps
    UniProt residue numbers to the 0-indexed positions in the extracted
    PDB sequence by matching against PDB residue numbers.

    When ``pdb_seq`` and ``expected_aas`` are supplied, the amino acid found
    at each mapped position is checked against the annotated catalytic-residue
    identity. A residue number can be present in a structure yet point to a
    different amino acid (numbering offset / renumbered construct); such a
    silent mismap previously produced FSI computed on the wrong residues.
    Mismatches are now reported loudly.

    Args:
        uniprot_positions: 1-indexed residue positions to map.
        pdb_resnums: PDB residue numbers (from extract_wildtype_sequence),
            ordered so pdb_resnums[i] is the number for sequence position i.
        pdb_seq: optional PDB chain sequence aligned to pdb_resnums.
        expected_aas: optional list parallel to uniprot_positions; each entry
            is the expected one-letter AA, or None to skip the check.

    Returns:
        List of 0-indexed positions in the PDB-extracted sequence.
        Positions that cannot be mapped are dropped (with a warning).
    """
    # Build PDB resnum -> 0-indexed position mapping
    pdb_to_idx = {resnum: i for i, resnum in enumerate(pdb_resnums)}

    mapped = []
    n_mismatch = 0
    for i, upos in enumerate(uniprot_positions):
        # Try direct match (PDB numbering often matches UniProt for well-resolved structures)
        if upos in pdb_to_idx:
            idx = pdb_to_idx[upos]
            mapped.append(idx)
            # Amino-acid identity check — catch silent mismaps
            if pdb_seq is not None and expected_aas is not None and i < len(expected_aas):
                exp = expected_aas[i]
                got = pdb_seq[idx] if idx < len(pdb_seq) else "?"
                if exp and got != exp:
                    n_mismatch += 1
                    print(f"    WARNING: RESIDUE MISMATCH at PDB position {upos} — "
                          f"structure has {got}, annotation expects {exp}. "
                          f"FSI for this site is computed on the WRONG residue.")
        else:
            # Log but don't fail — some positions may be outside the resolved structure
            print(f"    WARNING: UniProt position {upos} not found in PDB residue numbers")

    if n_mismatch:
        print(f"    WARNING: {n_mismatch}/{len(uniprot_positions)} functional "
              f"residues are mismapped — FSI for this structure is unreliable.")

    return mapped


# ============================================================================
# FSI analysis
# ============================================================================


def analyze_fsi_for_structure(
    pdb_id: str,
    pdb_info: dict,
    designed_sequences: list,
    wildtype_seq: str,
    functional_positions_0idx: list,
) -> dict:
    """Compute FSI metrics for a single toxin structure.

    Args:
        pdb_id: PDB identifier
        pdb_info: Metadata dict
        designed_sequences: List of ProteinMPNN-designed sequences
        wildtype_seq: Wild-type sequence
        functional_positions_0idx: 0-indexed positions mapped to PDB sequence

    Returns:
        dict with FSI metrics
    """
    # Convert to 1-indexed for compute_site_recovery (which expects 1-indexed)
    functional_residues_1idx = [p + 1 for p in functional_positions_0idx]

    func_recoveries = []
    overall_recoveries = []
    fsi_values = []

    for seq in designed_sequences:
        func_rec, overall_rec = compute_site_recovery(
            seq, wildtype_seq, functional_residues_1idx
        )
        fsi = compute_fsi(func_rec, overall_rec)

        func_recoveries.append(func_rec)
        overall_recoveries.append(overall_rec)
        fsi_values.append(fsi)

    # Sequence divergence from wildtype
    wt_identities = [compute_sequence_identity(s, wildtype_seq) for s in designed_sequences]

    return {
        "pdb_id": pdb_id,
        "description": pdb_info.get("description", ""),
        "uniprot": pdb_info.get("uniprot", ""),
        "n_designed_sequences": len(designed_sequences),
        "wildtype_length": len(wildtype_seq),
        "n_functional_sites": len(functional_residues_1idx),
        "functional_recovery": {
            "mean": float(np.mean(func_recoveries)),
            "std": float(np.std(func_recoveries)),
            "median": float(np.median(func_recoveries)),
        },
        "overall_recovery": {
            "mean": float(np.mean(overall_recoveries)),
            "std": float(np.std(overall_recoveries)),
            "median": float(np.median(overall_recoveries)),
        },
        "fsi": {
            "mean": float(np.mean(fsi_values)),
            "std": float(np.std(fsi_values)),
            "median": float(np.median(fsi_values)),
            "fraction_above_1": float(np.mean(np.array(fsi_values) > 1.0)),
            "per_sequence_values": [float(v) for v in fsi_values],
        },
        "sequence_divergence": {
            "mean_wt_identity": float(np.mean(wt_identities)),
            "std_wt_identity": float(np.std(wt_identities)),
            "min_wt_identity": float(np.min(wt_identities)),
        },
    }


# ============================================================================
# Visualization
# ============================================================================


def plot_fsi_results(all_results: list):
    """Plot FSI results across all evaluated structures."""
    import matplotlib.pyplot as plt

    pdb_ids = [r["pdb_id"] for r in all_results]
    fsi_means = [r["fsi"]["mean"] for r in all_results]
    fsi_stds = [r["fsi"]["std"] for r in all_results]
    func_rec = [r["functional_recovery"]["mean"] for r in all_results]
    overall_rec = [r["overall_recovery"]["mean"] for r in all_results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: FSI values
    ax = axes[0]
    x = np.arange(len(pdb_ids))
    ax.bar(x, fsi_means, yerr=fsi_stds, color="#6366f1", alpha=0.8, capsize=3)
    ax.axhline(1.0, color="black", linestyle="--", lw=1.5, label="FSI = 1.0 (no specificity)")
    ax.set_ylabel("Functional Specificity Index (FSI)", fontsize=12)
    ax.set_xlabel("Toxin Structure", fontsize=12)
    ax.set_title("FSI: Functional Recovery / Overall Recovery", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(pdb_ids, rotation=45, ha="right")
    ax.legend(fontsize=10)

    # Panel B: Recovery rates
    ax = axes[1]
    width = 0.35
    ax.bar(x - width / 2, func_rec, width, label="Functional sites", color="#ef4444", alpha=0.8)
    ax.bar(x + width / 2, overall_rec, width, label="Overall", color="#94a3b8", alpha=0.8)
    ax.set_ylabel("Sequence Recovery Rate", fontsize=12)
    ax.set_xlabel("Toxin Structure", fontsize=12)
    ax.set_title("Recovery at Functional vs. All Positions", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(pdb_ids, rotation=45, ha="right")
    ax.legend(fontsize=10)

    plt.tight_layout()
    path = FIGURES_DIR / "fsi_results.png"
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="ProteinMPNN FSI evaluation")
    parser.add_argument(
        "--proteinmpnn_dir",
        default=str(Path(__file__).parent.parent / "ProteinMPNN"),
        help="Path to cloned ProteinMPNN repository",
    )
    parser.add_argument("--num_seqs", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()

    print_header("ProteinMPNN Functional Specificity Evaluation")

    # Check ProteinMPNN installation
    mpnn_script = Path(args.proteinmpnn_dir) / "protein_mpnn_run.py"
    if not mpnn_script.exists():
        print(f"ERROR: ProteinMPNN not found at {args.proteinmpnn_dir}")
        print("Clone it: git clone https://github.com/dauparas/ProteinMPNN.git")
        sys.exit(1)

    # Load annotations
    func_sites = load_functional_sites()

    # PDB structures to evaluate
    pdb_structures = {}
    for uniprot_id, info in func_sites.items():
        if uniprot_id.startswith("_"):
            continue
        if info.get("exclude_from_fsi"):
            print(f"  Skipping {uniprot_id} ({info.get('name', '?')}): "
                  f"exclude_from_fsi is set")
            continue
        pdb_id = info.get("pdb_id")
        if pdb_id:
            pdb_path = STRUCT_DIR / f"{pdb_id}.pdb"
            if pdb_path.exists():
                fs = info["functional_sites"]
                cat_res = fs["catalytic_residues"]
                func_res = info.get("pdb_residues", cat_res)
                ann = fs.get("residue_annotations", {})
                # Expected amino acids, aligned by index to func_res. Valid only
                # when catalytic_residues and func_res are parallel (same length);
                # otherwise the identity check is skipped for this structure.
                if len(cat_res) == len(func_res):
                    expected_aas = [
                        expected_aa_from_annotation(ann.get(str(c))) for c in cat_res
                    ]
                else:
                    expected_aas = None
                pdb_structures[pdb_id] = {
                    "path": str(pdb_path),
                    "chain": info.get("pdb_chain", "A"),
                    "uniprot": uniprot_id,
                    "description": info["name"],
                    "functional_residues": func_res,
                    "use_pdb_numbering": info.get("use_pdb_numbering", False),
                    "expected_aas": expected_aas,
                }

    if not pdb_structures:
        print("ERROR: No PDB structures found. Run 01_collect_data.py first.")
        sys.exit(1)

    print(f"Evaluating {len(pdb_structures)} structures:")
    for pdb_id, info in pdb_structures.items():
        print(f"  {pdb_id}: {info['description']}")

    # Run ProteinMPNN and compute FSI for each structure
    all_results = []
    output_base = RESULTS_DIR / "proteinmpnn_output"
    output_base.mkdir(parents=True, exist_ok=True)

    for pdb_id, pdb_info in pdb_structures.items():
        print(f"\n{'='*50}")
        print(f"Structure: {pdb_id} — {pdb_info['description']}")
        print(f"{'='*50}")

        # Extract wild-type sequence and PDB residue numbers
        wt_seq, pdb_resnums = extract_wildtype_sequence(pdb_info["path"], pdb_info["chain"])
        if not wt_seq:
            print(f"  ERROR: Could not extract sequence from {pdb_id}")
            continue
        print(f"  Wild-type length: {len(wt_seq)}")
        print(f"  PDB residue range: {pdb_resnums[0]}-{pdb_resnums[-1]}")

        # Map functional residue positions to 0-indexed PDB sequence positions.
        # pdb_seq + expected_aas enable the amino-acid identity check.
        func_positions_0idx = map_uniprot_to_pdb_positions(
            pdb_info["functional_residues"],
            pdb_resnums,
            pdb_seq=wt_seq,
            expected_aas=pdb_info.get("expected_aas"),
        )
        if pdb_info.get("use_pdb_numbering", False):
            print(f"  Using PDB numbering: {len(func_positions_0idx)}/{len(pdb_info['functional_residues'])} sites resolved")
        print(f"  Mapped {len(func_positions_0idx)}/{len(pdb_info['functional_residues'])} functional sites to PDB")
        if not func_positions_0idx:
            print("  ERROR: No functional sites mapped, skipping")
            continue

        # Run ProteinMPNN
        output_dir = str(output_base / pdb_id)
        designed_seqs = run_proteinmpnn(
            pdb_info["path"],
            output_dir,
            args.proteinmpnn_dir,
            chain_id=pdb_info["chain"],
            num_seqs=args.num_seqs,
            temperature=args.temperature,
        )

        if not designed_seqs:
            print("  No designed sequences obtained, skipping")
            continue

        print(f"  Generated {len(designed_seqs)} designed sequences")

        # Compute FSI
        result = analyze_fsi_for_structure(
            pdb_id, pdb_info, designed_seqs, wt_seq,
            func_positions_0idx,
        )
        all_results.append(result)

        print(f"  Functional recovery: {result['functional_recovery']['mean']:.3f} ± {result['functional_recovery']['std']:.3f}")
        print(f"  Overall recovery:    {result['overall_recovery']['mean']:.3f} ± {result['overall_recovery']['std']:.3f}")
        print(f"  FSI:                 {result['fsi']['mean']:.3f} ± {result['fsi']['std']:.3f}")
        print(f"  Fraction FSI > 1:    {result['fsi']['fraction_above_1']:.3f}")

    # Visualize
    if all_results:
        print("\n--- Generating figures ---")
        plot_fsi_results(all_results)

    # Save results
    results_path = RESULTS_DIR / "fsi_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Summary
    print_header("FSI Summary")
    if all_results:
        fsi_means = [r["fsi"]["mean"] for r in all_results]
        overall_fsi = np.mean(fsi_means)
        print(f"  Overall mean FSI: {overall_fsi:.3f}")

        for r in all_results:
            print(f"  {r['pdb_id']}: FSI = {r['fsi']['mean']:.3f}, "
                  f"Func recovery = {r['functional_recovery']['mean']:.3f}, "
                  f"Overall = {r['overall_recovery']['mean']:.3f}")

        if overall_fsi > 1.0:
            print("\n  FSI > 1.0: ProteinMPNN specifically recovers functional residues")
            print("  beyond what structural similarity alone would predict.")
            print("  → The backbone structure of toxins encodes functional danger.")
        else:
            print("\n  FSI <= 1.0: ProteinMPNN does not specifically recover function.")
            print("  → Structural backbone alone may not encode dual-use specificity.")

    print("\nNext step: python src/07_fsi_analysis.py")


if __name__ == "__main__":
    main()
