#!/usr/bin/env python3
"""
12_ligandmpnn_fsi.py — FSI evaluation using LigandMPNN.

LigandMPNN extends ProteinMPNN with metal-ion / small-molecule context from
HETATM records. For metal-catalytic toxins (e.g. BoNT-A zinc protease), the
model receives the zinc coordinates alongside backbone — testing whether explicit
metal context further encodes dangerous function beyond backbone alone.

Hypothesis: FSI-LM >> FSI-PM for metal-binding active sites (e.g. BoNT-A),
because the HEXXH zinc motif is constrained by both backbone geometry AND
the zinc coordination sphere.

Protocol:
  - model_type "ligand_mpnn" for all proteins (automatically uses HETATM metal
    context if present; behaves like ProteinMPNN when no ligand is in PDB)
  - n=100 designs per protein at T=0.1

Requires:
  - LigandMPNN cloned at ../LigandMPNN/ (or --ligandmpnn_dir)
  - PDB files in data/structures/ (with HETATM records preserved)
  - Functional site annotations in data/annotations/functional_sites.json

Usage:
    python src/12_ligandmpnn_fsi.py [--ligandmpnn_dir /path/to/LigandMPNN]
                                    [--num_seqs 100] [--temperature 0.1]
                                    [--model_type ligand_mpnn]

Reference: Dauparas et al. (2023) "Atomic context-conditioned protein sequence
design using LigandMPNN." bioRxiv. https://doi.org/10.1101/2023.12.22.573103
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    RESULTS_DIR,
    STRUCT_DIR,
    add_schema_version,
    load_functional_sites,
    print_header,
)

# Reuse PDB parsing from 06_proteinmpnn_redesign
from importlib.util import spec_from_file_location, module_from_spec

_pm_path = Path(__file__).parent / "06_proteinmpnn_redesign.py"
_spec = spec_from_file_location("proteinmpnn_module", _pm_path)
_pm = module_from_spec(_spec)
_spec.loader.exec_module(_pm)

extract_wildtype_sequence = _pm.extract_wildtype_sequence
map_uniprot_to_pdb_positions = _pm.map_uniprot_to_pdb_positions
analyze_fsi_for_structure = _pm.analyze_fsi_for_structure


# ============================================================================
# LigandMPNN wrapper
# ============================================================================


def run_ligandmpnn(
    pdb_path: str,
    output_dir: str,
    ligandmpnn_dir: str,
    chain_id: str = "A",
    num_seqs: int = 100,
    temperature: float = 0.1,
    model_type: str = "ligand_mpnn",
) -> list:
    """Run LigandMPNN on a PDB file and return designed sequences.

    LigandMPNN's run.py script is the equivalent of ProteinMPNN's
    protein_mpnn_run.py. With model_type='ligand_mpnn', it parses HETATM
    records for metal / small-molecule context automatically.

    Args:
        pdb_path: Path to input PDB file (should include HETATM records)
        output_dir: Directory for LigandMPNN output
        ligandmpnn_dir: Path to cloned LigandMPNN repository
        chain_id: Chain to design
        num_seqs: Number of sequences to generate
        temperature: Sampling temperature
        model_type: 'ligand_mpnn' (metal-aware) or 'protein_mpnn' (backbone-only)

    Returns:
        List of designed sequences (strings)
    """
    os.makedirs(output_dir, exist_ok=True)
    pdb_name = Path(pdb_path).stem

    checkpoint_path = os.path.join(ligandmpnn_dir, "model_params", "ligandmpnn_v_32_010_25.pt")
    cmd = [
        sys.executable,
        os.path.join(ligandmpnn_dir, "run.py"),
        "--pdb_path", pdb_path,
        "--out_folder", output_dir,
        "--number_of_batches", str(num_seqs),
        "--batch_size", "1",
        "--temperature", str(temperature),
        "--model_type", model_type,
        "--chains_to_design", chain_id,
        "--pack_side_chains", "0",
        "--checkpoint_ligand_mpnn", checkpoint_path,
    ]

    print(f"  Running LigandMPNN ({model_type}): {pdb_name}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)

    if result.returncode != 0:
        print(f"  ERROR: LigandMPNN failed (returncode={result.returncode})")
        print(f"  stderr: {result.stderr[-2000:]}")
        print(f"  stdout: {result.stdout[-1000:]}")
        return []

    # Output FASTA — same directory structure as ProteinMPNN
    output_fasta = os.path.join(output_dir, "seqs", f"{pdb_name}.fa")
    if not os.path.exists(output_fasta):
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

    # First sequence is the wild-type reference; skip it
    return sequences[1:] if len(sequences) > 1 else sequences


def detect_metal_in_pdb(pdb_path: str) -> bool:
    """Check whether the PDB contains metal ion HETATM records."""
    metal_ions = {"ZN", "MG", "CA", "FE", "MN", "CU", "CO", "NI", "MO", "W"}
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("HETATM"):
                resname = line[17:20].strip().upper()
                if resname in metal_ions:
                    return True
    return False


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="LigandMPNN FSI evaluation (v2 Pillar 1A)")
    parser.add_argument(
        "--ligandmpnn_dir",
        default=str(Path(__file__).parent.parent / "LigandMPNN"),
        help="Path to cloned LigandMPNN repository",
    )
    parser.add_argument("--num_seqs", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--model_type",
        default="ligand_mpnn",
        choices=["ligand_mpnn", "protein_mpnn"],
        help="ligand_mpnn uses metal/ligand HETATM context; protein_mpnn is backbone-only",
    )
    args = parser.parse_args()

    print_header("LigandMPNN FSI Evaluation (v2 — Pillar 1A)")

    lmpnn_script = Path(args.ligandmpnn_dir) / "run.py"
    if not lmpnn_script.exists():
        print(f"ERROR: LigandMPNN not found at {args.ligandmpnn_dir}")
        print("Clone it: git clone https://github.com/dauparas/LigandMPNN.git")
        sys.exit(1)

    func_sites = load_functional_sites()

    pdb_structures = {}
    for uniprot_id, info in func_sites.items():
        if uniprot_id.startswith("_"):
            continue
        pdb_id = info.get("pdb_id")
        if pdb_id:
            pdb_path = STRUCT_DIR / f"{pdb_id}.pdb"
            if pdb_path.exists():
                fs = info["functional_sites"]
                pdb_structures[pdb_id] = {
                    "path": str(pdb_path),
                    "chain": info.get("pdb_chain", "A"),
                    "uniprot": uniprot_id,
                    "description": info["name"],
                    "functional_residues": info.get("pdb_residues", fs["catalytic_residues"]),
                    "use_pdb_numbering": info.get("use_pdb_numbering", False),
                }

    if not pdb_structures:
        print("ERROR: No PDB structures found. Run 01_collect_data.py first.")
        sys.exit(1)

    # Cache metal detection to avoid double file reads
    metal_cache = {pdb_id: detect_metal_in_pdb(info["path"]) for pdb_id, info in pdb_structures.items()}

    print(f"Evaluating {len(pdb_structures)} structures with LigandMPNN ({args.model_type}):")
    for pdb_id, info in pdb_structures.items():
        metal_str = " [METAL DETECTED]" if metal_cache[pdb_id] else ""
        print(f"  {pdb_id}: {info['description']}{metal_str}")

    all_results = []
    output_base = RESULTS_DIR / "ligandmpnn_output"
    output_base.mkdir(parents=True, exist_ok=True)

    for pdb_id, pdb_info in pdb_structures.items():
        has_metal = metal_cache[pdb_id]
        print(f"\n{'='*50}")
        print(f"Structure: {pdb_id} — {pdb_info['description']}")
        print(f"Metal context: {'YES' if has_metal else 'NO'}")
        print(f"{'='*50}")

        wt_seq, pdb_resnums = extract_wildtype_sequence(pdb_info["path"], pdb_info["chain"])
        if not wt_seq:
            print(f"  ERROR: Could not extract sequence from {pdb_id}")
            continue

        func_positions_0idx = map_uniprot_to_pdb_positions(
            pdb_info["functional_residues"], pdb_resnums
        )
        print(f"  Mapped {len(func_positions_0idx)}/{len(pdb_info['functional_residues'])} functional sites")
        if not func_positions_0idx:
            print(f"  ERROR: No functional sites mapped, skipping")
            continue

        output_dir = str(output_base / pdb_id)
        designed_seqs = run_ligandmpnn(
            pdb_info["path"],
            output_dir,
            args.ligandmpnn_dir,
            chain_id=pdb_info["chain"],
            num_seqs=args.num_seqs,
            temperature=args.temperature,
            model_type=args.model_type,
        )

        if not designed_seqs:
            print(f"  No designed sequences obtained, skipping")
            continue

        print(f"  Generated {len(designed_seqs)} designed sequences")

        result = analyze_fsi_for_structure(
            pdb_id, pdb_info, designed_seqs, wt_seq, func_positions_0idx
        )
        result["model"] = "ligandmpnn"
        result["model_type"] = args.model_type
        result["has_metal_context"] = has_metal
        all_results.append(result)

        print(f"  Functional recovery: {result['functional_recovery']['mean']:.3f} ± {result['functional_recovery']['std']:.3f}")
        print(f"  Overall recovery:    {result['overall_recovery']['mean']:.3f} ± {result['overall_recovery']['std']:.3f}")
        print(f"  FSI:                 {result['fsi']['mean']:.3f} ± {result['fsi']['std']:.3f}")
        print(f"  Fraction FSI > 1:    {result['fsi']['fraction_above_1']:.3f}")

    # Save results
    output = add_schema_version({
        "model": "ligandmpnn",
        "model_type": args.model_type,
        "num_seqs": args.num_seqs,
        "temperature": args.temperature,
        "results": all_results,
    })
    results_path = RESULTS_DIR / "fsi_ligandmpnn_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    print_header("LigandMPNN FSI Summary")
    if all_results:
        for r in all_results:
            metal_str = " [metal]" if r.get("has_metal_context") else ""
            print(f"  {r['pdb_id']}{metal_str}: FSI = {r['fsi']['mean']:.3f} ± {r['fsi']['std']:.3f}")
        print()
        print("Compare with v1 ProteinMPNN FSI values in results/fsi_results.json")
        print("Hypothesis: FSI-LM > FSI-PM for metal-containing structures (e.g. 3BTA BoNT-A)")


if __name__ == "__main__":
    main()
