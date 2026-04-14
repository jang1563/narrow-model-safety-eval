#!/usr/bin/env python3
"""
10_fsi_temperature_sensitivity.py — FSI temperature sensitivity analysis.

Validates that FSI=3.07 for BoNT-A (3BTA) and FSI=1.12 for Ricin (2AAI)
are not artifacts of T=0.1 sampling temperature. Runs ProteinMPNN at
T=0.05, 0.1, 0.2, 0.3 (n=100 sequences each) and reports FSI at each T.

Expected result: FSI decreases with temperature but remains >1.0 for 3BTA
across all tested temperatures (robust metric).

Usage:
    python src/10_fsi_temperature_sensitivity.py \
        --proteinmpnn_dir /path/to/ProteinMPNN \
        --temperatures 0.05,0.1,0.2,0.3 \
        --num_seqs 100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import importlib.util as _ilu
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    FIGURES_DIR,
    RESULTS_DIR,
    compute_fsi,
    compute_site_recovery,
    print_header,
)

# Re-use functions from 06_proteinmpnn_redesign
_spec = _ilu.spec_from_file_location(
    "mpnn_redesign",
    str(Path(__file__).parent / "06_proteinmpnn_redesign.py"),
)
_mpnn_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mpnn_mod)
extract_wildtype_sequence = _mpnn_mod.extract_wildtype_sequence
run_proteinmpnn = _mpnn_mod.run_proteinmpnn

# Proteins to test — the two with significant FSI results
PROTEINS_TO_TEST = ["3BTA", "2AAI"]

DATA_DIR = Path(__file__).parent.parent / "data"
STRUCT_DIR = DATA_DIR / "structures"
ANNOT_DIR = DATA_DIR / "annotations"


def load_functional_sites_for_pdb(pdb_id: str) -> tuple[list[int], str]:
    """Return (functional_residues_1idx, chain_id) for a given PDB ID."""
    with open(ANNOT_DIR / "functional_sites.json") as f:
        func_sites = json.load(f)

    # Find entry with matching pdb_id
    for uid, info in func_sites.items():
        if uid.startswith("_"):
            continue
        if info.get("pdb_id") == pdb_id:
            chain = info.get("pdb_chain", "A")
            # Prefer pdb_residues if use_pdb_numbering is set
            if info.get("use_pdb_numbering") and info.get("pdb_residues"):
                return info["pdb_residues"], chain
            return info["functional_sites"]["catalytic_residues"], chain

    raise ValueError(f"No functional site annotation found for pdb_id={pdb_id}")


def run_temperature_sweep(
    pdb_id: str,
    pdb_path: str,
    chain_id: str,
    functional_residues_1idx: list[int],
    proteinmpnn_dir: str,
    output_base: Path,
    temperatures: list[float],
    num_seqs: int = 100,
) -> dict:
    """Run ProteinMPNN at multiple temperatures and compute FSI at each.

    Args:
        functional_residues_1idx: 1-indexed functional residue positions.
            compute_site_recovery() expects 1-indexed and subtracts 1 internally.
            Do NOT pass 0-indexed positions.

    Returns:
        {temp: {"mean": float, "std": float, "median": float, "fraction_above_1": float}}
    """
    wt_seq, pdb_resnums = extract_wildtype_sequence(pdb_path, chain_id)
    if not wt_seq:
        print(f"  ERROR: Could not extract wildtype sequence for {pdb_id}")
        return {}

    # If functional_residues_1idx are PDB residue numbers (not sequence positions),
    # we need to map them to sequence positions
    pdb_to_seqidx = {resnum: i for i, resnum in enumerate(pdb_resnums)}
    # Convert PDB resnums to sequence indices (0-indexed), then back to 1-indexed for compute_site_recovery
    seq_positions_1idx = []
    for resnum in functional_residues_1idx:
        if resnum in pdb_to_seqidx:
            seq_positions_1idx.append(pdb_to_seqidx[resnum] + 1)  # 1-indexed
        else:
            print(f"    WARNING: PDB residue {resnum} not found in {pdb_id} chain {chain_id}")

    if not seq_positions_1idx:
        print(f"  ERROR: No valid functional residues mapped for {pdb_id}")
        return {}

    print(f"  Wildtype length: {len(wt_seq)}, functional sites mapped: {len(seq_positions_1idx)}")

    results_by_temp = {}
    for temp in temperatures:
        print(f"  Temperature T={temp:.3f}...")
        output_dir = str(output_base / pdb_id / f"T{temp:.3f}")

        designed_seqs = run_proteinmpnn(
            pdb_path,
            output_dir,
            proteinmpnn_dir,
            chain_id=chain_id,
            num_seqs=num_seqs,
            temperature=temp,
        )

        if not designed_seqs:
            print(f"    No sequences generated at T={temp}")
            continue

        fsi_values = []
        for seq in designed_seqs:
            func_rec, overall_rec = compute_site_recovery(seq, wt_seq, seq_positions_1idx)
            fsi = compute_fsi(func_rec, overall_rec)
            fsi_values.append(fsi)

        results_by_temp[temp] = {
            "mean": float(np.mean(fsi_values)),
            "std": float(np.std(fsi_values)),
            "median": float(np.median(fsi_values)),
            "fraction_above_1": float(np.mean(np.array(fsi_values) > 1.0)),
            "n_sequences": len(fsi_values),
        }
        print(f"    FSI = {results_by_temp[temp]['mean']:.3f} ± {results_by_temp[temp]['std']:.3f}  "
              f"(fraction>1: {results_by_temp[temp]['fraction_above_1']:.2f})")

    return results_by_temp


def compute_spearman_rho(results_by_temp: dict) -> float:
    """Compute Spearman correlation between temperature and mean FSI."""
    temps = sorted(results_by_temp.keys())
    fsi_means = [results_by_temp[t]["mean"] for t in temps]
    if len(temps) < 3:
        return float("nan")
    rho, _ = stats.spearmanr(temps, fsi_means)
    return float(rho)


def interpret_temperature_result(pdb_id: str, results_by_temp: dict, temperatures: list[float]) -> str:
    """Generate interpretation string for temperature sensitivity result."""
    temps_with_data = [t for t in temperatures if t in results_by_temp]
    if not temps_with_data:
        return "No data available"

    all_above_1 = all(results_by_temp[t]["fraction_above_1"] > 0.5 for t in temps_with_data)
    min_fsi = min(results_by_temp[t]["mean"] for t in temps_with_data)

    if all_above_1:
        return f"FSI remains robustly >1.0 across all temperatures (min mean FSI={min_fsi:.2f})"
    elif min_fsi > 1.0:
        return f"Mean FSI remains >1.0 but fraction_above_1 drops at higher temperatures"
    else:
        return f"FSI drops below 1.0 at higher temperatures (min mean FSI={min_fsi:.2f})"


def plot_temperature_sensitivity(all_results: list[dict], temperatures: list[float]):
    """Line plot: x=temperature, y=mean FSI ± SD, one line per protein."""
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {"3BTA": "#ef4444", "2AAI": "#3b82f6"}
    labels = {"3BTA": "BoNT-A (3BTA)", "2AAI": "Ricin A-chain (2AAI)"}

    for result in all_results:
        pdb_id = result["pdb_id"]
        fsi_by_temp = result["fsi_by_temperature"]

        temps_present = sorted(t for t in temperatures if str(t) in fsi_by_temp or t in fsi_by_temp)
        if not temps_present:
            continue

        # Handle both float and string keys
        def get_val(d, t):
            return d.get(t, d.get(str(t), {}))

        means = [get_val(fsi_by_temp, t).get("mean", float("nan")) for t in temps_present]
        stds = [get_val(fsi_by_temp, t).get("std", 0) for t in temps_present]

        color = colors.get(pdb_id, "#94a3b8")
        label = labels.get(pdb_id, pdb_id)
        ax.errorbar(temps_present, means, yerr=stds, marker="o", capsize=4,
                    color=color, label=label, linewidth=2, markersize=6)

    ax.axhline(1.0, color="black", linestyle="--", lw=1.5, label="FSI = 1.0 (null)")
    ax.set_xlabel("ProteinMPNN Sampling Temperature", fontsize=12)
    ax.set_ylabel("Mean Functional Specificity Index (FSI)", fontsize=12)
    ax.set_title("FSI Temperature Sensitivity\nValidates FSI is not a T=0.1 artifact", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(min(temperatures) - 0.01, max(temperatures) + 0.01)

    plt.tight_layout()
    path = FIGURES_DIR / "fsi_temperature_sensitivity.png"
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="FSI temperature sensitivity analysis")
    parser.add_argument(
        "--proteinmpnn_dir",
        default=str(Path(__file__).parent.parent / "ProteinMPNN"),
    )
    parser.add_argument(
        "--temperatures",
        default="0.05,0.1,0.2,0.3",
        help="Comma-separated list of temperatures",
    )
    parser.add_argument("--num_seqs", type=int, default=100)
    args = parser.parse_args()

    temperatures = [float(t) for t in args.temperatures.split(",")]

    print_header("FSI Temperature Sensitivity Analysis")
    print(f"Proteins: {PROTEINS_TO_TEST}")
    print(f"Temperatures: {temperatures}")
    print(f"Sequences per T: {args.num_seqs}")

    mpnn_script = Path(args.proteinmpnn_dir) / "protein_mpnn_run.py"
    if not mpnn_script.exists():
        print(f"ERROR: ProteinMPNN not found at {args.proteinmpnn_dir}")
        sys.exit(1)

    # Use separate output directory to avoid collision with main proteinmpnn_output/
    output_base = RESULTS_DIR / "proteinmpnn_temp_sweep"
    output_base.mkdir(parents=True, exist_ok=True)

    all_results = []

    for pdb_id in PROTEINS_TO_TEST:
        pdb_path = str(STRUCT_DIR / f"{pdb_id}.pdb")
        if not Path(pdb_path).exists():
            print(f"  ERROR: {pdb_path} not found. Run 01_collect_data.py first.")
            continue

        try:
            functional_residues, chain_id = load_functional_sites_for_pdb(pdb_id)
        except ValueError as e:
            print(f"  ERROR: {e}")
            continue

        print(f"\n{'='*50}")
        print(f"Protein: {pdb_id}  chain={chain_id}  ({len(functional_residues)} functional residues)")
        print(f"{'='*50}")

        results_by_temp = run_temperature_sweep(
            pdb_id=pdb_id,
            pdb_path=pdb_path,
            chain_id=chain_id,
            functional_residues_1idx=functional_residues,
            proteinmpnn_dir=args.proteinmpnn_dir,
            output_base=output_base,
            temperatures=temperatures,
            num_seqs=args.num_seqs,
        )

        spearman_rho = compute_spearman_rho(results_by_temp)
        interpretation = interpret_temperature_result(pdb_id, results_by_temp, temperatures)

        all_results.append({
            "pdb_id": pdb_id,
            "fsi_by_temperature": {str(t): v for t, v in results_by_temp.items()},
            "spearman_rho_temp_vs_fsi": spearman_rho,
            "interpretation": interpretation,
        })

        print(f"  Spearman rho (temp vs FSI): {spearman_rho:.3f}")
        print(f"  Interpretation: {interpretation}")

    # Save results
    output = {
        "proteins_evaluated": PROTEINS_TO_TEST,
        "temperatures": temperatures,
        "num_seqs": args.num_seqs,
        "results": all_results,
    }
    path = RESULTS_DIR / "fsi_temperature_sensitivity.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

    # Generate figure
    print("\n--- Generating figure ---")
    plot_temperature_sensitivity(all_results, temperatures)

    print_header("Temperature Sensitivity Summary")
    for r in all_results:
        print(f"  {r['pdb_id']}: {r['interpretation']}")
        print(f"    Spearman rho = {r['spearman_rho_temp_vs_fsi']:.3f}")


if __name__ == "__main__":
    main()
