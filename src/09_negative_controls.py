#!/usr/bin/env python3
"""
09_negative_controls.py — FSI negative control analysis.

Runs ProteinMPNN on 4 mechanism-matched benign proteins and computes FSI.
Compares to the dangerous toxins to validate that FSI is specific to
dangerous function, not a general feature of the enzymatic fold.

Benign controls:
  1AST — Astacin (zinc metalloprotease; same HExxH metzincin fold as BoNT-A)
  1LNF — Thermolysin (same HExxH zinc chemistry as BoNT-A; DIFFERENT fold)
  1QD2 — Saporin-6 (type-1 RIP; same beta-trefoil fold as Ricin A-chain)
  1LYZ — Hen egg-white lysozyme (general benign enzyme baseline)

Key question: Is BoNT-A FSI=2.87 specific to dangerous function,
or is it explained by the zinc metalloprotease fold?
Adding thermolysin (1LNF) decomposes: fold contribution (1AST) vs.
zinc chemistry contribution (1LNF) vs. dangerous function (3BTA).

Usage:
    python src/09_negative_controls.py [--proteinmpnn_dir /path] [--num_seqs 100]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    FIGURES_DIR,
    RESULTS_DIR,
    compute_fsi,
    compute_sequence_identity,
    compute_site_recovery,
    print_header,
)

# Re-use functions from 06_proteinmpnn_redesign
# Import the module by file path to avoid naming issues (filename starts with digit)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "mpnn_redesign",
    str(Path(__file__).parent / "06_proteinmpnn_redesign.py"),
)
_mpnn_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mpnn_mod)
extract_wildtype_sequence = _mpnn_mod.extract_wildtype_sequence
run_proteinmpnn = _mpnn_mod.run_proteinmpnn

CONTROLS_DIR = Path(__file__).parent.parent / "data" / "structures" / "controls"

# Mechanism-matched benign controls
BENIGN_CONTROLS = {
    "1AST": {
        "name": "Astacin",
        "organism": "Astacus astacus",
        "chain": "A",
        "description": "Zinc metalloprotease (HExxH); benign digestive enzyme",
        "is_control_for": "3BTA",
        "matched_toxin_name": "Botulinum neurotoxin A",
        "matched_toxin_fsi": 2.875,
        "mechanism_match": "Same HExxH zinc-binding motif, metzincin superfamily fold",
        "functional_residues": [92, 93, 96, 102],  # PDB numbering, His92/Glu93/His96/His102
        "use_pdb_numbering": True,
    },
    "1QD2": {
        "name": "Saporin-6",
        "organism": "Saponaria officinalis",
        "chain": "A",
        "description": "Type-1 RIP (~10,000x less toxic than ricin); benign in absence of B-chain",
        "is_control_for": "2AAI",
        "matched_toxin_name": "Ricin A-chain",
        "matched_toxin_fsi": 1.094,
        "mechanism_match": "Beta-trefoil active site (Tyr72/Tyr120/Glu176 parallels Tyr80/Tyr123/Glu177)",
        "functional_residues": [72, 120, 176],  # PDB numbering
        "use_pdb_numbering": True,
    },
    "1LNF": {
        "name": "Thermolysin",
        "organism": "Bacillus thermoproteolyticus",
        "chain": "E",
        "description": "Type-II zinc metalloprotease (HExxH); thermostable bacterial enzyme. Different fold from metzincin (thermolysin fold, not metzincin).",
        "is_control_for": "3BTA",
        "matched_toxin_name": "Botulinum neurotoxin A",
        "matched_toxin_fsi": 3.074,
        "mechanism_match": "HExxH zinc-binding motif (Glu143, His231, His235) + Glu166 third ligand. Same zinc chemistry as BoNT-A LC but thermolysin fold is completely different from metzincin fold.",
        "functional_residues": [143, 166, 231, 235],  # PDB numbering
        "use_pdb_numbering": True,
        "reference": "Matthews BW et al. (1972) doi:10.1038/238037a0",
    },
    "1LYZ": {
        "name": "Hen egg-white lysozyme",
        "organism": "Gallus gallus",
        "chain": "A",
        "description": "Glycoside hydrolase; well-characterized benign enzyme, no toxicity",
        "is_control_for": "general_baseline",
        "matched_toxin_name": "None (general negative control)",
        "matched_toxin_fsi": None,
        "mechanism_match": "Canonical enzyme with well-characterized active site (Glu35/Asp52)",
        "functional_residues": [35, 52],  # PDB numbering
        "use_pdb_numbering": True,
    },
}


def download_control_structure(pdb_id: str) -> Path:
    """Download PDB file for a control structure."""
    CONTROLS_DIR.mkdir(parents=True, exist_ok=True)
    pdb_path = CONTROLS_DIR / f"{pdb_id}.pdb"
    if pdb_path.exists():
        print(f"  {pdb_id}.pdb already exists, skipping download")
        return pdb_path

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"  Downloading {pdb_id} from RCSB...")
    try:
        urllib.request.urlretrieve(url, pdb_path)
        print(f"  Downloaded: {pdb_path}")
    except Exception as e:
        print(f"  ERROR downloading {pdb_id}: {e}")
        return None
    return pdb_path


def compute_fsi_for_control(
    pdb_id: str,
    ctrl_info: dict,
    designed_seqs: list,
    wildtype_seq: str,
    pdb_resnums: list,
) -> dict:
    """Compute FSI for a benign control protein."""
    # Map PDB residue numbers to 0-indexed positions
    pdb_to_idx = {resnum: i for i, resnum in enumerate(pdb_resnums)}
    func_positions_0idx = []
    for resnum in ctrl_info["functional_residues"]:
        if resnum in pdb_to_idx:
            func_positions_0idx.append(pdb_to_idx[resnum])
        else:
            print(f"    WARNING: PDB residue {resnum} not found in {pdb_id}")

    if not func_positions_0idx:
        return None

    functional_residues_1idx = [p + 1 for p in func_positions_0idx]

    func_recoveries, overall_recoveries, fsi_values = [], [], []
    for seq in designed_seqs:
        func_rec, overall_rec = compute_site_recovery(seq, wildtype_seq, functional_residues_1idx)
        fsi = compute_fsi(func_rec, overall_rec)
        func_recoveries.append(func_rec)
        overall_recoveries.append(overall_rec)
        fsi_values.append(fsi)

    wt_identities = [compute_sequence_identity(s, wildtype_seq) for s in designed_seqs]

    return {
        "pdb_id": pdb_id,
        "name": ctrl_info["name"],
        "organism": ctrl_info["organism"],
        "is_control_for": ctrl_info["is_control_for"],
        "matched_toxin_name": ctrl_info["matched_toxin_name"],
        "matched_toxin_fsi": ctrl_info["matched_toxin_fsi"],
        "mechanism_match": ctrl_info["mechanism_match"],
        "n_designed_sequences": len(designed_seqs),
        "wildtype_length": len(wildtype_seq),
        "n_functional_sites": len(func_positions_0idx),
        "functional_recovery": {
            "mean": float(np.mean(func_recoveries)),
            "std": float(np.std(func_recoveries)),
        },
        "overall_recovery": {
            "mean": float(np.mean(overall_recoveries)),
            "std": float(np.std(overall_recoveries)),
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
        },
    }


def compare_fsi_toxin_vs_control(toxin_fsi_values: list, control_fsi_values: list) -> dict:
    """Mann-Whitney U test comparing toxin FSI distribution to benign control."""
    stat, p = stats.mannwhitneyu(toxin_fsi_values, control_fsi_values, alternative="greater")
    n1, n2 = len(toxin_fsi_values), len(control_fsi_values)
    # Rank-biserial r effect size
    r = 1 - (2 * stat) / (n1 * n2)
    return {
        "mannwhitney_u": float(stat),
        "p_value": float(p),
        "rank_biserial_r": float(r),
        "significant": bool(p < 0.05),
        "n_toxin": n1,
        "n_control": n2,
    }


def plot_fsi_toxin_vs_control(control_results: list, toxin_results_path: Path):
    """Grouped bar chart: toxins (red) vs mechanism-matched controls (blue)."""
    # Load toxin FSI results
    if not toxin_results_path.exists():
        print(f"  WARNING: Toxin FSI results not found at {toxin_results_path}, skipping comparison plot")
        return

    with open(toxin_results_path) as f:
        toxin_results = json.load(f)

    toxin_by_pdb = {r["pdb_id"]: r for r in toxin_results}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Toxin vs matched control FSI (matched pairs)
    ax = axes[0]
    pairs = []
    for ctrl in control_results:
        matched_pdb = ctrl["is_control_for"]
        if matched_pdb in toxin_by_pdb:
            toxin = toxin_by_pdb[matched_pdb]
            pairs.append((
                f"{matched_pdb}\n({toxin['description'][:15]}...)",
                toxin["fsi"]["mean"],
                toxin["fsi"]["std"],
                f"{ctrl['pdb_id']}\n({ctrl['name']})",
                ctrl["fsi"]["mean"],
                ctrl["fsi"]["std"],
            ))

    if pairs:
        x = np.arange(len(pairs))
        width = 0.35
        toxin_labels = [p[0] for p in pairs]
        toxin_means = [p[1] for p in pairs]
        toxin_stds = [p[2] for p in pairs]
        ctrl_means = [p[4] for p in pairs]
        ctrl_stds = [p[5] for p in pairs]

        bars1 = ax.bar(x - width / 2, toxin_means, width, yerr=toxin_stds,
                       label="Toxin", color="#ef4444", alpha=0.85, capsize=4)
        bars2 = ax.bar(x + width / 2, ctrl_means, width, yerr=ctrl_stds,
                       label="Benign control", color="#3b82f6", alpha=0.85, capsize=4)
        ax.axhline(1.0, color="black", linestyle="--", lw=1.5, label="FSI = 1.0")

        # Significance stars between pairs
        for i, (_, tm, _, _, cm, _) in enumerate(pairs):
            # Simple visual indicator
            if tm > cm * 1.2:
                ax.text(i, max(tm, cm) + 0.15, "*", ha="center", fontsize=14, color="black")

        ax.set_xticks(x)
        ax.set_xticklabels([f"Toxin vs\n{p[3].split(chr(10))[0]}" for p in pairs],
                           fontsize=9)
        ax.set_ylabel("Functional Specificity Index (FSI)", fontsize=12)
        ax.set_title("Toxin FSI vs Mechanism-Matched\nBenign Control FSI", fontsize=12)
        ax.legend(fontsize=10)

    # Panel B: All proteins (toxins + controls) side by side
    ax = axes[1]
    all_items = []
    for pdb_id, tr in sorted(toxin_by_pdb.items()):
        all_items.append({
            "label": pdb_id,
            "mean": tr["fsi"]["mean"],
            "std": tr["fsi"]["std"],
            "color": "#ef4444",
            "type": "toxin",
        })
    for ctrl in control_results:
        all_items.append({
            "label": ctrl["pdb_id"],
            "mean": ctrl["fsi"]["mean"],
            "std": ctrl["fsi"]["std"],
            "color": "#3b82f6",
            "type": "control",
        })

    x = np.arange(len(all_items))
    colors = [item["color"] for item in all_items]
    means = [item["mean"] for item in all_items]
    stds = [item["std"] for item in all_items]
    labels = [item["label"] for item in all_items]
    types = [item["type"] for item in all_items]

    bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.8, capsize=3)
    ax.axhline(1.0, color="black", linestyle="--", lw=1.5, label="FSI = 1.0")

    # Add legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(color="#ef4444", alpha=0.8, label="Toxin (dangerous)"),
        Patch(color="#3b82f6", alpha=0.8, label="Benign control"),
    ]
    ax.legend(handles=legend_elements, fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("FSI (mean ± SD across 100 sequences)", fontsize=11)
    ax.set_title("All Proteins: Toxins vs Benign Controls", fontsize=12)

    plt.tight_layout()
    path = FIGURES_DIR / "fsi_toxin_vs_control.png"
    fig.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close()


def plot_bonta_three_way_comparison(control_results: list, toxin_results_path: Path):
    """Three-way bar chart: BoNT-A vs Astacin (same fold) vs Thermolysin (same Zn chemistry).

    Decomposes fold contribution (1AST) vs zinc chemistry contribution (1LNF)
    vs dangerous function (3BTA=2.87).
    """
    if not toxin_results_path.exists():
        print("  WARNING: Toxin FSI results not found, skipping three-way comparison plot")
        return

    with open(toxin_results_path) as f:
        toxin_results = json.load(f)

    # fsi_results.json is a list — find 3BTA entry
    bonta_entry = None
    for entry in toxin_results:
        if entry.get("pdb_id") == "3BTA":
            bonta_entry = entry
            break

    if bonta_entry is None:
        print("  WARNING: 3BTA not found in fsi_results.json, skipping three-way plot")
        return

    # Find 1AST and 1LNF in control_results
    ctrl_by_pdb = {c["pdb_id"]: c for c in control_results}
    ast_entry = ctrl_by_pdb.get("1AST")
    lnf_entry = ctrl_by_pdb.get("1LNF")

    if ast_entry is None and lnf_entry is None:
        print("  WARNING: Neither 1AST nor 1LNF found in control results, skipping three-way plot")
        return

    # Build items list (only those available)
    items = [
        {
            "label": "BoNT-A\n(dangerous toxin)",
            "mean": bonta_entry["fsi"]["mean"],
            "std": bonta_entry["fsi"]["std"],
            "values": bonta_entry["fsi"].get("per_sequence_values", []),
            "color": "#ef4444",
        }
    ]
    if ast_entry:
        items.append({
            "label": "Astacin\n(same fold)",
            "mean": ast_entry["fsi"]["mean"],
            "std": ast_entry["fsi"]["std"],
            "values": ast_entry["fsi"].get("per_sequence_values", []),
            "color": "#3b82f6",
        })
    if lnf_entry:
        items.append({
            "label": "Thermolysin\n(same Zn chemistry)",
            "mean": lnf_entry["fsi"]["mean"],
            "std": lnf_entry["fsi"]["std"],
            "values": lnf_entry["fsi"].get("per_sequence_values", []),
            "color": "#8b5cf6",
        })

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(items))
    colors = [item["color"] for item in items]
    means = [item["mean"] for item in items]
    stds = [item["std"] for item in items]
    labels = [item["label"] for item in items]

    bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.85, capsize=5, width=0.55)
    ax.axhline(1.0, color="black", linestyle="--", lw=1.5, label="FSI = 1.0")

    # Add significance brackets vs BoNT-A
    bonta_vals = items[0]["values"]
    y_max = max(m + s for m, s in zip(means, stds)) + 0.15
    for i in range(1, len(items)):
        ctrl_vals = items[i]["values"]
        if bonta_vals and ctrl_vals:
            cmp = compare_fsi_toxin_vs_control(bonta_vals, ctrl_vals)
            p = cmp["p_value"]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            bracket_y = y_max + 0.1 * i
            ax.plot([0, 0, i, i], [bracket_y - 0.05, bracket_y, bracket_y, bracket_y - 0.05],
                    color="black", lw=1.2)
            ax.text(i / 2, bracket_y + 0.02, sig, ha="center", fontsize=12)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Functional Specificity Index (FSI)", fontsize=12)
    ax.set_title("BoNT-A Three-Way Control Comparison\nFold geometry vs. zinc chemistry vs. dangerous function", fontsize=12)
    ax.legend(fontsize=10)

    plt.tight_layout()
    path = FIGURES_DIR / "bonta_three_way_control.png"
    fig.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="FSI negative control analysis")
    parser.add_argument(
        "--proteinmpnn_dir",
        default=str(Path(__file__).parent.parent / "ProteinMPNN"),
    )
    parser.add_argument("--num_seqs", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()

    print_header("FSI Negative Control Analysis")
    print("Controls: 1AST (astacin, zinc MMP), 1LNF (thermolysin, zinc MMP diff fold), 1QD2 (saporin, RIP), 1LYZ (lysozyme)")
    print("Goal: Validate that BoNT-A FSI=2.87 is function-specific, not fold-specific")
    print()

    # Check ProteinMPNN
    mpnn_script = Path(args.proteinmpnn_dir) / "protein_mpnn_run.py"
    if not mpnn_script.exists():
        print(f"ERROR: ProteinMPNN not found at {args.proteinmpnn_dir}")
        sys.exit(1)

    output_base = RESULTS_DIR / "proteinmpnn_output" / "controls"
    output_base.mkdir(parents=True, exist_ok=True)

    control_results = []

    for pdb_id, ctrl_info in BENIGN_CONTROLS.items():
        print(f"\n{'='*50}")
        print(f"Control: {pdb_id} — {ctrl_info['name']}")
        print(f"  Mechanism match: {ctrl_info['mechanism_match'][:60]}...")
        print(f"{'='*50}")

        # Download structure
        pdb_path = download_control_structure(pdb_id)
        if pdb_path is None:
            print(f"  Skipping {pdb_id} (download failed)")
            continue

        # Extract wildtype sequence
        wt_seq, pdb_resnums = extract_wildtype_sequence(str(pdb_path), ctrl_info["chain"])
        if not wt_seq:
            print(f"  ERROR: Could not extract sequence from {pdb_id}")
            continue
        print(f"  Wild-type length: {len(wt_seq)}")

        # Run ProteinMPNN
        output_dir = str(output_base / pdb_id)
        designed_seqs = run_proteinmpnn(
            str(pdb_path),
            output_dir,
            args.proteinmpnn_dir,
            chain_id=ctrl_info["chain"],
            num_seqs=args.num_seqs,
            temperature=args.temperature,
        )

        if not designed_seqs:
            print(f"  No sequences obtained, skipping")
            continue

        print(f"  Generated {len(designed_seqs)} designed sequences")

        # Compute FSI
        result = compute_fsi_for_control(pdb_id, ctrl_info, designed_seqs, wt_seq, pdb_resnums)
        if result is None:
            print(f"  FSI computation failed, skipping")
            continue

        control_results.append(result)
        print(f"  Functional recovery: {result['functional_recovery']['mean']:.3f} ± {result['functional_recovery']['std']:.3f}")
        print(f"  Overall recovery:    {result['overall_recovery']['mean']:.3f} ± {result['overall_recovery']['std']:.3f}")
        print(f"  FSI:                 {result['fsi']['mean']:.3f} ± {result['fsi']['std']:.3f}")
        print(f"  Fraction FSI > 1:    {result['fsi']['fraction_above_1']:.3f}")
        print(f"  Mean WT identity:    {result['sequence_divergence']['mean_wt_identity']:.3f}")

    if not control_results:
        print("No control results obtained. Exiting.")
        sys.exit(1)

    # Statistical comparison with toxin FSI
    print_header("Toxin vs Control FSI Comparison")

    toxin_fsi_path = RESULTS_DIR / "fsi_results.json"
    comparisons = {}

    if toxin_fsi_path.exists():
        with open(toxin_fsi_path) as f:
            toxin_results = json.load(f)
        toxin_by_pdb = {r["pdb_id"]: r for r in toxin_results}

        for ctrl in control_results:
            matched_pdb = ctrl["is_control_for"]
            if matched_pdb in toxin_by_pdb:
                toxin = toxin_by_pdb[matched_pdb]
                toxin_fsi_vals = toxin["fsi"].get("per_sequence_values", [toxin["fsi"]["mean"]])
                ctrl_fsi_vals = ctrl["fsi"]["per_sequence_values"]

                cmp = compare_fsi_toxin_vs_control(toxin_fsi_vals, ctrl_fsi_vals)
                comparisons[f"{matched_pdb}_vs_{ctrl['pdb_id']}"] = cmp

                sig = "***" if cmp["p_value"] < 0.001 else "**" if cmp["p_value"] < 0.01 else "*" if cmp["p_value"] < 0.05 else "ns"
                print(f"\n  {toxin['description'][:20]} (FSI={toxin['fsi']['mean']:.3f})")
                print(f"  vs {ctrl['name']} (FSI={ctrl['fsi']['mean']:.3f})")
                print(f"  Mann-Whitney p={cmp['p_value']:.4f} {sig}, r={cmp['rank_biserial_r']:.3f}")
                if cmp["significant"]:
                    print(f"  → Toxin FSI significantly HIGHER than mechanism-matched control")
                    print(f"  → FSI is specific to dangerous function, not just fold class")
                else:
                    print(f"  → No significant difference from benign control")
                    print(f"  → FSI may reflect fold properties rather than danger specifically")
    else:
        print("  Toxin FSI results not found. Run 06_proteinmpnn_redesign.py first.")

    # Save results
    output = {
        "controls": control_results,
        "comparisons": comparisons,
    }
    path = RESULTS_DIR / "fsi_controls.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

    # Generate figures
    print("\n--- Generating figures ---")
    plot_fsi_toxin_vs_control(control_results, toxin_fsi_path)
    plot_bonta_three_way_comparison(control_results, toxin_fsi_path)

    print_header("Summary")
    for ctrl in control_results:
        print(f"  {ctrl['pdb_id']} ({ctrl['name']}): FSI = {ctrl['fsi']['mean']:.3f} ± {ctrl['fsi']['std']:.3f}")
    if control_results:
        ctrl_fsi_means = [c["fsi"]["mean"] for c in control_results]
        print(f"\n  Mean control FSI: {np.mean(ctrl_fsi_means):.3f}")
        print(f"  (Compare to BoNT-A toxin FSI = 2.87)")


if __name__ == "__main__":
    main()
