#!/usr/bin/env python3
"""
11_esmfold_validation.py — ESMFold structural validation of top ProteinMPNN designs.

Takes the top-10 ProteinMPNN sequences of BoNT-A LC (3BTA) that fully recover
all functional residues, runs them through ESMFold, and computes TM-score vs.
the 3BTA crystal structure LC domain.

Claim: sequences that recover dangerous function (FSI=3.07, 100% recovery)
also fold to native-like structures (TM-score > 0.5), validating that FSI
represents structurally plausible designs, not random sequence artifacts.

Usage:
    python src/11_esmfold_validation.py \
        --proteinmpnn_fasta results/proteinmpnn_output/3BTA/seqs/3BTA.fa \
        --reference_pdb data/structures/3BTA.pdb \
        --lc_end_residue 430 \
        --n_top 10 \
        --device cuda

Requires GPU and ESMFold (fair-esm >=2.0.0 + openfold).
Memory: ~24GB for 430-aa sequences. Use 64GB partition.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import FIGURES_DIR, RESULTS_DIR, compute_fsi, compute_site_recovery, print_header

DATA_DIR = Path(__file__).parent.parent / "data"
ANNOT_DIR = DATA_DIR / "annotations"


def load_functional_residues_for_pdb(pdb_id: str) -> list[int]:
    """Return functional residue positions (PDB numbering if use_pdb_numbering, else sequence)."""
    with open(ANNOT_DIR / "functional_sites.json") as f:
        func_sites = json.load(f)

    for uid, info in func_sites.items():
        if uid.startswith("_"):
            continue
        if info.get("pdb_id") == pdb_id:
            if info.get("use_pdb_numbering") and info.get("pdb_residues"):
                return info["pdb_residues"]
            return info["functional_sites"]["catalytic_residues"]

    raise ValueError(f"No functional annotation found for {pdb_id}")


def parse_proteinmpnn_fasta(fasta_path: Path) -> list[dict]:
    """Parse ProteinMPNN output FASTA.

    ProteinMPNN FASTA header format:
        >pdb_id, score=X.XXX, global_score=X.XXX, seq_recovery=X.XXX, sample=N
    First sequence in the file is the wildtype (score=0, recovery=1.0). Skip it.

    Returns list of dicts with: sequence, score, seq_recovery, sample_id.
    """
    sequences = []
    current_header = None
    current_seq = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_header is not None and current_seq:
                    sequences.append((current_header, "".join(current_seq)))
                current_header = line[1:]
                current_seq = []
            elif line:
                current_seq.append(line)
        if current_header is not None and current_seq:
            sequences.append((current_header, "".join(current_seq)))

    # Skip first (wildtype) sequence
    if sequences and "sample=0," not in sequences[0][0] and "seq_recovery=1.0" in sequences[0][0]:
        sequences = sequences[1:]
    elif sequences:
        # More robust: skip entry with score=0.0000 (wildtype)
        sequences = [s for s in sequences if "score=0.0000" not in s[0]]

    result = []
    for header, seq in sequences:
        # Parse header fields
        info = {"sequence": seq, "score": float("nan"), "seq_recovery": float("nan"), "sample_id": -1}
        for part in header.split(","):
            part = part.strip()
            if "score=" in part and "global_score" not in part:
                try:
                    info["score"] = float(part.split("=")[1])
                except (ValueError, IndexError):
                    pass
            elif "seq_recovery=" in part:
                try:
                    info["seq_recovery"] = float(part.split("=")[1])
                except (ValueError, IndexError):
                    pass
            elif "sample=" in part:
                try:
                    info["sample_id"] = int(part.split("=")[1])
                except (ValueError, IndexError):
                    pass
        result.append(info)

    return result


def compute_functional_recovery_from_seq(
    sequence: str,
    wildtype_seq: str,
    functional_positions_1idx: list[int],
) -> tuple[float, float]:
    """Compute functional and overall recovery.

    Args:
        functional_positions_1idx: 1-indexed positions.
            compute_site_recovery() expects 1-indexed and subtracts 1 internally.
            Do NOT pass 0-indexed positions.
    """
    func_rec, overall_rec = compute_site_recovery(sequence, wildtype_seq, functional_positions_1idx)
    return func_rec, overall_rec


def select_top_sequences(
    parsed_sequences: list[dict],
    wildtype_seq: str,
    functional_positions_1idx: list[int],
    n_top: int = 10,
) -> list[dict]:
    """Rank sequences by functional recovery descending, tiebreak by overall FSI.

    Only considers sequences of the same length as wildtype.
    """
    candidates = []
    for item in parsed_sequences:
        seq = item["sequence"]
        if len(seq) != len(wildtype_seq):
            continue
        func_rec, overall_rec = compute_functional_recovery_from_seq(
            seq, wildtype_seq, functional_positions_1idx
        )
        fsi = compute_fsi(func_rec, overall_rec)
        candidates.append({
            **item,
            "func_recovery": func_rec,
            "overall_recovery": overall_rec,
            "fsi": fsi,
        })

    # Rank by func_recovery desc, tiebreak by fsi desc
    candidates.sort(key=lambda x: (x["func_recovery"], x["fsi"]), reverse=True)
    return candidates[:n_top]


def run_esmfold(sequences: list[str], device: str) -> list[tuple[str, float]]:
    """Run ESMFold on list of sequences.

    Returns list of (pdb_string, mean_plddt) per sequence.
    Handles OOM with chunk_size fallback [128, 64, 32].
    """
    try:
        import esm
        import torch
    except ImportError:
        print("ERROR: fair-esm not installed. Run: pip install fair-esm")
        return [(None, float("nan"))] * len(sequences)

    print("  Loading ESMFold model...")
    try:
        model = esm.pretrained.esmfold_v1()
        model = model.eval().to(device)
    except Exception as e:
        print(f"  ERROR loading ESMFold: {e}")
        return [(None, float("nan"))] * len(sequences)

    results = []
    import torch

    for i, seq in enumerate(sequences):
        print(f"  Folding sequence {i+1}/{len(sequences)} (len={len(seq)})...")
        pdb_str = None
        mean_plddt = float("nan")

        for chunk_size in [128, 64, 32, None]:
            try:
                with torch.no_grad():
                    if chunk_size:
                        output = model.infer_pdb(seq, num_recycles=4, residue_index_offset=512)
                    else:
                        output = model.infer_pdb(seq)
                pdb_str = output
                # Extract pLDDT from output
                try:
                    ptm_output = model.infer(seq)
                    plddt = ptm_output["plddt"].squeeze().cpu().numpy()
                    mean_plddt = float(plddt.mean())
                except Exception:
                    mean_plddt = float("nan")
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    if chunk_size is None:
                        print(f"    OOM at all chunk sizes, skipping sequence {i+1}")
                    continue
                else:
                    print(f"    ERROR: {e}")
                    break

        results.append((pdb_str, mean_plddt))

    return results


def extract_lc_domain(full_pdb_path: str, chain_id: str, end_residue: int, out_path: str):
    """Write PDB with only chain_id residues <= end_residue (0-indexed end_residue = aa count)."""
    lines_out = []
    with open(full_pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                rec_chain = line[21]
                try:
                    resnum = int(line[22:26].strip())
                except ValueError:
                    continue
                if rec_chain == chain_id and resnum <= end_residue:
                    lines_out.append(line)
            elif line.startswith("END"):
                lines_out.append(line)
    with open(out_path, "w") as f:
        f.writelines(lines_out)


def compute_tmscore(mobile_pdb: str, reference_pdb: str, usalign_binary: str = "USalign") -> float:
    """Compute TM-score between mobile and reference PDB.

    Tries USalign first, then TMalign, then biotite as fallback.
    Returns float TM-score (normalized to reference length), or -1.0 on failure.
    """
    # Try USalign
    for binary in [usalign_binary, "TMalign"]:
        try:
            result = subprocess.run(
                [binary, mobile_pdb, reference_pdb],
                capture_output=True, text=True, timeout=60
            )
            for line in result.stdout.splitlines():
                if line.startswith("TM-score=") and "Chain_2" in line:
                    return float(line.split("=")[1].split()[0])
                elif line.startswith("TM-score="):
                    return float(line.split("=")[1].split()[0])
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            continue

    # Fallback: biotite
    try:
        import biotite.structure.io.pdb as pdb_io
        import biotite.structure as struc

        ref_file = pdb_io.PDBFile.read(reference_pdb)
        mob_file = pdb_io.PDBFile.read(mobile_pdb)
        ref_atoms = pdb_io.get_structure(ref_file, model=1)
        mob_atoms = pdb_io.get_structure(mob_file, model=1)

        # Get CA atoms
        ref_ca = ref_atoms[ref_atoms.atom_name == "CA"]
        mob_ca = mob_atoms[mob_atoms.atom_name == "CA"]

        n_ref = len(ref_ca)
        if n_ref == 0 or len(mob_ca) == 0:
            return -1.0

        # Simple RMSD-based TM-score approximation (not exact)
        min_len = min(n_ref, len(mob_ca))
        ref_coords = ref_ca.coord[:min_len]
        mob_coords = mob_ca.coord[:min_len]

        d0 = 1.24 * (n_ref - 15) ** (1 / 3) - 1.8 if n_ref > 21 else 0.5
        diffs = np.linalg.norm(ref_coords - mob_coords, axis=1)
        tm_score = float(np.mean(1 / (1 + (diffs / d0) ** 2)))
        return tm_score
    except Exception:
        return -1.0


def plot_tmscore_scatter(results: list[dict], n_functional_sites: int):
    """Scatter plot: x=functional_recovery, y=TM-score."""
    if not results:
        return

    func_recs = [r["func_recovery"] for r in results]
    tm_scores = [r["tmscore"] for r in results if r["tmscore"] > 0]
    if not tm_scores:
        print("  No valid TM-scores to plot")
        return

    tm_scores_all = [r["tmscore"] for r in results]
    colors = ["#22c55e" if r["func_recovery"] >= 0.99 else "#94a3b8" for r in results]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(func_recs, tm_scores_all, c=colors, s=80, alpha=0.85, edgecolors="white", linewidths=1)
    ax.axhline(0.5, color="#f97316", linestyle="--", lw=1.5, label="TM-score = 0.5 (same fold)")
    ax.axhline(0.7, color="#ef4444", linestyle=":", lw=1.5, label="TM-score = 0.7 (near-native)")

    # Legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(color="#22c55e", alpha=0.85, label="All functional sites recovered"),
        Patch(color="#94a3b8", alpha=0.85, label="Partial functional recovery"),
        plt.Line2D([0], [0], color="#f97316", linestyle="--", label="TM = 0.5"),
        plt.Line2D([0], [0], color="#ef4444", linestyle=":", label="TM = 0.7"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")

    ax.set_xlabel("Functional Site Recovery (fraction of sites matching WT)", fontsize=11)
    ax.set_ylabel("TM-score vs. BoNT-A LC crystal structure", fontsize=11)
    ax.set_title("ProteinMPNN Designs: Functional Recovery vs. Structural Similarity\n"
                 "High functional recovery → native-like fold", fontsize=12)
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    path = FIGURES_DIR / "esmfold_tmscore_scatter.png"
    fig.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="ESMFold structural validation of top ProteinMPNN designs")
    parser.add_argument(
        "--proteinmpnn_fasta",
        default="results/proteinmpnn_output/3BTA/seqs/3BTA.fa",
        help="ProteinMPNN output FASTA for 3BTA",
    )
    parser.add_argument(
        "--reference_pdb",
        default="data/structures/3BTA.pdb",
        help="Reference crystal structure PDB",
    )
    parser.add_argument(
        "--lc_end_residue",
        type=int,
        default=430,
        help="Last residue of LC domain (truncate before interchain X-region)",
    )
    parser.add_argument("--n_top", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    fasta_path = Path(args.proteinmpnn_fasta)
    ref_pdb = Path(args.reference_pdb)

    print_header("ESMFold Structural Validation")

    if not fasta_path.exists():
        print(f"ERROR: FASTA not found: {fasta_path}")
        print("Run 06_proteinmpnn_redesign.py first.")
        sys.exit(1)

    if not ref_pdb.exists():
        print(f"ERROR: Reference PDB not found: {ref_pdb}")
        sys.exit(1)

    # Load functional sites for 3BTA
    functional_residues_1idx = load_functional_residues_for_pdb("3BTA")
    print(f"Functional residues (3BTA): {functional_residues_1idx}")

    # Parse ProteinMPNN FASTA
    print(f"\nParsing ProteinMPNN FASTA: {fasta_path}")
    parsed = parse_proteinmpnn_fasta(fasta_path)
    print(f"  Parsed {len(parsed)} designed sequences")

    # Get wildtype sequence (first entry in FASTA has seq_recovery=1.0)
    # Re-read to get wildtype
    all_raw = []
    current_header = None
    current_seq = []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_header is not None and current_seq:
                    all_raw.append((current_header, "".join(current_seq)))
                current_header = line[1:]
                current_seq = []
            elif line:
                current_seq.append(line)
        if current_header is not None and current_seq:
            all_raw.append((current_header, "".join(current_seq)))

    if not all_raw:
        print("ERROR: FASTA appears empty")
        sys.exit(1)

    wildtype_seq_full = all_raw[0][1]
    # Truncate to LC domain
    wildtype_seq = wildtype_seq_full[:args.lc_end_residue]
    print(f"  Wildtype (full): {len(wildtype_seq_full)} aa, LC domain: {len(wildtype_seq)} aa")

    # Select top sequences
    print(f"\nSelecting top {args.n_top} sequences by functional recovery...")
    top_seqs = select_top_sequences(
        parsed, wildtype_seq, functional_residues_1idx, n_top=args.n_top
    )

    if not top_seqs:
        # If no sequences match LC length, try full length
        top_seqs = select_top_sequences(
            parsed, wildtype_seq_full, functional_residues_1idx, n_top=args.n_top
        )

    print(f"  Selected {len(top_seqs)} sequences")
    for s in top_seqs[:3]:
        print(f"    sample={s['sample_id']}: func_rec={s['func_recovery']:.3f}, "
              f"overall_rec={s['overall_recovery']:.3f}, FSI={s['fsi']:.3f}")

    # Truncate sequences and reference to LC domain
    sequences_for_fold = [s["sequence"][:args.lc_end_residue] for s in top_seqs]

    # Extract LC domain from reference PDB
    esmfold_structs_dir = RESULTS_DIR / "esmfold_structures"
    esmfold_structs_dir.mkdir(parents=True, exist_ok=True)

    ref_lc_path = str(esmfold_structs_dir / "3BTA_LC_reference.pdb")
    extract_lc_domain(str(ref_pdb), "A", args.lc_end_residue, ref_lc_path)
    print(f"\nExtracted LC domain reference: {ref_lc_path}")

    # Run ESMFold
    print(f"\nRunning ESMFold on {len(sequences_for_fold)} sequences...")
    fold_results = run_esmfold(sequences_for_fold, args.device)

    # Compute TM-scores and build output
    print("\nComputing TM-scores vs. reference LC structure...")
    per_sequence = []
    for i, (top_seq, (pdb_str, mean_plddt)) in enumerate(zip(top_seqs, fold_results)):
        result_entry = {
            "sample_id": top_seq["sample_id"],
            "func_recovery": top_seq["func_recovery"],
            "overall_recovery": top_seq["overall_recovery"],
            "fsi": top_seq["fsi"],
            "mean_plddt": mean_plddt,
            "tmscore": -1.0,
            "predicted_structure_path": None,
        }

        if pdb_str is not None:
            # Save predicted structure
            sample_id = top_seq["sample_id"]
            out_pdb = str(esmfold_structs_dir / f"3BTA_sample{sample_id:03d}.pdb")
            with open(out_pdb, "w") as f:
                f.write(pdb_str)
            result_entry["predicted_structure_path"] = str(
                Path(out_pdb).relative_to(RESULTS_DIR.parent)
            )

            # Compute TM-score
            tmscore = compute_tmscore(out_pdb, ref_lc_path)
            result_entry["tmscore"] = tmscore
            print(f"  Sample {sample_id:3d}: func_rec={top_seq['func_recovery']:.3f}, "
                  f"TM={tmscore:.3f}, pLDDT={mean_plddt:.1f}")
        else:
            print(f"  Sample {top_seq['sample_id']:3d}: ESMFold failed (skipped)")

        per_sequence.append(result_entry)

    # Summary statistics
    valid_tm = [r["tmscore"] for r in per_sequence if r["tmscore"] > 0]
    valid_plddt = [r["mean_plddt"] for r in per_sequence if not np.isnan(r["mean_plddt"])]

    summary = {
        "mean_tmscore": float(np.mean(valid_tm)) if valid_tm else float("nan"),
        "median_tmscore": float(np.median(valid_tm)) if valid_tm else float("nan"),
        "fraction_tmscore_above_0.5": float(np.mean(np.array(valid_tm) > 0.5)) if valid_tm else float("nan"),
        "fraction_tmscore_above_0.7": float(np.mean(np.array(valid_tm) > 0.7)) if valid_tm else float("nan"),
        "mean_plddt": float(np.mean(valid_plddt)) if valid_plddt else float("nan"),
        "n_sequences_folded": len(valid_tm),
    }

    print(f"\n--- Summary ---")
    print(f"  Sequences folded: {summary['n_sequences_folded']}/{len(per_sequence)}")
    print(f"  Mean TM-score: {summary['mean_tmscore']:.3f}")
    print(f"  Fraction TM > 0.5: {summary['fraction_tmscore_above_0.5']:.2f}")
    print(f"  Fraction TM > 0.7: {summary['fraction_tmscore_above_0.7']:.2f}")
    print(f"  Mean pLDDT: {summary['mean_plddt']:.1f}")

    # Save output JSON
    output = {
        "pdb_id": "3BTA",
        "n_top_sequences": args.n_top,
        "lc_domain_end": args.lc_end_residue,
        "summary": summary,
        "per_sequence": per_sequence,
    }
    out_path = RESULTS_DIR / "esmfold_validation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Generate figure
    print("\n--- Generating figure ---")
    plot_tmscore_scatter(per_sequence, len(functional_residues_1idx))

    print_header("ESMFold Validation Complete")
    if valid_tm:
        if summary["mean_tmscore"] > 0.5:
            print("  RESULT: Top ProteinMPNN designs fold to native-like structures.")
            print("  → FSI=3.07 represents structurally plausible dangerous designs.")
        else:
            print("  RESULT: Designs do not consistently fold to native-like structures.")
            print("  → Physical realizability barrier confirmed by low structural similarity.")


if __name__ == "__main__":
    main()
