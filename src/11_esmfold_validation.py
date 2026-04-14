#!/usr/bin/env python3
"""
11_esmfold_validation.py — Structural compatibility validation of top ProteinMPNN designs.

Scores top ProteinMPNN sequences of BoNT-A LC (3BTA) using ESM-IF1
(esm.pretrained.esm_if1_gvp4_t16_142M_UR50), which measures
log P(sequence | structure) — structural compatibility of a sequence
given the 3BTA backbone, using a model independent of ProteinMPNN.

WHY ESM-IF1 OVER ESMFOLD:
  - ESMFold requires openfold (fails to compile on many HPC systems)
  - ESM-IF1 is built into fair-esm 2.0.0, no extra dependencies
  - ESM-IF1 directly answers: "does this sequence fit the dangerous backbone?"
  - Higher LL/L = more backbone-compatible = structurally plausible

CLAIM: Top functional-recovery designs (FSI > 3) receive significantly
higher ESM-IF1 scores than low-recovery designs, validating that FSI
captures structurally constrained functional recovery.

Usage:
    python src/11_esmfold_validation.py \\
        --proteinmpnn_fasta results/proteinmpnn_output/3BTA/seqs/3BTA.fa \\
        --reference_pdb data/structures/3BTA.pdb \\
        --lc_end_residue 430 \\
        --n_top 10 \\
        --n_bottom 10 \\
        --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

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


def parse_proteinmpnn_fasta(fasta_path: Path) -> tuple[str, list[dict]]:
    """Parse ProteinMPNN output FASTA.

    Returns (wildtype_seq, list of designed sequence dicts).
    Wildtype is the first entry (score=0 or seq_recovery=1.0).
    """
    entries = []
    current_header = None
    current_seq = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_header is not None and current_seq:
                    entries.append((current_header, "".join(current_seq)))
                current_header = line[1:]
                current_seq = []
            elif line:
                current_seq.append(line)
        if current_header is not None and current_seq:
            entries.append((current_header, "".join(current_seq)))

    if not entries:
        return "", []

    wildtype_seq = entries[0][1]

    result = []
    for header, seq in entries[1:]:  # skip wildtype
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

    return wildtype_seq, result


def annotate_with_fsi(
    parsed_sequences: list[dict],
    wildtype_seq: str,
    functional_positions_1idx: list[int],
) -> list[dict]:
    """Add func_recovery, overall_recovery, fsi to each sequence dict."""
    annotated = []
    for item in parsed_sequences:
        seq = item["sequence"]
        # Truncate to wildtype length if needed
        seq_trunc = seq[:len(wildtype_seq)]
        wt_trunc = wildtype_seq[:len(seq_trunc)]
        if len(seq_trunc) != len(wt_trunc):
            continue
        func_rec, overall_rec = compute_site_recovery(seq_trunc, wt_trunc, functional_positions_1idx)
        fsi = compute_fsi(func_rec, overall_rec)
        annotated.append({
            **item,
            "sequence_lc": seq_trunc,
            "func_recovery": func_rec,
            "overall_recovery": overall_rec,
            "fsi": fsi,
        })
    return annotated


def score_sequences_esmif1(
    sequences: list[str],
    pdb_path: str,
    chain_id: str,
    device: str,
) -> list[float]:
    """Score sequences using ESM-IF1 inverse folding model.

    Returns log-likelihood per residue (LL/L) for each sequence given
    the backbone structure. Higher = more structurally compatible.

    ESM-IF1 does NOT require openfold — uses GVP-Transformer architecture.
    """
    try:
        import esm
        import torch
        import torch.nn.functional as F
        from esm.inverse_folding.util import CoordBatchConverter
    except ImportError:
        print("ERROR: fair-esm not installed")
        return [float("nan")] * len(sequences)

    print("  Loading ESM-IF1 model (~600MB on first run)...")
    try:
        model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        model = model.eval().to(device)
    except Exception as e:
        print(f"  ERROR loading ESM-IF1: {e}")
        return [float("nan")] * len(sequences)

    # Load structure and extract coordinates
    try:
        structure = esm.inverse_folding.util.load_structure(pdb_path, chain_id)
        coords, native_seq = esm.inverse_folding.util.extract_coords_from_structure(structure)
    except Exception as e:
        print(f"  ERROR loading structure: {e}")
        return [float("nan")] * len(sequences)

    print(f"  Structure loaded: {len(native_seq)} residues, chain {chain_id}")

    def _score_one(seq_str):
        """Score a single sequence — CoordBatchConverter returns CPU tensors;
        we move everything to device before calling model.forward."""
        batch_converter = CoordBatchConverter(alphabet)
        coords_b, confidence, _, tokens, padding_mask = batch_converter(
            [(coords, None, seq_str)]
        )
        coords_b = coords_b.to(device)
        confidence = confidence.to(device)
        tokens = tokens.to(device)
        padding_mask = padding_mask.to(device)

        prev_output_tokens = tokens[:, :-1]
        target = tokens[:, 1:]
        target_padding_mask = (target == alphabet.padding_idx)

        logits, _ = model.forward(coords_b, padding_mask, confidence, prev_output_tokens)
        loss = F.cross_entropy(logits, target, reduction="none")
        loss_np = loss[0].detach().cpu().numpy()
        mask_np = target_padding_mask[0].cpu().numpy()
        ll = -np.sum(loss_np * ~mask_np) / max(np.sum(~mask_np), 1)
        return float(ll)

    log_likelihoods = []
    for i, seq in enumerate(sequences):
        try:
            with torch.no_grad():
                ll_per_residue = _score_one(seq)
            log_likelihoods.append(ll_per_residue)
            print(f"  Seq {i+1}/{len(sequences)}: LL/L = {ll_per_residue:.4f}")
        except Exception as e:
            print(f"  Seq {i+1}: ERROR {e}")
            log_likelihoods.append(float("nan"))

    return log_likelihoods


def plot_esm_if1_validation(top_seqs: list[dict], bottom_seqs: list[dict],
                            wildtype_ll: float):
    """Scatter and violin plot of ESM-IF1 LL/L vs functional recovery."""
    all_seqs = top_seqs + bottom_seqs
    valid = [s for s in all_seqs if not np.isnan(s.get("esm_if1_ll", float("nan")))]

    if not valid:
        print("  No valid ESM-IF1 scores to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: scatter functional recovery vs LL/L
    ax = axes[0]
    func_recs = [s["func_recovery"] for s in valid]
    ll_vals = [s["esm_if1_ll"] for s in valid]
    colors = ["#ef4444" if s in top_seqs else "#3b82f6" for s in valid]

    ax.scatter(func_recs, ll_vals, c=colors, s=80, alpha=0.85,
               edgecolors="white", linewidths=1)
    if not np.isnan(wildtype_ll):
        ax.axhline(wildtype_ll, color="black", linestyle="--", lw=1.5,
                   label=f"WT LL/L = {wildtype_ll:.3f}")
    rho, pval = stats.spearmanr(func_recs, ll_vals)
    ax.text(0.05, 0.95, f"Spearman rho = {rho:.3f}\np = {pval:.3e}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#ef4444", alpha=0.85, label="Top FSI (high func recovery)"),
        Patch(color="#3b82f6", alpha=0.85, label="Bottom FSI (low func recovery)"),
    ] + (ax.get_legend_handles_labels()[0] if not np.isnan(wildtype_ll) else []),
              fontsize=9)
    ax.set_xlabel("Functional Site Recovery", fontsize=12)
    ax.set_ylabel("ESM-IF1 LL/L (log P(seq|structure) per residue)", fontsize=11)
    ax.set_title("Structural Compatibility vs\nFunctional Recovery", fontsize=12)

    # Panel B: violin top vs bottom
    ax = axes[1]
    top_ll = [s["esm_if1_ll"] for s in top_seqs if not np.isnan(s.get("esm_if1_ll", float("nan")))]
    bot_ll = [s["esm_if1_ll"] for s in bottom_seqs if not np.isnan(s.get("esm_if1_ll", float("nan")))]
    if top_ll and bot_ll:
        parts = ax.violinplot([top_ll, bot_ll], positions=[1, 2],
                              showmedians=True, showextrema=True)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor("#ef4444" if i == 0 else "#3b82f6")
            pc.set_alpha(0.7)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Top FSI\n(high func recovery)", "Bottom FSI\n(low func recovery)"],
                           fontsize=11)
        stat, pval_mw = stats.mannwhitneyu(top_ll, bot_ll, alternative="greater")
        ax.text(0.5, 0.95, f"Mann-Whitney p = {pval_mw:.3e}",
                transform=ax.transAxes, ha="center", va="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        if not np.isnan(wildtype_ll):
            ax.axhline(wildtype_ll, color="black", linestyle="--", lw=1.5,
                       label=f"WT LL/L")
            ax.legend(fontsize=9)
    ax.set_ylabel("ESM-IF1 LL/L (log P(seq|structure) / L)", fontsize=11)
    ax.set_title("Top vs Bottom FSI Designs:\nStructural Compatibility", fontsize=12)

    plt.tight_layout()
    path = FIGURES_DIR / "esmif1_structural_compatibility.png"
    fig.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="ESM-IF1 structural compatibility validation")
    parser.add_argument("--proteinmpnn_fasta",
                        default="results/proteinmpnn_output/3BTA/seqs/3BTA.fa")
    parser.add_argument("--reference_pdb", default="data/structures/3BTA.pdb")
    parser.add_argument("--lc_end_residue", type=int, default=430)
    parser.add_argument("--n_top", type=int, default=10,
                        help="Top N sequences by functional recovery to score")
    parser.add_argument("--n_bottom", type=int, default=10,
                        help="Bottom N sequences by functional recovery (comparison group)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    fasta_path = Path(args.proteinmpnn_fasta)
    ref_pdb = Path(args.reference_pdb)

    print_header("ESM-IF1 Structural Compatibility Validation")
    print("Model: esm_if1_gvp4_t16_142M_UR50 (no openfold required)")
    print("Metric: log P(sequence | 3BTA backbone) per residue")

    if not fasta_path.exists():
        print(f"ERROR: FASTA not found: {fasta_path}")
        sys.exit(1)
    if not ref_pdb.exists():
        print(f"ERROR: Reference PDB not found: {ref_pdb}")
        sys.exit(1)

    functional_residues = load_functional_residues_for_pdb("3BTA")
    print(f"Functional residues (3BTA): {functional_residues}")

    # Parse FASTA
    print(f"\nParsing {fasta_path}...")
    wildtype_seq_full, parsed = parse_proteinmpnn_fasta(fasta_path)
    wildtype_seq = wildtype_seq_full[:args.lc_end_residue]
    print(f"  Wildtype LC: {len(wildtype_seq)} aa,  Designed: {len(parsed)} sequences")

    # Annotate with FSI
    annotated = annotate_with_fsi(parsed, wildtype_seq, functional_residues)
    annotated.sort(key=lambda x: (x["func_recovery"], x["fsi"]), reverse=True)

    top_seqs = annotated[:args.n_top]
    bottom_seqs = annotated[-args.n_bottom:]

    print(f"\n  Top {args.n_top} (highest func recovery): "
          f"func_rec range [{top_seqs[-1]['func_recovery']:.3f}, {top_seqs[0]['func_recovery']:.3f}]")
    print(f"  Bottom {args.n_bottom} (lowest func recovery): "
          f"func_rec range [{bottom_seqs[0]['func_recovery']:.3f}, {bottom_seqs[-1]['func_recovery']:.3f}]")

    # Score all sequences + wildtype with ESM-IF1
    all_to_score = [s["sequence_lc"] for s in top_seqs] + \
                   [s["sequence_lc"] for s in bottom_seqs] + \
                   [wildtype_seq]

    print(f"\nScoring {len(all_to_score)} sequences with ESM-IF1...")
    ll_scores = score_sequences_esmif1(
        all_to_score, str(ref_pdb), "A", args.device
    )

    # Assign scores back
    n_top = len(top_seqs)
    n_bot = len(bottom_seqs)
    for i, s in enumerate(top_seqs):
        s["esm_if1_ll"] = ll_scores[i]
    for i, s in enumerate(bottom_seqs):
        s["esm_if1_ll"] = ll_scores[n_top + i]
    wildtype_ll = ll_scores[n_top + n_bot]

    # Summary stats
    top_ll_valid = [s["esm_if1_ll"] for s in top_seqs if not np.isnan(s["esm_if1_ll"])]
    bot_ll_valid = [s["esm_if1_ll"] for s in bottom_seqs if not np.isnan(s["esm_if1_ll"])]

    summary = {
        "model": "ESM-IF1 (esm_if1_gvp4_t16_142M_UR50)",
        "metric": "log P(sequence | 3BTA backbone) per residue",
        "wildtype_ll_per_residue": wildtype_ll,
        "top_sequences_mean_ll": float(np.mean(top_ll_valid)) if top_ll_valid else float("nan"),
        "bottom_sequences_mean_ll": float(np.mean(bot_ll_valid)) if bot_ll_valid else float("nan"),
        "n_top_scored": len(top_ll_valid),
        "n_bottom_scored": len(bot_ll_valid),
    }

    if top_ll_valid and bot_ll_valid:
        stat, pval = stats.mannwhitneyu(top_ll_valid, bot_ll_valid, alternative="greater")
        r = 1 - (2 * stat) / (len(top_ll_valid) * len(bot_ll_valid))
        summary["mannwhitney_top_vs_bottom_pvalue"] = float(pval)
        summary["rank_biserial_r"] = float(r)
        print(f"\n  Top FSI LL/L = {summary['top_sequences_mean_ll']:.4f}")
        print(f"  Bottom FSI LL/L = {summary['bottom_sequences_mean_ll']:.4f}")
        print(f"  Wildtype LL/L = {wildtype_ll:.4f}")
        print(f"  Mann-Whitney (top > bottom): p = {pval:.4e}  r = {r:.3f}")
        if pval < 0.05:
            print("  → High functional-recovery designs are significantly more")
            print("    structurally compatible with 3BTA backbone (ESM-IF1)")
        else:
            print("  → No significant difference in structural compatibility")

    # Save
    per_sequence = []
    for s in top_seqs + bottom_seqs:
        per_sequence.append({
            "sample_id": s["sample_id"],
            "group": "top" if s in top_seqs else "bottom",
            "func_recovery": s["func_recovery"],
            "overall_recovery": s["overall_recovery"],
            "fsi": s["fsi"],
            "esm_if1_ll_per_residue": s.get("esm_if1_ll", float("nan")),
        })

    output = {
        "pdb_id": "3BTA",
        "validation_method": "ESM-IF1 inverse folding structural compatibility",
        "n_top_sequences": args.n_top,
        "n_bottom_sequences": args.n_bottom,
        "lc_domain_end": args.lc_end_residue,
        "summary": summary,
        "per_sequence": per_sequence,
    }
    out_path = RESULTS_DIR / "esmfold_validation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Figure
    print("\n--- Generating figure ---")
    (RESULTS_DIR / "esmfold_structures").mkdir(exist_ok=True)
    plot_esm_if1_validation(top_seqs, bottom_seqs, wildtype_ll)

    print_header("ESM-IF1 Validation Complete")


if __name__ == "__main__":
    main()
