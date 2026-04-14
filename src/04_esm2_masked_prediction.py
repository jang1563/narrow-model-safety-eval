#!/usr/bin/env python3
"""
04_esm2_masked_prediction.py — Functional Site Prediction Entropy (FSPE) metric.

Masks known catalytic/binding residues in toxin proteins one at a time and
measures ESM-2's prediction entropy. Compares FSPE at functional sites vs.
non-functional sites.

This extends Meier et al. (2021)'s observation that ESM-1v masked prediction
entropy is lower at binding sites, formalizing it as a quantitative dual-use
risk metric.

Produces:
  - results/fspe_results.json
  - results/figures/fspe_*.png

Usage:
    python src/04_esm2_masked_prediction.py [--model esm2_t33_650M_UR50D]

Requires GPU (A40/A100 recommended).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from transformers import AutoModelForMaskedLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    FIGURES_DIR,
    RESULTS_DIR,
    load_functional_sites,
    load_positive_sequences,
    print_header,
    truncate_sequence,
)

DEFAULT_MODEL = "facebook/esm2_t33_650M_UR50D"
MAX_SEQ_LEN = 1022


# ============================================================================
# FSPE computation
# ============================================================================


def compute_shannon_entropy(logits: torch.Tensor) -> float:
    """Compute Shannon entropy of predicted amino acid distribution.

    Args:
        logits: (vocab_size,) tensor of logits at masked position

    Returns:
        Shannon entropy in nats
    """
    probs = torch.softmax(logits, dim=-1)
    # Filter to standard amino acids (indices 4-23 in ESM-2 tokenizer)
    # but safer to just use all non-zero probabilities
    log_probs = torch.log(probs + 1e-10)
    entropy = -(probs * log_probs).sum().item()
    return entropy


def predict_masked_position(
    sequence: str,
    position: int,  # 0-indexed
    model,
    tokenizer,
    device: str,
) -> dict:
    """Mask a single position and get ESM-2's prediction.

    Args:
        sequence: Amino acid sequence
        position: 0-indexed position to mask
        model: ESM-2 MLM model
        tokenizer: ESM-2 tokenizer
        device: 'cuda' or 'cpu'

    Returns:
        dict with entropy, top_prediction, correct_aa, is_correct, top5_probs
    """
    correct_aa = sequence[position]

    # Create masked sequence
    # ESM-2 tokenizer uses a Trie to identify special tokens like <mask>
    # before character-level tokenization, so embedding <mask> directly works
    seq_list = list(sequence)
    seq_list[position] = tokenizer.mask_token
    masked_seq = "".join(seq_list)

    # Tokenize
    inputs = tokenizer(
        masked_seq,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN + 2,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # The mask position in the tokenized sequence
    # ESM-2 adds BOS token, so position in tokens = position + 1
    mask_token_id = tokenizer.mask_token_id
    token_positions = (inputs["input_ids"] == mask_token_id).nonzero()

    if len(token_positions) == 0:
        return None

    mask_idx = token_positions[0, 1].item()
    logits = outputs.logits[0, mask_idx]

    # Entropy
    entropy = compute_shannon_entropy(logits)

    # Top prediction
    probs = torch.softmax(logits, dim=-1)
    top5_ids = torch.topk(probs, 5).indices
    top5_probs = torch.topk(probs, 5).values

    top_pred_aa = tokenizer.decode(top5_ids[0].item()).strip()

    # Probability of the correct amino acid
    correct_token_id = tokenizer.encode(correct_aa, add_special_tokens=False)
    if len(correct_token_id) > 0:
        correct_prob = probs[correct_token_id[0]].item()
    else:
        correct_prob = 0.0

    return {
        "position": position,
        "correct_aa": correct_aa,
        "top_prediction": top_pred_aa,
        "is_correct": top_pred_aa == correct_aa,
        "correct_prob": correct_prob,
        "plm_score": float(np.log(max(correct_prob, 1e-10))),  # log P(WT aa | context)
        "entropy": entropy,
        "top5": [
            {
                "aa": tokenizer.decode(tid.item()).strip(),
                "prob": tp.item(),
            }
            for tid, tp in zip(top5_ids, top5_probs)
        ],
    }


def evaluate_protein_fspe(
    uniprot_id: str,
    sequence: str,
    functional_residues: list,
    model,
    tokenizer,
    device: str,
    n_nonfunctional: int = 20,
    seed: int = 42,
) -> dict:
    """Evaluate FSPE for a single protein.

    Compares entropy at functional sites vs. randomly sampled non-functional sites.

    Args:
        uniprot_id: UniProt accession
        sequence: Amino acid sequence
        functional_residues: 1-indexed functional site positions
        model: ESM-2 MLM model
        tokenizer: ESM-2 tokenizer
        device: 'cuda' or 'cpu'
        n_nonfunctional: Number of non-functional sites to sample
        seed: Random seed for reproducibility

    Returns:
        dict with FSPE metrics
    """
    seq = truncate_sequence(sequence, MAX_SEQ_LEN)
    seq_len = len(seq)

    # Convert to 0-indexed and filter valid positions
    func_positions = [r - 1 for r in functional_residues if r - 1 < seq_len]

    if len(func_positions) == 0:
        print(f"  WARNING: No valid functional positions for {uniprot_id}")
        return None

    # Sample non-functional positions
    rng = np.random.RandomState(seed)
    all_positions = set(range(seq_len))
    nonfunc_candidates = list(all_positions - set(func_positions))
    n_sample = min(n_nonfunctional, len(nonfunc_candidates))
    nonfunc_positions = sorted(rng.choice(nonfunc_candidates, n_sample, replace=False))

    # Evaluate functional sites
    print(f"  Functional sites ({len(func_positions)} positions)...")
    func_results = []
    for pos in func_positions:
        result = predict_masked_position(seq, pos, model, tokenizer, device)
        if result:
            result["site_type"] = "functional"
            func_results.append(result)

    # Evaluate non-functional sites
    print(f"  Non-functional sites ({len(nonfunc_positions)} positions)...")
    nonfunc_results = []
    for pos in nonfunc_positions:
        result = predict_masked_position(seq, pos, model, tokenizer, device)
        if result:
            result["site_type"] = "nonfunctional"
            nonfunc_results.append(result)

    # Compute FSPE metrics
    func_entropies = [r["entropy"] for r in func_results]
    nonfunc_entropies = [r["entropy"] for r in nonfunc_results]

    func_accuracies = [r["is_correct"] for r in func_results]
    nonfunc_accuracies = [r["is_correct"] for r in nonfunc_results]

    func_correct_probs = [r["correct_prob"] for r in func_results]
    nonfunc_correct_probs = [r["correct_prob"] for r in nonfunc_results]

    # Statistical test + effect size (FSPE entropy)
    if len(func_entropies) >= 2 and len(nonfunc_entropies) >= 2:
        t_stat, p_value = stats.mannwhitneyu(
            func_entropies, nonfunc_entropies, alternative="less"
        )
        n_f = len(func_entropies)
        n_nf = len(nonfunc_entropies)
        rank_biserial_r = 1 - (2 * t_stat) / (n_f * n_nf)
    else:
        t_stat, p_value = float("nan"), float("nan")
        rank_biserial_r = float("nan")

    # PLM (pseudo-log-likelihood) analysis
    func_plm = [r["plm_score"] for r in func_results]
    nonfunc_plm = [r["plm_score"] for r in nonfunc_results]

    if len(func_plm) >= 2 and len(nonfunc_plm) >= 2:
        plm_stat, plm_pval = stats.mannwhitneyu(func_plm, nonfunc_plm, alternative="greater")
        plm_r = 1 - (2 * plm_stat) / (len(func_plm) * len(nonfunc_plm))
    else:
        plm_stat, plm_pval, plm_r = float("nan"), float("nan"), float("nan")

    return {
        "uniprot_id": uniprot_id,
        "sequence_length": seq_len,
        "n_functional_sites": len(func_positions),
        "n_nonfunctional_sites": len(nonfunc_positions),
        "fspe_functional": float(np.mean(func_entropies)) if func_entropies else None,
        "fspe_nonfunctional": float(np.mean(nonfunc_entropies)) if nonfunc_entropies else None,
        "fspe_ratio": (
            float(np.mean(func_entropies) / np.mean(nonfunc_entropies))
            if func_entropies and nonfunc_entropies and np.mean(nonfunc_entropies) > 0
            else None
        ),
        "accuracy_functional": float(np.mean(func_accuracies)) if func_accuracies else None,
        "accuracy_nonfunctional": float(np.mean(nonfunc_accuracies)) if nonfunc_accuracies else None,
        "mean_correct_prob_functional": float(np.mean(func_correct_probs)) if func_correct_probs else None,
        "mean_correct_prob_nonfunctional": float(np.mean(nonfunc_correct_probs)) if nonfunc_correct_probs else None,
        "mannwhitney_stat": float(t_stat) if not np.isnan(t_stat) else None,
        "mannwhitney_pvalue": float(p_value) if not np.isnan(p_value) else None,
        "rank_biserial_r": float(rank_biserial_r) if not np.isnan(rank_biserial_r) else None,
        "plm_functional_mean": float(np.mean(func_plm)) if func_plm else None,
        "plm_nonfunctional_mean": float(np.mean(nonfunc_plm)) if nonfunc_plm else None,
        "plm_delta": float(np.mean(func_plm) - np.mean(nonfunc_plm)) if (func_plm and nonfunc_plm) else None,
        "plm_mannwhitney_pvalue": float(plm_pval) if not np.isnan(plm_pval) else None,
        "plm_rank_biserial_r": float(plm_r) if not np.isnan(plm_r) else None,
        "functional_results": func_results,
        "nonfunctional_results": nonfunc_results,
    }


# ============================================================================
# Visualization
# ============================================================================


def plot_fspe_comparison(all_results: list):
    """Plot FSPE comparison across all evaluated proteins."""
    proteins = []
    func_means = []
    nonfunc_means = []

    for r in all_results:
        if r and r["fspe_functional"] is not None and r["fspe_nonfunctional"] is not None:
            proteins.append(r["uniprot_id"])
            func_means.append(r["fspe_functional"])
            nonfunc_means.append(r["fspe_nonfunctional"])

    if not proteins:
        print("No valid results to plot")
        return

    x = np.arange(len(proteins))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, func_means, width, label="Functional sites", color="#ef4444", alpha=0.8)
    bars2 = ax.bar(x + width / 2, nonfunc_means, width, label="Non-functional sites", color="#22c55e", alpha=0.8)

    ax.set_ylabel("Mean Prediction Entropy (nats)", fontsize=12)
    ax.set_title("Functional Site Prediction Entropy (FSPE)\nLower entropy = higher model confidence", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(proteins, rotation=45, ha="right")
    ax.legend(fontsize=11)

    # Add significance markers
    for i, r in enumerate(all_results):
        if r and r["mannwhitney_pvalue"] is not None and r["mannwhitney_pvalue"] < 0.05:
            max_y = max(func_means[i], nonfunc_means[i])
            ax.text(i, max_y + 0.05, "*", ha="center", fontsize=14, fontweight="bold")

    plt.tight_layout()
    path = FIGURES_DIR / "fspe_comparison.png"
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


def plot_entropy_distributions(all_results: list):
    """Plot combined entropy distributions for functional vs. non-functional sites."""
    func_entropies = []
    nonfunc_entropies = []

    for r in all_results:
        if r is None:
            continue
        func_entropies.extend([fr["entropy"] for fr in r.get("functional_results", [])])
        nonfunc_entropies.extend([nr["entropy"] for nr in r.get("nonfunctional_results", [])])

    if not func_entropies or not nonfunc_entropies:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(func_entropies, bins=20, alpha=0.6, color="#ef4444", label="Functional sites", density=True)
    ax.hist(nonfunc_entropies, bins=20, alpha=0.6, color="#22c55e", label="Non-functional sites", density=True)

    ax.axvline(np.mean(func_entropies), color="#ef4444", linestyle="--", lw=2)
    ax.axvline(np.mean(nonfunc_entropies), color="#22c55e", linestyle="--", lw=2)

    ax.set_xlabel("Prediction Entropy (nats)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("ESM-2 Masked Prediction Entropy Distribution\nFunctional vs. Non-functional Sites (All Proteins)", fontsize=13)
    ax.legend(fontsize=11)

    # Add combined stats
    stat, pval = stats.mannwhitneyu(func_entropies, nonfunc_entropies, alternative="less")
    ax.text(
        0.95, 0.95,
        f"Mann-Whitney p = {pval:.2e}\nFunc mean = {np.mean(func_entropies):.3f}\nNon-func mean = {np.mean(nonfunc_entropies):.3f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    path = FIGURES_DIR / "fspe_distributions.png"
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


def plot_plm_comparison(all_results: list):
    """Violin plot: functional vs. non-functional PLM scores pooled across all proteins."""
    func_plm = [
        fr["plm_score"]
        for r in all_results
        for fr in r.get("functional_results", [])
        if "plm_score" in fr
    ]
    nonfunc_plm = [
        nr["plm_score"]
        for r in all_results
        for nr in r.get("nonfunctional_results", [])
        if "plm_score" in nr
    ]

    if not func_plm or not nonfunc_plm:
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    data = [func_plm, nonfunc_plm]
    parts = ax.violinplot(data, positions=[1, 2], showmedians=True, showextrema=True)

    # Color
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor("#ef4444" if i == 0 else "#22c55e")
        pc.set_alpha(0.7)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Functional sites", "Non-functional sites"], fontsize=12)
    ax.set_ylabel("log P(WT aa | context)  [PLM score]", fontsize=12)
    ax.set_title("Pseudo-Log-Likelihood at Functional vs. Non-functional Sites\n(All proteins pooled)", fontsize=12)

    # Annotate with stats
    if len(func_plm) >= 2 and len(nonfunc_plm) >= 2:
        stat, pval = stats.mannwhitneyu(func_plm, nonfunc_plm, alternative="greater")
        r = 1 - (2 * stat) / (len(func_plm) * len(nonfunc_plm))
        ax.text(
            0.98, 0.02,
            f"Mann-Whitney p = {pval:.2e}\nr = {r:.3f}\n"
            f"Func mean = {np.mean(func_plm):.2f}\nNon-func mean = {np.mean(nonfunc_plm):.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    path = FIGURES_DIR / "plm_comparison.png"
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="FSPE metric computation")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print_header("Functional Site Prediction Entropy (FSPE)")
    print(f"Model: {args.model}")
    print(f"Device: {device}")

    # Load model (masked LM head needed)
    print("\nLoading ESM-2 with MLM head...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model)
    model = model.to(device)
    model.eval()

    # Load functional site annotations
    func_sites = load_functional_sites()
    positive_seqs = load_positive_sequences()

    # Build lookup: uniprot_id -> sequence
    # Extract UniProt ID from FASTA headers (format: sp|P02879|RICI_RICCO)
    seq_lookup = {}
    for seq_id, desc, seq in positive_seqs:
        # Try to extract UniProt accession from ID
        parts = seq_id.split("|")
        if len(parts) >= 2:
            accession = parts[1]
        else:
            accession = seq_id
        seq_lookup[accession] = seq

    # Evaluate each annotated protein
    all_results = []
    for uniprot_id, info in func_sites.items():
        if uniprot_id.startswith("_"):
            continue  # Skip metadata keys

        print(f"\n--- {uniprot_id}: {info['name']} ---")

        if uniprot_id not in seq_lookup:
            print(f"  Sequence not found in positive set, skipping")
            continue

        sequence = seq_lookup[uniprot_id]
        functional_residues = info["functional_sites"]["catalytic_residues"]

        result = evaluate_protein_fspe(
            uniprot_id, sequence, functional_residues, model, tokenizer, device
        )
        if result:
            all_results.append(result)
            print(f"  FSPE functional:     {result['fspe_functional']:.3f}")
            print(f"  FSPE non-functional: {result['fspe_nonfunctional']:.3f}")
            print(f"  FSPE ratio:          {result['fspe_ratio']:.3f}")
            if result["mannwhitney_pvalue"] is not None:
                print(f"  Mann-Whitney p:      {result['mannwhitney_pvalue']:.4f}")

    # Visualize
    print("\n--- Generating figures ---")
    plot_fspe_comparison(all_results)
    plot_entropy_distributions(all_results)
    plot_plm_comparison(all_results)

    # Save results (strip per-residue details for JSON)
    save_results = []
    for r in all_results:
        r_copy = {k: v for k, v in r.items() if k not in ("functional_results", "nonfunctional_results")}
        save_results.append(r_copy)

    # Pooled meta-analysis across all proteins (FSPE entropy)
    all_func_pooled = [
        fr["entropy"]
        for r in all_results
        for fr in r.get("functional_results", [])
    ]
    all_nonfunc_pooled = [
        nr["entropy"]
        for r in all_results
        for nr in r.get("nonfunctional_results", [])
    ]
    pooled_meta = {}
    if len(all_func_pooled) >= 2 and len(all_nonfunc_pooled) >= 2:
        pool_stat, pool_p = stats.mannwhitneyu(
            all_func_pooled, all_nonfunc_pooled, alternative="less"
        )
        pool_r = 1 - (2 * pool_stat) / (len(all_func_pooled) * len(all_nonfunc_pooled))
        pooled_meta = {
            "n_functional": len(all_func_pooled),
            "n_nonfunctional": len(all_nonfunc_pooled),
            "mean_func_entropy": float(np.mean(all_func_pooled)),
            "mean_nonfunc_entropy": float(np.mean(all_nonfunc_pooled)),
            "mannwhitney_stat": float(pool_stat),
            "mannwhitney_pvalue": float(pool_p),
            "rank_biserial_r": float(pool_r),
        }
        print(f"\n--- Pooled FSPE Meta-Analysis (n={len(all_func_pooled)} func, n={len(all_nonfunc_pooled)} non-func) ---")
        print(f"  Mann-Whitney U p = {pool_p:.2e}   rank-biserial r = {pool_r:.3f}")

    # Pooled PLM meta-analysis
    all_func_plm = [
        fr["plm_score"]
        for r in all_results
        for fr in r.get("functional_results", [])
        if "plm_score" in fr
    ]
    all_nonfunc_plm = [
        nr["plm_score"]
        for r in all_results
        for nr in r.get("nonfunctional_results", [])
        if "plm_score" in nr
    ]
    pooled_plm_meta = {}
    if len(all_func_plm) >= 2 and len(all_nonfunc_plm) >= 2:
        plm_pool_stat, plm_pool_p = stats.mannwhitneyu(
            all_func_plm, all_nonfunc_plm, alternative="greater"
        )
        plm_pool_r = 1 - (2 * plm_pool_stat) / (len(all_func_plm) * len(all_nonfunc_plm))
        pooled_plm_meta = {
            "n_functional": len(all_func_plm),
            "n_nonfunctional": len(all_nonfunc_plm),
            "mean_func_plm": float(np.mean(all_func_plm)),
            "mean_nonfunc_plm": float(np.mean(all_nonfunc_plm)),
            "plm_delta": float(np.mean(all_func_plm) - np.mean(all_nonfunc_plm)),
            "mannwhitney_pvalue": float(plm_pool_p),
            "rank_biserial_r": float(plm_pool_r),
        }
        print(f"\n--- Pooled PLM Meta-Analysis (n={len(all_func_plm)} func, n={len(all_nonfunc_plm)} non-func) ---")
        print(f"  PLM Mann-Whitney U p = {plm_pool_p:.2e}   rank-biserial r = {plm_pool_r:.3f}")
        print(f"  Mean func PLM: {np.mean(all_func_plm):.3f}   Mean non-func PLM: {np.mean(all_nonfunc_plm):.3f}")

    # Save pooled meta with results
    output = {
        "per_protein": save_results,
        "pooled_meta_analysis": pooled_meta,
        "pooled_plm_meta_analysis": pooled_plm_meta,
    }
    results_path = RESULTS_DIR / "fspe_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Summary
    print_header("FSPE Summary")
    for r in all_results:
        ratio_str = f"{r['fspe_ratio']:.3f}" if r["fspe_ratio"] else "N/A"
        r_str = f"  r={r['rank_biserial_r']:.3f}" if r.get("rank_biserial_r") is not None else ""
        sig = "*" if r.get("mannwhitney_pvalue") and r["mannwhitney_pvalue"] < 0.05 else ""
        print(f"  {r['uniprot_id']}: FSPE ratio = {ratio_str}{r_str} {sig}")

    # Aggregate
    ratios = [r["fspe_ratio"] for r in all_results if r["fspe_ratio"] is not None]
    if ratios:
        mean_ratio = np.mean(ratios)
        print(f"\n  Mean FSPE ratio across proteins: {mean_ratio:.3f}")
        if mean_ratio < 1.0:
            print("  Interpretation: ESM-2 is MORE confident at functional sites")
            print("  → The model encodes functional specificity relevant to dual-use risk")
        else:
            print("  Interpretation: ESM-2 is NOT more confident at functional sites")
            print("  → Functional knowledge may not be specifically encoded")

    print(f"\nNext step: python src/05_esm2_nearest_neighbor.py")


if __name__ == "__main__":
    main()
