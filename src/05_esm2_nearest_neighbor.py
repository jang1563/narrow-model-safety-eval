#!/usr/bin/env python3
"""
05_esm2_nearest_neighbor.py — Nearest-neighbor retrieval analysis.

Tests whether ESM-2 embedding space can be used as a "toxin similarity search
engine": given a query sequence, do nearest neighbors include known toxins?

Uses pre-computed embeddings from 02_esm2_embed.py.

Produces:
  - results/nearest_neighbor_results.json
  - results/figures/nn_*.png

Usage:
    python src/05_esm2_nearest_neighbor.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from utils import FIGURES_DIR, RESULTS_DIR, print_header

# ============================================================================
# Nearest neighbor analysis
# ============================================================================


def compute_precision_at_k(
    distances: np.ndarray,
    query_labels: np.ndarray,
    db_labels: np.ndarray,
    k_values: list = [1, 3, 5, 10],
) -> dict:
    """Compute Precision@k: fraction of top-k neighbors that are positive.

    Args:
        distances: (n_queries, n_db) pairwise distance matrix
        query_labels: (n_queries,) binary labels for query sequences
        db_labels: (n_db,) binary labels for database sequences
        k_values: List of k values to evaluate

    Returns:
        dict mapping k -> precision
    """
    results = {}

    for k in k_values:
        if k > distances.shape[1]:
            continue

        precisions = []
        for i in range(distances.shape[0]):
            # Get top-k nearest neighbors (excluding self if present)
            sorted_indices = np.argsort(distances[i])
            top_k = sorted_indices[:k]
            precision = np.mean(db_labels[top_k] == 1)
            precisions.append(precision)

        results[k] = {
            "mean": float(np.mean(precisions)),
            "std": float(np.std(precisions)),
            # Broken down by query type
            "positive_queries": float(np.mean([p for p, l in zip(precisions, query_labels) if l == 1])),
            "negative_queries": float(np.mean([p for p, l in zip(precisions, query_labels) if l == 0])),
        }

    return results


def cross_retrieval_analysis(X: np.ndarray, y: np.ndarray) -> dict:
    """Leave-one-out cross-retrieval analysis.

    For each sequence, find its nearest neighbors among all OTHER sequences
    and measure whether dangerous neighbors cluster together.
    """
    n = len(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cosine distance
    distances = cdist(X_scaled, X_scaled, metric="cosine")

    # Set diagonal to infinity (don't retrieve self)
    np.fill_diagonal(distances, np.inf)

    k_values = [1, 3, 5, 10]
    # Filter k values to not exceed available neighbors
    k_values = [k for k in k_values if k < n]

    results = compute_precision_at_k(distances, y, y, k_values)
    return results


def retrieval_by_category(X: np.ndarray, y: np.ndarray, ids: dict) -> dict:
    """Analyze retrieval patterns: do dangerous sequences retrieve other
    dangerous sequences, and do benign sequences avoid dangerous ones?"""
    n_pos = len(ids["positive_ids"])
    n_neg = len(ids["negative_ids"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    distances = cdist(X_scaled, X_scaled, metric="cosine")
    np.fill_diagonal(distances, np.inf)

    # For each dangerous query, what fraction of top-5 are also dangerous?
    dangerous_retrieval = []
    for i in range(n_pos):
        top5 = np.argsort(distances[i])[:5]
        frac_dangerous = np.mean(y[top5] == 1)
        dangerous_retrieval.append(frac_dangerous)

    # For each benign query, what fraction of top-5 are dangerous?
    benign_retrieval = []
    for i in range(n_pos, n_pos + n_neg):
        top5 = np.argsort(distances[i])[:5]
        frac_dangerous = np.mean(y[top5] == 1)
        benign_retrieval.append(frac_dangerous)

    return {
        "dangerous_query_frac_dangerous_top5": {
            "mean": float(np.mean(dangerous_retrieval)),
            "std": float(np.std(dangerous_retrieval)),
        },
        "benign_query_frac_dangerous_top5": {
            "mean": float(np.mean(benign_retrieval)),
            "std": float(np.std(benign_retrieval)),
        },
    }


# ============================================================================
# Visualization
# ============================================================================


def plot_precision_at_k(precision_results: dict):
    """Plot Precision@k bar chart."""
    ks = sorted(precision_results.keys())
    pos_prec = [precision_results[k]["positive_queries"] for k in ks]
    neg_prec = [precision_results[k]["negative_queries"] for k in ks]

    x = np.arange(len(ks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, pos_prec, width, label="Dangerous queries", color="#ef4444", alpha=0.8)
    ax.bar(x + width / 2, neg_prec, width, label="Benign queries", color="#22c55e", alpha=0.8)

    ax.set_ylabel("Precision@k (fraction of neighbors that are dangerous)", fontsize=11)
    ax.set_xlabel("k (number of nearest neighbors)", fontsize=11)
    ax.set_title("ESM-2 Nearest-Neighbor Retrieval Analysis", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in ks])
    ax.legend(fontsize=11)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline")

    plt.tight_layout()
    path = FIGURES_DIR / "nn_precision_at_k.png"
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


# ============================================================================
# Main
# ============================================================================


def main():
    print_header("ESM-2 Nearest-Neighbor Retrieval Analysis")

    # Load embeddings
    pos = np.load(RESULTS_DIR / "embeddings_positive.npy")
    neg = np.load(RESULTS_DIR / "embeddings_negative.npy")
    with open(RESULTS_DIR / "embedding_ids.json") as f:
        ids = json.load(f)

    X = np.vstack([pos, neg])
    y = np.array([1] * len(pos) + [0] * len(neg))

    print(f"Total sequences: {len(X)} ({len(pos)} dangerous, {len(neg)} benign)")

    # Cross-retrieval analysis
    print("\n--- Precision@k (Leave-one-out) ---")
    precision_results = cross_retrieval_analysis(X, y)
    for k, vals in sorted(precision_results.items()):
        print(f"  Precision@{k}: {vals['mean']:.3f} ± {vals['std']:.3f}")
        print(f"    Dangerous queries → {vals['positive_queries']:.3f}")
        print(f"    Benign queries    → {vals['negative_queries']:.3f}")

    # Category-level analysis
    print("\n--- Category Retrieval Patterns ---")
    cat_results = retrieval_by_category(X, y, ids)
    print(f"  Dangerous queries, frac dangerous in top-5: "
          f"{cat_results['dangerous_query_frac_dangerous_top5']['mean']:.3f}")
    print(f"  Benign queries, frac dangerous in top-5: "
          f"{cat_results['benign_query_frac_dangerous_top5']['mean']:.3f}")

    # Visualize
    plot_precision_at_k(precision_results)

    # Save
    all_results = {
        "precision_at_k": {str(k): v for k, v in precision_results.items()},
        "category_retrieval": cat_results,
    }
    results_path = RESULTS_DIR / "nearest_neighbor_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Interpretation
    print_header("Interpretation")
    if precision_results:
        k1 = precision_results.get(1, {})
        if k1.get("positive_queries", 0) > 0.6:
            print("Dangerous proteins preferentially retrieve other dangerous proteins")
            print("→ ESM-2 embedding space clusters dual-use proteins together")
            print("→ This could be exploited as a 'toxin similarity search engine'")
        else:
            print("Dangerous proteins do not strongly cluster in ESM-2 space")
            print("→ Embedding-based toxin retrieval risk is lower")

    print(f"\nNext step: python src/06_proteinmpnn_redesign.py")


if __name__ == "__main__":
    main()
