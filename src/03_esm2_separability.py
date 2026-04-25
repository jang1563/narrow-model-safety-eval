#!/usr/bin/env python3
"""
03_esm2_separability.py — Test whether ESM-2 embeddings can distinguish
                          dangerous from benign proteins.

Uses pre-computed embeddings from 02_esm2_embed.py.
Trains a logistic regression classifier with 5-fold cross-validation.

Produces:
  - AUROC ± SD (cross-validated)
  - Per-category breakdown
  - UMAP/t-SNE visualization
  - results/separability_results.json
  - results/figures/separability_*.png

Usage:
    python src/03_esm2_separability.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from utils import FIGURES_DIR, RESULTS_DIR, print_header

# ============================================================================
# Load embeddings
# ============================================================================


def load_embeddings():
    """Load pre-computed ESM-2 embeddings."""
    pos = np.load(RESULTS_DIR / "embeddings_positive.npy")
    neg = np.load(RESULTS_DIR / "embeddings_negative.npy")

    with open(RESULTS_DIR / "embedding_ids.json") as f:
        ids = json.load(f)

    X = np.vstack([pos, neg])
    y = np.array([1] * len(pos) + [0] * len(neg))

    print(f"Positive (dangerous): {pos.shape}")
    print(f"Negative (benign):    {neg.shape}")
    print(f"Combined: {X.shape}")
    print(f"Embedding dim: {ids['embedding_dim']}")

    return X, y, ids


# ============================================================================
# Classification analysis
# ============================================================================


def run_separability_analysis(X: np.ndarray, y: np.ndarray) -> dict:
    """Run cross-validated logistic regression for separability."""
    print("\n--- Logistic Regression (5-fold CV) ---")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Cross-validated AUROC
    auroc_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="roc_auc")
    print(f"AUROC: {auroc_scores.mean():.3f} ± {auroc_scores.std():.3f}")
    print(f"  Per-fold: {[f'{s:.3f}' for s in auroc_scores]}")

    # Cross-validated accuracy
    acc_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="accuracy")
    print(f"Accuracy: {acc_scores.mean():.3f} ± {acc_scores.std():.3f}")

    # Cross-validated predictions for ROC curve
    y_proba = cross_val_predict(
        clf, X_scaled, y, cv=cv, method="predict_proba"
    )[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=["Benign", "Dangerous"]))

    # Overall ROC curve
    fpr, tpr, _ = roc_curve(y, y_proba)

    return {
        "auroc_mean": float(auroc_scores.mean()),
        "auroc_std": float(auroc_scores.std()),
        "auroc_per_fold": auroc_scores.tolist(),
        "accuracy_mean": float(acc_scores.mean()),
        "accuracy_std": float(acc_scores.std()),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }


# ============================================================================
# Visualization
# ============================================================================


def plot_roc_curve(results: dict):
    """Plot ROC curve."""
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(
        results["fpr"],
        results["tpr"],
        color="#2563eb",
        lw=2,
        label=f'ESM-2 LogReg (AUROC = {results["auroc_mean"]:.3f} ± {results["auroc_std"]:.3f})',
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ESM-2 Embedding Separability:\nDangerous vs. Benign Proteins", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    plt.tight_layout()
    path = FIGURES_DIR / "separability_roc.png"
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


def plot_embedding_tsne(X: np.ndarray, y: np.ndarray):
    """Plot t-SNE visualization of embeddings."""
    print("\nComputing t-SNE (this may take a moment)...")

    # PCA first for stability
    pca = PCA(n_components=min(50, X.shape[1], X.shape[0] - 1))
    X_pca = pca.fit_transform(X)

    # Adjust perplexity for small datasets
    perplexity = min(30, len(X) // 4)
    if perplexity < 5:
        perplexity = 5

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        max_iter=1000,
    )
    X_2d = tsne.fit_transform(X_pca)

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["#22c55e", "#ef4444"]
    labels = ["Benign", "Dangerous"]
    for label_val, color, label in zip([0, 1], colors, labels):
        mask = y == label_val
        ax.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            c=color,
            s=40,
            alpha=0.7,
            label=label,
            edgecolors="white",
            linewidths=0.5,
        )

    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.set_title("ESM-2 Embedding Space: Dangerous vs. Benign Proteins", fontsize=13)
    ax.legend(fontsize=11)

    plt.tight_layout()
    path = FIGURES_DIR / "separability_tsne.png"
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


# ============================================================================
# Main
# ============================================================================


def main():
    print_header("ESM-2 Separability Analysis")

    # Load
    X, y, ids = load_embeddings()

    # Classify
    results = run_separability_analysis(X, y)

    # Visualize
    plot_roc_curve(results)
    plot_embedding_tsne(X, y)

    # Save results
    results_path = RESULTS_DIR / "separability_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Summary
    print_header("Summary")
    auroc = results["auroc_mean"]
    if auroc > 0.8:
        interpretation = (
            "ESM-2's representations encode functional danger — a simple "
            "classifier on ESM-2 embeddings can identify dual-use proteins "
            f"with AUROC {auroc:.3f}, meaning the model's internal "
            "representation contains safety-relevant information."
        )
    elif auroc > 0.6:
        interpretation = (
            "ESM-2 shows moderate ability to separate dangerous from benign "
            f"proteins (AUROC {auroc:.3f}), suggesting partial encoding of "
            "functional danger in its representation space."
        )
    else:
        interpretation = (
            "ESM-2 does not inherently separate dangerous from benign "
            f"proteins in embedding space (AUROC {auroc:.3f}), suggesting "
            "that toxicity is an emergent property requiring specialized "
            "evaluation beyond representation similarity."
        )

    print(interpretation)
    print(f"\nNext step: python src/04_esm2_masked_prediction.py")


if __name__ == "__main__":
    main()
