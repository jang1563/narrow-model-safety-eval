#!/usr/bin/env python3
"""
02_esm2_embed.py — Extract ESM-2 embeddings for all sequences.

Runs on GPU (HPC: A40/A100). Produces:
  - results/embeddings_positive.npy  (N_pos × 1280)
  - results/embeddings_negative.npy  (N_neg × 1280)
  - results/embedding_ids.json       (sequence IDs in order)

Usage:
    python src/02_esm2_embed.py [--model esm2_t33_650M_UR50D] [--batch_size 8]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    RESULTS_DIR,
    load_negative_sequences,
    load_positive_sequences,
    print_header,
    truncate_sequence,
)

# ============================================================================
# ESM-2 embedding extraction
# ============================================================================

DEFAULT_MODEL = "facebook/esm2_t33_650M_UR50D"
MAX_SEQ_LEN = 1022  # ESM-2 max is 1024 tokens (including BOS/EOS)


def load_esm2_model(model_name: str, device: str):
    """Load ESM-2 model and tokenizer."""
    print(f"Loading model: {model_name}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    # Report model size
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.1f}M")

    return model, tokenizer


def extract_embeddings(
    sequences: list,
    model,
    tokenizer,
    device: str,
    batch_size: int = 8,
    max_length: int = MAX_SEQ_LEN,
) -> np.ndarray:
    """Extract mean-pooled last-layer embeddings from ESM-2.

    Args:
        sequences: List of (id, description, sequence) tuples
        model: ESM-2 model
        tokenizer: ESM-2 tokenizer
        device: 'cuda' or 'cpu'
        batch_size: Batch size for inference
        max_length: Maximum sequence length (truncate longer)

    Returns:
        embeddings: (N, hidden_dim) numpy array
    """
    all_embeddings = []
    n_total = len(sequences)
    n_truncated = 0

    print(f"Extracting embeddings for {n_total} sequences...")
    start_time = time.time()

    for batch_start in range(0, n_total, batch_size):
        batch_end = min(batch_start + batch_size, n_total)
        batch_seqs = [
            truncate_sequence(seq[2], max_length)
            for seq in sequences[batch_start:batch_end]
        ]

        # Count truncated
        for seq in sequences[batch_start:batch_end]:
            if len(seq[2]) > max_length:
                n_truncated += 1

        # Tokenize
        inputs = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length + 2,  # +2 for BOS/EOS
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # Mean pooling over sequence length (excluding padding)
        # attention_mask: 1 for real tokens, 0 for padding
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        hidden_states = outputs.last_hidden_state
        masked_hidden = hidden_states * attention_mask
        mean_pooled = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)

        all_embeddings.append(mean_pooled.cpu().numpy())

        # Progress
        done = batch_end
        elapsed = time.time() - start_time
        rate = done / elapsed if elapsed > 0 else 0
        print(
            f"  [{done}/{n_total}] {rate:.1f} seq/s | "
            f"Elapsed: {elapsed:.0f}s",
            end="\r",
        )

    print()  # newline after progress
    elapsed = time.time() - start_time
    print(f"Embedding extraction complete: {elapsed:.1f}s total")
    if n_truncated > 0:
        print(f"  {n_truncated} sequences truncated to {max_length} residues")

    return np.vstack(all_embeddings)


def main():
    parser = argparse.ArgumentParser(description="Extract ESM-2 embeddings")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"ESM-2 model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference (default: 8)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device (default: auto-detect cuda/cpu)",
    )
    args = parser.parse_args()

    # Device selection
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print_header("ESM-2 Embedding Extraction")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")

    # Load sequences
    positive_seqs = load_positive_sequences()
    negative_seqs = load_negative_sequences()
    print(f"\nPositive (dangerous) sequences: {len(positive_seqs)}")
    print(f"Negative (benign) sequences:    {len(negative_seqs)}")

    # Load model
    model, tokenizer = load_esm2_model(args.model, device)

    # Extract embeddings
    print("\n--- Positive sequences ---")
    pos_embeddings = extract_embeddings(
        positive_seqs, model, tokenizer, device, args.batch_size
    )

    print("\n--- Negative sequences ---")
    neg_embeddings = extract_embeddings(
        negative_seqs, model, tokenizer, device, args.batch_size
    )

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pos_path = RESULTS_DIR / "embeddings_positive.npy"
    neg_path = RESULTS_DIR / "embeddings_negative.npy"
    ids_path = RESULTS_DIR / "embedding_ids.json"

    np.save(pos_path, pos_embeddings)
    np.save(neg_path, neg_embeddings)

    ids_data = {
        "positive_ids": [s[0] for s in positive_seqs],
        "negative_ids": [s[0] for s in negative_seqs],
        "model": args.model,
        "embedding_dim": int(pos_embeddings.shape[1]),
    }
    with open(ids_path, "w") as f:
        json.dump(ids_data, f, indent=2)

    print(f"\nSaved embeddings:")
    print(f"  Positive: {pos_path} — shape {pos_embeddings.shape}")
    print(f"  Negative: {neg_path} — shape {neg_embeddings.shape}")
    print(f"  IDs:      {ids_path}")
    print(f"\nNext step: python src/03_esm2_separability.py")


if __name__ == "__main__":
    main()
