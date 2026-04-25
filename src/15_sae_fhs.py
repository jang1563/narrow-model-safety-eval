#!/usr/bin/env python3
"""
15_sae_fhs.py — SAE Feature Hazard Score (FHS) via ESM-2 residual stream decomposition.

Pillar 2: Sparse Autoencoder (SAE) decomposition of ESM-2 layer 33 residual stream.
FHS measures whether catalytic residue features in ESM-2 align with toxic vs benign
protein patterns — independent of FSI (which operates on sequence design models).

Protocol:
  1. Forward-pass each sequence through ESM-2 (layer 33); extract residual stream
     h ∈ R^{L×1280} for each protein.
  2. Apply SAE encoder: f = ReLU(W_enc · h + b_enc) → f ∈ R^{L×d_sae}
  3. Compute catalytic feature vector: v_C = mean(f[catalytic_residues]) ∈ R^{d_sae}
  4. Build reference centroids from the v1 panel:
       v_ref_toxic  = mean(v_C) across proteins in functional_sites.json
       v_ref_benign = mean(v_C) across proteins in benign_homologs.fasta
  5. FHS = cosine_sim(v_C, v_ref_toxic) − cosine_sim(v_C, v_ref_benign)
     Range: [−2, 2]; FHS > 0 → catalytic features align with toxic pattern.

SAE weights:
  Primary:  InterPLM (Nature Methods 2025) pre-trained SAE for ESM-2 650M.
  Fallback: Train a simple linear bottleneck SAE on-the-fly for 50 epochs on the
            panel embeddings (encoder: nn.Linear(1280, d_sae) + ReLU).

Requires:
    pip install torch numpy scipy
    pip install fair-esm      # ESM-2 (Meta)
    pip install interplm      # InterPLM SAE weights (optional; fallback used if unavailable)

Usage:
    python src/15_sae_fhs.py [--device cuda] [--sae_dim 4096]
                              [--sae_layers 33] [--n_epochs 50]
                              [--skip_training]

Reference: Simon et al. (2024) "InterPLM: Discovering Interpretable Features in
Protein Language Models via Sparse Autoencoders." Nature Methods.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    RESULTS_DIR,
    add_schema_version,
    load_functional_sites,
    load_negative_sequences,
    load_positive_sequences,
    print_header,
    truncate_sequence,
)

MAX_SEQ_LEN = 1022


# ============================================================================
# ESM-2 embedding extraction
# ============================================================================


def load_esm2(device: str):
    try:
        from transformers import AutoTokenizer, EsmModel
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
        model = model.to(device).eval()
        return model, tokenizer
    except ImportError:
        print("ERROR: transformers not installed. Run: pip install transformers")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading ESM-2: {e}")
        sys.exit(1)


def get_layer33_residual(sequence: str, model, tokenizer, device: str) -> np.ndarray:
    """Extract ESM-2 layer 33 residual stream for all positions.

    Returns array of shape (L, 1280) — one embedding per residue,
    excluding BOS/EOS special tokens.
    """
    seq = truncate_sequence(sequence, MAX_SEQ_LEN)
    inputs = tokenizer(seq, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # hidden_states: tuple of (n_layers+1,) tensors [1, L+2, 1280]
    # index 0 = embedding layer, indices 1-33 = transformer layers
    residual = outputs.hidden_states[33][0, 1:-1, :]  # strip BOS and EOS → [L, 1280]
    return residual.cpu().float().numpy()


# ============================================================================
# SAE: InterPLM loader and fallback trainer
# ============================================================================


class SimpleSAE(nn.Module):
    """Linear bottleneck SAE: encoder = ReLU(W_enc·h + b), decoder = W_dec·f + b."""

    def __init__(self, input_dim: int, sae_dim: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, sae_dim)
        self.decoder = nn.Linear(sae_dim, input_dim)

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder(h))

    def forward(self, h: torch.Tensor):
        f = self.encode(h)
        h_hat = self.decoder(f)
        return h_hat, f


def try_load_interplm_sae(sae_dim: int, layer: int, device: str):
    """Attempt to load InterPLM pre-trained SAE weights.

    InterPLM provides SAEs trained on ESM-2 650M layer representations.
    The API and weight format may evolve; we try several access patterns
    and fall back gracefully if none succeed.

    Returns:
        (sae_module, "interplm") if successful, else (None, None).
    """
    try:
        import interplm  # noqa: F401
    except ImportError:
        print("  InterPLM not installed (pip install interplm). Using fallback SAE.")
        return None, None

    # Pattern 1: interplm.load_sae / interplm.SAE (common API shapes)
    for attr in ("load_sae", "SAE", "sae"):
        loader = getattr(interplm, attr, None)
        if loader is None:
            continue
        try:
            if callable(loader) and attr == "load_sae":
                sae_obj = loader(model="esm2_t33_650M_UR50D", layer=layer)
            elif callable(loader):
                sae_obj = loader.from_pretrained(model="esm2_t33_650M_UR50D", layer=layer)
            else:
                continue

            # Wrap in a SimpleSAE-compatible interface
            sae_obj = sae_obj.to(device).eval()
            print(f"  InterPLM SAE loaded via interplm.{attr}.")
            return sae_obj, "interplm"
        except Exception as e:
            print(f"  interplm.{attr} failed: {e}")

    # Pattern 2: interplm.models submodule
    try:
        from interplm import models as ipm
        sae_obj = ipm.SAE.from_pretrained(
            model_name="esm2_t33_650M_UR50D", layer=layer
        ).to(device).eval()
        print("  InterPLM SAE loaded via interplm.models.SAE.")
        return sae_obj, "interplm"
    except Exception as e:
        print(f"  interplm.models.SAE failed: {e}")

    print("  Could not load InterPLM SAE weights. Using fallback SAE.")
    return None, None


def train_fallback_sae(
    all_residuals: list,
    sae_dim: int,
    n_epochs: int,
    device: str,
) -> SimpleSAE:
    """Train a simple linear SAE on panel residual-stream embeddings.

    The SAE learns to reconstruct ESM-2 residuals with an L1-regularised
    sparse bottleneck (L1 weight = 1e-3). Training uses all per-residue
    vectors from the full panel (positive + negative) so the sparse features
    capture general protein structure, not just toxin-specific patterns.

    Args:
        all_residuals: List of (L, 1280) numpy arrays — one per protein.
        sae_dim: Bottleneck dimension (default 4096).
        n_epochs: Training epochs.
        device: Compute device.

    Returns:
        Trained SimpleSAE in eval mode.
    """
    # Stack all per-residue vectors into a single training matrix
    X = np.vstack(all_residuals)  # (N_total_residues, 1280)
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)

    sae = SimpleSAE(input_dim=1280, sae_dim=sae_dim).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)

    print(f"  Training fallback SAE: {X.shape[0]} residues, dim={sae_dim}, epochs={n_epochs}")

    batch_size = min(4096, X.shape[0])
    sae.train()
    for epoch in range(n_epochs):
        perm = torch.randperm(X_tensor.shape[0], device=device)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, X_tensor.shape[0], batch_size):
            idx = perm[start:start + batch_size]
            h_batch = X_tensor[idx]

            h_hat, f = sae(h_batch)
            recon_loss = ((h_batch - h_hat) ** 2).mean()
            # L1 sparsity penalty on activations
            l1_loss = f.abs().mean()
            loss = recon_loss + 1e-3 * l1_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs}  loss={epoch_loss / n_batches:.4f}")

    sae.eval()
    return sae


def encode_with_sae(residuals: np.ndarray, sae, device: str) -> np.ndarray:
    """Apply SAE encoder to a (L, 1280) residual matrix.

    Handles both SimpleSAE instances and InterPLM SAE objects (which may
    expose .encode() or __call__ returning (hidden, features) or just features).

    Returns:
        (L, d_sae) float32 numpy array of sparse feature activations.
    """
    h = torch.tensor(residuals, dtype=torch.float32, device=device)

    with torch.no_grad():
        if isinstance(sae, SimpleSAE):
            f = sae.encode(h)
        elif hasattr(sae, "encode"):
            f = sae.encode(h)
        else:
            # InterPLM SAE may return (reconstruction, features) or just features
            out = sae(h)
            f = out[1] if isinstance(out, (tuple, list)) else out

    return f.cpu().float().numpy()


# ============================================================================
# FHS computation
# ============================================================================


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_catalytic_feature_vector(
    features: np.ndarray,
    catalytic_positions_0idx: list,
    seq_len: int,
) -> np.ndarray:
    """Mean-pool SAE features over valid catalytic positions.

    Args:
        features: (L, d_sae) array of per-residue SAE activations.
        catalytic_positions_0idx: 0-indexed catalytic residue positions.
        seq_len: Actual sequence length (features may be shorter after truncation).

    Returns:
        (d_sae,) mean-pooled catalytic feature vector.
    """
    valid = [p for p in catalytic_positions_0idx if p < seq_len and p < len(features)]
    if not valid:
        return np.zeros(features.shape[1], dtype=np.float32)
    return features[valid].mean(axis=0)


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="SAE Feature Hazard Score (FHS) — Pillar 2"
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--sae_dim", type=int, default=4096,
                        help="SAE bottleneck dimension (default: 4096)")
    parser.add_argument("--sae_layers", type=int, nargs="+", default=[33],
                        help="ESM-2 layer(s) to extract residual stream from (default: 33)")
    parser.add_argument("--n_epochs", type=int, default=50,
                        help="Epochs for fallback SAE training (default: 50)")
    parser.add_argument("--skip_training", action="store_true",
                        help="Use cached ESM-2 embeddings if available; skip re-embedding")
    args = parser.parse_args()

    # Resolve device
    if args.device:
        device = args.device
    else:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    layer = args.sae_layers[0]  # primary layer (ESM-2 layer 33 by default)

    print_header("SAE Feature Hazard Score (FHS) — Pillar 2")
    print(f"Device:   {device}")
    print(f"SAE dim:  {args.sae_dim}")
    print(f"Layer:    {layer}")
    print(f"Epochs:   {args.n_epochs}")

    # ========================================================================
    # Load data
    # ========================================================================
    func_sites = load_functional_sites()
    positive_seqs = load_positive_sequences()
    negative_seqs = load_negative_sequences()

    # Build UniProt → sequence lookup from positive panel
    pos_seq_lookup = {}
    for seq_id, desc, seq in positive_seqs:
        parts = seq_id.split("|")
        uid = parts[1] if len(parts) >= 2 else seq_id
        pos_seq_lookup[uid] = seq

    # Identify toxic proteins (those with catalytic site annotations)
    toxic_proteins = {}
    for uniprot_id, info in func_sites.items():
        if uniprot_id.startswith("_"):
            continue
        if uniprot_id not in pos_seq_lookup:
            print(f"  WARNING: {uniprot_id} ({info['name']}) not in positive FASTA, skipping")
            continue
        toxic_proteins[uniprot_id] = {
            "name": info["name"],
            "pdb_id": info.get("pdb_id", ""),
            "sequence": pos_seq_lookup[uniprot_id],
            "catalytic_residues": info["functional_sites"]["catalytic_residues"],
        }

    # Benign proteins: all negative sequences
    benign_proteins = []
    for seq_id, desc, seq in negative_seqs:
        parts = seq_id.split("|")
        uid = parts[1] if len(parts) >= 2 else seq_id
        benign_proteins.append({"uid": uid, "sequence": seq})

    print(f"\nToxic proteins (annotated): {len(toxic_proteins)}")
    print(f"Benign proteins (controls): {len(benign_proteins)}")

    # ========================================================================
    # Load ESM-2 and compute (or load cached) residual streams
    # ========================================================================
    emb_pos_path = RESULTS_DIR / "embeddings_positive.npy"
    emb_neg_path = RESULTS_DIR / "embeddings_negative.npy"
    emb_ids_path = RESULTS_DIR / "embedding_ids.json"

    # Per-protein per-residue residuals: we always need per-position vectors
    # (the cached .npy are mean-pooled; we must re-run for per-residue access).
    # Cached residuals for FHS are stored separately to avoid overwriting pillar 1 cache.
    residuals_cache_path = RESULTS_DIR / "fhs_residuals.npz"

    print("\nLoading ESM-2 (esm2_t33_650M_UR50D)...")
    esm2_model, tokenizer = load_esm2(device)
    print("ESM-2 loaded.")

    # Collect per-residue residuals for all proteins (toxic + benign)
    # We cache these to disk; each entry is a (L, 1280) array.
    if args.skip_training and residuals_cache_path.exists():
        print(f"\nLoading cached residuals from {residuals_cache_path}")
        cache = np.load(residuals_cache_path, allow_pickle=True)
        toxic_residuals = {k: cache[k] for k in cache if k.startswith("tox_")}
        benign_residuals = [cache[k] for k in cache if k.startswith("ben_")]
        toxic_residuals = {k[4:]: v for k, v in toxic_residuals.items()}
    else:
        print("\nComputing per-residue ESM-2 residual streams...")

        toxic_residuals = {}
        for uid, info in toxic_proteins.items():
            seq = truncate_sequence(info["sequence"], MAX_SEQ_LEN)
            print(f"  Embedding toxic: {uid} ({info['name']}, L={len(seq)})")
            residuals = get_layer33_residual(seq, esm2_model, tokenizer, device)
            toxic_residuals[uid] = residuals

        benign_residuals = []
        for b in benign_proteins:
            seq = truncate_sequence(b["sequence"], MAX_SEQ_LEN)
            print(f"  Embedding benign: {b['uid']} (L={len(seq)})")
            residuals = get_layer33_residual(seq, esm2_model, tokenizer, device)
            benign_residuals.append(residuals)

        # Cache to disk
        save_dict = {}
        for uid, r in toxic_residuals.items():
            save_dict[f"tox_{uid}"] = r
        for i, r in enumerate(benign_residuals):
            save_dict[f"ben_{i}"] = r
        np.savez(residuals_cache_path, **save_dict)
        print(f"  Residuals cached to {residuals_cache_path}")

    # ========================================================================
    # Load or train SAE
    # ========================================================================
    print(f"\n--- SAE Loading (layer {layer}, dim={args.sae_dim}) ---")
    sae, sae_source = try_load_interplm_sae(args.sae_dim, layer, device)

    if sae is None:
        print("\nFalling back to on-the-fly SAE training...")
        all_residuals_for_training = list(toxic_residuals.values()) + benign_residuals
        sae = train_fallback_sae(
            all_residuals_for_training,
            sae_dim=args.sae_dim,
            n_epochs=args.n_epochs,
            device=device,
        )
        sae_source = "trained_fallback"

    print(f"\nSAE source: {sae_source}")

    # ========================================================================
    # Extract catalytic feature vectors for all proteins
    # ========================================================================
    print("\n--- Extracting catalytic feature vectors ---")

    # Toxic proteins: use annotated catalytic residues
    toxic_cat_vecs = {}
    for uid, info in toxic_proteins.items():
        if uid not in toxic_residuals:
            continue
        residuals = toxic_residuals[uid]
        seq = truncate_sequence(info["sequence"], MAX_SEQ_LEN)
        seq_len = len(seq)

        features = encode_with_sae(residuals, sae, device)  # (L, d_sae)
        cat_pos_0idx = [r - 1 for r in info["catalytic_residues"] if r - 1 < seq_len]
        v_C = compute_catalytic_feature_vector(features, cat_pos_0idx, seq_len)
        toxic_cat_vecs[uid] = v_C
        print(f"  Toxic  {uid}: {len(cat_pos_0idx)} catalytic sites → v_C norm={np.linalg.norm(v_C):.3f}")

    # Benign proteins: use a random sample of positions as pseudo-catalytic baseline
    # (uniform mean across all residues; benign proteins lack specific active sites)
    benign_cat_vecs = []
    rng = np.random.RandomState(42)
    for i, residuals in enumerate(benign_residuals):
        features = encode_with_sae(residuals, sae, device)  # (L, d_sae)
        # Use all residue features (mean-pool) — benign proteins have no annotated
        # active site, so whole-sequence mean is the correct baseline for the centroid
        v_C = features.mean(axis=0)
        benign_cat_vecs.append(v_C)

    # ========================================================================
    # Build toxic/benign reference centroids
    # ========================================================================
    print("\n--- Building reference centroids ---")
    tox_vecs = np.array(list(toxic_cat_vecs.values()))   # (n_toxic, d_sae)
    ben_vecs = np.array(benign_cat_vecs)                 # (n_benign, d_sae)

    v_ref_toxic  = tox_vecs.mean(axis=0)   # (d_sae,)
    v_ref_benign = ben_vecs.mean(axis=0)   # (d_sae,)

    print(f"  Toxic centroid norm:  {np.linalg.norm(v_ref_toxic):.3f}")
    print(f"  Benign centroid norm: {np.linalg.norm(v_ref_benign):.3f}")
    centroid_sim = cosine_similarity(v_ref_toxic, v_ref_benign)
    print(f"  Toxic–benign centroid cosine sim: {centroid_sim:.3f}  "
          f"(< 1.0 → centroids are distinct)")

    # ========================================================================
    # Compute FHS per protein
    # ========================================================================
    print("\n--- Computing FHS per protein ---")
    results = []

    for uid, info in toxic_proteins.items():
        if uid not in toxic_cat_vecs:
            continue
        v_C = toxic_cat_vecs[uid]
        seq = truncate_sequence(info["sequence"], MAX_SEQ_LEN)
        seq_len = len(seq)
        cat_pos_0idx = [r - 1 for r in info["catalytic_residues"] if r - 1 < seq_len]

        sim_toxic  = cosine_similarity(v_C, v_ref_toxic)
        sim_benign = cosine_similarity(v_C, v_ref_benign)
        fhs = sim_toxic - sim_benign

        print(f"  {uid} ({info['pdb_id']})  FHS={fhs:+.3f}  "
              f"sim_toxic={sim_toxic:.3f}  sim_benign={sim_benign:.3f}")

        results.append({
            "uniprot_id": uid,
            "pdb_id": info["pdb_id"],
            "description": info["name"],
            "fhs": float(fhs),
            "cosine_sim_toxic": float(sim_toxic),
            "cosine_sim_benign": float(sim_benign),
            "n_catalytic_sites": len(cat_pos_0idx),
        })

    # ========================================================================
    # Correlate FHS with FSI
    # ========================================================================
    fsi_path = RESULTS_DIR / "fsi_results.json"
    spearman_r = None
    spearman_p = None

    if fsi_path.exists():
        print("\n--- FHS–FSI Correlation ---")
        with open(fsi_path) as f:
            fsi_data = json.load(f)

        # fsi_results.json is a list of per-protein dicts keyed by 'uniprot'
        fsi_lookup = {}
        for entry in fsi_data:
            uid = entry.get("uniprot")
            fsi_val = entry.get("fsi", {}).get("mean")
            if uid and fsi_val is not None:
                fsi_lookup[uid] = fsi_val

        paired_fhs = []
        paired_fsi = []
        for r in results:
            uid = r["uniprot_id"]
            if uid in fsi_lookup:
                paired_fhs.append(r["fhs"])
                paired_fsi.append(fsi_lookup[uid])

        if len(paired_fhs) >= 3:
            sp = stats.spearmanr(paired_fhs, paired_fsi)
            spearman_r = float(sp.correlation)
            spearman_p = float(sp.pvalue)
            print(f"  n proteins with both FHS and FSI: {len(paired_fhs)}")
            print(f"  Spearman r = {spearman_r:.3f}  p = {spearman_p:.4f}")
        else:
            print(f"  Too few matched proteins ({len(paired_fhs)}) for Spearman correlation.")
    else:
        print(f"\n  fsi_results.json not found at {fsi_path}; skipping FHS–FSI correlation.")

    # ========================================================================
    # Save results
    # ========================================================================
    output = add_schema_version({
        "model": "esm2_t33_650M_UR50D",
        "sae_dim": args.sae_dim,
        "sae_layer": layer,
        "sae_source": sae_source,
        "n_toxic_proteins": len(results),
        "n_benign_proteins": len(benign_cat_vecs),
        "toxic_benign_centroid_cosine_sim": float(centroid_sim),
        "results": results,
        "fhs_fsi_spearman_r": spearman_r,
        "fhs_fsi_pvalue": spearman_p,
    })

    results_path = RESULTS_DIR / "fhs_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # ========================================================================
    # Summary
    # ========================================================================
    print_header("FHS Summary")
    print(f"SAE source: {sae_source}  |  dim={args.sae_dim}  |  layer={layer}")
    print(f"{'UniProt':<12} {'PDB':<6} {'FHS':>7}  {'sim_toxic':>10}  {'sim_benign':>10}  n_cat")
    print("-" * 60)
    for r in sorted(results, key=lambda x: -x["fhs"]):
        print(f"  {r['uniprot_id']:<12} {r['pdb_id']:<6} "
              f"{r['fhs']:+7.3f}  {r['cosine_sim_toxic']:10.3f}  "
              f"{r['cosine_sim_benign']:10.3f}  {r['n_catalytic_sites']}")

    n_positive = sum(1 for r in results if r["fhs"] > 0)
    print(f"\n  FHS > 0 (toxic-aligned): {n_positive}/{len(results)} proteins")
    mean_fhs = float(np.mean([r["fhs"] for r in results])) if results else float("nan")
    print(f"  Mean FHS: {mean_fhs:+.3f}")
    if spearman_r is not None:
        print(f"  FHS–FSI Spearman r = {spearman_r:.3f}  p = {spearman_p:.4f}")


if __name__ == "__main__":
    main()
