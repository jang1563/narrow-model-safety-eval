#!/bin/bash
#SBATCH --job-name=sae_fhs
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/sae_%j.log

# ============================================================================
# SAE Feature Hazard Score — GPU Job (v2 Pillar 2)
# Runs: 15_sae_fhs.py
#
# Computes FHS (Feature Hazard Score) by:
#   1. Running ESM-2 layer-33 forward passes on all panel proteins
#   2. Training or loading a sparse autoencoder (SAE) on the residual stream
#   3. Extracting catalytic feature vectors and computing FHS vs toxic/benign
#   4. Correlating FHS with FSI (from prior v1 + v2 results)
#
# InterPLM SAE weights preferred; trains 4096-dim SAE fallback if unavailable.
# ESM-2 650M: ~1.3GB VRAM; SAE training: ~2GB; total ~4GB. A40 is sufficient.
# ============================================================================

echo "=== SAE Feature Hazard Score Job ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

SCRATCH="${SCRATCH:?ERROR: set the SCRATCH environment variable to your cluster scratch path}"
PROJECT_DIR="${SCRATCH}/Narrow_Model_Safety_Eval"

mkdir -p ${PROJECT_DIR}/logs

source ~/miniconda3/miniconda3/etc/profile.d/conda.sh
conda activate narrow_model_safety

cd ${PROJECT_DIR}

export HF_HOME="${SCRATCH}/hf_cache"

python src/15_sae_fhs.py \
    --device cuda \
    --sae_dim 4096 \
    --n_epochs 50

echo "=== Done: $(date) ==="
echo "Results: ${PROJECT_DIR}/results/fhs_results.json"
