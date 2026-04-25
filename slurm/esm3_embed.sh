#!/bin/bash
#SBATCH --job-name=esm3_fspe
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/esm3_%j.log

# ============================================================================
# ESM-3 Separability + FSPE — GPU Job (v2 Pillar 1C)
# Runs: 14_esm3_separability_fspe.py
#
# Computes:
#   A) ESM-3 embedding separability (AUROC) vs. ESM-2 (v1 baseline = 0.994)
#   B) ESM-3 FSPE: masked prediction entropy at catalytic vs non-catalytic sites
#
# ESM-3 (esm3_sm_open_v1): ~1.4B parameters, ~6GB VRAM for inference.
# Memory: request 64GB system RAM; 24GB VRAM GPU preferred (A40 or A100).
#
# For SaProt: run esm3_foldseek_preprocess.sh first (CPU-only), then rerun
# this script with --with_saprot.
# ============================================================================

echo "=== ESM-3 Separability + FSPE Job ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

SCRATCH="${SCRATCH:?ERROR: set the SCRATCH environment variable to your cluster scratch path}"
PROJECT_DIR="${SCRATCH}/Narrow_Model_Safety_Eval"

mkdir -p ${PROJECT_DIR}/logs

source ~/miniconda3/miniconda3/etc/profile.d/conda.sh
conda activate narrow_model_safety

# Install ESM-3 if not present
python -c "import esm" 2>/dev/null || {
    echo "Installing ESM-3..."
    pip install esm
}

cd ${PROJECT_DIR}

# ESM-3 weights downloaded automatically (~2.4GB for sm_open_v1)
export HF_HOME="${SCRATCH}/hf_cache"
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

# Run ESM-3 separability + FSPE
python src/14_esm3_separability_fspe.py \
    --device cuda

# If SaProt tokens are available (run esm3_foldseek_preprocess.sh first):
# --use_saved_embeddings reloads the ESM-3 .npy from the first pass so the
# separability result file is written correctly (avoids empty-results overwrite
# that occurred when --skip_separability was used here previously).
if [ -f "${PROJECT_DIR}/data/annotations/saprot_tokens.json" ]; then
    echo "=== SaProt tokens found. Running SaProt analysis... ==="
    python src/14_esm3_separability_fspe.py \
        --device cuda \
        --with_saprot \
        --use_saved_embeddings
else
    echo "=== SaProt tokens not found. Skipping SaProt. ==="
    echo "    Run: sbatch slurm/esm3_foldseek_preprocess.sh"
fi

echo "=== Done: $(date) ==="
echo "Results:"
echo "  Separability: ${PROJECT_DIR}/results/esm3_separability_results.json"
echo "  FSPE:         ${PROJECT_DIR}/results/esm3_fspe_results.json"
echo "  Embeddings:   ${PROJECT_DIR}/results/esm3_embeddings_*.npy"
