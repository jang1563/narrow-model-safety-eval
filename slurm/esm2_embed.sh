#!/bin/bash
#SBATCH --job-name=esm2_embed
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --exclude=g0001
# NOTE: Update --output path to your scratch directory before submitting
#SBATCH --output=logs/esm2_embed_%j.log

# ============================================================================
# ESM-2 Embedding Extraction — GPU Job
# Runs: 02_esm2_embed.py
# ============================================================================

echo "=== ESM-2 Embedding Job ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Setup
SCRATCH="${SCRATCH:-/path/to/scratch/narrow_model_safety_eval}"
PROJECT_DIR="${SCRATCH}/Narrow_Model_Safety_Eval"
mkdir -p ${SCRATCH}/logs

# Activate conda environment
source ~/miniconda3/miniconda3/etc/profile.d/conda.sh
conda activate narrow_model_safety  # Create this env first

# Set HuggingFace cache to scratch (avoid filling home)
export HF_HOME="${SCRATCH}/hf_cache"
export TRANSFORMERS_CACHE="${SCRATCH}/hf_cache"
mkdir -p ${HF_HOME}

# Run embedding extraction
cd ${PROJECT_DIR}
python src/02_esm2_embed.py \
    --model facebook/esm2_t33_650M_UR50D \
    --batch_size 8 \
    --device cuda

echo "=== Done: $(date) ==="
