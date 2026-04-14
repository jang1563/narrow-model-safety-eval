#!/bin/bash
#SBATCH --job-name=esm2_fspe
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/esm2_fspe_%j.log

# ============================================================================
# FSPE Metric Computation — GPU Job
# Runs: 04_esm2_masked_prediction.py
# ============================================================================

echo "=== FSPE Computation Job ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

SCRATCH="${SCRATCH:-/path/to/scratch/narrow_model_safety_eval}"
PROJECT_DIR="${SCRATCH}/Narrow_Model_Safety_Eval"
mkdir -p ${SCRATCH}/logs

source ~/miniconda3/miniconda3/etc/profile.d/conda.sh
conda activate narrow_model_safety

export HF_HOME="${SCRATCH}/hf_cache"
export TRANSFORMERS_CACHE="${SCRATCH}/hf_cache"

cd ${PROJECT_DIR}
python src/04_esm2_masked_prediction.py \
    --model facebook/esm2_t33_650M_UR50D \
    --device cuda

echo "=== Done: $(date) ==="
