#!/bin/bash
#SBATCH --job-name=cpu_analysis
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/cpu_analysis_%j.log

# ============================================================================
# CPU Analyses — Separability + Nearest Neighbor
# Runs after ESM-2 embeddings are extracted
# Runs: 03_esm2_separability.py, 05_esm2_nearest_neighbor.py
# ============================================================================

echo "=== CPU Analysis Job ==="
echo "Date: $(date)"
echo "Node: $(hostname)"

SCRATCH="${SCRATCH:-/path/to/scratch/narrow_model_safety_eval}"
PROJECT_DIR="${SCRATCH}/Narrow_Model_Safety_Eval"
mkdir -p ${SCRATCH}/logs

source ~/miniconda3/miniconda3/etc/profile.d/conda.sh
conda activate narrow_model_safety

cd ${PROJECT_DIR}

echo ""
echo "--- Separability Analysis ---"
python src/03_esm2_separability.py

echo ""
echo "--- Nearest Neighbor Analysis ---"
python src/05_esm2_nearest_neighbor.py

echo "=== Done: $(date) ==="
