#!/bin/bash
#SBATCH --job-name=fsi_agg
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/fsi_analysis_%j.log

# ============================================================================
# FSI Aggregate Analysis
# Runs after ProteinMPNN redesign completes
# Runs: 07_fsi_analysis.py
# ============================================================================

echo "=== FSI Analysis Job ==="
echo "Date: $(date)"
echo "Node: $(hostname)"

SCRATCH="${SCRATCH:-/path/to/scratch/narrow_model_safety_eval}"
PROJECT_DIR="${SCRATCH}/Narrow_Model_Safety_Eval"
mkdir -p ${SCRATCH}/logs

source ~/miniconda3/miniconda3/etc/profile.d/conda.sh
conda activate narrow_model_safety

cd ${PROJECT_DIR}
python src/07_fsi_analysis.py

echo "=== Done: $(date) ==="
