#!/bin/bash
#SBATCH --job-name=download_structs
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/download_structs_%j.log

# ============================================================================
# Download new PDB structures and UniProt sequences
# Runs: 01_collect_data.py (idempotent — skips already-downloaded files)
# ============================================================================

echo "=== Download Structures Job ==="
echo "Date: $(date)"
echo "Node: $(hostname)"

SCRATCH="${SCRATCH:-/path/to/scratch/narrow_model_safety_eval}"
PROJECT_DIR="${SCRATCH}/Narrow_Model_Safety_Eval"
mkdir -p ${SCRATCH}/logs

source ~/miniconda3/miniconda3/etc/profile.d/conda.sh
conda activate narrow_model_safety

cd ${PROJECT_DIR}
python src/01_collect_data.py

echo "=== Done: $(date) ==="
