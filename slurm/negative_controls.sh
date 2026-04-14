#!/bin/bash
#SBATCH --job-name=neg_controls
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --exclude=g0001
#SBATCH --output=logs/neg_controls_%j.log

# ============================================================================
# Negative Controls FSI Analysis — GPU Job
# Runs: 09_negative_controls.py
# Downloads 1AST/1QD2/1LYZ, runs ProteinMPNN, computes FSI, produces figure
# ============================================================================

echo "=== Negative Controls FSI Job ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

SCRATCH="${SCRATCH:-/path/to/scratch/narrow_model_safety_eval}"
PROJECT_DIR="${SCRATCH}/Narrow_Model_Safety_Eval"
PROTEINMPNN_DIR="${SCRATCH}/ProteinMPNN"
mkdir -p ${SCRATCH}/logs

source ~/miniconda3/miniconda3/etc/profile.d/conda.sh
conda activate narrow_model_safety

# Clone ProteinMPNN if not present
if [ ! -d "${PROTEINMPNN_DIR}" ]; then
    echo "Cloning ProteinMPNN..."
    git clone https://github.com/dauparas/ProteinMPNN.git ${PROTEINMPNN_DIR}
fi

cd ${PROJECT_DIR}
python src/09_negative_controls.py \
    --proteinmpnn_dir ${PROTEINMPNN_DIR} \
    --num_seqs 100 \
    --temperature 0.1

echo "=== Done: $(date) ==="
