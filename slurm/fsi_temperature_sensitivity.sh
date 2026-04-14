#!/bin/bash
#SBATCH --job-name=fsi_temp_sweep
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --exclude=g0001
#SBATCH --output=logs/fsi_temp_sweep_%j.log

# ============================================================================
# FSI Temperature Sensitivity — GPU Job
# Runs: 10_fsi_temperature_sensitivity.py
# Tests ProteinMPNN at T=0.05,0.1,0.2,0.3 for 3BTA and 2AAI (n=100 each)
# Runtime: ~160 min (4 temps x 2 proteins x ~20 min each)
# ============================================================================

echo "=== FSI Temperature Sensitivity Job ==="
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
python src/10_fsi_temperature_sensitivity.py \
    --proteinmpnn_dir ${PROTEINMPNN_DIR} \
    --temperatures 0.05,0.1,0.2,0.3 \
    --num_seqs 100

echo "=== Done: $(date) ==="
