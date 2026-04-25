#!/bin/bash
#SBATCH --job-name=lmpnn_fsi
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=logs/ligandmpnn_%j.log

# ============================================================================
# LigandMPNN FSI Evaluation — GPU Job (v2 Pillar 1A)
# Runs: 12_ligandmpnn_fsi.py
# Compare FSI-LM vs FSI-PM (in fsi_results.json from v1 proteinmpnn.sh)
# Hypothesis: FSI-LM >> FSI-PM for BoNT-A (3BTA) zinc-binding active site
# ============================================================================

echo "=== LigandMPNN FSI Job ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

SCRATCH="${SCRATCH:?ERROR: set the SCRATCH environment variable to your cluster scratch path}"
PROJECT_DIR="${SCRATCH}/Narrow_Model_Safety_Eval"
LIGANDMPNN_DIR="${SCRATCH}/LigandMPNN"

mkdir -p ${SCRATCH}/logs
mkdir -p ${PROJECT_DIR}/logs

source ~/miniconda3/miniconda3/etc/profile.d/conda.sh
conda activate narrow_model_safety

# Clone LigandMPNN if not present
if [ ! -d "${LIGANDMPNN_DIR}" ]; then
    echo "Cloning LigandMPNN..."
    git clone https://github.com/dauparas/LigandMPNN.git ${LIGANDMPNN_DIR}
fi

# Download LigandMPNN model weights if not present
# Weights are stored under LigandMPNN/model_params/
if [ ! -f "${LIGANDMPNN_DIR}/model_params/ligandmpnn_v_32_010_25.pt" ]; then
    echo "Downloading LigandMPNN weights..."
    cd ${LIGANDMPNN_DIR}
    bash get_model_params.sh "${LIGANDMPNN_DIR}/model_params"
    cd ${PROJECT_DIR}
fi

cd ${PROJECT_DIR}

# Run LigandMPNN FSI evaluation
# --model_type ligand_mpnn: uses HETATM metal/ligand context from PDB files
# PDB files in data/structures/ should retain HETATM records (zinc for 3BTA)
python src/12_ligandmpnn_fsi.py \
    --ligandmpnn_dir ${LIGANDMPNN_DIR} \
    --num_seqs 100 \
    --temperature 0.1 \
    --model_type ligand_mpnn

echo "=== Done: $(date) ==="
echo "Results: ${PROJECT_DIR}/results/fsi_ligandmpnn_results.json"
