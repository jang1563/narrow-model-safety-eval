#!/bin/bash
#SBATCH --job-name=esmif1_val
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --exclude=g0001
#SBATCH --output=logs/esmfold_val_%j.log

# ============================================================================
# ESM-IF1 Structural Compatibility Validation — GPU Job
# Runs: 11_esmfold_validation.py
# Uses ESM-IF1 (no openfold required) to score ProteinMPNN 3BTA designs
# Metric: log P(sequence | 3BTA backbone) per residue
# ============================================================================

echo "=== ESM-IF1 Structural Compatibility Job ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

SCRATCH="${SCRATCH:-/path/to/scratch/narrow_model_safety_eval}"
PROJECT_DIR="${SCRATCH}/Narrow_Model_Safety_Eval"
export TORCH_HOME="${SCRATCH}/torch_cache"
mkdir -p ${SCRATCH}/logs ${TORCH_HOME}

source ~/miniconda3/miniconda3/etc/profile.d/conda.sh
conda activate narrow_model_safety

cd ${PROJECT_DIR}
python src/11_esmfold_validation.py \
    --proteinmpnn_fasta results/proteinmpnn_output/3BTA/seqs/3BTA.fa \
    --reference_pdb data/structures/3BTA.pdb \
    --lc_end_residue 430 \
    --n_top 10 \
    --n_bottom 10 \
    --device cuda

echo "=== Done: $(date) ==="
