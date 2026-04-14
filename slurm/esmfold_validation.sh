#!/bin/bash
#SBATCH --job-name=esmfold_val
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=g0001
#SBATCH --output=logs/esmfold_val_%j.log

# ============================================================================
# ESMFold Structural Validation — GPU Job
# Runs: 11_esmfold_validation.py
# Folds top-10 ProteinMPNN designs of BoNT-A via ESMFold, computes TM-score
# Memory: ESMFold requires ~24GB for 430-aa sequences; 64GB gives headroom
# ============================================================================

echo "=== ESMFold Validation Job ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

SCRATCH="${SCRATCH:-/path/to/scratch/narrow_model_safety_eval}"
PROJECT_DIR="${SCRATCH}/Narrow_Model_Safety_Eval"
export TORCH_HOME="${SCRATCH}/torch_cache"
mkdir -p ${SCRATCH}/logs ${TORCH_HOME}

source ~/miniconda3/miniconda3/etc/profile.d/conda.sh
conda activate narrow_model_safety

# Check openfold dependency
python -c "import openfold" 2>/dev/null || {
    echo "WARNING: openfold not installed. Attempting install..."
    pip install git+https://github.com/aqlaboratory/openfold.git
}

cd ${PROJECT_DIR}
python src/11_esmfold_validation.py \
    --proteinmpnn_fasta results/proteinmpnn_output/3BTA/seqs/3BTA.fa \
    --reference_pdb data/structures/3BTA.pdb \
    --lc_end_residue 430 \
    --n_top 10 \
    --device cuda

echo "=== Done: $(date) ==="
