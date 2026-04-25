#!/bin/bash
#SBATCH --job-name=evodiff_fsi
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/evodiff_%j.log

# ============================================================================
# EvoDiff FSI Evaluation — GPU Job (v2 Pillar 1B)
# Runs: 13_evodiff_fsi.py
#
# Evaluates FSI-EvoD: EvoDiff OA_DM_640M motif-scaffolding with catalytic
# residues fixed (from functional_sites.json). Tests structure-free path to
# functional design.
#
# Initial run: evaluate BoNT-A (3BTA) and Ricin (2AAI) only.
# These have the highest v1 FSI and should show clearest motif-conditioned signal.
#
# Memory: OA_DM_640M requires ~20GB GPU VRAM. Request 48GB system RAM for
# sequence handling.
# ============================================================================

echo "=== EvoDiff FSI Job ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

SCRATCH="${SCRATCH:?ERROR: set the SCRATCH environment variable to your cluster scratch path}"
PROJECT_DIR="${SCRATCH}/Narrow_Model_Safety_Eval"

mkdir -p ${PROJECT_DIR}/logs

source ~/miniconda3/miniconda3/etc/profile.d/conda.sh
conda activate narrow_model_safety

# Install evodiff if not present
python -c "import evodiff" 2>/dev/null || {
    echo "Installing evodiff..."
    pip install evodiff
}

cd ${PROJECT_DIR}

# EvoDiff checkpoint pre-downloaded to torch hub cache on scratch
export TORCH_HOME="${SCRATCH}/torch_cache"

# Run BoNT-A, Ricin (main), and Anthrax PA (sanity check) in a single call.
# Single invocation avoids the second run overwriting fsi_evodiff_results.json.
# n=100 for 3BTA and 2AAI; Anthrax PA (1ACC) uses --num_seqs 100 too — it
# generates quickly since OA_DM has no structural constraint to resolve.
# Unconditional baseline is skipped for 1ACC to save time.
python src/13_evodiff_fsi.py \
    --num_seqs 100 \
    --proteins 3BTA 2AAI 1ACC \
    --device cuda

echo "=== Done: $(date) ==="
echo "Results: ${PROJECT_DIR}/results/fsi_evodiff_results.json"
