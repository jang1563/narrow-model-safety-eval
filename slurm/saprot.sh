#!/bin/bash
#SBATCH --job-name=nmse_saprot
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/nmse_saprot_%j.log

# SaProt, the structure-aware run. Requires 02h to have written
# data/annotations/structure_3di_v2.json first.
#
#   PROJECT_DIR=... PYTHON_BIN=... sbatch slurm/saprot.sh

set -euo pipefail
echo "=== NMSE SaProt ==="; date; hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

PROJECT_DIR="${PROJECT_DIR:?set PROJECT_DIR}"
PY="${PYTHON_BIN:?set PYTHON_BIN}"
SEEDS="${SEEDS:-20}"
cd "$PROJECT_DIR"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.hf_cache}"
mkdir -p "$HF_HOME" logs

test -f data/annotations/structure_3di_v2.json || {
    echo "missing structure_3di_v2.json; run src/02h_saprot_prepare.py first"; exit 1; }

"$PY" src/02i_saprot_embed.py --tag saprot_650M

for S in 03b_leave_one_mechanism_out 03c_ablation_baselines 03d_localization_confound; do
    "$PY" "src/${S}.py" --tag saprot_650M 2>&1 | grep -v Warning || true
done
"$PY" src/03h_probe_vs_similarity.py --tag saprot_650M --seeds 30 2>&1 | grep -v Warning || true
"$PY" src/03f_coverage_strictness.py --tag saprot_650M --seeds 30 2>&1 | grep -v Warning || true
"$PY" src/03j_classifier_sweep.py --tag saprot_650M --seeds "$SEEDS" 2>&1 | grep -v Warning || true

echo; echo "=== done ==="; date
