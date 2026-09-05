#!/bin/bash
#SBATCH --job-name=nmse_expand
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/nmse_expand_%j.log
set -euo pipefail
cd "$PROJECT_DIR"
export HF_HOME="$PROJECT_DIR/.hf_cache"
PY="$PYTHON_BIN"
echo "=== panel: $(grep -c '^>' data/sequences/toxins_positive_v2.fasta) pos / $(grep -c '^>' data/sequences/benign_negatives_v2.fasta) neg ==="
"$PY" src/02b_esm2_embed_v2.py --model facebook/esm2_t33_650M_UR50D
for S in 03b_leave_one_mechanism_out 03c_ablation_baselines; do
  "$PY" "src/${S}.py" 2>&1 | grep -v Warning || true
done
"$PY" src/03f_coverage_strictness.py --seeds 30 2>&1 | grep -v Warning || true
"$PY" src/03j_classifier_sweep.py --seeds 20 2>&1 | grep -v Warning || true
echo "=== done ==="
