#!/bin/bash
#SBATCH --job-name=nmse_esm3c
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/nmse_esm3c_%j.log

# Embed the v2 panel with ESM-C (300M, 600M) and ESM-3 (1.4B open), then run the
# same four analyses used for ESM-2 so the architecture axis is crossed with the
# negative-set axis and the per-member axis rather than reported beside them.
#
#   PROJECT_DIR=... PYTHON_BIN=.../envs/nmse/bin/python sbatch slurm/esm3_esmc_sweep.sh

set -euo pipefail
echo "=== NMSE ESM-3 / ESM-C sweep ==="; date; hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

PROJECT_DIR="${PROJECT_DIR:?set PROJECT_DIR}"
PY="${PYTHON_BIN:?set PYTHON_BIN}"
SEEDS="${SEEDS:-30}"
cd "$PROJECT_DIR"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.hf_cache}"
mkdir -p "$HF_HOME" logs

"$PY" -c 'import torch;print("torch",torch.__version__,"cuda",torch.cuda.is_available())'

run () {   # run <model> <tag>
    echo; echo "############ $2 ($1) ############"; date
    "$PY" src/02e_esm3_esmc_embed.py --model "$1" --tag "$2" 2>&1 | grep -v '^ESMC:' || return 1
    for s in 03b_leave_one_mechanism_out 03c_ablation_baselines 03d_localization_confound; do
        "$PY" "src/${s}.py" --tag "$2" 2>&1 | grep -v Warning || true
    done
    "$PY" src/03e_negative_difficulty_curve.py --tag "$2" --seeds "$SEEDS" 2>&1 | grep -v Warning || true
    "$PY" src/03f_coverage_strictness.py --tag "$2" --seeds "$SEEDS" 2>&1 | grep -v Warning || true
    "$PY" src/03g_member_separability.py --tag "$2" 2>&1 | grep -v Warning || true
}

run esmc_300m       esmc_300M
run esmc_600m       esmc_600M
run esm3_sm_open_v1 esm3_1_4B

echo; echo "=== done ==="; date
