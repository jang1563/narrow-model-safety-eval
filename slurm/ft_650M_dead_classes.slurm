#!/bin/bash
#SBATCH --job-name=ft650d
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=12-17,48-53
#SBATCH --output=logs/ft650d_%A_%a.log
# 🔒 The 650M arm, STAGED on the two dead classes. AY-1 -- the band that decides whether the
# headline's two failures are a frozen-probing artifact -- reads only `beta_lactamase` and
# `phage_peptidoglycan_hydrolase`, so these twelve (class x recipe) cells make AY-1 declarable
# on both arms. The indices come from `--list-tasks`, not from arithmetic.
# ⚠️ This is EXECUTION ORDER, not an amendment: no band, no prediction and no verdict rule
# changes. AY-2 quantifies over all sixteen classes and therefore stays UNDECLARABLE after
# this job, and must be reported as single-arm with that reason stated. Running the cheap
# subset and then quietly rescoping AY-2 would be the post-hoc move the method forbids.
# ⚠️ Sized from the probe (job 3550623), not from an estimate: ~9.2 min per fine-tune on an
# A40, 16 fine-tunes per cell, so ~2.4 h per cell against this 6 h limit. The full 80-cell
# arm would be ~195 GPU-hours, which is why it is staged.
# 🔒 Idempotent, so the probe's own cell (beta_lactamase/full, index 14) is skipped rather
# than recomputed.
# ⚠️ Written locally and rsynced. A double-quoted ssh payload expands ${CONDA_SETUP} on the
# LOCAL shell, to empty, and every task dies in one second.
set -uo pipefail
cd "$PROJECT_DIR"
source "${CONDA_SETUP}"
conda activate "${CONDA_ENV:-narrow_model_safety}"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.hf_cache}"
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -u src/147_finetune_recovery.py --arm esm2_650M \
  --task-index "$SLURM_ARRAY_TASK_ID" --seeds "${SEEDS:-10}"
