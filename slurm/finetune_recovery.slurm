#!/bin/bash
#SBATCH --job-name=ft_rec
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --array=0-79
#SBATCH --output=logs/ft_rec_%A_%a.log
# 🔴 The per-class array put 80 fine-tunes in one task: 11-16h on 32 CPUs, and three tasks
# timed out at the 16h limit -- one of them beta_lactamase, the class the paper's headline
# turns on. 🔒 One task per (class x recipe) is 16 fine-tunes, about 3h, and isolates failures.
# ⚠️ This file is written locally and rsynced. Writing it through a double-quoted ssh payload
# expanded ${CONDA_SETUP} on the LOCAL shell, to empty, and every task died in one second.
set -uo pipefail
cd "$PROJECT_DIR"
source "${CONDA_SETUP}"
conda activate "${CONDA_ENV:-narrow_model_safety}"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.hf_cache}"
export OMP_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false
python -u src/147_finetune_recovery.py --arm "${ARM:-esm2_35M}" \
  --task-index "$SLURM_ARRAY_TASK_ID" --seeds "${SEEDS:-10}"
