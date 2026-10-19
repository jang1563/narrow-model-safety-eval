#!/bin/bash
#SBATCH --job-name=ft_p8
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ft_p8_%j.log
# 🔒 A timing probe, not a result. Two mis-sizings have cost this study a day: the question
# is whether 8 CPUs per task is meaningfully slower than 32, given that 16 -> 32 bought
# almost nothing. If it is not, the cpu=300 cap allows 37 concurrent tasks instead of 9.
set -uo pipefail
cd "$PROJECT_DIR"
source "${CONDA_SETUP}"
conda activate "${CONDA_ENV:-narrow_model_safety}"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.hf_cache}"
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
python -u src/147_finetune_recovery.py --arm esm2_35M \
  --class-name virulence_associated_non_toxin --recipe mean_last2 --seeds 2
