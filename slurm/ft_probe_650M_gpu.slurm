#!/bin/bash
#SBATCH --job-name=ft650p
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/ft650p_%j.log
# 🔒 A SIZING PROBE, not a result run. The 650M arm is the arm the paper's headline rests on
# (frozen beta_lactamase 21.19%), and it has never run. This measures ONE cell so the 80-task
# array is sized from a measurement.
# 🔴 This study has been mis-sized twice already, both times from an estimate: once from a GPU
# figure applied to a CPU run (amendment 1), once by putting 80 fine-tunes in one task.
# ⚠️ The cell chosen is the WORST CASE on purpose -- `full` trains every parameter, so no other
# recipe can be slower, and `beta_lactamase` is the decisive class.
# ⚠️ 6h, not 12h. A long walltime is hard for the scheduler to BACKFILL, and at 12h this job sat
# at (Priority) while an A40 stood free on g0004. One cell is 16 fine-tunes of ~315 sequences;
# 35M needed ~8 min each on 16 CPUs, so an A40 has a wide margin inside 6h.
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
  --class-name beta_lactamase --recipe full --seeds "${SEEDS:-10}"
