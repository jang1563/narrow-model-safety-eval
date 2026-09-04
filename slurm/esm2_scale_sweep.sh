#!/bin/bash
#SBATCH --job-name=nmse_scale
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/nmse_scale_%j.log

# ESM-2 scale sweep on the current panel. Embeds the SAME panel at five model sizes so
# the model axis and the negative-set axis can be crossed rather than compared across
# different panels, which is what made the earlier 150M column not like-for-like.
#
#   PROJECT_DIR=... PYTHON_BIN=.../envs/nmse/bin/python sbatch slurm/esm2_scale_sweep.sh
#
# 650M is skipped by default because the untagged arrays already hold it for this panel.
# Set REDO_650M=1 to write a tagged copy as well.

set -euo pipefail
echo "=== NMSE ESM-2 scale sweep ==="; date; hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

PROJECT_DIR="${PROJECT_DIR:?set PROJECT_DIR}"
PY="${PYTHON_BIN:?set PYTHON_BIN to an interpreter with torch and transformers}"
cd "$PROJECT_DIR"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.hf_cache}"
mkdir -p "$HF_HOME" logs

echo "panel: $(grep -c '^>' data/sequences/toxins_positive_v2.fasta) positive / $(grep -c '^>' data/sequences/benign_negatives_v2.fasta) negative"

run () {   # run <model> <tag> <batch_size>
    echo; echo "--- $2 ($1) ---"; date
    "$PY" src/02b_esm2_embed_v2.py --model "$1" --tag "$2" --batch_size "$3"
}

run facebook/esm2_t6_8M_UR50D    esm2_8M    16
run facebook/esm2_t12_35M_UR50D  esm2_35M   16
run facebook/esm2_t30_150M_UR50D esm2_150M  8
[[ "${REDO_650M:-0}" == "1" ]] && run facebook/esm2_t33_650M_UR50D esm2_650M 8
run facebook/esm2_t36_3B_UR50D   esm2_3B    2

echo; echo "=== done ==="; date
ls -la results/v2/ | grep -E 'esm2_' || true
