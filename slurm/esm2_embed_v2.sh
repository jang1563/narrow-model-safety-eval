#!/bin/bash
#SBATCH --job-name=nmse_esm2_v2
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/nmse_esm2_v2_%j.log

# ESM-2 embeddings for the v2 panel. Writes results/v2/embeddings_*_v2*.npy plus a
# row-aligned manifest. The April arrays are not touched.
#
# Environment: set PYTHON_BIN to an interpreter that already has torch and
# transformers (this is how the Cayuga scratch env is used). If PYTHON_BIN is
# unset, fall back to activating a conda env.
#
#   PROJECT_DIR=... PYTHON_BIN=.../envs/nmse/bin/python sbatch slurm/esm2_embed_v2.sh
#
# MODEL defaults to ESM-2 650M; DRY_RUN_TAG suffixes the outputs for a throwaway
# pipeline check.

set -euo pipefail
echo "=== NMSE ESM-2 v2 ==="; date; hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

PROJECT_DIR="${PROJECT_DIR:?set PROJECT_DIR to the Narrow_Model_Safety_Eval checkout}"
cd "$PROJECT_DIR"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PY="$PYTHON_BIN"
else
    source "${CONDA_SETUP:-$HOME/miniconda3/etc/profile.d/conda.sh}"
    conda activate "${CONDA_ENV:-narrow_model_safety}"
    PY=python
fi

export HF_HOME="${HF_HOME:-$PROJECT_DIR/.hf_cache}"
mkdir -p "$HF_HOME" "$PROJECT_DIR/logs"

echo "python: $("$PY" -V 2>&1)  torch: $("$PY" -c 'import torch;print(torch.__version__, torch.cuda.is_available())')"
echo "panel: $(grep -c '^>' data/sequences/toxins_positive_v2.fasta) positive / $(grep -c '^>' data/sequences/benign_negatives_v2.fasta) negative"

ARGS=(--model "${MODEL:-facebook/esm2_t33_650M_UR50D}" --batch_size "${BATCH_SIZE:-8}")
[[ -n "${DRY_RUN_TAG:-}" ]] && ARGS+=(--dry-run-tag "$DRY_RUN_TAG")

"$PY" src/02b_esm2_embed_v2.py "${ARGS[@]}"

echo "=== done ==="; date
ls -la results/v2/
