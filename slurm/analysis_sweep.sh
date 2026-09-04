#!/bin/bash
#SBATCH --job-name=nmse_analysis
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/nmse_analysis_%j.log

# Run the leave-one-mechanism-out experiment and the negative-set decomposition once
# per embedded model, so the model axis and the negative-set axis are crossed on a
# single fixed panel. Assumes 02b has already written arrays for every TAG.
#
#   PROJECT_DIR=... PYTHON_BIN=... TAGS="esm2_8M esm2_35M esm2_150M '' esm2_3B" \
#     sbatch slurm/analysis_sweep.sh
#
# An empty tag means the untagged arrays, which hold the canonical 650M run.

set -euo pipefail
echo "=== NMSE analysis sweep ==="; date; hostname

PROJECT_DIR="${PROJECT_DIR:?set PROJECT_DIR}"
PY="${PYTHON_BIN:?set PYTHON_BIN}"
SEEDS="${SEEDS:-30}"
cd "$PROJECT_DIR"; mkdir -p logs

# shellcheck disable=SC2206
TAG_LIST=(${TAGS:-esm2_8M esm2_35M esm2_150M NONE esm2_3B})

for T in "${TAG_LIST[@]}"; do
    [[ "$T" == "NONE" ]] && ARG=() || ARG=(--tag "$T")
    LABEL=$([[ "$T" == "NONE" ]] && echo "untagged (650M)" || echo "$T")
    echo; echo "############ $LABEL ############"; date
    if [[ "$T" != "NONE" && ! -f "results/v2/embeddings_positive_v2_${T}.npy" ]]; then
        echo "  no embeddings for $T, skipping"; continue
    fi
    "$PY" src/03b_leave_one_mechanism_out.py "${ARG[@]}" 2>&1 | grep -v Warning || true
    "$PY" src/03e_negative_difficulty_curve.py "${ARG[@]}" --seeds "$SEEDS" 2>&1 | grep -v Warning || true
done

echo; echo "=== done ==="; date
