#!/bin/bash
#SBATCH --job-name=nmse_deepvic_loco
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/nmse_deepvic_loco_%j.log

# Study AI-2/AI-3/AI-4: leave-one-category-out on DeepVIC's published class axis.
# Bands and predictions are frozen in docs/DEEPVIC_BENCHMARK_PREREGISTRATION.md.
#
#   PROJECT_DIR=... ARMS=esm2_650M,esm2_35M sbatch slurm/deepvic_loco.sh
#
# CPU only: the fold is logistic regression over cached embeddings. Memory is sized for
# the chunked margin pass -- 12,989 x 12,989 would be 1.4 GB whole, so src/125 does it in
# 512-row chunks, but the embeddings themselves are 12,989 x 1280 plus the negative union.
#
# A verdict needs BOTH arms in the same band (the second-arm rule), which is why ARMS
# defaults to both and the artifact records each separately.

set -euo pipefail
echo "=== NMSE DeepVIC LOCO ==="; date; hostname

PROJECT_DIR="${PROJECT_DIR:?set PROJECT_DIR}"
cd "$PROJECT_DIR"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PY="$PYTHON_BIN"
else
    source "${CONDA_SETUP:-$HOME/miniconda3/etc/profile.d/conda.sh}"
    conda activate "${CONDA_ENV:-narrow_model_safety}"
    PY=python
fi
mkdir -p logs

ARMS="${ARMS:-esm2_650M,esm2_35M}"
for A in ${ARMS//,/ }; do
    SFX=$([[ "$A" == esm2_650M ]] && echo "" || echo "_$A")
    for f in "results/v3/embeddings_deepvic${SFX}.npy" \
             "results/v3/embedding_manifest_deepvic${SFX}.json"; do
        [[ -f "$f" ]] || { echo "MISSING: $f -- run slurm/deepvic_embed.sh for $A first"; exit 1; }
    done
done
echo "arms: $ARMS"

"$PY" -W ignore src/125_deepvic_loco.py --selftest
"$PY" -W ignore -u src/125_deepvic_loco.py --arms "$ARMS" --seeds "${SEEDS:-30}"

echo "=== done ==="; date
ls -la results/deepvic_loco.json
