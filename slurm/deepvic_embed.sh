#!/bin/bash
#SBATCH --job-name=nmse_deepvic_embed
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/nmse_deepvic_embed_%j.log

# Study AI embed step: ESM-2 over DeepVIC's published benchmark, one arm per job.
# docs/DEEPVIC_BENCHMARK_PREREGISTRATION.md section 3 (AI-2, AI-3, AI-4).
#
# 46,445 sequences: DeepVIC's 12,989 class-labelled positives then all 33,456
# negatives, in the fixed order src/124 records in the manifest.
#
#   PROJECT_DIR=... ARM=esm2_650M sbatch slurm/deepvic_embed.sh
#   PROJECT_DIR=... ARM=esm2_35M  sbatch slurm/deepvic_embed.sh
#
# Resumable: src/77's embed banks a shard every 250 sequences, so a preemption on
# preempt_gpu costs at most 250 sequences. Requeue with the same command.
#
# The checkpoint directory is stamped with build + model + DEVICE, so a resume that
# would mix two float backends in one array is refused rather than silently accepted.

set -euo pipefail
echo "=== NMSE DeepVIC embed ==="; date; hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

PROJECT_DIR="${PROJECT_DIR:?set PROJECT_DIR to the Narrow_Model_Safety_Eval checkout}"
ARM="${ARM:?set ARM to esm2_650M or esm2_35M}"
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

echo "python: $("$PY" -V 2>&1)"
echo "torch:  $("$PY" -c 'import torch;print(torch.__version__, torch.cuda.is_available())')"

# the inputs src/124 reads; fail loudly here rather than part-way through a GPU hour
for f in data/external/deepvic/DLDB_33456.faa data/external/deepvic/metadata.txt \
         data/sequences/benign_negatives_v3.fasta \
         results/deepvic_build.json results/external_class_axis_build.json; do
    [[ -f "$f" ]] || { echo "MISSING INPUT: $f"; exit 1; }
done
# 🔴 This was written as one line with $([[ $ARM == esm2_35M ]] && echo _esm2_35M). For
# ARM=esm2_650M the test is false, the && short-circuits, the command substitution returns 1,
# and `set -e` killed the job SILENTLY after 7 seconds -- on the arm the expression does not
# name, with no message in the log. `bash -n` passes it because it is not a syntax error.
case "$ARM" in
    esm2_650M) GATE="results/v3/embeddings_negative_v3.npy" ;;
    esm2_35M)  GATE="results/v3/embeddings_negative_v3_esm2_35M.npy" ;;
    *)         echo "unknown ARM: $ARM"; exit 1 ;;
esac
[[ -f "$GATE" ]] || { echo "MISSING GATE TARGET: $GATE"; exit 1; }
echo "gate target: $GATE"

"$PY" src/124_deepvic_embed.py --selftest
"$PY" -u src/124_deepvic_embed.py --arm "$ARM" --batch "${BATCH_SIZE:-16}"

echo "=== done ==="; date
ls -la results/v3/embeddings_deepvic* results/v3/embedding_manifest_deepvic* 2>/dev/null
