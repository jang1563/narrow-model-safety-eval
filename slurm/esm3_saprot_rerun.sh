#!/bin/bash
#SBATCH --job-name=esm3_saprot_rerun
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/esm3_saprot_rerun_%j.log

# =============================================================================
# Re-run the ESM-3 and SaProt FSPE columns with corrected coordinates.
#
# WHY: docs/EVALUATION_REPORT.md reports the cross-model sign-flip count as
# "2 of 9" rather than a figure over all 12 rows, because the 2026-05-22
# mature-chain numbering fix re-ran ESM-2 ONLY. Three proteins carry a corrected
# ESM-2 value beside uncorrected ESM-3 and SaProt values, so a count over all 12
# compares two coordinate systems against each other. Both columns live in one
# artifact (results/esm3_fspe_results.json) and src/14 routes through the shared
# offset resolver in utils, so ONE run of src/14 retires the indeterminate set.
#
# Three launcher defects in slurm/esm3_embed.sh are fixed here rather than there,
# because that script is the v2 record of how the published numbers were made:
#
#   1. env: it activates `narrow_model_safety`, which does not exist on this
#      cluster. The env is a PATH, not a name: ${NMSE_ENV} below.
#   2. SaProt gate: it tests for data/annotations/saprot_tokens.json, which no
#      longer exists and must not be used (it is an April 16 file, untracked,
#      8 proteins, one of them a corrected-away accession, and it predates the
#      numbering fix). The gate now tests the reproducible v2 3Di file.
#   3. PROJECT_DIR pointed at the old Narrow_Model_Safety_Eval clone.
#
# ⚠️ HF_HOME, not HF_HUB_CACHE, and this is the opposite of what the ESM-2 pool
# job needed. This cache root holds TWO layouts: ESM-3 and SaProt are under
# ${HF_CACHE}/hub, while the ESM-2 models sit directly under ${HF_CACHE}. So the
# correct variable depends on which model you want, which is why the earlier
# "HF_HOME was wrong" lesson does not transfer to this script.
# =============================================================================

set -euo pipefail

SCRATCH=/athena/masonlab/scratch/users/jak4013
PROJECT_DIR=${SCRATCH}/narrow_model_safety_eval/NMSE_v3_20260919
NMSE_ENV=${SCRATCH}/envs/nmse
HF_CACHE=${SCRATCH}/narrow_model_safety_eval/hf_cache

echo "=== ESM-3 + SaProt FSPE re-run ==="
echo "Date : $(date)"
echo "Node : $(hostname)"
echo "GPU  : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo N/A)"

cd "${PROJECT_DIR}"
mkdir -p logs

export HF_HOME="${HF_CACHE}"
export HF_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null || echo '')"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
# Both models are already in ${HF_CACHE}/hub, so run offline: a download attempt
# from a compute node is the failure mode that was once misdiagnosed as a hang.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PY="${NMSE_ENV}/bin/python"
echo "--- interpreter: $("${PY}" -V 2>&1) ---"
"${PY}" -c "import torch, esm, transformers; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| esm', esm.__version__, '| transformers', transformers.__version__)"

THREEDI="${PROJECT_DIR}/data/annotations/structure_3di_v2.json"
if [ ! -f "${THREEDI}" ]; then
    echo "FATAL: ${THREEDI} missing. SaProt cannot run reproducibly without it."
    exit 1
fi
echo "--- 3Di file present: $(basename "${THREEDI}") ---"

# One invocation does both columns: src/14 writes ESM-3 and SaProt rows into the
# same results/esm3_fspe_results.json, tagged by model.
echo "=== running src/14 with SaProt ==="
"${PY}" -u src/14_esm3_separability_fspe.py --device cuda --with_saprot

echo "=== Done: $(date) ==="
echo "Artifact: ${PROJECT_DIR}/results/esm3_fspe_results.json"
