#!/bin/bash
# ============================================================================
# Master pipeline orchestration script
# Submits all jobs with proper SLURM dependencies
#
# Usage:
#   bash slurm/run_all.sh          # Full pipeline
#   bash slurm/run_all.sh --skip-data  # Skip data collection (already done)
# ============================================================================

set -e

# Set SCRATCH to your HPC scratch directory
SCRATCH="${SCRATCH:-/path/to/scratch/narrow_model_safety_eval}"
PROJECT_DIR="${SCRATCH}/Narrow_Model_Safety_Eval"
SCRIPT_DIR="${PROJECT_DIR}/slurm"

# Use updated SLURM binaries (default v22.05.2 has version mismatch with scheduler)
SBATCH="/opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch"
SQUEUE="/opt/ohpc/pub/software/slurm/24.05.2/bin/squeue"

echo "=== Narrow Model Safety Evaluation — Full Pipeline ==="
echo "Project: ${PROJECT_DIR}"
echo "Date: $(date)"
echo ""

# Check project exists
if [ ! -d "${PROJECT_DIR}/src" ]; then
    echo "ERROR: Project not found at ${PROJECT_DIR}"
    echo "Sync first: rsync -av /path/to/Narrow_Model_Safety_Eval/ ${PROJECT_DIR}/"
    exit 1
fi

# ---- Step 1: Data collection (runs on login node, no GPU needed) ----
if [ "$1" != "--skip-data" ]; then
    echo "--- Step 1: Data collection ---"
    source ~/miniconda3/miniconda3/etc/profile.d/conda.sh
    conda activate narrow_model_safety
    cd ${PROJECT_DIR}
    python src/01_collect_data.py
    echo ""
fi

# ---- Step 2: Submit GPU jobs ----
echo "--- Step 2: Submitting GPU jobs ---"

# ESM-2 embeddings (prerequisite for steps 3, 5)
JOB_EMBED=$(${SBATCH} --parsable ${SCRIPT_DIR}/esm2_embed.sh)
echo "  Submitted esm2_embed.sh: Job ${JOB_EMBED}"

# ESM-2 masked prediction / FSPE (independent of embeddings)
JOB_FSPE=$(${SBATCH} --parsable ${SCRIPT_DIR}/esm2_masked.sh)
echo "  Submitted esm2_masked.sh: Job ${JOB_FSPE}"

# ProteinMPNN (independent of ESM-2 jobs)
JOB_PMPNN=$(${SBATCH} --parsable ${SCRIPT_DIR}/proteinmpnn.sh)
echo "  Submitted proteinmpnn.sh: Job ${JOB_PMPNN}"

# ---- Step 3: Submit CPU analyses (depend on GPU jobs) ----
echo ""
echo "--- Step 3: Submitting CPU analyses (with dependencies) ---"

# Separability + nearest-neighbor depend on embeddings
JOB_CPU=$(${SBATCH} --parsable \
    --dependency=afterok:${JOB_EMBED} \
    ${SCRIPT_DIR}/cpu_analyses.sh)
echo "  Submitted cpu_analyses.sh: Job ${JOB_CPU} (depends on ${JOB_EMBED})"

# FSI analysis depends on ProteinMPNN
JOB_FSI=$(${SBATCH} --parsable \
    --dependency=afterok:${JOB_PMPNN} \
    ${SCRIPT_DIR}/fsi_analysis.sh)
echo "  Submitted fsi_analysis.sh: Job ${JOB_FSI} (depends on ${JOB_PMPNN})"

# ---- Step 4: Final report (depends on all analysis jobs) ----
echo ""
echo "--- Step 4: Submitting evaluation report ---"

JOB_REPORT=$(${SBATCH} --parsable \
    --dependency=afterok:${JOB_CPU}:${JOB_FSI}:${JOB_FSPE} \
    ${SCRIPT_DIR}/eval_report.sh)
echo "  Submitted eval_report.sh: Job ${JOB_REPORT} (depends on ${JOB_CPU}, ${JOB_FSI}, ${JOB_FSPE})"

echo ""
echo "=== All jobs submitted ==="
echo ""
echo "Monitor with:"
echo "  ${SQUEUE} -u $(whoami)"
echo "  tail -f ${SCRATCH}/logs/*.log"
echo ""
echo "Job dependency chain:"
echo "  embed(${JOB_EMBED}) ──→ cpu_analyses(${JOB_CPU}) ──┐"
echo "  fspe(${JOB_FSPE})  ─────────────────────────────────┤──→ report(${JOB_REPORT})"
echo "  pmpnn(${JOB_PMPNN}) ──→ fsi(${JOB_FSI}) ───────────┘"
