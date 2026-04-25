#!/bin/bash
# ============================================================================
# Improvements pipeline orchestration script (Round 2)
# 5 improvements: thermolysin control, PLM, expanded proteins, T sensitivity, ESMFold
#
# Prerequisites: Sync project to HPC first.
#
# Usage (from HPC login node):
#   cd $SCRATCH/Narrow_Model_Safety_Eval
#   bash slurm/run_improvements.sh
# ============================================================================

set -e

# Set SCRATCH to your HPC scratch directory
SCRATCH="${SCRATCH:-/path/to/scratch/narrow_model_safety_eval}"
PROJECT_DIR="${SCRATCH}/Narrow_Model_Safety_Eval"
SCRIPT_DIR="${PROJECT_DIR}/slurm"

# Use updated SLURM binaries (default v22.05.2 has version mismatch with scheduler)
SBATCH="${SBATCH_BIN:-sbatch}"
SQUEUE="${SQUEUE_BIN:-squeue}"

echo "=== Narrow Model Safety Evaluation — Improvements Pipeline (Round 2) ==="
echo "Project: ${PROJECT_DIR}"
echo "Date: $(date)"
echo ""

# Check project exists
if [ ! -d "${PROJECT_DIR}/src" ]; then
    echo "ERROR: Project not found at ${PROJECT_DIR}"
    echo "Sync first: rsync -avz --delete /path/to/Narrow_Model_Safety_Eval/ ${PROJECT_DIR}/"
    exit 1
fi

mkdir -p ${SCRATCH}/logs

# ---- Step 0: Download new structures + sequences (CPU, no deps) ----
echo "--- Step 0: Downloading new structures (1ABR, 1Z7H, 4HSC) ---"

JOB_SETUP=$(${SBATCH} --parsable ${SCRIPT_DIR}/download_structures.sh)
echo "  Submitted download_structures.sh: Job ${JOB_SETUP}"

# ---- Step 1: Parallel GPU jobs (all depend on setup) ----
echo ""
echo "--- Step 1: Submitting parallel GPU jobs (depend on ${JOB_SETUP}) ---"

# Negative controls (1AST, 1LNF, 1QD2, 1LYZ) — now includes thermolysin
JOB_CONTROLS=$(${SBATCH} --parsable \
    --dependency=afterok:${JOB_SETUP} \
    ${SCRIPT_DIR}/negative_controls.sh)
echo "  Submitted negative_controls.sh (+ 1LNF thermolysin): Job ${JOB_CONTROLS}"

# ProteinMPNN re-run — now processes 8 proteins (added 1ABR, 1Z7H, 4HSC)
JOB_PMPNN=$(${SBATCH} --parsable \
    --dependency=afterok:${JOB_SETUP} \
    ${SCRIPT_DIR}/proteinmpnn.sh)
echo "  Submitted proteinmpnn.sh (8 proteins): Job ${JOB_PMPNN}"

# ESM-2 masked prediction — now processes 8 proteins + PLM score
JOB_FSPE=$(${SBATCH} --parsable \
    --dependency=afterok:${JOB_SETUP} \
    ${SCRIPT_DIR}/esm2_masked.sh)
echo "  Submitted esm2_masked.sh (8 proteins + PLM): Job ${JOB_FSPE}"

# Temperature sensitivity sweep (3BTA and 2AAI at T=0.05,0.1,0.2,0.3)
JOB_TEMP=$(${SBATCH} --parsable \
    --dependency=afterok:${JOB_SETUP} \
    ${SCRIPT_DIR}/fsi_temperature_sensitivity.sh)
echo "  Submitted fsi_temperature_sensitivity.sh: Job ${JOB_TEMP}"

# ---- Step 2: Analyses depending on ProteinMPNN ----
echo ""
echo "--- Step 2: Analyses depending on ProteinMPNN (depend on ${JOB_PMPNN}) ---"

# FSI aggregate analysis (Wilcoxon + bootstrap CI + Cohen's d) — now 8 proteins
JOB_FSI=$(${SBATCH} --parsable \
    --dependency=afterok:${JOB_PMPNN} \
    ${SCRIPT_DIR}/fsi_analysis.sh)
echo "  Submitted fsi_analysis.sh (Wilcoxon+bootstrap, 8 proteins): Job ${JOB_FSI} (depends on ${JOB_PMPNN})"

# ESMFold validation of top 3BTA designs
JOB_ESMFOLD=$(${SBATCH} --parsable \
    --dependency=afterok:${JOB_PMPNN} \
    ${SCRIPT_DIR}/esmfold_validation.sh)
echo "  Submitted esmfold_validation.sh: Job ${JOB_ESMFOLD} (depends on ${JOB_PMPNN})"

# ---- Step 3: Final report (all analyses complete) ----
echo ""
echo "--- Step 3: Updated evaluation report ---"

JOB_REPORT=$(${SBATCH} --parsable \
    --dependency=afterok:${JOB_CONTROLS}:${JOB_FSI}:${JOB_FSPE}:${JOB_TEMP}:${JOB_ESMFOLD} \
    ${SCRIPT_DIR}/eval_report.sh)
echo "  Submitted eval_report.sh: Job ${JOB_REPORT} (depends on all analysis jobs)"

echo ""
echo "=== All improvement jobs submitted ==="
echo ""
echo "Monitor with:"
echo "  ${SQUEUE} -u $(whoami)"
echo "  tail -f ${SCRATCH}/logs/*.log"
echo ""
echo "Job dependency chain:"
echo "  setup(${JOB_SETUP})"
echo "    ├── neg_controls(${JOB_CONTROLS}) ──────────────────────────────────┐"
echo "    ├── pmpnn(${JOB_PMPNN}) ──→ fsi_analysis(${JOB_FSI}) ──────────────┤"
echo "    │              └──→ esmfold(${JOB_ESMFOLD}) ───────────────────────┤──→ report(${JOB_REPORT})"
echo "    ├── esm2_masked(${JOB_FSPE}) ──────────────────────────────────────┤"
echo "    └── fsi_temp_sweep(${JOB_TEMP}) ──────────────────────────────────┘"
echo ""
echo "Key output files:"
echo "  results/fsi_controls.json              (negative controls incl. 1LNF)"
echo "  results/figures/bonta_three_way_control.png"
echo "  results/fsi_results.json               (8 proteins)"
echo "  results/fsi_aggregate_results.json     (8 proteins, Wilcoxon + bootstrap)"
echo "  results/fspe_results.json              (8 proteins + PLM analysis)"
echo "  results/figures/plm_comparison.png"
echo "  results/fsi_temperature_sensitivity.json"
echo "  results/figures/fsi_temperature_sensitivity.png"
echo "  results/esmfold_validation.json"
echo "  results/figures/esmfold_tmscore_scatter.png"
echo "  results/evaluation_report.txt          (updated with all improvements)"
