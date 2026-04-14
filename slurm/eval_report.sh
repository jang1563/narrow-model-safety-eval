#!/bin/bash
#SBATCH --job-name=eval_report
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/eval_report_%j.log

# ============================================================================
# Evaluation Report Generation
# Runs after all analyses complete
# Runs: 08_evaluation_report.py
# ============================================================================

echo "=== Evaluation Report Job ==="
echo "Date: $(date)"
echo "Node: $(hostname)"

SCRATCH="${SCRATCH:-/path/to/scratch/narrow_model_safety_eval}"
PROJECT_DIR="${SCRATCH}/Narrow_Model_Safety_Eval"
mkdir -p ${SCRATCH}/logs

source ~/miniconda3/miniconda3/etc/profile.d/conda.sh
conda activate narrow_model_safety

cd ${PROJECT_DIR}
python src/08_evaluation_report.py

echo ""
echo "=== Pipeline Complete ==="
echo "Results: ${PROJECT_DIR}/results/"
echo "Figures: ${PROJECT_DIR}/results/figures/"
echo "Report:  ${PROJECT_DIR}/results/evaluation_report.txt"

echo "=== Done: $(date) ==="
