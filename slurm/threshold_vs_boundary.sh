#!/bin/bash
#SBATCH --job-name=nmse_thr_bnd
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/nmse_thr_bnd_%j.log

# Decompose 37's surprising result into decision boundary versus threshold estimation, and
# finish the arm set that job 3377580 left incomplete.
#
# No GPU. Every embedding this needs already exists from job 3377580, and the work is logistic
# regression on cached arrays.
#
#   PROJECT_DIR=... PYTHON_BIN=... sbatch slurm/threshold_vs_boundary.sh

set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:?set PROJECT_DIR}"
PY="${PYTHON_BIN:?set PYTHON_BIN}"
# 🔴 Run python unbuffered. Without -u, every print from these scripts sits in a block buffer
# until the process exits, so a SLURM log shows only the bash `echo` lines and a job that is
# working looks identical to a job that is hung. Job 3377636 was polled four times over ten
# minutes with no way to tell which it was.
PY="$PY -u"
cd "$PROJECT_DIR"; mkdir -p logs
export HF_HOME="${HF_HOME:-/athena/masonlab/scratch/users/jak4013/narrow_model_safety_eval/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"

fail=0
step() {
  echo; echo "=== $* ==="
  if ! "$@"; then echo "STEP FAILED: $*"; fail=$((fail+1)); fi
}

for f in results/v3/embeddings_positive_v3.npy results/v3/embeddings_negative_v3.npy \
         results/v3/embeddings_pool_large_esm2_650M.npy; do
  [ -s "$f" ] || { echo "MISSING REQUIRED INPUT: $f (job 3377580 should have made it)"; exit 2; }
done

# The question that decides whether 37's result is a finding or an artifact.
step $PY src/38_threshold_vs_boundary.py --arm esm2_650M

# 🔴 Job 3377580's across-arms step ran on three arms, not five, and the audit caught it. The 8M
# and 35M embeddings exist only on the laptop, because .npy is gitignored, so this machine had
# canonical, 150M and 3B while the committed artifact describes canonical, 8M and 35M. Two
# machines each holding a different partial arm set is how a pinned claim drifts. These two are
# cheap, so the fix is to have every arm on the machine that does the analysis.
step $PY src/02b_esm2_embed_v2.py --panel v3 --model facebook/esm2_t6_8M_UR50D --tag esm2_8M
step $PY src/03b_leave_one_mechanism_out.py --panel v3 --tag esm2_8M
step $PY src/02b_esm2_embed_v2.py --panel v3 --model facebook/esm2_t12_35M_UR50D --tag esm2_35M
step $PY src/03b_leave_one_mechanism_out.py --panel v3 --tag esm2_35M
step $PY src/30_margin_across_arms.py --panel v3

echo; echo "failed steps: $fail"
exit $fail
