#!/bin/bash
#SBATCH --job-name=nmse_resp_pred
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/nmse_resp_pred_%j.log

# §10.9.1 leaves the per-class sign unexplained: at a fixed false-positive budget a larger benign set
# gives the phage class +23.5 points and beta-lactamase -14.0. This runs the boundary arm on all twelve
# classes and tests what predicts that, with the ceiling confound partialled out.
#
# 12 classes x 5 doses x 30 seeds = 1,800 logistic fits at up to 8,437 negatives. 39 did 1,350 in 25
# minutes on this partition, so budget about 35.
#
#   PROJECT_DIR=... PYTHON_BIN=... sbatch slurm/response_predictors.sh

set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:?set PROJECT_DIR}"
PY="${PYTHON_BIN:?set PYTHON_BIN} -u"
cd "$PROJECT_DIR"; mkdir -p logs

for f in results/v3/embeddings_pool_large_esm2_650M.npy \
         results/v3/embeddings_positive_v3.npy results/v3/embeddings_negative_v3.npy; do
  [ -s "$f" ] || { echo "MISSING $f -- produce it with slurm/negative_scaling_650M.sh first"; exit 2; }
done

$PY src/43_what_predicts_the_response.py --arm esm2_650M
rc=$?
echo; echo "exit: $rc"
exit $rc
