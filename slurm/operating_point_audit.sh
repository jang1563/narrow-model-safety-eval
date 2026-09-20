#!/bin/bash
#SBATCH --job-name=nmse_op_audit
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/nmse_op_audit_%j.log

# Settle whether 37's three to fourfold negative-set gain is a discrimination gain or an
# operating-point change. 38 established that the two single-factor arms together fall 30 to 42
# points short of `both`, so the effect is an interaction and 38's `share bnd` says nothing. 39
# tests the candidate identity of that interaction by rethresholding `both`'s own model on the
# panel's 118 hard held-out negatives.
#
# No GPU and no model weights: this reads only .npy embeddings that job 3377580 already wrote.
#
#   PROJECT_DIR=... PYTHON_BIN=... sbatch slurm/operating_point_audit.sh

set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:?set PROJECT_DIR}"
PY="${PYTHON_BIN:?set PYTHON_BIN}"
# Unbuffered, for the reason written into slurm/negative_scaling_650M.sh: a buffered log makes a
# working job and a hung job look identical.
PY="$PY -u"
cd "$PROJECT_DIR"; mkdir -p logs

# These two are gitignored regenerable artifacts, so a fresh clone will not have them. Naming the
# producer in the failure message rather than silently dying on np.load.
for f in results/v3/embeddings_pool_large_esm2_650M.npy \
         results/v3/embeddings_positive_v3.npy results/v3/embeddings_negative_v3.npy; do
  [ -s "$f" ] || { echo "MISSING $f -- produce it with slurm/negative_scaling_650M.sh first"; exit 2; }
done
# 38's artifact is not required, but without it self-test S4 cannot run and the fit order is
# unpinned. Say so loudly rather than letting the run look fully checked when it is not.
[ -s results/v3/threshold_vs_boundary_esm2_650M.json ] \
  || echo "WARNING: 38's artifact absent, self-test S4 will be skipped and fit order is unpinned"

# 🔴 Exit-code aggregation, written out rather than chained. An earlier version of these two lines
# ended with a bare `rc=$?`, which captures only the LAST command, so a failing 39 followed by a
# passing 40 would have exited 0 and the job would have looked clean. Each step's status is captured
# at the point it is produced and the job fails if any of them did.
#
# STEPS lets one step be re-run alone without editing the file: STEPS=40 sbatch ...
# 40 runs twice, once per reservation split. The two are not alternatives: the gap between them is
# the measurement of how much the pool's 2.33x name redundancy was inflating the background.
STEPS="${STEPS:-39 40}"
fail=0
run() {
  echo; echo "=== $* ==="
  if ! $PY "$@"; then echo "FAILED: $*"; fail=$((fail+1)); fi
}
for n in $STEPS; do
  case "$n" in
    39) run src/39_operating_point_audit.py --arm esm2_650M ;;
    40) run src/40_fixed_background_operating_point.py --arm esm2_650M --split random
        run src/40_fixed_background_operating_point.py --arm esm2_650M --split name-disjoint ;;
    *)  echo "unknown step $n"; fail=$((fail+1)) ;;
  esac
done
echo; echo "failed steps: $fail"
exit $fail
