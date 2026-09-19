#!/bin/bash
#SBATCH --job-name=nmse_neg_scaling
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/nmse_neg_scaling_%j.log

# Embed the 8,259-protein benign pool with the canonical ESM-2 650M arm, then run the
# negative-set scaling curve on it.
#
# Why this is a SLURM job and not a local run
# -------------------------------------------
# The development machine cannot do it. On 2026-09-19 a local attempt stalled at 99% of
# weight loading with 78 MB resident and no CPU activity: swap was at 9,045 of 10,240 MB
# with 67 MB of free RAM, so 2.6 GB of weights had nowhere to materialise. An earlier hang
# of the same kind was misdiagnosed as a network stall and "fixed" with offline mode, which
# is why the diagnosis is written down here.
#
# What only this arm can answer
# -----------------------------
# §10.7.1 predicts that ADDING benign proteins pushes the classes that sit in a dense benign
# region downwards. Testing that needs headroom in the class being watched. On v3 the
# canonical arm has it in both failures, beta-lactamase at 18.6% and phage peptidoglycan
# hydrolase at 10.0%. The small arms do not: beta-lactamase is 1.4% on esm2_35M and 0.0% on
# esm2_8M, already at the floor, so a local run can only watch the phage class fall and is
# half the preregistered test.
#
#   PROJECT_DIR=... PYTHON_BIN=... sbatch slurm/negative_scaling_650M.sh

set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:?set PROJECT_DIR}"
PY="${PYTHON_BIN:?set PYTHON_BIN}"
cd "$PROJECT_DIR"; mkdir -p logs

fail=0
step() {
  echo; echo "=== $* ==="
  if ! "$@"; then echo "STEP FAILED: $*"; fail=$((fail+1)); fi
}

# The pool and the v3 panel must both be present. A missing pool is a setup error rather
# than something to work around: 34 builds it from UniProt and is not a GPU job.
for f in data/sequences/benign_pool_large.fasta data/sequences/toxins_positive_v3.fasta \
         results/v3/embeddings_positive_v3.npy results/v3/lomo_results.json; do
  [ -s "$f" ] || { echo "MISSING REQUIRED INPUT: $f"; exit 2; }
done

step "$PY" src/35_negative_scaling_curve.py --embed --arm esm2_650M
step "$PY" src/35_negative_scaling_curve.py --arm esm2_650M

# The remaining v3 arms, which the local machine also could not finish. 150M was the fourth
# arm for the across-arms check; 3B and the ESM-C family need this partition regardless.
step "$PY" src/02b_esm2_embed_v2.py --panel v3 --model facebook/esm2_t30_150M_UR50D --tag esm2_150M
step "$PY" src/03b_leave_one_mechanism_out.py --panel v3 --tag esm2_150M
step "$PY" src/02b_esm2_embed_v2.py --panel v3 --model facebook/esm2_t36_3B_UR50D --tag esm2_3B
step "$PY" src/03b_leave_one_mechanism_out.py --panel v3 --tag esm2_3B
step "$PY" src/30_margin_across_arms.py --panel v3

echo; echo "=== audit ==="
step "$PY" src/22_claims_audit.py

echo; echo "failed steps: $fail"
exit $fail
