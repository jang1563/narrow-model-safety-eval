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
# 🔴 Run python unbuffered. Without -u, every print from these scripts sits in a block buffer
# until the process exits, so a SLURM log shows only the bash `echo` lines and a job that is
# working looks identical to a job that is hung. Job 3377636 was polled four times over ten
# minutes with no way to tell which it was.
PY="$PY -u"
cd "$PROJECT_DIR"; mkdir -p logs

# Point at the shared scratch model cache, the same one slurm/esm2_embed.sh uses. Without this,
# transformers defaults to $HOME/.cache/huggingface, which would re-download ESM-2 650M (2.5 GB)
# onto the home filesystem even though scratch already holds it.
export HF_HOME="${HF_HOME:-/athena/masonlab/scratch/users/jak4013/narrow_model_safety_eval/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"
echo "HF_HOME=$HF_HOME"

fail=0
step() {
  echo; echo "=== $* ==="
  if ! "$@"; then echo "STEP FAILED: $*"; fail=$((fail+1)); fi
}

# Only GIT-TRACKED inputs are preconditions. An earlier version of this list also required
# results/v3/embeddings_positive_v3.npy, which is gitignored as a regenerable artifact, so a
# fresh clone would have failed the check immediately. Embeddings are produced below, not
# required above.
for f in data/sequences/benign_pool_large.fasta data/sequences/toxins_positive_v3.fasta \
         data/sequences/benign_negatives_v3.fasta \
         data/annotations/mechanism_classes_v3.json results/v3/lomo_results.json; do
  [ -s "$f" ] || { echo "MISSING REQUIRED INPUT: $f"; exit 2; }
done

# The v3 panel's own embeddings, which a fresh clone does not have.
step $PY src/02b_esm2_embed_v2.py --panel v3

# ⚠️ Device reproducibility, checked rather than assumed. results/v3/lomo_results.json is
# committed and was computed from MPS embeddings on a laptop; this partition is CUDA. Rather
# than silently overwriting a published artifact with a slightly different one, the committed
# file is preserved and the CUDA recomputation is written beside it for comparison. Any
# difference is a float-precision difference between devices, not a finding, and the published
# numbers stay the ones the audit pins.
cp results/v3/lomo_results.json results/v3/lomo_results.committed_mps.json
step $PY src/03b_leave_one_mechanism_out.py --panel v3
mv results/v3/lomo_results.json results/v3/lomo_results.cuda650M.json
mv results/v3/lomo_results.committed_mps.json results/v3/lomo_results.json
echo "--- per-class MPS vs CUDA difference, recovery at 95% specificity ---"
$PY - <<'PYCHECK'
import json
a = json.load(open("results/v3/lomo_results.json"))["leave_one_mechanism_out"]
b = json.load(open("results/v3/lomo_results.cuda650M.json"))["leave_one_mechanism_out"]
for c in sorted(a):
    d = (b[c]["flagged_95_mean"] - a[c]["flagged_95_mean"]) * 100
    flag = "  <-- differs by more than a point" if abs(d) > 1 else ""
    print(f"  {c:<34}{a[c]['flagged_95_mean']*100:6.1f}% -> {b[c]['flagged_95_mean']*100:6.1f}%  ({d:+.1f}){flag}")
PYCHECK

step $PY src/35_negative_scaling_curve.py --embed --arm esm2_650M
step $PY src/35_negative_scaling_curve.py --arm esm2_650M

# 35's design has a confound found while running it locally on esm2_35M: its n=296 point is a
# random pool subsample, not the panel's real matched negatives, so its curve answers "replace
# the negatives" rather than "add to them". 37 fixes that by keeping the panel's true 296 as a
# fixed floor at every K and separates random addition from addition of the K NEAREST pool
# proteins, the actual converse of §10.7.1's removal experiment. It also needs this arm: on
# esm2_35M beta-lactamase's extreme seed variance (a 21-point swing across ten seeds on a
# 14-member class) made the run uninterpretable, and phage_peptidoglycan_hydrolase did not even
# clear the script's own failure threshold there.
step $PY src/37_negative_supplement_from_pool.py --arm esm2_650M

# The remaining v3 arms, which the local machine also could not finish. 150M was the fourth
# arm for the across-arms check; 3B and the ESM-C family need this partition regardless.
step $PY src/02b_esm2_embed_v2.py --panel v3 --model facebook/esm2_t30_150M_UR50D --tag esm2_150M
step $PY src/03b_leave_one_mechanism_out.py --panel v3 --tag esm2_150M
step $PY src/02b_esm2_embed_v2.py --panel v3 --model facebook/esm2_t36_3B_UR50D --tag esm2_3B
step $PY src/03b_leave_one_mechanism_out.py --panel v3 --tag esm2_3B
step $PY src/30_margin_across_arms.py --panel v3

echo; echo "=== audit ==="
step $PY src/22_claims_audit.py

echo; echo "failed steps: $fail"
exit $fail
