#!/bin/bash
#SBATCH --job-name=nmse_pool_t5
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=logs/nmse_pool_t5_%j.log

# Two experiments that share a GPU allocation.
#
#   1. Pooling variants for ESM-2 650M. §6i showed alignment beats every embedding
#      method on beta-lactamase and §6j showed no classifier head rescues it, which
#      leaves mean pooling over a conserved active site as the leading explanation.
#      mean, max and cls come from one forward pass.
#   2. ProtT5, the first model outside the EvolutionaryScale lineage, so the
#      cross-model claims can be checked against a different architecture,
#      objective and training corpus.
#
#   PROJECT_DIR=... PYTHON_BIN=... sbatch slurm/pooling_and_prott5.sh
#
# Written as a file rather than an inline heredoc: an earlier attempt lost a level
# of shell escaping and both jobs died in one second on "$PY: command not found".

set -euo pipefail
echo "=== NMSE pooling + ProtT5 ==="; date; hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

PROJECT_DIR="${PROJECT_DIR:?set PROJECT_DIR}"
PY="${PYTHON_BIN:?set PYTHON_BIN}"
SEEDS="${SEEDS:-20}"
cd "$PROJECT_DIR"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.hf_cache}"
mkdir -p "$HF_HOME" logs

echo; echo "############ pooling variants, ESM-2 650M ############"; date
"$PY" src/02f_pooling_variants.py --model facebook/esm2_t33_650M_UR50D --tag_prefix esm2_650M
for P in mean max cls; do
    echo; echo "##### pooling=$P"
    "$PY" src/03b_leave_one_mechanism_out.py --tag "esm2_650M_$P" 2>&1 | grep -v Warning || true
    "$PY" src/03j_classifier_sweep.py --tag "esm2_650M_$P" --seeds "$SEEDS" 2>&1 | grep -v Warning || true
done

echo; echo "############ ProtT5 ############"; date
"$PY" src/02g_prott5_embed.py --tag prott5_xl
for S in 03b_leave_one_mechanism_out 03c_ablation_baselines 03d_localization_confound; do
    "$PY" "src/${S}.py" --tag prott5_xl 2>&1 | grep -v Warning || true
done
"$PY" src/03h_probe_vs_similarity.py --tag prott5_xl --seeds 30 2>&1 | grep -v Warning || true
"$PY" src/03f_coverage_strictness.py --tag prott5_xl --seeds 30 2>&1 | grep -v Warning || true
"$PY" src/03j_classifier_sweep.py --tag prott5_xl --seeds "$SEEDS" 2>&1 | grep -v Warning || true

echo; echo "=== done ==="; date
