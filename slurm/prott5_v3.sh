#!/bin/bash
#SBATCH --job-name=nmse_t5_v3
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/nmse_t5_v3_%j.log

# ProtT5 on panel v3, the first arm outside the EvolutionaryScale lineage to see it.
#
# Why
# ---
# § 10.6.2 found that the two classes ESM-2 misses on v3 are each reached by a DIFFERENT model:
# ESM-C 600M takes beta-lactamase to 40.5% and leaves phage at 12%, ESM-3 1.4B takes phage to 31.7%
# and leaves beta-lactamase at 2.9%. Both are EvolutionaryScale models, so that dissociation is so
# far a statement about one lineage. ProtT5 is a Rostlab T5 encoder trained on UniRef50 with span
# corruption: different architecture, objective and corpus at once.
#
#   - if ProtT5 reaches one of the two classes, "reachable by some representation" is not an
#     EvolutionaryScale property and the joint-property reading in § 10.6.2 strengthens
#   - if it reaches neither while its overall separability holds, the two failures are harder than
#     the dissociation makes them look and lineage was the wrong variable
#
# SaProt is the other missing arm and is NOT run: structure_3di_v3.json carries the no_structure
# mask for the 214 members v3 added, so 48.1% of that panel would be scored sequence-only. src/02i
# computes that fraction and refuses. Fetching those structures is a separate job.
#
#   PROJECT_DIR=... PYTHON_BIN=.../envs/narrow_model_safety/bin/python \
#     /opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch slurm/prott5_v3.sh

set -uo pipefail          # NOT -e: a failing step must not cancel the ones after it
echo "=== NMSE ProtT5 on panel v3 ==="; date; hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

PROJECT_DIR="${PROJECT_DIR:?set PROJECT_DIR}"
PY="${PYTHON_BIN:?set PYTHON_BIN}"
cd "$PROJECT_DIR"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.hf_cache}"
mkdir -p "$HF_HOME" logs

"$PY" -c 'import torch,transformers;print("torch",torch.__version__,"transformers",transformers.__version__,"cuda",torch.cuda.is_available())'

echo; echo "############ embed v3 with ProtT5 ############"; date
"$PY" src/02g_prott5_embed.py --tag prott5_xl --panel v3 2>&1 | grep -v 'it/s\]'

echo; echo "############ leave-one-mechanism-out, v3 / prott5_xl ############"; date
"$PY" src/03b_leave_one_mechanism_out.py --panel v3 --tag prott5_xl 2>&1 | grep -v Warning

# src/30 discovers arms from the filesystem, so this picks up the new one on its own.
echo; echo "############ margin across arms, v3 ############"; date
"$PY" src/30_margin_across_arms.py --panel v3 2>&1 | grep -v Warning

echo; echo "=== done ==="; date
