#!/bin/bash
#SBATCH --job-name=nmse_esmc_v3
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/nmse_esmc_v3_%j.log

# Embed the v3 panel with the non-ESM-2 architectures, and run leave-one-mechanism-out on each.
#
# Why this job exists
# -------------------
# Every v3 arm in this repository is an ESM-2 arm, because src/02e was hardcoded to v2 while
# src/02b had already been given --panel. That is a loader gap, not a decision, and it scoped a
# scientific claim: ESM-C 600M is the only one of fourteen arms that recovers beta-lactamase
# above alignment (48.3% [43.9, 52.7] at 30 seeds against 29.5%), and it has never been run
# against v3, which contains BOTH beta-lactamase and the second unreachable class, phage
# peptidoglycan hydrolase. v3 therefore answers the question within a single embedding source.
#
# Both outcomes are publishable:
#   - if ESM-C 600M also recovers phage, the two unreachable classes are a property of the ESM-2
#     family rather than of the panel, and § 10.6's generality claim weakens.
#   - if it recovers beta-lactamase and not phage, the two failures have different causes, which
#     no result here currently distinguishes.
#
# Step 0 is the one that makes any of it quotable
# -----------------------------------------------
# The v2 ESM-C arrays were built under esm 3.4.0 / torch 2.11.0. That environment no longer
# exists on this cluster: the only env with the esm SDK is now 3.2.1 / torch 2.5.1, where
# LogitsConfig(return_mean_embedding=True) is rejected and src/02e pools the residue stack
# itself. So the v2 panel is re-embedded here under the new path and compared to the stored
# arrays, row by row, BEFORE any v3 number is read. The comparison does not gate the run — the
# v3 per-class question is answered entirely within v3 — it decides whether the v3 figure may be
# put beside the published v2 48.3%.
#
#   PROJECT_DIR=... PYTHON_BIN=.../envs/narrow_model_safety/bin/python \
#     /opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch slurm/esmc_v3_arms.sh

set -uo pipefail          # NOT -e: a failing arm must not silently cancel the arms after it
echo "=== NMSE ESM-C / ESM-3 on panel v3 ==="; date; hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

PROJECT_DIR="${PROJECT_DIR:?set PROJECT_DIR}"
PY="${PYTHON_BIN:?set PYTHON_BIN}"
cd "$PROJECT_DIR"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.hf_cache}"
mkdir -p "$HF_HOME" logs

"$PY" -c 'import torch,esm;print("torch",torch.__version__,"esm",esm.__version__,"cuda",torch.cuda.is_available())'

echo; echo "############ step 0: is this pooling path the published one? ############"; date
"$PY" src/02e_esm3_esmc_embed.py --model esmc_600m --tag esmc_600M_mp --panel v2 2>&1 \
    | grep -v '^ESMC:' | grep -v 'it/s\]'
"$PY" src/56_embedding_source_equivalence.py --panel v2 --a esmc_600M --b esmc_600M_mp
echo "equivalence exit: $?"

run () {   # run <model> <tag>
    echo; echo "############ v3 / $2 ($1) ############"; date
    "$PY" src/02e_esm3_esmc_embed.py --model "$1" --tag "$2" --panel v3 2>&1 \
        | grep -v '^ESMC:' | grep -v 'it/s\]' || { echo "EMBED FAILED: $2"; return 1; }
    "$PY" src/03b_leave_one_mechanism_out.py --panel v3 --tag "$2" 2>&1 | grep -v Warning
}

run esmc_600m       esmc_600M        # the arm that recovers beta-lactamase on v2
run esmc_300m       esmc_300M        # within-family capacity control
run esm3_sm_open_v1 esm3_1_4B        # second architecture

# src/30 discovers arms from the filesystem, so this covers whatever embedded above.
echo; echo "############ margin across arms, v3 ############"; date
"$PY" src/30_margin_across_arms.py --panel v3 2>&1 | grep -v Warning

echo; echo "=== done ==="; date
