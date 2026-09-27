#!/bin/bash
#SBATCH --job-name=nmse_saprot_v3
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/nmse_saprot_v3_%j.log

# SaProt on panel v3, the tenth arm, and the structure fetch it has been waiting for.
#
# Why this needs the cluster
# --------------------------
# structure_3di_v3.json carries real Foldseek strings for the 231 v2 members and the no_structure
# mask for the 214 v3 added, because src/27 built it by inheriting v2's entries: foldseek ships as a
# Linux static binary and the existing 3Di strings were produced here. SaProt reads an amino acid AND
# a structure token per position, so 48.1% of v3 would be scored sequence-only -- including the whole
# phage class, which is the class § 10.6.2 is about. src/02i computes that fraction and refuses above
# 5%, so the mask has to go before the arm can run.
#
# Steps: 02h fetches AlphaFold v6 cif for all 445 and runs foldseek over them, then 02i embeds, then
# 03b scores, then 30 and 41 pick the new arm up from the filesystem.
#
# 02h also checks itself. v3's 231 inherited strings are recomputed from AFDB and foldseek here, and
# if any comes back different the script exits 2 and says so, because that would put every 3Di number
# in the repository in question rather than just this arm's.
#
#   PROJECT_DIR=... PYTHON_BIN=.../envs/narrow_model_safety/bin/python \
#     /opt/ohpc/pub/software/slurm/24.05.2/bin/sbatch slurm/saprot_v3.sh

set -uo pipefail
echo "=== NMSE SaProt on panel v3 ==="; date; hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

PROJECT_DIR="${PROJECT_DIR:?set PROJECT_DIR}"
PY="${PYTHON_BIN:?set PYTHON_BIN}"
cd "$PROJECT_DIR"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.hf_cache}"
mkdir -p "$HF_HOME" logs

"$PY" -c 'import torch;print("torch",torch.__version__,"cuda",torch.cuda.is_available())'

echo; echo "############ step 1: structures and 3Di for panel v3 ############"; date
"$PY" src/02h_saprot_prepare.py --panel v3 --workdir "$PROJECT_DIR/.saprot_work" 2>&1 \
    | grep -vE 'it/s\]'
PREP=$?
echo "02h exit: $PREP"
if [ "$PREP" -ne 0 ]; then
    echo "!! 02h did not exit clean. Stopping before the embedding, because a changed 3Di string"
    echo "!! is a question about every 3Di number here and not just about this arm."
    echo "=== done ==="; date
    exit 0
fi

# --allow-masked 0.20 is deliberate and is not a way round the guard. AlphaFold DB has no model for
# any of the 32 phage_peptidoglycan_hydrolase members, so v3 cannot go below 18% masked by fetching
# harder: that class is unanswerable for a structure-aware model by construction. 02i now reports
# per-class coverage and names phage unreportable for this arm, so the beta-lactamase half of
# § 10.6.2's dissociation -- 14 of 14 with real structure -- can be measured while the phage half
# stays explicitly unmeasured rather than silently scored.
echo; echo "############ step 2: embed v3 with SaProt ############"; date
"$PY" src/02i_saprot_embed.py --tag saprot_650M --panel v3 --allow-masked 0.20 2>&1 \
    | grep -vE 'it/s\]'

echo; echo "############ step 3: leave-one-mechanism-out ############"; date
"$PY" src/03b_leave_one_mechanism_out.py --panel v3 --tag saprot_650M 2>&1 | grep -v Warning

echo; echo "############ step 4: margin across arms, v3 ############"; date
"$PY" src/30_margin_across_arms.py --panel v3 2>&1 | grep -v Warning

echo; echo "=== done ==="; date
