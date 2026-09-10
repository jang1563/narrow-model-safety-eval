#!/bin/bash
#SBATCH --job-name=nmse_esmc6b
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/nmse_esmc6b_%j.log
# Tests one hypothesis, not "another model arm".
#
# Beta-lactamase is recovered at 0-21% by every UniRef-only arm and at 51.4% by
# ESM-C 600M, the only arm above the alignment baseline. ESM-C was pretrained on
# UniRef 83M clusters plus MGnify 372M plus JGI 2B, metagenomic data 37.5% of the
# final mix, and beta-lactamases are among the most diverse families in
# environmental metagenomes. Corpus alone does not explain it: ESM-C 300M shares
# that corpus and reaches 15.7%. So any account needs corpus AND capacity.
#
# Prediction if that account is right: ESM-C 6B >= 600M on beta-lactamase.
# If 6B lands near 300M instead, the account is wrong and 600M is an anomaly whose
# cause is still unidentified. Both outcomes are informative, which is why this is
# worth one arm when "add a fourteenth model" was not.
set -uo pipefail
cd "$PROJECT_DIR"
export HF_HOME="$PROJECT_DIR/.hf_cache"
PY="$PYTHON_BIN"
TAG=esmc_6B

echo "=== which esmc names does the installed esm package expose? ==="
"$PY" - <<'PYEOF'
try:
    from esm.models.esmc import ESMC
    import inspect, esm
    print("esm package at:", esm.__file__)
    src = inspect.getsource(ESMC.from_pretrained)
    print(src[:600])
except Exception as e:
    print("probe failed:", type(e).__name__, e)
PYEOF

echo; echo "=== embed ==="
"$PY" src/02e_esm3_esmc_embed.py --model esmc_6b --tag "$TAG" 2>&1 | grep -vi warning | tail -20
if [ ! -f "results/v2/embeddings_positive_v2_${TAG}.npy" ]; then
  echo "!! embedding did not produce output; stopping"
  exit 1
fi

echo; echo "=== downstream, same nine analyses as every other arm ==="
for S in "03b_leave_one_mechanism_out" "03c_ablation_baselines" "03d_localization_confound" \
         "03g_member_separability" "03h_probe_vs_similarity" "03k_margin_holdout"; do
  echo "---- $TAG $S ----"
  "$PY" "src/${S}.py" --tag "$TAG" 2>&1 | grep -vi warning | tail -14 || echo "!! FAILED $S"
done
"$PY" src/03e_negative_difficulty_curve.py --tag "$TAG" --seeds 30 2>&1 | grep -vi warning | tail -6 || echo "!! FAILED 03e"
"$PY" src/03f_coverage_strictness.py --tag "$TAG" --seeds 30 2>&1 | grep -vi warning | tail -6 || echo "!! FAILED 03f"
"$PY" src/03j_classifier_sweep.py --tag "$TAG" --seeds 30 2>&1 | grep -vi warning | tail -8 || echo "!! FAILED 03j"
echo "=== done ==="
