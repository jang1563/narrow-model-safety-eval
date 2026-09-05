#!/bin/bash
#SBATCH --job-name=nmse_expanded_sweep
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/nmse_expanded_sweep_%j.log

# Re-run every model arm on the expanded 80-positive panel.
#
# Why this exists as one script rather than re-using the per-family scripts:
# after the 2026-09-05 class expansion the 650M mean-pool arm was regenerated on
# 80 positives while every other arm still described the 66-protein panel, so any
# cross-model comparison mixed two panels. The per-family scripts also ran
# DIFFERENT downstream analyses per arm (member_separability only for ESM-3/ESM-C,
# classifier_sweep only for pooling/ProtT5/SaProt, and so on), which is a second
# comparability problem: an arm could look different because it was measured
# differently. This runs the SAME downstream set for every arm.
#
# Failures are recorded and reported at the end rather than swallowed by `|| true`.
# A silently-skipped step is what left a stale lomo_results.json looking current.
#
#   PROJECT_DIR=... PYTHON_BIN=... sbatch slurm/expanded_panel_full_sweep.sh

set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:?set PROJECT_DIR}"
PY="${PYTHON_BIN:?set PYTHON_BIN}"
SEEDS="${SEEDS:-30}"
cd "$PROJECT_DIR"; mkdir -p logs
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.hf_cache}"

echo "=== NMSE expanded-panel sweep ==="; date; hostname
echo "panel: $(grep -c '^>' data/sequences/toxins_positive_v2.fasta) positives / $(grep -c '^>' data/sequences/benign_negatives_v2.fasta) negatives"
python3 -c "import json;d=json.load(open('data/annotations/mechanism_classes_v2.json'));print('annotation members:',len(d['proteins']))"

FAILED=()
run() {  # run <label> <cmd...>
    local label="$1"; shift
    echo; echo "---- $label ----"; date
    if ! "$@" 2>&1 | grep -v -e Warning -e '^ESMC:'; then
        echo "!! FAILED: $label"; FAILED+=("$label")
    fi
}

# ---- stage 0: structures for the 14 new positives -------------------------
# 02h skips every step whose output exists, so this fetches only the missing
# AlphaFold entries and rebuilds the 3Di table over all 234 panel proteins.
run "02h saprot prepare (3Di for new members)" "$PY" src/02h_saprot_prepare.py

# ---- stage 1: embeddings, one per arm -------------------------------------
run "embed esm2_650M (untagged, canonical)" "$PY" src/02b_esm2_embed_v2.py --model facebook/esm2_t33_650M_UR50D
for M in "esm2_t6_8M_UR50D:esm2_8M" "esm2_t12_35M_UR50D:esm2_35M" \
         "esm2_t30_150M_UR50D:esm2_150M" "esm2_t36_3B_UR50D:esm2_3B"; do
    run "embed ${M#*:}" "$PY" src/02b_esm2_embed_v2.py --model "facebook/${M%%:*}" --tag "${M#*:}"
done
run "embed esmc_300M" "$PY" src/02e_esm3_esmc_embed.py --model esmc_300m       --tag esmc_300M
run "embed esmc_600M" "$PY" src/02e_esm3_esmc_embed.py --model esmc_600m       --tag esmc_600M
run "embed esm3_1_4B" "$PY" src/02e_esm3_esmc_embed.py --model esm3_sm_open_v1 --tag esm3_1_4B
run "embed pooling variants" "$PY" src/02f_pooling_variants.py --model facebook/esm2_t33_650M_UR50D --tag_prefix esm2_650M
run "embed prott5_xl" "$PY" src/02g_prott5_embed.py --tag prott5_xl
run "embed saprot_650M" "$PY" src/02i_saprot_embed.py --tag saprot_650M

# ---- stage 2: the SAME downstream set for every arm -----------------------
ARMS=(NONE esm2_8M esm2_35M esm2_150M esm2_3B esmc_300M esmc_600M esm3_1_4B
      esm2_650M_mean esm2_650M_max esm2_650M_cls prott5_xl saprot_650M)

for T in "${ARMS[@]}"; do
    if [[ "$T" == "NONE" ]]; then ARG=(); LABEL="untagged (650M)"; ARR="results/v2/embeddings_positive_v2.npy"
    else ARG=(--tag "$T"); LABEL="$T"; ARR="results/v2/embeddings_positive_v2_${T}.npy"; fi
    echo; echo "############ $LABEL ############"
    if [[ ! -f "$ARR" ]]; then echo "  no embeddings, skipping"; FAILED+=("$LABEL: no embeddings"); continue; fi
    run "$LABEL 03b lomo"            "$PY" src/03b_leave_one_mechanism_out.py "${ARG[@]}"
    run "$LABEL 03c ablation"        "$PY" src/03c_ablation_baselines.py      "${ARG[@]}"
    run "$LABEL 03d localization"    "$PY" src/03d_localization_confound.py   "${ARG[@]}"
    run "$LABEL 03e negcurve"        "$PY" src/03e_negative_difficulty_curve.py "${ARG[@]}" --seeds "$SEEDS"
    run "$LABEL 03f coverage"        "$PY" src/03f_coverage_strictness.py     "${ARG[@]}" --seeds "$SEEDS"
    run "$LABEL 03g memb-separab"    "$PY" src/03g_member_separability.py     "${ARG[@]}"
    run "$LABEL 03h probe-vs-sim"    "$PY" src/03h_probe_vs_similarity.py     "${ARG[@]}" --seeds "$SEEDS"
    run "$LABEL 03j classifiers"     "$PY" src/03j_classifier_sweep.py        "${ARG[@]}" --seeds "$SEEDS"
    run "$LABEL 03k margin holdout"  "$PY" src/03k_margin_holdout.py          "${ARG[@]}"
done

# ---- stage 3: panel-level, model-free or cross-model ----------------------
run "03i alignment baseline" "$PY" src/03i_alignment_baseline.py
run "03l ensemble"           "$PY" src/03l_ensemble_alignment_embedding.py
run "04 scale sweep report"  "$PY" src/04_scale_sweep_report.py

echo; echo "=== done ==="; date
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "STEPS THAT FAILED (${#FAILED[@]}):"; printf '  - %s\n' "${FAILED[@]}"
else
    echo "every step succeeded"
fi
