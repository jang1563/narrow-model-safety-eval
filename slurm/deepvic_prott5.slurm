#!/bin/bash
#SBATCH --job-name=dv_prott5
#SBATCH --partition=preempt_gpu
#SBATCH --qos=low
# Was --gres=gpu:h100:1, which pins this to the ONE H100 node and put slurm estimated
# start 1.5 days out. ProtT5-XL is ~3B params: fp16 is ~6GB, so an A40 (48GB) or A100
# (40/80GB) is ample and the type was over-specified. Any GPU now.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --output=logs/dv_prott5_%j.log
set -euo pipefail
echo "=== DeepVIC x ProtT5: a second model FAMILY ==="; date; hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
cd "$PROJECT_DIR"
source "${CONDA_SETUP}"; conda activate "${CONDA_ENV:-narrow_model_safety}"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.hf_cache}"
mkdir -p logs
python -V; python -c 'import torch;print("torch",torch.__version__,torch.cuda.is_available())'
python src/137_deepvic_prott5.py --selftest
python -u src/137_deepvic_prott5.py --batch "${BATCH_SIZE:-8}"
echo "=== done ==="; date
