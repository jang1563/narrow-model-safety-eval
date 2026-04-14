#!/bin/bash
# ============================================================================
# One-time environment setup for Cayuga HPC
# Run interactively on login node or Phobos
# ============================================================================

echo "=== Setting up narrow_model_safety conda environment ==="

# Set SCRATCH to your HPC scratch directory
SCRATCH="${SCRATCH:-/path/to/scratch/narrow_model_safety_eval}"
mkdir -p ${SCRATCH}/{logs,hf_cache}

# Source conda
source ~/miniconda3/miniconda3/etc/profile.d/conda.sh

# Create environment
conda create -n narrow_model_safety python=3.10 -y
conda activate narrow_model_safety

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install \
    transformers \
    fair-esm \
    biopython \
    scikit-learn \
    numpy \
    scipy \
    matplotlib \
    pandas \
    requests \
    prody

echo ""
echo "=== Environment setup complete ==="
echo "Activate with: conda activate narrow_model_safety"
echo ""
echo "=== Next steps ==="
echo "1. Copy project to scratch:"
echo "   rsync -av /path/to/Narrow_Model_Safety_Eval/ ${SCRATCH}/Narrow_Model_Safety_Eval/"
echo ""
echo "2. Run data collection (local or login node):"
echo "   cd ${SCRATCH}/Narrow_Model_Safety_Eval && python src/01_collect_data.py"
echo ""
echo "3. Submit GPU jobs:"
echo "   sbatch slurm/esm2_embed.sh"
echo "   sbatch slurm/esm2_masked.sh"
echo "   sbatch slurm/proteinmpnn.sh"
