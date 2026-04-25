#!/bin/bash
#SBATCH --job-name=foldseek_saprot
#SBATCH --partition=scu-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/foldseek_%j.log

# ============================================================================
# Foldseek Tokenization for SaProt — CPU Job (v2 Pillar 1C prerequisite)
#
# SaProt uses structure-aware vocabulary: each residue position is encoded as
# a two-character token (amino acid + Foldseek 3Di structural neighborhood code).
# This script processes all PDB files in data/structures/ through Foldseek's
# structureto3didescriptor to generate the combined token strings.
#
# Output: data/annotations/saprot_tokens.json
# Format: {"UNIPROT_ID": "Ac#dE&fG...", ...}  (aa + 3Di token per residue)
#
# Run BEFORE esm3_embed.sh if SaProt analysis is needed.
# This is CPU-only — no GPU required.
# ============================================================================

echo "=== Foldseek SaProt Preprocessing ==="
echo "Date: $(date)"
echo "Node: $(hostname)"

SCRATCH="${SCRATCH:?ERROR: set the SCRATCH environment variable to your cluster scratch path}"
PROJECT_DIR="${SCRATCH}/Narrow_Model_Safety_Eval"
FOLDSEEK_BIN="${SCRATCH}/foldseek/bin/foldseek"

mkdir -p ${PROJECT_DIR}/logs
mkdir -p /tmp/foldseek_tmp

source ~/miniconda3/miniconda3/etc/profile.d/conda.sh
conda activate narrow_model_safety

# Install Foldseek if not present (conda-forge)
if [ ! -f "${FOLDSEEK_BIN}" ]; then
    echo "Installing Foldseek via conda..."
    conda install -c conda-forge -c bioconda foldseek -y 2>/dev/null || \
    pip install foldseek 2>/dev/null || {
        echo "Attempting binary download..."
        mkdir -p ${SCRATCH}/foldseek
        wget -q https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz -O /tmp/foldseek.tar.gz
        tar -xzf /tmp/foldseek.tar.gz -C ${SCRATCH}/foldseek --strip-components=1
    }
    FOLDSEEK_BIN=$(which foldseek 2>/dev/null || echo "${SCRATCH}/foldseek/bin/foldseek")
fi

echo "Foldseek binary: ${FOLDSEEK_BIN}"

# Verify foldseek works
${FOLDSEEK_BIN} --help 2>/dev/null | head -3 || {
    echo "ERROR: Foldseek not working. Check installation."
    exit 1
}

cd ${PROJECT_DIR}

# Generate 3Di tokens for all PDB files
python - <<'PYEOF'
import json
import subprocess
import tempfile
from pathlib import Path
import sys

project_root = Path(".")
struct_dir = project_root / "data" / "structures"
annot_dir = project_root / "data" / "annotations"
foldseek_bin = subprocess.run(["which", "foldseek"], capture_output=True, text=True).stdout.strip()
if not foldseek_bin:
    foldseek_bin = sys.argv[1] if len(sys.argv) > 1 else "foldseek"

# Load functional sites to get UniProt <-> PDB mapping
with open(annot_dir / "functional_sites.json") as f:
    func_sites = json.load(f)

pdb_to_uniprot = {
    info["pdb_id"]: uid
    for uid, info in func_sites.items()
    if not uid.startswith("_") and "pdb_id" in info
}

saprot_tokens = {}

for pdb_file in sorted(struct_dir.glob("*.pdb")):
    pdb_id = pdb_file.stem
    uniprot_id = pdb_to_uniprot.get(pdb_id)
    if not uniprot_id:
        print(f"  Skipping {pdb_id}: no UniProt mapping")
        continue

    print(f"  Processing {pdb_id} ({uniprot_id})...")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "out.tsv"
        cmd = [
            foldseek_bin,
            "structureto3didescriptor",
            str(pdb_file),
            str(output_file),
            "--threads", "4",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            print(f"  ERROR: Foldseek failed for {pdb_id}: {result.stderr[:200]}")
            continue

        # Parse output: tab-separated, columns include sequence and 3Di tokens
        # Format: chain_id  sequence  3di_sequence
        if output_file.exists():
            with open(output_file) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        aa_seq = parts[1]
                        di3_seq = parts[2]
                        # Interleave: A3Di token per residue
                        combined = "".join(aa + di for aa, di in zip(aa_seq, di3_seq))
                        saprot_tokens[uniprot_id] = combined
                        print(f"    Token string length: {len(combined)} ({len(aa_seq)} residues)")
                        break

output_path = annot_dir / "saprot_tokens.json"
with open(output_path, "w") as f:
    json.dump(saprot_tokens, f, indent=2)

print(f"\nSaProt tokens saved to: {output_path}")
print(f"Processed {len(saprot_tokens)} proteins")
PYEOF

echo "=== Done: $(date) ==="
echo "SaProt tokens: ${PROJECT_DIR}/data/annotations/saprot_tokens.json"
echo "Now run: sbatch slurm/esm3_embed.sh (with --with_saprot flag active automatically)"
