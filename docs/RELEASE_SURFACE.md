# Release Surface

This project is released as an evaluation framework, not as a design-output
corpus. The public surface is intentionally limited to inputs from public
reference databases, annotations needed to reproduce metrics, aggregate
statistics, figures, and pipeline code.

## Published

- Public reference FASTA/PDB inputs from UniProt and RCSB PDB
- Functional-site and physical-realizability annotations
- Aggregate JSON results, summary tables, and figures
- Source code, tests, SLURM launch scripts, and documentation
- The Hugging Face dataset card and aggregate dataset files

## Withheld

- Model-generated design FASTA outputs
- Temporary temperature-sweep outputs
- Local ESMFold/ESM-IF1 generated structure artifacts
- Embedding arrays, model weights, caches, logs, and local deployment files
- Any synthesis protocol, wet-lab procedure, expression vector, or operational recipe

## Enforcement

The release policy is enforced in two places:

- `.gitignore` excludes generated outputs such as `results/proteinmpnn_output/`,
  `results/proteinmpnn_temp_sweep/`, `results/esmfold_structures/`, `*.npy`,
  and model-weight files.
- CI fails if generated design-output directories are accidentally tracked again.

Local reproduction may create withheld artifacts on a user's machine. Those
files are implementation byproducts, not part of the GitHub or Hugging Face
release.
