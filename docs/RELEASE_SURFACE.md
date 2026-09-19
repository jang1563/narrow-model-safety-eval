# Release Surface

This project is released as an evaluation framework, not as a design-output
corpus. The public surface is intentionally limited to inputs from public
reference databases, annotations needed to reproduce metrics, aggregate
statistics, figures, and pipeline code.

## Published

- Public reference FASTA/PDB inputs from UniProt and RCSB PDB
- Functional-site and physical-realizability annotations
- Aggregate JSON results, summary tables, and figures
- Source code, tests, SLURM launch scripts, and documentation, **on GitHub**
- The Hugging Face dataset card and aggregate dataset files. ⚠️ The Hugging Face repository is a
  **dataset** surface: it carries the panels, annotations, aggregate results and documents, and only a
  partial copy of `src/`, because scripts land there when a sync touches them. GitHub is the code surface
  and the reproduction instructions on both surfaces clone it. A review pass on 2026-09-18 found 30 of 77
  scripts mirrored, which is why this is stated rather than left to be discovered.

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
- CI fails if generated design-output directories are accidentally tracked again,
  if result JSON files publish generated sequence payload keys, or if local
  Markdown links drift.

Local reproduction may create withheld artifacts on a user's machine. Those
files are implementation byproducts, not part of the GitHub or Hugging Face
release.

## Panel composition decisions, and why the v3 expansion is publishable

Added 2026-09-18 with panel v3. The policy above governs **what kinds of file** are published. It says
nothing about which proteins belong in a hazard panel, and the v3 expansion made that decision three times,
so the reasoning is recorded here rather than left in a script docstring.

**What was added.** Three non-animal-target mechanism classes, 69 positives: `bacteriocin` (15, kills
bacteria), `phage_peptidoglycan_hydrolase` (32, lyses bacterial cell wall), `cry_insecticidal` (22,
*Bacillus thuringiensis* delta-endotoxins acting on insect midgut). All are reviewed Swiss-Prot entries with
full public sequences, so they fall inside "public reference FASTA inputs from UniProt" above.

🔑 **The expansion does not raise the panel's human-hazard ceiling, and that is checkable rather than
asserted.** The already-published v2 panel contains anthrax edema factor and protective antigen
(`CYAA_BACAN`, `PAG_BACAN`), ricin (`RICI_RICCO`), Shiga toxin A (`STXA_SHIDY`), tetanus neurotoxin
(`TETX_CLOTE`), *Pseudomonas* exotoxin A (`TOXA_PSEAE`) and TSST-1 (`TSST_STAAU`), among botulinum
serotypes and staphylococcal enterotoxins. Every v3 addition targets **bacteria or insects**. Bt Cry
proteins in particular are sprayed on food crops and expressed in commercial Bt cultivars, so they are
among the most widely distributed insecticidal proteins in existence. Nothing in v3 is closer to a
human-health select agent than what v2 already publishes.

**Two candidates measured as admissible and deliberately not added.** `src/26_panel_growth_yield.py` found
five non-animal-target families that clear the panel's homology rule. Two were held back:

- **`plant_target_avirulence`, 32 admissible, the richest of the five.** Crop-targeting hazard is a
  different regulatory regime: USDA/APHIS PPQ select agents under 7 CFR 331.3(b) rather than the HHS
  human-health list. A panel that names plant-pathogen effectors as hazard positives is an agricultural
  biosecurity artifact, and the decision about whether that becomes public belongs at the start of that
  project rather than to whoever is adding sequences to this one. Not added.
- **`chitinase_antifungal`, 28 admissible.** Held back for a different reason, which is labelling rather
  than sensitivity: most reviewed chitinases are plant defence enzymes, so calling them hazard positives is
  a claim this panel's definition does not support. Sequence supply is not a hazard label.

⚠️ **The asymmetry this section fixes.** `src/27_expand_nonanimal_classes.py` documented why those two were
excluded and never stated affirmatively why the three admitted classes are acceptable to publish. An
exclusion rationale without an inclusion rationale reads as though nothing was decided about the classes
that went in.
