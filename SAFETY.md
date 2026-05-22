# Responsible Use and Safety Scope

The Narrow Scientific Model Safety Evaluation framework measures dual-use
risk in narrow scientific AI models (protein language models, structure-based
designers) using **statistical metrics over public reference proteins**. It
is intended for safety evaluation research, not for capability uplift.

See also [`DISCLAIMER.md`](DISCLAIMER.md) for the project-level safety scope.

## In Scope

- Measuring whether protein language model representations encode functional
  information about well-characterized public proteins (UniProt, PDB, VFDB)
- Quantifying whether structure-based design models recover known functional
  residues from backbone alone (FSPE, FSI metrics)
- Assessing physical realizability barriers (synthesis, folding, assembly,
  regulatory) as a bridge between computational predictions and real-world
  feasibility
- Reproducing the aggregate metrics on alternative narrow models and
  alternative protein sets to extend the framework

## Out of Scope

- Generation of novel dangerous sequences, structures, or design recipes
- Use of FSPE / FSI metric formulas as a target objective for an attack
  pipeline (the framework is designed as a *measurement*, not a generator)
- Sole reliance for any deployment risk-assessment decision; this is one
  evaluation framework among many that should be used together
- Reframing aggregate AUROC numbers (e.g. 0.981 embedding separability) as
  evidence that a specific deployment is or is not safe
- Adversarial reuse: probing for which proteins / sites / models are
  most "evaluable" using the framework as a recipe

## Withheld Content

- All evaluated proteins are from public databases; no novel dangerous
  sequence is generated or disclosed by this work.
- Per-residue functional-site coordinates are released only at the level
  required to reproduce the metric on a public dataset (UniProt / PDB
  identifiers + standard annotation databases).
- No synthesis pathways, codon-optimized sequences, or expression vectors
  appear in the repository.

## Reporting Concerns

- Open a GitHub issue with the `safety` label for: a metric definition
  that could be repurposed as a design objective; an evaluated protein
  whose presence in the corpus warrants redaction; a result table that
  could be misread as a recipe.
- For sensitive disclosures, email jak4013@med.cornell.edu with subject
  "NARROW-MODEL SAFETY"
- Do not paste operational sequence detail into public issues.

## Limitations Recap

- Single primary annotator for the functional-site corpus; expert review pending
- AUROC and FSPE results are model + protein-set specific; not a global
  capability claim about the protein language model in isolation
- Physical Realizability Tier mapping is conservative; real-world
  feasibility depends on specific facility/expertise/regulatory context
