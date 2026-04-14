# Safety & Ethics Disclaimer

This project evaluates the dual-use potential of narrow scientific AI models
(ESM-2, ProteinMPNN) for **defensive safety evaluation purposes only**.

## What this project does

- Measures whether protein language model representations encode functional
  information about well-characterized proteins from public databases
- Quantifies whether protein design models recover known functional residues
  from backbone structures alone
- Introduces metrics (FSPE, FSI) for evaluating dual-use risk in narrow
  scientific models
- Assesses physical realizability barriers (synthesis, folding, assembly,
  regulatory) to bridge computational predictions with real-world feasibility

## What this project does NOT do

- Does NOT provide synthesis protocols, lab procedures, or instructions for
  producing dangerous materials
- Does NOT publish individual dangerous sequences — only aggregate statistical
  metrics are reported
- Does NOT enhance the capability of any model — it evaluates existing
  capabilities

## Data sources

All protein sequences and structures used in this evaluation are from publicly
available databases (UniProt, RCSB PDB, VFDB) and published literature. No
novel dangerous sequences are generated or disclosed.

## Intended audience

AI safety researchers, biosecurity policy makers, and scientific AI developers
who need evaluation frameworks for narrow scientific models.
