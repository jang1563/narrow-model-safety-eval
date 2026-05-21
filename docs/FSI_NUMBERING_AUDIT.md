# FSI Residue-Numbering Audit

**Date:** 2026-05-21
**Scope:** the 8 structures with computed FSI values in `results/fsi_results.json`.
**Trigger:** the 2026-05-20 accession correction flagged "pre-existing FSI
numbering offsets" in Cholera/SEB/Abrin for a separate audit.

## How FSI maps residues

`06_proteinmpnn_redesign.py` computes FSI on functional residues as follows:

1. `extract_wildtype_sequence()` reads CA atoms of the configured PDB chain and
   returns the residue numbers as written in the PDB file.
2. `map_uniprot_to_pdb_positions()` takes each annotated functional residue and
   looks its **number** up directly in that PDB residue-number list.
3. The residues fed in are `pdb_residues` when `use_pdb_numbering` is set,
   otherwise `catalytic_residues` (from `functional_sites.json`).

There is **no offset correction and no sequence alignment** — only a literal
number match. This produces two distinct failure modes:

- **Drop** — the residue number is absent from the resolved structure, so the
  site is silently discarded (a `WARNING` is printed but FSI still runs).
- **Mismap** — the residue number *is* present but points to a different amino
  acid than the annotated catalytic residue. No warning; FSI is silently
  computed on the wrong residues.

## Method

For each FSI structure, every annotated catalytic residue was checked against
the actual amino acid at that PDB residue number, using the expected residue
identity parsed from `residue_annotations` (e.g. `"His110"` → His). Results
were cross-checked against the UniProt FASTA sequence.

## Findings

| PDB  | Protein            | FSI residues used        | Resolves correctly | FSI mean | Verdict |
|------|--------------------|--------------------------|--------------------|----------|---------|
| 2AAI | Ricin A-chain      | catalytic [80,123,177,180,211] | 5 / 5        | 1.097    | **clean** |
| 3BTA | BoNT/A             | pdb_residues [222,223,226,261] | 4 / 4        | 2.866    | **clean** (−1 offset handled by `pdb_residues`) |
| 1ACC | Anthrax PA         | pdb_residues [302,320,425,427] | 4 / 4        | 0.000    | **clean** (phi-clamp residues correct) |
| 1Z7H | Tetanus LC         | catalytic [233,234,237,270]    | 4 / 4        | 1.753    | **clean** |
| 4HSC | Streptolysin O     | catalytic [530,533,535,537,538] (chain X) | 5 / 5 | 0.452 | **clean** |
| 1XTC | Cholera A1         | catalytic [7,9,61,110,112]     | 3 / 5        | 0.220    | **CONTAMINATED** |
| 1ABR | Abrin A-chain      | catalytic [74,123,167,170,198] | 2 / 5        | 1.127    | **CONTAMINATED** |
| 3SEB | Staph. enterotoxin B | catalytic [23,25,44,45,89,90,91,93,94] | ~1 / 9 | 0.702 | **CONTAMINATED** |

### Contaminated entries (mismap mode)

- **Cholera 1XTC** — `Arg7`, `Ser61`, `Glu112` resolve correctly; `Ser9`
  (PDB#9 = Asp) and `His110` (PDB#110 = Glu) do not. 2 of 5 sites wrong.
- **Abrin 1ABR** — `Tyr74`, `Trp198` resolve correctly; `Tyr123` (PDB#123 =
  Ser), `Glu167` (PDB#167 = Arg), `Arg170` (PDB#170 = Tyr) do not. 3 of 5 wrong.
  The annotations explicitly parallel ricin (Tyr80/Tyr123/Glu177/Arg180/Trp211),
  which is verified clean in 2AAI — the abrin numbers were not re-derived for
  the abrin sequence/structure.
- **SEB 3SEB** — only `Tyr90` matches (by coincidence). The other 8 numbers
  point to unrelated residues in both the PDB and the UniProt FASTA, so the
  list is wrong in every numbering scheme. SEB is a superantigen (no catalytic
  site); the list is meant to be MHC-II / TCR interface residues and needs full
  re-curation.

Consequence: the FSI values 0.220 (Cholera), 1.127 (Abrin) and 0.702 (SEB) in
`fsi_results.json`, `evaluation_report.*`, `mdrp_risk_table.json` and the FSI
figures are **computed on the wrong residues and are not interpretable**.

## Secondary finding — Anthrax annotation is internally inconsistent

`functional_sites.json` P13423 carries three residue sets that disagree:

- `catalytic_residues`: `[162,163,164,165,166,305,307,309]`
- `residue_annotations` keys: `162–166` and `302,320,425,427`
- `pdb_residues`: `[302,320,425,427]`

`305,307,309` in `catalytic_residues` match neither the annotations nor the
structure. FSI is unaffected (it uses `pdb_residues`), but **FSPE and FHS use
`catalytic_residues` as sequence positions**, so the Anthrax FSPE/FHS values are
computed partly on unannotated positions. This entry needs curation.

## Proteins correctly excluded from FSI

ExoU (4QMK), ExoS (1HE1), YopH (1PA9) and Colicin E2 (3U43) are not in
`fsi_results.json`. The audit confirms why: their catalytic residues fall
outside the resolved/renumbered structures (drop mode) — 1HE1 covers only the
ExoS GAP domain; 1PA9 and 3U43 are renumbered single-domain constructs; 4QMK
has a gap at ExoU Asp344. FSI for these needs dedicated per-structure curation.

## Recommendation

1. Re-curate `catalytic_residues` (and add `pdb_residues` where the structure
   is renumbered) for **Cholera, Abrin, SEB** against UniProt active-site
   feature tables and the deposited structures, then re-run `06`/`07`.
2. Until then, treat the Cholera / Abrin / SEB FSI values as **invalid** —
   exclude them from any reported FSI aggregate or flag them explicitly.
3. Fix the Anthrax `catalytic_residues` inconsistency (affects FSPE/FHS).
4. Consider adding an amino-acid identity check to
   `map_uniprot_to_pdb_positions()` so future mismaps fail loudly instead of
   silently.
