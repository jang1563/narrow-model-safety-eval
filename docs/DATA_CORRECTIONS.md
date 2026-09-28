# Data Corrections Log

This file records corrections to the protein panel after data-integrity review.
It exists for reproducibility and publication transparency.

**How the entries are numbered, since the sequence is not a clean per-date index.** The `(nth entry)`
label is a **stable citation handle, not a position**. Scripts and public documents cite entries by
ordinal, so an ordinal is never reassigned once anything points at it, and the sequence therefore carries
its history: 2026-09-05's ordinals start at "second" on its third entry, 2026-09-20's continue 2026-09-18's
count rather than restarting, and 2026-09-18's fourth entry was written two days late because the ordinal
had been reserved for a write-up that did not happen while `src/03x` cited it by number the whole time. The
gap-closing entry says so itself. Tidying any of this would break the references it exists to serve.

## 2026-05-20 — Mislabeled UniProt accessions in the v2 panel

### Summary

A verification pass against UniProt and the RCSB PDB found that **5 of the 16
annotated panel proteins** in `data/annotations/functional_sites.json` carried
UniProt accessions (and, in two cases, PDB structures) that did **not** correspond
to the intended protein. The annotations themselves (catalytic residues, mechanism
descriptions, literature references) were written for the correct proteins; only
the accession/structure identifiers were wrong. All five were corrected.

### How it was found

While expanding the FASTA panel for full 16-protein coverage, the newly fetched
sequences were cross-checked against their `functional_sites.json` catalytic-residue
annotations. Several sequences were too short to contain the annotated residues
(e.g. ExoU "Asp344" on a 316-aa sequence). Fetching the FASTA headers revealed the
accessions resolved to unrelated proteins. PDB titles and EBI SIFTS mappings were
then used to confirm the correct accessions and structures.

### Corrections

| Panel protein | Wrong accession | Actually was | Corrected to | Wrong PDB | Corrected PDB |
|---|---|---|---|---|---|
| ExoU phospholipase | `Q9HXZ2` | Acetyl-CoA carboxylase (ACCA_PSEAE) | `O34208` (687 aa) | `3TU3` (SpcU chaperone) | `4QMK` |
| ExoS ADP-ribosyltransferase | `P26471` | O-antigen ligase WaaL (WAAL_SALTY) | `Q51451` (453 aa) | — | `1HE1` (unchanged; correct) |
| YopH phosphatase | `P0A030` | Cell division protein FtsZ (FTSZ_STAAW) | `P15273` (468 aa, reviewed) | `2Y53` (aldehyde dehydrogenase) | `1PA9` |
| Colicin E2 DNase | `P02978` | Colicin E1 — a pore-former, not a DNase (CEA1_ECOLX) | `P04419` (581 aa, reviewed) | — | `3U43` (unchanged; correct) |
| Streptolysin O | `P0C0I2` | (accession does not exist in UniProt) | `P0DF97` (571 aa, reviewed) | — | `4HSC` (unchanged; correct) |

### Re-keyed entry: Botulinum neurotoxin type A

`functional_sites.json` keyed BoNT/A under `P10844`, but `P10844` is BoNT type **B**
(BXB_CLOBO). The genuine BoNT/A — matching the `3BTA` structure and the annotated
HExxH catalytic motif — is `P0DPI1` (BXA1_CLOBH). The entry was re-keyed to `P0DPI1`.
`P10844` (BoNT/B) remains in the positive FASTA as a generic toxin.

### Catalytic-residue renumbering

Two entries had catalytic-residue numbers that matched no sequence and were
corrected to UniProt-curated numbering (UniProt Active site / metal-binding features):

- **Colicin E2 (`P04419`):** `[544, 558, 569, 573]` → `[549, 574, 578]`
  (UniProt Zn²⁺-binding sites of the H-N-H endonuclease motif).
- **BoNT/A (`P0DPI1`):** `[222, 223, 226, 261, 227]` → `[223, 224, 227, 262]`
  (P0DPI1 UniProt numbering: HExxH His223/Glu224/His227, Zn ligand Glu262).

### PDB residue-numbering validation

A separate validation parsed every panel PDB structure and checked that catalytic
residues land on the expected amino acids. Sequence-based analyses (FSPE, FHS,
embedding separability) use UniProt-numbered `catalytic_residues` and were verified
correct for all 16 proteins. Structure-based FSI, however, requires PDB numbering:

- **BoNT/A / 3BTA:** clean −1 offset (verified 4/4 by Cα-trace identity). Fixed by
  adding `pdb_residues: [222, 223, 226, 261]` + `use_pdb_numbering: true`.
- **Pre-existing FSI numbering offsets** were found in Cholera (1XTC), SEB (3SEB),
  and Abrin (1ABR); these affect structure-based FSI only and are flagged for a
  separate audit (FSI silently drops residues whose PDB number does not match).
- The corrected ExoU/ExoS/YopH/Colicin E2 were **not** added to the FSI analysis:
  4QMK has a gap at ExoU res 344; 1HE1 resolves only the ExoS GAP domain (the
  ADP-RT catalytic residues are absent); 1PA9 is a renumbered YopH construct;
  3U43 contains only the isolated, renumbered Colicin E2 DNase domain. FSI
  expansion to these proteins needs dedicated per-structure curation.

### Impact on results

Results computed before this correction — FSPE / FHS / separability entries for
ExoU, ExoS, YopH (and the BoNT/Colicin labels) — were computed on the wrong
sequences and are invalid for those proteins. They are regenerated by re-running
`14_esm3_separability_fspe.py` and `15_sae_fhs.py` on the corrected panel.

### Files changed

- `data/annotations/functional_sites.json` — 5 entries re-keyed; 2 residue lists
  corrected; BoNT/A `pdb_residues` added; provenance recorded in `_accession_note`.
- `data/annotations/physical_realizability.json` — 6 entries re-keyed.
- `src/01_collect_data.py` — `positive_accessions`, `benign_accessions`,
  `PDB_STRUCTURES` corrected; `fetch_sequences_by_accessions` no longer filters by
  `reviewed:true` (ExoU/ExoS have only TrEMBL entries); added `--append_missing`.
- `src/18_realizability_automation.py` — `V2_NEW_PROTEINS`, `PUBMED_QUERIES` corrected.
- `src/08_evaluation_report.py` — Streptolysin O citation accession corrected.
- `data/sequences/*.fasta` — wrong sequences removed, corrected sequences added.
- `data/structures/` — `3TU3.pdb`, `2Y53.pdb` removed; `4QMK.pdb`, `1PA9.pdb` added.

## 2026-05-21 — Colicin E2 FASTA placement fix (follow-up)

### Summary

The 2026-05-20 accession correction re-keyed Colicin E2 to `P04419` in the
annotations but placed the sequences in the **wrong FASTA files**: the genuine
Colicin-E2 (`P04419`, 581 aa, the DNase toxin) was written into
`benign_homologs.fasta`, while the stale Colicin-E1 (`P02978`, a pore-former —
the original mislabeled protein) was left in `toxins_positive.fasta`.

### Impact

`14_esm3_separability_fspe.py` and `15_sae_fhs.py` build their sequence lookup
from `toxins_positive.fasta`, so Colicin E2 (`P04419`) was **silently skipped**
from FSPE and FHS, and a genuine toxin was counted in the benign/negative set,
contaminating the ESM-3 separability AUROC.

### Fix

- `data/sequences/toxins_positive.fasta` — `P02978` Colicin-E1 entry replaced by
  `P04419` Colicin-E2.
- `data/sequences/benign_homologs.fasta` — stray `P04419` entry removed.
- FSPE / FHS / separability re-run on the corrected panel: FHS now covers all 16
  panel proteins; ESM-3 FSPE covers 15 (VacA `P55981` has no catalytic residues
  and is correctly skipped); separability `n_positive` 68→71, `n_negative` 63→62.

### Report regeneration (2026-05-21)

The full ESM-2 pipeline was re-run on the corrected panel so the downstream
reports no longer carry stale sequences/accessions:

- `02_esm2_embed.py` → embeddings recomputed (positive 71, negative 62).
- `04_esm2_masked_prediction.py` → `fspe_results.json` (ESM-2 FSPE) now covers
  the 15 catalytic panel proteins with corrected accessions (no stale `P10844`).
- `03/05` → `separability_results.json` (AUROC 0.994→0.981) and
  `nearest_neighbor_results.json` recomputed.
- `08_evaluation_report.py` → `evaluation_report.json/.txt` regenerated — the
  16-protein risk matrix now uses corrected accessions throughout.
- `19_risk_table.py` → `mdrp_risk_table.json` regenerated.

**Known limitation:** `mdrp_risk_table.json` is keyed by PDB ID and merges in
the v1 FSI/SER results, which were **not** re-run for the corrected proteins
(see the structural-numbering note above). It therefore still carries the old
YopH structure `2Y53` (with an empty `uniprot_id`, since `2Y53` no longer maps
to any panel entry) instead of the corrected `1PA9`, and the ExoU/ExoS/Colicin
FSI columns remain blank. Resolving this needs the deferred FSI re-run/audit.

## 2026-05-21 — FSI residue-numbering audit

The "separate audit" of the FSI numbering offsets was carried out — see
[`FSI_NUMBERING_AUDIT.md`](FSI_NUMBERING_AUDIT.md). Result: of the 8 structures
with computed FSI, 5 are clean (Ricin, BoNT/A, Anthrax, Tetanus, Streptolysin O)
and **3 are contaminated** — Cholera (1XTC), Abrin (1ABR) and SEB (3SEB) have
`catalytic_residues` that resolve to the wrong amino acids in the structure, so
their FSI values (0.220 / 1.127 / 0.702) are not interpretable. The audit also
found the Anthrax (P13423) annotation internally inconsistent (`catalytic_residues`
lists `305,307,309`, which match neither the annotations nor the structure;
this affects FSPE/FHS, which key off `catalytic_residues`).

## 2026-05-22 — FSI re-curation (resolution of the audit)

The audit's recommendations were carried out (see the Resolution section of
`FSI_NUMBERING_AUDIT.md`):

- An **amino-acid identity check** was added to `map_uniprot_to_pdb_positions()`
  in `06_proteinmpnn_redesign.py` — mismaps now fail loudly at run time.
- **Cholera (P01555)** and **Abrin (P11140)** `catalytic_residues` were
  re-curated against UniProt active-site features and verified per-residue
  against the structures (0 mismatches after the fix). Cholera gained
  `pdb_residues` for the −18-offset 1XTC numbering.
- **SEB (P01552)** was excluded from FSI (`exclude_from_fsi`) — a superantigen
  with no catalytic site and no UniProt-annotated functional residues.
- **Anthrax (P13423)** `catalytic_residues` were re-keyed to the UniProt-numbered
  counterpart of `pdb_residues`, removing the bogus `305/307/309`.

The FSI / FSPE / SER / negative-control pipeline was re-run on the corrected
panel. Headline change: the count of structures with significant FSI elevation
(Wilcoxon + Holm–Bonferroni) fell from 5 to **3** (BoNT/A, Tetanus, ExoS). All
result files and the README / Hugging Face card carry the corrected values.

## 2026-05-25 — Risk table ESM-3/SaProt column bug

### Summary

`19_risk_table.py::load_esm3_fspe()` iterated all entries in
`esm3_fspe_results.json` without filtering by `model`, so SaProt entries
(listed after ESM-3 in the JSON) silently overwrote ESM-3 values for 5
proteins: P02879 (Ricin), P01555 (Cholera), P01552 (SEB), P13423 (Anthrax
PA), P11140 (Abrin), P04958 (Tetanus). The `fspe_esm3` column in
`mdrp_risk_table.json` therefore contained SaProt values where both models
were available.

A second pre-existing bug was also found: `load_fspe()` used
`data.get("results", [])` but `fspe_results.json` stores ESM-2 FSPE under
the key `per_protein`, resulting in the `fspe_esm2` column being empty in
any newly generated risk table.

### Fix

- `load_esm3_fspe()` now filters by `model == "esm3_sm_open_v1"`.
- New `load_saprot_fspe()` function filters by `model == "saprot_650m_af2"`.
- `load_fspe()` now checks `per_protein` key before falling back to `results`.
- `build_risk_table()` adds `fspe_saprot` column.
- `mdrp_risk_table.json` regenerated with correct values.

### Impact

The risk table now has three separate FSPE columns: `fspe_esm2`, `fspe_esm3`,
`fspe_saprot`. Cross-model directional discordances are visible (3 proteins
flip ratio sign across models). The mislabeled values affected any downstream
analysis that interpreted the `fspe_esm3` column as ESM-3 when it was
actually SaProt for 5/12 proteins.

## 2026-06-04 — FSPE headline table completed with Streptolysin O

### Summary

The headline FSPE table (README, Hugging Face card, evaluation report §5) listed
**7 proteins** and omitted Streptolysin O. This was a residue of the
2026-05-20 accession correction: Streptolysin O originally carried the
non-existent accession `P0C0I2`, so it had no ESM-2 FSPE value when the table
was first built. After re-keying to the reviewed accession `P0DF97`, the FSPE
pipeline produced a valid, nominally significant result for it
(`fspe_ratio = 0.509`, Mann–Whitney `p = 0.025`, `r = +0.58`, expected
direction), present in `results/fspe_results.json` but never added to the
displayed table. The `results/summary_risk_table.csv` preview (the 8-toxin
panel) did include it, surfacing the gap.

### Fix

- Added the Streptolysin O (`P0DF97`) row to the FSPE table in `README.md`,
  `huggingface/README.md`, and `docs/EVALUATION_REPORT.md`.
- Updated the descriptive statistic from "mean 0.66, 5/7 below 1.0" to
  **"mean 0.64, 6/8 below 1.0"** in all three documents.
- The pooled meta-analysis (74 functional vs 300 background residues,
  `p = 2.6 × 10⁻⁸`, `r = 0.41`) is unchanged — it was already computed over
  the full residue set and did not depend on which rows the table displayed.

### Impact

No metric value changed; one already-computed result that had been omitted
from the headline table is now shown. The displayed FSPE panel (8 proteins)
and the curated `summary_risk_table.csv` preview are now consistent. The FSI
panel remains 7 scored structures (SEB excluded as a superantigen), so FSPE
(8) and FSI (7) panel sizes legitimately differ.

## 2026-09-04 — The headline FSPE p-value was pseudoreplicated

### Summary

Every public surface led with a "pooled meta-analysis" of the FSPE result,
`p = 2.6 × 10⁻⁸` over 74 functional against 300 background residues. That test
runs a Mann–Whitney across residues pooled from all proteins, treating each
residue as an independent observation. **Residues within one protein are not
independent** — they share a sequence, a fold, and a single model forward pass —
so the test is pseudoreplicated and its p-value is inflated. It is also not a
meta-analysis, which would combine per-protein effect sizes rather than raw
residues.

**The direction of the finding is unchanged. The confidence attached to it was
overstated by roughly five orders of magnitude.**

### How it was found

A claim-by-claim audit of how this work is cited externally traced the figure to
its source and asked what the independent unit of analysis is. It is the protein,
n = 15, not the residue, n = 374.

### Corrections

Re-run at the protein level in `src/21_fspe_protein_level_test.py`
(`results/fspe_protein_level_test.json`):

| Test | Result |
|---|---|
| Proteins with FSPE ratio < 1.0 | **12 of 15** |
| Exact one-sided sign test | **p = 0.018** |
| Sign-flip permutation on the mean log ratio, 20,000 draws | **p = 0.0010** |
| Residue-pooled Mann–Whitney, for contrast | p = 2.6 × 10⁻⁸ (pseudoreplicated) |

### Fix

- `README.md`, `huggingface/README.md`, `docs/EVALUATION_REPORT.md` and
  `docs/ARCHITECTURE.md` now lead with the protein-level tests.
- The residue-pooled figure is retained everywhere it appeared, but explicitly
  labelled as descriptive and as treating residues within a protein as
  independent.
- `docs/EVALUATION_REPORT.md` limitations section carries a dated note naming the
  error rather than silently replacing the number.

### Impact

No measurement was recomputed and no conclusion reversed: ESM-2 is still more
confident at annotated functional sites than at background positions. What
changes is the strength of the claim, from `10⁻⁸` to roughly `10⁻³`. Any prior
citation of the `2.6 × 10⁻⁸` figure as the headline result should be read as
overstated.

## 2026-09-05 — The temperature-sensitivity claim overstated its range and its scope

### Summary

Two errors in the same sentence, on two public surfaces.

1. **Range.** The documents reported the ProteinMPNN sampling-temperature sweep as
   `T ∈ {0.05, 0.1, 0.2, 0.5}`. The artifact
   (`results/fsi_temperature_sensitivity.json`) sweeps `{0.05, 0.1, 0.2, 0.3}`.
   The highest temperature actually tested is **0.3, not 0.5**.
2. **Scope.** The sentence read "FSI remains robustly above 1.0 across sampling
   temperatures", stated generally. The sweep covers **two structures**, 3BTA and
   2AAI, and the claim is true of only one of them. **2AAI has a minimum mean FSI
   of 0.99 and only 48% of its designs above 1.0 at T = 0.1**, so it crosses the
   threshold inside the swept range. The artifact's own `interpretation` field for
   2AAI says "FSI drops below 1.0 at higher temperatures", which the documents
   contradicted.

### How it was found

Expanding `src/22_claims_audit.py` from 5 registered claims to 10. The audit
recomputes each headline number from its artifact, so the temperature entry failed
on first run.

### Corrections

| | Reported | Artifact |
|---|---|---|
| Sweep range | {0.05, 0.1, 0.2, **0.5**} | {0.05, 0.1, 0.2, **0.3**} |
| Structures swept | implied panel-wide | **2** (3BTA, 2AAI) |
| 3BTA min mean FSI | 2.56 | 2.5566 ✅ |
| 3BTA Spearman ρ | −0.80 | −0.7999 ✅ |
| 2AAI min mean FSI | not reported | **0.9907**, crosses 1.0 |

### Fix

`docs/EVALUATION_REPORT.md` and `huggingface/README.md` now state the real range,
name both structures, and say explicitly that the stability result holds for
BoNT-A and **does not generalize to the panel**. The audit guards all of it,
including a forbidden-string check on the old `0.05, 0.1, 0.2, 0.5`.

### Impact

The BoNT-A robustness result is unaffected and its two numbers were already
correct. What changes is that a single-structure result is no longer presented as
a panel-wide property, and the swept range is no longer overstated.

### Also corrected in this pass

`src/22_claims_audit.py` itself had two defects, both of the same family as the
errors it exists to catch:

- Its FSI entry verified the 12-row file aggregate (0.881), a quantity **no
  document reports**, while the documents report the 7-toxin subset (1.02). It
  passed while checking the wrong thing.
- Its ESM-IF1 entry read a key that does not exist, received `None`, and passed
  because the assertion tolerated `None`.

Both are fixed, and every entry now has a negative control: breaking the
corresponding document, or reintroducing a retired string, makes the audit exit 1.

---

## 2026-09-05 — Class expansion clobbered a curated eligibility list, and a stale results file

### Summary

The v2 leave-one-mechanism-out panel was expanded from **66 to 80 positives** (154 negatives
unchanged) to raise the classes that had only three or four members. 14 proteins were added,
none removed, none reclassified, each screened at normalized Smith-Waterman ≤ 0.30 against
existing class members and against already-accepted candidates. Two defects were introduced
by the integration step and are corrected here.

### Defect 1: a curated list overwritten with a size rule

`data/annotations/mechanism_classes_v2.json` carries `holdout_eligible_classes`, a **curated**
list of which mechanism classes are treated as holdout targets. `src/23_expand_small_classes.py`
recomputed it as `n >= 3`. Two consequences, neither of which was intended by the expansion:

- **`other_toxin_mechanism`** — a grab-bag of unrelated mechanisms kept deliberately out of the
  holdout set — was promoted into the results table as though it were a mechanism class.
- **`virulence_associated_non_toxin`** flipped from `holdout_eligible: false` to `true`.
  `src/03b_leave_one_mechanism_out.py` runs this class on purpose, as
  `targets = eligible | {virulence_associated_non_toxin}`, and reports it with the flag set
  false precisely to mark it a **non-mechanism control**. The recompute destroyed the flag whose
  only job was to say "this row is not a mechanism."

No class actually crossed a curation boundary in the expansion, so the correct list is identical
to the one before it. `holdout_eligible_classes` is restored to its 8 curated entries, per-member
flags are re-derived from it, and `src/23` now **asserts** that eligibility is unchanged rather
than deriving it from membership counts. Eligibility is a curation decision and must be edited
deliberately.

### Defect 2: a stale results file that looked current

The appended annotation entries initially omitted `short_name` and `fasta_index`. Every
downstream script tolerated this except `03b`, which raised `KeyError: 'short_name'`. The result
was a stale `lomo_results.json` describing the 66-protein panel sitting beside freshly
regenerated companions, with nothing in the repository indicating the two had come apart. This
is the same failure mode as the four drift defects that motivated `src/22_claims_audit.py`.

### Fix

`src/22_claims_audit.py` gained two entries, taking it from 10 claims to 12:

- **`v2 panel and LOMO results describe the same panel`** — compares the annotation, the panel
  manifest and the results file, checking per-class membership **name by name** rather than by
  count, so a same-size substitution cannot pass.
- **`v2 class eligibility is curated, not a size rule`** — pins that no member's
  `holdout_eligible` flag disagrees with the class list, that the virulence control is present in
  the results and flagged as **not** eligible, and that the `other_toxin_mechanism` grab-bag is
  not eligible despite having n ≥ 3.

Verified against three negative controls: restoring the stale 66-protein results file, re-applying
the `n >= 3` recompute, and flipping a single member's flag each cause the audit to exit 1. CI runs
this audit, so the v2 panel is no longer outside the audited surface.

### Effect on results

Recovery numbers for the expanded classes are reported in the research log. Two classes whose own
membership did **not** change nonetheless moved — pore-forming cytolysin +11.4 points at the 95th
percentile threshold, beta-lactamase −2.9 at the 99th — because adding positives shifts the fitted
probe and therefore the calibrated threshold. Only members already near that threshold can cross
it; t3ss and virulence, whose members are all saturated or floored, did not move at all. **A
per-class recovery figure is a joint property of the class, the rest of the positive set, and the
operating point, and should not be quoted without them.**

### Also corrected

Within-class redundancy was reported during integration as "superantigen 0.813, clostridial 0.494"
under an unrecorded normalization. Recomputed with the definition stated (local Smith-Waterman,
BLOSUM62, gap −11/−1, normalized by the smaller self-score), the maxima are **0.974** and
**0.911**. The claim those figures supported — that the expansion added no redundancy — holds: no
class's maximum within-class similarity rose except `adp_ribosyl_ab_toxin`, 0.027 → 0.250, still
inside the screen.

---

## 2026-09-05 (second entry) — "Beta-lactamase resists every configuration" was wrong when written

### Summary

Earlier write-ups of the mechanism-generalization work stated that beta-lactamase **resists every model
configuration tested** and that **plain alignment beats every embedding method on it**. Both statements
are wrong. **ESM-C 600M recovers 51% of the class**, against alignment's 30% and ESM-2 650M's 21%.

### How it was found

Every model arm was re-run on the expanded 80-protein panel so that cross-model comparisons would not mix
two panels. Reading the resulting table row by row surfaced the ESM-C 600M value, which did not match the
claim the documents were making.

### It was not caused by the panel change

On the previous 66-protein panel ESM-C 600M already scored **48.6%** on beta-lactamase, with per-seed
values `[50, 71, 43, 43, 36]`. The claim was therefore incorrect at the time it was written. It survived
because the class was summarized from the ESM-2 arms — where it is genuinely hard at every scale and every
pooling — and the ESM-C row was never checked against the summary.

### Corrected statement

Beta-lactamase is the hardest class for **11 of 12** configurations, and the only class where alignment
beats the canonical ESM-2 probe (30% against 21%). **It is not unreachable.** One representation recovers
about half of it, and the jump is within a lineage rather than across scale: ESM-C 300M reaches 16% and
ESM-C 600M reaches 51%. Nothing measured here explains why that architecture at that size succeeds where a
3B ESM-2 does not.

### Fix

`src/22_claims_audit.py` gained an entry (16 claims total) that reads **every** arm rather than a summary:
it pins that ESM-C 600M exceeds the alignment baseline, that ESM-2 650M does not, and that ESM-C 600M is
the **only** arm above alignment. A future arm that changes any of those three facts fails CI instead of
being summarized around.

The same entry checks a robustness property that came free with the re-run. The untagged 650M arm and the
`esm2_650M_mean` pooling arm are the same model and pooling embedded by two different scripts at different
batch sizes (8 and 4). The arrays are **not** byte-identical — they differ numerically — yet per-class
recovery agrees to **0.0 points on all nine classes** at five seeds, and the 30-seed classifier sweep
agrees to within **0.2 points** on every head. That is the stronger statement: the conclusions are stable
under a numerical perturbation of the embeddings, not merely reproducible bit-for-bit.

### Also corrected: a mislabelled model pair

The scale-sensitivity figure "Spearman −0.10" was described as the class ordering "between the extremes",
implying ESM-2 8M against 3B. It is the ordering between **8M and 650M**; 8M against 3B was +0.12 on the
old panel. On the 80-protein panel the two are **−0.13** and **−0.17** respectively, so the claim that
scaling redistributes negative-set fragility rather than removing it is unaffected — only the model pair
was named wrongly.

---

## 2026-09-05 (third entry) — A permutation test drew 200 shuffles while advertising 20,000

### Summary

`perm_p()` in `src/03g_member_separability.py` had the signature `perm_p(x, y, a, n=20000, seed=0)` and
drew `range(n // 100)` — **200 permutations, not 20,000**. Every p-value in
`results/v2/member_separability*.json` therefore had a resolution floor of 0.005 while carrying a parameter
that said 0.00005. The two significant features both reported `perm_p = 0.0`, which meant "0 of 200".

### How it was found

While writing the feature table into `docs/MECHANISM_GENERALIZATION.md` the reported `0.0` was about to be
rendered as "p < 0.001". Checking what number of draws actually backed it showed 200, which supports
"p < 0.005" and nothing stronger.

### Fix

The loop now runs the full `n`, and the p-value is `(hits + 1) / (n + 1)` so that a null never exceeded in
a finite sample is not reported as exactly zero. Re-run across all 13 model arms.

| feature | AUROC | p at 200 draws | p at 20,000 draws |
|---|---|---|---|
| margin | 0.960 | 0.0000 | **0.00005** |
| nearest training positive (cosine) | 0.949 | 0.0000 | **0.00005** |
| 5-mer similarity | 0.373 | 0.135 | 0.104 |
| nearest training negative | 0.420 | 0.325 | 0.317 |
| signal peptide | 0.425 | 0.330 | 0.288 |
| exported | 0.455 | 0.530 | 0.565 |
| length | 0.516 | 0.915 | 0.844 |

**No conclusion changes.** The two significant features remain significant and are now at the floor of a
20,000-draw test, and the five null features stay null. What changes is that the reported precision is now
the precision that was actually computed.

### Also corrected

- `src/03h_probe_vs_similarity.py` carried a docstring stating that holding out beta-lactamase removes
  "14 of 66 positives, 21%". On the expanded panel it is 14 of 80, 17.5%.
- `docs/MECHANISM_GENERALIZATION.md` §4 reported that six of nine classes are perfectly binary, measured at
  five seeds. At thirty seeds the count is still six but **not the same six**: virulence leaves the set and
  contact-dependent inhibition enters it, and four of nine classes change their always/intermediate/never
  breakdown. Only five classes are binary under both. The document now reports the seed dependence and
  states the robust version of the claim: recovery is concentrated at the extremes, with 8 genuinely
  intermediate members out of 72 at thirty seeds.

---

## 2026-09-11 — Beta-lactamase entered the panel through an undocumented, different route

### Summary

`beta_lactamase` (14 of 80 positives, the largest class in the panel) does not carry either UniProt keyword
this panel's stated hazard definition uses. Checked directly against UniProt: it carries **KW-0046,
"Antibiotic resistance"**, not KW-0800 (Toxin) or KW-0843 (Virulence). `src/01_collect_data.py` shows why —
an explicit `antimicrobial_resistance` query block, `protein_name:"beta-lactamase"`, added the class by
name match, bypassing the hazard-keyword query entirely. This was never stated in any document describing
the panel's construction.

### Why this is more than a bookkeeping gap

The field's own controlled vocabulary for this kind of screening, [FunSoCs](https://pmc.ncbi.nlm.nih.gov/articles/PMC9119117/),
treats antibiotic resistance as a category distinct from toxin and pathogenesis mechanisms, not a subtype
of either: it "encompass[es] sequences involved in the mechanisms of microbial pathogenesis, antibiotic
resistance, and eukaryotic toxins." All eight other mechanism classes in this panel are defined by direct
interaction with a host — a ribosome inactivated, a membrane perforated, a receptor bridged, a secretion
apparatus injected through. Beta-lactamase requires none of that; it hydrolyzes a diffusing small molecule
in the periplasm and can exist in a fully non-pathogenic organism with no relationship to virulence.

### Effect on existing claims

**No numeric claim changes.** All reported beta-lactamase figures (21% recovery, ESM-C 600M at 51.4%, the
6B refutation at 4.3%, alignment at 30%, s90 undefined) are computed the same way regardless of why the
class was included, and none of the fixes applied to other panel defects this session apply here — the
class itself was never mislabeled, only its provenance was undocumented.

**One new candidate explanation, alongside two refuted ones.** The corpus/capacity hypothesis for the
ESM-C anomaly was tested on ESM-C 6B and refuted (§ 9.3 of `docs/MECHANISM_GENERALIZATION.md`; this sentence cited a "2026-09-10 entry" of this log until 2026-09-24, and no entry of that date has ever existed — entry twenty-two). This is a third, independent
candidate: beta-lactamase may resist generalization not because of any property of the models tested, but
because it was never the same *kind* of hazard as the other eight classes, and an embedding trained to
recognize host-interacting toxins has no principled reason to key on it. This is stated as a candidate,
not a finding — it has not been tested against a panel that separates host-interacting AMR mechanisms
(efflux pumps, target modification) from purely enzymatic ones (beta-lactamases), which would be the direct
test.

### Fix

Documented in `docs/MECHANISM_GENERALIZATION.md` §2 (provenance) and new §9.4 (the candidate explanation).
No code or panel membership change: the class is real, its members are correctly identified UniProt
entries, and removing it would discard a legitimate hard case rather than correct an error. The error was
in never having written down that its inclusion criterion differed from the other eight classes.

---

## 2026-09-11 (second entry) — Systematic keyword audit of all 80 positives, prompted by the beta-lactamase finding

### What was checked

The beta-lactamase finding (previous entry) raised an obvious follow-up: is beta-lactamase the *only*
class where members lack the panel's stated hazard keywords (KW-0800 Toxin, KW-0843 Virulence), or was
this found only because someone happened to ask about that one class? All 80 positives were batch-queried
against the live UniProt REST API and checked for KW-0800/KW-0843.

### Result

| class | n | missing both keywords |
|---|---|---|
| **beta_lactamase** | 14 | **14 (100%)** |
| nuclease_dnase_rnase | 2 | 2 |
| phospholipase | 2 | 1 |
| pore_forming_cytolysin | 7 | 1 |
| all other classes (9) | 55 | 0 |

Five members outside beta-lactamase also lack both keywords. Each was checked individually rather than
assumed to be the same issue:

- **RNBR_BACAM (Barnase)** — already adjudicated. The `reason` field records a dated 2026-09-03 decision
  to keep it as a positive (cytotoxic, requires barstar co-expression for safe handling, mechanistically
  close to the RIP toxins already in the panel). Not a new finding.
- **MU1_REOVD (reovirus outer-capsid mu-1)** — confirmed via UniProt function text: "involved in host cell
  membrane penetration." Genuinely host-interacting; UniProt simply uses viral structural-protein keywords
  rather than KW-0800/0843 for this functional class of protein.
- **O34208_PSEAI (ExoU)** — a well-characterized *Pseudomonas aeruginosa* T3SS-injected phospholipase; its
  own `reason` field notes it requires the SpcU secretion chaperone, confirming a host-injected mechanism.
  UniProt tags it by biochemical activity (lipid metabolism) rather than by toxin role.
- **CEA2_ECOLX (colicin E2 DNase domain)** — carries KW-0044 "Antibacterial" (UniProt's own definition:
  "protein with antibacterial activity"), not KW-0800/0843. This is a real, partial parallel to
  beta-lactamase: colicin E2 kills competing bacteria, not a eukaryotic host. It differs from
  beta-lactamase in one respect worth keeping straight — it still requires active, receptor-mediated
  targeting and translocation into a competitor cell, which beta-lactamase never does at all. n=1 in a
  class of 2 that is not LOMO-eligible, so this does not touch any reported number.

### Conclusion

**Beta-lactamase remains the only class-wide categorical mismatch.** The other four cases are individual
proteins where UniProt's keyword tagging is narrower than, or oriented differently from, the protein's
real functional category (viral structural keywords, biochemical-activity keywords, or the
antibacterial-specific keyword), not panel-construction errors — colicin E2 is the partial exception, and
it is too small a class to affect anything reported. No panel membership changed as a result of this
check.

---

## 2026-09-11 (third entry) — The AMR-category hypothesis for beta-lactamase, preregistered and refuted

### The test

§9.4's candidate explanation for the beta-lactamase anomaly — that it fails to generalize because
antibiotic resistance requires no host interaction, unlike the other eight mechanism classes — was written
as a hypothesis with a stated, falsifiable prediction before any new data was touched
(`src/24_amr_category_test.py`). A second, independently screened antibiotic-resistance family
(aminoglycoside-modifying enzymes: AAC, ANT, APH, and one AAC/APH fusion; 8 members across three fold
families; ≤0.30 normalized Smith-Waterman against each other and against all 234 existing panel members;
none carrying KW-0800 or KW-0843) was embedded with the same frozen ESM-2 650M pipeline and scored against
a probe trained on the full internal panel, at the same 95th-percentile threshold used throughout.

Predicted: recovery ≤40% would support the hypothesis; ≥70% would refute it.

### Result

**Recovery: 87.5% (7 of 8). Refuted.** A second antibiotic-resistance family, spanning more fold diversity
than beta-lactamase's own members, generalizes about as well as the classes that require host interaction.
The one member not recovered (AAC6_SALEN, a GNAT-fold acetyltransferase) does not track fold family — its
GNAT-fold sibling in the bifunctional fusion protein (AACA_ENTFA) was recovered.

### Standing

Three candidate explanations for the beta-lactamase anomaly have now been tested and refused: the
classifier head (§9.1), pretraining corpus and model capacity (§9.3, ESM-C 6B), and the toxin/AMR
categorical distinction (this entry). The anomaly is unexplained, exactly as it was after the ESM-C 6B
result — this closes off one more plausible story rather than finding the real one. No panel membership
changed; the test class was embedded and scored externally and does not appear in
`data/annotations/mechanism_classes_v2.json` or any internal panel file.

### Fix

Result recorded in `results/v2/amr_category_test.json`, documented in `docs/MECHANISM_GENERALIZATION.md`
§9.4, and pinned in the audit so the refutation cannot be silently reframed as inconclusive or supportive
in later editing.

🔴 **Superseded the same day by the fourth entry below.** The test behind this result had two defects, and
the corrected verdict is inconclusive rather than refuted. This entry stays as written because it is the
dated record of what was believed when it was published.

---

## 2026-09-11 (fourth entry) — The AMR-category test above was run wrong, and its refutation was overstated

### How it surfaced

A question about whether beta-lactamase is a hazard class at all, given that it confers antibiotic
resistance and carries no human toxicity, sent me back to `src/24_amr_category_test.py` to re-read what the
test had actually compared. Two defects are visible in the source, and both make the refutation look
stronger than the data supports.

### Defect one: the test class's own category was in the training set

`24` fits on `X = np.vstack([P, N])`, the full internal positive set. **14 of those 80 positives are
beta-lactamases**, 17.5% of the positives and the largest class in the panel. The probe had already seen
antibiotic-resistance enzymes when it was asked to reach a new antibiotic-resistance family, so the result
measures generalization inside a category the probe already knew. The hypothesis was about whether a
representation trained on host-interacting toxins reaches antibiotic resistance at all.

`src/03b_leave_one_mechanism_out.py` does the opposite for every internal class: line 226,
`tri = [i for i in range(len(P)) if i not in set(hi.tolist())]`, drops the held-out class from training.
The two numbers that were compared came from opposite training rules.

### Defect two: the comparison figure came from a different protocol

The 87.5% was set against beta-lactamase's **21%** from the LOMO table. LOMO holds out 40% of the negatives
and calibrates the threshold on the held-out ones. `24` trains on every negative and calibrates in sample.
Holding the protocol fixed at `24`'s own and removing each class from training in turn, beta-lactamase
recovers **50.0%**, and it stays the lowest class in that column. The preregistered bounds in `24`
("≤40%, in the same range as beta-lactamase's 21%") were anchored to a figure from the other protocol.

### The corrected test

`src/25_amr_category_test_v2.py`, preregistered before re-embedding, runs the 2×2 under one protocol:

| | test: beta-lactamase | test: aminoglycoside |
|---|---|---|
| train with AMR | 100.0% (in sample) | 87.5% |
| train without AMR | 50.0% | **75.0%** |

Re-embedding reproduced the original 87.5% (7 of 8) exactly, so the four cells are comparable.

**Decisive cell: 75.0% (6 of 8). Inconclusive** by the corrected bounds, where ≤62.5% would have meant
contamination drove the original result and ≥87.5% would have meant it stood. One member moved:
AACC3_PSEAI, 0.076 to 0.0053.

### What changes and what does not

**No number in the earlier entry was wrong.** 87.5%, 7 of 8, and the per-member scores all reproduce
exactly. The error was the inference drawn from them.

**The strong hypothesis stays unsupported**, now for a better reason: under an identical training mask,
aminoglycosides recover 75.0% where beta-lactamases recover 50.0%, so lacking host interaction does not by
itself produce beta-lactamase's failure.

**A weaker version is live and unproven.** 75.0% sits below the 90 to 100% that host-interacting classes
reach under the same protocol.

**The standing tally changes.** "Three candidates tested, three refused" becomes two refused and one
inconclusive. The anomaly is still unexplained.

### The meta-defect

This is the **second preregistration in this project whose falsification criteria failed to cover the
outcome it got**. The first is in `docs/EXTERNAL_VALIDATION_PREREGISTRATION.md`, under "A flaw in this
preregistration, recorded rather than quietly fixed", where attempt 2 returned 100% on every arm against
criteria that set a floor and no ceiling. Both failures share a cause: bounds written without checking
that the figure they are anchored to comes from the same machinery as the test.

### Fix

Corrected test in `src/25_amr_category_test_v2.py`, result in `results/v2/amr_category_test_v2.json`,
§9.4 rewritten, and a new audit claim pins the corrected verdict together with the protocol-24 column so
the stronger reading cannot return. The original script, artifact, and entry stay in place as the record.

---

## 2026-09-18 — Two traps in testing the pretraining caveat, and two guards that fired one step too late

### The date trap

Testing §9.3 means finding sequences ESM-2 never saw. ESM-2 was pretrained on UniRef50 2021_04, so the
filter looks obvious: take UniProt entries created after that release. **It is the wrong field.**
`date_created` is the date an entry entered *Swiss-Prot*, not the date its sequence appeared. `Q9JXM7`
carries `date_created` 2022-12-14 and a sequence last updated **2000-10-01**; it sat in TrEMBL for two
decades and is certainly inside UniRef50 2021_04, which is built from UniProtKB including TrEMBL.

Filtering on `date_created` selects recently **reviewed** proteins rather than recently **discovered**
ones. It inflated the bacterial virulence pool from 13 to 119, and had it gone unchecked the published
claim would have been that a set of sequences ESM-2 had entirely memorised was a set it had never seen.
The correct filter is `date_sequence_modified` together with sequence version 1, and homology screening is
still required on top: the *E. coli* OspC3 ortholog scores 0.955 normalized Smith-Waterman against the
panel's *Shigella* OspC3, and two *Bacillus* alveolysins score 0.567 against streptolysin O.

### The pseudoreplication trap

The second design correlates per-member recovery against UniRef50 cluster size, needing no new sequences.
Pooled over 72 members it gives Spearman **rho −0.399, p 0.0005**, opposite in sign to what the caveat
predicts and entirely presentable as a finding.

It is an artifact of pooling. Within class the mean rho is −0.139 with no class below p 0.05; after
removing class means rho is −0.136, p 0.2554; excluding beta-lactamase rho is −0.199, p 0.1334; and at the
class level, which is the real n of 9, rho is −0.331, p 0.3846. Recovery is dominated by class structure,
so members are not independent observations. **This repository has already propagated one pseudoreplicated
p-value across four public surfaces.** The controls were run before the number was written down, which is
the only reason this entry records a near miss rather than a correction.

### Two guards that fired one step too late

Both this session's validity guards were written as fixed thresholds, and both were set just past the
value that actually occurred.

| guard | threshold written | value measured | caught? |
|---|---|---|---|
| `03k` degeneracy: calibration margins at zero | > 0.50 | **0.49** | no |
| `03n` validity: AUROC on the external set | ≤ 0.55 | **0.556** | no |

Each needed a second pass to catch its own motivating case. `03n` now bootstraps the AUROC and refuses any
result whose 95% interval contains 0.5, which rejects the genus-matched run at 0.556 [0.345, 0.762] on the
statistic's own uncertainty rather than on a number chosen by hand. A fixed cutoff on a noisy statistic
encodes an assumption about sample size that small panels violate.

### Standing

Both pretraining-holdout runs are recorded as INVALID in
`results/v2/pretraining_holdout.json` rather than deleted, and the exposure analysis records
`survives_controls: false`. §9.3's caveat remains true and unquantified, and
`docs/MECHANISM_GENERALIZATION.md` §9.3.1 now states what would settle it: pretraining a model with a
family withheld, which is a training run rather than an analysis.

---

## 2026-09-18 (second entry) — The §9.4 test input existed only in /tmp, and was published that way

### What happened

`src/24_amr_category_test.py` and `src/25_amr_category_test_v2.py` were run against a
candidate FASTA at `/tmp/aminoglycoside_final.fasta`, and both scripts were committed and
mirrored to Hugging Face still pointing at that path. The file was gone the next day. For
about a day, the published 87.5% and 75.0% in §9.4 **could not be reproduced by anyone,
including their author.**

This file's own header lists four founding defects, one of which is "numbers cited in a
document as the reason for a decision that had been computed once in a shell and never
saved." That is exactly this, re-created by the person who wrote the header.

### Recovery, and it is exact

The eight accessions were recorded in `results/v2/amr_category_test.json`, so the sequences
were recoverable from UniProt. Three candidate formats were reconstructed and compared
against the size and hash recorded in the session log:

| reconstruction | bytes | matches |
|---|---|---|
| UniProt FASTA with descriptions, wrapped at 60 | 3304 | no |
| headers cut to the first token, wrapped at 60 | 2546 | no |
| **headers cut to the first token, sequence unwrapped** | **2511** | **yes** |

The third is byte-identical to the lost original:
**sha256 4df8a5c65ad684e31bebfb6a101cea7c6dca9bfd6307fa4f38a1a2a68edcc5d2**.
Re-running `25` against it reproduces 87.5% (7/8) and 75.0% (6/8) exactly, and the
protocol-24 column reproduces too, so the recovery is confirmed functionally as well as
by hash.

### Fix

The file is committed at `data/sequences/amr_category_test.fasta`, both scripts now default
to that path instead of naming `/tmp`, and a new audit entry pins the byte count and hash so
the input cannot vanish or drift again. The lesson generalises past this instance: an input
that lives outside the repository is not an input, it is a memory of one.

## 2026-09-18 (third entry) — Three defects in the FHS metric, found while checking whether it could help with the beta-lactamase anomaly

### What was being checked

Five candidate explanations for beta-lactamase's LOMO failure had been tested and refused
(classifier head, corpus/capacity, AMR-as-a-category, layer depth, training-set composition).
The next candidate needed a way to look inside the representation rather than around it, and
`src/15_sae_fhs.py`'s Feature Hazard Score (FHS) — a sparse-autoencoder decomposition of the
ESM-2 residual stream — was the tool already in the repository for exactly that. Checking
whether it was usable surfaced three separate problems before it could be applied.

### Defect one: `research/05_v2_related_work_survey.md` describes a metric that was not built

That survey states, in the present tense, that SAE weights from `Elana/InterPLM-esm2-650m`
are "the primary input for Pillar 2 FHS computation. No training required," and that the
feature catalog was "used to build `data/annotations/motif_reference_set.json`."

Neither is true of what is in the repository. `results/fhs_results.json` records
`"sae_source": "trained_fallback"` — a 4096-dimensional linear SAE trained from scratch on
the 78-protein panel, not InterPLM's pre-trained decomposition. `data/annotations/
motif_reference_set.json` does not exist. `docs/EVALUATION_REPORT.md` already describes this
accurately elsewhere in the repository ("Sparse-autoencoder probes … trained locally … as
part of an exploratory FHS metric"), so the survey is the one document out of step, written
in April as a plan and never reconciled with what shipped.

### Defect two: the published FHS–FSI correlation is stale, and the file that produced it can be named

`fhs_results.json` stores `"fhs_fsi_spearman_r": 0.7005, "fhs_fsi_pvalue": 0.0112`, n=12.
Re-running `15`'s own pairing code — join on `uniprot`, read `fsi.mean` from
`results/fsi_results.json` — against the files currently in the repository gives
**rho 0.6585, p 0.0199**, not the stored figure.

The cause is dateable. `fhs_results.json` was last written 2026-05-21
(`e2522d2`, "Fix Colicin E2 FASTA placement; re-run FSPE/FHS/separability"), and its stored
correlation is consistent with the `fsi_results.json` that existed at that commit.
`fsi_results.json` was then rewritten the next day by the 2026-05-22 FSI re-curation entry
above, which re-keyed Anthrax's (`P13423`) `catalytic_residues` and changed its FSI to 0.
That entry's own text says "The FSI / FSPE / SER … pipeline was re-run on the corrected
panel" — SAE/FHS is not in that list, and it was not re-run. The correlation sitting in
`fhs_results.json` is therefore a real number, computed correctly, paired against a version
of `fsi_results.json` that no longer exists.

**The correlation is also fragile at this n.** Dropping `P13423` alone from the current
12-point pairing moves it to rho 0.582, p 0.0604 — the single most FSI-corrected protein in
the panel is also the one most responsible for the correlation clearing p<0.05.

### Defect three: the fallback SAE has no seed, so its own output cannot be reproduced

`train_fallback_sae()` in `15_sae_fhs.py` initializes `SimpleSAE` and shuffles training
batches with no `torch.manual_seed` anywhere in the script. Re-running `15` end to end — not
just recomputing a downstream correlation, but retraining the SAE itself — produces a
different set of FHS values each time, because the encoder that produces them was never
pinned. A metric with this property cannot be checked by anyone re-running the pipeline,
including its author.

### Standing

None of the sixteen individual FHS values in `results/fhs_results.json` are alleged to be
computed wrong — the numbers there are what that run of that fallback SAE produced. What is
wrong is: a public document describing a different, undelivered metric; a correlation figure
paired against a since-superseded artifact; and a metric whose defining computation is not
reproducible even in principle until it is seeded. `docs/EVALUATION_REPORT.md`'s own framing
of FHS as **exploratory** turns out to have been the load-bearing caveat.

### Fix

`research/05_v2_related_work_survey.md`'s InterPLM claim is corrected to describe the
fallback that actually ran. The stale correlation is superseded by a recomputation against
the current `fsi_results.json`, reported with the single-point sensitivity above rather than
as a clean p<0.05. Seeding the fallback SAE, or replacing it with InterPLM's pre-trained
weights (available at layers 1, 9, 18, 24, 30, 33 for ESM-2 650M, and loadable directly from
their `.pt` state dicts without the `interplm` package, which is not installed here), is
tracked as the next step rather than folded into this entry.


---

## 2026-09-18 (fourth entry) — The audit imported scipy for one Spearman call and broke CI, and this entry was skipped for two days

### Why this ordinal was empty

🔴 **This entry did not exist until 2026-09-20, and `src/03x_seed_stability_all_arms.py` cited it by number
the whole time.** Its docstring explains why it computes Spearman by hand: "The release-surface CI job
installs numpy only, and importing scipy into an audited path has already broken CI once (see
docs/DATA_CORRECTIONS.md, 2026-09-18, fourth entry)." The incident was real, the fix was committed, and the
write-up was never done, so a script's design rationale pointed at nothing. Found by sweeping the log's
ordinals and noticing the sequence runs third, fifth.

### What happened

The **third entry** above added an FHS-against-FSI Spearman correlation to `src/22_claims_audit.py`, using
`scipy.stats.spearmanr` for one call. It passed locally and failed CI with `ModuleNotFoundError`.

The audit's CI step installs **numpy and nothing else**, deliberately, so that the gate runs anywhere:

```yaml
- name: Audit headline claims against their artifacts
  run: |
    pip install numpy
    python src/22_claims_audit.py
```

Adding scipy to that job would have been the smaller diff and the wrong one, because the constraint is the
point. Commit `405d2a7` replaced the call with a numpy-only implementation: average ranks for ties, Pearson
on the ranks, and the two-sided p-value from the standard t approximation on n−2 degrees of freedom via a
continued-fraction incomplete beta.

### That it is the same function, checked rather than asserted

Verified against `scipy.stats.spearmanr` on **200 random vectors of length 5 to 20**: maximum absolute
deviation **3.8 × 10⁻¹¹** across both rho and p. The third entry's own values are unchanged, rho 0.6585 at
p 0.0199 and rho 0.5818 at p 0.0604 excluding P13423, so the claim it pins is identical and only its
dependency footprint is smaller.

### Standing

⚠️ scipy is still used by five scripts outside the audited path, `03p`, `04_esm2_masked_prediction`,
`05_esm2_nearest_neighbor`, `07_fsi_analysis` and `10_fsi_temperature_sensitivity`. The constraint applies
to `22_claims_audit.py` and anything it imports, not to the repository.

⚠️ The numbering is **not** renumbered to close the gap. Entries five, six and seven are cited by ordinal in
this log, in `src/03x`, `src/41` and `src/43`, and in the public documents, so shifting them would break
every one of those references to tidy a sequence.

### Fix

The entry exists. `src/03x` and `src/43` both hand-roll Spearman for this reason and both now point at
something. The general lesson is the one this log keeps recording from a new direction: **local green is not
CI green**, and on 2026-09-20 it recurred as a lint gate, with a local ruff 0.15.4 passing where CI's pinned
0.15.16 failed.

## 2026-09-18 (fifth entry) — The published beta-lactamase recovery is a 5-seed mean that falls outside its own 30-seed confidence interval

### What was being checked

`src/15e_sae_feature_space_lomo.py` needed its own copy of the 03b fold logic in order to run
leave-one-mechanism-out in the SAE feature space (§9.7). A second implementation of a published
protocol is a chance to check the first, so before comparing anything it was run on the raw
embedding and matched against `results/v2/lomo_results.json`.

🟢 **It reproduces all nine published class numbers exactly** at 5 seeds, to machine precision.
The re-implementation is faithful and §9.7's comparisons rest on it.

### The defect

`src/03b_leave_one_mechanism_out.py` line 69 fixes `SEEDS = [0, 1, 2, 3, 4]`. Every recovery
percentage in §3 is therefore a mean over five negative-holdout splits. `src/03v_lomo_seed_stability.py`
re-runs the identical fold logic at 30:

| class | published (5 seeds) | 30 seeds | sd | seeds at exactly 0% | 30-seed 95% CI |
|---|---|---|---|---|---|
| **beta_lactamase** | **21.4%** | **15.7%** | 12.5 | **7 of 30** | **[11.2, 20.2]** |
| contact_dependent_inhibition | 35.0% | 37.5% | 22.5 | 1 of 30 | [29.4, 45.6] |
| the other seven | (unchanged) | within 1.4 pt | ≤13.2 | 0 | contains the published value |

**21.4% is outside [11.2, 20.2].** Eight of nine classes are fine, six of them having zero seed
variance. The single class whose number is seed-sensitive is the anomalous class the whole of §9 is
about, which is the worst possible place for it.

### Why it is not a fabrication and not silently rewritten

Five seeds is what the preregistered protocol specifies, the number reproduces exactly from the
committed script, and no result elsewhere in the document depends on it beyond the class ordering,
which does not change. Rewriting 21.4% to 15.7% would replace a reproducible protocol number with a
different-protocol number and break the audit pins that tie the text to `lomo_results.json`.

So the figure stays and the reading changes: **21% is the optimistic end of a wide distribution whose
centre is nearer 16%**, stated in §9.7 with the interval and the seven zero-recovery splits. Anyone
quoting the class should quote the interval.

### Direction of the error

🔴 Worth being explicit, because the direction is the opposite of the usual worry: this correction makes
the paper's central anomaly **stronger**. The true recovery is lower than published, so beta-lactamase is
harder to reach than §3 says, and the seven refused explanations in §9 are refusing something larger.
A correction that happens to favour the author's thesis gets the same treatment as one that does not,
which is why the interval and the per-seed zeros are published rather than the single corrected mean.

### Standing

⚠️ **Five seeds is too few for any class whose sd exceeds a few points, and two classes here exceed 12.**
The @99 operating point and the fourteen model arms in §9 are still 5-seed means and have not been
re-run at 30. They are not corrected because nothing in the document turns on their exact values, but a
reader should assume the same ±10-point seed noise applies to the low-recovery cells in those tables.

### Fix

`src/03v_lomo_seed_stability.py` is committed with its artifact `results/v2/lomo_seed_stability.json`,
§9.7 carries the interval, and `src/22_claims_audit.py` pins the 30-seed mean, the interval and the
9-of-9 reproduction check so none of the three can drift from the text.

---

## 2026-09-18 (sixth entry) — Three §9 sentences compared arms by their 5-seed points, and the comparisons do not hold

### What was being checked

The fifth entry left the fourteen model arms as a standing caveat: they are 5-seed means and
had not been re-run. That caveat covered a headline, so it was not left standing.
`src/03x_seed_stability_all_arms.py` re-runs the beta-lactamase hold-out on **every cached
arm at 30 seeds**, at both operating points, with an interval from the seed distribution.

🟢 **The headline survives and improves.** §9 states that ESM-C 600M is the only arm that
beats plain alignment on beta-lactamase. At 30 seeds it is **48.3%, 95% CI [43.9, 52.7]**,
against alignment's fixed 29.5%, and it is **still the only arm whose interval clears
alignment**. No other arm's interval even reaches it. The claim now rests on an interval
rather than on five draws.

### What does not survive

Five arms' published @95 values fall outside their own 30-seed intervals: the canonical
ESM-2 650M run and its named duplicate, `esm2_650M_max`, `esmc_6B`, and `esm3_1_4B`. Three
sentences built on those points are wrong as written.

**One. "Twenty times the parameters recovers less than a twelfth of what 600M does, and less
than 300M does."**

| ESM-C | 5 seeds | 30 seeds, 95% CI |
|---|---|---|
| 300M | 15.7% | 16.4% [10.7, 22.2] |
| 600M | 51.4% | 48.3% [43.9, 52.7] |
| 6B | 4.3% | **12.4% [7.4, 17.4]** |

The ratio is **3.9x, about a quarter rather than a twelfth**, and 6B's interval **overlaps
300M's**, so 6B being worse than 300M is unsupported. The load-bearing part holds: 600M and
6B do not overlap across a twentyfold capacity range on one corpus.

**Two. "Max pooling drives beta-lactamase to 0% and CLS to 13%, against mean at 21%."** At 30
seeds: mean 15.7% [11.2, 20.2], CLS 16.4% [10.6, 22.2], max 1.9% [0.6, 3.2]. **Mean and CLS
are indistinguishable** and the published ordering between them was noise. Max stays far
below both.

**Three. "Beta-lactamase runs 1%, 13%, 11%, 21%, 16% across the ESM-2 ladder, no trend."** At
30 seeds: 1.7%, 16.2%, 9.5%, 15.7%, 19.0%, a rank correlation against parameter count of
**+0.70** on five points. "No trend" is too strong. What carries the conclusion instead is
that **3B's whole interval, [14.5, 23.6], lies below alignment's 29.5%**.

### Why the conclusions stand while the sentences change

Each of the three headings claims that some axis **fails to fix** beta-lactamase, and each
still does: no scale, no pooling choice, and no ESM-C capacity setting other than 600M brings
the class near the baseline it has to beat. What was wrong was the precision of the
comparisons underneath, which asserted orderings between arms that five seeds cannot
resolve. The corrected wording compares intervals.

🔴 **The recurring error, stated once so it is not repeated.** A 5-seed mean of a quantity
whose seed sd runs 12 to 19 points supports a statement about whether an arm clears a
baseline by 20 points. It does not support a statement about which of two arms is higher when
they differ by 4. Three of §9's sentences did the second thing.

### Standing

⚠️ Still not re-run at 30 seeds: the other eight classes on the thirteen non-canonical arms,
and the "non-monotonic in three of nine classes" count in §9's closing block, which is a
5-seed claim about classes this entry did not touch. Nothing in the document's conclusions
rests on those counts, and a reader quoting one should assume the same 10-to-20-point seed
noise.

### Fix

`src/03x_seed_stability_all_arms.py` and `results/v2/seed_stability_all_arms.json` are
committed, §9's three sentences are rewritten with intervals, the surviving ESM-C 600M claim
now carries its interval, and `src/22_claims_audit.py` pins all of it: the single arm that
clears alignment, the 6B-against-300M overlap, the CLS-against-mean overlap, the ladder
correlation, and the count of five arms whose published value left its own interval.

---

## 2026-09-18 (seventh entry) — §10.4 called a correlation a mechanism, and the test of that sentence needed two of its own corrections

### The published overstatement

§10.4 was written this same day and said of the two failing classes sitting closer to benign proteins than
to any hazard class: *"That is a mechanism rather than a correlation."* Nothing at that point had
manipulated the geometry. Margin ranked the failures, tracked recovery, held across fourteen
representations and ordered three unseen mechanisms correctly, all of which is association.

`src/31_margin_causal_test.py` tested it by removing the ten training negatives nearest each held-out
class and comparing against random removals of the same size, paired within seed. The effect is real and
**small**: attributable +8.8 points for beta-lactamase and +7.1 for the phage class on v3, +6.7 for
beta-lactamase on v2, every interval excluding zero, and **8 to 11% of the distance to a recovered
class**. The sentence is corrected in place to say contributing cause, with §10.7 carrying the numbers.

🔑 The useful form of the finding is that prediction and repair dissociate. Margin says which mechanism
family to distrust; curating the negative set does not fix that family.

### 🔴 Two analysis changes in the test itself, both of which improved the result

Recording this is the point of the entry. A verdict that moves twice in the author's favour after the
analysis is edited has the shape of a result being fitted, whatever the merits of each edit.

**One. A pooled null returned REFUTED.** The first run pooled 30 seeds × 25 random draws into a single
distribution and compared the 30-value targeted mean against its 95th percentile. That null carries
fold-to-fold variance which the targeted mean has already averaged out, so it is wider than the targeted
arm's own sampling distribution and the comparison is not like for like. Per-seed pairing is what §5.1 in
the same document already used, reporting "winning on 47 of 60 seeds". With the pairing fixed the
attributable effect is positive and its interval excludes zero.

**Two. The failing classes were selected by margin.** `failures = the two lowest-margin classes` selects
the test set using the predictor under test. On v2 that admitted contact-dependent inhibition, which has a
negative margin and recovers at 37.5%, so it is not a failure; its −8.5-point result was briefly read as
evidence against the mechanism. Failures are now defined as **recovery below 25%**, the property the
mechanism exists to explain, which gives beta-lactamase and the phage class on v3 and beta-lactamase alone
on v2.

Both fixes are defensible without reference to their outcomes: a pooled null mixing two variance sources
is wrong whichever way it comes out, and selecting a test set with the predictor is circular whichever way
it comes out. Both are nonetheless changes made after seeing a result, and the sequence is published with
the numbers rather than behind them.

### Standing

⚠️ K is fixed at 10. Whether the effect scales with how many benign neighbours are removed is untested,
and a dose-response curve is the obvious next check: an effect that saturates at 10 and one that grows to
50 imply different things about how much of the failure is proximity.

### Fix

§10.4's sentence is replaced, §10.7 carries the design, the numbers and this disclosure,
`results/{v2,v3}/margin_causal_test.json` are committed, and `src/22_claims_audit.py` pins both the
positive attributable effects and the fraction of the gap they close, so neither half can be quoted
without the other.

## 2026-09-20 (eighth entry) — The first test of §10.7.1's converse varied two factors at once, and the section §6 exists to prevent that

### What was being tested

§10.7.1 read the dose-response curve as evidence that the failing classes sit inside a **dense** benign
region: removing the K benign proteins nearest a class keeps helping it all the way to K=80, so there is no
small nameable set of neighbours a curator could handle. That reading makes a converse prediction. If
density is the story, then **adding** benign proteins near these classes should push them down, and a pool
an order of magnitude larger than the panel should push them down a lot.

`src/34_scale_negative_set.py` harvested that pool, 8,259 reviewed Swiss-Prot proteins, and
`src/35_negative_scaling_curve.py` swept the size of the benign set across it.

### 🔴 The design flaw

35's smallest point is **n=296 drawn at random from the pool**, not the panel's own 296 matched negatives.
Its curve therefore varies **size and composition together**, and its low anchor is not the published
baseline. What it answers is "replace the panel's negatives with n pool proteins", not "add n pool proteins
to the panel", and those are different questions with different answers.

🔴 **The caveat was already written down, one script earlier.** `34`'s own docstring says the pool is
deliberately not taxon-matched, unlike the panel's negative blocks, and states it plainly: "Results at 296
from this pool and from the panel are therefore not the same experiment and are reported separately." 35
then built its curve on the pool's 296 and the reading treated it as the panel's. So this is not a caveat
nobody had thought of. It is a caveat that was recorded and then walked into by the next script, which is
the more common way a controlled comparison goes wrong and the reason it is written up here rather than
quietly fixed.

That conflation is the specific error §6 was built to rule out. §6 varied the negative set 2×2 over sample
size and decision boundary on v2 and found **sample size explains none of the effect** while the operating
point dominates, so for this panel composition is the live factor and size is not. Running a size sweep
whose points also change composition puts the two back together.

The size of the substitution is visible in 35's own output. At **identical n=296**, the canonical 650M arm
gives the phage class **46.5%** on a random pool sample and **12.2%** on the panel's real matched negatives,
a **34-point** gap with the sample size held exactly fixed. §6's conclusion reappears here at 28 times the
scale, from an experiment that was not built to test it.

A second property of the pool belongs with the first. The harvest's per-organism cap bit hard on
eukaryotic Swiss-Prot, so the pool is **87% bacterial** and resembles the panel's producer mix rather than
Swiss-Prot's. It is a larger benign set drawn from roughly the same world, which is the right material for
this question and not a neutral background sample. `pool_composition_note` in both artifacts records it.

### The fix, and what it changes

`src/37_negative_supplement_from_pool.py` keeps the panel's real 296 as a **fixed floor at every K** and
adds K pool proteins on top, so only composition grows and the K=0 point is the panel itself. It also
splits the addition two ways: K proteins drawn at random, and the K **nearest** pool proteins to the held-out
class, which is the actual converse of §10.7's removal experiment.

On the canonical 650M arm the comparison class is again what makes the result readable, and it does not
read the way proximity predicts. Adding the 500 **nearest** pool proteins costs the recovered class
**43.3 points**, 94.8% to 51.4%, while beta-lactamase falls only **5.7** and the phage class **rises 5.3**.
The proteins nearest a class hurt the class that was working far more than the two that were not. By K=8259
the two addition modes converge, because both have added the whole pool.

### ⚠️ A third design change made after seeing a result

37 was written after 35 returned REFUTED. The flaw is defensible without reference to that outcome, since
varying two factors at once when an earlier section has shown one of them dominates is wrong whichever way
it comes out, but it is still a design change made after seeing a result and it is the **third** in this
line of work. The first two are in the seventh entry. All three are published with the numbers rather than
behind them.

35's artifacts are kept for both arms rather than deleted, because its own question is a real one and its
answer is the composition effect §6 predicts, in the direction §6 predicts. On the 35M arm, where
beta-lactamase starts at the floor, a pool sample of 296 gives 9.0% and growing it to 8,259 gives 18.3%.

### Standing

⚠️ **The baseline in 37, 38 and 39 is a 30-seed recomputation, not the published figure.** At K=0 those
scripts give beta-lactamase **21.2%** and the phage class **12.2%**, against the published 5-seed **18.6%**
and **10.0%**. 37's own tolerance check flagged the gap rather than hiding it: `k0_matches_stored_lomo` is
False for both failing classes and True for the comparison class, at a 2-point tolerance. The cause is seed
noise on a 14-member and a 16-member class, the same effect as the fifth entry, and the published 5-seed
numbers stay the ones the audit pins.

### Fix

§10.9 carries the pool, both curves, the decomposition and this disclosure. `src/35` keeps its own
verdict string and its artifacts are committed for both arms, with `note_on_35` in every 37 artifact
pointing at the flaw from the fixed side.

## 2026-09-20 (ninth entry) — §10.6.1 compared five model arms by their 5-seed points, which is the same mistake as the sixth entry, made the same day it was written

### What was published

§10.6.1 was rewritten earlier on 2026-09-20 from three model arms to five, and the rewrite correctly
withdrew a capacity-monotone reading that the three-arm version had invented. It then made a new claim of
its own: **"3B and 150M are far worse on beta-lactamase than the canonical arm, 4.3% against 18.6%"**, and
for the phage class that it "sits between 6.9% and 10.0% across all three larger arms before jumping to
26.9% and 31.2% in the two small ones".

Every one of those figures comes from `lomo_results*.json`, which runs the published protocol at **five**
negative-holdout seeds.

### 🔴 Why that was already known to be unsafe

The fifth entry above established that a 5-seed mean of beta-lactamase recovery can sit outside its own
30-seed interval. The sixth entry withdrew **three separate §9 sentences** for comparing model arms by
their 5-seed points. §10.6.1's rewrite then did it again, on the same day, in the section whose entire
subject is comparing model arms.

On v2 that comparison does not survive at all. From `results/v2/seed_stability_all_arms.json`,
beta-lactamase across the ESM-2 ladder reads 1.4 / 12.9 / 11.4 / **21.4** / 15.7 at five seeds and
1.7 / 16.2 / 9.5 / 15.7 / **19.0** at thirty. **The peak moves from 650M to 3B**, and every interval among
the four larger arms overlaps every other, so on v2 "which arm is better" is a seed artefact.

### What the 30-seed check actually found on v3

`src/41_v3_arm_seed_stability.py` runs the v3 equivalent, both failing classes, five arms, 30 seeds, at
both operating points. The answer is the opposite of v2's:

- **The arms do separate.** 9 of 10 pairs are disjoint on beta-lactamase, 8 of 10 on the phage class, and
  the canonical arm separates from all four others on both classes. The direction of §10.6.1's claim
  therefore stands.
- **Three of five 5-seed figures for beta-lactamase sit outside their own 30-seed intervals** (3B, 150M,
  35M), and two of five do for the phage class (3B, 8M).
- 🔑 **Two coincidental ties dissolved.** 3B and 150M read 4.3% each on beta-lactamase and 6.9% each on
  the phage class, which is 3 of 70 arriving twice. At 30 seeds they are **9.3% [5.6, 13.0]** and
  **1.9% [0.8, 3.1]** on beta-lactamase, a factor of five apart with disjoint intervals, and 4.1% and 7.2%
  on the phage class, also disjoint. The sentence had treated them as the same result.
- **The phage class's best arm changes with the seed count**, 8M at five and 35M at thirty, and those two
  overlap, so the ranking between them was noise in both directions.

### Why v2 and v3 disagree, which is the useful part

v2 holds out 40% of 154 negatives, so its threshold is a quantile of **61** points. v3 holds out 40% of 296
and gets **118**. Doubling the calibration sample halves that noise source, and differences between arms
that v2 cannot resolve become resolvable on v3. §10.8's calibration constraint therefore also limits what
the study itself can measure, not only what a screen could deploy.

### Standing

⚠️ The published 5-seed figures stay the ones the audit pins, following the fifth entry's convention: the
protocol is 5 seeds and reproduces exactly, and the 30-seed recomputation is reported beside it rather than
over it. §10.6.1's table now carries both columns.

⚠️ The margin results are untouched. Margin orders classes **within** an arm and every rank correlation is
computed over twelve classes per arm, so none of them rests on an arm-to-arm recovery difference. What
needed the seed check was only the per-arm recovery figures quoted side by side.

### Fix

§10.6.1 carries the 30-seed column, the separation result and the dissolved ties; `src/41` is committed
with its preregistration; `results/v3/arm_seed_stability.json` is committed; and
`src/22_claims_audit.py` pins the pair counts, the three-and-two outside-interval lists, the dissolved tie
and the inverse-capacity effect on the phage class, so the direction cannot be quoted without the seed
check that supports it.

## 2026-09-20 (tenth entry) — §10.4 named the wrong number of classes under a chance figure only the right number produces, and truncated a p-value column

### The class count

§10.4 read: "Margin ranks the two failures as the two lowest of **eleven** classes, which has probability
**1/66** = 0.015 under a random ordering." Eleven classes give 1/55. The two figures could not both be
right and the chance figure was the correct one.

The margin ordering covers **twelve** classes: v3's eleven holdout-eligible mechanism classes **plus the
labelled virulence control**, which is not a mechanism, cannot be held out, and is ranked with the rest.
`results/v3/second_failure_class.json` carries `margin_order_low_to_high` as a list of twelve and
`chance` as 0.015151..., which is 1/C(12,2).

🔑 **Including the control is load-bearing rather than incidental**, which is why the correction is worth
more than a digit. The control sits **fourth-lowest** in the margin ordering, and it is the class that
takes second place from the phage class in **three of the five arms** in §10.6.1. A version of the test
run over the eleven eligible classes only would have removed the one class that explains §10.6.1's
misses.

### The p-value column

The same table gave margin's permutation p as **0.0001** where the artifact says **0.00015**, and the
nearest-negative's as 0.0034 where it says 0.00345. The column had been **truncated rather than rounded**,
which understates the margin p-value by a third. The other two rows, 0.0002 and 0.97, are unaffected by
the same truncation.

⚠️ Two different p-values for the same rho of +0.894 appear in this document and both are correct.
§10.4's table reports the decomposition's figure, 0.00015, and §10.6.1's table reports
`30_margin_across_arms.py`'s figure for the canonical arm, 0.0002. They are separate permutation runs of
separate scripts over the same ordering, and the difference is four draws out of 20,000.

### Fix

§10.4 carries twelve, 0.00015, 0.0035 and a note on why the control is in the ordering.
`huggingface/README.md` carried the same two errors and is corrected with it. The audit's pin on that
table row is updated, so the row cannot drift again without failing CI.

## 2026-09-20 (eleventh entry) — The benign pool was filtered by name and never by sequence, and it holds one 0.871 ortholog of a panel positive

### What the pool was screened for, and what it was not

`34` harvests 8,259 reviewed Swiss-Prot proteins as a large benign reference set. It excludes six hazard
keywords, re-checks them in Python, applies `02d`'s protein-name blocklist and a positive-class-term
blocklist, and dedups against both panels on **accession and sequence hash**.

Nothing in that screens a pool candidate against a panel positive **by sequence**.

⚠️ **That is not by itself a defect, and an earlier version of this entry said it was.** §2's ≤ 0.30
normalized Smith-Waterman rule governs **positives against positives**. `27` states plainly that negatives
are *not* screened against the positives, because mechanism-matched benign proteins are **wanted** as hard
negatives, and `data/sequences/benign_homologs.fasta` is the block that exists for exactly that. The pool
follows the panel's own policy for negatives rather than a weaker one.

🔑 **What makes a pool number an outlier is what the panel's own negatives reach under that policy.** `42`
measures it: the panel's 296 negatives against its 149 positives, 44,104 alignments, **zero** above 0.30,
highest **0.282**, second 0.170. A negative set assembled with no homology screen lands entirely below the
positives' own admission threshold. The pool's worst case is **3.1 times** that. So the finding below is an
outlier inside a documented policy, not a broken rule, and it is fixed at the point of use.

### 🔴 The name filter leaks, and asymmetrically between the two classes that matter

`CLASS_BLOCK` covers the phage class's canonical names, "endolysin", "lysozyme", "muramidase", "amidase"
and "holin". It does not cover "peptidoglycan hydrolase", "autolysin" or "peptidoglycan", so the pool holds
**eight genuine peptidoglycan hydrolases as negatives**: seven *Staphylococcus* "Bifunctional autolysin"
entries and "Peptidoglycan hydrolase PcsB". The autolysins are bifunctional amidase/glucosaminidases, so
the exact domain the block list names arrived under a protein name that does not contain the word.

Beta-lactamase is covered, with "lactamase", "beta-lactam", "penicillinase", "cephalosporinase" and
"carbapenemase" all blocked and zero matching entries. So the two classes §10.9.1 finds responding in
opposite directions had been filtered at different effective stringency, which is why this was checked at
all.

### What the census found, which was not what it was looking for

`src/42` runs every pool protein against every positive, **1,230,591** local alignments, no sampling.

- **Every mechanism class is clean.** The highest similarity any of them reaches is **0.174**
  (T3SS effectors), then 0.115 for beta-lactamase and 0.105 for the phage class. The eight cell-wall
  entries are functional analogues at about a third of the admission threshold, not sequence homologs, so
  §10.9.1's class split is not label contamination and its numbers stand.
- 🔴 **One violation, in the labelled virulence control.** `Q8X739` PHOQ_ECO57, *E. coli* O157:H7 sensor
  protein PhoQ, sits in the pool as a **negative** at **0.871** against `D0ZV89` PHOQ_SALT1, the
  *Salmonella* PhoQ that is a **positive** on the panel. Two orthologs of the same two-component sensor,
  0.87 identical, one labelled hazardous and one benign.

No keyword or name filter could have caught it. Both proteins are called "Sensor protein PhoQ" and neither
carries a hazard keyword, which is the same reason beta-lactamase needs its own §2 footnote. The defect is
reachable only by sequence.

### Why it is worth an entry rather than a silent fix

The census was built to test a hypothesis about the failing classes, both of which turned out clean, and it
found a defect in a class nobody was asking about. A targeted check would have returned a clean answer and
left it there.

⚠️ **And it was one script away from manufacturing a result.** `src/43` measures every class's response to
the pool, including the control. That class's pool proximity is extreme **because of this one protein**, and
letting a 0.871 homolog of one of its members enter training as a negative drives that member to the benign
side at high dose. The output would be a strongly negative response in the highest-proximity class, which
is precisely the correlation a crowding account predicts. `43` now drops the protein by default, keeps a
`--keep-pool-homologs` run for comparison, and the gap between the two is reported as the size of the
effect.

### Standing

⚠️ `CLASS_BLOCK` is deliberately **not** patched. Adding the missing terms changes which proteins the pool
holds, which invalidates every §10.9 and §10.9.1 number unless the pool is rebuilt and five scripts re-run,
and the census shows that would change the inputs without changing the conclusion. The terms a future
harvest should add are named at the site in `34`, along with the recommendation to use `42`'s census as the
admission gate instead of trusting names.

⚠️ The one violation is **not** removed from the committed pool either, for the same reason. It is removed
at use, by `43`, and any future script that trains on the pool should do the same.

### Fix

§10.9 carries the census table, the violation and what it reaches; `src/42` is committed with its
preregistration; `results/v3/pool_homology_against_panel.json` is committed; and the audit pins the
violation's accession pair, its class, its similarity and the fact that every mechanism class is clean, so
neither half can be quoted without the other.

## 2026-09-20 (twelfth entry) — A panel positive labelled "phospholipase" is Exoenzyme S, and its own FASTA header said so all along

### The flag that sat unresolved

`data/annotations/mechanism_classes_{v2,v3}.json` carried, for `tr|Q51451|Q51451_PSEAI` in the
`phospholipase` class, the reason **"P. aeruginosa TrEMBL entry, grouped with ExoU on organism and panel
context. Identity NOT independently confirmed"**, and the class note read "n=2, and one identity
unconfirmed". Both have been in the committed panel since it was built. Nothing resolved them.

Found while looking for an **unblocked** way to raise the class count above twelve, since n=12 is the
resolution every claim in §10.4 to §10.6 and §10.9.2 runs at. Two candidate classes are waiting on
labelling decisions, so the next place to look was the classes already annotated but too small to hold out,
which is where the flag was.

### Resolved, and the grouping is wrong

Checked against UniProt on 2026-09-20:

| accession | submission names | keywords |
|---|---|---|
| **Q51451** | Exoenzyme S, gene `exoS` | GTPase activation, NAD, Nucleotidyltransferase, Transferase, Glycosyltransferase, Secreted, Toxin, Virulence |
| **O34208** | ExoU, PepA, Type III effector protein | **Hydrolase, Lipid degradation, Lipid metabolism** |

Q51451 carries **no Hydrolase and no Lipid degradation keyword**. It is *Pseudomonas aeruginosa* Exoenzyme
S, an ADP-ribosyltransferase with an N-terminal Rho GAP domain, delivered by the type III secretion system.
O34208 is ExoU, a patatin-like phospholipase, and is correctly placed.

So the `phospholipase` class holds **one phospholipase and one ADP-ribosyltransferase**. What the two
members actually share is a producer organism and a delivery system, not a mechanism.

🔴 **The information needed to catch this was in the panel's own FASTA file from the start.** The header
reads `>tr|Q51451|Q51451_PSEAI Exoenzyme S OS=Pseudomonas aeruginosa OX=287 GN=exoS`. The entry was grouped
by organism while its own description named the protein.

### What it does and does not change

🟢 **No published number moves.** `phospholipase` is **not holdout-eligible** in v2 or v3 and appears in no
LOMO result in either, so no per-class recovery figure depends on the grouping. Q51451 contributes only as a
training positive in every other class's holdout and as one of the 80 or 149 positives behind the baseline
AUROC, which is unaffected by which label it carries.

⚠️ **The class assignment is therefore left UNCHANGED, deliberately.** Moving Q51451 into
`t3ss_effector_apparatus`, which is holdout-eligible at n=10, **would** change that class's recovery figure
and every number downstream of it. That is a labelling decision rather than a correction, and it is JK's to
make. The same reasoning kept the pool's PhoQ ortholog in place in the eleventh entry: document at the site,
fix at the point of use, do not mutate a frozen artifact to tidy a label.

⚠️ Consequences of the decision, so it can be made on the numbers: moving it makes `phospholipase` n=1 and
permanently ineligible, takes `t3ss_effector_apparatus` to n=11, and leaves the class count at twelve either
way. It does not help the n=12 limitation, which is what the search was for.

### Fix

Both annotations now record the confirmed identity, the evidence, and why the assignment stands. The edit is
a **text replacement**, four lines across the two files, verified to change exactly two parsed values and
nothing else: a parse-and-redump round trip had re-encoded an unrelated `§` escape in v3, which is more
than a frozen artifact should absorb for a documentation change.

## 2026-09-20 (thirteenth entry) — The last unassigned mechanism is assignable, and the family it belongs to cannot be a class

### The second flag the same sweep found

Sweeping both annotations for hedged reasons turned up exactly two across 80 and 149 proteins. The twelfth
entry is the first. The second is `sp|Q7NWF2|COPC_CHRVO`, in the remainder class `other_toxin_mechanism`
with the reason **"C. violaceum. Mechanism NOT confidently assigned"**.

### It is assignable now

Swiss-Prot gives Q7NWF2 the recommended name **Arginine ADP-riboxanase CopC**, EC **4.3.99.-**, family
**OspC**, catalysing

```
L-arginyl-[protein] + NAD(+) = ADP-riboxanated L-argininyl-[protein] + nicotinamide + NH4(+) + H(+)
```

which blocks host caspase processing. ⚠️ ADP-riboxanation is **not** ADP-ribosylation: it releases ammonia
and forms a different adduct, and Swiss-Prot names the two activities separately. So the panel's
`adp_ribosyl_ab_toxin` class is the wrong home for it on chemistry, quite apart from that class being an
AB-toxin architecture and this being a T3SS effector.

🔴 **The panel already holds the only other independent member of that mechanism, in a different class.**
`A0A0H2US87` OspC3 carries the same recommended-name pattern, the same EC and the same catalytic reaction,
and the panel labels it `t3ss_effector_apparatus` with the reason "Shigella OspC3, T3SS effector, caspase-4
inhibition". One mechanism, two classes: one member labelled by delivery, the other recorded as unassigned.

### Why this cannot become a class, which closes the route it was found on

The sweep was looking for an **unblocked** way to raise the class count above twelve, since n=12 is the
resolution every claim in §10.4 to §10.6 and §10.9.2 runs at. A mechanism class defined by one EC and one
catalytic reaction would have been the tightest definition in the panel. It fails the panel's own admission
rule.

Reviewed UniProt holds **seven** arginine ADP-riboxanases, all OspC family, all inside the panel's 100 to
1400 length window. Under normalized Smith-Waterman at ≤ 0.30 they collapse to **two** independent
sequences:

| cluster | members | mutual similarity |
|---|---|---|
| *Shigella* / *E. coli* | OspC1 Q8VSJ7, OspC2 Q8VSL8, **OspC3 A0A0H2US87**, OspC4 A0A0H2USP8, OspC3 P0DV36 | 0.638 to 0.970 |
| *Chromobacterium* | **CopC Q7NWF2**, OspC3 A0A2H5DV25 | 0.858 |

Between the clusters, 0.269 to 0.301. Effective n is **2** against an eligibility floor of **4**, so the
class is not buildable, and seven database entries are two independent sequences.

🟢 **And the two the panel already has are exactly those two representatives, at 0.279.** Whoever selected
them took one from each cluster and left nothing on the table, while labelling them into different classes.

### Standing

⚠️ Class assignments are left **UNCHANGED** for the same reason as the twelfth entry. Moving CopC into
`t3ss_effector_apparatus` would take an eligible class from n=10 to 11 and change its recovery figure; a
`arginine_adp_riboxanase` class of n=2 is ineligible and would take OspC3 **out** of an eligible class,
changing the same figure the other way. Both are labelling decisions rather than corrections.

⚠️ This route to raising n is now closed and the negative is worth recording, because the family looks like
a strong candidate from its entry count and is not one. The two candidate classes that could raise n,
`plant_target_avirulence` at 32 admissible and `chitinase_antifungal` at 29, are waiting on a crop-target
release policy and a hazard-label validity call respectively.

### Fix

Both annotations record the assigned mechanism, the EC, the reaction, the family's effective n and why no
class follows. No published number changes: `other_toxin_mechanism` is not holdout-eligible and appears in
no LOMO result.

## 2026-09-20 (fourteenth entry) — §10.9.2's one positive result was reported before it was replicated, and it does not replicate

### What was published, hours earlier on the same day

§10.9.2 ran the boundary arm on all twelve classes and tested five predictors of the per-class response. One
survived: **pool proximity minus the proximity the class already had to the panel's own negatives**, at
Spearman **−0.601** (permutation p 0.0398) and **−0.769** (p 0.0054) with the K=0 baseline partialled out.
The section stated the multiplicity problem plainly, that ten uncorrected tests put the Bonferroni threshold
at 0.005 and the surviving figure at 0.0054, and called the result "suggestive and not established".

🔴 **It was still published as one arm's correlation with no replication attempted**, in a section of a
document whose §10.6 exists precisely because a class-level statistic has to be recomputed inside every
model arm before it is believed. The discipline was available and was not applied before writing.

### The replication, and it fails

`src/43` re-run on `esm2_35M`, same twelve classes, same 30 seeds, same 20,000-permutation nulls:

| predictor | canonical 650M, rho (p) | partialled (p) | esm2_35M, rho (p) | partialled (p) |
|---|---|---|---|---|
| pool proximity − negative proximity | **−0.601** (0.040) | **−0.769** (0.0054) | −0.343 (0.276) | −0.385 (0.220) |
| margin, the preregistered null | −0.182 (0.571) | −0.217 (0.503) | −0.252 (0.425) | +0.350 (0.263) |

The **sign agrees** and nothing else does. The magnitude roughly halves and the significance is gone, so the
count is **1 arm of 2** against margin's 5 of 5 in §10.6.1. With the multiplicity miss on top, the honest
statement is that the predictor is **not supported**.

🟢 **The preregistered half does replicate.** Margin has no relationship with the response in either arm, at
p 0.571 and 0.425. So "margin says which mechanism classes a screen will miss, and nothing about which of
them a larger benign set makes worse" holds in both representations. The section's surviving content is that
separation and §10.9.1's class split, not the predictor.

### Standing

⚠️ Only two arms can be tested today. Pool embeddings exist for `esm2_650M` and `esm2_35M` only; the other
three v3 arms would need all 8,259 pool proteins embedded, which is GPU work rather than a rerun. Two arms
is weak either way, and a 1-of-2 is not evidence against the predictor so much as an absence of evidence
for it.

⚠️ The sign agreeing in both arms is the one thing worth keeping. It is not significance and it is not
reported as any.

### Fix

§10.9.2's heading now says the predictor does not replicate, its table carries both arms, and the
audit pins the second arm's rho, its non-significance, the sign agreement, the halved magnitude and margin's
null in both arms, so the −0.601 cannot be quoted without the −0.343 beside it.

## 2026-09-20 (fifteenth entry) — The negatives never had a test set, and the threshold estimator cannot deliver the specificity it reports

### The defect, found by reading the split rather than the results

`03b` splits the negatives once, 40% and 60%, trains on the 60% and takes the threshold as the 0.95
quantile of the 40%. It then reports the realised false-positive rate **on that same 40%**, which its own
docstring describes as "5% by construction". The variable is called `nte`, as in negative test; it is a
calibration set.

🔴 **A sweep of all 45 scripts in `src/` finds no three-way negative split anywhere.** Not one holds
negatives out of both training and threshold-fitting. So the positives have a test set, the held-out
mechanism class, and the negatives never have. Every "at 95% specificity" in this repository is a
within-calibration-set specificity.

### What the measurement found

`src/45` keeps training at the published 178 and divides the published 118 into m calibration and 118 − m
test, so the published protocol is the m = 118 endpoint with an empty test set. 300 seeds, two arms.

- At **m = 20** the out-of-sample rate is **8.64%** on the canonical arm and **8.79%** on 35M against a
  stated 5%, **1.7 to 1.8 times nominal**, with single splits reaching **36.7%**.
- 🔑 **`np.quantile(s, 0.95)` cannot return 5% at these sample sizes.** With m calibration points the
  achievable exceedance rates are `{j/(m+1)}`; at m=30 the neighbours are 3.2% and 6.5% and 5% is not among
  them. At **4 of 6** sizes the order-statistic bracket's lower edge already exceeds 5%. The observed mean
  falls inside the bracket at **6 of 6 sizes on both arms**, so the estimator is behaving predictably.
- 🟢 The conformal threshold, the k-th largest with k = ⌊(m+1)α⌋, **held its guarantee at every size on
  both arms**, running conservative at 2.9% to 4.9%.

### Three corrections to my own preregistration, all made before the numbers were believed

1. The first prediction said the estimator is near-unbiased and the mean out-of-sample rate is 5%. That
   treated `np.quantile` as if it returned a population quantile, which it does not on 20 points.
2. The replacement asserted the point prediction `k/(m+1)` and missed by 3.8 points at m=20, because the
   interpolation fraction was 0.05 there and the threshold sat almost on the (k+1)-th order statistic. The
   defensible quantity is the **bracket**, which holds 6 of 6.
3. The spread prediction used calibration noise only. The sweep trades calibration size against test size,
   so the total is the root of two variances and is U-shaped in m rather than monotone.

⚠️ A fourth was caught by the noise itself. At 30 seeds the canonical arm appeared to **violate** the
conformal guarantee at m=78, 4.08% against ≤3.80%. The standard error of that mean was 0.57 points, so the
excess was inside it. Panel A now runs at **300** seeds and every bracket and guarantee check is made
against two standard errors rather than at face value. A guarantee checked with too few seeds to see it is
not checked.

### What this does and does not change

🟢 **No published number is overturned.** The published protocol uses the largest calibration set
available, all 118, whose bracket is [5.04, 5.88]. Its true out-of-sample rate is within about 0.9 points
of what it claims. What was wrong is that the rate was never measured.

🔴 **The classes most sensitive to it are the ones §10 is about.** Between m=20 and m=118 the virulence
control moves 13.0 points, the phage class 8.9 and beta-lactamase 5.7, while the two classes at the ceiling
move exactly 0.0. Recovery and the realised false-positive rate rise together when the threshold loosens,
so "recovery at 95% specificity" conflates them whenever the calibration set is small. That is §10.9.1's
operating-point confusion arriving from calibration size alone.

🔑 It also explains §10.8 rather than restating it. §10.8's 250,003 requirement was derived from wanting
ten negatives above the threshold; k=10 granularity at α=10⁻⁴ needs m ≥ 99,999, which is **249,998** at a
40% split. The calibration set size fixes the **granularity** of the achievable operating points, not
merely the precision of one.

### Fix

§2.6 carries the split, the table, the bracket, the conformal arm and the §10.8 closure. `src/45` is
committed with its preregistration and its three corrections. The audit pins the m=20 rate on both arms,
the 6-of-6 bracket, the 6-of-6 conformal guarantee, the four sizes where 5% is unreachable and the
concentration of sensitivity in the low-recovery classes, so neither half can be quoted without the other.

⚠️ `03b` is **not** changed. Rewriting the protocol would invalidate every published number for a defect
that the m=118 bracket shows costs under a point. The three-way split belongs in the next panel, not
retrofitted to this one.

---

## 2026-09-20 (sixteenth entry) — Three FSPE annotations were in mature-chain coordinates while the pipeline indexed the precursor, and the audit that named this risk installed its guard on the other pathway

### The defect

`functional_sites.json` stores `catalytic_residues` for each panel protein. **Twelve scripts read that
one field, and they do not agree on what the numbers mean.** They split into two groups:

**Sequence-position consumers** — index the FASTA in `data/sequences/toxins_positive*.fasta` with
`r - 1`, so a mature-chain number silently lands on an unrelated residue:

| script | metric | state |
|---|---|---|
| `04_esm2_masked_prediction.py` | FSPE (ESM-2) | 🟢 fixed and re-run |
| `14_esm3_separability_fspe.py` | FSPE (ESM-3, SaProt) | 🟢 code fixed, ⚠️ results **not** re-runnable here |
| `15_sae_fhs.py` | FHS (SAE) | 🟢 code fixed, ⚠️ results **not** re-run |
| `13_evodiff_fsi.py` | EvoDiff site recovery | 🟢 code fixed, ⚠️ results **not** re-run |
| `utils.py::get_functional_residues()` | public helper | 🔴 handed out raw numbers with a docstring promising "1-indexed residue positions" and no coordinate caveat. Zero callers, so it never fired: a loaded gun, not a wound. Now documented and superseded. |

**PDB-numbering consumers** — resolve against `pdb_residues` / the structure, and are unaffected:
`06_proteinmpnn_redesign.py` (FSI, and the only one that ever had the identity guard),
`10_fsi_temperature_sensitivity.py`, `11_esmfold_validation.py`, `12_ligandmpnn_fsi.py`,
`17_stepping_stone.py`.

⚠️ Three of those five prefer `pdb_residues` and **silently fall back** to `catalytic_residues`
(`info.get("pdb_residues", fs["catalytic_residues"])`). Three entries carry `use_pdb_numbering: true`
with **no** `pdb_residues` to fall back to (Abrin, Tetanus LC, Streptolysin O). All three are verified
clean at offset 0, so nothing is currently wrong through that door, but the door is open.

Those FASTA records are full **precursors**. For a secreted toxin the structural literature numbers
residues on the **mature chain**, and three entries were curated straight from that literature:

| Accession | Protein | Annotated | UniProt boundary | Indexed as | Should be |
|---|---|---|---|---|---|
| `P02879` | Ricin A-chain | 80, 123, 177, 180, 211 | Signal 1-35, Chain 36-302 | those positions in the 576-aa precursor | **+35** → 115, 158, 212, 215, 246 |
| `P00648` | Barnase | 27, 73, 83, 87, 102 | Signal 1-34 + Propeptide 35-47, Chain 48-157 | ditto, 157-aa precursor | **+47** → 74, 120, 130, 134, 149 |
| `P00588` | Diphtheria toxin | 21, 65, 148 | Signal 1-32, Chain 33-225 (fragment A) | ditto, 567-aa precursor | **+32** → 53, 97, 180 |

So for four months every sequence-position consumer masked five residues of Ricin, five of Barnase and
three of diphtheria toxin at positions that were **not** the annotated catalytic residues, and reported
the resulting entropy ratio as the model's confidence at functional sites.

🔑 The count is the point. The 2026-05-21 audit read this field as having two consumers and guarded one.
It actually had twelve consumers in two coordinate systems, five of them indexing sequences. A field
whose meaning varies per entry needs one resolver, not a convention that each caller is trusted to
remember.

### Why it survived the audit that was looking for it

`docs/FSI_NUMBERING_AUDIT.md` (2026-05-21) is not silent on this. It states the mechanism outright:
"FSI is unaffected (it uses `pdb_residues`), but **FSPE and FHS use `catalytic_residues` as sequence
positions**." It then acted on that sentence for exactly one protein, Anthrax PA, and its scope line
reads "the 8 structures with computed FSI values" against an FSPE panel of 15. Three things followed:

1. 🔴 **Ricin was declared "clean, 5/5."** True in 2AAI, whose A-chain is numbered on the mature
   chain, so the FSI lookup succeeds on the same numbers that fail in the FASTA. One word, correct in
   its declared scope, read as global.
2. 🔴 **Barnase and diphtheria toxin were never examined at all.** Neither has an FSI entry, so both
   sat outside the audit's scope by construction.
3. 🔴 **Recommendation 4's loud-failure identity check went into `06` only.** The FSPE pathway, named
   in the audit's own text as the one that reads these numbers as sequence positions, got no check.

🔑 The sharpest illustration is that the audit's repairs **split two homologs into different
conventions**. Abrin and Ricin are both type-2 RIPs with the same catalytic tetrad, and the abrin
annotation was explicitly derived by parallel to ricin. Abrin was flagged, so on 2026-05-22 it was
re-curated to `[74, 113, 164, 167, 198]` in UniProt/precursor numbering. Ricin was called clean, so it
kept mature numbering. The two ended up on opposite coordinate systems by way of a correction.

### What the offsets rest on, since the correction favours the project's own claim

Every corrected statistic moved in the direction the report argues for, so the offsets need to be
fixed by something other than the outcome. Two independent criteria, neither of which is FSPE:

- **Residue identity, and it is unique.** Scanning every offset from 0 to the end of each sequence,
  the number that aligns *all* annotated identities is **+35 alone for Ricin (5/5), +47 alone for
  Barnase (5/5), +32 alone for diphtheria toxin (3/3)**. No alternative offset produces a full match.
- **It equals the UniProt feature boundary.** Each offset is exactly the signal (or signal +
  propeptide) length that precedes the mature chain, fetched from UniProt, not inferred.

An offset chosen to improve a ratio would be fitting. An offset over-determined by residue identity
and independently equal to a database boundary is not available for fitting. `src/46` records both.

### What changed

| Quantity | Before | After |
|---|---|---|
| Ricin `P02879` ratio | 1.226 | **1.230** (r −0.58 → −0.72) |
| Barnase `P00648` ratio | 1.283 | **0.051** (p 0.616 → **0.0004**) |
| Diphtheria `P00588` ratio | 0.955 | **0.562** (p 0.177 → 0.058) |
| Expanded panel mean (n=15) | 0.546 | **0.437** |
| Ratios below 1.0 | 12/15 | **13/15** |
| Protein-level sign test | p = 0.018 | **p = 0.0037** |
| Sign-flip permutation | p = 0.0010 | **p = 0.0002** |
| Residue-pooled Mann-Whitney | p = 2.6 × 10⁻⁸, r = 0.41 | **p = 4.5 × 10⁻¹⁰, r = 0.46** |
| Per-protein significant at p < 0.05 | 8/15 | **9/15** (Barnase gained; none lost) |

🟢 **The displayed eight-protein headline panel is materially unchanged**, mean 0.6386 → 0.6391, still
6/8 below 1.0. Ricin is its only affected member and Ricin barely moved. The strengthening is entirely
in the expanded panel and the protein-level test, because Barnase, the large correction, is not in the
displayed table. Anyone quoting the headline number was quoting a number the defect did not touch.

⚠️ The other twelve proteins agree between the pre- and post-correction runs to within **2.1 × 10⁻⁶**,
which is the forward-pass floating-point floor, not a change. Ricin's +3.9 × 10⁻³ is three orders
above that floor, so it is a real move on genuinely different residues that happens to land in the
same place: ESM-2 has no positional signal on the ricin A-chain active site under either numbering.

🔑 **The RIP exception survives the correction and is now mechanistic.** The two ratios still above
1.0 are Ricin (1.230) and Abrin (1.073), both type-2 ribosome-inactivating proteins. Abrin has **no
signal peptide** in UniProt, its chain starts at residue 1, so its numbering is verified correct at
offset 0 and cannot be explained away by this defect. Two independent RIPs, one with confirmed-clean
annotation, both showing no functional-site confidence, is a property of the mechanism class rather
than a curation artifact. This is consistent with the RIP class behaving separately elsewhere in the
panel and should be read as a finding, not as residual error.

### Two entries are flagged and deliberately not corrected

- 🔴 **SEB `P01552`.** No offset works. All nine annotated identities are parsed and all nine
  are scored, and **0 of 9 match at offset 0**; sweeping offsets 0-60 the best any offset
  reaches is 3 of 9, tied between two different offsets (+6 and +60), so there is no unique
  candidate of the kind that fixed the three entries above. Separately, SEB is a superantigen with no catalytic site, and these are MHC-II and
  TCR-Vβ **interface** residues, so `catalytic_residues` is the wrong field for this entry whatever the
  numbering. The 2026-05-21 audit's recommendation 1 asked for SEB re-curation; the resolution instead
  added `exclude_from_fsi`. That removed it from one consumer of the field and left it in the other,
  which is the same asymmetry as the rest of this entry. It is still **run at offset 0**, so its
  published 0.956 is unchanged and the corrected run isolates the three repaired entries; its
  contribution to the pooled statistic is unverified rather than corrected.
- 🔴 **ExoS `Q51451`.** 4 of 5 identities match at offset 0 (Arg146, Leu148, Glu379, Glu381), and
  offset 0 is also the **best** offset over the whole 0-60 sweep, so this entry is *not*
  mature-numbered and needs no offset. Position 234 is annotated Trp but the precursor
  carries Asp, and 1HE1 resolves only the GAP domain so it cannot adjudicate. Left in place and
  flagged rather than guessed. This entry is already recorded as mislabelled by the twelfth entry.
- ⚠️ **VacA `P55981`, documentation only.** Its `catalytic_residues` is deliberately empty (a
  pore-forming channel with no classical active site), so `src/04` skips it and no number depends on
  it. Two keys inside `residue_annotations` are nonetheless mature-numbered; both are explicitly
  labelled "not catalytic" and are never read. Left as-is. An audit pass that reads
  `residue_annotations` rather than `catalytic_residues` will flag this entry as broken; it is not.

### Fix

`functional_sites.json` carries an explicit `precursor_offset` per entry, with the UniProt boundary and
the verified match count in a sibling note, plus `_numbering_flag` on the three entries above. The
annotated numbers are **left in the cited reference's coordinates** rather than rewritten, so every
position can still be checked against the paper it came from. The offset is applied at index time by one
shared resolver in `utils.py`, and `src/04` records `precursor_offset`, `residues_annotated`,
`residues_indexed` and `numbering_flagged` on each result.

**One resolver, not a convention.** The missing half of recommendation 4 is installed in `utils.py`, not
in one script. `sequence_functional_positions()` applies `precursor_offset`, calls
`check_residue_identities()` and reports the `_numbering_flag` state; all four sequence-position
consumers (`src/04`, `src/13`, `src/14`, `src/15`) now go through it, so they cannot drift apart again,
and `get_functional_residues()` carries a docstring that names the coordinate hazard and points here. The
guard parses the expected amino acid out of `residue_annotations` and prints `WARNING: RESIDUE MISMATCH`
for any scored position whose identity does not land where the pipeline indexes it. On the corrected
panel it fires on exactly `P01552` (9/9) and `Q51451` (1/5) and is silent on the other thirteen, and it
is **print-only**: max |delta| between the pre- and post-guard runs is 0.00e+00. The helper reproduces
`src/04`'s recorded provenance for 15/15 proteins, and the provenance fields stay in `main()` rather than
inside `evaluate_protein_fspe()` because `src/46` imports that function for the bit-identical recompute.

`src/46` carries the offset derivation, the uniqueness scan (`offset_uniqueness()`, the anti-p-hacking
check described above) and the recompute. 🔴 It also had a bug of its own, introduced by this fix: it
read `published_ratio` from `fspe_results.json`, which this correction rewrote, so it would have compared
the fix against itself and reported no change. It now reads
`fspe_results_PRE_NUMBERING_FIX_2026_05_22.json` explicitly. Its module docstring separately claimed the
five mis-numbered proteins were "exactly the five weakest" in the published panel. 🔴 That is **false**
and is corrected in place: the mis-numbered set is ranks 1, 2, 4, 5 and 6, skipping rank 3 (Abrin, clean
5/5) and reaching past the weakest five. The gap at rank 3 is what makes the RIP finding above survive.

**Artifacts regenerated** from the corrected `fspe_results.json`, each with a `*_PRE_NUMBERING_FIX*`
sibling so the change is reproducible in both directions: `fspe_results.json`,
`fspe_protein_level_test.json`, `mdrp_risk_table.json` (`fspe_esm2` column) and
`evaluation_report.json`. Hand-curated files updated in lockstep, following the precedent set by the
2026-06-04 entry: `README.md`, `docs/EVALUATION_REPORT.md`, `docs/BIOHUB_RESEARCH_BRIEF.md`,
**`huggingface/README.md`** (the public model card) and **`results/summary_risk_table.csv`** (the public
Hugging Face dataset preview, which had Ricin at 1.226 / p 0.9788 and now reads 1.230 / 0.9952). The
model card was the one I missed on the first pass, which is the third time that file has needed a
separate sweep; the 2026-06-04 rule that it moves with the other two is not yet a habit.

### 🔴 Still wrong, and why it cannot be fixed here

The code is fixed for all four sequence consumers, but only ESM-2 could be re-run on this machine. The
numbers below are still computed on the wrong positions for Ricin, Barnase and diphtheria toxin and
**must not be quoted** until regenerated:

| artifact | blocker |
|---|---|
| `results/esm3_fspe_results.json`, and the `fspe_esm3` / `fspe_saprot` columns of `mdrp_risk_table.json` | the `esm` package is not installed here; SaProt additionally needs Foldseek 3Di preprocessing on the cluster |
| `results/fhs_results.json`, and the `fhs` column | needs the SAE. Independently, `docs/EVALUATION_REPORT.md` already records FHS as non-reproducible because the fallback SAE trains without a seed, so a re-run would move these numbers for a second, unrelated reason. FHS was already labelled a direction of work rather than a measurement; this is the second reason not to quote it. |
| `results/fsi_evodiff_results.json`, Ricin row only | needs EvoDiff. Barnase and diphtheria toxin have no EvoDiff run, so Ricin is the only affected row. |

⚠️ Ricin's ESM-2 correction was +0.004, and that is **not** a reason to assume the ESM-3, SaProt, FHS and
EvoDiff corrections are small. Those are different metrics over different representations. "The ricin
active site carries no positional signal under either numbering" is a measured fact about ESM-2, not a
prediction about the rest.

## 2026-09-21 (seventeenth entry) — Recreating the 650M pool embedding today matches 9/20's endpoints exactly, and diverges by sub-percent on the intermediate sizes of the two failing classes only

### The provenance question

`src/49_external_test_partition.py` and the accompanying canonical-arm run needed a 650M embedding of the
8,259-protein benign pool. The .npy was not on this machine, because *.npy is gitignored, and it had
been generated on Cayuga on 2026-09-19 by `src/35`'s `--embed --arm esm2_650M` path and then evicted from
local scratch. So today the 8,259 pool proteins were re-embedded on Cayuga (SLURM on an A40, importing
`src/02b_esm2_embed_v2`'s helper at MAX_LEN 1022, 293 sequences truncated exactly as before), transferred
back, and used by `src/49`. That leaves an open question: is today's embedding the same object as the
9/19 one, or a re-run whose numerical differences happen to be small?

### The check

`src/35_negative_scaling_curve.py` was rerun against today's embedding and compared against
`results/v3/negative_scaling_curve_esm2_650M.json`, which was committed on 2026-09-20 and generated from
the 9/19 embedding. It uses the pool differently at different sizes: **n=296 draws no random subsample
and n=8259 uses the whole pool**, so any embedding difference must appear in those two, while n=1000 and
n=3000 subsample from a `default_rng(0)` permutation of the pool.

    class                              n=296       n=1000     n=3000    n=8259
    phage_peptidoglycan_hydrolase      MATCH       +0.208pt   +0.104pt  MATCH
    beta_lactamase                     MATCH       +0.476pt   +0.238pt  MATCH
    rip_rrna_glycosidase               MATCH       EXACT      EXACT     MATCH

Endpoint agreement to six decimal places on all three classes rules out a different embedding: at
n=8259 the whole pool is used, and any per-protein difference would appear in the mean recovery. So
today's `embeddings_pool_large_esm2_650M.npy` is bit-for-bit the same object as the 9/19 embedding for
every downstream that uses the pool as a whole, and `src/49`'s canonical-arm figures are on the same
material as everything §10.9 reports.

### What the intermediate-size drift is, and is not

The two failing classes drift by 0.10–0.48 pt in the middle of the sweep and the recovered class is
exact. That direction is informative: `rip_rrna_glycosidase` has clean class separation, so the
logistic regression converges to the same probabilities under any competent BLAS. `phage` and
`beta_lactamase` do not, and their scores near the threshold are sensitive to floating-point pathways
that differ between the Cayuga node's BLAS and this laptop's. The 9/20 commit that pins this line of
work (`4adc12e`, "HPC results from job 3377580, and the first measurement that device does not change
these numbers") was measuring device stability on `03b`'s LOMO output, not `src/35`'s scaling curve, so
this rerun is the first measurement on the scaling curve. **The device claim, on this script, is
qualified rather than absolute: endpoints do not move, intermediate sizes on the failing classes move
by fractions of a point.** The verdict on this file is REFUTED on either machine, so the section's
reading does not shift.

`src/22_claims_audit.py` does not pin any value from this artifact, so the gate would not have
noticed the drift either way. The committed 9/20 artifact is restored in place and no downstream
number moves, but the observation is recorded because a future run on a third machine would otherwise
reopen this question, and because "device does not change these numbers" is a claim that has now been
checked on one script and is qualified rather than absolute on another.

### One check on `src/37`, which is where the drift is stronger

`src/37_negative_supplement_from_pool.py` was rerun with the same discipline and the drift is present
across every K, including **both endpoints**, up to 0.95 pt. That looks worse than `src/35` at first,
where K=296 and K=8259 matched to six decimals. It is not: `src/37` prepends the 296 panel negatives
before the K pool proteins in `np.vstack([N_panel, N_pool[order[:k]]])`, and that concatenation puts
the same rows in a different order in the matrix `StandardScaler` fits. Feature means and standard
deviations are floating-point sums whose result depends on the order of the summands, so LogReg
starts from a slightly different scaled input and converges to a slightly different solution. Same
mechanism as the HPC vs Mac drift, this time inside one machine, driven by concatenation order. The
verdict on `src/37` is REFUTED on both machines and no downstream number moves; the committed 9/20
artifact is restored and this observation joins the entry rather than opening a new one.

---

## 2026-09-21 (eighteenth entry) — A collateral-damage measurement was scored on the training positives, which reports 100% whatever the negative set does, and would have turned a partly indiscriminate attack into a clean targeted one

Found while writing criterion 18 of `docs/DETECTOR_CRITERIA.md`, before the number reached a
committed document but after it had been drafted into one.

### What happened

`src/37`'s canonical-arm run showed that appending 500 pool proteins chosen for similarity to
`rip_rrna_glycosidase`, labelled benign, cut that class from 94.8% to 51.9%. The obvious follow-up
question is whether the damage is confined to the targeted class, because a targeted suppression and
a general degradation have completely different security readings.

The first pass answered it with a scratch script that scored three quantities per negative set: the
held-out class's recovery, the false-positive rate on the pool, and "recovery on all other
positives", computed as `m.predict_proba(P[tri])`. That third quantity is wrong. `P[tri]` is the
training-positive matrix: every protein in it was just used to fit the model being scored. It
returned 99.9% for the panel-only set and 100.0% for both supplemented sets, which is not a finding
about collateral damage but a restatement of the fact that a logistic regression separates its own
training data.

The draft text that number produced said the targeted class fell 42.9 points while **"every other
hazard class is untouched at 100.0%"**, and concluded that targeted suppression is invisible to
aggregate monitoring.

### What the correct measurement says

Held out properly, each class in its own fit exactly as `03b` does it, the collateral is large and
often larger than the targeted damage. Under the same nearest-500 supplement aimed at rip:

| quantity | in-sample (wrong) | LOMO (correct) |
|---|---|---|
| targeted class, rip | 51.9% | 51.9% |
| worst other class | 100.0% ("untouched") | 78.6%, `adp_ribosyl_ab_toxin`, −21.0 pt |

Swept across all 13 classes in `src/50_reference_set_poisoning.py`, the targeted drop exceeds the
worst collateral drop on only **4 of 13** classes on the canonical arm and 2 of 13 on esm2_35M.
Targeting `adp_ribosyl_ab_toxin` costs the target 16.7 points and costs rip 34.3. The honest
description is broad degradation with a targeted component in a minority of classes, not a precision
attack, and the mechanism is geometric: the per-class nearest-500 sets overlap at mean Jaccard 0.324
and up to 0.883, spanning 1,999 distinct proteins across 6,500 slots.

### Why this one is worth an entry even though nothing shipped

Three reasons. The error produced a number that was **more publishable than the truth**, which is the
direction that does not self-correct. It was caught by asking what `P[tri]` actually contained rather
than by any check in the repository, and no gate would have caught it, because the quantity was new
and therefore pinned to nothing. And a 100.0% that does not move when the input changes is the
signature to look for: the first pass reported 99.9%, 100.0%, 100.0% across three quite different
negative sets and that invariance was the available clue, read at the time as a clean result rather
than as a dead instrument.

`src/50` now carries the warning in its module docstring so the next reader meets it before the
method, and criterion 18 leads with `cry_insecticidal` rather than rip because cry replicates on both
arms while rip's canonical 62.9-point drop inverts on esm2_35M. The claims audit pins both arms and
its assertion **requires** the 35M arm to be loud, so a future run that quietly made both arms
agreeable would fail the gate instead of strengthening the claim.

### A rule for when a drifted artifact is replaced and when it is restored

Entry seventeen restored both re-run artifacts and said no downstream number moved. On 2026-09-21
that stopped being true for one of them, so the two are now handled differently and the rule is
written down rather than decided case by case.

**An artifact is replaced when a public document quotes its exact values, and restored otherwise.**

- `results/v3/negative_scaling_curve_esm2_650M.json` is **restored** to the HPC version. The local
  re-run reproduced entry seventeen's signature precisely and independently: endpoints exact at
  n=296 and n=8259, intermediates drifting +0.21, −0.10, +0.48 and −0.24 pt at n=1000 and n=3000,
  on the two failing classes only, `rip_rrna_glycosidase` exact, verdict REFUTED either way. No
  document quotes these numbers, so there is nothing to gain by churning the file and the earlier
  provenance is worth keeping.
- `results/v3/negative_supplement_curve_esm2_650M.json` is **replaced** by the local run, because
  §10.7.2 of `docs/MECHANISM_GENERALIZATION.md` is the first document to quote this arm's table and
  the numbers in the document have to be the numbers in the artifact. The drift reaches 0.95 pt and
  is visible at one decimal place in 6 of 6 rows, so quoting one file's values beside the other's
  was not an option. The file also gains four diagnostic keys that did not exist before.
- `results/v3/negative_supplement_curve_esm2_35M.json` is replaced with zero risk: it reproduced its
  committed curve values to 0.0000 pt on every cell and changes only by the four new keys. That
  exact reproduction is itself the control showing `src/37` is deterministic on an unchanged
  embedding, which is what makes the 650M drift attributable to the re-generated pool embedding and
  the concatenation order rather than to the script.

### And the K=0 check in `src/37` was comparing a 30-seed mean to a 5-seed one

The script's docstring already said this, correctly, in September: `03b` publishes a five-seed mean
and `src/37`'s `recover()` uses thirty, so the printed "K=0 matches stored LOMO" line failed on the
failing classes every time it ran. The code was not doing what the prose said. It now recomputes K=0
at five seeds for the comparison and reports the thirty-seed value separately, and all three classes
pass: phage 10.0% against 10.0%, beta-lactamase 18.6% against 18.6%, rip 94.3% against 94.3%.

Worth stating because it bounds how the published per-class table should be read. At 200 seeds the
canonical arm gives phage 12.1% and beta-lactamase 20.8%, so the published five-seed 10.0% and 18.6%
sit about two points low. Every published class value is inside the 95% interval of a five-seed mean
around its 200-seed value, so nothing is wrong, but the intervals are wide: beta-lactamase's per-seed
standard deviation is 11.3%, giving a five-seed mean a ±9.9 pt interval. The classification of
beta-lactamase as a sub-25% failure is therefore **not** established by the five-seed table alone,
whose interval reaches 28.5%. It is established at 200 seeds, where the interval is [19.2, 22.4].

On esm2_35M the same recomputation is starker and it sharpens rather than weakens that arm's
documented limitation. The published 1.4% for beta-lactamase is a low five-seed draw of a 200-seed
mean of 4.7%, but **131 of 200 seeds return exactly zero**. So the standing caveat that this arm
"has headroom in only one of the two failing classes" survives verification and should be stated as
the stronger fact: a class that recovers nothing in two thirds of seeds has no room to decline, which
is a better reason than the 1.4% point estimate that was standing in for it.

---

## 2026-09-21 (nineteenth entry) — SEB's published FSPE ratio is computed at a coordinate offset that UniProt rules out, two of its masked positions sit inside a cleaved signal peptide, and the correct frame moves the protein-level headline

Found by running `src/52_flagged_entries_vs_uniprot.py`, which was written to close two long-standing
open items by reading the canonical record instead of reasoning about them further. One closed
favourably. This one did not.

### The finding

`P01552` (staphylococcal enterotoxin B) carries nine annotated positions, all nine of which fail the
residue-identity check at the published offset of 0. That was already recorded as unresolved. What
was not known is that offset 0 is **disproven**, not merely unsupported.

UniProt gives P01552 a **Signal peptide at 1-27** and the mature chain at 28-266. At offset 0,
positions **23 and 25** fall inside that signal peptide, `MYKRLFISHVILIFALILVISTPNVLA`, landing on
Pro23 and Val25. Both are annotated "MHC-II binding interface". A secreted superantigen cannot
present a receptor interface on a peptide that is cleaved off before secretion, so the published
numbering places two of the nine masked positions somewhere the annotation's own description says
they cannot be.

Three independent facts agree that the correct frame is **+27**:

- the signal-peptide boundary itself, Signal 1-27;
- the known mature N-terminus `ESQPDPKP`, which the precursor carries at 28-35;
- UniProt's **Disulfide bond at precursor 120-140**, which is mature **93-113** and puts a cysteine
  exactly at this entry's position 93. The entry calls that position "Gly93, TCR binding loop": the
  loop is right and the residue name is wrong.

### What it does to the headline

| panel | below 1.0 | exact sign test p |
|---|---|---|
| as published | 13/15 | 0.0037 |
| SEB dropped | 12/14 | 0.0065 |
| **SEB at +27, the only admissible frame** | **12/15** | **0.0176** |

SEB's ratio goes **0.9556 to 1.0417** and crosses 1.0, so it stops being one of the thirteen. The
0.9556 was already the weakest non-RIP ratio in the panel, which in hindsight is what a near-null
contribution looks like when the masked positions are largely arbitrary and two of them are in a
signal peptide.

### Why it is recorded and not applied

Applying +27 would violate this project's own acceptance rule for offsets, set in `src/46` and
pinned in the claims audit: an offset is accepted only when it is the **unique** integer placing
*every* annotated residue identity correctly. At +27, one of nine matches. The identity strings are
themselves corrupt, and the corruption is legible: at +27 the mature chain carries **Asn23 and
Tyr89** where this entry writes Tyr23 and Asn89, a transposition of the two residue names between
two positions. That is the reverse of the P02879/P00648/P00588 defect, where the identities were
right and the frame was wrong, and it is why a `precursor_offset` cannot repair this entry.

So the honest state is that **no annotation for SEB currently meets this project's standard**:
offset 0 is ruled out by biology, and +27 is ruled out by the project's own rule. That argues for
exclusion rather than correction, and exclusion already has a second and independent justification
sitting in `docs/EVALUATION_REPORT.md`: SEB is **excluded from FSI** on the grounds that a
superantigen has no discrete catalytic site to recover, and the identical objection applies to FSPE,
where the same residues are masked. Dropping it gives 12/14 at p = 0.0065 and resolves that
asymmetry at the same time.

The published panel stayed as published until that call was made deliberately, because it changes a
number on the Hugging Face model card and in the README. ✅ **The call was made on 2026-09-22 and SEB
is excluded; see the twentieth entry.** The open item had moved from "the numbering is unresolved" to
"the published numbering is ruled out", and those warrant different treatment: the first is a caveat,
the second is a correction. UniProt annotates no Site, Binding site or Active site features for SEB at all, so the
re-curation that would actually fix the entry still needs 3SEB or the superantigen literature.

### The other open item closed favourably, and its diagnosis was wrong

`Q51451` (ExoS) carried a hypothesis, marked in the annotation file as unchecked, that its spurious
position 234 was the **start of the ADP-RT domain** recorded as though it were a residue. UniProt
refutes it: the ADP-ribosyltransferase domain runs **243-429** and residue 243 is Lys. The entry's
own `function` field claims 233-453 and matches neither. Position 234 stays spurious with no
explanation for where it came from.

The real defect is larger than the open item named. UniProt annotates **Active site 319, 343, 381**
and **Binding site 146, 186, 187** on this entry. The annotation agrees on two of those, 146 and
381, and **omits four**: 186, 187, 319 and 343. Positions 148 and 379 have no UniProt feature,
though 379 is defensible as the first glutamate of the E-x-E motif whose second glutamate, 381, is
the annotated active site.

None of it changes the verdict, which is why this half resolves favourably. The ratio is 0.6618 as
published, 0.6034 without position 234, 0.6836 on UniProt's six sites alone, and 0.6195 on the union
of both sets. All four are below 1.0. ExoS's contribution to the headline is robust to an annotation
that is substantially incomplete, which is worth knowing and is the opposite of the SEB result.

### The general lesson, which is not about either protein

The identity check that `src/46` installed catches a wrong frame when the identities are right. It
cannot catch a wrong frame when the identities are also wrong, and SEB is that case: zero of nine
matched at offset 0 and the entry ran anyway for months. **A cheaper check would have caught it on
day one: no annotated functional position may fall inside a cleaved signal peptide or propeptide.**
That rule needs no residue identities, only the UniProt feature table, and it would have flagged
this entry immediately.

So it was run, over every entry, in `src/53_signal_peptide_sweep.py`. **Fifteen of sixteen entries
place every annotated position outside every cleaved region, and the single hit is P01552.** Nothing
lands inside a propeptide, which would have been the weaker case since a propeptide can be functional
before cleavage, and nothing indexes past the end of a sequence. Seven of the sixteen accessions
carry a signal peptide at all, so the check had real opportunity to fire and fired once.

Two details worth keeping from that run. The three entries repaired in entry sixteen store their
positions in **mature** coordinates and carry a `precursor_offset` the pipeline applies before
masking, so a sweep that reads `catalytic_residues` raw scores them in the wrong frame: the first run
of this script reported P00588 and P00648 as hits alongside P01552, which was the check firing on its
own repairs. Applying each entry's own offset, as the pipeline does, clears both. And P55981 (VacA)
has a deliberately **empty** position list because it is a pore-forming negative control with no
discrete active site, which is why the panel has sixteen annotation entries and fifteen FSPE
proteins.

The rule is cheap enough to be a standing gate rather than a one-off sweep, and it is a better first
check than the identity test because it holds regardless of whether the residue names are right. So it
is now a gate rather than a record: the claim in `src/22_claims_audit.py` **recomputes** the sweep from
`data/annotations/functional_sites.json` and the UniProt cache instead of reading `src/53`'s artifact,
because reading the artifact would let someone add a bad annotation and still pass on yesterday's
answer. It also fails when it cannot check: an entry whose accession has no cached UniProt record lands
in an `uncheckable` list that the assertion requires to be empty, so adding an entry without caching
its record is a failure and not a silent skip.

Both failure modes were negative-tested rather than assumed, because a gate never seen to fail is not
known to work. Injecting position 10 into `P01555`, whose signal peptide is 1-18, makes the audit exit
1 and name the entry. Adding an entry with no cached record makes it exit 1 with that accession in
`uncheckable`, and a second claim fires independently on the same edit. The annotation file was
restored from a byte-level backup afterwards and `git status` confirms it unchanged.

---

## 2026-09-22 (twentieth entry) — SEB is excluded from the FSPE protein-level test, the headline weakens to 12/14 at p = 0.0065, and the two protein-level tests move in opposite directions

The decision entry nineteen said should be taken. Taken deliberately rather than drifted into, because
it changes a number on the Hugging Face model card, in the README and in this report.

### What changed

`P01552` (staphylococcal enterotoxin B) now carries `fspe_excluded: true` in
`data/annotations/functional_sites.json`, with the reasoning in `_fspe_exclusion_reason`, and
`src/21_fspe_protein_level_test.py` honours the flag.

| statistic | before | after | direction |
|---|---|---|---|
| ratios below 1.0 | 13/15 | **12/14** | one fewer success, one fewer protein |
| exact sign test p | 0.0037 | **0.0065** | **weaker** |
| sign-flip permutation p | 0.0002 | **0.0001** | stronger |
| mean log ratio | −1.850 | −1.979 | more negative |

### Why exclusion rather than rescoring

Two independent reasons, either sufficient alone.

**No admissible annotation exists.** Entry nineteen established that the published offset of 0 is
disproven: positions 23 and 25, both annotated "MHC-II binding interface", fall inside the cleaved
signal peptide at 1-27, and a secreted superantigen cannot present a receptor interface on a peptide
removed before secretion. The only frame consistent with the record is +27, agreed independently by
the signal-peptide boundary, the `ESQPDPKP` mature N-terminus and UniProt's disulfide at precursor
120-140, which is mature 93-113. But +27 places only **1 of 9** annotated residue identities
correctly, because the identity strings are themselves transposed, so it fails the acceptance rule
`src/46` applied to every other corrected entry: an offset is accepted only when it is the unique
integer placing *every* identity correctly. Rescoring at +27 would mean holding this entry to a weaker
standard than the three that were repaired properly.

**The objection that already excludes it from FSI applies here unchanged.** SEB has been excluded from
FSI since the metric was written, on the grounds that a superantigen has no discrete catalytic site to
recover. Masking `catalytic_residues` is not defined for such a protein, and that is true whatever the
coordinates are. So excluding it from FSPE does not create an inconsistency, it **closes** one that
`docs/EVALUATION_REPORT.md` had carried as an open item for months.

### The direction, which is the part worth trusting

The sign test gets **worse**, 0.0037 to 0.0065. That is the headline figure on three public surfaces
and it is now weaker than it was. This is the direction-blind test of
`docs/DETECTOR_CRITERIA.md` criterion 11, and it is the third time this project has kept a correction
that hurt the metric: the Ricin numbering fix in entry sixteen pushed its ratio the wrong way
(1.226 to 1.230) and was kept, and one of three iso-FP corrections hurt and was kept.

⚠️ **But the permutation test gets better, and reporting only one of the two would be cherry-picking.**
It moves 0.0002 to 0.0001, because SEB's 0.956 was the closest to 1.0 of all thirteen successes, so
removing it makes the mean log ratio more negative. The count-based test loses a success while the
magnitude-based test loses its weakest contributor. Both figures are now stated together on every
surface, and the claims audit asserts **both directions** so neither can be quoted alone: it requires
the sign test to be the weaker one and the permutation p to be the smaller one.

### What was deliberately not done

`results/fspe_results.json` is **not** regenerated. Every per-protein ratio in it, SEB's 0.9556
included, is the record of what was computed and stays visible. Recomputing fourteen unaffected
proteins in order to drop one row would move them by the floating-point drift documented in entry
seventeen and force every published per-protein figure to be restated for no gain. The exclusion is an
analysis decision and is applied where the analysis happens, in `src/21`.

SEB's `catalytic_residues` and `residue_annotations` are also left in place rather than deleted. They
are the record of what was curated, and `src/53`'s signal-peptide gate reads them and must keep
flagging this entry: deleting them would make the panel look clean by removing the evidence.

`src/21` now writes a `without_exclusions` block alongside the reported figures, so the exclusion can
never become a silent one, and it reports any accession flagged for exclusion that is **absent** from
the results, because a flag that matches nothing is a no-op that would otherwise pass unnoticed.

---

## 2026-09-24 (twenty-first entry) — A length-dependent SaProt lookup silently dropped 3 of 15 proteins, the comment shipped with its fix mis-stated the census, and the figure that re-run retired survived in prose on two other sections

Three defects on one thread, written up together because they are the same failure repeated at three
altitudes: a silent drop in code, a wrong count in the comment about that drop, and a stale figure in
the document the drop was fixed to correct. The first two were found and fixed on 2026-09-23 but never
entered here; the third was found on 2026-09-24 while writing this entry, and is the one the gate
should have caught and did not.

### 1. The silent drop

`get_saprot_masked_entropy` in `src/14_esm3_separability_fspe.py` resolved a protein's structure
tokens by an **exact match on the full amino-acid string**, while `run_fspe_analysis` hands it
`truncate_sequence(sequence, MAX_SEQ_LEN)` with `MAX_SEQ_LEN = 1022`. Any protein longer than the
limit therefore missed the lookup, returned `None` at every position, and dropped out of the run with
`Could not compute entropies` — the generic message for a protein with no usable positions, not an
error naming the cause.

| accession | protein | length | visible as a loss? |
|---|---|---:|---|
| `P04958` | tetanus toxin | 1,315 | yes, it had a value and lost it |
| `P0DPI1` | botulinum neurotoxin A | 1,296 | **no**, never carried one |
| `Q99ZW2` | Cas9 (negative-class member) | 1,368 | **no**, never carried one |

Verified against the panel FASTA: 15 members, exactly these 3 over 1,022. The two invisible ones are
the point. A protein that never had a value reads as ordinary missing coverage, and SaProt had partial
coverage for unrelated reasons (missing AlphaFold structures), so the absence had a ready innocent
explanation sitting next to it.

**Older than it looks, and exposed by something getting more correct.** The legacy April token file was
built from already-truncated sequences, so its keys matched what the function is handed and the defect
stayed latent for five months. It surfaced only when the v2 3Di adapter began building tokens at full
length, which is the right behaviour: the adapter becoming more correct is what made the consumer's
bug observable.

⚠️ **The first diagnosis was wrong and is recorded because it was confidently stated.** The initial
claim was that `MAX_SEQ_LEN` was being applied to an interleaved amino-acid/structure string and so
halved the effective limit. SaProt's `vocab_size` is 446, about 21 amino acids times 21 structure
states, so the alphabet is combined and one interleaved pair is **one** token; 1,022 is correct in
residues. Reading the config settled in one minute what reasoning from the encoding had got backwards.

**Fix and its limit.** Both the full string and its truncated prefix are registered as keys (Cayuga job
3392513, 2m24s, purely additive: 27 rows to 30, zero change among shared rows, zero remaining warnings,
SaProt now covers all 15). Matching a model input by exact sequence string is fragile by construction;
**passing the accession down is the real repair and was not done**, because it is a larger change than
that run should carry. It remains open.

### 2. The census inside the fix

The comment shipped with that fix stated that `P04958` "is the only panel member over the limit and it
is exactly the one that vanished", while the commit message shipping the same change said three. The
comment held the count taken before the census; the message held the count after it. Nothing in
between checked them against each other, and the comment is the copy a future reader meets first.
Corrected 2026-09-24 against the panel FASTA, with the superseded count named in place rather than
deleted.

### 3. The figure that would not die

The 2026-09-23 re-run existed to retire a cross-model flip count that had been wrong twice: **3 / 12**
before the 2026-05-22 numbering fix, then, once that fix re-ran ESM-2 alone and left two columns in the
old coordinates, a narrowing to the nine unmoved rows giving **2 of 9** with three rows named
indeterminate. The re-run put all three columns on one numbering and the honest figure is **5 of 12**.

`docs/EVALUATION_REPORT.md` § 7 was updated to say so. **Two other sections of the same document were
not**, and kept asserting the superseded figure in words:

| line | section | text as it stood |
|---|---|---|
| 331 | § Cross-model FSPE | "Three of 12 proteins **flip ratio sign across models**" |
| 362 | § Where this framework currently loses | "Three of 12 proteins flip FSPE direction between ESM-2, ESM-3, and SaProt" |

So for one commit the report stated 5 of 12 in its limitations and three of twelve in its results, and
`src/22_claims_audit.py` passed all 69 claims while it did.

**Why the gate missed it, which is the transferable part.** The claim carried a `must` pin on the new
sentence and a `forbid` list containing `"honest current figure is **2 of 9**"` and
`"the figure is 2 of 9"`. The pin proved the new figure was **present**; nothing proved the old one was
**gone**, because that is what the forbid list is for, and the forbid list was written in the same
minute, by the same person, in the spelling that person had just used. The retiring commit even said
the forbid was "narrowed to its assertion form" — and the assertion form actually in the file was the
prose `Three of 12 proteins`, not the digits. **A forbid inherits the blind spot of whoever retires the
figure.** The rule that follows: when a figure is retired, grep the surface for every spelling of the
old number before writing the forbid, and forbid what the grep finds rather than what you remember
writing.

Both sentences now carry 5 of 12 and name the superseded value, and the forbid list covers the prose
form in both cases. Direction: this correction moves the framework's own cross-model consistency
**from three disagreements to five**, so § "Where this framework currently loses" got worse, which is
where it belongs. This is the fourth kept correction that hurts the metric under
`docs/DETECTOR_CRITERIA.md` criterion 11.

### What was deliberately not done

`results/esm3_fspe_results.json` is not regenerated: the 09-23 run is the record and the shared rows
were verified unchanged. No per-protein FSPE value moves in this entry — the SaProt fix is additive and
the flip count is a derived statistic over columns that already existed. The accession-keyed lookup that
would make §1 structurally impossible is still open, and is named here rather than quietly deferred.

---

## 2026-09-24 (twenty-second entry) — The Hugging Face card and this repository had diverged in both directions, so the correction telling readers not to quote the headline AUROC existed on one surface only

Found while preparing the first push in four days, by diffing the live dataset card against the
repository copy before overwriting it. That check was the whole finding: **a blind repository-to-Hub
sync would have deleted a correction.**

### What the diff showed

44 changed lines. The repository is newer on everything from the September work — the 12/14 FSPE
headline, the SEB and numbering caveats, Ricin at 1.230 rather than the pre-fix 1.226, the v2/v3
provenance-control numbers. But three things existed **only on the Hub**:

| Hub-only content | status |
|---|---|
| the `AUROC (v1, superseded)` / `AUROC (v2, screened panel)` rows and the **screening caveat** telling readers to quote 0.974 instead of 0.981 | a real correction, now in the repository |
| `benign_homologs.fasta` described as included in the dataset | correct: the file is in the dataset (33 KB); the repository copy said "see the GitHub repository" |
| the v1 pool (`toxins_positive.fasta`, 69 records) provenance bullet | correct: the file ships; the repository copy had dropped the bullet |

`git log -S "Screening caveat" --all` returns nothing, so that text was authored directly on the Hub
web interface and never came back. The Hub is a git remote that can be edited in a browser, and an
edit made there is invisible to every gate in this repository.

### The part that matters: the most-quoted number had no correction on the primary surface

`README.md` leads its Key Results with **AUROC 0.981 ± 0.016**. That is the v1 panel, whose 60-vs-60
membership is not shipped, which still contains the two identical-sequence pairs the v2 build removed,
and which predates the dated 2026-09-03 decision assigning barnase to the positive class and Cas9 to
the negative class. The screened figure is **0.974 ± 0.014**, and it entered this repository on
2026-09-05.

So for **nineteen days** the project's single most-quoted number was led with, uncorrected, on its
primary surface, while the correction sat on a secondary one. No gate could notice: the
`Embedding separability AUROC` claim recomputed 0.981 from its artifact and pinned it to **no
document at all**, and a claim with no document pin cannot fail on a document.

🔴 **Direction.** The quotable figure moves 0.981 to 0.974 and the panel it is computed on shrinks
from an unshipped set to a shipped one. This weakens the headline, and it is the fifth kept correction
that does so under `docs/DETECTOR_CRITERIA.md` criterion 11.

### Two more defects found by the same pass

**The interview brief was off the audited surface.** `docs/BIOHUB_RESEARCH_BRIEF.md` is the document
most likely to have its numbers spoken aloud, and it was the only one quoting headline figures that no
gate checked. Adding it to `PUBLIC` immediately surfaced a stale row: `15 proteins; mean ratio 0.437;
13/15 below 1.0`, the pre-exclusion headline the audit already forbids leading with. It survived
because the forbid string was the full sentence `13/15 below 1.0, sign test p = 0.0037` and the brief
carried the fragment. **Second time in two days that a forbid pinned to one spelling missed another**
(entry twenty-one was the same defect on `Three of 12 proteins`). The forbid is now the fragment.

**Three sentences cited an entry of this log that has never existed.** Two on the Hub card and one
inside this file referred to a "2026-09-10 entry". The dated headings here run 09-05, 09-11, 09-18; no
09-10 entry was ever written. The screening decisions those sentences meant are in
`data/sequences/panel_v2_manifest.json`, and the ESM-C 6B refutation the third one meant is § 9.3 of
`docs/MECHANISM_GENERALIZATION.md`. All three now point at what actually holds the record.

This is the **second occurrence** of this defect class: on 2026-09-20 an entry was cited by number for
two days before it was written. A citation is cheap to write, nothing downstream reads it, and a
reader who follows a dead one concludes the record is missing rather than misnamed. It is now gated:
`cited_entries_exist` resolves every `<date> entry` citation in the audited documents and in this log
against the headings actually present, skipping dates in quotes so a sentence can name the citation it
replaces.

### What is gated now, and negative-tested

Three checks were added or tightened, each verified to fail when the defect is reintroduced and to
pass when it is removed:

| check | fails on |
|---|---|
| `the v1 separability figure carries its screening caveat on every surface that prints it` | removing the caveat from the README, the card or the brief; 0.974 is recomputed from the v2 LOMO artifact rather than quoted |
| forbid `13/15 below 1.0` (was the full sentence) | the fragment reappearing on any audited surface |
| `every dated entry cited by a document exists in the corrections log` | any `<date> entry` citation with no matching heading |

71 claims.

### What was deliberately not done

The Hub card is **not** overwritten from the repository until the three Hub-only items above are in
the repository copy, which is the point of this entry. No result artifact is regenerated: every number
here already existed, on one surface or the other. And the underlying asymmetry is unfixed — the Hub
remains editable in a browser, so the next out-of-band edit will be just as invisible. The check that
would close it is a CI step diffing the live card against the repository copy, which is not written.

---

## 2026-09-24 (twenty-third entry) — The project's worst-scoring criterion was absent from both public surfaces, and two artifacts still called a two-arm result provisional

`docs/DETECTOR_CRITERIA.md` opens by arguing that the split comes first and that the other seventeen
criteria are downstream of it. This project scores a **fail** on that criterion and says so, in that
document and in § 2.6.1 of `docs/MECHANISM_GENERALIZATION.md`. Neither is a document a reader meets
first.

### What a reader of the headline surfaces could see

The dataset card prints a per-class recovery table whose every column is `flagged@95` or `flagged@99`,
under the heading **"Three cautions that belong with any number above"**. The three were provenance
leakage, amino-acid composition, and the classifier head. **The operating point itself was not among
them.** The README did not mention the split at all.

So the published position was: the 95 in `flagged@95` is a specificity measured on negatives the
pipeline had already fitted or calibrated on, and nothing on either surface said so. The honest
figures, 200 seeds, both arms, nominal 5%:

| | canonical 650M | esm2_35M |
|---|---|---|
| `np.quantile`, panel-calibrated, pool-tested | **7.87%** | **7.94%** |
| conformal, same conditions | 5.98% | 6.18% |
| `np.quantile`, distinct names only | **9.64%** | **10.29%** |

A nominal 5% budget costs about 8%, or about 10% once the pool's name redundancy stops hiding the
easy negatives. That is now the **first of four cautions** on the card and sits in the README's
reviewer framing, and the claims audit pins both sentences plus the § 2.6.1 table row, so removing any
of them fails the gate.

🔴 **Direction.** This correction publishes a worse number. `flagged@95` now has to be read against a
realized false-positive rate of about 8%, so every per-class recovery figure on the card is quoted at
an operating point looser than its label. Sixth kept correction that weakens a published claim under
criterion 11.

### The stale flag underneath it

`src/49_external_test_partition.py` wrote `"single_arm_provisional": True` as a **literal**, together
with a verdict string ending "SINGLE ARM, provisional". That was true when only `esm2_35M` had pool
embeddings. The canonical 650M pool embeddings were computed on 2026-09-21, the script was run on
them, and § 2.6.1 was updated to say route 2 runs on both arms — but the flag and the sentence are
written by the script, so **both artifacts went on declaring themselves provisional** while the
document citing them said the opposite.

A status that a run cannot update is a status that is wrong as soon as the situation changes. The arm
count is now read from disk: `external_test_partition_*.json` is globbed, the current arm is unioned
in, and provisional means fewer than two. Both arms were re-run at 200 seeds to regenerate the
artifacts.

✅ **Every number reproduced exactly.** The re-run diff on both files is the status fields and nothing
else — no false-positive rate, per-class recovery or confidence bound moved by any amount that shows
at the stored precision. That was not the purpose of the re-run, but it is the strongest
reproducibility evidence in this repository: two 200-seed runs, three days apart, on regenerated
embeddings for one arm, byte-identical in every quantity.

### The gate that could not have caught it

The claim covering this decomposition began `lambda v: v is None or (...)`, so a **missing artifact
passed vacuously**. On the claim that carries the project's worst-scoring criterion, the default was
"absent means fine". It now requires both arms, asserts the four figures the public surfaces quote,
and asserts that no artifact still calls itself provisional.

### What was deliberately not done

The published split is **not** changed: the panel keeps 178 train / 118 calibrate / 0 test, and the
per-class table keeps `np.quantile`. Re-thresholding the published table on conformal would break
comparability with every figure in `docs/MECHANISM_GENERALIZATION.md` and with the frozen v2 panel,
for a gain that is already reported alongside it. Criterion 1 therefore still scores **fail**:
measured, stated on the surfaces where it matters, and not repaired. The repair is a negative set
large enough to carve a test partition without halving the calibration resolution, which is
criterion 8's problem and is not solved here.

---

## 2026-09-24 (twenty-fourth entry) — A loader that could not see panel v3 had scoped a scientific claim to one model family, and closing it produced a double dissociation

`src/02b` was given `--panel` when v3 was built. `src/02e`, which loads ESM-C and ESM-3 through
EvolutionaryScale's SDK rather than HuggingFace, was not. So v3 could be embedded with the ESM-2
capacity ladder and with nothing else, and **every v3 arm in this repository was an ESM-2 arm** —
not because anyone decided the question was about ESM-2, but because the other loader could not
reach the file.

That is why this entry is not simply an added result. § 10.4 and § 10.6 describe beta-lactamase and
phage peptidoglycan hydrolase as the classes the probe **cannot reach**, and § 10.6.1 tested that
across "five arms", all five of them ESM-2. The word "unreachable" was doing work that the
measurement did not support.

### What the three new arms found

`slurm/esmc_v3_arms.sh`, Cayuga job 3397785. At 30 seeds, 95% specificity:

| arm | beta-lactamase | phage |
|---|---|---|
| canonical ESM-2 650M | 21.2% [16.5, 25.9] | 12.2% [10.0, 14.4] |
| **ESM-C 600M** | **40.5% [36.5, 44.5]** | 12.1% [9.4, 14.8] |
| **ESM-3 1.4B** | 2.9% [0.7, 5.0] | **31.7% [27.4, 36.0]** |
| ESM-C 300M | 5.2% [2.6, 7.8] | 4.9% [3.1, 6.7] |

A **double dissociation**, interval-separated in both directions: ESM-C 600M nearly doubles
beta-lactamase and is indistinguishable from the canonical arm on phage; ESM-3 1.4B is the best arm
on the panel for phage and is at the floor on beta-lactamase. Neither class is generally hard.
**Recovery is joint in the representation**, which is the same shape as § 5's point about the
positive set and § 10.4's about the operating point, with a third term.

Two further things fall out, and the second costs the margin story something:

- **The ESM-C 600M anomaly replicates on an independent panel and stays non-monotone in capacity:**
  48.3% on v2 and 40.5% on v3, against 300M's 16.4% and 5.2%. Whatever it is, it is not scale.
- **Margin does not transfer across arms for a fixed class.** ESM-C 600M has the second most
  negative beta-lactamase margin of any arm, −0.0231, and the best beta-lactamase recovery. The
  correlation in § 10.6.1 is across the twelve classes *within* an arm, and it is significant in
  8 of 8; it was never a statement about one class across arms, and inside the ESM-2 ladder the two
  moved together closely enough that the stronger reading looked available.
  ESM-3 1.4B is the one arm with a **positive** phage margin and the best phage recovery, so margin
  and recovery agree on the arm that breaks the pattern.

### Three hardcoded counts found while doing it

Each would have produced a wrong number silently rather than an error:

1. `src/41_v3_arm_seed_stability.py` carried a **fixed five-entry arm list**, while `src/30` already
   discovered arms from the filesystem. A new arm would have been embedded, scored by `03b`, picked
   up by `src/30`, and skipped here — so the 30-seed intervals every cross-arm comparison rests on
   would have covered the old five while the table showed eight. It discovers now.
2. The audit's `v3_arm_seed_stability` helper asserted separation as `== 4`, five arms minus one.
   With eight arms that literal reads False for a reason unrelated to separation. It is
   `len(arms) - 1` now, and the phage overlap is pinned **by name** (`esmc_600M`) rather than by a
   count, because that one overlap *is* the dissociation.
3. `src/02e` itself wrote its output filenames with a literal `v2`.

### A stale sentence whose pin kept passing, for the third time in three days

§ 10.6.1's stability paragraph said "**9 of the 10 arm pairs are disjoint**" and "the canonical
arm's interval separates from all four others on both". After the rerun the figures are 21 of 28,
and on the phage class the canonical arm does **not** separate from all others. The claim's numeric
assertions were updated; **the pinned sentence was not, so the gate passed on a document that now
contradicted its own artifact.** The pin quotes a sentence, so it can only fail when the sentence
changes, and a sentence whose numbers are stale is still the same sentence.

Entry twenty-one was this defect on `Three of 12 proteins`, entry twenty-two on `13/15 below 1.0`.
The mitigation is the same each time and is applied here: the superseded spelling goes in the forbid
list, `"9 of the 10 arm pairs"` and `"separates from\\nall four others"`, alongside the new pin. The
general lesson is that **updating a claim's assertions and updating the sentence it pins are two
edits, and doing only the first is invisible.**

### Environment, and a reproducibility fact worth stating plainly

The v2 ESM-C arrays were built on 2026-09-05 under **esm 3.4.0 / torch 2.11.0**. That environment no
longer exists on this cluster: the only env carrying the SDK is now esm 3.2.1 / torch 2.5.1, where
`LogitsConfig(return_mean_embedding=True)` is rejected and `src/02e` pools the residue stack itself.
**A published arm's environment is gone**, so those arrays cannot be reproduced bit for bit.

`src/56_embedding_source_equivalence.py` re-embedded the v2 panel through the new path and compared
it to the stored arrays: **not numerically identical** — max |Δ| 9.8e-3 against a coordinate scale of
9.8e-3 — but **geometrically equivalent**, minimum row cosine 0.99988 and pairwise-distance
correlation 0.99996. That is why the v2 48.3% and the v3 40.5% may be read as the same phenomenon,
and it is stated rather than assumed. The script reports both verdicts because its first version
reported only a single tolerance and returned "not equivalent" for a pair a probe cannot tell apart.

### Direction

This correction does not weaken a headline number; every recovery figure in the canonical arm is
unchanged. It **retracts a word**. "Unreachable" appears in `docs/MECHANISM_GENERALIZATION.md`, in
`docs/DETECTOR_CRITERIA.md` and in `docs/DETECTOR_EVALUATION_SUMMARY.md` — the last of which was
published earlier the same day — and should be read everywhere as **"not reached by the canonical
arm, and reached by a named other one"**. The summary and § 10.6.1 now say so; the criteria document
inherits it through criterion 3, whose "10% on a 32-member class" is a canonical-arm figure.

### What was deliberately not done

The v2 panel is not re-embedded for publication: the `_mp` arrays exist only as the equivalence
check's evidence and are excluded from the arm discovery in `src/41` and `src/30`. No published
recovery table is re-thresholded. ProtT5 and SaProt still cannot see v3 — `src/02g` and `src/02i`
have the same missing `--panel` that `src/02e` had — so the v3 arm count is eight and not ten, and
that is a known gap rather than a conclusion about those two models.

---

## 2026-09-26 (twenty-fifth entry) — The threshold estimator was validated on 2026-09-20 and applied on 2026-09-26, and applying it costs recovery in every class and gains it in none

Not a defect found, a defect **closed**, and the size of what it was hiding is the entry.

`src/45_negative_test_set_audit.py` established on 2026-09-20 that the published `np.quantile`
threshold does not return its own nominal rate out of sample. `src/48` and `src/49` confirmed the
conformal alternative on two routes and two arms on 09-21. Every one of those runs was a side
analysis. **The per-class table that this repository actually publishes kept `np.quantile`** — `03b`
line 93 and the same call in 03e, 03f, 03h, 03j and 15e — for six days after the estimator was known
to be wrong, because nothing connected the finding to the table.

### What applying it shows

`src/58_conformal_operating_point.py` recomputes the table at both estimators on identical folds,
models and scores. Its own quantile column reproduces `lomo_results.json` **exactly** on all four arms
tested, which is the gate that makes the comparison attributable to the threshold rule.

| run | budget | realized FP, `np.quantile` | realized FP, conformal | classes | dropped | rose | mean change |
|---|---|---:|---:|---:|---:|---:|---:|
| v2 canonical | 5% | **6.56%** | 4.92% | 13 | 7 | **0** | **−8.8 pt** |
| v3 canonical | 5% | 5.08% | 4.24% | 16 | 7 | **0** | −2.1 pt |
| v3 canonical | 1% | **1.69%** | 0.85% | 16 | **14** | **0** | **−15.4 pt** |
| v3 ESM-C 600M | 1% | 1.69% | 0.85% | 16 | 14 | **0** | −14.7 pt |
| v3 ESM-3 1.4B | 1% | 1.69% | 0.85% | 16 | 11 | **0** | −12.0 pt |

**Not one class, in any run, at either budget, gains recovery under the guaranteed threshold.** The
published estimator inflates recovery uniformly, and the inflation grows as the budget tightens:
−2.1 points at a nominal 5% against −15.4 at a nominal 1% on the same arm.

That inverts how the two columns should be read. `flagged@99` is the strictest budget in this
repository and therefore looks like the most conservative figure; it is in fact the figure that
depends most on an estimator known to overshoot. **Read `flagged@95` as recovery at a realized 5.1%
to 6.6%, and `flagged@99` at a realized 1.7%.**

🔴 **And on the frozen v2 panel the `@99` column cannot be computed with a guarantee at all.** At
m = 61 calibration negatives and α = 0.01, `floor((m+1)·α)` is **zero**: there is no k-th largest
score to return, so conformal declines. `np.quantile` returns a number anyway. Every `@99` figure on
the panel that carries every published result is an ungrounded threshold — not a wrong number, a
number with no finite-sample statement behind it. This is § 2.6's resolution ceiling arriving in the
column that quotes the strictest budget.

### Two figures that move enough to restate

**Beta-lactamase on the canonical v2 arm: 21% at a realized 6.56%, 10% at a guaranteed 4.84%.** The
class this repository describes as almost entirely missed is missed about twice as badly at an honest
operating point. At a nominal 1% on v3 the same class goes 7% to **0%**.

### What it does not overturn

🟢 **Entry twenty-four's dissociation survives, and sharpens.** At the guaranteed 4.24%: ESM-C 600M
recovers beta-lactamase at **30%** against the canonical arm's 16% and ESM-3's 1%; ESM-3 1.4B recovers
phage at **28%** against ESM-C's 9%. Roughly double and level-with, in both directions, exactly as at
the published threshold. That finding is one day old and this is the check it needed.

### What was deliberately not done

The published tables **keep** `np.quantile`. Re-thresholding them would invalidate every
cross-reference in `docs/MECHANISM_GENERALIZATION.md`, the frozen v2 panel's comparability, and the
external-validation record, for a result now reported in a parallel column beside them. The estimator
call sites in 03b/03e/03f/03h/03j/15e are unchanged for the same reason. What is removed is the option
of quoting a recovery figure without its realized rate: § 2.6.2 carries the table, the claims audit
asserts the realized rates and the one-directional cost, and the public summary states both.

---

## 2026-09-27 (twenty-sixth entry) — Three documents described a background no code builds, and measuring it found a larger problem than the one being measured

`docs/ARCHITECTURE.md`, `docs/EVALUATION_REPORT.md` and
`docs/MUTATION_EXTENSION_PREREGISTRATION.md` all stated that FSPE's background excludes the two
flanking positions on each side of every functional site. **No code in this repository has ever
implemented that.** `src/04` samples 20 positions from `all_positions - func_positions` and nothing
else. The defect surfaced while implementing FSPE-M, because the preregistration fixes FSPE-M's
background as "exactly as FSPE defines it" and that phrase pointed at two different backgrounds.

### Measured before anything was changed

`src/57_fspe_background_ablation.py` computes three backgrounds over one forward pass per distinct
position. The `published` arm reproduces `results/fspe_results.json` to 1e-6 on every field for all
15 proteins, zero drift, using `src/04`'s own masking code loaded through `importlib` — so the arms
differ only in position selection, and that check is simultaneously step 3 of the
preregistration's run order.

With `src/21`'s SEB exclusion applied, as the published figure applies it:

| arm | ratios below 1 | sign test *p* | permutation *p* |
|---|---|---|---|
| `published`, what `src/04` samples | **12/14** | **0.0065** | 0.00015 |
| same draw minus its flanking members | **12/14** | **0.0065** | 0.00015 |
| the documented metric, built properly | **12/14** | **0.0065** | 0.00055 |

**The headline does not depend on the definition.** The contamination is real and small: 6 of 15
proteins, 9 positions, at most **0.031** on a per-protein ratio and 0.012 on average, and nothing
crosses 1.0. Proteins with no flanking positions move by exactly zero, which is how the isolation
is known to work.

### 🔑 The finding that was not being looked for

Redrawing the background — 20 fresh positions from a candidate list that differs only by the
flanking exclusion — moves ratios by up to **0.345**. On the 9 proteins with **no flanking positions
to remove**, where the redraw is the only thing that changes, it moves them by up to 0.345 and
**0.097 on average**:

```
P13423   0.6499 -> 0.9950     to within 0.005 of the threshold
P11140   1.0726 -> 1.3732
P02879   1.2296 -> 1.3707
```

So **the per-protein FSPE ratio carries an order of magnitude more sampling variance at 20
background positions than the documentation defect that prompted the check**, and every per-protein
ratio this project has published is a single draw. The protein-level tests survive it on this panel
because the variance does not flip signs, but `P13423` at 0.995 shows how little margin some rows
have.

⚠️ **A per-protein FSPE ratio should be read as one draw from a distribution whose width has not
been characterised.** That is now stated in § 3.1.1 of the evaluation report rather than left
implicit. The repair is to average over several draws, or to use all non-functional positions
instead of a sample; neither is done here and neither changes the protein-level result.

### The decision, and why it could be made on the merits

**The code's background is kept and the three documents are corrected to describe it.** Switching to
the documented version would not move the headline, would break comparability with every published
per-protein value, and would differ from the published arm mostly through the sampling variance
above rather than through the exclusion it is named for — importing fresh noise for no measured gain.

Because the choice costs nothing either way, it was settled on the preregistration's own stated
reason instead: section 2.1 wants FSPE-M and FSPE to "differ in the reduction only, so a difference
between them cannot be a difference in position selection", and that is served by the background the
code builds. The preregistration's prose sentence is left in place with a pointer, since the
document is append-only; the other two documents are corrected in place with the measurement named.

### Direction

Neither weakens nor strengthens a published figure: 12/14 at p = 0.0065 under every background
tested. What it removes is a claim the documents were making about how the metric works, and what it
adds is a stated limit on how precisely a single per-protein ratio can be read.

### What was deliberately not done

`results/fspe_results.json` is not regenerated — the published arm reproduces it exactly, so there
is nothing to regenerate. No background is re-sampled for publication. The variance is characterised
at one redraw rather than many, which is enough to establish that it exceeds the flanking effect and
not enough to state its width; doing that properly is an open item and is named as one.

---

## 2026-09-27 (twenty-seventh entry) — The dissociation is model-specific, not lineage-specific, and the arm that shows it also cost an undeclared dependency and a stale assertion floor

Entry twenty-four reported that the two classes ESM-2 misses on v3 are each reached by a different
model: ESM-C 600M takes beta-lactamase to 40.5%, ESM-3 1.4B takes phage to 31.7%, neither takes both.
Both are EvolutionaryScale models, so a reading was available in which **different lineages see
different hazard classes**. That reading is now tested and wrong.

### ProtT5 reaches neither

`slurm/prott5_v3.sh`, Cayuga 3399923. ProtT5 XL is a Rostlab T5 encoder trained on UniRef50 with
span corruption, so architecture, objective and corpus all differ from the ESM family at once. At 30
seeds, 95% specificity:

| arm | beta-lactamase | phage |
|---|---|---|
| canonical ESM-2 650M | 21.2% [16.5, 25.9] | 12.2% [10.0, 14.4] |
| ESM-C 600M | **40.5% [36.5, 44.5]** | 12.1% [9.4, 14.8] |
| ESM-3 1.4B | 2.9% [0.7, 5.0] | **31.7% [27.4, 36.0]** |
| **ProtT5 XL** | **2.4% [0.6, 4.2]** | **11.2% [7.3, 15.2]** |

Beta-lactamase is at the floor, 23 of 30 splits at exactly 0%, with an interval disjoint from the
canonical arm's. Phage **overlaps** the canonical arm, so it is indistinguishable there. The arm is
not broken: bacteriocin 93%, Cry 99%, clostridial and RIP 100%, T3SS 80%.

So each class is reached by a **particular model** rather than by a family or by leaving one.
Whatever ESM-C 600M has, ESM-C 300M does not (5.2%) and ProtT5 does not; whatever ESM-3 has, its own
lineage siblings do not. **"Reachable by some representation" is true of both classes and "reachable
by representations like X" is true of neither** — weaker than what three arms made available, and
more useful, because it removes lineage as the organising variable.

⚠️ **Both of ProtT5's 5-seed figures fall outside their own 30-seed intervals**: 7.1% against
[0.6, 4.2] and 6.9% against [7.3, 15.2]. Fourth and fifth instance of that in this project.

### Three defects found on the way

**An undeclared dependency that a shipped script imports.** The first submission died on
`T5Tokenizer requires the SentencePiece library`. `sentencepiece` appears in neither
`pyproject.toml` nor `requirements.txt`, yet `src/02g` cannot construct its tokenizer without it.
The v2 ProtT5 arm was built on 2026-09-05 in an environment that happened to have it. It is declared
now.

🔴 **That is the second published arm whose build environment has since vanished.** Entry
twenty-four recorded that the v2 ESM-C arrays were built under esm 3.4.0 / torch 2.11.0 and that no
environment on the cluster has that any more. Now the v2 ProtT5 arm turns out to have been built with
a library the environment no longer carries. Two arms, two environments, both gone, both discovered
by trying to extend the arm rather than by any check. **The repository has no record of what
environment produced any published array beyond a torch version string in a manifest**, and that is
an open weakness rather than a fixed one.

**An assertion floor that encoded the panel it was written on.** The audit asserted
`min_rho > 0.75` for the across-arms margin correlation, written when the weakest of five ESM-2 arms
was +0.796. ProtT5 is **+0.739** — still significant at p = 0.0041, and below the floor. The floor
would have failed for a correct reason stated wrongly, so it is replaced by pinning the weakest arm's
value directly and letting `significant == 9` carry the claim that every arm's ordering holds. Same
species as the `== 4` separation literal in entry twenty-four: a number baked in from the panel
present when the line was written.

⚠️ **And margin's class ordering is weakest on the one arm outside the lineage.** +0.739 against
+0.796 to +0.894 for the eight ESM arms. Still significant, and not averaged away: it is the mildest
available caveat on § 10.6's generality claim and it is stated in § 10.6.1 rather than left in an
artifact.

### What was not done, and why

**SaProt is still not on v3, deliberately.** `structure_3di_v3.json` carries real Foldseek strings
for the 231 v2 members and the `no_structure` mask for the 214 that v3 added, so 48.1% of that panel
would be scored sequence-only — including the entire phage class, which is the class in question.
`src/02i` now computes that fraction itself and refuses above 5%, negative-tested at 48.1% and
exiting before the model loads. The prerequisite is fetching AlphaFold structures for those 214 and
running Foldseek, which needs the Linux binary on the HPC.

That file's own `stats` field said `no_structure: 3` against a real 214, inherited from v2 when
`src/27` built it. Recomputed in place with a maintenance note recording what it said, and `02i`
computes the fraction rather than trusting the field.

---

## 2026-09-27 (twenty-eighth entry) — Criterion 1 moves from fail to partial, and three figures for "distinct names in the pool" turn out to measure two different things

`docs/DETECTOR_CRITERIA.md` has scored the split as this project's worst failure since the document
was written, and the reason it stayed a failure after `src/48` and `src/49` measured what an honest
false-positive rate looks like is that both were **analysis-time draws**. Nothing in the repository
*was* a test partition. `src/60_negative_test_partition.py` freezes one.

### The partition, and the rule

8,258 pool proteins, **test only**, admitted under three conditions: in the reviewed Swiss-Prot pool
`src/34` built; accession in neither the panel's positive nor its negative set; and maximum
normalized local-alignment similarity to any panel positive **below the maximum the panel's own
negatives reach under the same screen**, 0.282, read from the screen artifact rather than written as
a literal.

That bound is deliberately tighter than the **0.30** rule governing positives against positives.
`src/27`'s policy leaves negatives unscreened against positives so that mechanism-matched benign
proteins can serve as hard negatives, which is right for curated calibration negatives and wrong for
a test partition: a test member closer to a hazard than any calibration negative is a different kind
of object, and admitting it would make the measured rate arguable.

🟢 **Tightening it is free.** Exactly one pool protein clears 0.30 (`Q8X739`, PhoQ, 0.871 against a
virulence-associated control) and exactly one clears 0.282 — the same one. **Nothing sits between the
two bounds**, so the stricter rule costs nothing and removes the argument. That is measured, not
assumed: `src/42` now dumps every pool protein's maximum, which it had been computing all along and
writing out only for entries above 0.30, so applying any other bound used to mean re-running 1.23
million alignments.

**Test-only is the load-bearing restriction.** Criterion 18's attack works through negatives the fit
sees; a set that appears only at evaluation time cannot suppress a class's recovery, it can only
change the rate that is reported. That asymmetry is why this is the safe half of the repair, and the
artifact says so in a `role` field rather than leaving it to a reader.

### What it does not fix, stated in the scorecard rather than implied

**The threshold is still set on 118 calibration points.** The panel's own negatives are still
178/118/0. What improves is the resolution of the rate that is **measured** — 1/296 to 1/3,549 by
distinct name — not of the threshold that is **set**. And exchangeability is **voided by design**:
the panel's negatives are three curated blocks, the partition is Swiss-Prot and 87% bacterial, so
the conformal guarantee does not hold across the shift. `src/49` measured that cost rather than
assuming it away, and a panel-to-pool rate is the deployment-shift number.

So criterion 1 is now **partial, not pass**, and the scorecard row says a partial here means "a test
set exists", not "the split is fixed". Tally moves from three fails and three partials to **two fails
(3, 12) and four partials (1, 4, 5, 18)**, propagated to the criteria document, the README and the
six-page summary, with the audit's own tally strings updated.

⚠️ **An obsolete forbid had to be retired for this.** The audit forbade the string
`(two fails, four partials)` on both public surfaces, because both had once carried that count while
the table did not support it. On 2026-09-27 it became true. The forbid is removed rather than worked
around: a forbid on a count is a forbid on a count the table does not support, and it has to be
retired when the table changes.

### Three figures for one quantity, and two of them were right

While computing the partition's effective size, three different "distinct names in this pool" figures
surfaced for the same 8,259 proteins:

| figure | source | what it groups by |
|---|---|---|
| 3,550 | `src/36_pool_effective_n.py` | the lowercased protein description |
| 3,407 | `src/49_external_test_partition.py` | the gene symbol, `sp|ACC|GENE_ORGANISM` → `GENE` |
| 3,552 | the first version of `src/60` | the description, **not lowercased** — a bug |

The first two are effective sizes of **different quantities** and both are legitimate; the third was
mine and is fixed. `src/60` now reports both conventions, 3,549 by description after the one
rejection and 3,407 by gene symbol, reconciling exactly with the two existing figures, and the
artifact carries a field saying they must not be conflated. A resolution claim that rests on
"distinct names" has to say which convention it means.

### What was deliberately not done

No pool protein is promoted into the train or calibration split. That is the direction criterion 18
warns about and it would change every published recovery number, so calibration resolution stays
where it is. The partition is not used to re-threshold anything: `src/49`'s measurements already
stand on the same membership, and § 2.6.2's parallel table is where a guaranteed-threshold figure
lives.

---

## 2026-09-27 (twenty-ninth entry) — P2's four benign controls become sequences, and thermolysin's UniProt features are 13 parts calcium

The 2026-09-27 prerequisite audit found P2 blocked three ways: thermolysin, one of the four controls
it names as "already in the repository", was absent; none of the other three had a sequence, because
they entered as PDB structures for the FSI controls; and all three carried `use_pdb_numbering: true`,
so each needed an identity-verified offset before it could be masked at all.
`src/61_benign_control_annotation.py` closes all three under `src/46`'s rule — the offset must be the
**unique** integer placing **every** expected residue identity correctly — and all four pass:

| control | UniProt | offset | positions | residues | identities from |
|---|---|---:|---|---|---|
| astacin `1AST` | P07584 | **+49** | 141, 142, 145, 151 | HEHH | annotation text |
| thermolysin `1LNF` | P00800 | **+0** | 374, 375, 378, 398, 463 | HEHEH | UniProt features |
| lysozyme `1LYZ` | P00698 | **+18** | 53, 70 | ED | annotation text |
| saporin-6 `1QD2` | P20656 | **+24** | 96, 144, 200 | YYE | annotation text |

Three of the four needed a non-zero offset, which is the point of having the rule: the annotations are
in PDB numbering and the sequences are canonical UniProt, and +49, +24 and +18 are the signal or
propeptide boundaries between them. Asserting any of those from memory would have masked the wrong
positions on every control in P2.

### 🔴 The defect in the first pass, and why it mattered

Thermolysin has no annotation text to read residues out of, so its identities came from UniProt's own
single-position features. Taking every `Active site` **and** `Binding site` gave it **18 positions**,
against astacin's 4 and lysozyme's 2. Reading them showed why: **13 of the 18 are structural calcium
sites.** Thermolysin is a calcium-stabilised thermophilic protease and UniProt annotates all of its
Ca²⁺ ligands, none of which is catalytic.

P2 compares **means** across controls. One control whose functional-site set mixed catalysis with
structural metal binding, while the other three were catalysis only, would have moved that comparison
without anyone deciding it should. The rule is now: every `Active site`, plus `Binding site` only
where the ligand is a catalytic metal. Thermolysin becomes the HExxH zinc motif plus the proton
donor — H374, E375, H378, E398, H463 — five positions, comparable in kind to astacin's four. The 13
excluded sites are logged per control in the artifact with their ligand, not dropped silently.

### What this is not

`src/61` supplies an input and takes no measurement: no model is loaded and no score is computed, so
it cannot have been tuned by looking at a result. **P2 itself is still not run.** What changes is that
its inputs now exist and are verified, so when it runs, a failure will be a statement about FSPE-M
rather than about missing annotations. § 4 calls P2 "the test most likely to fail, and the one that
decides whether this axis is worth building on", which is the reason to get its inputs right before
its numbers exist rather than after.

Also still open: the "benign enzyme set matched on length and catalytic-site count" P2 asks for
**beyond** these four does not exist, so P2 will run on n = 4 controls against 14 panel proteins
unless that set is built. That is annotation work and criterion 17 again.

---

## 2026-09-27 (thirtieth entry) — The mutation axis returns a preregistered negative, and step 4's gate turned out to be a coin

Three things, in the order they had to be decided.

### 1. Section 2.1's ratio was ill-defined and was retired before the panel ran

`s(i) = log p(wt) − log mean p(other)` is a **signed** log-odds. Section 2.1 wrapped it in the ratio
form FSPE uses, and FSPE's ratio works because Shannon entropy is non-negative with a meaningful
zero. `s(i)` has neither property. A five-point smoke run showed both failure modes, and the
amendment reproduces all five values so the change cannot be read as chosen for its answer: one panel
protein had a **negative** catalytic mean against a positive background, making "> 1" a test on a
quantity whose sign is not the sign of the contrast; and two controls with differences +8.42 and
+8.07 had ratios 6.48 and 3.40, while a control whose difference was +0.028 had a ratio of 1.16
because its denominator was 0.174. **The ratio was reporting the background level, not the contrast.**

Primary statistic became the **difference**, above 0 for the hazard-consistent direction. P1's
threshold is untouched — a sign test on "difference > 0" is the same test as one on "ratio > 1"
wherever the ratio is defined, and is defined where it is not. P2's 0.15 effect-size floor was
**voided**, because it was written on the retired scale and keeping the numeral on a scale of nats
would be arbitrariness dressed as continuity.

### 2. Step 4's gate: the leakage half passes, the AUROC half has no power

The gate reported FAIL. Decomposed:

| half | value | verdict |
|---|---|---|
| shuffled mean, expectation 0 | +0.1359, sd 1.2219, n 19 → **0.48 standard errors** | passes |
| shuffled panel-vs-control AUROC, tolerance 0.5 ± 0.05 | **0.2833** | not a test |

Permuting which proteins carry the control label, 20,000 times, gives that AUROC a null standard
deviation of **0.167**. A tolerance of ±0.05 on it therefore fails **77.3%** of the time on a
perfectly clean pipeline, and the observed 0.2833 has a two-sided p of 0.224 — entirely ordinary.
Section 4 wrote ±0.05 for P5's composition AUROC over the 234-protein panel, where it is about 1.7
standard errors; the same number was applied to a 15-versus-4 AUROC without checking its standard
error at that n. **Same defect class as the floor-with-no-ceiling the preregistration exists to
avoid**: a threshold carried across a change of sample size without a power argument.

Resolution: the gate is the centred-on-zero half, which passes; the AUROC is reported without a
pass/fail. ⚠️ That is a **real weakening** — the AUROC half would have caught a leak that preserved
within-protein exchangeability while still separating panel from control, and the surviving half does
not. It becomes testable somewhere above n_benign ≈ 30, so it is logged as a limitation rather than
repaired.

### 3. The result: P1 supported, P2 failed on its own ceiling

**P1.** 13 of 15 above 0, exact sign test **p = 0.0037** against a frozen ≥ 12 of 15 at p < 0.0083;
12 of 14 at p = 0.0065 with `P01552` excluded. Catalytic positions are more constrained than
background.

🔑 Its two exceptions are **FSPE's two exceptions**: ricin A-chain (−1.53) and abrin A-chain (−0.61)
are the only panel members below zero, and they are exactly the pair whose FSPE ratio exceeds 1.0
(1.230, 1.073). Two different reductions of one tensor pick out the same two proteins.

⚠️ The matching p-values, 13/15 at 0.0037 and 12/14 at 0.0065 on both metrics, carry **no information
beyond the matching counts**: an exact sign test on 15 items with 13 successes returns 0.0037 whatever
the statistic was. The coincidence looks like corroboration and is not.

**P2.** Panel mean **+4.32** against benign controls **+5.26**, difference **−0.94**, AUROC 0.393.
Section 4: *"Not supported if the benign controls match or exceed the toxins, in which case FSPE-M
measures evolutionary constraint and must not be called a hazard metric."* They exceed it. Astacin
(+8.42) and thermolysin (+8.07) rank above **14 of the 15** panel proteins.

**So the axis measures positional constraint and is not built on further as a hazard metric.** Section
0 set out to determine whether the extension was worth making; the determination is negative, and it
was fixed in advance by the test section 4 itself called "most likely to fail, and the one that
decides whether this axis is worth building on".

This reproduces the FSI controls lesson on an independent metric: 1AST at 1.85 and 1LNF at 1.69
sitting close under 3BTA at 2.24 was the same finding about fold and chemistry, and the mutation axis
finds it again in log-odds space.

### What the negative result does not license

**n_benign = 4 and the permutation p is 0.678.** The controls are not *significantly* above the panel.
The ceiling fires on **direction**, which is the right way round: the burden was on the panel to exceed
the controls by a stated margin, and it does not exceed them at all. What cannot be claimed is that
benign enzymes are reliably higher.

**P6 is unrun and would sharpen this.** If a PSSM from a homolog alignment matches the metric, section
4's own words apply — "a multiple sequence alignment does this as well as a protein language model" —
which is the natural reading of a constraint metric and would close the question. No alignment exists.

And dFSPE-M inherits FSPE's background, whose per-protein sampling variance the twenty-sixth entry
measures at up to 0.345 on a redraw. P1 is a sign test over quantities each of which is a single draw.

### What was deliberately not done

No threshold in section 4 was adjusted to fit an outcome. P1's stands as frozen and is met; P2's
effect-size floor was voided **before** the panel ran, on the grounds that its scale no longer exists,
with the five smoke values that forced the change disclosed in the amendment log. The ratio is still
reported per protein wherever both means are positive, so the frozen statistic stays visible.

---

## 2026-09-27 (thirty-first entry) — SaProt is the tenth arm, and the class it was added to settle has no structures at all

`slurm/saprot_v3.sh`. SaProt reads an amino acid and a structure token at every position, so it was
the one arm that could not simply be embedded: `structure_3di_v3.json` carried real Foldseek strings
for the 231 v2 members and the `no_structure` mask for the 214 v3 added, because `src/27` built the
file by inheriting v2's entries and foldseek ships as a Linux binary. Fetching AlphaFold models and
running Foldseek on the cluster raised coverage from **231 of 445 to 365 of 445**.

### The fact that closes the question

**AlphaFold DB has no model for any of the 32 phage peptidoglycan hydrolases.** The 80 entries still
masked are 45 negatives, one member each of three toxin classes, and the whole phage class. So the
class § 10.6.2 was about is **unanswerable for a structure-aware model by construction**.

🔴 **An aggregate masked fraction could not have caught that.** 18% masked overall reads as tolerable
and hides a class at 100%. `src/02i` now computes coverage **per class** into its manifest and names
any class below 80% as `unreportable` for the arm, with the note that a recovery figure for it would
be a statement about AlphaFold coverage rather than about the model. The 5% aggregate guard was right
to refuse and wrong to be the only check.

⚠️ **The transferable limit.** "Add more representations" produced § 10.6.2's dissociation and has a
boundary: a representation requiring an input the panel cannot supply for a class cannot be evaluated
on that class. **An arm count is not a coverage count**, and anything conditioned on annotation,
localisation or an external database hits the same wall somewhere else.

### What the measurable half says

Beta-lactamase has 14 of 14 real structures. At 30 seeds SaProt gives **12.4% [7.8, 17.0]**, which
overlaps the canonical arm's [16.5, 25.9] by half a point — **the first arm to overlap the canonical
one on this class** — and sits far below ESM-C 600M's [36.5, 44.5]. Across **ten** arms ESM-C 600M is
still the only one that reaches beta-lactamase. SaProt is not a weak arm: RIP, superantigen,
ADP-ribosyl and Cry all at 100%, bacteriocin 93%.

### The margin result that has to be discounted

`saprot_650M` returns the **highest** margin-against-recovery correlation of any arm, **+0.954** at
p = 0.0000, and is one of only three arms whose bottom-two margin classes are exactly the two
failures. Both are artefacts of the same gap: its phage margin is **−0.0368**, the most negative of
all ten and an order of magnitude below most, because those 32 proteins are the only ones it sees
sequence-only while every other protein carries structure. A class represented differently from every
other class sits at the edge of the embedding for that reason alone.

So § 10.6.1's bottom-two count is now **two of ten on interpretable arms plus one agreeing for a
structural reason**, and the audit pins all three with the exception named, so the discount cannot
drift into a confirmation. Two derived assertions moved with it and are recorded rather than adjusted
quietly: `beta_lowest_everywhere` is now false — SaProt's lowest-margin class is phage — and the
canonical arm gained its **first** overlap with another arm on beta-lactamase, which is SaProt's.

### 🟢 One thing the run settled cleanly

Rebuilding v3's 3Di recomputed the 231 strings inherited from v2, and **all 231 reproduced exactly**.
AlphaFold DB v6 and Foldseek have not moved under the published values. That check was added to
`src/02h` for this run and it is why the launcher stops before embedding if any string changes: a
changed string would be a question about every 3Di number in the repository, not about one arm.

### What was deliberately not done

`--allow-masked 0.20` is in the launcher with the reason written beside it, not as a way past the
guard: v3 cannot go below 18% by fetching harder. No phage figure is reported for this arm. The v2
SaProt arm is untouched.

---

## 2026-09-27 (thirty-second entry) — The margin result had no baseline simpler than itself, and reading prior work supplied one that is a real predictor on the published arm

Criterion 14 of `docs/DETECTOR_CRITERIA.md` is "report the boring baselines, including the ones that
make the model look worse". § 9 of `docs/MECHANISM_GENERALIZATION.md` applies it to the **recovery**
figures — composition 0.754, shuffled labels 0.506, lab-strain provenance 0.818. **It was never applied
to the margin result**, which § 10.4 compares to its own parts and to class size and to nothing else.

### Where the baseline came from

Not from introspection. **"Viral Proteins Reveal Geometry of Protein Language Models"**
([arXiv 2606.12609](https://arxiv.org/abs/2606.12609), ICML 2026 workshops) reports a *dominant
nativeness axis* in PLM embedding space aligned with masked reconstruction perplexity. If the geometry
is organised by typicality, then "the failing classes sit close to benign" could be "the failing classes
sit close to protein space in general", and margin would be reading that axis.

⚠️ **The paper postdates the April survey and the 2026-09-18 review pass did not catch it.** It surfaced
only from a search aimed at the findings produced *after* that pass. § 10.6.2, § 10.6.3 and § 2.6.2
currently carry zero citations between them, so this is one instance of a general gap rather than a
one-off.

### What the baseline does

`src/66_typicality_baseline.py`. `typicality(class)` = mean cosine of its members to the mean of the
8,259-protein benign pool. **No hazard label, no class label, no fit, one line.**

| arm | margin ↔ recovery | typicality ↔ recovery | margin, typicality held | typicality, margin held | typicality perm *p* |
|---|---:|---:|---:|---:|---:|
| canonical 650M | +0.894 | **−0.746** | **+0.776** | −0.346 | **0.0034** |
| esm2_35M | +0.796 | +0.021 | **+0.798** | +0.109 | 0.529 |

🔴 **On the arm every published figure uses, the label-free baseline reaches −0.746 at p = 0.0034.** The
sign reads directly: the more typical of general protein space a class is, the less of it is recovered.
Its top three by typicality are beta-lactamase, the labelled virulence control and the phage class, and
its bottom is the four classes recovered at 90 to 100%. That belonged in § 9's baseline list and was not
there.

### 🟢 Margin survives it, and the baseline does not replicate

Controlling for typicality, margin keeps **+0.776** of its +0.894 on the canonical arm and **+0.798** of
its +0.796 on the second — unchanged either way. Controlling for margin, typicality falls to −0.346 and
+0.109.

And the decisive difference runs the project's way: typicality is **+0.021 at p = 0.53 on `esm2_35M`**,
null and sign-flipped, while margin holds at +0.796. **Typicality is an arm-specific confound, not an
explanation.** A paper reporting only the canonical arm would have had a serious problem here, and
§ 10.6's rule of running every geometric claim across representations is what makes it answerable.

### What is still open

The proxy is cosine-to-centroid on embeddings; the paper's axis is **perplexity-aligned**. So the
crudest version of the concern is ruled out and the paper's actual construct is not. And the same
question applies to **FSPE**, which is itself a masked-prediction entropy metric measuring a
within-protein entropy contrast — whether that contrast is partly a typicality contrast has not been
asked. Their code is public, so this is available work rather than a standing caveat.

### A correction inside the correction

The first version of `src/66` ranked without tie-averaging and returned **+0.902** for margin against
§ 10.6.1's published **+0.894**. Three classes sit at exactly 100% recovery on the canonical arm, so the
ties are real and `src/30`'s tie-averaged ranks are correct. Fixed before anything was written down, and
margin now reproduces +0.894 and +0.796 exactly. **A baseline that disagrees with the number it is a
baseline for, for a reason unrelated to the baseline, is worse than no baseline** — and the disagreement
was in the flattering direction, which is the kind that survives.

### What was deliberately not done

No published figure changes: recovery, margin and every interval are untouched. The typicality baseline
is reported alongside margin rather than replacing any comparison, and § 9's baseline list is extended
rather than rewritten. Reproducing the paper's perplexity-aligned axis, and asking the same question of
FSPE, are named and not attempted.

---

## 2026-09-27 (thirty-third entry) — A benchmark attributed to the wrong paper, and three papers the survey did not have

Prompted by an instruction not to judge prior work from keywords or summaries. Acting on it meant reading
the primary sources for the external numbers this project compares itself against, and the first one
checked was wrong.

### 🔴 The correction

§ 11 of `docs/MECHANISM_GENERALIZATION.md` said, for as long as the sentence existed:

> DTVF (ProtT5 + LSTM/CNN) reports AUROC 0.92 on the standard 576/576 virulence benchmark.

| | what the paper actually says |
|---|---|
| **AUROC 0.92** | ✅ DTVF reports **0.9208** ([Genes 15(9) 1170, 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11430887/)) |
| **576/576** | ❌ **DeepVF's** partition ([BiB 22(3) bbaa125, 2021](https://academic.oup.com/bib/article/22/3/bbaa125/5864586)): "576 VFs and 576 non-VFs were randomly selected as the independent test dataset" |
| **the join** | DTVF reuses DeepVF's 3,576 / 4,910 pool and cites it as ref [9], but reports on "an independent test set" and **never states its split** |

So the sentence was an inference presented as a citation. It is a reasonable inference — same pool, same
reference — and that is exactly why it survived: **a plausible join between two true facts reads like a
fact.** § 11 now names DeepVF as the partition's author, gives its own AUC 0.896, gives DTVF's 0.9208, and
says in the text that the split is an inference.

⚠️ **Nothing in this repository could have caught this.** Every one of the other 80 claims recomputes a
number from an artifact; an external citation has no artifact, so the audit gate had no purchase on it.
Claim 81 (`external_baseline_numbers`) now pins each figure to the paper it was read from and forbids the
old sentence in both spellings. It still cannot verify a citation against its source — only a reader can —
but it can stop the three papers' numbers from re-merging.

### 🔑 And the comparison got harder, which is the right direction

**DeepVIC** ([Bioinformatics Advances 6(1) vbag237, 2026](https://academic.oup.com/bioinformaticsadvances/article/6/1/vbag237/8762933))
reports **AUROC 0.954** on a 13,384-sequence holdout from **33,456 VFs**. § 11's "not a better classifier"
disclaimer was pointing at a four-year-old number when a larger and better one existed, so the disclaimer
was true but weak. It now cites all three.

### What reading the sources turned up that the survey did not have

Three entries added to `research/05_v2_related_work_survey.md`, § 2.4 and § 5.0:

- 🔑 **DeepVIC carries an external class axis: 14 VFDB categories over 12,989 annotated VFs**, against this
  project's eleven-to-twelve hand-curated classes — the binding limitation § 1.1b-ter names. And it runs
  **no leave-one-category-out evaluation**; its generalization check is recall on two organism-specific
  positive-only sets (71.4%, 45/63). The largest virulence-factor classifier published has the class axis
  this project needs and does not ask the class-level question. Its repository is MIT inference code with
  **no data, labels or weights**, so the axis comes from VFDB directly (`setA_pro` 1.3 MB gzipped,
  `VFs.xls` for the categories, last updated 2026-09-25, no license stated).
- **VFUSE** ([arXiv 2606.10080](https://arxiv.org/html/2606.10080), June 2026), and its earlier form
  **SAEBER** (Apart Research hackathon, April 2026, same author): Matryoshka BatchTopK SAEs on
  RFDiffusion3 / RoseTTAFold3 activations to audit a protein *design* model for hazard features. AUROC
  **0.877 ± 0.025** random split, **0.817 ± 0.102** homology-clustered. This is the nearest neighbour in
  intent found so far — and its control is member-level homology clustering, not a held-out mechanism
  class, on n = 275 pairs. The class-level question stays unoccupied.
- ⚠️ **VFUSE draws its positives from SafeProtein with no route to them.** No data-availability statement,
  `github.com/jigang-fan/SafeProtein` still 404. Three incompatible n now exist for one unreleased
  dataset: the paper's **429**, VFUSE's **275 pairs**, and the **66** identities this project recovered
  from the paper's text.

### The part that generalises past this entry

§ 10.6.2, § 10.6.3 and § 2.6.2 still carry **zero** citations between them, and
`docs/DETECTOR_EVALUATION_SUMMARY.md` carries none at all. Entry thirty-two found a paper that produced a
real test of a headline result; this entry found a misattributed benchmark and a class axis that answers
the project's own stated limitation. Both came from the same activity, and neither came from introspection.
**Reading the literature has been the highest-yield thing done in the last two days, and the survey is
still not on the audited surface.**

---

## 2026-09-27 (thirty-fourth entry) — Two questions about what the hazard label means, and a stale answer to the second one

Asked directly: virulence and toxicity are not the same thing, and acting on a human is not the same
as acting on another species — are those distinguished, and does this project cover them? Both are
measurable rather than arguable, and answering them produced one correction to something written
minutes earlier in this session.

### 🔴 The correction: I put the species axis at n = 7, from the wrong panel

The first draft of `docs/VFDB_CLASS_AXIS_DESIGN.md` § 4 read the **v2** annotation —
`plant 1, bacteria 5, other_nonanimal 1` — and concluded that "the species axis rests on one
plant-target protein and five bacteria-target ones", then reasoned from there that § 2.4.1's
always-say-animal null was probably an n artifact and that rerunning it on v3 was available work.

**`data/annotations/target_host_v3.json` has 149 positives: bacteria 52, animal 51, insect 22, small
molecule 15, plant 1.** And the holdout has already been run on it, and is already in
`docs/MECHANISM_GENERALIZATION.md` at § 2.5:

| | v2, 9 classes | v3, 11 classes |
|---|---|---|
| balanced class-level accuracy | 0.500, the majority baseline | **0.804** (animal 0.86, non-animal 0.75) |
| class-level permutation *p* | 0.21 | **0.0150** |
| null draws reaching a perfect score | 8% | **0%** |

By the preregistered rule: **SUPPORTED**. So the work I proposed was done, the diagnosis I
independently arrived at was the one already recorded, and the n I quoted was three panels out of
date. ⚠️ **The pattern is the one this log keeps recording: a number read from the nearest file rather
than the current one.** `target_host_v2.json` and `target_host_v3.json` sit in the same directory, and
the v2 file is the one every `src/03*` script names in its docstring.

§ 4 is rewritten. `docs/VFDB_CLASS_AXIS_DESIGN.md` is now on the audited surface, which is the check
that would have caught it.

### The answers, from the artifacts

**Virulence ≠ toxicity, and the panel already treats it as a label rather than an assumption.**
`virulence_associated_non_toxin` is a labelled control class: v2 50%@95 / 32%@99 / AUROC 0.844, v3
**34% / 12% / 0.820**, against 100%@95 for four toxin classes. A probe trained on toxins does not
transfer to virulence factors that are not toxins. n = 10.

🔑 **VFDB's own ontology puts a number on how large that distinction is.** `src/67_vfdb_ingest.py`,
downloaded 2026-09-27: **Exotoxin is 248 of 4,755 verified records (5.2%) and 1,218 of 30,215 full
records (4.0%)** — one category of fourteen. The largest is Effector delivery system at 8,804. So on a
VFDB-derived axis the non-toxin control becomes 95% of the data, which is a construct change and not a
scale-up.

**Target host is annotated at species level, survives class holdout on v3, and does not predict
which class fails.** Recovery by target: insect Cry **87.3%**, bacteriocin 84.0%, CDI 75.0%, phage
peptidoglycan hydrolase **10.0%**, beta-lactamase **18.6%**, animal classes 80 to 100%.

🔑 **Both documented failures act on something other than an animal, and acting on something other
than an animal does not predict failure.** An insect-targeting Cry toxin is as detectable as a
mammalian neurotoxin; a phage lysin is not. Target host is not the axis the failures lie on — margin
is (§ 10.4). This is a sharper statement than § 2.4's 0.994-animal versus 0.898-non-animal, whose
non-animal side is 15 small-molecule and 4 regulatory proteins out of 22.

⚠️ **The one host with no support is plant, at n = 1** — the *Agrobacterium* T-pilus subunit, sitting
inside the mixed virulence control rather than in a class of its own. setB has **1,093**. So the gap
is plant specifically, not "other species" generally.

### What the download establishes for the next phase

| | records | categories ≥ 20 | Exotoxin | plant | insect |
|---|---:|---:|---:|---:|---:|
| **setA**, verified | 4,755 | 12 | 248 | **0** | **0** |
| **setB**, full | 30,215 | **14** | 1,218 | **1,093** | **511** |

🔴 **The host contrast is a setB-only property**: not one plant or insect pathogen appears in the
4,755 experimentally verified records. Asking the species question means accepting predicted VFs.

🔑 **And the axes cross.** Exotoxin by host in setB: **759 mammal, 181 insect, 61 plant** — Cry
toxins, *Photorhabdus* and *Xenorhabdus* toxin complexes, phytotoxins. That is a matched
toxin-versus-toxin, host-versus-host contrast, which neither this panel nor any paper in the survey
runs, and it is the design the two questions above point at.

⚠️ Two limits recorded before any probe is trained. VFDB's organism field names the **producing
pathogen, not the target**, and § 2.4 already records that these come apart — six of seven RIPs are
plant-produced and act on animal ribosomes. And **genus is the wrong resolution**: *Pseudomonas* holds
*aeruginosa* (935, human) beside *syringae* (681, plant) and *entomophila* (139, insect), *Bacillus*
holds *anthracis* beside *thuringiensis*. `src/67` assigns at species level from an explicit table and
leaves 5,635 setB records **unassigned** rather than defaulting them to the majority, because
defaulting to "animal" is the exact failure § 2.4.1 found in the probe.

Raw downloads are not committed: no license is stated on the download page, the full protein set is
19 MB, and `--download` reproduces both. Two parser failures worth keeping: one setB header
(`VFG042213`, a *Mesorhizobium loti* nodulation protein — VFDB contains symbiosis factors, not only
virulence) carries no accession block and tripped the deliberate hard failure on unparsed headers; and
the non-greedy category group kept its trailing space, so `"Exotoxin " != "Exotoxin"` reported **0
exotoxins out of 248** while still counting fourteen categories.

---

## 2026-09-27 (thirty-fifth entry) — "VFDB fixes n_benign = 4" was two fixes wearing one label

§ 5 of `docs/VFDB_CLASS_AXIS_DESIGN.md`, written earlier the same day, offered Option A as a single
move: hazard stays "toxin", VFDB's 4,507 non-toxin virulence factors become the negatives, and that
*"fixes the problem named earlier in this project as the binding one — **n_benign = 4**"*.

🔴 **It does not.** `n_benign = 4` is the count in `data/annotations/benign_control_sites.json`, and
what those four records carry is **per-residue catalytic positions** — 1AST, 1LNF, 1LYZ, 1QD2, each
verified by `src/46`'s unique-offset rule, because dFSPE-M averages a per-residue statistic over the
catalytic set and needs to know which residues those are. **VFDB carries no residue annotations at
all.** Its records are sequences with a category, an organism and a VF id.

Two separate fixes were sitting under one label:

| | what it needs | source | what it repairs |
|---|---|---|---|
| **A1** | 4,507 hard negatives | VFDB setA, already downloaded | **criterion 1**, the worst-scoring one: 0 test negatives, 7.87% out-of-sample FPR at a nominal 5% |
| **A2** | ≥ 30 benign enzymes **with `Active site` features** | Swiss-Prot, already local at `.external/db/uniprot_sprot.fasta` | **n_benign = 4**, which makes three of the mutation preregistration's AUROC halves untestable |

⚠️ **The tell was available and I did not read it.** The mutation preregistration says the threshold
*"becomes meaningful somewhere above n_benign of roughly 30"* and calls for a *"matched benign **enzyme**
set"* — enzyme, not negative. A virulence factor is not an enzyme with a known active site, and the two
words were treated as interchangeable because both mean "the not-hazardous side".

Recommendation order changed from **A → B** to **A2 → A1 → B**: A2 is cheapest, needs no new data, and
unblocks three frozen tests; A1 uses the download already made.

### And both are now preregistered to fail

`docs/NEGATIVE_EXPANSION_PREREGISTRATION.md`, frozen before either runs, six primary tests at
α = 0.05/6 = 0.0083, floors and ceilings on each.

🔴 **A2-1 predicts the benign controls will match or exceed the panel**, because the mutation axis
measures positional constraint rather than hazard — P2 failed its own ceiling and P6's alignment
reproduces both verdicts. At n = 4 the benign mean is **+5.26** against the panel, astacin **+8.42**,
permutation *p* = 0.678; at n ≥ 30 that stops being underpowered and becomes a verdict.

🔴 **A1-2 and A1-3 predict this project's own false-positive numbers get worse.** VFDB negatives are
pathogen-produced and largely secreted, which are the two features the provenance control already reads
at AUROC 0.818, so the honest prediction is a rate **above** the benign pool's 7.87% and 5.98%.

🔴 **A1-4 predicts that a bigger negative set makes the two known failures worse.** § 10.7 found that
*removing* benign neighbours repairs 8 to 11% of the beta-lactamase failure; adding 4,507
virulence-associated neighbours should move it the other way. **If recovery rises instead, margin is not
the mechanism it is written up as** — and that outcome would be the more interesting one.

⚠️ If A2's exclusions yield fewer than 30 candidates, the AUROC halves **stay untestable and that is
the reported result**. The exclusions do not get relaxed to reach the number.

---

## 2026-09-27 (thirty-sixth entry) — A2 built: 4 controls to 60, and the architecture question answered from the sweeps that already exist

### 🟢 A2 delivered, and every exclusion held

`src/68_benign_enzyme_set.py`, against the floor frozen hours earlier in
`docs/NEGATIVE_EXPANSION_PREREGISTRATION.md`. **60 controls admitted, floor 30**, from 66,275 reviewed
Swiss-Prot entries carrying an `Active site` in the panel's own length window [286, 1147]. That takes
`n_benign` from **4 to 60** and makes the mutation preregistration's three untestable AUROC halves
testable.

What bound and what did not: fewer than three Active sites rejected **332**, exact VFDB setB sequence
match **2**, hazard term in name or keywords **1**. Panel membership and the similarity bound rejected
**zero** — the highest similarity any of the 60 reaches against a v3 positive is **0.0495**, against a
bound of 0.282. **No exclusion was loosened to reach the number**, and claim 83 now pins those counts so
a later edit cannot loosen one quietly.

⚠️ **Selection order matters and was chosen before looking.** `sha256(accession)` ascending, a
reproducible draw from the whole pool, not the head of accession order — `P0xxxx` entries are older and
better characterised, so taking the first N would have been a selection choice wearing an ordering's
clothes.

🔴 **Three properties flagged in amendment 2 before anything is measured**, because noticing them
afterwards is indistinguishable from explaining a result: the set is **hydrolase-heavy (26 of 60 are
EC 3)**, which is the right direction since both unflaggable classes are hydrolases but means the benign
side is enriched for the panel's own failure family; **`P0CK11` is a 1,043-residue Turnip mosaic virus
polyprotein**, the only viral member and the longest, whose Active sites sit in a protease domain inside
a multi-domain precursor; and a few controls come from *Salmonella*, *Vibrio vulnificus* and
*P. aeruginosa*, **kept deliberately** because § 2.3's provenance probe reaches 0.818 from lab-strain
provenance alone, so pathogen-provenance benign enzymes make the control stronger against that confound.
**No pathogen-origin or domain-architecture exclusion was added**, both having been considered only
after seeing the set.

### ⚠️ Amendment 1: an exclusion that could not be applied as written

A2's exclusion 2 said "any accession appearing in VFDB setA or setB". **VFDB has no UniProt
accessions** — `VFG037176(gb|WP_001081735)` — so there is nothing to join on. Applied as an exact
sequence match against setB's 30,215 records instead, stricter in one direction and weaker in another,
and recorded as an amendment rather than reinterpreted silently. It caught O33407 and B0VMS2, so the
rule was not decorative. **The preregistration was written the same day and still contained a rule that
could not be executed**, which is the argument for running the build before treating a frozen document
as finished.

### The architecture question, answered from what is already here

Asked whether a mixture of experts is an option and whether classifier architecture has been explored
deeply. New § 8.2 of `docs/MECHANISM_GENERALIZATION.md`.

**It has been swept at two levels and neither is where the failure lives.** Four heads × 14 arms × 30
seeds (§ 9.1): logistic is the worst head on 7 of 14 arms and beaten on 12 of 14 by a median of 5.1
points, **but it moves only threshold-adjacent members** — the four saturated classes are unmoved to the
decimal and beta-lactamase reaches **21%** with the best of four. Ensembles twice (§ 8, § 8.1):
alignment + embedding gives 73.1% against the probe's 73.1%, the OR variant busts the budget at 6.5%
FPR, and fusing two language models costs 0.4 points.

🔑 **But § 8 tested *soft* combination, and its mechanism does not carry to a hard gate.** The reason
unions fail there is that **OR-ing raises the negatives' scores too, pushing the calibrated threshold
up**. A mixture of experts with a hard gate routes each sequence to exactly one expert, so no negative
is scored twice and the threshold is not inflated. **That architecture is untested here and § 8 is not
evidence against it** — it is the one combination the existing negative results leave standing, and the
complementarity is measured at Spearman **−0.25** between embedding and alignment recovery across
classes.

🔴 **The gate is the blocker, and this project has already failed at it.** A router needs a per-**sequence**
decision, and § 10.3 is the record of the **member**-level margin claim being preregistered, frozen and
**failing on two external panels**. The version that works is **class**-level (+0.894, transfers to
mechanisms it was not fit on). So routing works **if the family is known**, and for a novel sequence the
family is what is unknown.

⚠️ Which makes the gate a separate prediction problem the field has built: **DeepVIC classifies into 14
VFDB categories at 0.838 accuracy.** A category router feeding per-family experts is a concrete design,
downstream of study B rather than available now, and its honest error budget is the router's **16%
misroute rate compounded with each expert's own miss rate**. That compounding is the first thing to
measure and has not been.

---

## 2026-09-27 (thirty-seventh entry) — A refutation that covers less than the heading it licensed: no architecture here has seen more than one vector per protein

Asked about CNNs and LSTMs. Answering it required noticing that § 9's **"Pooling is not the fix"** is a
broader claim than the experiment under it supports, and that the overreach is why an entire class of
architecture has never been tried.

### 🔴 What the pooling experiment actually refuted

`src/02f_pooling_variants.py` asked the right question — its docstring is literally *"does mean pooling
throw away the class it cannot catch?"* — and tested three reductions on beta-lactamase, 30 seeds:

| reduction | recovery at 95% specificity |
|---|---|
| mean, the incumbent | **15.7%** [11.2, 20.2] |
| CLS | **16.4%** [10.6, 22.2] |
| **max, per dimension** | **1.9%** [0.6, 3.2] |

Its stated disjunction: *"If none of the three helps, mean pooling is exonerated and the information is
genuinely absent from the residue stack the probe sees."*

⚠️ **The second branch does not follow.** Per-dimension max takes an independent maximum in each of 1,280
dimensions, so dimension 5's value can come from residue 12 and dimension 6's from residue 300. **It does
not preserve a local signal — it destroys residue coherence**, which is why it landed worst of the three
rather than best. Mean and CLS are both whole-protein summaries. So three incoherent or global reductions
were refuted, and the residue stack was not tested at all.

🔴 **And the consequence is structural, not verbal: no architecture in this project has ever seen more
than one vector per protein.** § 9.1's four heads — logistic, SVM-RBF, random forest, k-NN — all consume
the pooled vector. A CNN or LSTM is not a fifth head; it consumes the `L × d` stack that every number here
discards before the classifier starts. The overreach made that look already-answered.

### 🔑 Three things in this repository say the residue-level signal is present

FSPE is a per-residue masked-entropy measurement and it resolves catalytic sites. **Smith-Waterman — an
inherently positional, local method — is the one baseline that beats the probe on beta-lactamase, 30%
against 21%**, while `phmmer` reaches 5% and `jackhmmer` 0%, so the advantage is specific to graded local
similarity rather than to homology search in general. And beta-lactamases are defined by a conserved
active-site motif in a conserved fold, which is the shape of signal a whole-protein average is worst at
keeping.

### The order this implies, which is not "start with a CNN"

New § 9.1.1. **Residue-coherent pooling first** — top-*k* mean over residues ranked by a within-fold
learned direction, and single-query attention pooling — because it needs one embedding pass that keeps
`L × d` plus the existing probe, and it is decisive either way: if beta-lactamase does not move under a
coherent reduction, a CNN is very unlikely to move it. **Then a 1D CNN** over a projected stack, which is
literally the motif detector the Smith-Waterman result points at. **LSTM last and least motivated** —
nothing here suggests long-range order carries the discrimination.

🔴 **All three are gated on n.** The panel is 149 positives / 296 negatives; DTVF trains on DeepVF's 3,000
VFs and 4,334 non-VFs, DeepVIC on 53,528 sequences. A convolutional net over a 1,280-dim residue stack
fitted on 149 positives is a different regime, not a smaller version of theirs, and the honest prior is
memorisation. **Studies A1 and B are prerequisites for the CNN, not alternatives to it.**

⚠️ Two constraints recorded with it. § 8's mechanism — under a fixed false-positive budget, capacity that
raises the **negatives'** scores costs threshold headroom — so recovery gains are not gains unless the
calibrated FPR holds. And § 2.3's provenance probe already reaches AUROC 0.818 from lab-strain provenance
alone: **local features make that confound easier to exploit, not harder**, because organism-specific
sequence idiosyncrasy is exactly what a motif detector latches onto.

⚠️ One stale number found while reading: `src/02f`'s docstring says Smith-Waterman beats the probe "31%
against 21%". The audited figure is 30%, 29.5% at 30 seeds. Both the overreach and the number are recorded
as an appended note in that file rather than edited into it, because the run was launched on the text as
written.

---

## 2026-09-27 (thirty-eighth entry) — A2 run: the predicted failure, harder at power, and a gate that can never be a test

### 🔴 P2 at n_benign = 60: the prediction held and the gap widened

`src/62_fspe_m.py --controls a2` on the 60-enzyme set built hours earlier. **The panel's dFSPE-M values
are bit-identical to the frozen-four run**, so the control set is the only thing that changed.

| | n_benign = 4 | n_benign = 60 |
|---|---:|---:|
| panel mean | +4.32 | +4.32 |
| benign mean | +5.26 | **+6.76** |
| difference | −0.94 | **−2.44** |
| AUROC | 0.393 | **0.265**, 95% CI [0.102, 0.445] |
| one-sided *p*, panel > benign | 0.678 | **0.9991** |

`docs/NEGATIVE_EXPANSION_PREREGISTRATION.md` predicted A2-1 would fail. **It failed harder than at
n = 4.** The fifth amendment's standing caveat — *"the controls are not significantly above the panel,
and the ceiling fires on direction rather than significance"* — is closed: the controls exceed the panel
at *p* = **0.0009**. **All 60 controls sit above the panel's worst protein and 59 of 60 above zero.**
A2-2 asked for AUROC ≥ 0.70 and got 0.265 with an interval that **excludes 0.50 on the wrong side**:
benign enzymes are not merely unseparated from toxins, they separate *from* them in the direction
opposite to the hazard hypothesis. A2-3, the leak check, is 0.430, inside its [0.40, 0.60] window.

🔑 **So dFSPE-M measures catalytic-site constraint, and ordinary benign enzymes have more of it than
toxins do.** Astacin and thermolysin were not unlucky picks; they were representative. The axis is not
built on further as a hazard metric, which is what section 4 of the mutation preregistration said this
outcome would mean.

⚠️ **The pre-flagged outlier behaved as flagged and in the conservative direction.** `P0CK11`, the viral
polyprotein amendment 2 named before the run, is the only control below zero at **−0.493**, and also the
only one whose sequence is truncated (1,043 against `MAX_SEQ_LEN` 1,022, with every catalytic position
inside the kept region). Keeping it **lowers** the benign mean, so it works against this result. Kept,
as amendment 2 said it would be.

### 🔴 And the same defect twice, the second time inside the amendment that named it

The fifth amendment of the mutation preregistration retired the P5 gate's AUROC half at n_benign = 4,
diagnosing it exactly: *"A ±0.05 tolerance on a statistic whose null standard deviation is 0.167 is not
a threshold, it is a coin weighted against passing"*, and naming the defect class — **"a threshold
carried across a change of sample size without a power argument."** It then predicted the repair: the
tolerance *"becomes meaningful somewhere above n_benign of roughly 30"*.

`src/71_a2_gate_power.py` reproduces that amendment's null **exactly** at n = 4 — sd 0.1669, clean
pipeline fails 77.3%, two-sided *p* 0.224 — so the construction is theirs, not a new one. Then it sweeps
n_benign with the panel fixed:

| n_benign | 4 | 10 | 30 | 60 | 120 | 500 | 5,000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| null sd | 0.170 | 0.119 | 0.094 | 0.088 | 0.083 | 0.079 | **0.077** |
| a clean pipeline passes ±0.05 | 20% | 31% | 39% | 43% | 45% | 48% | **47%** |

🔴 **The prediction is wrong, and it is wrong by the defect it had just named.** An AUROC's precision is
set by the **smaller** group. `functional_sites.json` has 16 entries, 15 with catalytic residues, and
that is the whole panel — it cannot grow. So the null sd floors near **0.077** and a ±0.05 tolerance
never passes more than about half the time however many controls are added. The amendment attributed its
power problem to the sample size that was about to grow, having just written down that thresholds must
not be carried across sample sizes without a power argument.

**Consequence**: the fifth amendment's resolution — report the shuffled AUROC without a pass/fail —
stands **permanently**, not provisionally. The observed 0.4300 at n = 60 is two-sided *p* = 0.4059 under
its own null, which says there is no leak evidence and says nothing more. ⚠️ And A2-3's own [0.40, 0.60]
window is ±0.10, about 1.2 null standard deviations, which a clean pipeline clears roughly 77% of the
time: **better than the gate it replaces and still not a strong test.** Written down rather than left
for a reader to derive.

### What this closes and what it costs

🟢 **Closed**: `n_benign = 4` as a limitation, listed in the mutation preregistration, in
`docs/EVALUATION_REPORT.md` § P2 and in `docs/DETECTOR_EVALUATION_SUMMARY.md`. All three now carry the
n = 60 numbers.

🔴 **Not closed, and now known to be uncloseable on this design**: the AUROC half of the P5 gate. The
route to a real leak test of that kind is more *panel* proteins with catalytic annotations, not more
controls — and the panel is 15 because 15 is how many the annotation set has.

---

## 2026-09-27 (thirty-ninth entry) — The residue stack tested at last: 13 of 14 reductions worse, and the one that works reallocates rather than adds

Entry thirty-seven established that § 9's "Pooling is not the fix" rested on three reductions, two of
them whole-protein summaries and one that destroys residue coherence, and that no architecture here had
ever seen more than one vector per protein. This ran the test that was missing.

### The negative nine tenths

`src/69` wrote the `L × d` stack behind a per-protein reproduction gate; `src/70` wrote fourteen
**label-free** coherent reductions as tagged artifacts `src/03b` evaluates **unchanged**; `src/72`
re-ran the winner at 30 seeds through `src/03x`'s imported fold logic. Control `mean_res` reproduces the
published mean to 3.9e-04 and its beta-lactamase recovery to the decimal.

🔴 **Twelve of fourteen are worse on the point estimate and most are far worse**: `dev_topk10` and
`dev_topk50` recover the class at **0.0% on all 30 seeds**, `dev_attn1` 1.4%, `win_best5` 3.8%. Under a
null where the reductions were equivalent about half the grid would beat the control. **Concentrating on
a few residues destroys the signal mean pooling captures** — so the hypothesis entry thirty-seven put
forward, that the signal is a local motif an average washes out, is wrong for thirteen of the fourteen
ways of being local that were tried.

### 🔑 And right for the fourteenth, which is not noise

`win_best25`, the 25-residue window whose mean deviates furthest from the protein's own mean, at 30 seeds:

| flagged@95 | `mean_res` | `win_best25` | Δ | intervals |
|---|---|---|---:|---|
| **beta_lactamase** | 15.5% [11.0, 19.9] | **35.0% [31.6, 38.4]** | **+19.5** | disjoint |
| **contact_dependent_inhibition** | 36.7% [29.3, 44.0] | **55.8% [50.7, 60.9]** | **+19.2** | disjoint |
| superantigen_enterotoxin | 100.0% | **56.2% [50.2, 62.2]** | **−43.8** | disjoint |
| pore_forming_cytolysin | 69.0% [64.2, 73.9] | 39.0% [36.1, 42.0] | −30.0 | disjoint |
| rip_rrna_glycosidase | 100.0% | 89.0% [84.9, 93.2] | −10.9 | disjoint |

🔑 **[31.6, 38.4] is the first pooling choice in this project whose whole interval clears alignment's
29.5% on beta-lactamase**, with 0 of 30 seeds at zero against the control's 7. The class that resisted
fourteen representations, four classifier heads, five scales, structure and lineage moves **+19.5 points
from a change of reduction on the canonical arm**.

🟢 Two confounds checked and neither explains it. Realised FPR at a nominal 5% is **0.0656 for both**, so
§ 8's fixed-budget objection is answered rather than dodged. And § 9.1.1 predicted local features would
make § 2.3's provenance confound *easier*; it goes the other way, **0.8150 → 0.7766**.

🔴 **But it takes as much as it gives, and § 9's heading survives on that** rather than on its original
evidence. Panel mean recovery falls **72.8% → 63.3%**. The two classes that gain are the two lowest; three
of those that lose were at or near saturation. **At a fixed budget you can move where the sensitivity goes
and not how much of it there is** — § 8's mechanism arriving a fourth time, after the panel (§ 4), the
negative set (§ 5) and the classifier head (§ 9.1). The gain and the cost are both reported at 30 seeds
**because reporting the gain at 30 and the cost at 5 is exactly the asymmetry this project keeps finding
in other people's tables**; `src/72` was extended to make them symmetric before anything was written.

⚠️ **What decides which classes gain is unexplained, and the mechanistic check is weak.** Not length:
beta-lactamase is the **shortest** class (median 278) and gains most, superantigen the second shortest
(257) and loses most, the clostridial neurotoxins the longest (1,296) and do not move. Across the 14
panel proteins with catalytic annotations the chosen window's centre sits a mean of **102** residues from
the nearest annotated catalytic residue against **143** expected for a uniform window, closer than chance
on **9 of 14** — sign test *p* ≈ 0.21, **not a result**. Six land essentially on the site (YopH 0,
`O34208` 1, cholera A 3, `Q51451` 3, colicin E2 9, anthrax PA 13); the two truncated 1,300-residue
neurotoxins land ~375 away. 🔴 **And it cannot be checked on the class that gained**: no beta-lactamase
in this panel carries functional-site annotations.

### Two sentences this falsified, and a claim it broke

🔴 § 9 read *"no pooling choice comes near alignment, which is what the heading claims"*. **False as of
today**, and rewritten: the heading now rests on the reallocation instead.

🔴 And claim "ESM-C 600M ... is the only arm that does" **failed the gate**, because `src/70`'s reductions
land in the same artifact namespace and the recomputation found two things above alignment. That is the
gate working exactly as intended: the public sentence said "the only one of the fourteen arms", and a
**re-pooling of the 650M arm** is not a fifteenth model. Both `docs/MECHANISM_GENERALIZATION.md` and the
Hugging Face card now say **model arm** and name the exception; the claim pins **both** facts rather than
excluding the new one, so neither can drift and the arm-versus-reduction distinction has to stay in the
prose.

### 🔑 The reading that survives, reached from an independent direction

A reduction that helps the lowest classes by as much as it hurts the highest is not a better global
choice. It is evidence that **the right reduction is per family**, and that the families needing locality
are the low-margin ones margin already identifies *before* training. That is § 8.2's mixture-of-experts
case, arrived at from a measured trade rather than from an analogy — and § 8.2's blocker still stands: the
gate has to work per sequence, and § 10.3 is the record of member-level margin failing preregistration on
two external panels.

⚠️ Bounds: v2 only, ESM-2 650M only, **label-free reductions only**. The supervised variant — residues
ranked by a direction fitted inside each fold, which is what a CNN would learn — is declared in `src/70`
and deliberately **not implemented**, and it is the one closest to the architecture that prompted all of
this.

---

## 2026-09-27 (fortieth entry) — Locality replicates on both failures; the supervised direction, which is what a CNN learns, cannot reach the harder one

### 🔑 The label-free window generalises

Entry thirty-nine reported `win_best25` on v2's beta-lactamase and said the bounds were "v2 only, 650M
only, label-free only". v3 closes the first at 30 seeds:

| flagged@95, 30 seeds | `mean_res` | `win_best25` | Δ |
|---|---|---|---:|
| beta_lactamase, **v2** | 15.5% [11.0, 19.9] | **35.0% [31.6, 38.4]** | **+19.5** |
| beta_lactamase, **v3** | 21.0% [16.3, 25.6] | **37.4% [35.5, 39.2]** | **+16.4** |
| **phage_peptidoglycan_hydrolase, v3** | 12.3% [10.0, 14.5] | **27.3% [23.3, 31.3]** | **+15.0** |

All three intervals disjoint from their controls. **Both of the project's documented failures move by 15
to 20 points**, and on the phage class it is not a lone grid point — **five of fourteen** reductions have
intervals clear of the control (`win_best5` 27.8% [24.7, 30.9], `win_best25`, `win_best15`, `win_best9`,
`dev_topk5`) against **one of fourteen** on v2's beta-lactamase.

🔴 **The reallocation replicates as well**, which is why this remains not a fix: v3's panel mean falls
**73.1% → 63.1%** against v2's 72.8% → 63.3%, with cry_insecticidal **−30.9**, pore_forming_cytolysin
**−40.0** and superantigen_enterotoxin **−25.7**, all interval-disjoint at 30 seeds.

### 🔴 And the supervised ranking, which is the one closest to a CNN, cannot touch beta-lactamase

`src/73_supervised_pooling_lomo.py` — declared in `src/70` and deliberately left unimplemented until it
could be done without leaking. The direction is fitted inside each fold on mean-pooled positives
**excluding the held-out class** plus the **training** negatives only, so it sees neither the class it is
tested on nor the negatives that set the threshold. **It reproduces `src/03b`'s fold loop rather than
approximating it**, and the reproduction is gated: at `k = 0` (every residue, i.e. the mean) the worst
per-class disagreement with the published run is **0.00e+00**.

| | v2, *k* = 100 | v3, *k* = 25 |
|---|---|---|
| panel mean | 72.4% → **75.2%** | 72.8% → **74.7%** |
| **beta_lactamase** | 15.5% → 15.5% (**+0.0**) | 21.0% → 15.5% (−5.5, ns) |
| **phage_peptidoglycan_hydrolase** | — | 12.3% → 23.6% (**+11.4**, n = 32) |
| contact_dependent_inhibition | +29.2 (**n = 4**) | +23.3 (**n = 4**) |
| cry_insecticidal / bacteriocin | — | +12.6 / +5.8 |
| pore_forming_cytolysin | −5.2 | −5.7 |

🔑 **Unlike the label-free window this raises the panel mean rather than lowering it** — +2.8 and +1.9 at
an unchanged realised FPR of 0.0656 — and on v2 it destroys nothing, the saturated classes staying at
100%. It moves the phage class **+11.4 on n = 32**, a far more trustworthy n than the +29.2 it gets on
contact-dependent inhibition, which has **four members** and so can only move in steps of 25%.

🔴 **But +0.0 on v2's beta-lactamase and −5.5 on v3's, and that has a mechanism.** The supervised
direction is fitted on the *other* classes, and beta-lactamase is the class whose members sit **closest to
benign** — § 10.4's negative margin. **A direction learned from classes it is unlike points the wrong way
for it.** The label-free "most deviant window" inherits no such bias, which is why the **unsupervised**
reduction beats the supervised one on exactly the class the supervised one cannot see.

⚠️ **Best-*k* is not stable and is not claimed to be**: 100 on v2, 25 on v3, both chosen after the fact.
The whole *k* curve is in the artifacts. What replicates is the **pattern** — mid-range *k* helps the low
classes, *k* = 1 is catastrophic on both panels (panel mean 55.6% and 57.0%), and no *k* helps
beta-lactamase.

### 🔑 What this settles about the architecture question

§ 8.2 argued for per-family routing from § 8's fixed-budget mechanism and § 10.3's failed member-level
gate. It now has two independent measurements instead of an analogy: **which reduction helps depends on
the class, and for the hardest class the supervised reduction is the wrong one.** A CNN learns supervised
local filters. On beta-lactamase, supervision is precisely what fails, because the supervision available
comes from families that class is unlike. A single global reduction — mean, window, or learned filters —
cannot be right for all twelve classes at a fixed false-positive budget.

### Three defects found while running this, all recorded rather than quietly fixed

⚠️ **`src/72` carried `_v2` inside its filenames after `--panel` was added**, so the v3 run read v2
artifacts from the v3 directory and died. A flag that switches a directory and not the filenames inside it
is a half-migration; six literals were replaced.

⚠️ **The cost loop rebound `a`, argparse's own namespace**, so `a.panel` died on the *second* class — after
the first had already printed a correct-looking result. Renamed. **A partial run that prints one good row
before crashing is the failure mode most likely to be read as a result.**

⚠️ **v3 has no `alignment_baseline.json`** — the alignment baseline was only ever computed on v2. The
script crashed on it; the fix reports "NONE for this panel" rather than borrowing v2's 29.5%, where both
the panel and the negative set differ.

⚠️ And one near-miss in the audit itself: the new claim first read the `k = 0` gate from the **30-seed**
artifact, where it is **0.0595** — which is not a gate failure but beta-lactamase's own 5-versus-30-seed
gap. The gate is only meaningful against `src/03b`'s five-seed run. Reading it from the wrong file would
have pinned a number that means something else while still passing.

---

## 2026-09-27 (forty-first entry) — The window's gain is one arm. Its cost is every arm. And margin explains the cost, not the gain.

Entry thirty-nine bounded the window result "v2 only, 650M only, label-free only". Entry forty closed the
first and third. This closes the second, and it is the one that mattered.

### 🔴 The gain does not survive a change of representation

`src/69 --model` and `src/70 --arm` put the same fourteen reductions on **`esm2_35M`**, gated the same way
(residue-only mean against the published `esm2_35M` artifact, 4.05e-06).

| beta-lactamase, 30 seeds | ESM-2 **650M** | ESM-2 **35M** |
|---|---|---|
| `mean_res`, control | 15.5% [11.0, 19.9] | 16.9% [10.0, 23.8] |
| **`win_best25`** | **35.0% [31.6, 38.4]** | **7.4% [4.7, 10.0]** |
| best on that arm | `win_best25` | `win_max15` 21.7% [15.8, 27.5] |
| intervals clear of the control | **1 of 14** | **0 of 14** |

🔴 **On the 35M arm the reduction costs beta-lactamase 9.5 points and nothing clears its control.** The two
controls are comparable — 16.9% against 15.5% — so this is not a dead representation, it is the same class
at the same difficulty with the reduction failing to help. **What entry forty replicated was the panel: v2
and v3 are both the 650M arm.** The representation had not been varied, and the write-up said so as a
bound rather than a result, which is the only reason this is a closure and not a retraction.

🔴 **The cost is general.** `win_best25` takes superantigen_enterotoxin 100% → **56.2%** (650M/v2),
95.2% → **69.5%** (650M/v3) and 100% → **54.8%** (35M/v2), all three interval-disjoint. **The reduction
reliably destroys the saturated classes everywhere and reliably helps on one arm.**

⚠️ **This is the second time the same rule caught the same shape of thing.** § 10.6.4's typicality baseline
reached −0.746 at *p* = 0.0034 on the canonical arm and **+0.021 at *p* = 0.53 on `esm2_35M`**. Twice now
an interval-clean effect on ESM-2 650M has vanished on the second arm. **The rule that every geometric
claim runs across representations is doing more work here than any single finding it has produced** — and
both times it was the cheapest arm in the project that did the work.

### Margin was the obvious mechanism and it explains the losses only

`src/30 --reductions` — the same margin code that produced § 10.6.1's arm table, pointed at the
reductions, writing its own artifact rather than a second implementation.

🟢 **Margin's ordering survives re-pooling, which strengthens § 10.4.** Across fifteen reductions of one
arm, Spearman(margin, recovery) runs **+0.536 to +0.941**, **14 of 15** significant at *p* < 0.05, and
**12 of 15** put beta-lactamase at the margin floor. § 10.6 established that across fourteen model
**arms**; it now also holds across fifteen **reductions** of a single arm — a second, different kind of
variation.

🔴 **But the gain is not a margin effect.** Change in beta-lactamase's margin against change in its
30-seed recovery, over the fifteen reductions: Spearman **+0.270, permutation *p* = 0.166** — **not a
result**. The pattern is asymmetric rather than absent: every reduction that drives the margin more than
0.019 below the control's loses 9 to 15 points, while `win_best25` gains 19.5 with a margin change of
**−0.0009**. **Preserving the margin is necessary for the gain and does not produce it**, since
`win_max9` and `win_max15` also leave the margin intact and still lose 6.9 and 8.3 points.

⚠️ Also worth recording: under `win_best5`, `win_best9` and `win_best25` beta-lactamase is **no longer the
lowest-margin class** — the labelled virulence control is. That looked like the mechanism for about a
minute. It is not: `win_best5` moves beta-lactamase off the margin floor and still loses 11.7 points.

### A namespace collision I created, and the published pipeline it would have broken

🔴 `src/70` writes `embeddings_<role>_<panel>_<tag>.npy` into the same directory as the model arms, and
`src/30`'s `discover_arms` globs exactly that pattern. **The next run of `src/30` would have turned
§ 10.6.1's ten-arm table into twenty-five**, silently, by counting fifteen re-poolings of one arm as
fifteen arms. This is the same arm-versus-reduction distinction entry thirty-nine had to put back into the
"only arm that clears alignment" sentence, arriving a second time through a glob.

Excluded by prefix in both `src/30` and `src/41`, and **the published artifact was re-derived to confirm
nothing moved**: 14 arms, identical verdict, identical per-arm numbers. The one difference is that
`arms_embedded_but_unscored` gained `_esmc_600M_mp` — a real embedding pair with no `lomo_results`,
produced by `src/56` after `src/30` last ran, and correctly reported. **Nothing was wrong; the list was
simply stale, which is its own small argument for re-deriving artifacts rather than trusting them.**

---

## 2026-09-27 (forty-second entry) — The supervised reduction fails the same test, so every gain in the pooling line is one arm

Entry forty-one closed the label-free window's representation bound and found the gain was ESM-2 650M
only. **Leaving § 9.1.3's supervised half unbounded while bounding the label-free half would have been
the same asymmetry this log keeps recording in other people's tables**, so `src/73 --arm` ran it on
`esm2_35M`, 30 seeds, gated at `k = 0` to **0.00e+00** against that arm's own published run.

| v2, 30 seeds | ESM-2 **650M** | ESM-2 **35M** |
|---|---|---|
| panel mean at *k* = 0 | 72.4% | 76.4% |
| best panel mean over *k* | **75.2%** (*k* = 100) | **75.4%** (*k* = 100) |
| *k* values above the *k* = 0 panel mean | **50 and 100** | 🔴 **none** |
| beta_lactamase | 15.5% → 15.5% | 16.9% → **2.6%** |
| contact_dependent_inhibition (**n = 4**) | 36.7% → 65.8% | 72.5% → 83.3% |

### 🔴 The unified result, which is negative

| | ESM-2 650M | ESM-2 35M |
|---|---|---|
| label-free `win_best25`, beta-lactamase | **+19.5** | **−9.5** |
| supervised top-*k*, panel mean | **+2.8** | **−1.0**, no *k* above control |
| supervised top-*k*, beta-lactamase | +0.0 | **−14.3** |
| cost: superantigen under `win_best25` | −43.8 | −45.2 |

**Every gain in §§ 9.1.2 to 9.1.4 is specific to ESM-2 650M. Every cost is general.** The only thing that
partly survives across arms is the supervised gain on contact-dependent inhibition — a class with **four
members**, where the statistic moves in steps of 25%.

🔑 **So the answer to the question that started this is negative.** A CNN learns local filters,
supervised. Both cheap stand-ins for that — a label-free local window and a supervised residue ranking —
gain on one representation out of two and cost on both. **Motivating a convolutional architecture from
either would be motivating it from a single-arm effect.** § 10.6's rule exists to stop exactly that, and
this is the third time in two days it has stopped something: the typicality baseline (§ 10.6.4), the
window (§ 9.1.4), and now the supervised ranking.

### What survives, and it is not small

🟢 **Margin's ordering holds across fifteen re-poolings of one arm as well as across fourteen model
arms** — Spearman +0.536 to +0.941, 14 of 15 significant, 12 of 15 with beta-lactamase at the floor.
That is a second and different kind of variation for § 10.4, and it is about the **diagnosis** rather than
a repair. 🔴 **But margin does not explain the one gain it was the obvious candidate for**: Δmargin
against Δrecovery over the fifteen reductions is Spearman **+0.270 at *p* = 0.166**, and `win_best25`
gains 19.5 points with a margin change of −0.0009. Preserving the margin is necessary and not sufficient.

⚠️ **A methodological note that is the real yield of this stretch.** Three findings in two days looked
interval-clean on ESM-2 650M and evaporated on the cheapest arm in the project. None of the three would
have been caught by more seeds, a bigger panel, or a better threshold — only by varying the
representation. **The rule is worth more than the findings it has killed**, and the cost of applying it
was one 103 MB residue stack and about twenty minutes of compute per test.

---

## 2026-09-27 (forty-third entry) — A review of the day's own work, which found two vacuous claims, one overlapping confirmation and one incomparable comparison

Asked to go back over the session for anything hasty, thin, or out of my depth. Five things checked, and
they were checked against the artifacts rather than reasoned about. **Three are real errors in text
already written, two of them mine from today.**

### 🔴 1. The false-positive check I cited as answering § 8 is vacuous

§ 9.1.2 and § 9.1.3 both read that the realised FPR is *"0.0656 for the control and 0.0656 for
`win_best25`, identical, so § 8's fixed-budget objection is answered."*

`03b` sets `t95 = quantile(s_nte, 0.95)` on the 40% negative holdout and then reports
`(s_nte >= t95).mean()` **on those same negatives**. With `m = int(154 × 0.40) = 61` that is 4/61 =
**0.0656 for any score vector whatsoever**. Checked: it is the single distinct value across every
reduction, every class and every seed, and a random normal vector returns it too. **That number is
vacuous.** It is a property of m and nothing else.

So **§ 8's objection is not answered.** § 8's mechanism is that capacity raising the *negatives'* scores
costs threshold headroom, and the quantity that tests it is an **out-of-sample** rate of the kind § 2.6.1
measures against the 8,259-protein pool — **never run for any reduction**. Both sentences are corrected in
place and the audit claim now pins the figure as *definitional* rather than as evidence.

⚠️ The other confound check in that paragraph does hold: the provenance probe going **0.8150 → 0.7766**
is a cross-validated AUROC on the features and does not involve the threshold.

### 🔴 2. A2's control set and the panel carry different *kinds* of annotation, and I never checked

The powered P2 verdict says benign enzymes exceed toxins on dFSPE-M by 2.44 points. `src/68` built its 60
controls from UniProt **`Active site`** features only. The panel's catalytic positions come from PDB and
literature curation, and **10 of 74 of them are a UniProt `Active site`** — the audit's own
`annotation_set_against_uniprot` line has been printing `frac_active = 0.32` on its comparable subset in
every run of the gate today.

| | positions from UniProt `Active site` |
|---|---|
| A2 controls (n = 60) | **100%** |
| v2 panel (74 annotated positions) | **13.5%** (10 of 74; ≈32% on the audit's comparable subset) |

🔴 **So "benign enzymes have more catalytic-site constraint than toxins do" is not established by this
comparison.** The alternative it does not exclude is that **UniProt's curated `Active site` positions are
more strongly conserved than a mixture of active sites, substrate contacts and functionally-important
residues** — which is a claim about annotation provenance, not about hazard.

What does survive: the **preregistered ceiling fires on direction**, and the burden was on the panel to
exceed the controls, which it does not, at *p* = 0.0009. The n = 4 controls had the same UniProt
provenance, so the powered run did not introduce the confound — **it inherited it and made it 15 times
larger without anyone noticing.** The repair is available and cheap: restrict the panel to its
UniProt-confirmed `Active site` positions and recompute. Not run today, named here.

⚠️ **This is the failure I am least comfortable with.** I built a 60-member control set to fix an
underpowered comparison and did not ask whether it was measuring the same thing on both sides. The number
that would have told me was on screen dozens of times.

### ⚠️ 3. The 30-seed "confirmation" of `win_best25` included the 5 seeds that selected it

`src/03b` screens on seeds 0–4; `src/03x` confirms on `range(30)`. The screen's seeds are a **subset** of
the confirmation's, so calling the second a confirmation was loose. 🟢 **Rerun on the 25 seeds that took
no part in the selection**: `win_best25` **36.0% [32.2, 39.8]** against the control's **14.3% [9.4, 19.1]**
— disjoint, **+21.7**, still clear of alignment's 29.5%. **The result does not depend on the overlap**, and
§ 9.1.2 now carries the held-out-seed figure. A screen-and-confirm that shares seeds is still worth
fixing even when the answer is unchanged, because next time it will not be.

### ⚠️ 4. The cross-reduction margin comparison used one scale for fifteen different geometries

§ 9.1.4 said *"every reduction that drives the margin more than 0.019 below the control's loses 9 to 15
points."* Margin is a difference of cosines, which is scale-free **within** a reduction — but the cosine
distribution itself moves enormously between them: mean pairwise cosine among positives is **0.256** under
`dev_topk1` and **0.931** under `mean_res`. A displacement of 0.019 is not one quantity across that grid.
Sentence withdrawn. What stands is the rank correlation, **+0.270 at *p* = 0.166**, which was null anyway,
and the within-geometry observation that `win_best25` gains 19.5 points with a margin change of −0.0009.

### ⚠️ 5. A1's negative set has no effective-*n*, which is the criticism § 10.9 made of the pool

`src/74` finished: **4,218 admitted** from VFDB setA, maximum similarity to any panel positive **0.2663**
against the 0.282 bound, rejections 248 Exotoxin / 222 length / 41 similarity / 12 panel-sequence-match /
12 alphabet / 2 duplicate. Categories are Effector delivery system 1,691, Immune modulation 658, Adherence
607, Motility 595. 38 genera, *Legionella* 440.

⚠️ **§ 10.9's whole point about the benign pool was that 8,259 raw records are 3,550 distinct names, a
redundancy factor of 2.33, and that the honest rate is the distinct-name one.** A1 currently reports raw
*n* only: exact-sequence duplicates were removed, near-duplicates were not, and `src/74` does not even
record VFDB's own `VF####` grouping id, which is the natural redundancy unit. **Before A1's rate is
quoted it needs the same distinct-unit accounting the pool got**, or it will invite exactly the criticism
this project levelled at itself.

### What the review did not find

The gates held everywhere they were checked. `src/73`'s reproduction of `03b`'s fold loop is exact at
`k = 0` on both arms (0.00e+00). `src/69`'s residue stacks reproduce the published pooled artifacts to
between 1.9e-06 and 9.1e-06 per protein on three arm/panel combinations. The A2 control set reruns
byte-identically. `src/30`'s published artifact was re-derived unchanged after the namespace fix. And the
cross-representation rule is the reason three of today's findings are bounded rather than believed.

🔴 **The pattern in the three errors is one thing, not three: I checked whether numbers were stable and
not whether they were measuring what I said they measured.** Seeds, intervals, gates and replication all
got attention. A definitionally constant quantity, an annotation-provenance mismatch and a cross-geometry
scale did not, and no amount of additional seeds would have surfaced any of them.

---

## 2026-09-27 (forty-fourth entry) — The A2 repair: half the gap was annotation, the hazard arm is a third non-hazard, and both repairs grew the side that was not binding

Entry forty-three named the A2 annotation confound and said the repair was available and not done.
`src/75_a2_annotation_matched.py` does it, and it needed **no new masked prediction**: `fspe_m_a2.json`
stores per-position `s(i)` and `src/62` builds them in sorted position order, so the panel's means
recompute directly on any subset of its positions.

### 🔴 Matching the annotation halves the gap and destroys the significance

| arm | panel *n* | panel mean | benign | difference | significance |
|---|---:|---:|---:|---:|---|
| as reported | 15 | +4.04 | +6.76 | **−2.72** | benign exceeds at *p* = 0.0009 |
| **annotation-matched** | **4** | **+5.57** | +6.76 | **−1.19** | AUROC **0.467 [0.117, 0.833]**, covers 0.50 |
| hazard-only, published annotation | 12 | +3.60 | +6.76 | −3.16 | — |
| **hazard-only *and* annotation-matched** | **2** | +4.37 | +6.76 | — | untestable |

Only **17 of the panel's 74 annotated positions (23%)** are a UniProt `Active site`, and only **4 of 15
proteins** carry two or more. On those four the panel mean rises by 1.53 points and the AUROC interval
**covers 0.50**. **So a substantial part of the powered result was annotation provenance.**

### 🔴 And a second defect the repair exposed, documented nowhere before today

**P2's hazard arm contains three BSL-1 entries with no hazard designation**: barnase and colicin E2
("no hazard designation") and Cas9 ("GRAS; widely used research tool"), as `functional_sites.json` itself
describes them. Section 4 of the mutation preregistration says P2 compares *"the panel mean"* against the
benign controls and never says that one fifth of the panel is not a hazard. Searched: no document flags it.

Excluding them moves the gap **away** from zero, to −3.16, so the dilution was working *against* the
finding rather than for it. ⚠️ But **two of the four annotation-matched survivors are barnase and Cas9**,
so the matched test's hazard arm is half non-hazard, and applying both restrictions leaves **ricin
(−1.27) and ExoU (+10.01)** — two proteins that disagree by 11 points.

### 🟢 What survives, and 🔴 what is withdrawn

The **preregistered verdict** survives every restriction because its ceiling is directional: the panel had
to *exceed* the controls and benign ≥ panel in all four arms. **P2 remains NOT SUPPORTED.**

🔴 **Withdrawn**: *"dFSPE-M measures catalytic-site constraint, and ordinary benign enzymes have more of
it than toxins do"*, together with the *p* = 0.0009 and the AUROC 0.265. Corrected in the two living
documents; it survives verbatim in three append-only records, which is why the audit pins the
**withdrawal** rather than forbidding the sentence — a forbid would fail the gate forever and the only way
to pass would be to rewrite history.

### 🔑 The synthesis, and it is uncomfortable

**A2 grew the wrong side of the comparison.** It was built to lift `n_benign` from 4, and it did: 60
controls, floor cleared, every frozen exclusion holding, the set reproducing byte-identically. The binding
constraint was the **panel** — 15 proteins, 74 positions, 17 confirmed, 4 proteins usable, 2 of those a
genuine hazard.

⚠️ **This is the second time today the constraint turned out to live on the panel side.** The fifth
amendment blamed `n_benign` for the P5 gate's AUROC having no power and predicted the tolerance would
become meaningful above 30 controls; `src/71` showed the null sd floors at 0.077 because an AUROC's
precision is set by the **smaller** group, which is the panel and cannot grow. **Two independent repairs,
both aimed at n_benign, both limited by n_panel.** What either would actually need is more panel proteins
carrying uniform, UniProt-confirmed catalytic annotation — one resource, two blocked tests, and it was
never the thing being bought.

---

## 2026-09-28 (forty-fifth entry) — The test § 8 needed: its mechanism is real, my reason for expecting it was wrong, and two "improvements" are not improvements

Entry forty-three retracted the claim that § 8's fixed-budget objection had been answered, because the
figure cited (0.0656) is 4/61 by construction. § 9.1.2 then froze a prediction before the pool embedding
finished — commit `84dca68` carries the timestamp — and this is it scored.

`src/76` embedded all 8,259 pool proteins under four reductions in 165 minutes, gate **4.05e-05** against
the published pool embedding (tolerance 5e-5, so it cleared and not by much — float32 accumulation across
runs, recorded because it is close). `src/49` then measured the rate on each with the published
118-negative calibration split.

| nominal 5%, 200 seeds | `np.quantile` | conformal | by distinct name |
|---|---|---|---|
| canonical, § 2.6.1 as published | 7.867% | 5.978% | 9.640% |
| **`mean_res`**, the control | **7.847%** | **5.964%** | **9.616%** |
| **`win_best25`** | 8.000% | **6.479%** | 9.374% |
| `win_best9` | 5.746% | 4.300% | 6.554% |
| `win_max9` | 7.193% | 5.436% | 8.806% |

🟢 The control reproduces the published arm to **0.02 points** on all three, which also retires § 9.1.4's
two-means worry for out-of-sample rates.

### 🟢 Half the prediction holds — § 8's mechanism is real

`win_best25` is worse on the conformal estimator, **5.964% → 6.479%, +0.52 points, intervals disjoint**.
Indistinguishable on `np.quantile` (+0.15, overlapping) and slightly better by distinct name (−0.24). So
the effect **exists, is small, and is estimator-dependent** — which is exactly why 0.0656 could not have
shown it and why the retraction was right.

### 🔴 The other half is refuted, and it was my reasoning, not the data

I predicted the rise partly because *"a representation that separates less well overshoots a nominal
budget by more."* It does not. `win_best9` has the **lowest** in-panel AUROC of the four (0.9470) and the
**best** out-of-sample calibration by a wide margin (−2.10, −1.66, −3.06 points, all disjoint);
`win_max9` has the **highest** (0.9735) and also beats the control. **In-panel separability does not order
the out-of-sample rates at all.** Pinned as such in claim 90, so the refuted reason cannot quietly return.

### 🔴 And the two lower rates are not improvements, which was the easy mistake here

`win_best9` cutting the out-of-sample rate from 5.96% to 4.30% on the project's **worst-scoring
criterion** is exactly the shape of thing that gets reported as a win. It is not one. In § 10.8's own
deployment terms — 10,000 screened, one hazard in a thousand, recall from the v3 panel mean:

| | conformal FPR | v3 recall | alerts | real hits | **precision** |
|---|---|---|---|---|---|
| `mean_res` | 5.96% | 73.1% | 603 | 7.3 | **1.21%** |
| `win_best25` | 6.48% | 63.1% | 654 | 6.3 | **0.97%** |
| `win_best9` | 4.30% | 54.0% | 435 | 5.4 | **1.24%** |
| `win_max9` | 5.44% | 69.5% | 550 | 6.9 | **1.26%** |

🔴 **Precision is flat at 0.97 to 1.26% across all four.** `win_best9` buys 1.7 points of FPR with **19
points of recall**: 603 alerts and 7.3 real hits become 435 and 5.4. That is a move **along the same ROC
curve**, obtainable by raising the threshold on `mean_res` and requiring no new reduction at all. **A
change that lowers the false-positive rate and the recovery together has not improved calibration**, and
reporting it as a repair of criterion 1 would have been wrong in the same way the 0.0656 claim was wrong:
a number that cannot distinguish what it is being used to distinguish.

### 🔑 What survives is § 8's reading, and it is coherent

`win_best25` is the **only** arm whose out-of-sample rate rises, and the **only** one that raises recovery
on the two classes the probe fails. It bought recovery and paid in the negatives' scores — § 8's mechanism
stated exactly, now measured rather than asserted. The bill: **+51 alerts per 10,000 screened, precision
1.21% → 0.97%**, against +19.5 points on beta-lactamase and +15.0 on the phage class.

⚠️ Both sides of that trade are ESM-2 650M only (§ 9.1.4), so it remains a measurement and not a
recommendation. And the flat precision column is the honest summary of this whole line of work: **four
reductions, three panels, two arms, and no configuration yet moves the screen's precision.**

---

## 2026-09-28 (forty-sixth entry) — A manuscript, and two things assembling it found

The recommendation taken was to write the methodological paper on what exists rather than spend weeks
enlarging the panel first, on the grounds that the argument for enlarging the panel **is** the paper's
content. `paper/`, 3,560 words, ~7 pages, assembled from `paper/0*.md` into `paper/MANUSCRIPT.md`.

**Its thesis is not the summary's.** `docs/DETECTOR_EVALUATION_SUMMARY.md` argues "the split comes first".
The paper argues the distance between **AUROC 0.974** on the panel and **1.2% precision** in a deployment
of ten thousand at a one-in-a-thousand hazard rate, and what three cheap practices did to make that
distance visible: ceilings as well as floors, every geometric claim across representations, and an
append-only log including the entries that hurt. The long documents become its supporting material rather
than being restated.

🔴 **It is on the audited surface**, which is the point of putting it there: a paper is the surface where a
stale figure travels furthest. Claim 91 recomputes its headline pair from the artifacts rather than reading
it out of the prose.

### 🔴 What assembling it found, which is why it was worth doing first

**1. The criteria scorecard was quoting a superseded number, unlabelled.** Its closing line read *"on a
framework whose headline aggregate number is 0.981."* **0.981 is the v1 separability**, superseded by the
2026-09-03 screening correction; `docs/BIOHUB_RESEARCH_BRIEF.md` carries the instruction to quote
**0.974 ± 0.014** instead, because the v1 membership is not shipped, contains two identical-sequence pairs,
and predates the barnase/Cas9 class decision. The other use of 0.981 in that document is explicitly
labelled v1 and is fine; this one was not. Corrected, and the unlabelled form is now forbidden by the gate.

⚠️ **Writing the argument is what surfaced it.** The number had been sitting in the scorecard's punchline
through every run of the gate, because no claim pinned that sentence. A figure is protected by the audit
only where a claim reaches it.

**2. My own two figures disagreed by two alerts, and I nearly shipped both.** The abstract quoted 603
alerts while citing the canonical arm's 7.87%; § 5's precision table also said 603. But § 5's table is
built on the **residue-only** mean, the control the reductions are compared against, and the abstract's
rate is the **published canonical** arm, which averages the special tokens in. The two differ by 0.014
points of false-positive rate and **two alerts**. The claim function computed 604.5 from the canonical
artifact and passed anyway, because I had written its tolerance as 600 to 606.

🔴 **A tolerance wide enough to hide the inconsistency it was meant to catch is the same defect as the
4/61 false-positive figure of entry forty-three**: a check that cannot distinguish what it is being used to
distinguish. Abstract now says 605 from the canonical arm, § 5 states that its control differs and why, and
the claim is tightened to ±1.0 alert.

### What the paper does not yet have

⚠️ A1's rate — the hard-negative test of criterion 1 — is still embedding, so the paper's § 3 carries the
pool figures only. ⚠️ The criteria table is referenced rather than reproduced, which is right for a paper
and means the eighteen verdicts are not independently pinned inside it. And it is a **draft**: nothing has
been through an outside reading.

---

## 2026-09-28 (forty-seventh entry) — A1: the same probe flags two fifths of non-toxin virulence factors, and study B's build

### 🔴 A1, and the prediction was right in direction and wrong by an order of magnitude

Same probe, same published **118-negative** calibration, two test sets. Method gate **9.06e-06**.

| nominal 5%, 200 seeds | `np.quantile` | conformal |
|---|---|---|
| benign pool, raw (8,258) | 7.867% | 5.978% |
| benign pool, distinct name (3,407) | 9.640% | 7.142% |
| **VFDB non-toxin virulence factors, raw (4,218)** | **46.698%** | **40.844%** |
| **VFDB, distinct VF group (565)** | **38.675%** | **32.612%** |

**A1-2 and A1-3: SUPPORTED**, by **+38.8** and **+34.9** points, intervals disjoint by 37.8 and 33.8. On
the like-for-like distinct-unit comparison — both sides then one representative per redundancy group —
**+29.0** and **+25.5**. The preregistration guessed "worse than 7.87%"; it is five to six times worse.

🔑 **This says what the screen separates: not toxins from other proteins, but virulence-associated
pathogen proteins from benign housekeeping ones.** § 2.3's provenance probe measures that at AUROC 0.818
on the panel; this is the same confound on a test set built to isolate it. In § 10.8's arithmetic the
605 alerts become **4,088** and precision **1.21% → 0.18%**.

⚠️ **What it does not establish**, stated because the number is quotable: it does not say the probe is
wrong to flag these proteins — VFDB virulence factors are not benign in any operational sense, and a
screen flagging them may be doing its job. It says **the 5% budget is not met against them**, and that a
false-positive rate quoted against housekeeping proteins **does not transfer to the population a screen
would meet**. Whether "virulence factor" should count as a positive is study B's question, not A1's.

🔑 **And it vindicates the pushback on the manuscript.** The paper was drafted with § 3 quoting 7.87% as
*the* out-of-sample rate, while the test that produces the most consequential number in the project was
still embedding. That number is now in § 3. **Writing before the results are in is what "premature"
meant, concretely.**

### Study B built

`src/78_external_class_axis_build.py`. 4,755 setA records → **746 representatives** at 6.37×
redundancy removed by the frozen rule (longest sequence per `VF####` group, ties by lowest VFG id).
**13 of 14 categories clear the panel's own eligibility floor of 7**, holding **740** across **72
species**; `Antimicrobial activity/Competitive advantage` has 6 and is reported rather than held out.

The pool partitions **1,000 train / 500 calibrate / 6,758 test** by `sha256(accession)`, contaminant
dropped. 🟢 **That is the split criterion 1 records as absent from the main panel** (178/118/**0**), so
study B's false-positive figures are out of sample by construction — and ⚠️ the test partition is
deliberately **unscreened**, because a deployed screen does not get to remove the proteins it will meet.

---

## 2026-09-28 (forty-eighth entry) — A length confound I built into a frozen rule, and three processes a kill silently failed to stop

Study B's implementation produced two problems worth recording, one about design and one about hygiene.
Both were caught by measuring rather than by re-reading.

### 🔴 The representative rule handed a probe 0.667 AUROC from length alone

§ 1 of `docs/EXTERNAL_CLASS_AXIS_PREREGISTRATION.md` froze "the longest sequence per `VF####` group,
ties by lowest VFG id". It was chosen for **determinism and nothing else**, and measuring the inputs
before fitting anything showed the cost:

| rule | mean length | **AUROC from length alone** |
|---|---:|---:|
| **longest, as frozen** | 803 | **0.6674** |
| median length | 589 | 0.4857 |
| **lowest VFG id** | 604 | **0.4835** |

Negatives average 462, so "longest" made the positives systematically longer and handed a probe two
thirds of an AUROC from sequence length — comparable to the composition baseline of 0.754 that § 9 of
`docs/MECHANISM_GENERALIZATION.md` treats as reason to discount a separability figure. A
margin-versus-recovery result on that set would have been partly a result about length.

**Amended to lowest VFG id** (amendment 3), which is chance on length. 🔑 **Legitimate because it is
outcome-independent**: no probe had been fitted, no recovery number existed, the decision rests entirely
on a length distribution computed before any label was used, and the alternatives were measured and
published in the amendment rather than picked quietly. The category structure is unchanged — the same 13
eligible categories holding the same 740 representatives, since grouping is by `VF####` and only the
choice *within* a group moved.

⚠️ And length now joins composition and shuffled labels as a reported control, because "it is near
chance" is itself a claim that can drift.

### 🔴 `pkill -f "A\|B"` matched nothing, and the verification used the same pattern

Cleaning up before the rerun, `pkill -f "79_external\|80_external"` was followed by
`pgrep -fl "79_external\|80_external" || echo "both stopped"`, which printed **both stopped**. macOS
`pgrep`/`pkill` take an **extended** regex, so `\|` is a literal and neither call matched anything.
**The kill and its check failed the same way**, so the check could not detect the failure.

Five processes were then running at once — three stale ones on the **previous** representative rule
(started 10:40, 10:43 and 10:47) and two new ones on the amended rule. That is why the screen's estimate
read 252 minutes: it was contending with its own zombies. With them gone the same screen reports **69**.
It also explains an embedding that had appeared to die silently an hour earlier: it had not died, there
were two of it.

🔴 **And there was real contamination.** A stale process wrote `shard_00000.npz` at 11:08 from the
old-rule positives. The new run counts complete shards and resumes after them, so it would have combined
**250 old-rule vectors with 496 new-rule ones** and written a finished-looking artifact with **no error
anywhere**. Caught by reading process start times, not by any check in the code.

**Fixed**: killed by PID and verified by PID; shard directory wiped; and `src/80` now writes a
`build.sha256` fingerprint of the build artifact into its checkpoint directory and **refuses to run** if
an existing directory's fingerprint disagrees, naming both hashes.

⚠️ **This is the fourth instance this session of one failure mode.** A false-positive figure that was
4/61 by construction; an alert tolerance of 600 to 606 that hid a 1.5-alert inconsistency; a `pgrep`
pattern that could not match; and a checkpoint that would inherit from a different experiment. **In all
four the verification shared an assumption with the thing it was verifying.** Seeds, intervals and
replication were checked; whether a check *could fail* was not.
