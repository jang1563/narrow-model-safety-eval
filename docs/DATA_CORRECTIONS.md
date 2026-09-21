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
ESM-C anomaly was tested on ESM-C 6B and refuted (2026-09-10 entry). This is a third, independent
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
