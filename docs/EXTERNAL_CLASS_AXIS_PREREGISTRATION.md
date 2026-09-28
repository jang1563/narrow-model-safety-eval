# Preregistration — margin on a class axis this project did not define

**Frozen 2026-09-28, before any embedding, probe or recovery figure exists for this design.**
Amendments append below and never edit what is above.

---

## 0. Why this study and not another representation

`docs/MECHANISM_GENERALIZATION.md` § 10.4 to § 10.6 report that class-level **margin** — nearest
other-class hazard minus nearest negative — ranks mechanism classes by leave-one-class-out recovery at
Spearman **+0.894** (permutation *p* = 0.00015), places both failing classes at the bottom, and holds
across **fourteen model arms** and **fifteen re-poolings** of one arm. It is the only result in this
project that replicated everywhere it was tested.

🔴 **It has never been tested on classes this project did not define.** Every one of those tests is on a
panel whose eleven-to-twelve mechanism classes were assigned by hand here. § 10.3 is the record of the
**member**-level version of the same claim being preregistered, frozen, and **failing on two external
panels**. A reviewer's first question about the class-level result is whether the classes were drawn to
suit it, and nothing in the repository answers that.

⚠️ **This study exists because writing the manuscript made the gap obvious.** The draft's § 4 rests on
margin and its § 5 reports that nothing improved the screen; a single positive finding with no external
test is not enough to carry a paper, and enlarging the evidence is the work that was skipped.

---

## 1. The axis, and why VFDB

VFDB assigns every record to one of **14 categories** maintained by its curators, and DeepVIC
([Bioinformatics Advances 6(1) vbag237, 2026](https://academic.oup.com/bioinformaticsadvances/article/6/1/vbag237/8762933))
uses that axis over 12,989 annotated VFs while running **no leave-one-category-out evaluation** of any
kind. The axis is published, externally maintained, and unoccupied for this question.

**Redundancy is removed before anything else.** VFDB lists orthologs of one virulence factor across
organisms under a shared `VF####` id, and setA's redundancy factor is **6.37**. One representative per
`VF####` group is kept — **the longest sequence, ties broken by lowest VFG id**, a rule fixed here and
not tuned.

| | setA raw | representatives |
|---|---:|---:|
| records | 4,755 | **746** |
| categories with ≥ 7 representatives | — | **13 of 14** |
| representatives in those 13 | — | **740** |
| distinct species | — | 72, the largest *E. coli* at 8% |

The eligibility floor of **7** is the panel's own, taken from `mechanism_classes_v3.json`, not chosen
here. `Antimicrobial activity/Competitive advantage` has 6 and is therefore **reported and not held out**.

⚠️ **This changes the hazard construct and that is stated, not hidden.** `Exotoxin` is one category of
fourteen and **102 of 746 representatives (14%)**. A VFDB-derived positive set is a *virulence-factor*
set, not a toxin set: adherence proteins, flagellar components and iron-uptake systems are in it. The
question being asked here is about **margin as a predictor of class-level generalization**, which is a
claim about geometry and not about hazard, and it is the only claim this study makes.

---

## 2. Negatives, and the split criterion 1 says the main panel lacks

Negatives come from the **8,259-protein benign pool** already built, screened and embedded. It is
partitioned **once, before any probe is fitted**, by a deterministic rule:

| partition | size | role |
|---|---:|---|
| **train** | 1,000 | fitted |
| **calibrate** | 500 | sets the threshold |
| **test** | the remainder, ≈ 6,700 | never seen by fitting or calibration |

Assignment is by `sha256(accession)` ascending, a reproducible draw from the whole pool rather than the
head of any ordering, with the contaminant `Q8X739` dropped as the repository's standing rule requires.

🟢 **This is the partition criterion 1 of `docs/DETECTOR_CRITERIA.md` records as absent from the main
panel** (178 train / 118 calibrate / **0 test**). It exists here by construction, so every
false-positive figure in this study is out of sample from the start.

**Homology screen.** The 1,500 train-plus-calibrate negatives are screened against all 746
representatives with the repository's aligner — `Bio.Align.PairwiseAligner` local, BLOSUM62, gap −11/−1,
normalized `score / sqrt(self_i · self_j)` — and any negative at or above **0.282**, the panel negatives'
own maximum similarity to a hazard, is excluded and logged. ⚠️ The **test** partition is deliberately
**not** screened: a deployed screen does not get to remove the proteins it will meet, and screening it
would make the out-of-sample rate optimistic in exactly the direction § 2.6.1 warns about.

---

## 3. What is measured

Leave-one-category-out, with `src/03b`'s fold logic unchanged: hold out a category, fit on the other
twelve plus the train negatives, threshold on the calibrate negatives, report recovery of the held-out
category and the false-positive rate on the test negatives. ESM-2 650M, mean pooling, logistic
regression, **30 seeds** — not five, because § 9 records that five is where this project's seed
instability hid.

Margin is computed exactly as `src/30_margin_across_arms.py` computes it, on the same embeddings.

---

## 4. Primary tests, frozen

Multiplicity: four primary tests, so **α = 0.05 / 4 = 0.0125**. Each has a **ceiling** as well as a
floor, because `docs/DETECTOR_CRITERIA.md` criterion 12 is a **fail** and this document is written to
stop that recurring.

| # | test | supported if | 🔴 not supported if |
|---|---|---|---|
| **B-1** | Spearman(margin, recovery) across the 13 categories | *ρ* > 0 at permutation *p* < 0.0125 | *ρ* ≤ 0, **or** *p* ≥ 0.0125. A null here means the panel result did not transfer to classes this project did not define |
| **B-2** | do the lowest-margin categories fail? | the bottom-2 by margin are among the bottom-3 by recovery | they are not. Reported whatever B-1 does |
| **B-3** | margin against its own parts | *ρ* exceeds both nearest-positive-alone and nearest-negative-alone | either part matches or beats it, in which case the difference is not carrying the result |
| **B-4** | out-of-sample FPR on the never-seen test negatives at a nominal 5% | reported with its interval — a measurement, not a hypothesis | — |

**Controls, run and reported regardless of outcome**: amino-acid composition alone; shuffled category
labels; and 🔴 **the provenance probe**, which matters more here than on the panel — VFDB positives are
pathogen-derived and the pool is Swiss-Prot, so a probe could separate on origin alone. § 2.3 measures
that confound at **AUROC 0.818** on the main panel and it will be **larger** here.

⚠️ **Why the confound does not automatically void B-1, and how that is checked.** B-1 is about the
**ordering** of thirteen categories, and a confound acting equally on all of them shifts every recovery
without reordering any. That is an argument, not a result, so it is tested: the provenance probe's
**per-category** accuracy is correlated against recovery, and if that correlation is itself significant
the ordering is confounded and B-1 is reported as uninterpretable.

---

## 5. Predictions, written before the first embedding

🔒 **B-1 will be supported, and more weakly than on the panel.** The panel gives *ρ* = +0.894 on twelve
hand-assigned mechanism classes. VFDB's categories are functional-role labels — "Adherence", "Motility" —
not mechanism classes, so several will be internally heterogeneous in a way the panel's are not, which
adds noise to both margin and recovery. **Predicted: *ρ* between +0.4 and +0.8, significant.**

🔒 **B-4 will be worse than the panel's 7.87%.** The test negatives are unscreened by design and the
positive set is five times larger and more diverse.

🔴 **If B-1 comes back null, the class-level margin result is a property of classes this project drew**,
and § 10.4 to § 10.6 need rewriting as a panel-specific observation rather than a mechanism. That outcome
is the reason to run it, and the manuscript stays a draft until this returns either way.
