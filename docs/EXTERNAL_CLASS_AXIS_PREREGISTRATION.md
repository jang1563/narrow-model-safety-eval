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

---

## Amendments

Append-only. Nothing above this line is edited.

### Amendment 1 — 2026-09-28, before any embedding: thirty seeds would have varied nothing

§ 2 fixes the membership of all three partitions by `sha256(accession)`, and § 3 asks for **30 seeds**.
🔴 **Those two are incompatible.** In `src/03b` the seed drives the train/test negative permutation; with
membership fixed and logistic regression deterministic, thirty seeds would be thirty identical runs and
every interval would be zero-width.

**The resolution, which keeps what § 2 was for.** The **test** partition stays fixed — that is the
criterion-1 point and it must not be resampled, or the out-of-sample figures stop being out of sample.
**Train and calibrate are redrawn per seed from their fixed 1,500-protein union**, at the frozen sizes
1,000 and 500. The test partition is never touched by any seed.

⚠️ **And a size rule, fixed now rather than when the screen comes back.** The union is screened at
0.282 against the 746 representatives before any split. If fewer than 1,500 survive, the survivors are
split **2:1** train to calibrate and the realised sizes are reported wherever the figures are. No
exclusion is relaxed to reach 1,500.

⚠️ This was found by writing the implementation, not by re-reading the document. **A preregistration
that has not been implemented against has not been checked**, which is the second time in two days the
same lesson has appeared — amendment 1 of `docs/NEGATIVE_EXPANSION_PREREGISTRATION.md` recorded a rule
that could not be executed for the same reason.

### Amendment 2 — 2026-09-28, during the screen: two residues BLOSUM62 does not have

The screen crashed on `ValueError: sequence contains letters not in the alphabet`. BLOSUM62's alphabet
is `ARNDCQEGHILKMFPSTWYVBZX*` — it carries X, B and Z but **not U (selenocysteine) or O
(pyrrolysine)**, and **10 of the 1,500** train-plus-calibrate negatives contain one of them. (22 more
are in the test partition, which is unscreened by design and therefore unaffected; ESM-2's tokenizer
handles them.)

**Resolved by substituting for the alignment only: U → C, O → K**, both conservative — U is a cysteine
analogue and O a lysine analogue — with the count logged in the artifact. **The stored sequences are
untouched**, so nothing the probe or the embedding sees changes.

⚠️ **The alternative was to drop those ten, and it was rejected**: it would shrink a partition frozen
before this was known, for a reason that has nothing to do with the proteins themselves. `src/74` did
exclude non-standard sequences, but it was choosing which candidates to *admit*; here the membership is
already fixed and the question is only whether a similarity can be computed for it.

### Amendment 3 — 2026-09-28, before any probe was fitted: the representative rule created a length confound

§ 1 froze "the longest sequence, ties by lowest VFG id" as the representative rule. It was chosen for
determinism and **nothing else**, and measuring the inputs before running anything shows what it cost:

| representative rule | mean length | 🔴 **AUROC from length alone** |
|---|---:|---:|
| **longest (as frozen)** | 803 | **0.6674** |
| median length | 589 | 0.4857 |
| **lowest VFG id** | 604 | **0.4835** |

Negatives average **462**. So "longest" makes study B's positives systematically longer than its
negatives and hands a probe **0.667 AUROC from sequence length alone** — comparable in size to the
composition baseline of 0.754 that § 9 of `docs/MECHANISM_GENERALIZATION.md` reports as a reason to
discount a separability figure. A margin-versus-recovery result obtained on that set would be partly a
result about length.

**Amended to: one representative per `VF####` group, the record with the lowest VFG id.** Length-alone
AUROC **0.4835**, essentially chance.

🔑 **Why this is a legitimate amendment and not a post-hoc choice**, stated so a reader can check it:
**no probe has been fitted and no recovery number exists.** The decision is driven entirely by a property
of the inputs — a length distribution — computed before any label was used, and the alternative was
measured and reported here rather than picked quietly. The category structure is **unchanged**: the same
13 eligible categories holding the same 740 representatives, because the grouping is by `VF####` and only
the choice within a group moved.

⚠️ **What it costs.** The lowest-VFG-id rule takes 23 representatives under 100 residues where "longest"
took 11, since some VF groups are genuinely short peptides; the minimum is 38 either way. And the screen
and the embedding both restart, about forty minutes of work discarded.

🔑 **And length joins the reported controls regardless**, as criterion 14 requires: length-alone AUROC is
reported beside composition and shuffled labels in every table, because the fact that it is now near
chance is itself a claim that can drift.

### Amendment 4 — 2026-09-28, before the LOMO runs: the confound test needed an operational definition

§ 4 says the confound is handled by correlating *"the provenance probe's **per-category** accuracy"*
against recovery. 🔴 **That is not a computable instruction.** § 2.3's provenance probe predicts a
lab-strain label for panel members; VFDB representatives and pool proteins do not share such a label, and
"accuracy per category" of a probe that does not have a per-category target is undefined.

**Operationalized as the leakage channel that actually exists here, and stated before any recovery
number:** a held-out category is easy to recover if the probe has already seen its **organisms** in
another category's members. So for each category,

    exclusivity(C) = fraction of C's members whose species appears in NO other category

is computed from the build artifact alone — no embeddings, no labels, no probe — and correlated against
recovery. 🔴 **If that correlation is significant at *p* < 0.0125, recovery is being driven by organism
overlap rather than by category, and B-1 is reported as uninterpretable.**

⚠️ **Two further controls are added here rather than discovered later**, both because § 9 of
`docs/MECHANISM_GENERALIZATION.md` reports them for the panel and criterion 14 requires them: amino-acid
**composition** alone, and **length** alone — the latter because amendment 3 changed the representative
rule to bring length-alone AUROC from 0.667 to 0.484, and "near chance" is a claim that can drift.

⚠️ This is the third amendment written while implementing against a frozen document, after a rule that
made thirty seeds vary nothing and a rule that could not be executed on BLOSUM62's alphabet. **The
pattern is that a preregistration is checked by implementing it, not by re-reading it**, and all three
were caught before any probe was fitted.

### Amendment 5 — 2026-09-28, before any recovery number: amendment 4's confound test has almost no variance, and the reason is the argument

Amendment 4 defined the confound test as Spearman(species **exclusivity**, recovery), where exclusivity
is the fraction of a category's members whose species appears in no other category. Computed on the
inputs, before any probe was fitted:

🔴 **Exclusivity ranges 0.00 to 0.11, with five of the thirteen categories at exactly 0.00.** There is
almost nothing to correlate, the statistic is dominated by ties, and a non-significant result from it
would mean "no variance", not "no confound". ⚠️ **Reporting that as a confound ruled out would be the
same defect as entry forty-three's 4/61 false-positive figure**: a check that cannot distinguish what it
is used to distinguish.

**So the direct quantity is computed instead**, and it is the one the concern was really about — for each
held-out category, the fraction of its members that have a **same-species protein among the training
positives**:

| | |
|---|---|
| range across the 13 categories | **88.9% (Motility) to 100.0%** (five categories) |
| mean | **96.9%** |

🔑 **Organism leakage is near-total and near-uniform, and that is what makes B-1 interpretable rather
than what threatens it.** The probe has seen essentially every held-out category's organisms while
training, to within 11 percentage points, for every category. A channel that is equally open for all
thirteen cannot explain why one recovers at 100% and another at 10%. **The ordering has to come from
something category-specific.**

⚠️ **This is an argument from the structure of the data, not a significance test, and it is weaker for
it.** It rules out organism identity as the *differentiating* factor; it does not rule out that all
thirteen recoveries are inflated together by pathogen-versus-Swiss-Prot provenance — § 2.3 measures that
at AUROC 0.818 on the panel and it will be larger here. **B-1 is a claim about ordering and is reported
only as that.** The absolute recoveries are reported beside it and are not claimed to be hazard
detection.

🔒 Both quantities are recorded here **before** the LOMO runs, so the reading above cannot be mistaken
for a reaction to whatever B-1 returns.

---

## Results, 2026-09-28

`src/81_external_class_axis_lomo.py`. 746 positives in 13 held-out categories, negatives **1,468 of
1,500** surviving the screen (32 at or above 0.282, highest admitted 0.2785) so amendment 1's 2:1 rule
applies: **978 train / 490 calibrate** redrawn per seed, **6,758 test** never fitted or calibrated on.
30 seeds.

| category | *n* | recovery | margin | exclusivity |
|---|---:|---:|---:|---:|
| Stress survival | 20 | **38.7%** | **−0.0003** | 0.000 |
| Regulation | 31 | 46.1% | +0.0015 | 0.000 |
| Post-translational modification | 8 | 46.7% | +0.0021 | 0.000 |
| Immune modulation | 77 | 53.1% | +0.0058 | 0.013 |
| Nutritional/Metabolic factor | 71 | 54.1% | +0.0010 | 0.014 |
| Enzyme | 59 | 58.2% | +0.0040 | 0.017 |
| Biofilm | 20 | 63.8% | +0.0048 | 0.050 |
| Others | 15 | 72.7% | +0.0039 | 0.000 |
| Invasion | 31 | 74.0% | +0.0128 | 0.000 |
| Adherence | 180 | 79.0% | +0.0097 | 0.050 |
| Effector delivery system | 108 | 79.2% | +0.0060 | 0.056 |
| Exotoxin | 102 | 79.4% | +0.0037 | 0.098 |
| Motility | 18 | **84.8%** | +0.0066 | 0.111 |

### The four tests

| | result | verdict |
|---|---|---|
| **B-1** Spearman(margin, recovery) | **+0.6813**, permutation *p* = **0.00590** | 🟢 **SUPPORTED** at α = 0.0125 |
| **B-2** bottom-2 margin within bottom-3 recovery | margin bottom-2 = Stress survival, Nutritional/Metabolic; recovery bottom-3 = Stress survival, Regulation, Post-translational | 🔴 **NOT SUPPORTED** — 1 of 2 |
| **B-3** margin against its own parts | margin **+0.681** vs nearest-positive **+0.319** vs nearest-negative **+0.115** | 🟢 **SUPPORTED** |
| **B-4** out-of-sample FPR on the never-seen test negatives | **7.06%** at a nominal 5%, range [6.74, 7.27] | measurement |

🔒 **B-1 landed inside the frozen prediction band.** § 5 predicted "+0.4 to +0.8, significant, and weaker
than the panel's +0.894 because VFDB's categories are functional-role labels". It is **+0.681**.

🔴 **B-4's prediction was wrong.** § 5 predicted worse than the panel's 7.87%; it is **better**, 7.06%.
The likely reason is mechanical and in this study's favour: the threshold is set on **490** calibration
points here against the panel's **118**, so it is estimated better. That is an argument for the split
design, not for the probe.

### 🔴 And the preregistered confound rule fires, so B-1 is reported UNINTERPRETABLE

§ 4 and amendment 4 fixed the rule: if the organism measure correlates with recovery at *p* < 0.0125,
**recovery is driven by organism overlap and B-1 is uninterpretable.** It does —
Spearman(exclusivity, recovery) = **+0.7613**, two-sided *p* = **0.0042**, and amendment 5's
same-species-in-train measure gives the mirror image, **−0.7613** at *p* = 0.0039. **By the rule as
written, that is the verdict, and it stands.**

⚠️ **But the sign is wrong for leakage, and that is worth stating without using it as a rescue.**
Leakage predicts that a category whose organisms the probe has already seen is **easier**. The data say
the opposite: higher same-species-in-train goes with **lower** recovery. So whatever this correlation is,
it is not the mechanism the rule was written to catch.

⚠️ **Post-hoc, and labelled as such**, using § 10.6.4's partial-correlation construction: margin keeps
**+0.6322** of its +0.6813 with exclusivity held (*p* = 0.0286) and exclusivity keeps **+0.7273** of its
+0.7613 with margin held (*p* = 0.0074); the two are correlated at only **+0.402** (*p* = 0.17). So they
are **not** redundant — both survive controlling for the other — and neither is explained away by the
other. 🔴 **This does not overturn the preregistered verdict.** It says the study cannot separate two
partially independent predictors on thirteen points, which is a statement about power, and the
separation needs a design where organism composition is held fixed across categories rather than a
larger *n* of the same shape.

### Amendment 6 — 2026-09-28: B-1's verdict, revisited by the design this study called for

The Results section above reports B-1 as **uninterpretable**, because the preregistered confound rule
fired, and says the separation "needs a design where organism composition is held fixed across
categories". That design is `docs/ORGANISM_STRATIFIED_PREREGISTRATION.md`, and it returned:

🟢 **Mean within-species Spearman(margin, recovery) = +0.3750, permutation *p* = 0.0104, 8 of 10 strata
positive.** Inside a species the organism is constant, so that stratum's ordering cannot be produced by
organism identity, exclusivity or same-species-in-training.

🔑 **So B-1 is reinstated as a claim about ordering**: margin's ordering of VFDB's categories is not an
artifact of organism composition. ⚠️ **The uninterpretable verdict above is not deleted** — it was the
correct call on the evidence available when it was made, and the rule that produced it is the reason a
targeted follow-up existed to run. 🔴 **And the reinstatement is narrow**: it covers the *ordering* only.
The absolute recoveries remain compatible with pathogen-versus-Swiss-Prot separation, which A1 measures
at 40.8% false positives on non-toxin virulence factors.

---

## Addendum — 2026-09-28: the standing caveat above has now been measured on both of its halves

The caveat this document leaves open — that the absolute recoveries stay compatible with
pathogen-versus-Swiss-Prot separation rather than hazard — has since been quantified by two controls
that hold the probe and its calibration fixed and vary only what the proteins are:

| control | benign comparison | share of the pool→VFDB gap |
|---|---|---:|
| pathogen origin (`docs/PROVENANCE_CONTROL_PREREGISTRATION.md`) | 6.98% → **22.08%** | **22.7%** |
| localization (`docs/LOCALIZATION_CONTROL_PREREGISTRATION.md`) | 7.08% → **21.73%** | **22.1%** |

🔴 **The localization control returned an adverse verdict on both arms**: benign *extracellular*
bacteria/archaea are flagged at **21.73%** against **4.15%** cytoplasmic — **5.36×**, replicating at
5.13× on ESM-2 35M and surviving length matching. **On the benign side this probe behaves substantially
like a localization detector**, and the recoveries in this document have to be read with that in hand.

🔒 **What survives**: against a localization-matched benign baseline of 21.73%, VFDB proteins are still
flagged at 73.49%, and against the pathogen-matched baseline of 22.08% likewise — so **neither confound
explains most of the separation.** ⚠️ **Nor may the two shares be added** — pathogen-derived proteins are
enriched for secretion, and the joint decomposition has not been run.
