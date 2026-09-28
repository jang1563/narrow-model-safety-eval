# Preregistration — margin with organism held constant

**Frozen 2026-09-28, before any per-member score has been saved or any within-species number computed.**
Append-only amendments below.

---

## 0. The question this exists to answer

Study B (`docs/EXTERNAL_CLASS_AXIS_PREREGISTRATION.md`) found Spearman(margin, recovery) = **+0.6813**
at *p* = 0.0059 across VFDB's 13 curator-maintained categories — margin's ordering reproducing on an
axis this project did not define. 🔴 **And its own preregistered confound rule fired**: an organism
measure correlates with recovery at **+0.7613**, *p* = 0.0042, so B-1's verdict is **uninterpretable**.
Post-hoc partials showed the two are not redundant (margin keeps +0.632 with the organism measure held;
the organism measure keeps +0.727 with margin held; they correlate at only +0.402) but thirteen points
cannot separate two partially independent predictors.

**Study B's own conclusion was that the separation needs a design holding organism composition fixed.**
This is that design.

---

## 1. Why matching was rejected, with the numbers

Three designs were evaluated on feasibility **before** this document was written, and two were discarded:

| design | why it fails |
|---|---|
| **within a single organism** | the most prolific species, *M. tuberculosis*, has 39 representatives across 9 categories but only **6** with ≥ 4 members. No species supports a 13-category holdout |
| **species-excluded folds** — drop from training every protein sharing a species with the held-out category | training collapses unevenly: **82 proteins (11%) for Adherence against 656 (88%) for Post-translational modification**. It trades the organism confound for a training-size confound |
| **a shared species pool** — draw all categories from the top-*N* species | restricting to the top 6 moves the mean pairwise species-profile total-variation distance only from **0.753 to 0.540**, and costs three categories. Profiles do not converge |

🔑 **The reason matching fails is biological, not statistical.** Categories genuinely have different
organism profiles — Motility factors come from flagellated organisms, exotoxins from toxin producers —
so no subsetting makes the profiles equal. **Organism is therefore held constant by stratification
rather than by matching**, which is the same move § 2.4.1 made when it held mechanism class constant to
test target host.

---

## 2. Design

The folds are **study B's, unchanged**: 13 categories held out one at a time, 978 train / 490 calibrate
negatives redrawn per seed from the screened union, 6,758 unscreened test negatives, 30 seeds, the same
`src/03b` pipeline. ⚠️ **No new embedding, screen or partition.** The only change is that per-member
outcomes are saved so recovery can be read **within a species**.

For each species *S* and each held-out category *C* where *S* contributes **≥ 3** members:

    recovery(C,S) = mean over seeds of the fraction of S's members of C that cleared the threshold
    margin(C,S)   = margin computed on S's members of C only, src/30's definition

Strata are the species with **≥ 4** such categories. On the built set that is **10 species covering 48
(species, category) cells** — fixed by the data, not chosen here.

🔑 **Inside a stratum the organism is constant**, so the ordering of that stratum's categories by
recovery cannot be produced by organism identity, exclusivity, or same-species-in-training. Those were
the three quantities that voided study B.

---

## 3. Primary test and predictions

One primary test, **α = 0.05**. Secondaries are reported with no claim attached.

| | test | supported if | 🔴 not supported if |
|---|---|---|---|
| **C-1** | the mean within-species Spearman(margin, recovery), against a null that permutes recovery **within** each species so the stratification is preserved | mean *ρ* > 0 at *p* < 0.05 | mean *ρ* ≤ 0, **or** *p* ≥ 0.05 |

**Secondaries**: the per-species *ρ* values and their spread; how many of the ten are positive; recovery
and margin per cell; and the number of cells, since a stratum with four categories contributes a rank
correlation over four points.

🔒 **Prediction.** Study B gave +0.681 over thirteen categories. Within a species there are four to six
categories and three or more members per cell, so this has far less power. **Predicted: a positive mean
*ρ* between +0.1 and +0.5, and I do not know whether it will clear *p* < 0.05.** That uncertainty is
stated rather than resolved by choosing a friendlier test.

🔴 **What each outcome licenses, fixed now so neither can be over-read.**

- **Supported**: margin's ordering survives with organism held constant. Organism identity is excluded
  as the *differentiating* factor, and study B's B-1 becomes interpretable as a claim about ordering.
  ⚠️ It still would not show the absolute recoveries are hazard detection rather than
  pathogen-versus-Swiss-Prot separation, which § 2.3 measures at AUROC 0.818 and A1 at 40.8%.
- **Not supported**: ⚠️ **this is the outcome that cannot be read cleanly, and saying so now is the
  point.** Four to six points per stratum cannot distinguish "margin does not work within an organism"
  from "there is not enough power here". If C-1 returns null, the honest report is that **study B's
  confound remains unresolved**, not that margin is refuted — and the next design would need more
  members per cell, which means a larger VFDB set than setA's experimentally verified core.

---

## Results, 2026-09-28

`src/82_organism_stratified.py`. Study B's folds unchanged — the per-category member flag rates
reproduce study B's recoveries exactly (54.1%, 72.7%, 46.7%, 46.1%, 38.7% …), which is the check that
these are the same folds. **104 cells at ≥ 3 members; 10 species with ≥ 4 categories, covering 48
cells** — the counts the feasibility pass predicted.

| stratum | categories | *ρ*(margin, recovery) |
|---|---:|---:|
| *Listeria monocytogenes* | 7 | **+0.821** |
| *Neisseria meningitidis* | 4 | +0.800 |
| *Salmonella enterica* | 4 | +0.800 |
| *Escherichia coli* | 6 | +0.600 |
| *Legionella pneumophila* | 5 | +0.500 |
| *Mycobacterium tuberculosis* | 6 | +0.429 |
| *Vibrio cholerae* | 4 | +0.400 |
| *Streptococcus pyogenes* | 4 | +0.200 |
| *Staphylococcus aureus* | 4 | 0.000 |
| *Porphyromonas gingivalis* | 4 | **−0.800** |

🟢 **C-1: SUPPORTED.** Mean within-species *ρ* = **+0.3750**, permutation *p* = **0.0104** at α = 0.05,
**8 of 10** strata positive, median *ρ* **+0.464**. The null has mean −0.0001 and sd 0.1675, so the
observed value sits **2.24 standard deviations** out.

🔒 **Inside the frozen band.** § 3 predicted "+0.1 to +0.5, and I do not know whether it will clear
*p* < 0.05". It is +0.375 and it does.

### 🔑 What this licenses, by the reading fixed in § 3

**Organism identity is excluded as the differentiating factor.** Inside a stratum the organism is
constant, so that stratum's ordering of categories by recovery cannot be produced by organism identity,
exclusivity, or same-species-in-training — the three quantities that voided study B. **Study B's B-1 is
therefore interpretable as a claim about ordering**, and the amendment to that document records it.

### ⚠️ And what it does not license

- **The effect is weaker and the margin of significance is not large**: +0.375 against B-1's +0.681,
  2.24 sd from the null, *p* = 0.0104 against α = 0.05.
- **One stratum runs hard the other way.** *P. gingivalis* gives **−0.800** on four categories, which is
  one swap from zero. With four points a single cell moves a stratum's *ρ* by 0.4 or more, and two
  strata sit at 0.000 and +0.200.
- 🔴 **It says nothing about the absolute recoveries.** Whether a flagged protein is flagged *because it
  is hazardous* is untouched: § 2.3's provenance probe reaches AUROC 0.818 on the panel, and A1 found
  this same probe flags **40.8%** of non-toxin virulence factors at a nominal 5%. **Margin ordering the
  categories correctly is compatible with the whole ordering sitting on top of pathogen-versus-Swiss-Prot
  separation.**
- ⚠️ **Sensitivity, reported because it was run**: the frozen cell floor of 3 gives 10 strata and 48
  cells; raising it to 4 leaves 5 strata and 23 cells, and to 5 leaves 3 strata and 13. The frozen
  setting is the one reported, and the rapid loss of strata is why it was set at 3 before the data were
  seen.

---

## Second arm, 2026-09-28 — and this is the first thing in the project to survive it

§ 9.1.4 of `docs/MECHANISM_GENERALIZATION.md` records **three** findings that were interval-clean on
ESM-2 650M and vanished on ESM-2 35M. The Results section above closed with "nothing here has been run
on a second arm". It has now. The screen is sequence-based and therefore arm-independent, the pool was
already embedded on both arms with identical row order, and the 746 representatives were re-embedded
with a gate of **1.67e-06** against that arm's published panel negatives.

| | ESM-2 **650M** | ESM-2 **35M** |
|---|---:|---:|
| **C-1** mean within-species *ρ* | **+0.3750** | **+0.4107** |
| permutation *p* (α = 0.05) | 0.0104 | **0.0069** |
| strata positive | **8 of 10** | **8 of 10** |
| verdict | 🟢 SUPPORTED | 🟢 **SUPPORTED** |
| | | |
| **B-1** Spearman(margin, recovery) | +0.6813 | **+0.7253** |
| permutation *p* (α = 0.0125) | 0.0059 | **0.0034** |
| **B-2** bottom-2 margin in bottom-3 recovery | 🔴 NOT SUPPORTED | 🟢 **SUPPORTED** |
| **B-3** margin against its parts | 🟢 SUPPORTED | 🟢 SUPPORTED |
| **B-4** out-of-sample FPR at a nominal 5% | 7.06% | **4.95%** |
| confound *ρ*, rule fires below *p* = 0.0125 | +0.7613 at *p* = **0.0042** | +0.5887 at *p* = 0.0371 |
| B-1 uninterpretable by the frozen rule? | **YES** | **no** |

🟢 **C-1 replicates, and slightly stronger.** 🟢 **B-1 replicates, and on the 35M arm the confound rule
does not fire at all**, so B-1 is interpretable there without the stratified follow-up. 🟢 **B-2, which
failed on 650M, passes on 35M.**

🔑 **B-4 is worth its own line: 4.95% against a nominal 5%**, on negatives that were never screened and
never seen by fitting or calibration. The panel's own out-of-sample figure is 7.87%. **The difference is
the split**: 490 calibration points here against 118 there. That is the clearest evidence in this project
that criterion 1's fail is about the *design* and not about the representation.

⚠️ **What does not replicate is which strata are weak**, and it is worth quantifying rather than
asserting. Across the ten strata the two arms' *ρ* values agree at only **+0.49** (Pearson), with a mean
absolute difference of **0.341**, a maximum of **1.000**, and **two strata moving by more than 0.5**:
*P. gingivalis* goes −0.800 → 0.000 and *S. pyogenes* +0.200 → −0.800. Meanwhile the **aggregate moves by
0.036** and the count of positive strata is identical at 8 of 10. 🔑 **The per-stratum values are noise
around a stable aggregate**, which is what four categories per stratum should produce, and they must not
be read individually.

🔴 **And the bound that does not move.** Both arms order the categories; neither says the ordering sits
on anything but pathogen-versus-Swiss-Prot separation. A1 measures that directly at **40.8%** false
positives against non-toxin virulence factors, and it is not addressed by anything in this document.

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

---

## 🔒 Superseded on this point — 2026-09-28, later the same day

The sentence above saying the joint decomposition **has not been run** was true when written and is
no longer. `docs/JOINT_DECOMPOSITION_PREREGISTRATION.md` ran it: provenance and localization are
**multiplicatively independent** (RR = 1.001 on 650M, 1.364 on 35M, both inside the frozen band), and
**together they span 42.1% of the pool→VFDB gap on 650M and 32.2% on 35M** — measured, not summed.
🔴 **22.7% + 22.1% remains wrong** and remains forbidden by claim 98. The text above is left as
written rather than edited, per this repository's append-only rule.

🔒 **Figures above superseded 2026-09-28 by entry 55** — `src/84` and `src/85` did not use `src/83`'s
fold, which falls back to 978/490 because the screen admits only 1,468 rows. Every number moved in the
third decimal and **no verdict changed**; the corrected tables are in the two preregistrations'
amendments.
