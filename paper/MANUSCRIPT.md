# An embedding hazard screen at AUROC 0.974 and precision 1.2%: what eighteen criteria and sixty corrections found

**Draft, 2026-09-28.** Every figure below is recomputed from an artifact by
`src/22_claims_audit.py` and pinned to the sentence that states it; the gate currently checks 90 such
claims. Section references without a document name are to `docs/MECHANISM_GENERALIZATION.md`.

---

## Abstract

A protein-language-model embedding separates a curated hazard panel from its negatives at **AUROC
0.974 ± 0.014**. Screened against 8,258 held-out benign proteins at a nominal 5% false-positive budget
the same probe returns **7.87%**, and translated into a deployment of ten thousand sequences at a
one-in-a-thousand hazard rate it produces **605 alerts of which about 7 are real — a precision of
1.2%**. This paper is about the distance between those two numbers, and about what it takes to see it.

We state **eighteen criteria** a hazard-screening claim should meet, score our own probe against them,
and report the result: **two fails, four partials and one mixed**. We then show that each criterion was
written because it caught something. The evidence is a **dated, append-only corrections log of sixty
entries** in which a published number moved, including entries that retract our own findings.

Three results matter beyond this panel.

**First, the aggregate is uninformative and the per-class picture is bimodal.** Holding out an entire
mechanism class, recovery runs from **100% to 10%**: four classes are fully recovered without ever
being trained on, while beta-lactamase reaches **15.5% [11.0, 19.9]** at thirty seeds and a
32-member phage peptidoglycan hydrolase class reaches **10%**. An AUROC of 0.974 is compatible with a
screen that cannot see two whole families.

**Second, a quantity computable before training predicts which class will fail.** Class-level
**margin** — nearest other-class hazard minus nearest negative in embedding space — ranks classes by
recovery at Spearman **+0.894** (permutation *p* = 0.00015), places both failures at the bottom, and
survives **fourteen model arms** and **fifteen re-poolings** of one arm (*ρ* +0.536 to +0.941, 14 of 15
significant). It is the only result here that replicated everywhere we looked. It is also
**mis-calibrated**: it over-predicts beta-lactamase recovery by 34 points.

**Third, and the finding we did not want: nothing we tried moves the screen's precision.** Four
classifier heads, fourteen representations, three panels, fifteen pooling reductions and a supervised
residue ranking later, deployment precision sits between **0.97% and 1.26%** in every configuration.
One reduction moves the two failing classes by +15 to +20 points — and pays for it in the negatives'
scores exactly as a fixed-budget argument predicts, and does so **only on one of two representations
tested**.

We close on a structural observation about evaluation rather than about protein models. Twice we
diagnosed a limit, built the repair, and found the limit was on the other side of the comparison: both
times we grew the negative set when the binding constraint was **panel annotation** — fifteen proteins,
seventy-four annotated catalytic positions, **seventeen** of them confirmed by a common source. An
evaluation can be improved in the wrong direction, and the cheapest way to find out is to write the
ceiling down before the run.

---

## 1. Introduction

A protein language model's embedding space separates toxins from benign proteins well enough that a
linear probe reaches AUROC 0.974 on a curated panel. It is a short step from there to a claim about
biosecurity screening, and the step is where the trouble is. This paper is an attempt to take that step
slowly, in public, with the failures written down as they were found.

Our contribution is not a better classifier. It is an account of what an evaluation has to do before a
separability number licenses a screening claim, together with the record of what that account cost us:
sixty dated corrections, two preregistrations of which one returned NOT SUPPORTED, and three findings
that looked clean on one representation and vanished on a second.

We make three empirical claims and one methodological one.

1. **The aggregate hides a bimodal failure structure.** Leave-one-mechanism-out recovery spans 100% to
   10%, and the classes that fail are not the small ones (§ 2).
2. **The failures are locatable in advance.** Class-level margin ranks them at *ρ* = +0.894 and
   transfers across representations and re-poolings, while over-predicting the hardest class by 34
   points (§ 4).
3. **Nothing we tried improved the screen.** Precision is flat at about 1% across every configuration
   tested (§ 5).
4. **Methodologically: the ordering of criteria matters more than the tally.** The split between
   calibration and test negatives is first because the other seventeen are measured through it (§ 3).

⚠️ **What this paper is not.** It is not a benchmark, a tool, or a deployment recommendation. § 7 lists
what it does not claim, and every number is on a self-built panel of 234 or 445 proteins that is **not
comparable** to the virulence-factor benchmarks in § 1.2.

---

## 1.2 Related work, and where this sits

### The virulence-factor classifier line

The established comparison is a sequence-classification line built on one dataset. **DeepVF**
([Briefings in Bioinformatics 22(3) bbaa125, 2021](https://academic.oup.com/bib/article/22/3/bbaa125/5864586))
assembled 3,576 virulence factors and 4,910 non-VFs, held out **576 of each** as an independent test
set, and reported **AUC 0.896**. **DTVF** ([Genes 15(9) 1170,
2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11430887/), ProtT5 with an LSTM–CNN dual channel) reuses
that pool and reports **AUROC 0.9208**. ⚠️ *DTVF states the shared 3,576/4,910 pool and cites DeepVF for
it but never states its own split, so "0.92 on the 576/576 benchmark" is an inference from DeepVF's
construction and is written here as one.* **DeepVIC** ([Bioinformatics Advances 6(1) vbag237,
2026](https://academic.oup.com/bioinformaticsadvances/article/6/1/vbag237/8762933)) is larger again:
ProtBert-BFD embeddings over **33,456 VFs**, **AUROC 0.954** on a 13,384-sequence holdout, plus
multiclass assignment to **14 VFDB categories** at 0.838 accuracy.

🔑 **The comparison we care about is not the AUROC.** DeepVIC carries a published 14-category class axis
over 12,989 annotated VFs and runs **no leave-one-category-out evaluation**; its generalization check is
recall on two organism-specific positive-only sets (71.4%, 45 of 63). The largest virulence-factor
classifier in the literature therefore holds the class axis this work needs and does not ask the
class-level question this work exists to ask. An adjacent paper does ask a question of that shape —
leave-one-EC3-class-out over 161 subclasses, 47.7% EC1 recovery against a 14.3% baseline
([arXiv 2606.12209](https://arxiv.org/abs/2606.12209)) — and reports per-class variation with a
narrative explanation, but no quantity computable *before* the class is held out.

### Hazard work on protein foundation models

**SafeProtein** ([arXiv 2509.03487](https://arxiv.org/abs/2509.03487)) red-teams generation, reporting
**up to 70% attack success** against ESM-3. That is a generation-time question; ours is whether hazard is
linearly readable in a frozen representation, which exists without any adversarial prompt. ⚠️ Its
429-protein benchmark is not downloadable: we recovered 66 identities from the paper's text, and a second
group reports **275 pairs** from the same unreleased set
([VFUSE, arXiv 2606.10080](https://arxiv.org/html/2606.10080)).

**VFUSE** is the nearest neighbour in intent — Matryoshka BatchTopK sparse autoencoders on
RoseTTAFold3 and RFDiffusion3 activations, AUROC **0.877 ± 0.025** on a random split and **0.817 ±
0.102** under homology-clustered cross-validation. 🔑 Its control is **member-level** homology clustering
at 30% identity, not a held-out mechanism class, on **n = 275 pairs**. The class-level question, and a
pre-hoc predictor of which class fails, are not occupied by it.

### What the framing borrows

The margin statistic is not new. Predicting generalization from margin distributions is a named
programme (Jiang et al., [ICLR 2019](https://arxiv.org/abs/1810.00113); the NeurIPS 2020 PGDL
competition), and *k*-nearest-neighbour distance in embedding space as an out-of-distribution score is
established (Sun et al., [ICML 2022](https://arxiv.org/abs/2204.06507)). 🔑 **What is ours is the level
and the target**: the statistic applied at **class** level, **before** the class is trained on, to a
safety screen's **false-negative** structure — and the finding that prediction and repair dissociate.

---

## 2. The panel, the protocol, and the bimodal failure

**Panels.** v2 is frozen at **80 hazard proteins / 154 negatives** and carries every published figure
outside the sections named in § 2.5 of `docs/MECHANISM_GENERALIZATION.md`. v3 adds three
non-animal-target mechanism classes — bacteriocin, phage peptidoglycan hydrolase, *B. thuringiensis* Cry
toxins — with organism-matched negatives, reaching **149 / 296** and eleven holdout-eligible classes.

**Protocol.** Hold out an entire mechanism class, train a probe on the rest plus the training negatives,
calibrate a threshold on held-out negatives, and ask how much of the unseen class is still flagged. ESM-2
650M, mean pooling, logistic regression, five seeds for the published table and thirty for every
stability claim.

**Result.** Baseline separability is **AUROC 0.974 ± 0.014**. Per class, at 95% specificity:

| recovered | classes |
|---|---|
| **100%** | superantigen enterotoxin, clostridial neurotoxin, ADP-ribosyl AB toxin, RIP rRNA glycosidase |
| 69–80% | T3SS effector apparatus, pore-forming cytolysin |
| 35–50% | contact-dependent inhibition, the labelled virulence control |
| **🔴 10–21%** | **phage peptidoglycan hydrolase (n = 32), beta-lactamase (n = 14)** |

🔴 **Neither failure is a small-*n* artifact.** The phage class is the **largest** eligible class in v3,
and beta-lactamase is the largest in v2. At thirty seeds beta-lactamase is **15.5% [11.0, 19.9]** with
**7 of 30 splits recovering it at exactly 0%**, and it is the only class where plain Smith-Waterman
alignment beats the probe (**29.5%** against 15.5%). Profile methods do not: `phmmer` reaches 5% there
and `jackhmmer` 0%.

⚠️ **An aggregate of 0.974 is compatible with a screen blind to two families.** That is the first reason
this paper exists.

---

## 3. Eighteen criteria, and the ordering that matters

`docs/DETECTOR_CRITERIA.md` states eighteen criteria and scores this project's own probe. The full table
is there; the scorecard is **two fails, four partials, one mixed**.

🔑 **Criterion 1 is first because the other seventeen are measured through it.** The panel's negatives
split **178 train / 118 calibrate / 0 test**, so every false-positive figure in the original design was
measured on negatives the pipeline had already seen. Measured out of sample against 8,258 pool proteins,
200 seeds:

| nominal 5% | `np.quantile` | conformal |
|---|---|---|
| calibrate on pool, test on pool (exchangeable control) | 5.87% | 4.32% |
| **calibrate on panel, test on pool (the deployment case)** | **7.87% [7.57, 8.16]** | **5.98% [5.73, 6.23]** |
| the same, one representative per distinct protein name | **9.64%** | **7.14%** |

The conformal threshold is the better estimator and still overshoots, because panel negatives and pool
negatives are not exchangeable — three curated blocks against Swiss-Prot. 🔑 **The guarantee is voided by
design, and that is the point**: a deployed screen calibrates on negatives it curated and then meets
whatever arrives.

🔴 **And what arrives is not housekeeping proteins.** Run the same probe, with the same 118-point
calibration, against **4,218 non-toxin virulence factors** from VFDB — pathogen-produced, largely
secreted, the two features a provenance probe already reads at AUROC 0.818 and the two that the
controls below quantify:

| nominal 5% | `np.quantile` | conformal |
|---|---|---|
| benign pool, raw | 7.87% | 5.98% |
| benign pool, one per distinct name | 9.64% | 7.14% |
| **VFDB virulence factors, raw** | **46.70% [45.93, 47.47]** | **40.84% [40.01, 41.68]** |
| **VFDB, one per distinct VF group** | **38.68%** | **32.61%** |

**At a nominal 5% the probe flags two fifths of them**, +34.9 points on conformal and +25.5 on the
like-for-like distinct-unit comparison, every interval disjoint. In the deployment arithmetic this turns
605 alerts into **4,088** and precision from 1.21% into **0.18%**.

🔑 **This says what the screen separates.** Not toxins from other proteins, but **virulence-associated
pathogen proteins from benign housekeeping ones**. ⚠️ It does not say the probe is wrong to flag them —
these proteins are not benign in any operational sense — but it does say the 5% budget is not met against
them, and that a false-positive rate quoted against housekeeping proteins **does not transfer to the
population a screen would actually meet**.

🔴 **And two controls have since measured how much of that separation is not about virulence at all.**
Both hold the probe and its calibration fixed and vary only what the proteins are
(`docs/PROVENANCE_CONTROL_PREREGISTRATION.md`, `docs/LOCALIZATION_CONTROL_PREREGISTRATION.md`):

| control | benign comparison | share of the pool→VFDB gap |
|---|---|---:|
| **pathogen origin** — 106 pathogen-derived non-VFDB proteins | 6.98% → **22.08%** | **23.6%** |
| **localization** — benign *extracellular* vs the pool | 7.06% → **21.75%** | **23.0%** |

🔴 **The localization gradient is the larger finding of the two, and it is adverse.** Among benign
Bacteria and Archaea in the pool, a **secreted or surface-exposed** protein is flagged at **21.75%**
against **4.13%** for a cytoplasmic one — a **5.38×** ratio [5.13, 5.63], replicating at **5.15×** on
ESM-2 35M. ⚠️ **So the nominal 5% is not merely exceeded on virulence factors; it is exceeded fourfold
on ordinary benign secreted proteins.**

🟢 **That was measured on the 45% of the pool a curator had localized, and it survives dropping that
restriction.** Adding GO cellular-component terms and signal-peptide and transmembrane features takes
coverage to 66%, so the unannotated stratum falls from **55.0% to 37.0%** — under a ceiling the
original preregistration set at 50% **and breached**, which is why every earlier statement of this
result carried the caveat that the annotated contrast might not represent the pool
(`docs/LOCALIZATION_COVERAGE_PREREGISTRATION.md`). The intracellular stratum grows from 1,666 proteins
to **2,739**, and `R` is **5.18** and **5.86** — **4.97** and **5.35** once VFDB contaminants are
dropped from the strata. 🔒 **The caveat is lifted and the finding is unchanged.**

⚠️ **Length was the obvious alternative explanation, and adjusting for it costs about a quarter.**
Pooling across ten length deciles gives a Mantel–Haenszel ratio of **3.98 [3.87, 4.09]** on 650M and
**4.70 [4.44, 4.97]** on 35M against crude ratios of 5.18 and 5.86 — 🔒 **still above the frozen 3.0 on
both arms** (`docs/LENGTH_ADJUSTMENT_PREREGISTRATION.md`).

⭐ **The ratio is not constant in length, which is what reconciles three different estimates of it.**
On 650M it runs from **1.40** in the longest band to **11.30** in the middle:

| length | intracellular | extracellular | ratio |
|---|---:|---:|---:|
| 0–250 | 14.15% | 46.12% | 3.26 |
| 350–450 | 1.91% | 21.60% | **11.30** |
| 450–550 | 1.69% | 17.10% | 10.13 |
| 700+ | 12.18% | 17.08% | **1.40** |

Mantel–Haenszel weights by cell size and gives 3.98; 1:1 length matching reproduces the *extracellular*
length distribution, which piles up at both extremes where the ratio is lowest, and gives 2.62; the
crude estimate is a third weighting. 🔒 **They are three weightings of one varying quantity, not three
answers.** What no weighting changes: **the ratio exceeds 1.0 in every band on both arms** — there is
no protein length at which the gradient disappears.

🔴 **The flag rate is U-shaped in length inside both strata** — intracellular 14.15% → 1.69% → 12.18%,
extracellular 46.12% → 17.10% → 17.08%. ⚠️ **A monotone summary cannot see that**: length's AUROC for
predicting the flag is **0.437**, below 0.5, for a relationship this strong. The confound rule that
cleared length in the first place measured a monotone quantity, and would have cleared it however
strong the U was.

🔑 **The U is a directional property of the representation, and two mechanical explanations have been
eliminated.** It is not the special tokens: the include-specials mean displaces an embedding by
`[(cls+eos) − 2·m_res]/(L+2)`, five times more for a short protein than a long one, but that
displacement is **0.34% of the embedding norm** and only **2.1%** of it lies along the probe's decision
direction, and refitting on residue-only embeddings leaves every rate unchanged to two decimals
(`docs/POOLING_ARTIFACT_PREREGISTRATION.md`). It is not proximity to the training positives either —
the maximum cosine to a class-axis positive is **lowest** in the band flagged most.

⭐ **What it is**: the score factors exactly as `‖z‖·cos(z,w)·‖w‖`, and holding the magnitude at its
median leaves the U almost intact (**7.61** against an observed 8.38) while holding the **direction** at
its median **inverts** it — the short band falls to **0.00%** and the mid band rises to 20.69%
(`docs/SCORE_DECOMPOSITION_PREREGISTRATION.md`). Short proteins have the *larger* standardized norm,
38.28 against 30.62, so magnitude pushes them **away** from the threshold; what lifts them over it is a
**less negative cosine**, −0.0198 against −0.0339. ⚠️ **So ESM-2 places short and long bacterial
proteins nearer the virulence-factor direction, and a linear probe reads that as hazard** — a second
population for which the nominal 5% is wrong, carried by a different mechanism than localization.

🔴 **And the effect is a function of model size.** Running the same measurement on four ESM-2 arms
spanning 80× in parameters gives a strictly monotone U-index — **44.66 → 18.68 → 9.65 → 8.38** for 8M,
35M, 150M and 650M, Spearman **−1.00** against log parameters (`docs/SIZE_SCALING_PREREGISTRATION.md`).
🔒 With four arms the smallest achievable one-tailed *p* is **0.0417**, which is what this returns, and
the study was frozen saying so.

⚠️ **It closes from below, which is the opposite of reassuring.** The short-band rate is **flat** across
all four arms (12.69–16.13%); what rises is the **mid-band** rate, **0.36% → 1.69%**, a factor of 4.7.
🔑 **Larger models do not flag short proteins less — they flag mid-length proteins more.** The ratio
improves with scale because the denominator gets worse. **"Use a bigger model" is not what this
licenses.**

| arm | short | mid | U-index | `R` |
|---|---:|---:|---:|---:|
| esm2_8M | 16.13% | **0.36%** | **44.66** | 6.93 |
| esm2_35M | 12.69% | 0.68% | 18.68 | 5.67 |
| esm2_150M | 14.51% | 1.50% | 9.65 | 4.50 |
| esm2_650M | 14.15% | **1.69%** | **8.38** | 5.15 |

⚠️ The localization ratio `R` does **not** follow strictly — 650M sits above 150M, Spearman −0.80 — and
the preregistration predicted it would be the noisy outcome and said why.

⚠️ **Part of that ratio was contamination, and the corrected figure is the one to quote.** The benign
pool holds **133 exact VFDB sequences** (1.61%), and they are **differentially** distributed across
exactly this contrast — 5.67% of the extracellular stratum against 1.20% of the intracellular one.
Removing every one of them from fitting, calibration and evaluation
(`docs/DECONTAMINATION_PREREGISTRATION.md`) leaves **R = 5.07 [4.91, 5.24]** on 650M and **4.35 [4.09,
4.61]** on 35M — 🟢 **the verdict holds well clear of its frozen 3.0 floor on both arms**, and
⚠️ **the published ratio was inflated by 5.7% of itself on 650M and 15.6% on 35M.** The decontaminated
extracellular rate is **20.90%**, so the fourfold overshoot of the nominal budget is unaffected.

🔒 **What survives.** Against a *localization-matched* benign baseline of 21.75% rather than the pool's
7.06%, VFDB virulence factors are still flagged at **70.91%**, so **roughly three quarters of the
separation is not localization** — the same fraction provenance left over.

⚠️ **The two shares must not be added, and the joint decomposition says what they actually buy.**
Pathogen-derived proteins are themselves enriched for secretion, so the controls overlap by
construction. Crossing them (`docs/JOINT_DECOMPOSITION_PREREGISTRATION.md`) gives a clean 2×2 on the
benign side:

| benign cell | *n* | ESM-2 650M | ESM-2 35M |
|---|---:|---:|---:|
| pathogen species × extracellular | 171 → **201** | **35.09%** | **27.39%** |
| pathogen species × intracellular | 597 → **679** | 6.33% | 4.82% |
| benign species × extracellular | 444 → **414** | 13.17% | 8.84% |
| benign species × intracellular | 1,049 → **967** | **2.36%** | **2.03%** |

🟢 **The two compose within a band frozen in advance** — ratio of ratios **0.692 [0.649, 0.735]** on
650M and **1.133 [1.048, 1.219]** on 35M. Localization is worth ≈5.5× and provenance ≈3.5×, each
roughly constant across the other's levels.

⚠️ **These replace an earlier, tidier pair.** Deciding "pathogen-derived" by whether the organism's
name appears in VFDB's species list misses every pathogen UniProt has since **renamed**, and rebuilding
the factor on canonical species (`docs/TAXID_PROVENANCE_PREREGISTRATION.md`) moved **5.05%** of
eligible proteins — **112 genuine pathogen proteins were in the benign cells**. The ratios were 1.009
and 1.327 under the name test. 🔴 **"Almost exactly independent" was a property of the defective
factor**: on the corrected one 650M's interval **crosses the band floor**, and the two arms now
straddle 1.0 in opposite directions. The composition claim survives its frozen band and is narrower
than it looked. 🟢 The correction strengthens provenance itself, as predicted in advance: with
localization held fixed it rises from **2.63× to 3.47×**.

⭐ **Together they span 43.9% of the pool→VFDB gap on 650M and 34.1% on 35M** — measured, not summed;
**23.6% + 23.0% is not a quantity this repository has.** 🔑 **So the majority of the separation belongs
to neither confound.**

🟢 **And a matched test attributes most of that majority to VFDB registration.** 133
pool proteins are exact VFDB sequence matches, which makes them the only population here that is both a
virulence factor *and* a Swiss-Prot entry carrying UniProt localization; they are held out under the
decontaminated fold (`docs/MATCHED_VFDB_PREREGISTRATION.md`). Holding pathogen origin **and**
extracellular localization fixed and varying only membership:

| | *n* | ESM-2 650M | ESM-2 35M |
|---|---:|---:|---:|
| benign × extracellular | 171 | 37.52% | 26.90% |
| ⭐ **VFDB × extracellular** | **38** | **73.77%** | **40.70%** |
| *reference*: full VFDB set, non-circular | 3,676 | 74.09% | — |

🔴 **That reference excludes 542 of the 4,218.** They share a sequence with one of the 746 class-axis
positives, so scoring them scored the probe's own training data; `src/74` screened this set against the
*panel* positives, which is what study A1 needed, and nothing screened it against the class-axis
positives that came later. Including them put the reference at 76.95% and made every share-of-gap
figure in this paper about 4.3% too small.

⚠️ **Those 38 are selected against the hypothesis**, which is what made this version small and its
verdict only *partial* (C/A 1.974 and 1.497 against a frozen 2.0): the pool's build query excludes the
*Virulence*, *Toxin*, *Cytolysis*, *Hemolysis*, *Bacteriocin* and *Bacteriolytic enzyme* keywords, so
every bridge protein is a virulence factor **UniProt declines to call virulent**.

🟢 **Removing that selection settles it.** Every VFDB record carries a RefSeq accession, so UniProt's
ID-mapping service gives the 4,218 admitted negatives a Swiss-Prot identity — **including** the
virulence-annotated members the bridge could not contain (`docs/MATCHED_VFDB_SCALEUP_PREREGISTRATION.md`).
After dropping class-axis positives and any protein whose length differs from its mapped entry, 778
remain:

| | *n* | ESM-2 650M | ESM-2 35M |
|---|---:|---:|---:|
| benign × extracellular | 171 | 37.52% | 26.90% |
| ⭐ **VFDB × extracellular** | **264** | **89.95%** | **69.41%** |
| benign × intracellular | 597 | 6.87% | 5.09% |
| ⭐ **VFDB × intracellular** | **134** | **54.08%** | **34.00%** |

**C/A = 2.418 [2.339, 2.497]** and **2.617 [2.495, 2.740]** — 🟢 **supported on both arms**, not
partial. And `D/B` = **7.988** and **6.875** on cells of 134 against a floor of 100: **within the
intracellular stratum, where neither secretion nor surface exposure can be doing the work, a VFDB
protein is seven to eight times more likely to be flagged than a benign one from the same kind of
organism.**

🔴 **That effect does not track hazard, and calling it a hazard signal would be wrong.** Splitting the
same population by whether a category is a mechanism of harm
(`docs/HAZARD_VS_MEMBERSHIP_PREREGISTRATION.md`) gives **2.410** for the *non-hazardous* categories —
Motility, Nutritional/Metabolic factor, Regulation, Stress survival — against **2.356** for the
partly-hazardous ones, on 650M. ⭐ **Motility alone is the most-flagged group on both arms** (2.665 and
2.682), at a raw rate of **99.05%**: essentially every flagellar protein is flagged, and flagella are
surface structures with no toxic function.

🔒 **So the finding is that the probe responds to VFDB *registration*.** ⚠️ And the population never
contained exotoxins — `src/74` built it from *"VFDB setA, non-Exotoxin records"* — so every claim in
this section is about **non-toxin** virulence factors. **The contrast is real and large; what it is
evidence for is narrower than membership-beyond-confounds suggested.**

🔴 **One earlier framing does not survive this.** Expressing membership's effect as a fraction of the
distance to the full-VFDB rate gave 99.1% at bridge scale and gives **143%** and **188%** here — a
fraction cannot exceed 100% unless the reference is not a ceiling, and it is not: the full-VFDB rate
averages over a **mixture of localizations** (89.95% extracellular down to 54.08% intracellular), which
a localization-matched cell is not bounded by. **The 99.1% meant the matched cell landed just under the
whole-set average, not that membership explained 99.1% of anything.** ⚠️ The contrast is unharmed; the
normalisation was.

⚠️ 🔴 **And none of this softens the localization finding** — in these same tables a benign secreted
protein is flagged at 37.5% against 6.9% cytoplasmic, so the nominal 5% is still wrong by a factor of
seven for secreted proteins.

🔴 **Criterion 12 fails, and the failure was expensive twice.** The first preregistration carried a floor
and no ceiling. Adding ceilings to the second one is what made its NOT SUPPORTED verdict readable — and
what later exposed that the same comparison was confounded by annotation provenance (§ 6).

🟢 **Criterion 14 is where the corrections log earns its place.** Baselines reported because they make the
model look worse: amino-acid composition alone reaches **AUROC 0.754**; shuffled labels **0.506**; a probe
trained on lab-strain provenance with the hazard label ignored reaches **0.818**, and the organism label
agrees with the hazard label on only 53.4% of v2. A label-free typicality proxy — mean cosine to the
benign pool's centroid — reaches **−0.746 against recovery at *p* = 0.0034** on the canonical arm, which
we had not tested until a paper we had missed prompted it.

---

## 4. Margin: the one result that replicated

Both failures share a geometric property computable **before** the class is trained on. For a class *C*,

    margin(C) = mean over members of [ max cosine to a hazard outside C − max cosine to any negative ]

| | |
|---|---|
| ranks classes by recovery | **Spearman +0.894**, permutation *p* = **0.00015** |
| the two lowest-margin classes of twelve | **exactly the two failures** (chance 1/66) |
| beats each of its own parts | nearest-positive alone, nearest-negative alone |
| class size runs the other way | −0.564 |
| **both failing classes** | **negative margin**: closer to a benign protein than to any trained hazard class |

🟢 **It transfers.** The ordering holds in **fourteen model arms** — ESM-2 at five scales, ESM-C at three,
ESM-3, ProtT5, SaProt, and two pooling variants — and, separately, across **fifteen re-poolings of one
arm** (*ρ* +0.536 to +0.941, **14 of 15** significant at *p* < 0.05, **12 of 15** placing beta-lactamase at
the margin floor). Those are two different kinds of variation and it survives both.

🔴 **It is a ranking, not a calibration.** Margin over-predicts beta-lactamase recovery by **34 points**
and errs optimistic on every out-of-sample class. And the member-level version of the same claim was
preregistered, frozen, and **failed on two external panels** — which is why the class-level result is
stated as a triage signal for deciding which mechanism families need their own validation, and not as a
deployment score.

⚠️ **Prediction and repair dissociate.** Removing the benign neighbours the geometry implicates repairs
**8 to 11%** of a failure the geometry predicts at *ρ* = +0.894. Knowing which class will fail does not
tell you how to fix it.

---

## 5. Nothing moved the precision

This section is the one we would omit if we were arguing for a method.

**What was varied.** Four classifier heads on fourteen arms at thirty seeds (logistic is the *worst* head
on 7 of 14 arms and beaten on 12 of 14 by a median of 5.1 points — every figure above is therefore close
to a lower bound). Five model scales. Two pooling variants, then **fifteen label-free reductions** that
keep whole residues or windows intact, then a **supervised residue ranking** fitted inside each fold.
Ensembles of alignment with embedding, and of two language models.

**What happened.** One reduction — the most deviant 25-residue window — moves both failing classes:
beta-lactamase **15.5% → 35.0% [31.6, 38.4]** and the phage class **12.3% → 27.3% [23.3, 31.3]** at thirty
seeds, intervals disjoint, the first pooling choice here whose interval clears alignment's 29.5%. It costs
**43.8 points** on superantigens and **9.5** on the panel mean.

🔴 **And it does not survive a change of representation.** On ESM-2 35M the same reduction costs
beta-lactamase 9.5 points and **no reduction clears its own control**, while the superantigen cost
replicates at −45.2. The supervised ranking behaves the same way: it raises the panel mean on 650M and
**no *k* raises it on 35M**. 🔑 **Every gain in this line is one representation; every cost is general.**

**The number that settles it.** In a deployment of ten thousand sequences at a one-in-a-thousand hazard
rate, with recall taken from the v3 panel mean and the false-positive rate measured out of sample:

| | conformal FPR | recall | alerts | real hits | **precision** |
|---|---|---|---|---|---|
| mean pooling (the reductions' control) | 5.96% | 73.1% | 603 | 7.3 | **1.21%** |
| the 25-residue window | 6.48% | 63.1% | 654 | 6.3 | **0.97%** |
| a 9-residue window | 4.30% | 54.0% | 435 | 5.4 | **1.24%** |
| windowed max | 5.44% | 69.5% | 550 | 6.9 | **1.26%** |

⚠️ *This table's control is the **residue-only** mean, which is what the reductions are compared against;
the published canonical arm averages the special tokens in and gives 5.98% and 605 alerts. The two differ
by 0.014 points of false-positive rate and two alerts, which is why § 1's figures say 605 and this table
says 603.*

🔴 **Precision is flat.** The reduction with the lowest false-positive rate buys it with nineteen points of
recall — a move along the same ROC curve, obtainable by raising a threshold and requiring no new method.
**A change that lowers the false-positive rate and the recovery together has not improved calibration.**

---

## 6. Two repairs that grew the wrong side

The most useful thing this project learned about itself is not about protein models.

**The first repair.** A preregistered metric extension compared the panel's masked-prediction constraint
against **four** benign controls. Three of its four AUROC tests were unusable at that *n*, which the
preregistration recorded as one finding about its own design. So we built the set it asked for: **60**
reviewed Swiss-Prot enzymes carrying `Active site` features, every frozen exclusion holding, the build
reproducing byte-identically. The verdict held — the controls exceed the panel, so the axis measures
positional constraint and not hazard — and then a review found the comparison **annotation-confounded**:
the controls' positions are UniProt `Active site` features exclusively, and **17 of the panel's 74
annotated positions (23%)** are. Restricting both sides to the same annotation type leaves **4 usable
panel proteins**, halves the gap, and gives an AUROC interval that **covers 0.50**.

🔴 **A second defect surfaced in the same pass and had been documented nowhere: the panel's hazard arm
contains three BSL-1 entries with no hazard designation** — barnase, colicin E2, and Cas9, the last
described in the annotation file itself as a widely used research tool. Two of the four
annotation-matched survivors are barnase and Cas9. With both restrictions applied the hazard arm is
**ricin and one phospholipase**.

**The second repair.** A gate on that same extension had an AUROC tolerance with no power at four
controls. The amendment that retired it diagnosed the defect precisely — *"a ±0.05 tolerance on a
statistic whose null standard deviation is 0.167 is not a threshold, it is a coin weighted against
passing"* — named the defect class as *"a threshold carried across a change of sample size without a power
argument"*, and predicted the tolerance would become meaningful above thirty controls. It does not. An
AUROC's precision is set by the **smaller** group; the null standard deviation floors near **0.077** and a
clean pipeline clears the tolerance under half the time **at any number of controls**.

🔑 **Both repairs grew the negative side. Both limits live in the panel.** Fifteen proteins, seventy-four
annotated positions, seventeen confirmed by a common source, four proteins usable for the comparison. One
resource blocks three tests, and it was never the thing being bought.

⚠️ **The generalizable claim is small and we think it is real: an evaluation can be improved in the wrong
direction, and writing the ceiling down before the run is what reveals which side is binding.** Both times
the diagnosis was in the record before the repair was built; both times it named the wrong sample size.

---

## 7. What this does not claim

- **Not a better classifier.** DeepVIC reports AUROC 0.954 on a 13,384-sequence holdout from 33,456 VFs.
  Our figures are on a self-built panel of 234 or 445 and are **not comparable**.
- **Not novel on homology control.** Homology-clustered evaluation is established practice.
- **Not a competence boundary.** The held-out class is removed from the *probe's* training, not from the
  foundation model's pretraining, and every class here is in the public databases these models saw.
- **Not deployment-ready.** § 10.8 of the long write-up puts it quantitatively: every specificity above
  0.9915 is extrapolation on this panel, and calibrating a one-in-ten-thousand budget would need a negative
  set roughly **850 times** larger. Common Mechanism and SecureDNA run in production; this does not.
- **Not evidence that hazard is what is being detected.** A provenance probe with the hazard label ignored
  reaches 0.818.
- **Not a controlled comparison of pretraining corpora.** The arms differ in corpus, architecture and
  scale at once.

---

## 8. Discussion

The gap this paper is named after — 0.974 separability, about 1% deployment precision — is not a defect of
one probe. It is what a class-imbalanced screen with a calibration set of 118 points looks like when the
false-positive rate is measured on negatives it has not seen. Three practices made it visible, and all
three are cheap.

**Write the ceiling, not only the floor.** A preregistration with a floor can only confirm. Both of ours
returned NOT SUPPORTED, and the second returned it *legibly* because its ceiling said in advance what a
benign control exceeding the panel would mean.

**Run every geometric claim across representations.** This rule killed three findings in two days: a
typicality baseline at −0.746 and *p* = 0.0034 that became +0.021 and *p* = 0.53 on a second arm; a
window reduction worth +19.5 points that cost 9.5 on the same arm's smaller sibling; and a supervised
ranking that raised the panel mean on one arm and on no *k* of the other. 🔑 **The rule was worth more
than the findings it killed**, and the cheapest arm in the project did all three kills for about twenty
minutes of compute each.

**Keep the log append-only, including the entries that hurt.** Sixty entries, several retracting our own
conclusions — a benchmark attributed to the wrong paper, a false-positive check that turned out to be
4/61 by construction and therefore identical for random scores, a cross-reduction comparison made across
geometries whose cosine distributions differ by a factor of three. ⚠️ **The pattern across those is one
thing, not three: we checked whether numbers were stable and not whether they measured what we said.**
Seeds, intervals and replication got attention; a definitionally constant quantity, an
annotation-provenance mismatch and an incomparable scale did not, and no number of additional seeds would
have surfaced any of them.

What we would want next is not another representation. It is **more panel**: mechanism classes from a
published ontology rather than hand curation, and catalytic positions from one source. VFDB supplies the
first — fourteen categories, of which **Exotoxin is one and 4% of the records** — and that number is its
own warning, because a virulence-factor axis is not a toxin axis, and adopting it changes what the screen
is for.

### Where this sits against data-governance proposals

The **Biosecurity Data Level** framework (Bloomfield, Berke, Pannu *et al.*,
[arXiv 2602.08061](https://arxiv.org/abs/2602.08061), endorsed by more than 100 researchers at the 50th
anniversary Asilomar conference) proposes tiered access control over the data AI models are trained on.
🔒 **It classifies data, not pathogens** — sequence data, functional annotations, functional assays,
protein–protein interaction data and model weights — and the tiers are set by what a model could learn
from them: **BDL-3** covers human-infecting viral families' *transmissibility, virulence, immune evasion
and resistance to medical countermeasures*; **BDL-4** covers data on pandemic-capable viral variants
showing those properties **enhanced relative to wild-type**.

⚠️ **This work is bacterial and sits outside those tier definitions**, which are written for viral
families. It is not an evaluation of BDL and does not test it.

🔑 **What it does bear on is an assumption underneath any such control**: that a model trained on
annotated pathogen data acquires a representation of the properties that made the data sensitive. This
paper measures that directly for one bacterial analogue, and the answer is unflattering. A linear probe
on ESM-2 separates **VFDB-registered proteins from matched benign ones at 2.3–2.7×** with provenance and
localization held fixed — but the separation **does not track harm**: the non-hazardous categories score
2.410 against the partly-hazardous 2.356, and **Motility — flagella, no toxic function — is the
most-flagged group of all at a 99.05% rate**. What the representation does respond to strongly is
**secretion** (5×) and **protein length** (a U-shaped effect that strengthens monotonically as the model
shrinks).

🔴 **The limit on that reading matters more than the finding.** A linear probe failing to separate hazard
does **not** show the model lacks the information; it shows this probe, on this representation, does not
expose it linearly. **Absence of linear separability is not absence of capability**, and a data-control
argument that leaned on this result in either direction would be leaning on the wrong thing. What the
result does support is narrower and still useful: **"the model was trained on virulence annotations, so
it represents virulence" is an assumption that can be checked, and here it did not hold.**

---

## 9. Reproduction

Every figure above is recomputed from an artifact by `src/22_claims_audit.py`, which currently checks
**90 claims** and fails if any number in a public document drifts from the artifact that produced it. The
panels, the protocol and the per-class tables are in `docs/MECHANISM_GENERALIZATION.md`; the criteria and
the scorecard in `docs/DETECTOR_CRITERIA.md`; the preregistrations in
`docs/MUTATION_EXTENSION_PREREGISTRATION.md` and `docs/NEGATIVE_EXPANSION_PREREGISTRATION.md`; and the
dated record of every number that moved in `docs/DATA_CORRECTIONS.md`.

🔴 **Status, 2026-09-28: this is an outline of an argument, not a submittable paper.** § 4 rests on a
single positive result — margin — that has **never been tested on classes this project did not define**,
and § 5 reports that nothing improved the screen.
`docs/EXTERNAL_CLASS_AXIS_PREREGISTRATION.md` freezes the external test, on VFDB's 13 eligible categories over 740 redundancy-deduplicated representatives, and
states in advance that a null there means §§ 4 and 6 of this draft must be rewritten as a panel-specific
observation. **This manuscript stays a draft until that returns, either way.**

⚠️ **It is a draft and is on the audited surface**, which means a figure here that drifts
from its artifact fails the gate in the same way a figure in the write-up does. It is assembled from
sections `paper/0*.md`; edit those, not this file.
