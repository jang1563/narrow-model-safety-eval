# An embedding hazard screen at AUROC 0.974 and precision 1.2%: what eighteen criteria and 124 corrections found

**Draft, 2026-09-28.** Every figure below is recomputed from an artifact by
`src/22_claims_audit.py` and pinned to the sentence that states it; the gate currently checks 152 such
claims. Section references without a document name are to `docs/MECHANISM_GENERALIZATION.md`.

---

## Abstract

A protein-language-model embedding separates a curated hazard panel from its negatives at **AUROC
0.974 ± 0.014**. Screened against 8,258 held-out benign proteins at a nominal 5% false-positive budget
the same probe returns **7.87%** with the interpolating `np.quantile` estimator and **5.98%** with the
conformal one, and translated into a deployment of ten thousand sequences at a one-in-a-thousand hazard rate
the conformal figure produces **605 alerts of which about 7 are real — a precision of
1.2%**. ⚠️ **The two estimators are named because the abstract's two numbers use different ones**: 7.87% is
the interpolating estimator and 605 alerts is the conformal one, so 7.87% × 9,990 gives 786 and not 605.
🔒 The gap is a calibration-size effect — it is 2.9 points on the 118-point panel fold and **0.6–2.4%** on
the 484-point fold every contrast in this paper uses, where the headline ratios are unchanged
(`docs/CONFORMAL_CONTRAST_PREREGISTRATION.md`). This paper is about the distance between those two numbers, and about what it takes to see it.

We state **eighteen criteria** a hazard-screening claim should meet, score our own probe against them,
and report the result: **two fails, four partials and one mixed**. We then show that each criterion was
written because it caught something. The evidence is a **dated, append-only corrections log of 124
entries** in which a published number moved, including entries that retract our own findings.

Five results matter beyond this panel.

**First, the aggregate is uninformative and the per-class picture is bimodal.** Holding out an entire
mechanism class, recovery runs from **100% to 12%**: three classes are fully recovered without ever being
trained on, while a 32-member phage peptidoglycan hydrolase class reaches **12.3%** and beta-lactamase
**21.0%** — on one panel, at thirty seeds, which is the internally consistent comparison. ⚠️ **The figures
this project published for two months were 10% and 15.5%, and each is the *minimum* of its row in a
four-cell panel × seed-count grid that was already computed and never printed** (§ 3). ⚠️ In that grid seed
count moves nothing by more than 1.7 points while panel moves three classes by 19 to 40, so per-class
recovery is a property of the panel. ⚠️ Member-level
resampling puts beta-lactamase at **[0, 36]**, so the *bimodality* is what survives — 100% and [0, 36] are
disjoint — while the precision of any single low number is limited by class size. An AUROC of 0.974 is
compatible with a screen that cannot see two whole families.

**Second, a quantity computable before training predicts which class will fail.** Class-level
**margin** — nearest other-class hazard minus nearest negative in embedding space — ranks classes by
recovery at Spearman **+0.894** (permutation *p* = 0.00015), places both failures at the bottom, and
survives **fourteen model arms** and **fifteen re-poolings** of one arm (*ρ* +0.536 to +0.941, 14 of 15
significant). It replicated across **every internal variation we tried**, and it is the only result here
that did. 🔒 **It now also replicates on a published benchmark with someone else's labels** — *ρ* =
**+0.947** and **+0.886** over DeepVIC's fourteen curator categories and 12,989 proteins, both arms, both
clearing a floor frozen in advance (§ 4). 🔒 **And it is not a fact about protein language models**: the
identical fold on **20 Newsgroups** with a frozen sentence encoder orders **13** held-out text classes at
*ρ* = **+0.951** and **+0.813** at the two classifier settings, both clearing the same floor, so what the
quantity indexes is frozen-representation geometry and proteins are one domain of it — ⚠️ **on one text
encoder, which is one arm.** ⚠️ **Its external record is otherwise mixed and is three further
tests.** The **member-level** version was
preregistered and **failed** externally. The **class-level** version — the one this paper claims — was
preregistered on **VFDB's 13 categories** and came back at *ρ* **+0.681** (*p* = 0.0059), where a frozen
confound rule then fired and made it **uninterpretable**; a targeted follow-up holding organism identity
constant reinstated it **as an ordering claim only**, at *ρ* **+0.375** and **+0.411** with **8 of 10**
species strata positive on both arms (§ 4). ⚠️ It is also **mis-calibrated** — over-predicting
beta-lactamase recovery by 34 points.

**Third, and the finding we did not want: nothing we tried moves the screen's precision.** Four
classifier heads, fourteen representations, three panels, fifteen pooling reductions and a supervised
residue ranking later, deployment precision sits between **0.97% and 1.26%** across reductions — and
**0.76% to 1.26%** once the two arbitrary choices in the arithmetic are varied too: whether the
false-positive rate is the conformal or the interpolating estimate, and whether recall is averaged over
**classes** or over **proteins**. ⚠️ **The headline 1.2% is the most favourable corner of that box**, and
the flatness is what survives all three axes.
One reduction moves one of the two failing classes — a consistent **+5 of 32** members on the phage class,
against a net **+1 of 14** on beta-lactamase, where a paired member-level test finds mostly swapping rather
than gain — and pays for it in the negatives'
scores exactly as a fixed-budget argument predicts, and does so **only on one of two representations
tested**.

**Fourth, the screen's benign-side behaviour is graded by protein localization, and that does not go
away.** Among benign bacterial and archaeal proteins, a secreted or surface-exposed one is flagged
**5.07×** as often as a cytoplasmic one; the ratio survives expanding annotation coverage, adjusting for
length, residue-only pooling, and an independent protein pool where it **grows**. Two metadata fields with
no sequence information — localization stratum and log length — reach **AUROC 0.8096** on an analogous
virulence task against the embedding's 0.9323, and add **+0.0007** on top of it. And the obvious
mitigation fails: after a **provably exact** linear erasure of the localization axis, the ratio is still
**4.79 [2.75, 9.29]**, because erasure equalizes class means while the screen fires on a tail.

**Fifth, what the screen responds to is database registration rather than hazard, and that now has an
external test.** On a hazard axis set by UniProt curators rather than by us — the Swiss-Prot Toxin,
Hemolysis, Cytolysis, Enterotoxin and Neurotoxin keywords — **31** curated toxins paired to
length-matched non-toxin members of the same database, with **both** groups held out of a single fit and
the entire exotoxin category held out as well, are ordered at **chance**: AUROC **0.4599** on ESM-2 650M
and **0.5247** on 35M, inside a registration band of 0.40–0.60 frozen before the study against a hazard
floor of 0.70. The screen flags **three quarters** of those toxins, so it is not blind to them; it simply
does not rank them above comparable non-toxins. ⚠️ The test rests on **31 pairs**, because only **30.6%**
of the population carries a reviewed UniProt entry — 🔒 **and the same ordering test on DeepVIC's axis
gives 0.4457 and 0.5047 on 828 pairs**, 26.7× the sample, inside the same band. ⚠️ **The one
arm whose interval excludes chance carries no model**: twenty-one amino-acid features reach 0.5553
[0.5204, 0.5919], so what ordering ability exists here is compositional. 🔴 **And the between-category
form of the same claim does not replicate.** On DeepVIC's published axis, where Exotoxin and Motility hold 1,126 and 1,121
members without any choosing, the length-matched recovery difference is **+6.04pp** on 650M and
**−5.30pp** on 35M — both intervals excluding zero, against a reading rule frozen before the run that
required both arms to agree. The disagreement is **exotoxins specifically**, which swing 13.7 points
between the two representations while the fourteen-category means differ by 0.3. **The screen does not
order toxins above matched non-toxins inside a fold; whether it recovers a held-out toxin category more
or less than a non-toxin one depends on which representation you ask.**

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

🔑 **And the published AUROC is substantially length-separable.** DeepVIC's repository ships its
sequences, so the benchmark can be scored with **no sequence semantics at all** (`src/122`, `src/123`;
bands frozen in `docs/DEEPVIC_BENCHMARK_PREREGISTRATION.md` before either was written). Its positives'
median length is **549** against its negatives' **268** — a **2.05×** ratio. On the same 66,912
sequences, **log length alone reaches AUROC 0.7547**, and **log length plus the twenty amino-acid
frequencies reaches 0.8705** — twenty-one numbers per protein, no order, no position, no model, and
**91.3%** of the published **0.954**. The embedding's margin over trivial features is **+0.0835**.
⚠️ **Not on DeepVIC's partition**: it states 80:20 but does not publish the split, so this is our own
80:20 at five seeds, on a holdout of **13,382** against their 13,384, and every seed clears both frozen
floors. 🔒 **And it is not circular** — 239 of this project's 746 class-axis representatives appear in
DeepVIC, **32.0% of ours but 0.71% of theirs**, and removing them moves these figures by **−0.24pp** and
**−0.08pp**.

🔑 **That is the same shape as § 5's finding on our own panel** — two metadata fields with no sequence
information reaching **0.8096** against this project's embedding at 0.9323 — now measured on the largest
published benchmark in the line instead of on a self-built panel of 445.

⚠️ **The same length difference truncates the two sides of the benchmark unequally.** ESM-2 cuts at
**1,022** residues, and on DeepVIC's own sequences that reaches **11.07%** of the class-labelled
positives against **0.90%** of the negatives — a **12×** asymmetry — rising to **22.02%** for the
**Exotoxin** class against **1.52%** for **Motility**. 🔑 **This is a property of the benchmark, not of
any model evaluated on it**: any sequence model with a context limit near 1,000 residues sees a
systematically larger fraction of the positive side cut, and the categories it cuts hardest are the
ones whose members are longest. It is why § 6's comparison on that axis is reported length-matched
rather than over full membership
(`docs/DEEPVIC_BENCHMARK_PREREGISTRATION.md` amendment 1).

🔑 **The comparison we care about is still not the AUROC.** DeepVIC carries a published 14-category class
axis over 12,989 annotated VFs and runs **no leave-one-category-out evaluation**; its generalization check is
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

🔑 **The surrounding literature predicts a different object, and the difference is consistent across it.**
Margin-based generalization prediction measures margins at the hidden layers of a network **already
trained on the task** and predicts an aggregate generalization gap across models and hyperparameters
(Jiang et al.; [arXiv 2405.17445](https://arxiv.org/abs/2405.17445)). Representation-geometry work on
frozen encoders predicts **aggregate** accuracy: effective manifold dimensionality and utility against
out-of-distribution accuracy under distribution shift
([arXiv 2603.01879](https://arxiv.org/abs/2603.01879)). The class-subset line asks **which subset of
training classes** to use for best overall performance, not which held-out class will fail
([arXiv 2403.03569](https://arxiv.org/abs/2403.03569)); and the leave-one-out-error line is
leave-one-**example**-out in the kernel regime, again against overall generalization
([arXiv 2203.03443](https://arxiv.org/abs/2203.03443)). ⚠️ **Two results in that literature are reasons
for caution rather than support.** Global geometric statistics — participation ratio, isotropy, local
isotropy — have been reported as **non-predictive** on a compositional-binding target (*r* = −0.00, 0.18
and 0.05) where a *functional* quantity, the input-output Jacobian's effective rank, reached *r* = 0.65
([arXiv 2602.03282](https://arxiv.org/abs/2602.03282)); and frozen-embedding separability was found **not
stable** as a training-free predictor of a fine-tuning label budget across 35 tasks and three encoders.
🔒 **Margin is neither of the quantities those two papers tested**: it is not a global summary of the
embedding cloud and not a derivative of the encoder, but a **local quantity relative to a class
partition** — which is the one cell of that design space with a positive result here, and the reason § 4
reports it as an ordering and not a score.

🔑 **The closest work by setup leaves the predictive question open.** A frozen-probing analysis of five
genomic language models reports exactly this paper's metric — recovery of fine-tuned performance by a
frozen probe — and finds it strongly task-dependent (95–100% on promoter tasks against 60–88% on
splice-site), concluding that local biological signal is "partially present … but is not always
accessible" through pooled embeddings ([arXiv 2608.05329](https://arxiv.org/abs/2608.05329)). ⚠️ **It
reports no quantity that predicts which task will be the inaccessible one, and no model-free baseline** —
which are the two things § 1.2's first subsection and § 4 are about.

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

**Result.** Baseline separability is **AUROC 0.974 ± 0.014**. Per class, at 95% specificity — ⚠️ **this
table is v2 at five seeds for nine classes, with bacteriocin, phage and Cry from v3, and it does not mark
which row is which**:

| recovered | classes |
|---|---|
| **100%** | superantigen enterotoxin, clostridial neurotoxin, ADP-ribosyl AB toxin, RIP rRNA glycosidase |
| 69–80% | T3SS effector apparatus, pore-forming cytolysin |
| 35–50% | contact-dependent inhibition, the labelled virulence control |
| **🔴 10–21%** | **phage peptidoglycan hydrolase (n = 32), beta-lactamase (n = 14)** |

🔴 **A single-panel table exists and was never published, so here it is.** All twelve classes on **v3 at
thirty seeds** (`results/v3/supervised_pooling_v3_30seeds.json`, leakage note: the direction is fitted on
positives excluding the held-out class plus the training negatives only):

| class | n | recovered | sd | splits at exactly 0% |
|---|---:|---:|---:|---:|
| phage peptidoglycan hydrolase | 32 | **12.29%** | 6.30 | 0 |
| beta-lactamase | 14 | **20.95%** | 12.99 | **2** |
| virulence-associated non-toxin | 10 | 32.00% | 13.75 | 0 |
| contact-dependent inhibition | 4 | 72.50% | 7.63 | 0 |
| Cry insecticidal | 22 | 79.24% | 16.87 | 0 |
| T3SS effector apparatus | 10 | 80.00% | 0.00 | 0 |
| bacteriocin | 15 | 86.44% | 3.71 | 0 |
| RIP rRNA glycosidase | 7 | 95.24% | 6.85 | 0 |
| superantigen enterotoxin | 7 | 95.24% | 9.44 | 0 |
| ADP-ribosyl AB toxin | 7 | 100.00% | 0.00 | 0 |
| clostridial neurotoxin | 6 | 100.00% | 0.00 | 0 |
| pore-forming cytolysin | 7 | 100.00% | 0.00 | 0 |

⭐ **And the full panel × seed-count grid was already in the artifacts.** It was never printed as one, so
here it is — the same twelve classes across both panels at five and thirty seeds:

| class | v2/5 | v2/30 | v3/5 | v3/30 | spread |
|---|---:|---:|---:|---:|---:|
| contact-dependent inhibition | 35.0 | 36.7 | 75.0 | 72.5 | **40.0** |
| pore-forming cytolysin | 68.6 | 69.0 | 100.0 | 100.0 | **31.4** |
| virulence-associated non-toxin | 50.0 | 51.3 | 34.0 | 32.0 | **19.3** |
| superantigen enterotoxin | 100.0 | 100.0 | 91.4 | 95.2 | 8.6 |
| Cry insecticidal | — | — | 87.3 | 79.2 | 8.0 |
| **beta-lactamase** | 21.4 | **15.5** | 18.6 | 21.0 | 6.0 |
| RIP rRNA glycosidase | 100.0 | 100.0 | 97.1 | 95.2 | 4.8 |
| bacteriocin | — | — | 84.0 | 86.4 | 2.4 |
| **phage peptidoglycan hydrolase** | — | — | **10.0** | 12.3 | 2.3 |
| ADP-ribosyl AB toxin | 100.0 | 99.5 | 100.0 | 100.0 | 0.5 |
| clostridial neurotoxin | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 |
| T3SS effector apparatus | 80.0 | 80.0 | 80.0 | 80.0 | 0.0 |

🔑 **The grid separates the two factors, and they are not the same size.** **Panel** moves three classes by
**19 to 40 points**. **Seed count** moves every class by **≤ 1.7 points — except beta-lactamase, which
moves 5.9** (21.4 → 15.5 on v2). ⚠️ **That exception is the class whose published figure is the minimum, so
its 15.5% is low on both counts at once**, and a sentence claiming seeds never matter would have been wrong
for exactly the row it mattered in. 🔴 **These are v3 numbers**; the v2 column is historical.

🔴 **Both published failure figures are the minimum cell of their row.** Beta-lactamase's four cells span
[15.5, 21.4] and **15.5 is the lowest**; phage's two span [10.0, 12.3] and **10.0 is the lowest**. ⚠️ **No
selection was intended and the effect is identical to selection.** 🔒 **The bimodality survives the move** —
phage and beta-lactamase are the two lowest in every cell where both exist, and three classes sit at
exactly 100% — **but which classes sit where in the middle does not.**

⚠️ **And "7 of 30 splits at exactly 0%" is a v2 figure.** On v3 at the same thirty seeds it is **2 of 30**.
The sentence below is kept because it was published, and it is now labelled.

🔴 **Neither failure is a small-*n* artifact.** The phage class is the **largest** eligible class in v3,
and beta-lactamase is the largest in v2. At thirty seeds **on v2** beta-lactamase is **15.5%** with **7 of
30 splits recovering it at exactly 0%** (**v3: 20.95%, 2 of 30**), and it is the only class where plain
Smith-Waterman alignment beats
the probe (**29.5%** against 15.5%). Profile methods do not: `phmmer` reaches 5% there and `jackhmmer` 0%.

⚠️ **The interval on that 15.5% needs stating precisely, because the obvious reading is wrong.** The
figure this project reported for two months, **[11.0, 19.9]**, is `mean ± 1.96·sd/√30` — a standard error
of the **seed mean**. On a **14-member** class its half-width of 4.5 points is **0.62 of one protein**, and
the seed spread it is derived from is **[−8.9, 39.9]**. Resampling members puts the recovery rate at
**[0, 36]**. 🔒 **The bimodal finding does not depend on this** — 100% and [0, 36] do not overlap, and
7 of 30 splits at exactly 0% is a count, not an interval. ⚠️ **What does not survive is the apparent
precision**, and the same construction underlies **21 intervals** in this paper from **14** source files;
they locate seed means and should be read as such.

🔴 **Two of them are impossible on their face.** Recomputed across all twelve holdout-eligible classes
(`src/111_lomo_member_ci.py`), the same construction puts RIP rRNA glycosidase at **[91.5, 102.7]** — above
100% for a bounded proportion — and contact-dependent inhibition at **[75.0, 75.0]**, a zero-width interval
on a **four-member** class. 🔒 **The member-level intervals are disjoint where the finding needs them to
be**: three classes at [100, 100] against phage peptidoglycan hydrolase at **[0, 21.9]**.

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

⚠️ **The brackets in that table are 200-seed intervals, and they are not bare seed standard errors:**
`src/49` redraws the **118 / 178** calibration split on every one of its two hundred seeds, so
calibration-split variability is already in them. 🔴 **What they omit is test-row variability on the
8,258 pool rows**, and sizing it binomially (`src/129`) puts it at **1.99×** and **2.05×** the published
half-width — giving **[7.22, 8.52]** and **[5.41, 6.55]**, about **2.2×** wider. 🟢 **Entry 84 predicted
"roughly 2×" for these and that is what they are.** ⚠️ **On the 3,407 distinct protein names** rather
than 8,258 rows the row term alone is **3.09×** and **3.19×**, giving **[6.92, 8.82]** and
**[5.14, 6.81]**. 🔴 **A binomial is a floor here** — pool rows are homologous — so an exact bootstrap
would be wider still, and `src/49` re-run with per-row flags retained is the outstanding work.

⚠️ **Study AC's honest intervals do not cover this quantity.** It fixed four intervals, and the two this
paper uses from it are the localization gradient and the registration contrast (§§ 5–6, where both are
named). Its own pool and VFDB rates are a **different fold** —
fitted on the 746 class-axis representatives, thresholded on the pool's calibrate split, tested on
**6,758** pool rows — so its 5.4–5.9× widenings belong to those quantities and not to the table above.

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

### Notation

Every symbol and short label used below, in one place, because each one is introduced in a table
elsewhere and none was defined before its first use:

| | what it is |
|---|---|
| **`R`** | extracellular over intracellular flag rate, among **benign** proteins. The localization gradient. Band frozen at ≥ 3.0 for "a major driver" |
| **`R_MH`** | the same ratio pooled over ten length deciles by Mantel–Haenszel — `R` with length held fixed |
| **`C/A`** | VFDB-extracellular over benign-extracellular flag rate. The registration contrast, with provenance and localization held fixed by stratification. Bands: ≥ 2.0 supported, ≥ 1.5 partial |
| **`D/B`** | the same contrast inside the **intracellular** stratum, where neither secretion nor surface exposure can be doing the work |
| **margin** | for a class *C*, the mean over its members of (max cosine to a hazard outside *C* − max cosine to any negative). Computable before *C* is trained on |
| **panels v1 / v2 / v3** | the three internal panel builds: 71/62, **80/154**, **149/296** hazard/benign proteins. v2 → v3 is strictly nested; v1 is superseded and is not quoted here |
| **studies E, K, L, M** | the preregistered confound studies, named by the file that froze each: **E** localization control, **K** matched VFDB at scale, **L** localization coverage, **M** length adjustment |
| **A1, B-1** | numbered hypotheses inside a preregistration — **A1** the hard-negative test of criterion 1, **B-1** the external class-axis test's primary Spearman |

⚠️ **The study letters are file labels, not a taxonomy.** Each names the document that froze its
protocol before the run, and the document is cited wherever its number is used.

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

🔴 **A second guard on the same strata failed four times and turned out to be measuring the wrong
thing**, which is reported here because the preregistrations carry the four failures. The check compared
the **unannotated** stratum's flag rate to the **pooled annotated** rate and returned **0.41, 0.32, 0.48
and 0.52** against a frozen [0.67, 1.5]. ⚠️ **The pool it divided by averages three classes that differ by
a factor of five.** Against **intracellular alone** the ratio is **1.066** and **1.171** length-adjusted
over ten deciles, an availability indicator adds **+0.0025 / +0.0074** AUROC over localization class with
a coefficient that **flips sign between arms**, and the probe separates unannotated from length-matched
intracellular at AUROC **0.5226 / 0.5371** — chance
(`docs/AVAILABILITY_DENOMINATOR_PREREGISTRATION.md`). 🔒 **The unannotated stratum behaves like the
cytoplasmic one, and `R` never used it**: it contrasts extracellular against intracellular, both
annotated.

⭐ **The obvious mitigation was tried, and it does not work.** If the probe reads localization, one could
orthogonalize the representation against the localization axis and ask what is left. Iterative nullspace
projection, with directions fitted on benign strata and cross-fitted so the rows that fit the axis never
measure it, **does not remove it**: across a frozen grid up to **160 of 1,280 dimensions (12.5%)**, the
best localization AUROC a classifier refitted on unseen rows reaches is **0.698**, against 0.980
unprojected and a frozen target of 0.55. 🔴 **And `R` never falls below 3.35 on either arm** — at k = 80 it
is **7.24**, *higher* than the 4.86 it started at (`docs/LOCALIZATION_NULLSPACE_PREREGISTRATION.md`).

⭐ **A provable erasure was then run, because INLP's under-erasure is a known limitation** — R-LACE
([Ravfogel et al., ICML 2022](https://proceedings.mlr.press/v162/ravfogel22a/ravfogel22a.pdf)) and
**LEACE** ([Belrose et al., NeurIPS 2023](https://arxiv.org/abs/2306.03819)) exist for it, and LEACE's
rank-1 oblique eraser *provably* stops every linear classifier from recovering a binary concept. 🔒 **It
does what it promises**: the residual correlation between the localization label and the most favourable
linear score of the erased representation is **3.1e-06**, against **0.58** unerased
(`docs/LEACE_ERASURE_PREREGISTRATION.md`).

🔴 **`R` is 4.79 after it.** On a representation that is **provably free of a linear localization signal**
on the rows the eraser was fitted to, a secreted benign protein is still flagged **4.79×** as often as a
cytoplasmic one (control 5.70; **4.55** unwhitened, **4.95** on 35M).

⚠️ **The honest interval on that 4.79 is [2.75, 9.29], not the ±0.17 a seed spread gives.** Cross-fitting
the localization axis leaves only **73 and 98** extracellular rows to measure on, so row sampling dominates,
and a bootstrap over rows as well as probe seeds is **19× wider** than the seed standard error this
repository reports by default (**14×** on 35M; `docs/LEACE_ERASURE_PREREGISTRATION.md`). 🔒 **It still
excludes 1.5 on both arms — the gradient is not removed — and it does *not* exclude 3.0**, so "survives"
is the point estimate's verdict and not the interval's.

⭐ **Across every erasure configuration tried** — INLP at 0 to 160 directions, LEACE with and without
whitening, concept directions from 200 to 1,664 labels, both arms — **`R` spans [3.35, 7.30] over 47
measured values and never approaches 1.5.** ⚠️ **It is above 4.5 in every LEACE configuration (minimum
4.55) but not in every INLP one**, where 14 of 47 values fall below 4.5.

🔑 **The reason is that erasure equalizes class means while the screen fires on a tail.** Out of sample the
guarantee does not transfer — erasure removes **29–53%** of the mean-based correlation, and a classifier
refitted on unseen rows still reaches **0.81** — and a label curve shows that gap **narrowing and then
flooring** rather than closing. ⚠️ **So orthogonalizing against localization is not a repair**, and whether
VFDB membership would survive an erasure that *did* transfer remains open; this paper does not claim it
either way.

⚠️ **Length was the obvious alternative explanation, and adjusting for it costs about a quarter.**
Pooling across ten length deciles gives a Mantel–Haenszel ratio of **3.98 [3.87, 4.09]** on 650M and
**4.70 [4.44, 4.97]** on 35M against crude ratios of 5.18 and 5.86, above the frozen 3.0 on both arms
(`docs/LENGTH_ADJUSTMENT_PREREGISTRATION.md`).

🔴 **On honest intervals the adjusted ratio does not clear that band, and the crude one does.** Resampling
the rows inside each decile and re-pooling gives **[2.871, 5.200]** and **[2.885, 7.116]** — 7.8 to 9.6×
wider, and **crossing 3.0 on both arms** (`docs/ADJUSTED_INTERVALS_PREREGISTRATION.md`). The crude ratio's
honest interval is **[3.714, 8.534]** and **[3.242, 9.839]**, which clears it.

🔒 **So the two verdicts separate and are stated separately from here on**: the localization gradient is a
**major driver** on the crude contrast and **partial** once length is adjusted for — *"survives adjustment
but not as a major driver"*, which is the middle band study M froze in advance. ⚠️ **The point estimates
are unchanged**; what changed is that the interval on a nine-cell stratified estimator was a seed standard
error.

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

⚠️ **Three qualifications on the U, before the numbers.** First, **the model truncates at 1,022
residues** and **293 of 8,259** pool proteins — **27.5% of the 700+ band** — are embedded from their first
1,022; 🟢 **truncation is ruled out as the cause** (those proteins are flagged at **4.37%** against a bound
of 39.8% required to carry the rise, `docs/TRUNCATION_BAND_PREREGISTRATION.md`). Second, **the magnitude is
650M-dominated**: the right arm reaches **12.18%** on 650M and **2.03%** on 35M, against troughs of 1.69%
and 0.68%, so the shape replicates and the size does not. Third, **the 700+ intracellular band is 189 of
288 proteins that only study L's rule calls intracellular**, and their rate relative to the rest is
**7.99×** on 650M and **0.21×** on 35M — 🔴 **inverted between arms, so the band's composition gets no
verdict.**

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

🔴 **And the U-index is a function of model size, though the shape is not.** Running the same
measurement on four ESM-2 arms spanning 80× in parameters gives a strictly monotone U-index —
**44.66 → 18.68 → 9.65 → 8.38** for 8M, 35M, 150M and 650M, Spearman **−1.00** against log parameters
(`docs/SIZE_SCALING_PREREGISTRATION.md`). 🔒 With four arms the smallest achievable one-tailed *p* is
**0.0417**, which is what this returns.

⚠️ **Removing the bands removes the trend.** Regressing the outcome on `log L` and `(log L)²` — which
contains no boundaries — gives a **positive** quadratic coefficient on **all four arms**, so the
curvature is in the data and not in the cut points; but its magnitude runs **+0.284, +0.212, +0.244,
+0.329**, Spearman **+0.40** (`docs/LENGTH_BAND_SENSITIVITY_PREREGISTRATION.md`). 🔑 **The two are
reconciled by the mechanism above**: a ratio whose denominator grows shrinks even when the shape does
not. 🔒 **So the U-index falls with scale; the curvature does not.** ⚠️ And the U-index is itself
band-sensitive — across six boundary grids on 650M it ranges **2.32 to 8.65** — so it is quoted with
that range.

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

**C/A = 2.418 [2.339, 2.497]** and **2.617 [2.495, 2.740]** — 🟢 **supported on both arms** *on those
intervals*.

🔴 **Those intervals are seed standard errors, and on honest ones the verdict is partial.** Resampling the
**264** VFDB and **171** benign rows as well as the probe seeds gives **[1.904, 3.252]** and
**[1.931, 3.938]** — **8.5× and 8.2× wider**, and **both cross the 2.0 band**
(`docs/HEADLINE_INTERVALS_PREREGISTRATION.md`). 🔒 **Study K's verdict on its own panel is therefore
partial, not supported** — which is where the independent pool had already put it, at 1.59 and 1.63.
⚠️ **The point estimates do not move; only the intervals do.**

🔒 **And that verdict has since been tested on 2.7× the population rather than left on the interval.**
Dropping the reviewed-only requirement on the VFDB side takes the numerator from **264 to 703**, and
pooling the independent benign draw takes the denominator from **171 to 301**
(`docs/SPENT_RULE_PREREGISTRATION.md`). `C/A` becomes **2.106** [1.772, 2.541] and **2.277**
[1.821, 3.223] — **partial on both arms, confirmed rather than overturned**, and above the frozen
retraction threshold of 1.2.

⭐ **The two expansions are informative in opposite ways.** Expanding the VFDB side 2.7× moves the point
estimate by **0.009** and **0.049** — **whether a VFDB protein has a curated Swiss-Prot entry does not
predict whether this probe flags it**, so a rule that cost **54.4%** of the population bought nothing
here. Pooling a second benign draw moves it by **0.30 on both arms**, between 6× and 33× as much.
🔑 **The denominator is the lever**, which is the conclusion the field reached independently when
negative-set choice alone swung AUPRC from 0.62 to 0.87 (§ 1.2). 🔑 **The adverse finding survives the same
treatment**: `R` = 5.53 and 5.43 with honest intervals **[3.714, 8.534]** and **[3.242, 9.839]**, clearing
the frozen 3.0 on both arms.

And `D/B` = **7.988** and **6.875** on cells of 134 against a floor of 100: **within the
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

⚠️ **That split is one author's judgment, and testing the same question on the curator's own labels does
not simply confirm it — it adds a caution.** The external class-axis test holds **each** of VFDB's
thirteen eligible categories out in turn, over 746 representatives that **do** include **102 Exotoxin**
(`src/119`, `results/class_axis_hazard_order.json`; recoveries from `src/81`, 30 seeds per arm). Applying
the *same frozen* four-category grouping to those thirteen labels:

| | 650M rank | 650M recovery | 35M rank | 35M recovery |
|---|---:|---:|---:|---:|
| ⭐ Motility — flagella, no toxic function | **1** | **84.81%** | 4 | 62.78% |
| **Exotoxin** — the curator's toxin category | 2 | 79.44% | **1** | **75.23%** |
| Adherence (n = 180, the largest) | 4 | 79.02% | 3 | 65.65% |
| Immune modulation (n = 77) | 10 | 53.07% | 9 | 43.33% |
| Stress survival — last on both arms | 13 | **38.67%** | 13 | **23.67%** |
| **hazard-relevant mean** (9 categories) | | **67.34%** | | **57.17%** |
| **non-hazardous mean** (4 categories) | | **55.92%** | | **37.26%** |

🔴 **The top slot is arm-dependent, so "the most-recovered category has no toxic function" is not
available as a claim**: Motility is first on 650M and *fourth* on 35M, where Exotoxin is first. Exotoxin
minus the best non-hazardous category is **−5.37pp on 650M and +12.45pp on 35M** — the sign itself flips
between arms, which under this paper's second-arm rule means no verdict.

⚠️ **And the group gap runs the other way from § 6's contrast.** Hazard-relevant categories recover
**higher** than the frozen non-hazardous four on **both** arms: **+11.41pp** and **+19.91pp** unweighted,
**+18.31pp** and **+26.04pp** member-weighted. 🟢 **It is indicative, not a verdict** — the category-level
bootstrap interval spans zero on 650M (**[−8.67, +28.10]**) and clears it only on 35M
(**[+1.82, +35.04]**), and the exact rank test is significant on *neither* arm (p = **0.330** and
**0.0755**, nine categories against four). ⚠️ Two confounds are larger than anything clean here: recovery
tracks a category's **median length** at ρ = **+0.53** / **+0.40** and its **species diversity** at
ρ = **+0.57** / **+0.51**, so the gap is not attributable to hazard content.

🔒 **So the finding is that the probe responds to VFDB *registration*** — and that narrowing rests on
the matched contrast, not on recovery. `C/A` compares
each VFDB protein against a length- and organism-matched benign one and comes out at parity across hazard
groups on both arms; per-category *recovery* has no matched comparator, tilts hazard-ward by 11–26
points, and cannot separate that tilt from length and diversity. ⚠️ The *matched* population also never
contained exotoxins — `src/74` built it from *"VFDB setA, non-Exotoxin records"* — so `C/A` is about
**non-toxin** virulence factors. **The contrast is real and large; what it is evidence for is narrower
than membership-beyond-confounds suggested.**

### The missing cell: a matched, toxin-inclusive contrast on a third-party axis

🟢 **Study AH closes it.** The axis is **UniProt curators' Swiss-Prot keywords** — Toxin (KW-0800),
Hemolysis, Cytolysis, Enterotoxin, Neurotoxin — set by neither the author (study Q) nor VFDB (study AG),
pulled for all **745** distinct RefSeq accessions of the 746 representatives in one ID-mapping job
(`src/120`, `src/121`; bands frozen in `docs/EXTERNAL_HAZARD_AXIS_PREREGISTRATION.md` before the study
was written). **228 of 746 carry a reviewed entry — 30.6%**, which is § 6's expensive reviewed-only rule
again on a new population — of which **31 are hazard-positive** and **197 are not**. 🔑 **The two curator
sets agree at 87%**: 26 of the 31 are VFDB **Exotoxin**, and 26 of the 30 reviewed Exotoxin
representatives carry a toxin keyword.

Each hazard-positive is paired without replacement to the nearest-length hazard-negative — **mean length
ratio 1.126, maximum 1.475, 17 of 31 pairs from the same species, none dropped** — and **both groups are
held out of the same fold**, so a single model scores both. 🔒 The **strict** variant also holds out the
entire Exotoxin category (**137** positives held, **609** fitted), so homology to a *trained* exotoxin
cannot help the hazard group.

| | 650M | 35M | frozen band |
|---|---:|---:|---|
| **AUROC**, hazard vs matched — **strict** | **0.4599** | **0.5247** | 🟢 **registration, 0.40–0.60** |
| AUROC — permissive | 0.5227 | 0.5468 | 🟢 registration |
| recovery, hazard group — strict | 74.2% | 66.2% | |
| recovery, matched negatives — strict | 71.0% | 56.0% | |
| Δrecovery, matched — strict | +3.23pp | **+10.22pp** | reg. / between bands |
| Δrecovery, pair bootstrap — strict | **[−10.75, +17.85]** | **[−6.99, +27.85]** | both span 0 |

🔒 **AUROC is a verdict, and it is the registration band on both arms and in both variants.** The screen
flags **three quarters of the curated toxins**, so it is not blind to them — but it **cannot order a
toxin above a length-matched non-toxin from the same database**: 0.46 and 0.52 against a hazard floor of
0.70. ⚠️ **Δrecovery is not a verdict.** The arms split in the strict variant — +3.23pp inside the band,
+10.22pp **0.22pp above its ceiling** — and all four pair bootstraps span zero at n = 31 pairs.

⭐ **The matching is not doing the work.** Expected recovery computed from study AG's own per-category
means puts the matched 31 within **0.41pp** of the full 197-protein negative set, so the contrast is not
carried by a shift in category mix.

🔴 **And study AG's category gap did not transfer to this axis.** The frozen prediction P1 — *derived*
from AG's +11 to +20pp group gap — was that the **unmatched** difference would exceed +10pp on both arms.
It is **−13.30pp on 650M** and **+8.10pp on 35M**: on the larger arm the curated toxins recover *worse*
than the non-toxin members. ⚠️ **AG's gap is a property of VFDB's category partition, not of hazard
content**, and a third-party axis cuts across that partition. P2 — that the matched figures would land
in the registration band — held on **all four AUROCs** and failed only on the 35M Δrecovery, by 0.22pp.
⭐ **P3, the one prediction labelled *reasoned*, is the one that held**: strict AUROC below permissive on
both arms, 0.4599 < 0.5227 and 0.5247 < 0.5468. Two derived predictions missed and the reasoned one
held — the exact inverse of this project's record, and the sample is one study.

🔑 **So § 6's narrowing now has the cell it was missing**: a matched contrast, on a population containing
toxins, along an axis this project did not define, with both groups unseen by the model that scored them
— and the screen orders toxins and length-matched non-toxins at chance. ⚠️ **On 31 pairs**, which is the
limit of what 30.6% reviewed coverage allows — 🔒 **and study AM repeats it on 828.**

### The ordering result at 26.7× the pairs

🔑 **This is the one hazard claim § 6 still makes, so it is the one worth repeating at scale.** DeepVIC's
axis holds **1,126 Exotoxin** against **1,121 Motility** and study AI-3 already matched **828** pairs from
them; AI-3 measured the *recovery difference* at a fixed threshold, which is a different quantity and
which did not replicate. Study AM measures the **ordering** quantity on the same pairs — both groups held
out of one fit, scored by one model, AUROC rather than recovery
(`docs/WITHIN_FOLD_ORDERING_PREREGISTRATION.md`, bands frozen first).

| | pairs | 650M | 35M | composition, 21 features |
|---|---:|---:|---:|---:|
| study AH (UniProt keywords) | 31 | 0.4599 | 0.5247 | — |
| **study AM (DeepVIC), strict** | **828** | **0.4457** | **0.5047** | **0.5553** |
| single-run prediction interval | | [0.3211, 0.6705] | [0.3872, 0.6605] | **[0.5204, 0.5919]** |
| 30-seed mean's own interval | | **[0.4147, 0.4767]** | [0.4800, 0.5295] | [0.5511, 0.5595] |

🔒 **Every cell is inside the frozen 0.40–0.60 band, on both arms and in both variants, against a
hazard floor of 0.70.** The screen flags these proteins at **44.7%** and **51.2%** against **48.5%** and
**46.1%** for their matched comparators, at out-of-sample false-positive rates of **4.99%** and
**4.87%**, so it is neither blind to them nor able to rank them. 🔑 **The 31-pair caveat is retired: the
result holds at 26.7× the sample.**

🔴 **And the only arm whose interval excludes chance is the one with no model in it.** Composition's
AUROC is **0.5553 [0.5204, 0.5919]** — inside the "cannot order" band, because that band is about
usefulness, but **reliably above 0.5**, while neither embedding arm's interval excludes it. ⚠️ **So such
ordering ability as exists here is compositional**, and twenty-one numbers have more of it than 1,280
dimensions do. 🔑 **That is the same finding as § 6's composition arm and § 6's increment table, arriving
a third time from a third direction.**

### We tried to test whether the increment is compositional, and could not resolve it

🔴 **Three descriptions are not a test.** Studies AJ, AK and AM each found composition doing something
the representation was credited with, and none of them asked whether the **+7-point increment is
information composition does not carry, or composition re-encoded.** Study AN removes the composition
subspace and re-measures — the move studies Y and Z made on localization — with
`X_resid = X − C(CᵗC)⁻¹CᵗX` over the same 21 features
(`docs/COMPOSITION_RESIDUAL_PREREGISTRATION.md`, bands frozen; **amendment 1** added mid-study and
explained below).

| projection | erasure, max &#124;corr&#124; | mean recovery, 650M | 35M | verdict |
|---|---:|---:|---:|---|
| none (the embedding) | 0.688 / 0.732 | 41.43% | 41.78% | — |
| composition alone | — | 34.25% | 34.25% | — |
| **fitted on negatives only** (frozen) | **0.239** | **37.69%** | **38.86%** | partly / not compositional |
| **fitted on all rows** (amendment 1) | **1.9e-14** | **9.12%** | **10.96%** | **is compositional** |

🔴 **The two projections disagree, so by a reading rule frozen before either was read, study AN does not
settle the question.** What it gives instead is a bracket: **[9.12%, 38.86%]** of mean recovery survives
removal of composition, against composition's own **34.25%** and the embedding's **41.4–41.8%**. ⚠️ **That
spans from almost nothing to almost everything.**

🔑 **Why neither bound is the answer, which is the useful part.** The frozen projection was fitted on the
negative union so it could not see the held-out category — and a least-squares map fitted on one set of
rows does not annihilate the dependence on rows outside it, so it **under-erases on exactly the rows whose
recovery is being measured** (residual correlation **0.239**, not zero). The all-rows projection erases
exactly, ⚠️ **but it is fitted across both classes, and composition reaches AUROC 0.8705 on this
benchmark's binary task (§ 1.2)** — so removing everything linearly predictable from composition removes
much of the class signal with it. 🔴 **Neither variant isolates what it was meant to.**

⚠️ **A post-hoc arithmetic that is suggestive and not evidence.** Composition's 34.25% plus the
exactly-erased residual's 9.12% is **43.37%** against the embedding's 41.43%, and 34.25 + 10.96 = 45.21%
against 41.78% — within 2 to 3 points of additive. 🔑 **If that near-additivity is real, most of what a
linear probe reads from this embedding for this task is compositional and a small part is not** — but
recovery rates do not add in general, the agreement is loose, and this is labelled post hoc because it
was noticed after the run.

🔒 **What the paper claims from AN is therefore only this**: the question is open, the instrument
available here cannot close it, and the bracket is wide. 🔑 **That is worth stating because the three
arrivals in §§ 1.2 and 6 make the compositional reading tempting, and this is the test that would have
licensed it.**

### The nested comparison answers it, and reinterprets the three arrivals

🔑 **Erasure was the wrong instrument.** Study AO asks what the embedding adds **given** composition by
fitting both together — no projection, no leak, every feature from the protein's own sequence
(`docs/NESTED_MODEL_PREREGISTRATION.md`, bands frozen first).

| | 650M | 35M |
|---|---:|---:|
| composition alone, 21 features | 34.25% | 34.25% |
| embedding alone | 41.43% | 41.78% |
| **concatenated**, 1,301 / 501 features | **41.27%** | **41.75%** |
| **AO-1**, concatenated − composition | **+7.02pp** | **+7.51pp** |
| **AO-2**, concatenated − embedding | **−0.16pp** | **−0.03pp** |

🔒 **AO-1 clears its frozen +6pp floor on both arms: the seven points are information composition does
not carry.** The increment is real and it is not re-encoding. 🔒 **AO-2 is zero to within two tenths of a
point on both arms: handing the probe composition alongside the embedding changes nothing.**

🔴 **And that is not because composition is useless — which is what makes it interesting.** On
**Exoenzyme** composition alone recovers **62.3%** against the embedding's **42.1%**, a 20-point win
(§ 6's increment table). Concatenating recovers **−0.86pp** of that gap on 650M and **+1.86pp** on 35M.
⚠️ **The information is sitting in twenty-one columns the probe has been handed, and it does not use
it.**

⭐ **A post-hoc diagnostic makes that literal.** In the concatenated model on that fold, the
twenty-one composition columns carry **1.54–1.91%** of the total absolute coefficient mass across five
seeds, against the **1.614%** they would get by column count alone — a ratio of **0.95 to 1.19×**. 🔑 **The
probe does not distinguish them from an average embedding column.** This was computed after the run and
is labelled so.

### The classifier was never tuned, and tuning it moves every level in this section

🔴 **`LogisticRegression(C=1.0)` is a default this project has used in every study and never examined.**
Study AP selects `C` per held-out category on a member split of the **other thirteen** — never on the
scored one — and re-measures (`docs/ESTIMATION_REPAIR_PREREGISTRATION.md`, bands and the amendment frozen
before each run).

| | 650M | 35M |
|---|---:|---:|
| mean recovery, fixed `C = 1.0` | 41.43% | 41.78% |
| **mean recovery, nested-selected `C`** | **55.39%** | **46.91%** |
| difference | **+13.96pp** | **+5.13pp** |
| `C` selected | 0.01 (14 of 14) | 0.01 (12), 0.1 (2) |
| out-of-sample false-positive rate | 4.88–5.05% | 4.91–5.23% |
| Exoenzyme: share of composition's lead closed | **119.3%** | **97.7%** |

🔒 **Both arms land in the frozen *repaired* band, at a matched false-positive rate**, and the selection
is **interior** to the grid after it was extended to 1e-4 — so these are tuned figures and not a lower
bound. ⚠️ **Every per-category recovery earlier in this section is therefore a statement about
`C = 1.0`, not about the representation.** On 650M the gains run from +0.99pp to **+24.18pp**, and
Exoenzyme reaches **66.2%**, **above** composition's 62.3% — 🔴 **so § 6's 20-point exoenzyme deficit is
largely a tuning artifact.**

🔑 **The one result that does not move is the one that was externally replicated.** *ρ*(margin, recovery)
goes from **+0.9473 to +0.8110** on 650M and **+0.8857 to +0.8637** on 35M, both still clearing the
**+0.60** floor frozen in § 4 — and even a parameter-free nearest-centroid classifier gives **+0.7451**
and **+0.6264**, also above it. 🔒 **The ordering survives the classifier; the levels do not.**

🔒 **And the comparison has since been made fair on both sides.** Study AQ applies the same selection
rule to composition and to the concatenation (`docs/FAIR_COMPARISON_PREREGISTRATION.md`, frozen first):

| | 650M | 35M |
|---|---:|---:|
| composition, fixed `C = 1.0` | 34.25% | 34.25% |
| **composition, tuned** | **36.41%** | **36.41%** |
| composition's gain from tuning | **+2.17pp** | **+2.17pp** |
| embedding's gain from tuning | +13.96pp | +5.13pp |
| **increment, both tuned** | **+18.98pp** | **+10.50pp** |
| increment, both untuned | +7.19pp | +7.53pp |

🔴 **The fixed-`C` comparison was *understating* the embedding's advantage, not overstating it.**
Composition has twenty-one features against 1,280 and therefore far less to gain from shrinkage — it
picks up **2.17pp** where the embedding picks up 13.96 — so tuning both sides **widens** the gap rather
than closing it. ⚠️ **The +21.14pp figure, which tuned only one side, is retired**; the fair figures are
**+18.98pp** and **+10.50pp**, and the **+7pp** figures are two untuned models.

🔒 **And AO-2's null survives tuning**: the concatenated model beats the tuned embedding by **−0.18pp**
and **+0.35pp**, both inside the ±2pp band frozen for it. 🔑 **Composition adds nothing on top of the
embedding even when every side is fitted properly**, so § 6's reinterpretation of the three arrivals
stands at a tuned classifier and not only at `C = 1.0`.

⭐ **One asymmetry in how the two are tuned is worth recording.** The embedding selects `C = 0.01` for
fourteen of fourteen categories; composition selects across **1e-4, 0.01, 0.1 and 10.0** — ⚠️ **with
twenty-one features the right amount of shrinkage varies by category, and with 1,280 it does not.** That
is a bias–variance signature rather than a result about proteins, and the reasoned prediction that
composition would need *less* shrinkage than the embedding **failed**.

⭐ **And a control splits the arms.** A nearest-centroid classifier — the difference of class means, no
regularization parameter at all — reaches **48.59%** against the swept probe's 55.39% on 650M (−6.80pp,
so `C` is doing something a centroid cannot) but **46.52%** against 46.91% on 35M (−0.39pp, so there the
whole repair is reproduced by a parameter-free model). ⚠️ **The arms disagree, so this is reported as a
quantity**: on the smaller representation the finding is about **model class**, and on the larger one it
is about **regularization strength**.

🔴 **All five of study AP's frozen predictions failed**, four of them derived from the coefficient-mass
mechanism measured one subsection above. 🔑 **The mechanism was real and the inference from it was wrong**:
what stronger regularization changes is not the relative weight on twenty-one columns but **which
direction is found**, and a direction fitted with less capacity transfers better to a held-out category.

🔑 **So the three arrivals are not information claims, and § 6 should not have been drifting toward
reading them as such.** Composition's wins — Exoenzyme's 20 points, the toxin ordering at AUROC 0.555
against 0.446 — happen **while the same information is available to the larger model and unused**. That
makes them a statement about **estimation**: with 1,301 standardized features and L2 regularization, the
fit on thirteen categories selects a direction that does not transfer to the fourteenth, and twenty-one
features cannot be fitted that way. ⚠️ **It is a bias–variance result, not an evidence-about-hazard
result**, and the distinction matters because only the second would say anything about protein language
models and biosecurity.

⚠️ **Two intervals are reported because they answer different questions and this project has confused
them before.** The **prediction** interval draws one seed and resamples pairs inside it — what a single
deployment fit would give, which is `src/110`'s idiom — and spans 0.5 on both embedding arms. The
**mean's** interval is `sd/√30` and excludes 0.5 on 650M, so *the average fit ranks toxins slightly below
their comparators* while *any one fit could land anywhere from 0.32 to 0.67*. 🔒 Both statements are true
and neither is the other.

### And on a published benchmark at eleven times the toxin count, the narrowing does not replicate

🔴 **This is the result we did not want, and it is reported because it was preregistered.** DeepVIC's
axis carries **1,126 Exotoxin** against **1,121 Motility** — sizes matched within **0.5%** without any
choosing — so study AI-3 asks § 6's question at **eleven times** the exotoxin count of study AH, on
curator labels from four databases. Amendment 1 made the **length-matched** variant primary before the
fold ran, because Exotoxin is truncated at 22.02% and Motility at 1.52% (§ 1.2).

| variant | 650M | 35M | signs agree |
|---|---:|---:|:--:|
| all members (1,121 / 1,126) | +2.87pp | **−15.04pp** | ❌ |
| untruncated only (1,104 / 878) | +5.02pp | **−8.76pp** | ❌ |
| **length-matched, primary** (828 pairs, ratio 1.148) | **+6.04pp** [+3.13, +8.96] | **−5.30pp** [−8.91, −1.58] | ❌ |

🔴 **The two arms disagree on the sign in all three variants, and both intervals exclude zero**, so this
is not noise: on 650M Motility is recovered **more** than Exotoxin and on 35M **less**. ⚠️ Neither arm
reaches the frozen hazard band of −20pp, so 35M is **between bands** rather than hazard-selective — but
amendment 1's reading rule required the primary variant to land in the replication band on **both** arms,
and it does not. 🔴 **AI-3 does not replicate § 6's narrowing.**

🔑 **The disagreement is one category, not a weaker arm.** The two arms have near-identical mean recovery
across the fourteen categories — **41.4%** against **41.8%** — and seven categories come out higher on
each. **Exotoxin** swings **−13.7pp** between them (44.7% on 650M, **58.4%** on 35M) while Motility swings
+4.2pp, and it is not seed instability: Exotoxin's across-seed standard deviation is 12.7 and 9.1 points
with **zero** zero-recovery seeds on either arm.

🔑 **A model-free classifier reproduces the smaller arm, including the category that moves.** Study AJ
reruns the identical fold — same categories, same seeds, and the negative partition reproduced
bit-for-bit — with the embedding replaced by **log length plus the twenty amino-acid frequencies**, the
same 21 numbers that reach AUROC 0.8705 on this benchmark (§ 1.2). Bands were frozen in
`docs/COMPOSITION_ARM_PREREGISTRATION.md` before the run.

| | 650M | 35M | **composition, 21 features** | length alone, 1 feature |
|---|---:|---:|---:|---:|
| Exotoxin recovery | 44.7% | **58.4%** | **58.3%** | 38.7% |
| Motility recovery | 47.6% | 43.3% | 38.3% | 8.2% |
| Motility − Exotoxin, length-matched | **+6.04pp** | **−5.30pp** | **−4.67pp** | −5.78pp |
| Exotoxin's rank of fourteen | 8 | 4 | **2** | 1 |
| mean absolute error over the 14 recoveries, vs 35M | — | — | **9.98pp** | 24.17pp |
| …vs 650M | — | — | 12.18pp | 23.82pp |

🔒 **Composition alone puts Exotoxin at 58.3% against the 35M arm's 58.4%** — a gap of **0.08
points** — and gives the length-matched difference the same sign and nearly the same size as 35M
(−4.67 against −5.30), while 650M sits at +6.04. Over all fourteen categories it is closer to 35M than
to 650M (9.98pp against 12.18pp). ⚠️ **So what the smaller representation is doing on exotoxins is
something a 21-number model-free classifier also does.**

🔴 **The preregistration's own verdict is nonetheless PARTIAL, and the reason is a band we chose
badly.** AJ-2 asked for the *difference* between composition's rank correlation with the 35M ordering
and with the 650M ordering to reach **+0.15**; it reached **+0.1319**. ⚠️ **But that criterion put a
floor on a difference of two correlations without requiring either to be large**, and the
length-alone control — included to separate length from composition — clears it at **+0.1714** while
sitting **24pp** from *both* embedding arms and marginally closer to 650M. 🔑 **A criterion that the
worse-fitting arm passes and the better-fitting arm fails is a bad criterion**, and the mean-absolute-
error comparison that shows this is **post hoc** and labelled so in the artifact. The frozen verdict
stands as *does not settle it*; what is reported as settled is the direction and the 0.08-point
Exotoxin match, not the composite rule.

⚠️ **And study AJ has no second arm and cannot have one** — a sequence has one composition. It is an
explanation offered, not a verdict, whichever way its bands had fallen.

### What the representation buys, category by category, at a matched false-positive rate

🔑 **Three folds calibrated alike make this subtractable.** Every pass above thresholds at the 95th
percentile of the same calibrate negatives and comes out at **4.7–5.3%** on the same untouched test
negatives, across all fourteen categories on both arms, so recovery rates compare without adjustment. Study AK subtracts
(`docs/REPRESENTATION_INCREMENT_PREREGISTRATION.md`, bands frozen; the arithmetic that motivated it was
disclosed there rather than hidden).

| | 650M | 35M |
|---|---:|---:|
| categories won, of 14 | **11** | **11** |
| mean increment over all 14 | **+7.19pp** [+0.57, +12.97] | **+7.53pp** [+2.17, +12.51] |
| mean increment where positive | +12.33pp | +11.15pp |
| mean increment over **length alone** | +23.82pp | +24.17pp |
| verdict against the frozen bands | **adds something and not much** | **adds something and not much** |

🔒 **The count criterion is met and the magnitude criterion is not.** Eleven of fourteen was the frozen
floor for *earns its place*; **+7.2 and +7.5 points** sit below its **+10pp** companion, so the composite
verdict on **both** arms is the middle one. 🔑 **That middle outcome is the useful number**: against 21
model-free features, ESM-2 buys about **seven points of recovery at a fixed 5% false-positive rate**, and
against length alone it buys **twenty-four**.

🔑 **Where it helps and where it hurts is interpretable.**

| category | composition | 650M | increment |
|---|---:|---:|---:|
| **Regulation** — cytoplasmic transcriptional regulators | **3.6%** | 23.0% | **+19.32pp** |
| Others | 44.2% | 70.0% | **+25.81pp** |
| … | | | |
| **Exotoxin** | **58.3%** | 44.7% | **−13.59pp** |
| **Exoenzyme** | **62.3%** | 42.1% | **−20.14pp** |

🔴 **On exoenzymes a 1,280-dimensional representation recovers twenty points fewer than twenty-one
numbers do, at the same false-positive rate, on both arms** (−20.14 and −16.75). ⚠️ **That is not "adds
nothing" — it is worse.** 🟢 And on categories with no compositional signature the representation does
nearly all of the work: composition finds **3.6%** of cytoplasmic regulators and the embedding finds
23.0%.

⚠️ **The loss sets are not the same on both arms**, which is why no claim is made about *which*
categories are compositionally marked beyond the one both arms agree on: **Exoenzyme**. On 650M Exotoxin
is also a loss (−13.59pp) and on 35M it is a gain by **0.08 points** — the same knife-edge study AJ
described from the other side. ⚠️ **The per-category intervals are unpaired**: the upstream artifacts
record across-seed SDs and not per-seed vectors, and seeds are shared between folds, so a paired interval
would be tighter and is unavailable. The frozen bands use the count and the mean, neither of which
depends on it.

⚠️ **This is not a contradiction of study AH, and the two should not be collapsed.** AH asks whether the
screen can **order** a curated toxin above a length-matched non-toxin *inside one fold*, and the answer is
chance on both arms. AI-3 asks whether a **held-out category** is recovered less than another held-out
category at a fixed 95%-specificity threshold, in two different folds. A class can be recovered more often
without being orderable above its comparator. 🔒 **What § 6 can claim is the first**: the narrowing holds
as a within-fold ordering result on a third-party axis. 🔴 **What it cannot claim is the second**: the
between-category version is representation-dependent, and on the smaller arm it points the other way.

🔴 **And it is narrower again on a benign population the probe has never seen.** Every result above,
and every result in the localization sections, is a re-analysis of **one fit on one pool of 8,259
proteins** — the "second arm" throughout this paper is a different *embedding of the same proteins*,
not different data. A second pool was drawn from the identical query, filters and per-organism cap,
with every pool-1 accession excluded: **11,301 proteins over 2,334 organisms, sharing 32% of its
species** (`docs/INDEPENDENT_POOL_PREREGISTRATION.md`). Refitting the whole procedure there, on ESM-2
35M:

| | pool 1 | pool 2 | |
|---|---:|---:|---|
| ⭐ localization `R`, 35M | 6.06 | **10.24** | replicates |
| ⭐ localization `R`, 650M | 7.68 | **11.77** | replicates |
| ⭐ registration `C/A`, 35M | 2.89 | **1.63** | partial |
| ⭐ registration `C/A`, 650M | 2.81 | **1.59** | partial |

🔑 **The two headline results move in opposite directions, on both arms.** The localization gradient —
the adverse finding — **replicates and is larger on the unseen pool**, against a band frozen at ≥ 3.0.
The registration contrast **falls out of the supported band into partial**, where this paper's earlier
sections read 2.418 and 2.617 as supported — 🔴 **and those sections were reading a seed standard error;
on an honest interval the contrast is partial on its own panel too** (§ 5) — the same script on pool 1 returns 2.81 and 2.89,
reproducing them closely, **so the drop is the population and not the code.**

🔒 The frozen retraction threshold (`C/A ≤ 1.2`) was not crossed on either arm, so nothing here is
withdrawn. ⚠️ **But "supported on both arms" described one pool**, and on a second pool of identical
construction the same quantity is partial on both arms. **That is the qualification the registration
finding carries from here on.**

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
agrees with the hazard label on only 53.4% of v2.

⭐ **And one baseline aimed at the comparison the literature makes.** § 1.2 cites DeepVF at 0.896, DTVF at
0.921 and DeepVIC at 0.954, and none of those papers reports what a model with **no sequence information**
achieves. On an analogous task here — 778 VFDB proteins against 1,988 matched benign, five-fold
cross-validated — **localization stratum plus log length reaches AUROC 0.8096**, against the ESM-2
embedding's **0.9323** (`docs/METADATA_BASELINE_PREREGISTRATION.md`). 🔑 **And the two are not additive:
metadata on top of the embedding adds +0.0007**, so those fields are not information the model lacks but a
cheap approximation of something it already has. **A paper reporting 0.93 has not shown that 0.81 of it
needed a protein language model.**

⚠️ **This is not their benchmark and does not re-analyse their numbers** — their negatives were chosen by
their own rules over different populations. **The claim is that a baseline of this kind is missing from
that literature**, shown on a task of the same shape. A label-free typicality proxy — mean cosine to the
benign pool's centroid — reaches **−0.746 against recovery at *p* = 0.0034** on the canonical arm, which
we had not tested until a paper we had missed prompted it.

---

## 4. Margin: the one result that replicated

Both failures share a geometric property computable **before** the class is trained on. For a class *C*,

    margin(C) = mean over members of [ max cosine to a hazard outside C − max cosine to any negative ]

| | |
|---|---|
| ranks classes by recovery | **Spearman +0.894**, permutation *p* = **0.00015** |
| the two lowest-margin classes of twelve | **exactly the two failures** (chance 1/66) — ⚠️ **canonical arm; 3 of 10 arms, see below** |
| beats each of its own parts | nearest-positive alone, nearest-negative alone |
| class size runs the other way | −0.564 |
| **both failing classes** | **negative margin**: closer to a benign protein than to any trained hazard class |

🟢 **The ordering transfers. Which classes sit at the floor transfers less, and the two are different
claims.**

| | panel | arms | significant | the floor property |
|---|---|---:|---:|---|
| **ordering**, *ρ*(margin, recovery) | v2, k = 1 | 14 | **13 / 14** (*ρ* 0.588 – 0.957) | beta-lactamase lowest on **12 / 14** |
| **ordering** | v3, k = 2 | 10 | **10 / 10** (*ρ* 0.739 – 0.954) | beta-lactamase lowest on **9 / 10** |
| **both failures in the bottom two** | v3, k = 2 | 10 | — | 🔴 **3 / 10** |

🔴 **The table row above is the canonical arm.** On the only panel where both failures exist, the property
*"the two lowest-margin classes are exactly the two failures"* holds on **three arms of ten** — canonical,
ESM-2 3B and SaProt — and the study's own preregistered test records **`majority: False`**. On **seven** of
the ten the bottom two are beta-lactamase and **virulence-associated non-toxin**, not the phage class.

🔒 **What does transfer is the ordering and the beta-lactamase floor**: significant on 13 of 14 arms on v2
and 10 of 10 on v3, with beta-lactamase lowest on 12 of 14 and 9 of 10. ⚠️ **"It transfers" was previously
stated once, against the v2 evidence, next to a v3 two-class property** — the same panel-mixing this paper
corrects in § 3.

### The ordering replicates on a published benchmark, with someone else's labels

🔒 **This is the first external confirmation of the ordering, and it is a verdict on both arms.** Study AI
runs the same fold on **DeepVIC's** benchmark — **12,989** class-labelled virulence factors in **14**
curator-maintained categories drawn from UniProt, VFDB, VICTORS and PATRIC, against its own 33,456
negatives — because DeepVIC carries that class axis and runs **no leave-one-category-out evaluation** of
its own (§ 1.2). Bands were frozen in `docs/DEEPVIC_BENCHMARK_PREREGISTRATION.md` before `src/125` existed.

| pass | *ρ*(margin, recovery) | permutation *p* | beats both controls |
|---|---:|---:|:--:|
| 650M, full | **+0.9473** | 0.00005 | ✅ |
| 650M, shared removed | **+0.9341** | 0.00005 | ✅ |
| 35M, full | **+0.8857** | 0.00020 | ✅ |
| 35M, shared removed | **+0.8593** | 0.00020 | ✅ |

🔒 **All four passes clear the frozen floor of +0.60 at *p* below α = 0.0125, and on all four margin beats
both of the controls it is built from** — nearest-positive alone reaches +0.63 to +0.79 and
nearest-negative alone −0.52 to −0.80. ⚠️ **That comparison is made at the untuned classifier, and the
paper said so nowhere until 2026-10-06.** `src/125` hardcodes `C = 1.0`; study AP measured *margin* at the
tuned setting — *ρ* falls to **+0.811** and **+0.864** — but never measured **the controls** there, so
"beats both of its own components" has been established at one operating point and not two. 🔴 **On text
it does not survive the tuning**: the nearest-positive half reaches **+0.808** against margin's **+0.769**
at the selected `C`, because the nearest-*negative* half carries real signal at `C = 1` (*ρ* **−0.231**,
worth **+0.11** over the positive half alone) and almost none once `C` is tuned (*ρ* **−0.049**, costing
**0.04**). 🔑 **The reading is about what tuning repairs**: at the untuned operating point the probe is
vulnerable to a class sitting near the negatives, which is exactly what the negative half measures, and
tuning removes that vulnerability — so the term stops predicting. 🔒 **Study AT measures the controls at
both settings in every cell**, which is the test this sentence needed. ⚠️ **One rival explanation is measured rather than assumed**: a
class's ESM-2 **truncation rate** also tracks its recovery, at *ρ* = +0.097 on 650M but **+0.365** on 35M,
so margin is not a truncation proxy — it exceeds that correlation on every pass — but the correlation is
not zero on the smaller arm. 🔒 **And it is not circular**: 239 of the 746 representatives this project
built its own panels from appear in DeepVIC, **0.71% of its positives**, and removing them moves *ρ* by
**0.013** and **0.026**.

🔴 **What does not replicate is the bimodality.** On our own panel the per-class picture is two clusters
with 100% and [0, 36] disjoint (§ 2). Across DeepVIC's fourteen categories recovery is **graded**: the
spread is 0.57 wide, the largest single jump is only **18–24%** of it, and that jump sits after rank 13 on
650M and after rank **1** on 35M — ⚠️ **the arms do not even agree where the gap is.** The ordering
transfers to a published benchmark; the shape of the failure distribution is a property of our panel.

🟢 **And the ordering is not carried by the two extreme points.** Leave-one-class-out gives *ρ* =
**+0.8325** (permutation *p* = 0.0006, 20,000 permutations) with a leave-one-out mean absolute error of
**0.134** against a **0.284** baseline; fitting on the nine pre-expansion classes and predicting the three
v3 added ranks the phage class lowest (chance 1/3). ⚠️ **That last one is a retrodiction and its artifact
says so** — *"not a sealed prediction; the author knew the answers"* — which is why it is reported after the
leave-one-out figure and not instead of it.

🔴 **It is a ranking, not a calibration.** Margin over-predicts beta-lactamase recovery by **34 points**
and errs optimistic on every out-of-sample class. And the member-level version of the same claim was
preregistered, frozen, and **failed on two external panels** — which is why the class-level result is
stated as a triage signal for deciding which mechanism families need their own validation, and not as a
deployment score.

⚠️ **Prediction and repair dissociate.** Removing the benign neighbours the geometry implicates repairs
**8 to 11%** of a failure the geometry predicts at *ρ* = +0.894. Knowing which class will fail does not
tell you how to fix it.

🔒 **The class-level claim has its own external record, and it is not the member-level one.**
`docs/EXTERNAL_CLASS_AXIS_PREREGISTRATION.md` froze a test on **746** VFDB proteins in **13** held-out
categories, 30 seeds, both arms: **B-1 Spearman(margin, recovery) = +0.6813**, permutation *p* =
**0.00590**, inside a band frozen at +0.4 to +0.8. 🔴 **And a frozen confound rule fired** — exclusivity
correlates with recovery at **+0.7613** (*p* = 0.0042) — so B-1 was reported **UNINTERPRETABLE**, which
is what the rule said to do. ⚠️ **The correlation's sign is wrong for leakage**: higher
same-species-in-training goes with *lower* recovery.

🟢 **The design that separates them was then run.** `docs/ORGANISM_STRATIFIED_PREREGISTRATION.md` holds
organism identity constant inside each of **10 species strata** (48 cells): mean within-species *ρ* =
**+0.3750** (*p* = 0.0104) on 650M and **+0.4107** (*p* = 0.0069) on 35M, **8 of 10 strata positive** on
both. 🔒 **B-1 is reinstated as a claim about ordering** — inside a species the organism cannot produce the
ordering — **and the reinstatement is narrow**: it covers ordering, not the absolute recoveries, which stay
compatible with the pathogen and localization separations § 6 quantifies at **42.1%** of the gap jointly.

⚠️ **And the "two external failures" of the member-level version should be read as one clear failure and one uninformative panel**,
because the preregistration says so about its own criteria. Attempt 2 returned **100% on every arm**: the
falsification rules specified a floor but **no ceiling**, so a panel with no room to move was scored NOT
SUPPORTED when *uninformative* is the accurate word
(`docs/EXTERNAL_VALIDATION_PREREGISTRATION.md`). 🔒 Attempt 1 stands as a genuine failure — 82% against a
predicted ≤ 40% at the **same absolute margin**, −0.012 external against −0.015 internal. ⚠️ The likely
cause of attempt 2's ceiling is visible in it: its low-margin holdout was **10 phospholipases out of 38**
in the panel, so margin — defined against out-of-class positives — came apart from a classifier that
trains on the within-class ones.

### And outside proteins entirely: the same fold on text

🔒 **Nothing in the formula mentions proteins.** It takes an embedding matrix and a class partition, so if
the result is about representation geometry it should appear in any domain with that structure, and if it
is about protein language models it should not. Study AS runs **`src/125`'s fold unchanged** — same
selection rule, same 95th-percentile threshold, same 30 seeds — on **20 Newsgroups** with
`all-MiniLM-L6-v2` frozen and mean-pooled, holding out one of **13** sub-classes at a time. Bands were
frozen in `docs/CROSS_MODALITY_PREREGISTRATION.md` before `src/138` existed.

| classifier | *ρ*(margin, recovery) | permutation *p* | mean recovery |
|---|---:|---:|---:|
| fixed `C = 1` | **+0.9505** | 0.00005 | 54.77% |
| nested-selected `C` | **+0.8132** | 0.0013 | 59.02% |

🔒 **Both settings clear the same +0.60 floor at the same α = 0.0125 that DeepVIC was judged by**, and the
floor was borrowed from the protein study deliberately so that the two domains are scored by one criterion
rather than two. 🟢 **The margin result is about frozen-representation geometry. It is not about protein
language models.**

⚠️ **This is one text encoder, so it is one arm, and it is reported as single-arm indicative** — the rule
this paper applies everywhere else is that no verdict stands on one arm (§ 3), and a second text encoder
is named as the follow-up rather than promised. ⚠️ **Thirteen classes is thirteen points**, and study AR
showed this *ρ* is the noisy quantity: it moved 0.105 between negative draws while the levels moved 0.023.

🔴 **The first design of this study was a null task and ran to completion before anyone looked at a
level.** It drew positives from 20 Newsgroups' train split and negatives from its test split, on the
stated reasoning that negatives should come from the same source — which made positives and negatives the
same kind of thing. Recoveries came back at **0.82% to 1.43%**, *below* the 5% nominal false-positive
rate. 🔒 **Nothing in the frozen bands would have caught it**, because every band was on *ρ*. Amendment 1
moved to the corpus's own hierarchy — `comp`/`rec`/`sci` as positives against `alt`/`misc`/`soc`/`talk` —
and added **AS-0**, a precondition that mean recovery exceed 20% before *ρ* is interpretable at all. Both
settings meet it. ⚠️ **The amendment is appended, not substituted**, and the dead design is in the
document above the fix.

🟢 **A second preregistered prediction held, and it was the one that could have gone either way.** Study
AP found an unexamined `C = 1.0` cost up to 24 points per category on ESM-2 650M, where 1,280 features
meet ~13k positives; the prediction was that MiniLM's **384** features on ~18k documents would over-fit
less, so the tuning gain would be **smaller**. It is **+4.25pp against +13.96pp**.

🔴 **A third prediction missed, and the miss locates a hole in the reasoning rather than in the result.**
We predicted the lowest-margin classes would be the near-duplicate sibling pairs — `comp.sys.*`,
`rec.sport.*` — because a class whose nearest other-class neighbour is its own sibling has little to
distinguish it. The three lowest are **`sci.crypt`, `sci.med`, `sci.space`**, and the `rec.sport` pair
sits **high** (+0.096, +0.114). ⚠️ **The prediction read only the first term of the formula.** Margin
subtracts the maximum cosine to any *negative*, and the negative side here is belief, politics and
commerce — which discuss medicine, cryptography policy and space. The classes at the floor are the ones
whose nearest neighbour is on the **other side of the axis**, not the ones with a close sibling on their
own. 🔒 **That is the same mechanism as the protein result**, where both failing classes carried negative
margin because they sat closer to a benign protein than to any trained hazard class.

⚠️ **One definitional difference between the two domains was found afterwards and is reported as a
sensitivity, not a result.** `src/125` computes margin against the *union* negatives — the pool the probe
trains from — while this study computed it against *all* of them. 🔒 **The two rank the thirteen classes
identically at *ρ* = +0.9945**, and under the union definition AS reads **+0.9231** and **+0.7692**
against +0.9505 and +0.8132 — all four above the floor. ⚠️ **Computed after the verdict was read**, so it
is a robustness number and not a preregistered one; study AT fixes the union definition for every cell.

⚠️ **What this does not do is make the screen work.** It widens what the geometric ordering is about; the
dissociation stated above is unchanged, and § 5 is still the section where nothing moved the precision.

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
seeds, the first pooling choice here whose interval clears alignment's 29.5%. It costs **43.8 points** on
superantigens and **9.5** on the panel mean.

🔴 **Those intervals were called disjoint, and they are seed standard errors.** A **paired** test on the
same members — the correct analysis, since both arms score the same proteins
(`src/112_paired_member_test.py`) — separates the two classes:

| | members gained / lost per seed | net median | paired Δ |
|---|---|---:|---|
| **phage peptidoglycan hydrolase** (n = 32) | gains **10, 4, 4, 7, 10**; loses 1, 2, 1, 2, 2 | **+5 of 32** | +17.0 [−3.1, 40.6] |
| **beta-lactamase** (n = 14) | gains 2, 2, 3, 5, 3; **loses 3, 2, 2, 0, 2** | **+1 of 14** | +8.6 [−28.6, 50.0] |

🔑 **On the phage class the reduction gains more than it loses in every seed**, and the effect is a
consistent **+5 of 32** members. ⚠️ **On beta-lactamase it mostly swaps members**: the net is **+1 of 14**,
one seed is negative, and the published +19.5 points does not appear. 🔒 **So the repair is real on one of
the two classes, not both**, and the member-level intervals **overlap** in both cases ([0.0, 50.0] against
[7.1, 57.1]; [0.0, 25.0] against [6.2, 53.1]).

⚠️ **Caveat, stated because it bounds the claim**: only **five** seeds of per-member decisions were stored,
against the published thirty. The gain/loss counts are directionally consistent across all five for phage
and inconsistent for beta-lactamase, which is a pattern rather than a point estimate — **but a thirty-seed
re-run retaining per-member flags is what would settle it, and it has not been done.**

🔴 **And it does not survive a change of representation.** On ESM-2 35M the same reduction costs
beta-lactamase 9.5 points and **no reduction clears its own control**, while the superantigen cost
replicates at −45.2. The supervised ranking behaves the same way: it raises the panel mean on 650M and
**no *k* raises it on 35M**. 🔑 **Every gain in this line is one representation; every cost is general.**

**The number that settles it.** In a deployment of ten thousand sequences at a one-in-a-thousand hazard
rate, with recall taken from the v3 panel's **unweighted class mean** and the false-positive rate measured
out of sample:

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

⚠️ **Two choices in that arithmetic are arbitrary, and both were made the favourable way.** Recall is the
**unweighted mean over twelve classes**; weighting by **members** gives **60.43%** instead of **73.12%**,
because the two failing classes hold **46 of the panel's 141 proteins** — a third of it — while being two
of twelve classes. And the false-positive rate can be the conformal estimate or the interpolating one
(§ 1). The four combinations:

| recall | conformal 5.98% | `np.quantile` 7.87% |
|---|---:|---:|
| **unweighted class mean, 73.12%** | **1.21%** ← published | 0.92% |
| member-weighted, 60.43% | 1.00% | **0.76%** |

🔒 **Neither weighting is obviously right** — unweighted assumes hazards arrive uniformly across mechanism
classes, member-weighted assumes they arrive in proportion to how many members each class has, and a
deployment fixes neither. ⭐ **The flatness is strengthened, not weakened, by the extra axis**: precision
stays inside **[0.76%, 1.26%]** across reductions, estimators *and* weightings. ⚠️ **The headline 1.2% is
the corner of that box most favourable to the screen.**

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

- 🔴 **Not measured on a panel the analysis never touched, and this is the limitation that bounds every
  other number here.** `docs/EXTERNAL_VALIDATION_PREREGISTRATION.md` states it in its own words: *"Every
  result in this project is internal cross-validation on a panel built here, and that panel grew from 91
  to 100 to 154 negatives **in response to confounds the analysis itself found**. Each expansion was
  principled and is documented, but the consequence is that the final numbers sit on a panel shaped by
  looking at intermediate results. No amount of additional internal control fixes that; only a set the
  project never touched can."* ⚠️ **What has since been run against that**: an independent benign pool
  drawn from the identical query with every pool-1 accession excluded, where the localization gradient
  **grew** and the registration contrast fell to partial; and an external class-axis test on VFDB's 13
  categories, supported at *ρ* +0.681, ruled uninterpretable by a frozen confound rule, and reinstated
  **for ordering only** by the organism-stratified follow-up. 🔒 **No externally curated panel has ever
  reproduced the localization or precision results**, and the member-level margin test that was tried
  externally **failed**.
- **Not a better classifier.** DeepVIC reports AUROC 0.954 on a 13,384-sequence holdout from 33,456 VFs.
  Our figures are on a self-built panel of 234 or 445 and are **not comparable**.
- **Not novel on homology control.** Homology-clustered evaluation is established practice.
- 🔴 **Not an externally validated hazard taxonomy.** § 6's narrowing from *hazard* to *VFDB registration*
  rests on a four-category non-hazardous group — Motility, Nutritional/Metabolic factor, Regulation, Stress
  survival — which `docs/HAZARD_VS_MEMBERSHIP_PREREGISTRATION.md` § 1.1 states is **one author's domain
  judgment**, frozen in advance with its reasoning and with the largest ambiguous class deliberately
  excluded. ⚠️ **There is no second annotator and no agreement statistic.** Disclosing the judgment is not
  the same as validating it. 🔴 **Testing the question on the curator's thirteen labels instead made this
  limitation worse, not better** (§ 6): the hazard-relevant categories recover **11–26 points higher**
  than the frozen non-hazardous four on both arms, the top slot flips between arms (Motility on 650M,
  Exotoxin on 35M), and length and species diversity track recovery at ρ = 0.4–0.57. That is indicative
  and unattributable rather than a counter-verdict, but it is the opposite of a confirmation, and it is
  why the narrowing is stated over the **matched contrast** alone. 🟢 **The matched contrast now exists on
  a third-party axis** (§ 6, study AH): UniProt curators' toxin keywords, 31 hazard-positives paired to
  length-matched non-toxin members of the same database, both groups held out of one fold, and the
  ordering is at chance on **both** arms — AUROC **0.4599** and **0.5247** against a frozen hazard floor
  of 0.70. ⚠️ **It rests on 31 pairs**, because only **30.6%** of the representatives carry a reviewed
  UniProt entry, and **a database keyword is not a regulatory hazard designation** any more than a
  database category is. 🔴 **There is still no second annotator for the author's four-category split**,
  and the honest statement is that the narrowing no longer depends on it rather than that it has been
  validated.
- 🔴 **The narrowing does not replicate between categories on the largest published benchmark.** On
  DeepVIC's axis — 1,126 Exotoxin against 1,121 Motility, sizes matched within 0.5% — the length-matched
  recovery difference is **+6.04pp** on 650M and **−5.30pp** on 35M, both intervals excluding zero, so the
  two representations disagree about which category is recovered more (§ 6). ⚠️ A reading rule frozen
  before the run required both arms in the replication band, and this fails it. The disagreement is
  **Exotoxin specifically** — it swings 13.7 points between arms while the fourteen-category means differ
  by 0.3 — so what is representation-dependent is how exotoxins sit relative to the rest of a registered
  set, which is exactly the quantity § 6's narrowing is about.
- **Not a competence boundary.** The held-out class is removed from the *probe's* training, not from the
  foundation model's pretraining, and every class here is in the public databases these models saw.
- 🟢 **The DeepVIC arm rested on one draw of negatives, and now it does not.** All seven studies there
  reproduce `default_rng(12345)` bit-for-bit so their recoveries are comparable, which left the arm on a
  single draw of 33,456 negatives — ⚠️ **and this project's own history said that mattered**, since study
  E's localization result changed on an independent pool (`C/A` 2.418 → 1.594, `R` 7.682 → 11.774). 🔒 On
  a second partition (`default_rng(67890)`, sharing 66.6% of its union with the first, as two independent
  two-thirds draws should) *ρ*(margin, recovery) is **+0.8418** and **+0.9033** at the fixed classifier
  and **+0.8198** and **+0.8549** at the tuned one — **all four cells above the +0.60 floor** — and the
  tuning gain reproduces on both arms. 🔑 **The levels move less than expected**: the mean absolute
  per-category shift between draws is **2.30pp** and **1.47pp**, with a maximum of **5.70pp** (Exotoxin)
  and **6.09pp** (Regulation), so the reasoned prediction that levels would move by more than 3pp
  **failed**. ⚠️ **Study AM's ordering AUROC and study AQ's fair comparison keep the original partition**,
  which is stated rather than implied.
- 🔴 **Thirteen preregistrations, fifty-one frozen predictions, and no correction across them.** Every
  preregistration here sets its own Bonferroni α over its own primary tests — α = 0.05/2, /3 or /4 — and
  **none of them corrects for the others.** With this many frozen bands, some verdicts are expected to
  fall by chance. ⚠️ **The exposure is stated rather than corrected retroactively**, because choosing a
  family-wise correction after seeing which bands were met is the error the bands exist to prevent.
  ⚠️ **Forty-one carry a verdict.** Ten belong to designs frozen for work not yet run — six to
  `MUTATION_EXTENSION` and four to `GENERALITY_GRID`, study AT's three-modality grid — and one was
  **withdrawn by amendment** when its referent moved to the other side of a redesigned axis; `results/prediction_record.json` holds the counts and discovers the studies
  rather than listing them. 🔒 **A withdrawn prediction is stored as unscored, not as a miss**, and the
  record now refuses to build if a withdrawal carries a scored verdict or a scored prediction carries
  none. ⚠️ **The declared, scored and unrun counts do not reconcile by subtraction, and the reason is
  structural**: `DEEPVIC_BENCHMARK` is scored by **two** artifacts — study AI-1's metadata baseline and
  study AI's verdicts — whose prediction numbers overlap, so it contributes **seven** scored rows against
  **six** declarations. 🔒 **The artifact reconciles the gap from its parts rather than asserting it** —
  *declared = scored + unrun + withdrawn − double-scored surplus*, which reads **51 = 41 + 10 + 1 − 1** —
  and names the preregistration the surplus comes from. 🔴 **Until today the declared total read 49**, because the count
  matched *lines* beginning with a prediction label and an amendment that restates "P1 and P2 are
  unchanged" was counted as two further declarations; it now counts distinct prediction numbers.
- ⚠️ **The prediction record, computed rather than carried.** Across those studies the frozen predictions
  held **17 of 30** where they were *derived* from a measurement and **3 of 11** where they were
  *reasoned* — so a derived prediction here is about a coin flip, and the paper's earlier reading that
  derived predictions reliably hold (§ 4) does not survive leaving the panel they were derived on. 🔴 **Two
  entries of the corrections log stated this tally wrongly** because it was maintained by hand; it is now
  recomputed from the artifacts, and a hand-maintained version no longer exists.
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

🔴 **It has since killed something larger: this paper's own § 6 narrowing, between categories.** On
DeepVIC's axis the length-matched recovery difference is **+6.04pp** on 650M and **−5.30pp** on 35M with
both intervals excluding zero (§ 6), so the two representations disagree about whether a held-out toxin
category is recovered more or less than a non-toxin one of the same size. ⚠️ **And the rule nearly failed
to fire in time**: a 650M-only ranking had already been written into § 6 before the second arm was read,
and nothing in the audit could have caught it, because every number in it was correct and traceable and
what was wrong was the scope of the sentence built on them.

**Keep the log append-only, including the entries that hurt.** The log holds 124 entries, many retracting
our own conclusions — a benchmark attributed to the wrong paper, a false-positive check that turned out to be
4/61 by construction and therefore identical for random scores, a cross-reduction comparison made across
geometries whose cosine distributions differ by a factor of three. ⚠️ **The pattern across those is one
thing, not three: we checked whether numbers were stable and not whether they measured what we said.**
Seeds, intervals and replication got attention; a definitionally constant quantity, an
annotation-provenance mismatch and an incomparable scale did not, and no number of additional seeds would
have surfaced any of them.

**What we said we wanted next was more panel, and more panel is what we got.** Mechanism classes from a
published ontology rather than hand curation: VFDB's fourteen categories (§ 6), then **DeepVIC's** same
fourteen over **12,989** annotated virulence factors from four databases, and a hazard axis from
**UniProt curators' keywords** rather than from us. 🔑 **Three things followed, and only the first was
the one we were hoping for.**

🟢 **The margin ordering survived the move.** *ρ* = **+0.947** and **+0.886** on the two arms of
someone else's benchmark, beating both of the controls it is built from on all four passes (§ 4). It is
the one result that replicated internally, and it is now the one result that has replicated externally.

🔑 **And it then survived leaving biology, which changes what the result is about.** The same fold on **20
Newsgroups** with a frozen sentence encoder orders thirteen held-out text classes at *ρ* = **+0.951** and
**+0.813** at the two classifier settings, both above the floor the protein study was judged by (§ 4). 🔒
**So the quantity is not indexing anything about protein language models, or about hazard**: it indexes
how far a held-out class sits from the trained positives relative to the negatives, in whatever frozen
representation is to hand. ⚠️ **That is a wider claim and a weaker one than it sounds.** Wider, because
the biosecurity screen becomes one instance of a general property rather than the subject; weaker,
because a property that holds for newsgroup topics is self-evidently not telling us anything specific
about toxins — and § 4's dissociation still stands, so knowing which class will fail does not say how to
repair it. 🔑 **The honest framing is a transfer-prediction result with a biosecurity screen as the
motivating case**, and this paper's title is still the screen because the screen is what the eighteen
criteria and the corrections log are about.

🔴 **The narrowing did not.** It holds as a *within-fold ordering* claim on a third-party hazard axis —
curated toxins against length-matched non-toxins at **AUROC 0.46 and 0.52**, both arms — and fails as a
*between-category recovery* claim on the larger benchmark. ⚠️ **Those are two different statements that
had been carried as one**, and separating them cost nothing but a preregistration that named which one
the matched contrast could support.

⚠️ **And the bimodality did not travel at all.** On our panel the per-class picture is two disjoint
clusters; across fourteen curator categories it is graded, and the arms do not even agree where the
largest gap falls. 🔑 **The shape of a failure distribution turned out to be a property of a panel of
445, not of the task** — which is the same lesson § 3 draws about per-class recovery, arriving a second
time from outside.

### What a protein language model is worth here, in one number

🔒 **Against twenty-one model-free features at a matched 5% false-positive rate, with the classifier
tuned on both sides, ESM-2 buys between ten and nineteen points of recovery** — **+18.98pp** on 650M and
**+10.50pp** on 35M (§ 6). ⚠️ **At the untuned `C = 1.0` both this project used everywhere, the same
comparison reads +7.19pp and +7.53pp**, so the honest form of the practical number depends on whether
the person asking will tune anything. Against length alone the untuned figure is **twenty-four** points.

🔴 **And one number this paper carried for several hours was a tuning artifact.** At `C = 1.0` exoenzymes
were recovered **20.14** and **16.75** points *worse* by the 1,280-dimensional representation than by
composition; tuned, the embedding reaches **66.2%** against composition's 62.3% and the deficit is gone
(§ 6). 🟢 The opposite case is not an artifact: on cytoplasmic regulators composition finds **3.6%** and
the embedding **23.0%**.

🔑 **That is the number we would have wanted at the start**, and it took a published benchmark, a
model-free control arm and a matched false-positive rate to get it. ⚠️ **It is also an argument for
reporting the control**: a paper that gives only the embedding's AUROC on this benchmark has not shown
that the embedding is what produced it.

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
**152 claims** and fails if any number in a public document drifts from the artifact that produced
it. ⚠️ **This sentence said 90 until 2026-09-30**, because the audit had no claim about its own size — the
same unaudited-number defect it exists to catch, in its own description. It is audited now. The
panels, the protocol and the per-class tables are in `docs/MECHANISM_GENERALIZATION.md`; the criteria and
the scorecard in `docs/DETECTOR_CRITERIA.md`; the preregistrations in
`docs/MUTATION_EXTENSION_PREREGISTRATION.md` and `docs/NEGATIVE_EXPANSION_PREREGISTRATION.md`; and the
dated record of every number that moved in `docs/DATA_CORRECTIONS.md`.

🔴 **Status, 2026-09-30: this is an outline of an argument, not a submittable paper.** § 5 reports that
nothing improved the screen, and § 4 rests on a single positive result — margin.

🔒 **The external test that paragraph was waiting for has returned.**
`docs/EXTERNAL_CLASS_AXIS_PREREGISTRATION.md` froze it on VFDB's **13** eligible categories over **740**
redundancy-deduplicated representatives, and stated in advance that a null there means §§ 4 and 6 become
panel-specific observations. It returned *ρ* **+0.6813** (*p* = 0.00590), a frozen confound rule then fired
and made it **uninterpretable**, and the organism-stratified follow-up that document called for reinstated
it **for ordering only** at *ρ* +0.3750 and +0.4107 with 8 of 10 strata positive (§ 4). ⚠️ **So "never
tested on classes this project did not define" is no longer true**, and it stood in this section until
2026-09-30.

### What kind of reproducibility this is

🔒 Using Gundersen's vocabulary ([*Phil. Trans. R. Soc. A* 379, 2021](https://arxiv.org/abs/2011.10098)),
this project is **R4** — text, code and data — and the claims audit **verifies** rather than **validates**:
it checks that no error entered the running or the reporting, not that the experiments answer the research
question. 🔑 **Every defect the audit has ever caught is a verification defect**; §§ 6 and 7 are where
validation lives, and they are where the adverse findings are.

⭐ **Two of this project's results are best described in that paper's three degrees.** The per-class
recovery table is **not outcome-reproducible across panels** — three classes move 19 to 40 points between
v2 and v3 — while it is **interpretation-reproducible**: the same two classes are lowest on both, and three
sit at exactly 100%. 🔒 And study K's `C/A` contrast was reproduced at **2.418** and **2.617** by a script
written independently on a different code path, which is a reproduction from the *description* rather than
by re-running the code — by that paper's argument the stronger of the two, because it shows the result does
not depend on the original implementation.

⚠️ **And the audit is one layer, not a guarantee.** Following Acuna's framing of layered defence, the thing
to report is the **residual error** of each check: this repository has measured one directly — a
manuscript-figure audit with a **94% false-pass rate** at three significant figures (`docs/DATA_CORRECTIONS.md`,
sixty-eighth entry) — and a single review pass in September 2026 found four further blind-spot classes:
predicates scoped narrower than the sentences they guard, numbers in public documents with no claim at all,
forbid strings that match their own disclaimers, and intervals that answer a different question from the one
the prose asks.

⚠️ **It is a draft and is on the audited surface**, which means a figure here that drifts
from its artifact fails the gate in the same way a figure in the write-up does. It is assembled from
sections `paper/0*.md`; edit those, not this file.
