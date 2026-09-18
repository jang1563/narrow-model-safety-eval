# Mechanism generalization: what a protein-embedding hazard probe does when the molecule is not on the list

**Written:** 2026-09-05 · **Panel:** v2, 80 positives / 154 negatives · **Author:** JangKeun Kim, Weill Cornell Medicine ([ORCID 0000-0002-8733-9925](https://orcid.org/0000-0002-8733-9925))

This document records the **leave-one-mechanism-out (LOMO)** line of work. It is separate from
[`docs/EVALUATION_REPORT.md`](EVALUATION_REPORT.md), which describes the structure-level metrics (FSPE,
FSI, Physical Realizability Tier) on a per-residue panel. The two lines ask different questions and should
not be read as one result.

Every number here is produced by a script in `src/` and stored in `results/v2/`. The project's headline
claims are recomputed from their artifacts by [`src/22_claims_audit.py`](../src/22_claims_audit.py), which
runs in CI and covers this panel explicitly: the recovery table in §3, the panel/results agreement, and the
class-eligibility curation each have an entry, and **this document is one of the surfaces the audit checks**,
so a figure edited here without its artifact fails the build. Corrections are logged in
[`docs/DATA_CORRECTIONS.md`](DATA_CORRECTIONS.md).

---

## 1. The question

A sequence screen is only as good as its list. The operationally interesting failure is not "does this
match a known toxin" but **"what happens when the hazardous molecule is not on the list at all"** — a new
serotype, a distant homolog, or a mechanism the screen was never built for.

Leave-one-mechanism-out is the direct measurement of that. Assign every hazardous protein in the panel to
a **mechanism class**, hold out an entire class, train a probe on the remaining classes plus the negatives,
and measure how much of the unseen class is recovered at a fixed false-positive budget.

> **Recovery here means: the class was never seen in training, and members are still flagged.** It does not
> mean the probe identified the mechanism, and it is not a deployment metric.

## 2. Panel

`data/sequences/panel_v2_manifest.json`, `data/annotations/mechanism_classes_v2.json`.

- **80 hazardous proteins** in 13 mechanism classes, **154 benign proteins** in three blocks
  (secreted cell-wall, cytoplasmic housekeeping, secreted-from-pathogen).
- Class membership is curated, with a written reason per protein. `holdout_eligible_classes` is a
  **curation decision, not a size threshold** — see [`docs/DATA_CORRECTIONS.md`](DATA_CORRECTIONS.md) for
  what happened when a script recomputed it as `n >= 3`.
- `virulence_associated_non_toxin` is run as a **labelled control**, not as a mechanism: it collects
  virulence-associated proteins that are not themselves toxins, and it is reported with
  `holdout_eligible: false` so the contrast against real mechanism classes stays visible.
- 🔴 **`beta_lactamase` entered the panel through a different query than the other eight mechanism
  classes, and this was not documented until now.** The panel's stated hazard definition is UniProt
  keywords KW-0800 (Toxin) or KW-0843 (Virulence). Beta-lactamase carries neither — it carries **KW-0046,
  "Antibiotic resistance"** — and was added through a separate `protein_name:"beta-lactamase"` query
  written for an explicit `antimicrobial_resistance` block in `src/01_collect_data.py`, bypassing the
  hazard-keyword definition entirely. See §9.3 for why this matters beyond bookkeeping.

Three confounds are measured rather than assumed away, because a hazard probe that is really an organism
detector or a secretion detector would look identical on a naive split:

| control | what it holds constant | result |
|---|---|---|
| **pathogen-matched** | organism: 104 benign proteins from the *same pathogens* as the positives | AUROC 0.968 ± 0.021 |
| **localization** | organism *and* secretion, lab strains only | AUROC 1.000 ± 0.000 |
| **provenance** | trains on lab-strain origin with the hazard label ignored | AUROC 0.818 ± 0.012 |

The provenance row is the important one. **A probe that ignores the hazard label entirely still reaches
0.818**, and the organism label agrees with the hazard label on 53% of the panel. Baseline separability
(AUROC 0.974 ± 0.014) therefore cannot be attributed to hazard alone, and no number in this document
should be read as one.

### 2.2 The confounds, stratified and doubly controlled

`src/03d_localization_confound.py`. The three controls in the table above are marginal. The stronger test
asks whether hazard separation survives **inside** a localization stratum, with organism held constant as
well. It does:

| stratum | positives / negatives | stratified AUROC | doubly controlled |
|---|---|---|---|
| exported | 57 / 71 | 0.944 ± 0.035 | 0.937 ± 0.021 |
| not exported | 23 / 83 | 0.968 ± 0.058 | 0.967 ± 0.040 |
| signal peptide | 44 / 68 | 0.969 ± 0.032 | 0.956 ± 0.032 |
| no signal peptide | 36 / 86 | 0.986 ± 0.022 | 0.991 ± 0.011 |

Being exported agrees with the hazard label on only 60% of the panel and having a signal peptide on 56%,
yet each axis is separately separable (localization alone AUROC 0.914, signal peptide alone 0.967). **The
hazard signal is not reducible to either axis**: separation holds in all four strata, including the two
where the confounding feature is absent.

### 2.1 Effective n, not n

Raw class size overstates independence. Clustering each class by single linkage at normalized
Smith-Waterman > 0.30 gives the number of genuinely independent sequences:

| class | n | effective n | note |
|---|---|---|---|
| beta_lactamase | 14 | 10 | |
| t3ss_effector_apparatus | 10 | 8 | |
| virulence_associated_non_toxin | 10 | 10 | labelled control, not a mechanism |
| adp_ribosyl_ab_toxin | 7 | 7 | |
| rip_rrna_glycosidase | 7 | 6 | |
| superantigen_enterotoxin | 7 | 5 | 4 under the stricter normalization |
| pore_forming_cytolysin | 7 | 6 | |
| clostridial_neurotoxin | 6 | 3 | |
| contact_dependent_inhibition | 4 | 4 | |

Before the 2026-09-05 expansion, **clostridial neurotoxin had effective n = 1** — all four members fell
into one cluster, so its perfect recovery rested on a single independent sequence — and superantigen had
effective n = 2. The expansion added 14 non-homologous members, each screened at ≤ 0.30 against existing
members *and* against already-accepted candidates. Perfect recovery survived (§3).

Effective n is threshold- and normalization-dependent. The screen normalizes by geometric mean of the
self-scores; min-self-score is stricter. The two agree on every class except superantigen (5 against 4),
over a single pair at 0.287 / 0.300.

---

### 2.3 🔴 The panel is 34% hazardous and an order queue is not

`src/03r_prevalence_adjusted.py`. Every AUROC above is computed on 80 positives against 154 negatives, so
**34.2%** of this panel is hazardous. AUROC is insensitive to prevalence, so it carries over unchanged
into a setting where it means something very different. AUPRC had never been computed here at all. It is
**0.964 ± 0.023**, against a 0.342 coin at this base rate, a 2.82× lift.

Precision is the quantity that moves, and it follows from the operating point with no new data:
`precision(π) = TPR·π / (TPR·π + FPR·(1−π))`.

| operating point | TPR | FPR | panel, 34.2% | 1 in 100 | 1 in 1,000 | 1 in 10,000 |
|---|---|---|---|---|---|---|
| **95% specificity**, used throughout this document | 89% | 5.2% | 89.9% | 14.7% | **1.7%** | 0.2% |
| 99% specificity | 80% | 1.3% | 97.0% | 38.4% | 5.8% | 0.6% |
| 99.9% specificity | 76% | 0.6% | 98.4% | 54.3% | 10.5% | 1.2% |

🔑 **At the operating point used everywhere else here, precision falls from 89.9% on the panel to 1.7%
at a one-in-a-thousand base rate**, which is 59 false alarms for every true one. AUROC 0.973 does not show
that. Tightening to 99.9% specificity buys precision back to 10.5% and costs 13 points of recall.

This is not a claim about any real screening queue's base rate, which is not public and varies by
provider; the columns are round numbers so a reader can locate their own. What it establishes is narrower
and enough: **a recovery figure at a fixed false-positive budget is not a deployment number.** §11 lists
deployment readiness among the things this work does not claim, and this is the arithmetic behind that
line.


### 2.4 🔴 The confound those three do not cover: what the protein acts on

All three controls above hold the protein's **origin** constant. None holds constant what it **acts on**,
and the panel is not uniform on that axis. `data/annotations/target_host_v2.json`,
`src/03s_target_host_control.py`.

Of the 80 positives: **51** act on an animal host; **15** act on a diffusing small molecule with no host
at all (14 beta-lactamases plus a teichoic-acid transferase whose annotated role is beta-lactam
resistance); **5** act on another bacterium (four contact-dependent inhibition systems plus colicin E2, a
bacteriocin); **1** acts on plant cells (the *Agrobacterium* T-pilus subunit); 4 act on the producing
organism itself; 4 are mixed or unassigned.

**This document never says which of those "hazardous" means.** That is the same kind of gap as the
beta-lactamase provenance entry in [`docs/DATA_CORRECTIONS.md`](DATA_CORRECTIONS.md): not a wrong number,
an undocumented definition. Target and producer also come apart, so both are recorded: six of the seven
ribosome-inactivating proteins are **plant-produced** and act on **animal** ribosomes.

| measurement | result |
|---|---|
| **target-host legibility**, positives only, hazard held constant | **AUROC 0.929 ± 0.065** |
| the provenance control above, for comparison | 0.818 |

🔑 **Target host is more legible in this representation than provenance is**, and §2 already treats
0.818 as reason enough not to attribute separation to hazard alone.

| hazard separation, against the same 154 negatives | n | AUROC |
|---|---|---|
| all positives | 80 | 0.973 ± 0.020 |
| **animal-target only** | 51 | **0.994 ± 0.009** |
| **non-animal-target only** | 22 | **0.898 ± 0.094** |

Not a sample-size artifact. Subsampling the animal set to n = 22, thirty draws give 0.979 ± 0.013 spanning
[0.947, 0.999], and the non-animal 0.898 sits **below that entire range**.

**So the headline is a blend**: the probe separates animal-directed hazards from benign proteins almost
perfectly, non-animal-directed ones considerably less well, and 0.973 averages the two.

**It also reorganizes §3's class dependence.** Ranking the eight mechanism classes by recovery, the two
whose target is not an animal are exactly the bottom two:

| class | recovery@95 | target |
|---|---|---|
| **beta_lactamase** | **21%** | non-animal |
| **contact_dependent_inhibition** | **35%** | non-animal |
| pore_forming_cytolysin | 69% | animal |
| t3ss_effector_apparatus | 80% | animal |
| adp_ribosyl / clostridial / rip / superantigen | 100% | animal |

The exact chance probability of that split is 1/C(8,2) = **0.036**, and class size does not explain it:
clostridial neurotoxin has the **smallest** effective n of any class, 3, and is recovered at 100%.

⚠️ **Post hoc, on eight classes that are not independent draws, and reported at that strength.** It was
noticed while asking what "hazard" means here, not predicted in advance.

🔴 **It bears directly on §9.4.** That section tested "no host interaction" with an external AMR family
and could not settle it. The reason is visible now: even its without-AMR condition removed only the 14
beta-lactamases, leaving contact-dependent inhibition and colicin E2 in training, so the probe still had
non-animal-target positives to learn from. A clean version removes **every** non-animal-target positive
from training before testing a non-animal class. That is registered here as the next test rather than run.


## 3. Result: recovery is class-dependent and spans the full range

ESM-2 650M, mean pooling, 5 seeds. `results/v2/lomo_results.json`, `src/03b_leave_one_mechanism_out.py`.
"flagged@95" is the fraction of held-out members scoring above the threshold that admits 5% of training
negatives.

| class | n | flagged@95 | flagged@99 | AUROC |
|---|---|---|---|---|
| adp_ribosyl_ab_toxin | 7 | **100%** | 91% | 0.994 |
| clostridial_neurotoxin | 6 | **100%** | **100%** | n < 7 |
| rip_rrna_glycosidase | 7 | **100%** | 89% | 0.997 |
| superantigen_enterotoxin | 7 | **100%** | **100%** | 1.000 |
| t3ss_effector_apparatus | 10 | 80% | 80% | 0.949 |
| pore_forming_cytolysin | 7 | 69% | 54% | 0.962 |
| contact_dependent_inhibition | 4 | 35% | 0% | n < 7 |
| **beta_lactamase** | 14 | **21%** | **1%** | 0.751 |
| *virulence_associated_non_toxin* | *10* | *50%* | *32%* | *0.844* |

**Mechanism proximity generalizes to unseen classes, but only for some mechanisms.** Four classes are
fully recovered without ever being trained on. Beta-lactamase — the largest class, effective n = 10, a
family defined by a conserved fold and active site — is almost entirely missed.

### 3.1 The representation is doing the work

`src/03c_ablation_baselines.py`. Before reading anything into the class differences, the embedding has to
beat trivial sequence features:

| feature | dim | AUROC |
|---|---|---|
| ESM-2 650M embedding | 1280 | **0.974 ± 0.014** |
| amino-acid composition | 20 | 0.754 ± 0.083 |
| composition + length | 21 | 0.753 ± 0.087 |
| length only | 1 | 0.453 ± 0.048 |
| **shuffled labels** | 1280 | **0.506** |

The shuffled-label row is the null: with the hazard labels permuted the same pipeline returns chance, so
the AUROC is not an artifact of the cross-validation. Composition alone is a strong baseline at 0.754 —
worth stating rather than hiding — and the embedding adds about 22 points over it. Composition + length
sits at 0.753 for every arm, since it does not depend on the model.

## 4. Recovery is binary per protein, not graded

Averaging hides the structure. Counting, per member, the fraction of seeds in which it is flagged:

| class | n | always flagged | never flagged | in between |
|---|---|---|---|---|
| adp_ribosyl_ab_toxin | 7 | 7 | 0 | **0** |
| clostridial_neurotoxin | 6 | 6 | 0 | **0** |
| rip_rrna_glycosidase | 7 | 7 | 0 | **0** |
| superantigen_enterotoxin | 7 | 7 | 0 | **0** |
| t3ss_effector_apparatus | 10 | 8 | 2 | **0** |
| virulence_associated_non_toxin | 10 | 5 | 5 | **0** |
| pore_forming_cytolysin | 7 | 4 | 1 | 2 |
| contact_dependent_inhibition | 4 | 1 | 1 | 2 |
| beta_lactamase | 14 | 0 | 9 | 5 |

**Six of nine classes are perfectly binary: every member is caught on all five seeds or on none.** T3SS at
80% is not a probe that is 80% sure about ten proteins; it is eight proteins it always catches and two it
never does. A screen tuned to a coverage target is not trading off uniformly — it is choosing how many
proteins fall on the wrong side of a hard split.

### 4.1 🔴 How binary a class looks depends on how many seeds you run

The table above is five seeds. `src/03f_coverage_strictness.py` runs the same measurement at the same
threshold with **thirty**, and four of nine classes change:

| class | 5 seeds | 30 seeds |
|---|---|---|
| beta_lactamase | 0 / 5 / 9 | 0 / 4 / 10 |
| contact_dependent_inhibition | 1 / **2** / 1 | 1 / **0** / 3 |
| pore_forming_cytolysin | 4 / 2 / 1 | 3 / 3 / 1 |
| virulence_associated_non_toxin | 5 / **0** / 5 | 5 / **1** / 4 |

The count of perfectly binary classes is six either way, **but it is not the same six**: virulence leaves
the set and contact-dependent inhibition enters it. Only **five classes are binary under both** —
adp_ribosyl, clostridial, RIP, superantigen, and T3SS.

**So "perfectly binary" is a property of the measurement as much as of the class.** A member flagged on
5 of 5 seeds is indistinguishable from one flagged on 28 of 30 until you run the extra seeds. The claim
that survives is the weaker and more useful one: **recovery is concentrated at the extremes, and the
number of genuinely intermediate members is small — at thirty seeds, 8 members out of 72 across all nine
classes.** The five-seed table above is kept because §5 is measured on the same basis; it should not be
read as saying those six classes are binary in general.

## 5. Recovery is not a property of the class

The 2026-09-05 expansion added members to five classes and left four untouched. **Two of the untouched
classes moved anyway:**

| class (own membership unchanged) | flagged@95 | flagged@99 |
|---|---|---|
| pore_forming_cytolysin | 57.1 → **68.6** | 45.7 → **54.3** |
| beta_lactamase | 21.4 → 21.4 | 4.3 → **1.4** |
| t3ss_effector_apparatus | 80.0 → 80.0 | 80.0 → 80.0 |
| virulence_associated_non_toxin | 50.0 → 50.0 | 32.0 → 32.0 |

Pore-forming cytolysin gained 11.4 points without a single member being added to it. The mechanism is
visible per member: adding positives shifts the fitted probe, which shifts the calibrated threshold
(pore-forming t95 0.707 → 0.666; beta-lactamase 0.436 → 0.470), and **only members already near that
threshold can cross it**. Anthrax protective antigen went 0.779 → 0.864 and from 2 of 5 seeds to 5 of 5.

**§4 predicts which classes can move, and it is right on all four.** Of the four classes whose membership
did not change, the two with the most in-between members moved (beta-lactamase 5 at five seeds and 4 at
thirty, pore-forming 2 and 3) and the two with the fewest did not (T3SS 0 and 0, virulence 0 and 1). Both
moved by exactly zero — not approximately zero, zero. A class with nothing near the threshold has nothing
to give.

Stated on intermediate-member **count** rather than on the binary/not-binary split, because §4.1 shows that
split is seed-dependent: virulence has one intermediate member at thirty seeds and still does not move.

> **A per-class recovery number is a joint property of the class, the rest of the positive set, and the
> operating point.** Reporting one without fixing the other two produces a figure that will not reproduce.

### 5.1 🔴 One class in the training set costs another 16 points

`src/03u_training_set_contamination.py`. §5 shows recovery is a joint property by changing the panel's
**size**. This is the same claim with a named cause: hold pore-forming cytolysin out as usual, then remove
the 14 beta-lactamases from what the probe trains on.

⚠️ **Exploratory, not preregistered.** It surfaced while running
`src/03t_animal_only_training.py`, which asked whether non-animal-target hazard is a category the probe
learns from non-animal examples and **refuted it**: the interaction was −0.0 points with a 95% CI of
[−3.4, +3.7]. That zero was two classes moving 20 points in opposite directions, and this attributes the
movement.

| pore_forming_cytolysin, 60 seeds | recovery |
|---|---|
| standard, every positive except the held-out class | 72.6% ± 14.4 |
| **minus the 14 beta-lactamases** | **89.0% ± 12.6** |
| 25 random removals of 14 | 71.2% ± 7.9, range [58.6, 85.5] |

Paired on the same splits the difference is **+16.4 points, 95% CI [+13.6, +19.3]**, winning on 47 of 60
seeds, and removing beta-lactamase lands **above all 25 random removals**, the 100th percentile, for an
effect attributable to that class of **+17.8 points**.

🔴 **A single random removal is not a control, and treating one as a control nearly produced a wrong
number here.** Two individual draws of 14 put pore-forming at 71.4% and 80.9%, a 9.5-point spread, which
made the attributable effect read as +25.7 on one draw and +8.4 on the other. Only the distribution settles
it.

The effect is two proteins rather than a class-wide shift:

| member | standard | minus beta-lactamase |
|---|---|---|
| **TACY_LISMO**, listeriolysin O | **30%** | **73%** |
| **TACY_STRPQ**, streptolysin O | **8%** | **52%** |
| HLA_STAAU, alpha-hemolysin | 78% | 98% |
| PAG_BACAN, anthrax protective antigen | 92% | 100% |
| HLYE_ECOLI / MU1_REOVD / VACA_HELPY | 100% | 100% |

Both movers are cholesterol-dependent cytolysins; the others sit at or near the ceiling in both conditions.
That is §4's binary structure and §5's joint property in one picture: **the training set decides which side
of a hard split two specific proteins fall on.**

**It does not generalise past this class.** The same test on the other seven: contact-dependent inhibition
moves +6.2 points against a random spread of ±8.7, so it is noise; T3SS and the labelled control move
+2.0 and +6.0; the four saturated classes do not move. Without the random-draw distribution,
contact-dependent inhibition would have been written up as a second case.

**What it changes above.** §3's table is the standard condition and is unaffected. The reading changes: a
per-class recovery figure is conditional on the rest of the positive set by more than 16 points, and
beta-lactamase, already the hardest class and the one whose provenance differs from the others (§2), is
also the one whose presence costs a neighbour most.


## 6. The negative set moves the answer, and mostly through the operating point

`src/03e_negative_difficulty_curve.py`. The negative set is varied 2×2 over **sample size** and
**decision boundary**, so the two candidate explanations for "a different negative set changes the answer"
are separated rather than confounded.

Widening the negative set from lab-strain only (49) to the full set (154) costs recovery, and the cost is
concentrated:

| tier | negatives | baseline AUROC | β-lact recovery@95 |
|---|---|---|---|
| T1 lab strain only | 49 | 0.989 ± 0.011 | 66.0% |
| T2 + pathogen cytoplasmic | 100 | 0.988 ± 0.010 | 74.3% |
| T3 + pathogen secreted | 154 | 0.974 ± 0.014 | **15.2%** |

Four of nine classes lose ground with a bootstrap CI excluding zero; beta-lactamase loses **50.7 points**,
and clostridial and superantigen lose exactly none. Separating the two candidate explanations:

| arm | what varies | mean T1→T3 | classes with CI excluding 0 |
|---|---|---|---|
| full | everything | −13.6 pts | 4 of 9 |
| **matched** | negative set, sample size held fixed | −14.6 pts | 6 of 9 |
| **calib_only** | only the calibration set (the operating point) | **−21.4 pts** | **8 of 9** |
| train_only | only the training set (the decision boundary) | −10.5 pts | 6 of 9 |

**Sample size explains none of it** — the size-matched arm is if anything slightly worse than the
unmatched one. The **operating point dominates**, and the decision boundary contributes for a subset. This
is the same route as §5, arriving from the other side of the panel.

An earlier version of this experiment leaked calibration negatives into training and inflated the
cross-arm baselines; it was fixed with an explicit set difference plus an assertion, after which three
arms agree. See [`docs/DATA_CORRECTIONS.md`](DATA_CORRECTIONS.md).

## 7. The honest baseline is alignment, and it is not always beaten

Every deployed nucleic-acid synthesis screen is alignment- or profile-based. Comparing a learned probe
against nothing is not informative, so the comparison here is against **Smith-Waterman** (BLOSUM62,
gap −11/−1, normalized), used both as a homology control and as an operational baseline.
`src/03i_alignment_baseline.py`, `src/03h_probe_vs_similarity.py`.

Across the panel the probe beats alignment by **+55.9 points** on average, and the homology screen holds:
maximum cross-class similarity 0.279, median 0.033, **zero pairs above 0.30**.

| class | n | probe@95 | alignment@95 | delta |
|---|---|---|---|---|
| superantigen_enterotoxin | 7 | 100% | 6% | +94 |
| adp_ribosyl_ab_toxin | 7 | 100% | 19% | +81 |
| clostridial_neurotoxin | 6 | 100% | 20% | +80 |
| rip_rrna_glycosidase | 7 | 100% | 29% | +71 |
| t3ss_effector_apparatus | 10 | 80% | 14% | +66 |
| pore_forming_cytolysin | 7 | 69% | 12% | +56 |
| *virulence_associated_non_toxin* | *10* | *50%* | *2%* | *+48* |
| contact_dependent_inhibition | 4 | 35% | 21% | +14 |
| **beta_lactamase** | 14 | **21%** | **30%** | **−8** |

**Beta-lactamase is the exception that matters**: it is the one class where alignment beats the canonical
ESM-2 probe. That is a statement about ESM-2, not about embeddings in general — one other representation
does better than alignment on it (§9). The two approaches fail on **disjoint** classes, which is what
motivated §8.

### 7.1 The profile half of that sentence, run

§7 opens by saying deployed screens are alignment- **or profile**-based, and only the alignment half was
supplied above. `src/03k_profile_hmm_baseline.py` supplies the other half with HMMER 3.4, written as a
strict swap-in for `03i`: the same leave-one-mechanism-out protocol, the same margin rule, the same
95th-percentile operating point on held-out negatives, the same 30 seeds, with the Smith-Waterman matrix
replaced by HMMER bitscores. Two modes, `phmmer` and `jackhmmer -N 3`, both run with `--max`.

| class | n | probe | SW alignment | phmmer | jackhmmer |
|---|---|---|---|---|---|
| adp_ribosyl_ab_toxin | 7 | 100% | 19% | 24% | 17% |
| **beta_lactamase** | 14 | **21%** | **30%** | **5%** | **0%** |
| clostridial_neurotoxin | 6 | 100% | 20% | 17% | 2% |
| contact_dependent_inhibition | 4 | 35% | 21% | 1% | 2% |
| pore_forming_cytolysin | 7 | 69% | 12% | 8% | 11% |
| rip_rrna_glycosidase | 7 | 100% | 29% | 21% | 8% |
| superantigen_enterotoxin | 7 | 100% | 6% | 15% | 1% |
| t3ss_effector_apparatus | 10 | 80% | 14% | 21% | 22% |
| *virulence_associated_non_toxin* | *10* | *50%* | *2%* | *5%* | *6%* |

The probe leads phmmer by **+59.8 points** and jackhmmer by **+65.1** on average, and **neither profile
method beats it on any of the nine classes**.

**Beta-lactamase does not stay an exception under profile methods.** §7 reports it as the one class where
alignment beats the probe, 30% against 21%. phmmer reaches 5% there and jackhmmer 0%, so the probe leads
profile-based homology search on that class by 17 to 21 points. The alignment result on beta-lactamase is
specific to Smith-Waterman, which returns a graded score for every pair, rather than a general property
of homology search.

⚠️ **The external-database variant was piloted, and it cannot work under this margin.** The obvious
objection to everything above is that the panel is homology-screened, so a held-out class has no
homologue present for a profile to recruit, and profile methods earn their sensitivity by recruiting.
`src/03m_swissprot_profile_pilot.py` builds the profile from Swiss-Prot instead: for each held-out
ribosome-inactivating protein it runs jackhmmer against 575,748 sequences, builds an HMM from the
recruited alignment, and scores the panel with that.

**Profile construction succeeds and the margin still fails.** All seven profiles found all six of their
held-out siblings, so these are strong family models. Their best score against a *training positive* was
0.2 to 5.1, against a *negative* 2.9 to 10.0, and the margin came out negative on **six of seven members,
mean −4.2**.

The reason is structural. The decision rule asks whether a query resembles a known hazard more than a
benign protein, and under leave-one-mechanism-out the known hazards are the *other* mechanism classes.
Enriching a query's profile makes it a sharper model of the query's own family, which is exactly the thing
held out of the comparison. A better profile is not a better hazard detector here.

🔑 **This is the cleanest statement of why homology search cannot do this task.** It recognizes what it
has already seen. The probe reaches unseen mechanisms because the representation places them near seen
ones in one shared space, and a profile has no shared space to place anything in.

It also settles the §9.3 parity idea rather than leaving it open. Giving the homology baseline a larger
database does not hand it the foundation model's advantage, because the two use a database differently:
one matches against it as a labelled reference, the other was shaped by it as a pretraining corpus.

🔴 **One run was discarded, and the reason is recorded rather than dropped.** jackhmmer without `--max`
returned a matrix 1.3% dense against phmmer's 6.4%. Most margins were then exactly 0, the calibrated
threshold became 0, and `margin >= 0` passed nearly everything, so every class read **100%, including the
labelled control**. That is absence of power, not sensitivity, and it repeats the signature of the
attempt-2 failure in [`docs/EXTERNAL_VALIDATION_PREREGISTRATION.md`](EXTERNAL_VALIDATION_PREREGISTRATION.md).
Matching the filters fixed it. The script now refuses to report any class whose calibrated threshold is 0,
and the first version of that guard also required more than half the calibration margins to be 0, which
would have missed this case at 49%.


## 8. The obvious ensemble is worse

`src/03l_ensemble_alignment_embedding.py`. Alignment and embeddings fail on **disjoint** classes, so
combining them looks free. It is not.

Across classes, embedding and alignment recovery are **negatively** correlated (Spearman −0.25), so the
complementarity is real. Under a 5% false-positive budget, 20 seeds:

| method | mean recovery |
|---|---|
| embedding probe alone | **73.1%** |
| learned stack over both features | 73.1% |
| split-FPR OR | 71.8% — and it busts the budget, achieving **6.5%** FPR |
| max-ensemble | 57.8% |
| alignment alone | 18.1% |

**No combination beats the embedding probe alone.** The learned stack ties it by ignoring the alignment
feature.

The general reason is worth stating: **OR-ing two detectors raises the negatives' scores too, which pushes
the calibrated threshold up, so a union is not free under a fixed false-positive budget.** Tuning further
until something looked better is the failure mode this project documents, so it was stopped and recorded.

### 8.1 Nor does fusing two protein language models

`src/03o_plm_fusion_baseline.py`. VF-Fuse ([Briefings in Bioinformatics 2025,
bbaf481](https://academic.oup.com/bib/article/26/5/bbaf481/8260786)) predicts virulence factors by fusing
ESM-2 with ProtT5 along two paths. This project runs both models, but only ever as separate arms, so the
fusion question was open. The two are genuinely complementary per class, which is what made it worth
running: ProtT5 leads ESM-2 by 17.6 points on pore-forming cytolysin and 14.2 on contact-dependent
inhibition, and trails it by 25.0 on the virulence control and 10.5 on beta-lactamase.

| arm | mean recovery |
|---|---|
| ESM-2 650M | **72.5%** |
| ProtT5-XL | 72.1% |
| concatenated, z-scored | 72.1% |

Fusion costs **0.4 points**. Per class it tracks the mean of its two inputs rather than the better of
them: it keeps neither ProtT5's advantage on pore-forming cytolysin nor ESM-2's on the virulence control.
So §8's result holds for a second kind of combination. Two complementary detectors do not make a better
one under a fixed false-positive budget, whether the second is an alignment score or another language
model.


## 9. Not the pooling, the head, the scale, the structure, or the lineage

Each of these was run as a candidate explanation for the beta-lactamase failure. All were run on the same
80-protein panel with the same downstream analyses, so an arm cannot look different merely because it was
measured differently.

| model | dim | baseline | β-lact | T3SS | pore | superAg | clostr | RIP | ADPr | CDI |
|---|---|---|---|---|---|---|---|---|---|---|
| ESM-2 650M (mean) | 1280 | 0.974 | 21% | 80% | 69% | 100% | 100% | 100% | 100% | 35% |
| ESM-2 8M | 320 | 0.945 | 1% | 38% | 71% | 89% | 100% | 71% | 69% | 50% |
| ESM-2 35M | 480 | 0.947 | 13% | 74% | 77% | 100% | 100% | 97% | 97% | 60% |
| ESM-2 150M | 640 | 0.938 | 11% | 54% | 91% | 100% | 100% | 100% | 60% | 30% |
| ESM-2 3B | 2560 | 0.974 | 16% | 80% | 74% | 100% | 100% | 94% | 100% | 50% |
| ESM-2 650M (max) | 1280 | 0.957 | 0% | 72% | 57% | 100% | 100% | 80% | 100% | 30% |
| ESM-2 650M (CLS) | 1280 | 0.946 | 13% | 84% | 43% | 100% | 90% | 100% | 97% | 50% |
| ESM-C 300M | 960 | 0.947 | 16% | 72% | 94% | 100% | 100% | 94% | 89% | 35% |
| **ESM-C 600M** | 1152 | 0.963 | **51%** | 80% | 94% | 100% | 100% | 100% | 100% | 50% |
| **ESM-C 6B** | 2560 | 0.957 | 4% | 84% | 60% | 100% | 100% | 100% | 100% | 45% |
| ESM-3 1.4B | 1536 | 0.938 | 1% | 72% | 97% | 100% | 100% | 100% | 100% | 50% |
| ProtT5-XL | 1024 | 0.949 | 3% | 80% | 86% | 100% | 100% | 94% | 100% | 40% |
| SaProt-650M | 1280 | 0.949 | 10% | 80% | 86% | 100% | 100% | 97% | 94% | 55% |

**Scale is not the fix.** Beta-lactamase runs 1%, 13%, 11%, 21%, 16% across the ESM-2 ladder from 8M to
3B — no trend. Nor is scale a general fix for negative-set fragility: the class ordering of that fragility
is uncorrelated between the smallest and largest ESM-2 (Spearman −0.13 for 8M against 650M, −0.17 for 8M
against 3B). Scaling redistributes which classes carry the fragility rather than removing it.

**Pooling is not the fix.** Max pooling drives beta-lactamase to 0% and CLS to 13%, against mean at 21%.

**Structure is not the fix.** SaProt with real AlphaFold structures for 231 of 234 panel proteins reaches
10%, below plain ESM-2.

**And not the lineage either, which took a second correction to establish.** Earlier write-ups said
beta-lactamase resists every configuration tested and that plain alignment beats every embedding method on
it. **Both were wrong:** ESM-C 600M recovers **51%**, above alignment's 30% and more than double ESM-2
650M. That error predated the panel expansion — ESM-C 600M already scored 48.6% on the 66-protein panel —
and survived because the class was summarized from the ESM-2 arms without checking the ESM-C row.

The obvious reading of that exception was that ESM-C's pretraining corpus explains it. ESM-C saw UniRef
83M clusters plus MGnify 372M plus JGI 2B, with metagenomic data at 37.5% of the final training mix, and
beta-lactamases are among the most diverse families in environmental metagenomes. That reading predicts
the effect should strengthen with capacity on the same corpus. **It was tested and it is wrong.**

| ESM-C, identical corpus, identical bf16, identical pipeline | β-lactamase@95 | @99 |
|---|---|---|
| 300M | 15.7% | 4.3% |
| **600M** | **51.4%** | **21.4%** |
| **6B** | **4.3%** | **0.0%** |

Twenty times the parameters on the same data recovers **less than a twelfth** of what 600M does, and less
than 300M does. So neither corpus nor capacity accounts for it, and the exception narrows rather than
resolves: **ESM-C 600M is a single anomalous configuration whose cause is not identified by anything
measured here.**

🟢 **This makes the scale conclusion stronger, not weaker.** The ESM-2 ladder already showed no trend from
8M to 3B, but it held architecture generation fixed while varying scale. The ESM-C family holds the
*corpus* fixed across a twentyfold range and the result is **non-monotonic in three of nine classes and
monotonically decreasing in a fourth**: pore-forming cytolysin loses 34 points going from 600M to 6B.
Scaling redistributes which classes a representation handles; it does not lift them together.

### 9.1 🔴 Every recovery number here uses the worst of four classifier heads

`src/03j_classifier_sweep.py`, 30 seeds. The probe throughout this document is **logistic regression**. It
is the *worst* of four heads on 7 of 14 arms and the best on 2, and on **12 of 14 arms some other head
does better** — by a median of 5.1 points:

| arm | logistic | SVM-RBF | random forest | k-NN (5) |
|---|---|---|---|---|
| ESM-2 650M | **72.5%** | 77.5% | 72.1% | 77.0% |
| ESM-2 8M | **54.8%** | 68.9% | 63.7% | 62.3% |
| ESM-2 3B | 74.8% | 78.6% | 74.8% | **82.7%** |
| ESM-C 600M | 80.6% | 82.2% | 80.8% | **84.1%** |
| **ESM-C 6B** | 71.9% | 73.7% | **79.0%** | 75.7% |
| ESM-3 1.4B | 70.7% | 74.7% | 75.8% | **76.1%** |
| ProtT5-XL | 72.1% | **74.0%** | 72.6% | 71.2% |
| SaProt-650M | 71.4% | 74.9% | **77.2%** | 73.4% |

The gap is 5.0 points at 650M — where logistic is second-worst, random forest is worst — and **14.1 points
at 8M**, where logistic is worst outright. The two arms where logistic wins are ESM-2 35M and CLS pooling,
both by 0.0 points, i.e. ties. Every recovery figure in this document is therefore close to a lower bound on
what the representation supports, and should be read that way.

**But the head does not change the conclusions**, and the reason is §4. Per class at 650M, the head only
moves the classes that have intermediate members:

| class | logistic | best head | delta |
|---|---|---|---|
| pore_forming_cytolysin | 68.6% | 92.9% (k-NN) | **+24.3** |
| contact_dependent_inhibition | 37.5% | 50.8% (k-NN) | +13.3 |
| virulence_associated_non_toxin | 51.3% | 60.7% (SVM) | +9.3 |
| beta_lactamase | 15.7% | **21.0%** (k-NN) | +5.2 |
| clostridial / RIP / superantigen / T3SS | 100 / 100 / 100 / 80% | identical | **+0.0** |

The four saturated classes and T3SS are unmoved to the decimal. **This is the same structure as §4 and §5,
arriving from a third direction: the head, the panel, and the negative set all move only the members near
the threshold.** And beta-lactamase reaches 21% with the best of four heads, so the head is not what
rescues it either — §9 shows what does.

### 9.2 Strictness: where each class stops being recoverable

`src/03f_coverage_strictness.py` sweeps the false-positive budget and reports **s90**, the strictest
specificity at which a class still reaches 90% recovery. The sweep stops at the **estimable ceiling**
(0.987 with 77 held-out negatives) rather than extrapolating past what the panel can measure.

| class | s90 |
|---|---|
| adp_ribosyl, clostridial, RIP, superantigen | **0.985** (at the ceiling) |
| pore_forming_cytolysin | 0.885 |
| contact_dependent_inhibition | 0.670 |
| t3ss_effector_apparatus | 0.665 |
| *virulence_associated_non_toxin* | *0.560* |
| **beta_lactamase** | **never** — does not reach 90% at any specificity |

### 9.3 🔴 The class is held out of the probe, not out of the pretraining

Every arm in the table above is a supervised probe on top of a pretrained representation, and
**leave-one-mechanism-out removes the class from the probe's training set, not from the foundation model's.**
Every model here has seen ricin, Shiga toxin, botulinum neurotoxin and thousands of beta-lactamases during
pretraining, because they are in the public sequence databases these models were trained on.

So what this document measures is: **can a probe trained without class X locate class X inside a
representation that has already seen it.** That is a real and useful question — it is the question a
screening operator faces, since they will be building on a public foundation model — but it is not the same
as asking whether the system generalises to a function no model has ever encountered. Nothing here speaks
to that stronger case.

🔴 **This also confounds the cross-model comparison in §9, and the confound is not uniform.** The arms were
pretrained on very different corpora:

| lineage | pretraining corpus |
|---|---|
| ESM-2 (8M – 3B) | UniRef, curated sequences only |
| **ESM-C (300M, 600M)** | UniRef 83M clusters **plus MGnify 372M plus JGI 2B**, metagenomic data 37.5% of the final training mix |
| ESM-3, ProtT5, SaProt | differ again, and are not compared on this axis here |

Beta-lactamases are among the most abundant and diverse families in environmental metagenomes, so an
ESM-C model has plausibly seen far more beta-lactamase diversity in pretraining than any ESM-2 model. That
was the obvious candidate explanation for the anomaly in §9, and it was tested rather than assumed.

🔴 **It does not survive.** ESM-C 6B shares the corpus and the loading precision with 300M and 600M and
recovers **4.3%** of beta-lactamase, below both. Corpus cannot explain an effect that reverses across a
twentyfold capacity range on that same corpus, and capacity cannot explain a non-monotonic curve. The
anomaly is narrower than it was: not a lineage, not a corpus, not a scale, but **one configuration**.

⚠️ **Two variables remain uncontrolled across lineages, and neither can be fixed after the fact.**
Pretraining corpus is one, as above. The other is numerical: the `esm` package loads ESM-C in **bfloat16**
on GPU, while the ESM-2 arms run in fp32, and the installed stack has neither Transformer Engine nor a
fused attention kernel so both fall back to pure PyTorch. The package's own warning states that residual
stream differences shrink to a few ULP after the final LayerNorm and perplexity stays within rounding
noise, and the mean embeddings used here are post-LayerNorm, so this is unlikely to move recovery numbers.
It is recorded because it is an uncontrolled difference between lineages, not because there is evidence it
mattered. Comparisons **within** ESM-C are unaffected: all three sizes share corpus and precision.

#### 9.3.1 Two attempts to put a number on that caveat, and why both failed

The paragraph above is a limitation with no measurement attached. Two designs were tried on 2026-09-18.
Both failed, for reasons worth recording, because together they say what it would actually take.

**Attempt one, an external set of sequences ESM-2 provably never saw** (`src/03n_pretraining_holdout.py`).
ESM-2 was pretrained on UniRef50 release 2021_04, so a sequence whose own first version postdates that
release cannot have been in it.

⚠️ The obvious filter is wrong, and the error is easy to repeat. UniProt's `date_created` is the date an
entry entered **Swiss-Prot**, not the date its sequence appeared. `Q9JXM7` has `date_created` 2022-12-14
and a sequence last updated **2000-10-01**: it sat in TrEMBL for two decades and is certainly inside
UniRef50 2021_04. Filtering on `date_created` selects recently *reviewed* proteins, not recently
*discovered* ones, and it inflated the bacterial pool from 13 to 119. The correct filter is
`date_sequence_modified` with sequence version 1. Homology screening is still needed on top: the *E. coli*
OspC3 ortholog scores 0.955 normalized Smith-Waterman against the panel's *Shigella* OspC3.

🔴 **Both runs came back invalid, and the script now says so itself.** Length-matched negatives gave AUROC
**0.424**, below chance. Genus-matched negatives, following §2's pathogen-matched logic, gave AUROC
**0.556 with a bootstrap 95% CI of [0.345, 0.762]**, which contains 0.5. Recovery read 85% and 89% in the
two runs, and both figures are a blanket false-positive rate rather than detection: at an operating point
calibrated on the new negatives, recovery falls to 2% and 11%.

The reason is structural. **Changing the era changes the population.** Sequences new since 2021 cluster
into anti-phage defence systems and animal venom, because that is where recent sequencing effort went, and
venomous organisms are in Swiss-Prot almost exclusively for their toxins, so a taxon-matched benign
control for them cannot be built from the database at all. Novelty cannot be separated from everything
else that moved with it.

**Attempt two, correlating recovery with how much of each family was in pretraining**
(`src/03p_pretraining_exposure.py`), using UniRef50 cluster size as the proxy and needing no new
sequences. Pooling all 72 members gives Spearman **rho −0.399, p 0.0005**, in the direction opposite to
the caveat, and it does not survive:

| test | rho | p |
|---|---|---|
| pooled over 72 members | −0.399 | **0.0005** |
| within class, 5 recovery-varying classes | −0.139 mean | none below 0.05 |
| residuals after removing class means | −0.136 | 0.2554 |
| excluding beta-lactamase | −0.199 | 0.1334 |
| **class level, which is the real n = 9** | −0.331 | **0.3846** |

🔴 **The pooled result is pseudoreplicated.** Recovery is dominated by class structure, so the effective
sample size is the nine classes rather than the 72 members. This repository has already published one
pseudoreplicated p-value across four public surfaces, which is why the controls were run before the number
was written down rather than after.

What survives is descriptive and is one class rather than a trend: beta-lactamase is the most heavily
represented family here (median UniRef50 cluster 224, largest member 2,535) and the worst recovered at
21%. The counterexample sits in the same table, since T3SS effectors have the largest median cluster of
all at 317 and are recovered at 80%.

**So the caveat stands as written, unquantified.** What would settle it is pretraining a protein language
model with one family withheld and comparing, which is a training run rather than an analysis and is
outside what this study can do. Saying that is more useful than a number from a design that cannot carry
it.


### 9.4 🔴 A third candidate, preregistered, tested, and then found to have been tested wrong

Two explanations for the beta-lactamase anomaly had already been tested and refused. Not the classifier
head (§9.1, still the hardest class with the best of four heads). Not corpus, not capacity, not lineage
(§9.3, refuted directly on ESM-C 6B). A third candidate did not require any of those to be wrong, because
it questioned whether beta-lactamase belonged in the same comparison at all — and it has now been tested
too.

**The candidate.** Beta-lactamase entered this panel through a different route than the other eight
mechanism classes (§2): it carries UniProt **KW-0046, "Antibiotic resistance,"** not KW-0800 (Toxin) or
KW-0843 (Virulence), and [FunSoCs](https://pmc.ncbi.nlm.nih.gov/articles/PMC9119117/) treats antibiotic
resistance as a category distinct from toxin and pathogenesis mechanisms. All eight other classes here
require interacting with a host — a ribosome inactivated, a membrane perforated, a receptor bridged.
Beta-lactamase requires none of that; it hydrolyzes a diffusing small molecule in the periplasm. The
hypothesis: if the representation is keying on a *host-interaction* signature, any class that lacks one
should generalize poorly, not just this specific enzyme family.

⚠️ **The input to this test lived only in `/tmp` when the result was first published, and was gone
the next day, so for about a day nobody could reproduce it.** The eight sequences were recovered from
UniProt by the accessions in `results/v2/amr_category_test.json` and the recovery is byte-identical to the
lost file: 2511 bytes, sha256 `4df8a5c65ad684e31bebfb6a101cea7c6dca9bfd6307fa4f38a1a2a68edcc5d2`. It is
committed at `data/sequences/amr_category_test.fasta`, both scripts default to that path, and the audit
pins the bytes. Re-running reproduces 87.5% and 75.0% exactly. Logged in
[`docs/DATA_CORRECTIONS.md`](DATA_CORRECTIONS.md).

**Preregistered prediction** (`src/24_amr_category_test.py`, written before embedding): a second,
sequence- and fold-distinct antibiotic-resistance family — aminoglycoside-modifying enzymes, 8 members
spanning three fold families (GNAT acetyltransferase, nucleotidyltransferase, protein-kinase-like) plus a
bifunctional fusion, screened at ≤0.30 normalized Smith-Waterman against each other and against all 234
existing panel members, none carrying KW-0800/KW-0843 — should recover at **≤40%** if the hypothesis holds,
and be **refuted at ≥70%**. Trained on the full internal panel (external test, not an internal holdout),
scored at the same 95th-percentile-of-negatives threshold used throughout.

**First result: 87.5% (7 of 8), published here on 2026-09-11 as a refutation.** A second
antibiotic-resistance family, one that spans *more* fold diversity than beta-lactamase's own class A/C/D
serine-hydrolase members and requires no host interaction at all, appeared to generalize about as well as
the classes that do. Scores in that first run:

| member | fold family | score | flagged @95 |
|---|---|---|---|
| KKA2_KLEPN | kinase-like (APH) | 0.099 | yes |
| KKA7_CAMJU | kinase-like (APH) | 0.592 | yes |
| KANU_STAAU | nucleotidyltransferase (ANT) | 0.076 | yes |
| S3AD_ECOLX | nucleotidyltransferase (ANT) | 0.985 | yes |
| AAC6_SALEN | GNAT (AAC) | 0.004 | **no** |
| AACC3_PSEAI | GNAT (AAC) | 0.076 | yes |
| EIS_MYCTU | divergent GNAT (Eis) | 0.186 | yes |
| AACA_ENTFA | bifunctional AAC/APH fusion | 0.128 | yes |

The one miss does not track fold family: the GNAT sibling inside the bifunctional fusion was caught.

🔴 **That refutation was overstated, and the defect was in the test rather than in the number.** It
surfaced on 2026-09-11, from the question of whether beta-lactamase belongs in this panel at all. Two
things were wrong.

**Defect one, training-set contamination.** `24` fits on `vstack([P, N])`, the full internal positive set.
14 of those 80 positives are beta-lactamases, 17.5% of the positives and the largest class here. The probe
had already seen antibiotic-resistance enzymes when it was asked to reach a new antibiotic-resistance
family, so the 87.5% measures generalization inside a category the probe already knew. The hypothesis was
about something else: whether a representation trained on host-interacting toxins reaches antibiotic
resistance at all. `src/03b_leave_one_mechanism_out.py` does the opposite for every internal class, at
line 226, where the held-out class is dropped from training. The two numbers being compared came from
opposite training rules.

**Defect two, protocol mismatch.** The 87.5% was set against beta-lactamase's 21% from the LOMO table.
LOMO holds out 40% of the negatives and calibrates on the held-out ones, while `24` trains on every
negative and calibrates in sample. Holding the protocol fixed at `24`'s own, with each class removed from
training in turn, gives a column the 87.5% can actually be read against:

| class | protocol-24 recovery | LOMO recovery |
|---|---|---|
| adp_ribosyl_ab_toxin | 100.0% | 100.0% |
| clostridial_neurotoxin | 100.0% | 100.0% |
| contact_dependent_inhibition | 100.0% | 35.0% |
| pore_forming_cytolysin | 100.0% | 68.6% |
| rip_rrna_glycosidase | 100.0% | 100.0% |
| superantigen_enterotoxin | 100.0% | 100.0% |
| t3ss_effector_apparatus | 90.0% | 80.0% |
| *virulence_associated_non_toxin* | *70.0%* | *50.0%* |
| **beta_lactamase** | **50.0%** | **21.4%** |

Under its own protocol beta-lactamase recovers 50.0%, where the comparison had been quoting 21%. The
preregistered bounds in `24` were anchored to a figure from the other protocol.

**The corrected test** (`src/25_amr_category_test_v2.py`, preregistered before re-embedding) runs the 2×2
that `24` should have run, one protocol throughout:

| | test: beta-lactamase | test: aminoglycoside |
|---|---|---|
| train **with** AMR | 100.0% (in sample, ceiling) | 87.5% |
| train **without** AMR | **50.0%** | **75.0%** |

Re-embedding reproduced the original 87.5% (7 of 8) exactly, which is what makes the other three cells
comparable to it.

🔴 **Corrected verdict: inconclusive.** The decisive cell is **75.0% (6 of 8)**, between the corrected
preregistration's bounds of ≤62.5% and ≥87.5%. One member moved, AACC3_PSEAI, from 0.076 to 0.0053.

What the corrected numbers support:

- **The strong form of the hypothesis fails.** Under an identical training mask and protocol,
  aminoglycosides recover 75.0% where beta-lactamases recover 50.0%. Lacking host interaction does not by
  itself produce beta-lactamase's failure.
- **Contamination contributed without accounting for the result.** Removing AMR from training cost one
  member, moving 87.5% to 75.0%.
- **A weaker AMR cost stays live and unproven.** 75.0% sits below the 90 to 100% that host-interacting
  classes reach under the same protocol, on 8 members with one of them at the threshold.
- **Beta-lactamase is still the outlier** in every matched comparison, and still the lowest class in the
  protocol-24 column.

🔴 **This is the second preregistration in this project whose falsification criteria failed to cover the
outcome it got.** The first is recorded in
[`docs/EXTERNAL_VALIDATION_PREREGISTRATION.md`](EXTERNAL_VALIDATION_PREREGISTRATION.md), where attempt 2
returned 100% on every arm against criteria that set a floor and no ceiling. Here the criteria were
anchored to a figure computed under different machinery than the test itself used. Both are the same class
of mistake.

Standing after three candidates: the classifier head is refused (§9.1), corpus and capacity are refused
(§9.3), and the toxin/AMR category distinction is inconclusive in its weak form and unsupported in its
strong form, where this document previously read it as refuted. The anomaly is unexplained. Logged in
[`docs/DATA_CORRECTIONS.md`](DATA_CORRECTIONS.md).

### 9.5 🔴 One axis that DOES change the answer: which layer

`src/02h_esm2_layer_sweep.py`, `src/03q_layer_depth_sweep.py`. Section 9 is a list of things that turn out
not to matter. This is the exception, and it was never checked until 2026-09-18: every embedding in this
document is pooled from the **final** layer of ESM-2 650M. Pooling was swept and scale was swept. Depth
was not.

One forward pass yields all hidden states, so the sweep costs a single pass over the panel. Protocol is
03b's exactly, 30 seeds, only the pooled layer differs.

⚠️ The canonical arm was embedded on the cluster and the layer arms locally, so depth and platform
could confound. Re-embedding a sample at the final layer locally agrees with the canonical rows to a
**maximum absolute deviation of 2.4e-06**, relative 2.9e-07, so the comparison is clean.

| class | final (33) | L6 | **L12** | L18 | L24 | L30 |
|---|---|---|---|---|---|---|
| adp_ribosyl_ab_toxin | 99.5% | 41.9% | 99.0% | 100.0% | 98.6% | 90.0% |
| **beta_lactamase** | **15.7%** | 10.5% | **0.0%** | **0.0%** | **0.0%** | 3.8% |
| clostridial_neurotoxin | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| **contact_dependent_inhibition** | **37.5%** | 13.3% | **100.0%** | 100.0% | 97.5% | 94.2% |
| **pore_forming_cytolysin** | **68.6%** | 73.3% | **99.5%** | 100.0% | 94.3% | 84.3% |
| rip_rrna_glycosidase | 100.0% | 87.6% | 100.0% | 100.0% | 100.0% | 100.0% |
| superantigen_enterotoxin | 100.0% | 98.6% | 100.0% | 97.1% | 100.0% | 100.0% |
| t3ss_effector_apparatus | 80.0% | 25.3% | 80.0% | 78.7% | 76.3% | 80.0% |
| *virulence_associated_non_toxin* | *51.3%* | *22.3%* | *17.0%* | *10.3%* | *13.7%* | *31.3%* |
| **mean** | **72.5%** | 52.5% | **77.3%** | 76.2% | 75.6% | 76.0% |

**Layer 12 beats the final layer by 4.8 points**, and the gap survives the seeds: paired on the same 30
splits the difference is **+0.048 with a bootstrap 95% CI of [+0.034, +0.060]**, L12 wins on **25 of 30**
seeds, Wilcoxon **p = 2e-05**. L12 is also four times more stable across seeds, ±0.010 against ±0.038.

🔑 **The mean understates it, because the composition changes.** Contact-dependent inhibition goes
37.5% → 100% and pore-forming cytolysin 68.6% → 99.5%, while the labelled control falls 51.3% → 17.0%.
Excluding the control, the eight mechanisms go **75.2% → 84.8%**; excluding beta-lactamase as well, the
seven remaining go **83.7% → 96.9%**.

That the control falls is the useful part rather than a loss. `virulence_associated_non_toxin` is a
labelled non-mechanism and should be harder than a real mechanism, yet at the final layer it is recovered
*better* than contact-dependent inhibition. At layer 12 the real mechanisms sit near 100% and the control
at 17%, so the contrast the control exists to draw is drawn much more sharply.

**Beta-lactamase goes the other way and reaches exactly 0.0% at layers 12, 18 and 24.** The anomaly gets
deeper rather than resolving, which rules out "wrong layer" as its explanation and adds a fourth refused
candidate to §9.4's three.

What this does not do is overturn anything above. Recovery stays class-dependent and spans the full range
at every depth tested. What it does mean is that the figures in this document are read off a layer that is
not the best available for the task, which is the same conclusion §9.1 reaches about the classifier head
and for the same reason: **these numbers are closer to a lower bound than to a ceiling.**

⚠️ Five depths on a coarse grid, on one model. Layer 12 is the best of those tested rather than an
optimum, and nothing here says the same depth would be best for another family.


## 10. The one claim that looked like a competence boundary, and failed

### 10.1 What predicts whether a member is caught

`src/03g_member_separability.py` asks which measurable property separates the members that are caught from
those that are not. It uses only members whose behaviour is unambiguous — the **8 intermediate members from
§4.1 are excluded by construction**, leaving 64 of 72, split 44 caught against 20 not. Two-sided
permutation p-values over 20,000 label shuffles:

| feature | AUROC | perm p |
|---|---|---|
| **margin** (embedding distance to the nearest training positive, minus to the nearest negative) | **0.960** | **0.00005** |
| **nearest training positive, cosine** | **0.949** | **0.00005** |
| 5-mer similarity to training positives | 0.373 | 0.104 |
| nearest training negative, cosine | 0.420 | 0.317 |
| has a signal peptide | 0.425 | 0.288 |
| is exported | 0.455 | 0.565 |
| sequence length | 0.516 | 0.844 |

The two significant rows are at the resolution floor: 0 of 20,000 shuffles reached the observed AUROC, and
0.00005 is `1 / (20000 + 1)`. An earlier version of `perm_p` drew `n // 100` permutations, so a function
advertising 20,000 delivered 200 and its p-values had a floor of 0.005 while being read as far smaller;
see [`docs/DATA_CORRECTIONS.md`](DATA_CORRECTIONS.md).

**Embedding-space proximity predicts it at AUROC 0.96; sequence similarity does not** — the 5-mer feature
is at 0.373, which is worse than chance in the same direction, and localization and length are null. So
whatever decides recovery is a property of the representation, not of surface sequence identity, and not
of the confounds in §2.

A pooled effect can be class identity in disguise, so the same test is run **within** class. Margin holds
at AUROC 1.000 within pore-forming and within T3SS, and weakens to 0.667 within contact-dependent and
0.600 within virulence — the two classes with the least internal structure.

### 10.2 Does fitting a probe help at all?

`src/03h_probe_vs_similarity.py`, 30 seeds. If margin is what decides recovery, a fair question is whether
the trained probe earns its place. Two controls answer it. The first replaces the probe with **plain
nearest-neighbour cosine similarity to the training positives**. The second keeps the probe but subsamples
the training positives to a **fixed count for every class**, so a class cannot look easy merely because
holding it out removed less training data.

| arm | probe − similarity-only (pts) | probe − size-fixed probe (pts) |
|---|---|---|
| ESM-2 35M | +47.9 | -0.4 |
| ESM-2 150M | +34.2 | +0.4 |
| ESM-2 8M | +32.7 | +0.3 |
| 650M CLS | +14.7 | -0.3 |
| ESM-2 650M | +11.5 | -0.4 |
| 650M mean | +11.4 | -0.3 |
| ESM-C 600M | +10.2 | -0.1 |
| 650M max | +8.3 | +0.0 |
| ESM-2 3B | +6.6 | -0.4 |
| SaProt-650M | +6.2 | +0.5 |
| ESM-C 300M | +2.6 | -1.1 |
| ProtT5-XL | -1.6 | -0.1 |
| ESM-3 1.4B | -4.3 | +0.3 |
| ESM-C 6B | -5.4 | -0.3 |

**The size control is null everywhere**, within about a point on every arm. Class size was never driving
the differences between classes, which is worth having ruled out.

**The first control is not null, and it does not point one way.** The benefit of fitting a probe runs from
**+47.9 points down to −5.4**, and on **3 of 14 arms it is negative**: ProtT5-XL, ESM-3 1.4B, ESM-C 6B. On those,
a trained linear probe is *worse* than simply asking which training positive a held-out protein sits
closest to.

🟢 **The pattern within ESM-2 is the interesting part.** The benefit shrinks as the model grows: +32.7 at
8M, +47.9 at 35M, +34.2 at 150M, +11.5 at 650M, +6.6 at 3B. Read alongside §10.1, that is what you would
expect if larger models place hazardous proteins in a geometry that already separates them — the more the
representation encodes, the less a supervised layer adds, until it adds nothing and then costs something.

⚠️ It is a pattern across arms, not a controlled experiment. The arms differ in corpus and precision as
well as scale (§9.3), five ESM-2 points are not a trend line, and the three negative arms come from three
different lineages. What can be said is narrow and still useful: **on nearly half the representations
tested the probe adds less than ten points over nearest-neighbour distance, and on three it adds nothing at
all.** Any claim that a learned classifier is doing the work has to survive this control first.

### 10.3 The claim, and its failure

`src/03k_margin_holdout.py` turns that into a holdout: rank positives by margin, hold out the lowest, and
compare against random holdouts of the same size.

| holdout | recovery@95 |
|---|---|
| lowest margin | **14.7% ± 10.2** |
| random, class-matched | 80.0% ± 14.6 |
| random | 91.4% ± 10.0 |
| highest margin | 100.0% ± 0.0 |

That is **−65.4 points** against a class-matched random holdout and −76.7 against an unmatched one. Across
all 14 arms the gap runs 8.1 to 69.6 points, median 45.1, and **12 of 14 exceed the 25-point threshold** the
preregistration later used. The two that do not are both non-mean poolings of the same model — CLS at
+17.4 and max at +8.1 — so the effect is a property of mean-pooled representations rather than of every
representation.

That is the shape of a **competence boundary**: a statement, computable in advance, of which molecules the
system will miss.

It was preregistered with falsification criteria, the pipeline was frozen at a tagged commit before any
external data was fetched, and it was tested on two external panels. **It failed both times**, and was
downgraded to a property of the internal panel as the preregistration required. A defect in the
preregistration itself — it specified a floor but no ceiling, so a uniform 100% result is recorded as NOT
SUPPORTED when *uninformative* is more accurate — is recorded there too.

Full record: [`docs/EXTERNAL_VALIDATION_PREREGISTRATION.md`](EXTERNAL_VALIDATION_PREREGISTRATION.md).

## 11. What this does not claim

- **Not a better classifier.** DTVF (ProtT5 + LSTM/CNN) reports AUROC 0.92 on the standard 576/576
  virulence benchmark. The numbers here are on a self-built panel and are **not comparable**; presenting
  them as a win would be wrong.
- **Not novel on homology control.** Homology-clustered evaluation is established practice.
- **Not a competence boundary.** §10 was the attempt, and it failed externally.
- **Not deployment-ready.** This is a 234-protein research panel. Common Mechanism and SecureDNA are
  running in production; this is not in that category.
- **Not evidence that hazard is what is being detected.** See the provenance control in §2.
- **Not a test of generalisation to functions no model has seen.** The held-out class is removed from the
  probe's training, not from the foundation model's pretraining, and every class here is in the public
  databases these models were trained on. See §9.3.
- **Not a controlled comparison of pretraining corpora.** The arms differ in what they were pretrained on
  as well as in architecture and scale, and this study cannot separate those. See §9.3.

## 12. Reproducing

```bash
# panel and annotations are in the repository; embeddings are regenerated
PROJECT_DIR=$PWD PYTHON_BIN=path/to/python sbatch slurm/expanded_panel_full_sweep.sh

# every headline claim, recomputed from its artifact and matched to the documents
python src/22_claims_audit.py
```

The audit exits 1 if any claim disagrees with its artifact, and CI runs it on every push. It was added
after four separate number-drift defects were found by hand, each because someone happened to look; the
two entries covering this panel were added after two more. All are documented in
[`docs/DATA_CORRECTIONS.md`](DATA_CORRECTIONS.md).
