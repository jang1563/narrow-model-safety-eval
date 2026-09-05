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

## 9. Not the pooling, the head, the scale, or the structure — but partly the lineage

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

🔴 **The lineage is, partly, and this corrects an earlier claim.** Earlier write-ups of this work said
beta-lactamase resists every configuration tested and that plain alignment beats every embedding method on
it. **Both statements are wrong.** ESM-C 600M recovers **51%** of beta-lactamase — above alignment's 30%
and more than double ESM-2 650M. The error was not introduced by the panel expansion: on the previous
66-protein panel ESM-C 600M already scored 48.6%, so the claim was wrong when it was written. It was
carried because the class was summarized from the ESM-2 arms and the ESM-C row was not checked against it.

The corrected statement: **beta-lactamase is the hardest class for 11 of 12 configurations and the only
class where alignment beats the canonical ESM-2 probe, but it is not unreachable.** One representation
recovers half of it, and the jump is within a lineage rather than across scale — ESM-C 300M gets 16% and
ESM-C 600M gets 51%. Why that architecture and that size succeed where a 3B ESM-2 does not is not
explained by anything measured here.

### 9.1 🔴 Every recovery number here uses the worst of four classifier heads

`src/03j_classifier_sweep.py`, 30 seeds. The probe throughout this document is **logistic regression**. It
is the *worst* of four heads on 6 of 13 arms and the best on 2, and on **11 of 13 arms some other head does
better** — by a median of 5.0 points:

| arm | logistic | SVM-RBF | random forest | k-NN (5) |
|---|---|---|---|---|
| ESM-2 650M | **72.5%** | 77.5% | 72.1% | 77.0% |
| ESM-2 8M | **54.8%** | 68.9% | 63.7% | 62.3% |
| ESM-2 3B | 74.8% | 78.6% | 74.8% | **82.7%** |
| ESM-C 600M | 80.6% | 82.2% | 80.8% | **84.1%** |
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

### 10.2 The claim, and its failure

`src/03k_margin_holdout.py` turns that into a holdout: rank positives by margin, hold out the lowest, and
compare against random holdouts of the same size.

| holdout | recovery@95 |
|---|---|
| lowest margin | **14.7% ± 10.2** |
| random, class-matched | 80.0% ± 14.6 |
| random | 91.4% ± 10.0 |
| highest margin | 100.0% ± 0.0 |

That is **−65.4 points** against a class-matched random holdout and −76.7 against an unmatched one. Across
all 13 arms the gap runs 8.1 to 69.6 points, median 45.1, and **11 of 13 exceed the 25-point threshold** the
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
