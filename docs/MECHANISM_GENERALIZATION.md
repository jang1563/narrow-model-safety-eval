# Mechanism generalization: what a protein-embedding hazard probe does when the molecule is not on the list

**Written:** 2026-09-05 · **Updated:** 2026-09-18 · **Author:** JangKeun Kim, Weill Cornell Medicine ([ORCID 0000-0002-8733-9925](https://orcid.org/0000-0002-8733-9925))

**Panel:** unless a section says otherwise, every number is on **v2, 80 positives / 154 negatives**, which is frozen. **v3, 149 / 296**, adds three non-animal-target mechanism classes and appears in §2.5, §10.4 and the §2.4.1 comparison; it lives in a parallel file set so that nothing here becomes unreproducible. Scripts take `--panel`.

This document records the **leave-one-mechanism-out (LOMO)** line of work. It is separate from
[`docs/EVALUATION_REPORT.md`](EVALUATION_REPORT.md), which describes the structure-level metrics (FSPE,
FSI, Physical Realizability Tier) on a per-residue panel. The two lines ask different questions and should
not be read as one result.

Every number here is produced by a script in `src/` and stored in `results/v2/` (or `results/v3/` for the
v3 sections). The project's headline
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
  (secreted cell-wall, cytoplasmic housekeeping, secreted-from-pathogen). §2.5 describes **v3**, which
  adds 69 positives in three classes and 142 organism-matched negatives in a fourth block.
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

🔴 **Read 0.929 with §2.4.1 next to it.** That figure is a per-protein cross-validation, and target host is
nearly determined by mechanism class here, so it does not survive holding a class out. It is enough to
establish target host as a **confound**, which is all this section uses it for, and not enough to call target
host a generalizable axis.

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
from training before testing a non-animal class. `src/03t_animal_only_training.py` ran exactly that and
**refuted it**: the interaction was −0.0 points, 95% CI [−3.4, +3.7], so having other non-animal positives
in training does not help a held-out non-animal class. §5.1 attributes the two classes that moved.


### 2.4.1 🔴 That target-host number is mostly within-class, and does not survive class holdout

`src/03w_target_host_class_holdout.py`. §2.4's **AUROC 0.929** for target-host legibility is the figure a
species-stratified follow-up would rest on: it says a frozen representation already encodes what a protein
acts on. `src/03s_target_host_control.py` computed it with `StratifiedKFold(5)`, which shuffles **proteins**.
That matters here more than usual.

**Target host is very nearly a function of mechanism class in this panel.** **Eleven of the thirteen classes
have a single target** across every member with an assigned one, leaving only `other_toxin_mechanism` and the
virulence control carrying more than one. So a probe scored by per-protein cross-validation can reach 0.929
by recognising the class and reading the target off it:

| target | classes carrying it |
|---|---|
| animal | adp-ribosyl 7/7, clostridial 6/6, phospholipase 2/2, pore-forming 7/7, RIP 7/7, superantigen 7/7, T3SS 10/10 |
| **non-animal** | **beta-lactamase 14/14** (a small molecule), **CDI 4/4** (another bacterium) |
| both | the labelled virulence control, 3 animal against 3 non-animal |

**Producer taxonomy is not the confound, and that was checked before anything else was run.** The scope
question is whether the probe reads the producing organism rather than the target, since six of seven RIPs
are plant-produced and act on animal ribosomes. But `producer_kingdom` in `data/annotations/target_host_v2.json`
is `bacteria_or_virus` for **74 of 80** positives, the other six being exactly those RIPs. There is almost no
producer variance to exploit at kingdom level. Finer producer structure is a separate question, and §2.3's
provenance probe reaching 0.818 on hazard says it is not nothing.

Three arms, one probe, positives only so hazard is held constant:

| arm | what it asks | result |
|---|---|---|
| pooled per-protein CV | replicate §2.4 | **AUROC 0.930 ± 0.062** against the published 0.929 ± 0.065 |
| leave-one-mechanism-class-out | does target host reach an unseen class | **balanced class accuracy 0.500** |
| within the one mixed class | class identity held constant | **AUROC 0.000** on 3 against 3 |

🔴 **The middle arm is the answer, and 0.500 is exactly what always saying "animal" scores.** Seven of seven
animal classes are called correctly and **zero of two non-animal classes are**. Every held-out class comes
back animal, including the two whose target is not:

| held-out class | target | members called right | mean p(animal) |
|---|---|---|---|
| clostridial / RIP / superantigen / adp-ribosyl / phospholipase | animal | 100% | 0.94 to 1.00 |
| t3ss_effector_apparatus | animal | 80% | 0.744 |
| pore_forming_cytolysin | animal | 71% | 0.691 |
| **beta_lactamase** | **non-animal** | **7%** | **0.821** |
| **contact_dependent_inhibition** | **non-animal** | **25%** | **0.700** |

⚠️ **The preregistered test for this was the wrong instrument, which is worth stating plainly.** The plan
was a class-level permutation null, permuting which classes carry which target, on the grounds that effective
n is the number of classes. At nine classes split seven to two, **8% of the 200 draws reach a balanced
accuracy of 1.000** and the null's 95th percentile is exactly 1.000. So its p of 0.21 carries no information
and the preregistered verdict is **INCONCLUSIVE** by its own rule. Recording that the statistic was
underpowered is the honest report, rather than reading 0.21 as an absence of effect.

**Post hoc, and labelled as such since it was added after seeing the null behave that way:** ranking the nine
held-out classes by mean p(animal) gives a class-level AUROC of **0.786**, exact one-tailed p **0.167** over
all 36 orderings. So the direction is right and the separation is not usable. Two animal classes,
pore-forming at 0.691 and T3SS at 0.744, score **below** beta-lactamase at 0.821, which means no threshold
assigns the held-out classes correctly.

**The third arm is the cleanest and the smallest.** The virulence control is the only class carrying both
target kinds, so class identity is constant inside it. A probe trained on every other class separates its
three animal members from its three non-animal members at **AUROC 0.000**, perfectly inverted: exact
one-tailed p is 1.000 in the predicted direction and 0.050 inverted, over the 20 orderings that exist at 3
against 3. **One class and six proteins is a curiosity rather than a result.** Its direction agrees with the
middle arm, which is the only weight it carries.

🔑 **What this changes.** §2.4's finding stands as written: target host **is** an uncontrolled confound in
the hazard numbers, and the animal/non-animal split of hazard separability is real. What does not stand is
reading 0.929 as evidence that target host is a *generalizable* axis. It is a within-class number. Whether a
representation can tell what a protein acts on when the mechanism is new is **untested here**.

🔴 **And this panel cannot test it.** Non-animal target is carried by **two** mechanism classes. Hold one out
and the training set contains a single example of the category, which is the same structural poverty that
made §9.4 unsettleable. A design that both trains and tests on non-animal hazard needs non-animal-target
positives from **four or five additional mechanism classes**, which is panel construction rather than
analysis.

### 2.5 Panel v3, and why v2 stays frozen

`src/27_expand_nonanimal_classes.py`. §2.4.1 ended with a prerequisite: non-animal target was carried by
**two** mechanism classes, so leave-one-class-out could not both train and test on non-animal hazard, and
the class-level test had no usable null. `src/26_panel_growth_yield.py` measured which candidate families
could supply new classes under the panel's own admission rule, and three were admitted:

| class added | n | target | organism-matched negatives |
|---|---|---|---|
| `phage_peptidoglycan_hydrolase` | 32 | bacteria | 44 benign phage proteins, 34 phage species |
| `cry_insecticidal` | 22 | insect | 42 benign *Bacillus thuringiensis* proteins |
| `bacteriocin` | 15 | bacteria | 56 benign lactic-acid-bacteria proteins |

**v3 is 149 positives against 296 negatives**, prevalence 33.5%, eligible classes **8 to 11**, non-animal
target classes **2 to 5**.

🔴 **v2 is frozen and v3 is a parallel file set, which is not a stylistic choice.** Every number in this
document, all fourteen arms in §9, §2.4.1, §9.7 and every pin in `src/22_claims_audit.py` is computed on
v2's 80 and 154. Growing those files in place would make this document unreproducible from its own
committed inputs. v3 is v2 plus the new members in that order, so v2's rows are a prefix of v3's: the v2
embedding matches the first 80 rows of the v3 one to 5e-06, and re-running `03b` on v2 reproduces every
class number exactly. `03b`, `03s` and `03w` take `--panel`, which switches the results directory and the
annotation files together so the two can never be mixed.

**The negatives are the reason this is not a one-afternoon job.** The panel held no phage proteins and no
*B. thuringiensis* before v3, so adding cry toxins and phage enzymes without matched negatives would have
made the producing organism the signal. That failure is measured twice in
`docs/DATA_CORRECTIONS.md`: length-matched-only negatives gave AUROC **0.424**, below chance, and
genus-matched gave 0.556 with an interval containing 0.5. Max similarity from a new negative to a new
positive is **0.062**, and none of the 142 new negatives carries a hazard keyword.

⚠️ Four label defects were caught during harvesting and are recorded because each would have put a wrong
label into a hazard panel. `protein_name:colicin` admitted eight entries that are not bacteriocins: the
colicin I receptor, which is a protein on the *target* cell, the colicin V secretion ATPase, and **six
immunity proteins, which are what a producer uses to survive its own colicin**. Selection moved to the
UniProt keyword, checked case by case: Colicin E1, N, B and A carry `KW-0078` and Colicin-M immunity
protein does not. One sequence, `O03979`, was admitted as a positive **and** as a negative in the same run,
because the deduplication sets were seeded from the old panel only. The phage class could not be called
`endolysin`, since `KW-0081` on viral producers also returns virion-associated hydrolases (T4 Gp5, T5 pb2,
phi29 morphogenesis protein 1, T7 gp16) that share the catalytic mechanism but are not endolysins, so the
class is named for what its members do. Three archaeal halocins were dropped because `producer_kingdom`
has no archaeal value and §2.4.1's producer argument depends on that coding being honest.

⚠️ **The SaProt arm is not evaluated on the three new classes.** `foldseek` is fetched as a Linux binary
and the existing 3Di strings were produced on the HPC, so new members carry the `no_structure` mask, which
the coverage audit accepts as coverage. Every other arm is unaffected. Only ESM-2 650M mean-pooled is
embedded for v3 so far.

🟢 **§2.4.1's verdict changes on v3, in the direction that section predicted.** Its class-holdout test came
back INCONCLUSIVE with a null so wide it was useless, and it said the cause was nine classes split seven to
two. With eleven classes and four non-animal:

| | v2, 9 classes | v3, 11 classes |
|---|---|---|
| balanced class-level accuracy | 0.500, the majority baseline | **0.804** (animal 0.86, non-animal 0.75) |
| class-level permutation p | 0.21 | **0.0150** |
| null draws reaching a perfect score | 8% | **0%** |
| null 95th percentile | 1.000, so uninformative | 0.682 |

By the preregistered rule that is **SUPPORTED**: target host survives leave-one-mechanism-class-out. So
§2.4.1's finding stands as written and its diagnosis was right. What was missing was class diversity, and
the pooled 0.929 remains a within-class number that never established this.

⚠️ One interpretable side effect. `pore_forming_cytolysin`'s target-host accuracy falls from 71% to 29%
once bacteriocins are in the panel labelled non-animal. Bacteriocins and pore-forming cytolysins both kill
by making holes in a membrane, so labelling one family non-animal pulls the other toward it. The axis is
real and it is not clean.

### 2.6 🔴 The negatives have no test set, and every specificity here is measured on the set that chose the threshold

`src/45_negative_test_set_audit.py`. `03b` produces every leave-one-mechanism-out number in this
document, and it splits the negatives **once**:

```python
nte, ntr = nperm[:ncut], nperm[ncut:]        # 40% / 60%
Xtr = np.vstack([P[tri], N[ntr]])            # train on the 60%
s_nte = model.predict_proba(N[nte])[:, 1]
t95 = threshold_at_specificity(s_nte, 0.95)  # threshold FROM the 40%
...
(s_nte >= t95).mean()                        # "realised FPR", on the SAME 40%
```

The variable is named `nte`, as in negative test, and it is a **calibration** set: the threshold is its
0.95 quantile. `03b`'s own docstring says as much, "makes FPR 5% by construction". So the positives have a
test set, the held-out mechanism class, and the negatives never do. A sweep of all 45 scripts in `src/`
finds **no three-way negative split anywhere**. Every "at 95% specificity" in this repository is a
within-calibration-set specificity, and the rate on negatives the threshold has never seen had never been
measured.

**The measurement.** Training negatives stay at the published 178, and the published 118 is divided into m
calibration and 118 − m test, so the published protocol is the **m = 118 endpoint with an empty test set**.
300 seeds, no class held out, on both arms that have the data:

| m | test n | in-sample FP | out-of-sample FP, 650M | out-of-sample FP, 35M | order-statistic bracket | conformal arm, 650M |
|---|---|---|---|---|---|---|
| 20 | 98 | 5.00% | **8.64% ± 0.35** | **8.79% ± 0.33** | [4.76, 9.52] | 4.65% (≤4.76 held) |
| 30 | 88 | 6.67% | 7.10% ± 0.27 | 7.58% ± 0.28 | [6.45, 9.68] | 2.86% (≤3.23 held) |
| 45 | 73 | 6.67% | 6.91% ± 0.26 | 6.81% ± 0.25 | [6.52, 8.70] | 4.27% (≤4.35 held) |
| 60 | 58 | 5.00% | 6.47% ± 0.26 | 6.20% ± 0.24 | [4.92, 6.56] | 4.87% (≤4.92 held) |
| 78 | 40 | 5.13% | 5.87% ± 0.27 | 5.47% ± 0.24 | [5.06, 6.33] | 3.63% (≤3.80 held) |
| 98 | 20 | 5.10% | 5.58% ± 0.34 | 5.38% ± 0.31 | [5.05, 6.06] | 3.87% (≤4.04 held) |
| **118** | **0** | 5.08% | **never measured, this is `03b`** | | [5.04, 5.88] | |

🔑 **`np.quantile(s, 0.95)` cannot deliver 5% out of sample at these sample sizes, and not because of
noise.** With m calibration points the achievable exceedance rates are the discrete set `{j/(m+1)}`. At
m=30 the neighbours are 1/31 = 3.2% and 2/31 = 6.5%, and **5% is not among them**. Interpolating does not
create the missing rate; it lands between two order statistics and the realised rate is bracketed by them.
At **4 of the 6** sizes the bracket's *lower* edge already exceeds 5%. The observed mean falls inside the
bracket at **6 of 6 sizes on both arms**, so this is the estimator behaving predictably rather than a
model failing.

At the smallest calibration set the gap is not subtle: **8.6%** measured against **5%** stated, 1.7 times
nominal, and individual splits reach **36.7%**.

🟢 **The fix is arithmetic and it is demonstrated rather than recommended.** Taking the k-th largest
calibration score with k = ⌊(m+1)α⌋ gives the conformal guarantee that a fresh negative exceeds it with
probability ≤ k/(m+1) ≤ α. That arm **held its guarantee at every size on both arms**, at the price of
running conservative, 2.9% to 4.9% where 5% was asked for. A finite-sample bound is available; an
interpolated point estimate is not.

⚠️ **The conformal half of this is not new, and the survey already says so.** CRC-Screen
([arXiv 2605.00074](https://arxiv.org/abs/2605.00074), MusIML @ ICML 2026) applies Conformal Risk Control
to hazard screening and states the same slack floor from the same arithmetic: its certifiable miss rate is
capped at `1/(n_cal + 1)`, which is 1.77% at their n_cal ≈ 55, and a procurement-grade budget needs an
order of magnitude more calibration data. That the calibration set, not the model, is the binding
constraint is **published**. What §2.6 adds is the same limit measured on the **specificity** side of a
mechanism panel rather than the miss-rate side of a taxonomic one, with the extra observation that the
achievable operating points are a discrete set and the nominal 5% is usually not in it. See
`research/05_v2_related_work_survey.md` §1.1b for the full comparison, including that CRC-Screen's own slack
term is a coarse approximation and not a bound.

⚠️ **What this does and does not do to the numbers in this document.** The published protocol uses the
largest calibration set available, all 118, which is the best case in the table: its bracket is
[5.04, 5.88], so its true out-of-sample rate is within about 0.9 points of what it claims. The published
figures are not overturned. What was wrong is that the rate was **never measured**, and what the table adds
is that the error grows sharply as calibration data shrinks.

🔴 **And the classes most sensitive to it are the ones §10 is about.** Panel B re-runs the recovery table
at each m. Sorted by how far each class moves between m=20 and m=118 on the canonical arm:

| class | m=20 | m=118 | swing |
|---|---|---|---|
| virulence_associated_non_toxin | 44.3% | 31.3% | **+13.0** |
| phage_peptidoglycan_hydrolase | 21.0% | 12.2% | **+8.9** |
| beta_lactamase | 26.9% | 21.2% | **+5.7** |
| adp_ribosyl_ab_toxin, clostridial_neurotoxin | 100.0% | 100.0% | 0.0 |

The classes at the ceiling do not move at all and the low-recovery classes move most, because recovery and
the realised false-positive rate rise together when the threshold loosens. So **"recovery at 95%
specificity" conflates the two whenever the calibration set is small**, which is the same operating-point
confusion §10.9.1 found in the pool experiment, arriving here from calibration size alone.

🔑 **This also explains §10.8's requirement rather than restating it.** §10.8 says a one-in-ten-thousand
budget needs **250,003** panel negatives, 100,001 of them in the calibration split, derived from wanting
ten negatives above the threshold. The order statistics give the same number for a stated reason: k=10
granularity at α=10⁻⁴ needs m ≥ 99,999 calibration points, which is **249,998** at a 40% split. Two
derivations, one number, and now with the reason attached: the calibration set size fixes the **granularity
of the achievable operating points**, not merely the precision of one.

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

### 2.6.1 🔑 The panel now has a test set, two ways, and the false-positive budget decomposes

`src/48_conformal_lomo_test_split.py` and `src/49_external_test_partition.py`, added 2026-09-21.
§2.6 established two things and then nothing acted on either: there is no test partition, and the
published `np.quantile` estimator does not return its own nominal rate on negatives held back from
calibration. It also identified the conformal threshold as the fix and **that fix was never applied**:
every recovery number in this document still comes from `np.quantile`, at `03b` line 93 and in the
same call in 03e, 03f, 03h, 03j and 15e.

**Route 1, test set from inside the panel (`src/48`).** Split the negatives 177 / 59 / 60 instead of
178 / 118 / 0, set the threshold on calibrate, measure the false-positive rate on test. 200 seeds, both
estimators on identical splits, models and scores, with the **seed as the unit of inference** because
all twelve classes share a seed's negative split and pooling class-by-seed observations would treat 30
independent splits as 360.

| arm | nominal | `np.quantile` | conformal |
|---|---|---|---|
| canonical 650M | 5% | 6.40% [5.82, 6.98] **exceeds** | 4.98% [4.45, 5.51] covers |
| canonical 650M | 1% | 2.84% [2.44, 3.24] **exceeds** | unreachable at m = 59 |
| esm2_35M | 5% | 6.67% [6.09, 7.24] **exceeds** | 5.15% [4.62, 5.69] covers |
| esm2_35M | 1% | 2.90% [2.52, 3.29] **exceeds** | unreachable at m = 59 |

⚠️ **Settling this took three attempts and the first two were wrong in opposite directions.** At 30
seeds on one arm conformal looked like it held; at 30 seeds on two arms the arms looked like they
disagreed; only at 200 seeds is the point estimate within 0.15 points of theory in both. The reason 30
seeds could not settle it is arithmetic: with m = 59 and alpha = 0.05, `(m+1)*alpha` is **exactly 3**,
so the guarantee is exactly 3/60 = 5.00% and the estimator is exactly calibrated with **zero
conservatism margin**, which puts about half of all point estimates above nominal by construction.
m = 59, 79, 99 and 119 all hit that integer at alpha = 0.05; m = 58 gives k = 2 and a genuinely
conservative 3.39%. **The margin is not monotone in m**, so a calibration size should be chosen to
miss the integer rather than to be as large as possible. Ties were ruled out first: 59 unique
calibration scores, zero exact ties at the threshold, strict `>` gives the same answer as `>=`.

Route 1 costs two things, and both are artefacts of where the test set came from rather than
properties of the panel: calibration halves to 59, so its recovery figures are **not** comparable to
the tables in this document, and conformal goes out of reach at a nominal 1%.

**Route 2, test set from outside the panel (`src/49`).** Take the test negatives from the
8,259-protein benign pool, dropping `Q8X739` per §10.9's rule. Calibration then stays at the
**published 118**, where conformal is reachable at both budgets with real conservatism: k = 5 for a
4.20% guarantee at nominal 5%, k = 1 for 0.84% at nominal 1%. Three arms separate the estimator from
the data, 200 seeds, nominal 5%, `esm2_35M`:

| arm | `np.quantile` | conformal |
|---|---|---|
| pool to pool, **exchangeable** | 5.86% [5.56, 6.16] **exceeds** | **4.23%** [3.98, 4.49] covers |
| panel to pool, deployment shift | 7.94% [7.67, 8.20] exceeds | 6.18% [5.96, 6.41] **exceeds** |
| panel to pool, **distinct names only** | **10.29%** [9.91, 10.67] exceeds | 7.79% [7.46, 8.13] exceeds |

Read down the conformal column and the budget decomposes:

```
4.20%   the guarantee the arithmetic promises
4.23%   what it delivers when the negatives really are exchangeable
6.18%   after calibration and test negatives come from different curations
7.79%   after the pool's duplicate names stop hiding the failures
```

**The theory is exact to within 0.03 points, and every point of overshoot above that is bought by
distribution shift and by name redundancy rather than by the estimator.** Same pattern at nominal 1%:
conformal gives 0.78% against a 0.84% guarantee under exchangeability and 1.31% under shift, against
`np.quantile`'s 1.92% and 3.10%.

Two results fall out of that table which §2.6 could not have seen. First, `np.quantile` **exceeds
nominal even under exchangeability**, 5.86% against 5%, so part of its overshoot was never about shift
at all. Second, and this is the one that should change how §10.9's pool numbers are read: collapsing
the pool to one protein per distinct name **raises** the measured false-positive rate, 7.94% to
10.29%. The duplicated entries are the easy ones, so a rate computed over 8,258 raw proteins flatters
itself by about 2.4 points against the same rate over 3,407 distinct names. **Effective n does not only
widen intervals, it moves point estimates**, which is stronger than the effective-n argument in §10.9.

So the defensible figure for a nominal 5% budget on this panel is close to **8%**: better estimator,
de-duplicated, out-of-distribution negatives. Not 5%.

**What replicates, what does not.** Across the two arms in route 1, the quantile overshoot replicates
at both budgets, conformal is closer in both, the 1% unreachability is structural, and within each arm
the estimator preserves the bottom-three ordering, so §10.4's failure story is **not** an artefact of
the threshold rule. What does not replicate, and survives 200 seeds so it is a property rather than
noise: the bottom-two **set** agrees, phage and beta-lactamase in both arms, but the order inside it
**flips**, phage worse on canonical (13.5% against 26.8%) and beta-lactamase worse on `esm2_35M` (9.1%
against 28.4%). Which class is worst is arm-dependent where which two classes are the problem is not,
so claims belong at the set and not the ranking. That is §10.4's joint-property point with the
representation as the joint term.

⚠️ **Route 2 is single-arm and PROVISIONAL.** Pool embeddings existed only for `esm2_35M`, and the
`embed_pool()` docstring in `src/35` already recorded why that arm is weak for per-class work:
beta-lactamase is **already floored at 1.4%** there, so the arm has headroom in only one of the two
failing classes. Its per-class column is half a test. The false-positive decomposition does not depend
on class headroom and is the durable part. What supports the run meanwhile is that the same script
reproduces this document's published per-class figures at the published seed count, beta-lactamase at
1.4% exactly, and diverges at 200 seeds only for the class §10.6.1 already documented as seed-fragile.

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

*Thirteen rows, and the text says "14 arms" throughout. The fourteenth artifact set is the canonical run
repeated under the explicit name `esm2_650M_mean`, kept as a consistency check and agreeing with the first
row to zero, so it is not given a duplicate row here. §10.6 reports the same deflation where it matters.*

**Scale is not the fix.** Beta-lactamase runs 1%, 13%, 11%, 21%, 16% across the ESM-2 ladder from 8M to
3B. 🔴 **"No trend" was how this read before the seeds were checked, and it is too strong.** At 30 seeds
(`src/03x_seed_stability_all_arms.py`) the ladder reads **1.7%, 16.2%, 9.5%, 15.7%, 19.0%**, a rank
correlation against parameter count of **+0.70** on five points. What carries the conclusion is not the
absence of a slope: it is that **3B's entire 95% interval, [14.5, 23.6], sits below alignment's 29.5%**, and
8M at 1.7% [0.2, 3.1] is the only arm clearly apart from the rest. Scale moves the class a little and does
not reach the baseline it has to beat. Nor is scale a general fix for negative-set fragility: the class ordering of that fragility
is uncorrelated between the smallest and largest ESM-2 (Spearman −0.13 for 8M against 650M, −0.17 for 8M
against 3B). Scaling redistributes which classes carry the fragility rather than removing it.

**Pooling is not the fix.** Max pooling drives beta-lactamase to 0% and CLS to 13%, against mean at 21%.
🔴 **The mean-beats-CLS part of that does not survive 30 seeds.** The three come out at **mean 15.7%
[11.2, 20.2], CLS 16.4% [10.6, 22.2], max 1.9% [0.6, 3.2]**, so mean and CLS are indistinguishable and the
published ordering between them was seed noise. Max stays far below both, and no pooling choice comes near
alignment, which is what the heading claims.

**Structure is not the fix.** SaProt with real AlphaFold structures for 231 of 234 panel proteins reaches
10%, below plain ESM-2.

**And not the lineage either, which took a second correction to establish.** Earlier write-ups said
beta-lactamase resists every configuration tested and that plain alignment beats every embedding method on
it. **Both were wrong:** ESM-C 600M recovers **51%**, above alignment's 30% and more than double ESM-2
650M. 🟢 **This is the one arm-level claim in §9 that gets stronger under seed checking.** At 30 seeds it is
**48.3%, 95% CI [43.9, 52.7]**, and it is still the **only** one of the fourteen arms whose interval lies
entirely above alignment's 29.5%, with no other arm's interval even reaching it.

That error predated the panel expansion — ESM-C 600M already scored 48.6% on the 66-protein panel —
and survived because the class was summarized from the ESM-2 arms without checking the ESM-C row.

The obvious reading of that exception was that ESM-C's pretraining corpus explains it. ESM-C saw UniRef
83M clusters plus MGnify 372M plus JGI 2B, with metagenomic data at 37.5% of the final training mix, and
beta-lactamases are among the most diverse families in environmental metagenomes. That reading predicts
the effect should strengthen with capacity on the same corpus. **It was tested and it is wrong.**

| ESM-C, identical corpus, identical bf16, identical pipeline | β-lactamase@95, 5 seeds | @99 | **@95 at 30 seeds, 95% CI** |
|---|---|---|---|
| 300M | 15.7% | 4.3% | **16.4% [10.7, 22.2]** |
| **600M** | **51.4%** | **21.4%** | **48.3% [43.9, 52.7]** |
| **6B** | **4.3%** | **0.0%** | **12.4% [7.4, 17.4]** |

🔴 **Two thirds of the original sentence here was seed noise, and the third that matters is now on firmer
ground.** It said twenty times the parameters recovers less than a twelfth of what 600M does, and less than
300M does. At 30 seeds the ratio is **3.9x, about a quarter rather than a twelfth**, and 6B's interval
**overlaps 300M's**, so 6B being worse than 300M is unsupported. What does survive is the part the argument
needs: **600M at [43.9, 52.7] and 6B at [7.4, 17.4] do not overlap**, across a twentyfold capacity range on
the same corpus. So neither corpus nor capacity accounts for it, and the exception narrows rather than
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
sequences. The per-protein cluster lookups are cached in `results/v2/uniref50_cluster_sizes.json`, which
is the input to `results/v2/pretraining_exposure.json` rather than a result of its own. Pooling all 72 members gives Spearman **rho −0.399, p 0.0005**, in the direction opposite to
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

### 9.6 🔴 A sixth candidate, inside the representation this time, also refused

`src/15b_interplm_sae.py`, `src/15c_interplm_power_check.py`, `src/15d_interplm_feature_overlap.py`.
Five explanations for beta-lactamase's 21% have now been refused: the classifier head (§9.1), corpus and
capacity (§9.3), the toxin/AMR category (§9.4), layer depth (§9.5), and training-set composition (§5.1).
Every one of those looked at the probe or the panel. This one looks at the **features**, using InterPLM's
pre-trained sparse autoencoder for ESM-2 650M — which `research/05` had claimed all along was Pillar 2's
input and which `src/15_sae_fhs.py` never actually loaded (see `docs/DATA_CORRECTIONS.md`, 2026-09-18).

**The hypothesis.** If the features that separate beta-lactamase from benign proteins are not *shared*
with the other hazard classes, a probe trained on those classes has no feature path to reach it, and the
recovery failure is explained.

**Power check first** (`15c`), because a feature-level story is worthless if the feature space cannot
separate hazard at all. At layer 18 the SAE's 10,240 features give **AUROC 0.910 ± 0.044** against the raw
embedding's 0.973, so the premise holds.

🔴 **And the hypothesis fails on a size control.** Unconditioned, it looks conclusive: beta-lactamase has
279 features significant against benign (Mann-Whitney, Bonferroni over 10,240), **264 of them — 95% —
appearing in no other class**, and no feature shared by all seven other classes. But beta-lactamase is the
largest class at n=14, and power scales with n. Subsampling every class to n=6 over 20 draws:

| class | sig. features | unique % | recovery@95 |
|---|---|---|---|
| **beta_lactamase** | **31** | **85%** | **21%** |
| **clostridial_neurotoxin** | **472** | **78%** | **100%** |
| t3ss_effector_apparatus | 12 | 50% | 80% |
| adp_ribosyl_ab_toxin | 131 | 48% | 100% |
| superantigen_enterotoxin | 59 | 41% | 100% |
| rip_rrna_glycosidase | 64 | 25% | 100% |
| pore_forming_cytolysin | 50 | 22% | 69% |

**Unique fraction against recovery: Spearman rho −0.217, p 0.641, n=7.** Beta-lactamase's significant
feature count collapses from 279 to **31** once n is matched, a ninefold drop, so most of the original
signal was power rather than biology. And clostridial neurotoxin settles it: **78% unique, nearly
beta-lactamase's 85%, and recovered at 100%.** Feature uniqueness does not predict recovery failure.

🔑 **What survives is the opposite of what was being looked for, and it is more useful.** Thirty-one
features separate beta-lactamase from benign proteins at n=6 under Bonferroni correction. **The
information is in the representation.** The probe's failure to recover beta-lactamase is therefore not the
representation lacking the class — it is the probe failing to reach information that is demonstrably
there. Six candidates in, that was the sharpest statement available about what the anomaly is not, and §9.7
sharpens it once more by testing whether a probe can find those features without being shown the class.

⚠️ Two caveats on the method. The SAE reconstructs only 87% of layer-18 activation (relative error 0.133,
rising monotonically with depth: 0.006 at layer 1 to 0.256 at layer 30), so a *negative* feature result at
this layer cannot be fully separated from what the SAE misses — which is why the positive result above is
the load-bearing one. And mean-pooling SAE features over a protein destroys the sparsity that makes them
interpretable: 146 features active per residue becomes 4,351 per protein, 42% of the dictionary, though
only 44 exceed the 99th percentile of activation values. Per-protein means are adequate for a classifier
and poor for naming individual features.


### 9.7 🔴 A seventh candidate: the probe cannot reach it in the feature space either

`src/15e_sae_feature_space_lomo.py`, `src/15f_feature_selection_control.py`,
`src/03v_lomo_seed_stability.py`.
§9.6 left one question open by construction. It proved 31 InterPLM features separate beta-lactamase from
benign proteins at matched size, so the information is present. But it found those features **while looking
at beta-lactamase**. Leave-one-mechanism-out has to find them without it. And §9.1's head comparison, which
showed no head rescues the class, was run on the **raw embedding**, not on the space where the discriminative
features were actually located. So: run LOMO in the SAE feature space.

**A dimension control is required, not optional.** The SAE matrix is 234 × 10,240, which is 43.8 dimensions
per positive against the raw embedding's 5.5. A logistic probe there is governed by regularisation, so "SAE
features do better" would be uninterpretable on its own. Three conditions, one protocol, C swept over
0.01/0.1/1/10, 30 seeds:

| condition | dims | best C | 9-class mean | β-lactamase |
|---|---|---|---|---|
| **raw** (the §3 baseline) | 1280 | 10 | 73.7% | **16.0%** |
| sae_full | 10240 | 10 | 63.0% | 0.5% |
| **sae_matched** (top-variance to 1280) | 1280 | 1 | **78.1%** | **0.0%** |

Feature selection for `sae_matched` is fit on **training rows only inside each fold**, so the held-out class
cannot influence which features survive.

🔴 **The feature space is better overall and worse than useless here.** It beats the raw embedding by 4.4
points on the nine-class mean and takes beta-lactamase to **exactly 0.0% at every value of C**. The mean
conceals a large rearrangement rather than a uniform lift:

| class | raw | sae_matched | Δ |
|---|---|---|---|
| contact_dependent_inhibition | 40.8% | 100.0% | **+59.2** |
| pore_forming_cytolysin | 72.9% | 100.0% | **+27.1** |
| adp_ribosyl_ab_toxin | 99.0% | 100.0% | +1.0 |
| t3ss_effector_apparatus | 80.3% | 78.0% | −2.3 |
| **beta_lactamase** | **16.0%** | **0.0%** | **−16.0** |
| *virulence_associated_non_toxin* (control) | *54.7%* | *25.3%* | *−29.3* |

The labelled control moving down 29 points is the useful part of that table: a space that lifted hazard
uniformly would raise the control too, and this one separates hazard classes from the control more sharply
while losing one hazard class entirely.

**The obvious confound was checked and is not the answer.** Top-variance selection keeps only **40 of the
279** beta-lactamase-discriminative features from §9.6, 14% of them, so 0.0% could be selection throwing away
exactly the features that matter. `15f` re-runs the whole comparison selecting on **discriminative power**
instead, |mean difference| / pooled sd against benign, again fit on training rows only:

| selection rule | 9-class mean | β-lactamase |
|---|---|---|
| top variance | 78.1% | **0.0%** |
| top discriminative | 68.4% | **0.0%** |

Selecting for exactly the property that is missing does not recover a single member. **Seventh candidate
refused.**

🔑 **What seven refusals have established.** Beta-lactamase's discriminative features demonstrably exist:
279 on the full panel, 31 at matched size under Bonferroni correction. They are not reachable from the other
eight classes by a linear probe on the raw embedding (§9.1), at any scale or pooling or architecture in
fourteen arms (§9), at any layer (§9.5), in the SAE feature space, or under feature selection that targets
discriminative power directly. The gap is not the information being absent and not the head being too weak.
It is that **nothing in the other eight classes points at where the information is.** That is a statement
about what generalisation across mechanism classes can and cannot be expected to do, and it is the useful
form of this result for anyone building a hazard screen on held-out mechanisms.

🔴 **One published number needs re-reading, found while validating the above.** `15e`'s fold logic is a
second implementation of 03b, so it was checked against the published table first: at 5 seeds it reproduces
**all nine class numbers exactly**. But 03b fixes 5 seeds, and 30 seeds moves exactly one class:

| class | published (5 seeds) | 30 seeds | sd | seeds at 0% | 30-seed 95% CI |
|---|---|---|---|---|---|
| **beta_lactamase** | **21.4%** | **15.7%** | 12.5 | **7 of 30** | **[11.2, 20.2]** |
| contact_dependent_inhibition | 35.0% | 37.5% | 22.5 | 1 of 30 | [29.4, 45.6] |
| the other seven classes | (unchanged) | within 1.4 pt | ≤13.2 | 0 | contains published |

**The published 21.4% falls outside its own 30-seed confidence interval**, and 7 of 30 negative-holdout
splits recover the class at exactly 0%. The figure is left in place because 5 seeds is what the preregistered
protocol specifies and it reproduces exactly; what changes is how to read it. **Treat 21% as the optimistic
end of a wide distribution whose centre is nearer 16%.** This does not soften anything in §9, since the
true recovery being lower than published makes the anomaly sharper. It is recorded in
`docs/DATA_CORRECTIONS.md` (2026-09-18) and the class-level conclusions elsewhere in this document are
unaffected, since no other class moves.

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

### 10.4 🔑 A second unreachable class, and margin locates both of them

`src/28_second_failure_class.py`, on panel v3 (§2.5). This is the most useful result in the document, and
it arrived by accident: the v3 expansion was built to fix a power problem in §2.4.1 and it produced a
second failure.

**Panel v3's leave-one-mechanism-out:**

| class | n | @95 | @99 | AUROC |
|---|---|---|---|---|
| **phage_peptidoglycan_hydrolase** | **32** | **10%** | **2%** | **0.660** |
| **beta_lactamase** | 14 | 19% | 7% | 0.820 |
| *virulence_associated_non_toxin* (control) | *10* | *34%* | *12%* | *0.820* |
| contact_dependent_inhibition | 4 | 75% | 75% | n<7 |
| t3ss_effector_apparatus | 10 | 80% | 70% | 0.910 |
| **bacteriocin** | 15 | **84%** | 81% | 0.976 |
| **cry_insecticidal** | 22 | **87%** | 41% | 0.975 |
| superantigen_enterotoxin | 7 | 91% | 37% | 0.980 |
| rip_rrna_glycosidase | 7 | 94% | 71% | 0.987 |
| adp-ribosyl / clostridial / pore-forming | 7 / 6 / 7 | 100% | 94 / 100 / 66% | 0.998 / n<7 / 0.987 |

Through §9, beta-lactamase was a **single** anomalous class that had survived seven refused explanations.
It is not single any more. A class the probe had never met recovers **10%**, worse than beta-lactamase, on
32 members, which is the largest class in the panel.

🔴 **The first explanation anyone reaches for is wrong, and it is written here rather than dropped.** Both
failures hydrolyse a molecular substrate instead of attacking a cell: beta-lactamase cleaves a small
molecule, the phage enzymes cleave a cell-wall polymer. **RIP refutes it.** Ribosome-inactivating proteins
hydrolyse the N-glycosidic bond of rRNA, which is as molecular a substrate as either, and they recover at
94%. Target host does not separate the pair either, since the other two classes added in v3 are also
non-animal target and recover at 84% and 87%.

**So instead of inventing a third axis to fit two points out of eleven, the test uses the predictor §10.1
already established** at the member level, before v3 existed and before this failure was known:

| predictor, class-mean | Spearman rho against recovery | perm p | are its two lowest classes the two failures? |
|---|---|---|---|
| **margin** = nearest other-class positive minus nearest negative | **+0.894** | **0.00015** | **yes** |
| nearest other-class positive alone | +0.852 | 0.0002 | no |
| nearest negative alone | +0.739 | 0.0035 | no |
| class size | −0.564 | 0.97 | no |

**Both predictions hold.** Margin ranks the two failures as the two lowest of **twelve** classes, which has
probability 1/66 = 0.015 under a random ordering, and margin tracks recovery at rho +0.894 against a
permutation null whose 95th percentile is +0.493. ⚠️ Twelve, not eleven: the ordering covers the eleven
holdout-eligible classes **plus the labelled virulence control**, which is not a mechanism and is not
eligible for holdout but is ranked with the rest. It matters that it is in there, because it sits
**fourth-lowest** and it is the class that displaces the phage class from the bottom two in three of the
five arms in §10.6.1. An earlier version of this sentence said eleven while quoting a 1/66 chance that
only twelve classes produce. Margin also beats each of its own parts, which matters:
without that column the claim would reduce to "classes near other hazards are recovered", which nn_pos
already says and which does not locate the failures.

🔑 **Both failing classes have a negative margin.** beta-lactamase −0.0082 and phage hydrolase −0.0055,
meaning **their members sit closer to a benign protein than to any other hazard class**. Every recovered
class is positive except the two controls. It is the same quantity §10.1 measured on individual proteins,
now holding one level up.

🔴 **An earlier version of this paragraph called that "a mechanism rather than a correlation". That was
overstated and §10.7 is the test that shows it.** Removing the nearest benign neighbours from training does
lift both failing classes beyond what removing the same number at random does, so the proximity is causal.
It closes **11% and 8%** of the distance to a fully recovered class. Margin predicts the failure far better
than removing the proximity repairs it.

**It replicates on v2**, where there is one failure instead of two: margin rho **+0.940**, permutation p
0.0003, and beta-lactamase is the single lowest-margin class of nine (chance 1/9). Notably nearest-negative
alone is useless on v2, rho +0.052 with p 0.45, and informative on v3 at +0.739. The bacterial-target
classes introduced benign-proximity variation that v2 did not contain.

⚠️ **Two honest limits.** The class-size row is the reason to trust the rest: it runs at rho −0.564 with
p 0.97, so the §9.6 power confound is absent and larger classes recover **less**, not more. But the
relationship is not a threshold rule. The **third** lowest margin, contact-dependent inhibition at −0.0052,
sits between the two failures and recovers at 75%. Margin locates the failures without predicting them from
a cutoff, and eleven classes is eleven observations.

🔑 **What this does to §9.** Seven explanations were refused for beta-lactamase, which read as seven dead
ends for one odd class. With a second class behaving the same way and a measured property that picks both
out, the refusals become informative rather than merely negative: **what §9 was failing to fix was never a
property of beta-lactamase.** It is what happens when a hazard class lies closer to benign proteins than to
any hazard the probe was trained on, and no amount of scale, pooling, architecture, corpus, layer depth,
feature space or feature selection changes that geometry. The remaining §9.7 statement holds and gets
sharper: the discriminative features exist, and nothing in the other classes points at where they are,
because the other classes are further away than the benign set is.

### 10.5 🔑 Margin ranks an unseen mechanism correctly and gets its miss rate badly wrong

`src/29_margin_predicts_new_classes.py`. §10.4 is a within-panel fit, and §10.3 is the standing warning
about exactly that: the member-level version of this claim was preregistered, frozen at a tagged commit,
tested externally and **failed twice**. So the class-level version has to be stated before the answer is
known, about classes that were not used to state it.

⚠️ **The external panels cannot supply that test, which is worth saying because it looks like they should.**
`external_mechanism_classes.json` holds 51 proteins in six classes and `safeprotein_mechanism_classes.json`
66 in seven, and **every one of those classes is already in v2**. They are new members of known mechanisms.
They test whether margin predicts for unseen *proteins*, which is what §10.3 already did and lost.

What can test it is the v3 expansion, because the relationship was measured on v2's nine classes and the
three new ones did not exist in the panel then. The model is fit on the **nine pre-expansion classes** and
asked about the **three new ones**, with margins and recoveries both taken from v3 so that only class
identity is out of sample.

| new mechanism class | margin | predicted | measured | error |
|---|---|---|---|---|
| **phage_peptidoglycan_hydrolase** | **−0.0055** | **44%** | **10%** | **+34** |
| cry_insecticidal | 0.0045 | 88% | 87% | +1 |
| bacteriocin | 0.0061 | 96% | 84% | +12 |

**The ordering is right.** Margin ranks the phage class lowest of the three, which is the whole claim in one
line: a mechanism nobody had tested, flagged as the one to distrust from its geometry alone, chance 1/3. The
relationship holds on the nine fitting classes at rho +0.898 and on panel v2 itself at +0.940.

**Out of sample it beats guessing.** Leave-one-class-out over all eleven classes gives mean absolute error
**13.4 points against a mean-recovery baseline of 28.4**, and the held-out predictions track measured
recovery at Spearman **+0.832, permutation p 0.0006**.

🔴 **And the calibration fails on the one class the boundary exists for.** Predicted 44%, measured 10%, a
**34-point** over-prediction. All three out-of-sample errors are optimistic, mean +15.7 points, and with
n=3 a uniform sign is p=0.125 on a sign test, so that bias is a caution rather than a measured effect. The
largest leave-one-out error anywhere is contact-dependent inhibition, predicted 28% against 75% measured,
which is the non-monotonicity §10.4 already flagged showing up as error.

🔑 **So the honest statement is narrower than "competence boundary" and still useful.** Margin says **which
mechanism class to distrust**, before any probe is trained on it, and it was right about a class nobody had
tested. It does **not** say how badly that class will be missed, and it errs optimistic on new mechanisms in
all three cases available. As an operational instrument that makes it a triage signal for deciding which
mechanism families need their own validation, and not a number to put in a specification.

⚠️ Eleven classes, three of them out of sample. §10.3's failure was a claim at this confidence level that
did not survive external data, and the corresponding external test for this one does not exist yet: it needs
a panel containing a mechanism class that v3 does not have.

### 10.6 🔑 The geometry is representation-general: in all fourteen arms the failing class sits closer to benign

`src/30_margin_across_arms.py`. §10.4 and §10.5 are both computed on one arm, ESM-2 650M mean-pooled, and
both are phrased as claims about **geometry**: a class fails when its members lie closer to a benign protein
than to any hazard the probe trained on. A claim in that language has to survive a change of representation,
or the language is overreach. Panel v2 has all fourteen arms embedded and scored, so this is answerable from
cache.

| arm | dim | rho, margin against recovery | perm p | lowest-margin class |
|---|---|---|---|---|
| esm3_1_4B | 1536 | **+0.957** | 0.0001 | beta_lactamase |
| esm2_3B | 2560 | +0.949 | 0.0003 | beta_lactamase |
| **esm2_650M mean** (canonical) | 1280 | **+0.940** | 0.0004 | beta_lactamase |
| esmc_6B | 2560 | +0.914 | 0.0003 | beta_lactamase |
| esmc_300M | 960 | +0.908 | 0.0010 | beta_lactamase |
| esm2_150M | 640 | +0.865 | 0.0027 | beta_lactamase |
| esmc_600M | 1152 | +0.853 | 0.0029 | beta_lactamase |
| prott5_xl | 1024 | +0.831 | 0.0047 | beta_lactamase |
| **saprot_650M** | 1280 | +0.828 | 0.0042 | *virulence control* |
| esm2_8M | 320 | +0.745 | 0.0127 | beta_lactamase |
| **esm2_650M CLS** | 1280 | +0.695 | 0.0228 | *contact_dependent_inhibition* |
| esm2_650M max | 1280 | +0.661 | 0.0323 | beta_lactamase |
| esm2_35M | 480 | +0.588 | *0.0513* | beta_lactamase |

🔑 **The strongest line in the table is the one that is not in it: beta-lactamase's margin is negative in
14 of 14 arms.** Across five model families, a twentyfold parameter range, a different tokenizer and
training objective in ProtT5, and a structure-aware representation in SaProt, the class the probe cannot
recover always sits closer to a benign protein than to any other hazard class. That is what makes §10.4's
geometric phrasing earned rather than decorative.

**12 of 14 arms rank beta-lactamase lowest**, against a per-arm chance of 1/9, and **13 of 14 have a
significant positive rank correlation**. The fourteenth, ESM-2 35M, is at rho +0.588 with **p 0.0513**, a
hair over the line and reported as it came out rather than rounded into the majority.

⚠️ **Two arms put a different class at the bottom, and they are the informative ones.** CLS pooling ranks
contact-dependent inhibition lowest and SaProt ranks the labelled virulence control lowest. Both still have
significant positive correlations, so the relationship survives in them while the specific ordering at the
bottom does not. Note this differs from §9's pattern: the **member**-level margin effect fails on CLS *and*
max, whereas here max pooling locates the class correctly and only CLS misses. The two effects are not the
same quantity and should not be quoted as one.

⚠️ **Two honest deflations.** The fourteen arms include one duplicate pair, the canonical run and
`esm2_650M_mean`, which agree to zero, so the independent count is thirteen. And this is panel **v2**, which
has one failure class rather than v3's two, because only the canonical arm is embedded for v3. Whether
`phage_peptidoglycan_hydrolase` is also lowest-margin in other representations is untested and needs
thirteen more embedding runs.

#### 10.6.1 🔴 On v3 the same test is stricter, and it comes back PARTIAL

`src/30_margin_across_arms.py --panel v3`. §10.6 asked whether margin locates the **one** failing
class on v2, where the bottom-1 of nine has a per-arm chance of 1/9. v3 has **two** failing classes, so
the test becomes whether the bottom-**two** of twelve are exactly those two, and the chance per arm falls
to **1/66**. That is a much stricter question and the answer is not as clean.

Five arms are embedded for v3. The three larger ones needed the GPU partition
(`slurm/negative_scaling_650M.sh`), because the development machine ran out of memory rather than
patience. Rows are ordered by capacity, not by result.

Recovery is given twice: the published protocol's 5 seeds, which the audit pins, and a 30-seed
recomputation with a 95% interval (`src/41_v3_arm_seed_stability.py`), because every side-by-side
comparison below is between arms and this project has retracted three such comparisons already.

| arm | dim | rho | perm p | β-lact margin | β-lact 5s → 30s [ci] | phage margin | phage 5s → 30s [ci] | bottom-2 |
|---|---|---|---|---|---|---|---|---|
| esm2_3B | 2560 | +0.831 | 0.0006 | −0.0030 | 4.3% → **9.3%** [5.6, 13.0] | −0.0014 | 6.9% → **4.1%** [2.4, 5.7] | **yes** |
| **canonical 650M** | 1280 | **+0.894** | 0.0002 | −0.0082 | 18.6% → **21.2%** [16.5, 25.9] | −0.0055 | 10.0% → **12.2%** [10.0, 14.4] | **yes** |
| esm2_150M | 640 | +0.861 | 0.0003 | −0.0090 | 4.3% → **1.9%** [0.8, 3.1] | −0.0008 | 6.9% → **7.2%** [5.4, 9.0] | no |
| esm2_35M | 480 | +0.796 | 0.0015 | −0.0172 | 1.4% → **6.9%** [3.2, 10.6] | −0.0021 | 26.9% → **27.5%** [24.6, 30.4] | no |
| esm2_8M | 320 | +0.846 | 0.0008 | −0.0278 | 0.0% → **0.0%** [0.0, 0.0] | −0.0035 | 31.2% → **26.0%** [22.8, 29.3] | no |

**What holds in all five.** Every arm has a **negative margin for both failing classes**, every arm has a
significant positive margin-against-recovery correlation (+0.796 to +0.894, every p ≤ 0.0016), and
**beta-lactamase is the single lowest-margin class in all five**. So §10.6's central claim, that the
failing class sits closer to benign than to any hazard class the probe trained on, survives the move to a
panel with two failures and holds across a **375-fold parameter range**, 8M to 3B.

🔴 **What does not hold is the exact bottom-two, and it holds in two arms of five.** In 150M, 35M and 8M
the **labelled virulence control** takes second-lowest and displaces the phage class. The displacer is the
same class in all three misses, which is the informative part: the control is a genuinely borderline set,
so the bottom of the margin ordering is where hazard and the control become hard to tell apart. That is the
same shape as §10.6's CLS and SaProt exceptions. The ordering and the negative-margin property are
representation-general; the identity of the bottom-k is not.

⚠️ **An earlier version of this section read the arm dependence as capacity-monotone and that was wrong.**
It was written from three arms, the canonical one plus 8M and 35M, and described beta-lactamase falling and
the phage class rising as the model shrinks. With 150M and 3B added there is no gradient, and the structure
is a split, three arms with two failing classes against two arms with one.

🟢 **The arms really do differ, and it took 30 seeds to say so.** At 30 seeds **9 of the 10 arm pairs are
disjoint on beta-lactamase and 8 of 10 on the phage class**, and the canonical arm's interval separates from
all four others on both. The direction of the claim survives: the canonical arm is genuinely the best arm
for beta-lactamase, [16.5, 25.9] against 3B's [5.6, 13.0], and the two smallest arms are genuinely the best
for the phage class, 26.0% and 27.5% against 4.1% for 3B, which is a **real inverse-capacity effect for one
class** and holds with disjoint intervals.

🔴 **Three of the five 5-seed figures for beta-lactamase and two for the phage class sit outside their own
30-seed intervals, and one coincidence dissolved.** The table's old reading had 3B and 150M tied at 4.3% on
beta-lactamase and tied again at 6.9% on the phage class. Both ties were 3 of 70 arriving twice: at 30 seeds
they are **9.3% and 1.9%** on beta-lactamase with disjoint intervals, a factor of five apart, and 4.1% and
7.2% on the phage class, also disjoint. The phage class's best arm also changes with the seed count, 8M at
five seeds and 35M at thirty, and those two overlap, so their ranking was noise either way.

🔴 **Every arm-to-arm comparison in the table above is a 5-seed comparison, and on v2 that kind of
comparison does not survive 30 seeds.** `results/v2/seed_stability_all_arms.json` has the same ESM-2 ladder
for beta-lactamase at both seed counts:

| arm | 5 seeds | 30 seeds | 95% interval | seeds at exactly 0% |
|---|---|---|---|---|
| esm2_8M | 1.4% | 1.7% | [0.2, 3.1] | 25 of 30 |
| esm2_35M | 12.9% | 16.2% | [9.3, 23.1] | 9 |
| esm2_150M | 11.4% | 9.5% | [6.1, 13.0] | 9 |
| canonical 650M | **21.4%** | 15.7% | [11.2, 20.2] | 7 |
| esm2_3B | 15.7% | **19.0%** | [14.5, 23.6] | 4 |

At 5 seeds the peak is the canonical arm; at 30 seeds it is 3B. **The location of the peak is a
seed artefact**, and the four larger arms' intervals all overlap, with only 8M clearly apart. The canonical
arm is also the one arm whose published 5-seed figure sits outside its own 30-seed interval, which is the
fifth entry in [`docs/DATA_CORRECTIONS.md`](DATA_CORRECTIONS.md).

⚠️ **v3 has no 30-seed ladder, so the v3 table's between-arm differences are untested in the same way.**
`src/03x_seed_stability_all_arms.py` is v2-only, so `src/41_v3_arm_seed_stability.py` runs the v3
equivalent for both failing classes across the five arms, and its figures are in the table above.

🔑 **The two panels answer the same question differently, and the reason is the negative set.** On v2 the
four larger arms' intervals all overlap and the peak moves with the seed count. On v3 nine of ten pairs
separate. v2 holds out 40% of 154 negatives, so its threshold is a quantile of **61** points; v3 holds out
40% of 296 and gets **118**. Doubling the calibration sample halves that noise source, and differences
between arms that v2 could not resolve become resolvable. This is §10.8's constraint showing up as a
limit on what the study can measure rather than on what a screen can deploy.

🔑 **A report that predates all of this says why the irregularity is not about capacity at all.**
`src/04_scale_sweep_report.py` ran these five arms on the frozen panel and
`results/v2/scale_sweep_report.json` has sat unreferenced since. The 150M arm sits below **both** its
neighbours in **5 of 9** v2 classes: adp-ribosylating (97.1 → 60.0 → 100.0), beta-lactamase,
contact-dependent inhibition, T3SS effectors and the virulence control. A dip that appears in five
unrelated mechanism classes at one arm is a property of that arm rather than of its parameter count. ⚠️
Those figures are 5-seed too, so the dip is a pattern across classes within one arm rather than a
established per-class difference.

⚠️ **The within-class check across arms fails once the middle arms are in.** Over three arms
beta-lactamase's margin and recovery moved together at every pair. Over five they do not: 150M has a more
negative margin than the canonical arm (−0.0090 against −0.0082) and 3B has a much less negative one
(−0.0030) while both recover at 4.3% against the canonical arm's 18.6%. Margin orders **classes within an
arm**, which is what every claim here rests on, and it does not order **arms within a class**. Reporting
the three-arm version, where it happened to do both, would have been the error this document keeps logging.

🔑 **The boundary of §10.5's triage claim, stated from this.** The triage is about ranking mechanism
classes inside one representation, and that is what survives: five of five significant, both failures
negative in five of five. It is not a statement about which mechanism is hardest in absolute terms, because
that depends on the arm: beta-lactamase is worst on three arms and the phage class on two, and the two
swap places between the canonical arm and every other one. An operator can use margin to rank which
mechanism families to distrust **in the representation they are actually deploying**, and cannot carry that
ranking to a different model.

### 10.7 🔴 Benign proximity is a contributing cause and closes a tenth of the gap

`src/31_margin_causal_test.py`. §10.4 to §10.6 are correlational: margin ranks the failures lowest, tracks
recovery at rho +0.894, holds in fourteen representations and orders an unseen mechanism correctly. None of
that shows the geometry does the work, and an earlier draft of §10.4 called it "a mechanism rather than a
correlation" anyway. This is the test of that sentence.

The design is §5.1's, moved to the negative side. §5.1 held pore-forming cytolysin out and removed the 14
beta-lactamases from **training**, gaining +16.4 points against a distribution of 25 random removals. Here
the removal is of the **K=10 training negatives closest to the held-out class**, against random removals of
the same size:

🔴 **Removal happens only inside the training split.** 03b calibrates its threshold on the 40% of negatives
held out of training. Removing the nearest negatives from that calibration set would lower the threshold and
raise recovery for a reason unrelated to the decision boundary. The calibration split is identical across
all three arms on a given seed.

| panel | class | recovery | nearest 10 removed | attributable | in-seed percentile | closes |
|---|---|---|---|---|---|---|
| v3 | **beta_lactamase** | 21.2% | 30.5% | **+8.8** | 79th | **11%** of the gap |
| v3 | **phage_peptidoglycan_hydrolase** | 12.2% | 19.2% | **+7.1** | 90th | **8%** of the gap |
| v3 | rip_rrna_glycosidase (comparison) | 94.8% | 98.6% | +3.6 | 62nd | 68% of its 5-point gap |
| v2 | **beta_lactamase** | 15.7% | 22.9% | **+6.7** | 73rd | **8%** of the gap |
| v2 | t3ss_effector_apparatus (comparison) | 80.0% | 80.3% | +0.3 | 52nd | 2% of its gap |

**Attributable** is the targeted removal minus that seed's own mean over 25 random removals, paired within
seed, and its 95% interval excludes zero for every failing class on both panels. So **benign proximity is
causal.** Removing exactly the benign proteins a class sits nearest to helps it more than removing the same
number of negatives at random.

🔴 **And it closes 8 to 11% of the distance to a recovered class.** Beta-lactamase goes from 21% to 30%
while the recovered classes sit at 94 to 100%. The targeted removal also lands at the 79th and 90th
percentile of the random draws rather than beyond the 95th, and it beats a given seed's own 95th percentile
in only 12 and 17 of 30 seeds. The effect is real, consistent, replicated across two panels, and **small**.

🔑 **Prediction and repair come apart, which is the useful part.** Margin predicts the failure strongly:
rho +0.894 across classes, correct out-of-sample ordering on three unseen mechanisms, negative in 14 of 14
representations. Removing the proximity repairs a tenth of it. So benign proximity is where the failure
**shows up** rather than all of what the failure **is**, and §10.4's earlier "mechanism rather than
correlation" has been corrected accordingly. An operator can use margin to decide which mechanism families
to distrust. An engineer cannot fix those families by curating the negative set.

🟢 **The open question in this section is now answered: see §10.7.1.** The effect keeps growing with how
many benign neighbours are removed rather than saturating, so the failure comes from a dense region rather
than a few specific proteins.

⚠️ **Two analysis changes were made after seeing a result, and both improved it. Recording that is not
optional.** The first run pooled 30 seeds × 25 draws into one null and compared the 30-value targeted mean
against its 95th percentile, which returned REFUTED. That pooled null carries fold-to-fold variance the
targeted mean has averaged out, so it is simply the wrong comparison, and §5.1 in this same document already
used per-seed pairing. The second run defined the failing classes as the two lowest-**margin** ones, which
selects the test set with the predictor under test; on v2 that admitted contact-dependent inhibition, which
has a negative margin and recovers at 37.5%, and its −8.5-point result was briefly read as evidence against
the mechanism. Failures are now defined by **recovery below 25%**, which is the property the mechanism is
meant to explain. Both fixes are defensible without reference to their outcome, and both outcomes moved in
the author's favour, so the sequence is stated here rather than only the final numbers.

#### 10.7.1 🔑 The dose-response answers §10.7's open question: the region is dense, so curation cannot fix it

`src/33_margin_dose_response.py`. §10.7 fixed K at 10 and left the shape of the curve as its standing
caveat, because the two readings differ operationally: an effect that saturates means a small, nameable set
of benign proteins carries the failure and a curator could handle them, while an effect that keeps growing
means the class sits inside a dense benign region and no curation of the negative set reaches it.

| removed | phage_peptidoglycan_hydrolase | beta_lactamase | rip_rrna_glycosidase (recovered, comparison) |
|---|---|---|---|
| K=5 | +5.2 [+3.7, +6.6] | +6.0 [+2.8, +9.2] | −0.4 [−3.4, +2.5] |
| K=10 | +7.1 [+5.4, +8.7] | +8.8 [+5.0, +12.6] | +3.6 [+1.4, +5.7] |
| K=20 | +14.3 [+11.8, +16.9] | +13.1 [+8.3, +18.0] | **+5.1** [+3.3, +6.9] |
| K=40 | +14.9 [+12.2, +17.6] | +14.0 [+8.3, +19.7] | +5.0 [+3.3, +6.6] |
| **K=80** | **+16.6** [+13.6, +19.6] | **+20.7** [+15.6, +25.8] | **+2.8** [−0.7, +6.3] |

🔑 **The comparison class is what makes the contrast readable.** Both failing classes climb monotonically to
the largest dose. The recovered class **peaks at K=20 and falls back to +2.8 by K=80**, with an interval that
then includes zero, which is the boundary degrading once too many negatives are gone. So the failures are not
simply benefiting from a looser boundary: whatever they gain keeps coming from the specific benign proteins
nearest them, and there are many of those.

**And the dose that gets furthest is not a dose anyone can apply.** K=80 removes 80 of 178 training
negatives, keeping **55%**. At that dose beta-lactamase reaches **41.9%** and the phage class **27.8%**,
closing **26%** and **19%** of the distance to a fully recovered class. Removing nearly half the benign
training set buys a quarter of the gap, and removing benign controls is precisely what a screen cannot do,
since they are what its false-positive rate is measured against.

🔑 **So §10.7's answer sharpens rather than changes.** Benign proximity is causal, it scales with how much of
the neighbourhood is removed, and it is still not most of the failure. The operational reading is the useful
one: **margin identifies mechanism families a screen will miss, and curating the negative set is not the
repair.** §10.8's finding that the panel is 850 times too small to calibrate a deployment threshold and this
one point the same way: the negative set is the binding constraint on this kind of screen, and it binds in
both directions at once.

### 10.8 🔑 The triage survives a tightening budget, and the panel cannot validate a deployable one

`src/32_deployment_operating_points.py`, panel v3. §10.4 to §10.7 measured margin at **95% specificity**,
which is a laboratory setting. Two questions follow, and the second one the panel answers by refusing to.

**Does the ordering hold as the false-positive budget tightens?** If margin only orders classes at a
threshold nobody would deploy at, the triage is not an operational instrument.

| specificity | rho, margin against catch rate | perm p |
|---|---|---|
| 0.90 | +0.885 | 0.0001 |
| 0.95 | +0.880 | 0.0003 |
| 0.98 | +0.874 | 0.0002 |
| **0.99** (strictest estimable) | **+0.734** | **0.0044** |

🔑 **It holds.** The ordering weakens at the tightest budget the panel can calibrate and stays significant.
So the triage is not an artifact of a lax threshold.

**Per class, as the budget tightens, the spread widens rather than shifting:**

| class | margin | @90 | @95 | @98 | @99 |
|---|---|---|---|---|---|
| **phage_peptidoglycan_hydrolase** | −0.0055 | 27% | 12% | 3% | **1%** |
| **beta_lactamase** | −0.0082 | 40% | 21% | 10% | **4%** |
| *virulence control* | *−0.0027* | *51%* | *31%* | *17%* | *13%* |
| contact_dependent_inhibition | −0.0052 | 78% | 72% | 72% | **69%** |
| cry_insecticidal | 0.0045 | 95% | 79% | 58% | 40% |
| bacteriocin | 0.0061 | 92% | 86% | 84% | 82% |
| clostridial_neurotoxin | 0.0079 | 100% | 100% | 100% | **100%** |

⚠️ **Contact-dependent inhibition is the exception at every operating point**, and it is the same class
§10.7 found does not respond to removing its nearest negatives. It has a negative margin and holds 69% at
99% specificity, flatter than classes with much better margins. Margin's ordering is a rank statistic over
twelve classes, not a per-class guarantee, and CDI is where that distinction bites.

🔴 **What the screen would actually put in a review queue, per 10,000 sequences:**

| specificity | TPR | FPR | alerts at 1-in-100 | at 1-in-1,000 | at 1-in-10,000 |
|---|---|---|---|---|---|
| 0.95 | 73% | 5.1% | 576 (13% real) | 515 (1% real) | 509 (0% real) |
| 0.98 | 63% | 2.5% | 315 (20% real) | 260 (2% real) | 255 (0% real) |
| **0.99** | **55%** | **1.7%** | **223 (25% real)** | **175 (3% real)** | **170 (0% real)** |

At the strictest calibrated point, screening ten thousand sequences against a one-in-a-thousand hazard rate
produces **175 alerts of which about 5 are real, while about 4 hazards go through**. §2.3 reported the
precision collapse; this is the same fact as a staffing number, and it is the form in which it decides
whether a screen is deployable.

🔴 **And the panel cannot calibrate a deployment budget at all.** The threshold is a quantile of the
held-out negatives, and 40% of 296 is 118, so the finest resolution available is one negative in 118, an
FPR of 0.85%. Every specificity above **0.9915** is extrapolation, which is why `03r`'s `spec_0.999` row
returns an FPR of 0.0065 rather than 0.001: it silently hit the ceiling. To calibrate a threshold with ten
negatives above it:

| target specificity | held-out negatives needed | panel size needed |
|---|---|---|
| 0.99 | 1,000 | 2,500 |
| 0.999 | 10,000 | 25,000 |
| **0.9999** | **100,001** | **250,003** |

This panel has **296**. A screen operating at one false positive in ten thousand needs a negative set
roughly **850 times** larger than this one to have its threshold calibrated rather than extrapolated. That
is a statement about what validating a biosecurity screen costs, and it is not a problem more modelling
solves.

⚠️ Volume figures are arithmetic on the measured TPR and FPR and assume the queue is drawn like the panel's
negatives, which no real order queue is. They are a scale check rather than a forecast.

⚠️ **This is the same conclusion CRC-Screen reached, measured from the other side.** That paper's abstract
already states that "the binding constraint on certifiable DNA-synthesis screening is calibration data, not
algorithms" ([arXiv 2605.00074](https://arxiv.org/abs/2605.00074)). §10.8 is an independent measurement of
the same limit on the **negative** side of a mechanism panel, quantile resolution for a specificity
threshold, rather than the miss-rate side of a taxonomic one, so it must be framed as a corroboration rather
than a first. §2.6 turns the same arithmetic into a per-split finite-sample bound.

### 10.9 🔑 A benign set 28 times larger appears to repair both unreachable classes

`src/34_scale_negative_set.py`, `src/35`, `src/36`, `src/37`, `src/38`, `src/39`, `src/40`, panel v3,
canonical 650M arm unless stated.

**Two reasons to build a benign set an order of magnitude larger, and one reason not to touch the
positive side.** §10.8 showed the threshold is a quantile of held-out negatives, so at 118 of them the
finest resolution available is one in 118 and every specificity above 0.9915 is extrapolation.
Calibrating a one-in-ten-thousand budget with ten negatives above the threshold needs roughly 250,000.
Reviewed Swiss-Prot supplies **503,488** eligible entries in the panel's length window once the six
hazard keywords are excluded, checked live on 2026-09-20, so on this one constraint the supply is about
twice the requirement. This is the only binding limit in the project that more data actually fixes.

The second reason is scientific. §10.7.1 read the dose-response as density, and density makes a
falsifiable prediction with a direction: if removing the nearest benign proteins keeps helping all the
way to K=80, then **adding** benign proteins should make these classes worse and leave the recovered
ones alone.

⚠️ **The positive side is deliberately left alone, and this is where "just use all of UniProt" stops.**
There are **6,720** reviewed entries in the length window carrying the Toxin or Virulence keyword, the
same figure live today as when `34` recorded it. Using
them as positives would replace 149 curated labels, each with a written reason, by whatever a curator
flagged. §2's own provenance control is the argument, and it was recomputed on v3 for this section
rather than carried over: a probe that discards the hazard label entirely and predicts lab-strain
origin reaches **AUROC 0.794 ± 0.062** on v3, against 0.818 ± 0.012 on v2, and the organism label
agrees with the hazard label on **43.6%** of v3 against 53.4% of v2. The v3 control is weaker and five
times noisier, and it still sits above chance by more than two standard deviations. At keyword scale
the hazard label *is* the annotation, so a classifier trained that way learns the annotation. Scaling
the negative side has no such problem, because a benign set is defined by the absence of the keywords
rather than their presence.

⚠️ Two published surfaces had this pair muddled and are corrected with it. `34`'s docstring quoted v2's
0.818 next to v3's 43.6%, two figures from two panels, and the Hugging Face card gave 0.818 and 53% against
"the panel" without naming which, at a point where the card describes both. Both pairs are now pinned
together in the audit for that reason.

**The harvest failed three times before it worked, and each failure produced a plausible-looking
result.** UniProt echoes the comma-separated `fields` parameter inside the `Link` header, so splitting
that header on commas found a truncated URL and the pagination loop ended quietly after one page: a
harvest that looks complete and is not. UniProt also returns results clustered by organism, so
sequential paging under a per-organism cap is a pathological sampler, and a first run drew 500 records
that were all *Homo sapiens*, rejected 493 of them, and kept 6. And a single
`TimeoutError: The read operation timed out` partway through the first kingdom discarded everything
collected so far, because there were no retries. The fixes are a `rel="next"` regex that asserts the
reported total was reached, superkingdom stratification with quotas proportional to the eligible corpus,
and exponential-backoff retries. All three are commented at the site in `34`, since a pagination loop
that ends quietly is worse than one that crashes.

**8,259 proteins, and the pool is not what its raw count says.** `36` applies §2.1's effective-n
discipline to the negative side, which the project had only ever applied to positives:

| count | effective n | held out at 40% | specificity ceiling |
|---|---|---|---|
| raw records | 8,259 | 3,303 | 0.99970 |
| by homology, 600-protein sample | **5,203** [4,884, 5,522] | 2,081 | **0.99952** |
| by distinct protein name, complete | **3,550** | 1,420 | 0.99930 |
| the panel, for comparison | 296 | 118 | 0.9915 |

🔴 **The name figure is the lower one and it is the complete one.** 8,259 records carry 3,550 distinct
names, a redundancy factor of **2.33**, and the most repeated name appears **389 times**. The homology
figure keeps 63% of a 600-protein sample [0.591, 0.669] and extrapolates; the name figure misses
homologs that curators named differently but counts every record. So the pool is worth an order of
magnitude more than the panel and roughly two orders less than a deployable budget needs, and the
figure to quote is 0.9995 rather than the 0.9997 the raw count implies.

**The pool is 87% bacterial, and it is 8,259 rather than the 12,000 requested, for the same reason.**
Realised composition is 7,226 Bacteria, 424 Archaea, 339 Viruses and **270 Eukaryota against a quota
of 4,011**. The harvest fetched 39,000 records and kept 8,259; the per-organism cap of 30 alone
rejected **28,728** of them. Swiss-Prot's eukaryotic half is dominated by a handful of model
organisms, so a cap that keeps any one organism from dominating removes most of that half by
construction. The result resembles the panel's own producer mix, **71 of v2's 80 positives are
bacterial** with six plant RIP sources and three viral entries, more closely than it resembles
Swiss-Prot, which is the right material for this question and is not a neutral background sample. The query is `reviewed:true AND length:[100 TO 1400]` excluding Virulence,
Toxin, Cytolysis, Hemolysis, Bacteriocin and Bacteriolytic enzyme.

⚠️ **The pool was never screened against the panel by sequence, and that matches the panel's own policy
rather than breaking it.** §2's **≤ 0.30** normalized Smith-Waterman rule governs **positives against
positives**, and §2.1 counts effective n at the same threshold. `27` says plainly that negatives are *not*
screened against the positives, because mechanism-matched benign proteins are **wanted** as hard negatives
and `data/sequences/benign_homologs.fasta` is the block that exists for exactly that. `34` follows the same
policy for the pool: it dedups on accession and sequence hash, which catches exact duplicates only, then
filters by UniProt keyword and by protein name.

🔑 **So what makes a pool number an outlier is what the panel's own negatives reach under that same
policy, and that is measurable.** `src/42` runs it: the panel's **296 negatives against its 149
positives**, 44,104 alignments. **Zero** reach 0.30. The highest is **0.282**, a sporulation-specific
N-acetylmuramoyl-L-alanine amidase against a phage endolysin, and the second is 0.170. A negative set
assembled with no homology screen at all lands entirely below the positives' own admission threshold.

**The filter that stood in for the screen does leak, though.**

The name filter is demonstrably porous, and it is porous asymmetrically between the two classes this
section is about. `CLASS_BLOCK` covers the phage class's canonical names, "endolysin", "lysozyme",
"muramidase", "amidase" and "holin". It does not cover "peptidoglycan hydrolase", "autolysin" or
"peptidoglycan", so the pool contains **eight genuine peptidoglycan hydrolases admitted as negatives**:
seven *Staphylococcus* "Bifunctional autolysin" entries and "Peptidoglycan hydrolase PcsB". The
autolysins are bifunctional amidase/glucosaminidases, so the exact domain `CLASS_BLOCK` names is present
under a protein name that does not contain the word. Beta-lactamase is covered, with "lactamase",
"beta-lactam", "penicillinase", "cephalosporinase" and "carbapenemase" all blocked and **zero** matching
entries in the pool.

🟢 And the panel does the same thing deliberately, which is how those eight should be read. Its own
negative set's **single closest protein to any positive** is `Q06320`, a **sporulation-specific
N-acetylmuramoyl-L-alanine amidase**, at 0.282 against a phage endolysin. An amidase, which is the exact
enzyme class most phage endolysins belong to, admitted on purpose as a mechanism-matched hard negative, and
it is the top of the list rather than a stray. `P36548` AmiA sits at 0.159 and `P0A3V1` spore cortex-lytic
enzyme at 0.080, both the same story. A functional analogue of a positive class sitting in the negative set
is this panel's design, not its failure mode.

The list is also a check that the aligner is finding real biology rather than noise: second and fourth are
ATP synthase subunits β and α at 0.170 and 0.153 against the T3SS ATPase, which is an F1-ATPase homolog.

🟢 **Checked by sequence rather than by name. Every mechanism class is clean.** `src/42` runs the census
the harvest skipped: all 8,259 pool proteins against all 149 positives, **1,230,591** local alignments
under the same BLOSUM62 screen `02d` and `27` use. No sampling, because this direction is 3% of the
all-pairs count that forced `36` to sample.

| class | max similarity to a pool protein | above 0.30 |
|---|---|---|
| t3ss_effector_apparatus | 0.174 | 0 |
| beta_lactamase | 0.115 | 0 |
| phage_peptidoglycan_hydrolase | 0.105 | 0 |
| the other twelve | ≤ 0.093 | 0 |
| **virulence_associated_non_toxin** (labelled control) | **0.871** | **1** |

So the eight cell-wall entries are functional analogues at roughly a third of the admission threshold
rather than sequence homologs, and the asymmetry §10.9.1 reports is not label contamination.

🔴 **The census found one violation, and it is not in the class the check was built for.** `Q8X739`
**PHOQ_ECO57**, *Escherichia coli* O157:H7 sensor protein PhoQ, sits in the pool as a negative at
**0.871** against `D0ZV89` **PHOQ_SALT1**, the *Salmonella* PhoQ that is a **positive** in the labelled
virulence control. One benign-labelled ortholog of a hazard-labelled panel member, 0.87 identical, which
no keyword or name filter could have caught because both are called "Sensor protein PhoQ" and neither
carries a hazard keyword. This is the difference between a targeted check and a census: the hypothesis was
about the failing classes, both of which are clean, and the defect was somewhere else.

**0.871 against the panel's own 0.282 is what makes it a defect rather than a policy.** The pool's single
worst case is **3.1 times** anything the panel's negative set contains, built under the same absence of a
screen. So this is an outlier inside a documented policy rather than a broken rule, and the fix belongs at
the point of use rather than in the policy.

**What it does and does not reach.** §10.9.1 reports beta-lactamase, the phage class and RIP, all three
clean, so its result stands unchanged. §10.6's margin is computed from the panel's own negatives and never
touches the pool. What it does reach is `src/43`, which measures every class's response to the pool,
including the control: that class's pool proximity is extreme **because of this one protein**, and letting
a 0.871 homolog of one of its members enter training as a negative would drive that member to the benign
side at high dose. The result would be a strongly negative response in the highest-proximity class, which
is exactly the correlation a crowding account predicts, manufactured. `43` therefore drops the offending
protein by default and reports the kept-in version beside it, so the size of that effect is measured
rather than argued about.

**The first attempt at the prediction varied two things at once.** `35` swept the size of the benign set
across the pool, and its smallest point was 296 proteins drawn from the pool rather than the panel's own
296. That changes size and composition together, which is the exact confound §6 exists to separate, and
`34`'s docstring had already written the caveat down before `35` walked into it. The substitution is
worth 34 points: at identical n=296 the phage class reads **46.5%** on a pool sample and **12.2%** on the
panel's real negatives. See [`docs/DATA_CORRECTIONS.md`](DATA_CORRECTIONS.md), eighth entry.

**The fixed test keeps the panel's 296 as a floor and adds on top.** `37` adds K pool proteins to the
panel's own negatives, two ways: K drawn at random, and the K **nearest** the held-out class, which is
the actual converse of §10.7's removal. Both arms are in
`results/v3/negative_supplement_curve_esm2_650M.json`, with the 35M arm beside it and `note_on_35` in
each recording the flaw above from the fixed side.

| class | mode | K=0 | 500 | 1,500 | 4,000 | 8,259 |
|---|---|---|---|---|---|---|
| phage_peptidoglycan_hydrolase | random | 12.2% | 29.0% | 36.9% | 42.9% | **49.3%** |
| phage_peptidoglycan_hydrolase | nearest | 12.2% | 17.5% | 21.4% | 46.2% | 47.9% |
| beta_lactamase | random | 21.2% | 10.5% | 25.2% | 42.4% | **41.9%** |
| beta_lactamase | nearest | 21.2% | 15.5% | 20.5% | 22.4% | 42.1% |
| rip_rrna_glycosidase (recovered) | random | 94.8% | 91.0% | 91.4% | 88.6% | 89.5% |
| rip_rrna_glycosidase (recovered) | nearest | 94.8% | **51.4%** | 68.1% | 78.1% | 90.0% |

Two readings are available and they point opposite ways. The density prediction is **refuted in the
direction it cared about**: both failing classes end three to four times higher rather than lower. But
the nearest arm behaves exactly as proximity predicts on the wrong class. Adding the 500 nearest pool
proteins costs the recovered class **43.3 points** while beta-lactamase falls 5.7 and the phage class
rises 5.3, so the proteins nearest a class hurt the class that was working far more than the two that
were not.

⚠️ The K=0 baseline here is a 30-seed recomputation of the published protocol, 21.2% and 12.2%, against
the published 5-seed 18.6% and 10.0%. `37`'s own tolerance check flagged the gap rather than hiding it.
The cause is seed noise on a 14-member and a 16-member class, the same effect as the fifth corrections
entry, and the published 5-seed figures stay the ones the audit pins.

**On §10.8's constraint the pool does exactly what it was built to do, and the cap that protects the
panel is what stops it going further.** A one-in-ten-thousand false-positive budget needs 250,003
panel negatives, 100,001 of them in the calibration split. Against that requirement:

| negative set | n | remaining gap |
|---|---|---|
| the panel | 296 | 845× |
| the pool, raw records | 8,259 | **30×** |
| the pool, by homology | 5,203 | 48× |
| the pool, by distinct protein name | 3,550 | **70×** |

🔑 **The two constraints pull against each other, which is the part that is not obvious.** The harvest
caps each organism at 30 records, and that cap exists because provenance is a live confound: the probe
that ignores the hazard label and predicts lab-strain origin still reaches AUROC 0.794 on v3. Letting a
few model organisms dominate the benign set would make organism the signal. But 1,996 contributing
organisms at 30 each tops out at **59,880** records, so the harvest design that keeps provenance
separable cannot itself reach a deployable calibration set: 250,003 under the same cap needs **8,334**
contributing organisms, 4.2 times as many as this harvest touched. The cap already rejected 28,728 of
39,000 fetched records.

So the supply exists, 503,488 eligible entries against a 250,003 requirement, and the binding
constraint moves from "is there enough reviewed benign protein" to "is there enough taxonomic breadth
to draw it without reintroducing provenance". That is a different problem than the one §10.8 named, and
it is the one a screen would actually have to solve.

#### 10.9.1 🔴 At a fixed false-positive budget it repairs one of them, and §6 says why the other looked repaired

`src/38`, `src/39`, `src/40`, panel v3, canonical 650M arm, 30 seeds throughout.

🔴 **Before §10.9's supplement table can be read as a repair, the false-positive budget has to be checked,
and it moved.** `38` split the gain into a decision-boundary arm and a threshold-estimation arm:

| class | K=0 | boundary_only | threshold_only | both |
|---|---|---|---|---|
| phage_peptidoglycan_hydrolase | 12.2% | 32.5% | 0.9% | 51.7% |
| beta_lactamase | 21.2% | 7.1% | 11.2% | 39.5% |
| rip_rrna_glycosidase (recovered) | 94.8% | 56.7% | 89.0% | 90.0% |

The two single-factor arms together fall **30.4 points** short of `both` for the phage class and **42.3**
short for beta-lactamase, so the effect is an interaction and the share-of-gain statistic says nothing.
`38` preregistered that case. Of its three arms only **`boundary_only` leaves the operating point
alone**: its threshold is the 0.95 quantile of the panel's own 118 held-out negatives, the same
population the published baseline uses. In `both` at K=8259 the calibration set is 118 panel negatives
plus 3,303 pool proteins, 96.5% pool, and the model was trained on those proteins too.

**`39` measures the budget instead of arguing about it**, and four self-tests make the run falsifiable.
`boundary_only`'s false-positive rate on the panel's 118 is **5.1% at every K and every class**, 6 of 118,
because that is its calibration set. `threshold_only`'s model never changes with K, so its rethresholded
recovery is flat at the baseline. Both hold in every cell. Every recovery figure also reproduces `38`'s
artifact to 1e-9, which pins the fit order and the random stream.

🔑 **At a fixed 5.1% false-positive rate on matched negatives, the pool helps exactly one class.**

| class | K=0 | 500 | 1,500 | 4,000 | 8,259 | best change |
|---|---|---|---|---|---|---|
| phage_peptidoglycan_hydrolase | 12.2% | 27.5% | 33.2% | **35.7%** | 32.5% | **+23.5** |
| beta_lactamase | 21.2% | 17.6% | 15.7% | 15.7% | **7.1%** | **−14.0** |
| rip_rrna_glycosidase (recovered) | 94.8% | 78.1% | 78.1% | 69.0% | **56.7%** | **−38.1** |

The phage rise is monotone across four increasing doses before it turns over, so it is not one lucky K
out of five. Beta-lactamase declines monotonically and the recovered class declines further. **The two
classes this project calls unreachable respond to the same intervention in opposite directions**, which
§10.4 to §10.7 had no way to see, because margin locates them together and every test until now treated
them as one phenomenon.

**So beta-lactamase's headline is the budget moving, and it moves a long way.** `both`'s threshold
incurs 11.1% to 14.5% false positives on the panel's matched negatives against a nominal 5.1%, so
`37`'s K>0 points were running at roughly **two to three times** the budget of its K=0 point. Rethreshold
`both`'s own model on the panel's 118 and beta-lactamase reads **13.3%** against its 21.2% baseline: worse,
not better, and its best value over all five doses is the baseline itself. The phage class survives the
same test at **37.3%**.

**A third framing, built to be the deployment one, lands on the same split.** `40` reserves 2,000 pool
proteins that no arm trains on, calibrates every arm on that same background, and compares each arm
against the K=0 model loosened to the **same** hard-negative false-positive rate. That iso-FP control is
what separates a better classifier from a looser threshold, and it needs one correction of its own: 118
hard negatives quantise the rate at 1/118, so two thresholds can flag the same number while sitting
either side of a gap in their scores, and at K=0 the arm and its baseline are the same model, so the
excess there is pure slack. It is +0.5 points for the phage class, +4.3 for beta-lactamase and +1.9 for
the recovered class, neither negligible nor constant, so every excess is reported net of it.

Every figure below is at the top dose, K=6,259, so the columns are comparable across rows.

| class | recovery on the fixed background | the same K=0 model at the same FP | net excess | hard-negative FP |
|---|---|---|---|---|
| phage_peptidoglycan_hydrolase | 1.1% → 52.0% | 28.0% | **+23.4** | 2.2% → 11.8%, **5.3×** |
| beta_lactamase | 12.4% → 56.4% | 49.0% | **+3.1** | 3.2% → 14.5%, **4.5×** |
| rip_rrna_glycosidase (recovered) | 90.0% → 100.0% | 100.0% | **−1.9** | 4.0% → 16.8%, **4.2×** |

🔴 **And the pool's own redundancy was holding up the one piece of evidence that beta-lactamase gained
anything.** 8,259 records carry 3,550 distinct names, so a random 2,000 / 6,259 reservation puts orthologs
of the same protein on both sides by construction, which inflates specificity on the background and
flatters large K. Splitting **name groups** instead of rows removes that, and the two splits disagree on
exactly one class:

| class | net excess, random split | net excess, name-disjoint split |
|---|---|---|
| phage_peptidoglycan_hydrolase | +23.4 | **+20.3** at K=1,500 |
| beta_lactamase | +3.1 | **+0.0**, and −6.4, −8.8, −24.0, −31.7 at the four positive doses |
| rip_rrna_glycosidase (recovered) | −1.9 | +0.0, falling to −18.1 at the top dose |

Beta-lactamase's +3.1 was the contamination. Under the name-disjoint split its best dose is **K=0**, the
baseline itself, and every dose above that is worse than moving the threshold, by up to 31.7 points. The
phage class loses about 3 points to the same correction and keeps the rest. The caveat written into `40`
before it ran said the bias flatters large K and that an R2 verdict would be safe against it while an R1
verdict would not; that is what happened, on the class where it mattered.

⚠️ **One thing this does not explain is why the two classes differ at all.** The obvious candidate, that
the pool contains homologs of one class and not the other, is measured and refuted in §10.9: zero pool
proteins reach the panel's 0.30 admission threshold against either, with maxima of 0.115 and 0.105. The
eight functional analogues that are in there hydrolyse the same bond as the class that **gains**, which is
the opposite of a contamination story. And margin does not predict the sign: it locates both classes
together, at −0.0082 and −0.0055, and says nothing about which of them a larger benign set will help.
§10.9.2 runs the same arm on all twelve classes and finds one candidate that does, suggestively.

🔑 **+20.3, +23.4 and +23.5 from three framings that share no threshold.** `39`'s boundary arm holds the panel's own operating
point; `40`'s two iso-FP controls hold a reserved pool background, contaminated and decontaminated. All
three agree on the phage class to within three points, and all three agree that beta-lactamase gains
nothing a threshold could not have bought: −14.0 at the panel's own budget, +3.1 with a contaminated
background and **+0.0 once the contamination is removed**.

⚠️ **The deployment reading is worse than either number suggests, and the direction of the drift is not
the obvious one.** At K=0 a threshold set on the broad background is *stricter* than nominal on matched
negatives, 2.2% to 4.0% against a nominal 5%, because pool proteins are out of distribution for a model
trained on the panel alone and score high enough to push the quantile up. As the training set grows to
resemble the background it is calibrated on, that reverses: the hard-negative rate rises to **4.2 to 5.3
times** its starting value, 11.8% to 16.8% against the same nominal 5%. So a screen whose specificity is
quoted against a broad benign background is quiet on matched negatives when it is trained narrowly and
loud on them once it is trained broadly, and the quoted figure moves for neither reason. `39`'s boundary
arm is the exception that proves the point: the phage gain is available at **exactly** the panel's budget,
because that arm never let the budget move.

**§10.7.1's closing line needs one word changed, and its experiment stands.** It said margin identifies
mechanism families a screen will miss, and *curating the negative set is not the repair*. §10.7 tested
curation by **removal** only, and the removal result is unchanged. Addition is a different operation and
it is not uniform: for one of the two failing classes, adding 4,000 benign proteins to the training set
buys 23.5 points at a fixed false-positive budget. For the other it costs 14.0 points, and for a class
that already worked it costs 38.1. So the corrected statement is narrower and more useful: **curating the
negative set is not a general repair, it is a per-class intervention with a sign that has to be measured,
and at panel scale its expected effect on a working class is negative.**

**None of this is a new mechanism. It is §6 running backwards at 28 times the scale.** §6 varied the
negative set 2×2 on v2 and found the operating point dominates, losing 21.4 points across 8 of 9 classes
when only the calibration set changed, against 10.5 points when only the boundary changed. That was
measured while making the negative set **harder**. Here the negative set is made **larger and easier**,
the operating point moves the other way, and it produces an apparent three to fourfold repair that is
mostly not there. A reader who takes one thing from §10.9 should take that: the same mechanism that
costs a screen 21 points when the benign set is sharpened will hand it back 20 or 30 points when the
benign set is broadened, and neither number is about the classifier.

#### 10.9.2 🔴 One predictor looked like it explained the sign, and it does not replicate in a second arm

`src/43_what_predicts_the_response.py`, panel v3, canonical 650M arm, boundary arm only so the
false-positive budget stays at the panel's 5.1%, all **twelve** classes, 30 seeds, 20,000-permutation
nulls.

§10.9.1 measured three classes. Run the same arm on all twelve and the spread is wider than those three
suggested:

| class | n | K=0 | at the top dose | change |
|---|---|---|---|---|
| phage_peptidoglycan_hydrolase | 32 | 12.2% | 30.6% | **+18.4** |
| contact_dependent_inhibition | 4 | 72.5% | 85.8% | **+13.3** |
| superantigen_enterotoxin | 7 | 95.2% | 100.0% | +4.8 |
| bacteriocin | 15 | 86.2% | 86.7% | +0.4 |
| clostridial_neurotoxin | 6 | 100.0% | 100.0% | +0.0 |
| virulence_associated_non_toxin | 10 | 31.3% | 31.0% | −0.3 |
| adp_ribosyl_ab_toxin | 7 | 100.0% | 98.6% | −1.4 |
| pore_forming_cytolysin | 7 | 100.0% | 98.1% | −1.9 |
| t3ss_effector_apparatus | 10 | 80.0% | 75.0% | −5.0 |
| cry_insecticidal | 22 | 79.4% | 73.9% | −5.5 |
| beta_lactamase | 14 | 21.2% | 6.9% | **−14.3** |
| rip_rrna_glycosidase | 7 | 94.8% | 57.1% | **−37.6** |

Two classes gain more than ten points, two lose more than ten, and eight move inside ±6. So §10.9.1's
phage class is not the only gainer: contact-dependent inhibition rises 13.3 points from a 72.5% baseline.
⚠️ That class has **four members**, so its figure is the noisiest in the table and should be read as a
direction rather than a magnitude.

**The predictors, each against the response, with the ceiling confound removed as well as reported.**
A class at 100% can only fall and a class at 12% has room, so every correlation is given twice: raw, and
after least-squares removal of the K=0 baseline from both sides.

| predictor | rho | perm p | rho with baseline partialled out | perm p |
|---|---|---|---|---|
| **pool proximity minus negative proximity** | **−0.601** | **0.0398** | **−0.769** | **0.0054** |
| pool proximity alone | −0.266 | 0.402 | −0.406 | 0.196 |
| **margin** (§10.4's predictor) | −0.182 | 0.571 | −0.217 | 0.503 |
| nearest-negative proximity alone | −0.098 | 0.763 | −0.077 | 0.812 |
| class size | −0.096 | 0.767 | +0.151 | 0.633 |
| baseline recovery | −0.120 | 0.713 | — | — |

🔴 **Read the multiplicity before reading the result.** That table is **ten** tests, and none of them is
corrected. The Bonferroni threshold at ten tests is 0.005, and the surviving partial correlation is at
**0.0054**. It misses. Worse for the finding, the version that does pass is the **contaminated** one: with
`Q8X739` PHOQ_ECO57 left in the pool the same figure reads −0.795 at p **0.0042**, and removing the one
protein §10.9 identified pushes it back over the line.

🔴 **And it does not replicate.** §10.6's discipline for a class-level statistic is to compute it inside each
arm and then count arms, which is how margin earns "significant in 5 of 5". Applied here, with the whole
measurement re-run on `esm2_35M`:

| predictor | canonical 650M, rho (p) | partialled (p) | esm2_35M, rho (p) | partialled (p) |
|---|---|---|---|---|
| pool proximity minus negative proximity | **−0.601** (0.040) | **−0.769** (0.0054) | −0.343 (0.276) | −0.385 (0.220) |
| margin, preregistered null | −0.182 (0.571) | −0.217 (0.503) | −0.252 (0.425) | +0.350 (0.263) |

The **sign agrees** in both arms and nothing else does: the magnitude roughly halves and the significance is
gone, so the count is **1 arm of 2**, against margin's 5 of 5 in §10.6.1. Combined with missing its own
multiplicity threshold, the honest position is that **`nn_pool − nn_neg` is not supported**, and §10.9.1's
question stays open.

🟢 What does replicate is the preregistered half. Margin has no relationship with the response in either
arm, at −0.182 and −0.252 with p 0.57 and 0.43. So the claim that margin says which classes a screen misses
and nothing about which of them a larger benign set makes worse holds in both representations, while the
exploratory positive result holds in one.

⚠️ Only two arms could be tested. Pool embeddings exist for `esm2_650M` and `esm2_35M`; the other three v3
arms would need all 8,259 pool proteins embedded, which is GPU work rather than a rerun.

🔑 **What made it worth testing at all is which predictor it was.** Pool proximity **alone** predicts
nothing on either arm. What moved was pool proximity **minus** the proximity the class already had to the
panel's own negatives, which is §10.4's margin with the pool substituted for the hazard side: margin is
`nn_pos − nn_neg`, this is `nn_pool − nn_neg`. The mechanism it would have described is readable, that a
class whose nearest benign neighbour was already a panel negative gains nothing when the pool arrives while
a class for which the pool brings benign proteins closer than anything the panel held loses ground. That is
what a second arm was for, and the second arm does not support it.

**Margin itself is null here, exactly as preregistered**, at −0.182 with p 0.571 on the canonical arm and
−0.252 with p 0.425 on the 35M arm. The two statistics answer different questions on the same panel:
**margin says which mechanism classes a screen will miss, and it says nothing about which of them a larger
benign set will make worse.** §10.9.1 showed those two questions come apart for the two failing classes,
and that separation is the part of this section that survives.

⚠️ Standing limits, all of them the same limit. n = 12 classes, which is the resolution every claim in
§10.4 to §10.6 runs at. `03p`'s entry in [`docs/DATA_CORRECTIONS.md`](DATA_CORRECTIONS.md) is the
precedent for what pooling the 149 members instead would do: a Spearman of −0.399 at p 0.0005 that was
pseudoreplication of nine classes. The route to settling this is more mechanism classes, not more members.

🟢 **One more class has been measured rather than guessed at, and it clears the floor on the panel's own
hazard definition.** `src/44_candidate_class_yield_v3.py` screens a candidate family against v3's 149
positives under §2's 0.30 rule and reports each survivor's hazard keywords, which is the number `26`
deliberately leaves out. **Bacterial phospholipase C, EC 3.1.4.3**, gives 27 reviewed entries in the length
window, **6 independent** at ≤ 0.30, and **all 6 novel** against the panel at a maximum similarity of 0.037:

| accession | protein | organism | hazard keywords |
|---|---|---|---|
| P06200 | Hemolytic phospholipase C, PLC-H | *P. aeruginosa* | Toxin, Virulence, Cytolysis, Hemolysis |
| P0C216 | Phospholipase C, **alpha-toxin** | *C. perfringens* | Toxin, Virulence, Cytolysis, Hemolysis |
| Q2FWP1 | Phospholipase C, **beta-hemolysin** | *S. aureus* | Toxin, Virulence, Cytolysis, Hemolysis |
| P09598 | Phospholipase C, cereolysin A | *B. cereus* | Cytolysis, Hemolysis |
| P9WIB3 | Phospholipase C B | *M. tuberculosis* | Virulence |
| O06350 | Carboxylesterase/phospholipase LipF | *M. tuberculosis* | **none** |

🔴 O06350 is excluded and that is not a judgement call: its own UniProt function line says it hydrolyses
short-chain esters and has **no activity on triacylglycerides**, so its EC 3.1.4.3 annotation does not
describe what it does. That leaves **5 under the panel's operative hazard definition and 3 under the
strictest one**, against an eligibility floor of 4, so the count straddles the floor and the reading decides
it.

**The panel's own harvests already use the loose reading.** `34`'s `BLOCK_KEYWORDS` keeps any protein
carrying Virulence, Toxin, Cytolysis or Hemolysis **out of the benign pool**, so P09598 and P9WIB3 are
already treated as hazard-side. Excluding them from a hazard class while also excluding them from the benign
set would leave them in neither, which is the inconsistency `27` flags for the OspC pair in the thirteenth
corrections entry.

⚠️ Two things this class would and would not do. It takes the count from twelve to **thirteen**, which is
what §10.9.2's resolution needs. It does **not** help §2.4.1's non-animal deficit, since its target host is
animal. It does not disturb the producer balance either, since every producer is bacterial, which is what
separates it from the animal-target candidates `26` measured and set aside. ⚠️ And it is a **proposal, not a
result**: nothing is admitted, no annotation moves, and building it is a panel change rather than a
correction.

**It is also the only one of six candidates that got that far**, and the pattern in the other five is the
more useful finding:

| candidate | reviewed | independent | with a hazard keyword | verdict |
|---|---|---|---|---|
| phospholipase C, EC 3.1.4.3 | 27 | 6 | **5** | buildable on the loose reading |
| hyaluronate lyase, EC 4.2.2.1 | 9 | 6 | 2 | no |
| protein-glutamine deamidase, EC 3.5.1.44 | 200 | 13 | 2 | no |
| bacterial collagenase, EC 3.4.24.3 | 10 | 3 | 1 | no |
| sialidase, EC 3.2.1.18 | 11 | **8** | **0** | no |
| clostridial glucosylating toxin, LCGT family | **0** | 0 | 0 | no, and for a different reason |

🔑 **Homology is not what stops these classes. Hazard annotation is.** Five of the six have between three
and thirteen independent members, comfortably past the n=4 floor on supply, and they fail because the same
catalytic activity is overwhelmingly carried by enzymes nobody calls a toxin. Sialidase is the clean case:
**8 independent members and not one hazard keyword between them**. EC 3.5.1.44 is the vivid one: 200
reviewed entries, 13 independent, and 11 of those 13 are CheB and CheD chemotaxis enzymes that share the
deamidase activity and nothing else. So the panel has twelve classes rather than thirty because
mechanism-defined **hazard** classes are rare, not because nobody has curated them.

⚠️ The glucosylating toxins fail on a third axis entirely, and it is the panel's own: `TcdA` is **2,710**
residues and `TcdB` **2,366**, against a length window of 100 to 1,400. C. difficile's large clostridial
toxins are excluded by §2's length rule, not by supply or homology.

🔴 **One of these verdicts was wrong before the function lines were read, and the tool cannot catch it.**
`44` counts hazard keywords, not mechanisms. Run on a name-based deamidase query it returned BUILDABLE at
four members; reading the six survivors' own UniProt entries cut that to two. `P9WNU9` deamidates Pup, a
*bacterial* protein, not a host target. `Q6TKR8` AvrRxo1's stated reaction is `NAD(+) + ATP = 3'-NADP(+) +
ADP`, an NAD kinase matched on a text mention. `P17452` has **no catalytic activity stated at all** and
entered on its protein name. The same reading is what excludes `O06350` LipF from the phospholipase C set,
since its own function line says it has no activity on triacylglycerides. **The counts in that table are
upper bounds**, and a candidate has to be defined by an EC or a stated reaction and then read entry by
entry.


## 11. What this does not claim

- **Not a better classifier.** DTVF (ProtT5 + LSTM/CNN) reports AUROC 0.92 on the standard 576/576
  virulence benchmark. The numbers here are on a self-built panel and are **not comparable**; presenting
  them as a win would be wrong.
- **Not novel on homology control.** Homology-clustered evaluation is established practice.
- **Not a competence boundary, and the shape of the failure has changed.** §10.3 is the record of the
  **member**-level claim being preregistered, frozen and failing on two external panels. The **class**-level
  version in §10.4 to §10.6 is in better shape and still not a boundary: margin orders mechanism classes by
  recovery, the ordering transfers to three mechanisms it was not fit on (§10.5) and holds in fourteen
  representations (§10.6), but it **mis-states the miss rate by 34 points on the class it exists to flag**
  and errs optimistic on every out-of-sample class. It also has **no external test**, because every class in
  both external panels is already in v2, so they contain new members of known mechanisms rather than new
  mechanisms. Read it as a triage signal for deciding which mechanism families need their own validation.
- **Not deployment-ready, and §10.8 says how far off.** These are research panels, 234 proteins in v2 and
  445 in v3. Common Mechanism and SecureDNA are running in production; this is not in that category. The
  quantitative version: the threshold is a quantile of held-out negatives, so **every specificity above
  0.9915 is extrapolation on this panel**, and calibrating a one-in-ten-thousand false-positive budget would
  need a negative set roughly **850 times** larger. At the strictest budget that can be calibrated, ten
  thousand screened sequences at a one-in-a-thousand hazard rate produce 175 alerts of which about five are
  real. §10.9 closes part of that gap and renames the rest of it: an 8,259-protein benign pool brings the
  shortfall to **30×** by raw count and **70×** by distinct protein name, and what then binds is taxonomic
  breadth rather than supply, because the per-organism cap that keeps provenance separable tops the harvest
  design out at 59,880 records.
- **Not evidence that hazard is what is being detected.** See the provenance control in §2.
- **Not a test of generalisation to functions no model has seen.** The held-out class is removed from the
  probe's training, not from the foundation model's pretraining, and every class here is in the public
  databases these models were trained on. See §9.3.
- **Not a controlled comparison of pretraining corpora.** The arms differ in what they were pretrained on
  as well as in architecture and scale, and this study cannot separate those. See §9.3.

## 12. Reproducing

Two panels live side by side. **v2 is frozen**: every number outside §2.5, §10.4 to §10.8 and the §2.4.1
comparison is computed on it, and it must stay reproducible. **v3** adds three non-animal-target classes.
`02b`, `03b`, `03s` and `03w` take `--panel`, which switches the input FASTA, the annotation files and the
output directory together, so a v3 result can never land on a v2 filename.

```bash
# v2, the frozen panel: all fourteen arms and every section up to §10.3
PROJECT_DIR=$PWD PYTHON_BIN=path/to/python sbatch slurm/expanded_panel_full_sweep.sh

# v3, built from UniProt with the panel's own admission rule. Dry run first: it writes a
# staging file and touches nothing. --from-stage then builds v3 from the records reviewed.
python src/26_panel_growth_yield.py                      # which families can supply a class, vs v2
python src/44_candidate_class_yield_v3.py --name phospholipase_c --query "ec:3.1.4.3" \
    --target animal --producer bacteria                  # supply AND hazard annotation, vs v3
python src/27_expand_nonanimal_classes.py                # dry run, screens and stages
python src/27_expand_nonanimal_classes.py --from-stage    # writes the v3 file set
python src/02b_esm2_embed_v2.py --panel v3               # v3 embeddings
python src/03b_leave_one_mechanism_out.py --panel v3     # §2.5's recovery table

# the margin line of work, §10.4 to §10.8
python src/28_second_failure_class.py --panel v3          # margin locates both failures
python src/29_margin_predicts_new_classes.py              # out-of-sample ordering
python src/30_margin_across_arms.py                       # all 14 arms, on v2
python src/30_margin_across_arms.py --panel v3            # five arms, on v3, §10.6.1
python src/31_margin_causal_test.py --panel v3            # causal, and how small
python src/33_margin_dose_response.py --panel v3          # dose-response
python src/32_deployment_operating_points.py --panel v3   # queue volume and the FP ceiling

# the negative set at scale, §10.9. The pool embeddings are 42 MB and each sweep is
# between 450 and 1,350 logistic fits, so these live on SLURM. A local attempt at the
# 650M pool embedding stalled with 67 MB of free RAM; the diagnosis is in the job script
# because an earlier hang of the same kind was misread as a network stall.
python src/34_scale_negative_set.py                       # harvest 8,259 reviewed Swiss-Prot proteins
python src/36_pool_effective_n.py                         # what the pool is worth after homology
PY=path/to/python; D=$PWD
PROJECT_DIR=$D PYTHON_BIN=$PY sbatch slurm/negative_scaling_650M.sh   # 35, then 37's fix of it
PROJECT_DIR=$D PYTHON_BIN=$PY sbatch slurm/threshold_vs_boundary.sh   # 38, boundary vs threshold
PROJECT_DIR=$D PYTHON_BIN=$PY sbatch slurm/operating_point_audit.sh   # 39, then 40 on both splits
python src/42_pool_homology_against_panel.py              # §10.9's homology census, ~15 min, local
PROJECT_DIR=$D PYTHON_BIN=$PY sbatch slurm/response_predictors.sh     # 43, §10.9.2, drops the homolog
KEEP_HOMOLOGS=1 PROJECT_DIR=$D PYTHON_BIN=$PY sbatch slurm/response_predictors.sh   # its comparison arm

# how the panel itself was built. These were unreferenced by any document until a
# review pass on 2026-09-18 found them, which matters because §10.3's failed
# preregistration and §10.5's argument that the external panels cannot test a NEW
# mechanism both rest on panels built here.
python src/02c_localization_annotate.py                    # localization, fetched blind to the label
python src/02d_secreted_pathogen_negatives.py              # the secreted-from-pathogen negative block
python src/05_external_panel_build.py                      # external validation panel, 51 positives
python src/05b_external_negatives.py                       # its 81 negatives
python src/05c_external_score.py                           # scoring the external panel
python src/06_safeprotein_panel_build.py                   # the SafeProtein set, 66 recovered from the paper

# seed stability, after two published numbers turned out to be 5-seed means
python src/03v_lomo_seed_stability.py
python src/03x_seed_stability_all_arms.py                 # v2, beta-lactamase, 14 arms
python src/41_v3_arm_seed_stability.py                    # v3, both failing classes, 5 arms, §10.6.1

# every headline claim, recomputed from its artifact and matched to the documents
python src/22_claims_audit.py
```

The audit exits 1 if any claim disagrees with its artifact, and CI runs it on every push. It was added
after four separate number-drift defects were found by hand, each because someone happened to look; the
two entries covering this panel were added after two more. All are documented in
[`docs/DATA_CORRECTIONS.md`](DATA_CORRECTIONS.md).

**The recovery figures do not depend on the device.** The v3 leave-one-mechanism-out run was repeated on
CUDA for the canonical 650M arm and on CPU for the 35M and 8M arms, against the published Apple MPS run.
All twelve classes agree **bit for bit** at both the 95% and the 99% operating point, on all three arms
across all three devices. The embeddings themselves do differ: per-seed calibrated thresholds move in the
fifth decimal. Recovery is a count over a dozen-odd class members, so a shift that small has to cross a
member's score to change anything, and here none of them did. The CUDA recomputation is kept **beside** the
published file as `results/v3/lomo_results.cuda650M.json` rather than over it, so the published numbers
stay the ones the audit pins.

The same holds for §10.6's margin, with one figure worth naming. Recomputing `margin_across_arms` on CPU
moves each class's margin in its **eighth significant figure**, for example beta-lactamase's canonical
−0.008177275769 against −0.008177250624, while every reported quantity is identical: the rank correlations,
the permutation p-values, the lowest-margin class and the bottom-two membership on all five arms. §10.6.1
quotes margins to four decimals, four orders of magnitude above that noise.
