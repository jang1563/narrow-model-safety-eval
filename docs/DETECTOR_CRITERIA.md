# What makes a good hazard detector

Written 2026-09-20. Every criterion below is anchored to a number this project
measured, usually to a place where this project's own detector failed. That is
deliberate: a criteria list assembled from good practice in the abstract is
cheap, and the ones that matter are the ones that cost something.

Scope note. "Detector" here means any system that takes a biological sequence
and returns a hazard score or a flag: an embedding probe, an alignment screen,
a classifier over annotations, a synthesis-order filter. The criteria are about
the evaluation of such a system, not about its architecture.

This document is **not yet on the audited surface** (`PUBLIC` in
`src/22_claims_audit.py`). Adding it there is the first follow-up once the
claims audit can be run, because a criteria document that drifts away from the
numbers it cites is exactly the failure mode described in criterion 10.

---

## The short version

A detector is worth deploying when you can state, before deployment, all of:

1. the split: whether a held-out **test** set exists at all, whether the
   threshold estimator delivers its own nominal rate out of sample, and the
   finest false-positive rate the calibration set can resolve,
2. its recovery **per hazard class**, not in aggregate,
3. which classes it is expected to fail on, with the accuracy of that
   prediction reported as an error and not only as a correlation,
4. the realized false-positive rate of every comparison you quote,
5. what its negative set actually contains, in effective rather than raw size,
6. its alert count and miss count at the deployment prevalence,
7. what it does to a known-safe near neighbour of a hazard.

Most published detector numbers, including several early ones in this
repository, fail at least three of these.

---

## 1. The split comes first: a calibration set is not a test set

Three separate failures live in the negative split, and they compound. This is
the criterion this project scores worst on, and it is listed first because the
other sixteen are downstream of it.

**(a) Check that a test set exists.** This panel's negatives divide as:

| partition | count |
|---|---|
| train | 178 |
| calibrate | 118 |
| **test** | **0** |

There is no held-out test set. Every negative is either fitted on or used to
place the threshold, so no false-positive number in this repository was ever
measured on negatives the pipeline had not already seen. That is not a subtle
statistical point. It means the reported specificity is an in-sample quantity,
and the first question to ask any detector is the one that is easiest to skip
because the answer is usually assumed.

**(b) Check that the threshold estimator delivers its own nominal rate out of
sample.** It does not here. Asked for a 5% false-positive budget and evaluated
on negatives held back from calibration, the published `np.quantile` estimator
returns **8.64%** on the canonical arm and **8.79%** on a second arm at a
calibration size of 20, and the nominal rate is unreachable at 4 of the sizes
tested. An estimator that misses 5% by 3.6 points is not a rounding problem when
the deployment budget is 1e-4.

That much was established as a property of the estimator. `src/48` then does the
thing that was missing, which is to carve an actual test partition out of the
negatives (177 train / 59 calibrate / 60 test) and rerun leave-one-mechanism-out
at a threshold set on calibrate and a false-positive rate measured on test, over
30 seeds, with the seed as the unit of inference:

| nominal | estimator | out-of-sample FP | seed-level 95% CI | excludes nominal |
|---|---|---|---|---|
| 5% | `np.quantile` | 7.33% | [5.76, 8.90] | **yes** |
| 5% | conformal | 6.31% | [4.73, 7.90] | no |
| 1% | `np.quantile` | 3.88% | [2.87, 4.89] | **yes** |
| 1% | conformal | **unreachable** | | |

Three things follow, and the third is the one worth carrying.

First, the published estimator's overshoot is now measured rather than inferred,
and at a nominal 1% it realizes **3.88%, close to four times the budget**.

Second, conformal is better but is **not** a drop-in fix. Its interval covers
nominal, so it is consistent with holding its guarantee, which is as much as 30
seeds can say. Anyone reading its 6.31% point estimate against 5% and concluding
it failed has compared a point to a target without an interval, which is the
error this document's criterion 6 exists to prevent.

Third, and this is the real result: at a nominal 1% the conformal threshold
**cannot be computed at all**, because `floor((m+1)*alpha)` is zero when m is 59.
`np.quantile` returns a number anyway, and that number realizes 3.88%.
**Conformal declines to answer where the quantile estimator invents an answer.**
So the binding constraint was never the estimator, it is the number of negatives,
which is (c). An estimator that refuses is strictly more useful than one that
extrapolates silently, but neither creates resolution that the data does not have.

Two riders on that run, both necessary.

The conclusions survive the estimator. Conformal lowers per-class recovery almost
everywhere, by 0.0 to 7.7 points, but the **ordering is preserved**: phage
peptidoglycan hydrolase stays lowest, beta-lactamase second, the labelled
virulence control third. The failure story that criteria 3 and 4 are built on is
therefore not an artefact of the threshold rule, which is worth knowing because
it easily could have been.

And the recovery figures from that run are **not** comparable to the published
table, because carving out a test partition also halved the calibration set from
118 to 59, so two things changed at once. Only the quantile-against-conformal
contrast within the run is clean, since both arms share the same splits, models
and scores. Quoting `src/48`'s recovery numbers against the published ones would
be the exact mistake criterion 2 describes, committed by the document that
describes it.

**(c) Check the resolution ceiling.** A threshold set as a quantile of held-out
negatives cannot resolve a false-positive rate finer than one over the number of
negatives in that quantile set. With 118 calibration proteins, one over 118 is
0.0085, so **every specificity above 0.9915 in this repository is extrapolation
from the tail of 118 points, not a measurement.** A 1e-4 budget, which is the
kind of number a synthesis screen actually needs, would require about 250,003
panel negatives, roughly 850 times the current panel.

The test for (c) is one division. Do it before quoting an operating point.
Corollary: reporting `@99` and `@99.9` side by side from the same calibration set
is reporting one measurement twice.

Taken together: (a) says the number is in-sample, (b) says the in-sample number
is also optimistic by about 3.6 points at small calibration sizes, and (c) says
the tail where a screen would actually operate is not resolvable at all. A
detector can satisfy every other criterion in this document and still be
unusable if it fails these three, which is why the split, and not the model, is
where a dataset audit should start.

## 2. Compare at a fixed realized false-positive rate, never at a nominal one

The most expensive error in this project's history was an apparent three-to-four
fold gain in recovery of the beta-lactamase class after adding negatives from a
large benign pool. At a fixed 5.1% realized false-positive rate the same
intervention gives:

| class | change at iso-FP |
|---|---|
| phage peptidoglycan hydrolase | **+23.5** |
| beta-lactamase | **-14.0** |
| RIP rRNA glycosidase (a healthy class) | **-38.1** |

The gain was the budget moving. Realized false positives had drifted to
between 11.1% and 14.5% while the nominal budget still read 5%. Under a
name-disjoint split beta-lactamase's best dose is no dose at all: +0.0 at K=0,
and -6.4, -8.8, -24.0, -31.7 above it.

So the defensible statement is neither "curating negatives repairs hard
classes" nor "curating negatives does not help". It is that the intervention is
**per class with a sign that has to be measured**, and that the same change
which helps one class by 23 points costs a healthy class 38.

Report the realized rate next to every recovery number. A recovery figure
without its realized false-positive rate is not comparable to anything.

## 3. Aggregate separability tells you nothing about the class you will meet

On this panel the aggregate numbers are strong: AUROC 0.981 on the v1
separability task, 0.974 on the v2 leave-one-mechanism-out baseline. Under the
same probe, at 95% specificity:

| class | n | flagged@95 | flagged@99 | AUROC |
|---|---|---|---|---|
| superantigen enterotoxin | 7 | 100% | 100% | 1.000 |
| RIP rRNA glycosidase | 7 | 100% | 89% | 0.997 |
| beta-lactamase | 14 | 19% | 7% | 0.820 |
| phage peptidoglycan hydrolase | 32 | 10% | 2% | 0.660 |

An aggregate AUROC near 0.98 coexists with 10% recovery on a 32-member class.
A detector's worst class is its deployment behaviour, because an adversary
picks the class.

Note also what does **not** explain the two failures: not target host (the two
other non-animal-target classes added at the same time recover at 84% and
87%), and not "hydrolyses a molecular substrate" (ribosome-inactivating
proteins do exactly that and recover at 94%).

## 4. A detector should predict which class it will fail on, and should not pretend to predict how badly

Margin, defined as nearest other-class positive minus nearest negative, ranks
this probe's failures well. Its two lowest classes out of twelve are exactly
the two failing classes, which is a 1-in-66 coincidence under chance, and it
tracks recovery at Spearman +0.894 with permutation p 0.00015. Both failing
classes have a **negative** margin: their members sit closer to a benign
protein than to any hazard class the probe was trained on.

Out of sample it is much weaker as a number. It predicted 44% for phage
hydrolase and the measured value was 10%, a 34 point over-prediction, and all
three out-of-sample classes erred optimistically. Leave-one-out mean absolute
error is 13.4 points against a 28.4 point baseline, so it beats predicting the
mean, and that is the honest size of the claim.

The usable form: **margin says which class to distrust, not how badly it will
do.** A triage signal, not a specification number. Any predictive signal
offered for a detector should be reported with its out-of-sample error, not
only with its correlation, because a rank correlation of +0.894 is compatible
with a 34 point miss on the class you care about.

## 5. Seed and configuration stability come before the headline, and arm ordering is the stricter test

Five seeds is this panel's preregistered protocol and it reproduces exactly.
It is still not enough to support the claims people want to make from it.

At 30 seeds, beta-lactamase recovery on the v2 canonical arm is 15.7%, sd
12.5, 95% CI [11.2, 20.2], and **7 of 30 splits recover the class at exactly
0%**. The published 5-seed figure of 21.4% sits outside its own 30-seed
interval. On v3, 30 seeds give canonical 21.2% [16.5, 25.9], 3B 9.3%
[5.6, 13.0], 150M 1.9% [0.8, 3.1].

The sharper lesson is about ordering, not intervals. On v2 the arm that looks
best at 5 seeds (canonical) is not the arm that looks best at 30 (3B). **The
location of the peak was a seed artefact.** A confidence interval on one arm
does not license a statement about which arm is best; that needs enough seeds
for the ordering itself to be stable, which is a stricter and more expensive
requirement.

And this criterion is criterion 1 wearing different clothes. On v2 the four
larger arms' intervals all overlap and the peak moves with the seed count; on
v3 nine of ten arm pairs separate. The difference is not the models and not the
seeds, it is that v2 calibrates on 61 held-out negatives and v3 on 118.
**Doubling the calibration sample made differences between arms resolvable that
were previously not resolvable at all.** Seed instability and threshold
resolution are one budget, so a study that cannot afford more negatives should
not expect to rank its own configurations either.

## 6. Fix the multiplicity threshold before looking, and treat a near miss as a miss

The cleanest demonstration in this repository of why the threshold goes first:
a follow-up correlation of -0.769, partialled, at p 0.0054, against a
ten-test Bonferroni threshold of 0.005. It misses. The earlier, contaminated
version of the same run gave -0.795 at p 0.0042, which passes.

If the threshold had been chosen after seeing those two numbers, there is no
version of the reasoning that ends anywhere except "the result holds". Written
down first, the verdict is NOT SUPPORTED, and that is the correct verdict.

## 7. One arm is not a result

The same follow-up does not replicate: -0.601 (p 0.0398) on the canonical arm
and -0.343 (p 0.276) on esm2_35M. One arm of two, plus a Bonferroni miss, is
not a finding.

The contrast is instructive. The preregistered **null** for margin at member
level replicates in both arms. A null that replicates is stronger evidence than
a positive that does not.

Where a claim does survive this test, say so at its real width: ESM-C 600M is
the only one of 14 arms whose 30-seed interval [43.9, 52.7] clears plain
alignment's 29.5% on beta-lactamase, and no other arm's interval reaches it.
That is a narrow, checkable statement, and three other arm-to-arm comparisons
in the same document did not survive the same check and were rewritten.

## 8. Negative-set quality is a measured property, reported as effective size

Raw counts overstate what a benign pool provides. This project's pool:

| property | value |
|---|---|
| raw proteins | 8,259 |
| effective n by homology clustering | 5,203 |
| effective n by distinct name | **3,550** |
| redundancy factor | 2.33 |
| most repeated name | 389 times |
| bacterial fraction | 87% |

Three things follow. First, quote effective n. Second, a per-organism cap is
not free: a cap of 30 rejected 28,728 of 39,000 candidates and puts a ceiling
of 59,880 on the whole design, which is well short of what criterion 1 asks
for. Third, **run a homology census against your own positives before using a
pool as negatives.** This project's census covered 1,230,591 alignments and
found no pool protein reaching 0.30 identity against any mechanism class
(maxima 0.115, 0.105, 0.174), with exactly one violation: Q8X739 at 0.871
identity to a labelled panel positive. One contaminated member out of 8,259 is
a good outcome, and it is only a known outcome because the census was run.

Anything that trains on that pool has to drop that member.

## 9. State the operating point in alerts and misses, not in rates

At 99% specificity this probe gives TPR 55% and FPR 1.7%. Translated to a
screening queue at a hazard prevalence of 1 in 1,000: about **175 alerts, of
which roughly 5 are real, while about 4 hazards pass.** That is the number a
reviewer can act on, and it is the number that makes the rate pair sound very
different.

Rates hide prevalence. An independent screening study reached the same
conclusion from the miss-rate side, so this is corroboration rather than a
first, and it should be cited as such.

## 10. Audit the metric as hard as the model, and make it fail loudly

Two metric defects in this project both moved headline numbers, and neither was
visible from the model's behaviour.

FSI mapped annotated catalytic residues onto structures by residue number. For
three structures the numbers were present but pointed at the **wrong amino
acids**, and the metric computed through it silently. After an amino-acid
identity check was added and the residues were re-curated, the count of
structures with significant FSI elevation fell from 5 to 3.

FSPE masked catalytic residues on sequences from the panel FASTA, which holds
the **precursor**, while three annotations were numbered on the **mature
chain**. The affected offsets are +35, +47 and +32, each equal to the UniProt
signal or signal-plus-propeptide length.

The transferable rule: a metric must assert its own preconditions and fail
loudly when they do not hold. A silently wrong score is worse than a crash,
because a crash gets fixed the same day.

A related, smaller version of the same rule inside the audit tooling itself: a
non-string pin once raised a bare `TypeError` and killed the whole claims audit
rather than reporting one failed claim, and a grep of the output for the
success line hid the crash. Check exit codes, not printed text.

## 11. A correction needs a criterion that does not mention the metric

The three FSPE offsets each move a ratio in the direction this project's own
claim wants, so the offsets cannot be justified by the outcome. They are pinned
by two criteria that never mention FSPE: each offset is the unique integer
that makes every annotated residue identity land correctly, and each
independently equals the UniProt chain boundary.

Uniqueness is swept rather than asserted. Over every offset the sequence admits,
the set achieving a **full** identity match is exactly `[35]` for Ricin at 5 of 5
checkable residues, `[47]` for barnase at 5 of 5, and `[32]` for diphtheria
toxin at 3 of 3. A singleton set is the whole point: the value was not available
to be chosen. The claims audit pins all three, and the assertion was
negative-tested by declaring one offset non-unique, which fails the gate.

The evidence that this discipline was real is that **one of the three
corrections pushed the metric the wrong way.** Ricin went from 1.226 to 1.230
under its own fix and stayed on the wrong side of 1.0. The entire headline gain
came from one protein, barnase, at 1.283 to 0.051.

A set of corrections that only ever helps is a set of corrections to distrust.

## 12. Preregistration needs a ceiling as well as a floor

The preregistered external test of the margin claim specified a gap of at least
25 points. Two external panels returned +13 and +0, and the claim was
downgraded as the preregistration required. The prereg's own defect is recorded
with it: it set a floor but no ceiling, so the second panel's uniform 100%
recovery is logged as NOT SUPPORTED when **uninformative** is the accurate
reading. A test that cannot distinguish "the signal is absent" from "the test
had no resolution" is a test with one criterion missing.

## 13. Recovery is a joint property, so it does not transfer

Expanding the panel from 66 to 80 positives moved pore-forming cytolysin by
+11.4 points **without adding a single member to that class**, because adding
positives shifts the calibrated threshold and members already near it cross.
Classes whose members were saturated or floored moved by exactly zero.

A per-class recovery figure is a property of the class, the rest of the
positive set, and the operating point together. Quoting one outside that
context is quoting a number that does not exist independently.

## 14. Report the boring baselines, including the ones that make the model look worse

Three that belong next to any number from this panel:

- amino-acid **composition alone** reaches AUROC 0.754, so the embedding adds
  about 22 points over counting residues, not 48 over chance;
- with labels shuffled the same pipeline returns 0.506, which is the sanity
  check that the pipeline is not leaking;
- a probe trained on **lab-strain provenance with the hazard label ignored**
  still reaches 0.818 on v2 and 0.794 on v3, while the organism label agrees
  with the hazard label on only 53.4% and 43.6% of members. Separability here
  cannot be attributed to hazard alone.

And the head choice: every headline figure uses logistic regression, which is
the worst of four heads on 6 of 13 configurations and beaten on 11 of 13 by a
median of 5.0 points. Saying so makes the numbers a near lower bound, which is
a stronger position than quoting the best head.

## 15. Computational signal is not physical realizability, and conflating them misallocates review

The two highest-FSI toxins in this panel, BoNT-A at 2.24 and tetanus light
chain at 1.77, also carry the **highest** physical barrier, Tier 4 on a
five-dimension expert scoring. A detector that ranks by computational signal
alone would put its top two findings at the top of an audit queue where they
are the least actionable entries in it.

Anthrax protective antigen, at FSI 0.00 across 100 designs, is the most
interpretable result in the set and also Tier 4.

## 16. Design the metric as a measurement, not as an objective

Every metric here is defined over public reference proteins with published
annotations, so it scores existing knowledge rather than providing a gradient
for a design pipeline to climb. Aggregate statistics are released; per-design
sequences are not.

This is a constraint on metric design and not only on release. A score that an
attack pipeline can optimize against is a different artifact from a score that
measures a model, even when the formula is identical.

## 17. The binding constraint on expansion is annotation, not data volume

Worth stating because it is counterintuitive and it cost this project a sweep.
When looking for new hazard mechanism classes to add, six candidate classes
were screened and only one was viable. The failures were **not** short of
homologs: sialidase had 8 independent members and zero hazard keywords in its
annotations. One candidate failed only on a length window (2,710 and 2,366
residues against a 100 to 1,400 aa panel rule).

So the panel's n of 12 eligible classes is structural, not a matter of
harvesting harder. Any plan to grow a hazard panel should budget for
**annotation** work, and candidate counts from a keyword query should be
treated as upper bounds that need entry-by-entry reading.

---

## Scorecard: how this project's own probe does

Applied honestly, against its own criteria.

| criterion | verdict |
|---|---|
| 1 the split | **Fail, and it is the worst one.** No test set at all (178/118/0), the estimator returns 8.6% for a nominal 5% out of sample, and the resolution ceiling is 0.9915. Stated honestly, not repaired |
| 2 iso-FP comparisons | **Pass now, failed before.** The three-to-four fold "gain" was a budget artefact |
| 3 per class not aggregate | **Pass** on reporting, **fail** on performance: 10% on a 32-member class |
| 4 predicts its own failures | **Partial.** Ranks them at +0.894, over-predicts recovery by 34 points |
| 5 seed and arm stability | **Partial.** 8 of 9 classes stable at 30 seeds; the interesting one is not |
| 6 multiplicity fixed first | **Pass**, and the resulting verdict was NOT SUPPORTED |
| 7 replication across arms | **Mixed.** The headline margin result holds in 5 of 5 arms; the follow-up in 1 of 2 |
| 8 negative set measured | **Pass.** Effective n 3,550 against a raw 8,259, one contaminant named |
| 9 alerts and misses | **Pass.** 175 alerts, about 5 real, about 4 hazards passing |
| 10 metric audited | **Pass, after two silent defects were found** |
| 11 direction-blind corrections | **Pass.** One of three corrections hurt the metric and was kept |
| 12 prereg with floor and ceiling | **Fail.** Floor only, and it cost a correct interpretation |
| 13 recovery quoted in context | **Pass** in the documents, easy to violate in a slide |
| 14 boring baselines reported | **Pass.** Composition 0.754, shuffled 0.506, provenance 0.818 |
| 15 realizability separated | **Pass.** This is the framework's original point |
| 16 measurement not objective | **Pass** by construction |
| 17 annotation is the constraint | **Pass**, learned from a wasted sweep |

Three fails and four partials on seventeen criteria, on a framework whose
headline aggregate number is 0.981. That ratio is the reason this document
exists.

And the ordering matters more than the tally. The worst failure is criterion 1,
the split, which no amount of work on the other sixteen can compensate for: a
per-class table, a multiplicity threshold and an iso-FP control are all
improvements to a number that was still never measured on unseen negatives. An
earlier draft of this document listed calibration resolution as criterion 1 and
scored it a pass, which was true of resolution and quietly omitted that the
panel has no test set. Getting the ordering wrong is the most likely way to
audit a dataset thoroughly and still miss the thing that decides whether any of
it is usable.

---

## Examples

**Alignment as a detector.** Plain Smith-Waterman is beaten by this probe by
55.9 points across classes, and beats it on exactly one: beta-lactamase, 30%
against 19%. That single exception is more informative than the average,
because it identifies the regime where a cheap, interpretable, fully auditable
method is the better choice. A detector comparison that reports only the mean
would have hidden it.

**A detector that passes criterion 3 and fails criterion 1.** Any screen
quoting a 1e-6 false-positive rate from a few thousand calibration negatives.
The per-class table may be impeccable and the headline number is still
fabricated by extrapolation.

**A detector that passes criterion 1 and fails criterion 8.** A screen
calibrated on 100,000 sequences drawn from one organism's proteome. The
resolution is real and the negatives are nearly one protein repeated, so the
threshold is tuned to a single genome's composition.

**The failure mode criterion 2 is named for.** Any "we improved recovery by
adding more negatives" claim where the realized false-positive rate is not
quoted. It costs nothing to check and it reversed the sign of the conclusion
here.

---

## References inside this repository

- `docs/EVALUATION_REPORT.md`, the evaluator-facing report for the structure line
- `docs/MECHANISM_GENERALIZATION.md`, the leave-one-mechanism-out panel, its
  controls and its negative results; sections 10.3 through 10.9.2 are the
  source of criteria 1 to 9
- `docs/EXTERNAL_VALIDATION_PREREGISTRATION.md`, the frozen prediction and both
  failed outcomes behind criterion 12
- `docs/FSI_NUMBERING_AUDIT.md` and `docs/DATA_CORRECTIONS.md`, behind criteria
  10 and 11
- `src/22_claims_audit.py`, the gate that holds these documents to their
  artifacts
