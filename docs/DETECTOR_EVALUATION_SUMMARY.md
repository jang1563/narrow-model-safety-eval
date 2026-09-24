# The split comes first

**What it takes to trust a sequence hazard detector, and where ours fails**

A standalone summary of the 2026-09 line of work, written 2026-09-24 for a reader who will not open
a 2,600-line results document. Every figure below is recomputed from its artifact by
`src/22_claims_audit.py` on every run, and a figure here that drifts from its artifact fails the
gate. Sources are named per section; the long form is `docs/MECHANISM_GENERALIZATION.md` and the
criteria are `docs/DETECTOR_CRITERIA.md`.

---

## Summary

A hazard detector takes a biological sequence and returns a score or a flag. This project built one
— a linear probe over frozen protein-language-model embeddings — for the narrow purpose of finding
out what it takes to know whether such a thing works. Five results, in descending order of how much
they should change a reader's practice:

1. **The split is the binding constraint, not the model.** This panel's negatives divide 178 train /
   118 calibrate / **0 test**. Every specificity ever published from it was measured on negatives the
   pipeline had already seen. Measured out of sample, a nominal 5% false-positive budget really costs
   **about 8%**, and about **10%** once the benign pool is collapsed to distinct names.
2. **Aggregate recovery hides everything that matters.** Across held-out mechanism classes recovery
   spans **100% to 10%** at the same operating point on the same panel, and plain Smith-Waterman
   alignment beats the embedding probe on the class the probe is worst at.
3. **The failures are geometric and locatable in advance, but the location is not a calibration.**
   Embedding margin ranks the unreachable classes correctly (Spearman **+0.894**, permutation
   *p* = 0.0001, holding in 12 of 14 representations) and **mis-states the miss rate of an unseen
   class by 34 points, always optimistically**.
4. **Whoever supplies the benign set can suppress a hazard class without touching the aggregate.**
   500 genuine Swiss-Prot proteins, none forged or mislabelled, cut one class from **63.5% to 16.7%**
   while the measured false-positive rate *improved*.
5. **A detector's own evaluation can be scored, and ours fails.** Against eighteen criteria, each
   anchored to a place this project fell short: **three fails, three partials, one mixed**. The worst
   is the first one, the split.

The transferable claim is not about protein models. It is that **a detector number without a stated
split, a per-class table, and a realized false-positive rate is not evidence**, and most published
detector numbers — including several early ones in this repository — are missing at least three of
those.

---

## 1. What is being evaluated

**The panel.** Two frozen versions. v2 is 80 hazardous proteins in 13 curated mechanism classes
against 154 benign negatives in three provenance-tagged blocks; v3 adds three non-animal-target
classes (bacteriocin, phage peptidoglycan hydrolase, *B. thuringiensis* Cry toxins) for
**149 positives / 296 negatives** and 12 classes in the recovery table. Every member is a reviewed
Swiss-Prot entry with a written admission reason. Baseline separability on v2 is
**AUROC 0.974 ± 0.014**.

**The protocol.** Leave-one-mechanism-out: hold out an entire hazard class, fit the probe on the rest
plus the negatives, set a threshold at a nominal specificity on calibration negatives, and measure
what fraction of the unseen class is flagged. This asks the deployment question — what happens when
the hazardous molecule is not on the list — rather than the benchmark question.

**The arms.** 14 representations: ESM-2 at 8M / 35M / 150M / 650M / 3B, ESM-C at 300M / 600M / 6B,
ESM-3, ProtT5, SaProt, and mean / CLS / max pooling variants of the canonical 650M arm. Where a
result is quoted for one arm, it is because it was tested on more than one and the others are named.

**Three baselines that belong beside every number.** Amino-acid composition alone reaches
AUROC **0.754**, so the embedding buys about 22 points over a bag of residues. With labels shuffled
the same pipeline returns **0.506**, so the machinery is not inventing signal. And a probe trained on
**lab-strain provenance with the hazard label ignored** reaches **0.818**, with the organism label
agreeing with the hazard label on **53%** of the panel — so separability on this panel cannot be
attributed to hazard alone. Every recovery figure also uses logistic regression, which is the worst
of four classifier heads on 6 of 13 configurations and beaten on 11 of 13 by a median of 5.0 points;
read the recovery numbers as close to a lower bound.

---

## 2. Recovery is class-dependent, and the aggregate is uninformative

At a nominal 95% specificity on the canonical 650M arm, v2:

| mechanism class | n | flagged@95 | flagged@99 | AUROC |
|---|---:|---:|---:|---:|
| adp-ribosyl AB toxin | 7 | 100% | 91% | 0.994 |
| clostridial neurotoxin | 6 | 100% | 100% | — |
| RIP rRNA glycosidase | 7 | 100% | 89% | 0.997 |
| superantigen enterotoxin | 7 | 100% | 100% | 1.000 |
| T3SS effector apparatus | 10 | 80% | 80% | 0.949 |
| pore-forming cytolysin | 7 | 69% | 54% | 0.962 |
| virulence-associated non-toxin *(labelled control)* | 10 | 50% | 32% | 0.844 |
| contact-dependent inhibition | 4 | 35% | 0% | — |
| **beta-lactamase** | 14 | **21%** | **1%** | 0.751 |

Four classes are fully recovered having never been trained on. The largest class in the panel, a
family defined by a conserved fold and an active site, is almost entirely missed. **A single
aggregate number over this table would describe none of its rows.**

Two consequences that are easy to state and easy to forget:

- **Alignment is the honest baseline and it is not always beaten.** Averaged across classes the probe
  beats Smith-Waterman by **+55.9 points**. On beta-lactamase alignment wins, **29.5% against 21.4%**.
  That single exception is the more useful half of the comparison: it identifies the regime where a
  cheap, interpretable, fully auditable method is the better choice, and a comparison reporting only
  the mean would have hidden it.
- **Recovery is not a property of the class.** Expanding the panel from 66 to 80 positives moved
  pore-forming cytolysin by **+11.4 points without adding a single member to it**, because adding
  positives moves the calibrated threshold and members already near it cross. A per-class recovery
  figure is a joint property of the class, the rest of the positive set, and the operating point.

**One arm reaches the hard class.** ESM-C 600M recovers beta-lactamase at **48.3% [43.9, 52.7]** over
30 seeds, the only one of 14 arms whose interval clears alignment. An earlier write-up here said the
class resisted every configuration tested; that was wrong when written and is corrected in the log.

---

## 3. The failures are geometric, and geometry locates them without calibrating them

v3 produced a **second** unreachable class: phage peptidoglycan hydrolase, **10% at 95% specificity**
on n = 32, worse than beta-lactamase's 18.6% on the same panel. That matters because it converts an
anomaly into a phenomenon. Seven candidate explanations for beta-lactamase alone were tested and
refused — pooling, classifier head, model scale, structure, lineage, pretraining exposure, AMR
category — and none of them was ever the question.

What locates both failures is **margin**: for each class, the distance to the nearest positive of
another class minus the distance to the nearest negative.

- Its two lowest classes out of twelve are **exactly** the two failures — chance 1 in 66.
- It tracks recovery at Spearman **+0.894**, permutation *p* = 0.0001.
- It beats each of its own parts (nearest-positive and nearest-negative alone locate nothing), while
  class size runs the other way at **−0.564**.
- Both failing classes have a **negative** margin: their members sit closer to a benign protein than
  to any hazard class the probe was trained on.
- The effect is representation-general: on v2, in **all 14 arms** the failing class has negative
  margin, the correlation is significant in 13, and margin locates the failure in 12. The two misses
  are the CLS and SaProt arms, which is a property of those representations rather than of the
  finding.
- 🔴 **But the stricter version of that test comes back partial, and the distinction matters.**
  v2 asks whether margin finds the bottom-1 of nine classes, chance 1/9. v3 asks for the bottom-**2**
  of twelve, chance 1/66, and only **5 arms** are embedded for v3. There, negative margin for both
  failing classes and a significant correlation (+0.796 to +0.894, every *p* ≤ 0.0016) hold in
  **5 of 5**, and beta-lactamase is the lowest-margin class in all five — but the exact bottom-two
  holds in only **2 of 5**. In the other three the labelled virulence control displaces phage, the
  same displacer each time, which says the bottom of the margin ordering is where hazard and a
  borderline control stop being distinguishable. **The ordering and the negative-margin property are
  representation-general; the identity of the bottom-k is not.** § 10.6.1.

**And it does not transfer to a miss-rate estimate.** Leave-one-class-out, margin ranks the unseen
mechanism correctly — phage is predicted lowest and is lowest — while predicting **44.3% recovery
against a measured 10.0%**, an error of **34.3 points**. Its cross-validated MAE, 0.134, beats a
mean-prediction baseline at 0.284, and **every error is optimistic**. So margin is a triage signal:
it says which classes to worry about, and it will tell you the damage is smaller than it is.

**Benign proximity is a contributing cause, not the cause.** Removing the benign proteins nearest a
failing class recovers **8.8 points for beta-lactamase and 7.1 for phage**, both with confidence
intervals excluding zero, which is about **a tenth** of each gap. The effect scales monotonically with
dose and at the largest dose still closes only 26% and 19%. Curating the negative set is therefore
not the repair.

---

## 4. The split, which is the result that should change practice

Three failures live in the negative split and they compound.

**(a) There is no test set.** 178 train / 118 calibrate / **0 test**. Every negative is either fitted
on or used to place the threshold, so no false-positive number in this repository was measured on
negatives the pipeline had not already seen.

**(b) The threshold estimator does not deliver its own nominal rate.** Carving a genuine test
partition out of the panel (177 / 59 / 60, 200 seeds, seed as the unit of inference):

| arm | nominal | `np.quantile` | conformal |
|---|---:|---|---|
| canonical 650M | 5% | 6.40% [5.82, 6.98] **exceeds** | 4.98% [4.45, 5.51] covers |
| canonical 650M | 1% | 2.84% [2.44, 3.24] **exceeds** | unreachable at m = 59 |
| esm2_35M | 5% | 6.67% [6.09, 7.24] **exceeds** | 5.15% [4.62, 5.69] covers |
| esm2_35M | 1% | 2.90% [2.52, 3.29] **exceeds** | unreachable at m = 59 |

The published estimator overshoots in both arms at both budgets, realizing roughly **three times** a
nominal 1%. Conformal attains its guarantee. It also **declines to answer** at 1% where the quantile
estimator invents a number, and an estimator that refuses is strictly more useful than one that
extrapolates silently.

⚠️ Settling this took three attempts, and the arithmetic is worth carrying. With m = 59 and
α = 0.05, `(m+1)·α` is **exactly 3**, so the guarantee is exactly 3/60 = 5.00% with **zero
conservatism margin** — about half of all point estimates land above nominal by construction, and two
arms both landing above is a coin flip coming up twice. m = 59, 79, 99 and 119 all hit that integer.
**The margin is not monotone in calibration size**, so a calibration set should be sized to miss the
integer rather than to be as large as possible.

**(c) The budget decomposes, and the estimator is the smallest term.** Taking the test negatives from
an 8,259-protein benign pool instead keeps calibration at the published 118, where conformal is
reachable at both budgets. Reading down the conformal column at a nominal 5%, canonical arm:

```
4.20%   the guarantee the arithmetic promises
4.32%   what it delivers when the negatives really are exchangeable
5.98%   after calibration and test negatives come from different curations
7.14%   after the pool's duplicate names stop hiding the failures
```

Theory is exact to within 0.03 points. **Every point of overshoot above that is bought by
distribution shift and by name redundancy, not by the estimator.** Both arms agree: the
`np.quantile` figures are 7.87% and 7.94% under shift, 9.64% and 10.29% de-duplicated.

Two things fall out that the earlier analysis could not see. First, `np.quantile` exceeds nominal
**even under exchangeability**, so part of its overshoot was never about shift. Second, collapsing
the pool to one protein per distinct name **raises** the measured false-positive rate: the duplicated
entries are the easy ones. **Effective n does not only widen intervals, it moves point estimates.**

**(d) The panel cannot calibrate a deployable threshold at all.** Its 296 negatives resolve a
false-positive rate no finer than 1/296; reaching 1e-4 would need **250,003** of them. At a deployment
prevalence of 1 in 1,000 and the strictest specification the panel can express, the arithmetic is
**175 alerts, about 5 of them real, and about 4 hazards passing**. That is the honest operating
picture, and it is available before any deployment decision rather than after.

---

## 5. The benign reference set is an attack surface

The negative set decides where the boundary sits, so whoever supplies it decides what the screen can
see. This is usually treated as a data-quality problem. It is also a security problem, and the
mechanism needs **no mislabelled data at all**.

Append 500 pool proteins chosen for proximity to one hazard class and distance from the others, label
them benign, and refit:

| arm | target recovery, panel only | with +500 | pool FP, panel only | with +500 |
|---|---:|---:|---:|---:|
| esm2_35M | 63.5% | **16.7%** | 8.30% | **5.87%** |
| canonical 650M | 79.4% | **54.8%** | 7.13% | 7.68% |

On the small arm the targeted class collapses **while the aggregate false-positive rate improves**.
An operator watching the only aggregate they have would record the supplemented reference set as the
better one. Every one of the 500 is a genuine reviewed Swiss-Prot entry passing the panel's own
hazard exclusions; nothing is forged, and the additions are 6% of the pool.

**The honest qualifier is that clean targeting is the minority case.** Across 13 classes the targeted
drop exceeds the worst collateral drop on only **5 of 13** on both arms. Nine times out of thirteen
the class hurt most is not the class aimed at. The reason is geometric and is the more durable
finding: the per-class nearest-500 sets are largely the same set — mean pairwise Jaccard **0.324**,
reaching **0.883** for one pair, and all thirteen together span **1,999 distinct pool proteins out of
6,500 slots**. "The benign neighbourhood of hazard class X" is mostly just "the benign neighbourhood
of the panel". An attacker cannot suppress one class quietly; a careless curator can degrade several
at once without noticing.

---

## 6. A preregistration that failed twice

One claim from this line of work was frozen before any external data was fetched: that embedding
margin predicts which unseen molecules a probe will miss, with a floor, a comparison and a gap
specified in advance.

| | predicted | internal | attempt 1 | attempt 2 |
|---|---|---:|---:|---:|
| low-margin holdout | ≤ 40% | 22% | 82% ❌ | 100% ❌ |
| class-matched random | ≥ 60% | 84% | 94% | 100% |
| gap | ≥ 25 pts | +62 | **+13** ❌ | **+0** ❌ |

Both external panels falsified it, and the claim was downgraded to a property of the internal panel
as the preregistration required. **A defect in the preregistration is recorded with it:** it
specified a floor and no ceiling, so attempt 2's uniform 100% is logged as NOT SUPPORTED where
*uninformative* is the accurate verdict — a test where everything is recovered cannot discriminate.
That is criterion 12, and this project fails it.

The preregistration is the reason this section can be written at all. Nothing about the outcome is
retrospectively reframed, because the frame was fixed first.

---

## 7. Scoring the evaluation itself

`docs/DETECTOR_CRITERIA.md` states eighteen criteria for evaluating a hazard detector, each anchored
to a number measured here — usually to a place this project's own probe fell short — and scores this
project against them. The result is **three fails (the split, per-class performance, the
preregistration ceiling), three partials, one mixed**, on a framework whose headline aggregate is
0.974.

The ordering matters more than the tally. A per-class table, a multiplicity threshold and an iso-FP
control are all improvements to a number that was still never measured on unseen negatives. An
earlier draft of that document listed calibration resolution first and scored it a pass, which was
true of resolution and quietly omitted that the panel has no test set. **Getting the ordering wrong
is the most likely way to audit a dataset thoroughly and still miss the thing that decides whether
any of it is usable.**

Three worked examples of the criteria discriminating between detectors:

- *Passes the per-class criterion, fails the split.* Any screen quoting a 1e-6 false-positive rate
  from a few thousand calibration negatives. The per-class table may be impeccable; the headline is
  fabricated by extrapolation.
- *Passes the split, fails the negative-provenance criterion.* A screen calibrated on 100,000
  sequences drawn from one organism's proteome. The resolution is real and the negatives are nearly
  one protein repeated.
- *The failure mode the iso-FP criterion is named for.* Any "we improved recovery by adding
  negatives" claim that does not quote the realized false-positive rate. It costs nothing to check,
  and checking it reversed the sign of a conclusion here.

---

## 8. What would change these conclusions

Stated as falsifiable conditions rather than future work:

- **A negative set large enough to carry a real test partition without halving calibration
  resolution.** That is the actual repair for the headline failure and it is not attempted here; the
  pool's effective size, 3,550 distinct names behind 8,259 records, is already the reason the
  de-duplicated rate is the honest one.
- **A second annotator.** All functional-site annotations, metric implementations and audits are by
  one person. Two silent numbering defects were found by internal audit; the rate at which such
  defects survive one annotator is unmeasured, and by construction cannot be measured from inside.
- **Cross-institution replication of the design-model half.** The FSI pipeline has never been run
  independently on the same structures.
- **A margin-like signal that is calibrated as well as ordinal.** Anything that ranks failures as
  well as margin does *and* states a miss rate within its own error bars would supersede § 3.
- **Any demonstration that the class-level margin effect is an artifact of mean pooling.** It already
  fails on the CLS and max arms of the canonical model, which is the sharpest existing lead against
  it.

---

## 9. Reproduction

```bash
git clone https://github.com/jang1563/narrow-model-safety-eval.git
pip install -e ".[dev]"

python src/03b_leave_one_mechanism_out.py --panel v3      # § 2, the recovery table
python src/29_margin_predicts_new_classes.py              # § 3, margin and its miss-rate error
python src/30_margin_across_arms.py                       # § 3, all 14 representations
python src/45_negative_test_set_audit.py                  # § 4a-b, the split and the estimator
python src/48_conformal_lomo_test_split.py --panel v3     # § 4b, test set from inside the panel
python src/49_external_test_partition.py --arm canonical  # § 4c, test set from the pool
python src/50_reference_set_poisoning.py                  # § 5, the attack surface
python src/22_claims_audit.py                             # every number above, against its artifact
```

The last command is the one that matters for reading the rest: it recomputes each headline figure
from its stored artifact and checks it against the sentence quoting it in every public document.
Exit status 1 if any claim has drifted.

---

*JangKeun Kim, Weill Cornell Medicine. Panels, annotations and aggregate results are public at
[`jang1563/narrow-model-safety-eval`](https://huggingface.co/datasets/jang1563/narrow-model-safety-eval);
code, documents and the corrections log are on
[GitHub](https://github.com/jang1563/narrow-model-safety-eval). No model-generated sequences,
synthesis routes or design protocols are released — see `SAFETY.md` and `docs/RELEASE_SURFACE.md`.*
