
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
