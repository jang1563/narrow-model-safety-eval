# A training-free rule for whether the negative term belongs — preregistration

**Frozen 2026-10-07, before any new embedding was computed.** Study AV.

## 1. The claim, and why it needs a sealed test

Studies AT and AU established, **post hoc**, that `margin = nearest_positive − nearest_negative` beats its
own first term only sometimes, and that when it does is predictable:

| | |
|---|---|
| gain from subtracting vs **\|*ρ*(nearest-negative, recovery)\|** | Spearman **+0.8118** over 16 cell-settings |
| gain from subtracting vs ***ρ*(nearest-positive, nearest-negative)** across classes | Spearman **−0.7219** |

🔑 **The second quantity needs no training.** It is a rank correlation between two per-class numbers, both
read off the frozen embedding and the class partition before a single probe is fitted. 🔑 **When the two
halves of the formula co-vary across classes, subtracting one from the other removes shared signal; when
they are independent or opposed, subtracting isolates it.**

🔴 **Both numbers were computed on cells whose results were already known**, and the second's threshold
would be chosen on the same 16 points. ⚠️ **A correlation fitted and tested on one set of cells is not a
rule.** 🔒 **This study seals a per-cell prediction before each fold runs and scores it afterwards.**

## 2. The rule, with its abstention region

🔒 Let **R = *ρ*(nearest-positive, nearest-negative)** over a cell's held-out classes, computed with
`src/30`'s Spearman on the frozen embedding, union negatives, no training.

| R | prediction |
|---|---|
| **< +0.15** | the gain is **positive** at both classifier settings — margin beats `nearest_positive` |
| **> +0.35** | the gain is **negative** at both settings — `nearest_positive` beats margin |
| **+0.15 to +0.35** | 🔒 **no prediction**, recorded as an **abstention** |

⚠️ **The two thresholds are chosen on the sixteen known cell-settings and are therefore fitted**, which is
exactly why the test is on cells not yet embedded. 🔒 **On those sixteen the rule is 6/6 below +0.15 and
4/4 above +0.35, and the middle band holds the four cells where the two settings disagree** — so the
abstention region is where the rule is known to be unreliable, declared rather than discovered.

🔑 **An abstention is not a free pass.** If the rule abstains on most new cells it has no content, so § 4
bands the abstention rate as well as the hit rate.

## 3. The new cells

🔒 **Four cells never embedded in this project**, chosen for training-objective diversity rather than
performance, on the two datasets whose folds are already frozen:

| cell | encoder | why it is a different thing |
|---|---|---|
| `text/distilroberta` | `sentence-transformers/all-distilroberta-v1` | a different base architecture from MiniLM and MPNet |
| `text/minilm_l3` | `sentence-transformers/paraphrase-MiniLM-L3-v2` | **3 layers**, a capacity extreme |
| `text/bge_base` | `BAAI/bge-base-en-v1.5` | a retrieval-contrastive objective |
| `image/vit_b16_sup` | `google/vit-base-patch16-224` | **supervised** ImageNet labels, against CLIP's language supervision and DINOv2's self-supervision |

🔒 **Everything else is unchanged**: `src/125`'s fold, union negatives, both the fixed `C = 1.0` and
`src/133`'s nested-selected `C`, 30 seeds, the 20 Newsgroups and CIFAR-100 splits already frozen in
`docs/CROSS_MODALITY_PREREGISTRATION.md` and `docs/GENERALITY_GRID_PREREGISTRATION.md` § 2.

🔒 **The procedure is sequenced so the prediction cannot follow the result:**

1. Embed all four cells. **No fold runs.**
2. Compute **R** for each and append the four sealed predictions to this document as **amendment 1**,
   with the commit that adds them preceding any fold.
3. Run the folds.
4. Score.

⚠️ **Step 2's amendment is the sealed record.** If it is missing, the study is void and says so.

## 4. Frozen bands

Bonferroni α = 0.05/2 over the two primary tests.

| | the rule has content | it does not |
|---|---|---|
| **AV-1** sealed sign predictions correct | **≥ 6 of 8** cell-settings | **≤ 4 of 8** |
| **AV-2** cells on which the rule abstained | **≤ 1 of 4** | **≥ 3 of 4** |
| **AV-3** *ρ*(R, gain) recomputed over **all** cell-settings, old and new | **≤ −0.60** | ≥ −0.20 |

🔒 **AV-1 and AV-2 are read together.** Eight of eight correct on one non-abstaining cell is not evidence;
🔑 **a rule that abstains is allowed to be silent, not allowed to be silent and still called predictive.**

⚠️ **Eight cell-settings is a small test and the two settings of a cell are not independent** — they share
an embedding and a fold. 🔒 **The per-cell count is reported alongside**: a rule that is right on 4 cells
is stronger evidence than one right on 6 of 8 settings across 2 cells, and both numbers appear.

## 5. Predictions about the predictions

**P1 (derived — from *ρ* = −0.7219 over sixteen cell-settings, and from the rule being 10 of 10 outside the
abstention band on those).** AV-1 lands at **≥ 6 of 8**.

**P2 (reasoned).** The **supervised** image encoder abstains or predicts a *negative* gain, because
ImageNet-label supervision should make a class's distance to the negative side track its distance to
everything — which is the co-variation R measures.

**P3 (derived — the three-layer encoder has 1/4 of MiniLM-L6's depth, and study AP's mechanism is that
weaker representations leave more for the classifier to do).** `text/minilm_l3` has the **lowest** mean
recovery of the four new cells.

⚠️ **The computed record stands at derived 17 of 33, reasoned 4 of 12.**

## 6. What each outcome changes

- **AV-1 ≥ 6 of 8 with AV-2 ≤ 1** ⇒ 🔑 **a quantity requiring no training predicts which form of the
  statistic to use**, which turns the post-hoc mechanism into a usable rule and is the result this project
  would lead an ML-venue paper with.
- **AV-1 ≤ 4 of 8** ⇒ the mechanism is a description of sixteen cells and not a rule. 🔑 **The post-hoc
  correlations stay in the paper as descriptions**, explicitly not as a procedure.
- **AV-2 ≥ 3 of 4** ⇒ the thresholds were fitted so tightly that the rule has no reach, and the honest
  report is that the abstention region swallows the claim.

---

## Amendment 0 — a factual error in § 2's description of the fitting set

**Added 2026-10-07, before any fold ran and before the sealed predictions below.** 🔒 **The rule and both
thresholds are unchanged.** This corrects a sentence *about* the calibration set, not the set or the
bands.

🔴 § 2 says the abstention region "holds the four cells where the two settings disagree". ⚠️ **It holds
three cells — six cell-settings — and only two of the three disagree.** Computed by the scorer's own
selftest: **6 below +0.15, 6 abstaining, 4 above +0.35.**

| cell | R | gain, fixed / swept |
|---|---:|---|
| `text/mpnet_base` | +0.1923 | +0.0440 / **−0.0440** |
| `text/minilm_l6` | +0.2253 | +0.1099 / **−0.0385** |
| `image_sc/dinov2_vits14` | +0.1879 | +0.0606 / +0.3030 — **both positive** |

🔑 **So the abstention band is conservative rather than tight**: one of the three cells inside it would
have been predicted correctly at both settings had the rule been willing to speak. ⚠️ **That is a reason
the band may be too wide, not too narrow**, and AV-2 bands the abstention rate for exactly this reason.
🔒 **The thresholds stay where they were frozen** — moving them now, with the direction of the error
known, is the move this project has declined twice today.

---

## Amendment 1 — the sealed predictions

**Added 2026-10-08. 🔒 No fold has run on any of these four cells.** R was computed by
`src/139 --embed-only`, which fits nothing, and `results/training_free_rule_sealed.json` asserts
`no_fold_ran: true`. 🔒 **The commit adding this section precedes every fold**, which is the only thing
separating a sealed prediction from a retrodiction.

| cell | encoder | dim | R | 🔒 **sealed prediction** |
|---|---|---:|---:|---|
| `text/distilroberta` | all-distilroberta-v1 | 768 | **+0.0714** | **positive gain** — margin beats `nearest_positive` at both settings |
| `image/vit_b16_sup` | google/vit-base-patch16-224 | 768 | **+0.1086** | **positive gain** at both settings |
| `text/bge_base` | BAAI/bge-base-en-v1.5 | 768 | **+0.1593** | 🔒 **ABSTAIN** |
| `text/minilm_l3` | paraphrase-MiniLM-L3-v2 | 384 | **+0.2473** | 🔒 **ABSTAIN** |

### 🔴 P2 has already failed, before any fold

**P2 (reasoned)** said the **supervised** image encoder would abstain or predict a *negative* gain, because
ImageNet-label supervision should make a class's distance to the negative side track its distance to
everything. 🔴 **R = +0.1086 predicts a positive gain**: on this measure the supervised ViT resembles
DINOv2 (+0.1105) and not CLIP (+0.3825). ⚠️ **Scored now rather than later**, because it is a claim about R
and R is settled.

### 🔴 AV-1 cannot clear, and this is visible from the abstentions alone

§ 4 bands **AV-1** at **≥ 6 of 8** cell-settings correct. 🔒 **Two of the four cells abstain, so only four
cell-settings are scorable and the maximum attainable is 4.** 🔴 **A perfect record would therefore land
in AV-1's *failure* band of "≤ 4 of 8".**

⚠️ **This is recorded with no knowledge of any outcome.** The abstentions follow from R alone; not one
gain has been computed. 🔑 **The defect is that AV-1 fixed a denominator which the rule's own abstention
region can shrink** — the same shape as amendment 1 of `GENERALITY_GRID`, where AT-1b could not bind.

🔒 **Both readings will be reported, and neither band is being moved:**

- **Literal**, as frozen: ≥ 6 of 8. **Unattainable at 2 abstentions, so AV-1 is void by construction.**
  This is the reading with no hindsight in it.
- **Reduced denominator**, the convention AT-0 and AU-0 already use, where an abstained unit leaves the
  numerator *and* the denominator and the reduced denominator is stated: **of 4**.

🔑 **AV-3 is unaffected and is the live test.** It is a rank correlation over every cell-setting, old and
new, and needs no thresholds — so the abstention region cannot reach it. ⚠️ **If AV-3 is the only band
that survives, then the honest summary of this study is that the *continuous* relationship holds or does
not, and that the thresholded rule was never testable at this sample size.**

### ⚠️ And the abstention band's reach is now measured

🔑 **Three of the seven text cell-pairs in this project sit inside [+0.15, +0.35]** — `minilm_l6`
(+0.2253), `mpnet_base` (+0.1923), `bge_base` (+0.1593), `minilm_l3` (+0.2473) — against `distilroberta`
at +0.0714. ⚠️ **The band swallows most sentence encoders**, which amendment 0 already suspected from the
other direction. 🔒 **AV-2 reads 2 of 4: between bands, confirming nothing.**
