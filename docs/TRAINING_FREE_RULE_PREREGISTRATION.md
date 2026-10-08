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
