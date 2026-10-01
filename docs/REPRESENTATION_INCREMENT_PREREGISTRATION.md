# What does the representation buy, category by category? — preregistration

**Frozen 2026-10-02, before `src/128` existed.** Study AK.

## 1. Disclosure: what is already known at freezing time

🔴 **This is not a sealed prediction and the paper must not present it as one.** Studies AI-2 and
AJ are already in the record, so the following are known:

| | unweighted mean recovery over the 14 categories |
|---|---:|
| ESM-2 650M | **41.4%** |
| ESM-2 35M | **41.8%** |
| composition, 21 features | **34.2%** |
| length alone, 1 feature | **17.6%** |

and that composition beats **both** embedding arms on **Exotoxin** (58.3% against 44.7% and 58.4%).

🔒 **What is not known, and is what this study measures:** the per-category differences, how many
of the fourteen categories the embedding wins, whether the categories it loses are the **same set**
on both arms, and the size of the win where it wins. Knowing three means constrains none of these.

## 2. Why it is worth a study rather than a sentence

Three populations now say the same thing from different directions:

- § 5: two metadata fields with no sequence information reach **AUROC 0.8096** against this project's
  embedding at 0.9323, and add **+0.0007** on top of it.
- § 1.2 (AI-1): on DeepVIC's benchmark, 21 model-free numbers reach **AUROC 0.8705**, **91.3%** of its
  published 0.954.
- § 6 (AJ): composition alone reproduces the 35M arm's Exotoxin recovery to **0.08 points**.

🔑 **None of those is the number a person deciding whether to use a protein language model for this
task would want.** That number is *per category, at a matched false-positive rate*: how much recovery
does the representation add where it adds any, and where does it add nothing. 🔒 **All three folds are
calibrated to the 95th percentile of the same calibrate negatives and come out at 4.9–5.1% on the same
untouched test negatives**, so their recovery rates are directly comparable without any adjustment.

## 3. The measurement

No new fitting. For each of DeepVIC's 14 categories, from `results/deepvic_loco.json` and
`results/composition_arm.json`:

    increment(category, arm) = recovery(arm) − recovery(composition)

reported per category with a seed-level interval on the difference, plus:

- **AK-1** the count of categories where the increment is positive, per arm
- **AK-2** whether the loss set — the categories where composition wins — is the same on both arms
- **AK-3** the mean increment over categories where it is positive, and over all fourteen
- **AK-4** the increment against the **length-alone** arm, as the floor of the comparison

⚠️ **The interval is over the 30 probe seeds**, which locates a seed mean and does not bound the
quantity (entry 83). A category-level bootstrap is reported for the aggregate in AK-3, where the unit
is the category.

## 4. Frozen bands

Floors **and** ceilings, on **both** arms. Bonferroni α = 0.05/4.

| | representation earns its place | representation is not load-bearing |
|---|---|---|
| **AK-1** categories won, of 14 | **≥ 11** | **≤ 7** |
| **AK-3** mean increment over all 14 | **≥ +10pp** | **≤ +3pp** |
| **AK-2** loss sets identical across arms | — | — (diagnostic, no band) |

🔒 **A verdict requires both arms in the same column.** Between the columns is reported as
*the representation adds something and not much*, which is a finding rather than a failure.

## 5. Predictions

**P1 (derived — from the 41.4/41.8 against 34.2 means).** AK-3 lands **between** the bands: the mean
increment is above +3pp and below +10pp on both arms. ⚠️ This is a derivation from arithmetic I have
already seen, and it is written down to be scored rather than to be impressive.

**P2 (derived — from study AJ).** **Exotoxin** is in the loss set on both arms.

**P3 (reasoned).** The loss sets are **identical** across the two arms, and contain at most three
categories, all of them ones a compositional signature would mark — Exotoxin and Exoenzyme among them.

⚠️ **The record this is scored against**: across studies AH, AI and AJ, derived predictions have held
3 of 7 and reasoned 2 of 3, and entry 108's reading is that the **direction** of a derived prediction
has held 5 of 7 times while its **magnitude** almost never has. 🔑 **P1 is a magnitude prediction with
a floor and a ceiling**, so it is the harder kind, and P2 is a direction prediction.

## 6. What each outcome changes

- **Both arms "earns its place"** ⇒ § 5's "nothing moved the precision" is a statement about our
  panel's precision and not about the representation's value, and § 1.2's no-semantics baseline is a
  weaker criticism than it reads.
- **Both arms "not load-bearing"** ⇒ 🔴 the strongest claim this paper could make about protein
  language models for registration screening is that **21 numbers are most of the way there**, and § 8
  must say it.
- **Between** ⇒ reported as a quantity, per category, with the loss set named. 🔑 That is the useful
  form for a reader deciding whether to spend a GPU on this, and it is the outcome P1 predicts.
