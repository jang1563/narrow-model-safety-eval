# The surviving hazard claim, at twenty-seven times the pairs — preregistration

**Frozen 2026-10-02, before `src/130` existed.** Study AM.

## 1. The one claim § 6 still makes, and how thin its evidence is

After studies AI-3 and AJ, § 6 claims exactly one thing about hazard:

> 🔒 The screen **cannot order a curated toxin above a length-matched non-toxin inside one fold.**

Its whole evidence is study AH: **31 pairs**, AUROC **0.4599** and **0.5247**, both arms inside a band
frozen at 0.40–0.60 against a hazard floor of 0.70. ⚠️ **Thirty-one pairs is the limit of what 30.6%
reviewed UniProt coverage allowed**, and the paper says so.

🔑 **DeepVIC's axis has 1,126 Exotoxin and 1,121 Motility members, and study AI-3 already built 828
length-matched pairs from them.** AI-3 measured the **recovery difference** on those pairs at a fixed
threshold, which is a *different quantity* and which did **not** replicate (the arms disagree on its
sign). 🔴 **Nobody has measured the ordering quantity — the AUROC — on that population.** This does.

## 2. Why the two quantities differ, restated so the result cannot be misread

- **Recovery difference** (AI-3): the fraction flagged above a 95%-specificity threshold, in one group
  minus the other, from two separate folds. Sensitive to where the whole score distribution sits.
- **AUROC** (AH, and this study): the probability that a randomly chosen hazard member scores above a
  randomly chosen matched non-hazard member, **within one fold, from one model**. Threshold-free.

⚠️ **A class can be flagged more often while not being orderable above its comparator**, which is why
§ 6 reports AH and declines AI-3. 🔑 **If AUROC at 828 pairs lands where it did at 31, § 6's surviving
claim is no longer resting on 31 pairs.** If it does not, § 6 has nothing left on hazard and must say so.

## 3. The design

`src/125`'s fold, unchanged, with one difference: both groups of each pair are held out of the **same**
fit and scored by the **same** model, exactly as `src/121` does, and the reported statistic is AUROC
rather than recovery.

- pairs: study AI-3's **828**, rebuilt by the same matcher (nearest length, no replacement, ratio ≤ 2.0)
- held out: Exotoxin ∪ the matched Motility members, in one fold; **strict** variant also holds out the
  whole Motility and Exotoxin categories
- negatives: the partition from `src/125`, reproduced bit-for-bit
- 30 seeds, both arms
- ⚠️ a **composition arm** is run on the identical pairs, because study AJ showed a 21-number classifier
  reproduces the 35M arm on this category; if the embedding arms sit at chance and composition does too,
  that is one finding, and if composition orders them while the embeddings do not, that is another.

## 4. Frozen bands

Floors **and** ceilings. **Both arms in the same band for a verdict**, as everywhere here.

| | cannot order | can order | retraction |
|---|---|---|---|
| **AM-1** AUROC, hazard vs matched, strict | **0.40 – 0.60** | **≥ 0.70** | ≤ 0.30 or ≥ 0.80 |
| **AM-2** paired bootstrap on AUROC excludes 0.50 | — | — | diagnostic, no band |

🔒 **AM-1 in the "cannot order" band on both arms ⇒ § 6's surviving claim replicates at 828 pairs**, and
the 31-pair caveat is removed from the abstract. 🔴 **AM-1 in the "can order" band on either arm ⇒ the
claim is wrong at scale and § 6 loses its last hazard statement.** Between ⇒ indicative.

⚠️ **n = 828 pairs makes the interval narrow, which cuts both ways**: an AUROC of 0.56 would be
*distinguishable from chance* here and still inside the "cannot order" band. The band is about
usefulness, not about significance, and that is stated in advance.

## 5. Predictions

**P1 (derived — from AH's 0.4599 and 0.5247 at 31 pairs).** AM-1 lands in 0.40–0.60 on both arms.

**P2 (derived — from study AJ's composition arm reproducing the 35M Exotoxin recovery to 0.08 points).**
The composition arm's AUROC is **further from 0.50** than the 650M arm's, in the direction that favours
exotoxins.

**P3 (reasoned).** The paired bootstrap **excludes 0.50** on at least one arm, because 828 pairs is
twenty-seven times AH's sample and AH's own intervals were wide enough to hide a real effect of this size.

⚠️ **The record these are scored against**: across AH, AI, AJ and AK, derived predictions have held
**4 of 9** and reasoned **2 of 4**, and entry 109 withdrew the "direction holds, magnitude does not"
reading as over-fit. 🔑 **P1 is a direction-and-band prediction carried from a different population**,
which is the kind that has done worst here.

## 6. What each outcome changes

- **Replicates** ⇒ the abstract's fifth result drops "on 31 pairs" and gains "and at 828".
- **Fails** ⇒ 🔴 § 6 states that the screen *can* order toxins above matched non-toxins at scale, the
  narrowing to registration is withdrawn in both its forms, and § 7 gains the retraction.
- **Composition orders them while the embeddings do not** ⇒ the ordering is compositional, which would
  make the screen's hazard-blindness a property of the representation rather than of the task.
