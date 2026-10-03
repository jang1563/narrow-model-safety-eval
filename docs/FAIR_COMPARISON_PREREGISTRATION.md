# The comparison at a tuned classifier on both sides — preregistration

**Frozen 2026-10-03, before `src/135` existed.** Study AQ.

## 1. The asymmetry study AP left

AP tuned `C` for the **embedding** and left it at 1.0 for **composition**, so two figures in § 6 are
now comparisons of a tuned model against an untuned one:

| | fixed `C = 1.0` on both | tuned embedding, untuned composition |
|---|---:|---:|
| **AK's increment** = E − C | **+7.19 / +7.53pp** | +21.14 / +12.66pp ⚠️ unfair |
| **AO-2** = (C+E) − E | **−0.16 / −0.03pp** | not computed |

🔒 **The +7pp figures are two untuned models and are at least internally consistent**; the +21pp figure
is not a statement anyone should quote. This study tunes **every** side with the same rule.

## 2. The measurement

`src/133`'s selection rule, unchanged and applied to each feature set separately: for each held-out
category, `C` is chosen on a member split of the **other thirteen** plus train negatives, at 5 inner
seeds, over `C ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100}`; the selected value is then scored on the
held-out category at 30 seeds. 🔴 **No number from the scored category reaches any selection.**

Three feature sets, each tuned: **composition** (21), **embedding** (1,280 / 480), **concatenated**
(1,301 / 501). The embedding's tuned figures already exist in `results/estimation_repair.json` and are
reused rather than recomputed.

## 3. Frozen bands

Floors **and** ceilings, on **both** arms. Bonferroni α = 0.05/2.

| | | |
|---|---|---|
| **AQ-1** tuned E − tuned C | **≥ +6pp** ⇒ the increment survives a fair comparison | **≤ +2pp** ⇒ it does not |
| **AQ-2** tuned (C+E) − tuned E | **&#124;·&#124; ≤ 2pp** ⇒ AO-2's null survives tuning | **≥ +5pp** ⇒ composition adds something once both are tuned |

🔒 **A verdict needs both arms in the same column.** Between ⇒ reported as a quantity.

## 4. Predictions

**P1 (derived — composition has 21 features against the embedding's 1,280, so it has far less capacity
to over-fit and far less to gain from shrinkage).** Tuned composition gains **less than +5pp** over its
own `C = 1.0` value of 34.25%, against the embedding's **+13.96pp**.

**P2 (derived — from P1).** AQ-1 **exceeds** the untuned +7.19pp on 650M: 🔑 **the fixed-`C` comparison
was understating the embedding's advantage**, not overstating it.

**P3 (derived).** AQ-2 stays inside ±2pp — adding twenty-one columns to 1,280 should not help once both
are tuned, as it did not when both were untuned.

**P4 (reasoned).** The `C` selected for composition is **larger** than the 0.01 selected for the
embedding, because less capacity needs less shrinkage.

⚠️ **Every one of study AP's five predictions failed, four of them mechanism-derived**, and entry 117
recorded that mechanism-derived predictions are not safer than any other kind. 🔑 **P1 to P4 here are
mechanism-derived too, and are written in that knowledge.**

🔒 **The record is computed, not carried**: `results/prediction_record.json`, currently **derived 10 of
22, reasoned 3 of 9**, and claim 147 requires § 7 to match it.

## 5. What each outcome changes

- **AQ-1 ≥ +6pp** ⇒ § 6 restates the increment at a fair comparison and the +21pp figure is retired.
- **AQ-1 ≤ +2pp** ⇒ 🔴 the embedding's advantage over twenty-one numbers was an artifact of leaving
  composition untuned, and § 8's practical number collapses.
- **AQ-2 ≥ +5pp** ⇒ composition carries something the embedding lacks **once both are fitted properly**,
  which is the claim studies AJ, AK and AM circled and study AO ruled out at `C = 1.0` only.
