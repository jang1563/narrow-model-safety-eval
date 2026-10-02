# Is the representation's increment compositional? — preregistration

**Frozen 2026-10-02, before `src/131` existed.** Study AN.

## 1. Three arrivals, no test

The same finding has now reached this paper three independent times:

| study | finding |
|---|---|
| **AJ** | composition alone reproduces the 35M arm's Exotoxin recovery to **0.08 points** |
| **AK** | the embedding wins 11 of 14 categories by **+7.19 / +7.53pp**, and **loses 20.14 / 16.75pp** on Exoenzyme |
| **AM** | composition is the **only** arm whose ordering interval excludes chance (0.5553 [0.5204, 0.5919]) |

🔴 **All three are descriptions of a correlation, and none is a test.** The question they raise is
causal in the weak sense that matters here: **is the embedding's +7-point increment information that
composition does not carry, or is it composition re-encoded?**

🔑 **This project has the instrument already.** Studies Y and Z removed the localization axis from the
representation and re-measured — INLP, then LEACE's provably exact rank-1 erasure — and found `R` still
at 4.79. The same move applies to composition.

## 2. The measurement

`src/125`'s fold over DeepVIC's 14 categories, 30 seeds, both arms, with one new feature set:

    X_resid  =  X  −  C (CᵗC)⁻¹ Cᵗ X

where `C` is the **21 composition features** (log length plus the twenty amino-acid frequencies) and
`X` the embedding. 🔒 **Everything linearly predictable from composition is gone from `X_resid`**, and
the residual correlation is reported as the evidence rather than assumed, as study Z did.

⚠️ **The projection is fitted on the negative union only** — no labels, no held-out positives — so it
cannot leak the held-out category. That is weaker than per-fold refitting and it is stated, not hidden.

Reported per arm, against figures already in the record:

| | source |
|---|---|
| recovery, embedding | study AI-2 |
| recovery, composition | study AJ |
| **recovery, residualized embedding** | **this study** |
| increment over composition, before and after | arithmetic |

## 3. Frozen bands

Floors **and** ceilings, on **both** arms. Bonferroni α = 0.05/2.

| | the increment is NOT compositional | the increment IS compositional |
|---|---|---|
| **AN-1** mean recovery of the residualized embedding, over 14 categories | **≥ 38%** (within 4pp of the embedding's 41.4 / 41.8) | **≤ 35%** (at or below composition's 34.2) |
| **AN-2** categories where the residualized embedding still beats composition, of 14 | **≥ 10** | **≤ 7** |

🔒 **A verdict needs both arms in the same column.** Between ⇒ the increment is **partly** compositional,
reported as a quantity.

⚠️ **One thing this design cannot do.** Removing a linear subspace cannot remove information the
embedding encodes **non-linearly** in composition, and the probe is linear, so AN measures what a linear
probe can still use. 🔑 That is the right scope — every result in this paper is from a linear probe —
and it is a limit on the conclusion, stated in advance.

## 4. Predictions

**P1 (derived — from study AK's +7.19/+7.53pp increment and study AM's finding that composition
out-orders both embeddings).** AN-1 lands **between** the bands: the residualized embedding keeps some
of the increment and not all of it, between 35% and 38% mean recovery.

**P2 (derived — from study AK's loss set, where only Exoenzyme is shared across arms).** AN-2 is **≥ 10**
on at least one arm: most categories' increments survive residualization even if the mean falls.

**P3 (reasoned).** The residual correlation between the composition features and the residualized
embedding is below **1e-06**, as study Z's LEACE erasure achieved (3.1e-06 whitened).

⚠️ **The record**: across AH, AI, AJ, AK and AM, derived predictions have held **6 of 11** and reasoned
**2 of 5**. 🔑 **Entry 109 withdrew the direction-versus-magnitude reading as over-fit**, and the
replacement was that predictions made on the same population do better than ones carried across. **P1
and P2 are derived from measurements on this exact benchmark**, so they are the favourable case.

## 5. What each outcome changes

- **Not compositional** ⇒ § 6's increment table stands as a measure of what the representation adds,
  and the three composition arrivals are about *which categories*, not about the whole.
- **Is compositional** ⇒ 🔴 the strongest statement this paper can make about protein language models
  for registration screening becomes: **the part of the signal a linear probe can use is substantially
  compositional**, and § 8's "seven points" is seven points of re-encoding.
- **Partly** ⇒ reported as the fraction of the increment that survives, which is the number a
  practitioner wants and which no study here has produced.
