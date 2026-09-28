# Preregistration — is the localization gradient partly a length gradient?

**Frozen 2026-09-29, after a declared mechanism check and before any length-adjusted rate has been
computed.** One primary, one second-arm replication, one diagnostic. Append-only.

---

## 0. The number that moved against study E

Study L cleared study E's coverage ceiling and its verdict survived at `R` = **5.18** and **5.86**.
🔴 **One number moved the other way**: the 1:1 length-matched ratio came in at **2.623** on 650M — in
the *partial* band — against study E's 4.382.

🔑 **And study E's confound rule cannot see why**, because it asks the wrong question. It measures
whether **length separates the strata** — AUROC **0.562**, comfortably inside its 0.65 tolerance — and
never measures whether **length predicts the flag**. ⚠️ **A variable can separate two groups weakly and
still predict the outcome strongly**, and then matching on it moves the estimate a great deal. That is
the gap this study closes.

## 🔒 0.1 Declared mechanism check, run before freezing

Matching keeps all 657 extracellular proteins and selects 657 of the 2,739 intracellular ones, so
**only the denominator changes**:

| | mean length | median |
|---|---:|---:|
| extracellular | 529 | 476 |
| intracellular, all 2,739 | 461 | 451 |
| ⭐ intracellular, the 657 matching selects | **524** | **476** |

🔒 **So matching selects longer intracellular proteins**, and the matched ratio falls only if the flag
rate **rises with length inside a stratum**. Nothing about the flag's relationship to length has been
looked at.

---

## 1. Design

⚠️ **No new probe and no new inference.** Study G's clean fold, 30 seeds, study L's strata, the same
pool rows. **Only the analysis changes.**

🔒 **Length deciles are cut on the pooled extracellular ∪ intracellular population**, so both strata
are sliced at identical boundaries and no decile is defined by the group it will be compared in.

---

## 2. Primary test, frozen bands — study E's, unchanged

**M-1**: the **Mantel–Haenszel** risk ratio of extracellular to intracellular flag rates, pooled across
the ten length deciles, ESM-2 650M. **M-2**: the same on 35M.

| band | reading |
|---|---|
| **R_MH ≤ 1.5** | 🟢 the gradient was length |
| **1.5 < R_MH < 3.0** | ⚠️ partial — localization survives adjustment but not as a *major* driver |
| **R_MH ≥ 3.0** | 🔴 localization is a major driver independent of length — study E's verdict, adjusted |

🔴 **What the middle and lower bands oblige, written before the number exists**: if `R_MH < 3.0` on
**both** arms, then **"the probe is substantially a localization detector" is not supportable as
stated** — it is a localization *and length* detector, and every document carrying study E's verdict
must be restated in the same commit, on the terms entry 53's obligation was executed.

🔒 **What no outcome changes**: the *operational* claim. A benign secreted protein really is flagged at
~24% against a nominal 5%, whatever explains it. **Length adjustment speaks to the mechanism, not to
the deployment fact**, and that distinction is held in every statement of the result.

## 3. Diagnostic — the number study E never measured

**M-3**: AUROC of **length for the flag**, computed overall and **within each stratum**. 🔑 If this is
high while the strata-separating AUROC is 0.562, the frozen confound rule in study E was measuring the
wrong quantity, and that is worth recording independently of how M-1 lands.

## 4. Frozen predictions

- **M-3: length→flag AUROC ≥ 0.70 overall**, and ≥ 0.65 within the intracellular stratum. This is what
  would explain the matching effect.
- **M-1: R_MH between 2.5 and 4.5 on 650M** — below the unadjusted 5.18, and I expect it to land
  **above 3.0**, so the verdict holds in adjusted form.
- **The arms may disagree**, since 35M's matched ratio (3.553) already sits above 650M's (2.623).
- ⚠️ 🔒 **Checkable against a wrong answer**: if `R_MH` comes back **equal to the unadjusted R to three
  decimals**, the stratification did not happen and that is a bug, not a finding.

## 5. Floors

🔒 A decile contributes only if it holds **≥ 20** proteins in **each** stratum; dropped deciles are
counted and reported. If fewer than **6** of the 10 deciles contribute, M-1 is indicative rather than a
verdict.

---

## Results, 2026-09-29

`src/93_length_adjustment.py`, study G's clean fold, study L's strata, 30 seeds, both arms.
`results/length_adjustment.json`, `results/length_adjustment_esm2_35M.json`.

Nine of ten deciles contribute; decile 5 (452–476) is dropped for holding 15 extracellular proteins
against a floor of 20.

| | ESM-2 **650M** | ESM-2 **35M** |
|---|---:|---:|
| crude `R` | 5.182 [5.034, 5.330] | 5.855 [5.500, 6.211] |
| ⭐ **M-1 / M-2, Mantel–Haenszel `R`** | **3.984 [3.874, 4.094]** | **4.704 [4.438, 4.970]** |
| attenuation of the excess risk | 28.6% | 23.7% |

### 🔴 The verdict survives adjustment, on both arms

**R_MH = 3.984 and 4.704**, both above the frozen 3.0. 🔒 **"Localization is a major driver" holds
independent of length**, and the obligation in § 2 is not triggered. Adjustment removes about a quarter
of the excess risk and leaves the rest.

### ⭐ And the reconciliation, which is the substance of this study

The ratio is **not constant across length** — it runs from **1.40 to 11.30** on 650M:

| length | intracellular *n* | rate | extracellular *n* | rate | ratio |
|---|---:|---:|---:|---:|---:|
| 0–250 | 274 | 14.15% | 86 | 46.12% | 3.26 |
| 250–350 | 478 | 3.76% | 104 | 30.35% | 8.08 |
| 350–450 | 605 | 1.91% | 123 | 21.60% | **11.30** |
| 450–550 | 849 | 1.69% | 77 | 17.10% | 10.13 |
| 550–700 | 236 | 3.59% | 72 | 17.96% | 5.01 |
| 700+ | 297 | 12.18% | 195 | 17.08% | **1.40** |

🔑 **So the three estimates are not in conflict; they weight length differently.** Mantel–Haenszel
weights by cell size and lands at 3.98. 1:1 matching reproduces the *extracellular* length
distribution, which is concentrated at both extremes — 86 proteins under 250 and 195 over 700 — and
those are exactly the bands where the ratio is lowest, so it lands at 2.62. **Both are correct answers
to slightly different questions**, and the crude 5.18 is a third weighting again.

🔒 **What does not depend on the weighting**: the ratio exceeds **1.0 in every band on both arms**, and
on 35M its minimum is **2.54**. **There is no length at which the localization gradient disappears.**

### 🔴 The flag rate is U-shaped in length, which is why § 3's diagnostic failed

Inside **both** strata the rate falls and then rises: intracellular **14.15% → 1.69% → 12.18%**,
extracellular **46.12% → 17.10% → 17.08%**.

§ 4 predicted the length→flag AUROC would be **≥ 0.70**. It is **0.437** on 650M and **0.338** on 35M —
🔴 **below 0.5, and badly wrong in both magnitude and direction.** ⚠️ **The reason is that AUROC is a
monotone summary and this relationship is not monotone**: short and long proteins are both flagged
more, so the two halves cancel and AUROC reports "no relationship" for a strong U.

🔑 **That is the same shape of error as the one this study was written to fix.** Study E's confound
rule asked whether length separates the *strata* (0.562) rather than whether it predicts the *flag*;
§ 3 then asked the right question with an instrument that cannot see the answer. **The length-band
table above is the diagnostic that works, and it is now in the artifact.**

### 🔒 Scoring the frozen predictions

| | frozen | actual | |
|---|---|---|---|
| M-1 `R_MH` 2.5 – 4.5 on 650M, above 3.0 | | **3.984** | ✅ |
| the arms may disagree | | 3.984 and 4.704, same band | ❌ they agree |
| M-3 AUROC ≥ 0.70 | | **0.437 / 0.338** | ❌ wrong size *and* sign |
| `R_MH` equal to crude ⇒ a bug | | 3.984 vs 5.182 | ✅ guard not triggered |

### ⚠️ One line of this study's own output asserted something its neighbour refuted

The band table printed **"the ratio exceeds 1.5 in every band"** beside a computed minimum of **1.40**.
🔒 Fixed to print the number rather than a claim about it. **A hardcoded assertion standing next to a
figure that can contradict it is an assertion that will eventually be wrong.**
