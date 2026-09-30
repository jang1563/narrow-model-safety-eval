# Study Z — erase the localization axis with a guarantee, and measure the tail separately

**Frozen 2026-09-30, before any of Z-0 – Z-4 was computed.** Code: `src/109_leace_erasure.py`.
Artifacts: `results/leace_erasure.json`, `results/leace_erasure_esm2_35M.json`.

---

## 1. Why this study exists

Study Y ran **INLP** ([Ravfogel et al., ACL 2020](https://aclanthology.org/2020.acl-main.647/)) and it did
not remove localization: at up to **160 of 1,280** directions a classifier refitted on unseen rows still
reached **AUROC 0.698**, and `R` stayed **above 3.35**. 🔴 **Entry 78 retired the conclusion drawn from
that**, because INLP's under-erasure is published and **LEACE** ([Belrose et al., NeurIPS
2023](https://arxiv.org/abs/2306.03819)) is closed-form and **provably** prevents every linear classifier
from recovering a binary concept.

🔑 **So study Y's question was never answered, and the obstacle was the method.** Study Z removes the
obstacle. It has no iterations and **no stopping rule** — the thing that broke study Y's gate.

---

## 2. The construction, frozen

For a binary concept the LEACE eraser is a **rank-1 oblique** map:

> `r(x) = x − A (x − μ)`, with `A = Σ^½ u uᵀ Σ^−½` and `u ∝ Σ^−½ Σ_XZ`

- **Σ (covariance) is estimated on the whole pool, 8,259 rows, and uses no labels.** 🔒 Covariance is a
  label-free quantity, so estimating it off-fold leaks nothing about localization, and n ≫ d makes it far
  better conditioned than a fold estimate (1,663 rows against 1,280 dimensions) — which is where
  whitening would otherwise amplify noise.
- **Σ_XZ (the concept direction) comes from the direction-fitting fold only**, under the same 50/50
  cross-fitting and the same measurement rows as study Y, so every number is comparable to Y's.
- ⚠️ **Because Σ and Σ_XZ come from different row sets, the algebraic exactness is not assumed. It is
  measured**: § 3.

---

## 3. Z-0 — the gate, in two halves

**Half one, in-fold, which the construction should make exact.** The residual correlation between the
concept label and the *most favourable* linear score of the erased representation, on the fitting fold.
🔒 **Frozen tolerance: |corr| ≤ 0.02, and the achieved value is reported whatever it is.** If it exceeds
that, the pool-covariance variant is abandoned for a fold-covariance one, and **both are reported.**

**Half two, out-of-fold, which no theorem covers.** Study Y's Y-0b, unchanged: a fresh localization
classifier fitted on the residual of the **measurement** rows, held-out AUROC.

| band | reading |
|---|---|
| **≤ 0.55** | the erasure generalizes |
| 0.55 – 0.60 | marginal, and Z-1/Z-2 are reported as indicative |
| **> 0.60** | ❌ the erasure does not generalize either |

🔒 **Prediction: in-fold |corr| ≤ 0.02; out-of-fold AUROC ≤ 0.55.** Derived: with equal class-conditional
means, **every** linear score has equal expectations across classes, so its AUROC is 0.5 for symmetric
distributions — the only slippage is estimating the mean difference on 619 against 2,708 rows.

---

## 4. Z-1 — the question study Y could not reach

`R` = extracellular over intracellular benign flag rate, after erasure. **Study Y's bands, unchanged**:
≤ 1.5 removed, ≤ 3.0 partly, > 3.0 not removed. Unprojected it is **4.86 / 6.38** (650M, per fold).

🔒 **Prediction: `R` > 1.5, and quite possibly > 3.0.** ⚠️ **This is derived from three measurements, not
reasoned:**

1. **Entry 76**: a chance AUROC and a parity tail-rate are different claims — established by study X,
   where the two dissociated in the opposite direction.
2. **Study X**: at equal means, *dispersion* drives tail behaviour — VFDB standardized ‖z‖ median **39.87**
   against benign **32.52**, per-dimension SD **1.106** against **0.931**.
3. **Study O**: the availability gap was carried by the **magnitude** factor as much as the direction one,
   so a mean-matching erasure has no reason to flatten a tail ratio.

🔑 **LEACE equalizes means. The flag is a 95th-percentile tail event. Those are not the same object**, and
if `R` survives a *provably* mean-erased representation, that is the strongest form of this paper's
localization finding rather than a weaker one.

---

## 5. Z-2 — and the answer to Y-2, if Z-0 holds

CA = VFDB-extracellular over benign-extracellular, and ρ = (CA − 1)/(CA_original − 1) against **2.418**
(650M) and **2.617** (35M). **Study Y's bands, unchanged**: ρ ≥ 0.70 survives, ≥ 0.30 partial, below that
the contrast was largely the axis.

🔒 **Prediction: ρ ≥ 0.30. The ≥ 0.70 band is declined**, for the reason study Y § 5 gave and which still
holds: study K's contrast is already within-stratum, but VFDB proteins may sit further along the axis than
benign secreted ones, and nothing in the record settles it.

🔒 **If Z-0 holds on both halves, this answers Y-2** — the first time the question is answerable — and the
answer is reported whichever way it goes.

---

## 6. Z-3 — the control, same rank, same construction

A **random** unit vector in place of `Σ^−½ Σ_XZ`, everything else identical, so the control is a rank-1
oblique projection of the same family. Band: **ρ_random ≥ 0.85**, else Z-2 is capacity-confounded.

🔒 **Prediction: ρ_random ≥ 0.85, and `R_random` stays near its unprojected 4.86–6.38.** Derived: study N
put a 2.1%-along-w perturbation at no cost to any conclusion, and this removes **one** direction of 1,280.

---

## 7. Z-4 — is the survivor hazard-shaped?

ρ for study Q's **non-hazardous** categories, its **partly-hazardous** ones, and **Motility** alone. Band:
**|Δ| ≤ 0.20** means no more hazard-shaped than what went in. 🔒 **Prediction: |Δ| ≤ 0.20**, from study Q's
unprojected 2.410 against 2.356. ⚠️ **Motility's cell is 28 rows, under the frozen floor of 30, and is
reported as indicative in every arm.**

---

## 8. Obligations

- 🔒 **If Z-0's in-fold half misses 0.02**, the pool-covariance choice is wrong; the fold-covariance
  variant runs and **both** are reported.
- 🔒 **If Z-0's out-of-fold half exceeds 0.60**, Z-1/Z-2/Z-4 are indicative and **Y-2 stays open** — and
  the entry says the guarantee is in-sample only, which is a real limit of the method rather than a
  failure of this run.
- ❌ **If ρ < 0.30 with Z-0 held and the control clean**, the project's positive result narrows again and
  `paper/MANUSCRIPT.md` § 6 and the abstract say so.
- 🔒 **If ρ ≥ 0.70 with `R` collapsed**, that is a **positive** result for the probe and is reported with
  equal prominence. ⚠️ It would still not make the probe a hazard detector: study Q's Motility finding is
  untouched.
- 🔒 **`R` and the AUROC are reported side by side at every step**, because the whole point of § 4 is that
  they can disagree, and a single summary number would hide it.
- ⚠️ **Nothing here is a mitigation.** A provable erasure that leaves `R` high is evidence *against*
  erasure as a fix, and the claims audit keeps the forbid that bans calling it one.

---

## 9. Results — the guarantee is exact in-sample and transfers only partly

**Run 2026-09-30**, 30 seeds, two cross-fitted folds. `results/leace_erasure{,_esm2_35M,_nowhiten}.json`.

| | 650M whitened | 650M **no whitening** | 35M whitened |
|---|---:|---:|---:|
| **Z-0** in-fold \|corr\| (tol 0.02) | **3.1e-06** ✅ | **1.5e-08** ✅ | **3.5e-06** ✅ |
| the same check, unerased control | 0.580 / 0.572 | 0.580 / 0.572 | 0.529 / 0.526 |
| **Z-0 out-of-fold \|corr\|** | **0.400 / 0.354** | 0.524 / 0.429 | **0.262 / 0.247** |
| out-of-fold control | 0.562 / 0.597 | 0.562 / 0.597 | 0.517 / 0.525 |
| **Z-0** unseen-row AUROC | 0.809 / 0.879 | 0.826 / 0.846 | 0.803 / 0.856 |
| unseen-row control | 0.895 / 0.980 | 0.896 / 0.980 | 0.901 / 0.987 |
| Z-0 band | ❌ does not generalize | ❌ | ❌ |
| **Z-1** `R` (control) | **4.791** (5.695) | **4.546** (5.641) | **4.954** (5.808) |
| **Z-2** ρ | 1.162 | 1.185 | 1.103 |
| **Z-3** control ρ | 1.002 ✅ | 1.007 ✅ | 1.010 ✅ |
| **Z-4** Δ | +0.051 | +0.056 | −0.190 |

### 🟢 The guarantee is real, and the check can fail

In-fold residual correlation is **3.1e-06** whitened and **1.5e-08** unwhitened — exact to numerical
precision — while the same check on the unerased representation returns **0.53–0.58**. 🔒 **So Z-0's
in-fold half is a working measurement, not a tautology.**

### 🔴 And it transfers only partly, which is the finding

Out-of-fold, erasure removes **29%** and **41%** of the mean-based correlation on 650M
(0.562→0.400, 0.597→0.354) and **49%** and **53%** on 35M. **Unseen-row AUROC falls 0.90→0.81 and
0.98→0.88.** ⚠️ **A provably exact linear erasure leaves a classifier refitted on unseen rows at 0.81.**

🔑 **The decisive diagnostic is the out-of-fold correlation, and it rules out the interesting
explanation.** If the residual separation were *not* mean-based — classes with equal means and different
covariances — out-of-fold \|corr\| would be near zero while AUROC stayed high. **It is 0.25–0.52.** So the
mean difference is still there out-of-fold: **the estimate did not transfer.**

### 🔴 Whitening is not the amplifier, and the direction of the effect names the cause

I expected whitening to amplify estimation error in low-variance directions. ❌ **Wrong**: whitened
transfers **better** than unwhitened (0.400/0.354 against 0.524/0.429). 🔑 **What does track is dimension**
— 35M (**480** dims) erases about half the out-of-fold correlation, 650M (**1,280** dims) about a third,
with the same 619/2,708 labels.

⚠️ **That points at estimating the concept direction, not at the erasure algorithm** — and it supersedes
*both* earlier diagnoses: entry 77's "redundantly encoded" and entry 78's "INLP's known under-erasure".
🔴 **It also rests on two dimension points, which is the inference that has missed repeatedly in this
session.** § 10 tests it before it is asserted.

### 🟢 Z-1 is the result that does not depend on any of that

`R` = **4.79 / 4.55 / 4.95** against controls of **5.70 / 5.64 / 5.81**, so the ablation costs some of the
gradient and **nowhere near all of it**. ⭐ **In-sample the representation is provably free of a linear
localization signal, and the flag-rate ratio is still above 4.5.** 🔑 **That is the dissociation § 4
predicted from three measurements, and it held.**

⚠️ **Z-2's ρ of 1.10–1.19 is not an answer to Y-2**, because Z-0's out-of-fold half failed: a contrast that
survives an erasure which did not transfer has not survived an erasure. 🔒 **Y-2 stays open**, and the
artifacts carry `answers_Y2: false`.

### 🔒 Predictions

| | frozen | actual | |
|---|---|---|---|
| Z-0 in-fold \|corr\| ≤ 0.02 | | 3.1e-06 / 1.5e-08 / 3.5e-06 | ✅ |
| Z-0 out-of-fold AUROC ≤ 0.55 | | **0.81 – 0.88** | ❌ |
| **Z-1 `R` > 1.5, "quite possibly > 3.0"** | | **4.55 – 4.95** | ✅ **and beyond the hedge** |
| Z-2 ρ ≥ 0.30 | | 1.10 – 1.19 | ⚠️ inside, uninterpretable |
| Z-3 ρ_random ≥ 0.85 | | 1.002 – 1.010 | ✅ |
| Z-4 \|Δ\| ≤ 0.20 | | +0.051, +0.056, −0.190 | ✅ all three, 35M barely |

🔑 **Z-1 was derived from three measurements and held with room to spare. Z-0's second half was derived
from a theorem — and the theorem is about the sample it was fitted on**, which § 3 said in its own words
("which no theorem covers") and then predicted ≤ 0.55 anyway. ⚠️ **Eighth time: the prediction that came
from measurements held, the one that came from an argument did not.**

---

## 10. Amendment 1 — frozen 2026-09-30, before the label curve was computed

🔒 **The claim "it is the concept-direction estimate, not the algorithm" rests on two dimension points and
is not asserted until tested.** If the cause is estimation error in the mean difference, then **holding
dimension fixed and varying the number of labels must move the out-of-fold correlation monotonically.**

- **Grid**: n_labels ∈ {200, 400, 800, 1600, all}, subsampled from the fitting fold at a fixed seed,
  class balance preserved. Both arms. Whitened. 10 seeds for `R`.
- **Reported at each n**: out-of-fold \|corr\|, unseen-row AUROC, and `R`.
- **Read-off**, frozen: Spearman ρ between n and out-of-fold \|corr\| across the grid.

| band | reading |
|---|---|
| **ρ ≤ −0.80 on both arms** | the obstacle is the concept-direction estimate |
| −0.80 < ρ ≤ −0.40 | partly the estimate |
| **ρ > −0.40** | ❌ not an estimation-error story; the earlier diagnoses are not superseded |

🔒 **Prediction: ρ ≤ −0.80 on both arms, and the curve does not reach zero at the largest n.** Derived from
the two facts in hand: the 480-dimension arm transfers better than the 1,280-dimension arm at equal labels,
and the unwhitened variant — which has *less* noise amplification and *more* bias — transfers worse, so the
binding constraint behaves like variance in the estimate rather than like a property of the geometry.

⚠️ **If the curve is flat, the estimation-error account is wrong and this document says so** — and the
reason it would matter is that a flat curve would mean more labels cannot fix it, which is a statement
about the representation after all.


---

## 11. Amendment 1 Results — the estimate is part of it, and there is a floor

**Run 2026-09-30**, `results/leace_erasure_label_curve{,_esm2_35M}.json`, 10 seeds for `R`.

Out-of-fold \|corr\| per fold, and `R`, against the number of labels used for the concept direction:

| n labels | 200 | 400 | 800 | 1,600 | 1,664 (all) |
|---|---:|---:|---:|---:|---:|
| **650M** out-of-fold \|corr\| | 0.493 / 0.519 | 0.449 / 0.475 | 0.381 / 0.388 | 0.382 / 0.344 | 0.400 / 0.354 |
| **650M** unseen AUROC | 0.929 / 0.961 | 0.941 / 0.966 | 0.895 / 0.877 | 0.871 / 0.824 | 0.809 / 0.879 |
| **650M** `R` | 5.47 | 5.48 | 5.41 | **4.57** | 4.69 |
| **35M** out-of-fold \|corr\| | 0.395 / 0.412 | 0.328 / 0.341 | 0.280 / 0.268 | 0.270 / 0.239 | 0.262 / 0.247 |
| **35M** unseen AUROC | 0.941 / 0.914 | 0.924 / 0.929 | 0.852 / 0.857 | 0.866 / 0.839 | 0.803 / 0.856 |
| **35M** `R` | 5.15 | 4.67 | 5.07 | 4.90 | 4.67 |

**Spearman(n, out-of-fold \|corr\|)**: 650M **−0.6 / −0.9**, 35M **−1.0 / −0.9**.

### 🟡 The frozen band is met on one arm, not both

❌ **The prediction was ρ ≤ −0.80 on both arms.** 35M gives −1.0 / −0.9 → *the obstacle is the
concept-direction estimate*; 650M's worse fold gives **−0.6** → *partly the estimate*. 🔒 **So the
estimation-error account is supported on the 480-dimension arm and only partly on the 1,280-dimension
one**, which is the direction the § 10 reasoning predicted but not the strength.

### 🔴 And the curves plateau, which no earlier diagnosis anticipated

More labels help, then stop helping. 650M runs **0.493 → 0.449 → 0.381** and then **flattens and ticks
back up** (0.382, 0.400); 35M runs **0.395 → 0.328 → 0.280** and settles near **0.26**. ✅ **The frozen
prediction that the curve would not reach zero held** — but the shape is a **floor**, not a slow descent.

🔑 **That floor is the reconciliation of three diagnoses, none of which was right alone:**

1. Entry 77 said **redundantly encoded**. There *is* a floor, so something of this survives — but it is
   about the mean-difference direction being unestimable out-of-sample, **not** about linear removability
   in principle, which LEACE settles in-sample at **3e-06**.
2. Entry 78 said **INLP's known under-erasure**. True and published, but **not the binding cause**: ⭐
   **INLP's rank-80 ablation reduced unseen AUROC further (0.698) than LEACE's provable rank-1 did
   (0.809).** A guarantee bought exactness on the fitted sample, not transfer.
3. This study said **the concept-direction estimate**. Supported on 35M, partial on 650M, with a floor.

🔒 **The honest synthesis is that out-of-sample erasure is governed by rank and by estimation quality, and
a guarantee is neither.** ⚠️ **It is in-sample by construction, and that is a property of the theorem, not
a defect of the implementation** — § 3 said so and then predicted otherwise anyway.

### ⭐ And the one number that survived every configuration

`R` is **above 4.5 in all of it**: INLP at k = 0 … 160, LEACE whitened and unwhitened, label counts 200 …
1,664, both arms. Range across the label curve **[4.57, 5.48]** and **[4.67, 5.15]**.

🔒 **The localization gradient does not depend on which diagnosis is right.** ⚠️ **And it is not a
mitigation story in either direction**: an erasure that is provably exact in-sample leaves the flag-rate
ratio above 4.5, so orthogonalization is not a repair, and the reason it is not has now been measured
three ways instead of guessed at once.
