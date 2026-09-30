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

## 9. Results

*(to be appended after the run, append-only)*
