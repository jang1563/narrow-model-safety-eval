# Study AB — do the headline contrasts survive the operating point the project calls better?

**Frozen 2026-09-30, before any contrast was recomputed.** Code: `src/114_conformal_contrasts.py`.
Artifacts: `results/conformal_contrasts{,_esm2_35M}.json`.

---

## 1. Why this exists

**48** source files set their operating point with `np.quantile(s_cal, 0.95)`. `src/48`'s own docstring
calls the conformal estimator *"the estimator that was validated and never used"* and says `np.quantile`
*"interpolates instead, which is why it misses nominal from below."* 🔒 The project reports both for the
**out-of-sample FPR** (7.87% against 5.98%) and § 6's text calls conformal **the better estimator**.

🔴 **No contrast in this paper has ever been recomputed at that threshold.** `R`, the VFDB `C/A` contrast
and the LOMO recoveries all sit on the interpolating estimator.

⚠️ **The order statistics say the gap should be small here and large there**, and that is the reason to
measure rather than assume:

| calibration n | conformal k = ⌊(m+1)·0.05⌋ | conformal rank | `np.quantile` rank |
|---|---:|---|---|
| **118** (the published FPR table) | 5 | 5th largest | ≈ **6.9th** largest |
| **490** (every study's fold) | 24 | 24th largest | ≈ **25.5th** largest |

🔑 **So the 7.87 / 5.98 gap comes from n = 118, and the studies calibrate on 490**, where the two
thresholds are 24 against ~25.5 of 490.

---

## 2. What is measured

`R` = extracellular / intracellular benign flag rate (study E's keyword strata) and `C/A` =
VFDB-extracellular / benign-extracellular (study K's exclusions), each under **both** thresholds on the
**same** calibration fold, 30 seeds, both arms. Also the realized specificity of each threshold on the
held-out calibration rows, so the estimator's own behaviour is visible.

| band on the relative change in `R` | reading |
|---|---|
| **< 5%** | the contrasts do not depend on the estimator |
| 5 – 15% | mildly dependent |
| **> 15%** | ❌ the headline contrasts are estimator-dependent |

🔒 **Prediction: < 5% on both arms, and both rates fall.** Derived from the table above — 24 against 25.5
of 490 is a **6%** shift in the tail index, and a ratio of two rates that both move the same way absorbs
most of it.

🔒 **Second prediction: the realized false-positive rate is lower under conformal on both arms**, because
the conformal threshold is the higher of the two by construction at these n.

---

## 3. Obligations

- ❌ **If `R` moves more than 15%**, every flag rate in this project is an artifact of an estimator its own
  code calls wrong, and `paper/MANUSCRIPT.md` § 6 says so.
- 🔒 **Whatever the outcome**, the abstract gains a clause naming which estimator each of its two headline
  numbers uses: **7.87%** is `np.quantile` and **605 alerts** is conformal, which § 10.8 states and § 1
  does not.
- ⚠️ **This is not a claim that conformal is correct for this design.** Both panels and pools are
  non-exchangeable with the calibration set — § 6 says so — and the guarantee is marginal. The point is
  whether the contrasts care.

---

## 4. Results — the contrasts do not care, and the estimator that was never used now has been

**Run 2026-09-30**, 30 seeds, both arms, calibration fold **n = 484**, conformal **k = 24**.
`results/conformal_contrasts{,_esm2_35M}.json`.

| | threshold | realized calibration FPR | `R` | `C/A` |
|---|---:|---:|---:|---:|
| **650M** `np.quantile` | 0.89525 | **5.17%** | 5.529 | **2.418** |
| **650M** conformal | 0.91498 | **4.96%** | 5.574 | 2.477 |
| **35M** `np.quantile` | 0.94967 | **5.17%** | 5.425 | **2.617** |
| **35M** conformal | 0.95642 | **4.96%** | 5.459 | 2.666 |

### 🟢 Both predictions held

Relative change in `R`: **0.81%** and **0.62%**, inside the frozen < 5% band. `C/A` moves **2.43%** and
**1.84%**. 🔒 **The verdicts are unchanged on both arms**, and both rates move in the same direction so the
ratio absorbs the shift — which is the arithmetic § 2 predicted.

🔒 **And the estimator behaves as advertised**: the conformal threshold is the higher of the two on both
arms, and the realized calibration false-positive rate is **4.96%** against `np.quantile`'s **5.17%**.
🔑 **Conformal lands inside the 5% budget and `np.quantile` does not** — the overshoot `src/48`'s docstring
describes, measured here on the studies' own fold.

### ⭐ An unplanned verification

`C/A` under `np.quantile` comes out at **2.418** on 650M and **2.617** on 35M — **study K's published
figures, to three decimals**, from an independently written script on a different code path. 🔒 **Not a
preregistered test and reported as what it is**: a reproduction that had no reason to succeed if either
implementation were wrong.

### 🔑 Why the FPR gap is large and the contrast gap is not

Both follow from calibration size, and the order statistics were tabulated before the run:

| calibration n | conformal rank | `np.quantile` rank | gap |
|---|---|---|---|
| **118** — the published out-of-sample FPR table | 5th largest | ≈ 6.9th | **7.87% vs 5.98%** |
| **484** — every study's fold | 24th largest | ≈ 25.5th | **0.6–2.4%** on the contrasts |

🔒 **So "the probe overshoots its budget by 2.9 points" and "the contrasts do not care which estimator is
used" are both true, and they are the same fact at two sample sizes.**

### 🔒 Obligation discharged

The abstract now names which estimator each of its two headline numbers uses. ⚠️ **It did not**: **7.87%**
is `np.quantile` and **605 alerts** is conformal, so a reader multiplying 7.87% by 9,990 gets **786** and
not 605. 🔒 § 10.8 stated this and § 1 did not.
