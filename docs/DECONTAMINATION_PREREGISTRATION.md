# Preregistration — the decontaminated re-run that entries 54 and 55 owe

**Frozen 2026-09-28, after a declared diagnostic count and before the probe has been fitted on a
decontaminated pool.** Append-only.

---

## 0. What this owes and to whom

Entry 54 found **133 exact VFDB sequences in the benign pool** and entry 55 corrected how they are
distributed. Both said a decontaminated re-run was owed, and entry 55 said so in the words *"this
reduces the number, not the reason."* 🔒 **This is that run.**

## 🔒 0.1 Declared diagnostic, taken before freezing

Study F excluded contaminants from its 2×2 cells. ⚠️ **Study E never did**, and the contamination is
**differential across exactly the contrast E measures**:

| stratum | *n* | VFDB contaminants | |
|---|---:|---:|---|
| **extracellular** | 652 | **37** | **5.67%** |
| membrane | 495 | 11 | 2.22% |
| **intracellular** | 1,666 | **20** | **1.20%** |
| unannotated | 3,434 | 40 | 1.16% |

🔴 **E's numerator is nearly five times more contaminated than its denominator**, so E-1's ratio is
inflated by construction and **the size of that inflation has never been measured.** Nothing about the
probe's behaviour on a decontaminated pool has been looked at.

---

## 1. Design

Two effects run in opposite directions and this study separates them:

| effect | mechanism | direction |
|---|---|---|
| **calibration** | contaminants score high, so they raise the 95th-percentile threshold | removing them **raises** every rate |
| **evaluation** | contaminants inside a stratum are near-certain flags | removing them **lowers** that stratum's rate |

🔒 **Frozen procedure**: drop all 133 exact-VFDB pool proteins from the pool entirely — from the
screen's admitted rows (so no contaminant can be fitted or set a threshold) **and** from the evaluated
strata and cells. Fold sizes recomputed by `src/83`'s rule on the decontaminated admitted set. 30
seeds, both arms, and the mean threshold is reported so the mechanism is visible rather than inferred.

⚠️ **`src/84` and `src/85` are not modified.** They are the record of the preregistered analyses;
a flag that changes their output would make that record ambiguous. `src/86` reads the same inputs and
reports both conditions side by side.

---

## 2. Primary tests, frozen bands

| | test | band that preserves the finding |
|---|---|---|
| **G-1** | study E's `R` on a clean pool, both arms | **R ≥ 3.0** — E's adverse verdict holds |
| **G-2** | study F's `RR`, both arms | **RR ∈ [0.67, 1.5]** — the confounds still compose |
| **G-3** | study F's joint share, 650M | **within ±10 pp of 42.2%** |

🔴 **If G-1 falls below 3.0**, study E's verdict was an artifact of contamination, and the
qualifications entry 53 wrote into the manuscript and four preregistrations must be revisited — in the
same commit, on the same terms the original obligation was written on.

## 3. Frozen predictions

- **G-1: R rises**, to **4.5 – 7.0**. The evaluation effect removes more from the numerator than the
  calibration effect adds back, and the worst-case bound already computed (all contaminants flag) puts
  a clean R at **≥ 5.74**, above the observed 5.381.
- **extracellular falls to 17 – 21%**, intracellular to **2.9 – 4.1%**.
- **The mean threshold falls**, since ≈6 of 490 calibration proteins were inflating it.
- **G-2 and G-3 move little** — study F already excluded contaminants from its cells, so only the
  calibration effect reaches them.

## 4. Floors

🔒 Every study F cell must still hold ≥ 150 proteins after decontamination, and E's strata ≥ 300, or
the affected test is reported as indicative rather than as a verdict.

---

## Results, 2026-09-28

`src/86_decontamination_sensitivity.py`, both conditions in one run, 30 seeds, both arms.
`results/decontamination_sensitivity.json`, `results/decontamination_sensitivity_esm2_35M.json`.

18 contaminants leave the admitted set (1,468 → 1,450), so the fold moves 978/490 → 966/484. Floors
met in both conditions on both arms.

| | ESM-2 650M dirty → clean | ESM-2 35M dirty → clean |
|---|---|---|
| threshold | 0.9131 → **0.8953** | 0.9499 → 0.9497 |
| extracellular | 21.75% → **20.90%** | 15.30% → **13.62%** |
| intracellular | 4.13% → 4.15% | 3.08% → 3.23% |
| pool overall | 7.06% → 6.86% | 4.90% → 4.78% |
| ⭐ **R** | 5.381 → **5.074 [4.913, 5.235]** | 5.151 → **4.347 [4.085, 4.609]** |
| **RR** | 1.009 → **0.999** | 1.327 → **1.358** |
| joint share | 42.2% → **46.0%** | 32.8% → 32.2% |

### 🟢 All three primaries hold, on both arms

- **G-1**: clean **R = 5.074** and **4.347**, both far above the 3.0 floor. 🔒 **Study E's adverse
  verdict is not an artifact of contamination**, and the qualifications entry 53 wrote into the
  manuscript and four preregistrations stand as written.
- **G-2**: clean **RR = 0.999** and **1.358**, both inside the frozen band. The confounds still
  compose, and on 650M the decontaminated estimate is almost exactly 1.
- **G-3**: the joint share moves +3.8 pp and −0.6 pp, inside the ±10 pp band.

### ⚠️ But R fell on both arms, so part of the original ratio was contamination

**−0.307 on 650M (5.7% of it) and −0.804 on 35M (15.6%).** 🔑 **The localization finding is real and
was modestly inflated**, and the inflation is larger on the arm where the effect is smaller. The
mechanism is visible in the table: the evaluation effect took 0.85 pp and 1.68 pp off the
extracellular rate while the calibration effect moved intracellular *up* slightly.

### 🔴 The prediction record here is the worst of the four studies, and one prediction was invalid

| | frozen | actual | |
|---|---|---|---|
| G-1 band 4.5 – 7.0 | | 5.074 / **4.347** | ✅ 650M, ❌ 35M |
| "R **rises**" | | fell on both | ❌ |
| extracellular 17 – 21% | | 20.90% / **13.62%** | ✅ 650M, ❌ 35M |
| intracellular 2.9 – 4.1% | | **4.15%** / 3.23% | ❌ 650M, ✅ 35M |
| threshold falls | | fell on both | ✅ |

🔴 **And § 3's supporting bound was derived wrongly.** It said a clean R was **≥ 5.74** "since all
contaminants flagging puts extracellular at ≥ 17.05% and intracellular at ≥ 2.97%". That **divides two
lower bounds, which bounds nothing.** To bound a ratio from below you need the numerator's lower bound
over the denominator's **upper** bound: 17.04% / 4.18% = **≥ 4.08**. Both observations satisfy the
correct bound, and the prediction of an *increase* came directly from the invalid one.

⚠️ **A second defect: the ranges never said which arm they were about**, so "extracellular 17–21%" is
scored generously above as a 650M prediction. A prediction that does not name its population cannot be
scored cleanly, and this one could not.

### 🔴 A coincidence that must not be read as a vindication

The decontaminated joint share on 650M is **46.0%**, and the sum claim 98 forbids — 22.7% + 22.1% — is
**44.8%**. 🔴 **These are unrelated.** The 46.0% is measured on the (pathogen × extracellular) cell
against the pool; the sum is still not a quantity this repository has, and the two landing near each
other on one arm of one condition is arithmetic coincidence — the 35M arm puts the same figure at
**32.2%**.

---

## 🔒 Superseded on one number — 2026-09-28: the provenance factor matched on names

This study decides "pathogen-derived" by testing whether the first two words of the organism string
appear in VFDB's species list. 🔴 **That misses every pathogen UniProt has renamed.** Rebuilding the
factor on canonical species (`docs/TAXID_PROVENANCE_PREREGISTRATION.md`) moves **5.05%** of eligible
proteins and puts **112 genuine pathogen proteins** into the pathogen cells where they belong.

Every frozen criterion here survives the rebuild — the ratio of ratios stays inside its band on both
arms, the joint share moves under 10 pp, and the provenance effect **grows** from 2.63× to 3.47× on
650M. ⚠️ **But the ratio of ratios moves from 1.009 to 0.692 on 650M and 1.327 to 1.133 on 35M**, so
"almost exactly independent" was a property of the defective factor. The numbers above are left as
written, per the append-only rule; the corrected ones are in that document's results.

---

## 🔴 Superseded on the share-of-gap figures — 2026-09-28, entry 61

**542 of the 4,218 VFDB negatives share a sequence with one of the 746 class-axis positives.**
`src/74` screened that set against the **panel** positives, which is what study A1 needed, and nothing
screened it against the **class-axis** positives that came later — so every evaluation of the study-B
probe on this population was scoring the probe's own training data on **12.85%** of the rows.

The VFDB reference drops from **76.95% to 74.09%** on 650M under study G's clean fold (and study D's
own figure from **73.49% to 70.91%**). Every share-of-gap divides by `(vfdb − pool)`, so **all of them
were about 4.3% too small**. 🔒 **No ratio moved** — R, RR, C/A, D/B and F-5 do not involve the
reference and are unchanged to four decimals.

| this document's figures | before | after |
|---|---:|---:|
| clean joint share, 650M | 46.0% | **47.9%** |
| clean joint share, 35M | 32.2% | **33.4%** |

⚠️ The § "coincidence that must not be read as a vindication" section compared 46.0% with a
forbidden sum of 44.8%. On the corrected reference the figures are **47.9%** and **46.6%**, so the
coincidence is *closer*, and the reason it is still a coincidence is unchanged: the 35M arm puts
the same quantity at **33.4%**.
