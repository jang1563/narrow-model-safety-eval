# Study AC — honest intervals for the headline contrasts

**Frozen 2026-10-01, before any interval was recomputed.** Code: `src/115_headline_intervals.py`.
Artifacts: `results/headline_intervals{,_esm2_35M}.json`.

---

## 1. Why this exists

Entry 81 established that `stat()` reports **mean ± 1.96 · SE over probe seeds on a fixed row set** — the
location of a seed mean, not the uncertainty of the quantity. Entry 84 enumerated **eleven** published
intervals built that way and said each needs its study re-run with rows retained. 🔴 **They were listed
and not fixed.** This study fixes the four that carry the paper's argument:

| | published | rows |
|---|---|---|
| localization `R` | **5.38× [5.13, 5.63]** | 652 extracellular / 1,666 intracellular |
| registration `C/A` | **2.418 [2.339, 2.497]** and **2.617 [2.495, 2.740]** | 264 VFDB / ~171 benign |
| out-of-sample FPR | **7.87% [7.57, 8.16]**, **5.98% [5.73, 6.23]** | 8,258 |
| VFDB flag rate | **46.70% [45.93, 47.47]** | 4,218 |

---

## 2. What is computed

A **hierarchical bootstrap**: resample a (seed, fold) draw, then resample **rows with replacement inside
each stratum**, exactly as `src/110` does for the erasure contrast. Reported beside the seed-only interval
so the difference is visible, 30 seeds, both arms, B = 20,000.

🔒 **The point estimates do not move.** Only the intervals do, and the question is whether any **verdict**
moves with them.

| quantity | its frozen band | the verdict at risk |
|---|---|---|
| `R` | ≥ 3.0 | "localization is a major driver" |
| `C/A` | ≥ 2.0 supported, ≥ 1.5 partial | study K's **supported** |

---

## 3. Predictions

🔒 **`R` stays above 3.0 on both arms.** Derived: 652 and 1,666 rows give binomial relative standard
errors near **4%** and **7%** at the observed rates, so the ratio's interval widens by roughly a factor of
two from a base that clears 3.0 by 2.4 points.

⚠️ **`C/A` is the one at risk, and I do not predict which side it lands.** Its benign denominator is
about **171** rows and its numerator **264**; the point estimate 2.418 clears the 2.0 band by 0.418, and a
two-fold widening of a ±0.08 interval does not obviously cover that — but the same contrast **already fell
to 1.59 and 1.63 on the independent pool**, where the verdict moved from supported to partial. 🔒 **If the
honest interval crosses 2.0, study K's verdict becomes partial on its own panel too**, and § 6 says so.

🔒 **The two large-*n* rates widen by about 2× and no verdict depends on them** — they are reported
quantities, not banded ones.

---

## 4. Obligations

- ❌ **If `R`'s interval crosses 3.0**, the localization finding is stated as partial everywhere it appears.
- ❌ **If `C/A`'s crosses 2.0**, study K's supported verdict becomes partial and the abstract says so.
- 🔒 **Whatever happens, both intervals are printed side by side** — seed-only and row-inclusive — because
  the gap is the finding for the eleven intervals this does not reach.
- ⚠️ **The remaining seven** — the ratio of ratios, the two Mantel–Haenszel ratios and the deduplicated
  rates — stay on seed intervals and are labelled as such until they are re-run.

---

## 5. Results — the adverse finding survives the honest interval and the positive one does not

**Run 2026-10-01**, 30 seeds, B = 20,000, both arms.
`results/headline_intervals{,_esm2_35M}.json`.

| | point | seed interval | **row + seed (honest)** | wider by |
|---|---:|---|---|---:|
| **650M** `R` | 5.529 | [5.325, 5.734] | **[3.714, 8.534]** | **11.8×** |
| **650M** `C/A` | 2.418 | [2.339, 2.497] | **[1.904, 3.252]** | **8.5×** |
| 650M pool FPR | 7.61% | [7.21, 8.01] | [5.74, 10.45] | 5.9× |
| 650M VFDB rate | 76.95% | [75.80, 78.11] | [70.03, 83.57] | 5.9× |
| **35M** `R` | 5.425 | [5.100, 5.750] | **[3.242, 9.839]** | **10.1×** |
| **35M** `C/A` | 2.617 | [2.495, 2.740] | **[1.931, 3.938]** | **8.2×** |
| 35M pool FPR | 5.20% | [4.90, 5.49] | [3.48, 6.69] | 5.4× |
| 35M VFDB rate | 51.49% | [49.79, 53.19] | [41.56, 60.38] | 5.5× |

### 🟢 `R` clears its band on both arms

Honest lower bounds **3.714** and **3.242**, against a band frozen at **≥ 3.0**. ✅ **The prediction held**,
and it was derived: 652 and 1,666 rows give relative standard errors near 4% and 7%, so a roughly two-fold
widening from a base clearing 3.0 by 2.4 points was never going to cross it. ⚠️ **The widening is larger
than predicted — 10 to 12× rather than 2×** — and the verdict survives anyway.

### 🔴 `C/A` crosses 2.0 on both arms, so study K's verdict is partial

Honest lower bounds **1.904** and **1.931**, against **≥ 2.0 supported** and **≥ 1.5 partial**.
🔒 **Obligation § 4 fires: study K's supported verdict becomes partial on its own panel**, and § 5 of the
manuscript says so.

🔑 **This is the second time the same contrast has moved the same way.** The independent pool put it at
**1.59** and **1.63** — partial — and the honest interval now puts the original panel there too. ⚠️ **§ 4
declined to predict which side it would land on**, and the reason given was exactly this: the pool had
already moved it once.

### 🔒 What this does and does not change

**Does**: the registration contrast is **partial** everywhere it is stated, on both arms, on both panels.
**Does not**: any point estimate. 🔑 **And the asymmetry is the result** — the same treatment applied to
both headline findings leaves the **adverse** one (localization) inside its band and moves the
**positive** one (registration) out of its. ⚠️ **Seven of the eleven intervals entry 84 enumerated are
still seed intervals**, and they are labelled as such until they are re-run.
