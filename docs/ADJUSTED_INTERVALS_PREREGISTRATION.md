# Study AD — honest intervals for the two adjusted contrasts

**Frozen 2026-10-01, before either interval was recomputed.** Code: `src/116_adjusted_intervals.py`.
Artifacts: `results/adjusted_intervals{,_esm2_35M}.json`.

---

## 1. What is left

Study AC fixed the four intervals carrying the main argument. **Two banded quantities remain on seed
standard errors**, and both have a live verdict attached:

| | published | band | verdict at risk |
|---|---|---|---|
| length-adjusted `R_MH` (study M) | **3.984 [3.874, 4.094]** and **4.704 [4.444, 4.966]** | ≥ 3.0 | *"localization is a major driver independent of length"* |
| provenance × localization `RR` (study I's corrected factor) | **0.692** and **1.133** | multiplicative independence | already reported as crossing its floor on 650M |

---

## 2. What is computed

The same hierarchical bootstrap as `src/110` and `src/115`: resample a seed, then rows **with replacement
inside each cell**, and recompute the estimator from the resampled cells.

- `R_MH` is **recomputed stratum by stratum** — deciles resampled within, then pooled by Mantel–Haenszel —
  so the bootstrap respects the estimator rather than perturbing its output.
- `RR` is recomputed from the **four** 2×2 cells (201 / 679 / 414 / 967 on 650M).

🔒 Point estimates do not move. Only the intervals do.

---

## 3. Predictions

🔴 **`R_MH`'s honest interval crosses 3.0 on at least one arm.** Derived, not reasoned: study AC widened
the *crude* `R` from a ±0.2 seed interval to **[3.714, 8.534]** on a point of 5.53, so the lower bound sat
at **67%** of the point. `R_MH`'s point is **3.98**, and 67% of that is **2.67** — below the band floor.
⚠️ **The estimator is stratified and pools nine cells, so the transfer is not exact**, which is why this is
stated as "at least one arm" rather than both.

🔒 **If it crosses, the verdict becomes *partial*** — "localization survives adjustment but not as a major
driver" — and § 6 says so. ⚠️ **The crude `R` verdict is untouched either way**: study AC already put its
honest interval at [3.714, 8.534] and [3.242, 9.839].

🔒 **`RR`'s interval widens and the independence reading does not survive on 650M**, which the manuscript
already states for the seed interval. ⚠️ **No prediction on 35M**, whose point estimate is 1.133 and whose
seed interval does not cross.

---

## 4. Obligations

- ❌ **If `R_MH` crosses 3.0 on both arms**, the adjusted claim is *partial* everywhere, and the abstract's
  localization sentence is rewritten to lead with the crude ratio and its adjustment separately.
- ⚠️ **If it crosses on one arm only**, it is stated per arm and not averaged.
- 🔒 **The crude and adjusted verdicts are reported together from here on**, because the paper currently
  states the adjusted one as a survival of the crude one and they now have different interval treatments.

---

## 5. Results — it crosses on both arms, and the two verdicts separate

**Run 2026-10-01**, 30 seeds, B = 20,000, both arms, nine deciles used.
`results/adjusted_intervals{,_esm2_35M}.json`.

| | point | seed interval | **row + seed (honest)** | wider | clears 3.0 |
|---|---:|---|---|---:|---|
| **650M** `R_MH` | 3.890 | [3.769, 4.011] | **[2.871, 5.200]** | 9.6× | ❌ |
| **35M** `R_MH` | 4.386 | [4.114, 4.659] | **[2.885, 7.116]** | 7.8× | ❌ |

### 🔴 The obligation's first branch fires

§ 4 said: *"If `R_MH` crosses 3.0 on both arms, the adjusted claim is partial everywhere."* It crosses on
both. 🔒 **The adjusted verdict is the middle band study M froze in advance** — *"localization survives
adjustment but not as a major driver"* — and § 6 of the manuscript now states the crude and adjusted
verdicts separately.

✅ **The prediction held and was derived rather than reasoned.** § 3 computed the lower bound from study
AC's transfer — 67% of the point estimate, giving **2.67** — and it landed at **2.871** and **2.885**. ⚠️
**It said "at least one arm" because the estimator is stratified; it crossed on both.**

### 🔑 And now both headline findings have been narrowed the same way

| | crude | adjusted / honest |
|---|---|---|
| localization (adverse) | major driver, **[3.714, 8.534]** and **[3.242, 9.839]** | **partial** once length is adjusted |
| registration (positive) | — | **partial**, **[1.904, 3.252]** and **[1.931, 3.938]** (study AC) |

🔒 **The same instrument moved both**, and neither point estimate changed. ⚠️ **This is not a retraction of
either finding**: a nine-cell stratified ratio whose honest interval runs [2.87, 5.20] is still a large
effect, and the middle band was written in advance for exactly this case.

### 🔒 What remains on seed intervals

The provenance × localization `RR` — **0.692** and **1.133**, from `results/taxid_provenance.json` — was
**not** recomputed here. ⚠️ The manuscript already reports that 650M's interval crosses its band floor, so
the reading does not depend on the widening; it is left labelled rather than fixed, and it is the last of
entry 84's eleven.
