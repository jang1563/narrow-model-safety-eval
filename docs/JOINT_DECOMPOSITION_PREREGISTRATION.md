# Preregistration — do provenance and localization overlap, and how much do they buy together?

**Frozen 2026-09-28, after a declared feasibility count and before any probe has been applied to
these cells.** One primary test, one second-arm replication, two secondaries. Append-only.

---

## 0. The gap this exists to close, which three documents now name

Two controls have each been measured at about a quarter of the benign→VFDB gap:

| control | share of the gap |
|---|---:|
| pathogen origin (`docs/PROVENANCE_CONTROL_PREREGISTRATION.md`) | 22.7% |
| localization (`docs/LOCALIZATION_CONTROL_PREREGISTRATION.md`) | 22.1% |

🔴 **They may not be added**, and claim 98's forbid list bans the sum in three forms — pathogen-derived
proteins are themselves enriched for secretion, so the controls overlap by construction. Entry 53, the
manuscript, and the studies B/C/D preregistrations all currently say *"the joint decomposition has not
been run."* **This runs it.**

---

## 🔒 0.1 Declared: a feasibility count was run before this document was written

Cell sizes decide whether this design is possible at all, and three rules in
`docs/EXTERNAL_CLASS_AXIS_PREREGISTRATION.md` had to be amended because they were frozen before anyone
checked they could execute. **So the counts were taken first and are disclosed here**, before the probe
touches any cell:

| | extracellular | intracellular |
|---|---:|---:|
| **VFDB species** | 206 | 609 |
| non-VFDB species | 446 | 1,057 |

Nothing about the probe's behaviour on these cells has been looked at. 🔒 **The floor below is set at
150 rather than the 300 used for the localization control**, because the smallest cell is 206 and the
exact-VFDB-sequence exclusion will shave it — **set with the counts visible and stated as such**, which
is the honest version of a floor that could otherwise be tuned after the fact.

---

## 1. Design

⚠️ **No new probe and no new inference**, for the third study running. Study B's fold exactly as
`src/83` and `src/84` use it: 1,000 train / 500 calibrate, 30 seeds, threshold at the 95th percentile
of calibration scores. Both factors are annotations on proteins that are **already embedded on both
arms**.

**Population**: the pool's test partition, Bacteria and Archaea, localization `extracellular` or
`intracellular` (the frozen rule in `docs/LOCALIZATION_CONTROL_PREREGISTRATION.md` § 1.1). `membrane`
and `unannotated` are reported but are not in the 2×2.

🔒 **Exact VFDB sequence matches are excluded**, as in study D — a positive sitting in a negative set
would inflate exactly the cell the design is about. Study D found four this way.

### 1.1 The provenance factor, and the conservative choice inside it

**Factor P** = the protein's species appears in VFDB (405 species across setA and setB), against a
species that does not. 🔒 Externally defined by VFDB's own species list, at **species** resolution,
the same resolution study D used.

⚠️ **This is conservative and that is deliberate.** *E. coli* K-12 is a benign laboratory strain of a
species that contains pathogens, so it counts as pathogen-derived here. **The effect this measures is
therefore a lower bound on a strain-resolved one**, and is stated that way wherever it appears.

---

## 2. Primary test — is the overlap multiplicative?

**F-1**: the **ratio of ratios**

`RR = (extracellular/intracellular | VFDB species) / (extracellular/intracellular | non-VFDB species)`

mean over 30 seeds, ESM-2 650M. **F-2**: the same on ESM-2 35M. Per the standing rule, **a verdict
needs both arms in the same band.**

| band | reading |
|---|---|
| **RR ∈ [0.67, 1.5]** | 🟢 multiplicatively independent — the two effects compose, and their *log* contributions add |
| **RR < 0.67** | ⚠️ sub-multiplicative — they overlap; each is partly the other |
| **RR > 1.5** | ⚠️ super-multiplicative — being a pathogen protein *amplifies* the localization effect |

🔒 **Whatever the band, the sum 22.7% + 22.1% stays forbidden.** Independence on a *ratio* scale is not
additivity on a *share* scale, and the share the two buy together is measured directly in § 3 rather
than inferred from this test.

---

## 3. The joint share, measured rather than added

**F-3**: the flag rate in the **(VFDB species × extracellular)** cell, against the pool's overall rate
and VFDB's 73.49%, in study D's arithmetic:

`joint share = (rate[pathogen × extracellular] − rate[pool overall]) / (0.7349 − rate[pool overall])`

🔒 **This is the number that replaces the forbidden sum**, and it is the only one this repository will
quote for "how much do both confounds together explain".

---

## 4. Secondaries

- **F-4 — study D's upper-bound claim, tested.** Study D declared 22.08% an **upper bound** on the
  provenance effect because its 106 controls are the panel's *curated hard* negatives. The pool's
  VFDB-species proteins are not curated for difficulty. 🔒 **Prediction: their flag rate is below
  22.08%.** If it is not, study D's bound was wrong and entry 52 needs a correction.
- **F-5 — provenance with localization held fixed.** The pathogen/non-pathogen ratio **within the
  intracellular stratum only**, where the localization confound is removed by construction.

---

## 5. Frozen predictions, so they can be scored

- **F-1: RR ≈ 0.8**, inside the independent band, slightly sub-multiplicative.
- **F-3: the joint share is 30–45%** — more than either alone, well under their sum.
- **F-4: below 22.08%**, confirming study D's bound.
- **F-5: a ratio above 1.0** — provenance survives with localization held fixed.

## 6. Floors

- 🔒 Every one of the four cells must hold **≥ 150** proteins after the VFDB-sequence exclusion, or
  F-1 is reported as indicative and not as a verdict.
- 🔒 The § 1.1 conservative-definition caveat travels with every number this study produces.

---

## Results, 2026-09-28

`src/85_joint_decomposition.py`, study B's fold unchanged, 30 seeds, no new inference.
`results/joint_decomposition.json`, `results/joint_decomposition_esm2_35M.json`.

| cell | *n* | ESM-2 **650M** | ESM-2 **35M** |
|---|---:|---:|---:|
| **pathogen × extracellular** | 171 | **35.03% [33.30, 36.76]** | **26.88% [25.60, 28.16]** |
| pathogen × intracellular | 597 | 6.37% [5.82, 6.92] | 4.67% [4.31, 5.03] |
| benign-species × extracellular | 444 | 13.14% [12.04, 14.24] | 8.46% [7.85, 9.08] |
| **benign-species × intracellular** | 1,049 | **2.36% [2.08, 2.65]** | **1.95% [1.76, 2.15]** |

🔒 Floor met — the smallest cell is 171, over the frozen 150. 57 exact VFDB sequence matches were
excluded from these cells first (§ 5.1 below).

### 🟢 F-1 / F-2: the two confounds are separable, and on 650M almost exactly so

**RR = 1.001 [0.923, 1.079]** on 650M and **1.364 [1.249, 1.480]** on 35M. Both are inside the frozen
independent band, on both arms, so **the verdict is that provenance and localization compose
multiplicatively.** Each is close to a constant factor on the other's cells:

| | ESM-2 650M | ESM-2 35M |
|---|---:|---:|
| localization, within pathogen species | 5.50× | 5.76× |
| localization, within benign species | 5.57× | 4.34× |
| provenance, within extracellular | 2.67× | 3.18× |
| provenance, within intracellular (**F-5**) | **2.70×** | **2.39×** |

⚠️ **The arms are not identical and the band should not hide it.** 35M's interval **excludes 1.0**, so
there is a real super-multiplicative interaction on that arm; it is simply too small to leave the band
the rule froze. Reported rather than absorbed into the word "independent".

### ⭐ F-3: what they buy together, measured

**42.1%** of the pool→VFDB gap on 650M, **32.2%** on 35M — against 22.7% for provenance alone and
22.1% for localization alone. 🔴 **This is the number that replaces the forbidden sum**, and it is
comfortably below it on both arms.

### 🟢 F-4: study D's upper bound was real

The pool's VFDB-species proteins are flagged at **12.75%** (650M) and **9.61%** (35M), against study
D's **22.08%** from the panel's *curated hard* negatives. 🔒 **Study D declared in advance that 22.08%
was an upper bound, and it is** — the unbiased figure is roughly half of it. Entry 52 needs no
correction.

### 🔒 Scoring the frozen predictions — 3½ of 4

| | frozen | actual | |
|---|---|---|---|
| F-1 | RR ≈ 0.8, independent band, slightly sub-multiplicative | 1.001 / 1.364 | band ✅, direction ❌ |
| F-3 | joint share 30–45% | 42.1% / 32.2% | ✅ both arms |
| F-4 | below 22.08% | 12.75% / 9.61% | ✅ |
| F-5 | above 1.0 | 2.70× / 2.39× | ✅ |

**The best prediction record of the three controls**, after study D's band-right/characterisation-wrong
and study E's miss by a factor of nearly three.

---

## 🔴 5.1 What the exclusion count found: the benign pool contains 133 VFDB proteins

The § 1 exclusion removed **57** exact VFDB sequence matches from the 2×2 cells. That prompted the
count nobody had run, and it is worse than the cells:

| partition | rows | exact VFDB matches | |
|---|---:|---:|---|
| **train** | 1,000 | **13** | 1.30% — fitted as negatives |
| **calibrate** | 500 | **12** | 2.40% — **they set the threshold** |
| test | 6,758 | 108 | 1.60% |
| **whole pool** | **8,259** | **133** | 1.61% |

🔴 **Twelve true virulence factors are in the 500 proteins that set the 95% threshold.** At a 95th
percentile only 25 proteins sit above the cut, so the contaminants can occupy a large share of the tail
that defines it.

🔒 **The direction of that bias is knowable in advance and is conservative**: contaminants score high,
so they push the threshold **up**, and every flag rate this repository reports is **too low**, not too
high. ⚠️ **Direction is not magnitude**, and this study does not establish the magnitude — a
decontaminated re-run does, and is the next thing run rather than an acknowledgement left standing.
