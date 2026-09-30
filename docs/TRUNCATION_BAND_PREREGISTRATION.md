# Study AA — is the length-U's right arm a truncation artifact?

**Frozen 2026-09-30, before the bands were computed.** Code: `src/113_truncation_length_band.py`.
Artifact: `results/truncation_length_band.json`.

---

## 1. Why this exists

ESM-2 truncates at **1,022 residues** (`max_len` in every embedding manifest). In the benign pool
**293 of 8,259** proteins exceed that and are embedded from their **first 1,022 residues**; they are
**27.5% of the 700+ length band**, which is the band the length-U rises in.

🔴 **None of the three length studies mentions truncation.** `LENGTH_ADJUSTMENT`,
`LENGTH_BAND_SENSITIVITY` and `POOLING_ARTIFACT` contain neither the word nor the number.

⚠️ **An asymmetry worth recording separately**: `src/74` **excludes** VFDB proteins outside [50, 1022]
rather than truncating them (10 dropped for length mismatch, 178 for circularity, 778 kept of 966), while
the pool **truncates**. The two sides of any pool-versus-VFDB comparison treat long proteins differently.

---

## 2. The test

Split the 700+ band at **1,022** — the truncation boundary — into **700–1022 (intact)** and **>1022
(truncated)**, and report the flag rate in each, for benign Bacteria/Archaea, with the localization split
held as in study E. 30 seeds, the same probe, fold and calibration as study E.

🔒 **The arithmetic bound, computed before running.** The published 700+ intracellular rate is **12.18%**
against the mid-band's **1.69%**. If the intact sub-band matched the mid-band, truncation would have to
carry the whole rise:

> 0.725 × 1.69 + 0.275 × r = 12.18  ⇒  **r = 39.8%**

⚠️ **So truncation can only explain the U's right arm if proteins over 1,022 residues are flagged at about
40%.**

| band | reading |
|---|---|
| intact 700–1022 rate **≥ 8%** | the rise is **not** truncation; it is in the intact proteins |
| intact rate **4–8%** | partly truncation |
| intact rate **< 4%** with truncated **> 30%** | ❌ the right arm is a truncation artifact |

🔒 **Prediction: intact ≥ 8%, and truncated below 40%.** Derived from the bound above: 27.5% of a band
cannot move it sevenfold unless its own rate is extreme, and no rate near 40% appears anywhere in this
project's benign measurements — the highest benign stratum rate on record is **21.75%** (extracellular).

🔒 **Second prediction: the truncated group is *not* the lowest** either. Truncation discards C-terminal
residues, which is a perturbation, and study N found perturbations of this kind move the score without
reversing contrasts.

---

## 3. Obligations

- ❌ **If the right arm is a truncation artifact**, studies M, N, O, P and U all inherit it, the U-shape
  becomes a statement about a 1,022-residue window rather than about length, and
  `paper/MANUSCRIPT.md` § 6 says so.
- 🔒 **Whatever the outcome**, the three length preregistrations gain a note that truncation exists and
  where the boundary falls, because none of them mentions it.
- 🔒 **The exclude-versus-truncate asymmetry is recorded in the paper** regardless of this test's result.

---

## 4. Results — truncation is not the cause, and the band's composition is not stable either

**Run 2026-09-30**, 30 seeds, both arms. `results/truncation_length_band{,_esm2_35M}.json`.

| band | n | **650M** | **35M** |
|---|---:|---:|---:|
| 0–250 | 710 | 16.03% | 15.04% |
| 250–700 | 4,707 | 4.95% | 3.48% |
| **700–1022 (intact)** | 535 | **12.32%** | **3.41%** |
| **>1022 (truncated)** | 187 | **4.37%** | **2.50%** |

### 🟢 Truncation does not carry the right arm, on either arm

The frozen bound required proteins over 1,022 residues to be flagged at about **39.8%** for truncation to
explain the rise. They are flagged at **4.37%** and **2.50%**. 🔒 **Ruled out on both arms.**

### 🔴 And the aggregate drop across the boundary is composition, not truncation

The 4.37% looks like suppression until localization is held fixed, and then the sign **reverses between
strata**:

| | intact 700–1022 | >1022 truncated | n |
|---|---:|---:|---|
| extracellular | 12.76% | **24.76%** | 163 vs **21** |
| intracellular | 2.02% | 0.83% | 79 vs **20** |

⚠️ **Opposite signs on 21 and 20 rows.** The aggregate difference is the **extracellular share**: **30.5%**
of the intact band against **11.2%** of the truncated one. 🔑 **This is entry 76's defect exactly** — a
ratio against a mixture — and it was caught only because that entry exists. **A "2.8× truncation blind
spot" was one paragraph away from being written.**

### 🔴 The right arm itself is 650M-dominated

Study M's published band table, both arms, intracellular rate: **650M** 14.15 → 3.76 → 1.91 → **1.69** →
3.59 → **12.18**; **35M** 12.69 → 3.03 → 1.26 → **0.68** → 1.62 → **2.03**. 🔒 **The U's shape replicates**
— both rise from trough to last band — **but its magnitude does not**: **7.2×** the trough on 650M against
**3.0×** on 35M, and 12.18% against 2.03% in absolute terms. ⚠️ **The paper quotes 650M only.**

### 🔴 And the band's composition inverts between arms, so it gets no verdict

Study M's 700+ intracellular band uses **study L's** GO-augmented rule. **189 of its 288 members are
proteins the keyword rule calls `unannotated`** — every one of them:

| | n | **650M** | **35M** |
|---|---:|---:|---:|
| keyword-intracellular too | 99 | **1.78%** | 4.28% |
| **study-L-added (all keyword-unannotated)** | 189 | **14.27%** | **0.90%** |
| ratio added / agree | | **7.99×** | **0.21×** |

🔴 **Eight times higher on one arm and five times lower on the other.** 🔒 **By this project's second-arm
rule that is no verdict**, and it is reported as an instability rather than a finding: **the right arm sits
on a subpopulation whose behaviour is arm-dependent.**

⚠️ **It also qualifies study X.** Entry 76 established that `unannotated` behaves like `intracellular`
overall — length-adjusted ratio **1.066**. That pooled over ten deciles; **inside the 700+ band the two
subsets differ by 8× on 650M.** 🔒 **The aggregate equivalence holds; it does not hold band by band.**

### 🔒 Predictions

| | frozen | 650M | 35M | |
|---|---|---|---|---|
| truncated below the 39.8% bound | | 4.37% | 2.50% | ✅ both |
| intact ≥ 8% | | 12.32% | **3.41%** | ✅ / ❌ |
| the truncated group is **not** the lowest band | | **4.37% is lowest** | **2.50% is lowest** | ❌ **both** |

❌ **The second prediction failed on both arms and it was the reasoned one** — "truncation is a
perturbation and study N found perturbations move scores without reversing contrasts". ⚠️ **Ninth time this
session that the reasoned prediction missed and the derived one held**: the 39.8% bound was arithmetic from
published rates and it held on both arms.

### 🔒 Obligations discharged

- 🔒 The three length preregistrations gain a note that truncation exists, where the boundary falls, and
  that it was ruled out here.
- 🔒 The **exclude-versus-truncate asymmetry** is recorded: `src/74` drops VFDB proteins outside
  [50, 1022] — **778 kept of 966**, 178 for circularity and **10** for length mismatch — while `src/02b`
  truncates pool proteins above it. **293 of 8,259** pool proteins are affected.
