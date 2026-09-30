# Study AF — are the honest intervals still too narrow, and is the between-pool gap draw noise?

**Frozen 2026-10-01, before any clustered interval was computed.** Code: `src/118_organism_cluster.py`.
Artifacts: `results/organism_cluster{,_esm2_35M}.json`.

---

## 1. Two questions, one instrument

Studies AC–AE replaced seed intervals with **protein-level** ones, 8–12× wider, and settled `C/A` at
**partial**. 🔴 **Proteins from the same organism are not independent**: they share provenance,
secretion machinery and homology, which are the three things this paper's own confound studies identify as
driving the flag. **A bootstrap that resamples proteins independently assumes away exactly what § 6
measures.**

| cell | proteins | organisms | mean cluster |
|---|---:|---:|---:|
| benign extracellular, pool 1 | 171 | **65** | 2.6 |
| VFDB extracellular, expanded | 703 | **65** | **10.8** |

⚠️ **Neither cell is dominated by one organism** — the largest contributes 8.8% and 9.0% — so this is a
design-effect question and not an outlier question.

And the second question, which the same instrument answers: study AE found the two benign draws differ by
**0.30** in `C/A`. 🔒 **If 0.30 sits inside the organism-clustered interval of one pool, it is draw noise
at the organism level; if it sits outside, the draws differ systematically.**

---

## 2. What is computed

A three-level bootstrap: resample a **seed**, then resample **organisms with replacement** inside each
cell, then take **all** proteins of the chosen organisms. Reported beside study AE's protein-level
interval, both arms, B = 20,000.

🔒 **The point estimate is unchanged by construction.** What is reported is the **design effect** — the
ratio of interval widths — and where the between-pool gap falls.

| band on the width ratio | reading |
|---|---|
| **< 1.2** | organism clustering is negligible; the protein-level intervals stand |
| 1.2 – 2.0 | material; the published intervals are optimistic and are restated |
| **> 2.0** | ❌ the protein-level intervals are the wrong instrument too |

---

## 3. Predictions

🔒 **Width ratio > 1.3 on both arms, and larger for `C/A` than for `R`.** Derived: the design effect is
approximately 1 + (m̄ − 1)·ICC, and `C/A`'s numerator has **m̄ = 10.8** against the benign cells' 2.6, so
even a modest intra-organism correlation inflates the numerator's variance several-fold.

🔒 **The 0.30 between-pool gap sits inside the clustered interval** — draw noise at the organism level, not
a systematic difference between the draws. Derived: 0.30 is 12.5% of a 2.4 point estimate, and a clustered
half-width of ±0.9 or more covers it comfortably.

🔒 **`C/A`'s verdict stays partial.** ⚠️ **If the clustered lower bound falls below 1.2**, the frozen
retraction threshold is crossed and § 5 says so — **that is the one outcome that would change a
conclusion, and it is possible**: a 2× widening of [1.772, 2.541] reaches about 1.4, and a 3× widening
reaches 1.1.

---

## 4. Obligations

- ❌ **If the clustered lower bound crosses 1.2**, the registration contrast is **retracted** by the
  threshold frozen in study H, and the abstract and § 5 say so.
- 🔒 **If the width ratio exceeds 1.2**, every interval in studies AC–AE is restated as protein-level and
  the clustered one is given beside it, because "honest" was the word used for the protein-level ones.
- ⚠️ **`R` is computed the same way** — its cells are benign extracellular against benign intracellular,
  both clustered by organism — so the localization verdict is exposed to the same test.

---

## 5. Results — clustering is negligible, and my prediction was wrong against the project's own evidence

**Run 2026-10-01**, 30 seeds, B = 20,000, both arms.
`results/organism_cluster{,_esm2_35M}.json`.

| | point | protein-level | **organism-clustered** | design effect |
|---|---:|---|---|---:|
| **650M** `C/A` | 2.409 | [1.903, 3.231] | [1.870, 3.337] | **1.11** |
| **650M** `R` | 4.965 | [3.679, 6.702] | [3.469, 7.033] | **1.18** |
| **35M** `C/A` | 2.568 | [1.938, 3.892] | [1.893, 4.089] | **1.12** |
| **35M** `R` | 5.353 | [3.468, 8.750] | [3.282, 9.062] | **1.09** |

### 🟢 The protein-level intervals stand

Design effects **1.09 – 1.18**, all under the 1.2 the document froze as *negligible*. 🔒 **So studies
AC–AE's intervals do not need restating**, and that is an earned result rather than an assumption — the
test could have gone the other way and § 4's first obligation was written for that case.

🔒 **Both verdicts survive**: `C/A` **partial** on clustered lower bounds **1.870** and **1.893**, above
the frozen retraction threshold of 1.2; `R` clears its 3.0 band at **3.469** and **3.282**.

### 🔴 And both halves of my prediction were wrong

❌ § 3 predicted the width ratio **> 1.3 on both arms**; it is **1.09 – 1.18**.
❌ § 3 predicted `C/A`'s ratio would exceed `R`'s, because `C/A`'s numerator has mean cluster size **10.8**
against `R`'s 1.8 and 3.9. On 650M it is the **other way round** — 1.11 against 1.18.

🔑 **The model was wrong, not the arithmetic.** A design effect is 1 + (m̄ − 1)·ICC, and I reasoned from
m̄ while treating the ICC as a free parameter. **The ICC is what matters here and it is small: organism
identity does not carry the flag.**

⚠️ **And the evidence for that was already in this project.**
`docs/ORGANISM_STRATIFIED_PREREGISTRATION.md` found margin's ordering holds **within species** — mean
*ρ* +0.3750 and +0.4107, 8 of 10 strata positive — which is a direct statement that organism identity is
not what the score tracks. 🔴 **I predicted a large organism design effect against a result this project
had already established**, and the correct prior was one document away.

### 🔒 And study AE's leftover question is answered

The **0.30** between-pool gap in `C/A` sits inside a clustered half-width of **0.734** (650M) and
**1.098** (35M). 🔒 **Draw noise at the organism level, on both arms** — the two benign draws do not
differ systematically, and a third draw would not settle anything this does not.
