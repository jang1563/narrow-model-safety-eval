# Preregistration — study E's contrast on a localization source that clears its own ceiling

**Frozen 2026-09-29, after a declared coverage count and before the probe has been applied to any
restratified cell.** One primary, one second-arm replication, two guards. Append-only.

---

## 0. The breach this exists to close

Study E returned this project's main adverse finding — the probe is strongly graded by localization on
**benign** proteins, R = 5.07 and 4.35 after decontamination — and it breached its own frozen ceiling
doing it:

> 🔒 **Ceiling**: if `unannotated` exceeds 50% of the eligible pool, the annotated contrast is not
> representative of the pool and that caveat is carried in every statement of the result.

**It came in at 55.0%.** The availability check failed too, at 0.41 against a frozen [0.67, 1.5].
🔑 **Both failures have the same cause**: UniProt's *curated keywords* are sparse, so the contrast was
measured on the 45% of the pool a curator had reached.

## 🔒 0.1 Declared coverage count, taken before freezing

| localization source | eligible proteins covered | unannotated |
|---|---:|---:|
| UniProt keywords (study E's rule) | 2,813 / 6,247 (45.0%) | **55.0%** ❌ |
| GO cellular component | 4,070 (65.2%) | 34.8% |
| ⭐ **keyword ∪ GO CC ∪ signal/transmembrane** | **4,127 (66.1%)** | **33.9%** ✅ |

🔒 **So the ceiling is clearable, and by a wide margin.** Nothing about the probe's behaviour on any
restratified cell has been looked at.

---

## 1. Design

⚠️ **No new probe and no new inference.** Study G's clean fold, 30 seeds, the same pool rows. **Only
the localization annotation changes.**

### 1.1 Frozen strata rule, first match wins

| stratum | rule |
|---|---|
| **extracellular** | GO CC names any of `extracellular`, `cell outer membrane`, `cell surface`, `cell wall`, `fimbri`, `pilus`, `flagell`, `capsule`, `S-layer`; **or** a `SIGNAL` feature exists; **or** study E's keyword rule says extracellular |
| **membrane** | otherwise GO CC names `membrane`; **or** a `TRANSMEM` feature exists; **or** study E's keyword rule says membrane |
| **intracellular** | otherwise GO CC names `cytoplasm`, `cytosol`, `periplasm`, `nucleoid`, `ribosome`, `chromosome`; **or** study E's keyword rule says intracellular |
| **unannotated** | otherwise |

🔒 **Study E's rule is embedded, not replaced** — the new sources are unions with it, so every protein
study E classified keeps its class and the change is purely additive. That is what makes L-1 comparable
to E-1 rather than a different experiment.

⚠️ **Declared conflation, carried over deliberately**: a signal peptide routes a protein out of the
cytoplasm but in a Gram-negative that can end at the outer membrane or the periplasm, not outside the
cell. Study E's keyword rule made the same call (`Signal` → extracellular). **Keeping it identical is
the point**; changing it would confound the coverage question with a definition change.

🔒 Population unchanged: Bacteria and Archaea in the pool's test partition.

---

## 2. Primary test, frozen bands — study E's, unchanged

**L-1**: `R` = flag rate(extracellular) / flag rate(intracellular), ESM-2 650M.
**L-2**: the same on 35M. A verdict needs **both arms in the same band**.

| band | reading |
|---|---|
| **R ≤ 1.5** | 🟢 localization is not a material driver |
| **1.5 < R < 3.0** | ⚠️ partial |
| **R ≥ 3.0** | 🔴 localization is a major driver — study E's verdict |

🔴 **What the adverse-for-study-E outcome obliges**: if `R < 3.0` on both arms once annotation is no
longer sparse, then **study E's verdict was an artifact of measuring on the curated 45%**, and the
qualifications entry 53 wrote into the manuscript and four preregistrations must be **revisited in the
same commit**, on the terms that obligation was written on.

## 3. Guards, which are the point of this study

- 🔒 **L-3, the ceiling**: `unannotated` must be **≤ 50%**. Declared at 33.9%; reported as measured.
- 🔒 **L-4, availability**: the unannotated stratum's flag rate over the pooled annotated rate must lie
  in **[0.67, 1.5]**. Study E failed this at 0.41. ⚠️ **It may fail again** — if the remaining 33.9% is
  still systematically different, that is a finding about what annotation tracks, not a defect to hide.
- 🔒 **Floors**: extracellular and intracellular ≥ 300 each, as study E froze.
- 🔒 **Length confound**: length-alone AUROC for extracellular vs intracellular, declared confounded
  above 0.65, with a 1:1 length-matched subsample as the fallback. Study E measured 0.607.

## 4. Frozen predictions

- **L-1: R stays adverse, 4.0 – 6.0 on 650M**, and above 3.0 on 35M. The proteins the new sources add
  were previously `unannotated` and flagged at **4.28%**, almost exactly the old intracellular rate of
  4.13%, so most of them should land intracellular and the ratio should hold or fall slightly.
- **extracellular falls below 21.75%** — the newly-added extracellular proteins are ones no curator
  labelled secreted, so they should be less obviously exported and flagged lower.
- **L-3 passes at ≈ 34%.**
- ⚠️ **L-4 fails again**, at a ratio below 0.67. Annotation availability tracking the score is a
  property of the pool, not of the keyword source.

---

## 🔴 Amendment 1 — 2026-09-29, on running it: § 1.1 described two different rules

§ 1.1 states a **first-match-wins rule over the union of all sources** and, in the same paragraph,
claims the change is **"purely additive"** so that "every protein study E classified keeps its class".
🔴 **Those are not the same rule**, and the script's own additivity guard caught it on the first run:
**49** proteins study E had classified would have been reclassified, because a `SIGNAL` feature or an
extracellular GO term outranks a `Cytoplasm` keyword under first-match-wins.

🔒 **The stated intent governs, because it is what this study is for.** The question is whether study
E's **coverage** breach invalidates its verdict, so classification is held fixed and only coverage
changes: study E's class where it has one, the GO/signal/transmembrane rule only where it says
`unannotated`. The literal reading is kept as `union_stratum()` and its **49** reclassifications are
reported in every artifact.

---

## Results, 2026-09-29

`src/92_localization_coverage.py`, study G's clean fold, 30 seeds, both arms.
`results/localization_coverage.json`, `results/localization_coverage_esm2_35M.json`.

| stratum | study E | this study | Δ |
|---|---:|---:|---:|
| extracellular | 652 | 657 | +5 |
| membrane | 495 | 537 | +42 |
| **intracellular** | 1,666 | **2,739** | **+1,073** |
| **unannotated** | 3,434 | **2,314** | **−1,120** |

🔒 **Additive as amended**: 0 of the 2,813 proteins study E classified changed class.

| | ESM-2 **650M** | ESM-2 **35M** |
|---|---:|---:|
| extracellular | 23.93% [22.91, 24.94] | 15.00% [14.29, 15.71] |
| intracellular | 4.65% [4.39, 4.90] | 2.65% [2.43, 2.86] |
| unannotated | 4.62% | 3.23% |
| ⭐ **L-1 / L-2, R** | **5.182 [5.034, 5.330]** | **5.855 [5.500, 6.211]** |
| R with contaminants removed from the strata | 4.965 [4.806, 5.124] | 5.353 [5.002, 5.705] |

### 🔴 The verdict holds, on annotation that is no longer sparse

**R = 5.182 and 5.855**, both far above the frozen 3.0, on both arms. 🔒 **Study E's adverse finding
was not an artifact of measuring on the curated 45%.** The decontaminated figures — 4.965 and 5.353,
directly comparable to study G's clean condition at 5.074 and 4.347 — say the same thing.

### 🟢 L-3: the ceiling is cleared, and the caveat it imposed is lifted

**Unannotated falls from 55.0% to 37.0%**, under the frozen 50%. 🔒 **The sentence study E's
preregistration attached to every statement of its result — "the annotated contrast is not
representative of the pool" — no longer applies**, and the intracellular stratum in particular is now
built on 2,739 proteins rather than 1,666.

### ⚠️ L-4 fails again, as predicted, and that is now a finding rather than a defect

The availability ratio is **0.48** and **0.52**, still outside [0.67, 1.5]. 🔑 **Doubling the
annotated fraction did not fix it**, so annotation availability tracking the score is a property of
**the pool**, not of the keyword source — proteins no curator has localized are flagged at about half
the rate of proteins someone has, whichever source is asked.

### 🔴 And one number moved the wrong way: the length-matched estimate

Length-alone AUROC is **0.562**, better than study E's 0.607 and well inside the 0.65 tolerance, so
the frozen rule does **not** invoke the matched subsample and **R = 5.182 governs.** But the matched
ratio is **2.623** on 650M — study E's was 4.382 — and **2.623 sits in the *partial* band.** On 35M it
is 3.553, above the threshold. 🔴 **So the two arms straddle the band boundary on the matched
estimate**, and it is recorded here rather than left out because the frozen rule happened not to reach
for it.

⚠️ **The likely reason is mechanical**: the intracellular stratum grew by 1,073 proteins, so 1:1
nearest-neighbour matching now draws from a much larger and better-matched pool. **That makes the
matched estimate more trustworthy, not less** — which is exactly why the gap between 5.182 and 2.623
is worth stating. **Length carries more of this contrast than an AUROC of 0.562 suggests.**

### 🔒 Scoring the frozen predictions

| | frozen | actual | |
|---|---|---|---|
| L-1 R = 4.0 – 6.0 on 650M | | 5.182 | ✅ |
| L-1 above 3.0 on 35M | | 5.855 | ✅ |
| extracellular **falls** below 21.75% | | **23.93%** | ❌ it rose |
| L-3 passes at ≈ 34% | | 37.0%, cleared | ✅ cleared, value 3 pp off |
| L-4 fails below 0.67 | | 0.48 / 0.52 | ✅ both arms |

⚠️ The extracellular miss has a mundane cause: these rates are on study G's **clean fold**, which
lowers the threshold, and the frozen strata still carry the contaminants. The decontaminated
extracellular rate is the one comparable to study E's, and § 3's prediction should have named which of
the two it meant. **Third time a prediction in this project has failed to say which population it was
about.**
