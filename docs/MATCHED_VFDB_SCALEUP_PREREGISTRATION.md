# Preregistration — study H's matched contrast, at seven times the size and without its selection

**Frozen 2026-09-28, after a declared feasibility count and before the probe has been applied to any
of these cells.** One primary, one second-arm replication, two secondaries. Append-only.

---

## 0. The two things study H could not do

Study H found that with pathogen origin **and** extracellular localization held fixed, VFDB membership
covers **99.1%** of the remaining distance to the full-VFDB rate. It carried two limits it could not
remove:

1. 🔴 **Its VFDB side was selected against the hypothesis.** The bridge is the pool∩VFDB overlap, and
   the pool's build query excludes the *Virulence*, *Toxin*, *Cytolysis*, *Hemolysis*, *Bacteriocin*
   and *Bacteriolytic enzyme* keywords — so every bridge protein is a virulence factor **UniProt
   declines to call virulent**.
2. ⚠️ **It was small**: 38 extracellular and **24** intracellular, one protein under its own floor, so
   D/B was indicative rather than a verdict.

🔑 **VFDB headers carry a RefSeq or GenBank accession for every record**, so UniProt's ID-mapping
service gives the 4,218 admitted negatives a Swiss-Prot identity — *including* the virulence-annotated
members the pool query removed. That is what this study uses.

## 🔒 0.1 Declared feasibility count, taken before freezing

| | |
|---|---:|
| admitted VFDB negatives | 4,218 |
| map to a **reviewed** Swiss-Prot entry | 966 |
| of those, **non-circular** (not a class-axis positive) | **788** |
| of those, **length identical** to the mapped entry | **778** |
| ⭐ carry the **Virulence** keyword | 300 |

| stratum | length-matched | length differs |
|---|---:|---:|
| **extracellular** | **264** | 3 |
| **intracellular** | **134** | 4 |
| membrane | 160 | 1 |
| unannotated | 220 | 2 |

Study H's cells were **38** and **24**. Nothing about the probe's behaviour on any of these has been
looked at.

---

## 1. Design

⚠️ **No new probe.** Study G's clean fold exactly — contaminants dropped from the admitted rows, sizes
by `src/83`'s rule, 30 seeds, threshold at the 95th percentile of calibration scores.

**New inference, once**: the 4,218 were embedded on 650M only, so the 35M arm is embedded by
`src/77 --model facebook/esm2_t12_35M_UR50D`, whose gate reproduces the published 35M panel negatives.
🔒 A verdict needs both arms, which is why this is worth the run.

| side | population |
|---|---|
| **benign** | pool test-partition proteins, Bacteria from VFDB species, **not** in VFDB |
| **VFDB** | the 778 mapped, reviewed, non-circular, length-identical negatives |

🔒 **Three exclusions, all frozen and all applied before any scoring:**
- **reviewed only** — both sides then sit on the same annotation pipeline, which is the confound that
  bit A2 in entry 36;
- **non-circular** — a protein that is also a class-axis positive is training data (entries 57, 61);
- **length identical** to the mapped UniProt entry, so the annotation describes the sequence that was
  embedded. The 10 length-mismatched are **reported, not silently dropped**.

🔒 **No species test on the VFDB side**: a VFDB record is pathogen-derived by construction, and a
taxonomic synonym is not evidence about provenance (study H, Amendment 1).

---

## 2. Primary test, frozen bands — deliberately identical to study H's

**K-1**: `C/A` = flag rate(VFDB × extracellular) / flag rate(benign × extracellular).
**K-2**: the same on ESM-2 35M. **K-1b**: `D/B` within intracellular, now above its floor.

| band | reading |
|---|---|
| **≥ 2.0** | 🟢 VFDB membership matters beyond both confounds — **supported**, not partial |
| **1.2 – 2.0** | ⚠️ partial, as study H found |
| **≤ 1.2** | 🔴 membership adds little once provenance and localization are held fixed |

🔒 **The bands are study H's, unchanged, so the two results are directly comparable.** A verdict
requires **both arms in the same band**.

🔴 **What the adverse band obliges**: if `C/A ≤ 1.2` on both arms, then study H's result was an
artifact of its small, oddly-selected bridge, and **H-5's 99.1% must be withdrawn** from the manuscript
and from `docs/MATCHED_VFDB_PREREGISTRATION.md` in the same commit.

## 3. Secondaries

- **K-3 — the selection that study H could not escape.** Flag rate of the **300 Virulence-keyword**
  members against the rest of the mapped set. 🔑 If the keyword-carrying members are flagged higher,
  that is the direct measurement of what study H's bridge was missing.
- **K-4 — the H-5 analogue**: what fraction of the distance from the matched benign cell to the
  decontaminated full-VFDB reference (74.09%, 3,676 proteins) does membership cover at this scale.

## 4. Frozen predictions

- **K-1: C/A ≈ 2.2 on 650M, in the supported band**, higher than study H's 1.974 because the
  virulence-annotated members are now present. **35M ≈ 1.7**, still partial — the arms will likely
  **disagree on the band**, and if they do, the frozen rule gives no verdict and that is the correct
  outcome rather than a disappointment.
- **K-1b: D/B ≈ 3.0**, and now a verdict rather than indicative.
- **K-3: the Virulence-keyword members are flagged higher**, by 10–25 points.
- **K-4: 95–105%**, close to study H's 99.1%.

## 5. Floors

🔒 Every VFDB cell must hold **≥ 100** after all three exclusions, or the affected test is indicative.
Observed: 264 and 134, both clear.
🔒 The 10 length-mismatched proteins are reported in the artifact whatever the outcome.

---

## Results, 2026-09-28

`src/91_matched_vfdb_scaleup.py`, study G's clean fold, 30 seeds, both arms.
`results/matched_vfdb_scaleup.json`, `results/matched_vfdb_scaleup_esm2_35M.json`.

966 mapped-reviewed → **778** kept: 178 dropped as class-axis positives, 10 as length-mismatched (all
ten listed in the artifact). Cells: VFDB **264** extracellular and **134** intracellular against benign
171 and 597. Floor of 100 met on both.

| population | *n* | ESM-2 **650M** | ESM-2 **35M** |
|---|---:|---:|---:|
| benign × extracellular | 171 | 37.52% | 26.90% |
| ⭐ **VFDB × extracellular** | **264** | **89.95% [89.39, 90.51]** | **69.41% [67.69, 71.12]** |
| benign × intracellular | 597 | 6.87% | 5.09% |
| ⭐ **VFDB × intracellular** | **134** | **54.08% [52.41, 55.75]** | **34.00% [32.45, 35.56]** |
| VFDB × membrane | 160 | 71.75% | 41.65% |
| VFDB × unannotated | 220 | 54.86% | 28.97% |
| *reference*: non-circular VFDB set | 3,676 | 74.09% | 49.57% |

### 🟢 K-1 / K-2: supported on both arms — study H's "partial" was its bridge, not the effect

**C/A = 2.418 [2.339, 2.497]** on 650M and **2.617 [2.495, 2.740]** on 35M. Both clear the frozen
**2.0** threshold, on both arms, so **the verdict is supported, not partial.** Study H's bridge gave
1.974 and 1.497 — 🔑 **the difference is the selection study H could not escape**, and K-3 measures it
directly.

🟢 **K-1b is now a verdict too**: `D/B` = **7.988 [7.665, 8.312]** and **6.875 [6.434, 7.317]** on cells
of 134 against a floor of 100, where study H had 24 against a floor of 25. Within the intracellular
stratum — where neither secretion nor surface exposure can be doing the work — **a VFDB protein is
seven to eight times more likely to be flagged than a benign one from the same kind of organism.**

### ⭐ K-3: the members study H's bridge could not contain are flagged 15–19 points higher

The **228** kept proteins carrying UniProt's *Virulence* keyword are flagged at **83.67%** against
**64.48%** for the other 550 on 650M (**+19.18 pp**), and **56.89%** against **41.72%** on 35M
(**+15.16 pp**). 🔑 **The pool's build query excluded exactly these**, which is why study H's bridge
read partial and this reads supported. **The selection caveat study H declared in advance was real and
is now quantified.**

### 🔴 K-4 broke, and what it breaks is study H's H-5 framing

K-4 returns **143.4%** on 650M and **187.5%** on 35M. A "fraction of the distance to the full-VFDB
rate" cannot exceed 100% unless **the reference is not a ceiling** — and it is not. The full-VFDB rate
averages over a **mixture of localizations**: 89.95% extracellular, 71.75% membrane, 54.86%
unannotated, 54.08% intracellular. A localization-**matched** cell is not bounded by that mixture and
here sits well above it.

🔴 **So H-5's 99.1% must be read differently than it was written.** It does not mean membership
explains 99.1% of the residual; it means the matched cell happened to land just under the whole-set
average at study H's scale. **At this scale it lands far above.** The quantity itself is not
well-founded, and `docs/MATCHED_VFDB_PREREGISTRATION.md` and the manuscript are amended accordingly in
the same commit.

⚠️ **The underlying finding is unharmed and is stronger here**: with provenance and localization both
held fixed, membership raises the flag rate by **2.4–2.6×** in the extracellular stratum and **6.9–8.0×**
in the intracellular one. **It is the normalisation that was wrong, not the contrast.**

### 🔒 Scoring the frozen predictions

| | frozen | actual | |
|---|---|---|---|
| K-1 650M ≈ 2.2, supported band | | **2.418** | ✅ band and close |
| K-1 35M ≈ 1.7, partial; **arms disagree** | | **2.617**, supported | ❌ — 35M came in *above* 650M and the arms **agree** |
| K-1b D/B ≈ 3.0 | | 7.988 / 6.875 | ❌ badly under-predicted |
| K-3 +10 to +25 pp | | +19.18 / +15.16 | ✅ both arms |
| K-4 95–105% | | 143.4% / 187.5% | ❌ — and the miss is what exposed the metric |

⚠️ **Two of five, and the most useful thing this study did came out of a prediction that failed.**

---

## 🔴 Superseded on what the effect IS — 2026-09-29, entry 69

This document reads its result as **VFDB membership mattering beyond provenance and localization**.
`docs/HAZARD_VS_MEMBERSHIP_PREREGISTRATION.md` split the same population by whether a category is a
mechanism of harm, and the effect **does not track hazard at all**:

| | 650M | 35M |
|---|---:|---:|
| non-hazardous categories (Motility, Nutritional/Metabolic, Regulation, Stress survival) | **2.410** | **2.309** |
| partly-hazardous (Effector delivery, Immune modulation, Invasion, Enzyme) | 2.356 | 2.530 |
| ⭐ **Motility alone** — flagella, no toxic function | **2.665** | **2.682** |

🔴 **Motility is the most-flagged group on both arms, at a 99.05% raw rate on 650M.** The correct
statement is that **the probe responds to VFDB registration**, not to hazard.

🔒 **And a second correction owed regardless**: `src/74` built this population from *"VFDB setA,
non-Exotoxin records"*, so **exotoxins were never in it**. Every claim here is about **non-toxin**
virulence factors and is restated that way.

⚠️ **What still stands**: the *contrast* is real and large — VFDB-registered proteins are flagged
2.3–2.7× above matched benign ones with provenance and localization held fixed, and 7–9× inside the
intracellular stratum. **What changed is what it is evidence for.**
