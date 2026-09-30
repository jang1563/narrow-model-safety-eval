# Preregistration — why are unannotated proteins flagged at half the rate?

**Frozen 2026-09-30, after a declared diagnostic and before any adjusted rate has been computed.**
One primary, two secondaries. Append-only.

---

## 0. The guard that has failed every time it has been run

Study E froze an availability check: the `unannotated` stratum's flag rate over the pooled annotated
rate must lie in **[0.67, 1.5]**. It has failed on **every** version:

| | ratio | |
|---|---:|---|
| study E, 650M | **0.41** | ❌ |
| study E, 35M | **0.32** | ❌ |
| study L, 650M | **0.48** | ❌ |
| study L, 35M | **0.52** | ❌ |

🔑 **Doubling the annotated fraction from 45% to 66% did not fix it**, which is why study L recorded it
as a property of the pool rather than of the keyword vocabulary. **It has been reported four times and
explained zero times**, and it sits underneath every localization number this project has.

## 🔒 0.1 Declared diagnostic, taken before freezing

Over the 6,247 eligible proteins — 3,933 annotated, 2,314 unannotated:

| | annotated | unannotated |
|---|---:|---:|
| median length | 449 | 416 |
| **share over 700 residues** | **14.7%** | **7.3%** |
| share under 250 | 11.3% | 11.9% |
| "uncharacterised"-style name | 4.2% | 0.2% |
| median proteins per organism | 12 | 11 |

🔑 **One candidate stands out and one is eliminated.** The long tail differs **twofold** — and study M
established that the flag rate is **U-shaped in length**, high at both ends, so a population with twice
as many long proteins should be flagged more. ⭐ **Organism curation depth is not it**: 12 against 11.
⚠️ The name check runs *against* the obvious story — annotated proteins are **more** likely to carry an
"uncharacterised" style name, not less.

🔒 **Arithmetic done forward, before running**: annotated holds **26.0%** of its mass in the high-flag
bands (under 250 plus over 700) against unannotated's **19.2%** — a **1.35×** difference in high-band
share, against an observed rate ratio of about **2.1×**. **So length composition should explain part of
the gap and not all of it.** Nothing about adjusted rates has been computed.

---

## 1. Design

⚠️ **No new probe and no new inference.** Study G's clean fold, study L's strata, the same pool rows,
30 seeds.

---

## 2. Primary test, frozen bands

**T-1**: the **Mantel–Haenszel** ratio of unannotated to annotated flag rates, pooled across ten length
deciles cut on the combined population, both arms.

| band | reading |
|---|---|
| **0.67 – 1.5** | 🟢 **length composition explains the guard's failure** — the guard should be computed length-adjusted |
| **0.50 – 0.67** | ⚠️ length explains part of it; a residual remains |
| **< 0.50** | 🔴 length explains essentially none of it, and the cause is something else |

🔒 **Decile floor**: a decile contributes only with ≥ 20 proteins in **each** group; fewer than 6
contributing deciles makes T-1 indicative.

## 3. Secondaries

- **T-2 — magnitude or direction?** Study O's decomposition, `‖z‖ · cos(z, w) · ‖w‖`, with each factor
  held at its pooled median in turn. 🔑 Study O found the length-U was **directional**; if the
  availability gap is directional too, the two are plausibly the same phenomenon.
- **T-3 — embedding-space density.** Mean cosine to the 10 nearest pool neighbours, annotated against
  unannotated. ⚠️ **The hypothesis this tests is uncomfortable**: a protein UniProt has localized is one
  that has been *studied*, which correlates with belonging to a well-known family, and a probe trained
  on VFDB representatives may respond to family recognisability rather than to anything about
  localization. **If annotated proteins sit in denser regions, the guard's failure and part of the
  localization result could share that cause.**

## 4. Frozen predictions

- **T-1: between 0.50 and 0.67 — length explains part, not all.** The forward arithmetic in § 0.1 gives
  1.35× of an observed 2.1×, so roughly half the gap on a log scale.
- **T-2: directional**, as study O found for length.
- **T-3: annotated proteins sit in denser regions**, mean nearest-neighbour cosine higher by 0.005 or
  more.
- ⚠️ 🔒 **Checkable against a wrong answer**: if the Mantel–Haenszel ratio equals the crude ratio to
  three decimals, the stratification did not happen and that is a bug.

## 5. What no outcome licenses

🔒 **None of this rehabilitates or retracts the localization finding.** That finding compares
**extracellular against intracellular**, both of which are *annotated*; the unannotated stratum is not
in the ratio. ⚠️ **What a T-3 result would touch is the interpretation** — if the probe partly reads
family recognisability, "the probe is a localization detector" needs company — and **that is a
qualification, not a refutation, and is to be written as one.**

---

## Results, 2026-09-30

`src/103_annotation_availability.py`, study G's clean fold, study L's strata, 30 seeds, both arms.
All ten deciles contribute on both.

| | ESM-2 **650M** | ESM-2 **35M** |
|---|---:|---:|
| crude unannotated / annotated | 0.482 [0.465, 0.498] | 0.520 [0.503, 0.536] |
| **T-1, length-adjusted (Mantel–Haenszel)** | **0.532 [0.514, 0.550]** | **0.522 [0.506, 0.538]** |
| ⭐ **share of the gap that closes** | **10%** | **0%** |
| T-2, direction-only | 0.508 | 0.547 |
| T-2, magnitude-only | 0.421 | 0.355 |
| T-3, mean cosine to 10 nearest neighbours | +0.0004 | **−0.0007** |

### 🔴 Length is not the cause, and the band label says otherwise

The adjusted ratio lands at **0.532** and **0.522**, which the frozen bands call *"length explains part
of it"* — but adjusting moves it by **10%** of the distance to parity on 650M and by **0%** on 35M.

🔴 **That is a defect in § 2's bands, not a finding.** They were set on the adjusted ratio's **level**
and labelled with a conclusion about **how much length explains**, which are different quantities —
exactly the error study E's § 2 made when it set bands on a ratio and labelled them with conclusions
about a share. 🔒 **The honest statement is the explained fraction: length accounts for at most a tenth
of the gap, and on one arm for none of it.**

⚠️ **And § 4's forward arithmetic was wrong.** It computed a 1.35× difference in high-flag-band share
against an observed 2.1× and predicted "roughly half the gap on a log scale". The actual answer is a
tenth. **The arithmetic was right about the composition difference and wrong about what it buys**,
because the flag rate inside a decile also differs between the two groups — which is the thing
adjustment removes and the forward calculation ignored.

### 🔴 T-2: the decomposition does not localise it either

Neither counterfactual approaches parity: direction-only gives **0.508** and **0.547**, magnitude-only
**0.421** and **0.355**. 🔑 **Both factors carry the gap, roughly equally** — which makes it *unlike*
the length-U, where holding direction fixed **inverted** the effect and magnitude worked against it
(study O). **So the availability gap and the length-U are not the same phenomenon.**

### 🔴 T-3: the uncomfortable hypothesis is refuted

Annotated proteins do **not** sit in denser regions. The mean cosine to the ten nearest pool neighbours
differs by **+0.0004** on 650M and **−0.0007** on 35M — **zero, with the sign flipping between arms**,
against a prediction of ≥ 0.005 in one direction. 🟢 **"The probe reads family recognisability" is not
supported**, and § 5's qualification of the localization finding is therefore **not** owed.

### 🔒 What this study establishes

**A negative result, and it is the useful kind**: three candidate explanations for a guard that has
failed four times are now eliminated or bounded — **length at ≤ 10%, a single score component at
neither, embedding density at zero**. ⚠️ **The cause remains unknown**, and the guard stays failed and
reported wherever the localization numbers appear.

🔒 **What did not change**: the localization finding compares extracellular against intracellular, both
**annotated**. The unannotated stratum is not in that ratio, and nothing here touches it.

### 🔒 Predictions: one band, three characterisations wrong

| | frozen | actual | |
|---|---|---|---|
| T-1 ratio 0.50 – 0.67 | | 0.532 / 0.522 | ✅ band |
| T-1 "roughly half the gap" | | **10% / 0%** | ❌ |
| T-2 directional, as study O | | both factors equally | ❌ |
| T-3 annotated denser by ≥ 0.005 | | +0.0004 / −0.0007 | ❌ sign flips |
| adjusted == crude ⇒ a bug | | 0.532 vs 0.482 | ✅ guard silent |

⚠️ **Every mechanism I guessed at was wrong, and the one quantity I computed forward was wrong too.**
This is the fourth study in the sequence where guessing at a mechanism failed; the difference from
study O is that there the arithmetic ran forward from *measured rates*, and here it ran forward from
*composition* while ignoring that the rates within each stratum also differ.
