# Preregistration — does the probe respond to hazard, or to VFDB membership?

**Frozen 2026-09-29, after a declared category count and before the probe has been applied to any
category split.** One primary, two secondaries, one obligation. Append-only.

---

## 0. The claim at risk

Study K found that with pathogen origin **and** extracellular localization held fixed, VFDB membership
raises the flag rate by **2.418×** (650M) and **2.617×** (35M) — *supported* on both arms. 🔴 **That is
the project's main positive result, and it is the one most at risk of overreach**, because **VFDB
membership is a curator label, not a hazard label.**

VFDB contains **Motility**: flagellar and chemotaxis proteins, which are surface structures and are not
toxic. ⚠️ Study B already flagged this — on 650M **Motility had the single highest category recovery,
0.8481**, above Exotoxin — and nothing followed it up.

## 🔒 0.1 Declared: study K's population excludes exotoxins by construction

`src/74` built the 4,218-protein set from **"VFDB setA, non-Exotoxin records"**. 🔴 **So study K never
contained the most hazardous class**, and its result is properly stated as *"non-toxin virulence-factor
membership matters"*, not *"membership matters"*. **That correction is owed regardless of what this
study returns**, and is made in the same commit.

It also means the hazard contrast available here is **partly-hazardous versus non-hazardous**, not
toxin versus non-toxin. That is a weaker contrast and is declared as such.

## 🔒 0.2 Declared category count, taken before freezing

Over study K's mapped, non-circular set:

| category | *n* | extracellular | intracellular |
|---|---:|---:|---:|
| Effector delivery system | 316 | 132 | 37 |
| **Motility** | **121** | **28** | **24** |
| Adherence | 112 | 71 | 16 |
| Immune modulation | 110 | 7 | 23 |
| **Nutritional/Metabolic factor** | **62** | **14** | **8** |
| Enzyme | 30 | 4 | 25 |
| Biofilm / Invasion / Regulation / Stress survival | 37 | 11 | 5 |

Nothing about the probe's behaviour on any of these has been looked at.

---

## 1. Design

⚠️ **No new probe and no new inference.** Study G's clean fold, study L's strata, study K's three
exclusions (reviewed, non-circular, length-identical), the same 30 seeds. **Only the VFDB side is
split.**

### 1.1 🔴 The category assignment is my judgment, and it is frozen here

VFDB does not label categories by hazard, so this is a domain call and the reasoning is written down
rather than assumed:

| group | categories | why |
|---|---|---|
| ⭐ **non-hazardous** | **Motility**, **Nutritional/Metabolic factor**, **Regulation**, **Stress survival** | flagella and chemotaxis; siderophores and nutrient acquisition; cytoplasmic transcriptional regulators; catalase and superoxide dismutase. **None of these harms a host directly** — they are required for an organism to establish itself, which is not the same thing |
| **partly-hazardous** | Effector delivery system, Immune modulation, Invasion, Enzyme | secretion systems that deliver effectors; capsule and LPS that subvert host defence; host-cell entry; proteases and lipases. **Each is a mechanism of harm or delivers one** |
| *ambiguous, reported not used* | Adherence, Biofilm, Others, Post-translational modification, Antimicrobial activity | adhesins are essential to infection but are not themselves harmful; the rest are small or heterogeneous |

🔒 **Adherence is excluded from both groups despite being the third largest**, because forcing the
largest ambiguous class into either arm would decide the result by that choice alone.

---

## 2. Primary test, frozen bands — study K's, unchanged

**Q-1**: `C/A` for the **non-hazardous** group against the benign extracellular cell, ESM-2 650M, and
**Q-2** the same on 35M. Bands are study K's so the two are directly comparable:

| band | reading |
|---|---|
| **≥ 2.0** | 🔴 **non-hazardous VFDB categories are flagged like hazardous ones — the probe responds to membership, not hazard** |
| **1.2 – 2.0** | ⚠️ partial |
| **≤ 1.2** | 🟢 the non-hazardous categories are not elevated — study K's effect is hazard-related |

🔴 **The obligation, written before the number exists.** If `C/A ≥ 2.0` for the non-hazardous group on
both arms — that is, statistically indistinguishable from study K's 2.418 — then **"VFDB membership
matters beyond both confounds" must be restated as "the probe responds to VFDB registration"**, and
every document carrying study K's result is amended in the same commit, on the terms entry 53's
obligation was executed. ⚠️ **This is the outcome I expect** (§ 4), and writing the obligation for the
expected outcome rather than the convenient one is the point of writing it first.

## 3. Secondaries

- **Q-3 — Motility alone** (121 proteins, 28 / 24). The single cleanest cell: flagellar proteins are
  surface structures with no toxic function, so a high flag rate here cannot be read as hazard
  detection.
- **Q-4 — non-hazardous against partly-hazardous directly**, within each localization stratum, so
  localization cannot carry the contrast.

## 4. Frozen predictions

Computing forward from study B's recorded per-category recoveries rather than guessing at a mechanism,
which is the approach that worked in study O and failed in studies M and N:

- **Q-1: `C/A` ≈ 2.2 – 2.6 on 650M, in the supported band** — indistinguishable from study K's 2.418.
  **I expect the deflationary outcome.**
- **Q-3: Motility is flagged at or above the set average.** Study B put its recovery at **0.8481**, the
  highest of all 13 categories, above Exotoxin's 0.7944.
- **Q-4: the non-hazardous/partly-hazardous ratio is within 0.8 – 1.25 of 1.0** — no hazard gradient.
- ⚠️ 🔒 **Checkable against a wrong answer**: if the non-hazardous and partly-hazardous groups return
  flag rates equal to three decimals, the split did not happen and that is a bug.

## 5. Floors

🔒 Each group needs **≥ 30** proteins in the stratum being compared, above study H's 25 because that
study came in one protein short and had to report D/B as indicative. Motility's intracellular cell holds
**24** and is therefore **indicative from the outset**, declared now rather than discovered later.

---

## Results, 2026-09-29

`src/99_hazard_vs_membership.py`, study G's clean fold, study K's exclusions, 30 seeds, both arms.
`results/hazard_vs_membership.json`, `results/hazard_vs_membership_esm2_35M.json`.

| population | *n* | ESM-2 **650M** | ESM-2 **35M** |
|---|---:|---:|---:|
| benign × extracellular | 171 | 37.52% | 26.90% |
| ⭐ **non-hazardous × extracellular** | **42** | **89.60%** | **61.67%** |
| ⭐ **Motility alone × extracellular** | **28** | **99.05%** | **71.55%** |
| partly-hazardous × extracellular | 143 | 87.67% | 67.16% |
| ambiguous (Adherence etc.) × extracellular | 79 | 94.26% | 77.59% |
| benign × intracellular | 597 | 6.87% | 5.09% |
| non-hazardous × intracellular | 33 | 54.24% | 46.57% |
| partly-hazardous × intracellular | 83 | 46.43% | 27.35% |

### 🔴 Q-1 / Q-2: the probe responds to VFDB registration, not to hazard

| | 650M | 35M |
|---|---:|---:|
| **non-hazardous `C/A`** | **2.410 [2.323, 2.497]** | **2.309 [2.156, 2.462]** |
| partly-hazardous `C/A` | 2.356 | 2.530 |
| study K, all categories | 2.418 | 2.617 |

🔴 **Both arms clear the 2.0 band, so § 2's obligation fires.** On 650M the non-hazardous group's
2.410 is **indistinguishable from study K's all-category 2.418** and sits **above** the
partly-hazardous group's 2.356. **There is no hazard gradient to find.**

### ⭐ Q-3: flagellar proteins are the single most-flagged group

**Motility alone reaches `C/A` = 2.665 and 2.682** — the highest of any group on both arms — and its
raw flag rate on 650M is **99.05%**. 🔑 **Essentially every flagellar protein in this set is flagged.**
Flagella are surface structures with no toxic function; study B had already put Motility's recovery at
0.8481, above Exotoxin's 0.7944, and this is the follow-up that observation needed.
⚠️ n = 28 and 23, below the frozen floor of 30, so Q-3 is **indicative** — declared in § 5 before the
run, not after.

### 🔴 Q-4: no hazard gradient in either direction

Non-hazardous over partly-hazardous: **1.022** (extracellular) and **1.178** (intracellular) on 650M;
**0.919** and **1.734** on 35M. 🔑 **Three of the four are at or above parity** — the non-hazardous
categories are flagged as much as, or more than, the mechanisms of harm.

### 🔒 The obligation, executed

§ 2 said: *if `C/A` ≥ 2.0 for the non-hazardous group on both arms, then "VFDB membership matters
beyond both confounds" must be restated as "the probe responds to VFDB registration".* **It does, on
both arms.** `docs/MATCHED_VFDB_SCALEUP_PREREGISTRATION.md`, `docs/MATCHED_VFDB_PREREGISTRATION.md` and
`paper/MANUSCRIPT.md` are amended in the same commit.

🔒 **And the correction owed regardless of this result is made with it**: study K's population excludes
exotoxins by construction, so its finding was never about membership in general.

### 🔒 Scoring the frozen predictions

| | frozen | actual | |
|---|---|---|---|
| Q-1 `C/A` 2.2 – 2.6, supported band | | 2.410 / 2.309 | ✅ both arms |
| Q-3 Motility at or above the set average | | highest of any group | ✅ |
| Q-4 within 0.8 – 1.25 of parity | | 1.022, 0.919, 1.178 | ✅ three of four |
| Q-4, 35M intracellular | | **1.734** | ❌ outside — and **more** deflationary, not less |
| equal rates ⇒ the split did not happen | | groups differ | ✅ guard silent |

🔑 **I predicted the deflationary outcome and wrote the obligation for it before running.** That is the
only reason this reads as a result rather than a retreat.
