# Study Y — remove the localization axis, and see what is left

**Frozen 2026-09-30, before any of Y-0 – Y-4 was computed.** Code: `src/107_localization_nullspace.py`.
Artifacts: `results/localization_nullspace.json`, `results/localization_nullspace_esm2_35M.json`.

---

## 1. The question nine studies have made unavoidable

The probe's benign-side behaviour is graded by **localization** (`R` = 5.07 clean, 3.98 length-adjusted,
growing to 10.24 on an independent pool), by **length** in a U-shape, and by **provenance** (2.7×). Two
metadata fields reach **AUROC 0.8096** with no sequence information. The positive finding is **VFDB
registration**, not hazard: Motility is the most-flagged category at **99.05%**, and non-hazardous
categories are elevated **2.410×** against **2.356×** for partly-hazardous ones.

🔑 **Every one of those results is a decomposition of the same score. None of them asks the constructive
question**: if the localization axis is removed from the representation, does the membership contrast
survive, and is what survives any more hazard-shaped than what went in?

⚠️ **This study is not a proposed fix.** Removing a direction from an embedding is a measurement
technique, not a mitigation, and nothing here should be read as a repaired detector.

---

## 2. The intervention, frozen

**Iterative nullspace projection** (the standard construction: fit a linear localization classifier,
project its direction out, repeat, because localization is not a one-dimensional property and a rank-1
ablation does not remove it).

- **Labels**: benign pool proteins, **extracellular** against **intracellular**, under study E's *keyword*
  strata — the same rule study K and study Q used, so the numbers stay comparable to 2.418 and 2.410.
- **Direction fitting never sees the rows it is measured on.** Eligible test rows are split **50/50 by a
  fixed seed into folds A and B**; directions come from fold A, `R` is measured on fold B, and the whole
  thing is repeated with the folds swapped and averaged. Within fold A, an internal **75/25** split
  decides when to stop, so fold B is untouched until measurement.
- **Stopping rule**: stop when held-out localization AUROC **≤ 0.55**, or at **k = 20** directions,
  whichever comes first. `k` is reported.
- **Then** the class-axis probe is refitted on projected embeddings (positives + the benign training fold),
  recalibrated at the 95th percentile on the projected calibration fold, and every rate is remeasured.
  **VFDB never enters direction fitting.**

---

## 3. Y-0 — the gate, which is not a result

Localization AUROC on the held-out fold, before and after projection.

🔒 **If the post-projection AUROC exceeds 0.55, the method did not do its job and Y-1, Y-2 and Y-4 are
reported as indicative only** — as a failure of the technique, **not** as evidence the probe's signal
survives removal. That distinction is the whole reason this section exists.

🔒 **Prediction: the gate is met on both arms at k ≤ 20.**

---

## 4. Y-1 — does the localization gradient actually collapse?

`R` = extracellular over intracellular flag rate among benign, after projection, crude.

| band | reading |
|---|---|
| **≤ 1.5** | the gradient is removed |
| **1.5 – 3.0** | partly removed |
| **> 3.0** | ❌ the projection did not remove the gradient |

🔒 **Prediction: ≤ 3.0. I decline the ≤ 1.5 band**, and the reason is entry 76: **an AUROC at chance and a
tail rate at parity are different claims**, and study X needed a separate measurement to connect them.
Y-0's gate is about AUROC; `R` is a ratio of tail rates. **The gate being met does not entail `R` ≈ 1.**

---

## 5. Y-2 — the finding: does VFDB membership survive?

The contrast study K reported with provenance and localization held fixed: **VFDB-extracellular over
benign-extracellular**, which is **2.418** (650M) and **2.617** (35M) unprojected.

Retained fraction **ρ = (CA_projected − 1) / (CA_original − 1)**.

| band | reading |
|---|---|
| **ρ ≥ 0.70** | membership survives localization removal |
| **0.30 ≤ ρ < 0.70** | partial |
| **ρ < 0.30** | ❌ the membership contrast was largely the localization axis |

🔒 **Prediction: ρ ≥ 0.30, and the ≥ 0.70 band is declined.** The argument for a high ρ is real and comes
from the design: study K's contrast is **already within-stratum**, comparing extracellular VFDB proteins
to extracellular benign ones, so gross localization is controlled before this study starts. ⚠️ **The
argument against it is equally real**: within-stratum comparison does not remove the axis, and VFDB
proteins may simply sit **further along it** than benign secreted proteins. **I cannot settle that from
what is in the record, so the band stays wide.**

---

## 6. Y-3 — the control without which Y-2 means nothing

🔴 **Removing directions removes capacity.** If Y-2's ρ is low, that could be ablation damage rather than
anything about localization. So the same pipeline runs again with **k random orthonormal directions** —
same `k`, same folds, same seeds, everything else identical.

| band on ρ_random | consequence |
|---|---|
| **≥ 0.85** | 🔒 Y-2 is attributable to the localization axis |
| **< 0.85** | ❌ Y-2 is confounded by capacity loss and is reported as such |

🔒 **Prediction: ρ_random ≥ 0.85.** Derived: k ≤ 20 of 1,280 dimensions is **≤ 1.6%** of the space (4.2% on
35M), and study N found a perturbation of **2.1% along w** changed no conclusion.

---

## 7. Y-4 — is what survives any more hazard-shaped?

ρ computed separately for study Q's **non-hazardous** categories and its **partly-hazardous** ones, and for
**Motility** alone.

| band on ρ_non-hazardous − ρ_partly-hazardous | reading |
|---|---|
| **|Δ| ≤ 0.20** | what survives is no more hazard-shaped than what went in |
| **Δ ≤ −0.20** | ⭐ projection preferentially removes the non-hazardous signal |
| **Δ ≥ +0.20** | the non-hazardous signal is the more robust one |

🔒 **Prediction: |Δ| ≤ 0.20.** Derived from study Q's own numbers: unprojected, the two groups sit at
**2.410 and 2.356** (650M) and **2.309 and 2.530** (35M) — already within ~0.2 of each other, and a
projection acting on the shared axis should move them together.

---

## 8. Obligations, written before the data exists

- 🔒 **If Y-0's gate fails**, Y-1/Y-2/Y-4 are indicative, the entry leads with the **method's** failure,
  and no claim is made about signal surviving.
- 🔒 **If Y-3's control falls below 0.85**, Y-2 is reported as capacity-confounded, whatever its value.
- ❌ **If Y-2's ρ < 0.30 with the control clean**, the project's main positive result narrows again — from
  "partial on an independent pool" to "largely the localization axis" — and `paper/MANUSCRIPT.md` says so
  in § 6 and in the abstract.
- 🔒 **If ρ ≥ 0.70 with the gradient collapsed**, that is a **positive** result for the probe and is
  reported with the same prominence as the adverse ones. ⚠️ **It would not make the probe a hazard
  detector** — study Q's Motility finding is untouched by anything here.
- 🔒 **`k` is reported on both arms**, and if the two arms need very different `k`, that is stated rather
  than averaged.
- ⚠️ **No sentence anywhere may describe this projection as a fix, a mitigation, or a repaired probe.**
  The claims audit enforces that with a forbid.

---

## 9. Results — the frozen run, and why its verdict is refused

**Run 2026-09-30**, 30 seeds, two cross-fitted folds, both arms.
`results/localization_nullspace{,_esm2_35M}.json`.

| | 650M | 35M |
|---|---:|---:|
| directions removed, per fold | 9, 13 (**1.0%** of 1,280) | 8, 12 (**2.5%** of 480) |
| **Y-0** localization AUROC before → after | 0.9979 → **0.5443** | 0.9982 → **0.4966** |
| **Y-0 gate [≤ 0.55]** | ✅ **met** | ✅ **met** |
| **Y-1** `R` after projection | **4.590** [4.335, 4.845] | **4.313** [3.989, 4.638] |
| Y-1 band | ❌ **did not remove the gradient** | ❌ **did not remove the gradient** |
| **Y-2** CA after projection, ρ | 2.645, **ρ = 1.160** | 3.131, **ρ = 1.318** |
| **Y-3** control ρ | 1.017 ✅ clean | 0.987 ✅ clean |
| **Y-4** Δ(non-haz − partly) | +0.071 | −0.353 |

🔴 **Read literally, this document's § 5 now says "membership survives localization removal" with a clean
control.** ⚠️ **That claim is refused, and the reason is a check § 3 did not contain.**

### 🔴 Amendment 1 — Y-0b, and why the frozen gate was optimistic

§ 3's gate is measured on a **holdout of the fitting fold**, and the procedure **stops at the first
iteration where that estimate crosses 0.55**. 🔑 **A stopping rule that halts when a noisy held-out
estimate first dips below a threshold guarantees an optimistic final value** — it is a minimum over a
noisy sequence, and with 21 extracellular against 75 intracellular rows in the inner holdout, that noise
is around ±0.07.

**Y-0b asks the same question on rows the axis never saw**: fit a fresh localization classifier on the
residual of the *measurement* rows, hold out a quarter, report its AUROC. Nothing is tuned on it; it
decides only whether Y-1 and Y-2 are interpretable.

| | 650M | 35M |
|---|---:|---:|
| Y-0's gate, on the fitting fold's holdout | 0.5443 | 0.4966 |
| **Y-0b, on rows the axis never saw** | **0.7780** [0.706, 0.850] | **0.8247** [0.726, 0.923] |
| the random-direction control, same measure | 0.9582 | 0.9674 |

🔴 **The projection removed some localization and nowhere near enough.** ⚠️ **Y-1 is the same fact in a
different statistic**: `R` = 4.59 at a gate-measured AUROC of 0.544, which is precisely the dissociation
§ 4 declined the tight band for.

🔒 **So the verdict is refused**: Y-1, Y-2 and Y-4 are **indicative only**, and no claim is made that
membership survives localization removal — **because localization was not removed.** The artifacts carry
`Y0b_generalized: false` and `verdict_eligible: false`. ⚠️ **§ 8's first obligation keys on Y-0, which
passed, so the letter of this document permitted the claim. Its intent did not, and the intent is what is
followed.**

### 🔴 Amendment 2 — more labels did not help, which rules out one explanation

The obvious reading of Y-0b is **too few labels**: the keyword rule gives only **171** extracellular rows,
so directions are fitted on ~128 points in 1,280 dimensions. Study L's GO-augmented strata give **619**
and **2,708**. Re-run with those for the **axis only** — every measured rate stays on the keyword strata,
so 2.418 remains the comparator (`--labels coverage`,
`results/localization_nullspace{,_esm2_35M}_coverage_axis.json`):

| | 650M | 35M |
|---|---:|---:|
| axis labels | 619 / 2,708 | 619 / 2,708 |
| directions removed | 13, 15 | 18, 17 |
| Y-0 gate | 0.5452 ✅ | 0.5380 ✅ |
| **Y-0b on unseen rows** | **0.8901** | **0.8143** |
| Y-1 `R` | 4.670 | 4.300 |
| Y-2 ρ | 1.189 | 1.203 |

🔴 **Y-0b got *worse* on the larger arm, not better** — 0.778 → 0.890, against a control of 0.939. 🔑 **So
the failure is not a sample-size problem.** With 3.6× the labels and more directions removed, held-out
localization is **nearly as separable as under a random ablation of the same rank**.

🔒 **What that leaves is redundancy**: localization is encoded across enough of the space that removing
13–18 linear directions — 1–4% of it — does not flatten it, and a classifier refitted on the residual
finds it again. ⚠️ **This is a statement about the representation, not about the probe**, and it is the one
thing the frozen run does establish.

### 🔒 Predictions, scored honestly

| | frozen | actual | |
|---|---|---|---|
| Y-0 gate met at k ≤ 20 | | met, k = 8–18 | ✅ on its own terms |
| Y-1 ≤ 3.0, tight band declined | | **4.30 – 4.67** | ❌ **missed the wide band** |
| Y-2 ρ ≥ 0.30, ≥ 0.70 declined | | 1.16 – 1.32 | ⚠️ inside, but uninterpretable |
| Y-3 ρ_random ≥ 0.85 | | 0.987 – 1.017 | ✅ |
| Y-4 \|Δ\| ≤ 0.20 | | +0.071, −0.353, +0.012, +0.099 | ⚠️ 3 of 4, and none is a verdict |

🔑 **Y-3's band was derived from study N's measured 2.1%-along-w perturbation and held. Y-1's was reasoned
about, and missed.** ⚠️ **And Y-1's miss was foreseeable from § 4's own sentence**: having written that a
chance AUROC and a parity tail-rate are different claims, I still predicted a band for `R` from a gate
about AUROC. **The argument was in the document and I did not apply it to my own prediction.**

---

## 10. Y-5 — frozen 2026-09-30, after the run above and before any curve was computed

🔒 **The question survives its first method.** Y-5 removes the stopping rule, which is the thing that
failed, and replaces it with a **curve**.

- **No stopping rule, so no selection.** The iterative procedure runs to the largest grid value, giving a
  **nested** sequence of directions, and every prefix is read off the same pass.
- **Grid**: k ∈ {0, 2, 5, 10, 20, 40, 80, 160}, capped at **d / 4** (320 on 650M, 120 on 35M).
- **At each k**: Y-0b's unseen-row localization AUROC, `R`, and CA with ρ. 10 seeds on the curve.
- **Read-off rule, frozen**: `k*` is the smallest grid k whose unseen-row AUROC is **≤ 0.55 on both
  folds**. At `k*`, everything is recomputed at **30 seeds**, with a random-direction control of the same
  rank, and Y-1's and Y-2's **existing bands** apply unchanged.
- **If no grid k reaches 0.55**, that is the result: **the localization axis is not linearly removable
  within d/4 directions**, and Y-2 is not answerable by this method. 🔒 **Reported as a finding, not as a
  failed run.**

### 🔒 Predictions

- 🔒 **`k*` exists and lands in [20, 160] on 650M.** ⚠️ **This one is reasoned, not derived** — 13
  directions leave AUROC at 0.890, and flattening a redundantly encoded concept is known to take tens of
  directions — **and this session's record says reasoned predictions are the ones that miss.**
- 🔒 **At `k*`, `R` > 1.5.** Derived: `R` is 4.59 at AUROC 0.778 and 4.67 at 0.890, so the two statistics
  are already visibly decoupled, and entry 76 established why they can be.
- 🔒 **ρ at `k*` ≥ 0.30.** ⚠️ Wide on purpose; the record cannot narrow it.
- 🔒 **The control at `k*` keeps ρ_random ≥ 0.85**, and if `k*` is large enough that it does not, **the
  curve is reported and Y-2 stays unanswered** rather than being read off a damaged representation.
