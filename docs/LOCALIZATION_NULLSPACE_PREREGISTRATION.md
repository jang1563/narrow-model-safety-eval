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

## 9. Results

*(to be appended after the run, append-only)*
