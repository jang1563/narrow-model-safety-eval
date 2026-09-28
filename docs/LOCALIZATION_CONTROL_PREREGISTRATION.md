# Preregistration — is it hazard, or is it just being on the outside of the cell?

**Frozen 2026-09-28, before any localization annotation has been fetched.** One primary test, one
second-arm replication, one declared-exploratory observation. Append-only.

---

## 0. The caveat this exists to close

Entry 52 closed most of the provenance question and left this in its place, verbatim:

> ⚠️ **What this does not establish**: that the remaining three quarters is *hazard* rather than
> another property virulence factors share — secretion, surface exposure, host interaction.
> Separating those needs negatives matched on **localization** as well as organism.

Study D removed pathogen origin as the explanation for about a quarter of the benign-to-virulence
gap. **Localization is the next candidate for the rest**, and it is the more dangerous one: a probe
that separates *secreted and surface-exposed* proteins from *cytoplasmic* ones would reproduce most
of this project's results without containing any notion of hazard at all.

---

## 🔴 0.1 One input to this document is already contaminated, and is quarantined accordingly

Study B's per-category recoveries are **already published** in `results/external_class_axis_lomo.json`
and I have read them before writing this. Grouped by hand, they look like this on both arms:

| | ESM-2 650M | ESM-2 35M |
|---|---:|---:|
| Motility, Exotoxin, Effector delivery system, Adherence, Invasion | 0.74 – 0.85 | 0.58 – 0.75 |
| Regulation, Stress survival, Post-translational modification, Nutritional/Metabolic | 0.39 – 0.54 | 0.24 – 0.40 |

🔴 **That grouping was made after seeing those numbers, so it cannot be evidence for the hypothesis it
suggested.** It is recorded here as **E-0, exploratory**, and is barred from the verdict. Naming it in
advance is what stops it being quietly re-presented later as a confirmation.

It is also **not obviously about hazard**: Motility at 0.85 is the single best-recovered category on
650M, and flagellar proteins are surface-exposed but not toxic. That is the observation that makes the
localization reading worth a real test.

---

## 1. Design

⚠️ **No new probe and no new inference.** Study B's fold is used exactly as `src/83` uses it —
1,000 train / 500 calibrate / 6,758 test, 30 seeds, threshold at the 95th percentile of the
calibration scores. The pool is already embedded on both arms. **The only new thing is an
annotation join.**

The test is on the **benign side**, where the hypothesis makes a sharp prediction that nothing
already computed can have leaked into: *if the probe is a localization detector, then benign
extracellular proteins from non-pathogens must be flagged far above benign cytoplasmic ones.*

### 1.1 Strata — frozen UniProt controlled-vocabulary rule, evaluated in this order

| stratum | rule (first match wins) |
|---|---|
| **extracellular** | keyword `Secreted`, `Signal`, `Cell wall`, `Cell outer membrane`, `Fimbrium`, or `Flagellum` |
| **membrane** | otherwise, keyword `Cell membrane` or `Membrane` |
| **intracellular** | otherwise, keyword `Cytoplasm`, `Periplasm`, `Nucleus`, or `Cytoplasmic vesicle` |
| **unannotated** | otherwise — no localization keyword of any kind |

🔒 **Population**: Bacteria and Archaea only — the kingdoms VFDB draws from. "Secreted" does not mean
the same thing in a virus. The all-kingdom version is secondary and reported alongside.

🔒 **`unannotated` is reported, never dropped.** Conditioning on annotation availability is exactly
the error that produced A2's provenance confound (entry 36); a stratum that exists only where a
curator wrote something is not a random sample of anything.

---

## 2. Primary test, frozen bands

**E-1 (primary)**: `R = ` flag rate(extracellular) `/` flag rate(intracellular) in the pool test
partition, mean over 30 seeds, ESM-2 650M.
**E-2 (second arm)**: the same on ESM-2 35M. Per this project's standing rule, **a verdict requires
both arms to land in the same band.**

| band | reading |
|---|---|
| **R ≤ 1.5** | 🟢 localization is not a material driver; the virulence elevation survives it |
| **1.5 < R < 3.0** | ⚠️ partial — report as a fraction of the pool→VFDB gap, in study D's arithmetic |
| **R ≥ 3.0** | 🔴 localization is a major driver |

🔴 **What the adverse band obliges, written down before the number exists**: if R ≥ 3.0 on both arms,
then the hazard reading of A1, B, C and D is **not supportable as stated**, and this repository's
claim that the probe responds to virulence must be qualified wherever it appears — the audited
documents, `paper/MANUSCRIPT.md`, and the abstract — in the same commit that records the result.

### 2.1 Frozen prediction, so it can be scored the way study D's was

**R ≈ 2.0** — extracellular ≈ 11%, intracellular ≈ 5.5% — landing in the **partial** band, with
localization explaining **more of the gap than provenance's 22.7% but still a minority (25–45%)**.

### 2.2 Floors and ceilings

- 🔒 **Floor**: if either the extracellular or the intracellular stratum holds < 300 proteins, E-1 is
  **underpowered** and is reported as indicative, not as a verdict.
- 🔒 **Ceiling**: if `unannotated` exceeds 50% of the eligible pool, the annotated contrast is not
  representative of the pool and that caveat is carried in every statement of the result.
- 🔒 **Annotation-availability check**: if the `unannotated` flag rate differs from the pooled
  annotated rate by more than 1.5×, annotation availability itself tracks the score, and E-1's
  strata are confounded by it.

### 2.3 Frozen confound rule — length

Length has already produced one false lead in this project (the representative rule, entry 45).
🔒 Compute length-alone AUROC for extracellular vs intracellular. **If it exceeds 0.65, E-1 is
declared length-confounded** and the verdict comes instead from a length-matched subsample
(nearest-neighbour matching on length, 1:1, without replacement).

---

## 3. E-3, the decomposition

To keep this commensurable with study D, report the same arithmetic: what fraction of the distance
from the pool's overall rate to the VFDB rate (73.49% on 650M) is spanned by the extracellular
stratum's elevation above the intracellular one.

---

## 4. E-0 — the exploratory observation, and the one thing that could make it informative

Margin already predicts category recovery at ρ = +0.68 (B-1). 🔒 **So E-0 reports the
localization grouping's correlation with recovery both raw and partialled on margin.** If the
grouping adds nothing beyond margin, it is not an independent explanation of anything — it is
B-1 relabelled. This stays exploratory in either case.
