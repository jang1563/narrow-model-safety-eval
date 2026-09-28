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

---

## Amendment 1 — 2026-09-28, on running it: § 3 asked for one thing and called it another

§ 3 specified "the extracellular stratum's elevation **above the intracellular one**" and in the same
sentence called it "the same arithmetic" as study D. It is not. Study D computed
`(control − pool) / (vfdb − pool)`, against the **pool's overall rate**, not against another stratum.

🔒 Both are now reported and neither replaces the other: `contrast` is § 3 as written, `study_d_form`
is the figure that is actually comparable to study D's 22.7%. **Named rather than silently switched
to whichever one reads better.**

---

## Results, 2026-09-28

`src/84_localization_control.py`, study B's fold unchanged, 30 seeds, no new inference.
`results/localization_control.json`, `results/localization_control_esm2_35M.json`.

| stratum | *n* | ESM-2 **650M** | ESM-2 **35M** |
|---|---:|---:|---:|
| ⭐ **extracellular** | **652** | **21.73% [20.40, 23.06]** | **14.83% [14.05, 15.60]** |
| membrane | 495 | 16.84% [15.45, 18.23] | 13.45% [12.45, 14.46] |
| ⭐ **intracellular** | **1,666** | **4.15% [3.77, 4.53]** | **2.98% [2.73, 3.23]** |
| unannotated | 3,434 | 4.32% [3.91, 4.73] | 2.42% [2.24, 2.59] |
| all eligible (Bacteria + Archaea) | 6,247 | 7.08% | 4.74% |

### 🔴 E-1 and E-2: the adverse band, on both arms

**R = 5.356 [5.108, 5.604]** on 650M and **5.130 [4.814, 5.446]** on 35M. The frozen ceiling was
**3.0**. Both arms are in the adverse band, so by § 2's second-arm rule **this is a verdict, not an
indication.**

🔒 **The length rule did not save it.** Length-alone AUROC is **0.607**, inside the frozen 0.65
tolerance, so E-1 is not declared length-confounded — and the length-matched subsample (652 pairs,
1:1 nearest-neighbour) still gives **R = 4.378** on 650M and **3.400** on 35M, both above the ceiling.

🔒 **The prediction in § 2.1 was wrong, and not by a little.** It said **R ≈ 2.0**, partial band.
The answer is **5.36**, nearly three times that, in the band the document called adverse. Recorded as
a miss.

### 🔴 Two frozen guards fired, and they cut against the strata

- **Ceiling breached**: `unannotated` is **55.0%** of the eligible pool, over the frozen 50%. The
  annotated contrast is **not representative of the pool**, and that caveat travels with every
  statement of this result.
- **Availability check failed**: the availability ratio is **0.41** (650M) and **0.32** (35M), outside
  the frozen [0.67, 1.5]. **Annotation availability itself tracks the score.**
- Floor met: 652 and 1,666, both over 300.

⚠️ The unannotated rate (4.32%) sits almost exactly on the intracellular rate (4.15%), which is
*consistent with* unannotated Swiss-Prot bacterial proteins being predominantly cytoplasmic proteins
no curator got to. **That is a reading, not a demonstration** — nothing here establishes it, and the
ceiling stands regardless.

### ⚠️ E-3: a huge ratio, and a share of the gap that is nonetheless about a quarter

| | share of the pool→VFDB gap |
|---|---:|
| localization, § 3 as written (extracellular − intracellular) | **26.5%** |
| localization, study D's arithmetic (extracellular − pool) | **22.1%** |
| provenance, study D | 22.7% |

🔴 **The band labels in § 2 conflated two different things and this result separates them.** The
bands were set on a **ratio** and labelled with a conclusion about **share**. Both readings are true
at once, and both are now on the record:

- **The gradient is enormous.** A benign secreted or surface-exposed bacterial protein is **5.4×**
  more likely to be flagged than a benign cytoplasmic one. On the benign side the probe behaves
  substantially like a localization detector.
- **The share is about a quarter.** Against a localization-matched benign baseline of **21.73%**,
  VFDB virulence factors are still flagged at **73.49%** — so **roughly three quarters of the gap
  is still not localization**, almost exactly what was left over after provenance.

🔒 **The rule as written governs.** R ≥ 3.0 on both arms, so § 2's obligation is live and is being
executed: every claim that the probe responds to virulence is qualified in the same commit that
records this. **Switching to the share statistic now, because it reads better, is precisely the move
the frozen band exists to prevent.**

### What this does and does not license

🟢 Licensed: the probe's **false positives are strongly structured by localization**, and any
deployment reading of its 5% nominal rate is wrong for secreted and surface proteins, where the real
rate is 22%.
🔴 Licensed: **"the probe detects virulence" is not supportable unqualified**, and is now qualified
everywhere it appears.
⚠️ Not licensed: that localization *explains* the virulence separation. It accounts for about a
quarter of it. Provenance accounted for about another quarter. **The two have not been shown to be
independent** — pathogen-derived proteins are themselves enriched for secretion, so these shares
cannot simply be added, and the joint decomposition has not been run.

---

## 🔒 Superseded on this point — 2026-09-28, later the same day

The sentence above saying the joint decomposition **has not been run** was true when written and is
no longer. `docs/JOINT_DECOMPOSITION_PREREGISTRATION.md` ran it: provenance and localization are
**multiplicatively independent** (RR = 1.001 on 650M, 1.364 on 35M, both inside the frozen band), and
**together they span 42.1% of the pool→VFDB gap on 650M and 32.2% on 35M** — measured, not summed.
🔴 **22.7% + 22.1% remains wrong** and remains forbidden by claim 98. The text above is left as
written rather than edited, per this repository's append-only rule.

---

## 🔴 Amendment 2 — 2026-09-28: this study did not use the fold it said it used

`src/83` takes a fallback branch that `src/84` and `src/85` did not have. The screen rejects 32 of the
1,500 partition rows for similarity to positives, leaving **1,468 admitted rows — fewer than
`N_TRAIN + N_CAL`** — so `src/83` splits 978 train / 490 calibrate, while a fixed `perm[:1000]` /
`perm[1000:1500]` slice silently trained on 1,000 and **calibrated on 468**.

🔴 **"Study B's fold unchanged" was therefore false** in this document, in the script's docstring, in
entries 53 and 54, in the manuscript and in two commit messages. Corrected in `src/84` and `src/85`,
which now take the same fallback, and everything re-run on both arms.

🟢 **Nothing it affected changed a verdict.** Every quantity moved in the third decimal; the largest
single move was 35M's RR, by −0.037. **The corrected numbers are below and supersede the tables above.**

| stratum | *n* | ESM-2 **650M** | ESM-2 **35M** |
|---|---:|---:|---:|
| ⭐ **extracellular** | **652** | **21.75% [20.46, 23.05]** | **15.30% [14.47, 16.12]** |
| membrane | 495 | 16.90% [15.55, 18.25] | 13.89% [12.90, 14.89] |
| ⭐ **intracellular** | **1,666** | **4.13% [3.77, 4.49]** | **3.08% [2.82, 3.34]** |
| unannotated | 3,434 | 4.28% [3.90, 4.66] | 2.52% [2.33, 2.70] |
| all eligible | 6,247 | 7.06% | 4.90% |

🔒 **Corrected E-1 / E-2: R = 5.381 [5.134, 5.628] on 650M and 5.151 [4.786, 5.516] on 35M.** Both
still in the adverse band, so the verdict and its § 2 obligation are unchanged. Length-matched
**4.382** and **3.430**, both still over the 3.0 ceiling. Length AUROC **0.607**, unannotated
**55.0%**, availability **0.41** / **0.32** — every guard resolves exactly as before.
🔒 **E-3 is unchanged at one decimal**: **26.5%** as § 3 wrote it, **22.1%** in study D's arithmetic.
