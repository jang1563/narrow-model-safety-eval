# Can the estimation gap be repaired? — preregistration

**Frozen 2026-10-02, before `src/133` existed.** Study AP. 🔑 **The first study here that attempts an
improvement rather than a measurement.**

## 1. What study AO located

AO showed the +7-point increment is real information (**AO-1** +7.02 / +7.51pp) and that composition
adds nothing on top of the embedding (**AO-2** −0.16 / −0.03pp) — ⚠️ **while composition alone beats the
embedding by 20 points on Exoenzyme and the concatenated model recovers −0.86pp of that gap.** The
twenty-one composition columns carry **1.54–1.91%** of the coefficient mass against **1.614%** by column
count: the probe does not distinguish them from an average embedding column.

🔒 **So the deficit is estimation, not information**, and estimation deficits are the kind that a
modelling choice can sometimes fix. This asks whether this one can.

## 2. The two levers, and the one rule that matters

- **Regularization strength.** `LogisticRegression(C=1.0)` has been fixed in every study here. Sweep
  `C ∈ {0.01, 0.1, 1.0, 10.0, 100.0}`.
- **Dimensionality.** PCA the embedding to **k = 50**, fitted on the **negative union only** so no
  held-out positive informs the basis.

🔴 **The rule that makes this a study and not a search: the hyperparameter is selected without the
held-out category.** For each held-out category, `C` is chosen on a random member split of the **other
thirteen** categories plus train negatives, at 5 inner seeds; the selected value is then scored on the
held-out category at the usual 30 seeds. ⚠️ **No number from the scored category reaches the selection.**

Baselines, already in the record: fixed `C = 1.0` gives **41.43% / 41.78%** mean recovery (study AI-2),
and composition gives **34.25%** (study AJ).

## 3. Frozen bands

Floors **and** ceilings, on **both** arms. Bonferroni α = 0.05/2.

| | repaired | not repaired |
|---|---|---|
| **AP-1** mean recovery, nested-selected `C`, minus the fixed-`C` baseline | **≥ +3pp** | **≤ +0.5pp** |
| **AP-2** share of the Exoenzyme gap closed (composition 62.3% − embedding) | **≥ 50%** | **≤ 10%** |

🔒 **A verdict needs both arms in the same column.** Between ⇒ reported as a quantity.
⚠️ **AP-1 may be negative**: nested selection can pick worse than a fixed default, and that is a result
rather than a bug.

## 4. Predictions

**P1 (derived — from AO's coefficient-mass measurement).** AP-1 lands in the **not repaired** band,
**≤ +0.5pp**, on both arms. The mechanism is explicit: under standardization L2 shrinks all coefficients
**proportionally**, so no choice of `C` can selectively raise twenty-one columns relative to 1,280. 🔑
**If this is right, regularization strength is the wrong lever and the paper should say so with a
number.**

**P2 (derived — same mechanism).** AP-2 lands in the **not repaired** band, **≤ 10%** of the Exoenzyme
gap closed by the `C` sweep.

**P3 (reasoned).** The PCA arm at k = 50 does **better** than the `C` sweep on Exoenzyme, because
reducing to fifty dimensions changes the ratio of parameters to transferable signal in a way `C` cannot.

⚠️ **This project's repairs have a record and it is poor**: two grew the wrong side of the comparison,
and the one that worked (`win_best25`) is real on one class of two. 🔑 **P1 and P2 predict failure**, and
they are derived from a mechanism measured two entries ago rather than from the record's mood.

🔑 **Running total across AH through AO: derived 9 of 16, reasoned 2 of 7.**

## 5. What each outcome changes

- **Repaired** ⇒ 🔴 every per-category recovery in this paper is a statement about `C = 1.0` and not about
  the representation, and § 3's panel-dependence finding gains a hyperparameter dimension. **That would be
  the most consequential result in the paper and it is the one P1 and P2 predict against.**
- **Not repaired** ⇒ the estimation deficit is real and not a tuning artifact; § 6 can say the probe
  ignores available information and no obvious knob changes it.
- **PCA wins where `C` does not** ⇒ the lever is dimensionality, and the paper reports which.

---

## Amendment 1 — the grid selected at its own edge, and the control that would reframe the result

**Added 2026-10-03, after the § 3 bands were applied and before the extended grid was run.** The only
facts used here are the selected values and the verdict, both already recorded.

### 🔴 The sweep is not finished

`C ∈ {0.01, 0.1, 1, 10, 100}` returned **repaired** on both arms — 41.43% → **55.39%** on 650M and
41.78% → **46.91%** on 35M, at out-of-sample false-positive rates of 4.88–5.23%. ⚠️ **But the selection
sat at the grid's lower edge: 0.01 for 14 of 14 categories on 650M and 12 of 14 on 35M.** 🔴 **A boundary
selection means the optimum may lie outside the grid, so none of those numbers can be stated as what
tuning achieves — only as a lower bound on it.**

🔒 The grid is extended downward to **`C ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100}`**, with the selection
rule unchanged: chosen on a member split of the other thirteen categories, never on the scored one.

**AP-3 (frozen here).** Does the extended grid select in the **interior**? ⚠️ **If it again selects at
the new edge (1e-4) for a majority of categories, the sweep is still unfinished and every figure is
reported as a lower bound, not as a tuned result.**

### 🔑 The control that may reframe the whole thing

At `C = 0.01` on standardized features a logistic regression is close to a **shrunken mean-difference
classifier**: the penalty dominates the likelihood and the solution approaches a scaled difference of
class means. 🔴 **If that is what is happening, "tune `C`" is the wrong description and "the paper used an
over-parameterized classifier" is the right one.**

🔒 **AP-4 (frozen here).** A **nearest-centroid** classifier — the difference of class means, no
regularization parameter at all — is run on the identical fold and scored the same way.

| | reading |
|---|---|
| nearest-centroid **within 2pp** of the swept probe, both arms | the finding is about **model class**, not about a hyperparameter, and the paper should say so |
| nearest-centroid **more than 5pp below** the swept probe | the shrunken solution is doing something a centroid cannot, and `C` is a real lever |

⚠️ Between ⇒ reported as a quantity.

### Predictions

**P4 (derived — from 0.01 beating 0.1 on 14 of 14 categories, which is a monotone trend at the edge).**
The extended grid selects **below 0.01** for a majority of categories on both arms, and mean recovery
rises by a further **≥ 1pp** on at least one arm.

**P5 (derived — from the penalty dominating at small `C` on standardized features).** AP-4's
nearest-centroid lands **within 2pp** of the swept probe on both arms.

⚠️ **The record, now computed from the artifacts rather than carried in prose** (entry 112's defect,
which recurred in entry 115): **derived 10 of 20, reasoned 3 of 9.** 🔑 **P1 and P2 of this study were
mechanism-derived and both failed**, which is why P4 and P5 are stated as mechanism claims that can fail
the same way.
