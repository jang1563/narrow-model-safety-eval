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
