# Do the verdicts survive a second draw of negatives? — preregistration

**Frozen 2026-10-06, before `src/136` existed.** Study AR.

## 1. The weakness this closes

🔴 **All seven DeepVIC studies share one negative partition.** `src/125`, `127`, `130`, `131`, `132`,
`133` and `135` each reproduce `default_rng(12345)` bit-for-bit — deliberately, so their recoveries are
comparable — ⚠️ **with the consequence that the whole arm rests on one draw of 33,456 negatives.**

🔑 **This project's own history says that matters.** Study E's localization result was re-measured on an
**independent pool** and changed: `C/A` fell from 2.418 to 1.594 and `R` **grew** from 7.682 to 11.774
(§ 6). A second draw is not a formality here.

## 2. The measurement

One new partition — `default_rng(67890)` in place of 12345, same two-thirds/one-third split of the same
33,456 negatives — and `src/133`'s fold re-run on it at **both** classifier settings:

- **fixed `C = 1.0`**, which is what studies AI-2 through AO used;
- **nested-selected `C`**, which is study AP's repair.

🔒 One pass therefore re-tests two things the paper leans on hardest: **AI-2's margin ordering** (the only
externally replicated result) and **AP's tuning repair** (the newest and most consequential).

⚠️ **What this does not re-test**: study AM's within-fold ordering AUROC and study AQ's fair comparison
keep the original partition, and that is stated rather than implied.

## 3. Frozen bands

Floors **and** ceilings, on **both** arms and at **both** classifier settings. Bonferroni α = 0.05/3.

| | survives | does not |
|---|---|---|
| **AR-1** *ρ*(margin, recovery) on the new draw | **≥ +0.60** in all four cells | **≤ +0.20** in any cell |
| **AR-2** tuning gain, new draw | **≥ +3pp** on both arms | **≤ +0.5pp** on either |
| **AR-3** mean absolute per-category shift between draws | no band — reported as the sensitivity | |

🔒 **A verdict needs both arms, at both settings, in the same column.** Between ⇒ indicative.

🔑 **AR-3 is the number this study exists to produce.** AI-2's ρ and AP's repair are *verdicts about
orderings and differences*; the paper also prints **levels**, and nothing here has ever measured how much
a level depends on which negatives were drawn.

## 4. Predictions

**P1 (derived — the ordering survived a change of classifier entirely, from `C = 1.0` to a tuned `C` to a
parameter-free centroid, at ρ +0.75 to +0.95).** AR-1 clears **+0.60** in all four cells: a re-draw of
negatives is a weaker perturbation than replacing the classifier.

**P2 (derived — the repair came from capacity, not from the particular negatives).** AR-2 reproduces at
**≥ +3pp** on both arms.

**P3 (reasoned).** AR-3 exceeds **3pp**: the levels are draw-sensitive even where the verdicts are not,
which is why the paper should quote orderings and differences rather than levels.

⚠️ **The computed record stands at derived 13 of 25 and reasoned 3 of 10**
(`results/prediction_record.json`), and claim 147 requires § 7 to match it. 🔑 **P1 and P2 are
mechanism-derived, and entry 118 recorded that a mechanism claim is neither reliably safe nor reliably
wrong.**

## 5. What each outcome changes

- **All three survive** ⇒ § 7's "one negative partition" limitation is replaced by a measurement, and the
  arm no longer rests on one draw.
- **AR-1 fails in any cell** ⇒ 🔴 the externally replicated margin result is draw-dependent, and § 4's
  external confirmation is withdrawn to indicative.
- **AR-2 fails** ⇒ study AP's repair is a property of one partition, and § 6's tuned figures come out.
- **AR-3 large** ⇒ every level in the arm carries a draw-sensitivity figure beside it.
