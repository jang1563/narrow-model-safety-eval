# Does the geometry let you do anything? — preregistration

**Frozen 2026-10-08, before `src/145` existed.** Study AW. 🔒 **Bands checked for reachability before
freezing** — see § 5.

## 1. The question this project has not asked

Every result so far **measures**. Margin orders which held-out class a probe will fail on; R says when the
negative half of the formula is worth subtracting. 🔑 **Nothing has tested whether any of it is useful.**

🔴 **And this project's own evidence predicts that it is not.** Removing the benign neighbours the
geometry implicates repairs **8–11%** of a failure the geometry predicts at *ρ* = +0.894 (§ 4 of the
paper): **prediction and repair dissociate.** ⚠️ **A margin-routed mixture is a repair built on a
prediction, which is the move the dissociation says does not pay.** 🔒 **So this study expects to fail,
and says so before running.**

## 2. What is NOT being claimed

🔴 **The member-level margin was preregistered, frozen, and failed on two external panels** — 82% against
a predicted ≤ 40% at the same absolute margin. ⚠️ **So nothing here may rest on a per-query *hazard*
claim.** 🔒 **Routing is not a hazard claim**: a query is assigned to an expert by its nearest cluster
centroid, which is a question about where it sits, not about what it is. 🔑 **The allocation decision —
which clusters get an expert — is made once, offline, before any query exists.**

## 3. The design

🔒 **`src/125`'s fold on DeepVIC, unchanged**: 12,989 positives in 14 curator categories, 33,456
negatives, union/test partition at seed 12345, 30 seeds, leave-one-**category**-out, both ESM-2 arms.

1. **Cluster**, unsupervised: k-means with **k = 7** on the frozen positive embeddings, fitted on the
   *trained* positives only. 🔑 **k = 7 is half of 14 so clusters cannot align with the curator
   categories** — if they did, the mixture would be given the answer.
2. **Score each cluster** with `margin(cluster)`, the same formula with clusters in place of categories.
3. **Allocate** a budget of **m = 3** experts by one of four strategies, then fit:
   - 🔒 **`margin`** — the 3 **lowest-margin** clusters get their own probe;
   - **`random`** — 3 clusters chosen by seed, the budget control;
   - **`size`** — the 3 **largest** clusters, the size control;
   - **`none`** — one probe for everything, which is **study AP's *tuned* single probe**.
4. **Route** every row, positive and negative, to the expert of its nearest cluster centroid; rows whose
   cluster has no expert go to the global probe.
5. **Threshold the mixture's scores** at the 95th percentile of the calibrate negatives, so the operating
   point is matched exactly as everywhere else in this project.
6. **Score** mean recovery over the 14 held-out categories.

🔒 **The comparison is against the TUNED single probe.** 🔴 **An unexamined `C = 1.0` cost up to 24
points per category** (study AP), so a mixture that beats the untuned probe has beaten a bug this project
has already corrected. ⚠️ **Every arm here selects `C` by `src/133`'s nested rule.**

⚠️ **A mixture has more parameters than one probe.** 🔒 **Both comparisons are reported**: at matched
false-positive rate, and with the parameter count stated beside each — because a win on the first and a
loss on the second is the likely shape and the paper should not print only one.

## 4. Frozen bands

Bonferroni α = 0.05/3.

| | routing works | it does not |
|---|---|---|
| **AW-1** `margin` − `none`, mean recovery | **≥ +5pp** | **≤ +1pp** |
| **AW-2** `margin` − `random`, same budget | **≥ +3pp** | **≤ +1pp** |
| **AW-3** `margin` − `size`, same budget | **≥ +3pp** | **≤ +1pp** |

🔒 **Both arms must agree on all three**, and 🔑 **AW-2 is the one that matters**: AW-1 can be won by
having more parameters, and only AW-2 isolates whether *margin* picked the right clusters.

## 5. 🔒 Reachability, checked before freezing

🔑 **Three preregistrations in a row carried a band that could not do what it was written to do**, so
`src/143_band_reachability.py` was run on these before they were frozen. ⚠️ **It flags the two-arm
conjunction**:

    AW-1/AW-2 (two-arm agreement)  n_units: 2  attainable_max: 2
    🔴 SINGLE-UNIT VETO: stated as a conjunction over all 2 units, so one marginal unit
       decides the study-level verdict

🔒 **The warning is accepted, with a reason.** The second-arm rule exists *precisely* so that one arm can
veto: it killed three findings in two days and then killed this paper's own § 6 narrowing. 🔑 **A veto by
one arm is the intended behaviour here and not a defect** — which is what a reachability warning is for,
to be answered rather than to be obeyed.

⚠️ **The three bands are continuous differences, not counts against a denominator**, so the defect that
sank AT-1b, AU-3 and AV-1 cannot arise. 🔒 **Stated because that was checked, not assumed.**

## 6. Predictions

**P1 (reasoned — the dissociation: repair recovers 8–11% of what prediction predicts).** 🔴 **AW-2 lands
in the "does not" band, ≤ +1pp.** The mixture will not beat a random allocation of the same budget.

**P2 (derived — from study AP: the single probe's `C` was the binding constraint, worth up to 24 points,
and a mixture gives each expert fewer positives to fit).** `margin` − `none` is **positive but under
+5pp**, i.e. AW-1 lands *between* bands.

**P3 (derived — k-means on 12,989 embeddings with k = 7 against 14 curator categories).** The mean
cluster-to-category purity is **below 0.50**, confirming the clusters are not the categories in disguise.

**P4 (reasoned).** The three lowest-margin **clusters** contain a disproportionate share of the members of
beta-lactamase and the phage class — the two categories § 2 of the paper reports as the screen's failures.

⚠️ **The computed record stands at derived 18 of 35, reasoned 4 of 13**, so a reasoned prediction here is
close to a coin flip and P1 is labelled reasoned for that reason.

## 7. What each outcome changes

- **AW-2 ≤ +1pp**, as predicted ⇒ 🔑 **the geometry predicts and does not repair, a second time and by a
  second route.** The paper's dissociation stops being one measurement and becomes the finding; margin is
  a **triage signal for where to spend evaluation effort** and explicitly not a component to build with.
- **AW-2 ≥ +3pp on both arms** ⇒ 🔑 **the prediction is actionable after all**, which would be the most
  consequential result in the project and would need its own replication on a second benchmark before the
  paper leads with it. 🔒 **Named now so the bar is set before the number exists.**
- **AW-1 ≥ +5pp with AW-2 ≤ +1pp** ⇒ the mixture helps because it has more parameters, not because margin
  chose. ⚠️ **That is a result about model capacity and must not be reported as a result about geometry.**
