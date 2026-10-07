# Margin as a routing signal: what would have to be true

**Sketch, 2026-10-06. Not a preregistration** — the decisive experiment depends on study AT, and this
document exists so the design is written down before the result that would motivate it.

## 1. Why this is even a candidate

🔒 **Margin is computable before any training.** That is the whole content of the result: a quantity read
off a frozen representation and a class partition orders which held-out class a probe will fail on, at
*ρ* +0.947/+0.886 on DeepVIC and +0.951/+0.813 on 20 Newsgroups. 🔑 **A quantity available before
training is, in principle, a routing signal** — routing is exactly a decision made before the expert runs.

🔑 **And the MoE literature has arrived at the same geometry from the other side.** Routers in sparse MoE
have been shown to accumulate toward *centroids* of the inputs routed to each expert, to the point that
"much of the router's role should be recoverable by replacing standard gating weights with non-learnable
centroids" ([arXiv 2605.12476](https://arxiv.org/abs/2605.12476)); rank-1 experts under cosine-similarity
routing are monosemantic by construction and their specialization directly inspectable
([arXiv 2604.14434](https://arxiv.org/abs/2604.14434)). ⚠️ **Those are claims about routing inside a
trained MoE, learned end to end** — not about using a pre-training geometric quantity to decide which
classes need an expert at all.

🔒 **This project already has the centroid number.** Study AP ran a centroid classifier against the tuned
probe on DeepVIC: it trails the swept probe by **6.80pp** on ESM-2 650M and by **0.39pp** on 35M — a tie
on the smaller arm — and margin's ordering survives on centroid scores at *ρ* **+0.745** and **+0.626**,
both above the +0.60 floor. 🔑 **So the object a router approximates performs about as well as a tuned
linear probe here, and the margin ordering holds on it.**

## 2. 🔴 The constraint that kills the obvious version

The obvious design is per-query abstention: for a query *x*, compute

    margin(x) = max cos(x, trained positives) − max cos(x, negatives)

and route or abstain on it. 🔴 **That is the member-level margin, and this project preregistered it,
froze it, and it FAILED on two external panels** (§ 4 of the paper; one genuine failure at 82% against a
predicted ≤ 40% at the same absolute margin, one uninformative panel at a ceiling). ⚠️ **The class-level
result does not license a per-query signal**, and any MoE design that needs one is already refuted by this
project's own record.

🔑 **So the only honest MoE reading is about allocation, not about queries.** Margin says which *class* a
single global probe will fail to generalize to. The decision it can inform is **which clusters deserve
their own expert, their own calibration, or their own validation set** — made once, offline, before
training, with no per-query claim attached.

## 3. The design

🔒 **Unsupervised, because the trained-class labels are the thing a deployment would not have for the
held-out family.**

1. Cluster the positive embeddings into *k* groups with no labels (k-means on the frozen vectors).
2. For each cluster, compute `margin(cluster)` against the remaining positives and the negatives — the
   same formula, with clusters in place of curator categories.
3. Allocate a fixed budget of *m* experts to the *m* **lowest-margin** clusters; everything else is served
   by one global probe.
4. Score the mixture, leave-one-true-class-out, at the matched 5% false-positive rate.

### 🔒 What it must be compared against, or the result is meaningless

| rival | why it is the one that matters |
|---|---|
| **AP's *tuned* single probe** | 🔴 An unexamined `C = 1.0` cost up to **24pp** per category. A mixture that beats the *untuned* probe has beaten a bug, and this project has already published the correction. |
| **the same budget allocated at random** | the claim is that *margin* picks the right clusters, not that more experts help |
| **the same budget allocated by cluster size** | class size runs the other way at −0.564 on our panel, so this should lose — and if it wins, margin was a size proxy |
| **one expert per cluster** (full mixture) | bounds what the budget is costing |

⚠️ **A mixture has more parameters than a single probe**, so a win at matched FPR is not a win on equal
terms unless the parameter count is reported. 🔒 **Report both**: matched FPR *and* matched total
parameters, with the second as the honest comparison.

## 4. What would make this a negative result worth having

🔑 **The dissociation already found predicts failure here.** Removing the benign neighbours the geometry
implicates repairs only **8–11%** of a failure the geometry predicts at *ρ* = +0.894: knowing which class
will fail does not tell you how to fix it. ⚠️ **A margin-routed mixture is a repair built on a
prediction**, which is exactly the move the dissociation says does not pay.

🔒 **So the preregistration must carry a ceiling, not only a floor** — the error
`docs/EXTERNAL_VALIDATION_PREREGISTRATION.md` made in the other direction. A sketch of the bands:

| | margin-routing works | it does not |
|---|---|---|
| mean recovery, margin-allocated vs **tuned** single probe | **≥ +5pp** | **≤ +1pp** |
| margin-allocated vs **random**-allocated, same budget | **≥ +3pp** | **≤ +1pp** |
| the two arms | must **agree**, as everywhere | — |

⚠️ **Nothing above is frozen.** The thresholds are a sketch and would be set against the measured
seed-to-seed spread before any mixture is fitted, not chosen to be clearable.

## 5. 🔑 Why this waits for study AT

If AT returns **general**, clusters-not-categories is the natural next step and the claim is about frozen
representations, so the mixture experiment is worth its compute in any of the three modalities — and text
or images would be far cheaper to run it in than proteins.

If AT returns **modality-dependent**, the routing question narrows to wherever margin held, and the
cheapest honest version of this study is in that modality alone.

🔴 **If AT returns "does not generalize", this document is moot** and should be read as a record of a
direction that the generality test closed, not as work pending.
