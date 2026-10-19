# Does fine-tuning recover the classes the frozen probe cannot see? — preregistration

**Frozen 2026-10-08, before `src/147` existed.** Study AY.

## 1. The question the paper's headline rests on

The abstract says: 🔑 **"An AUROC of 0.974 is compatible with a screen that cannot see two whole
families."** Holding out an entire mechanism class, a 32-member phage peptidoglycan hydrolase class
reaches **12.3%** and beta-lactamase **21.0%**.

🔴 **Every number in this project comes from a FROZEN representation.** "frozen" appears 79 times in the
paper; fine-tuning appears twice, both citing other people's work. ⚠️ **So nobody has shown whether those
two failures are a fact about the task or a fact about frozen probing**, and that is the first question a
reader should ask.

🔒 **What is already ruled out is the head.** Four classifier heads across fourteen arms at thirty seeds,
including SVC and random forest; logistic is the **worst** head on 7 of 14 arms and beaten on 12 of 14 by
a median of **5.1 points**. ⚠️ **Non-linearity in the head buys about five points against a 21% and a
12.3% failure.** 🔑 **The representation is the only untested axis.**

## 2. The measurement

🔒 **`15e`'s `recover()` fold, unchanged, on the v3 panel**: 149 positives in 16 mechanism classes, 296
negatives, `FRAC = 0.40` of negatives held out per seed, threshold at the **95th percentile** of the
held-out negatives, leave-one-mechanism-out. 🔒 **The frozen arm is reproduced in the same run rather
than quoted**, and it already matches: **21.19%** on beta-lactamase and **12.19%** on phage against the
paper's 21.0% and 12.3%.

🔒 **Only the scorer changes.** For the same seed, the same negatives split the same way; the fine-tuned
arm trains `EsmForSequenceClassification` on the **sequences** of the trained positives plus the training
negatives, thresholds on the same held-out negatives, and reports the same flagged fraction. ⚠️ **This is
a paired comparison on one fold, not two protocols.**

| recipe | what trains | why it is here |
|---|---|---|
| **`head_only`** | classification head, trunk frozen | 🔑 this *is* the linear probe with a different optimiser — an internal sanity anchor, not a result |
| **`last2`** | head + final two transformer blocks | the cheapest real change to the representation |
| **`full`** | everything | the strongest change, and the most exposed to overfitting |

🔒 **Arms: ESM-2 35M and 650M.** ⚠️ **~137 training positives and ~178 training negatives against 35M+
parameters**, so overfitting is this study's main threat, not its incidental risk.

### 🔒 Study AP's lesson, in reverse

🔴 **AP found an unexamined `C = 1.0` cost up to 24 points per category**, and every frozen arm here now
selects `C` by nested rule. ⚠️ **If the fine-tune is given no equivalent care and loses, the result is a
bug of my own making, not a finding.** 🔒 **Learning rate and epoch count are selected by nested
hold-out on the TRAINED classes only**, over a fixed grid stated here — LR ∈ {1e-5, 5e-5, 1e-4}, epochs
∈ {3, 10} — **once per held-out class, not per seed**, because the hyperparameters must not depend on the
negative draw.

## 3. Frozen bands, set from the measured variance

🔴 **Study AW banded a difference at 1pp against intervals 4–9pp wide**, and its verdict became an
artifact of that. 🔒 **The variance was consulted first this time.** At 30 seeds the frozen arm gives

| class | n | mean | sd | SE |
|---|---|---:|---:|---:|
| `beta_lactamase` | 14 | 21.19% | **13.20pp** | 2.41 |
| `phage_peptidoglycan_hydrolase` | 32 | 12.19% | 6.16pp | 1.12 |

🔑 **So the two-class mean has a per-seed sd near 7.3pp**, and at **10 seeds** its interval is roughly
**9pp** wide. ⚠️ **My first instinct was a ±5pp ceiling, which sits inside that** — the AW error exactly.

| | the failures were a frozen-probing artifact | they survive fine-tuning |
|---|---|---|
| **AY-1** fine-tuned − frozen, mean of the two dead classes | **≥ +20pp** | **≤ +7pp** |
| **AY-2** fine-tuned − frozen, mean over all 16 classes | **≥ +10pp** | **≤ +3pp** |

🔒 **AY-1's 13pp gap exceeds the interval width**, and its pass threshold sits outside it.
🔒 **Both arms must agree**, and `src/143_band_reachability.py` flags that two-unit conjunction as a
single-unit veto — ⚠️ **accepted with the same reason as study AW: the second-arm rule exists so that one
arm can veto.**

🔒 **AY-3, a precondition**: realized false-positive rate on the held-out negatives must sit in
**[4%, 6%]** for every arm and recipe. ⚠️ **It is 5% by construction for the frozen arm; a fine-tuned
model that collapses to one class would break it**, and a recovery figure at an unmatched operating point
is not comparable.

## 4. Predictions

**P1 (reasoned).** 🔴 **AY-1 lands in "they survive", ≤ +7pp.** With ~137 training positives spread over
15 remaining classes and a held-out class that is by construction unlike anything in training, there is
very little signal with which to reshape the representation.

**P2 (derived — 315 training sequences against 35M parameters).** `full` **underperforms** `last2` on the
all-class mean: more trainable parameters is worse here.

**P3 (derived — `head_only` is the linear probe with a different optimiser).** `head_only`'s all-class
mean lands within **10pp** of the frozen arm's.

**P4 (derived — § 6's fair comparison puts ESM-2 at +18.98/+10.50pp over 21 model-free features, so the
representation carries real signal).** AY-2 is **positive**: fine-tuning moves the aggregate even if it
leaves the two dead classes dead.

⚠️ **The record stands at derived 21 of 39, reasoned 4 of 15.**

## 5. What each outcome changes

- **AY-1 ≤ +7pp**, as predicted ⇒ 🔑 **the paper's headline survives its hardest test.** "A screen that
  cannot see two whole families" becomes a statement about the task and the data, not about the probing
  method, and the abstract can say so.
- **AY-1 ≥ +20pp** ⇒ 🔴 **the headline weakens and must be rewritten.** "An AUROC of 0.974 is compatible
  with a screen that cannot see two whole families" would become "…with a *frozen probe* that cannot",
  which is a much narrower claim. 🔒 **Named now so the rewrite is not a negotiation later.**
- **AY-3 fails on an arm** ⇒ that arm is uninterpretable and leaves the comparison, with the reduced
  denominator stated.

---

## Amendment 1 — the inner selection criterion could not discriminate, and the compute was mis-sized

**Added 2026-10-08, after ~50 minutes of the array and before any AY-1 number existed.** 🔒 **No
recovery figure for any held-out class had been computed when this was written** — only the frozen
baseline, which is the published number, and the inner-validation diagnostics below.

### 🔴 Recall at 95% specificity cannot separate the grid on 27 positives

§ 2 selects learning rate and epochs by nested hold-out, scoring each combination by **recall at the 95th
percentile of held-out negatives** — the same quantity the study reports. 🔴 **The inner validation set is
a fifth of the trained positives, about 27 proteins**, so that recall moves in steps of **1/27 = 3.7pp**
and ties are the normal case. **Observed in the first 74 evaluations**: `beta_lactamase` returned
`0.630` for both `epochs = 3` and `epochs = 10`; `adp_ribosyl_ab_toxin` returned `0.500` twice.

⚠️ **A tie resolves to whichever combination was tried first, so the selection is effectively
arbitrary.** 🔴 **That is exactly the risk § 2 exists to avoid**: study AP found an unexamined `C = 1.0`
cost up to 24 points, and a fine-tune whose hyperparameters are chosen by a coin flip has not been given
the equivalent care — **a loss would then be partly my doing and not a finding.**

🔒 **The inner criterion becomes AUROC** over the inner validation positives against the held-out
negatives. 🔑 **It is a *selection* criterion, not a reported metric**: nothing the study reports changes,
and AUROC on 27 positives against ~118 negatives is far more granular than a 27-step recall.
⚠️ **The grid itself is unchanged** — LR ∈ {1e-5, 5e-5, 1e-4}, epochs ∈ {3, 10} — and so are all four
bands.

### ⚠️ And the array was sized from a GPU estimate for a CPU run

🔴 **Measured: ~11 minutes per fine-tune on 16 CPUs**, so 18 inner evaluations plus 30 outer runs is about
**8.8 hours per class** against a **6-hour** limit. ⚠️ **Every task would have been killed at the
deadline**, which is a failure mode that produces no result and no error anyone reads.

🔒 **Resubmitted at 32 CPUs with a 16-hour limit.** 🔑 **The seed count stays at 10**, because § 3's bands
were set from the 10-seed interval and cutting seeds to fit a wall clock would widen that interval past
the band gap — the error study AW made from the other direction.

---

## Amendment 2 — the fine-tuned arm was handicapped by the head, and P3's failure found it

**Added 2026-10-09, after 14 of 16 classes landed.** 🔒 **AY-1 is already known and is not changed by
this**: +2.23pp on `full`, +0.36pp on `last2`, so **P1 holds** either way. 🔑 **This amendment is driven
by a frozen check that failed, not by an outcome I wanted.**

### 🔴 `EsmForSequenceClassification` reads one token, and this project mean-pools

**P3** predicted `head_only` would land within **10pp** of the frozen arm, on the stated reasoning that it
"*is* the linear probe with a different optimiser". 🔴 **It is 37.54pp below it** — 37.80% against 75.35%
on the all-class mean, and **0.00%** on five classes.

🔒 **The cause is in `transformers`' own source:**

    class EsmClassificationHead(nn.Module):
        def forward(self, features, **kwargs):
            x = features[:, 0, :]  # take <s> token (equiv. to [CLS])

⚠️ **ESM-2 is masked-LM pretrained with no sequence-level objective, so `<cls>` is barely populated** —
which is precisely why every frozen arm in this project **mean-pools over residue positions**. 🔑 **So
`head_only` is not the linear probe with a different optimiser; it is a probe on a different and much
poorer representation**, and `last2` and `full` must first spend capacity teaching the trunk to put
signal into `<cls>` before they can exceed it.

🔴 **That means study AY as frozen UNDERSTATES fine-tuning**, and the understatement is mine: the design
chose the library default head and compared it against a mean-pooled baseline. ⚠️ **"Fine-tuning does not
fix the dead classes" is, as it stands, confounded with "the HF default head is bad on this task."**

### 🔒 Two recipes added, pooling held to the frozen arm's

| recipe | what changes |
|---|---|
| **`mean_last2`** | head **mean-pools over residue positions**, final two blocks trainable |
| **`mean_full`** | the same head, every block trainable |

🔒 **Everything else is unchanged** — same fold, same seeds, same negatives, same 95th-percentile
threshold, same nested LR/epoch grid, same bands. 🔑 **Only the pooling moves, and it moves to match the
baseline**, so the comparison is between representations rather than between pooling conventions.

⚠️ **The original three recipes stay and are reported.** 🔑 **`head_only` is now the measurement of the
CLS handicap rather than a sanity anchor**, which is a more useful thing than what it was meant to be.

### 🔒 What this cannot change

⚠️ **AY-1's bands are not touched**, and the verdict will be read on the recipe set as a whole with the
per-class split printed beside it. 🔴 **Whatever the mean does, the composition is the finding**: `phage`
moves **+23.75pp** and `beta_lactamase` collapses from 19.29% to **0.00%** on all three original recipes.
🔑 **The two dead classes do not survive identically — they trade**, and no mean can show that.

---

## Amendment 3 — the frozen-trunk mean-pooled cell is missing, and without it the CLS handicap and the fine-tuning path stay confounded

**Added 2026-10-19, AFTER the 35M arm was scored.** 🔴 **This amendment was written with AY-1, AY-2 and
P1–P4 already read on `esm2_35M`**, and says so rather than implying otherwise. 🔒 **It therefore touches
no band, no prediction and no verdict**: AY-1 and AY-2 keep their definitions and their frozen edges,
P1–P4 keep their wording and their already-computed outcomes, and the recipe added below is declared a
**diagnostic that is excluded from the verdict's recipe set**.

### 🔴 What the scored table shows

| recipe | pooling | trunk | AY-2 |
|---|---|---|---:|
| `head_only` | `<cls>` | frozen | **−33.54pp** |
| `last2` | `<cls>` | final two blocks trainable | −3.23pp |
| `mean_last2` | masked mean | final two blocks trainable | −4.13pp |

⚠️ **With the trunk trainable, pooling is worth about 1pp** (−3.23 against −4.13). 🔴 **With the trunk
frozen, the only pooling present is `<cls>`.** 🔑 **So the ~30pp between `head_only` and `last2` cannot
be decomposed**: it is some mixture of "the final two blocks carry real signal" and "training anything
at all repairs a `<cls>` token that masked-LM pretraining barely populates", and amendment 2's `mean_*`
recipes cannot separate them because they move pooling and the trunk together.

### 🔒 `mean_head_only`: masked mean, frozen trunk, trained head

🔒 **The head architecture is `head_only`'s** — dense → tanh → output, same dropout — and the trunk is
frozen exactly as `head_only` freezes it. 🔑 **Only the pooling differs, so the pair isolates pooling at
a frozen trunk**, and `mean_head_only` against the frozen arm isolates everything the fine-tuning path
contributes that `sklearn` does not.

| contrast | what it measures |
|---|---|
| `mean_head_only` − `head_only` | the `<cls>` handicap, at a frozen trunk |
| `mean_head_only` − frozen arm | what the head, the optimiser and the absence of standardisation cost |

### ⚠️ It is a NEAR anchor, not an exact one

⚠️ **This will not reproduce the frozen arm exactly and is not expected to.** The frozen arm is
`StandardScaler` → `LogisticRegression(C, max_iter=5000)` with `C` selected by nested rule; this is a
one-hidden-layer head with dropout and no standardisation, fitted by AdamW at batch 4 over the nested
LR/epoch grid. 🔑 **Agreement establishes that the two arms are comparable; it does not make them the
same estimator.**

🔒 **The threshold is the one the preregistration already set for this exact quantity** — P3's **10pp**
on the all-class mean against the frozen arm. ⚠️ **No new edge is being invented after the fact**; the
existing one is being applied to the cell that should have carried it.

### 🔒 What each outcome changes

- **`mean_head_only` within 10pp of frozen** ⇒ 🔑 **the ~33.54pp is pooling, and the fine-tuning path is
  sound.** `last2`'s advantage over `head_only` is then mostly `<cls>` repair rather than representation
  change, and that reading goes in the paper beside AY-1.
- **`mean_head_only` also ~30pp below frozen** ⇒ 🔴 **the fine-tuning path itself is weaker than the
  frozen arm's estimator**, and every recipe's comparison to frozen is then partly a comparison of
  optimisers. ⚠️ **That would not move AY-1's verdict, which is already "they survive" on both edges of
  every recipe, but it would become a stated limitation of the whole study** rather than a footnote about
  one recipe.
- **Either way the trade stands.** 🔑 **`beta_lactamase` reaches 0.00% or 0.71% on all five scored
  recipes while `phage` gains 13–26pp**, and a sixth recipe cannot unmake five.
