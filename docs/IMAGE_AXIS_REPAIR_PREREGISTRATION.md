# The image axis had no room to move — preregistration

**Frozen 2026-10-07, before `src/141` existed.** Study AU.

## 1. What this is repairing, and what it is not

Study AT's two image cells were the grid's weakest. `image/clip_vit_b32` landed **between bands** at *ρ*
+0.5231 and +0.5422; `image/dinov2_vits14` cleared by **0.016**. ⚠️ **Amendment 3 of
`docs/GENERALITY_GRID_PREREGISTRATION.md` declined to reclassify either as uninterpretable**, because
doing so after seeing a cell fail converts a negative result into a non-result. 🔒 **Those verdicts stand
and this study does not touch them.**

🔑 **What amendment 3 did do is name the diagnosis and the remedy.** The measured problem:

| cell | recovery range | sd | classes ≥ 99% |
|---|---|---:|---:|
| `image/clip_vit_b32` fixed | 73.0 – 99.6% | **6.6** | 2 / 50 |
| `image/clip_vit_b32` swept | 77.3 – 100.0% | **5.0** | **16 / 50** |
| every protein and text cell | 13 – 88% | 12.0 – 19.5 | 0 |

🔴 **A held-out CIFAR-100 *fine* class has forty-nine siblings still in training.** Hold out `leopard` and
the probe has seen `lion`, `tiger`, `bear` and `wolf`; it recovers the held-out class at 96% on average
and there is almost no variance left for any quantity to predict. ⚠️ **That is a property of the axis, not
of images**, and the AT design cannot tell those two apart.

## 2. The one change

🔒 **The held-out unit becomes the superclass, not the fine class.** Everything else is identical:
CIFAR-100, the same **animal vs non-animal** split frozen in AT § 2 (10 superclasses each, 25,000 rows
each), the same two frozen encoders, the same **already-built embedding caches**, `src/125`'s fold
unchanged, the union negatives, both the fixed `C = 1.0` and `src/133`'s nested-selected `C`, 30 seeds.

| | AT (study AT) | AU (this study) |
|---|---|---|
| held-out unit | fine class | **superclass** |
| units held out | 50 | **10** |
| rows removed per fold | 500 | **2,500** |
| siblings left in training | 49 fine classes | **none from that superclass** |

🔑 **Holding out `large_carnivores` removes `bear`, `leopard`, `lion`, `tiger` and `wolf` together**, so
the held-out group has no near-duplicate left behind. 🔑 **That is what the protein and text folds already
do** — a mechanism category and a newsgroup sub-hierarchy are both coherent groups, not one member of a
tight cluster.

⚠️ **Ten units is ten points, and *ρ* is coarser there than anywhere else in this project**: one adjacent
swap moves it by 12/(10 × 99) = **0.0121**, and study AT measured draw-to-draw movement near ±0.10 at
thirteen classes. 🔒 **A second negative draw is part of this study and not an afterthought.**

## 3. Preconditions, with a ceiling this time

🔴 **AT-0a had a floor and no ceiling, which is the defect the paper criticises in
`docs/EXTERNAL_VALIDATION_PREREGISTRATION.md` and which amendment 3 recorded rather than repaired.**
🔒 **AU-0 is two-sided:**

- **AU-0a** — mean recovery inside **[20%, 90%]**. Below 20% the fold is near null; **above 90% it is at
  the ceiling and *ρ* is uninterpretable whatever it comes out as.**
- **AU-0b** — realized test false-positive rate inside **[4.5%, 5.6%]**, which both AT image cells met.
- 🔒 A cell failing either is **UNINTERPRETABLE** and leaves the numerator *and* the denominator, with the
  reduced denominator stated. ⚠️ **It is not a failure**, and it is not a pass.

## 4. Frozen bands

Bonferroni α = 0.05/3.

| | repaired | not repaired |
|---|---|---|
| **AU-1** *ρ*(margin, recovery), **both** encoders, **both** settings | **≥ +0.60** | **≤ +0.20** on either encoder |
| **AU-2** margin's \|*ρ*\| exceeds class size and both halves of its own definition | **≥ 3 of 4** cell-settings | ≤ 1 of 4 |
| **AU-3** mean recovery and its spread | mean **< 90%** and sd **> 10pp**, both encoders | sd ≤ 6pp on either |

🔒 **AU-1 requires both encoders.** One encoder is one arm, and study AT's two image encoders disagreed at
the fine-class axis — +0.52 against +0.62 — which is exactly why a single-encoder answer here would not
settle anything.

⚠️ **AU-3 is the precondition restated as a test.** If the spread does not open up, the axis change did not
do what § 1 says it should, and AU-1 is uninterpretable *by AU-0a* rather than informative.

## 5. Predictions

**P1 (derived — from the measured saturation: 16 of 50 classes at or above 99% at the tuned setting, and
a held-out fine class retaining 49 siblings).** AU-3 holds: mean recovery falls **below 90%** and sd rises
**above 10pp** on both encoders.

**P2 (derived — from study AT's mechanism result, *ρ* = +0.7413 between \|*ρ*(nearest-negative)\| and the
gain from subtracting, and from DINOv2's negative term already being strong at −0.4015 and −0.5176).**
Margin beats `nearest_positive` on **both** encoders at **both** settings — 4 of 4.

**P3 (reasoned).** AU-1 clears **+0.60** on both encoders, i.e. 🔑 **study AT's image result was about the
axis and not about images.**

**P4 (derived — ten points, and AT measured ±0.10 draw-to-draw at thirteen).** The second negative draw
moves *ρ* by **more** than the 0.0029–0.0473 seen on the fifty-class cells.

⚠️ **The computed record stands at derived 17 of 30, reasoned 3 of 11.** ⚠️ **And this design was written
knowing the fine-class outcome**, which is disclosed here: the direction was named in amendment 3 before
any of it was designed, but the numbers above were chosen with AT's spread already on the table.

## 6. What each outcome changes

- **AU-1 clears and AU-3 holds** ⇒ 🔑 **study AT's image cells were measuring a saturated axis**, the
  three-modality claim is about representation geometry in all three, and the paper reports both axes with
  the fine-class one as the cautionary case. 🔒 **AT's verdict is still computed on its own frozen cells.**
- **AU-3 holds but AU-1 fails** ⇒ 🔑 **the strongest outcome available here**: the axis was repaired, the
  variance is real, and margin still does not order it — so the quantity is genuinely weaker on images and
  the grid's modality-dependence is a fact about images rather than about CIFAR-100's hierarchy.
- **AU-3 fails** ⇒ the repair did not repair anything, AU-1 is uninterpretable, and the honest report is
  that this project could not build a non-saturated image axis from CIFAR-100.
