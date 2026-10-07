# Three modalities, two encoder families each — preregistration

**Frozen 2026-10-06, before `src/139` existed.** Study AT.

## 1. What study AS left open

AS ran `src/125`'s fold on **20 Newsgroups** with one frozen sentence encoder and returned *ρ*(margin,
recovery) = **+0.9505** and **+0.8132** at the two classifier settings, both clearing the **+0.60** floor
borrowed from the DeepVIC study. 🔑 **That moved the result from "a fact about protein language models" to
"a fact about frozen representations"** — but on evidence of exactly **one** encoder per non-protein
modality, and the protein evidence is **one family at two sizes**.

⚠️ **This project's own rule is that no verdict stands on one arm.** AS is labelled single-arm
indicative for that reason, and § 4 of the paper says a second encoder is the named follow-up. 🔒 **This
is that follow-up, widened to the smallest grid that could settle the generality question.**

## 2. The grid

🔒 **`src/125`'s fold, unchanged** in all six cells: `StandardScaler` → `LogisticRegression`, threshold at
the 95th percentile of calibrate negatives, 1,000 train / 500 calibrate redrawn per seed, **30 seeds**,
fixed test partition, one class held out entirely at a time. 🔒 **`margin` is `src/30`'s, imported**, and
both the fixed `C = 1.0` and `src/133`'s nested-selected `C` are run in every cell.

| modality | positives | negatives | classes held out | encoders |
|---|---|---|---|---|
| **protein** | DeepVIC's 12,989 virulence factors | its 33,456 non-virulence | **14** curator categories | **ESM-2 650M**, **ProtT5-XL** |
| **text** | 20NG `comp`/`rec`/`sci` | 20NG `alt`/`misc`/`soc`/`talk` | **13** sub-classes | **all-MiniLM-L6-v2**, **all-mpnet-base-v2** |
| **image** | CIFAR-100 animal superclasses incl. `people` (10) | its plant, artifact and scene superclasses (10) | **50** fine classes | **CLIP ViT-B/32**, **DINOv2 ViT-S/14** |

🔑 **The protein row is deliberately ESM-2 650M and ProtT5, not 650M and 35M.** The grid is about encoder
**families**; 650M/35M is one family twice, and its agreement is already reported. 🔒 **ESM-2 35M stays in
the record as a third protein cell** and is reported, but the two-family row is what AT-1 is scored on.

🔑 **The image axis is animal against non-animal**, stated now rather than chosen later, and `people` sits
on the **animal** side because people are animals — the alternative placement was considered and
rejected before any embedding was computed. ⚠️ **CIFAR-100 is 32×32 and both encoders pretrained at
224×224**, so images are upsampled; this is standard for frozen CLIP probes and is a **stated
methodological choice**, not a neutral one.

🔑 **Fifty held-out classes is 50 points**, against text's 13 and protein's 14. ⚠️ **That makes the image
cells the best-powered and the protein cells the worst**, which is the reverse of this project's usual
asymmetry and is worth saying out loud before the numbers arrive.

## 3. Preconditions, frozen before the run

🔴 **AS's first design was a null task that ran to completion because every band was on *ρ*.** Each cell
must therefore pass **both** of these before its *ρ* is interpretable at all:

- **AT-0a** — mean recovery over the cell's held-out classes **> 20%**. Below that the fold is close to
  null and a rank correlation over near-zero numbers is noise.
- **AT-0b** — realized calibration false-positive rate inside **[4.5%, 5.6%]**, the matched operating
  point every contrast in this project uses.

🔒 **A cell failing either is reported UNINTERPRETABLE and removed from AT-1's numerator *and*
denominator**, with the reduced denominator stated in the artifact. ⚠️ **It is not counted as a failure**,
because an uninterpretable cell is not evidence either way — the error
`docs/EXTERNAL_VALIDATION_PREREGISTRATION.md` made in the other direction.

## 4. Frozen bands

Floors **and** ceilings. Bonferroni α = 0.05/3 over the three primary tests.

| | the result is general | it is modality-dependent | it does not generalize |
|---|---|---|---|
| **AT-1** cells with *ρ* ≥ +0.60 at **both** classifier settings | **≥ 5 of 6** | 3–4 of 6 | **≤ 2 of 6** |
| **AT-1b** any modality with **both** encoders ≤ +0.20 | **none** | — | **one or more** |
| **AT-2** cells where margin's \|*ρ*\| exceeds class size **and** both of its own components | **≥ 5 of 6** | 3–4 | **≤ 2** |

🔒 **AT-1 and AT-1b must agree.** A 5-of-6 count with one modality dead on both encoders is
**modality-dependent**, not general, and the conjunction is what is read — the same both-arms rule that
caught § 6's 650M-only ranking.

⚠️ **AT-2 is the test that matters for an outside reader.** A quantity that correlates with recovery is
uninteresting if class size does it too; margin must beat **class size** and **both halves of its own
definition** — max-cosine-to-positive alone and max-cosine-to-negative alone — as it does on all four
DeepVIC passes.

## 5. Predictions

**P1 (derived — AS cleared +0.60 at both classifier settings on text, and DeepVIC at both arms and both
passes).** AT-1 lands at **≥ 5 of 6**.

**P2 (reasoned).** The **image** cells are the weakest of the three modalities, because 32×32 upsampled to
224 sits furthest from either encoder's pretraining distribution.

**P3 (derived — from AS's P4 miss, which showed the floor classes are those nearest the *negative* side
rather than those with a near-duplicate sibling).** In every interpretable cell the lowest-margin class
has a margin at or below **zero** and sits in the **bottom third** of that cell's recovery ranking.

**P4 (derived — AS's tuning gain was +4.25pp on 384 features against ESM-2 650M's +13.96pp on 1,280, and
study AP's mechanism is over-fitting at fixed `C`).** Across the six cells the tuning gain correlates
**positively** with embedding dimension, at *ρ* ≥ **+0.5**.

⚠️ **The computed record stands at derived 17 of 30, reasoned 3 of 11**, so a derived prediction here is
near a coin flip and P1 is not a safe bet because it is labelled derived.

## 6. What each outcome changes

- **AT-1 ≥ 5 of 6 with AT-1b clean** ⇒ 🔑 **the paper's lead result is a transfer-prediction claim across
  three modalities and six frozen encoders**, with the biosecurity screen as the motivating case, and the
  single-arm caveat on AS is discharged.
- **one modality dead on both encoders** ⇒ the claim is **bounded to the modalities where it holds**, and
  the paper names the boundary instead of implying generality past it. 🔑 **This is a result, not a
  failure**, and it is the outcome that would most change what the quantity is understood to measure.
- **AT-2 below 5 of 6** ⇒ margin is not separable from class size or from its own parts outside protein,
  and the class-level claim narrows to the domain where the controls were run.

## 7. Compute, stated in advance

⚠️ **One text encoder (~3 min), two image encoders over 50,000 CIFAR-100 images (~20–40 min each,
locally, checkpointed), and ProtT5 on DeepVIC's 46,445 sequences (Cayuga, queued).** 🔒 **No cell is
re-run after its *ρ* is seen**, and the embedding checkpoints hash build, model and device so a resume
cannot mix backends — the defect the ESM-2 arm hit.
