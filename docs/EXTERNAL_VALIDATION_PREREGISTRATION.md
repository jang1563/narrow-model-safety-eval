# External validation: frozen protocol and pre-registered prediction

**Written:** 2026-09-04
**Frozen at commit:** `f1a5860`
**Status when written:** no external data has been fetched, inspected or scored. This document exists
so that it cannot be written afterwards.

## Why this exists

Every result in this project is internal cross-validation on a panel built here, and that panel grew
from 91 to 100 to 154 negatives **in response to confounds the analysis itself found**. Each expansion
was principled and is documented, but the consequence is that the final numbers sit on a panel shaped by
looking at intermediate results. No amount of additional internal control fixes that; only a set the
project never touched can.

The purpose of this document is to fix the pipeline, the hypothesis and the falsification criteria
**before** the external panel is assembled, so the result cannot be rationalized in either direction.

## What is frozen

Pipeline, exactly as at `f1a5860`, with no parameter changes permitted:

| Stage | Script | Frozen settings |
|---|---|---|
| Embedding | `02b_esm2_embed_v2.py` | `facebook/esm2_t33_650M_UR50D`, mean pooling, `MAX_LEN = 1022` |
| Holdout scoring | `03k_margin_holdout.py` | `NEG_HOLDOUT_FRAC = 0.40`, `SPEC = 0.95`, `k = 10`, 30 seeds, 15 random repeats |
| Margin definition | `03k` | cosine to positives **outside the protein's own mechanism class**, minus cosine to the nearest negative |
| Head | all | L2 logistic regression on standardized embeddings |

ESM-2 650M is the primary model because it is the one the internal numbers were developed on, which
makes it the least favourable choice and therefore the honest one. Other models may be reported as
secondary, but the pass or fail decision is taken on ESM-2 650M alone.

## Panel construction rules, fixed in advance

1. Positives and negatives come from a **public, independently assembled source**, preference order:
   SafeProtein-Bench (as used by SafeBench-Seq), then OmniTox.
2. **Any accession already in `panel_v2_manifest.json` is excluded**, positive or negative.
3. **Any external protein whose normalized Smith-Waterman similarity to any internal panel member
   exceeds 0.30 is excluded**, using the same aligner settings as `03i`. This threshold is chosen now,
   before seeing the distribution.
4. Mechanism classes are assigned from the source's own labels where they exist. Where they do not,
   classes are assigned by the source's annotation only, never by inspecting model output.
5. The panel is assembled once. If it must be rebuilt for a technical reason, that is recorded here with
   the reason, and the reason may not be "the result came out wrong".

## Primary hypothesis

§6l found that recovery is governed by margin to already-seen positives, with an effect of 36 to 62
points across five models, and 62 points on ESM-2 650M with class composition held fixed.

**Prediction, stated before any external data is seen:**

> On the external panel, a holdout of the 10 lowest-margin positives will be recovered at **≤ 40%**,
> a class-composition-matched random holdout of the same size at **≥ 60%**, and the gap between them
> will be **≥ 25 percentage points**.

The predicted bounds are deliberately wider than the internal result (22%, 84%, 62 points), because an
external panel has different class structure and a different negative distribution. Narrow bounds would
be false precision.

## What would falsify it

- **Gap < 25 points**: the margin effect does not transfer, and §6l describes this panel rather than the
  method. That would be the headline result and would be reported as such.
- **Low-margin holdout > 40%**: the low-margin regime is not a failure regime outside this panel.
- **Class-matched random < 60%**: the pipeline does not work on the external panel at all, which would
  make the margin comparison uninterpretable rather than negative. Reported as inconclusive, not as
  support.

## Rules for after the result

- The pipeline is **not** modified in response to the external result. If it fails, it fails.
- No post-hoc exclusion of external proteins. The exclusion rules are the four above and nothing else.
- The external numbers are reported in memo 44 whatever they are, next to the internal ones, with this
  document linked.
- If the result is negative, §6l's claim is downgraded from a property of the method to a property of
  the internal panel, and the log says so in §0.

## Secondary, non-preregistered

Anything else computed on the external panel is exploratory and will be labelled that way. That includes
per-class recovery, the alignment baseline, and any other model. Only the three numbers in the
prediction above are confirmatory.
