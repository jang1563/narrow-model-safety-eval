# Narrow Model — one-pager for the ICML candidate decision

**Written 2026-10-19** for the Oct 17–23 deliverable in
`Conferences/2026_AI/CONFERENCE_SUBMISSION_STRATEGY_2026-10-17.md` §6, which asks each candidate for
its core question, its current main result, and its most important unfinished comparison. The
decision it feeds is **by 2026-11-15: one of Narrow or Arena as the ICML main candidate.**

Every number here is from `paper/MANUSCRIPT.md` or a named artifact, not from recollection.

---

## 1. The core question

**Does proximity in a frozen representation predict which unobserved class a detector will fail on —
and can you know it before training?**

The project arrived at this question from the other direction, by building a hazard screen and
finding that its aggregate score said nothing about its per-class behaviour. That origin is why the
manuscript currently leads with the screen. The screen is the *case*, not the contribution.

## 2. Current main result

**Three findings carry, in this order.**

**(a) The aggregate is uninformative and the per-class picture is bimodal.** Holding out an entire
mechanism class, recovery runs **100% to 12%**: three classes fully recovered untrained, while a
32-member phage peptidoglycan hydrolase class reaches **12.3%** and beta-lactamase **21.0%** (one
panel, 30 seeds). Member-level resampling puts beta-lactamase at **[0, 36]**, so what survives is
the *bimodality* — 100% and [0, 36] are disjoint — not the precision of any single low number.
**AUROC 0.974 ± 0.014 is compatible with a screen that cannot see two whole families.**

**(b) A quantity computable before training predicts which class fails, and it is not a fact about
protein language models.** Class-level **margin** — nearest other-class hazard minus nearest
negative — ranks classes by recovery at **ρ +0.894** (permutation *p* = 0.00015), places both
failures at the bottom, and survives 14 model arms and 15 re-poolings (ρ +0.536 to +0.941, 14 of 15
significant). On a published benchmark with someone else's labels: **ρ +0.947 / +0.886** over
DeepVIC's 14 curator categories and 12,989 proteins. A preregistered **3-modality × 2-encoder-family
grid** — proteins, text, images; ESM-2, ProtT5-XL, MiniLM, MPNet, DINOv2, CLIP — clears its frozen
floor at **5 of 6** cells and returns `general`; **read under the preconditions exactly as frozen it
is 3 of 4**, and both counts are reported. The protein row is two *families*, not one family at two
sizes, so what survives is a change of representation rather than of scale.

**(c) The formula is conditional, and a training-free quantity says which form to use.** `margin`
and its own first term each clear the floor in **10 of 12** cell-settings and fail on *different*
cells. The subtraction is earned only where the negative term carries signal: **ρ +0.8118** between
that term's own correlation with recovery and the gain from subtracting it. And
**ρ(nearest-positive, nearest-negative)** across a cell's classes — sealed before the folds ran on
four unseen encoders — predicts which at **ρ −0.661** over 24 cell-settings.

> This third finding is the one the strategy document asked for and the one the manuscript currently
> buries. It is written as a caveat ("what does not generalize is the formula") when it is the
> answer to *when is a simple distance enough*. **Restructuring, not new research.**

**What is closed negatively, and should be stated as such rather than defended:** no training-free
quantity predicts the *magnitude* of the gain — nine candidates against a permutation p95 of
**0.551**, best **0.360**. The paper predicts an **ordering**, not an improvement size. Saying so
plainly is stronger than letting a reviewer find it.

## 3. Most important unfinished comparison

**Ranked. Only the first is load-bearing for the Nov 15 decision.**

**(1) The matched-pooling fine-tuning comparison — in flight today.** The strategy document names
this precisely: the fine-tuning arms differ in pooling and head, so the comparison is
interpretation-limited and its matched conclusion is not complete. Status as of today:

- The **35M arm is complete** and every recipe lands in the preregistered "they survive" band
  (worst AY-1 **+3.84pp** against a **+7pp** ceiling; AY-2 negative on all five recipes). AY-3
  passes on all 78 cells.
- **It is a trade, not a failure to help.** `beta_lactamase` goes 19.29% → **0.00% or 0.71%** under
  *all five* recipes while `phage` gains **13–26pp**; the spread between the two is 32.7–44.8pp, so
  a tidy +3.84pp mean is a −18.57pp and a +26.25pp averaged together.
- **The verdict is `undeclarable`.** The preregistration freezes "both arms must agree" and the
  **650M arm has never run** (it needs a GPU; 35M costs ~2.1 h per cell on 16 CPUs). The scoring
  code refuses to declare rather than quietly reporting one arm.
- Amendment 3's **`mean_head_only`** — masked mean, frozen trunk, `head_only`'s own head — is the
  matched-pooling cell that closes the interpretation gap. 16 cells running; declared a post-read
  diagnostic, excluded from the verdict and from the scored predictions.

**(2) The thresholded form of the training-free selector was not established** — 1 of 2 independent
cells. The ordering claim (ρ −0.661) stands; the decision rule does not.

**(3) External class-level replication is mixed, in three tests.** VFDB's 13 categories returned
**ρ +0.681** (*p* = 0.0059), a frozen confound rule then fired and made it uninterpretable, and a
follow-up holding organism identity constant reinstated it **as an ordering claim only** at
**ρ +0.375 / +0.411** with 8 of 10 species strata positive on both arms. The member-level version was
preregistered and **failed** externally. And the predictor is **mis-calibrated** — it over-predicts
beta-lactamase recovery by **34 points**.

**(4) Named and not promised:** a second text/image encoder for the axis-repair study; a second
annotator for the hazard taxonomy; six of twelve intervals in corrections entry 84; DeepVIC's
shipped `.keras` weights cannot be run.

## 4. Against the document's own six criteria

| criterion | honest score |
|---|---|
| New scientific question | **Strong.** "What predicts transfer in a frozen representation" is not the hazard-screening literature's question. |
| Independent evaluation unit | **Strong.** Class-level, leave-one-mechanism-out, plus two external label sets (DeepVIC, VFDB) and three modalities. |
| **Strong simple baseline** | **Mixed — and this is the weak one.** Nearest-positive alone matches `margin` in 10 of 12 cell-settings. The honest resolution is that the simple baseline *is* a result (ρ +0.8118 says when subtraction is earned), not a threat to be minimised. |
| Real usage meaning | **Weak as written.** Deployment precision sits in a **0.76%–1.26%** box and 1.2% is its most favourable corner. Must be stated at a usage criterion, not as a headline. |
| Generalization matching claim scope | **Adequate if restructured.** Ordering yes, magnitude no, calibration no. The paper must claim exactly that. |
| Completed reading flow | **Fails today.** 158 audited claims and a 135-entry corrections log are apparatus; by JK's own standing rule they are not the paper. |

## 5. What the restructure is

Lead with (b) and (c): proximity in a frozen representation predicts per-class transfer, the
functional form is conditional, and a training-free quantity selects the form. Demote the hazard
screen to the motivating case and the source of (a). Keep the eighteen criteria and the corrections
log as an appendix and a methods statement, not as the spine. State three limits in the abstract
rather than the discussion: **ordering not magnitude**, **mis-calibrated**, and **precision at a
usage criterion**.

**Do not** inflate any of this into a safety guarantee or a direct measurement of biological hazard.
The strategy document says so explicitly and it is right.
