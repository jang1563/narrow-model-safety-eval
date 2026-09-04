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

## Amendment 1, 2026-09-04: the preferred sources do not exist in usable form

Recorded **before** any external panel was built or scored, and before the query below was run. The
preference order above was SafeProtein-Bench, then OmniTox. Both fail, and so does the obvious third
option:

| Source | Outcome |
|---|---|
| **SafeProtein-Bench** | `github.com/HARISKHAN-1729/SafeBench-Seq`, cited in the paper as "Code & metadata", contains only `LICENSE` and `README.md`. The README reads "Working on organizing the code". No metadata is released. Every SafeProtein repository path tried returns 404. |
| **OmniTox** | No public repository found. |
| **ToxinPred3** (HF `tanthinhdt/toxinpred3`, the natural third option) | 🔴 **Wrong domain.** Its sequences are peptides: median 21 residues, range 4 to 35. This panel holds proteins: median 375, range 96 to 2085. Scoring a protein pipeline on 21-residue peptides would test a different task while being labelled external validation. |

**Substitution, specified here before it is run.** The external panel becomes a **held-out draw from
UniProt under a query fixed in this document**, executed once. This is weaker than a third-party
benchmark and is labelled as such wherever it is reported: it is not independent curation, it is
independent *proteins*. The property it does preserve is the one that matters for the hypothesis, that
none of these proteins was seen or selected while the method was developed.

**The query, fixed now:**

1. **Positives.** UniProt, `reviewed:true`, `keyword:KW-0800` (Toxin), bacterial, sequence length 100 to
   1022. Excluded: any accession in `panel_v2_manifest.json`; any protein with normalized Smith-Waterman
   similarity above **0.30** to any internal panel member; more than 3 per species.
2. **Mechanism classes** are assigned from the UniProt protein name using the same vocabulary already
   written in `02d`'s `CLASS_BLOCK`, and nothing else. A protein matching no class is dropped rather
   than placed in a catch-all.
3. **Negatives.** For each positive's organism, benign proteins fetched by `02d`'s existing filters
   unchanged, including filter 4b, excluding every internal panel accession.
4. **Target size** roughly 60 positives and 120 negatives, to keep the ratio near the internal panel's
   66 to 154. Whatever the query returns is what is used.

The prediction below is unchanged. It was written before this amendment and is not revised in light of
the substitution.

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

## OUTCOME, 2026-09-04: NOT SUPPORTED

Run once, on the panel built by the query above, with the pipeline frozen at `f1a5860`. The scorer
refused to touch the external panel until its copy of the margin reproduced the published internal
numbers; that gate passed at 22 / 84 / 100.

| | predicted | internal | **external** | |
|---|---|---|---|---|
| low-margin holdout of 10 | ≤ 40% | 22% | **82%** | 🔴 FAIL |
| class-matched random | ≥ 60% | 84% | 94% | 🟢 PASS |
| gap | ≥ 25 pts | +62 | **+13** | 🔴 FAIL |
| high-margin holdout | not preregistered | 100% | 100% | |

**The prediction failed.** Two of the three confirmatory criteria were missed.

**The failure is not explained by the low-margin group being less extreme.** Its mean margin is
**−0.012 externally against −0.015 internally**, essentially the same absolute position. At the same
margin, the internal panel recovers at 22% and the external at 82%. **Margin is therefore not the
governing variable it appeared to be.**

**Consequence, applied as written above:** §6l is downgraded from a property of the method to a property
of the internal panel, and memo 44 §0 says so.

### Exploratory observations, not a rescue

Labelled per the rules below, and none of them change the verdict:

- The external panel is easier overall: class-matched random is 94% against 84%, so the usable range
  above the low group is compressed.
- Its class structure is very different: 27 of 51 positives are pore-forming, and it has 4
  holdout-eligible classes against the internal panel's 9. Holding out 10 of 51 removes a fifth of the
  positives from a much more homogeneous pool.
- The external margin range is narrower at the top, +0.007 against +0.029, so "high margin" means
  something weaker there, yet it still recovers at 100%.

These are candidate explanations for **why** the effect did not transfer. They are hypotheses generated
after seeing the result and carry no evidential weight until they are themselves tested on a panel that
does not yet exist.

## Amendment 2, 2026-09-04: attempt 2, registered before scoring

🔴 **Attempt 1 failed and that failure stands permanently.** It is reported in memo 44 §6m and is not
retracted, softened or replaced by anything below. Running a second external test after a failed first
one is exactly the move a preregistration exists to control, so the rule is stated here before the second
run: **both attempts are reported, in order, with their verdicts, whatever they are.** If attempt 2
passes, the honest summary is "failed on a self-drawn panel, passed on an externally curated one", never
"validated externally".

**Why a second attempt is justified rather than opportunistic.** Attempt 1's substitution was forced by
three sources being unavailable, and its own Amendment 1 recorded the substitute as "independent
proteins, not independent curation", which is the weaker property. An externally curated panel tests
something attempt 1 could not.

**The source.** SafeProtein-Bench's own repository is also unavailable: `github.com/jigang-fan/SafeProtein`
returns 404 and the author account has zero public repositories, so that is now the third paper in this
niche claiming a public release that does not exist. However, SafeProtein's panel identity is recoverable
from artifacts that VFUSE and SAEBER published on Hugging Face: the per-protein filenames in
`michaelwaves/saeber-rfd3-safeprotein-activations` encode UniProt accessions with hazard or benign
labels. **280 accessions recovered, 166 hazard and 114 benign.** Sequences are then fetched from UniProt
by accession. The curation is therefore third-party; only the sequence retrieval is ours.

**Rules for attempt 2, fixed here:**

1. Labels come from the recovered filenames and are **not** reviewed, corrected or re-assigned.
2. **Accessions already present in `panel_v2_manifest.json` are excluded**, and the overlap count is
   reported.
3. ⚠️ **No homology screen against the internal panel**, deliberately, and this differs from attempt 1.
   The scoring for an external panel trains and tests entirely within that panel, so the internal set
   contributes nothing and cannot leak. Attempt 1's screen existed to make a self-drawn panel genuinely
   different, a concern that does not apply to a panel curated by someone else. Screening here would mean
   editing a third-party benchmark to suit us.
4. Mechanism classes are assigned from UniProt protein names using the same `CLASS_TERMS` mapping as
   attempt 1, because SafeProtein's own class labels were not recovered. A protein matching no class is
   dropped.
5. The pipeline, thresholds and the three confirmatory numbers are **unchanged**: low-margin holdout of
   10 at ≤ 40%, class-matched random at ≥ 60%, gap ≥ 25 points.

⚠️ Known differences from both previous panels, recorded now rather than discovered later: this set is
hazard-heavy (166 against 114) where the internal panel is negative-heavy (66 against 154), and its
hazards include viral proteins where the internal panel is bacterial.

## Secondary, non-preregistered

Anything else computed on the external panel is exploratory and will be labelled that way. That includes
per-class recovery, the alignment baseline, and any other model. Only the three numbers in the
prediction above are confirmatory.
