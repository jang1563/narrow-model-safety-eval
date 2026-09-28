# An embedding hazard screen at AUROC 0.974 and precision 1.2%: what eighteen criteria and sixty corrections found

**Draft, 2026-09-28.** Every figure below is recomputed from an artifact by
`src/22_claims_audit.py` and pinned to the sentence that states it; the gate currently checks 90 such
claims. Section references without a document name are to `docs/MECHANISM_GENERALIZATION.md`.

---

## Abstract

A protein-language-model embedding separates a curated hazard panel from its negatives at **AUROC
0.974 ± 0.014**. Screened against 8,258 held-out benign proteins at a nominal 5% false-positive budget
the same probe returns **7.87%**, and translated into a deployment of ten thousand sequences at a
one-in-a-thousand hazard rate it produces **605 alerts of which about 7 are real — a precision of
1.2%**. This paper is about the distance between those two numbers, and about what it takes to see it.

We state **eighteen criteria** a hazard-screening claim should meet, score our own probe against them,
and report the result: **two fails, four partials and one mixed**. We then show that each criterion was
written because it caught something. The evidence is a **dated, append-only corrections log of sixty
entries** in which a published number moved, including entries that retract our own findings.

Three results matter beyond this panel.

**First, the aggregate is uninformative and the per-class picture is bimodal.** Holding out an entire
mechanism class, recovery runs from **100% to 10%**: four classes are fully recovered without ever
being trained on, while beta-lactamase reaches **15.5% [11.0, 19.9]** at thirty seeds and a
32-member phage peptidoglycan hydrolase class reaches **10%**. An AUROC of 0.974 is compatible with a
screen that cannot see two whole families.

**Second, a quantity computable before training predicts which class will fail.** Class-level
**margin** — nearest other-class hazard minus nearest negative in embedding space — ranks classes by
recovery at Spearman **+0.894** (permutation *p* = 0.00015), places both failures at the bottom, and
survives **fourteen model arms** and **fifteen re-poolings** of one arm (*ρ* +0.536 to +0.941, 14 of 15
significant). It is the only result here that replicated everywhere we looked. It is also
**mis-calibrated**: it over-predicts beta-lactamase recovery by 34 points.

**Third, and the finding we did not want: nothing we tried moves the screen's precision.** Four
classifier heads, fourteen representations, three panels, fifteen pooling reductions and a supervised
residue ranking later, deployment precision sits between **0.97% and 1.26%** in every configuration.
One reduction moves the two failing classes by +15 to +20 points — and pays for it in the negatives'
scores exactly as a fixed-budget argument predicts, and does so **only on one of two representations
tested**.

We close on a structural observation about evaluation rather than about protein models. Twice we
diagnosed a limit, built the repair, and found the limit was on the other side of the comparison: both
times we grew the negative set when the binding constraint was **panel annotation** — fifteen proteins,
seventy-four annotated catalytic positions, **seventeen** of them confirmed by a common source. An
evaluation can be improved in the wrong direction, and the cheapest way to find out is to write the
ceiling down before the run.
