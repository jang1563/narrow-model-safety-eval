# What does the embedding add given composition? — preregistration

**Frozen 2026-10-02, before `src/132` existed.** Study AO.

## 1. Why study AN could not answer this and this can

AN removed the composition subspace and re-measured, and returned **does not settle it** under a rule
frozen before either variant was read. 🔑 **The reason is structural, not a bug**: a projection that
erases completely must be fitted on the rows it is applied to, and fitting across both classes makes it
absorb the between-class difference — composition alone reaches **AUROC 0.8705** on this benchmark's
binary task, so removing everything linearly predictable from it removes much of the class signal. The
leak-free variant under-erases on exactly the measured rows (residual correlation **0.239**).

🔒 **A nested model comparison has neither problem.** Ask what the embedding adds *given* composition by
fitting both together, instead of trying to subtract one from the other:

| model | features | recovery |
|---|---:|---|
| **C** composition alone | 21 | **34.25%** (study AJ) |
| **E** embedding alone | 1,280 / 480 | **41.43% / 41.78%** (study AI-2) |
| **C+E** concatenated | 1,301 / 501 | **this study** |

⚠️ **No erasure, no projection, no leak**: every feature is computed from the protein's own sequence and
the fold is `src/125`'s, with the negative partition reproduced bit-for-bit. 🔑 **The quantity is the one
a practitioner has**: you can always compute composition, so what matters is whether paying for a
protein language model buys anything on top of it.

## 2. Frozen bands

Floors **and** ceilings, on **both** arms. Bonferroni α = 0.05/2.

| | the embedding adds information composition lacks | it does not |
|---|---|---|
| **AO-1** recovery(C+E) − recovery(C) | **≥ +6pp** | **≤ +2pp** |
| **AO-2** recovery(C+E) − recovery(E) | — | — (diagnostic, no band) |

🔒 **A verdict needs both arms in the same column.** Between ⇒ reported as a quantity.

🔑 **AO-2 is the symmetric check and it is why this design is worth running even though AO-1 is nearly
implied by study AK.** If the embedding already contains composition's information, adding composition
to it should change nothing: **AO-2 near zero means the embedding subsumes composition**, and **AO-2
clearly positive means it does not** — that composition carries something 1,280 dimensions of ESM-2 do
not, which is the claim studies AJ, AK and AM have been circling without testing.

⚠️ **What this cannot do.** It compares what a **linear** probe extracts, as every result in this paper
does, and it says nothing about non-linear structure. And a null AO-1 would not show the embedding is
uninformative in general — only that a linear probe gains nothing over 21 numbers **on this task, at
this threshold, on this benchmark**.

## 3. Predictions

**P1 (derived — from study AK's +7.19 / +7.53pp increment of E over C).** AO-1 clears **+6pp** on both
arms: concatenating cannot lose information a linear probe already had, so C+E should be at least E.

**P2 (derived — from study AM, where composition out-ordered both embeddings on the toxin pairs, and
study AK, where composition beat the embedding on Exoenzyme by 20 points).** **AO-2 is positive on both
arms** — composition carries something the embedding does not.

**P3 (reasoned).** AO-2 exceeds **+1pp** on at least one arm, concentrated in the categories where
study AK found composition winning: **Exoenzyme** on both arms, Exotoxin on 650M.

⚠️ **Running record: derived 8 of 14, reasoned 2 of 6.** 🔑 **P1 is close to arithmetic and should hold;
P2 is the one that matters**, because the three composition arrivals in §§ 1.2 and 6 all imply it and
none has tested it.

## 4. What each outcome changes

- **AO-1 clears +6pp and AO-2 is positive** ⇒ § 6 can state the thing it has been circling: the two
  feature sets are **complementary**, each carrying signal the other lacks, and the paper's seven-point
  increment is real rather than re-encoding.
- **AO-1 clears +6pp and AO-2 is zero** ⇒ the embedding subsumes composition and the three arrivals are
  about **which categories**, not about information.
- **AO-1 fails** ⇒ 🔴 a linear probe on 1,280 ESM-2 dimensions gains nothing over 21 numbers once both
  are available, and § 8's practical number becomes a statement about redundancy.
