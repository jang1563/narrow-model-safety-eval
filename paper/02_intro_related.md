
---

## 1. Introduction

A protein language model's embedding space separates toxins from benign proteins well enough that a
linear probe reaches AUROC 0.974 on a curated panel. It is a short step from there to a claim about
biosecurity screening, and the step is where the trouble is. This paper is an attempt to take that step
slowly, in public, with the failures written down as they were found.

Our contribution is not a better classifier. It is an account of what an evaluation has to do before a
separability number licenses a screening claim, together with the record of what that account cost us:
sixty dated corrections, two preregistrations of which one returned NOT SUPPORTED, and three findings
that looked clean on one representation and vanished on a second.

We make three empirical claims and one methodological one.

1. **The aggregate hides a bimodal failure structure.** Leave-one-mechanism-out recovery spans 100% to
   10%, and the classes that fail are not the small ones (§ 2).
2. **The failures are locatable in advance.** Class-level margin ranks them at *ρ* = +0.894 and
   transfers across representations and re-poolings, while over-predicting the hardest class by 34
   points (§ 4).
3. **Nothing we tried improved the screen.** Precision is flat at about 1% across every configuration
   tested (§ 5).
4. **Methodologically: the ordering of criteria matters more than the tally.** The split between
   calibration and test negatives is first because the other seventeen are measured through it (§ 3).

⚠️ **What this paper is not.** It is not a benchmark, a tool, or a deployment recommendation. § 7 lists
what it does not claim, and every number is on a self-built panel of 234 or 445 proteins that is **not
comparable** to the virulence-factor benchmarks in § 1.2.

---

## 1.2 Related work, and where this sits

### The virulence-factor classifier line

The established comparison is a sequence-classification line built on one dataset. **DeepVF**
([Briefings in Bioinformatics 22(3) bbaa125, 2021](https://academic.oup.com/bib/article/22/3/bbaa125/5864586))
assembled 3,576 virulence factors and 4,910 non-VFs, held out **576 of each** as an independent test
set, and reported **AUC 0.896**. **DTVF** ([Genes 15(9) 1170,
2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11430887/), ProtT5 with an LSTM–CNN dual channel) reuses
that pool and reports **AUROC 0.9208**. ⚠️ *DTVF states the shared 3,576/4,910 pool and cites DeepVF for
it but never states its own split, so "0.92 on the 576/576 benchmark" is an inference from DeepVF's
construction and is written here as one.* **DeepVIC** ([Bioinformatics Advances 6(1) vbag237,
2026](https://academic.oup.com/bioinformaticsadvances/article/6/1/vbag237/8762933)) is larger again:
ProtBert-BFD embeddings over **33,456 VFs**, **AUROC 0.954** on a 13,384-sequence holdout, plus
multiclass assignment to **14 VFDB categories** at 0.838 accuracy.

🔑 **The comparison we care about is not the AUROC.** DeepVIC carries a published 14-category class axis
over 12,989 annotated VFs and runs **no leave-one-category-out evaluation**; its generalization check is
recall on two organism-specific positive-only sets (71.4%, 45 of 63). The largest virulence-factor
classifier in the literature therefore holds the class axis this work needs and does not ask the
class-level question this work exists to ask. An adjacent paper does ask a question of that shape —
leave-one-EC3-class-out over 161 subclasses, 47.7% EC1 recovery against a 14.3% baseline
([arXiv 2606.12209](https://arxiv.org/abs/2606.12209)) — and reports per-class variation with a
narrative explanation, but no quantity computable *before* the class is held out.

### Hazard work on protein foundation models

**SafeProtein** ([arXiv 2509.03487](https://arxiv.org/abs/2509.03487)) red-teams generation, reporting
**up to 70% attack success** against ESM-3. That is a generation-time question; ours is whether hazard is
linearly readable in a frozen representation, which exists without any adversarial prompt. ⚠️ Its
429-protein benchmark is not downloadable: we recovered 66 identities from the paper's text, and a second
group reports **275 pairs** from the same unreleased set
([VFUSE, arXiv 2606.10080](https://arxiv.org/html/2606.10080)).

**VFUSE** is the nearest neighbour in intent — Matryoshka BatchTopK sparse autoencoders on
RoseTTAFold3 and RFDiffusion3 activations, AUROC **0.877 ± 0.025** on a random split and **0.817 ±
0.102** under homology-clustered cross-validation. 🔑 Its control is **member-level** homology clustering
at 30% identity, not a held-out mechanism class, on **n = 275 pairs**. The class-level question, and a
pre-hoc predictor of which class fails, are not occupied by it.

### What the framing borrows

The margin statistic is not new. Predicting generalization from margin distributions is a named
programme (Jiang et al., [ICLR 2019](https://arxiv.org/abs/1810.00113); the NeurIPS 2020 PGDL
competition), and *k*-nearest-neighbour distance in embedding space as an out-of-distribution score is
established (Sun et al., [ICML 2022](https://arxiv.org/abs/2204.06507)). 🔑 **What is ours is the level
and the target**: the statistic applied at **class** level, **before** the class is trained on, to a
safety screen's **false-negative** structure — and the finding that prediction and repair dissociate.
