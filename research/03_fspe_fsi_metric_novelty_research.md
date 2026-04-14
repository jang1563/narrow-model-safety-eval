# Deep Research: Novelty Assessment of FSPE and FSI Metrics

*Research date: 2026-04-06*

---

## Executive Summary

After exhaustive searching across PubMed, bioRxiv, arXiv, and the web: **both FSPE and FSI are genuinely novel as formalized metrics**, but they build on well-established individual observations. Neither the exact metric names nor their specific formulations exist in the literature. The underlying phenomena they capture are partially known, meaning the contribution lies in (a) formalizing these observations into quantitative metrics and (b) applying them to dual-use risk evaluation — a framing entirely absent from the literature.

---

## Metric 1: Functional Site Prediction Entropy (FSPE)

### CRITICAL PRIOR ART: Meier et al. (2021) — ESM-1v Paper

**Most important citation.** "Language models enable zero-shot prediction of the effects of mutations on protein function" (NeurIPS 2021):

- Computed **entropy of ESM-1v's masked prediction distribution at each position** as "a measure of [the model's] estimation of conservation."
- Showed **"the lowest entropy predictions cluster at binding sites"** (Figure 5A).
- Demonstrated **"a significant difference between the entropy assignment to binding and non-binding site residues"** — this is essentially the FSPE_functional < FSPE_nonfunctional comparison we propose.
- Illustrated with DNA methyltransferase M.HaeIII: 10 lowest-entropy residues cluster in active site, interact with cytosine substrate.
- Also showed entropy gradients from surface to core in TIM barrel.

**Source:** [bioRxiv](https://www.biorxiv.org/content/10.1101/2021.07.09.450648v1.full) / [NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/f51338d736f95dd42427296047067694-Paper.pdf)

### What is genuinely novel about FSPE

1. **Meier et al. observed the phenomenon but did not formalize a metric.** They showed entropy differs at binding vs. non-binding sites as supporting analysis for variant effect prediction — not as a standalone, named, reproducible evaluation metric.
2. **No one has applied this specifically to toxin proteins** or framed it as a dual-use risk measure.
3. **The application to ESM-2 specifically** (rather than ESM-1v) has not been published with this analysis.
4. **No one has proposed the functional-vs-nonfunctional entropy ratio as a quantitative risk indicator.**

### Additional Related Work

| Tool/Paper | Year | Relevance | Difference from FSPE |
|------------|------|-----------|---------------------|
| **ESM-Scan** (Totaro et al.) | 2023/2024 | Per-residue variant scanning with ESM-2 masked-marginal scoring | Does NOT compute entropy at functional sites. Outputs fitness scores, not entropy. |
| **Pseudo-perplexity / OFS** | 2024 | Pseudo-perplexity from masked predictions as fitness estimator | Whole-sequence level, not per-residue at functional sites. |
| **Masked marginal scoring** (Meier et al.) | 2021 | Standard method: `sum[log p(x_i^mut | x_{-M}) - log p(x_i^wt | x_{-M})]` | Different quantity — log-likelihood ratio, not Shannon entropy of full distribution. |
| **PLM identifies disordered motifs** | 2025 (eLife) | Per-residue entropy from PLMs for conserved functional motifs in disordered regions | Different context (IDRs, not toxin functional sites). |

### Novelty Assessment: **MODERATE-HIGH**

The entropy-at-functional-sites observation exists (Meier et al. 2021). Our contribution is the **formalization as a named metric, specific protocol for toxin proteins, and biosecurity risk framing**. Must cite Meier et al. prominently.

### Recommended Citation Strategy

> "Meier et al. (2021) observed that ESM-1v masked prediction entropy is significantly lower at binding sites than non-binding sites. We formalize this observation into a quantitative metric, FSPE, and apply it specifically to evaluate dual-use risk in protein language models by measuring whether these models encode functional specificity for toxin proteins."

---

## Metric 2: Functional Specificity Index (FSI)

### Related Prior Art

**No one has defined this specific metric.** The ratio (functional site recovery rate) / (mean overall sequence identity) for ProteinMPNN designs does not appear anywhere in the literature.

However, the individual components are well-characterized:

| Component | Value | Source |
|-----------|-------|--------|
| ProteinMPNN overall sequence recovery | 52.4% on native backbones | Dauparas et al., Science 2022 |
| ProteinMPNN at ligand-contacting positions | 50.5% (small mol), 40.6% (metals), 34.0% (nucleotides) — **lower** without ligand context | LigandMPNN, Nature Methods 2025 |
| LigandMPNN at ligand-contacting positions | 63.3% (small mol), 77.5% (metals), 50.5% (nucleotides) — **higher** with ligand context | LigandMPNN, Nature Methods 2025 |
| ABACUS-T pocket residue recovery | Median 0.76 | Nature Communications 2025 |
| ABACUS-T catalytic residue recovery | Median 0.80-1.0 (with ESM integration) | Nature Communications 2025 |
| ProteinMPNN by burial depth | 90-95% deep core; ~35% surface | Dauparas et al. supplementary |

### Critical Note on ABACUS-T

ABACUS-T reports functional and overall recovery **separately** but **does not compute a ratio normalizing functional recovery by overall recovery**. This is the closest existing work.

### Microsoft "Paraphrase Project" (Science, October 2025)

- Used ProteinMPNN and EvoDiff to redesign toxins.
- Demonstrated "structure and active sites preserved while amino acid sequence was rewritten."
- Generated ~70,000 toxin variant sequences evading screening.
- **Did NOT define or publish a quantitative metric** like FSI — evaluation was binary (evade screening or not). Detailed methods deliberately withheld for biosecurity reasons.
- [Science](https://www.science.org/doi/10.1126/science.adu8578)

### "Beyond native sequence recovery" (bioRxiv, January 2026)

- Critiques sequence recovery as a sole metric, arguing for energy-landscape modeling.
- Does NOT propose a functional-vs-overall ratio.
- [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.01.14.699067v1.full)

### Novelty Assessment: **HIGH**

FSI as a defined metric does not exist in the literature. The closest work (ABACUS-T) reports functional and overall recovery separately but does not compute their ratio as a risk indicator. The Microsoft study is the most relevant context but deliberately withheld quantitative metrics.

### Recommended Citation Strategy

> "While sequence recovery at different position types has been characterized for ProteinMPNN (Dauparas et al. 2022) and LigandMPNN (Dauparas et al. 2025), and ABACUS-T (2025) reports pocket and catalytic residue recovery separately, no prior work has defined the ratio of functional site recovery to overall recovery as a metric. We introduce FSI to quantify whether inverse folding models specifically preserve dangerous function beyond what structural similarity alone would predict."

---

## "Stepping Stone" Analysis

### Related Prior Art

| Work | Year | Relevance | Difference |
|------|------|-----------|------------|
| **HalluDesign** | 2025 | Iterative AlphaFold2 hallucination + ProteinMPNN; measures convergence toward structural targets | Tracks structure, not functional recovery |
| **Scalable protein design via relaxed sequence space** | 2024/2025 | Iterative gradient-descent hallucination + ProteinMPNN | Tracks structural convergence, not functional recovery |
| **RFdiffusion + iterative pipelines** | Various | ProteinMPNN used iteratively to improve designability scores | Tracks pLDDT/pTM, not functional site identity |

### Novelty Assessment: **HIGH**

No one has specifically proposed running ProteinMPNN iteratively on redesigned structures and measuring whether **functional site recovery increases toward wild-type identity** over iterations. Existing iterative design work tracks structural metrics, not functional site identity recovery across iterations.

### Recommended Framing

> "Iterative protein design pipelines (HalluDesign, relaxed-sequence hallucination) optimize for structural convergence. We propose tracking functional site identity recovery across iterations as a measure of whether these tools converge toward reconstructing wild-type toxin function."

---

## Broader Dual-Use Evaluation Landscape

| Framework | Year | Focus | Provides Protein-Specific Metrics? |
|-----------|------|-------|----------------------------------|
| **BioRiskEval** | 2025 | Genomic foundation models | No (genome-level perplexity, LD50 prediction) |
| **Pannu et al.** | 2025 | Capability framework for bio AI | No ("developing evaluations will require determining how to concretely measure") |
| **Protein design & biosecurity** | 2026 | Risks review (Frontiers) | No (identifies risks, no quantitative metrics) |
| **Moremi Bio** | 2025 | 1,020 toxic proteins scored | Whole-protein toxicity scoring, not functional-site |
| **Evo 2** | 2025 | Biosecurity evaluation | Used ProteinGym proxy — no protein-functional-site metrics |

---

## Summary Table

| Metric | Exact metric exists? | Underlying phenomenon known? | Applied to toxins/biosecurity? | Overall novelty |
|--------|---------------------|------------------------------|-------------------------------|-----------------|
| **FSPE** | No | Yes (Meier et al. 2021 — entropy at binding sites) | No | **Moderate-High** |
| **FSI** | No | Partially (recovery rates reported separately) | No (except Microsoft with withheld details) | **High** |
| **Stepping Stone** | No | No (iterative design exists but tracks structure) | No | **High** |

---

## Key Sources

- Meier et al. 2021 — ESM-1v. [bioRxiv](https://www.biorxiv.org/content/10.1101/2021.07.09.450648v1.full) / [NeurIPS](https://proceedings.neurips.cc/paper/2021/file/f51338d736f95dd42427296047067694-Paper.pdf)
- ESM-Scan. [Protein Science 2024](https://onlinelibrary.wiley.com/doi/full/10.1002/pro.5221)
- ProteinMPNN. [Science 2022](https://www.science.org/doi/10.1126/science.add2187)
- LigandMPNN. [Nature Methods 2025](https://www.nature.com/articles/s41592-025-02626-1)
- ABACUS-T. [Nature Communications 2025](https://www.nature.com/articles/s41467-025-65175-3)
- Microsoft Paraphrase Project. [Science 2025](https://www.science.org/doi/10.1126/science.adu8578)
- BioRiskEval. [arXiv 2025](https://arxiv.org/abs/2510.27629)
- Pannu et al. [PLOS CompBio 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12061118/)
- HalluDesign. [bioRxiv 2025](https://www.biorxiv.org/content/10.1101/2025.11.08.686881v2.full)
- Beyond native recovery. [bioRxiv 2026](https://www.biorxiv.org/content/10.64898/2026.01.14.699067v1.full)
- Moremi Bio. [arXiv 2025](https://arxiv.org/abs/2505.17154)
- Protein design & biosecurity. [Frontiers 2026](https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2026.1817535/full)
