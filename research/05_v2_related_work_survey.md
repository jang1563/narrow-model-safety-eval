# v2 Related Work Survey — GitHub / HuggingFace / arXiv / bioRxiv / Journals

**Date:** 2026-04-16  
**Purpose:** Comprehensive landscape of existing projects similar or related to Narrow Model Safety Eval v2, for positioning, differentiation, and potential integration.

---

## Table of Contents

1. [Directly Competitive / Most Similar Projects](#1-directly-competitive--most-similar-projects)
2. [Mechanistic Interpretability of Protein LMs](#2-mechanistic-interpretability-of-protein-lms)
3. [Provenance, Watermarking & Traceability](#3-provenance-watermarking--traceability)
4. [Agentic Bio-Capability Benchmarks](#4-agentic-bio-capability-benchmarks)
5. [Toxin Classification & Hazard Screening (Sequence-Level)](#5-toxin-classification--hazard-screening-sequence-level)
6. [DNA Synthesis Screening & Evasion](#6-dna-synthesis-screening--evasion)
7. [Comprehensive Protein Model Evaluation Frameworks](#7-comprehensive-protein-model-evaluation-frameworks)
8. [Policy, Governance & Framework Papers](#8-policy-governance--framework-papers)
9. [Curated Resources & Datasets](#9-curated-resources--datasets)
10. [Differentiation Summary](#10-differentiation-summary)

---

## 1. Directly Competitive / Most Similar Projects

### 1.1 SafeProtein (September 2025) — **CLOSEST COMPETITOR**

| Field | Detail |
|---|---|
| **arXiv** | [2509.03487](https://arxiv.org/abs/2509.03487) |
| **GitHub** | [jigang-fan/SafeProtein](https://github.com/jigang-fan/SafeProtein) |
| **Status** | Public, active |

**What it does:** The first red-teaming framework for protein foundation models. Combines multimodal prompt engineering + heuristic beam search to attack ESM-3, ESMFold, and others.

**SafeProtein-Bench:** 429 experimentally resolved harmful proteins (viral + toxins). Dual-criteria evaluation. Up to 70% attack success rate against ESM-3.

**Approach:** Input-level adversarial attack. Uses Foldseek structural similarity (non-pathogenic proxy) + sequence-masking strategies + beam search score function to generate harmful sequences that fool models.

**Key difference from v2:** SafeProtein attacks INPUT-LEVEL prompts to model (adversarial jailbreak). v2 measures LATENT ENCODING of function in model representations and design outputs (FSI/FSPE/FHS). Complementary, not redundant: SafeProtein tests "can you attack the model", v2 tests "is the function encoded even without attack."

**Relevant v2 cross-reference:** SafeProtein-Bench's 429 protein dataset could extend v2's panel in Pillar 5. Their Foldseek similarity strategy maps to v2's SaProt Foldseek prerequisite.

---

### 1.2 SafeBench-Seq (December 2025)

| Field | Detail |
|---|---|
| **arXiv** | [2512.17527](https://arxiv.org/abs/2512.17527) |
| **GitHub** | [HARISKHAN-1729/SafeBench-Seq](https://github.com/HARISKHAN-1729/SafeBench-Seq) |
| **Author** | Muhammad Haris Khan |

**What it does:** CPU-only baseline for sequence-level protein hazard screening using physicochemical/composition features (NOT protein LMs). Evaluates under homology-clustered holdouts (≤40% identity train/test split) to avoid leakage.

**Key methodological contribution:** Points out that random train/test splits substantially overestimate robustness vs homology-clustered evaluation.

**Key difference from v2:** SafeBench-Seq is a *screening* benchmark (binary toxin/benign classification from sequence features). v2 measures *latent functional encoding* inside generative design models. SafeBench-Seq is downstream screening; v2 is upstream representation analysis.

**Relevant v2 cross-reference:** Homology-clustering protocol directly applicable to v2's expanded protein panel. The ≤40% identity cutoff is the correct standard for panel construction in Pillar 5.

---

### 1.3 NIST TEVV Study — Experimental Evaluation (May 2025)

| Field | Detail |
|---|---|
| **bioRxiv** | [2025.05.15.654077](https://www.biorxiv.org/content/10.1101/2025.05.15.654077v2.full) |
| **NIST page** | [nist.gov/publications/experimental-evaluation-ai-driven-protein-design-risks](https://www.nist.gov/publications/experimental-evaluation-ai-driven-protein-design-risks-using-safe-biological-proxies) |
| **Authors** | Ikonomova, Wittmann et al. |

**What it does:** TESTING, EVALUATION, VALIDATION & VERIFICATION (TEVV) framework for AI-assisted protein design (AIPD). Uses SAFE proteins as proxies for sequences of concern. Experimental wet-lab validation.

**Key findings:** AI biodesign tools generate synthetic homologs with predicted structure similar to native templates *without necessarily retaining function*. When activity IS preserved, the sequence is typically detectable via sequence homology. Highlights importance of key residues and motifs.

**Key difference from v2:** Wet-lab experimental; uses safe proxy proteins. v2 is computational-only on actual toxins (with realized no-synthesis constraint). NIST TEVV validates the physical realizability dimension that v2 only scores theoretically.

**Citation value for v2:** NIST TEVV is the best available wet-lab validation of the computational-physical gap. Directly supports v2's Physical Realizability Tier argument.

---

## 2. Mechanistic Interpretability of Protein LMs

### 2.1 InterPLM (Nature Methods 2025) — DIRECTLY USED IN V2 PILLAR 2

| Field | Detail |
|---|---|
| **Paper** | [Nature Methods — InterPLM](https://pubmed.ncbi.nlm.nih.gov/41023434/) |
| **arXiv** | [2412.12101](https://arxiv.org/abs/2412.12101) |
| **bioRxiv** | [2024.11.14.623630](https://www.biorxiv.org/content/10.1101/2024.11.14.623630v1) |
| **GitHub** | [ElanaPearl/InterPLM](https://github.com/ElanaPearl/InterPLM) |
| **HuggingFace** | [Elana/InterPLM-esm2-650m](https://huggingface.co/Elana/InterPLM-esm2-650m) + [Elana/InterPLM-esm2-8m](https://huggingface.co/Elana/InterPLM-esm2-8m) |
| **Dashboard** | [interplm.ai](https://interplm.ai) |

**What it does:** Trains sparse autoencoders (SAEs) on ESM-2 residual stream. Discovers up to 2,548 interpretable features per layer correlating with 143 biological concepts: binding sites, structural motifs, functional domains.

**Key assets:**
- Pre-trained SAE weights for ESM-2 8M and **650M** (the exact model v2 uses)
- Interactive feature catalog at interplm.ai (searchable by UniProt annotation)
- Every ESM-2 layer explored (layers 6, 18, 30, 33 primarily)

**v2 integration:** SAE weights from `Elana/InterPLM-esm2-650m` are the **primary input for Pillar 2 FHS computation**. No training required. Feature catalog used to build `data/annotations/motif_reference_set.json`.

**Key difference from v2:** InterPLM does not connect interpretability to biosecurity risk. v2 is the first to apply SAE feature analysis to dual-use risk quantification (FHS metric).

---

### 2.2 ProtoMech — Protein Circuit Tracing (February 2026)

| Field | Detail |
|---|---|
| **arXiv** | [2602.12026](https://arxiv.org/abs/2602.12026) |
| **GitHub** | Not yet public (as of April 2026) |

**What it does:** Cross-layer transcoders (CLTs) for computational circuit discovery in ESM-2. Learns sparse joint representations across all transformer layers. Recovers 82–89% of ESM-2 performance on protein family classification with <1% of latent space. Identifies circuits corresponding to structural and functional motifs.

**Notable:** Found a circuit for the HRD catalytic motif in protein kinases — directly relevant to FHS concept.

**v2 relevance:** Stronger interpretability tool than InterPLM for circuit-level analysis. If GitHub code becomes available, could replace or supplement InterPLM SAEs in Pillar 2. Track for code release.

---

### 2.3 Mechanistic Interpretability → Mechanistic Biology (ICML 2025)

| Field | Detail |
|---|---|
| **Paper** | [ICML 2025 spotlight — OpenReview zdOGBRQEbz](https://openreview.net/forum?id=zdOGBRQEbz) |
| **bioRxiv** | [2025.02.06.636901](https://www.biorxiv.org/content/10.1101/2025.02.06.636901v1.full.pdf) |
| **PMC** | [PMC11839115](https://pmc.ncbi.nlm.nih.gov/articles/PMC11839115/) |

**What it does:** Trains and evaluates SAEs on protein LMs, evaluates feature interpretability against known biological annotations. Demonstrates that PLMs store concepts in superposition; SAEs decompose them into biologically coherent features.

**v2 relevance:** Methodological background for Pillar 2. Cited alongside InterPLM to establish SAE interpretability for protein LMs.

---

## 3. Provenance, Watermarking & Traceability

### 3.1 FoldMark (bioRxiv 2024 → PMC 2025)

| Field | Detail |
|---|---|
| **arXiv** | [2410.20354](https://arxiv.org/abs/2410.20354) |
| **bioRxiv** | [2024.10.23.619960](https://www.biorxiv.org/content/10.1101/2024.10.23.619960v7) |
| **PMC** | [PMC11565776](https://pmc.ncbi.nlm.nih.gov/articles/PMC11565776/) |
| **GitHub** | [zaixizhang/FoldMark](https://github.com/zaixizhang/FoldMark) |

**What it does:** Watermarking for protein structure generative models. Distributional + evolutionary watermarking embeds codes across all residues. >95% bit accuracy at 32 bits, scTM >0.9. Supports AlphaFold3, ESMFold, RFDiffusion, RFDiffusionAA.

**v2 relevance:** FoldMark watermarks designs at the structure level. Orthogonal to v2 (v2 evaluates risk; FoldMark traces provenance). **Complementary defensive layer** to the v2 framework: FoldMark could be applied to ProteinMPNN/LigandMPNN designs to enable post-hoc attribution. Mention in v2 Discussion as a downstream defensive tool.

---

### 3.2 StrucTrace (bioRxiv October 2025)

| Field | Detail |
|---|---|
| **bioRxiv** | [2025.10.18.683214](https://www.biorxiv.org/content/10.1101/2025.10.18.683214v1.full.pdf) |

**What it does:** Fourier-domain watermarking for 3D biomolecular structures. Perturbs only flexible backbone atoms; embeds info via frequency modulation. Validated on 40K+ protein structures; perfect recovery; deviations below biological thresholds.

**v2 relevance:** Same as FoldMark — orthogonal but complementary. Mention as defensive infrastructure.

---

### 3.3 ProteinWatermark (Bioinformatics 2025)

| Field | Detail |
|---|---|
| **Journal** | [Bioinformatics, btaf141 (2025)](https://academic.oup.com/bioinformatics/article/41/7/btaf141/8124073) |
| **GitHub** | [poseidonchan/ProteinWatermark](https://github.com/poseidonchan/ProteinWatermark) |

**What it does:** Injects watermarks into ProteinMPNN output sequences by modifying logit distribution at each sampling step. Includes tutorials for modifying ProteinMPNN. Tested on 60 protein structures.

**v2 relevance:** Directly modifies ProteinMPNN — the primary design model in v2. v2 could adopt watermarking as an optional output layer in `src/12_ligandmpnn_fsi.py` to demonstrate responsible design output handling.

---

## 4. Agentic Bio-Capability Benchmarks

### 4.1 ABC-Bench (NeurIPS 2025, October 2025)

| Field | Detail |
|---|---|
| **OpenReview** | [mo5H9VAr6r](https://openreview.net/forum?id=mo5H9VAr6r) |
| **NeurIPS** | [neurips.cc 131273](https://neurips.cc/virtual/2025/loc/san-diego/131273) |
| **GitHub** | [uiuc-kang-lab/agentic-benchmarks](https://github.com/uiuc-kang-lab/agentic-benchmarks) |

**What it does:** Agentic Bio-Capabilities Benchmark for Biosecurity. LLM agents on: (1) liquid handling robot code, (2) DNA fragment design for in vitro assembly, (3) DNA synthesis screening evasion. Grok 3 achieves 53%, outperforming PhD biologists (24%).

**Key result:** GPT-4o-mini-high produced code that, run on an OpenTrons robot, successfully assembled DNA in 3 independent experiments.

**v2 relevance:** Directly motivates Pillar 4 (Stepping Stone trajectory). ABC-Bench measures LLM-level capability; Pillar 4 measures protein-model-level iterative exploitation capability. Cite ABC-Bench as evidence that the agentic threat model is real.

---

### 4.2 ABLE — Agentic BAIM–LLM Evaluation (NeurIPS 2025)

| Field | Detail |
|---|---|
| **OpenReview** | [3fd094f3a011ca4820836bd6abf0dd01ca1e28f8](https://openreview.net/pdf/3fd094f3a011ca4820836bd6abf0dd01ca1e28f8.pdf) |
| **NeurIPS** | [neurips.cc 131277](https://neurips.cc/virtual/2025/loc/san-diego/131277) |

**What it does:** Benchmarks LLM agent's ability to use Biological AI Models (BAIMs — ProteinMPNN, AlphaFold3) in a dual-use protein design workflow. Evaluates: structure retrieval, design approach, sequence generation via ProteinMPNN, validation via AlphaFold3.

**Key finding:** Agents can effectively use protein design tools but are inconsistent at multi-step computational workflows.

**v2 relevance:** ABLE evaluates the same tool chain (ProteinMPNN → folding → interpretation) that Pillar 4 uses for Stepping Stone trajectory. ABLE is external-agent perspective; Pillar 4 is internal-model-mechanics perspective. Complementary framing.

---

## 5. Toxin Classification & Hazard Screening (Sequence-Level)

### 5.1 BioLMTox — Fine-Tuned ESM-2 Toxin Classifier (bioRxiv 2024)

| Field | Detail |
|---|---|
| **bioRxiv** | [2024.04.14.589430](https://www.biorxiv.org/content/10.1101/2024.04.14.589430v1.full) |
| **HuggingFace** | Available via biolm.ai API |

**What it does:** Fine-tunes ESM-2 on improved unified toxin dataset. Achieves validation accuracy 0.964, recall 0.984. Identifies toxins from multiple domains of life in sub-second time.

**Key difference from v2:** BioLMTox is a TOXIN CLASSIFIER (sequence → is it toxic?). v2 probes WHAT IS ENCODED and HOW SPECIFICALLY in design model representations. BioLMTox is downstream classification; v2 is upstream representation analysis.

**v2 cross-reference:** BioLMTox could serve as a reference positive-classification baseline against which SER-P is measured in Pillar 3.

---

### 5.2 ToxClassifier (GitHub)

| Field | Detail |
|---|---|
| **GitHub** | [rgacesa/ToxClassifier](https://github.com/rgacesa/ToxClassifier) |

**What it does:** ML classifier for toxin identification from protein sequence.

---

### 5.3 VF-Fuse — Virulence Factor Prediction (Briefings in Bioinformatics 2025)

| Field | Detail |
|---|---|
| **Journal** | [Briefings in Bioinformatics, bbaf481 (2025)](https://academic.oup.com/bib/article/26/5/bbaf481/8260786) |

**What it does:** Dual-path feature fusion (ESM-2 + ProtT5) for virulence factor prediction. F1=87.15%. Uses ensemble of 15 combination methods; Majority Voting best.

**v2 relevance:** Demonstrates ESM-2 + ProtT5 ensemble for hazard prediction. v2's Pillar 1C could add ProtT5 embeddings alongside ESM-3 to replicate VF-Fuse's dual-path approach for FSPE.

---

### 5.4 NTxPred2 — Neurotoxic Peptide Prediction (bioRxiv 2025)

| Field | Detail |
|---|---|
| **bioRxiv** | [2025.03.01.640936](https://www.biorxiv.org/content/10.1101/2025.03.01.640936v1.full.pdf) |

**What it does:** Multiple PLMs for neurotoxic peptide/protein prediction.

---

## 6. DNA Synthesis Screening & Evasion

### 6.1 Wittmann et al. — Microsoft Paraphrase Project (Science October 2025) — KEY PAPER

| Field | Detail |
|---|---|
| **Journal** | [Science, adu8578 (2025)](https://www.science.org/doi/10.1126/science.adu8578) |
| **PubMed** | [41037625](https://pubmed.ncbi.nlm.nih.gov/41037625/) |
| **Microsoft Research** | [Paraphrase Project](https://www.microsoft.com/en-us/research/project/paraphrase-project/) |

**What it does:** 76,080 toxin sequence variants from EvoDiff/ProteinMPNN/ESM-based models. Initial DNA screening flag rate: 23–70%. After patches: ~72% average / 97% for top-risk sequences.

**v2 relationship:** This is the DOWNSTREAM paper that v2's SER-N (Pillar 3) bridges to. Wittmann measures evasion AFTER codon optimization and DNA synthesis screening. v2 measures whether function is encoded in model representations BEFORE any sequence output. Together they span the full upstream-to-downstream risk picture.

**Critical** for positioning: cite explicitly in v2's SER section as "our work provides the upstream representation-level signal that Wittmann's DNA-screening-level work cannot capture."

---

### 6.2 Securing Dual-Use Pathogen Data (arXiv February 2026, NeurIPS 2025)

| Field | Detail |
|---|---|
| **arXiv** | [2602.08061](https://arxiv.org/abs/2602.08061) |
| **OpenReview** | [ZgD951FVe7](https://openreview.net/forum?id=ZgD951FVe7) |
| **NeurIPS** | [neurips.cc 131281](https://neurips.cc/virtual/2025/loc/san-diego/131281) |
| **Authors** | Doni Bloomfield et al. (endorsed by 100+ researchers at Asilomar 50th anniversary) |

**What it does:** Five-tier Biosecurity Data Level (BDL) framework for categorizing pathogen data used in AI training. Proposes technical restrictions per tier.

**v2 relevance:** BDL framework directly applicable to v2's Physical Realizability Tier design. Consider whether v2's Realizability Tiers should align with BDL tiers for cross-comparability.

---

### 6.3 AI Bioweapons and the Failure of Inference-Time Filters (NeurIPS 2025)

| Field | Detail |
|---|---|
| **OpenReview** | [NAr5wbHghS](https://openreview.net/forum?id=NAr5wbHghS) |
| **NeurIPS** | [neurips.cc abstract](https://neurips.cc/virtual/2025/loc/san-diego/131286) |

**What it does:** Shows inference-time content filters for bioweapon-relevant outputs fail systematically. Relevant to LLM biosafety but informs the broader landscape.

---

## 7. Comprehensive Protein Model Evaluation Frameworks

### 7.1 ProteinBench (ICLR 2025)

| Field | Detail |
|---|---|
| **arXiv** | [2409.06744](https://arxiv.org/abs/2409.06744) |
| **Website** | [proteinbench.github.io](https://proteinbench.github.io/) |
| **Leaderboard** | [proteinbench-proteinbench.hf.space](https://proteinbench-proteinbench.hf.space) |
| **HuggingFace** | Interactive leaderboard |

**What it does:** First holistic benchmark for protein foundation models. Evaluates quality, novelty, diversity, robustness across: inverse folding, structure design, sequence design, co-design, motif scaffolding, antibody design, folding, multi-state prediction, distribution prediction.

**v2 relevance:** ProteinBench provides the motif scaffolding leaderboard that situates EvoDiff and ProteinMPNN performance. v2's FSI-EvoD and FSI-LM can be contextualized against ProteinBench's motif scaffolding metrics (they measure design *quality*; v2 measures design *functional specificity for dangerous sites*).

**Differentiation:** ProteinBench does not evaluate biosafety dimensions (FSI, SER, FHS). v2 adds the safety orthogonal to ProteinBench's quality metrics.

---

### 7.2 WMDP — Weapons of Mass Destruction Proxy Benchmark (2024)

| Field | Detail |
|---|---|
| **arXiv** | [2403.03218](https://arxiv.org/abs/2403.03218) |
| **HuggingFace** | [cais/wmdp](https://huggingface.co/datasets/cais/wmdp) |
| **Website** | [wmdp.ai](https://www.wmdp.ai/) |

**What it does:** 3,668 multiple-choice questions as proxy for hazardous knowledge in biosecurity, cybersecurity, chemical security. Used for unlearning evaluation.

**Key difference from v2:** WMDP tests *text-level knowledge* in LLMs. v2 tests *structural/functional encoding* in narrow protein models. Complementary: WMDP is the LLM equivalent; v2 is the narrow-model equivalent.

---

## 8. Policy, Governance & Framework Papers

### 8.1 Pannu et al. — Dual-Use Capabilities of Biological AI Models (PLOS Computational Biology 2025)

| Field | Detail |
|---|---|
| **Journal** | [PLOS Computational Biology, e1012975 (May 2025)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012975) |
| **PMC** | [PMC12061118](https://pmc.ncbi.nlm.nih.gov/articles/PMC12061118/) |
| **Authors** | Pannu, Bloomfield, Zhu, MacKnight, Gomes, Cicero, Inglesby (Johns Hopkins CHS, Stanford, Fordham, CMU) |

**What it does:** Proposes categories of dual-use capabilities for standardized AI biosecurity evaluation. Recommends evaluating prior to deployment. Distinguishes information hazard (LLMs) from generative capability hazard (protein design models).

**v2 relevance:** The PLOS paper provides the POLICY FRAMING that v2 exemplifies technically. Cite as foundational policy motivation for why FSI/SER/FHS matter.

---

### 8.2 Resilient Biosecurity in the Era of AI-Enabled Bioweapons (arXiv August 2025)

| Field | Detail |
|---|---|
| **arXiv** | [2509.02610](https://arxiv.org/abs/2509.02610) |
| **Authors** | Feldman & Feldman |

**What it does:** Evaluates PPI prediction tools (AlphaFold3, AF3Complex, SpatialPPIv2) on viral-host interactions. Shows they fail systematically on viral proteins. Argues this creates exploitable blind spots.

**v2 relevance:** Confirms that computational tools have structural failure modes for biological threats. Complements v2's finding that FSI=0 for anthrax PA is a meaningful safety result (not a model deficiency).

---

### 8.3 Generative AI for Biosciences: Emerging Threats and Roadmap to Biosecurity (arXiv October 2025)

| Field | Detail |
|---|---|
| **arXiv** | [2510.15975](https://arxiv.org/abs/2510.15975) |

**What it does:** Comprehensive threat analysis from 130 expert interviews. AI-native biosafety safeguards must be embedded directly into model behavior. Reviews jailbreaks, dual-use challenges, autonomous AI agents.

---

### 8.4 NeurIPS 2025 Workshop — Biosecurity Safeguards for Generative AI

| Field | Detail |
|---|---|
| **Website** | [biosafe-gen-ai.github.io](https://biosafe-gen-ai.github.io/) |
| **OpenReview** | [NeurIPS.cc/2025/Workshop/BioSafe_GenAI](https://openreview.net/group?id=NeurIPS.cc%2F2025%2FWorkshop%2FBioSafe_GenAI) |

**What it does:** NeurIPS 2025 workshop uniting AI, synthetic biology, biosecurity policy, and ethics. Featured: biosafety benchmarking for protein design, DNA LM watermarking, securing dual-use pathogen data, agentic AI risk evaluation.

**v2 relevance:** This workshop is the PRIMARY SUBMISSION VENUE for a v2 conference paper. Highly relevant community. Accepted papers overlap directly with v2 scope.

---

### 8.5 Biosecurity for Synthetic Nucleic Acid Sequences — NIST Program

| Field | Detail |
|---|---|
| **NIST** | [nist.gov/programs-projects/biosecurity-synthetic-nucleic-acid-sequences](https://www.nist.gov/programs-projects/biosecurity-synthetic-nucleic-acid-sequences) |

The broader NIST program providing policy and measurement standards for v2's SER metric framework.

---

## 9. Curated Resources & Datasets

### 9.1 awesome-biosecurity-datasets (GitHub)

| Field | Detail |
|---|---|
| **GitHub** | [martinholub/awesome-biosecurity-datasets](https://github.com/martinholub/awesome-biosecurity-datasets) |

Curated list of public biosecurity datasets and resources for building biosecurity tools. Useful for finding reference databases for v2's SER computation and panel expansion.

---

### 9.2 BioRAM — Sandia National Labs Biosecurity Risk Assessment Tool

| Field | Detail |
|---|---|
| **GitHub** | [sandialabs/BioRAM](https://github.com/sandialabs/BioRAM) |
| **GCBS Program** | [gcbs.sandia.gov/tools/biosecurity-biosafety-ram](https://gcbs.sandia.gov/tools/biosecurity-biosafety-ram/) |

**What it does:** Systematic biorisk assessment tool for BSL laboratories. Factors: biological agent properties, lab security, mitigation measures, local threat likelihood. Excel + Java versions. BioRAM-2022 merges safety + security assessment.

**v2 relevance:** BioRAM's 5-dimensional risk framework is a physical-world validation of v2's Physical Realizability Tier design. v2's manual 5-dimension scoring is philosophically aligned with BioRAM's approach. Can cite BioRAM to legitimize the tier framework methodology.

---

## 10. Differentiation Summary

| Project | Scope | Level | v2 Relation |
|---|---|---|---|
| **SafeProtein** | Input-level adversarial jailbreak of protein models | Model input/output | Competitive but orthogonal; attacks inputs, v2 measures latent encoding |
| **SafeBench-Seq** | Sequence-level binary hazard classification | Sequence features | Downstream screening; v2 is upstream representation |
| **NIST TEVV** | Wet-lab TEVV with safe proxies | Experimental | Ground truth for physical realizability dimension |
| **InterPLM** | SAE interpretability of ESM-2 | Representations | v2 USES InterPLM weights directly (Pillar 2) |
| **ProtoMech** | Circuit tracing in ESM-2 | Representations | Stronger Pillar 2 tool; track for code release |
| **FoldMark/StrucTrace** | Structure provenance watermarking | Output | Defensive infrastructure orthogonal to v2 |
| **ABC-Bench/ABLE** | Agentic bio-capability benchmarks | LLM+Tool agents | Motivates Pillar 4; LLM-level vs v2's model-mechanics level |
| **Wittmann/Paraphrase** | DNA synthesis screening evasion | DNA-level | v2's SER bridges upstream (v2) to downstream (Wittmann) |
| **ProteinBench** | Holistic protein model quality evaluation | Quality metrics | v2 adds safety dimension to ProteinBench quality metrics |
| **WMDP** | Hazardous knowledge in LLMs (MCQ) | LLM text-level | v2 is the narrow-model structural analogue of WMDP |
| **Pannu et al.** | Policy framework for AI dual-use categories | Policy | Foundation policy paper motivating v2's metric design |

### v2's Unique Position

**No existing project:**
1. Applies SAE interpretability (InterPLM-style) to biosecurity risk quantification (FHS — Pillar 2)
2. Defines FSI as a systematic metric across multiple design model families (Pillar 1)
3. Connects model-latent functional encoding directly to DNA synthesis evasion (SER, Pillar 3)
4. Measures iterative/agentic exploitation depth quantitatively (Stepping Stone N*, Pillar 4)
5. Integrates all four dimensions into a single Multi-Dimensional Risk Profile per protein × model

v2 occupies the upstream, representation-level, cross-model safety analysis niche that lies between interpretability work (InterPLM) and downstream screening work (Wittmann/SafeBench-Seq).

---

*Survey compiled 2026-04-16. Sources: GitHub, HuggingFace, arXiv, bioRxiv, PubMed, Science, PLOS CompBio, NeurIPS 2025 OpenReview.*
