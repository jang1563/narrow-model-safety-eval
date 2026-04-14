# Deep Research: ESM-2 Dual-Use Safety Evaluation — Prior Work & Technical Details

*Research date: 2026-04-06*

---

## 1. Prior Work on ESM-2 Safety / Dual-Use Evaluation

### The short answer: Nobody has done a systematic dual-use safety evaluation of ESM-2 specifically. Our work would be novel.

However, several closely related efforts exist:

### A. "Dual-use capabilities of concern of biological AI models" — Pannu, Bloomfield, Cicero, Inglesby et al. (PLoS Computational Biology, May 2025)

- From Johns Hopkins Center for Health Security. Argues that AI model evaluations should prioritize capabilities enabling high-consequence risks (e.g., transmissible disease outbreaks).
- General framework for thinking about dual-use biological AI but does NOT perform any empirical evaluation of ESM-2.
- [DOI: 10.1371/journal.pcbi.1012975](https://doi.org/10.1371/journal.pcbi.1012975)

### B. "Without safeguards, AI-Biology integration risks accelerating future pandemics" — Wang, Huot, Zhang, Jiang, Shakhnovich & Esvelt (Frontiers in Microbiology, Jan 2026)

- Harvard/MIT paper. Maps progress in using protein language models for fitness optimization, assesses dual-use risks and laboratory automation workflows.
- Introduces "Intelligent Automated Biology" (IAB) concept — coupling model-guided sequence design with robotic synthesis and experimental feedback.
- Proposes capability-oriented framework for training- and inference-time safeguards.
- Does NOT empirically evaluate ESM-2 for toxin design/classification.
- [DOI: 10.3389/fmicb.2025.1734561](https://doi.org/10.3389/fmicb.2025.1734561)

### C. "Security challenges by AI-assisted protein design" (EMBO Reports, May 2024)

- Discusses how ProteinMPNN can redesign known toxins using different amino acid building blocks, potentially bypassing homology-based DNA screening.
- Focused on generative design tools, not ESM-2's representation/classification capabilities.
- [EMBO Reports](https://www.embopress.org/doi/10.1038/s44319-024-00124-7)

### D. "Strengthening nucleic acid biosecurity screening against generative protein design tools" — Wittmann et al. (Science, October 2025)

- **The "Paraphrase Project"** (Microsoft Research + Baker Lab).
- Generated >75,000 AI-redesigned variants of hazardous proteins and tested against 4 biosecurity screening tools.
- Found detection of AI-redesigned synthetic homologs was inconsistent.
- Evaluates *generative tools* (ProteinMPNN, RFdiffusion) rather than ESM-2's *discriminative/representation* capabilities.
- [Science](https://www.science.org/doi/10.1126/science.adu8578)

### E. "Experimental evaluation of AI-driven protein design risks using safe biological proxies" (bioRxiv, May 2025; NIST)

- Uses safe proxy proteins to experimentally test whether AIPD tools can redesign sequences to maintain function while evading screening.
- Key finding: current AIPD systems are "not yet powerful enough to reliably rewrite the sequence of a given protein while both maintaining activity and evading detection."
- [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.05.15.654077v2)

### Gap our work fills

No one has evaluated ESM-2's *discriminative* capabilities (fill-mask, embeddings-based classification) as a dual-use risk. Existing literature focuses on generative tools (ProteinMPNN, RFdiffusion) or policy frameworks. Evaluating whether ESM-2 can *identify*, *classify*, or *improve* toxin sequences from embeddings is an unstudied angle.

---

## 2. ESM-2 Technical Details

### 2a. Masked Language Modeling (MLM)

ESM-2 uses standard BERT-style masked language modeling:
- **Training**: 15% of amino acid tokens are selected; of those, 80% replaced with `<mask>`, 10% replaced with random amino acid, 10% left unchanged.
- **Fill-mask capability**: Given a sequence with `<mask>` tokens, produces per-position logits over the 20 standard amino acids (plus special tokens). Enables:
  - Mutational scanning (pseudolikelihood scoring of all single-point mutations)
  - Design scoring (evaluating how "natural" a position is in context)
  - Zero-shot fitness prediction (log-likelihood scores correlate with experimental fitness)

### 2b. Model Sizes

| Model | Layers | Parameters | Hidden Dim | GPU Memory (Inference) |
|-------|--------|-----------|------------|----------------------|
| esm2_t6_8M_UR50D | 6 | 8M | 320 | <1 GB |
| esm2_t12_35M_UR50D | 12 | 35M | 480 | ~1 GB |
| esm2_t30_150M_UR50D | 30 | 150M | 640 | ~2 GB |
| **esm2_t33_650M_UR50D** | 33 | 650M | 1280 | **~4 GB** |
| esm2_t36_3B_UR50D | 36 | 3B | 2560 | ~12-16 GB |
| esm2_t48_15B_UR50D | 48 | 15B | 5120 | ~30 GB (fp16) |

**Recommendation**: 650M and 3B models are practical sweet spots on A40 (48GB) / A100 (40/80GB).

### 2c. Embedding Extraction Best Practices

- **Layer selection**: Last layer is most common. For functional classification, mid-to-late layers (e.g., layers 20-33 of 650M) can outperform final layer by up to 32% in unsupervised settings.
- **Pooling**: Mean pooling over per-residue embeddings consistently outperforms max pooling, BOS/CLS token, PCA, and iDCT.
- **Per-residue embeddings**: Appropriate for predicting individual residue properties (binding sites, active sites).
- **Practical recommendation**: Extract last-layer mean-pooled embeddings (dim 1280 for 650M) as baseline; optionally probe intermediate layers.

Sources: [Nature Scientific Reports](https://www.nature.com/articles/s41598-025-05674-x), [Briefings in Bioinformatics](https://academic.oup.com/bib/article/25/2/bbad534/7590319)

### 2d. ESM-2 for Functional Classification (SOTA)

- ESM-2 650M and ESM-C 600M show "consistently good performance, falling only slightly behind their larger counterparts."
- For downstream classification, ESM-2 embeddings + simple classifiers (linear probes, SVMs, shallow MLPs) achieve strong performance.

---

## 3. Toxin / Dual-Use Protein Databases

### 3a. UniProt Toxin Keyword (KW-0800)

- **Reviewed (Swiss-Prot)**: ~6,300 manually curated toxin proteins (Tox-Prot program)
- **Total (reviewed + unreviewed)**: ~109,442
- **API**: Full REST API, FASTA download supported
- Query: `https://rest.uniprot.org/uniprotkb/search?query=(keyword:KW-0800)%20AND%20(reviewed:true)&format=json&size=1`

### 3b. VFDB (Virulence Factor Database)

- **VFDB 2025** update published in NAR (2025). URL: http://www.mgc.ac.cn/VFs/
- 32 genera of common bacterial pathogens; 14 basal VF categories with >100 subcategories
- **setA** (core, experimentally verified) and **setB** (full, including predicted VFs)
- **No formal REST API.** Access via web interface, BLAST, bulk download.

### 3c. T3DB (Toxin and Toxin Target Database)

- Active at https://www.t3db.ca/
- 3,678 toxins; primarily **environmental/chemical toxins** (small molecules), NOT protein toxins
- **Not directly relevant** for our protein-level evaluation

### 3d. CDC/APHIS Select Agent Toxin List — Well-Characterized Proteins

| Toxin | PDB Entries | Key Features |
|-------|-------------|--------------|
| **Botulinum neurotoxins (A-G)** | 3BTA, 1S0D, 7QFQ | Most potent known protein toxins |
| **Ricin** | 2AAI, 5E1H | Heterodimeric A-B toxin; ~60-65 kDa |
| **Abrin** | 1ABR | Structurally homologous to ricin |
| **Staphylococcal enterotoxins (SEB)** | Multiple | Superantigens |
| **Clostridium perfringens toxins** | Available | Pore-forming toxins |
| **Shiga toxins** | Multiple | Ribosome-inactivating |

### 3e. CARD (Comprehensive Antibiotic Resistance Database)

- Active at https://card.mcmaster.ca/
- v3.2.4: 6,627 ontology terms, 5,010 reference sequences, 5,057 AMR detection models
- **No formal REST API.** Bulk download + RGI command-line tool
- GitHub: https://github.com/arpcard

---

## 4. ESM-2 for Toxin Classification — Existing Work

### Several groups have used PLM embeddings for toxin/virulence classification:

| Tool | Year | PLM Used | Performance | Application |
|------|------|----------|-------------|-------------|
| **Exo-Tox** | 2025 | ProtT5 | MCC > 0.9 | Bacterial exotoxin identification |
| **VF-Fuse** | 2025 | ESM-2 + ProtT5 | F1=87.15%, MCC=73.61% | Virulence factor prediction |
| **ToxDL 2.0** | 2025 | PLM + AlphaFold2 GCN | SOTA on multiple test sets | Protein toxicity prediction |
| **HyPepTox-Fuse** | 2025 | ESM-1, ESM-2, ProtT5 | Competitive | Peptide toxicity prediction |

### Key Insight for Our Work

While ESM-2 embeddings have been used for toxin *classification*, **no one has framed this as a dual-use biosecurity risk evaluation**. The question "how well can ESM-2 distinguish dangerous proteins from safe ones, and does this capability itself constitute a risk?" has not been posed.

---

## 5. Existing Biosecurity Evaluation Frameworks for Protein Models

### No direct equivalent to BioThreat-Eval for narrow protein models exists.

| Framework | Year | Focus | Provides Benchmarks? |
|-----------|------|-------|---------------------|
| **CLTR-RAND Global Risk Index** | 2025 | 57 tools scored Red/Amber/Green | No |
| **EBRC Risk Mitigation for BDTs** | 2025 | Policy, capability-based risk | No |
| **CLTR Capability-Based Assessment** | 2024 | Methodology framework | No |
| **BBG Framework** | 2025 | Biothreat benchmark generation | LLM-focused, not protein models |
| **NIST TEVV (safe proxies)** | 2025 | Experimental validation of AIPD risk | Proxy-based, not metric-based |
| **BioRiskEval** | 2025 | Genomic foundation models | Genome-level, not protein-functional |
| **Moremi Bio** | 2025 | 1,020 toxic protein scoring | Whole-protein, not functional-site |

---

## 6. Novelty Assessment Summary

Our "Narrow Scientific Model Safety Evaluation" would be novel in:

1. **First empirical dual-use safety evaluation of ESM-2's discriminative capabilities** (fill-mask, embeddings)
2. **First concrete evaluation protocol** for narrow protein models' biosecurity risk (all existing work is frameworks/policy)
3. **Novel framing**: ESM-2's classification accuracy on toxins as a biosecurity metric (existing work treats toxin classification as beneficial bioinformatics)
4. **Combined ESM-2 (representation) + ProteinMPNN (generation) evaluation** — first to assess both "understanding" and "design" sides of the dual-use coin

### Recommended Positioning

Our work bridges the gap between:
- (a) the biosecurity policy literature calling for capability-based evaluation of biological AI tools, and
- (b) the bioinformatics literature demonstrating PLM embeddings can classify toxins with high accuracy

— asking the critical question of whether these classification capabilities themselves constitute or enable dual-use risks.
