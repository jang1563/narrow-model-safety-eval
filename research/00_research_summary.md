# Research Summary — Narrow Scientific Model Safety Evaluation

*Compiled: 2026-04-06*

---

## Research Files

| File | Topic | Key Findings |
|------|-------|-------------|
| [01_esm2_dual_use_safety_research.md](01_esm2_dual_use_safety_research.md) | ESM-2 prior work, technical details, toxin databases | No prior empirical dual-use evaluation of ESM-2 exists; our work is novel |
| [02_biosecurity_ai_policy_research.md](02_biosecurity_ai_policy_research.md) | Policy landscape, Lila Sciences context, stepping stones | Field calls for narrow-model-specific evaluations; Lila building safety team now |
| [03_fspe_fsi_metric_novelty_research.md](03_fspe_fsi_metric_novelty_research.md) | Novelty assessment of FSPE, FSI, Stepping Stone | FSI = high novelty; FSPE = moderate-high (cite Meier et al. 2021); Stepping Stone = high |
| [04_proteinmpnn_technical_details.md](04_proteinmpnn_technical_details.md) | ProteinMPNN technical setup, LigandMPNN, PDB structures | Recovery 52.4%; 90-95% buried, ~35% surface; LigandMPNN is key successor |

---

## Critical Findings for Project Design

### 1. Novelty is confirmed

- **No one has evaluated ESM-2's discriminative capabilities for dual-use risk.** All existing work evaluates generative tools (ProteinMPNN, RFdiffusion) or provides policy frameworks.
- **FSI metric is genuinely novel** — functional/overall recovery ratio not defined anywhere.
- **FSPE builds on Meier et al. (2021)** who observed entropy differs at binding sites, but never formalized it as a named metric or applied it to toxin/biosecurity context.

### 2. Key papers to cite

| Paper | Year | Why Critical |
|-------|------|-------------|
| **Meier et al. — ESM-1v** | 2021 | Observed entropy at binding sites; FSPE builds on this |
| **Dauparas et al. — ProteinMPNN** | 2022 | Our evaluation target; recovery benchmarks |
| **Wittmann et al. — Paraphrase Project** | 2025 | Microsoft's 75K toxin variants; closest related work |
| **NIST TEVV — Safe proxies** | 2025 | Experimental validation approach; our work is computational |
| **Sandbrink — LMs vs BDTs** | 2023 | Foundational taxonomy for narrow vs general-purpose safety |
| **Wang et al. (Esvelt) — IAB** | 2026 | PLM + automation risks; stepping stone concept |
| **RAND Global Risk Index** | 2025 | 13 Red-flagged tools; policy context |
| **VF-Fuse** | 2025 | Used ESM-2 for virulence factor prediction; shows embeddings work |

### 3. Technical decisions confirmed

| Decision | Rationale |
|----------|-----------|
| **ESM-2 650M** as primary model | Fits easily on A40; sufficient for proof-of-concept; widely used |
| **ESM-2 3B** as optional upgrade | Fits on A40/A100; stronger representations |
| **Mean pooling, last layer** for embeddings | Best practice per literature |
| **ProteinMPNN (not LigandMPNN)** for initial eval | Tests backbone-only information — the pure dual-use question |
| **100 sequences per structure, temp 0.1** | Standard for recovery analysis |
| **UniProt KW-0800 for toxin sequences** | ~6,300 reviewed entries; REST API available |
| **VFDB for virulence factors** | Bulk download (no API); setA for experimentally verified |

### 4. Positioning relative to existing work

Our work is the first to:
1. Evaluate ESM-2's discriminative capabilities (fill-mask, embeddings) as dual-use risk
2. Define quantitative metrics (FSI, FSPE) for narrow model biosecurity evaluation
3. Combine representation analysis (ESM-2) + generative analysis (ProteinMPNN) in a single evaluation
4. Propose trajectory-based risk assessment (stepping stone analysis) for iterative protein design

We bridge the gap between:
- **Policy literature** calling for capability-based evaluation of biological AI tools (RAND, NIST, EBRC)
- **Bioinformatics literature** demonstrating PLM embeddings can classify toxins (VF-Fuse, ToxDL 2.0, Exo-Tox)

### 5. Connection to applicant's prior work

| Our Finding (Expected) | Prior Work Connection |
|------------------------|---------------------|
| ESM-2 separability AUROC | Extends BioThreat-Eval to narrow models |
| FSPE (entropy at functional sites) | Extends BioRLHF calibration insight to protein predictions |
| FSI (functional recovery ratio) | Operationalizes BioThreat-Eval's "specificity drives risk" for protein design |
| Stepping stone convergence | Connects to Safety Vision's trajectory monitoring concept |
| Physical realizability tier | Leverages BSL-2 wet-lab expertise (Perturb-seq, CRISPR) |

---

## Next Steps

1. **Data collection**: UniProt API for toxin/benign sequences; PDB downloads for structures
2. **HPC setup**: conda env with transformers, torch, fair-esm, scikit-learn on SLURM GPU partition
3. **ESM-2 pipeline**: embed → separability → masked prediction + FSPE
4. **ProteinMPNN pipeline**: redesign → FSI → stepping stone
5. **Report**: figures, evaluation_report.md, README
