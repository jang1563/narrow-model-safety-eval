# Deep Research: AI Biosecurity Policy Landscape & Lila Sciences Context

*Research date: 2026-04-06*

---

## 1. Current Policy Landscape for AI Biosecurity Evaluation (2024-2026)

### Think Tank Publications

#### RAND Corporation

- **Global Risk Index for AI-enabled Biological Tools** (September 2025, with Centre for Long-Term Resilience): First structured, scalable framework for assessing AI-enabled biological tools. From 1,107 tools, shortlisted 57 for detailed examination. **13 tools flagged "Red" (requiring immediate action)**; 61.5% of Red-flagged tools are fully open-sourced. Methodology uses eight functional categories with misuse-relevant capabilities and maturity/availability scoring.
  - [RAND report](https://www.rand.org/pubs/external_publications/EP71093.html)
  - [CLTR page](https://www.longtermresilience.org/reports/global-risk-index-for-ai-enabled-biological-tools/)

- **Developing a Risk-Scoring Tool for AI-Enabled Biological Design**: Dual-component risk-scoring tool combining biological modification risk factors with actor capability assessments.
  - [RAND](https://www.rand.org/pubs/research_reports/RRA4490-1.html)

- **Dissecting America's AI Action Plan: A Primer for Biosecurity Researchers** (August 2025)
  - [RAND](https://www.rand.org/pubs/commentary/2025/08/dissecting-americas-ai-action-plan-a-primer-for-biosecurity.html)

#### CSET (Georgetown)

- **AI Safety Evaluations: An Explainer**: Evaluations should assess whether an AI model could enhance biological risk by helping a non-expert make a bioweapon.
  - [CSET](https://cset.georgetown.edu/article/ai-safety-evaluations-an-explainer/)

- **AI and Biorisk: An Explainer**: Distinguishes that frontier LLMs "pose different kinds of biosecurity risks than biological AI models (BAIMs)." LLMs expand the *number* of bad actors; some BAIMs raise the *ceiling* of possible harm.
  - [CSET](https://cset.georgetown.edu/publication/ai-and-biorisk-an-explainer/)

- **Anticipating Biological Risk: A Toolkit for Strategic Biosecurity Policy** (December 2024)
  - [CSET PDF](https://cset.georgetown.edu/wp-content/uploads/CSET-Anticipating-Biological-Risk.pdf)

#### Open Philanthropy

- Funded the **Biosecurity AI Research Fund Program** (via AI Safety Fund). Grants $350-600K range.
- [RFP on improving capability evaluations for AI governance](https://www.openphilanthropy.org/request-for-proposals-technical-ai-safety-research/)
- Funded Blueprint Biosecurity, Mirror Biology Dialogues Fund, individual biosecurity researchers.

---

### Executive Order 14110 (October 2023)

- **Section 4.4** calls for evaluating/mitigating CBRN risks from AI.
- Directs OSTP to establish nucleic acid synthesis screening framework.
- Makes screening a condition of federal research funding.
- **Revoked by Trump administration January 2025**; partially replaced by May 2025 EO "Improving the Safety and Security of Biological Research."
- [EO full text](https://www.presidency.ucsb.edu/documents/executive-order-14110-safe-secure-and-trustworthy-development-and-use-artificial)
- [CSET breakdown](https://cset.georgetown.edu/article/breaking-down-the-biden-ai-eo-screening-dna-synthesis-and-biorisk/)

### OSTP Nucleic Acid Synthesis Screening Framework

- Released April 2024, finalized September 2024.
- Requires screening every 200-nucleotide window, decreasing to 50 nucleotides by October 2026.
- **Critical gap (Baker & Church)**: "Screening sequences alone may not be sufficient because proteins generated through de novo design may have little or no sequence similarity to any natural proteins."
- [OSTP Framework PDF](https://bidenwhitehouse.archives.gov/wp-content/uploads/2024/10/OSTP-Nucleic-Acid_Synthesis_Screening_Framework-Sep2024-Final.pdf)

### NIST AI Risk Management Framework and AISI

- **NIST AI 600-1** (July 2024): Notes chemical/biological design tools (BDTs) are "highly specialized AI systems" requiring strict oversight.
  - [NIST AI 600-1 PDF](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf)

- **October 2024**: AISI issued **RFI on "Safety Considerations for Chemical and/or Biological AI Models"** (Federal Register 89 FR 80886). Explicitly names protein design tools, small biomolecule design tools, viral vector design tools, genome assembly tools, experimental simulation tools, and autonomous experimental platforms.
  - [Federal Register](https://www.federalregister.gov/documents/2024/10/04/2024-22974/safety-considerations-for-chemical-andor-biological-ai-models)

- **EBRC response to AISI RFI**: Argued inputs/outputs are not standardized across chem-bio models, making generalized evaluations challenging. Advocated for risk-tiered deployment.
  - [EBRC PDF](https://ebrc.org/wp-content/uploads/2024/12/NIST-AISI-RFI.pdf)

- **February 2024**: NIST partnered with nonprofit research consortium for synthetic biology safety tools.

---

## 2. Key Papers on AI-Enabled Biological Risk

### Soice et al. (2023) — "Can large language models democratize access to dual-use biotechnology?"

[arXiv:2306.03809](https://arxiv.org/abs/2306.03809)

- MIT study: non-scientist students used LLM chatbots. In **one hour**, chatbots:
  1. Suggested four potential pandemic pathogens
  2. Explained reverse genetics generation from synthetic DNA
  3. Supplied DNA synthesis companies unlikely to screen orders
  4. Provided detailed protocols and troubleshooting
  5. Recommended core facilities for those lacking lab skills
- Core argument: LLMs dramatically increase the number of potential bad actors.

### Mouton et al. (RAND, 2024) — LLM Uplift for Bioweapons

[RAND report](https://www.rand.org/pubs/research_reports/RRA2977-2.html)

- **No statistically significant difference** in viability of plans generated with or without LLM assistance.
- LLM outputs generally mirrored information readily available on the internet.
- **2025 updates**: Anthropic found "some amount of uplift to novices"; OpenAI expects "high risks" in the near future.

### Papers on Protein Design Models and Biosecurity

| Paper | Year | Key Finding |
|-------|------|-------------|
| **Sandbrink — "Differentiating risks of LMs and BDTs"** | 2023 | LLMs = democratization; BDTs = ceiling-raising. BDTs "could enable creation of pandemic pathogens substantially worse than anything seen to date." [arXiv:2306.13952](https://arxiv.org/abs/2306.13952) |
| **Baker & Church — "Protein design meets biosecurity"** | 2024 | AI protein design tools can generate functional variants that evade synthesis screening. Called for enhanced screening, universal logging. [Science](https://www.science.org/doi/10.1126/science.ado1671) |
| **Wittmann et al. — "Paraphrase Project"** | 2025 | ~75,000 AI-redesigned variants of hazardous proteins. ~3% evade all screening. [Science](https://www.science.org/doi/10.1126/science.adu8578) |
| **Schubert et al. — "Security challenges by AI-assisted protein design"** | 2024 | De novo protein design poses new biosecurity/biosafety threats. [EMBO Reports](https://pmc.ncbi.nlm.nih.gov/articles/PMC11094011/) |
| **NIST TEVV — Safe biological proxies** | 2025 | AIPD can generate structural homologs but without necessarily retaining activity. [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.05.15.654077v2.full) |
| **Pannu et al. — "Dual-use capabilities of concern"** | 2025 | Capability-based evaluation framework. Recommends focusing on models trained >10^26 operations. [PLOS CompBio](https://pmc.ncbi.nlm.nih.gov/articles/PMC12061118/) |
| **Responsible AI x Biodesign** | 2024 | 170+ scientists (Baker, Arnold, Church, Horvitz) signed community standards. [responsiblebiodesign.ai](https://responsiblebiodesign.ai/) |

### Kevin Esvelt's Work

- MIT Media Lab professor, founder of [SecureBio](https://securebio.org/) (stepped back September 2024).
- Co-authored 2025 Frontiers paper warning that PLMs trained on millions of sequences can predict, generate, and optimize functional proteins with minimal human input.
- Key position: Traditional regulatory measures (DURC policy, static lists) are globally insufficient.
- **SecureBio 2025**: Developed the **Virology Capabilities Test (VCT)**: 322 multimodal questions. OpenAI's o3 achieved 43.8% accuracy, outperforming 94% of expert virologists (human average: 22.1%).

---

## 3. The "Narrow Model" vs "General-Purpose Model" Safety Distinction

### Key Articulations

1. **Sandbrink (2023)**: LLMs = information hazards (*tell* people how); BDTs = capability hazards (*do* dangerous things directly).
2. **CSET**: LLMs expand *breadth* of who can do harm; BAIMs expand *depth* of what can be done.
3. **EBRC NIST RFI response**: "Inputs and outputs are not standardized across chem-bio models, making generalized evaluations challenging."
4. **NIST AISI RFI (2024)**: Predicated on the recognition that chemical/biological AI models need their own safety evaluation approach.
5. **PLOS CompBio (2025)**: "Dual-use capabilities of concern should be translated into model type-specific evaluation methods."

### Methodological Differences for Narrow Models

| Dimension | LLM Safety | Narrow Model Safety |
|-----------|-----------|-------------------|
| I/O format | Text in / text out | Sequence / structure in / design out |
| Dual-use separation | Can separate "safe" from "dangerous" capabilities | Core therapeutic functions = same as misuse functions |
| Validation | Text analysis sufficient | Wet-lab synthesis required for definitive validation |
| Convergence | Standalone | LLMs can orchestrate BAIM workflows, blurring the distinction |

### Current Consensus on PLM Dual-Use Risk (2024-2026)

- **2024 (early)**: NIST TEVV found current tools "not yet powerful enough" to reliably rewrite + maintain activity + evade screening.
- **2024-2025**: Baker & Church and Paraphrase Project showed gap is closing (~3% evade all screening).
- **2025**: RAND flagged 13 tools as "Red"; OpenAI flagged "substantial and immediate biorisks" from near-future models.
- **Key nuance**: AI tools currently cannot design self-replicating pathogens (insufficient viral training data). Highest-concern capability remains out of reach but forecasted to arrive.

### Information Hazard Debate

- For LLMs: standard information hazard (provides instructions).
- For PLMs: generates **novel molecular designs** that never existed — not "information" in traditional sense, but **generative capability**.
- AlphaFold 3's restricted release (API-only, 10 submissions/day) reflected Google DeepMind's implicit judgment that structure prediction constitutes information hazard — met with backlash from >1,000 scientists.

---

## 4. Lila Sciences Safety Context

### Company Overview

- Founded in Flagship Pioneering's labs (2023), unveiled March 2025.
- "World's first scientific superintelligence platform and fully autonomous labs."
- **CEO**: Geoffrey von Maltzahn (Flagship General Partner since 2009)
- **Funding**: ~$550M total ($200M seed + ~$350M Series A, valuation >$1.3B)
- **Philosophy**: Rejects hard-coding expert knowledge; follows Rich Sutton's "Bitter Lesson" — single general platform for autonomous science.

### Public Safety Statements

> "We scale experimentation and accelerate timelines without compromising the rigor that makes science reliable."
> "A culture of safety, attuned to human impact, and grounded in scientific rigor, not reckless experimentation."
> "It is essential that the mission be pursued ethically, safely, and responsibly."

### Safety Team (Building Now)

| Role | Salary | Focus |
|------|--------|-------|
| Scientist/Sr. Scientist, AI Safety | $228K-$358K | Build evaluations, proof-of-concept safeguards |
| **Senior/Principal RS, AI Safety (Bio/Phys)** | $268K-$384K | **Set safety research strategy**, threat modeling, capability evaluation |
| Research Scientist I/II, AI Safety (Bio/Phys) | — | Execute evaluations |

Key from job postings: "evaluations to test for scientific risks (both known but especially novel) from cutting edge scientific models integrated with automated physical labs."

### Kenneth Stanley and Open-Endedness

- **SVP of Open-Endedness** at Lila (joined 2025).
- Known for: NEAT, **Novelty Search**, "Why Greatness Cannot Be Planned."
- Core insight: **"stepping stones" to great discoveries are often not recognized as such by objective-based search** — intermediate steps appear to make no progress toward the goal.
- At Lila: building open-endedness team combining pre-training, RLHF, distillation, mechanistic interpretability, quality diversity.

### AI Science Factory (AISF)

- Fully autonomous labs running the **entire scientific method 24/7** — hypotheses, experiments, learning.
- Standardized template: general-purpose robots, automated liquid handling, bioreactors, integrated analytical tools.
- **Demonstrated result**: Discovered novel non-platinum-group metal catalysts for green hydrogen in **4 months** (experts estimated a decade).
- Human/partner firm uploads high-level research goal; system autonomously executes.

---

## 5. The "Stepping Stone" Risk Concept

### Theoretical Foundation (Kenneth Stanley)

- **Novelty search** (Lehman & Stanley, 2011): Fitness functions typically do not reward intermediate stepping stones. Exploring novel behavioral space often discovers better solutions than direct optimization.
- **Objective paradox**: The most ambitious goals are often unreachable by direct optimization — only through serendipitous discovery of stepping stones.
- **Open-endedness** (2019): "Unlike algorithms in ML that learn to solve problems we ask them to solve, open-ended algorithms could produce surprises beyond our imagination."

### Safety Implication

If individually benign discoveries serve as stepping stones in unforeseeable ways, then **an open-ended system might converge on dangerous capabilities through a sequence of individually harmless steps**. Extremely difficult to monitor or prevent.

### Formalization as Safety Risk

| Source | Year | Contribution |
|--------|------|-------------|
| **Darwin Godel Machine** | 2025 | "Allowing unfettered general-purpose agents to iteratively refine themselves heightens safety concerns." Simple safe testbeds necessary. |
| **Wang et al. (Esvelt)** | 2025 | IAB "couples model-guided sequence design with robotic synthesis in iterative cycles." Active learning "dramatically reduces experiments to identify high-risk variants." |
| **PLOS CompBio** | 2025 | Defines capability levels where stepping-stone risk becomes acute (Levels 3-5). |

### Application to Protein Design

- **Iterative convergence**: AI-driven directed evolution involves iterative mutation+selection cycles. Each variant may be benign, but optimization trajectory can converge on dangerous function.
- **Sequence-function decoupling**: AI-generated proteins may be functionally equivalent to toxins while sharing little sequence similarity. Each step passes screening; endpoint is dangerous.
- **Reduced iteration requirements**: Advances are reducing iterations needed — stepping-stone path getting shorter.

### Open Research Problem

Can safety systems evaluate not just individual model outputs but **trajectories of iterative design** to detect convergence toward dangerous capabilities?
