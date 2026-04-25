#!/usr/bin/env python3
"""
18_realizability_automation.py — Automated Physical Realizability Annotation (v2 Pillar 5B)

Uses Claude API to annotate new proteins with Physical Realizability scores,
validated against the existing manual annotations in physical_realizability.json.

Protocol:
  1. Load existing manual annotations as ground truth
  2. For each new protein: fetch UniProt JSON + PDB remarks + PubMed abstracts
  3. Prompt Claude with structured schema, parse output as JSON
  4. Validate against Pydantic schema; flag Tier 1-2 entries for human review
  5. Compute Spearman correlation against manual scores for benchmarking

Usage:
    python src/18_realizability_automation.py --proteins Q9HXZ2 P26471 P0A030
    python src/18_realizability_automation.py --all_new
    python src/18_realizability_automation.py --validate_only   # score vs manual only

Output:
    results/realizability_auto_annotations.json
    results/realizability_auto_vs_manual.json  (correlation benchmark)

Requires:
    pip install anthropic pydantic requests scipy
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

import requests
from pydantic import BaseModel, Field, field_validator
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
ANNOTATIONS_DIR = DATA_DIR / "annotations"

PHYSICAL_REAL_PATH = ANNOTATIONS_DIR / "physical_realizability.json"
FUNCTIONAL_SITES_PATH = ANNOTATIONS_DIR / "functional_sites.json"
AUTO_OUTPUT_PATH = RESULTS_DIR / "realizability_auto_annotations.json"
CORRELATION_OUTPUT_PATH = RESULTS_DIR / "realizability_auto_vs_manual.json"

# UniProt + PubMed APIs
UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb"
PUBMED_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


# ============================================================================
# Pydantic schema for structured LLM output
# ============================================================================

class RealizabilityAnnotation(BaseModel):
    name: str = Field(..., description="Protein common name")
    synthesis_feasibility: int = Field(..., ge=1, le=5)
    synthesis_notes: str = Field(..., max_length=500)
    folding_complexity: int = Field(..., ge=1, le=5)
    folding_notes: str = Field(..., max_length=500)
    assembly_requirements: int = Field(..., ge=1, le=5)
    assembly_notes: str = Field(..., max_length=500)
    activity_assay_barrier: int = Field(..., ge=1, le=5)
    assay_notes: str = Field(..., max_length=500)
    regulatory_barrier: int = Field(..., ge=1, le=5)
    regulatory_notes: str = Field(..., max_length=500)
    overall_realizability_tier: int = Field(..., ge=1, le=5)
    tier_rationale: str = Field(..., max_length=600)
    key_bottleneck: str = Field(..., max_length=200)
    human_review_required: bool = Field(
        ..., description="True if Tier 1-2 (most accessible) — needs human expert check"
    )
    confidence: str = Field(..., pattern="^(high|medium|low)$")

    @field_validator("overall_realizability_tier")
    @classmethod
    def tier_must_reflect_barriers(cls, v, info):
        data = info.data
        # Tier 1 requires ALL individual scores <= 2
        if v == 1:
            scores = [
                data.get("synthesis_feasibility", 3),
                data.get("folding_complexity", 3),
                data.get("assembly_requirements", 3),
                data.get("activity_assay_barrier", 3),
                data.get("regulatory_barrier", 3),
            ]
            if any(s > 2 for s in scores):
                raise ValueError(
                    "Tier 1 requires all individual scores <= 2"
                )
        return v


# ============================================================================
# Data fetching helpers
# ============================================================================

def fetch_uniprot_json(accession: str) -> dict:
    url = f"{UNIPROT_BASE}/{accession}"
    for attempt in range(3):
        try:
            resp = requests.get(url, params={"format": "json"}, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt < 2:
                time.sleep(5)
            else:
                print(f"  WARNING: UniProt fetch failed for {accession}: {e}")
                return {}


def extract_uniprot_summary(data: dict) -> str:
    """Extract key fields from UniProt JSON entry for LLM context."""
    if not data:
        return "UniProt data unavailable."

    lines = []
    # Protein name
    try:
        pname = data["proteinDescription"]["recommendedName"]["fullName"]["value"]
        lines.append(f"Protein: {pname}")
    except (KeyError, TypeError):
        pass

    # Organism
    try:
        org = data["organism"]["scientificName"]
        lines.append(f"Organism: {org}")
    except (KeyError, TypeError):
        pass

    # Function
    try:
        for comment in data.get("comments", []):
            if comment.get("commentType") == "FUNCTION":
                for text_block in comment.get("texts", []):
                    lines.append(f"Function: {text_block['value'][:500]}")
                    break
                break
    except (KeyError, TypeError):
        pass

    # Keywords
    try:
        kws = [kw["name"] for kw in data.get("keywords", [])[:10]]
        lines.append(f"Keywords: {', '.join(kws)}")
    except (KeyError, TypeError):
        pass

    # Subcellular location
    try:
        for comment in data.get("comments", []):
            if comment.get("commentType") == "SUBCELLULAR LOCATION":
                locs = []
                for loc_block in comment.get("subcellularLocations", []):
                    loc_val = loc_block.get("location", {}).get("value", "")
                    if loc_val:
                        locs.append(loc_val)
                if locs:
                    lines.append(f"Subcellular location: {', '.join(locs)}")
                break
    except (KeyError, TypeError):
        pass

    # Length
    try:
        length = data["sequence"]["length"]
        lines.append(f"Sequence length: {length} aa")
    except (KeyError, TypeError):
        pass

    return "\n".join(lines)


def fetch_pubmed_abstracts(query: str, n: int = 2) -> str:
    """Fetch top-N PubMed abstract snippets for context."""
    try:
        search_url = f"{PUBMED_EUTILS}/esearch.fcgi"
        params = {"db": "pubmed", "term": query, "retmax": n, "retmode": "json"}
        resp = requests.get(search_url, params=params, timeout=20)
        resp.raise_for_status()
        ids = resp.json().get("esearchresult", {}).get("idlist", [])[:n]
        if not ids:
            return ""
        time.sleep(0.5)

        fetch_url = f"{PUBMED_EUTILS}/efetch.fcgi"
        fetch_params = {"db": "pubmed", "id": ",".join(ids), "rettype": "abstract", "retmode": "text"}
        fetch_resp = requests.get(fetch_url, params=fetch_params, timeout=30)
        fetch_resp.raise_for_status()
        # Truncate to first 1000 chars of combined abstracts
        return fetch_resp.text[:1000]
    except Exception as e:
        print(f"  WARNING: PubMed fetch failed: {e}")
        return ""


# ============================================================================
# LLM annotation
# ============================================================================

SYSTEM_PROMPT = """You are a biosafety expert annotating proteins for physical realizability —
the practical ease or difficulty of converting a computationally-designed sequence into a functional protein.

You must respond with ONLY a valid JSON object matching the schema provided. No markdown, no explanation.

Scoring scale for each dimension (1=trivial, 5=extreme barrier):
- synthesis_feasibility: 1=standard gene synthesis; 5=specialized synthesis, host-toxic gene
- folding_complexity: 1=folds in E. coli cytoplasm autonomously; 5=requires chaperones, disulfide bonds, mammalian expression, co-factors
- assembly_requirements: 1=active as monomer; 5=multi-subunit complex requiring co-expression
- activity_assay_barrier: 1=simple colorimetric/gel assay; 5=live animal or complex cell-based assay required
- regulatory_barrier: 1=no restrictions; 5=Select Agent Tier 1, BSL-3+, IACUC/IBC required

overall_realizability_tier: composite 1-5 (higher = harder to realize physically)
human_review_required: true if Tier 1 or 2 (most accessible = most safety-relevant)
confidence: "high" if literature is clear, "medium" if some uncertainty, "low" if speculative"""

ANNOTATION_PROMPT_TEMPLATE = """Annotate the physical realizability of this protein.

## Protein: {accession}

{uniprot_summary}

## Relevant Literature Context:
{pubmed_context}

## Output JSON schema:
{{
  "name": "<protein common name>",
  "synthesis_feasibility": <1-5>,
  "synthesis_notes": "<1-2 sentences>",
  "folding_complexity": <1-5>,
  "folding_notes": "<1-2 sentences>",
  "assembly_requirements": <1-5>,
  "assembly_notes": "<1-2 sentences>",
  "activity_assay_barrier": <1-5>,
  "assay_notes": "<1-2 sentences>",
  "regulatory_barrier": <1-5>,
  "regulatory_notes": "<1-2 sentences>",
  "overall_realizability_tier": <1-5>,
  "tier_rationale": "<2-3 sentences>",
  "key_bottleneck": "<single phrase identifying the main barrier>",
  "human_review_required": <true if Tier 1-2>,
  "confidence": "<high|medium|low>"
}}

Respond with ONLY the JSON object."""


def annotate_protein_llm(
    accession: str,
    uniprot_data: dict,
    pubmed_context: str,
    model: str = "claude-opus-4-7",
    api_key: Optional[str] = None,
) -> Optional[RealizabilityAnnotation]:
    """Call Claude API to annotate one protein, parse and validate output."""
    try:
        import anthropic
    except ImportError:
        print("ERROR: anthropic package not installed. Run: pip install anthropic")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    uniprot_summary = extract_uniprot_summary(uniprot_data)
    prompt = ANNOTATION_PROMPT_TEMPLATE.format(
        accession=accession,
        uniprot_summary=uniprot_summary,
        pubmed_context=pubmed_context if pubmed_context else "No PubMed context available.",
    )

    for attempt in range(3):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()

            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = re.sub(r"^```[^\n]*\n", "", raw)
                raw = re.sub(r"\n```$", "", raw)

            parsed = json.loads(raw)
            annotation = RealizabilityAnnotation(**parsed)
            return annotation

        except json.JSONDecodeError as e:
            print(f"  Attempt {attempt+1}: JSON parse error: {e}")
            if attempt < 2:
                time.sleep(3)
        except Exception as e:
            print(f"  Attempt {attempt+1}: Error: {e}")
            if attempt < 2:
                time.sleep(3)

    return None


# ============================================================================
# Correlation benchmark vs. manual annotations
# ============================================================================

def compute_correlation_vs_manual(
    auto_results: dict,
    manual_data: dict,
    dimensions: list,
) -> dict:
    """Spearman correlation between automated and manual scores per dimension."""
    shared_accessions = [
        k for k in auto_results
        if k in manual_data and not k.startswith("_")
    ]

    if len(shared_accessions) < 3:
        print(f"  Only {len(shared_accessions)} shared proteins — insufficient for correlation")
        return {}

    correlation_results = {}
    for dim in dimensions:
        auto_scores = [auto_results[acc].get(dim) for acc in shared_accessions
                       if auto_results[acc].get(dim) is not None]
        manual_scores = [manual_data[acc].get(dim) for acc in shared_accessions
                         if manual_data.get(acc, {}).get(dim) is not None]

        if len(auto_scores) < 3 or len(auto_scores) != len(manual_scores):
            continue

        r, p = stats.spearmanr(auto_scores, manual_scores)
        correlation_results[dim] = {
            "spearman_r": round(float(r), 3),
            "p_value": round(float(p), 4),
            "n": len(auto_scores),
            "interpretation": (
                "strong agreement" if abs(r) >= 0.7 else
                "moderate agreement" if abs(r) >= 0.5 else
                "weak agreement"
            ),
        }

    # Overall tier correlation
    auto_tiers = [auto_results[acc].get("overall_realizability_tier")
                  for acc in shared_accessions
                  if auto_results[acc].get("overall_realizability_tier")]
    manual_tiers = [manual_data[acc].get("overall_realizability_tier")
                    for acc in shared_accessions
                    if manual_data.get(acc, {}).get("overall_realizability_tier")]

    if len(auto_tiers) >= 3 and len(auto_tiers) == len(manual_tiers):
        r, p = stats.spearmanr(auto_tiers, manual_tiers)
        correlation_results["overall_realizability_tier"] = {
            "spearman_r": round(float(r), 3),
            "p_value": round(float(p), 4),
            "n": len(auto_tiers),
            "interpretation": (
                "strong agreement" if abs(r) >= 0.7 else
                "moderate agreement" if abs(r) >= 0.5 else
                "weak agreement"
            ),
        }

    return {
        "shared_proteins": shared_accessions,
        "n_shared": len(shared_accessions),
        "dimension_correlations": correlation_results,
    }


# ============================================================================
# Main
# ============================================================================

SCORE_DIMENSIONS = [
    "synthesis_feasibility",
    "folding_complexity",
    "assembly_requirements",
    "activity_assay_barrier",
    "regulatory_barrier",
]

# Proteins to annotate automatically in v2 panel expansion
V2_NEW_PROTEINS = [
    "Q9HXZ2",  # ExoU
    "P26471",  # ExoS
    "P0A030",  # YopH
    "P00648",  # Barnase
    "P02978",  # Colicin E2
    "Q99ZW2",  # SpCas9
    "P00588",  # Diphtheria toxin
    "P55981",  # VacA
]

# PubMed search terms per protein for context retrieval
PUBMED_QUERIES = {
    "Q9HXZ2": "ExoU Pseudomonas phospholipase type III effector",
    "P26471": "ExoS Pseudomonas ADP-ribosyltransferase type III secretion",
    "P0A030": "YopH Yersinia protein tyrosine phosphatase virulence",
    "P00648": "barnase Bacillus ribonuclease barstar expression",
    "P02978": "colicin E2 DNase HNH endonuclease mechanism",
    "Q99ZW2": "Cas9 SpCas9 CRISPR nuclease mechanism crystal structure",
    "P00588": "diphtheria toxin ADP-ribosylation EF-2 mechanism",
    "P55981": "VacA Helicobacter pylori vacuolating cytotoxin pore-forming channel",
}


def main():
    parser = argparse.ArgumentParser(description="Automated Physical Realizability Annotation")
    parser.add_argument("--proteins", nargs="+", help="UniProt accessions to annotate")
    parser.add_argument("--all_new", action="store_true", help="Annotate all v2 new proteins")
    parser.add_argument("--validate_only", action="store_true",
                        help="Only run correlation benchmark (no new annotations)")
    parser.add_argument("--model", default="claude-opus-4-7",
                        help="Claude model ID (default: claude-opus-4-7)")
    parser.add_argument("--skip_llm", action="store_true",
                        help="Skip LLM calls; only fetch data and print prompts")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing manual annotations
    with open(PHYSICAL_REAL_PATH) as f:
        manual_data = json.load(f)
    manual_proteins = {k: v for k, v in manual_data.items() if not k.startswith("_")}
    print(f"Loaded {len(manual_proteins)} manually annotated proteins")

    if args.validate_only:
        # Load existing auto annotations if present
        if AUTO_OUTPUT_PATH.exists():
            with open(AUTO_OUTPUT_PATH) as f:
                auto_results = json.load(f)
            auto_results = {k: v for k, v in auto_results.items() if not k.startswith("_")}
            print(f"Loaded {len(auto_results)} auto-annotated proteins for correlation")
        else:
            print("No auto annotations found. Run without --validate_only first.")
            sys.exit(1)

        corr = compute_correlation_vs_manual(auto_results, manual_proteins, SCORE_DIMENSIONS)
        with open(CORRELATION_OUTPUT_PATH, "w") as f:
            json.dump(corr, f, indent=2)
        print(f"\n=== Correlation vs. Manual ===")
        for dim, res in corr.get("dimension_correlations", {}).items():
            print(f"  {dim:<40} r={res['spearman_r']:+.3f} p={res['p_value']:.4f}  {res['interpretation']}")
        return

    # Determine proteins to annotate
    if args.all_new:
        targets = V2_NEW_PROTEINS
    elif args.proteins:
        targets = args.proteins
    else:
        print("Specify --proteins <accessions> or --all_new")
        parser.print_help()
        sys.exit(1)

    print(f"\nAnnotating {len(targets)} proteins: {', '.join(targets)}")

    # Load existing auto annotations if present (to avoid re-running)
    if AUTO_OUTPUT_PATH.exists():
        with open(AUTO_OUTPUT_PATH) as f:
            auto_results = json.load(f)
    else:
        auto_results = {"_schema_version": "2.0", "_source": "automated_llm_annotation"}

    review_required = []

    for accession in targets:
        if accession in auto_results:
            print(f"\n{accession}: already annotated, skipping")
            continue

        print(f"\n--- {accession} ---")

        # Fetch UniProt data
        print("  Fetching UniProt...")
        uniprot_data = fetch_uniprot_json(accession)
        time.sleep(0.5)

        # Fetch PubMed context
        query = PUBMED_QUERIES.get(accession, f"{accession} protein mechanism expression")
        print(f"  Fetching PubMed: '{query[:60]}'...")
        pubmed_text = fetch_pubmed_abstracts(query, n=2)
        time.sleep(0.5)

        if args.skip_llm:
            print("  [--skip_llm] Would call LLM with this context:")
            print("  " + extract_uniprot_summary(uniprot_data)[:200])
            continue

        print("  Calling Claude API...")
        annotation = annotate_protein_llm(
            accession=accession,
            uniprot_data=uniprot_data,
            pubmed_context=pubmed_text,
            model=args.model,
        )

        if annotation is None:
            print(f"  ERROR: annotation failed for {accession}")
            auto_results[accession] = {"error": "annotation_failed"}
            continue

        result_dict = annotation.model_dump()
        auto_results[accession] = result_dict

        tier = annotation.overall_realizability_tier
        print(f"  Tier={tier} | {annotation.key_bottleneck}")

        if annotation.human_review_required:
            review_required.append(accession)
            print(f"  *** HUMAN REVIEW REQUIRED (Tier {tier}) ***")

        # Save after each protein
        with open(AUTO_OUTPUT_PATH, "w") as f:
            json.dump(auto_results, f, indent=2)

    print(f"\n=== Annotation complete ===")
    completed = [a for a in targets if a in auto_results and "error" not in auto_results.get(a, {})]
    print(f"Annotated: {len(completed)}/{len(targets)}")
    if review_required:
        print(f"HUMAN REVIEW REQUIRED: {', '.join(review_required)}")

    # Compute correlation with manual annotations on overlapping proteins
    corr = compute_correlation_vs_manual(
        {k: v for k, v in auto_results.items() if not k.startswith("_")},
        manual_proteins,
        SCORE_DIMENSIONS,
    )
    if corr:
        with open(CORRELATION_OUTPUT_PATH, "w") as f:
            json.dump(corr, f, indent=2)
        print(f"\n=== Correlation vs. Manual (n={corr['n_shared']} shared) ===")
        for dim, res in corr.get("dimension_correlations", {}).items():
            print(f"  {dim:<40} r={res['spearman_r']:+.3f} p={res['p_value']:.4f}")

    print(f"\nResults: {AUTO_OUTPUT_PATH}")
    if CORRELATION_OUTPUT_PATH.exists():
        print(f"Correlation: {CORRELATION_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
