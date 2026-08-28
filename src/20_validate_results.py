#!/usr/bin/env python3
"""
20_validate_results.py - release-facing result validation.

This script checks cross-artifact consistency that normal unit tests do not
always catch: JSON parseability, withheld sequence payload keys, stale narrative
phrases, FSPE report drift, and ESM-3/SaProt MDRP model-label consistency.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

FORBIDDEN_SEQUENCE_KEYS = {
    "sequence",
    "sequences",
    "designed_sequence",
    "designed_sequences",
    "aa_sequence",
    "nt_sequence",
    "dna_sequence",
    "codon_optimized_sequence",
}

NARRATIVE_PATHS = [
    ROOT / "README.md",
    ROOT / "huggingface" / "README.md",
    ROOT / "docs" / "EVALUATION_REPORT.md",
    ROOT / "src" / "08_evaluation_report.py",
    ROOT / "dashboard" / "app.py",
    RESULTS_DIR / "evaluation_report.txt",
]

STALE_PATTERNS = [
    "BoNT-A FSI=2.87",
    "BoNT-A FSI=3.07",
    "3BTA FSI=3.07",
    "FSI=0.70",
    "4/5 proteins",
    "5/7 proteins",
    "mean ratio=0.928",
    "100% catalytic residue recovery",
    "TM-score > 0.5",
    "job 2808653 pending",
    "7-Dimensional MDRP Radar Chart",
    "Biohub / Presentation Framing",
    "Biohub / Interview Framing",
    "BIOHUB_RESEARCH_BRIEF",
    "Do not say",
    "interview brief",
    "research-talk outline",
    "Talk outline",
    "Slide-ready takeaway",
]


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _load_json(path: Path, errors: list[str]) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - validation should report all parse failures.
        errors.append(f"{_rel(path)} is not valid JSON: {exc}")
        return None


def _walk_forbidden_keys(value: Any, source_path: Path, errors: list[str]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in FORBIDDEN_SEQUENCE_KEYS:
                errors.append(f"{_rel(source_path)} publishes generated sequence key: {key}")
            _walk_forbidden_keys(nested, source_path, errors)
    elif isinstance(value, list):
        for nested in value:
            _walk_forbidden_keys(nested, source_path, errors)


def validate_json_files(errors: list[str]) -> dict[Path, Any]:
    loaded = {}
    paths = sorted(DATA_DIR.glob("**/*.json")) + sorted(RESULTS_DIR.glob("*.json"))
    for path in paths:
        loaded[path] = _load_json(path, errors)
    return loaded


def validate_release_surface(loaded: dict[Path, Any], errors: list[str]) -> None:
    for path, data in loaded.items():
        if path.parent == RESULTS_DIR and data is not None:
            _walk_forbidden_keys(data, path, errors)


def validate_narrative_staleness(errors: list[str]) -> None:
    for path in NARRATIVE_PATHS:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in STALE_PATTERNS:
            if pattern in text:
                errors.append(f"{_rel(path)} contains stale narrative: {pattern}")


def validate_fspe_report_consistency(loaded: dict[Path, Any], errors: list[str]) -> None:
    fspe = loaded.get(RESULTS_DIR / "fspe_results.json")
    report_path = RESULTS_DIR / "evaluation_report.txt"
    if not isinstance(fspe, dict) or not report_path.exists():
        return

    rows = fspe.get("per_protein", [])
    ratios = [row["fspe_ratio"] for row in rows if row.get("fspe_ratio") is not None]
    if not ratios:
        return

    expected = (
        f"FSPE is directional ({sum(r < 1.0 for r in ratios)}/{len(ratios)} proteins, "
        f"mean ratio={sum(ratios) / len(ratios):.3f})"
    )
    report = report_path.read_text(encoding="utf-8")
    if expected not in report:
        errors.append(
            "results/evaluation_report.txt FSPE headline does not match "
            f"results/fspe_results.json; expected '{expected}'"
        )


def validate_mdrp_fspe_models(loaded: dict[Path, Any], errors: list[str]) -> None:
    source = loaded.get(RESULTS_DIR / "esm3_fspe_results.json")
    mdrp = loaded.get(RESULTS_DIR / "mdrp_risk_table.json")
    if not isinstance(source, dict) or not isinstance(mdrp, dict):
        return

    by_model: dict[str, dict[str, float]] = {"esm3_sm_open_v1": {}, "saprot_650m_af2": {}}
    for row in source.get("results", []):
        model = row.get("model")
        ratio = row.get("fspe_ratio")
        if model in by_model and ratio is not None and ratio != 1.0:
            by_model[model][row["uniprot_id"]] = ratio

    for row in mdrp.get("proteins", []):
        uid = row.get("uniprot_id")
        if not uid:
            continue
        for key, model in [("fspe_esm3", "esm3_sm_open_v1"), ("fspe_saprot", "saprot_650m_af2")]:
            observed = row.get(key)
            expected = by_model[model].get(uid)
            if observed is None and expected is None:
                continue
            if observed is None or expected is None or abs(observed - expected) > 1e-12:
                errors.append(
                    f"mdrp_risk_table.json {key} mismatch for {uid}: "
                    f"observed={observed}, expected={expected}"
                )


def validate_ser_qc(loaded: dict[Path, Any], warnings: list[str]) -> None:
    ser = loaded.get(RESULTS_DIR / "ser_results.json")
    if not isinstance(ser, dict):
        return

    rows = ser.get("results", [])
    if rows and any("protein_search_complete" not in row for row in rows):
        warnings.append(
            "ser_results.json was generated before BLAST search-completion QC fields were added; "
            "regenerate step 16 before making SER a headline claim."
        )


def run_validations() -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    loaded = validate_json_files(errors)
    validate_release_surface(loaded, errors)
    validate_narrative_staleness(errors)
    validate_fspe_report_consistency(loaded, errors)
    validate_mdrp_fspe_models(loaded, errors)
    validate_ser_qc(loaded, warnings)
    return errors, warnings


def main() -> int:
    errors, warnings = run_validations()

    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        print("Result validation failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("Result validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
