"""Release-surface and metadata integrity checks."""

import importlib.util
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def load_script_module(path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_no_withheld_artifacts_are_tracked():
    forbidden_patterns = [
        "results/proteinmpnn_output/**",
        "results/proteinmpnn_temp_sweep/**",
        "results/esmfold_structures/**",
        "*.npy",
        "*.pt",
        "*.bin",
        "*.safetensors",
        "models/**",
    ]
    result = subprocess.run(
        ["git", "ls-files", *forbidden_patterns],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == ""


def test_citation_metadata_matches_project_metadata():
    citation = read("CITATION.cff")
    pyproject = read("pyproject.toml")
    license_text = read("LICENSE")

    assert 'version: "2.0.0"' in citation
    assert 'license: "CC-BY-4.0"' in citation
    assert 'version = "2.0.0"' in pyproject
    assert 'name = "JangKeun Kim"' in pyproject
    assert "Creative Commons Attribution 4.0" in license_text
    assert "MIT" not in citation


def test_readme_and_dataset_card_citations_match():
    readme = read("README.md")
    dataset_card = read("huggingface/README.md")

    expected = [
        "@misc{kim2026narrowmodelsafety",
        "author  = {Kim, JangKeun}",
        "year    = {2026}",
        "note    = {Version 2.0.0}",
    ]
    for needle in expected:
        assert needle in readme
        assert needle in dataset_card

    assert "author  = {Jang, Jaewon}" not in dataset_card


def test_huggingface_card_uses_correct_bonta_accession_and_paths():
    card = read("huggingface/README.md")

    assert "| P0DPI1 | Botulinum neurotoxin A light chain | 3BTA |" in card
    assert "| P10844 | Botulinum neurotoxin A light chain |" not in card
    assert 'filename="results/fsi_results.json"' in card
    assert 'print(sites["P0DPI1"]["functional_sites"]["catalytic_residues"])' in card
    assert "2.87" not in card


def test_release_narrative_has_no_stale_headline_numbers():
    narrative_paths = [
        "README.md",
        "huggingface/README.md",
        "src/08_evaluation_report.py",
        "src/09_negative_controls.py",
        "src/10_fsi_temperature_sensitivity.py",
        "results/evaluation_report.txt",
    ]
    stale_patterns = [
        "BoNT-A FSI=2.87",
        "BoNT-A FSI=3.07",
        "3BTA FSI=3.07",
        "FSI=0.70",
        "4/5 proteins",
        "mean ratio=0.928",
        "author  = {Jang, Jaewon}",
    ]
    for path in narrative_paths:
        text = read(path)
        for stale in stale_patterns:
            assert stale not in text, f"{path} contains stale narrative: {stale}"


def test_release_surface_docs_are_cross_linked():
    readme = read("README.md")
    dataset_card = read("huggingface/README.md")
    safety = read("SAFETY.md")
    contributing = read("CONTRIBUTING.md")
    release_surface = read("docs/RELEASE_SURFACE.md")

    assert "docs/RELEASE_SURFACE.md" in readme
    assert "docs/PUBLISHING_CHECKLIST.md" in readme
    assert "docs/PUBLISHING_CHECKLIST.md" in dataset_card
    assert "docs/RELEASE_SURFACE.md" in safety
    assert "docs/RELEASE_SURFACE.md" in contributing
    assert re.search(r"results/proteinmpnn_output/", release_surface)
    assert re.search(r"GitHub or Hugging Face\s+release", release_surface)


def test_tracked_json_files_parse():
    result = subprocess.run(
        ["git", "ls-files", "data/**/*.json", "results/*.json"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    paths = [line for line in result.stdout.splitlines() if line]
    assert paths
    for path in paths:
        json.loads(read(path))


def test_result_json_files_do_not_publish_generated_sequences():
    forbidden_keys = {
        "sequence",
        "sequences",
        "designed_sequence",
        "designed_sequences",
        "aa_sequence",
        "nt_sequence",
        "dna_sequence",
        "codon_optimized_sequence",
    }
    result = subprocess.run(
        ["git", "ls-files", "results/*.json"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    paths = [line for line in result.stdout.splitlines() if line]
    assert paths

    def walk(value, path):
        if isinstance(value, dict):
            for key, nested in value.items():
                assert key not in forbidden_keys, f"{path} publishes generated sequence key: {key}"
                walk(nested, path)
        elif isinstance(value, list):
            for nested in value:
                walk(nested, path)

    for path in paths:
        walk(json.loads(read(path)), path)


def test_ser_skip_blastn_omits_nucleotide_screening(monkeypatch, tmp_path):
    ser = load_script_module("src/16_screening_evasion.py", "screening_evasion")
    monkeypatch.setattr(ser, "blastp_max_identity", lambda aa, db, tmp: 0.40)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("BLASTn should not run when skip_blastn=True")

    monkeypatch.setattr(ser, "blastn_max_identity", fail_if_called)

    result = ser.compute_ser(
        sequences=["ACDE", "WXYZ"],
        prot_db_path="protein_db",
        nt_db_path=None,
        max_seqs=2,
        tmp_dir=str(tmp_path),
        skip_blastn=True,
    )

    assert result["ser_p"] == 1.0
    assert result["ser_n"] is None
    assert result["mean_prot_identity"] == 0.40
    assert "mean_nt_identity" not in result
    assert "nt_identity_per_seq" not in result


def test_tracked_markdown_local_links_exist():
    result = subprocess.run(
        ["git", "ls-files", "*.md"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    paths = [line for line in result.stdout.splitlines() if line]
    link_pattern = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
    missing = []

    for path in paths:
        text = read(path)
        for raw_target in link_pattern.findall(text):
            target = raw_target.split()[0].strip("<>")
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path_part = target.split("#", 1)[0]
            if not path_part:
                continue
            candidate = (ROOT / path).parent / path_part
            if not candidate.exists():
                missing.append((path, raw_target))

    assert missing == []
