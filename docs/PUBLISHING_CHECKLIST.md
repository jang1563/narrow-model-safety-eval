# Publishing Checklist

Use this checklist before promoting a GitHub release, Hugging Face dataset
update, paper supplement, or public demo.

## Required Gates

- Run `pytest tests/ -v --tb=short`.
- Run `ruff check src/ tests/`.
- Confirm `git status --short` contains only intentional release files.
- Confirm `docs/RELEASE_SURFACE.md`, `SAFETY.md`, `DISCLAIMER.md`, and
  `SECURITY.md` still describe the actual release surface.
- Confirm `CITATION.cff`, `pyproject.toml`, `README.md`, and
  `huggingface/README.md` use the same version, title, author, and repository
  URLs.
- Confirm all Markdown links pass `tests/test_release_integrity.py`.

## GitHub Readiness

- CI badge points to `.github/workflows/ci.yml` and the workflow is green.
- Issue templates route public bug reports, documentation fixes, and
  responsible-use concerns to the right channel.
- The PR template includes tests, linting, citation/annotation checks, and
  release-surface checks.
- `CONTRIBUTING.md` explains how to add proteins without weakening the
  responsible-use boundary.
- `SECURITY.md` tells reporters not to post operational sequence detail in
  public issues.

## Hugging Face Dataset Readiness

- Dataset card lives at `huggingface/README.md` and has valid YAML front matter.
- **Sync the live card after any edit.** Editing `huggingface/README.md` in
  this repo does **not** update the live dataset card — the Hub does not pull
  from GitHub. Push it explicitly:
  ```python
  from huggingface_hub import upload_file
  upload_file(path_or_fileobj="huggingface/README.md", path_in_repo="README.md",
              repo_id="jang1563/narrow-model-safety-eval", repo_type="dataset",
              commit_message="Update dataset card")
  ```
  Before pushing, diff the live card against this repo's pre-edit version to
  confirm they were in sync (no independent Hub-side drift to clobber):
  `huggingface_hub.hf_hub_download(..., force_download=True)` then `diff`.
  After pushing, re-download with `force_download=True` and confirm the new
  sections are present.
- The card states that no model-generated dangerous sequences, synthesis routes,
  expression vectors, or design protocols are included.
- Example downloads use `repo_type="dataset"` and paths that exist in the
  published dataset.
- Public files are limited to reference inputs, annotations, aggregate result
  JSON, figures, documentation, and reproducible pipeline code.
- Generated design FASTA/PDB outputs, embeddings, model weights, local BLAST
  databases, caches, and temporary HPC artifacts are absent.

## Safety Review

- No generated amino-acid or nucleotide sequence payloads appear in
  `results/*.json`.
- No codon-optimized sequence payloads are written to release artifacts.
- Any newly added evaluated protein has DOI-backed functional-site annotations.
- Any Tier 1-2 realizability annotation is manually reviewed before publication.
- New analyses preserve the measurement-not-objective framing in `SAFETY.md`.

## Release Note Skeleton

```markdown
## Version X.Y.Z

### Highlights
- TBD

### Scientific changes
- TBD

### Safety and release-surface changes
- TBD

### Verification
- `pytest tests/ -v --tb=short`
- `ruff check src/ tests/`
```
