# Contributing

Thank you for your interest in contributing. This project sits at the intersection of AI safety, computational biology, and biosecurity policy — contributions that maintain scientific rigor and responsible disclosure are especially welcome.

## Getting started

```bash
git clone https://github.com/jang1563/narrow-model-safety-eval.git
cd narrow-model-safety-eval
pip install -e ".[dev]"
```

## Running tests and linting

```bash
pytest tests/ -v          # Unit tests
ruff check src/ tests/    # Linting
ruff format src/ tests/   # Auto-format
```

All CI checks must pass before a pull request is reviewed.

## What contributions are welcome

- **New evaluation proteins**: extending the toxin or benign homolog set with properly annotated functional sites (DOI citations required)
- **Metric improvements**: statistical refinements to FSPE, FSI, or the realizability tier framework
- **Extended model coverage**: applying the FSPE/FSI pipeline to additional protein language models or inverse-folding tools
- **Documentation and reproducibility**: setup guides, tutorial notebooks, clearer docstrings
- **Bug fixes**: correctness issues in statistical tests, data loading, or figure generation

## What is out of scope

- Any contribution that discloses specific dangerous sequences, recovery protocols, or synthesis routes
- Changes that weaken the ethical framework or DISCLAIMER.md
- Dependencies that introduce supply-chain risk (prefer well-audited packages)

## Code style

- Follow ruff defaults (`line-length = 100`, `target-version = "py39"`)
- No multi-paragraph docstrings — one short line is enough unless the logic is non-obvious
- Prefer explicit over implicit; avoid premature abstractions

## Adding a new protein

1. Add FASTA sequences to `data/sequences/`
2. Add PDB structure(s) to `data/structures/`
3. Add functional site annotations to `data/annotations/functional_sites.json` with at least one DOI citation per site
4. Add physical realizability scoring to `data/annotations/physical_realizability.json`
5. Verify `python src/01_collect_data.py` succeeds end-to-end

## Pull request checklist

- [ ] Tests pass (`pytest tests/`)
- [ ] Lint passes (`ruff check src/ tests/`)
- [ ] New annotations include DOI citations
- [ ] No dangerous sequences or protocols are disclosed
- [ ] PR description explains the scientific rationale

## Responsible disclosure

If you discover a significant safety capability gap in a widely-used protein model that goes beyond the scope of this public repository, please open a [private security advisory](https://github.com/jang1563/narrow-model-safety-eval/security/advisories/new) before making anything public. We follow coordinated disclosure best practices.
