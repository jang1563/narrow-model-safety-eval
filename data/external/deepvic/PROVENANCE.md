# DeepVIC — third-party material, provenance

**This directory holds material from someone else's published work.** It is not covered by this
repository's CC BY 4.0 licence, which applies to the authors' own content only.

Source: **DeepVIC**, *Bioinformatics Advances* **6**(1) vbag237, 2026 —
<https://academic.oup.com/bioinformaticsadvances/article/6/1/vbag237/8762933>

## What is here, and what is redistributed

| file | tracked in git | why |
|---|---|---|
| `metadata.txt` | **yes** | The curator category labels and accessions this project's DeepVIC analyses are computed against. Redistributed so the external replication in `docs/DEEPVIC_BENCHMARK_PREREGISTRATION.md` can be reproduced from the repository rather than taken on trust. |
| `DLDB_33456.faa` | no, gitignored | The sequence database. Obtain it from the source above. |
| `binary.keras`, `multiclass.keras` | no, gitignored | The published model weights. Obtain them from the source above. ⚠️ These weights could not be run in this project's environment; the comparison here uses DeepVIC's published **labels**, not its predictions. |

## If you are a DeepVIC author

`metadata.txt` is included only to make a published comparison checkable. If you would prefer it not
be redistributed, open an issue and it will be removed and replaced with a download step.

## How it is used

This project uses DeepVIC's category axis as an **external label set** — a hazard axis set by other
people's curators rather than by this project — to test whether a class-level ordering result holds
on labels it was not built on. Nothing here reimplements or evaluates DeepVIC as a classifier.
