# The coverage ledger — what each frozen exclusion costs and what it buys

**Written 2026-10-01**, from artifacts only. 🔑 **Every number here was already recorded on the day the
rule that produced it was frozen. What was missing was the column that adds them up.**

---

## 1. Why this document exists

Asked why the matched population is small and whether raising it had been attempted. 🔴 **The premise of
the question was wrong in a way worth recording: it had been raised, repeatedly, and each raise was
preregistered and measured.**

- Study H reached VFDB through the **benign pool's build query**, which *excludes virulence-annotated
  proteins* — so its population was VFDB records that had survived a filter designed to remove exactly
  what the study was about.
- Study K replaced that bridge with UniProt's ID-mapping service, **seven times the size**, and the
  verdict moved **PARTIAL → SUPPORTED**. Its K-3 test explains why: the **228** Virulence-keyword members
  the pool query had excluded are flagged **19.2** and **15.2** points above the rest.
- Study L raised localization annotation coverage from **45% to 66%**, and reported the 50% ceiling it
  breached.

🔒 **So coverage was treated as a first-class quantity three times.** ⚠️ **What was never done is add up
what the remaining rules cost**, and one of them turns out to be most of the loss.

---

## 2. The VFDB side

| rule | before → after | kept | what it buys |
|---|---|---:|---|
| admitted to the negative set | 4,218 → 4,218 | 100% | length in [50, 1022], screened against the panel |
| **maps to a reviewed Swiss-Prot entry** | 4,218 → **966** | **22.9%** | curated keywords, localization, length |
| non-circular (not a class-axis positive) | 966 → 788 | 81.6% | no training-data leakage |
| length identical to the mapped entry | 788 → **778** | 98.7% | no mapping error |

**Overall: 4,218 → 778 = 18.4% of the population survives.**

### 🔴 The decomposition of the one expensive rule

| | rows | share |
|---|---:|---:|
| reviewed Swiss-Prot hit | 966 | 22.9% |
| ⭐ **unreviewed-only hit — excluded by the rule** | **2,295** | **54.4%** |
| no UniProt hit at all — unavoidable | 957 | 22.7% |

🔑 **The reviewed-only rule alone costs 2,295 rows.** Spending it would take the mapped population from
**966 to 3,261 — 22.9% to 77.3%**. ⚠️ **`n_unreviewed_only` was written into
`results/vfdb_uniprot_map.json` on 2026-09-28, the day the rule was frozen, and no study has spent it.**

🔒 **The other two exclusions are cheap and should stay**: non-circularity costs 178 rows and buys
freedom from training-data leakage — the defect entry 61 found — and length-identity costs **10** and
buys freedom from mapping error.

---

## 3. The benign side, which is the other half of `C/A`

| | rows |
|---|---:|
| pool test partition | 6,758 |
| eligible: Bacteria/Archaea, not in VFDB | 6,247 |
| study L's extracellular stratum | 657 |
| ⭐ **the matched benign cell `C/A` actually divides by** | **171** |

🔑 **`C/A` = 264 / 171.** Its honest interval — **[1.904, 3.252]** and **[1.931, 3.938]**, study AC — is
as much the 171 as the 264. ⚠️ **Spending the reviewed-only rule grows the numerator and not the
denominator**, so it narrows the interval by roughly √3.4 on one side only.

🔒 **The 171 is structural**: the benign cell is restricted to Bacteria whose organism appears in VFDB's
species list, which *is* the provenance match the contrast exists to hold fixed. **The only way to grow it
is another pool**, and one exists — the independent 11,301-protein draw, which would roughly double it.

---

## 4. Hazard labels, and why a hazard axis cannot come from UniProt here

| | rows | share of the 966 |
|---|---:|---:|
| **Virulence** keyword | 300 | 31.1% |
| ⭐ **Toxin** keyword | **7** | **0.7%** |

🔴 **Seven proteins.** Two frozen choices compose to produce that: `src/74` built the negative set from
**non-Exotoxin** records, and the reviewed-only rule then kept the best-curated fifth of what remained.
⚠️ **Both were right individually and together they make a UniProt-keyword hazard axis impossible on this
population.** 🔒 A hazard axis has to come from somewhere else — the class-axis set's **102** Exotoxin
representatives, the 80-protein panel, or an external regulatory list.

---

## 5. What this says about the project's design

🔑 **Five exclusions, each individually correct, compose to 18.4%.** Reviewed-only buys curation,
non-Exotoxin avoids trivial positives, length-identity avoids mapping error, non-circularity avoids
leakage, and the keyword strata buy comparability with study K. ⚠️ **The project bought
precision-of-definition with population five times and priced it once** — in this session, when the honest
intervals were computed and came out 8 to 12 times wider than the seed ones.

🔒 **The actionable conclusion is not "another study". It is that the ledger now exists**, so the next
expansion can be chosen by what it costs rather than by what is convenient: **the reviewed-only rule is
54.4% of the loss and buys the least, because the fields it protects — length and organism — are present
on unreviewed entries too.**
