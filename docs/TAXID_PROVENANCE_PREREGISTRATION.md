# Preregistration — the provenance factor matched on names, and names change

**Frozen 2026-09-28, after a declared resolver feasibility test and before any VFDB name has been
resolved in bulk or any probe applied to a rebuilt cell.** Append-only.

---

## 0. The defect this exists to quantify

Entry 57 found that study H dropped ten bridge proteins because **UniProt had renamed the genus while
VFDB kept the old name** — the bridge alone spans **12** such species, including *Mycoplasmoides
pneumoniae* (VFDB: *Mycoplasma*), *Mycobacteroides abscessus* (VFDB: *Mycobacterium*) and *Klebsiella
aerogenes* (VFDB: *Enterobacter*).

🔴 **The same problem runs the other way in studies F and G.** Their provenance factor is the string
test at `src/85_joint_decomposition.py:125` — *does the first two words of the organism string appear
in VFDB's species list* — so a **renamed pathogen fails it and is scored `benign_species`**, putting
genuine pathogen proteins into the benign cells.

Entry 57 recorded the direction — **dilution, therefore conservative** — and recorded that it was
**not quantified**. 🔒 **This quantifies it.**

⚠️ **Conservative is not harmless.** The provenance effect is one of the two numbers in the joint
decomposition, and the ratio of ratios that says the two confounds compose is built from the very cells
this misassignment moves proteins between.

---

## 🔒 0.1 Declared feasibility test of the resolver, run before freezing

Three routes were tried and two were rejected, on eight names:

| route | verdict |
|---|---|
| `scientific:"<name>"` alone | 🔴 **rejected** — matches only the *current* name, so every renamed pathogen fails |
| free-text `"<name>"`, take the top hit | 🔴 **rejected and dangerous** — `"Escherichia coli"` returns **Escherichia phage 1**, and `"Staphylococcus aureus"` returns **Staphylococcus phage P68** |
| ⭐ **two-stage, exact-verified** | 🟢 adopted |

🔒 **The adopted resolver**: try `scientific:"<name>" AND rank:species` and accept only an exact
scientific-name match; otherwise search `"<name>" AND rank:species` and accept a hit **only if the
queried name appears exactly in its `scientificName`, `synonyms` or `otherNames`**. 🔴 **An unverified
hit is never accepted** — that is the rule that stops *E. coli* becoming a phage.

Test result: `Escherichia coli`→562, `Staphylococcus aureus`→1280, `Bacillus anthracis`→1392,
`Vibrio cholerae`→666 by scientific name; `Mycoplasma pneumoniae`→2104, `Enterobacter aerogenes`→548,
`Mycobacterium abscessus`→36809 by **synonym** — the three renamed genera, resolved.
`Salmonella typhimurium`→**UNRESOLVED**, correctly: it is a serovar of *S. enterica*, not a species.

🔒 **The pool side needs no resolver at all.** UniProtKB returns a rank-annotated lineage per
accession, so a pool protein's **current canonical species name** is read directly from its
`(species)` lineage entry in the batched request that already runs. Both sides therefore become
current canonical species names and are matched name-to-name.

⚠️ Nothing about the probe's behaviour on any rebuilt cell has been looked at, and the resolver has not
been run over VFDB's 405 names.

---

## 1. Design

⚠️ **No new inference.** Same rows, same embeddings, study G's clean fold. **Only the definition of one
binary factor changes.**

---

## 2. Primary tests, frozen bands

**I-1 — how much moved.** The share of eligible pool proteins that change side under the rebuilt factor.

| band | reading |
|---|---|
| **< 1%** | 🟢 the string test was adequate; studies F and G take a footnote |
| **1 – 5%** | ⚠️ material — F and G's cells are restated |
| **> 5%** | 🔴 the string factor was substantially wrong, and every number built on it is restated |

**I-2 — did it change a conclusion?** Studies F and G re-run under the rebuilt factor, both arms.

| | preserved if |
|---|---|
| ratio of ratios | **RR ∈ [0.67, 1.5]** on both arms, as frozen for study F |
| joint share | within **±10 pp** of the string-factor value |
| provenance within intracellular (F-5) | still **> 1.0** on both arms |

🔴 **If RR leaves the band on both arms**, the finding that provenance and localization compose was an
artifact of name matching, and entries 54 and 56 and `paper/MANUSCRIPT.md` are amended in the same
commit that records it.

## 3. Frozen predictions

- **I-1: 1 – 5%**, the material band. The bridge's 12 renamed species are a hint at the rate, not a
  measurement of it, and the pool spans 1,996 organisms.
- **The provenance effect grows**, because dilution is being removed: **F-5 rises above 2.685**.
- **RR stays inside the band**; the joint share moves by **less than 5 pp**.
- ⚠️ 🔒 **A prediction deliberately checkable against a wrong answer**: if the measured movement is
  **exactly 0**, that is a **bug** — a failed fetch or a factor that was not actually rebuilt — **not a
  finding**. Seven "checks that cannot fail" have been caught in this repository; this one names its
  own failure mode in advance.

## 4. Floors

- 🔒 Every cell in the rebuilt 2×2 must hold **≥ 150** proteins, as frozen for study F, or I-2 is
  indicative rather than a verdict.
- 🔒 If more than **10%** of VFDB's 405 species names fail to resolve, the rebuilt factor is
  **worse-founded than the string factor it replaces**, and this study reports that instead of a
  verdict.
- 🔒 Unresolved names are **listed**, not silently dropped.
