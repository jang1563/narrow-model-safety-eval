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

---

## 🔴 Amendment 1 — 2026-09-28, before resolving in bulk: there were never 405 species

This document was frozen saying VFDB has **405** species names, and § 4's unresolved ceiling is stated
against that number. 🔴 **Both come from a defective regex**, found by this study's own resolver on its
first output, which listed *"Accessory secretion"* and *"Acid phosphatase"* as species (entry 58).

A VFDB header has two bracket groups and `re.search` took the first, so **133 of the 405 were VF names,
not organisms**, and **11 real pathogens were missing**. The corrected extractor — `vfdb_species()` in
`src/83`, taking the *last* bracket group — gives **283** species.

🔒 **§ 2's bands, § 3's predictions and § 4's floors are unchanged.** The only substitutions are
**405 → 283** wherever the count appears, and the 10% unresolved ceiling now applies to 283, i.e. a
budget of **28 names**.

⚠️ **This makes I-1's prediction harder, not easier.** § 3 predicted 1–5% movement partly on the
strength of the bridge's 12 renamed species; the corrected set already recovers 11 organisms the buggy
one lost, and studies F, G and H re-ran **with every quantity identical to four decimal places**
(entry 58). 🔒 **So the movement I-1 measures is now purely the genus-rename effect**, with the
regex defect removed from underneath it — and if that is small, § 3's band will be missed low.
**Recorded before the number exists.**

---

## Results, 2026-09-28

`src/88_taxid_provenance.py`, study G's clean fold, 30 seeds, no new inference.
`results/taxid_provenance.json`, `results/taxid_provenance_esm2_35M.json`.

**Resolution**: 283 VFDB names → **224 by scientific name, 45 by synonym, 14 unresolved (4.9%,
under the 10% ceiling)** → **268 canonical species**. The 45 synonym resolutions are the renamed
pathogens the string factor was losing. Unresolved names are listed in the artifact and include
*Clostridium difficile*, *Mycobacterium bovis* and *Francisella novicida* — reclassified or
subspecies-rank entries.

🔒 **Reproduction gate passed on both arms.** Under the string factor this script returns
**RR = 0.999** (650M) and **1.358** (35M), reproducing study G's clean condition to |Δ| ≤ 0.0003. The
two conditions differ in one binary factor and nothing else.

### 🔴 I-1: the string factor was substantially wrong

| cell | string | taxid | Δ |
|---|---:|---:|---:|
| pathogen × extracellular | 171 | **201** | **+30** |
| pathogen × intracellular | 597 | **679** | **+82** |
| benign × extracellular | 444 | 414 | −30 |
| benign × intracellular | 1,049 | 967 | −82 |

**310 of 6,139 eligible proteins changed side = 5.05%**, over the frozen 5% boundary and therefore in
the band the document calls **"substantially wrong"**. 🔴 **112 genuine pathogen proteins were sitting
in the benign cells.**

### 🟢 I-2: every frozen criterion survives, and the provenance effect grows as predicted

| | string | taxid | |
|---|---:|---:|---|
| RR, 650M | 0.999 | **0.692** | in band [0.67, 1.5] |
| RR, 35M | 1.358 | **1.133** | in band |
| joint share, 650M | 0.460 | 0.412 | within ±10 pp |
| joint share, 35M | 0.322 | 0.276 | within ±10 pp |
| **F-5, 650M** | 2.632 | **3.469** | ⭐ grew, as § 3 predicted |
| F-5, 35M | 2.350 | 2.417 | grew |

🟢 **Removing the dilution makes provenance stronger**, which is the direction § 3 named in advance:
with localization held fixed, a pathogen protein is now **3.47×** more likely to be flagged than a
benign one on 650M, against 2.63× under the string factor.

### ⚠️ But the independence finding is weaker than it was, and that has to be said

**RR moved −0.307 on 650M and −0.225 on 35M.** Both point estimates remain inside the frozen band, so
I-2 passes **as written** — but:

- 🔴 650M's interval is **[0.649, 0.735]**, which **crosses the band floor of 0.67**. Independence is
  no longer cleanly established on that arm.
- 🔴 **The two arms now straddle 1.0 in opposite directions** — 0.692 is sub-multiplicative, 1.133 is
  super-multiplicative (its interval excludes 1.0). Under the string factor they were 0.999 and 1.358,
  both at or above 1.

🔒 **The frozen rule governs and the claim stands, narrowed**: provenance and localization still
compose within the band, but **"almost exactly independent" was a property of the defective factor**,
not of the data. Entry 54's 650M figure of 1.009 is superseded by 0.692.

### 🔒 Scoring the frozen predictions

| | frozen | actual | |
|---|---|---|---|
| I-1 | 1 – 5% | **5.05%** | ❌ missed **high**, by 0.05 pp |
| provenance grows, F-5 > 2.685 | | 3.469 / 2.417 | ✅ 650M, ⚠️ 35M rose but stayed below |
| RR stays in band | | 0.692 / 1.133 | ✅ both arms |
| joint moves < 5 pp | | 4.9 pp / 4.6 pp | ✅ both, narrowly |
| "0 movement is a bug, not a finding" | | 310 moved | ✅ guard not triggered |

⚠️ Amendment 1 predicted that removing the regex defect would make I-1 **miss low**. It missed
**high**. Recorded.

### ⚠️ Declared limitation: 5.05% is a lower bound

The pool side reads its species from a rank-annotated lineage, and **2,122 of 6,758 proteins have no
`(species)` entry** because the organism is itself at species rank or below. Those fall back to the
organism string:

| | *n* | |
|---|---:|---|
| bare two-word name | 941 | already the canonical species — fallback is correct |
| ⚠️ **parenthetical synonyms** | **823** | e.g. *Enterobacter agglomerans (Erwinia herbicola) (Pantoea agglomerans)* — **the first name is the old one** |
| unclassified *Genus* sp. | 358 | no species rank exists to resolve to |

🔑 **So for 823 proteins the pool side still matches on a superseded name**, which is the very defect
this study removes from the VFDB side. Fixing it would move **more** proteins, not fewer.
🔒 **5.05% is therefore a lower bound on the string factor's error**, and the "substantially wrong"
verdict is the conservative reading.

---

## 🔴 Amendment 2 — 2026-09-28: the declared limitation above was wrong, by about fifty-five fold

The **"5.05% is a lower bound"** section says 823 pool proteins still match on a superseded name
because their organism string reads *Enterobacter agglomerans (Erwinia herbicola) (Pantoea
agglomerans)* and the fallback takes the first two words. 🔴 **That is not a defect, and checking it
before building the follow-up study is what showed it.**

**UniProt's taxonomy keeps the same primary name the protein record displays.** Taxid 549's
`scientificName` is **`Enterobacter agglomerans`** — the old name — not *Pantoea agglomerans*. And the
resolver maps VFDB's current-name entry onto that same primary: `resolve_species("Pantoea
agglomerans")` → **`(549, 'Enterobacter agglomerans', 'synonym')`**. 🔑 **Both sides were already in
one vocabulary and already agreed.**

🔒 **Verified exhaustively rather than by example.** Every fallback protein whose display name carries
parenthetical synonyms — **799** of them, across **254 distinct taxids**, all resolved — was compared
against its taxon's `scientificName`:

| | count |
|---|---:|
| ⭐ **fallback name == UniProt `scientificName`** | **799** |
| disagree | **0** |
| taxid unresolved | 0 |

### 🔒 The real residual, which is small and not fixable by a better resolver

Of the **14** VFDB names that failed to resolve, exactly **two** have any pool protein:
*Clostridium difficile* (UniProt: *Clostridioides difficile*, **13** proteins) and *Borrelia
bavariensis* (UniProt: *Borreliella bavariensis*, **2**). That is **15 of 6,139 eligible = 0.24%**,
against study I's measured 5.05% — and only **4** reach the 2×2 cells at all, every one of them
*C. difficile* in the intracellular stratum.

⚠️ **No resolver change recovers them.** UniProt carries neither old name as a searchable synonym,
checked at `size=200` both with and without the `rank:species` filter. *Mycobacterium bovis* resolves
only at rank **`biotype`**, below species, and contributes no pool proteins.

🔒 **The conclusion survives and the magnitude does not.** Those 15 are still pathogens scored benign,
so **5.05% remains a lower bound** and "substantially wrong" is still the conservative reading — but
the gap is **0.24%, not 823 proteins**, and the earlier section overstated it by roughly fifty-five
fold.

### ⚠️ Housekeeping found alongside it

The resolver cache still held **68 entries from the defective 405-name list** of entry 58 — junk like
*"Accessory secretion"*, all marked `UNRESOLVED`. Study I filtered them correctly by construction (it
iterates the current 283 names), so **no reported number was affected**, and re-running after the prune
reproduces study I byte-for-byte. 🔒 **Pruned anyway**: a cache holding keys from a superseded build is
the same shape of trap as entry 45's stale shard.

---

## 🔴 Superseded on the share-of-gap figures — 2026-09-28, entry 61

**542 of the 4,218 VFDB negatives share a sequence with one of the 746 class-axis positives.**
`src/74` screened that set against the **panel** positives, which is what study A1 needed, and nothing
screened it against the **class-axis** positives that came later — so every evaluation of the study-B
probe on this population was scoring the probe's own training data on **12.85%** of the rows.

The VFDB reference drops from **76.95% to 74.09%** on 650M under study G's clean fold (and study D's
own figure from **73.49% to 70.91%**). Every share-of-gap divides by `(vfdb − pool)`, so **all of them
were about 4.3% too small**. 🔒 **No ratio moved** — R, RR, C/A, D/B and F-5 do not involve the
reference and are unchanged to four decimals.

| this document's figures | before | after |
|---|---:|---:|
| taxid joint share, 650M | 41.2% | **42.8%** |
| taxid joint share, 35M | 27.6% | **28.7%** |

🔒 I-1's 5.05% and both ratios of ratios are unchanged — they do not involve the reference.
