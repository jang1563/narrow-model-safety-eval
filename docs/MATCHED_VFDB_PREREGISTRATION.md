# Preregistration — with provenance *and* localization held fixed, does VFDB membership still matter?

**Frozen 2026-09-28, after a declared feasibility count and before the probe has been applied to these
populations.** One primary test, one second-arm replication, one reference. Append-only.

---

## 0. The question three studies have now left standing

| control | share of the benign→VFDB gap |
|---|---:|
| pathogen origin (study D) | 22.7% |
| localization (study E) | 22.1% |
| ⭐ **both together, measured** (studies F, G) | **46.0%** on 650M, 32.2% on 35M |

🔑 **So the majority belongs to neither, and every one of those documents says what it belongs to is
not established.** The candidates are *hazard* — being a virulence factor — or some further property
the curated sets share. **This is the first test in the project that can separate them**, because it
holds both measured confounds fixed and varies only VFDB membership.

## 🔒 0.1 The instrument is the contamination study G removed

133 pool proteins are exact VFDB sequence matches. 🔑 **They are simultaneously VFDB virulence factors
*and* Swiss-Prot entries with UniProt localization annotation**, which is the pairing no other
population in this repository has: the 4,218-protein VFDB set has no UniProt accessions to annotate.
**The contamination becomes the bridge.** All 133 are held out under study G's clean fold, which drops
every contaminant from the admitted rows, so none is ever fitted or used to set a threshold.

### 🔴 0.2 The bridge is selected *against* the hypothesis, and that is declared, not discovered later

The pool's build query is
`reviewed:true AND length:[100 TO 1400] NOT keyword:KW-0843 NOT KW-0800 NOT KW-0204 NOT KW-0354 NOT
KW-0078 NOT KW-0081`, which excludes **Virulence, Toxin, Cytolysis, Hemolysis, Bacteriocin** and
**Bacteriolytic enzyme** (names read from UniProt, not assumed).

🔴 **So every bridge protein is a VFDB virulence factor that UniProt declines to annotate as virulent,
toxic, cytolytic or haemolytic.** They are the least obviously hazardous members VFDB has.

- 🟢 **If they are still elevated, that is strong** — it survives a subset chosen to be hard.
- ⚠️ **If they are not, it is weak evidence of absence**, because the informative members were removed
  by a query written for another purpose. **Stated now so the null cannot be over-read later.**

## 🔒 0.3 Declared feasibility count, taken before freezing

| | extracellular | intracellular |
|---|---:|---:|
| **bridge (pool ∩ VFDB)** | **45** | **26** |
| non-VFDB, VFDB-species | 171 | 597 |

Also 12 membrane and 50 unannotated on the bridge side, reported but not in the primary. Nothing about
the probe's behaviour on any of these has been looked at. 🔒 **The floor below is set at 25** with those
counts visible and stated as such.

---

## 1. Design

⚠️ **No new inference.** All four populations are pool rows, already embedded on **both** arms — which
is why the bridge is used rather than the 4,218-protein VFDB set, which exists on 650M only.

**Fold**: study G's clean condition exactly — contaminants dropped from the admitted rows, sizes by
`src/83`'s rule, 30 seeds, threshold at the 95th percentile of calibration scores.

🔒 **Both sides are restricted to Bacteria from VFDB species**, so provenance is held fixed, and
compared **within** a localization stratum, so localization is held fixed. **The only thing that
differs is VFDB membership.** Both sides come from the *same build query and the same annotation
pipeline*, which no earlier comparison in this project could say.

🔒 **A positive-side contamination check runs first**: any bridge protein whose sequence is also one of
the 746 class-axis positives is excluded. A positive evaluated as a test case would be circular, and
this has not been checked before.

---

## 2. Primary test, frozen bands

**H-1**: `C/A` = flag rate(bridge × extracellular) / flag rate(non-VFDB × extracellular), and `D/B`
the same within intracellular. **H-2**: both on ESM-2 35M. Extracellular is primary; it has the larger
cells. A verdict requires **both arms in the same band**.

| band | reading |
|---|---|
| **≥ 2.0** | 🟢 VFDB membership matters beyond both confounds — the residual is virulence-associated |
| **1.2 – 2.0** | ⚠️ partial |
| **≤ 1.2** | 🔴 membership adds little once provenance and localization are held fixed |

🔴 **What the adverse band obliges, written before the number exists**: if `C/A ≤ 1.2` on both arms,
then the residual majority of the gap is **not** shown to be about virulence, and every document that
currently says "roughly three quarters is not localization" must be amended to say that the remainder
is also not attributable to VFDB membership — in the same commit, on the terms study E's obligation was
executed.

## 3. Reference and secondaries

- **H-3**: the 4,218-protein VFDB set under the *clean* fold on 650M, so the bridge can be placed
  relative to the full set rather than against 73.49%, which was computed on a contaminated fold.
- **H-4**: the bridge's membrane and unannotated strata, reported for completeness.

## 4. Frozen predictions

- **H-1: C/A ≈ 1.8**, in the partial band, with the bridge's extracellular rate near **60%** against
  the benign 35%. **D/B ≈ 2.5.**
- **H-3**: the bridge sits **below** the full VFDB set, because § 0.2's query removed the obvious
  members.
- ⚠️ **Both cells are small**; intervals are expected to be wide and that is not a defect to be
  reported as a surprise.

## 5. Floors

🔒 Bridge cells ≥ 25 after the positive-side exclusion, or the affected test is indicative only.
🔒 § 0.2's selection caveat travels with every number this study produces.

---

## 🔴 Amendment 1 — 2026-09-28, on running it: a taxonomic synonym is not evidence about provenance

The first run applied the "Bacteria from a VFDB species" test to **both** sides. 🔴 **That is wrong on
the bridge side**: a protein whose sequence matches VFDB exactly is pathogen-derived *by construction*,
and the species test dropped ten of them because **UniProt has renamed the genus while VFDB still uses
the old name**. The bridge spans **12** such renamed species in total: `Borreliella burgdorferi`,
`Klebsiella aerogenes`, `Mycobacteroides abscessus`, `Mycolicibacterium gilvum`, `M. paratuberculosis`,
`M. smegmatis`, `M. vanbaalenii`, `Mycoplasmoides genitalium`, `M. pneumoniae`, `Ralstonia nicotianae`,
`Salmonella typhi`, `Salmonella typhimurium`.

🔒 **Corrected**: the bridge requires only that the protein is bacterial. The benign side keeps the
species test, because that is where provenance has to be held fixed.

🔒 **Both versions are reported so the change cannot read as tuning.** It moved a cell *up* to just
below the floor, which is the direction that would be suspicious, so here is every affected number:

| | before (buggy) | after (corrected) |
|---|---|---|
| bridge extracellular *n* | 37 | **38** |
| bridge intracellular *n* | 15 | **24** |
| C/A, 650M | 1.954 | **1.974** |
| C/A, 35M | 1.435 | **1.497** |
| D/B, 650M | 1.915 | **3.204** |
| ⭐ **D/B, 35M** | **0.532** | **1.311** |

🔑 **The correction is what made the intracellular stratum replicate.** Before it, D/B was 1.92 on 650M
and **0.53 on 35M** — opposite signs. After it, 3.20 and 1.31, the same sign on both arms. The
sign flip was nine proteins removed by a genus rename.

⚠️ **The same synonym problem runs the other way in studies F and G**, whose provenance factor is
"species appears in VFDB": renamed pathogens were scored as benign-species, which **dilutes** the
measured provenance effect. That direction is conservative, and **it is not quantified here.**

---

## 🔒 Reconciling § 0.3's declared counts with the delivered ones

§ 0.3 declared 45 extracellular and 26 intracellular. Delivered: **38** and **24**. The whole
difference is the § 1 positive-side check, which § 0.3's count was taken before:

| | declared | − class-axis positives | delivered |
|---|---:|---:|---:|
| extracellular | 45 | −7 | **38** |
| intracellular | 26 | −2 | **24** |

## 🔴 The circularity check found 13, and it had never been run anywhere

⚠️ The positive FASTA holds **745 distinct sequences for 746 records**, so one positive is an exact
duplicate of another — noted in passing, found only because this check counted them.

**13 of the 133 bridge proteins are themselves class-axis positives** — `A5U8S6`, `P9WGG7`, `P9WGH9`,
`P9WGI3`, `P9WKK6`, `Q8DQ36`, `P05431`, `Q833V7`, `Q9WXB9`, `Q9RQJ2`, `P0A609`, `A1KQD8`, `E8XDJ8`,
mostly *M. tuberculosis* and *M. bovis*. 🔴 **Evaluating those as test cases would have been
circular**, and nothing in this repository had checked for it before this study's § 1 required it.

---

## Results, 2026-09-28

`src/87_matched_vfdb.py`, study G's clean fold, 30 seeds, no new inference.
`results/matched_vfdb.json`, `results/matched_vfdb_esm2_35M.json`.

| population | *n* | ESM-2 **650M** | ESM-2 **35M** |
|---|---:|---:|---:|
| benign × extracellular | 171 | 37.52% [36.18, 38.86] | 26.90% [25.65, 28.15] |
| ⭐ **VFDB × extracellular** | **38** | **73.77% [70.41, 77.13]** | **40.70% [36.34, 45.07]** |
| benign × intracellular | 597 | 6.87% [6.47, 7.27] | 5.09% [4.72, 5.47] |
| **VFDB × intracellular** | 24 | **21.81% [20.47, 23.14]** | 6.67% [5.33, 8.00] |
| VFDB × membrane | 12 | 53.89% | 38.06% |
| VFDB × unannotated | 46 | 42.54% | 12.03% |
| *reference*: full VFDB set | 4,218 | 76.95% [75.80, 78.10] | — |

### ⚠️ H-1 / H-2: partial on both arms, and 650M cannot be told from the supported band

**C/A = 1.974 [1.889, 2.059]** on 650M and **1.497 [1.360, 1.634]** on 35M. Both land in the frozen
**partial** band [1.2, 2.0), so **the band verdict agrees on both arms: partial.** ⚠️ 650M's interval
**straddles 2.0**, so on that arm partial and supported cannot be distinguished; the verdict is the
band the point estimate falls in, as frozen.

⚠️ **D/B is indicative only**: the bridge intracellular cell holds **24**, one short of the frozen
floor of 25. 🔒 **The floor is not being moved for one protein.** It gives 3.204 and 1.311 — elevated on
both arms, and see Amendment 1 for why that is a corrected number.

### ⭐ H-5: membership closes 91.9% of the residual that studies D–G could not attribute

With pathogen origin **and** extracellular localization both held fixed, the benign cell sits at
**37.52%** and the full VFDB set at **76.95%**. VFDB membership alone takes a protein to **73.77%** —
**91.9% of the way.**

🟢 **This is the first positive evidence in the project that the residual majority of the separation is
about being a virulence factor**, rather than a further property the curated sets happen to share. And
🔑 **it is measured on the subset § 0.2 declared to be biased against it**: every one of these 38 is a
VFDB virulence factor that UniProt declines to annotate as virulent, toxic, cytolytic or haemolytic,
and they still reach within 3.2 points of the full set.

### 🔒 What this does not overturn

⚠️ **Study E stands.** The probe remains strongly localization-graded — that is the *benign* finding and
this is a *membership* finding, and both are true: a benign secreted protein is flagged at 37.5% here
against 6.9% cytoplasmic. **The 5% budget is still wrong for secreted proteins by a factor of seven.**
⚠️ **And 35M is much weaker than 650M** — 1.497 against 1.974, with the bridge at 40.70% against 73.77%.
The finding is a verdict on the band, not on the magnitude.

### 🔒 Frozen predictions

| | frozen | actual | |
|---|---|---|---|
| H-1 C/A ≈ 1.8, partial band | | 1.974 / 1.497 | band ✅ both arms |
| bridge extracellular ≈ 60% | | **73.77%** / 40.70% | ❌ underestimated on 650M |
| D/B ≈ 2.5 | | 3.204 / 1.311 | straddles it, underpowered |
| H-3 bridge below the full set | | 73.77% < 76.95% | ✅ |
| wide intervals on small cells | | yes | ✅ |

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
| full VFDB reference | 76.95% (4,218) | **74.09% (3,676)** |
| ⭐ **H-5, membership closes** | 91.9% | **99.1%** |

🔑 **H-5 is the figure this changes most.** With pathogen origin and extracellular localization
both held fixed, VFDB membership covers **essentially all** of the remaining distance to the
decontaminated full-VFDB rate — 73.77% against a reference of 74.09%. ⚠️ **The three
qualifications are untouched**: the band is still *partial* (C/A 1.974 and 1.497, unchanged), D/B
still sits one protein under its floor, and study E is still not softened.
