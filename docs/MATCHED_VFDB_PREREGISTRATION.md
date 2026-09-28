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
