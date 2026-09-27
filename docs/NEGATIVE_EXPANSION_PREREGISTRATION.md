# Preregistration — expanding the negative side

**Frozen 2026-09-27, before any of it is run.** Amendments append below and never edit what is above,
following `docs/MUTATION_EXTENSION_PREREGISTRATION.md`'s discipline. Two studies, both keeping hazard =
**toxin**. The class-axis study that changes the construct to "virulence factor" is study B of
`docs/VFDB_CLASS_AXIS_DESIGN.md` § 5 and is **not** preregistered here.

Multiplicity: six primary tests below, so **α = 0.05 / 6 = 0.0083** one-sided where a test is
directional, and every threshold carries a **ceiling** as well as a floor. A study that can only
confirm is not a test — § 10.3 of `docs/MECHANISM_GENERALIZATION.md` is the record of what happens when
a preregistered claim is allowed to fail, and both studies here are written so they can.

---

## A2 — the matched benign enzyme set

### Why

`benign_control_sites.json` holds **four** verified controls (1AST astacin, 1LNF, 1LYZ lysozyme,
1QD2). Three of the mutation preregistration's four AUROC halves are unusable at that n, which the
preregistration records as **one finding about its own design, not three about the data**, and it puts
the threshold at *"somewhere above n_benign of roughly 30"*.

⚠️ **The existing n = 4 result points the wrong way for this project, and that is why it is worth
doing.** The benign controls **exceed** the panel on dFSPE-M: benign mean **+5.26**, astacin **+8.42**,
permutation *p* = 0.678. Section 4 of that preregistration says *"Not supported if the benign controls
match or exceed the toxins"*. At n = 4 that is not a verdict. At n ≥ 30 it will be.

### Construction, frozen

Source: Swiss-Prot `Active site` features, `.external/db/uniprot_sprot.fasta` (already local, no new
download). Acceptance is `src/46`'s existing rule, unchanged: **the offset must be the unique integer
in [0, 60] placing every expected residue identity correctly.** A candidate failing it is **excluded,
not rescored**.

Exclusions, applied in this order and logged per candidate:

1. Any accession in the v2 or v3 panel, positive or negative.
2. Any accession appearing in VFDB setA or setB.
3. Any entry whose name or keywords carry a toxin, virulence, hemolysin, or select-agent term.
4. Any entry above the panel negatives' own maximum similarity to a panel positive — the same bound
   `src/60_negative_test_partition.py` reads from the screen artifact, not a fresh number.
5. Fewer than **3** annotated catalytic residues, since dFSPE-M averages over the catalytic set.

Matching to the 14 FSPE panel proteins: **median length within a factor of 2**, and the EC first digit
distribution reported rather than forced, because forcing it would shrink the achievable n and the
whole point is n.

Target **n ≥ 30**. 🔴 **If the pipeline yields fewer than 30 after exclusions, the AUROC halves stay
untestable and that is reported as the result** — the exclusions are not relaxed to reach the number.

### Primary tests

| # | test | supported if | not supported if |
|---|---|---|---|
| **A2-1** | dFSPE-M, panel vs benign, one-sided | panel > benign, *p* < 0.0083 | 🔴 **benign ≥ panel**, which is what n = 4 shows |
| **A2-2** | AUROC, panel vs benign | ≥ 0.70 with the 95% interval clear of 0.50 | interval covers 0.50 |
| **A2-3** | shuffled-label AUROC, same pipeline | within [0.40, 0.60] | outside it ⇒ leak, and A2-2 is void |

🔴 **The prediction this document commits to is that A2-1 fails.** The mutation axis measures positional
constraint, not hazard — P2 failed its own ceiling and P6's alignment reproduces both verdicts — so a
properly powered benign set should **match or exceed** the panel. If A2-1 is supported, that is a
surprise and § 10 of the mutation preregistration needs rewriting, not this document.

---

## A1 — VFDB non-toxin virulence factors as hard negatives

### Why

Criterion 1 of `docs/DETECTOR_CRITERIA.md` is the project's worst: the panel's negatives split
**178 train / 118 calibrate / 0 test**, so every false-positive figure was measured on negatives the
pipeline had seen. Out of sample against the 8,259-protein benign pool the nominal 5% delivers
**7.87%** (`np.quantile`) and **5.98%** (conformal), and collapsing to distinct names raises it to
9.64%.

**VFDB's 4,507 verified non-toxin virulence factors are the adversarial version of that test.** They
are pathogen-produced and largely secreted — the two features the provenance control (AUROC 0.818) and
the localization controls already show this representation reads. A benign pool of cytosolic housekeeping
proteins is an easy negative set. These are not.

### Construction, frozen

Negatives: VFDB setA, the 4,507 records whose category is not `Exotoxin`, deduplicated by VFG id, with
exclusion rules 1 and 4 of A2 applied (panel membership, and the panel negatives' own maximum
similarity bound). Positives for the FPR measurement: unchanged v3 panel, so the comparison is against
published numbers.

### Primary tests

| # | test | supported if | not supported if |
|---|---|---|---|
| **A1-1** | out-of-sample FPR at nominal 5%, quantile estimator, on VFDB negatives | reported with its interval — this is a measurement, not a hypothesis | — |
| **A1-2** | is the VFDB rate **worse** than the benign pool's 7.87%? one-sided | VFDB FPR > 7.87%, *p* < 0.0083 | VFDB FPR ≤ 7.87% ⇒ 🟢 the hard-negative worry is unfounded and § 2.6.1's figure is the honest ceiling |
| **A1-3** | same on the conformal estimator, against 5.98% | as A1-2 | as A1-2 |
| **A1-4** | adding VFDB negatives to LOMO training: beta-lactamase and phage-lysin recovery | 🔴 recovery **falls** for both, as margin predicts | recovery rises ⇒ margin's account is incomplete and § 10.7's dose-response needs re-reading |

🔴 **A1-2 and A1-3 predict the project's own numbers get worse**, and A1-4 predicts that a bigger
negative set makes the two known failures worse rather than better. § 10.7 found that *removing* benign
neighbours repairs 8 to 11% of the beta-lactamase failure; adding 4,507 virulence-associated neighbours
should move it the other way. **If recovery rises instead, margin is not the mechanism it is written up
as**, and that outcome is more interesting than the predicted one.

### What neither study touches

v2 stays frozen: no published recovery figure, margin value or interval is recomputed. A1 adds a
**second** negative set reported alongside the existing one; it does not replace the 178/118 split or
restate any § 2 number. A2 adds controls to the mutation extension and changes no FSPE panel figure.

### Host and species

Out of scope here and deliberately. § 4.2 of `docs/VFDB_CLASS_AXIS_DESIGN.md` establishes that the host
axis already survives class holdout at 0.804 and that recovery does not follow it, so the species
question needs setB and its own preregistration, not a rider on these two.

---

## Amendments

Append-only. Nothing above this line is edited.

### Amendment 1 — 2026-09-27, before A2 was run: exclusion 2 cannot be applied as written

A2's exclusion 2 says *"any accession appearing in VFDB setA or setB"*. **VFDB has no UniProt
accessions.** Its records are identified by VFG ids and GenBank or RefSeq accessions
(`VFG037176(gb|WP_001081735)`), so there is nothing to join on.

Applied instead as an **exact sequence match against setB's 30,215 protein records**, which is stricter
in one direction (it catches a VFDB protein deposited under a different accession) and weaker in
another (it misses a VFDB entry whose sequence differs by one residue from the Swiss-Prot canonical
form). Recorded here rather than reinterpreted silently. **It caught two candidates**, O33407 and
B0VMS2, so the rule was not decorative.

### Amendment 2 — 2026-09-27, after A2 was built and before anything is measured on it

`src/68_benign_enzyme_set.py`. **60 controls admitted against a floor of 30**, from a pool of 66,275
reviewed Swiss-Prot entries carrying an `Active site` in the panel's own length window [286, 1147].

| | |
|---|---|
| admitted | **60** (compute cap; the floor was 30) |
| examined after the cheap rules | 62 |
| rejected: fewer than 3 Active sites | 332 |
| rejected: exact VFDB setB sequence | 2 |
| rejected: hazard term in name or keywords | 1 (Q8FHF4) |
| rejected: in the panel / above the similarity bound | **0 / 0** |
| length | 289 to 1,043, median 440 |
| EC first digit | 3:26, 2:16, 6:8, 1:6, 4:1, 5:1, unassigned:2 |
| catalytic sites per control | 3 for 52 of them, 4 for five, 5 for one, 6 for two |

**Selection order is `sha256(accession)` ascending**, a reproducible draw from the whole pool rather
than the head of accession order, which would have been biased toward old, well-characterised `P0xxxx`
entries. Maximum similarity to any v3 positive across all 60 is **0.047**, far below the 0.282 bound,
so rule 4 never bound.

🔴 **Three properties of the delivered set are flagged now, before any measurement, because noticing
them afterwards would be indistinguishable from explaining a result.**

1. **The set is hydrolase-heavy: 26 of 60 are EC 3.** That is not a defect — it is the right direction,
   since both classes the probe cannot flag are hydrolases (§ 1.1b-ter of the survey) — but it means the
   benign side is enriched for exactly the fold family the panel's failures come from, and A2-1's
   comparison inherits that.
2. **`P0CK11` is a 1,043-residue Turnip mosaic virus P3N-PIPO polyprotein**, the only viral-origin
   member and the longest. Its `Active site` features belong to a protease domain inside a multi-domain
   precursor, so its masked-position context is unlike the single-domain enzymes around it. **If it is
   an outlier in A2, that was predicted here and is not a post-hoc exclusion.** It is kept: no
   domain-architecture rule was frozen, and adding one after seeing the set would be tuning.
3. **A few controls come from pathogens** — *Salmonella*, *Vibrio vulnificus*, *Pseudomonas
   aeruginosa*. Deliberately kept. § 2.3's provenance probe reaches AUROC 0.818 on hazard from
   lab-strain provenance alone, so benign enzymes with pathogen provenance make the control **stronger**
   against that confound, not weaker.

⚠️ **No pathogen-origin or domain-architecture exclusion was added.** Both were considered after seeing
the set, which is precisely when a new exclusion stops being a rule and becomes a result.

### Amendment 3 — 2026-09-27: A2 run, and all three tests resolve

`src/62_fspe_m.py --controls a2`, artifact `results/fspe_m_a2.json`, ESM-2 650M, 15 panel proteins and
60 controls, 20 background positions each. The panel's dFSPE-M values are **bit-identical** to the
frozen-four run, which is the check that the control set is the only thing that changed.

| test | frozen at the top of this document | result | verdict |
|---|---|---|---|
| **A2-1** | panel > benign, one-sided *p* < 0.0083 | panel **+4.32**, benign **+6.76**, difference **−2.44**, one-sided *p* = **0.9991** | 🔴 **NOT SUPPORTED**, on the ceiling |
| **A2-2** | AUROC ≥ 0.70, interval clear of 0.50 | **0.265**, 95% CI **[0.102, 0.445]** | 🔴 **NOT SUPPORTED**, and the interval is clear of 0.50 **on the wrong side** |
| **A2-3** | shuffled AUROC inside [0.40, 0.60] | **0.430** | 🟢 **SUPPORTED** — no leak |

🔴 **The prediction this document committed to was that A2-1 would fail, and it failed harder at power
than it did at n = 4.** The benign mean rose from +5.26 to +6.76 and the gap from −0.94 to −2.44. At
n = 4 the ceiling fired on direction only, *p* = 0.678; at n = 60 the controls exceed the panel at
*p* = 0.0009. **All 60 controls sit above the panel's worst protein and 59 of 60 above zero.**
dFSPE-M measures catalytic-site constraint, and ordinary benign enzymes have more of it than toxins do.

⚠️ **The pre-flagged outlier behaved as flagged.** `P0CK11`, amendment 2's viral polyprotein, is the
only control below zero at **−0.493** and the only one whose sequence is truncated by `MAX_SEQ_LEN`
(1,043 residues against 1,022, with all its catalytic positions inside the kept region). Keeping it
**lowers** the benign mean, so it works against this result; it is kept, as amendment 2 said it would
be.

🔴 **And the gate's AUROC half is still not a test, for a reason the mutation preregistration's fifth
amendment got wrong.** `src/71_a2_gate_power.py`, 20,000 label permutations:

| | n_panel 15, n_benign 4 | n_panel 15, n_benign 60 |
|---|---:|---:|
| observed shuffled AUROC | 0.2833 | **0.4300** |
| null sd | 0.1669 | **0.0834** |
| a clean pipeline fails the ±0.05 gate | 77.3% | **55.5%** |
| two-sided *p* for the observed value | 0.2243 | **0.4059** |

The n = 4 column **reproduces the fifth amendment's numbers exactly** (0.1669, 77.3%, 0.224), so the
construction is theirs. That amendment predicted the tolerance "becomes meaningful somewhere above
n_benign of roughly 30". **It does not, because n_benign is not the binding constraint.** An AUROC's
precision is set by the *smaller* group, and sweeping n_benign at a fixed panel of 14 scorable proteins:

| n_benign | 4 | 10 | 30 | 60 | 120 | 500 | 5,000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| null sd | 0.170 | 0.119 | 0.094 | 0.088 | 0.083 | 0.079 | **0.077** |
| a clean pipeline passes | 20% | 31% | 39% | 43% | 45% | 48% | **47%** |

🔴 **The null sd floors at about 0.077 and the ±0.05 gate never passes more than about half the time,
no matter how many controls are added.** `functional_sites.json` has 16 entries, 15 with catalytic
residues, and that is the whole panel. So the fifth amendment's resolution — report the shuffled AUROC
without a pass/fail — stands permanently rather than provisionally, and its stated repair condition was
attributed to the wrong sample size. A2-3's own [0.40, 0.60] window is ±0.10, about 1.2 null standard
deviations, which a clean pipeline clears roughly 77% of the time: **better than the gate it replaces
and still not a strong test.** Said here rather than left for a reader to work out.


### Amendment 4 — 2026-09-27, on review, after A2 was reported

🔴 **A2's two sides are annotated by different methods, and nothing above says so.** Its construction
specifies "Swiss-Prot `Active site` features" for the controls and says nothing about the panel, whose
catalytic positions come from PDB and literature curation. Measured: **10 of the panel's 74 annotated
positions are a UniProt `Active site`** (≈32% on the annotation audit's comparable subset) against
**100%** of the controls'.

So A2-1's result — benign mean +6.76 against the panel's +4.32 — **does not distinguish "benign enzymes
are more constrained at their catalytic sites" from "UniProt's curated active sites are more constrained
than mixed literature annotations."** A2-1's *verdict* is unaffected, because the preregistered ceiling
asks whether the panel exceeds the controls and the burden was on the panel; the *interpretation* is.

⚠️ **This should have been an exclusion rule in A2's construction and was not.** The frozen rules covered
panel membership, VFDB membership, hazard terms, similarity and site count — every axis except whether the
two sides' positions mean the same thing. Recorded here rather than repaired silently. **The repair:
restrict the panel to its UniProt-confirmed `Active site` positions and recompute both means**, which
costs one rerun of `src/62` on a filtered annotation and is not done here.


### Amendment 5 — 2026-09-27: A1's negative set is built, and it needs an effective-*n* before its rate is quoted

`src/74_vfdb_negative_set.py`. **4,218 admitted** from VFDB setA, maximum similarity to any v3 positive
**0.2663** against the frozen 0.282 bound, so rule 4 bound on 41 candidates and nothing admitted comes
close to the limit. Rejections: 248 Exotoxin (A1's positive class), 222 length outside [50, 1022], 41
similarity, 12 exact sequence match to a panel protein, 12 non-standard residues, 2 duplicate sequence.
Categories: Effector delivery system 1,691 · Immune modulation 658 · Adherence 607 · Motility 595 ·
Nutritional/Metabolic 320. 38 genera, *Legionella* 440.

🔴 **A1's rate must not be quoted on the raw 4,218.** § 10.9 established the rule for the benign pool:
8,259 raw records are 3,550 distinct names, a redundancy factor of 2.33, and **the distinct-unit rate is
the honest one** — it raised the out-of-sample figure from 7.87% to 9.64%. A1 currently removes only
**exact** sequence duplicates, and `src/74` does not record VFDB's own `VF####` id, which is the natural
redundancy unit for a database that lists one representative gene per virulence factor per organism.

**So before `src/49` is run on this set**, `src/74` gains the `VF####` grouping and A1-1 to A1-3 report
both rates, raw and per distinct unit, exactly as § 2.6.1 does for the pool. Stated now because quoting
the raw rate first and the corrected one later is how a number escapes.
