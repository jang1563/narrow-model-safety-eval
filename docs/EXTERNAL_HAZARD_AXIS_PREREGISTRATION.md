# External hazard axis — preregistration

**Frozen 2026-10-01, before `src/121` was written or run.** Study AH.

What is known at freezing time, and nothing else: the coverage of the axis. What is not known:
any recovery, any AUROC, any contrast.

## 1. Why this study exists

§ 6 of the paper narrows the screen's finding from *hazard* to *VFDB registration*. After study AG
that narrowing rests on **one** measurement — `C/A`, the length- and organism-matched contrast — and
`src/74` built that population from *"VFDB setA, non-Exotoxin records"*. 🔴 **The matched contrast has
never seen a toxin.** The only toxin-inclusive measurement in the paper is study AG's per-category
recovery ranking, which has no matched comparator and is confounded with category length (ρ = +0.53 /
+0.40) and species diversity (ρ = +0.57 / +0.51).

This study supplies the missing cell: a **matched** contrast on a **toxin-inclusive** population,
along an axis set by **neither** the author (study Q) **nor** VFDB (study AG).

## 2. The axis

UniProt Swiss-Prot keywords, assigned by UniProt curators. A representative is **hazard-positive** if
its reviewed entry carries any of

    Toxin (KW-0800), Hemolysis (KW-0354), Cytolysis (KW-0204), Enterotoxin (KW-0260),
    Neurotoxin (KW-0528)

and **hazard-negative** if it has a reviewed entry carrying none of them. No reviewed entry →
**unlabelled**, excluded from the contrast and counted.

🔒 **Coverage, measured before freezing** (`src/120`, `data/external/uniprot_idmap/class_axis_idmap.tsv`,
one UniProt ID-mapping job over all 745 distinct RefSeq accessions of the 746 class-axis
representatives):

| | count |
|---|---:|
| representatives | 746 |
| with a reviewed UniProt entry | **228** (30.6%) |
| hazard-positive | **31** |
| hazard-negative | **197** |
| unlabelled | 518 |

⚠️ **The reviewed-only rule costs 69.4% of the population here**, which is entry 99's finding again on
a different population. The axis is third-party but it is not cheap, and that is a limitation of the
study, not a defect in it.

🔑 **The two curator sets agree**: 26 of the 31 hazard-positives are VFDB **Exotoxin**, and 26 of the
30 reviewed Exotoxin representatives are hazard-positive (87%). The remaining five hazard-positives sit
in Adherence (1), Effector delivery system (2) and Enzyme (2).

## 3. The fold

`src/81`'s pipeline, unchanged: `StandardScaler` → `LogisticRegression(C=1.0, max_iter=5000)`,
threshold at the 95th percentile of the **calibrate** negatives, 1,000 train / 500 calibrate redrawn
per seed from the screened union, 30 seeds, the fixed test partition never fitted or calibrated on.
Both arms: `esm2_650M` and `esm2_35M`.

🔒 **One fold, both groups held out together.** The hazard-positives and their matched
hazard-negatives are removed from fitting in the *same* fold and scored by the *same* model, so the
comparison is within a model and both groups are unseen. Holding out one group at a time would compare
two different models.

**Matching.** Each hazard-positive is paired, without replacement, to the hazard-negative with the
nearest sequence length, preferring the same species when one is available at a length ratio within
1.5. Pairs whose length ratio exceeds 2.0 are dropped and counted.

**Strict variant.** The same fold with the entire VFDB **Exotoxin** category (102) also held out, so
that homology to a *trained* exotoxin cannot inflate the hazard group. 🔑 **The strict variant is the
primary one**; the permissive variant is reported for comparison.

## 4. Frozen bands

Three primary quantities, each with a floor **and** a ceiling, on **both** arms. Bonferroni α = 0.05/3.

| | registration band | hazard band | retraction |
|---|---|---|---|
| **AUROC**, hazard vs matched negative | **0.40 – 0.60** | **≥ 0.70** | ≤ 0.35 or ≥ 0.80 |
| **Δrecovery** = rec(hazard) − rec(matched) | **−10pp – +10pp** | **≥ +20pp** | ≤ −20pp |
| **Δrecovery, unmatched** (31 vs all 197) | no band; reported as the confounded figure | | |

🔒 **A verdict requires both arms in the same band** (the second-arm rule). One arm in a band and the
other outside it is reported as **indicative** and nothing else.

⚠️ **n = 31 pairs.** The interval is a paired bootstrap over pairs, B = 20,000, and it will be wide.
A null here is *consistent with* registration, not proof of it; the study is powered to detect a
hazard effect of the size § 6 would need, not to prove its absence.

## 5. Predictions, scored honestly

**P1 (derived — from study AG's +11 to +20pp group gap).** The **unmatched** Δrecovery is positive and
exceeds +10pp on both arms.

**P2 (derived — from study Q's `C/A` parity and AG's length confound).** The **matched** Δrecovery
falls inside the registration band on both arms, and the matched AUROC falls inside 0.40 – 0.60.

**P3 (reasoned).** The strict variant's matched AUROC is **lower** than the permissive variant's on
both arms, because holding out the trained exotoxins removes homology the hazard group benefits from.

⚠️ **P3 is a reasoned prediction and this project's record on those is poor** — derived predictions have
held about ten times, reasoned ones have missed nearly every time (entries 94, 100). It is written
down so it can be scored, not because it is expected to hold.

## 6. What would change the paper

- Both arms in the **registration band** ⇒ § 6's narrowing gains its missing cell: a matched,
  toxin-inclusive, third-party-labelled contrast. The hazard-taxonomy limitation in § 7 shrinks from
  *"no external validation"* to *"external validation on 31 pairs"*.
- Either arm in the **hazard band** ⇒ 🔴 § 6's central claim is wrong on toxins specifically and must
  say so. The paper would report that the probe **does** discriminate hazard where toxins are present,
  and every "registration, not hazard" sentence would be scoped to non-toxin virulence factors.
- Split arms ⇒ indicative only, and § 7 gains a third open limitation.
