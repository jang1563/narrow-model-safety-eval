# DeepVIC's published benchmark — preregistration

**Frozen 2026-10-01, before `src/123` existed and before any AUROC, recovery or margin was
computed on this data.** Study AI.

What is known at freezing time: the partition and its **composition**, from `src/122`
(`results/deepvic_build.json`). What is not known: any outcome.

## 1. Why

§ 1.2 states the comparison problem in full: **DeepVIC** ([Bioinformatics Advances 6(1)
vbag237, 2026](https://academic.oup.com/bioinformaticsadvances/article/6/1/vbag237/8762933))
is the largest virulence-factor classifier in the literature, it carries a published
**14-category** class axis over **12,989** annotated VFs, and it **runs no
leave-one-category-out evaluation**. § 1 of the paper disclaims comparability because every
number here is on a self-built panel of 234 or 445 proteins.

Its repository ships the sequences, the class labels **and** the trained weights. So the
class-level question this project exists to ask can be asked on **someone else's benchmark,
with someone else's labels, at 17× the panel size.**

## 2. What `src/122` measured before this document was written

| | |
|---|---:|
| FASTA records | 66,912 |
| positives / negatives | 33,456 / 33,456 |
| positives carrying a class label | **12,989** in **14** classes |
| class `NA` | 20,467 |
| source databases | UNIPROT 21,544 · VFDB 9,143 · VICTORS 1,752 · PATRIC 1,017 |
| **median length, positives** | **549** |
| **median length, negatives** | **268** |
| median length, labelled positives | 369 |

🔴 **The positives' median length is 2.05× the negatives'.** That is this paper's own
documented defect — a benign-side gradient that metadata alone can exploit — sitting in the
published benchmark. ⚠️ It is a *composition* fact, measured before freezing. What a
length-only classifier **scores** on it is an outcome and is predicted below, not peeked.

### The circularity price

**239** of this project's 746 class-axis representatives appear in DeepVIC — **32.0% of ours**
but only **0.71% of theirs**. 🔒 **Every reported figure is computed twice**: once on DeepVIC's
full set and once with all 239 shared VFG ids removed. A result that moves between those two
is reported as circular and nothing else.

## 3. The four tests

**AI-1 — the metadata-only baseline.** No embeddings. Two no-semantics classifiers on the
binary task: (a) **log length alone**, (b) log length **plus** the 20 amino-acid frequencies.
Our own 80:20 split at five seeds, because DeepVIC states the ratio but does not publish the
partition. Reported against DeepVIC's published **AUROC 0.954**.

**AI-2 — leave-one-category-out on their axis.** `src/81`'s pipeline over the 14 classes:
fit on the other thirteen plus train negatives, threshold at 95% specificity on calibrate
negatives, report per-class recovery and out-of-sample FPR. Primary: **Spearman(margin,
recovery)** over the 14 classes, and whether recovery is **bimodal** as it is on our panel.

**AI-3 — registration against hazard at 11× scale.** DeepVIC's **Exotoxin** (1,126) and
**Motility** (1,121) classes are within 0.5% of each other in size. 🔑 **This is the paper's
central narrowing, on someone else's labels, with the two groups naturally matched for n.**
Primary: recovery(Motility) − recovery(Exotoxin), with a protein-level bootstrap.

**AI-4 — the same four figures with the 239 shared representatives removed.**

## 4. Frozen bands

Floors **and** ceilings. Bonferroni α = 0.05/4 = 0.0125.

| | confound band | clean band | retraction |
|---|---|---|---|
| **AI-1a** log-length-only AUROC | **≥ 0.75** ⇒ the benchmark is substantially length-separable | **≤ 0.60** ⇒ length is not the story | — |
| **AI-1b** length + composition AUROC | **≥ 0.85** ⇒ no-semantics features approach the published 0.954 | ≤ 0.70 | — |
| **AI-2** Spearman(margin, recovery) | **≥ +0.60** at p < 0.0125 ⇒ replicates | **≤ +0.20** ⇒ fails to replicate | ≤ −0.40 ⇒ inverts |
| **AI-3** rec(Motility) − rec(Exotoxin) | **≥ 0** ⇒ replicates the narrowing | **≤ −20pp** ⇒ hazard-selective on their labels | — |

🔒 **A verdict requires both arms** — `esm2_650M` and `esm2_35M` — in the same band, for every
test that uses embeddings (AI-2, AI-3, AI-4). AI-1 uses no embeddings and has no second arm, so
it is reported as a **single-arm** figure whatever it shows.

⚠️ **AI-2's two smallest classes have 24 and 66 members.** The eligibility floor is **7**, as on
our own panel, and per-class recovery on a class of 24 is reported with its own interval, not
pooled.

## 5. Predictions, scored honestly

**P1 (derived — from the 549/268 median-length ratio in § 2).** AI-1a clears **0.75**.

**P2 (derived — from this project's own metadata baseline reaching AUROC 0.8096 with two
fields).** AI-1b clears **0.85**.

**P3 (derived — from ρ = +0.894 on our mechanism classes and +0.681 / +0.725 on VFDB's).**
AI-2 clears **+0.60** on both arms.

**P4 (derived — from Motility ranking 1st of 13 on 650M and 4th on 35M in study AG, and from
`C/A` parity across hazard groups).** AI-3 is **≥ 0** on both arms.

**P5 (reasoned).** Removing the 239 shared representatives (AI-4) moves no figure by more than
**2pp**, because they are 0.71% of DeepVIC's positives.

⚠️ **Four of these five are derived and one is reasoned.** This project's record is that derived
predictions hold and reasoned ones miss — ⚠️ **except in study AH, where it inverted**: two
derived missed and the reasoned one held. The record is not a law.

## 6. What would change the paper

- **AI-1a or AI-1b in the confound band** ⇒ 🔴 § 1.2's "the comparison we care about is not the
  AUROC" gains a measured reason: the published 0.954 is partly a length artifact, and the paper
  can say so with a number instead of a disclaimer.
- **AI-2 replicating** ⇒ the margin result stops being a property of this project's panel. It
  would be the first external, published-benchmark confirmation of the one finding that
  replicated.
- **AI-3 ≥ 0** ⇒ § 6's narrowing holds at 11× the exotoxin count, on curator labels from four
  databases, with naturally matched group sizes. 🔴 **AI-3 ≤ −20pp** ⇒ the narrowing is wrong at
  scale and § 6 must be rewritten.
- **AI-4 moving a figure** ⇒ the arm is circular and is withdrawn.

---

## Amendment 1 — the truncation asymmetry, and what AI-3 must report because of it

**Added 2026-10-01, after the embeddings were written and before `src/125` was run.** No recovery,
rho or AI-3 difference existed when this was written; the only new facts are sequence lengths, which
are composition.

ESM-2 truncates at **1,022** residues. `src/124`'s log reports **1,738** of 46,445 sequences
truncated. Decomposed:

| | truncated | of | rate | median length |
|---|---:|---:|---:|---:|
| labelled positives | 1,438 | 12,989 | **11.07%** | 369 |
| negatives | 300 | 33,456 | **0.90%** | 268 |

🔴 **A 12× asymmetry between the two sides of the binary task**, and it is worse along the axis AI-3
is built on:

| class | truncated | of | rate | median |
|---|---:|---:|---:|---:|
| **Exotoxin** | 248 | 1,126 | **22.02%** — the highest of all fourteen | 532 |
| Exoenzyme | 46 | 210 | 21.90% | 494 |
| Adherence | 482 | 2,259 | 21.34% | 420 |
| … | | | | |
| **Motility** | 17 | 1,121 | **1.52%** | 308 |
| Stress survival | 0 | 186 | 0.00% | 540 |

🔴 **AI-3 compares Motility against Exotoxin because their sizes match within 0.5%. Their truncation
rates differ by 14×.** If truncation costs recovery, the asymmetry pushes AI-3 **toward** `≥ 0`, which
is the direction § 5's P4 predicts. ⚠️ **A confirmation of P4 on the full membership would therefore
be uninterpretable on its own**, and that is stated here rather than discovered afterwards.

### What AI-3 now reports: three variants, not one

- **AI-3a — all members.** As frozen in § 3. 1,121 against 1,126.
- **AI-3b — untruncated only.** Members of length ≤ 1,022, so no sequence is cut. Sizes stop
  matching (about 1,104 against 878) and that is the price.
- **AI-3c — length-matched pairs.** Each Exotoxin member paired without replacement to the
  nearest-length Motility member, pairs above a 2.0 length ratio dropped and counted, as `src/121`
  does. 🔑 **This is the primary variant**, because it removes the length difference and the
  truncation difference together.

🔒 **The bands are unchanged** and apply to each variant: `≥ 0` replicates the narrowing, `≤ −20pp`
is hazard-selective on their labels, and a **verdict needs both arms in the same band**.

🔑 **The reading rule, frozen here:** AI-3 replicates § 6's narrowing only if **AI-3c** lands in the
replication band on both arms. If AI-3a is `≥ 0` while AI-3c is not, the result is reported as a
**truncation and length artifact** and § 6 gains nothing from it.

**P6 (derived — from the 22.02% against 1.52% rates above).** AI-3a exceeds AI-3c on both arms: the
full-membership difference is more favourable to the narrowing than the matched one.

⚠️ AI-2's Spearman is also exposed to this — truncation rate correlates with class median length, and
`src/125` already reports margin against the two controls it is meant to beat. The **truncation rate
per class** is added to the artifact so the correlation can be computed rather than assumed, and
Spearman(truncation rate, recovery) is reported alongside. No band is set for it; it is a diagnostic.
