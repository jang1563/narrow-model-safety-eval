# Is the arm disagreement a composition effect? — preregistration

**Frozen 2026-10-02, before `src/127` existed.** Study AJ. Nothing but the facts in § 1 was known
at freezing time; no composition recovery, correlation or Exotoxin figure had been computed.

## 1. The thing to explain

Study AI-3 left one unexplained disagreement, and it is the sharpest open question in the project:

| | 650M | 35M | swing |
|---|---:|---:|---:|
| **Exotoxin** recovery (n = 1,126) | 44.7% | **58.4%** | **−13.7pp** |
| Motility recovery (n = 1,121) | 47.6% | 43.3% | +4.2pp |
| mean over all 14 categories | 41.4% | 41.8% | −0.3pp |

🔒 **It is not a globally weaker arm** — the means differ by 0.3 points and seven of fourteen
categories come out higher on each. 🔒 **It is not seed instability** — Exotoxin's across-seed SD is
12.7 and 9.1 points with zero zero-recovery seeds. 🔒 **It is not length matching** — the
length-matched variant still gives +6.04pp against −5.30pp, with paired medians 404 against 405.

⚠️ **And § 6's narrowing is exactly the quantity that moves.** If how exotoxins sit relative to the
rest of a registered set depends on the representation, then "the probe responds to registration, not
hazard" is a statement about a representation and not about the task.

## 2. The hypothesis

**The 35M representation is closer to amino-acid composition than the 650M representation is, and
exotoxins are compositionally distinctive.**

Two facts already in the record make this worth testing rather than asserting:

- **AI-1**: log length plus the twenty amino-acid frequencies reach **AUROC 0.8705** on DeepVIC's
  binary task, **91.3%** of its published 0.954. Composition is a strong signal on this benchmark.
- **AI-2**: a class's truncation rate tracks its recovery at *ρ* = **+0.097** on 650M but **+0.365**
  on 35M. The smaller arm's recovery is already measurably more length-coupled.

## 3. The design

`src/125`'s fold, unchanged, with the embedding replaced by features that contain **no model at
all**: **log length plus the twenty amino-acid frequencies** — the same 21 numbers as AI-1b. Each of
the 14 categories held out in turn, fit on the other thirteen plus train negatives, threshold at the
95th percentile of the calibrate negatives, 30 seeds.

🔒 **The negative partition is reproduced bit-for-bit** from `src/125` — `default_rng(12345)` over the
manifest's negative block, two thirds as the union, the remainder as the fixed test set — so the
composition fold and the two embedding folds differ **only** in the features.

A second reference arm uses **log length alone**, one feature, to separate "composition" from "length".

## 4. Frozen bands

Floors **and** ceilings. Bonferroni α = 0.05/3.

| | supports the hypothesis | refutes it |
|---|---|---|
| **AJ-1** sign of rec(Motility) − rec(Exotoxin) under composition | **negative**, i.e. matches 35M | ≥ 0, i.e. matches 650M |
| **AJ-2** *ρ*(composition recovery, 35M recovery) − *ρ*(composition recovery, 650M recovery) | **≥ +0.15** | **≤ −0.15** |
| **AJ-3** Exotoxin's rank under composition, of 14 | **top 4** | outside the top 7 |

⚠️ **Between the two columns is "between bands" and is reported as not settling the question.**
AJ-1's sign is checked on the length-matched pairing as well as full membership, because that is
amendment 1's primary form.

🔴 **This study has no second arm and cannot have one**: there is only one composition of a sequence.
Its result is therefore **an explanation offered, not a verdict**, however it comes out, and the
paper must say so. What it can do is rule the hypothesis **out**.

## 5. Predictions

**P1 (derived — from AI-1b's 0.8705 and AI-2's +0.365 against +0.097).** AJ-1 is negative: under
composition alone, Exotoxin is recovered **more** than Motility, as on 35M.

**P2 (derived — same two facts).** AJ-2 is **≥ +0.15**: composition recovery resembles the 35M
ordering more than the 650M ordering.

**P3 (reasoned).** AJ-3 puts Exotoxin in the **top 4** of fourteen under composition, because
secreted toxins carry signal peptides and disulfide-rich cores that are compositionally marked.

⚠️ **This project's prediction record has just collapsed** — across studies AH and AI, derived
predictions held 2 of 5 and reasoned 1 of 2, after about ten derived holds in a row. The reason given
in entry 107 is that both studies left the panel the earlier derivations came from. 🔑 **P1 and P2
here are derived from measurements on *this* benchmark**, which is the population they are about, so
if the entry-107 explanation is right they should hold. **That is itself a test of the explanation.**

## 6. What each outcome changes

- **Hypothesis supported** ⇒ § 6 gains a mechanism: the between-category hazard ordering is
  representation-dependent **because** the smaller representation is closer to composition, and
  composition favours exotoxins. The narrowing's within-fold form (study AH) is unaffected.
- **Hypothesis refuted** ⇒ the disagreement stays unexplained and § 7 says so. 🔑 **A refutation is
  worth as much as a confirmation here**, because it removes the most obvious explanation and leaves
  the finding standing as a bare fact about two representations.
- **Between bands** ⇒ reported as not settling it, with the figures given.
