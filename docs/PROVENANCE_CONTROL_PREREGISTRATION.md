# Preregistration — is it virulence, or is it just pathogen origin?

**Frozen 2026-09-28, before the probe has been applied to this set.** One primary test. Append-only.

---

## 0. The caveat this exists to close

Every write-up of studies B and C carries the same sentence: the ordering result *"says nothing about
the absolute recoveries, which remain compatible with pathogen-versus-Swiss-Prot separation."* A1
sharpened it — the same probe flags **40.84%** of non-toxin virulence factors at a nominal 5%, against
**5.98%** on the Swiss-Prot benign pool — and offered two readings without deciding between them:

1. the probe responds to **virulence**, and virulence factors are legitimately flagged;
2. the probe responds to **pathogen origin**, and "virulence factor" is incidental.

🔑 **The control that separates them is pathogen-derived proteins that are not virulence factors**, and
the repository already has them: **106** proteins from the v3 negative set drawn from **20 species that
also contribute study B representatives**, none of them matching a VFDB setA or setB sequence. (Four
more did match and are excluded — they were positives sitting in a negative set.)

---

## 1. Design

**No new probe.** Study B's fold is used unchanged: fit on the VFDB representatives plus the pool
training negatives, threshold at 95% specificity on the pool calibration negatives, 30 seeds, both arms.
The only new thing is a third population scored at that threshold.

| population | *n* | what it is |
|---|---:|---|
| Swiss-Prot benign pool, test partition | 6,758 | the easy negatives, never fitted or calibrated on |
| **pathogen-derived, not in VFDB** | **106** | ⭐ the control: same 20 species as the positives |
| VFDB non-toxin virulence factors (A1) | 4,218 | pathogen-derived **and** virulence-annotated |

All three are scored by the same models at the same thresholds, so the only thing that differs between
them is what the proteins are.

---

## 2. Primary test and predictions

One primary test, **α = 0.05**.

| | test | reading |
|---|---|---|
| **D-1** | flag rate on the 106 matched pathogen non-VF proteins, against the pool's rate and A1's rate | see the three outcomes below |

🔒 **The three outcomes and what each licenses, fixed now:**

- **Near A1's 40.8%** → the probe responds to **pathogen origin**. Virulence annotation adds little, A1's
  result is a provenance result, and every absolute recovery in studies B and C should be read as
  pathogen-versus-Swiss-Prot separation. 🔴 **The ordering results survive** — C-1 held organism constant
  — **but the claim that the probe detects anything hazard-specific does not.**
- **Near the pool's 6.0%** → the probe responds to something **virulence-specific**. A1's 40.8% is then a
  fact about virulence factors rather than about their organisms, and the standing caveat in studies B
  and C can be narrowed.
- **Between** → partial, and the split is reported as a proportion rather than argued about.

🔒 **Prediction.** § 2.3's provenance probe reaches AUROC **0.818** on the panel using lab-strain
provenance with the hazard label ignored, and these 106 were curated as *hard* negatives for the panel's
toxins rather than sampled at random from pathogen proteomes — both push the rate up. **Predicted: 20%
to 45%, i.e. much closer to A1's 40.8% than to the pool's 6.0%.**

⚠️ **The known weakness of this control, stated in advance.** These 106 are the panel's own curated
negatives, chosen to be *difficult*. A random sample of pathogen proteomes would be an easier control and
would give a lower rate. **So a high result here is an upper bound on the provenance effect, not a point
estimate of it**, and the follow-up if D-1 lands high is a random pathogen-proteome sample — which needs
a fetch this study deliberately does not do.

---

## Amendment 1 — 2026-09-28, on running it: § 2's anchors named the wrong probe

§ 2 defined D-1's three outcomes against **A1's 40.8%**. 🔴 **That figure is the *panel* probe's rate on
VFDB proteins; this study uses *study B's* probe, which is trained on VFDB representatives.** For study
B's probe a VFDB virulence factor is a near-positive, not a false positive, and it returns **73.49%** —
a recall figure, not comparable to A1's 40.8%.

**The anchors that are correct for this probe, and both are measured here**: its own false-positive rate
on the Swiss-Prot pool, and its own rate on VFDB virulence factors. D-1 is then reported as **where the
control sits between them**, which is the "between → report as a proportion" branch § 2 already
specified. The three-outcome framing stands; only the numbers anchoring it were wrong, and they were
wrong because the preregistration was written before the script named which probe it would use.

## Results, 2026-09-28

`src/83_provenance_control.py`. Study B's fold unchanged, 30 seeds, three populations scored by the same
models at the same thresholds.

| population | *n* | ESM-2 **650M** | ESM-2 **35M** |
|---|---:|---:|---:|
| Swiss-Prot pool, test partition | 6,758 | 6.98% [6.45, 7.50] | 4.98% [4.67, 5.28] |
| ⭐ **pathogen-derived, not in VFDB** | **106** | **22.08% [20.98, 23.17]** | **19.06% [17.93, 20.18]** |
| VFDB virulence factors (near-positives) | 4,218 | 73.49% [71.65, 75.32] | — |

### D-1: both readings in § 2 are refuted, and the answer is the proportion

🔴 **Pathogen origin alone is worth a 3.2× elevation** — 6.98% → 22.08% on 650M, 4.98% → 19.06% on 35M
(3.8×), replicating across arms. **Provenance is real and substantial.**

🟢 **And it does not explain most of the separation.** On the arm where all three are measured, the
control sits **22.7%** of the way from the pool's rate to the virulence factors' rate. **Roughly three
quarters of the benign-to-virulence gap is not accounted for by pathogen origin.**

🔒 **Scoring the prediction.** § 2 predicted **20% to 45%**, and said the result would be "much closer to
A1's 40.8% than to the pool's 6.0%". **The band was right — 22.08% — and the characterization was
wrong**: it lands near the bottom of the band, roughly midway between the two anchors rather than close
to the upper one.

### What this licenses, and the bound that makes it stronger

The standing caveat in studies B and C — *"the absolute recoveries remain compatible with
pathogen-versus-Swiss-Prot separation"* — is **narrowed, not removed**. Pathogen origin contributes about
a fifth to a quarter of the effect and the rest does not have that explanation.

🔑 **And the direction of the known weakness helps.** § 2 recorded in advance that these 106 are the
panel's own **curated hard** negatives rather than a random pathogen-proteome sample, so 22.08% is an
**upper bound** on the provenance effect. A random sample would be easier and score lower. **Even at an
upper bound, provenance explains under a quarter.**

⚠️ **What is still not established**: that the remaining three quarters is *hazard* rather than some
other property virulence factors share — secretion, surface exposure, host interaction. Separating those
needs negatives matched on **localization** as well as organism, which this study does not have.

---

## Addendum — 2026-09-28, later the same day: the localization half of that caveat has now been measured

`docs/LOCALIZATION_CONTROL_PREREGISTRATION.md` ran the missing control on the benign side, and it
returned an **adverse verdict on both arms**: benign *extracellular* bacterial and archaeal proteins are
flagged at **21.73%** against **4.15%** for cytoplasmic ones, a **5.36×** ratio that replicates at 5.13×
on ESM-2 35M and survives length matching.

🔑 **The share arithmetic lands in almost exactly the same place this study did**: localization spans
**22.1%** of the pool→VFDB gap in this document's own formula, against provenance's **22.7%**.

⚠️ 🔴 **The two shares must not be added.** Pathogen-derived proteins are themselves enriched for
secretion, so provenance and localization are expected to overlap and the joint decomposition has not
been run. **"About 45% explained" is not a result this repository has, and is the obvious way for these
two numbers to be misread together.**

---

## 🔒 Superseded on this point — 2026-09-28, later the same day

The sentence above saying the joint decomposition **has not been run** was true when written and is
no longer. `docs/JOINT_DECOMPOSITION_PREREGISTRATION.md` ran it: provenance and localization are
**multiplicatively independent** (RR = 1.001 on 650M, 1.364 on 35M, both inside the frozen band), and
**together they span 42.1% of the pool→VFDB gap on 650M and 32.2% on 35M** — measured, not summed.
🔴 **22.7% + 22.1% remains wrong** and remains forbidden by claim 98. The text above is left as
written rather than edited, per this repository's append-only rule.

🔒 **Figures above superseded 2026-09-28 by entry 55** — `src/84` and `src/85` did not use `src/83`'s
fold, which falls back to 978/490 because the screen admits only 1,468 rows. Every number moved in the
third decimal and **no verdict changed**; the corrected tables are in the two preregistrations'
amendments.
