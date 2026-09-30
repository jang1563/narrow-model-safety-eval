# Preregistration — are the arm differences a function of model size?

**Frozen 2026-09-29, while the 8M and 150M embeddings are still running and before any quantity has
been computed on them.** One primary over two outcomes, one post-hoc gate. Append-only.

---

## 0. What has accumulated

ESM-2 35M has behaved differently from 650M in the same direction in study after study, and the
differences are not small:

| quantity | ESM-2 650M | ESM-2 35M | |
|---|---:|---:|---|
| length U-index, intracellular (study O) | 8.38 | **18.66** | 2.2× stronger on the small arm |
| localization `R` (study L) | 5.18 | **5.86** | stronger |
| matched-VFDB `C/A` (study K) | 2.418 | **2.617** | stronger |
| length-matched ratio (study L) | 2.62 | **3.55** | stronger |

🔑 **Every one of them points the same way: the coarse effects are stronger on the smaller model.**
⚠️ **Two points cannot tell a size trend from an arm quirk**, and every one of those studies recorded the
difference as unexplained. This adds two more arms.

## 🔒 0.1 What is being built, and what it costs

The pool and the class-axis positives existed at **two** sizes. `src/35 --embed` and `src/80` are
extended to **esm2_8M** and **esm2_150M**, giving four arms spanning **8M → 650M, about 80× in
parameters**. 🔒 **esm2_3B is deliberately excluded**: an 11 GB model over 8,259 proteins on this
machine is hours, and four points already decide monotonicity.

---

## 1. Design

⚠️ **Nothing changes but the embedding.** The same pool rows, the same study-L strata, the same
study-G clean fold, the same 30 seeds, the same length bands. 🔑 **Cell sizes are therefore identical
at every arm** — 274 short, 849 mid, 657 extracellular, 2,739 intracellular — which is a strength: no
arm can differ because its population differed.

## 🔒 1.1 The gate this needs, and why it is post-hoc

`src/80` gates each arm against the published panel-negative array for that arm. 🔴 **`src/35`'s pool
embedding has no gate of its own** — it calls `src/02b`'s `embed` directly, which is the function that
produced the published arrays, but nothing checks that.

🔒 **So the study re-embeds the 296 panel negatives through `src/35`'s path at each new arm and compares
against `embeddings_negative_v3_{arm}.npy`, tolerance 5e-5.** Post-hoc rather than in-line, because the
pool run was already under way when this was written and re-running it to add an in-line gate would
have cost an hour for the same evidence. **If an arm fails that gate, it is dropped and the curve is
reported on the arms that pass.**

---

## 2. Primary test, frozen bands

**P-1**: Spearman ρ between **log parameter count** and each of the two outcomes — the intracellular
**U-index** and the localization **`R`** — across the four arms.

| band | reading |
|---|---|
| **ρ = −1.0** | 🔴 strictly monotone: the effect is a size trend |
| **−1.0 < ρ ≤ −0.6** | ⚠️ consistent with a size trend, not strict |
| **\|ρ\| < 0.6** | 🟢 no size trend — the 35M/650M gap is not about size |
| **ρ ≥ +0.6** | 🔴 the opposite trend, and every earlier reading of these arm gaps is backwards |

🔒 **What n = 4 can and cannot do, stated before the result**: with four points the only Spearman values
available are 0, ±0.2, ±0.4, ±0.6, ±0.8, ±1.0, and the best achievable one-tailed permutation *p* is
**1/24 = 0.042**. **This study is suggestive by construction and cannot be otherwise**; it is worth
running because four aligned points are a much better reason to believe a trend than two, not because
it will reach significance.

## 3. Frozen predictions

- **P-1 on the U-index: ρ = −1.0.** Predicted values, interpolating the two known arms: 8M ≈ 25,
  150M ≈ 12, against 35M's 18.66 and 650M's 8.38.
- **P-1 on `R`: ρ = −1.0**, but with a much smaller spread — 8M ≈ 6.2, 150M ≈ 5.5, against 5.86 and 5.18.
- ⚠️ **The U-index is the outcome I expect to behave; `R` is the one I expect to be noisy**, because
  its 650M–35M gap is 0.68 against the U-index's 10.3.
- ⚠️ 🔒 **Checkable against a wrong answer**: if any arm returns a U-index within 0.01 of another arm's,
  the two arms are being scored on the same embedding and that is a bug.

## 4. Floors

🔒 Each arm must pass § 1.1's gate at 5e-5 or be dropped. With fewer than **three** surviving arms,
P-1 is not computed at all.
🔒 The frozen band verdict is reported for **both** outcomes separately; a trend in one and not the
other is the likely result and is not to be presented as a trend in general.

---

## Results, 2026-09-29

`src/97_size_scaling.py`, four arms, study G's clean fold, study L's strata, 30 seeds.
`results/size_scaling.json`. 🔒 **Both new arms passed § 1.1's post-hoc gate** — the 296 panel
negatives re-embedded through `src/35`'s path match the published arrays at **0.000e+00** (8M) and
**1.907e-06** (150M); `src/80`'s own gate on the positives path passed at 1.907e-06 for both. Cells are
identical at every arm: 274 / 849 / 657 / 2,739.

| arm | dim | short | mid | **U-index** | **`R`** |
|---|---:|---:|---:|---:|---:|
| **esm2_8M** | 320 | 16.13% | **0.36%** | **44.66** | 6.93 |
| esm2_35M | 480 | 12.69% | 0.68% | 18.68 | 5.67 |
| esm2_150M | 640 | 14.51% | 1.50% | 9.65 | 4.50 |
| esm2_650M | 1280 | 14.15% | **1.69%** | **8.38** | 5.15 |

### 🔴 P-1 on the U-index: ρ = −1.00, strictly monotone

**44.66 → 18.68 → 9.65 → 8.38** over 80× in parameters, without a single inversion. One-tailed
permutation *p* = **0.0417**, which is **the smallest value n = 4 can produce** — exactly as § 2 said
in advance. 🔒 **The length effect is a size trend**: the smaller the model, the more strongly its
representation separates short proteins from mid-length ones along the virulence axis.

### ⭐ And it closes from below, which is not what "the effect weakens" would look like

| | 8M | 650M | |
|---|---:|---:|---|
| short-band rate | 16.13% | 14.15% | **flat** — range 12.69–16.13 across all four arms, no trend |
| **mid-band rate** | **0.36%** | **1.69%** | ⭐ **strictly increasing, 4.7×** |

🔑 **The U-index falls because bigger models flag mid-length proteins *more*, not because they flag
short ones less.** The short band is where the small models and the large ones agree; what scale buys
is a *higher* false-positive rate in the middle of the length range. ⚠️ **That is a worse reading for
deployment than "the artifact goes away"**, and it is the opposite of what a "larger models are
cleaner" story would predict.

### ⚠️ P-1 on `R`: ρ = −0.80, and 650M breaks the order

6.93 → 5.67 → 4.50 → **5.15**: the 650M arm sits *above* 150M. Consistent with a size trend, not
strict, *p* = 0.1667. 🔒 **§ 3 predicted `R` would be the noisy outcome and said why** — its 650M–35M
gap was 0.68 against the U-index's 10.3 — so this is the outcome the document expected to misbehave,
and the two are reported separately exactly as § 4 required.

### 🔒 Scoring the frozen predictions

| | frozen | actual | |
|---|---|---|---|
| U-index ρ = −1.0 | | **−1.00** | ✅ |
| U-index 8M ≈ 25, 150M ≈ 12 | | **44.66**, 9.65 | ⚠️ 150M close, 8M badly under-predicted |
| `R` ρ = −1.0 | | −0.80 | ❌ |
| `R` 8M ≈ 6.2, 150M ≈ 5.5 | | 6.93, **4.50** | ❌ both outside |
| "`R` is the one I expect to be noisy" | | it was | ✅ |
| two arms with the same U-index ⇒ a bug | | 44.66 / 18.68 / 9.65 / 8.38 | ✅ guard silent |

⚠️ **The interpolation was wrong because it assumed the curve was roughly linear in log-parameters, and
it is not** — the 8M→35M step drops the U-index by 26 points and the 150M→650M step by 1.3.

### 🔑 What this does and does not settle

🟢 **It settles that the arm gaps recorded as unexplained in studies E, K, L and O are a size trend on
the length outcome**, and not an ESM-2-35M quirk.
⚠️ **It does not settle *why*.** A monotone relationship over four points with *p* = 0.042 is what this
design can deliver and no more, as § 2 recorded before the data existed.
🔴 **And it does not license "use a bigger model".** The quantity that improves with scale is a ratio;
the *rate* that scale changes is the mid-band false-positive rate, and it goes **up**.

---

## 🔴 Superseded on what the trend is in — 2026-09-30, entry 74

This document reads its result as **"the length effect is a size trend"**.
`docs/LENGTH_BAND_SENSITIVITY_PREREGISTRATION.md` removed the bands entirely — regressing the outcome on
`log L` and `(log L)²`, which contains no boundaries — and **the trend does not survive**:

| | 8M | 35M | 150M | 650M | ρ |
|---|---:|---:|---:|---:|---:|
| band-based **U-index** (this document) | 44.66 | 18.68 | 9.65 | 8.38 | **−1.00** |
| band-free **curvature** | +0.284 | +0.212 | +0.244 | +0.329 | **+0.40** |

🔑 **And this document's own observation reconciles them.** It found the U-index falls because the
**trough rises** — the mid-band rate 0.36% → 1.69%, a factor of 4.7 — while the short band stays flat.
**A ratio whose denominator grows shrinks even when the shape does not.**

🔒 **So the correct statement is narrower than the one above**: the **U-index** falls with model size
because its denominator rises; the **curvature** does not fall, and is positive on all four arms. The
U-shape is real and band-free; **its trend with scale is a property of the statistic, not of the shape.**

⚠️ **And the U-index is band-sensitive**: across six boundary grids on 650M it ranges **2.32 to 8.65**,
falling below 3.0 at one of them. Every quotation of it carries that range from here on.
