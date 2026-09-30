# Preregistration — is the length-U an artifact of where the bands were drawn?

**Frozen 2026-09-30, before any coefficient or perturbed band has been computed.** One primary, two
secondaries. Append-only.

---

## 0. The dependency nobody has tested

Studies M, N, O and P all measure the length effect through **fixed residue cuts** — 0–250, 250–350,
350–450, 450–550, 550–700, 700+ — and the U-index through **rate(0–250) / rate(450–550)**. 🔴 **Those
boundaries were chosen once, in study M, and reused in four studies without a sensitivity check.** The
step-by-step review listed it as the last untested item.

🔑 **Perturbing the boundaries is the weak version of this test.** The strong version removes them: a
U-shape in length is exactly a **positive quadratic coefficient** in a regression of the outcome on
`log L` and `(log L)²`, which has no bands in it at all. If that coefficient is positive on every arm,
the U is not a cut-point artifact, and if its magnitude falls with model size then study P's trend
survives without bands too.

---

## 1. Design

⚠️ **No new probe and no new inference.** Study G's clean fold, study L's strata, all four arms already
embedded, 30 seeds.

Within the **intracellular** stratum — the cleanest, where localization cannot be doing the work and
where studies O and P measured the U — regress per-protein outcomes on `x = log10(L)` centred, and
`x²`:

- **on the flag** (binary, as studies M–P used, so the result is comparable), and
- **on the standardized score** (continuous, threshold-free, so the cut at the 95th percentile cannot
  shape it).

🔒 Coefficients are averaged over the 30 seeds, with the seed-to-seed interval reported.

---

## 2. Primary test, frozen bands

**U-1**: the sign of the **quadratic coefficient on the score**, on each of the four arms.

| outcome | reading |
|---|---|
| **positive on all four arms** | 🟢 the U is not a cut-point artifact |
| positive on 650M but not all | ⚠️ partial — it holds where it was measured, not generally |
| **not positive on 650M** | 🔴 the U as reported is an artifact of the bands, and studies M, N, O and P need restating |

**U-2**: Spearman ρ between log parameter count and the quadratic coefficient's magnitude, four arms —
study P's trend, band-free.

| band | reading |
|---|---|
| **ρ = −1.0** | 🟢 study P's size trend survives without bands |
| ρ ≤ −0.6 | ⚠️ consistent, not strict |
| \|ρ\| < 0.6 | 🔴 the trend was a band artifact |

🔒 As in study P, **n = 4 gives a best one-tailed p of 1/24 = 0.042** and this is suggestive by
construction.

## 3. Secondary — the weak version, run anyway

**U-3**: the U-index recomputed with the short/mid boundaries shifted by **±50** and **±100** residues,
and with **quantile** bands (lowest and middle fifths) replacing residue cuts. 🔒 The **range** across
all of those is reported, not the best one.

## 4. Frozen predictions

- **U-1: positive on all four arms.** The band table already shows the rate falling then rising inside
  the intracellular stratum on both arms measured, and a monotone summary of it (AUROC 0.437) failed
  precisely because the relationship is not monotone — which is a positive quadratic by another name.
- **U-2: ρ = −1.0**, matching study P.
- **U-3: the U-index varies but stays above 3.0 on 650M under every perturbation.**
- ⚠️ 🔒 **Checkable against a wrong answer**: if the quadratic coefficient is identical on two arms to
  three significant figures, the arms are being fitted on the same embedding and that is a bug.

## 5. Floors

🔒 The intracellular stratum holds 2,739 proteins on every arm, so no cell floor applies to U-1 or U-2.
For U-3, a perturbed band holding fewer than **100** proteins is reported but excluded from the range.

---

## Results, 2026-09-30

`src/104_length_band_sensitivity.py`, study G's clean fold, study L's strata, four arms, 30 seeds,
2,739 intracellular proteins on every arm.

| arm | dim | quadratic on the **score** | quadratic on the **flag** | U-index as published |
|---|---:|---:|---:|---:|
| esm2_8M | 320 | **+5.499** | **+0.284** | 54.38 |
| esm2_35M | 480 | **+5.622** | **+0.212** | 24.54 |
| esm2_150M | 640 | **+3.884** | **+0.244** | 11.66 |
| esm2_650M | 1280 | **+10.463** | **+0.329** | 8.65 |

### 🟢 U-1: the U-shape is not a cut-point artifact

**The quadratic coefficient is positive on all four arms**, on the score and on the flag alike. 🔒 **The
curvature is in the data, not in the boundaries** — studies M, N and O do not need restating on this
count.

### 🔴 U-2: study P's size trend does **not** survive band removal

**ρ = +0.40** against log parameters — on **both** measures, so the concern that score scales differ by
dimension (√dim runs 17.9 → 35.8) does not change it. The frozen reading for |ρ| < 0.6 is *"the trend
was a band artifact"*, and 🔒 **the obligation is executed here.**

| | 8M | 35M | 150M | 650M | ρ |
|---|---:|---:|---:|---:|---:|
| band-based **U-index** (study P) | 44.66 | 18.68 | 9.65 | 8.38 | **−1.00** |
| band-free **curvature** (flag) | +0.284 | +0.212 | +0.244 | +0.329 | **+0.40** |

🔑 **And the two are reconciled by study P's own observation.** Study P found the U-index falls because
the **trough rises** — the mid-band rate goes 0.36% → 1.69%, a factor of 4.7 — while the short-band rate
stays flat. **A ratio whose denominator grows shrinks even when the shape does not change.** 🔒 **So
"the length effect is a size trend" is wrong as stated; the correct statement is that the U-INDEX falls
with model size because its denominator rises, and the curvature does not fall.**

### 🔴 U-3: the U-index itself is highly band-sensitive

650M, across the six frozen grids:

| grid | U-index |
|---|---:|
| as published | 8.65 |
| shift −100 | n/a — a band fell under the 100-protein floor |
| shift −50 | 7.32 |
| shift +50 | 7.60 |
| ⭐ **shift +100** | **2.32** |
| quintiles | 6.29 |

**Range [2.32, 8.65]** — a factor of 3.7 on the choice of boundary. 🔴 **At shift +100 it falls below
3.0**, the level study E's bands treat as the line between "a major driver" and "partial" for a
different quantity. ⚠️ **The U-index must be quoted with that range attached from here on**, and § 4's
prediction that it would stay above 3.0 under every perturbation is wrong.

### 🔒 Scoring the frozen predictions

| | frozen | actual | |
|---|---|---|---|
| U-1 positive on all four arms | | all positive, both measures | ✅ |
| U-2 ρ = −1.0 | | **+0.40** | ❌ wrong **sign** |
| U-3 stays above 3.0 everywhere | | **2.32** at shift +100 | ❌ |
| two arms sharing a coefficient ⇒ a bug | | all four differ | ✅ guard silent |

⚠️ **One of three substantive predictions held.** Both misses were on quantities I reasoned about
qualitatively — "the trend should survive", "the index should be robust" — rather than computing. The
one that held was the one derived from a property already in the record: a non-monotone relationship
that defeated an AUROC is a positive quadratic by definition.
