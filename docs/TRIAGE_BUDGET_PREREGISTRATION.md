# Does margin triage an evaluation budget? — preregistration

**Frozen 2026-10-08, before `src/146` existed.** Study AX.

## 1. The claim the paper actually makes

§ 4 states margin is **"a triage signal for deciding which mechanism families need their own validation,
and not as a deployment score."** 🔴 **That has never been tested.** Study AW tested the other use —
building a mixture — and found no advantage. 🔑 **This tests the one the paper asserts.**

⚠️ **It is not redundant with *ρ*.** A rank correlation of +0.6 says the ordering is better than chance;
it does not say whether picking the **two** worst classes out of fourteen works. 🔑 **The decision a
reader faces is a budget**, and the decision-relevant number is recall at that budget, not a correlation.

## 2. The measurement

🔒 **No fitting.** Every recovery and every margin already exists in the artifacts of studies AT, AU and
AV: **24 cell-settings** over **3 modalities** and **8 frozen encoders**.

For a cell with *k* held-out classes and a budget *m*:

- **margin picks** the *m* lowest-margin classes;
- **truth** is the *m* lowest-**recovery** classes;
- **recall@m** = |picked ∩ truth| / *m*;
- **chance** = *m*/*k*, and the exact null is the hypergeometric distribution.

🔒 Reported at **m = 1, 2, 3** and at **m = ⌈k/5⌉** (a 20% budget), per cell-setting and pooled.
🔒 **Also reported: the mean recovery of the picked classes against the mean of *m* at random** — the
quantity a reader spending the budget actually experiences.

## 3. Frozen bands

Bonferroni α = 0.05/2.

| | triage works | it does not |
|---|---|---|
| **AX-1** pooled recall@2 over the 24 cell-settings | **≥ 2× chance** | **≤ 1.2× chance** |
| **AX-2** cell-settings where recall@2 beats chance | **≥ 18 of 24** | **≤ 12 of 24** |

🔒 **Bands are RATES with their denominators named**, and 🔒 **checked with
`src/143_band_reachability.py` before freezing** — the five-band lesson of entries 120–127. ⚠️ **AX-1's
2× is against a chance that varies by *k*** (2/14 on protein, 2/13 on text, 2/10 and 2/50 on images), so
the pooled figure is a mean of per-cell ratios and not a ratio of pooled counts; **stated because the two
differ.**

## 4. Predictions

**P1 (derived — *ρ* runs +0.52 to +0.95 across these cells, and a positive rank correlation must lift
recall at a small budget).** AX-1 clears **2× chance**.

**P2 (derived — from study AU: the image cells' recoveries are compressed, sd 5–20pp against 12–19pp
elsewhere, so their ordering carries less signal per rank).** The **image** cell-settings have the
**lowest** mean recall@2 of the three modalities.

**P3 (reasoned).** Recall@1 is **worse** than recall@2 relative to chance, because the single
lowest-margin class is the noisiest point of the ordering and § 4 already reports margin over-predicting
beta-lactamase recovery by 34 points.

⚠️ **The record stands at derived 19 of 37, reasoned 4 of 14.**

## 5. What each outcome changes

- **AX-1 clears** ⇒ 🔑 **§ 4's triage claim is tested rather than asserted**, and the paper can state a
  budget number a reader can act on instead of a correlation.
- **AX-1 fails** ⇒ 🔴 **the triage framing goes too.** With AW showing no advantage in building and this
  showing none in triaging, margin would be a measurement with no demonstrated use — and § 4 would have
  to say so.
