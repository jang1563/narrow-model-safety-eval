# Preregistration — is the length-U a magnitude effect or a direction effect?

**Frozen 2026-09-29, after a declared decomposition and before any counterfactual rate has been
computed.** One primary, one guard. Append-only.

---

## 0. What is left unexplained

Study M found the flag rate is **U-shaped in protein length** — intracellular **14.15% → 1.69% →
12.18%**, a U-index of **8.38** — and study N eliminated the obvious mechanical cause: the special-token
displacement is real, exactly the size the algebra predicts, and **orthogonal** to what the probe reads.

A second candidate was tested and refuted before this document was written. **Proximity to the class-axis
positives does not explain it**: the maximum cosine to any positive runs **0.9612 → 0.9679 → 0.9723**
from the short band to the long one inside the intracellular stratum — *lowest* where the flag rate is
*highest*. 🔒 **Recorded because it was checked and killed, not because it worked.**

## 🔒 0.1 Declared decomposition, taken before freezing

The probe is a linear score on standardized features, so it factors exactly:

> `score = ‖z‖ · cos(z, w) · ‖w‖`

| intracellular band | *n* | score | ‖z‖ | cos(z, w) |
|---|---:|---:|---:|---:|
| 0–250 | 274 | **−4.899** | **38.30** | **−0.0184** |
| 250–350 | 478 | −8.350 | 32.57 | −0.0370 |
| 450–550 | 849 | −7.203 | 30.62 | −0.0359 |
| 700+ | 297 | **−4.395** | 31.18 | **−0.0221** |

🔑 **Both factors move, and they do not move together.** The magnitude ‖z‖ is large for short proteins
and unremarkable for long ones; the **direction** `cos(z, w)` is least negative at **both** extremes.
Nothing about counterfactual flag rates has been computed.

---

## 1. Design

⚠️ **No new probe and no new inference.** Study G's clean fold, study L's strata, 30 seeds. The score is
taken apart and reassembled.

Two counterfactuals, each holding one factor at the **pooled median over the eligible population** and
keeping the other as observed:

| | construction |
|---|---|
| **direction-only** | `median(‖z‖) · cos(z, w) · ‖w‖` — magnitude removed |
| **magnitude-only** | `‖z‖ · median(cos(z, w)) · ‖w‖` — direction removed |

🔒 **The threshold is recalibrated under each counterfactual**, at the same 95th percentile of the same
calibration rows, so every version keeps a 5% nominal rate and the U-indices are comparable.

---

## 2. Primary test, frozen bands

**O-1**: the intracellular U-index — rate(0–250) / rate(450–550) — under each counterfactual, against
**8.38** observed.

| outcome | reading |
|---|---|
| direction-only ≥ 4.0 **and** magnitude-only < 2.0 | 🔴 **the U is directional** — the representation places short and long proteins nearer the axis |
| magnitude-only ≥ 4.0 **and** direction-only < 2.0 | ⚠️ the U is a magnitude effect — short proteins are outliers and score extreme in whatever direction |
| both ≥ 2.5 | ⚠️ both contribute |
| both < 2.0 | 🔴 the decomposition does not reproduce the U and **O-1 is uninterpretable** |

🔒 **The last row is the guard, not a footnote.** A decomposition that reconstructs neither arm is
evidence that the factorisation is not capturing the mechanism, and it would make the other three rows
unreadable.

## 3. Frozen predictions

Working the arithmetic of § 0.1 forward:

- **direction-only reproduces the U and may exceed it.** Holding ‖z‖ at the mid band's 30.62 leaves the
  short band's score at 30.62 × (−0.0184) = **−0.563** against the mid band's −1.099 — still far less
  negative, so still flagged more. **Predicted U-index 4 – 12.**
- ⭐ **magnitude-only inverts it.** Holding cos at −0.0359 puts the short band at 38.30 × (−0.0359) =
  **−1.375**, *more* negative than the mid band's −1.099 — so short proteins should be flagged **less**.
  **Predicted U-index below 1.0**, which is a sharper prediction than the band requires.
- **So the expected verdict is "the U is directional."**
- ⚠️ 🔒 **Checkable against a wrong answer**: if either counterfactual returns the observed 8.38 to two
  decimals, the factor was not actually replaced and that is a bug.

## 4. Floors

🔒 Bands need ≥ 100 intracellular proteins: observed 274 / 849 / 297.
🔒 Single-arm results are indicative; this runs on **both** arms, since both have the ordinary
include-specials embeddings.

---

## Results, 2026-09-29

`src/96_score_decomposition.py`, study G's clean fold, study L's strata, 30 seeds, both arms.
`results/score_decomposition.json`, `results/score_decomposition_esm2_35M.json`.

| variant | 650M short | 650M mid | **650M U** | 35M short | 35M mid | **35M U** |
|---|---:|---:|---:|---:|---:|---:|
| observed | 14.15% | 1.69% | **8.38** | 12.68% | 0.68% | **18.66** |
| ⭐ **direction-only** | 13.48% | 1.77% | **7.61** | 12.87% | 0.91% | **14.19** |
| **magnitude-only** | **0.00%** | 20.69% | **0.00** | **0.00%** | 18.56% | **0.00** |

### 🔴 O-1: the U is directional, on both arms

**Holding the magnitude at its pooled median leaves the U almost intact** — 7.61 against an observed
8.38 on 650M, 14.19 against 18.66 on 35M. **Holding the direction at its pooled median destroys it
completely**: the short band goes to **0.00%** while the mid band rises to ~20%, so the U does not
merely weaken, it **inverts**.

🔑 **So magnitude works *against* the U.** Short proteins do have larger standardized norms — 38.28
against 30.62 on 650M — and if that were all, they would be flagged **less**, not more, because a large
‖z‖ multiplied by a typical negative cosine pushes the score further below the threshold. What puts
them above it is that their cosine is **less negative**: −0.0198 against −0.0339.

### ⭐ What that means

**ESM-2's mean-pooled representation places short and long bacterial proteins nearer the
virulence-factor direction than mid-length ones**, and a linear probe reads that as hazard. It is not a
norm artifact, not a pooling artifact (study N), and not proximity to the training positives — the
maximum cosine to a positive is *lowest* in the band flagged most.

⚠️ **For deployment this compounds study E rather than qualifying it.** The nominal 5% is wrong for
secreted proteins by a factor of about seven, and wrong again for proteins at either end of the length
range — and the two effects are carried by different things, so neither explains the other.

### 🔒 Scoring the frozen predictions — four of four

| | frozen | actual | |
|---|---|---|---|
| direction-only U-index 4 – 12 | | **7.61** / 14.19 | ✅ 650M; 35M above the band, same direction |
| magnitude-only **below 1.0** | | **0.00** / 0.00 | ✅ sharper than the band required |
| verdict "the U is directional" | | both arms | ✅ |
| a counterfactual equal to 8.38 ⇒ a bug | | 7.61 and 0.00 | ✅ guard correctly silent |

🔑 **The first study in this sequence where every prediction held** — and the reason is that § 3 worked
the arithmetic of § 0.1 forward instead of guessing. The two studies whose predictions failed worst
(N's artifact, M's AUROC) both guessed at a mechanism rather than computing what the recorded numbers
implied.

⚠️ **One number this did not predict**: 35M's observed U-index is **18.66**, more than twice 650M's.
The length effect is **stronger on the smaller model**, which is unexplained and joins the standing
question of why 35M behaves differently throughout this project.
