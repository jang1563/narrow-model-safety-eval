# Preregistration — is the short arm of the length-U a special-token artifact?

**Frozen 2026-09-29, after a declared geometric measurement and before the probe has been refitted
under residue-only pooling.** One primary, two guards. Append-only.

---

## 0. What study M found and could not explain

Study M found the flag rate is **U-shaped in protein length** inside both strata — intracellular
**14.15% → 1.69% → 12.18%** from the shortest band to the longest. 🔑 **The short arm has a mechanical
candidate.** Every embedding in this project is `src/02b`'s mean over the **full attention mask**, so
`<cls>` and `<eos>` are averaged in, and for a short protein they are a large share of the mean.

## 🔒 0.1 Declared geometric measurement, taken before freezing

The algebra is exact:

> `m_full − m_res = [(cls + eos) − 2·m_res] / (L + 2)`

so the displacement should fall as `1/(L+2)`. The pool's two published arrays — include-specials and
residue-only, 8,259 rows each — say it does:

| | |
|---|---|
| correlation of ‖m_full − m_res‖ with 1/(L+2) | **r = 0.977** |
| ‖m_full − m_res‖ × (L+2), median | **9.51**, CV **0.105** |
| mean displacement, proteins under 250 | **0.0555** |
| mean displacement, proteins over 700 | **0.0113** — ⭐ **5× smaller** |

🔒 **So the mechanism exists and is exactly the size the algebra predicts.** What is not established is
whether it **drives** the U in the flag rate. Nothing about the probe under residue-only pooling has
been looked at.

## 🔴 0.2 This study has one arm, and is therefore indicative

The pool's residue-only array exists on **ESM-2 650M only**; 35M has none, and the 746 class-axis
positives had none on either arm until `src/94` wrote the 650M one. 🔒 **By this project's standing
rule a single arm is not a verdict**, and no outcome here changes a claim on its own. A second arm
would need the 8,259-protein pool re-embedded at 35M with residue-only pooling.

---

## 1. Design

⚠️ **No new probe design.** Study G's clean fold, study L's strata, the same pool rows — **only the
pooling of the embeddings changes**, for the positives and the pool alike.

🔴 **The comparison must be within-pooling, not across.** A residue-only embedding is a different
representation, so absolute flag rates will move for reasons that have nothing to do with the U.
What is compared is the **shape**:

> **U-index** = flag rate(length 0–250) / flag rate(length 450–550), inside the intracellular stratum.

Under include-specials pooling the U-index is **14.15 / 1.69 = 8.37**.

---

## 2. Primary test, frozen bands

**N-2**: the U-index under residue-only pooling.

| band | reading |
|---|---|
| **≤ 2.0** | 🟢 the short arm was substantially a special-token artifact |
| **2.0 – 5.0** | ⚠️ partial — the artifact contributes but does not account for it |
| **≥ 5.0** | 🔴 the short arm is not a pooling artifact and needs another explanation |

🔒 **The long arm is the control.** `flag rate(700+) / flag rate(450–550)` is **7.21** under
include-specials, and the displacement shrinks with length, so **a special-token mechanism cannot
explain the long arm.** If the long-arm index moves as much as the short-arm index does, the
comparison is picking up something other than the special tokens and **N-2 is uninterpretable** — that
is the guard, not a footnote.

## 3. Guards

- 🔒 **Gate on the embedding**: `src/94` computes the include-specials mean in the same forward pass
  and must reproduce the published positives within 5e-5, or nothing is written.
- 🔒 **Floors**: each length band needs ≥ 100 intracellular proteins. Observed 274 / 849 / 297.
- 🔒 **Sanity**: the localization ratio `R` under residue-only pooling is reported. If it collapses
  below 1.5, the residue-only representation is not supporting the probe at all and N-2 says nothing
  about the U.

## 4. Frozen predictions

- **N-2: U-index between 3.0 and 6.0** — the artifact contributes, the short arm does not vanish. The
  partial band.
- **The long-arm index stays above 5.0**, because nothing in the mechanism touches it.
- **R stays above 3.0** under residue-only pooling.
- ⚠️ 🔒 **Checkable against a wrong answer**: if the U-index comes back **equal to 8.37 to two
  decimals**, the pooling did not change and that is a bug, not a finding.

---

## Results, 2026-09-29

`src/94_class_axis_positives_mean_res.py` wrote the missing residue-only positives — its gate
reproduced the published include-specials array at **0.000e+00**, and the cross-artifact check
`d × (L+2)` gave **9.09** on the positives against the pool's **9.51**. `src/95_pooling_artifact.py`
then refitted the probe under each pooling. `results/pooling_artifact.json`.

| pooling | short (0–250) | mid (450–550) | long (700+) | **U-index** | long-index | `R` |
|---|---:|---:|---:|---:|---:|---:|
| include-specials | 14.15% | 1.69% | 12.18% | **8.38** | 7.21 | 5.15 |
| residues only | 14.16% | 1.68% | 12.18% | **8.41** | 7.23 | 5.15 |

### 🔴 N-2: not a pooling artifact — nothing moved at all

**U-index 8.38 → 8.41**, in the band the document calls *not a pooling artifact*. Every rate is
unchanged to two decimal places, and so is `R`. ⚠️ **The frozen bug-guard did not fire** — the two
arrays genuinely differ, by up to **6.35e-02** on the first 200 rows — so this is a real null, not a
comparison of an array with itself.

### ⭐ And the geometry says why, which is the part worth keeping

| | |
|---|---:|
| median embedding norm ‖m_full‖ | 6.912 |
| median displacement ‖m_full − m_res‖ | 0.0232 |
| displacement as a share of the norm | **0.34%** (0.71% for proteins under 250) |
| ⭐ **share of the displacement lying along the probe's decision direction** | **2.1%** |
| resulting shift in the score | **1.07% of an IQR** |

🔑 **The mechanism is real, exactly the size the algebra predicts, and almost perfectly orthogonal to
what the probe reads.** Two independent reasons it cannot drive the U: it is a third of a percent of
the embedding, and 98% of even that lies in directions the probe ignores.

### 🔴 The control is uninformative here, and that is not the same as passing

§ 2 made the long-arm index a guard: if it moved as much as the short-arm index, the comparison would
be picking up something other than the special tokens. **Neither index moved** — 0.2% against 0.3% —
so the guard **had nothing to discriminate**. 🔒 **Reported as uninformative rather than as passed.**
It would have mattered had the short arm moved.

### 🔒 Scoring the frozen predictions

| | frozen | actual | |
|---|---|---|---|
| U-index 3.0 – 6.0, partial band | | **8.41** | ❌ the artifact contributes **nothing** |
| long-arm index stays above 5.0 | | 7.23 | ✅ |
| `R` above 3.0 under residue-only pooling | | 5.15 | ✅ |
| U-index unchanged to 2 dp ⇒ a bug | | 8.38 → 8.41, arrays differ by 6.35e-02 | ✅ guard correctly silent |

⚠️ **The primary prediction was wrong in the direction of expecting an artifact.** I proposed this
study on the strength of a mechanism that turned out to be real, measurable, and irrelevant.

### 🔑 What is now open

**The U-shape is a property of the representation and the probe, not of the pooling.** Why proteins
under 250 residues and over 700 are both flagged several times more often than mid-length ones is
**not explained by anything in this repository**, and the most obvious mechanical explanation has been
eliminated. ⚠️ Single-arm, so indicative: a second arm needs the 8,259-protein pool re-embedded at 35M
with residue-only pooling.
