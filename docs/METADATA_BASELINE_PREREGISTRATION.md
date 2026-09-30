# Preregistration — how much of the virulence-factor task do two metadata features do?

**Frozen 2026-09-30, after a declared single-feature diagnostic and before any fitted model has been
scored.** One primary, one comparison, one declared limitation. Append-only.

---

## 0. The gap this addresses, and the one it cannot

The step-by-step review's remaining criticism was **no external validation**: § 1.2 cites DeepVF
(AUC 0.896), DTVF (0.921) and DeepVIC (0.954) and this project has never run anything comparable.

🔴 **This study does not close that gap and does not claim to.** Their benchmarks are different
populations — 3,576 VFs against 4,910 non-VFs for DeepVF, 33,456 against a 13,384 holdout for DeepVIC —
and **nothing here re-analyses their numbers.** What it can do is ask a question their papers do not:
**what does a model with no sequence information at all achieve on a task of this shape?**

🔑 **The project already keeps baselines of this kind** — § 14 reports amino-acid composition alone at
AUROC 0.754 and shuffled labels at 0.506, "reported because they make the model look worse". This is the
same practice applied to the comparison the literature makes.

## 🔒 0.1 Declared diagnostic, taken before freezing

Task: **778** VFDB proteins (study K's mapped, non-circular, length-matched set) against **1,988** pool
proteins that are Bacteria from VFDB species and not in VFDB.

| stratum | VFDB | benign |
|---|---:|---:|
| extracellular | 33.9% | 8.6% |
| membrane | 20.6% | 8.1% |
| intracellular | 17.2% | 30.0% |
| unannotated | 28.3% | 53.3% |

| single feature, no model fitted | AUROC |
|---|---:|
| localization stratum | **0.6973** |
| length | **0.3072** — i.e. **0.693** with the sign flipped; VFDB proteins are *shorter* |

⚠️ Nothing combined has been fitted. **A crude rank-sum of the two was computed and is discarded rather
than reported**: it scored 0.638, worse than either feature alone, because it used length in the wrong
direction. **A bad combination is not a result.**

---

## 1. Design

⚠️ **No new inference.** Five-fold cross-validation over the same 2,766 proteins, the same folds for
every feature set, AUROC averaged across folds with the fold-to-fold interval reported.

| feature set | contents |
|---|---|
| **metadata-only** | localization stratum (one-hot, 4) + `log10 length` — **no sequence information** |
| **ESM-2** | the arm's pooled embedding, 1,280 or 480 dimensions |
| metadata + ESM-2 | both, to see whether metadata adds anything the embedding lacks |
| shuffled labels | the floor, as § 14 keeps one |

🔒 Logistic regression with standardization in every case, so the comparison is between **inputs**, not
between model classes.

---

## 2. Primary test, frozen bands

**W-1**: the metadata-only AUROC. **W-2**: the ESM-2 increment over it, same folds.

| outcome | reading |
|---|---|
| metadata ≥ 0.70 **and** increment < 0.15 | 🔴 **two annotation fields do most of what this task rewards**, and a sequence model's headline AUROC on a task of this shape is not evidence about sequence |
| metadata ≥ 0.70 **and** increment ≥ 0.15 | ⚠️ metadata is a large floor and the embedding clears it |
| metadata < 0.70 | 🟢 the metadata floor is modest and the published numbers are not in question from this direction |

🔒 **What no outcome licenses.** ⚠️ **This is not their benchmark**, their negatives were chosen by their
own rules, and **a high metadata floor here does not establish one there.** The claim available is about
**a missing baseline in a literature**, demonstrated on an analogous task — and it must be written that
way or it is an overreach.

## 3. Frozen predictions

- **W-1: metadata-only between 0.78 and 0.85.** The two single features sit at 0.697 and 0.693 and are
  not obviously redundant — localization is annotation-derived, length is not — so a fitted combination
  should clear both.
- **W-2: the increment is large, 0.10 – 0.20**, and the embedding lands above 0.92. Study K measured a
  2.4× flag-rate contrast on this population; that is a strong separation and should show up as AUROC.
- **So the expected verdict is the middle band**, not the alarming one.
- ⚠️ 🔒 **Checkable against a wrong answer**: shuffled labels must return 0.50 ± 0.03, or the
  cross-validation is leaking.

## 4. Floors

🔒 Every fold must contain at least 100 VFDB proteins, or the fold count is reduced and reported.
🔒 The § 2 limitation travels with every number this study produces.

---

## Results, 2026-09-30

`src/105_metadata_baseline.py`, five-fold cross-validation over 778 VFDB proteins and 1,988 benign,
identical folds for every feature set, logistic regression throughout.

| feature set | ESM-2 **650M** | ESM-2 **35M** |
|---|---:|---:|
| ⭐ **metadata only** (localization one-hot + log length) | **0.8096 [0.7937, 0.8256]** | **0.8096** |
| ESM-2 embedding | 0.9323 [0.9236, 0.9410] | 0.9035 |
| metadata + ESM-2 | 0.9330 | — |
| shuffled labels, 20 repetitions | **0.4455** | **0.4668** |

*for context, not comparison:* DeepVF 0.896, DTVF 0.921, DeepVIC 0.954

### 🔴 W-1: two annotation fields reach 0.81 with no sequence information

**Localization stratum and log length** — one field from UniProt and one integer — give **AUROC 0.8096**
on separating VFDB proteins from matched benign ones. 🔒 **No sequence model, no embedding, no training
on sequence at all.**

### ⚠️ W-2: the increment is +0.123 and +0.094, and the verdict hangs on a threshold I chose

ESM-2 reaches **0.9323** and **0.9035**, an increment of **+0.1227** and **+0.0939** over metadata. Both
fall under the frozen 0.15, so the bands read *"two annotation fields do most of what this task
rewards"*. 🔴 **But 0.1227 against a 0.15 boundary is a knife edge, and the boundary was my choice.**
The honest statement is the numbers: **metadata gets to 0.81, the embedding adds a tenth of an AUROC.**

⭐ **And the embedding subsumes the metadata.** Adding both gives **0.9330** against the embedding's
0.9323 — **+0.0007**. 🔑 So the two annotation fields are not extra information the model lacks; **they
are a cheap approximation of something it already has**, which is the more interesting reading and the
one that makes the missing baseline matter: a paper reporting 0.93 has not shown that 0.81 of it needed
a protein language model.

### 🔴 The shuffled-label floor is not 0.50 here, and that took three controls to establish

One shuffle returned **0.4663**, tripping a frozen tolerance of ±0.03 whose own fold interval covered
0.5. ⚠️ **That tolerance was tighter than the estimator's spread — a defect in the guard.** Twenty
repetitions give **0.4455** and **0.4668** with intervals that **exclude 0.50**, so it is systematic
rather than noise.

🔒 **Three controls establish the procedure is clean**: random features at the same shapes and imbalance
give **0.5013**, random features with a real class offset **0.5030**, balanced classes **0.4990**.

🔑 **The cause is in the data**: standardized over the pooled set, VFDB embeddings are systematically
**more dispersed** — ‖z‖ median **39.87** against **32.52**, per-dimension SD **1.106** against
**0.931**. Under permuted labels a fit lands on a high-leverage direction, and the more dispersed class
sits at one end of it **systematically, not at random**.

⚠️ **This does not threaten W-1 or W-2** — both sit far above either floor — **but the guard was
mis-specified twice over**: its tolerance was tighter than its estimator's spread, and it was written
against 0.50 when this design's null is not 0.50.

### 🔒 Predictions

| | frozen | actual | |
|---|---|---|---|
| W-1 metadata 0.78 – 0.85 | | **0.8096** | ✅ |
| W-2 increment 0.10 – 0.20 | | +0.1227 / **+0.0939** | ✅ 650M, ❌ 35M below |
| the expected verdict is the middle band | | the stronger band, on both arms | ❌ |
| shuffled labels 0.50 ± 0.03 | | 0.4455 / 0.4668 | ❌ — and the guard, not the data, was wrong |

### 🔴 What this does not establish, restated because it is the whole risk

⚠️ **This is not DeepVF's, DTVF's or DeepVIC's benchmark.** Their negatives were chosen by their own
rules over different populations, and **a metadata floor here does not establish one there.** The claim
available is that **a baseline of this kind is missing from that literature**, demonstrated on an
analogous task — not that their reported numbers are inflated.
