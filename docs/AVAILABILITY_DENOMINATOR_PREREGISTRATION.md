# Study X — is the availability gap a denominator, not a phenomenon?

**Frozen 2026-09-30, before any of X-1, X-2 or X-3 was computed.** Code: `src/106_availability_denominator.py`.
Artifacts: `results/availability_denominator.json`, `results/availability_denominator_esm2_35M.json`.

---

## 1. The guard that has failed four times

Study E froze an availability check: the **unannotated** stratum's flag rate over the **annotated** rate
must sit in **[0.67, 1.50]**. It has returned **0.409**, **0.322**, **0.484** and **0.518** — four runs,
four failures, across two stratum rules and two model arms. Entry 73 ruled out three causes: length
explains **≤ 10%** of it, neither score factor localises it, and embedding density is **zero**. The entry
closed by recording that the cause was unknown.

⚠️ **This study does not propose a fourth cause. It questions the denominator.**

The pooled `annotated` rate averages three classes that study E itself showed are graded by a factor of
**five** — extracellular **0.239**, membrane **0.169**, intracellular **0.046** on the 650M arm. **A ratio
against that pool is a ratio against a composition.** If the unannotated stratum behaves like one of the
three rather than like their weighted mean, the guard was never measuring what its name says.

---

## 2. X-0 — a derivation, and deliberately **not** a prediction

🔴 **I read these numbers out of four existing artifact files before writing this document, so nothing
here is scored as a prediction.** It is recorded as a derivation, with provenance, because the rest of the
design depends on it and the reader is owed the order of events.

Unannotated flag rate divided by each annotated class, from `results/localization_control{,_esm2_35M}.json`
and `results/localization_coverage{,_esm2_35M}.json`:

| stratum rule | arm | /extracellular | /membrane | **/intracellular** | /pooled annotated |
|---|---|---:|---:|---:|---:|
| E (keywords) | 650M | 0.197 | 0.253 | **1.035** | 0.409 |
| E (keywords) | 35M | 0.164 | 0.181 | **0.817** | 0.322 |
| L (GO + signal + TM) | 650M | 0.193 | 0.274 | **0.994** | 0.484 |
| L (GO + signal + TM) | 35M | 0.215 | 0.235 | **1.218** | 0.518 |

🔑 **All four /intracellular values sit inside the guard's own band. All four /pooled values sit outside
it.** The guard's band was not wrong; **its denominator was.**

⚠️ **This is a re-reading of published files, and re-readings are cheap.** It establishes agreement of
**flag rates at one threshold**, which is much weaker than agreement of populations. Sections 3–5 are the
part that can fail.

### 🔒 Why unannotated should resemble intracellular, stated before testing it

UniProt records extracellular or membrane localization **because a feature is detectable** — a signal
peptide, a transmembrane helix, a curated GO term. A protein with none of the three is not a protein whose
localization is unknown in the uniform sense; **absence of all three is itself weak evidence of a
cytoplasmic protein.** So `unannotated` is expected to be a **low-information proxy for intracellular**,
and a probe that reads localization (study E, R = 5.07) should score it accordingly.

⚠️ **This reasoning is partly circular for study L's rule**, whose unannotated set is by construction free
of signal and transmembrane annotation. **It is not circular for study E's rule**, which used keywords
only — and study E's arms give 1.035 and 0.817.

---

## 3. X-1 — the length-adjusted ratio against intracellular

Mantel–Haenszel over **10 length deciles**, minimum **20** rows per cell per group, minimum **6**
contributing deciles for a verdict — study T's floors, unchanged and imported, not restated in code.
Denominator: **intracellular only**. Study L's stratum rule, both arms, 30 seeds.

| band | reading |
|---|---|
| **[0.80, 1.25]** | unannotated is indistinguishable from intracellular |
| **[0.67, 1.50]** outside the above | unannotated is within the guard's own band of intracellular |
| outside **[0.67, 1.50]** | ❌ the availability gap is **not** a denominator artifact |

🔒 **Prediction: inside [0.67, 1.50] on both arms. I do *not* predict the tighter band.** The crude values
already span **0.817 – 1.218**, and study T's adjustment moved its ratio by only **0.05**, so a value
outside [0.80, 1.25] on one arm is entirely possible and would not disturb the reading.

---

## 4. X-2 — does availability add anything once localization is in the model?

The nested test, in the shape study W used. Features are **three localization dummies with unannotated
folded into intracellular** — that coding *is* the hypothesis — and the increment is an added
**unannotated** indicator that lets it depart from intracellular.

- model A: `extracellular`, `membrane`, `intracellular ∪ unannotated`
- model B: the same, plus `unannotated`

Reported: ΔAUROC (B − A) under 5-fold cross-validation on the flag, and model B's fitted coefficient on
the unannotated indicator.

| band on ΔAUROC | reading |
|---|---|
| **< 0.005** | availability adds nothing over localization |
| **0.005 – 0.020** | availability adds a little |
| **> 0.020** | ❌ availability is not reducible to localization |

🔒 **Prediction: < 0.005 on both arms, and the unannotated coefficient within ±0.25.** This one is derived
from a measurement rather than reasoned about: X-0 puts the unannotated rate on top of the intracellular
rate, and intracellular ∪ unannotated is the large majority of eligible rows, so an indicator that
re-labels part of a class whose rate it matches cannot add discrimination.

---

## 5. X-3 — the test that can actually fail

🔴 **Equal flag rates at the 95th percentile do not imply equal score distributions.** Two populations can
agree in one tail and differ everywhere else. X-3 is therefore the open question and is stated as such.

**AUROC of the probe score, unannotated against intracellular, 1:1 length-matched** (study E's
`length_matched`, imported), 30 seeds, both arms.

| band | reading |
|---|---|
| **[0.45, 0.55]** | the probe does not separate them |
| **[0.40, 0.60]** outside the above | weak separation |
| outside **[0.40, 0.60]** | ❌ different populations to the probe; X-1's agreement is a tail coincidence |

🔒 **Prediction: inside [0.40, 0.60]. I decline to predict the tight band**, for the reason stated at the
head of this section: the evidence in hand constrains one quantile, and an AUROC integrates all of them.

---

## 6. X-4 — the replacement guard, and its null calibration

The guard is re-specified against intracellular. ⚠️ **A guard that has never failed is worth nothing**, so
the artifact must record that the new one **fails on input it should reject**:

1. fed the **pooled** denominator, it must fail (the four published values, 0.32 – 0.52, are outside)
2. fed a **shuffled** stratum assignment, it must fail

🔒 Both are computed and written to the artifact. If either passes, the guard is not reported.

---

## 7. Obligations, written before the data exists

- ❌ **If X-1 lands outside [0.67, 1.50]**, the denominator explanation is refuted, entry 73's "cause
  unknown" stands unamended, and this document says so in those words.
- ❌ **If X-2's ΔAUROC exceeds 0.020**, availability carries something localization does not, and the
  denominator reading is at most partial.
- ❌ **If X-3 falls outside [0.40, 0.60]**, then X-1's agreement is a threshold coincidence. The guard is
  still mis-specified, but **"unannotated is intracellular" is not the explanation**, and the entry must
  lead with the failure rather than with X-0's table.
- 🔒 **Regardless of every outcome above**, study E's availability check is amended, because a denominator
  that pools three classes spanning a factor of five is wrong independently of what replaces it. ⚠️ **The
  four published failing values stay in the record**; they are not deleted, they are re-read.
- 🔒 **No number in § 2 may be presented as a confirmed prediction**, in this document or in
  `paper/MANUSCRIPT.md`. The claims audit enforces that with a forbid.

---

## 8. Results

*(to be appended after the run, append-only)*
