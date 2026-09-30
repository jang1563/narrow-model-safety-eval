# Study AE — spend the reviewed-only rule, pool both benign draws, and settle `C/A`

**Frozen 2026-10-01, before any expanded contrast was computed.** Code: `src/117_spent_rule.py`.
Artifacts: `results/spent_rule{,_esm2_35M}.json`.

---

## 1. What this spends and what it pools

`docs/COVERAGE_LEDGER.md` located **54.4%** of the VFDB loss in one rule — reviewed-only — and showed the
fields it protects, **length** and **organism**, are present on unreviewed entries too. Study AC then left
`C/A`'s verdict at **partial** on an honest interval, **[1.904, 3.252]** and **[1.931, 3.938]**, straddling
the 2.0 band.

Two expansions, both from data already on disk:

| | now | after |
|---|---:|---:|
| VFDB accessions with a usable UniProt entry | 966 | **3,261** |
| benign draws contributing to the denominator | 1 | **2** (pool 1 + the independent 11,301) |

🔒 **No new network fetch for the numerator**: `vfdb_negatives_idmap.tsv` already holds **3,601** unreviewed
rows, with **Keywords on 96.9%** and **Length on 100%**, so the keyword strata and the length-identity
check apply unchanged.

🔒 **The two cheap exclusions stay**: non-circularity (178 rows, buys freedom from training-data leakage)
and length-identity (10 rows, buys freedom from mapping error). **Only the curation rule is spent.**

---

## 2. The thing that makes this not merely a precision gain

🔴 **The two benign draws disagree about `C/A`.** Pool 1 gives **2.81 / 2.89**; the independent pool gives
**1.59 / 1.63**. Pooling them therefore **moves the point estimate**, it does not only narrow the interval,
and the pooled value must land between the two.

⚠️ **Which way it lands is governed by the denominator**: `C/A` falls when the benign extracellular rate
rises, and pool 2's is the higher one. 🔒 **So this study can lower the point estimate while narrowing the
interval, and both effects are reported separately.**

---

## 3. Predictions

🔒 **Pooled point estimate in [1.6, 2.3]**, below pool 1's 2.418. Derived: the two draws bracket it at
**1.59** and **2.81**, and pool 2 contributes the larger benign cell, so the pooled value sits nearer the
lower end.

🔒 **The honest interval's lower bound clears 1.5 and does not clear 2.0** — *partial* confirmed rather
than overturned. Derived from the bracket above: a pooled point near 1.8–2.0 with an interval narrowed by
roughly **1.7×** from study AC's ±0.67 cannot reach 2.0 from below.

⚠️ **If the numerator expansion raises the VFDB rate**, the point estimate rises instead, and the
prediction fails. That is possible: unreviewed entries are less-studied proteins, and study K's K-3 found
that the members *previous* coverage raises brought in were flagged **19.2 points above** the rest. 🔒
**This is the reason the band is [1.6, 2.3] and not a point.**

| band on the honest lower bound | reading |
|---|---|
| **> 2.0** | 🟢 supported returns — study AC's partial was a small-sample artifact |
| **1.5 – 2.0** | ⚠️ **partial confirmed** on 4× the population |
| **< 1.5** | 🔴 the contrast falls out of partial too |

---

## 4. Obligations

- 🔒 **The numerator and denominator effects are reported separately** — expanded VFDB against pool-1
  benign, and original VFDB against pooled benign — so a reader can see which expansion moved what.
- ❌ **If the honest lower bound falls below 1.5**, the registration contrast is stated as *not supported*
  in § 5 and the abstract, and the frozen retraction threshold (`C/A ≤ 1.2`) is checked explicitly.
- ⚠️ **The unreviewed rows' stratum assignment is from automatic annotation**, and that is a new confound
  this study introduces rather than removes. 🔒 **It is reported as a caveat and the reviewed-only subset
  is always shown beside the expanded one**, so the cost of spending the rule is visible.

---

## 5. Results — partial confirmed on 2.7× the VFDB population, and the rule was inert

**Run 2026-10-01**, 30 seeds, B = 20,000, both arms.
`results/spent_rule{,_esm2_35M}.json`. Benign extracellular: pool 1 **171**, pool 2 **130**, pooled
**301**. VFDB extracellular: reviewed-only **264**, expanded **703**.

| cell | 650M point | 650M honest | 35M point | 35M honest | verdict |
|---|---:|---|---:|---|---|
| as published (reviewed / pool 1) | 2.418 | [1.904, 3.239] | 2.617 | [1.943, 3.940] | partial |
| **numerator only** (expanded / pool 1) | **2.409** | [1.904, 3.223] | **2.568** | [1.934, 3.921] | partial |
| **denominator only** (reviewed / pooled) | **2.114** | [1.775, 2.571] | **2.319** | [1.824, 3.252] | partial |
| **both** | **2.106** | **[1.772, 2.541]** | **2.277** | **[1.821, 3.223]** | **partial** |

### 🟢 All three predictions held, and all three were derived

| | frozen | 650M | 35M |
|---|---|---|---|
| pooled point in [1.6, 2.3] | | **2.106** ✅ | 2.277 ✅ |
| honest lower bound > 1.5 and < 2.0 | | **1.772** ✅ | **1.821** ✅ |
| interval narrows ≈ 1.7× | | **1.74×** ✅ | 1.43× ⚠️ |

🔒 **`C/A` is partial, confirmed rather than overturned**, on **2.7×** the VFDB population and **1.8×** the
benign one, on both arms, and comfortably above the frozen retraction threshold of **1.2**.

### ⭐ The rule cost 54.4% of the population and bought nothing here

Expanding the numerator from **264 to 703** moved the point estimate by **0.009** on 650M and **0.049** on
35M. 🔑 **Whether a VFDB protein has a curated Swiss-Prot entry does not predict whether this probe flags
it.** ⚠️ **So the reviewed-only rule was expensive and inert for this contrast** — which is not an argument
that it was wrong when frozen, because nothing had measured it; it is the measurement.

### 🔑 What does move it is which benign proteins you drew

Pooling a **second independent draw** moved the point estimate by **0.30 on both arms** — between **6×**
and **33×** the numerator expansion's effect. 🔒 **The denominator is the lever**, which is the same
conclusion the field reached independently: *Evaluation of machine learning and negative data strategies*
([Brief Bioinform 21(5) 1596](https://academic.oup.com/bib/article/21/5/1596/5574719)) swings AUPRC from
**0.62 to 0.87** on negative-set choice alone.

### ⚠️ The caveat this study introduces

**439 of the 703** VFDB rows now come from **unreviewed** entries whose stratum is assigned from automatic
annotation. 🔒 The reviewed-only column is printed beside every expanded one, so the cost of spending the
rule is visible — and in this case the two columns agree to within 0.05, which is itself the reason to
trust the expansion.
