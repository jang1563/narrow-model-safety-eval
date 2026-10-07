# Does the margin result hold outside proteins? — preregistration

**Frozen 2026-10-06, before `src/138` existed.** Study AS.

## 1. What is actually being claimed

The one result this project has that replicated — internally across fourteen model arms, then
externally on DeepVIC's fourteen curator categories, then across a change of classifier, then across a
re-draw of negatives — is:

> 🔒 **A quantity computable from a frozen representation before any training predicts which held-out
> class a linear probe will fail on.**

    margin(C) = mean over members of C of [ max cos to a positive outside C − max cos to any negative ]

🔑 **Nothing in that formula mentions proteins.** It takes an embedding matrix and a class partition. If
the result is about representation geometry it should appear in any domain with the same structure; if it
is about protein language models it should not.

⚠️ **Every test so far has been on one modality and, on DeepVIC, one model family at two sizes.** That is
the weakest part of the claim and this is the cheapest decisive test of it.

## 2. The measurement

The **identical fold** — `src/125`'s, with the selection and threshold rules unchanged — on text:

- **data**: 20 Newsgroups (scikit-learn's `fetch_20newsgroups`), **20 classes**, headers, footers and
  quotes stripped so the task is topical content and not metadata;
- **positives**: the documents of the 20 classes, held out one class at a time;
- **negatives**: a disjoint background drawn from the same corpus's *other* split, so negatives are text
  from the same source and not a different distribution;
- **representation**: `sentence-transformers/all-MiniLM-L6-v2`, frozen, mean-pooled — the analogue of a
  frozen protein language model;
- **probe**: `StandardScaler` → `LogisticRegression`, threshold at the 95th percentile of calibrate
  negatives, 30 seeds.

🔒 **`margin` is `src/30`'s, imported.** 🔒 Both the fixed `C = 1.0` and `src/133`'s nested-selected `C`
are run, because study AP showed the default moves levels by up to 24 points.

## 3. Frozen bands

Floors **and** ceilings. Bonferroni α = 0.05/2.

| | the result generalizes | it does not |
|---|---|---|
| **AS-1** *ρ*(margin, recovery) over the 20 classes | **≥ +0.60** at both classifier settings | **≤ +0.20** at either |
| **AS-2** permutation *p* for AS-1 | **< 0.0125** | ≥ 0.05 |

⚠️ **There is no second arm here.** One text encoder is one arm, so AS-1 is reported as **single-arm
indicative** however it falls, exactly as study AJ's composition arm was. 🔑 **A second encoder is the
obvious follow-up and is named rather than promised.**

⚠️ **And twenty classes is twenty points.** Study AR showed this ρ is the noisy part — it moved 0.105
between negative draws while the levels moved 0.023 — so a single ρ near the floor should not be read as
a near-miss.

## 4. Predictions

**P1 (derived — the quantity is a statement about nearest-neighbour structure, and nearest-neighbour
structure is what every contrastively or masked-pretrained encoder has).** AS-1 clears **+0.60** at both
settings.

**P2 (derived — from study AP: 1,280 features on ~13k positives over-fit, and MiniLM's 384 on ~18k
documents should over-fit less).** The tuning gain on text is **smaller** than the +13.96pp seen on
ESM-2 650M.

**P3 (reasoned).** The classes this fails on are the two notoriously confusable newsgroup pairs
(`talk.religion.misc` with `soc.religion.christian`, `comp.os.ms-windows.misc` with `comp.sys.*`) rather
than the small ones.

⚠️ **The computed record stands at derived 15 of 27, reasoned 3 of 11.**

## 5. What each outcome changes

- **AS-1 clears the floor** ⇒ 🔑 **the result is about frozen-representation geometry and not about
  protein language models**, and the paper's framing changes from a biosecurity evaluation with a
  geometric aside to a transfer-prediction result with biosecurity as its motivating domain.
- **AS-1 fails** ⇒ the result is protein-specific, which is a narrower but still publishable claim, and
  the paper says so instead of implying generality it has not tested.
