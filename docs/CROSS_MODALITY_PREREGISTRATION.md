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

---

## Amendment 1 — the first design had no positive/negative axis, and a precondition is added

**Added 2026-10-06, after the first run's recoveries were seen and before the fixed design was run.**

### 🔴 The fold was measuring noise

§ 2 drew positives from 20 Newsgroups' **train** split and negatives from its **test** split, with the
stated justification that negatives should be "text from the same source and not a different
distribution". ⚠️ **That made positives and negatives the same kind of thing.** The probe was being asked
to separate train-split newsgroup posts from test-split newsgroup posts, which is a null task.

🔴 **The recoveries say so**: `rec.autos` 1.26%, `rec.motorcycles` 1.43%, `rec.sport.baseball` 0.82%,
`rec.sport.hockey` 0.82% — **below the 5% nominal**, i.e. a held-out class was *less* like the trained
positives than the negatives were. 🔑 **The protein fold has a real contrast** — registered virulence
factors against benign proteins — **and the text fold had none.**

### 🔒 The fixed design uses the corpus's own hierarchy

| | |
|---|---|
| **positive side** | `comp.*` (5), `rec.*` (4), `sci.*` (4) = **13 sub-classes**, held out one at a time |
| **negative side** | `alt.atheism`, `misc.forsale`, `soc.religion.christian`, `talk.*` (4) = **7 groups** |

🔑 **That is the protein structure**: a genuine positive/negative axis — technical, hobby and science
content against belief, politics and commerce — with sub-classes *inside* the positive side playing the
role of mechanism categories. **Thirteen classes against DeepVIC's fourteen.**

### 🔒 AS-0, a precondition frozen before the run

**Mean recovery over the thirteen classes must exceed 20%** at the fixed classifier. ⚠️ **Below that the
fold is still close to null and AS-1 is uninterpretable whatever ρ comes out**, because a rank
correlation over thirteen near-zero numbers is noise. 🔴 **The first design would have failed this
precondition at ~1%, and nothing in § 3 would have caught it** — the bands were all on ρ.

🔑 **This is the check the original preregistration should have had**, and its absence is why a broken
fold ran to completion before anyone looked at a level.

### Predictions, restated for the fixed design

**P1 and P2 are unchanged.** **P3 is withdrawn**: the confusable-pair reasoning was about all twenty
newsgroups, and `talk.religion.misc` and `soc.religion.christian` are now both on the **negative** side,
so the prediction no longer has a referent. ⚠️ **Withdrawn rather than quietly rewritten**, and it counts
as neither held nor missed.

**P4 (derived — from the hierarchy: `comp.sys.ibm.pc.hardware` and `comp.sys.mac.hardware` are near
neighbours, as are the two `rec.sport.*` groups).** The lowest-margin classes are drawn from those two
near-duplicate pairs, because a class whose nearest other-class neighbour is its own sibling has little
to distinguish it.
