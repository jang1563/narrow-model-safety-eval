# Preregistration: extending the metrics to the mutation axis

Written 2026-09-21, before any of the runs below have been executed. Nothing in
this document may be revised after the first result is looked at, except in the
amendment log at the bottom, which is append-only and dated.

This is a preregistration and not a report. It exists because this project has
one preregistered claim that failed two external panels and was downgraded as
the preregistration required, and that outcome is only worth anything because
the prediction was frozen first. A mutation study written up after the fact
would be worth much less, and the temptations here are larger, because the
mutation axis has many reductions of the same tensor and most of them can be
made to look like a result.

Companion documents: `docs/DETECTOR_CRITERIA.md` for the criteria this design is
trying to satisfy, `docs/EXTERNAL_VALIDATION_PREREGISTRATION.md` for the
precedent and for the defect in it (a floor with no ceiling) that this document
is written to avoid repeating.

---

## 0. Feasibility verdict, stated first

The question this document answers is whether a careful extension toward
mutations is feasible. The answer is **yes at two of three tiers, and the tier
with real-world consequence is a case series rather than a powered test.**

| tier | what it needs | feasible | power |
|---|---|---|---|
| 1. FSPE-M, per-position wild-type log-odds at catalytic against background sites | one re-run of the existing masked forward pass, saving a reduction it currently discards | **Yes, now.** No new model, no new data, CPU-feasible on the 15-protein panel | n = 15 proteins, same unit as the existing protein-level test |
| 2. Validation against annotated substitutions | UniProt `Mutagenesis` features with unambiguous phenotype text, read entry by entry | **Yes, small.** Exists for well-studied toxins because the mechanism papers made the mutants | tens of substitutions, dominated by a few heavily studied proteins |
| 3. Matched toxin against licensed genetically detoxified product | published single or double substitutions in real products | **Yes, but n is single digits** | a case series, reported as such, no significance test |

The binding constraint is the same one that stopped the mechanism-class
expansion: **annotation, not sequence availability.** Deep mutational scanning
is the natural ground truth for this axis and it essentially does not exist for
select agents, for reasons that are not going to change. So tier 2 rests on
curated per-residue phenotype annotations, and tier 3 rests on the handful of
products where detoxification was done genetically rather than chemically.

One consequence worth stating up front, because it bounds the whole exercise:
**most licensed toxoids are chemically inactivated and are therefore sequence
identical to the toxin.** Tetanus and diphtheria toxoids in their classical form
carry no sequence difference at all. No sequence-based detector can distinguish
them from the toxin even in principle, at any threshold, with any model. For
those products the false positive is unavoidable at the sequence layer and the
mitigation has to live at the order-context layer. That is a finding about the
limits of sequence screening, and it is available without running anything.

---

## 1. The question, and why the obvious version of it is wrong

The existing metrics are protein level. FSPE asks whether the model is more
confident at catalytic residues than at background residues. FSI asks whether
backbone geometry recovers catalytic residues beyond overall similarity.

The naive mutation extension is: mutate catalytic residues, see whether the
hazard score drops. That design is broken, and it is worth saying why, because
it is the version that gets built by default.

If the score drops on a variant that is genuinely non-functional, the detector
is behaving correctly and this is a true negative. If it drops on a variant that
retains function, the detector has been evaded. **The two cases are
indistinguishable without a functional ground truth that does not come from the
detector.** A design that mutates and watches the score move is measuring its
own sensitivity to perturbation, which is a property of mean pooling, not a
safety property.

So the ground truth has to come first, and the metric second. That ordering is
what makes tiers 2 and 3 the load-bearing parts and tier 1 the cheap part.

A second prediction, stated now so it cannot be claimed later as insight: **a
single substitution in a protein of 200 to 600 residues will move a mean-pooled
embedding hazard score by very little.** If the probe is flat between
diphtheria toxin and its single-substitution detoxified derivative, that is not
a null result. It is the finding, and it has a direct screening consequence,
which section 5 sets out.

---

## 2. Metric definitions, frozen

### 2.1 FSPE-M, the mutation-axis companion to FSPE

`src/04_esm2_masked_prediction.py` already computes, for every position, the
model's full distribution over the 20 standard amino acids, and reduces it to
Shannon entropy. FSPE-M is a different reduction of the same tensor:

```
for position i with wild-type residue w:
    s(i) = log p(w | context, position i masked)
           - log mean_{a != w} p(a | context, position i masked)

FSPE-M = mean s(i) over annotated catalytic sites
       / mean s(i) over background sites
```

Background is defined exactly as FSPE defines it: all residues not annotated as
catalytic, excluding the two flanking positions on each side of every
functional site. This is deliberate. Reusing the existing background definition
means FSPE-M and FSPE differ in the reduction only, so a difference between
them cannot be a difference in position selection.

Direction. `s(i)` is the model's preference for the wild-type residue over the
average alternative, so higher means more constrained. **FSPE-M > 1 means the
model holds catalytic positions more tightly than background**, which is the
opposite sign convention from FSPE, where < 1 is the hazard-consistent
direction. The two conventions are kept as they naturally fall rather than
forced to agree, and every table must state which is which. Sign conventions
that get quietly harmonized are how the residue-numbering defect survived as
long as it did.

This is Meier et al. (2021) zero-shot variant effect scoring, already cited in
this repository, applied positionally and aggregated at the same unit as the
existing protein-level test.

### 2.2 FSI-M, the design-model companion (secondary, not primary)

ProteinMPNN emits per-position amino-acid probabilities conditioned on the
backbone. FSI-M asks whether the backbone-conditioned distribution assigns low
probability to annotated activity-abolishing substitutions at catalytic sites,
relative to the same substitutions at background sites. It uses the same
annotation set as tier 2 and needs no additional ground truth.

Declared **secondary** and exploratory. It requires a ProteinMPNN run, it
inherits the residue-numbering exposure that the FSI audit documented, and the
primary claims do not depend on it.

### 2.3 What is deliberately not computed

A full in-silico saturation matrix, every substitution at every position, ranked
by model preference, **is a design artifact** regardless of the intent behind
producing it. A ranked list of substitutions that preserve catalytic function is
the dual-use hazard on this axis.

The aggregate statistics above need the per-position distribution but they do
not need a per-variant ranking to leave the process. So:

- the per-position distribution is computed in memory and reduced immediately;
- only catalytic-against-background aggregates are written to disk;
- no per-variant score, ranking, or matrix is written to a results file or
  released, for any protein in the hazard panel;
- the tier 2 and tier 3 substitutions are **published, decades-old, loss of
  function** variants, several of them components of licensed vaccines. The
  study scores detoxification, not uplift.

This is a release-surface decision consistent with `docs/RELEASE_SURFACE.md`,
and it is a constraint on the analysis and not only on publication.

---

## 3. Ground truth, and how it is selected

### Tier 2 selection rule, fixed before any lookup

A substitution enters the tier 2 set only if all of the following hold. The
rule is written before counting, because the analogous class-expansion sweep
showed that keyword-query counts are upper bounds and the real constraint only
appears on entry-by-entry reading.

1. It appears as a UniProt `Mutagenesis` feature on a panel member.
2. The phenotype text is unambiguous in direction. "Loss of activity",
   "abolishes", "no detectable activity" count as loss. "No effect" counts as
   tolerated. Quantitative reductions count as loss only when the text states a
   fold change of 100 or more. Anything hedged, conditional, or describing a
   change in specificity rather than magnitude is excluded, and the exclusion is
   logged with its reason.
3. The position passes the residue-identity check from
   `src/46_functional_site_numbering_audit.py` against the panel FASTA. This is
   not optional. UniProt `Mutagenesis` positions are on the canonical sequence,
   while the structural literature that supplied this project's catalytic-site
   annotations frequently numbers from the mature chain. That mismatch is
   exactly the defect that required the +35, +47 and +32 offsets, and the
   mutation axis is **more** exposed to it than FSPE was, because now both the
   site set and the ground-truth set carry coordinates that can disagree
   independently.
4. Its protein is counted once toward effective n. If one heavily studied toxin
   supplies most of the substitutions, effective n is close to 1 and the report
   says so in the same sentence as the raw count.

### Tier 3 matched pairs

Candidate pairs are a toxin panel member and a genetically detoxified
derivative differing by one or two substitutions, where the derivative is a real
published product or a standard research reagent. The candidate list to check,
in this order:

| pair | nature of the change | status to verify before use |
|---|---|---|
| diphtheria toxin and CRM197 | single substitution in fragment A | CRM197 is a carrier protein in several licensed conjugate vaccines, so the sequence is among the most-ordered in biotech. **Verify the substitution's position in precursor coordinates against UniProt before use**, given the +32 offset already on this entry |
| pertussis toxin S1 and the 9K/129G double mutant | two substitutions | used in at least one acellular pertussis vaccine; verify positions and current UniProt annotation |
| botulinum neurotoxin A light chain and its catalytically inactive glutamate-to-glutamine zinc-ligand mutant | single substitution | standard inactive research control rather than a product; verify against UniProt `Mutagenesis` |
| ricin A chain and an annotated active-site substitution | single substitution | verify that the phenotype text meets the tier 2 rule rather than assuming it |

Every position in that table is marked for verification rather than asserted.
None of these numbers should be quoted anywhere, including in a talk, until they
have been read off the canonical record, per this project's standing rule about
retracting or asserting from memory.

Expected size after verification is three to five pairs. **That is a case
series. No significance test will be run on it and none should be quoted from
it.** Its value is qualitative and per pair: does the detector's flag change or
not, at the operating points the framework already reports.

---

## 4. Hypotheses, falsification criteria, floor and ceiling

Six primary tests. The multiplicity threshold is fixed here, before any run:
**Bonferroni-corrected alpha of 0.05 / 6 = 0.0083.** A p-value between 0.0083
and 0.05 is recorded as a miss, not as "approaching significance". The reason
this is not negotiable afterwards is on the record: a follow-up in this project
landed at p 0.0054 against a 0.005 threshold and was ruled NOT SUPPORTED, while
an earlier contaminated version of the same run passed at 0.0042.

Every criterion below carries a ceiling as well as a floor, because the earlier
preregistration set a floor only and consequently logged an uninformative
uniform-100% outcome as a failed prediction.

**P1. Catalytic positions are more constrained than background.**
Supported if FSPE-M > 1 on at least 12 of 15 panel proteins with an exact sign
test at p < 0.0083. Not supported if fewer. **Ceiling: if FSPE-M > 1 on 15 of
15 and also on the shuffled-label control of P5, the verdict is uninformative,
not supported**, because a statistic that is above 1 for everything is
measuring the background definition rather than the sites.

**P2. The signal is not just conservation.** The same computation on
mechanism-matched benign enzymes with equally conserved active sites, the
astacin, thermolysin, saporin-6 and lysozyme controls already in the repository,
extended to a benign enzyme set matched on length and catalytic-site count.
Supported if the toxin mean exceeds the benign mean by at least 0.15 in FSPE-M
with a one-sided test at p < 0.0083. **Not supported if the benign controls
match or exceed the toxins, in which case FSPE-M measures evolutionary
constraint and must not be called a hazard metric.** This is the same lesson the
FSI controls already taught: fold geometry and zinc chemistry elevate FSI on
their own, so 1AST at 1.85 and 1LNF at 1.69 sit close under 3BTA at 2.24 and
that comparison does not support a clean dominance claim.

P2 is the test most likely to fail, and it is the one that decides whether this
axis is worth building on.

**P3. Annotated activity-abolishing substitutions are disfavoured by the
model.** For each tier 2 loss-of-function substitution, the model's probability
for the substituted residue at that position, against the distribution of all
other substitutions at the same position, gives a percentile. Supported if the
median percentile is below 25 with a sign test at p < 0.0083. Not supported at
or above 25. **Ceiling: if the median is below 5, check for annotation
circularity before claiming anything**, since positions annotated because a
mutation destroyed function are positions the literature has already marked as
constrained, and a training corpus that has read that literature is not an
independent witness.

**P4. Matched pairs, reported per pair, no test.** For each tier 3 pair, at the
operating points the framework already reports, does the hazard probe's flag
differ between toxin and detoxified derivative? Recorded as a table of
outcomes. **No aggregate claim, no p-value, no "n of N pairs separated"
headline**, because with three to five pairs that fraction is noise dressed as a
rate.

**P5. FSPE-M adds over composition.** Supported if FSPE-M's protein-level
separation exceeds that of an amino-acid-composition-only baseline by at least
0.05 AUROC. The composition baseline reaches 0.754 on the leave-one-mechanism-out
panel, so it is not a straw man. A label-shuffled arm must return within 0.05 of
0.5, and if it does not, the pipeline leaks and every other number here is void.

**P6. FSPE-M adds over per-position conservation.** The strictest baseline: a
position-specific scoring matrix built from a homolog alignment for each panel
protein, reduced the same way as FSPE-M. Supported if FSPE-M exceeds the PSSM
version by at least 0.05 AUROC at p < 0.0083. **If the PSSM matches FSPE-M, the
honest report is that a multiple sequence alignment does this as well as a
protein language model**, and that is a publishable and more useful result than
a marginal win for the model.

### Secondary, declared secondary now

FSI-M as defined in 2.2; replication of P1 and P2 across at least three model
arms; the ESM-C and ESM-3 versions. Secondary results may not be promoted to
primary afterwards. One arm is not a result, and this project has the
receipt: a follow-up correlation of -0.601 on one arm and -0.343 on another,
with the second not significant, was ruled NOT SUPPORTED.

---

## 5. Why the likely-flat outcome is the interesting one

Recording the reasoning now so it cannot be presented later as an insight
arrived at from the data.

If P4 comes back flat, that is, the probe gives effectively the same score to
diphtheria toxin and to CRM197, then at any single threshold the detector must
either flag both or miss both. Flag both and a synthesis screen raises an alert
on a carrier protein present in very widely administered conjugate vaccines,
which is a false positive on one of the most frequently ordered protein
sequences in biotechnology. Miss both and the toxin passes.

**The over-refusal failure and the miss failure are the same measurement.** The
detector has one number and the two sequences differ at one residue, so there
is no threshold placement that separates them. That connects this axis directly
to the over-refusal line of work: an over-block on a legitimate biological
sequence and an under-block on a hazard are two readings of a single
insufficient resolution, not two independent problems to be traded off.

And the chemically inactivated toxoids make the bound sharper, because they are
sequence identical. For those, the resolution is not merely insufficient, it is
zero. A sequence-layer detector cannot do this job, so the claim that belongs in
a safety case is about where in the pipeline the discrimination has to happen,
not about how well the sequence model scores.

---

## 6. Confounders, each with its control fixed in advance

| confounder | why it bites here | control |
|---|---|---|
| evolutionary conservation | catalytic residues are conserved in benign enzymes too, so any evolution-trained model is confident there for reasons unrelated to hazard | P2 and P6 |
| annotation circularity | a residue is annotated catalytic partly because mutating it destroyed function, and that literature is in the training corpus | P3's ceiling; report the subset of positions whose annotation predates the model's corpus cutoff separately |
| composition and length | composition alone reaches 0.754 on this panel's separability task | P5 |
| coordinate mismatch | two independently-coordinated annotation sources, site set and ground truth, either of which can be on mature or precursor numbering | the identity check from `src/46` as a hard precondition, failing loudly; the offsets already found are +35, +47 and +32 |
| effective n | a handful of well-studied toxins supply most annotated substitutions | report effective n by distinct protein next to the raw substitution count, as the negative pool reports 3,550 distinct names against 8,259 raw entries |
| head and pooling choice | logistic regression is the worst of four heads on 6 of 13 configurations in the existing sweep | fix mean pooling and logistic regression in advance, report as a near lower bound, do not sweep for the best |
| seed instability | the 5-seed peak arm in the existing panel is not the 30-seed peak arm | 30 seeds for any arm-to-arm statement, and no claim about which arm is best from 5 |
| the experimenter | every reduction of this tensor is a candidate metric, and several will look good | FSPE-M as written in 2.1 is the only primary reduction; any other is exploratory and labelled so |

---

## 7. Execution order, and what stops the study

1. Verify the tier 3 substitution positions against UniProt. If none survive the
   identity check, tier 3 is dropped and the document is amended to say so.
2. Build the tier 2 set under the section 3 rule, logging every exclusion with
   its reason. **If fewer than 12 substitutions survive across at least 4
   distinct proteins, P3 is dropped rather than run underpowered**, and the
   remaining primary tests drop to five with the threshold recomputed to 0.01.
3. Re-run the masked forward pass saving the FSPE-M reduction. Confirm FSPE is
   reproduced bit for bit from the same run, since a changed entropy on a
   re-run means something else moved and the study stops until that is
   explained.
4. P5's label-shuffled arm before anything else is interpreted. If it does not
   return near 0.5, stop.
5. P1, P2, P6, then P3, then the tier 3 table.
6. Pin every surviving number in `src/22_claims_audit.py` with a recomputation
   from the artifact, not a hand-copied figure.

The study also stops, with the partial result reported, if the numbering
precondition in step 3 fails on any panel member, because the entire reason this
document exists in its current form is a numbering defect that computed through
silently for months.

---

## 8. What would make this worth publishing, and what would not

Worth it: P2 failing, with a clear statement that the metric tracks
conservation rather than hazard. P6 failing, with a clear statement that an
alignment does this as well as a protein language model. P4 flat, with the
single-threshold argument of section 5 and the sequence-identical-toxoid bound.
Any of those is a usable, checkable limit on what sequence-level detectors can
do, and all three are negative results.

Not worth it: FSPE-M above 1 on most proteins with no benign control, which is a
restatement of the fact that active sites are conserved. A per-variant ranking,
for the reasons in 2.3. A matched-pair fraction quoted as a rate. Any claim from
one model arm.

---

## Amendment log

Append only. Each entry dated, with what changed and why, written before the
next run rather than after.

- 2026-09-21: document created. No run has been executed. No result has been
  looked at.

- 2026-09-21: **step 1 of section 7's run order executed: tier 3 coordinates
  verified against UniProt.** `src/51_tier3_coordinate_verification.py`,
  artifact `results/v3/tier3_coordinate_verification.json`, UniProt records
  cached under `data/uniprot_cache/` so the check is reproducible without a
  network. No model was run and no score was computed. What changed:

  **Tier 3 is not dropped.** All four panel sequences are byte-identical to
  their UniProt entries, every offset comes from UniProt's own Signal or
  Propeptide boundary rather than from assumption, and the wild-type residue is
  correct at mature+offset in all four cases. The verified positions, in
  **precursor** coordinates, which is what the pipeline indexes:

  | pair | mature | offset | precursor | residue | UniProt Mutagenesis | tier 2 LOF |
  |---|---|---|---|---|---|---|
  | BoNT-A light chain E->Q | 223 | +1 | **224** | E | **E->K,Q**, "Light chain no longer cleaves SNAP25" | yes |
  | pertussis S1 9K | 9 | +34 | **43** | R | absent | n/a |
  | pertussis S1 129G | 129 | +34 | **163** | E | E->D only, "Reduction of several orders of magnitude" | yes |
  | diphtheria CRM197 G52E | 52 | +32 | **84** | G | absent | n/a |
  | ricin E177 | 177 | +35 | **212** | E | **absent** | no |

  **The ricin row does not survive as written and is amended.** Its stated check
  was "verify that the phenotype text meets the tier 2 rule rather than assuming
  it". There is no phenotype text: UniProt annotates no Mutagenesis at precursor
  212 at all. The only loss-of-function variant P02879 carries is D110 in
  precursor coordinates, D75 mature, "Suppresses the toxic activity", which is
  part of the vascular-leak-syndrome LDV motif at mature 74-76 and not an
  active-site residue. The other four annotated variants on that entry open with
  "No effect on the toxic activity", which is a real phenotype and not a loss of
  the toxic function. So ricin is demoted from a tier 3 pair to a candidate that
  would need a literature source outside UniProt, and it is **last** in the run
  order rather than fourth. If it is used at all, the pair must be stated as
  D75/D110 with the VLS phenotype, not as an active-site substitution.

  **A limit on the identity check, recorded because it would otherwise be
  over-read.** The offset sweep pins the offset on identity evidence alone for
  only one of the four, P04977, which is the only pair carrying two
  substitutions. One substitution is one constraint, and over 121 candidate
  offsets roughly six will satisfy it by chance: the sweeps return 15, 9 and 7
  admissible offsets for diphtheria, BoNT-A and ricin respectively. Those three
  offsets rest on the UniProt feature boundary, which is independent of the
  identity check but is a single source. This is weaker than the functional-site
  numbering fix in `src/46`, where each offset was the unique integer satisfying
  three to five simultaneous constraints, and the difference is stated rather
  than smoothed over.

  **One number is now fixed that two coordinate systems both have a claim on.**
  The BoNT-A zinc-ligand glutamate is **precursor 224**; the HExxH motif reads
  H223-E224-L225-I226-H227 there, and UniProt places its Mutagenesis feature at
  224. The research literature numbers the same residue **223**, counting from
  the light chain's own first residue, because the light chain is Chain 2-448 of
  a precursor whose residue 1 is the initiator methionine. Both are correct in
  their own frame and only 224 is usable here. This is `docs/DATA_CORRECTIONS.md`
  entry sixteen's failure mode on a different protein, caught before a run
  rather than after, which is the reason section 7 put this step first.

  **Strength ordering for the run, which is new information the document did not
  have.** BoNT-A is the strongest pair: the exact substitution is
  UniProt-annotated with a loss-of-function phenotype. Pertussis is second: both
  positions verify and the position carries a loss-of-function phenotype, but
  UniProt annotates E163D rather than the vaccine mutant's E163G, so the
  substitution itself is sourced from the vaccine literature and not from
  UniProt. Diphtheria is third: position and residue verify, no UniProt
  Mutagenesis annotation exists, and CRM197's identity rests entirely on the
  product literature. Ricin is last, as above. Section 7's order is superseded by
  this one.
