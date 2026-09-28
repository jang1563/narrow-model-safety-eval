
---

## 6. Two repairs that grew the wrong side

The most useful thing this project learned about itself is not about protein models.

**The first repair.** A preregistered metric extension compared the panel's masked-prediction constraint
against **four** benign controls. Three of its four AUROC tests were unusable at that *n*, which the
preregistration recorded as one finding about its own design. So we built the set it asked for: **60**
reviewed Swiss-Prot enzymes carrying `Active site` features, every frozen exclusion holding, the build
reproducing byte-identically. The verdict held — the controls exceed the panel, so the axis measures
positional constraint and not hazard — and then a review found the comparison **annotation-confounded**:
the controls' positions are UniProt `Active site` features exclusively, and **17 of the panel's 74
annotated positions (23%)** are. Restricting both sides to the same annotation type leaves **4 usable
panel proteins**, halves the gap, and gives an AUROC interval that **covers 0.50**.

🔴 **A second defect surfaced in the same pass and had been documented nowhere: the panel's hazard arm
contains three BSL-1 entries with no hazard designation** — barnase, colicin E2, and Cas9, the last
described in the annotation file itself as a widely used research tool. Two of the four
annotation-matched survivors are barnase and Cas9. With both restrictions applied the hazard arm is
**ricin and one phospholipase**.

**The second repair.** A gate on that same extension had an AUROC tolerance with no power at four
controls. The amendment that retired it diagnosed the defect precisely — *"a ±0.05 tolerance on a
statistic whose null standard deviation is 0.167 is not a threshold, it is a coin weighted against
passing"* — named the defect class as *"a threshold carried across a change of sample size without a power
argument"*, and predicted the tolerance would become meaningful above thirty controls. It does not. An
AUROC's precision is set by the **smaller** group; the null standard deviation floors near **0.077** and a
clean pipeline clears the tolerance under half the time **at any number of controls**.

🔑 **Both repairs grew the negative side. Both limits live in the panel.** Fifteen proteins, seventy-four
annotated positions, seventeen confirmed by a common source, four proteins usable for the comparison. One
resource blocks three tests, and it was never the thing being bought.

⚠️ **The generalizable claim is small and we think it is real: an evaluation can be improved in the wrong
direction, and writing the ceiling down before the run is what reveals which side is binding.** Both times
the diagnosis was in the record before the repair was built; both times it named the wrong sample size.

---

## 7. What this does not claim

- **Not a better classifier.** DeepVIC reports AUROC 0.954 on a 13,384-sequence holdout from 33,456 VFs.
  Our figures are on a self-built panel of 234 or 445 and are **not comparable**.
- **Not novel on homology control.** Homology-clustered evaluation is established practice.
- **Not a competence boundary.** The held-out class is removed from the *probe's* training, not from the
  foundation model's pretraining, and every class here is in the public databases these models saw.
- **Not deployment-ready.** § 10.8 of the long write-up puts it quantitatively: every specificity above
  0.9915 is extrapolation on this panel, and calibrating a one-in-ten-thousand budget would need a negative
  set roughly **850 times** larger. Common Mechanism and SecureDNA run in production; this does not.
- **Not evidence that hazard is what is being detected.** A provenance probe with the hazard label ignored
  reaches 0.818.
- **Not a controlled comparison of pretraining corpora.** The arms differ in corpus, architecture and
  scale at once.

---

## 8. Discussion

The gap this paper is named after — 0.974 separability, about 1% deployment precision — is not a defect of
one probe. It is what a class-imbalanced screen with a calibration set of 118 points looks like when the
false-positive rate is measured on negatives it has not seen. Three practices made it visible, and all
three are cheap.

**Write the ceiling, not only the floor.** A preregistration with a floor can only confirm. Both of ours
returned NOT SUPPORTED, and the second returned it *legibly* because its ceiling said in advance what a
benign control exceeding the panel would mean.

**Run every geometric claim across representations.** This rule killed three findings in two days: a
typicality baseline at −0.746 and *p* = 0.0034 that became +0.021 and *p* = 0.53 on a second arm; a
window reduction worth +19.5 points that cost 9.5 on the same arm's smaller sibling; and a supervised
ranking that raised the panel mean on one arm and on no *k* of the other. 🔑 **The rule was worth more
than the findings it killed**, and the cheapest arm in the project did all three kills for about twenty
minutes of compute each.

**Keep the log append-only, including the entries that hurt.** Sixty entries, several retracting our own
conclusions — a benchmark attributed to the wrong paper, a false-positive check that turned out to be
4/61 by construction and therefore identical for random scores, a cross-reduction comparison made across
geometries whose cosine distributions differ by a factor of three. ⚠️ **The pattern across those is one
thing, not three: we checked whether numbers were stable and not whether they measured what we said.**
Seeds, intervals and replication got attention; a definitionally constant quantity, an
annotation-provenance mismatch and an incomparable scale did not, and no number of additional seeds would
have surfaced any of them.

What we would want next is not another representation. It is **more panel**: mechanism classes from a
published ontology rather than hand curation, and catalytic positions from one source. VFDB supplies the
first — fourteen categories, of which **Exotoxin is one and 4% of the records** — and that number is its
own warning, because a virulence-factor axis is not a toxin axis, and adopting it changes what the screen
is for.
