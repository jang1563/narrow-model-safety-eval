# What external review and AI4Sci Korea 2026 imply for this project

**Written 2026-09-30.** Sources: three NeurIPS 2026 reviews of BioReview-Bench (submission 1932, ratings
**2 / 2 / 2**, confidence 4 / 4 / 3) and their synthesis; AI4Sci Korea 2026 session notes for 28–30
September; the FLMSec 2026 workshop review form. 🔒 **Nothing here is a claim about those papers or
venues** — it is a record of which criticisms transfer to *this* repository and what was done about each.

---

## 1. Criticisms of a different paper that land on this one

### 🔴 1.1 "The evaluated units were never independently validated" — it lands

The reviewers' release-blocking gap was that structured targets were LLM-produced and never checked by
blinded annotators, and that the existing validation file was inadmissible because its human fields were
**identical** to the machine labels by construction.

🔑 **The analogue here is study Q's hazard taxonomy.** § 6's narrowing from *hazard* to *VFDB
registration* rests on a four-category non-hazardous group, and
`docs/HAZARD_VS_MEMBERSHIP_PREREGISTRATION.md` § 1.1 already says in bold that **the assignment is one
author's judgment**, frozen in advance, with the largest ambiguous class excluded so it cannot decide the
result.

🟢 **Better than the criticised case in two ways**: the judgment is declared rather than presented as
ground truth, and it was frozen before the run. 🔴 **Worse in one**: there is still **no second annotator
and no agreement statistic**. ⚠️ **Disclosure is not validation**, and § 7 now says so.

### 🔴 1.2 "Remove internal experiment codes and dramatic claims" — it lands hard

Reviewer S4zR asked for plain language, defined abbreviations and no internal codes. 🔴 **This paper is
written almost entirely in them**: *study K*, *study Q*, *B-1*, *A1*, *§ 6l*, *F-DNA3*, *C/A*, *S1b*.
⚠️ **Recorded as an open editorial debt**, not fixed today, because renaming across 36 documents and 126
audited claims is a change that needs its own pass with the gate green at every step.

### 🟢 1.3 "Treat a strong cheap baseline as benchmark-specific until controls separate it" — already done

Reviewer j6RR's point about BM25 is the same shape as this project's **metadata floor** (AUROC 0.8096 from
localization stratum and log length). 🔒 The controls that reviewer asked for exist here: matched benign,
the localization control, the provenance control and the joint decomposition, which is why the floor is
reported as *"a baseline of this kind is missing from that literature"* and not as a claim about anyone's
numbers.

### 🟢 1.4 "Publish full quality control, not selected examples" — mostly done, one gap closed today

The per-class table now carries **all twelve** classes on one panel at one seed count, and the panel ×
seed-count grid is printed rather than described (§ 3).

---

## 2. AI4Sci Korea 2026 — vocabulary this project needed and did not have

### ⭐ 2.1 Gundersen (NTNU): verify vs validate, and the three degrees

- **Verify** = no errors in running or reporting. **Validate** = the experiments answer the question.
  🔑 **Every defect the claims audit has ever caught is a verification defect.** §§ 6–7 are where
  validation lives, and that is where the adverse findings are. § 9 now says this.
- **R1–R4** documentation types, and the trade-off that a reproduction from *less* documentation is
  stronger evidence. 🔒 Study K's `C/A` was reproduced at **2.418 / 2.617** by a script written on a
  different code path — a reproduction from the description, which is the stronger kind.
- **Outcome / analysis / interpretation reproducible.** 🔑 **This names segment 1's finding exactly**: the
  per-class table is *not* outcome-reproducible across panels (19–40 points) and *is* interpretation-
  reproducible (same two classes lowest, three at 100%).

### ⭐ 2.2 Acuna (CU Boulder): layered defence, and measuring residual error

*"No single check catches everything, so combine complementary checks and measure their residual errors and
shared blind spots."* 🔒 This repository has measured one residual directly — a **94% false-pass rate** on
a manuscript-figure audit at three significant figures (entry 68) — and a September 2026 review pass found
**four blind-spot classes**: predicates scoped narrower than their sentences, public numbers with no claim
at all, forbid strings matching their own disclaimers, and intervals answering a different question from
the prose. ⚠️ **Reporting the gate's size without its residual error was the omission**, and § 9 now
carries both.

### 🔑 2.3 Korea AI Safety Institute: the gap, not the pace

*"The central problem is not the pace of capability but the gap between that pace and our capacity to
evaluate what has been built."* 🔒 That is this project's subject in one sentence — **AUROC 0.974 against
what the probe actually responds to** — and it is a better framing for § 1 than the deployment-precision
lead alone.

### ⚠️ 2.4 FLMSec review form: candid negative results count as a strength

The form lists *"candid negative results"* among the things a reviewer may cite as a concrete strength.
🔒 Recorded because this paper is mostly negative results, and the corrections log is the evidence that
they were not selected after the fact.

---

## 3. What this changed today

| | change |
|---|---|
| § 7 | a new bullet: the hazard taxonomy is one author's judgment with **no second annotator** |
| § 9 | verify-vs-validate, R4, the three degrees, and the audit's **measured residual error** |
| § 9 | the stale **"90 claims"** corrected, and claim 126 now makes the paper state the registry's real size |
| § 9 | the stale *"never tested on classes this project did not define"* corrected — it was, and entry 85 says how |
| open | the internal-code vocabulary, recorded as editorial debt rather than fixed |
| open | a second annotator for the hazard split, which is the one item here that needs new human work |
